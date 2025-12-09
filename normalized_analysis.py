import pandas as pd
import numpy as np
import configuration as cf
import utils as utils
import calcScore as csf
import postBo as pb
from datetime import datetime


def normalize_bometrics(BoMetric_df, lower_q=0.01, upper_q=0.99):
    """Return a normalized copy of BoMetric_df where numeric metric columns are
    winsorized (per-date) and scaled to cross-sectional z-scores per date.

    Keeps `date` and `source` columns intact.
    """
    df = BoMetric_df.copy()
    if df.empty:
        return df

    # ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    metric_cols = [c for c in df.columns if c not in ['date', 'source']]

    def norm_group(g):
        for col in metric_cols:
            s = pd.to_numeric(g[col], errors='coerce')
            if s.dropna().empty:
                g[col] = np.nan
                continue
            lo = s.quantile(lower_q)
            hi = s.quantile(upper_q)
            s_clip = s.clip(lower=lo, upper=hi)
            mu = s_clip.mean()
            sigma = s_clip.std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                g[col] = (s_clip - mu) * 0.0
            else:
                g[col] = (s_clip - mu) / sigma
        return g

    df_norm = df.groupby('date', group_keys=False).apply(norm_group)
    # keep original ordering
    df_norm = df_norm.reset_index(drop=True)
    return df_norm


def run_normalized_pipeline(loadfname=None, save_results=True):
    """Load metric data via utils and run the analysis on normalized metrics.

    - Creates `BoMetric_norm_df` and runs the same scoring/post-processing pipeline using
      the normalized metrics so you can compare results with the original pipeline.
    - Adds a timing flag per ticker if data includes current-year records.
    - Returns a tuple (resdic_norm, datandmetricdic_norm)
    """
    # get defaults from configuration
    cfg = cf.getDataFetchConfiguration([])
    if loadfname is None:
        loadfname = cfg.get('loadBoMetricfname')
    load_dic = {'loadBoMetric': 1, 'loadBoMetricfname': loadfname}
    dmdic = utils.loadWrapper('metric', load_dic)

    BoMetric_df = dmdic['BoMetric_df']
    cdx_df = dmdic.get('cdx_df')

    # make normalized dataframe
    BoMetric_norm_df = normalize_bometrics(BoMetric_df)

    # detect timing issues: tickers that have a row with date in current year
    current_year = datetime.today().year
    timing_flag = []
    for src in BoMetric_df['source'].unique():
        df_t = BoMetric_df[BoMetric_df['source'] == src]
        if not df_t.empty and df_t['date'].dt.year.max() == current_year:
            timing_flag.append(src)

    # assemble new datandmetricdic and run existing pipeline
    dmdic_norm = dict(dmdic)  # shallow copy
    dmdic_norm['BoMetric_df'] = BoMetric_norm_df
    # recompute averages used by calcScore/postBo
    meandic = csf.getAves2(BoMetric_norm_df)
    dmdic_norm.update(meandic)

    # attach timing flag list for auditing
    dmdic_norm['timing_maybe_off'] = timing_flag

    # run post-analysis on normalized metrics
    resdic_norm = pb.postBoWrapper(dmdic_norm)
    # also attach the timing flag and normalized dataframe to the results
    resdic_norm['BoMetric_norm_df'] = BoMetric_norm_df
    resdic_norm['timing_maybe_off'] = timing_flag

    if save_results:
        # prepare a minimal wrapper similar to utils.saveWrapper but for normalized results
        try:
            fname = f'results_normalized_{datetime.today().strftime("%Y-%m-%d")}.pickle'
            pd.to_pickle(resdic_norm, fname)
        except Exception:
            pass

    return resdic_norm, dmdic_norm


if __name__ == '__main__':
    r, d = run_normalized_pipeline()
    print('Normalized pipeline finished. Results keys:', list(r.keys()))
