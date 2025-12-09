import pandas as pd
import numpy as np
import configuration as cf
import utils as utils
import getData_gen as gdg
import gains as gains
from datetime import datetime

# Simple rolling backtest harness that (1) builds scores from BoMetric_df per rebalance date,
# (2) selects top-N tickers, (3) computes portfolio returns between rebalance dates,
# (4) reports monthly returns. This is a simple, readable harness — not ultra-performant.


def latest_metrics_as_of(BoMetric_df, asof_date):
    # for each ticker, pick the latest row with date <= asof_date
    df = BoMetric_df.copy()
    df = df[df['date'] <= asof_date]
    if df.empty:
        return pd.DataFrame()
    df_sorted = df.sort_values(['source', 'date']).groupby('source').tail(1)
    return df_sorted.set_index('source')


def compute_scores_from_metrics(metrics_df):
    # metrics_df: indexed by source, columns are metrics (no date)
    # compute cross-sectional z-score across tickers per metric and average
    if metrics_df.empty:
        return pd.Series(dtype=float)
    metric_cols = [c for c in metrics_df.columns if c not in ['date', 'source']]
    # ensure numeric
    metrics = metrics_df[metric_cols].apply(pd.to_numeric, errors='coerce')
    z = (metrics - metrics.mean()) / metrics.std(ddof=0)
    # average z across metrics (mean) ignoring NaNs
    score = z.mean(axis=1)
    return score


def price_on_or_after(cdx_df, symbol, asof_date):
    tmp = cdx_df[(cdx_df['source'] == symbol) & (cdx_df['date'] >= asof_date)]
    if tmp.empty:
        return np.nan
    return tmp.sort_values('date').iloc[0]['price']


def run_backtest(loadfname=None, rebalance='A', topn_list=[3,10], weight_style='equal'):
    # load metrics
    cfg = cf.getDataFetchConfiguration([])
    if loadfname is None:
        loadfname = cfg.get('loadBoMetricfname')
    load_dic = {'loadBoMetric': 1, 'loadBoMetricfname': loadfname}
    dmdic = utils.loadWrapper('metric', load_dic)
    BoMetric_df = dmdic['BoMetric_df'].copy()
    cdx_df = dmdic['cdx_df'].copy()

    # ensure dates
    BoMetric_df['date'] = pd.to_datetime(BoMetric_df['date'])
    cdx_df['date'] = pd.to_datetime(cdx_df['date'])

    # build rebalance dates from BoMetric_df date range
    # rebalance: 'A' = annual (default), 'Q' = quarterly (max), 'M' = monthly (will be capped to quarterly)
    dates = sorted(BoMetric_df['date'].unique())
    if not dates:
        print('No dates found in metrics')
        return
    all_dates = pd.DatetimeIndex(dates)
    rebalance = (rebalance or 'A').upper()
    if rebalance not in ('A', 'Q', 'M'):
        print(f"Unknown rebalance '{rebalance}', defaulting to annual ('A').")
        rebalance = 'A'
    if rebalance == 'M':
        # user requested monthly — cap to quarterly (we prefer less trading)
        print('Monthly rebalance requested; capping to quarterly to limit turnover.')
        rebalance = 'Q'

    if rebalance == 'A':
        # annual anchors: use year-end dates present in metrics
        rb_dates = sorted({pd.Timestamp(d.year, 12, 31) for d in all_dates})
        rb_dates = [ (d if d in all_dates else max(all_dates[all_dates <= d])) for d in rb_dates if any(all_dates <= pd.Timestamp(d))]
    elif rebalance == 'Q':
        # quarterly anchors: use quarter-end dates present in metrics
        rb_dates = sorted({pd.Timestamp(d.year, ((d.month-1)//3+1)*3, 1) for d in all_dates})
        rb_dates = [ (d + pd.offsets.MonthEnd(0)) for d in rb_dates]
        rb_dates = [ (d if d in all_dates else max(all_dates[all_dates <= d])) for d in rb_dates if any(all_dates <= pd.Timestamp(d))]

    # Build series of rebalance points where we have metrics (ensure ascending and present)
    rb_points = []
    for d in rb_dates:
        # pick the last metric date <= anchor d
        candidates = all_dates[all_dates <= d]
        if len(candidates) > 0:
            rb_points.append(candidates.max())
    rb_points = sorted(list(dict.fromkeys(rb_points)))
    if len(rb_points) < 2:
        # fallback to using available unique dates
        rb_points = dates

    results = {n: [] for n in topn_list}
    results_dates = []

    for i in range(len(rb_points)-1):
        asof = rb_points[i]
        nex = rb_points[i+1]
        # get latest metrics as of 'asof'
        metrics = latest_metrics_as_of(BoMetric_df, asof)
        if metrics.empty:
            continue
        scores = compute_scores_from_metrics(metrics)
        scores = scores.dropna()
        # get price as of or after asof and price as of or after nex
        entry_prices = {}
        exit_prices = {}
        symbols = list(scores.index)
        for s in symbols:
            entry_prices[s] = price_on_or_after(cdx_df, s, asof)
            exit_prices[s] = price_on_or_after(cdx_df, s, nex)
        # compute returns per symbol
        rets = {}
        # small in-memory cache for dividend history per symbol to avoid repeated API calls
        div_cache = {}
        for s in symbols:
            p0 = entry_prices.get(s)
            p1 = exit_prices.get(s)
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                rets[s] = np.nan
            else:
                # compute dividends per share paid during (asof, nex]
                try:
                    baseurl = cfg.get('baseurl')
                    api_key = cfg.get('api_key')
                except Exception:
                    baseurl = None
                    api_key = None

                div_sum = 0.0
                if baseurl and api_key:
                    if s in div_cache:
                        hist_df = div_cache[s]
                    else:
                        hist_df = gains.getHistDivs(s, api_key, baseurl)
                        div_cache[s] = hist_df

                    if hist_df is not None and not hist_df.empty:
                        div_sum = gains.getDividends(s, hist_df, asof, nex)

                rets[s] = (p1 - p0 + div_sum) / p0
        # for each topn, compute portfolio return
        for topn in topn_list:
            top_symbols = scores.sort_values(ascending=False).head(topn).index.tolist()
            # weights
            if weight_style == 'equal':
                w = np.array([1.0/topn]*topn)
            elif weight_style == 'linear_decay':
                ranks = np.arange(topn,0,-1)
                w = ranks / ranks.sum()
            else:
                w = np.array([1.0/topn]*topn)
            vals = np.array([rets.get(s, np.nan) for s in top_symbols], dtype=float)
            # ignore nan positions and renormalize weights
            valid_mask = ~np.isnan(vals)
            if not valid_mask.any():
                portret = np.nan
            else:
                w_valid = w[valid_mask]
                w_valid = w_valid / w_valid.sum()
                portret = (w_valid * vals[valid_mask]).sum()
            results[topn].append(portret)
        results_dates.append(nex)

    # build DataFrame and print monthly/yearly gains
    out = {}
    for topn in topn_list:
        ser = pd.Series(results[topn], index=pd.to_datetime(results_dates))
        out[topn] = ser
    # produce a summary table: monthly returns and yearly aggregated returns
    summary_rows = []
    for topn, ser in out.items():
        monthly = ser
        yearly = monthly.groupby(monthly.index.year).apply(lambda x: (1 + x.fillna(0)).prod() - 1)
        summary_rows.append({'topn': topn, 'months': monthly, 'yearly': yearly})

    return out


if __name__ == '__main__':
    out = run_backtest()
    for topn, ser in out.items():
        print(f'--- Top {topn} monthly returns (first 12 shown) ---')
        print(ser.head(12))
