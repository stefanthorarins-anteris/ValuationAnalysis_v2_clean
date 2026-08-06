"""READ-ONLY per-exchange data-completeness diagnostic (offline, no network).

WHY IT EXISTS.  Two dead exchange codes (EURONEXT / OSE) meant 1,046 Paris / Amsterdam /
Brussels / Lisbon / Oslo names were never in any run, and are about to enter the default
universe.  The question this answers is NOT "does FMP serve those names" (unanswerable
offline) but "do the non-US exchanges ALREADY in the panel behave like the US ones on data
completeness, and if not, does the pipeline's missing-data handling REWARD the gap?"

The reward channel is specific and it is the reason this is not cosmetic:
  * STAGE 1 (calcScore.calcByTier)     -- a NaN criterion scores as a FAIL.  Missingness is
                                          a PENALTY, and it operates as a top-100 GATE.
  * STAGE 2 (postBoRank.normalizeAndDropNA) -- a NaN metric is filled at the POOL MEAN after
                                          z-scoring (fillna(0)), which on a right-skewed
                                          column sits ABOVE the observed median.  Missingness
                                          is a REWARD, conditional on clearing Stage 1.

EMITS ONLY.  Nothing is written back into any pipeline frame; no pipeline module is mutated;
no HTTP call is made.  Every table is printed and optionally written to CSV under
baseline_tools/pxc_*.csv.
"""

import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import createDicts as cdic
import scoringWeights as sw
import postBoRank as pbr
import stage2_metrics as sm
import reporting_period as rp
import carveOut as co
import data_quality as dq

PANEL = os.path.join(_HERE, 'resdic_2026-07-17_CORRECTED.pickle')

STALE_STORED_COLS = ('incomeQuality', 'EPStoEPSmean')


def suffix(s):
    if not isinstance(s, str) or '.' not in s:
        return 'US'
    return s.rsplit('.', 1)[1].strip().upper()


def lgroup(s):
    g = suffix(s)
    if g == 'L' and co._IOB_LSE_SYMBOL_RE.match(str(s).strip()):
        return 'L_IOB'
    return g


def load_panel(path=PANEL):
    with open(path, 'rb') as f:
        return pickle.load(f)


STAGE2_INPUTS = {
    'RoA':                        ['returnOnAssets'],
    'earnYield':                  ['earningsYield'],
    'grahamNumberToPrice':        ['grahamNumber', 'price'],
    'bVpRatio':                   ['pbRatio'],
    'revenueGrowth':              ['revenue'],
    'incomeQuality':              ['netIncome', 'netCashProvidedByOperatingActivities',
                                   'totalAssets'],
    'returnOnEquity':             ['returnOnEquity'],
    'returnOnCapitalEmployed':    ['returnOnCapitalEmployed'],
    'currentRatio':               ['currentRatio'],
    'grossProfitMargin':          ['grossProfitMargin'],
    'freeCashFlowYield':          ['freeCashFlow', 'marketCap'],
    'freeCashFlowPerShareGrowth': ['freeCashFlow', 'weightedAverageShsOut'],
    'marketCapRevQuants':         ['marketCap'],
    'Altman-Z':                   ['totalAssets', 'totalLiabilities', 'totalCurrentAssets',
                                   'totalCurrentLiabilities', 'totalStockholdersEquity',
                                   'operatingIncome', 'revenue', 'marketCap'],
    'Piotroski':                  ['totalAssets', 'netIncome',
                                   'netCashProvidedByOperatingActivities', 'longTermDebt',
                                   'currentRatio', 'weightedAverageShsOut',
                                   'grossProfitMargin', 'revenue'],
    #  E-2's two extracted Piotroski components -- the SAME cdx fields p7 and p5 read.
    'shareCountChange':           ['weightedAverageShsOut'],
    'longTermDebtChange':         ['longTermDebt', 'totalAssets'],
    'tbVpRatio':                  ['tangibleBookValuePerShare', 'price'],
    'EPStoEPSmean':               ['netIncome', 'weightedAverageShsOut'],
    'CycleHeat':                  ['netIncome', 'weightedAverageShsOut', 'date'],
    #  2026-08-06.  Without these two the completeness report SILENTLY omitted them: line ~148
    #  builds its cdx column set from this dict's VALUES, so a missing key drops the raw legs
    #  from the panel and the metric simply never appears -- no error, just an under-reported
    #  coverage table for the S block's new Tier-1 instrument.
    'interestCoverage':           ['operatingIncome', 'interestExpense'],
    'navPerShareGrowth':          ['bookValuePerShare'],
    'DcfToPrice':                 ['LIVE_DCF_ENDPOINT'],
    'BoScore':                    ['STAGE1_SCORE'],
    'priceGrowth':                ['price'],
}


def weight_vector():
    return dict(sw.DEPLOYED)


def weighted_metrics():
    W = weight_vector()
    return [k for k in sw.METRIC_KEYS if float(W[k]) != 0.0]


def stage1_criteria():
    db, dm, dd, du, ds = cdic.getBaseMeanDiffUnitySpecialDicts()
    tw = {'S': 1.0, 'A': 0.75, 'B': 0.5, 'C': 0.3, 'D': 0.1}
    out = []
    for k, v in db.items():
        out.append((k, tw.get(v['Tier'], 0.0)))
    for k, v in dm.items():
        out.append(('m' + k[0].upper() + k[1:], tw.get(v['Tier'], 0.0)))
    for k, v in dd.items():
        out.append(('d' + k[0].upper() + k[1:], tw.get(v['Tier'], 0.0)))
    for k, v in du.items():
        out.append(('u' + k[0].upper() + k[1:], tw.get(v['Tier'], 0.0)))
    for k, v in ds.items():
        out.append((k, tw.get(v['Tier'], 0.0)))
    return out


def emit(df, name, csv=True):
    print('\n' + '=' * 110)
    print(name)
    print('=' * 110)
    with pd.option_context('display.width', 260, 'display.max_columns', 80,
                           'display.max_rows', 500):
        print(df.round(4).to_string())
    if csv:
        safe = ''.join(ch if (ch.isalnum() or ch in '_-') else '_' for ch in name)[:70]
        p = os.path.join(_HERE, 'pxc_%s.csv' % safe)
        df.to_csv(p)
        print('  -> %s' % p)


def field_completeness(cdx, key='grp'):
    """NaN rate per exchange on every Stage-2 raw input column, two ways.

    rowNaN    : share of PANEL ROWS with no value.
    allNaNsrc : share of COMPANIES for which the field is 100 pct NaN, i.e. the metric
                that needs it CANNOT be computed for that company at all.
    """
    cols = sorted({c for v in STAGE2_INPUTS.values() for c in v if c in cdx.columns})
    rows = []
    for col in cols:
        num = pd.to_numeric(cdx[col], errors='coerce')
        tmp = pd.DataFrame({key: cdx[key], 'source': cdx['source'], 'na': num.isna()})
        row = {'column': col}
        for grp, sub in tmp.groupby(key):
            row['%s_rowNaN' % grp] = 100.0 * sub['na'].mean()
        per_src = tmp.groupby([key, 'source'])['na'].mean()
        for grp, sub in per_src.groupby(level=0):
            row['%s_allNaNsrc' % grp] = 100.0 * (sub >= 1.0).mean()
        rows.append(row)
    out = pd.DataFrame(rows).set_index('column')
    return out.reindex(sorted(out.columns), axis=1)


def stage1_field_completeness(bm, key='grp'):
    crit = stage1_criteria()
    rows = []
    for col, w in crit:
        row = {'criterion': col, 'tier_w': w}
        if col not in bm.columns:
            row['MISSING_COLUMN'] = True
            rows.append(row)
            continue
        num = pd.to_numeric(bm[col], errors='coerce')
        tmp = pd.DataFrame({key: bm[key], 'source': bm['source'], 'na': num.isna()})
        for grp, sub in tmp.groupby(key):
            row['%s_rowNaN' % grp] = 100.0 * sub['na'].mean()
        per_src = tmp.groupby([key, 'source'])['na'].mean()
        for grp, sub in per_src.groupby(level=0):
            row['%s_allNaNsrc' % grp] = 100.0 * (sub >= 1.0).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index('criterion')


def stage1_nan_weight(bm, n=8, key='grp'):
    """Replicates the nan_sink accounting in calcScore.calcByTier: per source, per
    criterion, is the head(n) scoring window entirely non-computable? Sums the tier
    weight of every such criterion. That summed weight is the Stage-1 score the name
    forfeits to missingness alone.
    """
    crit = [(c, w) for c, w in stage1_criteria() if c in bm.columns]
    total_w = sum(w for _c, w in crit)
    bm = bm.copy()
    bm['date'] = pd.to_datetime(bm['date'], errors='coerce')
    bm = bm.sort_values(['source', 'date'], ascending=[True, False])
    recs = []
    for src, sub in bm.groupby('source', sort=False):
        head = sub.head(n)
        nan_w = 0.0
        nan_n = 0
        for c, w in crit:
            v = pd.to_numeric(head[c], errors='coerce')
            if len(v) == 0 or v.isna().all():
                nan_w += w
                nan_n += 1
        recs.append({'source': src, 'grp': sub[key].iloc[0], 'rows': len(sub),
                     'nan_crit': nan_n, 'nan_tier_w': nan_w,
                     'nan_w_share': nan_w / total_w})
    return pd.DataFrame(recs), total_w


def stage2_metric_frame(cdx_pool, boscore, nq=16, verbose=True):
    """Recompute the Stage-2 metric frame offline for an arbitrary pool.

    Calls the SAME stage2_metrics functions the live scorer calls, so no formula can
    diverge; cdx is pre-grouped by source purely for speed. DcfToPrice is left NaN
    (weight 0.000, needs the live endpoint).
    """
    postBm, postNew = cdic.getPostDict()
    cols = ['source'] + list(postBm.keys()) + list(postNew.keys())
    cdx_pool = cdx_pool.copy()
    cdx_pool['date'] = pd.to_datetime(cdx_pool['date'], errors='coerce')
    cdx_pool = cdx_pool.sort_values(['source', 'date'], ascending=[True, False])
    cdx_pool['mcapQuants'] = sm.add_mcap_quants(cdx_pool)
    freq_map = rp.frequency_by_source(cdx_pool, verbose=False)
    bs = dict(zip(boscore['source'], boscore['score']))
    groups = {k: v for k, v in cdx_pool.groupby('source', sort=False)}
    recs = []
    srcs = [s for s in boscore['source'] if s in groups]
    for i, ticker in enumerate(srcs):
        if verbose and i % 1000 == 0:
            print('  stage2 %d/%d' % (i, len(srcs)), flush=True)
        tempcdx = groups[ticker]
        _rpy = rp.rows_per_year(freq_map, ticker)
        r = {'source': ticker}
        for key1 in postBm:
            try:
                r[key1] = sm.postbm_metric(key1, postBm[key1]['eqMet'], tempcdx, nq,
                                           rpy=_rpy)
            except Exception:
                r[key1] = np.nan
        tempfcf = tempcdx.freeCashFlow
        tempshares = tempcdx.weightedAverageShsOut
        r['freeCashFlowYield'] = sm.free_cash_flow_yield(tempfcf, tempcdx.marketCap, nq,
                                                        rpy=_rpy, tempcdx=tempcdx)
        r['freeCashFlowPerShareGrowth'] = sm.free_cash_flow_per_share_growth(
            tempfcf, tempshares, nq, rpy=_rpy, tempcdx=tempcdx)
        r['DcfToPrice'] = np.nan
        r['marketCapRevQuants'] = tempcdx.mcapQuants.iloc[0]
        r['tbVpRatio'] = sm.tbv_p_ratio(tempcdx, nq, rpy=_rpy)
        r['Altman-Z'] = sm.altman_z(tempcdx, rpy=_rpy)
        r['Piotroski'] = sm.piotroski(tempcdx, rpy=_rpy)
        r['EPStoEPSmean'] = sm.eps_to_eps_mean(tempcdx, rpy=_rpy)
        r['priceGrowth'] = sm.price_growth(tempcdx, nq, rpy=_rpy)
        r['CycleHeat'] = sm.cycleheat(tempcdx, rpy=_rpy)
        r['BoScore'] = bs.get(ticker, np.nan)
        recs.append(r)
    out = pd.DataFrame(recs)
    return out.reindex(columns=cols)


def fill_advantage(raw, weights, pool_label=''):
    """Per-NAME AggScore advantage attributable PURELY to the mean-fill of missing metrics.

    Mechanism (postBoRank.normalizeAndDropNA): mu and sigma are computed on the OBSERVED
    cells, so the post-z fillna(0) puts a missing metric at the pool MEAN, z = 0.

    Two counterfactuals, both reported, because they answer different questions.

    MEDIAN counterfactual (headline). Had the value been observed, the point guess for a
    typical name is the observed MEDIAN z of that column, m. Advantage = (0 - m) * w. On a
    right-skewed column m < 0, so a positive-weight column yields a POSITIVE advantage.
    This is the counterfactual that missing_data_fill_report uses (percentile vs median) and
    it is the RANK-relevant one: against a randomly drawn peer the filled name wins with
    probability P(z < 0), which exceeds 0.5 whenever the fill sits above the median.

    MEAN counterfactual. Had the value been a random draw from the observed distribution,
    E[z] = 0 exactly, so the EXPECTED advantage is ZERO and the only effect is variance
    reduction. Reported so the headline is not over-read.
    """
    W = dict(weights)
    wcols = [c for c in raw.columns if c != 'source' and float(W.get(c, 0) or 0) != 0.0]
    norm, _outl = pbr.normalizeAndDropNA(raw.copy(), weight_series=pd.Series(W))
    per_col = []
    adv = pd.Series(0.0, index=raw.index)
    n_imp = pd.Series(0, index=raw.index)
    imp_w = pd.Series(0.0, index=raw.index)
    tot_absw = sum(abs(float(W[c])) for c in wcols)
    for c in wcols:
        w = float(W[c])
        rawc = pd.to_numeric(raw[c], errors='coerce')
        zc = pd.to_numeric(norm[c], errors='coerce')
        imputed = rawc.isna()
        obs = zc[~imputed].dropna()
        if len(obs) == 0:
            continue
        med = float(obs.median())
        pct = 0.5 * ((obs < 0.0).sum() + (obs <= 0.0).sum()) / len(obs)
        contrib = (0.0 - med) * w
        adv = adv + imputed.astype(float) * contrib
        n_imp = n_imp + imputed.astype(int)
        imp_w = imp_w + imputed.astype(float) * abs(w)
        eff = (pct - 0.5) * (1.0 if w > 0 else -1.0)
        per_col.append({'pool': pool_label, 'column': c, 'weight': w,
                        'n_imputed': int(imputed.sum()),
                        'pct_imputed': 100.0 * imputed.mean(),
                        'observed_z_median': med,
                        'fill_pctile_in_observed_z': pct,
                        'fill_dir': ('ADVANTAGE' if eff > 0 else
                                     'PENALTY' if eff < 0 else 'neutral'),
                        'agg_adv_per_imputed_name': contrib})
    name_df = pd.DataFrame({'source': raw['source'].values,
                            'n_imputed_cols': n_imp.values,
                            'imputed_weight_share': (imp_w / tot_absw).values,
                            'fill_advantage_aggscore': adv.values})
    return pd.DataFrame(per_col), name_df, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=PANEL)
    ap.add_argument('--no-csv', action='store_true')
    ap.add_argument('--skip-stage2', action='store_true')
    ap.add_argument('--iob-split', action='store_true',
                    help='report LSE IOB depositary lines as their own group L_IOB')
    a = ap.parse_args()
    csv = not a.no_csv
    keyfn = lgroup if a.iob_split else suffix

    d = load_panel(a.panel)
    tick = d['Tickers_df'].copy()
    cdx = d['cdx_df'].copy()
    bm = d['BoMetric_df'].copy()
    bsc = d['BoScore_df'].copy()
    labels = d['carveout_labels']
    memb = d['carve_full_membership']
    diag = d['carveout_diagnostics']
    postRank = d['postRank'].copy()

    for f in (cdx, bm):
        f['grp'] = f['source'].map(keyfn)
    tick['grp'] = tick['symbol'].map(keyfn)
    bsc['grp'] = bsc['source'].map(keyfn)

    groups = sorted(tick['grp'].unique())
    print('PANEL          : %s' % a.panel)
    print('EXCHANGE GROUPS: %s' % groups)
    print('STALE STORED COLUMNS (never read here, Stage-2 is RECOMPUTED): %s'
          % list(STALE_STORED_COLS))

    # ---------- populations ----------
    ded = diag.get('dedup', {})
    pop = pd.DataFrame(index=groups)
    pop['1_tickers_prefetch'] = tick.groupby('grp').size()
    pop['2_panel_sources'] = cdx.groupby('grp')['source'].nunique()
    pop['3_scored_lines'] = bsc.groupby('grp').size()
    iss = pd.Series({g: 0 for g in groups}, dtype=float)
    lab_grp = pd.Series(labels.index.map(keyfn), index=labels.index)
    pop['4_issuers_postdedup'] = lab_grp.value_counts()
    for lab in ('general', 'REIT', 'Mining', 'InvestmentVehicle', 'FinManager',
                'BalanceSheetFin'):
        s = pd.Series(list(memb.get(lab, [])))
        pop['5_%s' % lab] = s.map(keyfn).value_counts() if len(s) else 0
    below = set(labels.index) - set().union(*[set(memb.get(l, [])) for l in memb])
    pop['5_below_25M_floor'] = pd.Series(sorted(below)).map(keyfn).value_counts()
    pop['6_top100_general'] = postRank['source'].map(keyfn).value_counts()
    pop = pop.fillna(0).astype(int)
    pop.loc['TOTAL'] = pop.sum()
    pop['fetch_pass_pct'] = 100.0 * pop['2_panel_sources'] / pop['1_tickers_prefetch']
    pop['dedup_survive_pct'] = 100.0 * pop['4_issuers_postdedup'] / pop['3_scored_lines']
    pop['general_pct_of_issuers'] = 100.0 * pop['5_general'] / pop['4_issuers_postdedup']
    pop['top100_per_1000_general'] = 1000.0 * pop['6_top100_general'] / pop['5_general']
    emit(pop, '1_populations_by_exchange', csv)

    # ---------- history depth ----------
    rows_per_src = cdx.groupby(['grp', 'source']).size().rename('rows')
    hist = rows_per_src.groupby(level=0).describe()[['count', 'mean', 'min', '25%',
                                                     '50%', '75%', 'max']]
    for thr in (8, 12, 16, 20, 28):
        hist['pct_lt_%d' % thr] = rows_per_src.groupby(level=0).apply(
            lambda s, t=thr: 100.0 * (s < t).mean())
    span = cdx.copy()
    span['date'] = pd.to_datetime(span['date'], errors='coerce')
    sp = span.groupby(['grp', 'source'])['date'].agg(['min', 'max'])
    sp['years'] = (sp['max'] - sp['min']).dt.days / 365.25
    hist['median_years_span'] = sp.groupby(level=0)['years'].median()
    hist['median_newest_date'] = sp.groupby(level=0)['max'].median()
    emit(hist, '2_history_depth_by_exchange', csv)

    # ---------- reporting frequency ----------
    if 'reportingFrequency' in cdx.columns:
        fq = cdx.groupby(['grp', 'source'])['reportingFrequency'].first()
        fqt = pd.crosstab(fq.index.get_level_values(0), fq.values, normalize='index') * 100
        fqn = pd.crosstab(fq.index.get_level_values(0), fq.values)
        fqt.columns = ['pct_' + str(c) for c in fqt.columns]
        fqn.columns = ['n_' + str(c) for c in fqn.columns]
        emit(pd.concat([fqn, fqt], axis=1), '3_reporting_frequency_by_exchange', csv)

    # ---------- sector / industry coverage ----------
    smap = co._load_sector_map()
    imap = co._load_industry_map()
    iss_syms = pd.Series(list(labels.index))
    cov = pd.DataFrame(index=groups)
    g = iss_syms.map(keyfn)
    cov['n_issuers'] = g.value_counts()
    cov['sector_in_map_pct'] = (iss_syms.map(lambda s: s in smap)
                                .groupby(g).mean() * 100)
    cov['sector_KNOWN_pct'] = (iss_syms.map(lambda s: co._is_known_sector(smap.get(s)))
                               .groupby(g).mean() * 100)
    cov['industry_pct'] = (iss_syms.map(lambda s: bool(imap.get(s)))
                           .groupby(g).mean() * 100)
    # same, on the pre-dedup scored universe
    g2 = bsc['source'].map(keyfn)
    cov['scored_lines'] = bsc.groupby('grp').size()
    cov['scored_sector_KNOWN_pct'] = (bsc['source']
                                      .map(lambda s: co._is_known_sector(smap.get(s)))
                                      .groupby(g2).mean() * 100)
    emit(cov, '4_sector_industry_coverage_by_exchange', csv)
    return d, cdx, bm, bsc, labels, memb, postRank, keyfn, csv, a


def phase_fields(cdx, bm, csv=True):
    fc = field_completeness(cdx)
    emit(fc, '5_stage2_input_field_NaN_by_exchange', csv)
    s1 = stage1_field_completeness(bm)
    emit(s1, '6_stage1_criterion_NaN_by_exchange', csv)
    return fc, s1


def phase_stage1(bm, bsc, csv=True, n=8):
    nanw, total_w = stage1_nan_weight(bm, n=n)
    agg = nanw.groupby('grp').agg(
        n_sources=('source', 'size'),
        mean_nan_crit=('nan_crit', 'mean'),
        mean_nan_tier_w=('nan_tier_w', 'mean'),
        median_nan_tier_w=('nan_tier_w', 'median'),
        p90_nan_tier_w=('nan_tier_w', lambda s: s.quantile(0.90)),
        pct_any_nan_crit=('nan_crit', lambda s: 100.0 * (s > 0).mean()))
    agg['total_stage1_tier_weight'] = total_w
    agg['mean_nan_w_pct_of_total'] = 100.0 * agg['mean_nan_tier_w'] / total_w
    b = bsc.groupby('grp')['score'].agg(['mean', 'median',
                                         lambda s: s.quantile(0.95)])
    b.columns = ['BoScore_mean', 'BoScore_median', 'BoScore_p95']
    agg = agg.join(b)
    emit(agg, '7_stage1_NaN_penalty_and_BoScore_by_exchange', csv)
    return nanw, agg, total_w


def phase_price_sanity(cdx, csv=True):
    """Re-run the price-plausibility screen on the SURVIVING panel.

    The panel is already POST-filter (data_quality runs twice in Sbocker), so a nonzero
    detection here would mean the filter is not idempotent. What this measures instead is
    HOW CLOSE each exchange sits to the thresholds, i.e. whether a convention difference
    puts non-US names near a cliff.
    """
    df = cdx.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['source', 'date'])
    recs = []
    prev_row = {}
    prev_valid = {}
    for r in df.itertuples(index=False):
        row = {k: getattr(r, k) for k in ('price', 'marketCap', 'earningsYield',
                                          'dividendYield', 'weightedAverageShsOut')}
        src = r.source
        pp, pm, _pd = prev_valid.get(src, (None, None, None))
        prm, prs = prev_row.get(src, (None, None))
        ok, reason = dq.check_price_sanity(row, prev_price=pp, prev_mcap=pm,
                                          prev_row_mcap=prm, prev_row_shares=prs)
        prev_row[src] = (row['marketCap'], row['weightedAverageShsOut'])
        if ok and pd.notna(row['price']) and row['price'] > 0:
            prev_valid[src] = (row['price'], row['marketCap'], r.date)
        if not ok:
            recs.append({'grp': r.grp, 'source': src, 'date': r.date,
                         'reason': str(reason).split('(')[0].strip()})
    det = pd.DataFrame(recs)
    if len(det):
        emit(pd.crosstab(det['grp'], det['reason']),
             '8_price_sanity_detections_on_surviving_panel', csv)
    else:
        print('\n8_price_sanity_detections_on_surviving_panel: ZERO detections in every '
              'exchange group (the saved panel is post-filter and the filter is idempotent).')
    # proximity to the two cross-field cliffs
    prox = pd.DataFrame(index=sorted(cdx['grp'].unique()))
    ey = pd.to_numeric(cdx['earningsYield'], errors='coerce')
    dy = pd.to_numeric(cdx['dividendYield'], errors='coerce')
    prox['rows'] = cdx.groupby('grp').size()
    prox['pct_ey_ge_1'] = (ey >= 1.0).groupby(cdx['grp']).mean() * 100
    prox['pct_ey_ge_2'] = (ey >= 2.0).groupby(cdx['grp']).mean() * 100
    prox['pct_ey_na'] = ey.isna().groupby(cdx['grp']).mean() * 100
    prox['pct_dy_na'] = dy.isna().groupby(cdx['grp']).mean() * 100
    prox['pct_dy_gt0'] = (dy > 0).groupby(cdx['grp']).mean() * 100
    emit(prox, '9_price_screen_proximity_by_exchange', csv)
    return det, prox


def phase_stage2(cdx, bsc, memb, postRank, keyfn, csv=True):
    """THE CONSEQUENCE. Recompute Stage-2 on two pools and price the fill per exchange.

    POOL A -- the GENERAL POOL (4,287 issuers). This is the population Stage-1 ranks and
      cuts head(100) from, so it is the right pool for a per-exchange estimate with enough
      names per group. NOTE it is NOT the pool the live z-scores are computed on.
    POOL B -- the DEPLOYED top-100. This IS the live Stage-2 pool: mu, sigma and therefore
      the fill are computed on exactly these 100 names. Per-exchange n is tiny here, which
      is a property of the design, not of this measurement.
    """
    W = weight_vector()
    out = {}
    for label, srcs in (('A_general_pool', list(memb['general'])),
                        ('B_deployed_top100', list(postRank['source']))):
        bs_pool = bsc[bsc['source'].isin(srcs)].sort_values('score', ascending=False)
        cdx_pool = cdx[cdx['source'].isin(srcs)]
        print('\n### Stage-2 recompute on %s: %d names' % (label, len(bs_pool)))
        raw = stage2_metric_frame(cdx_pool, bs_pool)
        col_df, name_df, norm = fill_advantage(raw, W, pool_label=label)
        emit(col_df.set_index('column'), '10_fill_effect_per_column_%s' % label, csv)
        name_df['grp'] = name_df['source'].map(keyfn)
        # AggScore as the pipeline builds it: sum of z*w over the weighted columns
        zw = norm.copy()
        for c in [c for c in zw.columns if c != 'source']:
            zw[c] = pd.to_numeric(zw[c], errors='coerce') * float(W.get(c, 0) or 0)
        agg = pbr.getAggScore(zw.copy())
        agg['grp'] = agg['source'].map(keyfn)
        aggs = agg['AggScore'].sort_values(ascending=False).reset_index(drop=True)
        med = float(aggs.median())
        r20 = float(aggs.iloc[min(19, len(aggs) - 1)])
        dist = r20 - med
        print('  %s: median AggScore %.4f, rank-20 AggScore %.4f, median->rank20 '
              'distance %.4f' % (label, med, r20, dist))
        g = name_df.groupby('grp')
        res = pd.DataFrame({
            'n_names': g.size(),
            'pct_with_any_imputed': g['n_imputed_cols'].apply(lambda s: 100.0 * (s > 0).mean()),
            'mean_imputed_cols': g['n_imputed_cols'].mean(),
            'mean_imputed_weight_share': g['imputed_weight_share'].mean(),
            'mean_fill_advantage_agg': g['fill_advantage_aggscore'].mean(),
            'median_fill_advantage_agg': g['fill_advantage_aggscore'].median(),
            'max_fill_advantage_agg': g['fill_advantage_aggscore'].max(),
        })
        res['median_to_rank20_distance'] = dist
        res['mean_adv_pct_of_that_distance'] = 100.0 * res['mean_fill_advantage_agg'] / dist
        res['mean_AggScore'] = agg.groupby('grp')['AggScore'].mean()
        emit(res, '11_fill_advantage_by_exchange_%s' % label, csv)
        out[label] = (raw, col_df, name_df, norm, agg, res, dist)
    return out


def run():
    d, cdx, bm, bsc, labels, memb, postRank, keyfn, csv, a = main()
    phase_fields(cdx, bm, csv)
    phase_stage1(bm, bsc, csv)
    phase_price_sanity(cdx, csv)
    if not a.skip_stage2:
        phase_stage2(cdx, bsc, memb, postRank, keyfn, csv)


if __name__ == '__main__':
    run()


def phase_stage2_percol_by_exchange(raw, keyfn, weights, label, csv=True):
    """Which Stage-2 METRIC (not raw field) is imputed, per exchange. This is the table
    that answers the question the CEO actually asked: is there a metric that is simply not
    computable for a whole market?
    """
    W = dict(weights)
    wcols = [c for c in sw.METRIC_KEYS
             if c in raw.columns and float(W.get(c, 0) or 0) != 0.0]
    g = raw['source'].map(keyfn)
    rows = []
    for c in wcols:
        s = pd.to_numeric(raw[c], errors='coerce').isna()
        r = {'metric': c, 'weight': float(W[c])}
        for grp, sub in s.groupby(g):
            r['%s_imp_pct' % grp] = 100.0 * sub.mean()
        rows.append(r)
    out = pd.DataFrame(rows).set_index('metric')
    emit(out, '12_stage2_metric_imputation_pct_by_exchange_%s' % label, csv)
    return out


def phase_graham_reason(cdx, csv=True):
    """grahamNumber is the ONE sparse Stage-2 input. Is that an FMP gap or is the number
    genuinely UNDEFINED for the company? getData_fmp stamps the reason, so it can be read
    rather than assumed.
    """
    if 'grahamUndefinedReason' not in cdx.columns:
        print('\ngrahamUndefinedReason not present on this panel.')
        return None
    gn = pd.to_numeric(cdx['grahamNumber'], errors='coerce')
    sub = cdx[gn.isna()]
    ct = pd.crosstab(sub['grp'], sub['grahamUndefinedReason'].fillna('(no reason stamped)'))
    ct['TOTAL_missing_rows'] = ct.sum(axis=1)
    ct['rows_in_panel'] = cdx.groupby('grp').size()
    ct['pct_of_rows_missing'] = 100.0 * ct['TOTAL_missing_rows'] / ct['rows_in_panel']
    emit(ct, '13_grahamNumber_undefined_reason_by_exchange', csv)
    return ct


def phase_stage1_cut(bsc, memb, postRank, keyfn, csv=True):
    """Where the head(100) Stage-1 cut lands, and how far each exchange sits from it."""
    gen = bsc[bsc['source'].isin(list(memb['general']))].sort_values(
        'score', ascending=False).reset_index(drop=True)
    cut = float(gen['score'].iloc[99])
    gen['grp'] = gen['source'].map(keyfn)
    res = gen.groupby('grp')['score'].agg(
        n='size', mean='mean', p50='median',
        p90=lambda s: s.quantile(0.90),
        p99=lambda s: s.quantile(0.99), max='max')
    res['stage1_cut_score'] = cut
    res['n_at_or_above_cut'] = gen[gen['score'] >= cut].groupby('grp').size()
    res['n_at_or_above_cut'] = res['n_at_or_above_cut'].fillna(0)
    res['best_minus_cut'] = res['max'] - cut
    res['expected_at_base_rate'] = res['n'] * 100.0 / len(gen)
    emit(res, '14_stage1_top100_cut_by_exchange', csv)
    return res
