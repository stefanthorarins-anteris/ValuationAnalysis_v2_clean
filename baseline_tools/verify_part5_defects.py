"""READ-ONLY verification harness for metric-rationale.md Part 5 defects D1/D3/D6/D8/D10/D11.

Writes nothing into the pipeline; reads the saved resdic pickle + the saved panel and
re-derives every quantitative claim independently.  Analysis tool only.
"""
import os, sys, numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path: sys.path.insert(0, _p)
import stage2_metrics as sm, reporting_period as rp, createDicts as cdic

PK = os.path.join(_HERE, "resdic_2026-07-17_CORRECTED.pickle")
d = pd.read_pickle(PK)
pr, raw, cdx = d['postRank'], d['postScoreMetric_raw'], d['cdx_dftop100']
postBm, postNew = cdic.getPostDict()
W = {**{k: postBm[k]['w'] for k in postBm}, **{k: postNew[k]['w'] for k in postNew}}


def unw(series_or_val, metric):
    """Un-weight a postRank column (which holds z x w) back to z -- NaN, never inf, on w == 0.

    HARDENED 2026-08-01 (audit item 3).  Un-weighting divides by the metric's own weight, and
    four of the five divisions in this file were unguarded while the fifth guarded correctly
    (`c.min()/W[m] if W[m] else np.nan`).  All seven weights used here are non-zero TODAY, so
    nothing was broken -- but `DcfToPrice`, `BoScore` and `priceGrowth` are ALREADY w = 0.000
    in this very vector, so a zero weight is a live shape in this dict, not a hypothetical.
    A future re-weighting that zeroes one of the seven would have produced `inf` (Series) or
    `ZeroDivisionError` (scalar) instead of a clean NaN, and an `inf` propagating into a
    printed z-range is a silently wrong number rather than a visibly missing one.
    """
    w = float(W.get(metric, 0) or 0)
    if w == 0:
        return (series_or_val * np.nan if hasattr(series_or_val, '__len__')
                else float('nan'))
    return series_or_val / w

top20 = list(pr['source'].head(20))
print("== PROVENANCE ==")
print("pool n =", pr.shape[0], "| top20 =", top20)
print("cdx_dftop100 sources", cdx['source'].nunique(), "rows", len(cdx))
print("has period col:", 'period' in cdx.columns, "| has reportedCurrency:", 'reportedCurrency' in cdx.columns)

# newest-first re-sort exactly as the live scorer does
cdxs = cdx.copy(); cdxs['date'] = pd.to_datetime(cdxs['date'], errors='coerce')
cdxs = cdxs.sort_values(['source','date'], ascending=[True,False]).reset_index(drop=True)
fmap = rp.frequency_by_source(cdxs, verbose=False)
NQ = 16

# ---------------------------------------------------------------- fidelity check
rep = {}
for t in pr['source']:
    tc = cdxs.loc[cdxs['source'] == t]
    rpy = rp.rows_per_year(fmap, t)
    row = {}
    for k in postBm: row[k] = sm.postbm_metric(k, postBm[k]['eqMet'], tc, NQ, rpy=rpy)
    row['freeCashFlowYield'] = sm.free_cash_flow_yield(tc.freeCashFlow, tc.marketCap, NQ, rpy=rpy)
    row['freeCashFlowPerShareGrowth'] = sm.free_cash_flow_per_share_growth(tc.freeCashFlow, tc.weightedAverageShsOut, NQ, rpy=rpy)
    row['tbVpRatio'] = sm.tbv_p_ratio(tc, NQ, rpy=rpy)
    row['Altman-Z'] = sm.altman_z(tc, rpy=rpy)
    row['Piotroski'] = sm.piotroski(tc, rpy=rpy)
    row['EPStoEPSmean'] = sm.eps_to_eps_mean(tc, rpy=rpy)
    row['CycleHeat'] = sm.cycleheat(tc, rpy=rpy)
    row['_rpy'] = rpy
    rep[t] = row
R = pd.DataFrame(rep).T
rawi = raw.set_index('source')
print("\n== FIDELITY vs saved postScoreMetric_raw (max abs diff per column) ==")
for c in sorted(set(R.columns) & set(rawi.columns)):
    a = pd.to_numeric(R[c], errors='coerce'); b = pd.to_numeric(rawi[c].reindex(R.index), errors='coerce')
    both = a.notna() & b.notna()
    md = float((a[both]-b[both]).abs().max()) if both.any() else float('nan')
    print(f"  {c:30s} maxdiff={md:.3e}  nan_mismatch={int((a.isna()!=b.isna()).sum())}")

# ---------------------------------------------------------------- D3 coverage
print("\n== D3: per-metric coverage in the head(w) window [pool n=100] ==")
def win_series(t, kind):
    tc = cdxs.loc[cdxs['source'] == t]; rpy = rp.rows_per_year(fmap, t); w = rp.scale_window(NQ, rpy)
    if kind == 'grahamNumberToPrice': s = tc['grahamNumber']/tc['price']
    elif kind == 'bVpRatio':          s = 1/tc['pbRatio']
    elif kind == 'revenueGrowth':     s = tc['revenue'].pct_change(-int(rpy), fill_method=None)
    elif kind == 'freeCashFlowYield': s = tc.freeCashFlow/tc.marketCap
    elif kind == 'freeCashFlowPerShareGrowth':
        s = (tc.freeCashFlow/tc.weightedAverageShsOut).pct_change(-int(rpy), fill_method=None)
    elif kind == 'tbVpRatio':         s = tc['tangibleBookValuePerShare']/tc['price']
    elif kind == 'priceGrowth':       s = tc['price'].pct_change(-1, fill_method=None)
    else:                             s = pd.to_numeric(tc[postBm[kind]['eqMet']], errors='coerce')
    s = pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf], np.nan).head(w)
    return s, w
WINDOWED = ['grahamNumberToPrice','bVpRatio','revenueGrowth','RoA','earnYield','incomeQuality',
            'returnOnEquity','returnOnCapitalEmployed','currentRatio','grossProfitMargin',
            'freeCashFlowYield','freeCashFlowPerShareGrowth','tbVpRatio','priceGrowth']
cov = {}
for k in WINDOWED:
    rows = {}
    for t in pr['source']:
        s, w = win_series(t, k)
        rows[t] = {'k': int(s.notna().sum()), 'w': int(w), 'frac': (s.notna().sum()/w if w else np.nan),
                   'mean_computable': float(s.mean()) if s.notna().any() else np.nan,
                   'mean_zerofill': float(s.fillna(0).sum()/w) if w else np.nan}
    cov[k] = pd.DataFrame(rows).T
    c = cov[k]
    print("  %-30s w=%2d | full-coverage=%3d | <full=%3d | k<8=%3d | mink=%2d | meanfrac=%.3f | w=%.4f"
          % (k, int(c['w'].mode().iloc[0]), int((c['frac'] >= 1.0).sum()), int((c['frac'] < 1.0).sum()),
             int((c['k'] < 8).sum()), int(c['k'].min()), float(c['frac'].mean()), W.get(k, 0)))

# ---------------------------------------------------------------- D3 decomposition
print("\n== D3: decomposing grahamNumberToPrice coverage [pool n=100] ==")
rows = []
for t in pr['source']:
    tc = cdxs.loc[cdxs['source'] == t]
    rpy = rp.rows_per_year(fmap, t); w = rp.scale_window(NQ, rpy)
    s_all = pd.to_numeric(tc['grahamNumber']/tc['price'], errors='coerce').replace([np.inf,-np.inf], np.nan)
    s = s_all.head(w)
    gn_all = pd.to_numeric(tc['grahamNumber'], errors='coerce')
    eps = pd.to_numeric(tc['netIncomePerShare'], errors='coerce')
    rows.append(dict(source=t, rpy=rpy, w=w, n_rows=len(tc), k_win=int(s.notna().sum()),
                     k_full_panel=int(s_all.notna().sum()), n_panel=len(s_all),
                     gn_nan_full=int(gn_all.isna().sum()),
                     neg_eps_rows=int((eps <= 0).sum()),
                     mean_comp=float(s.mean()) if s.notna().any() else np.nan,
                     mean_zero=float(s.fillna(0).sum()/w),
                     in_top20=(t in top20)))
G = pd.DataFrame(rows).set_index('source')
print("names with n_rows < 16 (short panel):", int((G['n_rows'] < 16).sum()))
print("names with k_win < w (any NaN inside the window):", int((G['k_win'] < G['w']).sum()))
print("names with k_win < 16 (doc's literal wording, window basis):", int((G['k_win'] < 16).sum()))
print("names with k_win < 8:", int((G['k_win'] < 8).sum()), "| min k_win:", int(G['k_win'].min()))
print("names with k_full_panel < 16 (whole per-ticker panel basis):", int((G['k_full_panel'] < 16).sum()))
print("names with >=1 NaN grahamNumber anywhere in panel:", int((G['gn_nan_full'] > 0).sum()))
print("names with >=1 non-positive EPS row anywhere in panel:", int((G['neg_eps_rows'] > 0).sum()))
print("panel row-count distribution:", G['n_rows'].value_counts().to_dict())

# direction of the bias
sub = G[G['mean_comp'].notna()].copy()
sub['ratio'] = sub['mean_comp']/sub['mean_zero'].replace(0, np.nan)
from scipy import stats
part = sub[sub['k_win'] < sub['w']]
print("\n-- DIRECTION: is the NaN-skipping mean HIGHER (=cheaper) than a coverage-weighted one? --")
print("partial-coverage names n=%d: mean_comp median %.4f vs coverage-weighted median %.4f"
      % (len(part), part['mean_comp'].median(), part['mean_zero'].median()))
print("mean overstatement factor w/k on partial names: median %.3f max %.3f"
      % ((part['w']/part['k_win']).median(), (part['w']/part['k_win']).max()))
ok = sub[['k_win','mean_comp']].dropna()
rho, p = stats.spearmanr(ok['k_win'], ok['mean_comp'])
print("cross-sectional rho(coverage k, grahamNumberToPrice raw) = %.4f  p=%.4f  n=%d" % (rho, p, len(ok)))
covf = sub['k_win']/sub['w']
rho2, p2 = stats.spearmanr(covf, sub['mean_comp'])
print("cross-sectional rho(coverage FRACTION, raw metric)      = %.4f  p=%.4f" % (rho2, p2))

# ---------------------------------------------------------------- D3 alt definitions for "40"
print("\n== D3: hunting the document's '40 of 100' figure under alternative definitions ==")
bm = d['BoM_dftop100']
gcol = [c for c in bm.columns if 'raham' in c]
print("BoMetric_df graham-ish columns:", gcol)
for c in gcol:
    cnt16, cntpanel = 0, 0
    for t in pr['source']:
        s = pd.to_numeric(bm.loc[bm['source'] == t, c], errors='coerce').replace([np.inf,-np.inf], np.nan)
        if int(s.head(16).notna().sum()) < 16: cnt16 += 1
        if int(s.notna().sum()) < 16: cntpanel += 1
    print("  BoMetric_df[%s]: <16 computable in head(16)=%d ; <16 in whole panel=%d" % (c, cnt16, cntpanel))
gn_only16 = sum(1 for t in pr['source'] if int(pd.to_numeric(
    cdxs.loc[cdxs['source'] == t, 'grahamNumber'], errors='coerce').head(16).notna().sum()) < 16)
print("  cdx grahamNumber alone, <16 non-NaN in head(16):", gn_only16)
gnp_pos = sum(1 for t in pr['source'] if int((pd.to_numeric(
    cdxs.loc[cdxs['source'] == t, 'grahamNumber'], errors='coerce').head(16) > 0).sum()) < 16)
print("  cdx grahamNumber > 0, <16 in head(16):", gnp_pos)

# ---------------------------------------------------------------- D3 genuine NaN-inside-available
print("\n== D3: NaN INSIDE AVAILABLE ROWS (true adverse-selection exposure, not short panel) ==")
for k in WINDOWED:
    n_sel, worst = 0, 0.0
    for t in pr['source']:
        s, w = win_series(t, k)
        avail = len(s)                      # rows actually present in the window
        kk = int(s.notna().sum())
        if kk < avail:
            n_sel += 1
            worst = max(worst, (avail - kk)/avail)
    print("  %-30s names with NaN inside available window rows = %3d | worst missing frac %.2f | w=%.4f"
          % (k, n_sel, worst, W.get(k, 0)))

# ---------------------------------------------------------------- D3 counterfactual through deployed code
print("\n== D3: counterfactual re-score through the DEPLOYED normalise/weight/aggregate path ==")
import postBoRank as pbr
def rescore(raw_frame):
    f = raw_frame.copy()
    normed, _ = pbr.normalizeAndDropNA(f.copy(), weight_series=W)
    wt = normed.drop('source', axis=1).copy()
    for col in wt.columns: wt[col] = normed[col].values * W.get(col, 1)
    out = pd.concat([normed[normed.columns.difference(wt.columns)], wt], axis=1)
    return pbr.getAggScore(out)
import io, contextlib
base_raw = rawi.reindex(pr['source']).reset_index()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    base = rescore(base_raw)
print("baseline reproduction top20 == saved top20:", list(base['source'].head(20)) == top20)

alt_cw = base_raw.copy()
alt_cw['grahamNumberToPrice'] = [G.loc[s, 'mean_zero'] for s in alt_cw['source']]
alt_mc = base_raw.copy()
MINCOV = 12
alt_mc['grahamNumberToPrice'] = [G.loc[s, 'mean_comp'] if G.loc[s, 'k_win'] >= MINCOV else np.nan
                                 for s in alt_mc['source']]
for lbl, fr in (('coverage-weighted (NaN->0)', alt_cw), ('min-coverage 12/16 -> NaN', alt_mc)):
    with contextlib.redirect_stdout(buf):
        alt = rescore(fr)
    a = base.set_index('source')['AggScore']; b = alt.set_index('source')['AggScore']
    dz = (b - a).reindex(a.index)
    t20a, t20b = list(base['source'].head(20)), list(alt['source'].head(20))
    print("\n  [%s]" % lbl)
    print("   max |dAggScore| = %.4f ; median |d| = %.4f" % (dz.abs().max(), dz.abs().median()))
    print("   top-20 membership changes: IN %s | OUT %s" % (sorted(set(t20b)-set(t20a)), sorted(set(t20a)-set(t20b))))
    rk_a = base.reset_index().set_index('source').index.get_indexer_for
    ra = {s: i+1 for i, s in enumerate(base['source'])}; rb = {s: i+1 for i, s in enumerate(alt['source'])}
    mv = sorted(((abs(rb[s]-ra[s]), s, ra[s], rb[s]) for s in ra), reverse=True)[:8]
    print("   biggest rank moves (|d|, name, old, new):", [(m[1], m[2], m[3]) for m in mv])
    aff = [s for s in top20 if G.loc[s, 'k_win'] < G.loc[s, 'w']]
    print("   TOP-20 names with partial graham coverage: "
          + ", ".join("%s k=%d/%d dAgg=%+.4f rank %d->%d" % (s, G.loc[s,'k_win'], G.loc[s,'w'], dz[s], ra[s], rb[s]) for s in aff))

# ---------------------------------------------------------------- D3 extra detail
print("\n== D3: extra detail ==")
nsa = sum(1 for t in pr['source'] if rp.rows_per_year(fmap, t) == 2)
print("semi-annual-classified names in pool (w=8):", nsa, "| quarterly:", 100-nsa)
print("of the 27 NaN-inside-window names, semi-annual:",
      sum(1 for t in pr['source'] if G.loc[t,'k_win'] < G.loc[t,'w'] and rp.rows_per_year(fmap,t) == 2))
print("\nWHERE the NaN sits (window position) for the 3 growth-type metrics:")
for k in ('revenueGrowth', 'freeCashFlowPerShareGrowth'):
    tail_only, mixed = 0, 0
    for t in pr['source']:
        s, w = win_series(t, k)
        if s.notna().sum() < len(s):
            nanpos = list(np.where(s.isna().to_numpy())[0])
            if min(nanpos) >= len(s) - len(nanpos): tail_only += 1
            else: mixed += 1
    print("   %-28s NaNs only at the OLD end of the window: %d ; interior/other: %d" % (k, tail_only, mixed))
print("\nIMPP detail: raw graham/price mean over its 6 computable q = %.4f ; pool median = %.4f ; pool max = %.4f"
      % (G.loc['IMPP','mean_comp'], G['mean_comp'].median(), G['mean_comp'].max()))
print("IMPP z on grahamNumberToPrice (from saved postRank / w):",
      unw(float(pr.loc[pr['source']=='IMPP','grahamNumberToPrice'].iloc[0]), 'grahamNumberToPrice'))

# ---------------------------------------------------------------- D11
print("\n== D11: marketCapRevQuants pool-relativity magnitude [pool n=100] ==")
mq = pd.to_numeric(raw.set_index('source')['marketCapRevQuants'], errors='coerce')
print("raw value counts:", mq.value_counts().sort_index().to_dict())
wcol = pd.to_numeric(pr.set_index('source')['marketCapRevQuants'], errors='coerce')
zcol = unw(wcol, 'marketCapRevQuants')
print("z range: %.4f .. %.4f (span %.4f)" % (zcol.min(), zcol.max(), zcol.max()-zcol.min()))
print("WEIGHTED contribution range: %.4f .. %.4f (span %.4f of AggScore)"
      % (wcol.min(), wcol.max(), wcol.max()-wcol.min()))
step = (wcol.max()-wcol.min())/3
print("one-quartile step = %.4f of AggScore" % step)
med, t20d = pr['AggScore'].median(), pr['AggScore'].iloc[19]
print("pool median AggScore %.4f ; rank-20 AggScore %.4f ; median->top20 distance %.4f" % (med, t20d, t20d-med))
print("one-quartile step as a fraction of the median->top20 distance: %.2f" % (step/(t20d-med)))
# churn simulation: drop k names at random from the pool, re-cut the quartiles
usd = sm._mcap_for_quants(cdxs.drop_duplicates('source').set_index('source').loc[pr['source']].reset_index())
usd.index = pr['source']
rng = np.random.default_rng(7)
flips = []
for _ in range(200):
    keep = pr['source'].sample(frac=0.85, random_state=int(rng.integers(1e6)))
    sub = usd.loc[keep]
    codes_sub = pd.qcut(sub, 4, duplicates='drop').cat.codes
    vals_sub = (-1)*((codes_sub/3)-0.5)
    base_vals = mq.loc[keep]
    d = (vals_sub - base_vals).abs()
    flips.append(float((d > 1e-9).mean()))
print("15%%-of-pool composition change: fraction of SURVIVING names whose size score MOVES = %.3f (mean over 200 draws)"
      % np.mean(flips))

# ---------------------------------------------------------------- D11b: what population is qcut cut over?
print("\n== D11b: the qcut population is ROWS, not NAMES ==")
mqrows = sm.add_mcap_quants(cdxs)
print("cdxtop rows passed to add_mcap_quants:", len(cdxs), "| unique sources:", cdxs['source'].nunique())
print("row-level mcapQuants value counts:", mqrows.value_counts().sort_index().to_dict())
newest = cdxs.groupby('source').head(1).index
print("per-NAME (newest row) value counts:", mqrows.loc[newest].value_counts().sort_index().to_dict())
rows_per_name = cdxs.groupby('source').size()
print("rows per name: min %d max %d -- a long-history name contributes %.1fx as many observations "
      "to the quartile edges as the shortest" % (rows_per_name.min(), rows_per_name.max(),
                                                 rows_per_name.max()/rows_per_name.min()))
# faithful churn simulation on the DEPLOYED row-level basis
flips2 = []
for _ in range(200):
    keep = set(pr['source'].sample(frac=0.85, random_state=int(rng.integers(1e6))))
    sub = cdxs[cdxs['source'].isin(keep)]
    v = sm.add_mcap_quants(sub)
    newv = v.loc[sub.groupby('source').head(1).index]
    newv.index = sub.groupby('source').head(1)['source'].values
    oldv = mq.loc[newv.index]
    flips2.append(float(((newv - oldv).abs() > 1e-9).mean()))
print("DEPLOYED-basis churn: 15%% pool turnover -> %.3f of surviving names change size score" % np.mean(flips2))

# ---------------------------------------------------------------- D10
print("\n== D10: capitalExpenditureCoverageRatio fillna(0) [panel] ==")
_D = pd.read_pickle(PK)
full = _D['cdx_df']
print("cdx_df (universe panel) rows:", len(full), "sources:", full['source'].nunique())
c = pd.to_numeric(full['capitalExpenditureCoverageRatio'], errors='coerce')
n_nan = int(c.isna().sum())
print("NaN capitalExpenditureCoverageRatio rows: %d of %d (%.2f%%)" % (n_nan, len(c), 100*n_nan/len(c)))
print("sources with >=1 NaN: %d of %d" % (int(full.loc[c.isna(),'source'].nunique()), full['source'].nunique()))
allnan = full.groupby('source')['capitalExpenditureCoverageRatio'].apply(
    lambda s: pd.to_numeric(s, errors='coerce').isna().all())
print("sources where it is NaN on EVERY row: %d" % int(allnan.sum()))
# what the special metric yields, and the pass rate
tested = -c.fillna(0) - 2
tested_computable = -c.dropna() - 2
print("pass rate (Sign=+1, value>0) on ALL rows incl. filled: %.4f" % float((tested > 0).mean()))
print("pass rate on COMPUTABLE rows only:                     %.4f" % float((tested_computable > 0).mean()))
print("filled-NaN rows land at -2 -> pass = %s (so every missing row is a FAIL)"
      % bool((-0.0 - 2) > 0))
print("Tier of capitalExpenditureCoverageRatio in Stage-1:", cdic.getDicts()[6]['capitalExpenditureCoverageRatio'])
sp = cdic.getDicts()[6]
print("dict_special keys:", list(sp.keys()), "| 'EPStoEPSmean' present:", 'EPStoEPSmean' in sp)

print("\n== D10 extra: exposure in the Stage-1 head(8) SCORING window, and the x==0 limb ==")
bmfull = _D['BoMetric_df']
col = 'capitalExpenditureCoverageRatio'
print("BoMetric_df rows:", len(bmfull), "sources:", bmfull['source'].nunique(), "| has col:", col in bmfull.columns)
cc = pd.to_numeric(full[col], errors='coerce')
print("rows where the RAW value is exactly 0.0 (capex-free -> also tested as -2 -> FAIL): %d (%.3f%%)"
      % (int((cc == 0).sum()), 100*float((cc == 0).mean())))
g = full.copy(); g['_v'] = cc
g['_dt'] = pd.to_datetime(g['date'], errors='coerce')
g = g.sort_values(['source','_dt'], ascending=[True, False])
win = g.groupby('source').head(8)
per = win.groupby('source')['_v'].agg(['size', lambda s: int(s.isna().sum())])
per.columns = ['n','n_nan']
print("sources with >=1 NaN inside the head(8) Stage-1 window: %d of %d (%.2f%%)"
      % (int((per['n_nan'] > 0).sum()), len(per), 100*float((per['n_nan'] > 0).mean())))
print("sources with ALL 8 window rows NaN (max damage: loses the whole w=0.3): %d" % int((per['n_nan'] == per['n']).sum()))
print("worst-case Stage-1 score loss from this defect = %.3f of the 18.65 gate total (%.2f%%)"
      % (0.3, 100*0.3/18.65))

print("\n== D10 corrected basis: BoMetric_df (the frame Stage-1 actually scores) ==")
bb = bmfull.copy()
bb['_v'] = pd.to_numeric(bb[col], errors='coerce')
bb['_dt'] = pd.to_datetime(bb['date'], errors='coerce')
bb = bb.sort_values(['source','_dt'], ascending=[True, False])
print("BoMetric_df NaN rows in this column: %d of %d (%.2f%%)" % (int(bb['_v'].isna().sum()), len(bb), 100*float(bb['_v'].isna().mean())))
w8 = bb.groupby('source').head(8)
p8 = w8.groupby('source')['_v'].agg(n='size', n_nan=lambda s: int(s.isna().sum()))
print("sources with >=1 NaN in head(8): %d ; all-NaN head(8): %d" % (int((p8['n_nan']>0).sum()), int((p8['n_nan']==p8['n']).sum())))
print("NOTE: BoMetric_df stores the ALREADY-TRANSFORMED special value (%s), so check what fillna produced:" % col)
print("  value == -2.0 exactly (i.e. the filled/zero case): %d rows (%.2f%%)"
      % (int((bb['_v'] == -2.0).sum()), 100*float((bb['_v'] == -2.0).mean())))
w8v = w8.groupby('source')['_v'].apply(lambda s: int((s == -2.0).sum()))
print("  sources with >=1 '-2.0' row in head(8): %d of %d (%.1f%%)" % (int((w8v>0).sum()), len(w8v), 100*float((w8v>0).mean())))
print("  sources where ALL 8 window rows are exactly -2.0: %d" % int((w8v>=8).sum()))
# is the raw 0 a real zero or a missing-as-zero?
z = full.loc[pd.to_numeric(full[col], errors='coerce') == 0]
for cand in ('netCashProvidedByOperatingActivities','capexPerShare','freeCashFlow'):
    if cand in z.columns:
        v = pd.to_numeric(z[cand], errors='coerce')
        print("  among raw==0 rows: %-40s zero=%.1f%% nan=%.1f%% median=%s"
              % (cand, 100*float((v == 0).mean()), 100*float(v.isna().mean()), v.median()))

# ---------------------------------------------------------------- D6
print("\n== D6: leverage / interest coverage in Stage-2 ==")
print("Stage-2 metric keys (postBm):", list(postBm.keys()))
print("Stage-2 metric keys (postNew):", list(postNew.keys()))
LEVTOK = ('debt','Debt','lever','Lever','interest','Interest','coverage','Coverage','gearing','EBITDA','solvenc')
print("Stage-2 keys matching any leverage/interest token:",
      [k for k in list(postBm)+list(postNew) if any(t in k for t in LEVTOK)])
print("nonzero-weight Stage-2 metrics:", sorted([k for k,v in W.items() if v != 0]))
print("zero-weight:", sorted([k for k,v in W.items() if v == 0]))
# Altman term decomposition (what leverage content DOES reach Stage-2)
terms = {}
for t in pr['source']:
    tc = cdxs.loc[cdxs['source'] == t]; rpy = rp.rows_per_year(fmap, t); n = int(rpy)
    try:
        cur = tc.iloc[0]; ta = cur['totalAssets']; tl = cur['totalLiabilities']
        if not (ta > 0 and tl > 0) or len(tc) < n: continue
        oi = pd.to_numeric(tc['operatingIncome'], errors='coerce').head(n)
        rv = pd.to_numeric(tc['revenue'], errors='coerce').head(n)
        if oi.isna().any() or rv.isna().any(): continue
        terms[t] = dict(t1=1.2*(cur['totalCurrentAssets']-cur['totalCurrentLiabilities'])/ta,
                        t2=1.4*cur['totalStockholdersEquity']/ta, t3=3.3*oi.sum()/ta,
                        t4=0.6*cur['marketCap']/tl, t5=1.0*rv.sum()/ta)
    except Exception: pass
T = pd.DataFrame(terms).T
absmean = T.abs().mean()
print("\nAltman-Z mean |contribution| share [pool n=%d]: %s"
      % (len(T), {k: round(100*absmean[k]/absmean.sum(),1) for k in absmean.index}))
zsum = T.sum(axis=1)
print("corr(0.6*x4, Z) = %.4f ; corr(1.4*x2 [equity/TA, the LEVERAGE limb], Z) = %.4f"
      % (T['t4'].corr(zsum), T['t2'].corr(zsum)))
print("leverage-bearing share of Z's mean |contribution| (t2 + t4) = %.1f%% ; of which t4 is price-bearing = %.1f%%"
      % (100*(absmean['t2']+absmean['t4'])/absmean.sum(), 100*absmean['t4']/absmean.sum()))
# what the top-20 actually looks like on leverage
print("\nTOP-20 leverage profile [pool] (newest row):")
lev = []
for t in top20:
    tc = cdxs.loc[cdxs['source'] == t]; cur = tc.iloc[0]
    de = pd.to_numeric(tc['debtEquityRatio'], errors='coerce').head(16).mean()
    nd = pd.to_numeric(tc['netDebtToEBITDA'], errors='coerce').head(16).mean()
    tle = cur['totalLiabilities']/cur['totalStockholdersEquity'] if cur['totalStockholdersEquity'] else np.nan
    ltd = cur['longTermDebt']/cur['totalAssets'] if cur['totalAssets'] else np.nan
    lev.append(dict(source=t, DE_16qmean=de, netDebtToEBITDA_16qmean=nd, TL_over_equity=tle, LTD_over_TA=ltd))
L = pd.DataFrame(lev).set_index('source')
print(L.round(3).to_string())
uni_de = pd.to_numeric(full['debtEquityRatio'], errors='coerce')
print("\nuniverse [panel] debtEquityRatio median %.3f ; top-20 median of 16q-mean %.3f"
      % (uni_de.median(), L['DE_16qmean'].median()))
print("top-20 names with 16q-mean D/E > 1.0: %d of 20 ; > 2.0: %d ; netDebtToEBITDA > 3: %d"
      % (int((L['DE_16qmean']>1).sum()), int((L['DE_16qmean']>2).sum()), int((L['netDebtToEBITDA_16qmean']>3).sum())))

# ---------------------------------------------------------------- D1
print("\n== D1: DCF call count per run ==")
print("DcfToPrice weight in decisional vector:", W['DcfToPrice'])
sl = _D.get('carveout_sidelists', {})
print("general pool scored: 100 names -> 100 DCF calls")
tot = 100
for lab, r in (sl or {}).items():
    n = 0 if r is None else len(r['postRank'])
    print("  cohort side-list %-22s scored %d names -> %d DCF calls" % (lab, n, n))
    tot += n
print("TOTAL per-ticker DCF GETs per run on this pool =", tot,
      "(one v3/discounted-cash-flow GET per scored name per pool; postBoScoreRanking is called once per pool)")
print("VA_OFFLINE_NO_DCF default:", repr(os.environ.get('VA_OFFLINE_NO_DCF', '<unset>')), "(set by this harness's parent? see run_corrected_current)")

print("\n== D6b: how the levered names cleared Stage-1's uNetDebtToEBITDA (Tier A, w=0.75, Sign -1) ==")
u = bmfull.copy(); u['_dt'] = pd.to_datetime(u['date'], errors='coerce')
u = u.sort_values(['source','_dt'], ascending=[True, False])
for t in ['DLNG','TRTN-PA','PBYI','GSL','HAFN','GASS']:
    s = pd.to_numeric(u.loc[u['source']==t,'uNetDebtToEBITDA'], errors='coerce').head(8)
    passes = [(-1)*v > 0 for v in s]           # Sign=-1, testvec = metric - 1
    print("  %-9s head(8) uNetDebtToEBITDA=%s -> passes %d/8 -> earns %.3f of w=0.75"
          % (t, [None if pd.isna(v) else round(v,2) for v in s], sum(bool(p) for p in passes),
             0.75*np.mean([1.0 if p else 0.0 for p in passes])))
print("  (Stage-1 is a weighted PASS-RATE sum over 18.65, not a gate: failing this criterion outright costs 0.75/18.65 = %.2f%%)"
      % (100*0.75/18.65))

print("\n== D8: psbrfilter inertness ==")
pf = _D['psbrfilter']
print("psbrfilter type:", type(pf).__name__, "shape:", getattr(pf,'shape',None))
print("names it would EXCLUDE from the 100:", sorted(set(pr['source']) - set(pf['source'])))
print("i.e. it passes %d of %d names on this pool" % (len(pf), len(pr)))
print("NOTE the filter runs on postRank whose columns are z*w, so the -1.5 cutoff is applied to a WEIGHTED z:")
for m in ['earnYield','grahamNumberToPrice','RoA','EPStoEPSmean','freeCashFlowYield','revenueGrowth']:
    c = pd.to_numeric(pr[m], errors='coerce')
    print("   %-22s w=%.4f  min(z*w)=%.4f  min(z)=%.4f  -> would -1.5 on z*w ever bind? %s ; on z? %s"
          % (m, W[m], c.min(), unw(c.min(), m), c.min() < -1.5, unw(c.min(), m) < -1.5))

print("\n== D8b: what the gate would do on the UN-weighted z scale (its evident intent) ==")
zz = pr.set_index('source')[['earnYield','grahamNumberToPrice','RoA','EPStoEPSmean','freeCashFlowYield','revenueGrowth']].apply(pd.to_numeric, errors='coerce')
for c in zz.columns: zz[c] = unw(zz[c], c)
excl = zz[(zz < -1.5).any(axis=1)]
print("names excluded if the -1.5 cutoff were applied to z (not z*w): %d of 100" % len(excl))
print("  ", sorted(excl.index.tolist()))
print("of the deployed TOP-20, would be excluded:", [s for s in top20 if s in set(excl.index)])

print("\n== D3 final: sign of the correction across all partial-coverage names ==")
with contextlib.redirect_stdout(buf):
    alt = rescore(alt_cw)
a = base.set_index('source')['AggScore']; b = alt.set_index('source')['AggScore']
dz = (b - a)
partial = [t for t in pr['source'] if G.loc[t,'k_win'] < G.loc[t,'w']]
print("partial-coverage names n=%d ; dAggScore NEGATIVE (i.e. they were INFLATED) on %d of them"
      % (len(partial), sum(1 for t in partial if dz[t] < 0)))
print("full-coverage names n=%d ; dAggScore POSITIVE (relative gain) on %d"
      % (100-len(partial), sum(1 for t in pr['source'] if t not in partial and dz[t] > 0)))
gz = unw(pd.to_numeric(pr.set_index('source')['grahamNumberToPrice'], errors='coerce'), 'grahamNumberToPrice')
print("names whose grahamNumberToPrice z is exactly +-3.0 (winsorizer bound): %d -> %s"
      % (int((gz.abs().round(6) == 3.0).sum()), list(gz[gz.abs().round(6) == 3.0].index)))
print("of those, partial-coverage:", [t for t in gz[gz.abs().round(6)==3.0].index if t in partial])
