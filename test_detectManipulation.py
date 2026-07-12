"""
Offline known-answer tests for detectManipulation.py (Beneish M / Montier C).

SYNTHETIC fixtures, deterministic, NO network. Run:
    python test_detectManipulation.py

These assert the PUBLISHED Beneish/Montier expectation computed BY HAND -- NOT the
prior code behaviour. The root cause fixed here: the working frame is oldest-first,
but the forensic formulas + recency window (head(...)) had been written as if it were
newest-first, so (a) five of the eight M components computed the reciprocal (wrong)
direction and (b) the score summarized the OLDEST quarters. The fix normalizes each
per-symbol frame to newest-first locally and writes every component in its published
direction; head(...) then reads the MOST RECENT quarters.

Published component directions (neutral = 1.0; TATA neutral = 0.0):
  DSRI = (DSO/Sales)_t / _{t-1}          current/prior   (rising receivables -> >1)
  GMI  = GrossMargin_{t-1} / _t          PRIOR/current   (margin decline     -> >1)
  AQI  = AssetQuality_t / _{t-1}         current/prior   (asset quality down -> >1)
  SGI  = Sales_t / _{t-1}                current/prior   (sales growth       -> >1)
  DEPI = DepRate_{t-1} / _t              PRIOR/current   (dep-rate decline   -> >1)
  SGAI = (SGA/Sales)_t / _{t-1}          current/prior   (SG&A intensity up  -> >1)
  LVGI = Leverage_t / _{t-1}             current/prior   (leverage up        -> >1)
  TATA = (NI - CFO)/TA  (level; higher accruals -> more suspicious)
GMI and DEPI use prior/current; the other five use current/prior. A single global
orientation flip therefore CANNOT make all eight correct at once -- each component
must compute ITS OWN published direction, which is what is asserted below.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import detectManipulation as dm
import forensicFlags as ff
from detectManipulation import invrollsumTTM

TOL = 1e-9

# Baseline (clean) annual levels, one value per metric.
_BASE = dict(
    daysSalesOutstanding=100, revenue=1000, grossProfitMargin=0.40,
    totalCurrentAssets=300, propertyPlantEquipmentNet=300, totalAssets=1000,
    depreciationAndAmortization=100, sellingGeneralAndAdministrativeExpenses=100,
    longTermDebt=200, totalCurrentLiabilities=100, netIncome=40,
    netCashProvidedByOperatingActivities=40,
    netCashUsedProvidedByFinancingActivities=0,
    daysOfInventoryOutstanding=100, otherCurrentAssets=20,
    capexPerShare=0, weightedAverageShsOut=4,
)
# Deteriorated (dirty) annual levels -- every axis moves to its suspicious side.
_DIRTY = dict(_BASE)
_DIRTY.update(
    daysSalesOutstanding=300,       # DSO up faster than sales   -> DSRI up
    revenue=1100,                   # mild sales growth          -> SGI  up
    grossProfitMargin=0.20,         # margin halves              -> GMI  up (prior/cur)
    totalCurrentAssets=100,         # softer assets              -> AQI  up
    propertyPlantEquipmentNet=100,
    depreciationAndAmortization=20, # dep rate falls             -> DEPI up (prior/cur)
    sellingGeneralAndAdministrativeExpenses=220,  # SG&A intensity up -> SGAI up
    longTermDebt=400, totalCurrentLiabilities=200,  # leverage up -> LVGI up
    netIncome=300, netCashProvidedByOperatingActivities=50,  # NI>>CFO -> TATA up
    daysOfInventoryOutstanding=300,  # inventory days up   -> DSIinc fires
    otherCurrentAssets=200,          # other-CA/rev up      -> OCARinc fires
)


def _build(annual_by_year, symbol, oldest_first=True):
    """annual_by_year: dict metric -> list of ANNUAL levels, OLDEST year first.
    Expands to 4 quarters/year (quarterly = annual/4). Returns an oldest-first
    per-symbol frame (or newest-first if oldest_first=False, to test robustness)."""
    nyears = len(next(iter(annual_by_year.values())))
    nq = 4 * nyears
    dates = pd.date_range('2015-03-31', periods=nq, freq='QE').strftime('%Y-%m-%d').tolist()
    data = {}
    for metric, levels in annual_by_year.items():
        q = []
        for yr_level in levels:
            q += [yr_level / 4.0] * 4
        data[metric] = q
    df = pd.DataFrame(data)
    df['date'] = dates
    df['source'] = symbol
    if not oldest_first:
        df = df.iloc[::-1].reset_index(drop=True)
    return df


def _years(metric_dict_per_year):
    """Turn {metric: value} baseline/dirty dicts into per-metric year lists."""
    return metric_dict_per_year


def _annual(per_year_specs):
    """per_year_specs: list of dicts (one per year, oldest first). Return dict
    metric -> [year0, year1, ...]."""
    metrics = per_year_specs[0].keys()
    return {m: [spec[m] for spec in per_year_specs] for m in metrics}


def _run(annual, symbol, oldest_first=True):
    df = _build(annual, symbol, oldest_first=oldest_first)
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': [symbol]})}
    mdf, slm, pm = dm.calcBeneishM(resdic, [symbol])
    cdf, slc, pc = dm.calcMontierC(resdic, [symbol])
    return dict(mdf=mdf, m_mean=slm['M_Score_mean'].iloc[0], m_flagged=symbol in pm,
                cdf=cdf, c_mean=slc['C_Score_mean'].iloc[0], c_flagged=symbol in pc)


# ---------------------------------------------------------------------------

def test_component_directions_exact_row0():
    """Each M component, most-recent full-year YoY (stored row 0), equals its
    HAND-computed published value. Baseline flat for 5 years, dirty in year 6, so
    row0's current window (newest 4 q) and prior window (next 4 q) are each a clean
    single-year block -> exact ratios."""
    annual = _annual([_BASE]*5 + [_DIRTY])
    r = _run(annual, 'DIRTY')
    row0 = r['mdf'].iloc[0]

    # Hand values (dirty year vs baseline year):
    #  DSRI = (300/1100)/(100/1000) = 2.727272...
    #  GMI  = 0.40/0.20 = 2.0            (prior/current: margin fell)
    #  AQI  = (1-200/1000)/(1-600/1000) = 0.8/0.4 = 2.0
    #  SGI  = 1100/1000 = 1.1
    #  DEPI = [20/(20+100)] inverse: rate_prior/rate_cur = 0.25/(1/6) = 1.5
    #  SGAI = (220/1100)/(100/1000) = 0.2/0.1 = 2.0
    #  LVGI = (600/1000)/(300/1000) = 2.0
    #  TATA = (300-50-0)/1000 = 0.25
    expect = dict(DSRI=(300/1100)/(100/1000), GMI=2.0, AQI=2.0, SGI=1.1,
                  DEPI=0.25/(20/120), SGAI=2.0, LVGI=2.0, TATA=0.25)
    for k, v in expect.items():
        assert abs(float(row0[k]) - v) < TOL, (k, float(row0[k]), v)

    # Every published-direction check: the ratio indices sit ABOVE neutral 1.0 and
    # TATA above 0.0 -- i.e. each moved to its fraud-suspicious side.
    for k in ('DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI'):
        assert float(row0[k]) > 1.0, (k, float(row0[k]))
    assert float(row0['TATA']) > 0.0
    print("PASS test_component_directions_exact_row0")


def test_mscore_fold_and_flag_dirty():
    """Stored M row0 equals the -4.84..-0.327*LVGI standard M PLUS the 1.78 fold,
    so stored M>0 <=> standard M>-1.78. The deteriorating firm's recent-window mean
    exceeds 0 (flagged manipulator)."""
    annual = _annual([_BASE]*5 + [_DIRTY])
    r = _run(annual, 'DIRTY')
    row0 = r['mdf'].iloc[0]
    standard = (-4.84 + 0.92*row0.DSRI + 0.528*row0.GMI + 0.404*row0.AQI
                + 0.892*row0.SGI + 0.115*row0.DEPI - 0.172*row0.SGAI
                + 4.679*row0.TATA - 0.327*row0.LVGI)
    assert abs(float(row0['M_Score']) - (standard + 1.78)) < TOL, (row0['M_Score'], standard)
    # standard M > -1.78 <=> stored > 0
    assert (standard > -1.78) == (float(row0['M_Score']) > 0)
    assert standard > -1.78, standard          # this firm IS a manipulator on published M
    assert r['m_mean'] > 0 and r['m_flagged'], (r['m_mean'], r['m_flagged'])
    print("PASS test_mscore_fold_and_flag_dirty")


def test_cscore_flags_dirty():
    """Montier flags fire on the deteriorating firm (recent window): rising DSO,
    rising inventory, rising other-CA/rev, NI>>CFO divergence, falling dep-rate.
    C_Score_mean clears the >4 review cutoff."""
    annual = _annual([_BASE]*5 + [_DIRTY])
    r = _run(annual, 'DIRTY')
    crow0 = r['cdf'].iloc[0]
    for flag in ('NICFOdiv', 'DSOinc', 'DSIinc', 'OCARinc'):
        assert float(crow0[flag]) > 0, (flag, float(crow0[flag]))
    assert float(crow0['DAPPdec']) > 0                # dep-rate decline
    assert r['c_mean'] > 4 and r['c_flagged'], (r['c_mean'], r['c_flagged'])
    print("PASS test_cscore_flags_dirty")


def test_clean_firm_below_thresholds():
    """A perfectly flat firm: every M ratio = neutral 1.0, TATA = 0, M well below 0;
    C below the >4 cutoff (a small residual DAPP flag from the accumulated-depreciation
    proxy is expected and stays far below cutoff)."""
    annual = _annual([_BASE]*6)
    r = _run(annual, 'CLEAN')
    row0 = r['mdf'].iloc[0]
    for k in ('DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI'):
        assert abs(float(row0[k]) - 1.0) < TOL, (k, float(row0[k]))
    assert abs(float(row0['TATA'])) < TOL
    assert r['m_mean'] < 0 and not r['m_flagged'], (r['m_mean'], r['m_flagged'])
    assert r['c_mean'] < 4 and not r['c_flagged'], (r['c_mean'], r['c_flagged'])
    print("PASS test_clean_firm_below_thresholds")


def test_recency_window_reads_recent_not_oldest():
    """THE recency test. Two 8-year firms:
      - old-CLEAN / recent-DIRTY  -> must be FLAGGED (score reflects recent dirt)
      - old-DIRTY / recent-CLEAN  -> must be CLEAN   (old dirt no longer dominates)
    If head(...) still read the oldest quarters (the bug), these verdicts would be
    swapped."""
    oc_rd = _annual([_BASE]*7 + [_DIRTY])
    od_rc = _annual([_DIRTY] + [_BASE]*7)
    r1 = _run(oc_rd, 'OC_RD')
    r2 = _run(od_rc, 'OD_RC')
    assert r1['m_mean'] > 0 and r1['m_flagged'], ('old-clean/recent-dirty', r1['m_mean'])
    assert r1['c_mean'] > 4, ('old-clean/recent-dirty C', r1['c_mean'])
    assert r2['m_mean'] < 0 and not r2['m_flagged'], ('old-dirty/recent-clean', r2['m_mean'])
    assert r2['c_mean'] < 4, ('old-dirty/recent-clean C', r2['c_mean'])
    print("PASS test_recency_window_reads_recent_not_oldest")


def test_stored_frames_are_newest_first():
    """Contract relied on by forensicFlags (its head(M_WINDOW)/head(C_WINDOW) reads
    of mscore_df/cscore_df must land on the recent quarters): the stored per-symbol
    frames are newest-first (row 0 date is the latest)."""
    annual = _annual([_BASE]*5 + [_DIRTY])
    r = _run(annual, 'DIRTY')
    md = pd.to_datetime(r['mdf']['date'])
    cd = pd.to_datetime(r['cdf']['date'])
    assert (md.values[:-1] >= md.values[1:]).all(), "mscore_df not newest-first"
    assert (cd.values[:-1] >= cd.values[1:]).all(), "cscore_df not newest-first"
    print("PASS test_stored_frames_are_newest_first")


def test_orientation_robust_to_input_order():
    """The local sort makes the result independent of how cdx_df rows arrive:
    the same data fed oldest-first vs newest-first yields identical M/C means."""
    annual = _annual([_BASE]*5 + [_DIRTY])
    a = _run(annual, 'DIRTY', oldest_first=True)
    b = _run(annual, 'DIRTY', oldest_first=False)
    assert abs(a['m_mean'] - b['m_mean']) < TOL, (a['m_mean'], b['m_mean'])
    assert abs(a['c_mean'] - b['c_mean']) < TOL, (a['c_mean'], b['c_mean'])
    print("PASS test_orientation_robust_to_input_order")


def test_sloan_accruals_reads_recent_quarter():
    """forensicFlags.computeSloanAccruals must compute Sloan accruals on the MOST
    RECENT TTM window, not the oldest. Firm: 5 flat years with NI==CFO (zero
    accruals) then a newest year with big accruals (NI=400, CFO=0), TA flat at 1000.

    Hand values (quarterly = annual/4; TTM of a flat year = that year's annual):
      recent (correct):  NI_ttm=400, CFO_ttm=0, avg_TA=(250+250)/2=250 -> 400/250 = 1.6
      oldest  (the bug): NI_ttm=40,  CFO_ttm=40                        -> 0/250   = 0.0
    So the fixed value is 1.6 and the old iloc[0]-on-oldest read would have been 0.0."""
    annual = _annual([_BASE]*5 + [_BASE])   # start from baseline shape, then override the accrual axes
    annual['netIncome'] = [40, 40, 40, 40, 40, 400]
    annual['netCashProvidedByOperatingActivities'] = [40, 40, 40, 40, 40, 0]
    annual['totalAssets'] = [1000]*6
    df = _build(annual, 'SLOAN', oldest_first=True)   # oldest-first, the real frame orientation

    got = ff.computeSloanAccruals(df, ['SLOAN'])['sloanAccruals'].iloc[0]
    assert abs(got - 1.6) < TOL, ('fixed Sloan should reflect recent quarter', got)

    # Prove the OLD read (iloc[0] on the raw oldest-first frame) would have been wrong:
    raw = df[df['source'] == 'SLOAN']            # NOT normalized -> oldest at iloc[0]
    ni0 = invrollsumTTM(pd.to_numeric(raw['netIncome'], errors='coerce')).iloc[0]
    cfo0 = invrollsumTTM(pd.to_numeric(raw['netCashProvidedByOperatingActivities'], errors='coerce')).iloc[0]
    old_val = (ni0 - cfo0) / 250.0
    assert abs(old_val - 0.0) < TOL, ('old read lands on the clean OLDEST quarter', old_val)
    assert abs(got - old_val) > 1.0, ('fix must change the answer', got, old_val)

    # Orientation-robust: same data fed newest-first yields the same (recent) Sloan.
    df_nf = _build(annual, 'SLOAN', oldest_first=False)
    got_nf = ff.computeSloanAccruals(df_nf, ['SLOAN'])['sloanAccruals'].iloc[0]
    assert abs(got - got_nf) < TOL, (got, got_nf)
    print("PASS test_sloan_accruals_reads_recent_quarter")


if __name__ == '__main__':
    test_component_directions_exact_row0()
    test_mscore_fold_and_flag_dirty()
    test_cscore_flags_dirty()
    test_clean_firm_below_thresholds()
    test_recency_window_reads_recent_not_oldest()
    test_stored_frames_are_newest_first()
    test_orientation_robust_to_input_order()
    test_sloan_accruals_reads_recent_quarter()
    print("\nALL detectManipulation KNOWN-ANSWER TESTS PASSED")
