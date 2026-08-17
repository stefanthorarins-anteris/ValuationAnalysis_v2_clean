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
  DSRI = DSO_t / DSO_{t-1}               current/prior   (rising receivables -> >1)
         (daysSalesOutstanding IS ALREADY AR/Sales; dividing by Sales again -- which the
          code did until the 2026-07-19 audit fix -- made DSRI = DSRI_true / SGI)
  GMI  = GrossMargin_{t-1} / _t          PRIOR/current   (margin decline     -> >1)
  AQI  = AssetQuality_t / _{t-1}         current/prior   (asset quality down -> >1)
  SGI  = Sales_t / _{t-1}                current/prior   (sales growth       -> >1)
  DEPI = DepRate_{t-1} / _t              PRIOR/current   (dep-rate decline   -> >1)
  SGAI = (SGA/Sales)_t / _{t-1}          current/prior   (SG&A intensity up  -> >1)
  LVGI = Leverage_t / _{t-1}             current/prior   (leverage up        -> >1)
  TATA = (NI_ttm - CFO_ttm)/TotalAssets  (TA as a LEVEL, not a 4-quarter SUM, and with
         NO financing-cash-flow term -- both were audit fixes on 2026-07-19; higher
         accruals -> more suspicious)
GMI and DEPI use prior/current; the other five use current/prior. A single global
orientation flip therefore CANNOT make all eight correct at once -- each component
must compute ITS OWN published direction, which is what is asserted below.
"""
import inspect
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import detectManipulation as dm
import nan_policy as npol
import forensicFlags as ff
import reporting_period as rp
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
    #  DSRI = 300/100 = 3.0              (published: DSO_t/DSO_{t-1}; daysSalesOutstanding
    #                                     is already AR/Sales, so there is no second
    #                                     division by Sales -- the old (300/1100)/(100/1000)
    #                                     = 2.727 was DSRI_true/SGI)
    #  GMI  = 0.40/0.20 = 2.0            (prior/current: margin fell)
    #  AQI  = (1-200/1000)/(1-600/1000) = 0.8/0.4 = 2.0
    #  SGI  = 1100/1000 = 1.1
    #  DEPI = [20/(20+100)] inverse: rate_prior/rate_cur = 0.25/(1/6) = 1.5
    #  SGAI = (220/1100)/(100/1000) = 0.2/0.1 = 2.0
    #  LVGI = (600/1000)/(300/1000) = 2.0
    #  TATA = (300-50)/250 = 1.0         (NI_ttm - CFO_ttm over the TA LEVEL. NOTE: this
    #                                     harness expands every annual level to annual/4
    #                                     per quarter, INCLUDING balance-sheet stocks, so
    #                                     here the TA level is 1000/4 = 250 and the old
    #                                     4-quarter SUM coincidentally equalled the true
    #                                     annual level -- which is precisely why this
    #                                     harness could not see the 1/4-scale defect. See
    #                                     test_tata_uses_asset_LEVEL_not_4q_sum for the
    #                                     realistic-stock case. No -CFF term any more.)
    expect = dict(DSRI=300/100, GMI=2.0, AQI=2.0, SGI=1.1,
                  DEPI=0.25/(20/120), SGAI=2.0, LVGI=2.0, TATA=(300-50)/250)
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
    # DAPPdec is GONE (deleted 2026-07-26 -- positive by construction), so the deteriorating
    # firm now scores at most 5 and the cutoff is >= 4 of 5.
    assert r['c_mean'] >= dm.C_FLAG_CUTOFF and r['c_flagged'], (r['c_mean'], r['c_flagged'])
    print("PASS test_cscore_flags_dirty")


def test_clean_firm_below_thresholds():
    """A perfectly flat firm: every M ratio = neutral 1.0, TATA = 0, M well below 0;
    C below the cutoff (the DAPP flag that used to add a residual here was deleted
    2026-07-26)."""
    annual = _annual([_BASE]*6)
    r = _run(annual, 'CLEAN')
    row0 = r['mdf'].iloc[0]
    for k in ('DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI'):
        assert abs(float(row0[k]) - 1.0) < TOL, (k, float(row0[k]))
    assert abs(float(row0['TATA'])) < TOL
    assert r['m_mean'] < 0 and not r['m_flagged'], (r['m_mean'], r['m_flagged'])
    assert r['c_mean'] < dm.C_FLAG_CUTOFF and not r['c_flagged'], (r['c_mean'], r['c_flagged'])
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
    # >= the shared cutoff, not a literal 4: with DAPPdec deleted (2026-07-26) the
    # deteriorating firm tops out at 5 flags and lands exactly ON the cutoff.
    assert r1['c_mean'] >= dm.C_FLAG_CUTOFF, ('old-clean/recent-dirty C', r1['c_mean'])
    assert r2['m_mean'] < 0 and not r2['m_flagged'], ('old-dirty/recent-clean', r2['m_mean'])
    assert r2['c_mean'] < dm.C_FLAG_CUTOFF, ('old-dirty/recent-clean C', r2['c_mean'])
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



def test_tata_uses_asset_LEVEL_not_4q_sum():
    """TATA's denominator must be the TotalAssets LEVEL, not invrollsumTTM(TotalAssets).

    The main harness expands annual levels to annual/4 per quarter for EVERY metric
    including balance-sheet stocks, so there the 4-quarter sum happens to equal the true
    annual level and the 1/4-scale defect is invisible.  Real FMP data carries the FULL
    asset level in every quarter, so the sum is ~4x the level.  This fixture builds
    totalAssets that way (constant 1000 per quarter) and pins TATA to the published scale.

    Hand values: NI_ttm = 4*100 = 400, CFO_ttm = 4*25 = 100, TA level = 1000
      published TATA = (400 - 100)/1000                 = 0.30
      old (4q-sum denominator, minus CFF) = (400-100-4*50)/4000 = 0.025  -- 12x too small
    """
    nq = 12
    dates = pd.date_range('2015-03-31', periods=nq, freq='QE').strftime('%Y-%m-%d').tolist()
    df = pd.DataFrame({
        'date': dates, 'source': 'LEVELS',
        'netIncome': [100.0]*nq,
        'netCashProvidedByOperatingActivities': [25.0]*nq,
        'netCashUsedProvidedByFinancingActivities': [50.0]*nq,
        'totalAssets': [1000.0]*nq,            # a STOCK: full level every quarter
        'totalCurrentAssets': [300.0]*nq, 'propertyPlantEquipmentNet': [300.0]*nq,
        'daysSalesOutstanding': [100.0]*nq, 'revenue': [250.0]*nq,
        'grossProfitMargin': [0.4]*nq, 'depreciationAndAmortization': [25.0]*nq,
        'sellingGeneralAndAdministrativeExpenses': [25.0]*nq,
        'longTermDebt': [200.0]*nq, 'totalCurrentLiabilities': [100.0]*nq,
        'daysOfInventoryOutstanding': [100.0]*nq, 'otherCurrentAssets': [20.0]*nq,
        'capexPerShare': [0.0]*nq, 'weightedAverageShsOut': [4.0]*nq,
    })
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': ['LEVELS']})}
    mdf, _slm, _pm = dm.calcBeneishM(resdic, ['LEVELS'])
    got = float(mdf.iloc[0]['TATA'])
    assert abs(got - 0.30) < TOL, ('TATA must use the TA LEVEL', got)
    old = (400 - 100 - 4*50) / 4000.0
    assert abs(got - old) > 0.2, ('fix must change the answer materially', got, old)
    print("PASS test_tata_uses_asset_LEVEL_not_4q_sum")


def test_dsri_is_not_contaminated_by_sales_growth():
    """DSRI must be the published DSO ratio, so it is INDEPENDENT of sales growth.

    Two firms with the SAME receivables-days path (100 -> 300) but different sales growth
    (flat vs +10%) must get the SAME DSRI.  Under the old DSO/Sales form the growing firm
    got DSRI / 1.1, i.e. sales growth partially cancelled the receivables red flag even
    though SGI enters the M-score with the other large positive coefficient."""
    flat = _annual([_BASE]*5 + [dict(_DIRTY, revenue=_BASE['revenue'])])
    grow = _annual([_BASE]*5 + [dict(_DIRTY, revenue=1100)])
    d_flat = float(_run(flat, 'FLAT')['mdf'].iloc[0]['DSRI'])
    d_grow = float(_run(grow, 'GROW')['mdf'].iloc[0]['DSRI'])
    assert abs(d_flat - 3.0) < TOL, d_flat
    assert abs(d_grow - 3.0) < TOL, d_grow
    assert abs(d_flat - d_grow) < TOL, ('DSRI must not depend on sales growth',
                                        d_flat, d_grow)
    print("PASS test_dsri_is_not_contaminated_by_sales_growth")


def test_c_cutoff_is_one_constant_everywhere():
    """ONE C-score cutoff for the whole pipeline.  detectManipulation's problemlist used a
    strict `> 4` while forensicFlags/the presentation used `>= 4`, so the two shipped
    columns contradicted each other for every name scoring exactly 4.0 (12 of 90 on
    2026-07-17).  A C_Score_mean of exactly the cutoff must now flag on BOTH sides."""
    assert ff.C_FLAG_CUTOFF == dm.C_FLAG_CUTOFF, (ff.C_FLAG_CUTOFF, dm.C_FLAG_CUTOFF)
    assert list(ff.C_FLAGS) == list(dm.C_FLAG_COLS)
    cut = dm.C_FLAG_CUTOFF

    # detectManipulation's own problemlist must use that SAME cutoff, inclusively: for every
    # finite score, membership == (score >= cut).  Under the old strict `> 4` a score of
    # exactly 4.0 was excluded here while forensicFlags flagged it, and both columns shipped.
    fixtures = {'CLEAN': _annual([_BASE]*6),
                'DIRTY': _annual([_BASE]*5 + [_DIRTY]),
                'OD_RC': _annual([_DIRTY] + [_BASE]*7)}
    for sym, annual in fixtures.items():
        df = _build(annual, sym, oldest_first=True)
        resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': [sym]})}
        _cdf, slc, pc = dm.calcMontierC(resdic, [sym])
        score = float(slc['C_Score_mean'].iloc[0])
        assert (sym in pc) == (score >= cut), (sym, score, cut, sym in pc)

        # forensicFlags derives its flag from the SAME constant, so the two must agree.
        assert (score >= ff.C_FLAG_CUTOFF) == (score >= cut), (sym, score)
    print("PASS test_c_cutoff_is_one_constant_everywhere")


def test_missing_component_does_not_fire_a_c_flag():
    """Absent data must NOT manufacture a Montier red flag.

    Every C flag used to end in `.fillna(99999)`, so a period whose YoY change could not be
    COMPUTED counted as a maximal INCREASE -- the suspicious side.  Here a clean flat firm
    has its inventory-days field entirely absent (a services company with no inventory
    line): DSIinc must be NaN and must NOT fire, and the C_Score must DROP relative to the
    old fill behaviour rather than rise."""
    annual = _annual([_BASE]*6)
    df = _build(annual, 'NOINV', oldest_first=True)
    df['daysOfInventoryOutstanding'] = np.nan
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': ['NOINV']})}
    cdf, slc, pc = dm.calcMontierC(resdic, ['NOINV'])
    row0 = cdf.iloc[0]
    assert pd.isna(row0['DSIinc']), ('non-computable flag must stay NaN', row0['DSIinc'])
    assert not (float(row0['DSIinc']) > 0 if pd.notna(row0['DSIinc']) else False)
    # and the score counts only the flags that could actually be evaluated
    fired = [c for c in dm.C_FLAG_COLS
             if pd.notna(row0[c]) and float(row0[c]) > 0]
    assert float(row0['C_Score']) == len(fired), (row0['C_Score'], fired)
    assert 'DSIinc' not in fired
    print("PASS test_missing_component_does_not_fire_a_c_flag")



def _build_semiannual(annual_by_year, symbol):
    """Same fixture shape as _build but SEMI-ANNUAL: 2 rows per year, each carrying HALF
    the annual level (a real 6-month flow), dated at ~183-day spacing so the cadence
    classifier sees a semi-annual reporter."""
    nyears = len(next(iter(annual_by_year.values())))
    dates, data = [], {m: [] for m in annual_by_year}
    for yi in range(nyears):
        for half in (0, 1):
            dates.append('%d-%s' % (2015 + yi, '06-30' if half == 0 else '12-31'))
            for m, levels in annual_by_year.items():
                data[m].append(levels[yi] / 2.0)
    df = pd.DataFrame(data)
    df['date'] = dates
    df['source'] = symbol
    return df


def test_semiannual_windows_are_period_aware():
    """A SEMI-ANNUAL fixture must be classified as such and scored on 2-row windows.

    Built so the ANNUAL economics are identical to the quarterly _BASE/_DIRTY fixture
    (each half carries half the annual level), so the Beneish components should land on
    the SAME published values as the quarterly case -- which is the whole point: the same
    company reporting twice a year instead of four times must not score differently.
    """
    annual = _annual([_BASE] * 5 + [_DIRTY])
    df = _build_semiannual(annual, 'SEMI')

    assert rp.classify_from_cadence(df['date']) == rp.SEMIANNUAL, 'cadence must say semiannual'
    fmap = rp.frequency_by_source(df)
    assert fmap['SEMI'] == rp.SEMIANNUAL
    assert rp.rows_per_year(fmap, 'SEMI') == 2

    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': ['SEMI']})}
    mdf, slm, pm = dm.calcBeneishM(resdic, ['SEMI'], fmap)
    row0 = mdf.iloc[0]

    # Same hand values as the quarterly test: 2 half-rows == 1 year of flow.
    expect = dict(DSRI=300 / 100, GMI=2.0, AQI=2.0, SGI=1.1,
                  DEPI=0.25 / (20 / 120), SGAI=2.0, LVGI=2.0)
    for k, v in expect.items():
        assert abs(float(row0[k]) - v) < TOL, ('semiannual ' + k, float(row0[k]), v)
    assert float(row0['TATA']) > 0.0

    # And the WRONG (fixed-4-row) path must differ -- otherwise the test proves nothing.
    mdf4, _s4, _p4 = dm.calcBeneishM(resdic, ['SEMI'], {'SEMI': rp.QUARTERLY})
    assert abs(float(mdf4.iloc[0]['SGI']) - 1.1) > TOL,         'a 4-row window on semi-annual data must NOT reproduce the 1-year SGI'
    print("PASS test_semiannual_windows_are_period_aware")


def test_quarterly_path_is_bit_identical_under_rpy4():
    """The branch must be a NO-OP for quarterly data: explicit rpy=4 == the default."""
    annual = _annual([_BASE] * 5 + [_DIRTY])
    df = _build(annual, 'Q', oldest_first=True)
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': ['Q']})}
    a_m, a_s, _ = dm.calcBeneishM(resdic, ['Q'])                          # default map
    b_m, b_s, _ = dm.calcBeneishM(resdic, ['Q'], {'Q': rp.QUARTERLY})     # explicit 4
    for c in ('DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 'M_Score'):
        x = pd.to_numeric(a_m[c], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(b_m[c], errors='coerce').to_numpy(dtype=float)
        assert np.array_equal(x, y, equal_nan=True), ('quarterly not bit-identical', c)
    a_c, a_cs, _ = dm.calcMontierC(resdic, ['Q'])
    b_c, b_cs, _ = dm.calcMontierC(resdic, ['Q'], {'Q': rp.QUARTERLY})
    assert np.array_equal(pd.to_numeric(a_c['C_Score']).to_numpy(float),
                          pd.to_numeric(b_c['C_Score']).to_numpy(float), equal_nan=True)
    print("PASS test_quarterly_path_is_bit_identical_under_rpy4")



def test_dappdec_is_gone_and_cannot_come_back():
    """PINS the 2026-07-26 ruling: Montier is scored over FIVE flags and DAPPdec appears
    NOWHERE in the emitted forensic frame.

    It is a test and not just a comment because the deletion site is exactly where a future
    reader will "helpfully" restore it -- DAPPdec was positive BY CONSTRUCTION (a
    since-panel-start cumsum in its denominator), fired on 0.666 of names against
    0.271-0.472 for the other five, and carried 13 of the 16 C>=4 flags on the 2026-07-17
    top-100.  Restoring it must break a test.
    """
    assert len(dm.C_FLAG_COLS) == 5, ('Montier must be scored over 5 flags',
                                      dm.C_FLAG_COLS)
    assert 'DAPPdec' not in dm.C_FLAG_COLS
    assert dm.C_FLAG_CUTOFF == 4, 'cutoff holds at 4 (of 5) -- NOT rescaled to 3'
    assert list(ff.C_FLAGS) == list(dm.C_FLAG_COLS), 'consumers must share the constant'

    annual = _annual([_BASE] * 5 + [_DIRTY])
    r = _run(annual, 'DIRTY')
    cdf = r['cdf']
    assert 'DAPPdec' not in cdf.columns, ('DAPPdec must not be emitted at all',
                                          list(cdf.columns))
    # C_Score can never exceed the number of scored flags
    cs = pd.to_numeric(cdf['C_Score'], errors='coerce').dropna()
    assert (cs <= 5).all(), ('C_Score exceeded 5', cs.max())
    assert (cs >= 0).all()
    # and the whole emitted frame is free of the string, not just the score
    for col in cdf.columns:
        assert 'DAPP' not in str(col)
    print("PASS test_dappdec_is_gone_and_cannot_come_back")


def test_beneish_depi_is_deliberately_left_alone():
    """The Beneish DEPI term makes the SAME net-PP&E substitution and is deliberately NOT
    harmonised with the C-score deletion: it carries the smallest coefficient (0.115) inside
    a continuous weighted sum, which is tolerable, where DAPPdec was one binary vote of six
    against a hard cutoff, which was not.  This test exists so a future "consistency" pass
    does not delete DEPI too."""
    annual = _annual([_BASE] * 5 + [_DIRTY])
    r = _run(annual, 'DIRTY')
    assert 'DEPI' in r['mdf'].columns, 'Beneish DEPI must still be computed'
    assert pd.to_numeric(r['mdf']['DEPI'], errors='coerce').notna().any()
    print("PASS test_beneish_depi_is_deliberately_left_alone")


#  ---- THE BENEISH BASE GUARDS (undefended-denominator fix, 2026-08-14) ------------------
#  Three independent rules, each with its own failure it catches -- see the block at
#  `detectManipulation.BENEISH_AQI_SHARE_FLOOR` for the panel measurements behind each:
#    (a) a MAGNITUDE FLOOR on aqiTTM / depiTTM  -- the base is too small to be a measurement;
#    (b) an IDENTITY guard on aqiTTM only       -- the base is outside [0,1], i.e. impossible;
#    (c) a COVERAGE gate on the M window        -- too few periods survived to be a year.
#  aqiTTM and depiTTM carry SEPARATE floors, each derived from its own density.

def test_a_vanishing_aqi_base_ABSTAINS_instead_of_emitting_a_huge_index():
    """(a) AQI must REFUSE a near-zero asset-quality base, not divide by it.

    This is the WSE case from the shipped 2026-08-13 run reduced to a fixture: the base (the
    share of total assets that is neither current assets nor net PP&E) is 0.2% of the balance
    sheet in the prior year, so `AQI = a_t / a_{t-1}` is ~200 and the +0.404 coefficient alone
    carries M ~80 points past a cutoff of 0.  The published index is calibrated on a population
    centred at ~1.04; a value two orders of magnitude outside it is not evidence, it is a
    division.

    ASSERTED IN BOTH DIRECTIONS, because "it is NaN now" is worth nothing on its own -- the
    same NaN would appear if the fixture were simply broken.  With the guard DISABLED the
    fixture must produce the huge index; with it enabled the component must abstain.
    """
    prior = dict(_BASE, totalCurrentAssets=698, propertyPlantEquipmentNet=300)
    annual = _annual([_BASE] * 4 + [prior, _BASE])
    saved = dm.BENEISH_AQI_SHARE_FLOOR
    try:
        dm.BENEISH_AQI_SHARE_FLOOR = 0.0                 # pre-fix behaviour
        raw = pd.to_numeric(_run(annual, 'VANISH')['mdf']['AQI'], errors='coerce').iloc[0]
        assert raw > 100, ('fixture must reproduce the defect it guards against; AQI was %r'
                           % raw)
        dm.BENEISH_AQI_SHARE_FLOOR = 0.01
        r = _run(annual, 'VANISH')
        got = pd.to_numeric(r['mdf']['AQI'], errors='coerce').iloc[0]
        assert pd.isna(got), ('a base of 0.002 is below the 0.01 floor, so AQI must ABSTAIN; '
                              'got %r' % got)
        assert pd.isna(pd.to_numeric(r['mdf']['M_Score'], errors='coerce').iloc[0])
        assert raw * 0.404 > 40, raw
    finally:
        dm.BENEISH_AQI_SHARE_FLOOR = saved
    print("PASS test_a_vanishing_aqi_base_ABSTAINS_instead_of_emitting_a_huge_index")


def test_an_IMPOSSIBLE_aqi_base_is_refused_even_though_it_is_LARGE():
    """(b) The limb a magnitude floor cannot reach, and the one that actually bounds the term.

    `totalCurrentAssets + propertyPlantEquipmentNet <= totalAssets` is an identity, so an
    aqiTTM outside [0,1] is arithmetically impossible -- the panel carries 160 such rows
    (0.291%), reaching -14,550.37.  Those rows are not SMALL, they are WRONG, so every
    magnitude floor from 0.001 to 0.02 leaves them untouched: measured, the largest single-row
    |0.404 x AQI| contribution to M is 97,671,954 with no guard, still 136,644 with the floor
    alone, and 36.32 only when the identity guard is added.

    An earlier version of the block claimed aqiTTM was "bounded in [0,1] by construction".
    This test is the standing refutation of that.
    """
    #  totalCurrentAssets + netPP&E = 1400 against totalAssets 1000 -> aqi = -0.4, impossible
    bad = dict(_BASE, totalCurrentAssets=1100, propertyPlantEquipmentNet=300)
    annual = _annual([_BASE] * 4 + [bad, _BASE])
    saved = dm.BENEISH_AQI_SHARE_FLOOR
    try:
        dm.BENEISH_AQI_SHARE_FLOOR = 0.01
        got = pd.to_numeric(_run(annual, 'IMPOSS')['mdf']['AQI'], errors='coerce').iloc[0]
        assert pd.isna(got), ('an impossible base must be refused; got %r' % got)
        #  ...and it is NOT the floor doing it: |−0.4| is far ABOVE any floor tested.
        dm.BENEISH_AQI_SHARE_FLOOR = 0.0
        still = pd.to_numeric(_run(annual, 'IMPOSS')['mdf']['AQI'], errors='coerce').iloc[0]
        assert pd.isna(still), ('the identity guard must fire independently of the floor; '
                                'got %r' % still)
    finally:
        dm.BENEISH_AQI_SHARE_FLOOR = saved
    print("PASS test_an_IMPOSSIBLE_aqi_base_is_refused_even_though_it_is_LARGE")


def test_depi_carries_its_OWN_floor_and_does_not_inherit_AQI_s():
    """(b2) The two bases are different quantities with different degenerate populations.

    depiTTM's density shows a sharp isolated spike of 1,797 rows in [0, 0.002) against 340 in
    the next bin, so its floor is 0.002; aqiTTM's excess ends at 0.010.  Sharing one constant
    cost 4,320 refused DEPI rows (8.0% of the column) while changing ZERO verdicts on the
    shipped shortlist.  Pinned so the two cannot be silently re-merged into one symbol."""
    assert dm.BENEISH_AQI_SHARE_FLOOR == 0.01
    assert dm.BENEISH_DEPI_SHARE_FLOOR == 0.002
    assert dm.BENEISH_DEPI_SHARE_FLOOR < dm.BENEISH_AQI_SHARE_FLOOR
    #  a depi base of 0.005 is below AQI's floor but ABOVE its own -> it must survive
    v, n = dm._floor_share_base(pd.Series([0.005, 0.5]), dm.BENEISH_DEPI_SHARE_FLOOR)
    assert n == 0 and v.notna().all(), v.tolist()
    v, n = dm._floor_share_base(pd.Series([0.005, 0.5]), dm.BENEISH_AQI_SHARE_FLOOR)
    assert n == 1, n
    print("PASS test_depi_carries_its_OWN_floor_and_does_not_inherit_AQI_s")


def test_a_partially_refused_M_window_ABSTAINS_on_COVERAGE():
    """(c) THE DEFECT THE GUARD ITSELF CREATED, now closed (review F-1).

    `M_Score_mean` is `M_Score.head(scale_window(4, rpy)).mean()`, and pandas' `mean()` SKIPS
    NaN -- so refusing SOME rows of the window silently shortens it while the column keeps its
    trailing-year label.  On the shipped top-100 that converted HWX.TO (RANK 5) from a
    manipulation flag (+0.9618) to a clean -1.3159 on a mean over ONE of its four quarters.

    An earlier version of this file PINNED that behaviour as acceptable, on the stated ground
    that applying `nan_policy.COVERAGE_MIN` "would also change the three names the floor never
    touched".  It would not: that test is a STRICT `<` and the three sit at exactly 0.50.  The
    deferral was wrong and this test replaces it.
    """
    prior = dict(_BASE, totalCurrentAssets=698, propertyPlantEquipmentNet=300)
    annual = _annual([_BASE] * 4 + [prior, _BASE])       # refuses 1 of the 4 window rows
    saved = dm.BENEISH_AQI_SHARE_FLOOR
    try:
        dm.BENEISH_AQI_SHARE_FLOOR = 0.01
        r = _run(annual, 'PARTIAL')
        m = pd.to_numeric(r['mdf']['M_Score'], errors='coerce').head(4)
    finally:
        dm.BENEISH_AQI_SHARE_FLOOR = saved
    n_ok, n = int(m.notna().sum()), len(m)
    assert 0 < n_ok < n, m.tolist()          # genuinely partial, or the test is vacuous
    if n_ok / float(n) < npol.COVERAGE_MIN:
        assert pd.isna(r['m_mean']), (
            'a window below COVERAGE_MIN must abstain, not publish a short mean: %r'
            % r['m_mean'])
    else:
        assert not pd.isna(r['m_mean'])
        assert abs(float(r['m_mean']) - float(m.dropna().mean())) < TOL
    #  and the rule is the REPO'S rule, not a second copy of it
    src = inspect.getsource(dm.calcBeneishM)
    assert 'COVERAGE_MIN' in src and 'npol' in src


def test_the_coverage_gate_uses_a_STRICT_comparison_like_nan_policy():
    """Exactly-at-threshold must PASS, matching `nan_policy.py`'s own strict `<`.

    This is the fact the earlier deferral got wrong, so it is pinned rather than trusted: on
    the shipped top-100 the three semi-annual names LOUP.PA, NEDAP.AS and MAU.PA sit at exactly
    0.50 coverage, and it is precisely because 0.50 passes that this change touches ONE name
    instead of four."""
    src = inspect.getsource(dm.calcBeneishM)
    assert '>= npol.COVERAGE_MIN' in src, (
        'the M-window coverage test must ADMIT exactly-at-threshold, as nan_policy does')
    assert npol.COVERAGE_MIN == 0.50


def test_the_floors_are_read_at_CALL_time_not_bound_at_import():
    """Both constants must be overridable by assignment, the way every other constant in this
    repo is overridden for a measurement.

    Bound as a default argument a constant is evaluated ONCE at import, so the override
    silently does nothing and a before/after measurement reports "no change" for a reason that
    has nothing to do with the data.  That happened while this fix was being measured."""
    s = pd.Series([0.5, 0.004, np.nan, -0.002, 0.02])
    v, n = dm._floor_share_base(s, 0.0)
    assert n == 0 and int(v.notna().sum()) == 4, (n, v.tolist())
    v, n = dm._floor_share_base(s, 0.01)
    assert n == 2, n                       # 0.004 and -0.002
    assert v.isna().tolist() == [False, True, True, True, False], v.tolist()
    #  a row that was ALREADY NaN is not counted as a refusal by these rules
    assert dm._floor_share_base(pd.Series([np.nan, np.nan]), 0.01)[1] == 0
    #  the call sites read the module constants by NAME, so assignment reaches them
    src = inspect.getsource(dm.calcBeneishM)
    assert 'BENEISH_AQI_SHARE_FLOOR' in src and 'BENEISH_DEPI_SHARE_FLOOR' in src
    print("PASS test_the_floors_are_read_at_CALL_time_not_bound_at_import")


def test_the_floor_is_SYMMETRIC_and_guards_the_numerator_leg_too():
    """A vanishing base in the NUMERATOR gives an index near 0 -- bounded, so it looks
    harmless -- but it is the same non-measurement, and 0 is the index's BEST (least
    suspicious) side.  The 2026-07-26 ruling is that an uncomputable component must never score
    as the favourable one, so the guard is applied to the SERIES, not to the divisor."""
    current = dict(_BASE, totalCurrentAssets=698, propertyPlantEquipmentNet=300)
    annual = _annual([_BASE] * 5 + [current])       # newest year has the 0.002 base
    saved = dm.BENEISH_AQI_SHARE_FLOOR
    try:
        dm.BENEISH_AQI_SHARE_FLOOR = 0.0
        raw = pd.to_numeric(_run(annual, 'NUMER')['mdf']['AQI'], errors='coerce').iloc[0]
        assert 0 < raw < 0.02, ('fixture must put the vanishing base on the NUMERATOR; '
                                'AQI was %r' % raw)
        dm.BENEISH_AQI_SHARE_FLOOR = 0.01
        got = pd.to_numeric(_run(annual, 'NUMER')['mdf']['AQI'], errors='coerce').iloc[0]
        assert pd.isna(got), ('a near-zero numerator must abstain, not score as the best '
                              'possible asset quality; got %r' % got)
    finally:
        dm.BENEISH_AQI_SHARE_FLOOR = saved
    print("PASS test_the_floor_is_SYMMETRIC_and_guards_the_numerator_leg_too")


def test_the_guards_do_NOT_touch_an_ordinary_firm():
    """The guards must be inert on a normal balance sheet -- the baseline fixture's asset-
    quality base is 0.40 of total assets and its depreciation rate 0.25, both far above their
    floors and inside [0,1].  Without this, a floor set too high would pass every test above by
    refusing everything."""
    annual = _annual([_BASE] * 5 + [_DIRTY])
    sa, sd = dm.BENEISH_AQI_SHARE_FLOOR, dm.BENEISH_DEPI_SHARE_FLOOR
    try:
        dm.BENEISH_AQI_SHARE_FLOOR = dm.BENEISH_DEPI_SHARE_FLOOR = 0.0
        a = _run(annual, 'ORD')['mdf']
        dm.BENEISH_AQI_SHARE_FLOOR, dm.BENEISH_DEPI_SHARE_FLOOR = sa, sd
        b = _run(annual, 'ORD')['mdf']
    finally:
        dm.BENEISH_AQI_SHARE_FLOOR, dm.BENEISH_DEPI_SHARE_FLOOR = sa, sd
    for col in ('AQI', 'DEPI', 'M_Score'):
        x = pd.to_numeric(a[col], errors='coerce').to_numpy(dtype='float64')
        y = pd.to_numeric(b[col], errors='coerce').to_numpy(dtype='float64')
        assert np.allclose(x, y, equal_nan=True, rtol=0, atol=0), (col, x, y)
    print("PASS test_the_guards_do_NOT_touch_an_ordinary_firm")


# ---------------------------------------------------------------------------
#  O-13: the other five indices (DSRI, GMI, SGI, SGAI, LVGI), 2026-08-15.
#  These fixtures are PER QUARTER, not per year: the whole point of the guard is that ONE
#  unreported period inside a trailing-year window poisons the base, which an annual fixture
#  (four identical quarters) cannot express.
# ---------------------------------------------------------------------------

_Q = {k: (v / 4.0) for k, v in _BASE.items()}     # the per-QUARTER baseline `_build` implies


def _build_quarters(rows, symbol):
    """rows: per-QUARTER dicts, OLDEST first.  Returns an oldest-first per-symbol frame."""
    dates = pd.date_range('2015-03-31', periods=len(rows),
                          freq='QE').strftime('%Y-%m-%d').tolist()
    df = pd.DataFrame(list(rows))
    df['date'] = dates
    df['source'] = symbol
    return df


def _runq(rows, symbol='Q'):
    df = _build_quarters(rows, symbol)
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': [symbol]})}
    mdf, slm, pm = dm.calcBeneishM(resdic, [symbol])
    return dict(mdf=mdf.reset_index(drop=True), m_mean=slm['M_Score_mean'].iloc[0],
                m_flagged=symbol in pm)


def _col(mdf, name):
    return pd.to_numeric(mdf[name], errors='coerce')


class _domain_off(object):
    """Neutralise one or more period domains for the OFF arm of a two-sided assertion.

    The domains are read at CALL time (see the test of that below), so this is the same
    override mechanism `BENEISH_AQI_SHARE_FLOOR` uses -- and it is what lets every test here
    assert the defect REPRODUCES before asserting the guard removes it.
    """

    def __init__(self, *fields):
        self.fields = fields

    def __enter__(self):
        self.saved = dict(dm.PERIOD_DOMAIN)
        d = dict(self.saved)
        for f in (self.fields or list(d)):
            d[f] = (-np.inf, np.inf)
        dm.PERIOD_DOMAIN = d
        return self

    def __exit__(self, *a):
        dm.PERIOD_DOMAIN = self.saved
        return False


def test_a_SENTINEL_zero_DSO_period_is_not_summed_into_the_trailing_year_base():
    """DSRI: the CFG / 0HYP.L case (one issuer, two listings) reduced to a fixture.

    Its `daysSalesOutstanding` reads 0, 0, 5922.68, 0, 0, 0, 30.5453, 0 across eight quarters,
    so the "base" DSRI divided was a single spike surrounded by periods FMP could not compute,
    smeared over four windows by `invrollsumTTM`.  DSRI came out at 194 and carried M to +132.
    A magnitude floor cannot reach this: the base is 5,922, not small.
    """
    rows = [dict(_Q, daysSalesOutstanding=0.0) for _ in range(12)]
    rows[6]['daysSalesOutstanding'] = 30.0        # the prior-year window's only reported period
    rows[10]['daysSalesOutstanding'] = 5922.0     # the current window's
    with _domain_off('daysSalesOutstanding'):
        off = _col(_runq(rows, 'SENT')['mdf'], 'DSRI').iloc[0]
    assert off > 100, ('fixture must reproduce the defect it guards against; DSRI was %r' % off)
    on = _runq(rows, 'SENT')['mdf']
    assert pd.isna(_col(on, 'DSRI').iloc[0]), _col(on, 'DSRI').iloc[0]
    assert pd.isna(_col(on, 'M_Score').iloc[0])
    print("PASS test_a_SENTINEL_zero_DSO_period_is_not_summed_into_the_trailing_year_base")


def test_GMI_is_refused_on_its_DOMAIN_and_a_magnitude_floor_would_be_the_WRONG_instrument():
    """GMI: the BLDP case, and the reason the brief's "floor the base" prescription fails here.

    Two rows, one fixture:
      * the SIGN-CHANGE row -- prior-year margins negative, current-year margins summing to a
        hair above zero.  GMI is a large NEGATIVE number, and because the coefficient is
        POSITIVE (+0.528) it SUBTRACTS hundreds of points from M: the index that exists to
        catch deteriorating margins scores an IMPROVING one as maximally clean.
      * the BOTH-NEGATIVE row -- margins negative in both years, base |1.4| and |0.3|, i.e.
        FAR above any magnitude floor anyone would write.  GMI is a plausible-looking +4.96,
        read as a 5x margin deterioration, while the margin actually IMPROVED from -36.9% to
        -7.4%.  This is the row that proves a floor is the wrong instrument: no floor refuses
        it, and it is wrong anyway.
    """
    #  the SIGN-CHANGE fixture: prior year 4 x -0.3458, current year summing to +0.0025
    sign = [dict(_Q, grossProfitMargin=-0.3458) for _ in range(8)]
    sign += [dict(_Q, grossProfitMargin=-0.10), dict(_Q, grossProfitMargin=-0.10),
             dict(_Q, grossProfitMargin=-0.10), dict(_Q, grossProfitMargin=+0.3025)]
    #  the BOTH-NEGATIVE fixture: -34.58% -> -7.43% a quarter, i.e. the margin IMPROVED
    bothneg = [dict(_Q, grossProfitMargin=-0.3458) for _ in range(8)]
    bothneg += [dict(_Q, grossProfitMargin=-0.0743) for _ in range(4)]
    with _domain_off('grossProfitMargin'):
        gmi_sign = _col(_runq(sign, 'SIGNX')['mdf'], 'GMI').iloc[0]
        gmi_bothneg = _col(_runq(bothneg, 'BOTHNEG')['mdf'], 'GMI').iloc[0]
    assert gmi_sign < -100, gmi_sign
    assert 0.528 * gmi_sign < -50, gmi_sign          # it CLEARS the name, by a lot
    assert 3.0 < gmi_bothneg < 8.0, gmi_bothneg      # plausible-looking, and inverted
    assert pd.isna(_col(_runq(sign, 'SIGNX')['mdf'], 'GMI').iloc[0])
    assert pd.isna(_col(_runq(bothneg, 'BOTHNEG')['mdf'], 'GMI').iloc[0])
    print("PASS test_GMI_is_refused_on_its_DOMAIN_and_a_magnitude_floor_would_be_the_WRONG_"
          "instrument")


def test_a_gross_margin_of_exactly_1_is_the_MISSING_COST_line_and_must_not_score_as_NEUTRAL():
    """GMI: 2,497 panel rows report grossProfitMargin == 1 -- gross profit equal to revenue to
    the cent, i.e. no cost of revenue.  Of the 323 sources carrying one, exactly TWO report 1 in
    every period; the rest alternate between 1 and a real margin (GXAI: 1, 0.9516, 1, 0.669, 1).

    Left alone it is worse than noise: with both legs at 1 the index is EXACTLY 1.0, the neutral
    value, so an unmeasured component silently scores as the benign one -- the case the
    2026-07-26 domain review ruled on ("UNCOMPUTABLE != MAXIMALLY SUSPICIOUS" has a mirror).
    """
    rows = [dict(_Q, grossProfitMargin=1.0) for _ in range(12)]
    with _domain_off('grossProfitMargin'):
        off = _col(_runq(rows, 'NOCOGS')['mdf'], 'GMI').iloc[0]
    assert off == 1.0, ('the sentinel scores as dead neutral, which is why it is invisible; '
                        'got %r' % off)
    assert pd.isna(_col(_runq(rows, 'NOCOGS')['mdf'], 'GMI').iloc[0])
    #  ...and a REAL high margin just below the sentinel is untouched.
    real = [dict(_Q, grossProfitMargin=0.999) for _ in range(12)]
    assert _col(_runq(real, 'HIGHM')['mdf'], 'GMI').iloc[0] == 1.0
    assert not pd.isna(_col(_runq(real, 'HIGHM')['mdf'], 'M_Score').iloc[0])
    print("PASS test_a_gross_margin_of_exactly_1_is_the_MISSING_COST_line_and_must_not_score_"
          "as_NEUTRAL")


def test_SGAI_needs_no_magnitude_floor_because_the_PERIOD_domain_already_reaches_SAP_TO():
    """SGAI: the brief's near-zero-base exemplar was SAP.TO at base 5.2e-5.

    It is not a small measurement.  The window behind that base is SG&A = 0, 0, 1,000,000, 0
    over sales of ~1.9e10, in a series whose reported quarters carry 4.8e8 to 1.1e9 -- three
    sentinels and a crumb.  Refusing the PERIODS reaches it, so no SGAI magnitude constant is
    introduced, and this test is what stops one being added on the theory that the period rule
    missed the case it was written for.
    """
    rows = [dict(_Q, sellingGeneralAndAdministrativeExpenses=0.0, revenue=4.6e9)
            for _ in range(8)]
    rows[6]['sellingGeneralAndAdministrativeExpenses'] = 1.0e6      # the crumb
    rows += [dict(_Q, sellingGeneralAndAdministrativeExpenses=4.8e8, revenue=4.6e9)
             for _ in range(4)]
    with _domain_off('sellingGeneralAndAdministrativeExpenses'):
        off = _runq(rows, 'SAP_LIKE')['mdf']
        base_prior = _col(off, 'SGAI').iloc[0]
    assert base_prior > 1000, ('fixture must reproduce the defect; SGAI was %r' % base_prior)
    assert abs(-0.172 * base_prior) > 100, base_prior
    on = _runq(rows, 'SAP_LIKE')['mdf']
    assert pd.isna(_col(on, 'SGAI').iloc[0]), _col(on, 'SGAI').iloc[0]
    #  no floor constant exists for this base, and none is smuggled in through the domain
    assert dm.PERIOD_DOMAIN['sellingGeneralAndAdministrativeExpenses'] == (0.0, np.inf)
    print("PASS test_SGAI_needs_no_magnitude_floor_because_the_PERIOD_domain_already_reaches_"
          "SAP_TO")


def test_SGI_has_NO_floor_and_a_REAL_21800x_revenue_ramp_STILL_SCORES():
    """SGI: the over-reach test, and the reason SGI gets no floor at all.

    GXAI's revenue genuinely goes 275 -> 6,005,051 across the window.  Its SGI is enormous and
    LEGITIMATE, and its base is a CURRENCY LEVEL, so there is no unit-free constant to place on
    it and a self-relative test would refuse exactly this name.  The index must survive, be
    finite, and still be able to FLAG -- an abstention here would be the guard over-reaching
    into the signal it exists to protect.
    """
    rows = [dict(_Q, revenue=68.75) for _ in range(8)]               # 275/yr
    rows += [dict(_Q, revenue=1.50126e6) for _ in range(4)]          # 6,005,051/yr
    r = _runq(rows, 'RAMP')
    sgi = _col(r['mdf'], 'SGI').iloc[0]
    assert 21000 < sgi < 22500, sgi
    assert not pd.isna(_col(r['mdf'], 'M_Score').iloc[0]), 'the ramp must still SCORE'
    assert r['m_flagged'], 'and it must still be able to flag'
    print("PASS test_SGI_has_NO_floor_and_a_REAL_21800x_revenue_ramp_STILL_SCORES")


def test_SGI_refuses_only_the_ARITHMETIC_domain_of_its_base():
    """SGI: trailing-year sales of zero or less is not a level to index against (3,493 bases on
    the panel).  That is arithmetic, not a threshold -- and it is the ONLY refusal SGI gets.

    The NEGATIVE base is the limb worth pinning: a zero base already produced NaN through 0/0,
    but a trailing year that nets to -400 of contra-revenue against +4,000 this year used to
    emit SGI = -10, which at +0.892 SUBTRACTS 9 points from M -- an unmeasurable sales level
    scoring as the clean side, the mirror of the 2026-07-26 ruling.  The panel carries 379
    negative revenue rows and 3,057 zero ones.
    """
    zero = [dict(_Q, revenue=0.0) for _ in range(8)] + [dict(_Q) for _ in range(4)]
    assert pd.isna(_col(_runq(zero, 'NOSALES')['mdf'], 'SGI').iloc[0])
    neg = [dict(_Q, revenue=-100.0) for _ in range(8)] + [dict(_Q, revenue=1000.0)
                                                          for _ in range(4)]
    on = _col(_runq(neg, 'NEGSALES')['mdf'], 'SGI').iloc[0]
    assert pd.isna(on), ('a negative trailing-year sales base must ABSTAIN, not emit a '
                         'negative growth index; got %r' % on)
    print("PASS test_SGI_refuses_only_the_ARITHMETIC_domain_of_its_base")


def test_LVGI_gets_NO_FLOOR_because_its_density_walks_flat_into_zero():
    """LVGI: the one place a floor was expected and the panel refuses it.

    (LTD+CL)/TA has a per-0.001 density of 51.6, 60.4, 56.9, 48.7, 46.7 walking INTO zero --
    flat, no excess.  Compare aqiTTM, where the same measurement gave 2,164 rows in the first
    0.005 bin against a flat ~740/bin above 0.010.  So a leverage share of 0.002 is UNUSUAL,
    not degenerate, and it must still produce an index.  Only the impossible side is refused.
    """
    #  leverage share 0.002 -> 0.004: a real, if extreme, doubling of a tiny liability book
    thin = [dict(_Q, longTermDebt=0.0, totalCurrentLiabilities=0.5) for _ in range(8)]
    thin += [dict(_Q, longTermDebt=0.0, totalCurrentLiabilities=1.0) for _ in range(4)]
    r = _runq(thin, 'THIN')
    lvgi = _col(r['mdf'], 'LVGI').iloc[0]
    assert abs(lvgi - 2.0) < 1e-9, lvgi
    assert not pd.isna(_col(r['mdf'], 'M_Score').iloc[0])
    #  a DEBT-FREE firm is a real firm: longTermDebt == 0 is a measurement (4,869 panel rows),
    #  NOT a sentinel, and nothing here may treat it as one.
    assert 'longTermDebt' not in dm.PERIOD_DOMAIN
    #  ...and the impossible side is still refused
    zero = [dict(_Q, longTermDebt=0.0, totalCurrentLiabilities=0.0) for _ in range(12)]
    assert pd.isna(_col(_runq(zero, 'NOLIAB')['mdf'], 'LVGI').iloc[0])
    print("PASS test_LVGI_gets_NO_FLOOR_because_its_density_walks_flat_into_zero")


def test_ONE_refused_period_takes_the_WHOLE_trailing_year_window_not_just_its_own_row():
    """The refusal is at the SUMMAND, before `invrollsumTTM`, so the NaN propagates through
    every window that contains it -- the `ttm_sum` rule of 2026-08-14 ("the newest rpy rows,
    every one present, else NaN") applied to the forensic layer.  A base assembled from periods
    the vendor did not report is not a trailing-year measurement whatever its magnitude."""
    rows = [dict(_Q) for _ in range(16)]
    rows[8]['daysSalesOutstanding'] = 0.0          # exactly ONE bad period, in the interior
    bad = int(_col(_runq(rows, 'ONEBAD')['mdf'], 'DSRI').isna().sum())
    #  the CLEAN arm still has NaN at the OLD end -- the oldest rows have no prior-year window
    #  -- so the claim is the DIFFERENCE, not the count.
    clean = int(_col(_runq([dict(_Q) for _ in range(16)], 'CLEAN')['mdf'], 'DSRI').isna().sum())
    assert bad - clean >= 4, ('one refused period must take its whole trailing-year window, '
                              'not one row; %d vs %d NaN' % (bad, clean))
    print("PASS test_ONE_refused_period_takes_the_WHOLE_trailing_year_window_not_just_its_own_"
          "row")


def test_the_period_domains_are_read_at_CALL_time_not_bound_at_import():
    """The same trap `_floor_share_base` documents: a default argument is evaluated ONCE at
    import, so overriding the module constant would appear to work and do nothing -- which is
    how an earlier measurement in this file's history reported "no change" for a reason that had
    nothing to do with the data.  Every test above depends on this holding."""
    rows = [dict(_Q, daysSalesOutstanding=0.0) for _ in range(12)]
    rows[6]['daysSalesOutstanding'] = 30.0
    rows[10]['daysSalesOutstanding'] = 5922.0
    with _domain_off('daysSalesOutstanding'):
        assert not pd.isna(_col(_runq(rows, 'CALLT')['mdf'], 'DSRI').iloc[0])
    assert pd.isna(_col(_runq(rows, 'CALLT')['mdf'], 'DSRI').iloc[0])
    print("PASS test_the_period_domains_are_read_at_CALL_time_not_bound_at_import")


def test_the_period_domains_do_NOT_touch_an_ordinary_firm():
    """Inertness on a normal filer -- without this, a domain set too wide would pass every test
    above by refusing everything.  The baseline fixture's DSO is 25 days, its margin 0.10 and
    its SG&A 25, all strictly inside their domains."""
    rows = [dict(_Q) for _ in range(20)]
    with _domain_off():
        a = _runq(rows, 'ORD2')['mdf']
    b = _runq(rows, 'ORD2')['mdf']
    for col in ('DSRI', 'GMI', 'SGI', 'SGAI', 'LVGI', 'M_Score'):
        x = _col(a, col).to_numpy(dtype='float64')
        y = _col(b, col).to_numpy(dtype='float64')
        assert np.allclose(x, y, equal_nan=True, rtol=0, atol=0), (col, x, y)
    print("PASS test_the_period_domains_do_NOT_touch_an_ordinary_firm")


def test_a_beneish_refusal_can_NEVER_reach_primary_eject():
    """THE COUPLING THAT NEARLY SHIPPED IN 72298ab, asserted at this layer too.

    Blanking a field that appears in `nan_policy.PRIMARY_PRESENT` ejects the whole SOURCE via
    `primary_eject` inside `data_quality.filter_invalid_data` -- that is how the input-sanity
    change came within a review of deleting JHX and SZZL from the universe.  These guards are
    structurally incapable of it because they refuse LOCAL COPIES and never write to the shared
    panel; this test pins that structure rather than trusting it.  Asserted THREE ways: the
    frame is byte-identical after scoring, no `sanityRefusedFields` stamp appears, and the eject
    verdict on the frame is unchanged and empty.
    """
    rows = [dict(_Q, daysSalesOutstanding=0.0, grossProfitMargin=0.0,
                 sellingGeneralAndAdministrativeExpenses=0.0, revenue=0.0)
            for _ in range(12)]
    df = _build_quarters(rows, 'EJECT')
    df['totalLiabilities'] = 100.0
    df['totalStockholdersEquity'] = 150.0
    df['price'] = 10.0
    df['marketCap'] = 40.0
    before = df.copy(deep=True)
    eject_before = npol.primary_eject(df)
    dm.calcBeneishM({'cdx_df': df, 'postRank': pd.DataFrame({'source': ['EJECT']})}, ['EJECT'])
    eject_after = npol.primary_eject(df)
    assert df.equals(before), 'calcBeneishM must not mutate the shared panel'
    assert npol.SANITY_REFUSED_COLUMN not in df.columns
    assert len(eject_before) == 0 and len(eject_after) == 0, (eject_before, eject_after)
    print("PASS test_a_beneish_refusal_can_NEVER_reach_primary_eject")


# ---------------------------------------------------------------------------
#  THE FORENSIC DATA GAP IN THE AD-HOC PENALTY BUCKET (CEO, 2026-08-16).
#  Beneish itself stays display-only; what enters the score is the ABSENCE of an assessment.
#  The bucket's own invariants are pinned in test_adhoc_penalty.py -- what is pinned HERE is
#  the CHECK that feeds it, beside the layer that raises it.
# ---------------------------------------------------------------------------

import adhoc_penalty as ap


def _charge(rows, symbol='GAP', sector_map=None):
    """(book, gaps frame) for one synthetic source, through the shipped function.

    `sector_map={}` BY DEFAULT, and it is not cosmetic: the production default reads the
    repo-root `sectorsdic_fmp.pickle` (40,164 real symbols), and these fixtures use short
    synthetic tickers that COLLIDE with real ones -- `TWO` resolves to a financial there, so
    the forensic-validity gate silently exempted a fixture and two point tests failed for a
    reason that had nothing to do with points.  An empty map makes every fixture 'Unknown',
    i.e. forensically valid, which is the case these tests are about.
    """
    df = _build_quarters(rows, symbol)
    book = ap.PenaltyBook()
    gaps = dm.contribute_forensic_gap_points(df, [symbol], book, verbose=False,
                                             sector_map={} if sector_map is None
                                             else sector_map)
    return book, gaps


def test_the_forensic_gap_CHARGES_the_bucket_and_SCALES_with_how_much_is_missing():
    """CEO, 2026-08-16: *"we should probably punish it slightly. Straight to the ad-hoc penalty
    bucket. But have it very slight."*  And: one absent input is a different statement from
    three, so the charge is a COUNT of gap events, exactly as `stage1_veto` counts missing rows
    -- the weight stays 0.01 and the scaling lives in the amount (the 2026-08-10 ruling).
    """
    full = [dict(_Q) for _ in range(12)]
    one = [dict(_Q, grossProfitMargin=0.0) for _ in range(12)]
    two = [dict(_Q, grossProfitMargin=0.0,
                sellingGeneralAndAdministrativeExpenses=0.0) for _ in range(12)]
    _b, g_full = _charge(full, 'FULL')
    _b, g_one = _charge(one, 'ONE')
    _b, g_two = _charge(two, 'TWO')
    assert float(g_full['points'].iloc[0]) == 0.0, g_full.to_dict('records')
    #  one absent component + the missing verdict it causes
    assert float(g_one['points'].iloc[0]) == 2.0, g_one.to_dict('records')
    assert float(g_two['points'].iloc[0]) == 3.0, g_two.to_dict('records')
    #  and the AggScore consequence, at the fixed weight
    b2, _ = _charge(two, 'TWO2')
    assert abs(float(b2.penalty_series(['TWO2']).iloc[0]) - (-0.03)) < 1e-12
    print("PASS test_the_forensic_gap_CHARGES_the_bucket_and_SCALES_with_how_much_is_missing")


def test_the_charge_is_MONOTONE_and_CAPPED_and_can_never_be_a_BONUS():
    """MONOTONICITY IS THE PROPERTY THIS BUCKET HAS BROKEN BEFORE.  Its first version read the
    SHAPE of the refusals and gave a FULLY refused window a free pass, so the charge climbed to
    -0.07 at seven missing rows of eight and fell to 0.00 at eight of eight -- the WORST data
    paid LEAST.  Walked here across 0..8 absent components.

    The cap at 3 component points is a DECISION, not a rounding: beyond three the statement has
    saturated ("we could not assess this company") and the `no_verdict` point is what says it.
    It binds on 162 of 2,629 panel names (6.2%).
    """
    last = -1.0
    for n in range(0, 9):
        absent = list(dm.M_COMPONENTS)[:n]
        pts, items = dm.forensic_gap_points(absent, has_verdict=(n == 0))
        assert pts >= last, ('the charge FELL as the data got worse: %d absent -> %.1f after '
                             '%.1f' % (n, pts, last))
        last = pts
        assert all(p > 0 for _c, p in items), items      # `PenaltyBook.add` refuses <= 0
    assert dm.forensic_gap_points(list(dm.M_COMPONENTS), False)[0] == 4.0
    assert dm.forensic_gap_points(list(dm.M_COMPONENTS)[:3], False)[0] == 4.0
    #  "Very slight" is the CEO's word and this is where it is pinned: the maximum forensic
    #  charge must stay STRICTLY BELOW the largest charge ONE Stage-1 veto flag can raise --
    #  a missing forensic assessment is a smaller finding than a persistent solvency red flag.
    #  ASSERTED AGAINST THE VETO'S OWN CONSTANT (review L-7: the previous
    #  `assert 0.01 * 4.0 * 2 == 0.08` was a tautology on literals and could not fail if the
    #  veto's window changed underneath it).
    #  AND THE "HALF" CLAIM WAS WRONG, which is what writing this assertion exposed: the veto's
    #  worst SINGLE FLAG is 7 points (refused rows on all but one row of its 8-row window),
    #  i.e. -0.07, not -0.08.  -0.08 is the worst per-SOURCE total observed on the 2026-08-13
    #  run, across several checks.  So 4 points is 57% of one veto flag and exactly half the
    #  worst observed source total -- both stated in the block, neither of them "half the
    #  veto's maximum".
    import stage1_veto as sv
    veto_one_flag_max = float(sv.WINDOW_ROWS - 1)
    ours = dm.forensic_gap_points(list(dm.M_COMPONENTS), False)[0]
    assert ours < veto_one_flag_max, (ours, veto_one_flag_max)
    print("PASS test_the_charge_is_MONOTONE_and_CAPPED_and_can_never_be_a_BONUS")


def test_the_COVERAGE_ONLY_abstention_is_charged_TOO_and_it_is_the_only_charge_that_reaches_it():
    """The 84 panel names whose components are ALL computable somewhere and which still abstain
    on `nan_policy.COVERAGE_MIN`.  Under a components-only rule they would be FREE -- an
    abstention that costs nothing, which is the exact case the ruling exists to remove.
    """
    rows = [dict(_Q) for _ in range(8)]            # too short for a full M window
    book, g = _charge(rows, 'COV')
    assert int(g['n_absent'].iloc[0]) == 0, g.to_dict('records')
    assert bool(g['has_verdict'].iloc[0]) is False
    assert float(g['points'].iloc[0]) == 1.0, g.to_dict('records')
    items = book.itemised()
    assert list(items['check']) == [dm.CHECK_GAP_NO_VERDICT], list(items['check'])
    assert 'trailing-year' in items['reason'].iloc[0] or 'computable' in items['reason'].iloc[0]
    print("PASS test_the_COVERAGE_ONLY_abstention_is_charged_TOO_and_it_is_the_only_charge_"
          "that_reaches_it")


def test_a_name_whose_assessment_WAS_made_is_charged_NOTHING():
    """Inertness, and it is the half that makes the charge mean something: if an ordinary
    filer were charged too, the penalty would be a constant and would reorder nobody."""
    rows = [dict(_Q) for _ in range(16)]
    book, g = _charge(rows, 'CLEANFIRM')
    assert bool(g['has_verdict'].iloc[0]) is True
    assert float(g['points'].iloc[0]) == 0.0
    assert len(book) == 0 and book.penalty_series(['CLEANFIRM']).iloc[0] == 0.0
    assert g['reason'].iloc[0] == ''
    print("PASS test_a_name_whose_assessment_WAS_made_is_charged_NOTHING")


def test_the_points_and_the_SCORE_come_from_ONE_computation_and_cannot_drift():
    """The bucket charges a name for not having an M; the CSV beside it explains why.  If the
    two were computed separately the artifact could show a gross margin the bucket had just
    charged the name for lacking.  `contribute_forensic_gap_points` calls `calcBeneishM` rather
    than re-deriving the gap, and this is what pins that: every name the scorer leaves without
    an M_Score_mean is charged, and no name that has one is.
    """
    rows = [dict(_Q, grossProfitMargin=0.0) for _ in range(6)] + [dict(_Q) for _ in range(6)]
    df = _build_quarters(rows, 'DRIFT')
    _mdf, SL, _p = dm.calcBeneishM({'cdx_df': df}, ['DRIFT'], verbose=False)
    scored = pd.notna(pd.to_numeric(SL['M_Score_mean'], errors='coerce').iloc[0])
    book = ap.PenaltyBook()
    g = dm.contribute_forensic_gap_points(df, ['DRIFT'], book, verbose=False)
    charged = float(g['points'].iloc[0]) > 0
    assert charged == (not scored), (scored, charged, g.to_dict('records'))
    print("PASS test_the_points_and_the_SCORE_come_from_ONE_computation_and_cannot_drift")


def test_verbose_False_changes_the_LOG_and_never_the_NUMBERS():
    """The gap charge computes the pool's M a second time and must not print a second set of
    guard counts over a DIFFERENT name set -- an operator reading two different refusal counts
    for one run cannot tell which describes the names the CEO is shown.  What it must not do is
    change an answer."""
    rows = [dict(_Q, grossProfitMargin=0.0) for _ in range(12)]
    df = _build_quarters(rows, 'QUIET')
    a = dm.calcBeneishM({'cdx_df': df}, ['QUIET'], verbose=True)
    b = dm.calcBeneishM({'cdx_df': df}, ['QUIET'], verbose=False)
    for col in dm.M_COMPONENTS + ('M_Score',):
        x = pd.to_numeric(a[0][col], errors='coerce').to_numpy(dtype='float64')
        y = pd.to_numeric(b[0][col], errors='coerce').to_numpy(dtype='float64')
        assert np.allclose(x, y, equal_nan=True, rtol=0, atol=0), col
    assert a[2] == b[2]
    print("PASS test_verbose_False_changes_the_LOG_and_never_the_NUMBERS")


def test_the_abstention_SAYS_WHY_and_names_the_missing_vendor_input():
    """CEO, 2026-08-16: an abstention must say why -- a 34% blank rate reads as a broken tool
    unless it explains itself.  The reason names the INPUT in English, not the component code,
    because the reader of the deck is not required to know what SGAI is."""
    rows = [dict(_Q, grossProfitMargin=0.0,
                 sellingGeneralAndAdministrativeExpenses=0.0) for _ in range(12)]
    _b, g = _charge(rows, 'WHY')
    reason = g['reason'].iloc[0]
    assert 'gross margin' in reason and 'SG&A intensity' in reason, reason
    assert 'GMI' in reason and 'SGAI' in reason, reason
    #  ...and a name that HAS a verdict says nothing, so the column is self-limiting
    assert dm.abstention_reason([], True) == ''
    print("PASS test_the_abstention_SAYS_WHY_and_names_the_missing_vendor_input")


def test_the_CSV_reason_is_the_SAME_STRING_the_bucket_charged_on():
    """One function, two artifacts.  If `ForensicFlagsTop100.csv` and
    `AdHocPenaltyBucket.csv` explained the same abstention differently, the CEO would have two
    accounts of one fact and no way to tell which the score used."""
    rows = [dict(_Q, grossProfitMargin=0.0) for _ in range(12)]
    df = _build_quarters(rows, 'SAME')
    book = ap.PenaltyBook()
    g = dm.contribute_forensic_gap_points(df, ['SAME'], book, verbose=False)
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': ['SAME']})}
    det = dm.detectManipulationWrapper(resdic)
    tbl = ff.buildForensicFlagTable({**resdic, **det}, 1)
    csv_reason = tbl['M_abstain_reason'].iloc[0]
    charged_reason = book.itemised()
    charged_reason = charged_reason[
        charged_reason['check'] == dm.CHECK_GAP_COMPONENTS]['reason'].iloc[0]
    assert csv_reason == charged_reason == g['reason'].iloc[0], (csv_reason, charged_reason)
    assert tbl['forensicTag'].iloc[0] == 'data-incomplete: dig-deeper'
    print("PASS test_the_CSV_reason_is_the_SAME_STRING_the_bucket_charged_on")


def test_the_gap_is_raised_ONCE_per_run_so_a_name_in_TWO_pools_is_not_charged_TWICE():
    """`PenaltyBook.penalty_series` sums a source's contributions across every pool tag, so a
    finding raised for the general pool AND again for a carve-out cohort would charge a name
    that sits in both twice for ONE gap.  The gap is a property of the name's data and is
    identical in every pool, so it is raised once with `pool=None` -- and the call site passes
    the UNION of the pools rather than calling per pool.  A duplicate source in that union must
    still only be charged once.
    """
    rows = [dict(_Q, grossProfitMargin=0.0) for _ in range(12)]
    df = _build_quarters(rows, 'DUP')
    book = ap.PenaltyBook()
    dm.contribute_forensic_gap_points(df, ['DUP', 'DUP', 'DUP'], book, verbose=False)
    assert float(book.penalty_series(['DUP']).iloc[0]) == -0.02, book.itemised().to_dict()
    assert set(book.itemised()['pool'].fillna('')) == {''}, book.itemised().to_dict()
    print("PASS test_the_gap_is_raised_ONCE_per_run_so_a_name_in_TWO_pools_is_not_charged_TWICE")


def test_the_gap_charge_is_raised_BEFORE_head_100_so_it_cannot_change_MEMBERSHIP():
    """*** review L-8. ***  "`head(100)` runs BEFORE Stage-2, so this penalty CANNOT change
    top-100 membership" is load-bearing -- it is what bounds the blast radius of the whole
    charge to a REORDER -- and it was true only by the current statement order in `postBo`.
    Nothing asserted it, so a future reorder would silently turn a reordering penalty into a
    selection one.  Same idiom as `test_published_columns.test_moatScore_is_raw_because_it_is_
    merged_after_scoring`, which pins the mirror-image ordering fact for moatScore.
    """
    src = open(os.path.join(REPO, 'postBo.py'), encoding='utf-8').read()
    i_charge = src.index('contribute_forensic_gap_points')
    #  the ACTUAL cut, by its statement -- not the first `.head(100)` in the file, which is a
    #  comment about the legacy no-carve path 200 lines earlier
    i_cut = src.index('BoS_dftop100 = general_scores.head(100)')
    i_stage2 = src.index('pbr.postBoScoreRanking')
    assert i_charge < i_cut < i_stage2, (
        'the forensic gap charge must be raised BEFORE the head(100) cut and the cut BEFORE '
        'Stage-2 (charge %d, cut %d, stage-2 %d). If the charge ever moves after the cut it '
        'still only reorders; if the CUT moves after Stage-2, the penalty starts deciding '
        'top-100 MEMBERSHIP, which is a different and much larger claim than the one measured.'
        % (i_charge, i_cut, i_stage2))
    print("PASS test_the_gap_charge_is_raised_BEFORE_head_100_so_it_cannot_change_MEMBERSHIP")


def test_a_forensically_INVALID_name_is_NOT_charged_for_lacking_a_Beneish_score():
    """*** review H-2. ***  `forensicFlags._classify_financial` already rules Beneish
    structurally undefined for banks, insurers and REITs, and the forensic tag says so instead
    of scoring them.  Charging a REIT for not having a Beneish score is charging it for being a
    REIT, and the reason written against it ("the assessment is ABSENT -- not clean") is false:
    it is INAPPLICABLE.  Measured before this gate, the five shipped cohort side-lists charged
    71 names of which 57 were forensically invalid (REIT 16 of 17, InvestmentVehicle 16 of 20,
    FinManager 13 of 16, BalanceSheetFin 12 of 13); the general top-100 was already clean.

    Asserted in BOTH directions -- the exemption must not become a blanket amnesty: the SAME
    fixture with a non-financial sector is still charged.
    """
    rows = [dict(_Q, grossProfitMargin=0.0) for _ in range(12)]
    df = _build_quarters(rows, 'AREIT')
    book = ap.PenaltyBook()
    g = dm.contribute_forensic_gap_points(df, ['AREIT'], book, verbose=False,
                                          sector_map={'AREIT': 'Real Estate'})
    assert float(g['points'].iloc[0]) == 0.0, g.to_dict('records')
    assert bool(g['forensically_valid'].iloc[0]) is False
    assert len(book) == 0 and float(book.penalty_series(['AREIT']).iloc[0]) == 0.0
    #  ...and the exemption is DECLARED, not silent
    un = book.unmeasured
    assert len(un) == 1 and 'EXEMPT' in un[0]['reason'] and un[0]['points'] == 0.0
    #  the same name with an ordinary sector IS charged -- the gate is a classification, not an
    #  amnesty, and the gap itself is unchanged
    book2 = ap.PenaltyBook()
    g2 = dm.contribute_forensic_gap_points(df, ['AREIT'], book2, verbose=False,
                                           sector_map={'AREIT': 'Technology'})
    assert float(g2['points'].iloc[0]) == 2.0, g2.to_dict('records')
    assert int(g2['n_absent'].iloc[0]) == int(g['n_absent'].iloc[0]) == 1
    #  ...and the DEFAULT map is the forensic layer's own, so the charge and the shipped tag
    #  cannot be classifying names off two different sources.  (Every test above passes an
    #  explicit map to stay hermetic, which would otherwise leave this wiring unpinned.)
    src = inspect.getsource(dm.contribute_forensic_gap_points)
    assert 'from forensicFlags import _classify_financial, _load_sector_map' in src
    assert 'sector_map = _load_sector_map()' in src
    #  ...and the CSV explains such a row the SAME way the bucket does: the model does not
    #  apply, rather than "no usable vendor data", which would describe a measurement nobody
    #  attempted.  Two artifacts, one account of the row.
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': ['AREIT']})}
    det = dm.detectManipulationWrapper(resdic)
    tbl = ff.buildForensicFlagTable({**resdic, **det}, 1,
                                    sector_fallback={'AREIT': 'Real Estate'})
    assert tbl['forensicTag'].iloc[0].startswith('financial: forensic-invalid')
    assert 'does not apply' in tbl['M_abstain_reason'].iloc[0], tbl['M_abstain_reason'].iloc[0]
    assert 'no usable vendor data' not in tbl['M_abstain_reason'].iloc[0]
    print("PASS test_a_forensically_INVALID_name_is_NOT_charged_for_lacking_a_Beneish_score")


def test_a_partial_failure_commits_NOTHING_to_the_book():
    """*** review L-1. ***  The caller wraps this in one `except` whose banner says the gap
    "cost nothing -- 'charged nothing' here does not mean 'no gaps found'".  If the loop raised
    partway, names 1..k-1 were already in the book and would ship in the evidence CSV
    underneath that caveat -- a false statement in the one file that exists to be argued with.
    Contributions are buffered and committed only after the loop completes.
    """
    rows = [dict(_Q, grossProfitMargin=0.0) for _ in range(12)]
    df = pd.concat([_build_quarters(rows, 'A1'), _build_quarters(rows, 'A2')],
                   ignore_index=True)
    book = ap.PenaltyBook()
    saved = dm.abstention_reason
    calls = {'n': 0}

    def _boom(*a, **k):
        calls['n'] += 1
        if calls['n'] > 1:
            raise RuntimeError('vendor frame went missing mid-loop')
        return saved(*a, **k)
    try:
        dm.abstention_reason = _boom
        try:
            dm.contribute_forensic_gap_points(df, ['A1', 'A2'], book, verbose=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError('the fixture did not raise; the test proves nothing')
    finally:
        dm.abstention_reason = saved
    assert len(book) == 0, ('a partial charge reached the book: %s' % book.itemised().to_dict())
    print("PASS test_a_partial_failure_commits_NOTHING_to_the_book")


def test_no_penalty_book_means_NO_CHARGE_and_no_crash():
    """A caller that passes no book gets an empty frame and no exception -- the same shape
    `postBoScoreRanking` already has for a pool with no book, where the honest reading is
    "this pool was scored with no penalty", not "this pool has no data gaps"."""
    rows = [dict(_Q, grossProfitMargin=0.0) for _ in range(12)]
    out = dm.contribute_forensic_gap_points(_build_quarters(rows, 'NB'), ['NB'], None)
    assert out.empty and list(out.columns)[:2] == ['source', 'n_absent']
    print("PASS test_no_penalty_book_means_NO_CHARGE_and_no_crash")


# ---------------------------------------------------------------------------
#  P-1  --  `calcMontierC.DSOinc` READS THE SAME SENTINEL COLUMN (2026-08-17)
#  P-2  --  AN EXPLANATION OF A VERDICT THAT DOES NOT EXIST
#
#  Both are the same shape as defects already fixed in this module, in a second consumer.
#  Every test below is two-sided: the OFF arm must REPRODUCE the defect before the ON arm is
#  allowed to claim it is gone, because a guard that only ever runs on clean fixtures proves
#  nothing.
# ---------------------------------------------------------------------------

#  A quarterly series whose DSO is a vendor SENTINEL in the newest year and REPORTED in the
#  prior one.  `DSOinc` = TTM(DSO)_t - TTM(DSO)_{t-4}, so with the zeros summed in as if they
#  were collection periods the newest window's base COLLAPSES and the flag reads as a large
#  DECREASE; drop the sentinels to the prior year instead and it reads as a large INCREASE and
#  FIRES.  The second arrangement is the one that matters: the sentinel MANUFACTURES a red flag.
def _dso_rows(newest_year, prior_year, n=12):
    """12 quarters, oldest first: the newest 4 carry `newest_year`, the 4 before `prior_year`."""
    rows = [dict(_Q) for _ in range(n)]
    for i in range(n):
        if i >= n - 4:
            rows[i]['daysSalesOutstanding'] = newest_year
        elif i >= n - 8:
            rows[i]['daysSalesOutstanding'] = prior_year
    return rows


def _runc(rows, symbol='CQ'):
    df = _build_quarters(rows, symbol)
    cdf, slc, pc = dm.calcMontierC({'cdx_df': df}, [symbol])
    return dict(cdf=cdf.reset_index(drop=True), c_mean=slc['C_Score_mean'].iloc[0],
                c_flagged=symbol in pc)


def test_P1_a_SENTINEL_zero_DSO_period_does_not_reach_MontierC_DSOinc():
    """*** P-1. ***  `calcMontierC` differenced the SAME `daysSalesOutstanding` column
    `calcBeneishM` guards, with no domain at all.  A zero there is FMP's placeholder -- 7,174
    panel rows sit at exactly zero against 35 anywhere in (0, 0.01) days -- and
    `invrollsumTTM` summed it into the trailing-year base as though it were a reported
    collection period.

    The fixture is the flag-MANUFACTURING arrangement: sentinels in the PRIOR year, a real 100
    days in the newest, so `DSOinc` reads as receivables ballooning from nothing.
    """
    rows = _dso_rows(newest_year=100.0, prior_year=0.0)
    with _domain_off('daysSalesOutstanding'):
        off = _runc(rows, 'SENTC')
    off_dso = pd.to_numeric(off['cdf']['DSOinc'], errors='coerce').iloc[0]
    assert off_dso > 0, ('the fixture must reproduce the defect it guards against; DSOinc '
                         'was %r' % off_dso)
    on = _runc(rows, 'SENTC')
    on_dso = pd.to_numeric(on['cdf']['DSOinc'], errors='coerce').iloc[0]
    assert pd.isna(on_dso), on_dso
    #  ...and the manufactured flag is gone from the score, not merely from the column
    assert float(on['c_mean']) < float(off['c_mean']), (off['c_mean'], on['c_mean'])
    print("PASS test_P1_a_SENTINEL_zero_DSO_period_does_not_reach_MontierC_DSOinc")


def test_P1_MontierC_reads_the_SAME_domain_dict_as_calcBeneishM():
    """ONE instrument, not a second implementation.  A second copy of "which DSO periods the
    vendor actually reported" is how the two forensic models start disagreeing about one
    column -- so the constant is shared, read at CALL time, and `calcMontierC` goes through
    the same `_domain_period_input` helper.

    Asserted BEHAVIOURALLY (override the dict, watch the C-score move) and then structurally,
    because the behavioural half alone would still pass if someone copy-pasted the helper.
    """
    rows = _dso_rows(newest_year=100.0, prior_year=0.0)
    saved = dict(dm.PERIOD_DOMAIN)
    try:
        #  a domain that refuses NOTHING must restore the unguarded answer...
        dm.PERIOD_DOMAIN = dict(saved, daysSalesOutstanding=(-np.inf, np.inf))
        wide = pd.to_numeric(_runc(rows, 'DOMC')['cdf']['DSOinc'], errors='coerce').iloc[0]
        #  ...and one that refuses everything below 200 days must refuse the reported 100 too,
        #  which no hard-coded `> 0` inside calcMontierC could ever do.
        dm.PERIOD_DOMAIN = dict(saved, daysSalesOutstanding=(200.0, np.inf))
        narrow = pd.to_numeric(_runc(rows, 'DOMC')['cdf']['DSOinc'], errors='coerce').iloc[0]
    finally:
        dm.PERIOD_DOMAIN = saved
    assert wide > 0 and pd.isna(narrow), (wide, narrow)
    src = inspect.getsource(dm.calcMontierC)
    assert '_domain_period_input' in src and 'MONTIER_DSO_DOMAIN' in src, src[:400]
    assert dm.MONTIER_DSO_DOMAIN == 'daysSalesOutstanding'
    assert dm.PERIOD_DOMAIN['daysSalesOutstanding'] == (0.0, np.inf)
    #  ONE definition of the dict, and no BENEISH_-prefixed alias left behind: an alias would
    #  let `dm.<old name> = ...` in a test or a measurement silently patch nothing.
    mod = open(os.path.join(REPO, 'detectManipulation.py'), encoding='utf-8').read()
    assert mod.count('\nPERIOD_DOMAIN = {') == 1, mod.count('\nPERIOD_DOMAIN = {')
    assert 'BENEISH_PERIOD_DOMAIN' not in mod
    print("PASS test_P1_MontierC_reads_the_SAME_domain_dict_as_calcBeneishM")


def test_P1_the_guard_does_NOT_touch_a_name_whose_DSO_is_always_reported():
    """The negative control.  Every column of the C-score frame must be BIT-identical for a
    filer with no sentinel anywhere -- the guard refuses periods, it does not re-specify the
    model."""
    rows = _dso_rows(newest_year=300.0, prior_year=100.0)
    with _domain_off('daysSalesOutstanding'):
        off = _runc(rows, 'CLEANC')
    on = _runc(rows, 'CLEANC')
    for col in dm.C_FLAG_COLS + ['C_Score']:
        x = pd.to_numeric(off['cdf'][col], errors='coerce').to_numpy(dtype='float64')
        y = pd.to_numeric(on['cdf'][col], errors='coerce').to_numpy(dtype='float64')
        assert np.allclose(x, y, equal_nan=True, rtol=0, atol=0), col
    assert float(off['c_mean']) == float(on['c_mean'])
    #  and the fixture is a live one: DSOinc genuinely fires here, so "identical" is not
    #  "identically empty"
    assert pd.to_numeric(on['cdf']['DSOinc'], errors='coerce').iloc[0] > 0
    print("PASS test_P1_the_guard_does_NOT_touch_a_name_whose_DSO_is_always_reported")


def test_P1_the_refusal_does_NOT_abstain_it_scores_the_name_CLEANER():
    """THE DECLARED COST, PINNED SO IT CANNOT BE FORGOTTEN.  Unlike Beneish, this model has no
    abstention: `C_Score` is `(cols > 0).sum()` and NaN > 0 is False, so a refused period does
    not make the name UNSCORED -- it makes one of five flags not fire, and the name reads
    CLEANER.  Measured on the 2026-08-13 panel `C_Score_mean` is non-NaN for all 2,629 sources
    on BOTH arms, and every one of the 167 names that moves moves DOWN.

    This is the ratified safe direction for a review flag (2026-07-19), not a free correction.
    If a future change gives the C-score a real abstention, this test SHOULD fail -- and it
    should fail loudly, because that is a CEO decision about what a blank C-score means, not a
    refactor.
    """
    rows = _dso_rows(newest_year=100.0, prior_year=0.0)
    with _domain_off('daysSalesOutstanding'):
        off = _runc(rows, 'COSTC')
    on = _runc(rows, 'COSTC')
    assert not pd.isna(float(on['c_mean'])), on['c_mean']
    assert float(on['c_mean']) < float(off['c_mean'])
    #  the refused flag reads exactly like a flag that was computed and did not fire
    assert pd.isna(pd.to_numeric(on['cdf']['DSOinc'], errors='coerce').iloc[0])
    assert int(on['cdf']['C_Score'].iloc[0]) == int(off['cdf']['C_Score'].iloc[0]) - 1
    print("PASS test_P1_the_refusal_does_NOT_abstain_it_scores_the_name_CLEANER")


def test_P1_daysOfInventoryOutstanding_is_deliberately_NOT_guarded():
    """A JUDGEMENT, RAISED RATHER THAN TAKEN -- pinned so that guarding it later is a decision
    and not a drift.  `DSIinc` reads `daysOfInventoryOutstanding`, which carries exact zeros
    too, but a services filer genuinely holds no inventory: the arithmetic corroboration that
    carries the gross-margin limb (a zero margin sits on a zero-revenue row) has no analogue
    here.  So the field is NOT in the domain dict and `DSIinc` still fires on a zero-to-real
    transition."""
    assert 'daysOfInventoryOutstanding' not in dm.PERIOD_DOMAIN
    rows = [dict(_Q) for _ in range(12)]
    for i in range(12):
        rows[i]['daysOfInventoryOutstanding'] = 100.0 if i >= 8 else 0.0
    r = _runc(rows, 'DSIC')
    assert pd.to_numeric(r['cdf']['DSIinc'], errors='coerce').iloc[0] > 0
    print("PASS test_P1_daysOfInventoryOutstanding_is_deliberately_NOT_guarded")


def test_the_C_score_is_DISPLAY_ONLY_and_cannot_reach_the_RANKING():
    """*** The question P-1 had to answer before it could be scoped. ***  Beneish's ABSENCE
    now enters the score through the ad-hoc bucket, so "is the C-score in the ranking too?"
    decides whether a guard on it is a display change or a scoring change.  It is NOT: the
    whole forensic layer is computed AFTER the ranking is fixed.

    Same idiom as `test_the_gap_charge_is_raised_BEFORE_head_100...` -- the fact is true only
    by statement ORDER in `Sbocker`, so the order is what is asserted.  If this ever fails,
    the C-score has become a scoring input and every measurement of a C-side change has to be
    redone against AggScore, not against a CSV column.
    """
    sb = open(os.path.join(REPO, 'Sbocker.py'), encoding='utf-8').read()
    assert sb.index('pb.postBoWrapper') < sb.index('dm.detectManipulationWrapper'), (
        'detectManipulationWrapper must run AFTER postBoWrapper (which contains Stage-1, the '
        'head(100) cut and Stage-2); if it moves earlier its output could be fed into the '
        'score.')
    #  ...and nothing on the scoring path reads the C-score by name.
    for mod in ('postBoRank.py', 'calcScore.py', 'adhoc_penalty.py', 'stage1_veto.py',
                'carveOut.py'):
        s = open(os.path.join(REPO, mod), encoding='utf-8').read()
        for token in ('SLmeanCscore', 'C_Score_mean', 'C_flag_ge_4', 'calcMontierC'):
            assert token not in s, (mod, token)
    #  the ONE consumer that is allowed to see it publishes a column, and does so from the
    #  frame `head(ntopagg)` already fixed.
    pbsrc = open(os.path.join(REPO, 'postBo.py'), encoding='utf-8').read()
    assert "BoComp_tocsv['C-Score'] = cscoreVec" in pbsrc
    print("PASS test_the_C_score_is_DISPLAY_ONLY_and_cannot_reach_the_RANKING")


# ---------------------------------------------------------------------------
#  P-2  --  a driver breakdown for a verdict that does not exist
# ---------------------------------------------------------------------------

class _no_sector_pickle(object):
    """Close the world for `buildForensicFlagTable`.

    Its default `_load_sector_map()` reads the repo-root `sectorsdic_fmp.pickle` (40,164 REAL
    symbols), so a short synthetic ticker can collide with a real financial and be tagged
    `financial: forensic-invalid` -- which is a DIFFERENT row shape from the one these tests
    are about.  The same trap `_charge` documents, on the other function.
    """

    def __enter__(self):
        self.saved = ff._load_sector_map
        ff._load_sector_map = lambda *a, **k: {}
        return self

    def __exit__(self, *a):
        ff._load_sector_map = self.saved
        return False


def _flagtable(rows, symbol):
    df = _build_quarters(rows, symbol)
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': [symbol]})}
    det = dm.detectManipulationWrapper(resdic)
    with _no_sector_pickle():
        return ff.buildForensicFlagTable({**resdic, **det}, 1)


def test_P2_M_drivers_is_BLANK_on_a_row_that_has_no_M_score():
    """*** P-2, CEO 2026-08-17: blank it. ***  `_mscore_drivers` averages each component
    INDEPENDENTLY, so it returns a decomposition even when `M_Score_mean` is NaN -- listing
    the components that WERE computable and silently omitting the one that caused the
    abstention.  On the shipped 2026-08-13 top-100 all 21 `data-incomplete` rows carried one.
    A breakdown asserts "here is what drove the verdict" where there is no verdict; the CEO's
    ruling is to say nothing rather than something unfounded, and `M_abstain_reason` already
    tells the reader why the cell is empty.

    TWO-SIDED, and the second half is the one that matters: the column must still be populated
    for every name that HAS a verdict, or the fix has traded one wrong artifact for a blank one.
    """
    #  Abstains on GMI (no usable gross margin anywhere in the window) while DSRI is not only
    #  computable but strongly ADVERSE -- DSO 100 -> 300 -- so the fixture is the real case:
    #  a name with no verdict whose surviving components would happily produce a breakdown.
    rows = [dict(r, grossProfitMargin=0.0) for r in _dso_rows(300.0, 100.0)]
    t = _flagtable(rows, 'NOVERD')
    assert pd.isna(t['M_score_mean'].iloc[0]), t['M_score_mean'].iloc[0]
    assert t['M_drivers'].iloc[0] == '', repr(t['M_drivers'].iloc[0])
    assert t['M_abstain_reason'].iloc[0], 'the blank must still be EXPLAINED'
    assert t['forensicTag'].iloc[0] == 'data-incomplete: dig-deeper'
    #  ...and the underlying decomposition still EXISTS -- this is a publication rule, not a
    #  deletion of the computation, so a future reader can still ask for it deliberately.
    det = dm.detectManipulationWrapper(
        {'cdx_df': _build_quarters(rows, 'NOVERD'),
         'postRank': pd.DataFrame({'source': ['NOVERD']})})
    assert ff._mscore_drivers(det['mscore_df'], 'NOVERD') != ''

    #  the other side: a scored name keeps its drivers
    annual = _annual([_BASE] * 5 + [_DIRTY])
    df = _build(annual, 'HASVERD')
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': ['HASVERD']})}
    det2 = dm.detectManipulationWrapper(resdic)
    with _no_sector_pickle():
        t2 = ff.buildForensicFlagTable({**resdic, **det2}, 1)
    assert pd.notna(t2['M_score_mean'].iloc[0])
    assert t2['M_drivers'].iloc[0] != '', 'a name WITH a verdict must keep its breakdown'
    assert t2['M_abstain_reason'].iloc[0] == ''
    print("PASS test_P2_M_drivers_is_BLANK_on_a_row_that_has_no_M_score")


def test_P2_C_flags_fired_is_BLANK_when_there_is_no_C_score():
    """The same rule on the neighbouring column.  ITS MEASURED EFFECT TODAY IS ZERO and that
    is stated rather than implied: `C_Score` is a COUNT, so `C_Score_mean` is NaN only when a
    name produces no forensic rows at all -- 0 of 2,629 on the 2026-08-13 panel.  The guard is
    here because an asymmetry between two adjacent columns (one self-limiting, one not) is
    exactly how the M side acquired P-2 in the first place.
    """
    rows = _dso_rows(300.0, 100.0)      # DSOinc genuinely fires, so the blank is a REFUSAL
    df = _build_quarters(rows, 'NOC')
    resdic = {'cdx_df': df, 'postRank': pd.DataFrame({'source': ['NOC']})}
    det = dm.detectManipulationWrapper(resdic)
    #  force the no-C-score row that the panel does not currently produce: the mean is absent
    #  while the per-period frame still holds fired flags.
    det['SLmeanCscore'] = det['SLmeanCscore'].assign(C_Score_mean=np.nan)
    assert (pd.to_numeric(det['cscore_df'][dm.C_FLAG_COLS].stack(), errors='coerce') > 0).any()
    with _no_sector_pickle():
        t = ff.buildForensicFlagTable({**resdic, **det}, 1)
    assert pd.isna(t['C_score_mean'].iloc[0])
    assert t['C_flags_fired'].iloc[0] == '', repr(t['C_flags_fired'].iloc[0])
    assert t['forensicTag'].iloc[0] == 'data-incomplete: dig-deeper'
    #  two-sided: with the mean present the fired set is published as before
    det2 = dm.detectManipulationWrapper(resdic)
    with _no_sector_pickle():
        t2 = ff.buildForensicFlagTable({**resdic, **det2}, 1)
    assert pd.notna(t2['C_score_mean'].iloc[0]) and t2['C_flags_fired'].iloc[0] != ''
    print("PASS test_P2_C_flags_fired_is_BLANK_when_there_is_no_C_score")


def test_P2_blanking_at_SOURCE_reaches_every_artifact_the_CEO_reads():
    """ONE place, four surfaces.  `M_drivers` is blanked in `buildForensicFlagTable`, which is
    upstream of `ForensicFlagsTop100.csv`, of the `AggScoreTop100` merge, of the XLSX forensic
    block and of the HTML deck's R5 rule -- so there is no second site to keep in step.  This
    asserts the wiring, because the measurement that proved the count goes to zero was run on
    the CSV and would not have caught a deck that recomputes its own drivers.
    """
    pb = open(os.path.join(REPO, 'postBo.py'), encoding='utf-8').read()
    #  the AggScore CSV carries the column through the merge (so the deck inherits the blank)
    assert "'M_drivers', 'M_abstain_reason'" in pb
    #  the XLSX forensic block reads the same frame, not a recomputation
    assert "frow.get('M_drivers', '')" in pb
    assert '_mscore_drivers' not in pb, 'postBo must not recompute the breakdown'
    gp = open(os.path.join(REPO, 'generate_presentation.py'), encoding='utf-8').read()
    assert "r0.get('M_drivers')" in gp, 'the deck must READ the published column'
    assert '_mscore_drivers' not in gp, 'the deck must not recompute the breakdown'
    #  and the R5 rule is the deck consumer that a stale breakdown would have fired
    assert "mdriver_hit = any(tok in m_drivers.upper()" in gp
    print("PASS test_P2_blanking_at_SOURCE_reaches_every_artifact_the_CEO_reads")


if __name__ == '__main__':
    test_component_directions_exact_row0()
    test_mscore_fold_and_flag_dirty()
    test_cscore_flags_dirty()
    test_clean_firm_below_thresholds()
    test_recency_window_reads_recent_not_oldest()
    test_stored_frames_are_newest_first()
    test_orientation_robust_to_input_order()
    test_sloan_accruals_reads_recent_quarter()
    test_tata_uses_asset_LEVEL_not_4q_sum()
    test_dsri_is_not_contaminated_by_sales_growth()
    test_c_cutoff_is_one_constant_everywhere()
    test_missing_component_does_not_fire_a_c_flag()
    test_semiannual_windows_are_period_aware()
    test_quarterly_path_is_bit_identical_under_rpy4()
    test_dappdec_is_gone_and_cannot_come_back()
    test_beneish_depi_is_deliberately_left_alone()
    test_a_vanishing_aqi_base_ABSTAINS_instead_of_emitting_a_huge_index()
    test_an_IMPOSSIBLE_aqi_base_is_refused_even_though_it_is_LARGE()
    test_depi_carries_its_OWN_floor_and_does_not_inherit_AQI_s()
    test_a_partially_refused_M_window_ABSTAINS_on_COVERAGE()
    test_the_coverage_gate_uses_a_STRICT_comparison_like_nan_policy()
    test_the_floors_are_read_at_CALL_time_not_bound_at_import()
    test_the_floor_is_SYMMETRIC_and_guards_the_numerator_leg_too()
    test_the_guards_do_NOT_touch_an_ordinary_firm()
    test_a_SENTINEL_zero_DSO_period_is_not_summed_into_the_trailing_year_base()
    test_GMI_is_refused_on_its_DOMAIN_and_a_magnitude_floor_would_be_the_WRONG_instrument()
    test_a_gross_margin_of_exactly_1_is_the_MISSING_COST_line_and_must_not_score_as_NEUTRAL()
    test_SGAI_needs_no_magnitude_floor_because_the_PERIOD_domain_already_reaches_SAP_TO()
    test_SGI_has_NO_floor_and_a_REAL_21800x_revenue_ramp_STILL_SCORES()
    test_SGI_refuses_only_the_ARITHMETIC_domain_of_its_base()
    test_LVGI_gets_NO_FLOOR_because_its_density_walks_flat_into_zero()
    test_ONE_refused_period_takes_the_WHOLE_trailing_year_window_not_just_its_own_row()
    test_the_period_domains_are_read_at_CALL_time_not_bound_at_import()
    test_the_period_domains_do_NOT_touch_an_ordinary_firm()
    test_a_beneish_refusal_can_NEVER_reach_primary_eject()
    test_the_forensic_gap_CHARGES_the_bucket_and_SCALES_with_how_much_is_missing()
    test_the_charge_is_MONOTONE_and_CAPPED_and_can_never_be_a_BONUS()
    test_the_COVERAGE_ONLY_abstention_is_charged_TOO_and_it_is_the_only_charge_that_reaches_it()
    test_a_name_whose_assessment_WAS_made_is_charged_NOTHING()
    test_the_points_and_the_SCORE_come_from_ONE_computation_and_cannot_drift()
    test_verbose_False_changes_the_LOG_and_never_the_NUMBERS()
    test_the_abstention_SAYS_WHY_and_names_the_missing_vendor_input()
    test_the_CSV_reason_is_the_SAME_STRING_the_bucket_charged_on()
    test_the_gap_is_raised_ONCE_per_run_so_a_name_in_TWO_pools_is_not_charged_TWICE()
    test_the_gap_charge_is_raised_BEFORE_head_100_so_it_cannot_change_MEMBERSHIP()
    test_a_forensically_INVALID_name_is_NOT_charged_for_lacking_a_Beneish_score()
    test_a_partial_failure_commits_NOTHING_to_the_book()
    test_no_penalty_book_means_NO_CHARGE_and_no_crash()
    test_P1_a_SENTINEL_zero_DSO_period_does_not_reach_MontierC_DSOinc()
    test_P1_MontierC_reads_the_SAME_domain_dict_as_calcBeneishM()
    test_P1_the_guard_does_NOT_touch_a_name_whose_DSO_is_always_reported()
    test_P1_the_refusal_does_NOT_abstain_it_scores_the_name_CLEANER()
    test_P1_daysOfInventoryOutstanding_is_deliberately_NOT_guarded()
    test_the_C_score_is_DISPLAY_ONLY_and_cannot_reach_the_RANKING()
    test_P2_M_drivers_is_BLANK_on_a_row_that_has_no_M_score()
    test_P2_C_flags_fired_is_BLANK_when_there_is_no_C_score()
    test_P2_blanking_at_SOURCE_reaches_every_artifact_the_CEO_reads()
    print("\nALL detectManipulation KNOWN-ANSWER TESTS PASSED")
