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
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import detectManipulation as dm
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
    print("\nALL detectManipulation KNOWN-ANSWER TESTS PASSED")
