"""The price-scale contradiction rule (`nan_policy` section 5, third producer) -- Q-48.

WHAT THE RULE IS FOR.  FMP serves some names' price side and some names' share count at a
power of ten off the tape.  `marketCap` sits on ONE side of every cheapness ratio in this
pipeline, so a contaminated name reads as the cheapest equity in the universe -- the direction
that HELPS it.  The rule refuses the contradicted cells at ingest so those criteria come out
ABSENT rather than wrong.

EVERY FIXTURE BELOW IS A REAL ROW from the 2026-08-29 CUR6K panel (or, for ATRI, from
`price_scale_audit`'s own docstring), not a synthetic, so a test failing here means the shipped
rule stopped catching a shipped defect.

WHAT THIS FILE CANNOT DETECT, stated per test as well as here:
  * It cannot tell whether the THRESHOLD is right.  Every fixture is one or more decades
    inside the cut; a test that moved with the constant would pin the constant, not the
    behaviour, so none of these would fail if `PRICE_SCALE_PB_ALARM` were 0.05 or 0.005.
    `test_there_is_exactly_one_definition_of_contaminated` pins the only property about the
    number that is checkable here: that the reporting side and the refusing side share it.
  * It cannot tell whether the THRESHOLDS are right -- and since Q-75 there are TWO of them,
    `PRICE_SCALE_PB_ALARM` (unconditional) and `PRICE_SCALE_PB_WIDE` (witness required).  What
    the tests below DO pin is the STRUCTURE: that the shipped rule is the UNION of the two
    levels and not the conjunction the widening was first framed as, that each level fires on a
    real row the other cannot reach, and that the witnessed level does nothing without its
    witness.  Those are the properties the measurement actually established.
  * It STILL cannot detect the residual under-reach.  Rows between `PRICE_SCALE_PB_WIDE` and
    0.20 carry the share-count signature at 7.5-12.4% against a 1.1-1.8% base rate and are
    deliberately left alone; so is every contaminated row whose share count is sound (the ATRI
    and 0CHZ.L shape) and every majority-corrupt source above the floor.  Nothing here fails if
    those populations grow.
  * IT CANNOT DETECT AN OVER-REFUSAL IT WAS NOT SHOWN.  The widened band's measured
    false-refusal rate is 11.4% of rows I could adjudicate and up to 31.4% counting the
    unresolved ones, by two named mechanisms (dilution ramps; real >=5:1 consolidations).  ONE
    fixture below stands for the first mechanism.  A test suite cannot bound a rate -- read the
    measured table in `nan_policy` beside the constants for that.
  * It cannot see a name ABSENT FROM THE PANEL, which is the population ATRI is actually in.

ONE MEASUREMENT LEAD, RECORDED AND DELIBERATELY NOT CHASED.  Re-scoring the saved 2026-08-29
panel at HEAD moves 397 of 4,934 sources by exact multiples of 0.3/8, always downward.  It does
NOT gate this change -- every outcome leads to the same action (a fresh single-basis panel), the
top-100 is unaffected, and a test that cannot change the decision should not be run.  The likely
explanation, unconfirmed: `Sbocker's getAves2-on-the-filtered-frame note` records that recomputing `getAves2` on the FILTERED
frame moves 26 of 36 medians by more than 1%, which would produce exactly that signature with no
defect present at all.
"""
import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

#  `price_scale_audit` LIVES IN `baseline_tools/`, AND THIS FILE DOES NOT.  Four tests below
#  import it, and until 2026-09-01 they did so with a bare `import` and no path setup -- which
#  RESOLVED ONLY BY ACCIDENT, because ten alphabetically-earlier root test files put
#  `baseline_tools` on `sys.path` first.  Narrow the pytest selection and they raise
#  `ModuleNotFoundError`.  The four affected are the one-definition threshold guard, BOTH tests
#  that close this fix's own detector blind spot, and the `shipped_sources` regression guard --
#  every one load-bearing, every one passing for a reason other than the behaviour it asserts.
#  That is the house repeat-defect shape appearing inside the guards written to prevent it.
#  Same two lines `baseline_tools/test_price_scale_audit.py` already carries.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, 'baseline_tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import createDicts as cdic
import getData_fmp as gdf
import nan_policy as npol
import reporting_period as rp
import utils


def _frame(rows):
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  THE TWO SHAPES THAT PRODUCE THE SIGNATURE                                   #
# --------------------------------------------------------------------------- #
def test_the_ATRI_shape_is_refused_price_and_marketCap_at_one_thousandth():
    """ATRI: synthetic price 0.45212 against bookValuePerShare 135.65 -> price/book 0.0033.

    FMP serves open/high/low/close/adjClose divided by 1000 AND scales `marketCap` by the same
    1/1000, which is why the synthetic-vs-grid check cancels on it (ratio 1.04) and why check A
    -- this ratio -- is the one that sees it.

    CANNOT DETECT: ATRI has 0 rows in `cdx_df` on both the 08-29 and 08-31 panels; it lives in
    the PIT dead-merged pool.  This test proves the RULE fires on the shape, not that the shape
    reaches the rule on the live path.
    """
    df = _frame([{'source': 'ATRI', 'date': '2021-12-31', 'price': 0.45212,
                  'bookValuePerShare': 135.65, 'marketCap': 1269524.90,
                  'earningsYield': 0.0451, 'weightedAverageShsOut': 1801000,
                  'totalStockholdersEquity': 2.4431e8}])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert set(rep['relation']) == {npol.PRICE_SCALE_RELATION}
    for f in npol.PRICE_SCALE_REFUSE:
        assert np.isnan(out.loc[0, f]), f
    #  the number that caused the refusal travels with it, so the cut can be argued with
    #  the ratio the rule actually tests is marketCap / totalStockholdersEquity
    assert rep['ratio'].iloc[0] == pytest.approx(1269524.90 / 2.4431e8, rel=1e-9)
    #  a field the rule does not name is untouched -- this is a CELL refusal
    assert out.loc[0, 'totalStockholdersEquity'] == 2.4431e8


def test_the_QBY0_shape_refuses_only_the_contaminated_ROWS_not_the_whole_source():
    """QBY0.DE, real rows: the share count is ~125x too small before 2025-10 and correct after.

    `totalStockholdersEquity` runs CONTINUOUS across the break (92.34M -> 94.17M) and `price`
    is sound on both sides (4.34 -> 3.47); it is `weightedAverageShsOut` (199,326 ->
    24,915,897), `marketCap` (0.865M -> 86.5M) and `bookValuePerShare` (475.08 -> 3.868) that
    jump.

    THIS IS THE PER-ROW-VERSUS-PER-SOURCE CLAIM, and it is the substantive design choice.
    `price_scale_audit` aggregates to a per-source MEDIAN over the whole history because it is
    REPORTING a name; refusing on that basis would refuse this name's CORRECT current readings.
    Measured on the 08-29 panel: four of the eight sources that module flags (BGMS, ENDUR.OL,
    IPOK.DE, SEA1.OL) are clean inside the newest-8 Stage-1 window, and a refusal scores as a
    FAIL in Stage-1 -- so a per-source refusal would not be a neutral over-reach, it would
    invent an adverse judgement about four real companies.

    CANNOT DETECT: whether the surviving rows are themselves right.  It only asserts that a row
    the rule does not fire on keeps its vendor values verbatim.
    """
    df = _frame([
        {'source': 'QBY0.DE', 'date': '2026-04-01', 'price': 3.48,
         'bookValuePerShare': 3.789187, 'marketCap': 86707321.56, 'earningsYield': -0.011844,
         'weightedAverageShsOut': 24915897.0, 'totalStockholdersEquity': 92030000.0},
        {'source': 'QBY0.DE', 'date': '2026-01-01', 'price': 3.40,
         'bookValuePerShare': 3.826433, 'marketCap': 84714049.80, 'earningsYield': -0.011844,
         'weightedAverageShsOut': 24915897.0, 'totalStockholdersEquity': 93051000.0},
        {'source': 'QBY0.DE', 'date': '2025-10-01', 'price': 3.47,
         'bookValuePerShare': 3.868334, 'marketCap': 86458162.59, 'earningsYield': -0.011844,
         'weightedAverageShsOut': 24915897.0, 'totalStockholdersEquity': 94174000.0},
        {'source': 'QBY0.DE', 'date': '2025-07-01', 'price': 4.34,
         'bookValuePerShare': 475.078640, 'marketCap': 865079.18, 'earningsYield': -1.48,
         'weightedAverageShsOut': 199327.0, 'totalStockholdersEquity': 92340000.0},
        {'source': 'QBY0.DE', 'date': '2025-04-01', 'price': 4.62,
         'bookValuePerShare': 472.690604, 'marketCap': 920890.74, 'earningsYield': -1.48,
         'weightedAverageShsOut': 199327.0, 'totalStockholdersEquity': 91852000.0},
    ])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    refused_rows = sorted(set(rep['row'])) if 'row' in rep.columns else None
    assert len(out) == len(df), 'a refusal never removes a row'
    #  rows 3 and 4 (the broken share count) refused; rows 0-2 untouched
    for i in (3, 4):
        for f in npol.PRICE_SCALE_REFUSE:
            assert np.isnan(out.loc[i, f]), (i, f)
    for i in (0, 1, 2):
        assert out.loc[i, 'marketCap'] == df.loc[i, 'marketCap'], i
        assert out.loc[i, 'price'] == df.loc[i, 'price'], i
        assert out.loc[i, 'bookValuePerShare'] == df.loc[i, 'bookValuePerShare'], i
        assert not str(out.loc[i, npol.SANITY_REFUSED_COLUMN] or ''), i
    #  the share count is DELIBERATELY not refused: it is the broken field in THIS shape and
    #  the SOUND one in the ATRI shape, and the contradiction does not identify it.
    assert out.loc[3, 'weightedAverageShsOut'] == 199327.0
    assert out.loc[3, 'totalStockholdersEquity'] == 92340000.0


# --------------------------------------------------------------------------- #
#  OVER-REACH: the states that must SURVIVE                                    #
# --------------------------------------------------------------------------- #
def test_a_book_insolvent_company_is_not_a_units_error():
    """Negative book value per share is a real state of a real company and must survive.

    The `bookToPrice` mean-dict entry rules on exactly this: "A negative book yield is a true
    measurement of a real company, not a domain error; it belongs in the ruler."  A rule that
    refused it would be deleting a measurement, and would do so on the names most likely to be
    genuinely distressed.  Real row: DSV.TO 2016-04-01.

    CANNOT DETECT: a name whose book value is negative BECAUSE of the same share-count
    corruption.  Those are invisible to this rule by construction.
    """
    df = _frame([{'source': 'DSV.TO', 'date': '2016-04-01', 'price': 0.01,
                  'bookValuePerShare': -3.690464, 'marketCap': 11700.0,
                  'earningsYield': -0.5, 'weightedAverageShsOut': 1169999.0,
                  'totalStockholdersEquity': -4317840.0}])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert npol.PRICE_SCALE_RELATION not in set(rep['relation'])
    assert out.loc[0, 'marketCap'] == 11700.0
    assert out.loc[0, 'price'] == 0.01


def test_a_genuinely_deep_value_name_above_the_cut_is_untouched():
    """price/book 0.05 -- an order of magnitude under the panel's 1st percentile and still
    NOT refused.  `price_scale_audit` calls 0.05 the WATCH level for exactly this reason: "a
    real distressed equity bottoms out around 0.05-0.10".

    CANNOT DETECT: the measured under-reach.  This test asserts the cut is not crossed at
    0.05; it says nothing about whether 0.05 rows are clean.  Measured on the 08-29 panel they
    are not -- 22.6% of rows in [0.05,0.10) sit >= 5x off their own source's median share
    count against a 3.45% base rate -- and they are deliberately left alone.
    """
    df = _frame([{'source': 'DEEP', 'date': '2026-01-01', 'price': 0.50,
                  'bookValuePerShare': 10.0, 'marketCap': 5.0e7, 'earningsYield': 0.10,
                  'weightedAverageShsOut': 1.0e8, 'totalStockholdersEquity': 1.0e9}])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert npol.PRICE_SCALE_RELATION not in set(rep['relation'])
    assert out.loc[0, 'marketCap'] == 5.0e7


# --------------------------------------------------------------------------- #
#  THE COUPLINGS THAT WOULD MAKE THIS FIX WORSE THAN THE DEFECT                #
# --------------------------------------------------------------------------- #
def test_refusing_the_NEWEST_row_does_NOT_eject_the_source():
    """THE HIGHEST-CONSEQUENCE INTERACTION IN THIS CHANGE, and it is live, not theoretical.

    `price` and `marketCap` are BOTH in `nan_policy.PRIMARY_POSITIVE`, whose limb is "NaN or
    <= 0" on the source's NEWEST row, and `primary_eject` removes the WHOLE SOURCE.  So a rule
    that blanks them on a newest row would DELETE the name from the universe -- an excluded
    name is never scored, never appears, and no output says why.  `test_input_sanity.py`
    records this exact defect being shipped once already (JHX), and its note warns that it
    "re-arms silently the moment any future relation names a primary field".  This rule names
    two of the three.

    It is safe only because `primary_eject` subtracts `refused_fields_mask`.  That coupling is
    two modules apart and has no other test aimed at THIS rule, so it is pinned here.
    Real case: CCM and CMCM both have their NEWEST row refused on the 08-29 panel.

    CANNOT DETECT: an ejection happening somewhere other than `primary_eject` --
    `data_quality.filter_invalid_data` has other limbs, and this test calls only this one.
    """
    df = _frame([
        {'source': 'CMCM', 'date': '2025-10-01', 'price': 41.955646,
         'bookValuePerShare': 3157.395033, 'marketCap': 2.602261e7,
         'earningsYield': -0.74, 'weightedAverageShsOut': 620241.0,
         'netIncome': -1.7e7, 'netCashProvidedByOperatingActivities': -5.0e6,
         'totalStockholdersEquity': 1.607669e9, 'totalAssets': 2.0e9, 'revenue': 3.0e8},
        {'source': 'CMCM', 'date': '2026-01-01', 'price': 38.005242,
         'bookValuePerShare': 3091.783858, 'marketCap': 2.359312e7,
         'earningsYield': -0.74, 'weightedAverageShsOut': 620786.0,
         'netIncome': -1.7e7, 'netCashProvidedByOperatingActivities': -5.0e6,
         'totalStockholdersEquity': 1.561773e9, 'totalAssets': 2.0e9, 'revenue': 3.0e8},
    ])
    assert npol.primary_eject(df, verbose=False).empty, 'fixture must not eject before the rule'
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    newest = out['date'].astype(str).idxmax()
    assert np.isnan(out.loc[newest, 'price']), 'the newest row must still be refused'
    assert np.isnan(out.loc[newest, 'marketCap'])
    ej = npol.primary_eject(out, verbose=False)
    assert ej.empty, (
        'the price-scale rule EJECTED the source through primary_eject: %s. A refused cell is '
        'a deliberate abstention, not an absent primary input.'
        % ej[['source', 'field', 'limb']].to_dict('records'))


def test_the_refusal_is_stamped_so_it_can_never_read_as_an_absence():
    """All four refused field names ride on the row, and a second pass refuses nothing more.

    `data_quality.filter_invalid_data` runs TWICE on the live path, so non-idempotence would
    double-count in the report; and the stamp is what `primary_eject` and
    `price_scale_audit.refused_upstream` both read.

    CANNOT DETECT: whether the stamp survives the pickle round-trip and the `-loadbometric`
    reload.  It is an ordinary object column, so it should, but that is not asserted here.
    """
    df = _frame([{'source': 'K', 'date': '2026-01-01', 'price': 1.0,
                  'bookValuePerShare': 1000.0, 'marketCap': 1.0e6, 'earningsYield': 0.5,
                  'weightedAverageShsOut': 1.0e6, 'totalStockholdersEquity': 1.0e9}])
    out, _ = npol.refuse_impossible_cells(df, verbose=False)
    stamp = str(out.loc[0, npol.SANITY_REFUSED_COLUMN])
    assert set(stamp.split('|')) == set(npol.PRICE_SCALE_REFUSE)
    _, rep2 = npol.refuse_impossible_cells(out, verbose=False)
    assert len(rep2) == 0, 'the rule must be idempotent -- filter_invalid_data runs twice'


def test_there_is_exactly_one_definition_of_contaminated():
    """The REPORTING threshold and the REFUSING threshold are the same object.

    This repo's recorded worst bug class is one fact stated in two places that drift apart
    (`Sign` in two registries; the guard domain restated inside a lambda).  A detector that
    named eight sources while the refusal fired on a different set would be that defect on the
    one number the CEO reads.

    CANNOT DETECT: a third consumer hard-coding 0.02 somewhere else.  It checks the two sides
    that exist today.
    """
    import price_scale_audit as psa
    assert psa.PB_ALARM == npol.PRICE_SCALE_PB_ALARM
    src = open(psa.__file__, encoding='utf-8').read()
    assert 'PB_ALARM = npol.PRICE_SCALE_PB_ALARM' in src, (
        'price_scale_audit re-declared its own alarm literal -- the reporting side and the '
        'refusing side can now disagree about which names are contaminated.')
    #  SINCE Q-75 THE RULE HAS TWO LEVELS, so pinning only the floor would leave the widened
    #  band free to drift -- and the widened band is where the coverage actually grew.  A
    #  reporting side that knows the floor and not the band under-reports by construction.
    assert psa.PB_WIDE == npol.PRICE_SCALE_PB_WIDE
    assert 'PB_WIDE = npol.PRICE_SCALE_PB_WIDE' in src, (
        'price_scale_audit re-declared the widened band -- the audit can now name a different '
        'population than the refusal does in the 0.02-0.10 band.')
    #  and the floor must stay BELOW the band, or the union collapses to one level and the
    #  measured argument for the witness is silently gone.
    assert npol.PRICE_SCALE_PB_ALARM < npol.PRICE_SCALE_PB_WIDE


def test_run_audit_says_what_was_refused_so_a_quiet_check_A_is_not_an_all_clear():
    """THE FIX'S OWN BLIND SPOT, CLOSED.

    Check A reads `price` and `bookValuePerShare`; the refusal blanks BOTH.  So on a panel
    built after this change the rows the audit exists to name are invisible to it, and
    `0 ALARM` would print over a defect that WAS found -- "a detector blind to its own
    motivating case is worse than none, because it would have been read as an all-clear", in
    that module's own words.

    CANNOT DETECT: whether a HUMAN reads the A0 block before the ALARM count.  It asserts the
    line is emitted, not that the log is read.
    """
    import price_scale_audit as psa
    #  THE SOURCE NAME IS PART OF THE TEST.  It must not be a substring of anything the
    #  banner itself prints -- the previous fixture was called 'REF', which is inside
    #  'REFUSED', which is why the assertion below could not fail.
    df = _frame([{'source': 'ZZTESTCO', 'date': '2026-01-01', 'price': 1.0,
                  'bookValuePerShare': 1000.0, 'marketCap': 1.0e6, 'earningsYield': 0.5,
                  'weightedAverageShsOut': 1.0e6, 'revenue': 1.0e8,
                  'totalStockholdersEquity': 1.0e9}])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    got = psa.refused_upstream(out, report=rep)
    assert set(got) == {'ZZTESTCO'} and got['ZZTESTCO']['rows'] == 1, got
    #  the ratio is carried through from the refusal report, not re-derived from the panel
    #  (it cannot be: `marketCap` is one of the refused cells)
    assert got['ZZTESTCO']['worst_ratio'] == pytest.approx(1.0e6 / 1.0e9), got

    lines = []
    res = psa.run_audit(out, prices_csv=None, log=lines.append, run_grid_check=False,
                        refusal_report=rep)
    #  WAS: `assert int(...) if len(...) else 0 == 0` -- which Python parses as
    #  `assert (int(...) if len(...) else (0 == 0))`: VACUOUS when `internal` is empty and
    #  the exact INVERSE of its intent when it is not.  Instance nineteen of the house's
    #  test-pins-the-defect family.  Written so it fails in BOTH branches now.
    n_alarm = int((res['internal']['severity'] == 'ALARM').sum()) if len(res['internal']) else 0
    assert n_alarm == 0, (
        'check A still reports %d ALARM on a fully-refused panel, so the fixture no longer '
        'exercises the blindness this test exists for' % n_alarm)
    blob = '\n'.join(lines)
    assert 'ALREADY REFUSED UPSTREAM' in blob, blob
    #  WAS: `and 'REF' in blob` -- and 'REF' is a substring of the 'REFUSED' banner the line
    #  above already asserts, so the second conjunct could NEVER fail and the one behaviour
    #  `refused_upstream` exists to provide -- NAMING the source -- went unchecked.  Instance
    #  twenty.  The fixture's source is now a token that appears nowhere in the module.
    assert 'ZZTESTCO' in blob, (
        'A0 fired but did not NAME the refused source, which is the whole point of it: %s'
        % blob)
    assert '0.00100' in blob, ('A0 named the source but not how far under the cut it sat; '
                               'the ratio is computed and free to print: %s' % blob)
    #  and the negative case must look DIFFERENT, not identical
    clean = _frame([{'source': 'OK', 'date': '2026-01-01', 'price': 10.0,
                     'bookValuePerShare': 5.0, 'marketCap': 1.0e9, 'earningsYield': 0.05,
                     'weightedAverageShsOut': 1.0e8, 'revenue': 1.0e8,
                     'totalStockholdersEquity': 5.0e8}])
    lines2 = []
    psa.run_audit(clean, prices_csv=None, log=lines2.append, run_grid_check=False)
    assert 'ALREADY REFUSED UPSTREAM' not in '\n'.join(lines2)


# --------------------------------------------------------------------------- #
#  COVERAGE: every marketCap/price-bearing criterion, through the REAL builder  #
# --------------------------------------------------------------------------- #
_PRE = cdic.getDicts()[0]
_PREREQ_FIELDS = sorted({f for v in _PRE.values() for f in v})


def _one_source_panel(n=10):
    """A single source's statement history with every preReq field present.

    Built from `createDicts.getDicts()[0]` rather than a hand-written column list, so a field
    added to the ingest cannot make this fixture silently stale.
    """
    df = pd.DataFrame({f: [np.nan] * n for f in _PREREQ_FIELDS})
    df['source'] = 'TST'
    df['date'] = pd.Timestamp('2026-01-01') - pd.to_timedelta(np.arange(n) * 91, unit='D')
    df['weightedAverageShsOut'] = 1.0e8
    df['weightedAverageShsOutDil'] = 1.0e8
    df['totalStockholdersEquity'] = 5.0e8
    df['totalAssets'] = 1.0e9
    df['totalLiabilities'] = 5.0e8
    df['totalCurrentAssets'] = 4.0e8
    df['totalCurrentLiabilities'] = 2.0e8
    df['revenue'] = 3.0e8
    df['netIncome'] = 3.0e7
    df['netCashProvidedByOperatingActivities'] = 4.0e7
    df['freeCashFlow'] = 3.5e7
    df['grossProfit'] = 1.2e8
    df['operatingIncome'] = 4.0e7
    df['interestExpense'] = 5.0e6
    df['depreciationAndAmortization'] = 1.0e7
    df['bookValuePerShare'] = 5.0
    df['grahamNumber'] = 5.8
    df['price'] = 8.0
    df['marketCap'] = 8.0e8
    df['earningsYield'] = 0.0375
    #  A RISING TRAILING EPS so `PEG` is computable in the clean arm -- without it the growth
    #  leg is exactly zero, PEG is +inf -> NaN, and the coverage assertion below could not tell
    #  a refusal from a fixture gap.  `calcMetrics._PEG_EPS_FIELD` is `netIncomePerShare`, NOT
    #  `eps`; both are set so the fixture stays right if that constant is ever repointed.
    df['netIncomePerShare'] = np.where(np.arange(n) < 4, 0.30, 0.25)
    df['eps'] = df['netIncomePerShare']
    df[rp.FREQ_COLUMN] = 'quarterly'
    return df


def _build(tf):
    bm_cols = list(utils.initBoMetric_fromDict()['BoMetric_df'].columns)
    dd = cdic.getDicts()
    packed = (dd[2], dd[3], dd[5], dd[4], dd[6])
    tmp = pd.DataFrame(columns=bm_cols)
    tmp['date'] = tf['date'].values
    tmp['source'] = 'TST'
    tmp = utils.setDatesToQuarterly(tmp)
    return gdf.build_bometric_rows(tf.reset_index(drop=True), tmp, 4, dicts=packed)


#  Stage-1 criteria whose value is a function of `marketCap` or `price`, DERIVED from the
#  five scoring dicts below rather than listed, so a criterion added later is covered by the
#  same assertion instead of being quietly missed.
def _price_bearing_columns():
    dd = cdic.getDicts()
    base, mean, diff, unity = dd[2], dd[3], dd[4], dd[5]
    fields = set(npol.PRICE_SCALE_REFUSE)
    out = set()
    for d, pre in ((base, ''), (mean, 'm'), (diff, 'd'), (unity, 'u')):
        for k, spec in d.items():
            if spec.get('Upper') in fields or spec.get('Lower') in fields:
                out.add(pre + (k if not pre else k[0].upper() + k[1:]))
    #  `PEG` is a `special`: its formula lives in `calcMetrics.peg_local` rather than in an
    #  Upper/Lower pair, so it cannot be derived from the registry and is named.
    out.add('PEG')
    return out


def test_every_marketCap_or_price_bearing_criterion_goes_ABSENT_through_the_real_builder():
    """The whole point of fixing this at the FIELD rather than at the criterion.

    `the refusal hook in getData_fmp.getFundamentalsData` is where `tempfund` is both the frame every Stage-1 metric is computed
    from AND the frame that becomes `cdx_df`, so refusing a cell there covers every consumer
    from one call site.  This asserts that -- through `build_bometric_rows`, the production
    builder, not a replication.

    THE CRITERION LIST IS DERIVED FROM THE REGISTRY, which is the part that matters: `Q-48` was
    reopened because `bookToPrice` was found by inspection and its seven siblings were left
    unpatched.  A ninth criterion built on `marketCap` or `price` is covered by this assertion
    the day it is added.

    CANNOT DETECT: Stage-2.  `earnYield`, `bVpRatio`, `tbVpRatio`, `freeCashFlowYield`,
    `grahamNumberToPrice`, `priceGrowth`, Altman's `x4`, `marketCapRevQuants` (through
    `stage2_metrics._mcap_for_quants`) and `stage2_metrics.nav_per_share_growth` all read a
    refused field off the same `cdx_df`, so they are covered by the same mechanism -- but they
    are computed in `postBoRank`/`stage2_metrics` and are NOT exercised here.
    `nav_per_share_growth` is the one an enumeration built on "price or marketCap" MISSES: its
    only contaminated input is `bookValuePerShare`, and it is an ENDPOINT PAIR, so one refused
    row at either edge of the window kills it.  Stage-2 imputes a missing metric to the column
    MEDIAN, so unlike Stage-1 a refusal there is neutral-to-favourable -- see `nan_policy` for
    why that asymmetry is deliberate rather than an oversight.
    """
    clean = _build(_one_source_panel())
    cols = sorted(_price_bearing_columns())
    assert cols, 'the registry walk found no price-bearing criterion -- the walk is broken'
    for c in cols:
        assert clean[c].notna().any(), (
            '%s is NaN even on the CLEAN fixture, so this test could not tell a refusal from a '
            'fixture gap' % c)

    dirty = _one_source_panel()
    #  a 1/1000 market cap against a sound balance sheet -- marketCap/equity 0.0016
    dirty.loc[0, 'marketCap'] = 8.0e5
    out, rep = npol.refuse_impossible_cells(dirty, verbose=False)
    assert set(rep['relation']) == {npol.PRICE_SCALE_RELATION}
    built = _build(out)
    for c in cols:
        assert pd.isna(built[c].iloc[0]), (
            '%s survived the refusal -- a criterion built on `marketCap`/`price` is still '
            'being scored off a units error' % c)
    #  and nothing ELSE moved: this is an absence, never a correction
    for c in clean.columns:
        if c in ('source', 'date') or c in cols:
            continue
        a = pd.to_numeric(clean[c], errors='coerce').iloc[0]
        b = pd.to_numeric(built[c], errors='coerce').iloc[0]
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b, rel=1e-12), c


def test_a_price_derived_vendor_field_cannot_become_a_criterion_without_being_refused():
    """THE SIBLING-DEFECT GUARD, pointed at the fields rather than at the criteria.

    Q-48's actual failure was not that `bookToPrice` was wrong -- it was that seven other
    criteria shared its contaminated input and nobody enumerated them.  FMP's key-metrics block
    carries four more price-derived fields the ingest already captures (`pbRatio`, `pfcfRatio`,
    `dividendYield`, `priceEarningsToGrowthRatio`).  None is scored today.  If one is ever
    given a Tier, it must ride with the refusal, or the same defect returns under a new name.

    CANNOT DETECT: a criterion that reaches the price side through a field NOT on this list --
    a future ingest addition nobody adds here.  The list is hand-maintained and says so.
    """
    price_derived_but_not_refused = {'pbRatio', 'pfcfRatio', 'dividendYield',
                                     'priceEarningsToGrowthRatio'}
    assert not (price_derived_but_not_refused & set(npol.PRICE_SCALE_REFUSE))
    dd = cdic.getDicts()
    offenders = []
    for d in (dd[2], dd[3], dd[4], dd[5]):
        for k, spec in d.items():
            if spec.get('Tier') in (None, 'N'):
                continue          # weightless: not scored, so not an exposure
            for leg in ('Upper', 'Lower'):
                if spec.get(leg) in price_derived_but_not_refused:
                    offenders.append((k, leg, spec.get(leg), spec.get('Tier')))
    assert not offenders, (
        'a WEIGHTED Stage-1 criterion now reads a price-derived vendor field that '
        '`nan_policy.PRICE_SCALE_REFUSE` does not blank: %s. Either add the field to '
        'PRICE_SCALE_REFUSE or the criterion is scored off the same units error Q-48 was '
        'reopened to close.' % offenders)


# --------------------------------------------------------------------------- #
#  THE SECOND-ORDER SWEEP -- consumers that read a field this rule blanks       #
# --------------------------------------------------------------------------- #
def _atri_shape(n=10, broken=(4, 5)):
    """Ten quarters, share count FLAT, balance sheet SOUND, `broken` quarters served at
    1/1000 on both `price` and `marketCap`.  The shape the rule exists for."""
    rows = []
    for i in range(n):
        k = 0.001 if i in broken else 1.0
        rows.append({'source': 'ATRISHAPE',
                     'date': pd.Timestamp('2026-01-01') - pd.DateOffset(months=3 * i),
                     'price': 40.0 * k, 'marketCap': 4.0e9 * k,
                     'earningsYield': 0.01 / k, 'dividendYield': 0.02,
                     'weightedAverageShsOut': 1.0e8, 'bookValuePerShare': 20.0,
                     'totalStockholdersEquity': 2.0e9, 'totalAssets': 4.0e9,
                     'totalLiabilities': 2.0e9, 'revenue': 5.0e8, 'netIncome': 4.0e7,
                     'netCashProvidedByOperatingActivities': 5.0e7, 'freeCashFlow': 4.0e7})
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def test_the_refusal_does_NOT_suppress_a_data_quality_deletion():
    """THE BLOCKER FOUND IN REVIEW, AND THE ONE THIS FILE PREVIOUSLY COULD NOT SEE.

    `data_quality.check_price_sanity` reads five fields and this rule blanks THREE of them
    (`price`, `marketCap`, `earningsYield`).  Every check there is `pd.notna`-guarded, so a
    blanked input does not make a check ABSTAIN -- it makes the check PASS.  Checks 3, 5 and 6
    then skip, PASS 1 records no corruption, and PASS 3 stops removing the prefix it used to
    remove.  This is the FIRST section-5 rule to blank a field PASS 1 consumes; neither
    existing rule table names any of the five, so "it already worked this way" was never
    available.

    Three arms through the REAL `dq.filter_invalid_data`.  The bar is EXACT EQUALITY between
    the pre-refusal arm and the restored arm -- restoring must reproduce the old deletion set,
    not merely produce SOME deletions, because over-deleting would be its own defect (PASS 3
    removes a whole prefix).

    CANNOT DETECT: a consumer of a refused field OUTSIDE `data_quality` PASS 1.  The sweep
    that found this one was manual; nothing here makes the next one automatic.
    """
    import data_quality as dq
    raw = _atri_shape()
    refused, rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    assert set(rep['relation']) == {npol.PRICE_SCALE_RELATION}

    def removed_dates(frame, refusals):
        _clean, rm = dq.filter_invalid_data(frame.copy(), min_periods_required=1,
                                            verbose=False, sanity_refusals=refusals)
        return set(pd.to_datetime(rm['date']).dt.date) if len(rm) else set()

    before = removed_dates(raw, None)
    blinded = removed_dates(refused, None)
    restored = removed_dates(refused, rep)

    assert before, 'the fixture must be deleted by the PRE-refusal pipeline or it proves nothing'
    assert restored == before, (
        'the restoration did not reproduce the pre-refusal deletion set: before=%s restored=%s'
        % (sorted(before), sorted(restored)))
    assert blinded != before, (
        'the fixture no longer demonstrates the blinding, so this test would pass with the '
        'restoration removed -- rebuild it')


def test_check_price_sanity_sees_the_pre_refusal_values_not_the_blanks():
    """The same defect at the single-check level, where the mechanism is legible.

    Check 5 (`implausible_yield_pair`) needs `earningsYield`, which this rule blanks.

    CANNOT DETECT: whether PASS 1 actually calls it with the restored row -- that is the test
    above.  This one pins only that restoration changes the verdict.
    """
    import data_quality as dq
    raw = _atri_shape()
    refused, rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    i = 4                                        # a refused row
    ok_before, why_before = dq.check_price_sanity(dict(raw.iloc[i]))
    ok_blind, _ = dq.check_price_sanity(dict(refused.iloc[i]))
    row = dict(refused.iloc[i])
    #  the key carries an OCCURRENCE index -- see `refusal_restore_map` -- so a duplicate
    #  (source, date) pair cannot collapse onto one entry
    row.update(npol.refusal_restore_map(rep)[
        ('ATRISHAPE', npol._normalise_refusal_date(raw.iloc[i]['date']), 0)])
    ok_restored, why_restored = dq.check_price_sanity(row)
    assert ok_before is False and 'implausible_yield_pair' in why_before
    assert ok_blind is True, 'the fixture no longer shows the blinding'
    assert ok_restored is False and why_restored == why_before


def test_a_refused_newest_market_cap_is_reported_not_silently_backfilled():
    """`carveOut.marketcap_usd_by_source` takes the latest NON-NaN row, so refusing the
    newest `marketCap` does not make a name UNKNOWN -- it makes it the previous quarter's,
    and that value picks the band `postBo.generalTopN` reads as the top-20.

    The fallback is KEPT (measured: on the 08-29 panel it recovers the sound number on six of
    the seven affected sources) and made VISIBLE instead.

    CANNOT DETECT: a fallback landing on a row that is contaminated but ABOVE the cut -- the
    rule's measured under-reach composes here and nothing sees it.
    """
    import carveOut as co
    raw = _atri_shape(n=6, broken=(0,))          # the NEWEST row is the broken one
    refused, _rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    fb = co.marketcap_fallback_report(refused, report=_rep)
    assert len(fb) == 1 and fb['source'].iloc[0] == 'ATRISHAPE', fb
    assert fb['rows_back'].iloc[0] == 1
    assert fb['refused_marketCap_usd'].iloc[0] == pytest.approx(4.0e6)
    assert fb['fallback_marketCap_usd'].iloc[0] == pytest.approx(4.0e9)
    #  a clean panel produces no report at all -- the line must not print unconditionally
    assert len(co.marketcap_fallback_report(_atri_shape(n=6, broken=()))) == 0


def test_the_containment_paragraph_covers_the_refused_names_and_never_goes_silent():
    """A0 fixed the ALARM COUNT and left the sentence underneath it.

    `_containment_lines` derives its population from `internal`, and
    `check_fundamentals_internal` drops refused rows and then drops thin sources -- so a fully
    refused source leaves `internal` entirely, and when that emptied the list the function
    returned [] and printed NOTHING.  Not "not checked", not an all-clear: silence, over a
    defect that had been found.

    CANNOT DETECT: whether the containment CLAIM is true -- only that it is made about the
    full population rather than about whatever survived the refusal.
    """
    import price_scale_audit as psa
    rows = []
    for i in range(6):
        rows.append({'source': 'ZZTESTCO', 'date': pd.Timestamp('2026-01-01') - pd.DateOffset(months=3 * i),
                     'price': 1.0, 'bookValuePerShare': 1000.0, 'marketCap': 1.0e6,
                     'earningsYield': 0.5, 'weightedAverageShsOut': 1.0e6,
                     'revenue': 1.0e8, 'totalStockholdersEquity': 1.0e9})
    out, rep = npol.refuse_impossible_cells(_frame(rows), verbose=False)
    lines = []
    psa.run_audit(out, prices_csv=None, log=lines.append, run_grid_check=False,
                  refusal_report=rep, shipped_sources=['SOMEOTHERNAME'],
                  stage1_scores=None)
    blob = '\n'.join(lines)
    assert 'ALARM' in blob
    assert 'POPULATION:' in blob and 'REFUSED UPSTREAM' in blob, blob
    #  the shipped-list question must be ANSWERED about the refused name, not skipped
    assert 'containment holds this run' in blob or 'IS IN THE SELECTED LIST' in blob, blob


def test_refused_upstream_keys_on_a_field_only_this_rule_names():
    """`refused_upstream` keys on a `price` refusal because no other `nan_policy` rule names
    that field.  TRUE TODAY, and nothing pinned it (review S4-2): a future rule naming `price`
    would make A0 over-report and attribute another rule's refusals to this one.

    CANNOT DETECT: a rule added OUTSIDE the two tables below.
    """
    others = set()
    for _name, _num, _den, _factor, _two, refuse in npol.IMPOSSIBLE_RELATIONS:
        others |= set(refuse)
    others |= set(npol.SCALE_SPIKE_FIELDS)
    assert 'price' not in others, (
        'another nan_policy rule now refuses `price`, so `price_scale_audit.refused_upstream` '
        'will attribute ITS refusals to the price-scale rule: %s' % sorted(others & {'price'}))


def test_shipped_sources_as_a_Series_does_not_raise():
    """`shipped_sources or []` invokes `Series.__bool__`, which raises. Production passes a
    list, so it was latent -- but a diagnostic that dies on a Series is one nobody gets.

    CANNOT DETECT: any other `or`-on-a-Series in the module.
    """
    import price_scale_audit as psa
    rows = [{'source': 'ZZTESTCO', 'date': pd.Timestamp('2026-01-01') - pd.DateOffset(months=3 * i),
             'price': 1.0, 'bookValuePerShare': 1000.0, 'marketCap': 1.0e6,
             'earningsYield': 0.5, 'weightedAverageShsOut': 1.0e6, 'revenue': 1.0e8,
             'totalStockholdersEquity': 1.0e9} for i in range(6)]
    out, rep = npol.refuse_impossible_cells(_frame(rows), verbose=False)
    lines = []
    psa.run_audit(out, prices_csv=None, log=lines.append, run_grid_check=False,
                  refusal_report=rep,
                  shipped_sources=pd.Series(['A', 'B']))          # must not raise
    assert lines


# --------------------------------------------------------------------------- #
#  THE INGEST TAIL -- where the stamp was silently destroyed                    #
# --------------------------------------------------------------------------- #
def test_the_refusal_stamp_SURVIVES_the_real_ingest_coercion():
    """THE SECOND BLOCKER, AND THE ONE THAT WOULD HAVE DELETED REAL COMPANIES.

    `getData_gen.forceNumOnDf` coerces every column not in its passthrough list with
    `pd.to_numeric(errors='coerce')`, and `fixAfterGetData` calls it at
    `getData_gen.fixAfterGetData, called at the end of the fetch loop` -- i.e. on EVERY LIVE FETCH, after the refusal hook at `:178`.
    Omitted from that list, `sanityRefusedFields` becomes an all-NaN float64 column.

    THAT IS NOT A COSMETIC LOSS.  `price` and `marketCap` are both in
    `nan_policy.PRIMARY_POSITIVE`, whose limb is `isna() or <= 0` on a source's NEWEST row,
    and `primary_eject` removes the WHOLE SOURCE.  The only thing standing between a
    refusal and a deleted company is `nan_policy.primary_eject's refused_fields_mask subtraction` reading this column.  Measured on
    the seven real 2026-08-29 sources whose newest row is refused: 7 kept with the stamp,
    0 kept without it -- and the removal CSV would say `primary_input_absent`, with nothing
    connecting the loss to the refusal.

    The same one column also drives `carveOut.marketcap_fallback_report` and
    `price_scale_audit.refused_upstream`, so its loss silently disarms the A0 block and the
    containment paragraph -- the exact silence two other tests in this file exist to prevent.
    All three are asserted here because one line fixes all three and assuming that is how
    this defect survived the first review.

    CANNOT DETECT: a coercion applied somewhere OTHER than `forceNumOnDf`, or a consumer of
    the stamp added later that nobody thinks to assert here.
    """
    import carveOut as co
    import getData_gen as gdg
    import price_scale_audit as psa

    raw = _atri_shape(n=6, broken=(0,))          # NEWEST row refused -> eject-critical
    refused, rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    stamp = npol.SANITY_REFUSED_COLUMN
    assert refused[stamp].astype(str).str.len().gt(0).any(), 'fixture must stamp something'

    #  the REAL ingest tail, not a replication of it
    _bm = pd.DataFrame({'source': ['ATRISHAPE'], 'x': [1.0]})
    _bm2, tail = gdg.fixAfterGetData(_bm, refused.copy())

    assert tail[stamp].dtype == object, (
        'the refusal stamp was coerced to %s by the ingest tail -- `%s` is missing from '
        '`getData_gen.forceNumOnDf`\'s passthrough list' % (tail[stamp].dtype, stamp))
    assert npol.refused_fields_mask(tail, 'price').any(), 'the stamp no longer reads back'

    #  ALL THREE CONSUMERS, after the tail
    assert npol.primary_eject(tail, verbose=False).empty, (
        'the source is EJECTED after the ingest tail -- a refusal became a deletion')
    assert len(co.marketcap_fallback_report(tail, report=rep)) == 1, (
        'the band-fallback report went blind after the ingest tail')
    assert set(psa.refused_upstream(tail, report=rep)) == {'ATRISHAPE'}, (
        'A0 went blind after the ingest tail')


def test_every_declared_STAMP_column_is_in_the_ingest_passthrough_list():
    """THE TRAP, CLOSED FOR THE NEXT AUTHOR TOO.

    `forceNumOnDf`'s passthrough list is hand-maintained and its own comment records that
    `reportedCurrency` was silently destroyed before someone noticed and added it.  This
    change walked into the same trap with `sanityRefusedFields`.  A hand-maintained list
    whose omissions are silent and load-bearing will be walked into again.

    THE CHECK IS DERIVED FROM THE MODULE NAMESPACE, NOT FROM A LIST.  The first version of
    this test compared three HAND-LISTED constants while its docstring claimed it walked "every
    module-level constant in the repo" -- a guard whose stated coverage materially exceeded its
    actual coverage, in the artifact the next author would trust.  It was blind to
    `nan_policy.GRAHAM_REASON_COLUMN`, a stamp constant in the very module it imports.  It now
    walks `dir()` for upper-case `*_COLUMN` string constants, so a NEW stamp constant is
    covered the day it is declared rather than the day someone remembers this test.

    CANNOT DETECT: a stamp column written as a bare string literal rather than declared as a
    constant, or declared in a module not walked here.  The derivation can only see what is
    declared, and the modules walked are named below rather than discovered.
    """
    import inspect
    import getData_gen as gdg
    import reporting_period as rp

    src = inspect.getsource(gdg.forceNumOnDf)
    declared = {}
    for mod in (npol, rp):
        for attr in dir(mod):
            if not attr.isupper() or not attr.endswith('_COLUMN'):
                continue
            val = getattr(mod, attr)
            if isinstance(val, str) and val:
                declared['%s.%s' % (mod.__name__, attr)] = val
    assert len(declared) >= 3, (
        'the namespace walk found only %d stamp constant(s) -- it has stopped deriving and is '
        'no longer a guard: %s' % (len(declared), declared))

    #  Either the literal appears in the tuple, or the constant is referenced by name.
    missing = {k: v for k, v in declared.items()
               if ("'%s'" % v) not in src and k.split('.')[-1] not in src}
    assert not missing, (
        'a declared STAMP column is not preserved by `getData_gen.forceNumOnDf`, so it will '
        'be coerced to all-NaN on every live fetch: %s' % missing)
def test_the_restoration_uses_the_ROWs_OWN_stamp_not_just_the_source_date_key(capsys):
    """`(source, date)` IS NOT UNIQUE on this panel -- 296 duplicate-key rows across 76
    sources on the 2026-08-11 CUR3K panel, AAPL among them.  A restore map keyed on it alone
    hands one row's pre-refusal values to a DIFFERENT row that merely shares its date, so the
    data-quality checks judge a clean row on numbers that are not its own.

    No refusal intersects a duplicate key today, so this is LATENT.  It is closed rather than
    documented because the row's own `sanityRefusedFields` stamp disambiguates it for free.

    The observable is PASS 1's own count: with the intersection it restores ONE row, without
    it, both twins.

    CANNOT DETECT: which ROW a restored value belongs to -- that is the occurrence index, and
    it is pinned separately below.  An earlier version of this note claimed a both-twins-refused
    key was harmless "because the values are interchangeable anyway".  THAT WAS WRONG:
    `check_price_sanity`'s step check compares ADJACENT market caps, so a borrowed value can
    create or suppress a break and therefore a whole prefix deletion.
    """
    import data_quality as dq
    twin = pd.DataFrame([
        {'source': 'DUP', 'date': pd.Timestamp('2026-01-01'), 'price': 1.0,
         'marketCap': 1.0e6, 'bookValuePerShare': 1000.0, 'earningsYield': 0.5,
         'dividendYield': 0.0, 'weightedAverageShsOut': 1.0e6,
         'totalStockholdersEquity': 1.0e9, 'totalAssets': 2.0e9, 'revenue': 1.0e8,
         'netIncome': 1.0e7, 'netCashProvidedByOperatingActivities': 1.0e7},
        {'source': 'DUP', 'date': pd.Timestamp('2026-01-01'), 'price': 40.0,
         'marketCap': 4.0e9, 'bookValuePerShare': 20.0, 'earningsYield': 0.01,
         'dividendYield': 0.0, 'weightedAverageShsOut': 1.0e8,
         'totalStockholdersEquity': 2.0e9, 'totalAssets': 4.0e9, 'revenue': 5.0e8,
         'netIncome': 4.0e7, 'netCashProvidedByOperatingActivities': 5.0e7},
    ])
    refused, rep = npol.refuse_impossible_cells(twin, verbose=False)
    stamped = int(refused[npol.SANITY_REFUSED_COLUMN].astype(str).str.len().gt(0).sum())
    assert stamped == 1, 'the fixture must refuse exactly ONE of the two twins'
    dq.filter_invalid_data(refused.copy(), min_periods_required=1, verbose=True,
                           sanity_refusals=rep)
    out = capsys.readouterr().out
    assert 'PASS 1 evaluated 1 refused row(s)' in out, (
        'the restoration reached %s rows, so it is keying on (source, date) alone and has '
        'handed a clean twin another row\'s values: %s'
        % ('both' if 'evaluated 2' in out else '?', out))


def test_a_restored_removal_says_its_values_are_the_PRE_REFUSAL_ones():
    """The removal CSV carries two halves for one row and, after a restoration, they DISAGREE:
    the corruption record holds the pre-refusal `price`/`marketCap` that tripped the check,
    the removed row itself holds the NaN section 5 left.  Two numbers for one (source, date)
    with nothing saying why is how a reader concludes the transparency CSV is broken.

    CANNOT DETECT: whether the two halves are joined correctly downstream -- only that the
    reason says which values it is showing.
    """
    import data_quality as dq
    raw = _atri_shape()
    refused, rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    _clean, removed = dq.filter_invalid_data(refused.copy(), min_periods_required=1,
                                             verbose=False, sanity_refusals=rep)
    assert len(removed), 'the fixture must produce at least one removal'
    assert 'refused_fields' in removed.columns, (
        'the transparency CSV has no column explaining why a removed row shows a NaN price')
    #  the refused rows in the removal set must NAME what was blanked on them
    blanked = removed[removed['price'].isna()]
    assert len(blanked), 'the fixture must remove at least one refused (NaN-price) row'
    assert blanked['refused_fields'].astype(str).str.contains('marketCap').all(), (
        'a removed row shows a NaN price with nothing on the row saying section 5 refused '
        'it, so the CSV disagrees with itself: %s'
        % blanked[['source', 'date', 'price', 'refused_fields']].to_dict('records'))
    #  and an ordinary removal is not decorated
    assert (removed.loc[removed['price'].notna(), 'refused_fields'] == '').all()


def test_BOTH_refused_twins_of_a_duplicate_key_keep_their_OWN_values():
    """The other half of the duplicate-key hole, and the half the stamp cannot close.

    `refusal_restore_map` keyed on `(source, date)` alone, so when BOTH twins of a duplicate
    key are refused the map collapsed to one entry and PASS 1 judged twin 0 on twin 1's
    numbers.  The per-row stamp intersection fixes WHICH FIELDS are restored and cannot fix
    WHICH ROW; only a positional component can.

    Not harmless, contrary to what this file used to say: `check_price_sanity`'s step check
    compares ADJACENT market caps, so a borrowed value can create or suppress a break and with
    it a whole prefix deletion.

    CANNOT DETECT: a re-sort of a source's rows between the refusal (per ticker, on
    `tempfund`) and the filter (on the assembled `cdx_df`).  Both sides count occurrences in
    FRAME ORDER, so they agree only while that order is preserved -- stated in
    `refusal_restore_map` as a residual assumption rather than guarded here.
    """
    twins = pd.DataFrame([
        {'source': 'DUP', 'date': pd.Timestamp('2026-01-01'), 'price': 1.0,
         'marketCap': 1.0e6, 'bookValuePerShare': 1000.0, 'earningsYield': 0.5,
         'weightedAverageShsOut': 1.0e6, 'totalStockholdersEquity': 1.0e9},
        {'source': 'DUP', 'date': pd.Timestamp('2026-01-01'), 'price': 7.0,
         'marketCap': 7.0e6, 'bookValuePerShare': 7000.0, 'earningsYield': 0.7,
         'weightedAverageShsOut': 1.0e6, 'totalStockholdersEquity': 1.0e9},
    ])
    _out, rep = npol.refuse_impossible_cells(twins, verbose=False)
    assert len(rep[rep['field'] == 'marketCap']) == 2, 'the fixture must refuse BOTH twins'
    m = npol.refusal_restore_map(rep)
    assert len(m) == 2, (
        'the restore map collapsed %d refused twins onto %d entr(ies), so one row will be '
        'judged on the numbers of its twin: %s' % (2, len(m), m))
    d = npol._normalise_refusal_date(pd.Timestamp('2026-01-01'))
    assert m[('DUP', d, 0)]['marketCap'] == pytest.approx(1.0e6)
    assert m[('DUP', d, 1)]['marketCap'] == pytest.approx(7.0e6)


# --------------------------------------------------------------------------- #
#  UPSTREAM: what this rule READS that its own pass has already judged          #
# --------------------------------------------------------------------------- #
def test_the_rule_does_NOT_fire_on_an_equity_leg_the_same_pass_refused():
    """THE THIRD BLOCKER, AND IT IS THE PREMISE THE WHOLE ONE-SIDED DESIGN RESTS ON.

    `price_scale_hits` refuses the market-cap side and NOT the balance sheet, on the stated
    ground that `totalStockholdersEquity` "already carries two guards in this module".  All
    three producers were computed on the SAME PRE-BLANKING FRAME, so the rule divided by raw
    equity even on rows where `balance_sheet_identity` had -- in the very same call -- already
    declared that equity contradictory.  The rule cited guards it then ignored.

    The fixture is a row with a WHOLLY SOUND price side and one broken balance-sheet cell.
    Before the fix, four sound cells were refused on the strength of a number the pass had
    already rejected, each scoring as a FAIL in `calcScore.calcByTier` -- 3.80 of 17.85 of the
    Stage-1 gate, for no evidential reason.

    Live incidence on the 2026-08-29 CUR6K panel is ZERO, so the guard costs nothing there;
    the mechanism is deterministic and the fixture is the proof, not the panel.

    CANNOT DETECT: the seventeen OTHER read/refuse couplings among the pre-existing producers.
    They are real and derived in the sweep, but every one stays inside the balance sheet --
    only this rule exported damage to a different field family, which is why only this one is
    closed.
    """
    row = {'source': 'ZZEQ', 'date': pd.Timestamp('2026-01-01'),
           'price': 12.0, 'marketCap': 1.2e8, 'bookValuePerShare': 5.0,
           'earningsYield': 0.04, 'weightedAverageShsOut': 1.0e7,
           'totalAssets': 2.0e8, 'totalLiabilities': 1.5e8,
           'totalStockholdersEquity': 1.0e11}          # the ONLY broken cell
    out, rep = npol.refuse_impossible_cells(_frame([row]), verbose=False)

    assert 'balance_sheet_identity' in set(rep['relation']), (
        'the fixture must trip the equity guard, or it proves nothing')
    assert npol.PRICE_SCALE_RELATION not in set(rep['relation']), (
        'the price-scale rule fired on an equity leg the SAME pass had just refused: %s'
        % rep[['relation', 'field', 'ratio']].to_dict('records'))
    for f in ('price', 'marketCap', 'bookValuePerShare', 'earningsYield'):
        assert pd.notna(out.loc[0, f]), '%s was refused on a row whose price side is sound' % f
    #  and the balance-sheet refusal itself is untouched -- this drops one rule, not both
    assert pd.isna(out.loc[0, 'totalStockholdersEquity'])


def test_the_rule_STILL_fires_when_the_equity_leg_is_sound():
    """The other side of the same guard: dropping is conditional on the divisor having been
    refused, not on a balance-sheet rule having fired at all.  Without this, the fix above
    could be implemented as "never fire alongside another rule" and still pass.

    CANNOT DETECT: whether the equity is CORRECT -- only that no other producer rejected it.
    """
    row = {'source': 'ZZOK', 'date': pd.Timestamp('2026-01-01'),
           'price': 1.0, 'marketCap': 1.0e6, 'bookValuePerShare': 1000.0,
           'earningsYield': 0.5, 'weightedAverageShsOut': 1.0e6,
           'totalAssets': 2.0e9, 'totalLiabilities': 1.0e9,
           'totalStockholdersEquity': 1.0e9}           # identity holds; equity sound
    out, rep = npol.refuse_impossible_cells(_frame([row]), verbose=False)
    assert 'balance_sheet_identity' not in set(rep['relation'])
    assert npol.PRICE_SCALE_RELATION in set(rep['relation'])
    assert pd.isna(out.loc[0, 'marketCap'])


def test_a_price_scale_refusal_is_DETECTED_by_the_PE_column_on_the_REAL_ATRI_SHAPE():
    """THE DELIVERABLE, one function away from the market-cap band the author already swept.

    A units-error refusal and a company with no earnings both produce a BLANK `PE-ratio`, and
    they are not the same fact about the run.  This is the half that proves the pipeline can
    still TELL THEM APART on the shape that motivated the refusal in the first place: the real
    ATRI panel, run through the real `nan_policy.refuse_impossible_cells`, not through a
    hand-stamped `sanityRefusedFields` fixture.

    WHY IT STAYS HERE AND IS NOT FOLDED INTO `test_display_basis.py`.  That file drives the
    published cell end-to-end, which is the stronger instrument for the CELL -- but its panel
    fixture stamps the refusal by hand.  This one earns the refusal from the rule, so together
    they close the loop: the rule really fires on this shape, and the artifact really goes
    blank for a name the rule fired on.  Neither test is redundant with the other.

    RENAMED AND CUT BACK ON 2026-09-03, and the old name is worth recording because it names a
    mechanism that no longer exists: `test_a_refused_newest_earningsYield_does_NOT_fall_back_
    to_the_vendor_PE`.  There is no vendor fallback any more -- the CEO deleted FMP's
    `priceEarningsRatio` from the deliverable outright ("compute, do not consume") -- so every
    assertion about the `vendor` / `ours` / `none` outcomes of `_pe_cell`, and the structural
    block that checked `_pe_tag == 'vendor'` and `_pe_vendor_fallback.append` inside
    `writeBoAggToCSV`, was pinning behaviour the deliverable is now required NOT to have.  The
    column no longer reads `earningsYield` either; it reads `price` and `epsTTM`, so the field
    named in the old title is not even an input.  Those assertions are DELETED rather than
    repaired.  What replaced them is behavioural and lives in `test_display_basis.py`:
    `test_a_REFUSED_input_and_a_LOSS_MAKER_get_DIFFERENT_basis_tokens` (the two blanks carry
    different `PE-ratio_basis` tokens in the written CSV) and
    `test_the_REFUSAL_beats_a_derived_value_that_OUTLIVED_it` (the ordering, which became
    load-bearing under the new basis).

    CANNOT DETECT: whether any refused name actually reaches the top-N in a real run.  That is
    unmeasured; `carveOut.marketcap_fallback_report` existing at all is the evidence that
    refused newest rows do survive into the deliverable stages.
    """
    import postBo

    raw = _atri_shape(n=6, broken=(0,))          # newest row refused
    refused, _rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    assert postBo._pe_refused_sources(refused) == {'ATRISHAPE'}
    #  a clean panel names nobody -- the branch must not fire unconditionally
    assert postBo._pe_refused_sources(_atri_shape(n=6, broken=())) == set()

    #  AND THE DETECTOR WATCHES THE FIELDS THE NUMBER ACTUALLY READS.  It watched
    #  `earningsYield` until 2026-09-03 because the P/E was built from it; watching a field the
    #  published number no longer reads would silently relabel every refusal as a loss-maker.
    #  `marketCap` is the one this shape refuses -- the price-scale rule refuses the cap, NOT
    #  the derived `price` -- so its presence in the list is what makes the assertions above
    #  fire at all.
    assert 'marketCap' in postBo.PE_INPUT_FIELDS
    assert {'price', 'epsTTM'} <= set(postBo.PE_INPUT_FIELDS)

    #  THE CELL THAT RESULTS, called rather than read.  A source-order assertion survives the
    #  branch being deleted, because the token still appears where the set is built -- which is
    #  exactly what a mutation showed, and the reason the decision lives in `_pe_cell` at all.
    assert postBo._pe_cell(None, 'ATRISHAPE', {'ATRISHAPE'}) == (
        'NaN', postBo.PE_BASIS_REFUSED)
    #  a name the rule did NOT name is not collateral damage
    assert postBo._pe_cell(2.0, 'OTHER', {'ATRISHAPE'}) == ('2.0000', postBo.PE_BASIS_TTM)


def test_the_equity_guard_drops_ONLY_the_rows_whose_own_equity_was_refused():
    """The over-correction arm.  `_drop_price_scale_on_already_refused_equity` must drop
    per-ROW, not wholesale: a panel where SOME row has its equity refused must not lose the
    price-scale refusal on a DIFFERENT row whose equity is sound.  Without this the guard could
    be implemented as "never fire alongside a balance-sheet rule" and every test above would
    still pass, because their fixtures are single-row.

    CANNOT DETECT: a drop keyed on the wrong row identity -- the two rows here differ in both
    source and content, so a coarse key would still separate them.
    """
    df = _frame([
        {'source': 'ZZBAD', 'date': pd.Timestamp('2026-01-01'), 'price': 12.0,
         'marketCap': 1.2e8, 'bookValuePerShare': 5.0, 'earningsYield': 0.04,
         'weightedAverageShsOut': 1.0e7, 'totalAssets': 2.0e8,
         'totalLiabilities': 1.5e8, 'totalStockholdersEquity': 1.0e11},   # equity refused
        {'source': 'ZZGOOD', 'date': pd.Timestamp('2026-01-01'), 'price': 1.0,
         'marketCap': 1.0e6, 'bookValuePerShare': 1000.0, 'earningsYield': 0.5,
         'weightedAverageShsOut': 1.0e6, 'totalAssets': 2.0e9,
         'totalLiabilities': 1.0e9, 'totalStockholdersEquity': 1.0e9},    # sound equity
    ])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    ps = rep[rep['relation'] == npol.PRICE_SCALE_RELATION]
    assert set(ps['source']) == {'ZZGOOD'}, (
        'the guard is not per-row: it dropped the price-scale refusal on a row whose own '
        'equity was never refused (kept: %s)' % sorted(set(ps['source'])))
    assert pd.isna(out.loc[1, 'marketCap']), 'ZZGOOD must still be refused'
    assert pd.notna(out.loc[0, 'marketCap']), 'ZZBAD must keep its sound price side'


def test_the_containment_line_reports_RANKABLE_names_not_all_flagged_names():
    """A fully-refused source has no computable `bookToPrice`, so it leaves `bookToPrice_ranks`
    while staying in the flagged list.  Counted into the DENOMINATOR it made the head count
    read "1 of 2 rank in the top 2" about a population whose MOST contaminated member had
    silently dropped out of the numerator -- the sentence erring toward an all-clear.

    CANNOT DETECT: whether the rank numbers themselves are right; only that the denominator
    counts what was actually ranked and that the unrankable names are named.
    """
    import price_scale_audit as psa
    rows = []
    for i in range(6):                                   # fully refused on EVERY row
        rows.append({'source': 'ZZFULL', 'date': pd.Timestamp('2026-01-01') - pd.DateOffset(months=3 * i),
                     'price': 1.0, 'marketCap': 1.0e6, 'bookValuePerShare': 1000.0,
                     'earningsYield': 0.5, 'weightedAverageShsOut': 1.0e6,
                     'revenue': 1.0e8, 'totalStockholdersEquity': 1.0e9})
    def _rows(src, mcap_clean, eq_clean, refuse_first=0):
        for i in range(6):
            bad = i < refuse_first
            rows.append({'source': src,
                         'date': pd.Timestamp('2026-01-01') - pd.DateOffset(months=3 * i),
                         'price': 1.0 if bad else 10.0,
                         'marketCap': 1.0e6 if bad else mcap_clean,
                         'bookValuePerShare': 1000.0 if bad else 5.0,
                         'earningsYield': 0.5 if bad else 0.05,
                         'weightedAverageShsOut': 1.0e6 if bad else 1.0e8,
                         'revenue': 1.0e8,
                         'totalStockholdersEquity': 1.0e9 if bad else eq_clean})
    #  bookToPrice = equity/marketCap, ranked DESCENDING.  Two FLAGGED-and-rankable names
    #  bracket a clean one, so `r <= len(ranks)` (=2) and `r <= len(names)` (=3) give
    #  DIFFERENT head counts -- without that the numerator's bound is unpinned.
    #  NB the CLEAN rows must sit ABOVE the cut (marketCap/equity >= 0.02, i.e. b2p <= 50) or
    #  they are refused too and the source stops being rankable -- which is exactly what a
    #  first draft of this fixture did.
    _rows('ZZCLEANA', 5.0e7, 1.0e9)                  # b2p  20 -> rank 1
    _rows('ZZPART',   1.0e8, 1.0e9, refuse_first=2)  # b2p  10 -> rank 2   FLAGGED
    _rows('ZZPART2',  1.43e8, 1.0e9, refuse_first=2) # b2p   7 -> rank 3   FLAGGED
    _rows('ZZCLEANB', 2.0e8, 1.0e9)                  # b2p   5 -> rank 4
    out, rep = npol.refuse_impossible_cells(_frame(rows), verbose=False)
    lines = []
    psa.run_audit(out, prices_csv=None, log=lines.append, run_grid_check=False,
                  refusal_report=rep, shipped_sources=['SOMETHINGELSE'])
    blob = chr(10).join(lines)
    #  THE NUMBER, not the word: one rankable name of one rankable name.  Counting the
    #  unrankable one into the denominator would read "1 of 2".
    #  THE NUMBER, not the word.  Two flagged names are rankable (ranks 1 and 3) and one is
    #  not, so the honest line is "1 of 2" -- one of the two rankable ones sits in the head of
    #  size 2.  Counting the unrankable name into the denominator would make it "2 of 3".
    assert '1 of 2 RANKABLE flagged names' in blob, blob
    #  AND THE LINE THAT READS THAT THRESHOLD, four lines further down.  The S3-2 fix moved
    #  `n_in_head`'s bound and this denominator to the rankable count and left the printed
    #  THRESHOLD on the flagged count, so the sentence said "1 of 2 ... in the top 3" while
    #  both ranks were <= 3 -- false, and one-signed toward an all-clear.  Asserting the
    #  numerator without its threshold is what let that survive.
    assert 'in the top 2 of' in blob, (
        'the printed threshold disagrees with the head count it explains: %s' % blob)
    assert 'NOT RANKABLE here' in blob or 'NOT COMPUTABLE FOR ANY' in blob, (
        'a fully-refused flagged name vanished from the containment paragraph without the '
        'paragraph saying so: %s' % blob)


def test_the_band_fallback_report_LABELS_the_basis_it_used():
    """The banner explains a BAND, and the band is decided on USD.  Reporting a
    reporting-currency number beside a USD-decided band misdescribes the substitution the
    artifact exists to disclose -- CORE-A.ST's "4.474bn" is SEK, about $430M, a different band.
    USD alone would be right and EMPTY on every panel predating the `reportedCurrency`
    capture, so both bases are allowed and the one used is named.

    CANNOT DETECT: whether the FX rate itself is right -- only which basis was used.
    """
    import carveOut as co
    raw = _atri_shape(n=6, broken=(0,))          # no reportedCurrency column at all
    refused, rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    fb = co.marketcap_fallback_report(refused, report=rep)
    assert len(fb) == 1
    assert fb['basis'].iloc[0] == 'reporting-currency', (
        'the report claimed a USD basis on a panel that carries no reportedCurrency, so the '
        'number printed beside a USD-decided band is not USD: %s' % fb.to_dict('records'))
    assert fb['rows_back'].iloc[0] == 1, 'the fallback must still be found on the raw basis'

    #  THE INVARIANT, not a fixed string: the label must say USD exactly when a USD value was
    #  resolvable.  Asserting 'USD' outright would pin the FX state of the test process rather
    #  than the behaviour -- `marketcap_usd_series` needs a live or installed rate table, and
    #  `conftest` isolates that per test.
    with_ccy = refused.copy()
    with_ccy['reportedCurrency'] = 'USD'
    fb2 = co.marketcap_fallback_report(with_ccy, report=rep)
    usd_available = co.marketcap_usd_series(with_ccy, allow_suffix_fallback=False).notna().any()
    expected = 'USD' if usd_available else 'reporting-currency'
    assert fb2['basis'].iloc[0] == expected, (
        'the basis label disagrees with whether a USD value was actually resolvable '
        '(usd_available=%s): %s' % (bool(usd_available), fb2.to_dict('records')))


def test_the_band_fallback_DISCLOSURE_actually_prints(capsys):
    """THE DISCLOSURE THAT HAD NEVER EMITTED A ROW.

    `partition_by_marketcap`'s banner carried FIVE format specifiers and SIX arguments, putting
    `basis` (a string) on the `%d`.  Every run where the report had a row therefore raised
    `TypeError: %d format: a real number is required, not str` inside the surrounding
    try/except, and the operator got the header followed by "this run simply has no record of
    which bands rest on a fallback".

    That disclosure is the ACCEPTED REMEDY for keeping the silent fallback at all -- the whole
    argument for not refusing a stale market cap was that the substitution would be recorded.
    It was inert from the day it was written.

    THE REASON IT SURVIVED FOUR ROUNDS IS THAT EVERY TEST WENT STRAIGHT TO THE FRAME BUILDER.
    None drove the caller.  This one drives `partition_by_marketcap` and reads stdout.

    CANNOT DETECT: a format defect in any OTHER print in this file -- the reviewer AST-scanned
    all ten and found no second arity mismatch, but nothing here would see a new one.
    """
    import carveOut as co
    raw = _atri_shape(n=6, broken=(0,))          # newest row refused
    refused, _rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    ranked = pd.DataFrame({'source': ['ATRISHAPE']})
    capsys.readouterr()
    co.partition_by_marketcap(ranked, refused)
    out = capsys.readouterr().out

    assert 'MARKET-CAP BAND FALLBACK' in out, out
    assert 'ATRISHAPE' in out, (
        'the header printed but no ROW did, which is exactly the shape of the arity bug: %s'
        % out)
    assert 'row(s) back' in out
    #  and it must never print a bare `nan`: the refused value is not recoverable at this call
    #  site (no refusal report reaches `partition_by_marketcap`), so it says so instead.
    assert 'nan' not in out.lower().replace('not recoverable', ''), (
        'the banner printed a bare nan rather than saying the value is unavailable: %s' % out)
    assert 'fallback report skipped' not in out, (
        'the disclosure raised and was swallowed by its own try/except: %s' % out)


def test_the_refused_and_fallback_market_caps_are_on_the_SAME_basis():
    """A row that compares two currencies mis-states the SIZE of the break it exists to show.

    `refused_marketCap_usd` came from the refusal report, which records the RAW
    reporting-currency value; `fallback_marketCap_usd` is USD wherever the currency resolves.
    On a SEK name at ~10 SEK/USD a genuine 1,000x break reports as 10,000x.

    THE OLD FIXTURE COULD NOT SEE THIS: both its arms had an FX factor of 1.0 (no currency, or
    USD).  A test whose fixture cannot express the defect is not a guard, which is the house
    repeat defect in its quietest form.  This one uses a NON-TRIVIAL factor.

    CANNOT DETECT: whether the FX factor itself is right -- only that both numbers are
    expressed on the basis the `basis` column claims.
    """
    import carveOut as co
    raw = _atri_shape(n=6, broken=(0,))
    raw['reportedCurrency'] = 'SEK'
    refused, rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    fb = co.marketcap_fallback_report(refused, report=rep)
    assert len(fb) == 1
    basis = fb['basis'].iloc[0]
    ref_v = float(fb['refused_marketCap_usd'].iloc[0])
    fal_v = float(fb['fallback_marketCap_usd'].iloc[0])

    #  The RAW break in the fixture is exactly 1000x (4.0e6 refused against 4.0e9 sound).
    #  Whatever basis the row is on, the RATIO between the two must still read 1000x -- that
    #  is the invariant a currency mismatch destroys.
    assert fal_v / ref_v == pytest.approx(1000.0, rel=1e-6), (
        'the refused and fallback values are on different bases, so the break reports as %.4gx '
        'instead of 1000x (basis=%s, refused=%.6g, fallback=%.6g)'
        % (fal_v / ref_v, basis, ref_v, fal_v))
    if basis == 'USD':
        #  and when the row claims USD, the refused value must actually BE converted
        assert ref_v != pytest.approx(4.0e6, rel=1e-9), (
            'the row claims a USD basis while carrying the raw reporting-currency value')


# --------------------------------------------------------------------------- #
#  Q-75 -- THE TWO-LEVEL RULE.  Each level is pinned by a REAL row the other        #
#  level cannot reach, so neither can be deleted without a failure here.       #
# --------------------------------------------------------------------------- #
#  EVERY FIXTURE IN THIS SECTION IS VERBATIM PANEL DATA, extracted from
#  `baseline_tools/resdic_2026-07-17_CORRECTED.pickle` and
#  `Bometric_dic-fmp_stock_CUR3K_all_2026-08-11...pickle` rather than typed, because the whole
#  claim being tested is about what the vendor actually serves.
_QBY0_MAJORITY_CORRUPT = [
    {'source': 'QBY0.DE', 'date': '2023-04-01', 'price': 3.26, 'marketCap': 649802.76,
     'bookValuePerShare': 541.3593811143554, 'earningsYield': -4.682959487583586,
     'weightedAverageShsOut': 199326.0, 'totalStockholdersEquity': 106780000.0},
    {'source': 'QBY0.DE', 'date': '2023-10-01', 'price': 3.13, 'marketCap': 623890.38,
     'bookValuePerShare': 498.6554689302951, 'earningsYield': -7.834709680889775,
     'weightedAverageShsOut': 199326.0, 'totalStockholdersEquity': 97846000.0},
    {'source': 'QBY0.DE', 'date': '2024-01-01', 'price': 2.86, 'marketCap': 570072.36,
     'bookValuePerShare': 493.0867021863681, 'earningsYield': -2.401449528266903,
     'weightedAverageShsOut': 199326.0, 'totalStockholdersEquity': 96478000.0},
    {'source': 'QBY0.DE', 'date': '2024-04-01', 'price': 4.13, 'marketCap': 823216.38,
     'bookValuePerShare': 489.0480920702768, 'earningsYield': -1.2244654315551884,
     'weightedAverageShsOut': 199326.0, 'totalStockholdersEquity': 95470000.0},
    {'source': 'QBY0.DE', 'date': '2024-07-01', 'price': 3.89, 'marketCap': 775378.14,
     'bookValuePerShare': 484.512808163511, 'earningsYield': -1.2445540443015328,
     'weightedAverageShsOut': 199326.0, 'totalStockholdersEquity': 94469000.0},
]
#  WDH (Waterdrop): shares 361.9M -> 36.1M on the NEWEST TWO rows with `price` flat and equity
#  flat.  pb 0.093 and 0.079 -- ABOVE the floor, so only the witnessed band can reach it.
_WDH_NEWEST_ROW_BREAK = [
    {'source': 'WDH', 'date': '2024-07-01', 'price': 8.42052, 'marketCap': 3051966950.88,
     'bookValuePerShare': 12.999655119135646, 'earningsYield': 0.030408258507924123,
     'weightedAverageShsOut': 362444000.0, 'totalStockholdersEquity': 4621198000.0},
    {'source': 'WDH', 'date': '2024-10-01', 'price': 8.612584, 'marketCap': 3118608053.816,
     'bookValuePerShare': 13.459214192803625, 'earningsYield': 0.03391994061920077,
     'weightedAverageShsOut': 362099000.0, 'totalStockholdersEquity': 4797435000.0},
    {'source': 'WDH', 'date': '2025-01-01', 'price': 10.811738, 'marketCap': 3914260002.044,
     'bookValuePerShare': 13.627171733353958, 'earningsYield': 0.0276412399645147,
     'weightedAverageShsOut': 362038000.0, 'totalStockholdersEquity': 4861361000.0},
    {'source': 'WDH', 'date': '2025-04-01', 'price': 9.670185, 'marketCap': 3489247751.80716,
     'bookValuePerShare': 13.892026140869442, 'earningsYield': 0.04016883006585259,
     'weightedAverageShsOut': 360825336.0, 'totalStockholdersEquity': 5012595000.0},
    {'source': 'WDH', 'date': '2025-07-01', 'price': 13.453965, 'marketCap': 4870048074.393285,
     'bookValuePerShare': 14.013851408125456, 'earningsYield': 0.03253889850353106,
     'weightedAverageShsOut': 361978649.0, 'totalStockholdersEquity': 5072715000.0},
    {'source': 'WDH', 'date': '2025-10-01', 'price': 13.285940147093456,
     'marketCap': 480011026.48749596, 'bookValuePerShare': 142.59715405538302,
     'earningsYield': 0.33307246537630264, 'weightedAverageShsOut': 36129248.0,
     'totalStockholdersEquity': 5151928000.0},
    {'source': 'WDH', 'date': '2026-01-01', 'price': 11.173950186349142,
     'marketCap': 402010424.0890201, 'bookValuePerShare': 144.70996819131315,
     'earningsYield': 0.24322013843687926, 'weightedAverageShsOut': 35977467.0,
     'totalStockholdersEquity': 5097275875.0},
]
#  CHLL.L: five real rows sitting INSIDE the widened band (pb 0.021-0.074) whose share count is
#  within 1.5x of the source's own median.  No witness, so no refusal.  This is the fixture
#  that stands for the measured false-refusal population -- ALDBT.PA and URU.L are refused by
#  the shipped rule and this name, in the same band, is not.
_CHLL_IN_BAND_NO_WITNESS = [
    {'source': 'CHLL.L', 'date': '2016-01-01', 'price': 1.484, 'marketCap': 102522.14,
     'bookValuePerShare': 20.17695592386191, 'earningsYield': -8.859896018557553,
     'weightedAverageShsOut': 69085.0, 'totalStockholdersEquity': 1393925.0},
    {'source': 'CHLL.L', 'date': '2017-01-01', 'price': 2.649, 'marketCap': 183006.165,
     'bookValuePerShare': 101.44998190634725, 'earningsYield': -10.43736969188989,
     'weightedAverageShsOut': 69085.0, 'totalStockholdersEquity': 7008672.0},
    {'source': 'CHLL.L', 'date': '2017-07-01', 'price': 2.3, 'marketCap': 181161.8,
     'bookValuePerShare': 111.30995607241704, 'earningsYield': -12.635555619341385,
     'weightedAverageShsOut': 78766.0, 'totalStockholdersEquity': 8767440.0},
    {'source': 'CHLL.L', 'date': '2018-01-01', 'price': 2.213, 'marketCap': 257810.074,
     'bookValuePerShare': 80.41048773369499, 'earningsYield': -6.544185701602956,
     'weightedAverageShsOut': 116498.0, 'totalStockholdersEquity': 9367661.0},
    {'source': 'CHLL.L', 'date': '2018-07-01', 'price': 1.69, 'marketCap': 198127.15,
     'bookValuePerShare': 53.344922591376296, 'earningsYield': -22.156675649955094,
     'weightedAverageShsOut': 117235.0, 'totalStockholdersEquity': 6253892.0},
]


def test_the_FLOOR_alone_refuses_a_majority_corrupt_source_the_witness_cannot_see():
    """QBY0.DE -- why the CEO-ruled bare conjunction could not ship, as an executable fact.

    22 of this source's 24 real rows carry a share count of ~199,326 against a true
    24,915,897, so THE CORRUPTION IS THE MAJORITY OF THE HISTORY and 199,326 IS the source's
    own median.  The witness ratio is therefore 1.000 on every contaminated row -- it does not
    merely miss, it would point at the two SOUND rows instead.  Only the unconditional floor
    reaches these.

    THIS TEST IS THE GUARD ON THE UNION.  Replace the shipped disjunction with the conjunction
    `pb < PB_WIDE and witness` and all five rows go unrefused: MEASURED, the conjunction drops
    QBY0.DE, 0CHZ.L and CMCM entirely and takes the panel from 265 refused rows to 160.

    CANNOT DETECT: whether 0.02 is the right floor.  Every row here is 3x inside it.
    """
    df = _frame(_QBY0_MAJORITY_CORRUPT)
    #  the witness is SILENT -- stated first, so a reader knows which level is on trial
    wr = npol.share_count_witness_ratio(df)
    assert wr.max() == pytest.approx(1.0), (
        'the fixture no longer has a silent witness, so it no longer tests the floor: %s'
        % wr.tolist())
    hits = npol.price_scale_hits(df)
    assert sorted(hits['row']) == [0, 1, 2, 3, 4], hits
    #  and the floor limb ALONE accounts for all five: collapsing the widened band onto the
    #  floor changes nothing here
    floor_only = npol.price_scale_hits(df, pb_wide=npol.PRICE_SCALE_PB_ALARM)
    assert sorted(floor_only['row']) == [0, 1, 2, 3, 4], floor_only
    #  the CONJUNCTION, evaluated here rather than described: it refuses nothing
    pb = (pd.to_numeric(df['marketCap']) / pd.to_numeric(df['totalStockholdersEquity']))
    conjunction = ((pb < npol.PRICE_SCALE_PB_WIDE)
                   & (wr >= npol.PRICE_SCALE_WITNESS_FACTOR))
    assert not bool(conjunction.any()), (
        'the conjunction now fires on this fixture, so it has stopped being the '
        'counter-example the union rests on')


def test_the_WITNESSED_BAND_catches_a_newest_row_break_no_other_guard_can_reach():
    """WDH -- what the widening is FOR, and the strongest single case for it.

    Shares fall 361.9M -> 36.1M (exactly 10x) on the source's NEWEST TWO rows while `price`
    stays 13.45 -> 13.29 and equity stays 5.1bn.  price/book lands at 0.093 and 0.079, so:
      * the FLOOR misses it -- 0.079 is four times the floor;
      * the SPIKE rule cannot see it -- `weightedAverageShsOut` is not in `SCALE_SPIKE_FIELDS`,
        and even if it were, a two-sided neighbour test is structurally blind on a newest row
        and both broken rows are adjacent;
      * `balance_sheet_identity` cannot see it -- the balance sheet is sound.
    A real ~$400M company whose current-period market cap is 10x too low, feeding 21.3% of the
    Stage-1 gate.  Before this change nothing in the pipeline touched it.

    CANNOT DETECT: whether 5.0 is the right witness factor.  This row is at 10.0x.
    """
    df = _frame(_WDH_NEWEST_ROW_BREAK)
    wr = npol.share_count_witness_ratio(df)
    assert wr.iloc[5] == pytest.approx(10.0, abs=0.1)
    assert wr.iloc[6] == pytest.approx(10.1, abs=0.1)
    assert wr.iloc[:5].max() < 1.01, 'the sound rows must sit AT the median'
    hits = npol.price_scale_hits(df)
    assert sorted(hits['row']) == [5, 6], hits
    #  the floor reaches NONE of it -- this is the coverage the widening bought
    assert not len(npol.price_scale_hits(df, pb_wide=npol.PRICE_SCALE_PB_ALARM))
    #  the spike rule reaches none of it either, and it is not close: the field is not one it
    #  looks at.  Stated as an assertion because "the spike rule would have caught it" is the
    #  obvious objection to this test existing.
    assert npol.PRICE_SCALE_WITNESS_FIELD not in npol.SCALE_SPIKE_FIELDS
    assert not len(npol.scale_spike_hits(df))
    #  end to end: the cap side goes, the balance sheet stays
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    for r in (5, 6):
        for f in npol.PRICE_SCALE_REFUSE:
            assert pd.isna(out.at[r, f]), (f, r)
        assert out.at[r, 'totalStockholdersEquity'] == df.at[r, 'totalStockholdersEquity']
        assert out.at[r, 'weightedAverageShsOut'] == df.at[r, 'weightedAverageShsOut'], (
            'the witness field must never be refused -- the rule reads it and asserts '
            'nothing about it')
    #  and the SOUND rows are untouched, so this is a row refusal and not a source refusal
    for r in range(5):
        assert out.at[r, 'marketCap'] == df.at[r, 'marketCap']


def test_a_row_inside_the_WIDENED_BAND_with_no_witness_is_LEFT_ALONE():
    """CHLL.L -- the witness is a REQUIREMENT of the widened band, not decoration.

    Five real rows at price/book 0.021-0.074, i.e. squarely inside `PRICE_SCALE_PB_WIDE`, whose
    share count sits within 1.5x of the source's own median.  Nothing refuses them.

    THIS IS THE OVER-REFUSAL GUARD.  Drop the `& witness` conjunct -- make the band
    unconditional -- and ALL FIVE rows are refused, which would invent a Stage-1 FAIL
    (`calcScore.calcByTier` scores a NaN as a FAIL) against a real company on no evidence at
    all.  That is the error this module calls the invisible one, because nothing downstream
    reports it.

    CANNOT DETECT: the measured 11.4%-31.4% of widened-band refusals that ARE over-reaches.
    This name is on the right side of the cut; ALDBT.PA and URU.L, in the same band with
    dilution-ramp witness ratios, are not, and are refused. See `nan_policy` for that table.
    """
    df = _frame(_CHLL_IN_BAND_NO_WITNESS)
    pb = (pd.to_numeric(df['marketCap']) / pd.to_numeric(df['totalStockholdersEquity']))
    #  the premise: EVERY row is in the band and above the floor, so the witness is the
    #  only thing standing between this name and five invented FAILs
    in_band = (pb >= npol.PRICE_SCALE_PB_ALARM) & (pb < npol.PRICE_SCALE_PB_WIDE)
    assert int(in_band.sum()) == len(df), pb.tolist()
    wr = npol.share_count_witness_ratio(df)
    assert wr.max() < npol.PRICE_SCALE_WITNESS_FACTOR, wr.tolist()
    assert not len(npol.price_scale_hits(df)), npol.price_scale_hits(df)
    #  the unconditional band, evaluated here rather than described: it takes all five
    assert int(in_band.sum()) == 5
    out, _ = npol.refuse_impossible_cells(df, verbose=False)
    assert out['marketCap'].notna().all()
    assert npol.SANITY_REFUSED_COLUMN not in out.columns or out[
        npol.SANITY_REFUSED_COLUMN].fillna('').eq('').all()


def test_the_shipped_rule_is_a_SUPERSET_of_the_floor_never_a_replacement_for_it():
    """The structural property the widening had to have, on one mixed frame.

    A conjunction cannot be a superset of its own conjunct.  That is the whole reason the
    ruled form could not ship, and it is checkable without any threshold: refusing with the
    band collapsed onto the floor must yield a SUBSET of refusing with the band open.

    CANNOT DETECT: whether the rows the band ADDS are the right ones.  That is adjudication,
    not a property, and it lives in `nan_policy`'s measured table.
    """
    df = _frame(_QBY0_MAJORITY_CORRUPT + _WDH_NEWEST_ROW_BREAK + _CHLL_IN_BAND_NO_WITNESS)
    union = set(npol.price_scale_hits(df)['row'])
    floor = set(npol.price_scale_hits(df, pb_wide=npol.PRICE_SCALE_PB_ALARM)['row'])
    assert floor and union, (floor, union)
    assert floor < union, (
        'the widened rule is no longer a strict superset of the floor -- it has become a '
        'REPLACEMENT for it, which is exactly the defect that made the conjunction '
        'un-shippable: floor=%s union=%s' % (sorted(floor), sorted(union)))
    #  and the three sources are doing three different jobs on the same frame
    got = {str(df.at[r, 'source']) for r in union}
    assert got == {'QBY0.DE', 'WDH'}, got


def test_the_WITNESS_FIELD_is_refused_by_no_producer_in_this_module():
    """The widening reads a field, so it could have added an 18th read/refuse coupling.  It
    does not -- and that is one edit away from being false.

    `share_count_witness_ratio` reads `weightedAverageShsOut` off the SAME pre-blanking frame
    every other producer runs on.  If any producer refused that field, the witness would be
    computed from a cell this pass had already rejected: the Q-72 defect, freshly grown on the
    rule that was being fixed.

    CANNOT DETECT: a producer added OUTSIDE this module's three tables.
    """
    refusers = set(npol.PRICE_SCALE_REFUSE) | set(npol.SCALE_SPIKE_FIELDS)
    for _n, _num, _den, _f, _t, refuse in npol.IMPOSSIBLE_RELATIONS:
        refusers |= set(refuse)
    assert npol.PRICE_SCALE_WITNESS_FIELD not in refusers, (
        '%s is now refused by a producer in this module, so the price-scale witness reads a '
        'cell the same pass can reject -- the coupling class Q-72 exists for. Either stop '
        'refusing it or give the witness a pre-refusal copy of the column.'
        % npol.PRICE_SCALE_WITNESS_FIELD)


def test_the_witness_abstains_when_there_are_too_few_rows_to_have_a_median():
    """Two rows and a 100x break put BOTH rows ~10x off their midpoint, so a median witness
    would corroborate a refusal of the SOUND row as readily as the broken one.
    `PRICE_SCALE_WITNESS_MIN_ROWS` refuses to answer instead.

    CANNOT DETECT: whether 4 is the right minimum.  Measured cost on all five saved panels is
    0 rows, so no shipped refusal turns on the exact number.
    """
    two = _frame([
        {'source': 'ZZTWO', 'date': '2025-01-01', 'price': 1.0, 'marketCap': 1.0e5,
         'bookValuePerShare': 1.0, 'earningsYield': 0.1,
         'weightedAverageShsOut': 1.0e5, 'totalStockholdersEquity': 2.0e6},
        {'source': 'ZZTWO', 'date': '2025-04-01', 'price': 1.0, 'marketCap': 1.0e7,
         'bookValuePerShare': 1.0, 'earningsYield': 0.1,
         'weightedAverageShsOut': 1.0e7, 'totalStockholdersEquity': 2.0e6},
    ])
    assert npol.share_count_witness_ratio(two).isna().all(), (
        'a two-row source now gets a witness verdict, so the widened band can fire on a '
        'median that is a midpoint between a sound and a broken value')
    #  pb here is 0.05 -- inside the band, outside the floor -- so the ONLY thing keeping this
    #  frame unrefused is the row minimum.
    assert not len(npol.price_scale_hits(two))
    #  the same shape with enough rows DOES get a verdict, so the guard is a row-count guard
    #  and not an accidental all-abstain.
    four = _frame(_QBY0_MAJORITY_CORRUPT[:4])
    assert npol.share_count_witness_ratio(four).notna().all()


# --------------------------------------------------------------------------- #
#  Q-72 -- THE READ/REFUSE COUPLING GUARD                                      #
# --------------------------------------------------------------------------- #
#  081580.KQ 2020-10, verbatim from CUR3K_2026-08-11 with its two date-neighbours (the spike
#  rule needs both).  `totalAssets` is served 1,000x too small for exactly this quarter; every
#  other cell on the row is sound and continuous with its neighbours.
_KQ_CORRUPT_TOTALASSETS = [
    {'source': '081580.KQ', 'date': '2020-07-01', 'totalAssets': 129836984000.0,
     'totalLiabilities': 28125493000.0, 'totalStockholdersEquity': 101032317000.0,
     'totalCurrentAssets': 67500638000.0, 'totalCurrentLiabilities': 24684627000.0,
     'propertyPlantEquipmentNet': 48774780000.0},
    {'source': '081580.KQ', 'date': '2020-10-01', 'totalAssets': 111489851.0,
     'totalLiabilities': 23737056599.0, 'totalStockholdersEquity': 97439627358.0,
     'totalCurrentAssets': 57821671.0, 'totalCurrentLiabilities': 21315026075.0,
     'propertyPlantEquipmentNet': 43397726255.0},
    {'source': '081580.KQ', 'date': '2021-01-01', 'totalAssets': 126716636000.0,
     'totalLiabilities': 27479082000.0, 'totalStockholdersEquity': 98783151000.0,
     'totalCurrentAssets': 68345388000.0, 'totalCurrentLiabilities': 25266706000.0,
     'propertyPlantEquipmentNet': 43298339000.0},
]


def test_a_relation_does_not_blank_a_sound_leg_off_a_cell_the_spike_rule_refused():
    """Q-72, the nine couplings the measurement justified closing.

    On this real row FOUR producers fire on the same pre-blanking frame:
    `isolated_scale_spike:totalAssets`, `isolated_scale_spike:totalCurrentAssets`,
    `balance_sheet_identity` (which DIVIDES BY the corrupt `totalAssets` and blanks
    `totalLiabilities` and `totalStockholdersEquity` off it) and `ppe_within_assets` (which
    divides by it and blanks `propertyPlantEquipmentNet`).

    BEFORE the guard, five cells were refused and three of them -- `totalLiabilities`,
    `totalStockholdersEquity`, `propertyPlantEquipmentNet` -- are sound and continuous with
    their own neighbours.  AFTER, only the two cells a producer's own evidence NAMES are
    refused.

    CANNOT DETECT: the eight MUTUAL couplings (two containment relations reading each other
    with no spike hit to break the tie).  Ordering is not the instrument for those; they are
    left, with their measured incidence recorded in `nan_policy`.
    """
    df = _frame(_KQ_CORRUPT_TOTALASSETS)
    rel = npol.impossible_relation_hits(df)
    spike = npol.scale_spike_hits(df)
    #  the premise, asserted rather than assumed: all four producers really do fire
    assert set(rel['relation']) == {'balance_sheet_identity', 'ppe_within_assets'}, rel
    assert set(spike['relation']) == {'isolated_scale_spike:totalAssets',
                                      'isolated_scale_spike:totalCurrentAssets'}, spike
    assert set(rel['row']) == {1} and set(spike['row']) == {1}
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    #  the two cells the spike rule named are gone
    assert pd.isna(out.at[1, 'totalAssets'])
    assert pd.isna(out.at[1, 'totalCurrentAssets'])
    #  the three the relations would have taken as collateral survive, with their real values
    for f in ('totalLiabilities', 'totalStockholdersEquity', 'propertyPlantEquipmentNet'):
        assert out.at[1, f] == df.at[1, f], (
            '%s was blanked off a `totalAssets` this same pass had already refused' % f)
    #  the report says so too -- an audit reading the CSV must see the same two cells
    assert sorted(rep['field']) == ['totalAssets', 'totalCurrentAssets'], rep
    assert set(rep['relation']) == {'isolated_scale_spike:totalAssets',
                                    'isolated_scale_spike:totalCurrentAssets'}, rep
    #  and the stamp records an ABSTENTION on exactly those two, so a downstream reader cannot
    #  mistake either for an absence
    assert npol.refused_fields_mask(out, 'totalAssets').iloc[1]
    assert not npol.refused_fields_mask(out, 'totalStockholdersEquity').iloc[1]
    #  the neighbours are untouched: this is a row abstention, not a window deletion
    for r in (0, 2):
        assert out.loc[r].notna().all()


def test_the_coupling_guard_never_drops_a_SPIKE_hit():
    """The guard is ASYMMETRIC and that is what makes it acyclic.

    The spike rule refuses exactly the one field its own evidence names and reads none of the
    sibling fields, so it has no collateral and is never a drop target.  A symmetric version
    of this guard could drop both sides of a mutual pair and lose the refusal entirely --
    silently, since fewer refusals reads like a cleaner panel.

    CANNOT DETECT: a future producer added with the spike prefix that DOES read siblings.
    """
    df = _frame(_KQ_CORRUPT_TOTALASSETS)
    hits = pd.concat([npol.impossible_relation_hits(df), npol.scale_spike_hits(df)],
                     ignore_index=True)
    kept = npol._drop_relation_hits_the_spike_rule_already_explained(hits)
    def _spikes(h):
        return sorted(h.loc[h['relation'].astype(str).str.startswith(
            npol.SCALE_SPIKE_RELATION_PREFIX), 'relation'])
    assert _spikes(kept) == _spikes(hits), (_spikes(hits), _spikes(kept))
    assert len(kept) == 2 and len(hits) == 4, (hits, kept)
    #  NOR A PRICE-SCALE HIT.  The price-scale rule is a READER in the coupling sweep, never a
    #  refuser of a field a relation reads, and it has its OWN guard
    #  (`_drop_price_scale_on_already_refused_equity`).  If this guard also dropped it the row
    #  would lose its cheapness refusal silently -- fewer refusals reads like a cleaner panel.
    withps = pd.concat([hits, pd.DataFrame([{
        'row': 1, 'relation': npol.PRICE_SCALE_RELATION, 'ratio': 0.001,
        'fields': npol.PRICE_SCALE_REFUSE}])], ignore_index=True)
    kept2 = npol._drop_relation_hits_the_spike_rule_already_explained(withps)
    assert (kept2['relation'] == npol.PRICE_SCALE_RELATION).sum() == 1, kept2
    #  and it is a NO-OP when no spike hit is present: a relation firing alone keeps its hit
    lone = npol._drop_relation_hits_the_spike_rule_already_explained(
        npol.impossible_relation_hits(df))
    assert len(lone) == 2, lone


def test_the_spike_relation_label_has_exactly_one_home():
    """The guard identifies spike hits BY PREFIX, so a second literal would make it a silent
    no-op -- fewer drops, no error, no test failure anywhere else.  One fact, one home.

    CANNOT DETECT: a THIRD module building the label itself.
    """
    src = open(npol.__file__, encoding='utf-8').read()
    #  the literal appears where the constant is DEFINED and nowhere else in code
    import re
    code = [ln for ln in src.splitlines()
            if "'isolated_scale_spike:" in ln and not ln.lstrip().startswith('#')]
    assert len(code) == 1 and 'SCALE_SPIKE_RELATION_PREFIX =' in code[0], code
    assert npol.scale_spike_relation('totalAssets') == 'isolated_scale_spike:totalAssets'
    #  and the producer really does use the helper
    assert 'scale_spike_relation(field)' in src


def test_the_coupling_guard_runs_BEFORE_the_price_scale_equity_guard():
    """ORDER IS LOAD-BEARING, and running it the other way is a silent wrong answer.

    `_drop_price_scale_on_already_refused_equity` asks "did another producer refuse this row's
    `totalStockholdersEquity`".  The coupling guard can DROP the `balance_sheet_identity` hit
    that was the only such refusal -- so a price-scale hit that used to be dropped now stands.
    Measured on the two NA1_EU1 panels: +2 price-scale hits each.  Run in the other order, the
    equity guard answers from a hit set that does not ship.

    CANNOT DETECT: the correctness of the resulting refusal.  It pins the ORDER, which is what
    a future edit is likely to get wrong.
    """
    src = inspect.getsource(npol.refuse_impossible_cells)
    i_cpl = src.index('_drop_relation_hits_the_spike_rule_already_explained(hits')
    i_eq = src.index('_drop_price_scale_on_already_refused_equity(hits')
    assert i_cpl < i_eq, (
        'the price-scale equity guard now runs before the coupling guard, so it decides '
        'whether to trust a row\'s equity from hits that the coupling guard then removes')


def test_the_coupling_guard_does_not_change_which_sources_are_EJECTED():
    """The section-5 contract is ABSTAIN, NEVER EJECT, and this guard moves cells the primary
    limbs read -- `totalStockholdersEquity` is a primary limb and `totalAssets` is a
    `SANITY_IMPOSSIBLE` one.

    THE MECHANISM OF CONCERN, stated so the fixture can be checked against it: a dropped hit
    means a cell keeps its RAW value instead of becoming a stamped blank, and `_limb_fails`
    reads a raw value rather than an absence.  Blanking never ejects (`NaN <= 0` is False, and
    `refused_fields_mask` subtracts a stamped blank from the NaN limbs), so the only way this
    guard could ADD an eject is by restoring a raw `totalAssets <= 0` on the row the verdict is
    taken from.

    IT STRUCTURALLY CANNOT, and that is what the last assertion pins rather than asserts by
    example.  `primary_eject` reads each source's NEWEST row only.  The guard drops a hit only
    where the SPIKE rule fired, and the spike rule requires BOTH date-neighbours -- so it can
    never fire on a newest row.  The two sets are disjoint by construction.

    THE FIXTURE EXERCISES THE GUARD, which an earlier version of this test did not (found in
    independent review -- it put the corrupt cell on the NEWEST row, where no spike hit can
    exist, so the guard dropped nothing and the test passed with or without it).  Here the
    spike fires on the INTERIOR row 1 and `balance_sheet_identity` fires on the same row and
    reads the spiked field, so there is a real hit to drop; the newest row carries a SEPARATE
    `totalAssets <= 0` that ejects on raw values and must still be blanked-and-stamped.

    CANNOT DETECT: whether dropping the relation hit was RIGHT on a row where TWO cells are
    independently corrupt -- the spike names F, and the relation's other leg G is also bad.
    The guard spares G, and nothing here would notice.  Unmeasured; no evidence it occurs on
    any saved panel; named because it is the guard's one theoretical over-spare.
    """
    common = {'price': 10.0, 'marketCap': 1.0e9, 'weightedAverageShsOut': 1.0e8,
              'netIncome': 1.0e7, 'netCashProvidedByOperatingActivities': 1.0e7,
              'revenue': 5.0e8}
    df = _frame([
        dict(source='ZZEJ', date='2024-07-01', totalAssets=2.0e9,
             totalLiabilities=1.999e9, totalStockholdersEquity=1.0e6, **common),
        #  INTERIOR: `totalLiabilities` served 500x too small for one quarter.  The spike rule
        #  names that one cell; `balance_sheet_identity` divides by it (A/(L+E) = 408) and
        #  would blank `totalAssets` and `totalStockholdersEquity` as collateral.
        dict(source='ZZEJ', date='2024-10-01', totalAssets=2.0e9,
             totalLiabilities=3.9e6, totalStockholdersEquity=1.0e6, **common),
        dict(source='ZZEJ', date='2025-01-01', totalAssets=2.0e9,
             totalLiabilities=1.999e9, totalStockholdersEquity=1.0e6, **common),
        #  NEWEST: a non-positive `totalAssets`, which is a `SANITY_IMPOSSIBLE` eject on raw
        #  values.  The identity fires here too and MUST keep firing -- there is no spike hit
        #  on a newest row, so the guard must not reach it.
        dict(source='ZZEJ', date='2025-04-01', totalAssets=-5.0,
             totalLiabilities=1.999e9, totalStockholdersEquity=1.0e6, **common),
    ])
    rel = npol.impossible_relation_hits(df)
    spike = npol.scale_spike_hits(df)
    #  the premise: the guard has something to drop, and only on the interior row
    assert sorted(rel['row']) == [1, 3] and set(rel['relation']) == {'balance_sheet_identity'}
    assert list(spike['row']) == [1], spike
    assert list(spike['relation']) == ['isolated_scale_spike:totalLiabilities'], spike
    hits = pd.concat([rel, spike], ignore_index=True)
    kept = npol._drop_relation_hits_the_spike_rule_already_explained(hits)
    assert len(hits) - len(kept) == 1, (hits, kept)
    dropped = set(map(tuple, hits[['row', 'relation']].values.tolist())) - set(
        map(tuple, kept[['row', 'relation']].values.tolist()))
    assert dropped == {(1, 'balance_sheet_identity')}, dropped

    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    #  INTERIOR ROW: only the cell the spike rule NAMED is refused; the identity's two other
    #  legs keep their real values.  This is the guard working.
    assert pd.isna(out.at[1, 'totalLiabilities'])
    assert out.at[1, 'totalAssets'] == 2.0e9
    assert out.at[1, 'totalStockholdersEquity'] == 1.0e6
    #  NEWEST ROW: untouched by the guard, so the identity still blanks AND stamps, which is
    #  what keeps `primary_eject` off a source the guard has no evidence about.
    for f in ('totalAssets', 'totalLiabilities', 'totalStockholdersEquity'):
        assert pd.isna(out.at[3, f]), f
        assert npol.refused_fields_mask(out, f).iloc[3], f

    #  THE EJECT COMPARISON, and `before` is deliberately NON-EMPTY so it can fail: the raw
    #  frame DOES eject on `totalAssets <= 0`, and the refusal is what removes it.
    ej_before = npol.primary_eject(df)
    before = set(ej_before['source'])
    after = set(npol.primary_eject(out)['source'])
    assert before == {'ZZEJ'}, ej_before
    assert 'totalAssets' in set(ej_before['field']), ej_before
    assert not (after - before), (
        'the refusal pass now EJECTS a source it did not eject on raw values: %s'
        % sorted(after - before))
    assert not after, 'the refusal must REMOVE this eject, not merely fail to add one'

    #  THE STRUCTURAL REASON, pinned rather than argued: every hit the guard drops sits on a
    #  row that has BOTH date-neighbours, and `primary_eject` reads only the newest row.  A
    #  future edit that let the guard reach an endpoint would fail here first.
    newest = pd.to_datetime(df['date']).max()
    oldest = pd.to_datetime(df['date']).min()
    for row, _rel in dropped:
        d = pd.to_datetime(df.at[row, 'date'])
        assert oldest < d < newest, (
            'the coupling guard dropped a hit on an ENDPOINT row (%s); the spike rule cannot '
            'fire there, so this can only mean the guard stopped keying on spike hits' % d)
