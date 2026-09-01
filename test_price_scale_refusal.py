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
  * It cannot detect the rule's measured UNDER-REACH.  Rows between 0.02 and 0.10 carry the
    same share-count corruption signature at the same rate and are deliberately left alone
    (density table in `nan_policy`, beside the constant).  Nothing here fails if that
    population grows.
  * It cannot see a name ABSENT FROM THE PANEL, which is the population ATRI is actually in.

ONE MEASUREMENT LEAD, RECORDED AND DELIBERATELY NOT CHASED.  Re-scoring the saved 2026-08-29
panel at HEAD moves 397 of 4,934 sources by exact multiples of 0.3/8, always downward.  It does
NOT gate this change -- every outcome leads to the same action (a fresh single-basis panel), the
top-100 is unaffected, and a test that cannot change the decision should not be run.  The likely
explanation, unconfirmed: `Sbocker's getAves2-on-the-filtered-frame note` records that recomputing `getAves2` on the FILTERED
frame moves 26 of 36 medians by more than 1%, which would produce exactly that signature with no
defect present at all.
"""
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


def test_a_refused_newest_earningsYield_does_NOT_fall_back_to_the_vendor_PE():
    """THE DELIVERABLE, one function away from the market-cap band the author already swept.

    `_pe_ratio_from_panel` returns None for a missing OR non-positive yield and the caller then
    publishes FMP's `priceEarningsRatio`.  That is right for a LOSS-MAKER.  It is wrong for a
    units-error refusal: the vendor's P/E is derived from the SAME price the refusal rejected,
    so on the ATRI shape it is 1/1000 of the truth and on the QBY0.DE shape ~100x too small --
    an absurdly CHEAP positive P/E beside a name in the CEO's list.

    CANNOT DETECT: whether any refused name actually reaches the top-N in a real run.  That is
    unmeasured; `carveOut.marketcap_fallback_report` existing at all is the evidence that
    refused newest rows do survive into the deliverable stages.
    """
    import inspect
    import postBo

    raw = _atri_shape(n=6, broken=(0,))          # newest row refused
    refused, _rep = npol.refuse_impossible_cells(raw.copy(), verbose=False)
    assert postBo._pe_refused_sources(refused) == {'ATRISHAPE'}
    #  a clean panel names nobody -- the branch must not fire unconditionally
    assert postBo._pe_refused_sources(_atri_shape(n=6, broken=())) == set()
    #  ... and our own P/E is indeed unavailable for it, i.e. the vendor branch is reachable
    assert postBo._pe_ratio_from_panel(postBo._pe_panel_table(refused), 'ATRISHAPE') is None

    #  THE ORDERING IS THE BEHAVIOUR: the refusal check must be consulted BEFORE the vendor
    #  fallback, or the fallback wins and the check is decoration.  Same idiom as
    #  `test_vendor_contamination.test_the_quarantine_runs_BEFORE_the_arithmetic_checks`.
    #  THE DECISION ITSELF, called rather than read.  A source-order assertion survives the
    #  branch being deleted, because the token still appears where the set is built -- which is
    #  exactly what a mutation showed.
    assert postBo._pe_cell(None, 'ATRISHAPE', {'ATRISHAPE'}, 5.0) == ('NaN', 'refused'), (
        'a REFUSED name still publishes the vendor P/E, which is derived from the price the '
        'refusal rejected')
    #  a loss-maker is NOT collateral damage: it keeps the vendor fallback
    assert postBo._pe_cell(None, 'OTHER', {'ATRISHAPE'}, 5.0) == ('5.0000', 'vendor')
    #  and our own answer still wins outright
    assert postBo._pe_cell(2.0, 'ATRISHAPE', {'ATRISHAPE'}, 5.0) == ('2.0000', 'ours')
    #  the vendor sign test survives the extraction
    assert postBo._pe_cell(None, 'OTHER', set(), -3.0) == ('NaN', 'none')

    #  THE LOG'S BOOKKEEPING, which decides whether a refused name is REPORTED as a refusal or
    #  misattributed as a loss-maker.  The cell itself is pinned behaviourally above; this half
    #  lives inside a ~300-line function and is asserted STRUCTURALLY, which is weaker and is
    #  said so.  Both branches must exist and must feed DIFFERENT lists.
    src = inspect.getsource(postBo.writeBoAggToCSV)
    assert "_pe_tag == 'refused'" in src and '_pe_refused.append' in src, (
        'a refused name no longer reaches the refused list, so the log will report it as a '
        'loss-maker that fell back -- the misattribution this fix exists to remove')
    assert "_pe_tag == 'vendor'" in src and '_pe_vendor_fallback.append' in src
    assert src.index("_pe_tag == 'refused'") < src.index("_pe_tag == 'vendor'")


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
