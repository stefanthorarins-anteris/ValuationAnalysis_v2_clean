"""Cross-field impossibility (nan_policy section 5) -- it must ABSTAIN, and it must not
over-reach.

Every fixture below is the REAL row from the 2026-08-13 CUR3K panel, not a synthetic, so a
test failing here means the shipped rule stopped catching the shipped defect.
"""
import numpy as np
import pandas as pd
import pytest

import nan_policy as npol


def _frame(rows):
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  THE TWO NAMES THAT MOTIVATED THE CHANGE                                     #
# --------------------------------------------------------------------------- #
def test_jhx_newest_row_assets_are_one_millionth_of_liabilities_plus_equity():
    """JHX 2026-04-01: totalAssets 1.3493e4 while L + E = 1.34934e10.

    THE IDENTITY LIMB MUST BE TWO-SIDED.  The first implementation tested only
    `ratio >= factor` and MISSED this -- the ratio is 1e-6, not 1e6 -- which is the whole
    reason `IMPOSSIBLE_RELATIONS` carries a `two_sided` flag.
    """
    df = _frame([{'source': 'JHX', 'date': '2026-04-01', 'totalAssets': 1.3493e4,
                  'totalLiabilities': 6.9528e9, 'totalStockholdersEquity': 6.5406e9}])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert set(rep['relation']) == {'balance_sheet_identity'}
    assert np.isnan(out.loc[0, 'totalAssets'])
    #  BOTH sides refused: the panel does not say which of the two is wrong.
    assert np.isnan(out.loc[0, 'totalLiabilities'])
    assert np.isnan(out.loc[0, 'totalStockholdersEquity'])


def test_rdzn_current_liabilities_exceed_total_liabilities_by_a_million():
    """RDZN 2024-10-01: totalCurrentLiabilities 6.136344e13, totalLiabilities 6.2476e7."""
    df = _frame([{'source': 'RDZN', 'date': '2024-10-01',
                  'totalCurrentLiabilities': 6.136344e13, 'totalLiabilities': 6.247641e7}])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert 'current_liabilities_within_liabilities' in set(rep['relation'])
    assert np.isnan(out.loc[0, 'totalCurrentLiabilities'])


def test_rdzn_sga_spike_is_caught_by_the_time_rule_not_by_a_ratio_bound():
    """RDZN 2025-10-01: SG&A 1.090059e13 between 1.006e7 and 1.542e7.

    No cross-field identity contains it -- `SGA <= grossProfit - operatingIncome` was
    measured and REFUSED because its denominator degenerates on real companies -- so this is
    what the isolated-spike rule exists for.
    """
    df = _frame([
        {'source': 'RDZN', 'date': '2025-07-01',
         'sellingGeneralAndAdministrativeExpenses': 1.005913e7},
        {'source': 'RDZN', 'date': '2025-10-01',
         'sellingGeneralAndAdministrativeExpenses': 1.090059e13},
        {'source': 'RDZN', 'date': '2026-01-01',
         'sellingGeneralAndAdministrativeExpenses': 1.541902e7},
    ])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert list(rep['relation']) == [
        'isolated_scale_spike:sellingGeneralAndAdministrativeExpenses']
    assert np.isnan(out.loc[1, 'sellingGeneralAndAdministrativeExpenses'])
    #  the neighbours are untouched -- a spike refuses ONE cell, not the series
    assert out.loc[0, 'sellingGeneralAndAdministrativeExpenses'] == 1.005913e7
    assert out.loc[2, 'sellingGeneralAndAdministrativeExpenses'] == 1.541902e7


def test_hban_current_liabilities_collapse_is_caught():
    """HBAN 2025-07-01: 2.52e8 between two ~1.6e11 quarters -- a real $200bn bank.

    Included because it is the evidence that these unit errors are NOT confined to
    micro-caps, which is the finding that most justifies shipping the rule.
    """
    df = _frame([
        {'source': 'HBAN', 'date': '2025-04-01', 'totalCurrentLiabilities': 1.63956e11},
        {'source': 'HBAN', 'date': '2025-07-01', 'totalCurrentLiabilities': 2.52e8},
        {'source': 'HBAN', 'date': '2025-10-01', 'totalCurrentLiabilities': 1.47099e11},
    ])
    _, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 1 and rep.iloc[0]['field'] == 'totalCurrentLiabilities'


# --------------------------------------------------------------------------- #
#  IT MUST NOT OVER-REACH -- a genuinely unusual company still scores          #
# --------------------------------------------------------------------------- #
def test_pre_revenue_company_with_sga_thousands_of_times_revenue_is_untouched():
    """GXAI: sgaTTM/salesTTM of 9,456 -- LEGITIMATE (pre-revenue, ramping).

    This is why the proposed `SGA / revenue` threshold was refused: GXAI's legitimate value
    and RDZN's corrupt one are only 20x apart on that axis.  An identity has no such tail, so
    a pre-revenue company must pass every relation.
    """
    df = _frame([{'source': 'GXAI', 'date': '2024-04-01', 'revenue': 275.0,
                  'sellingGeneralAndAdministrativeExpenses': 2.6e6,
                  'totalAssets': 1.0e7, 'totalLiabilities': 7.2e5,
                  'totalStockholdersEquity': 9.28e6,
                  'totalCurrentAssets': 8.0e6, 'totalCurrentLiabilities': 5.0e5,
                  'propertyPlantEquipmentNet': 1.0e5}])
    _, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 0, ('a pre-revenue company was refused: %s'
                           % rep[['relation', 'field']].to_dict('records'))


def test_leveraged_balance_sheet_with_negative_equity_is_untouched():
    """Liabilities far exceeding equity is DISTRESS, not corruption; the identity still holds."""
    df = _frame([{'source': 'LEVERED', 'date': '2026-01-01', 'totalAssets': 3.2e7,
                  'totalLiabilities': 6.2e7, 'totalStockholdersEquity': -3.0e7,
                  'totalCurrentLiabilities': 5.8e7,
                  'totalCurrentAssets': 3.0e7, 'propertyPlantEquipmentNet': 1.9e6}])
    _, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 0


def test_bank_with_current_liabilities_far_above_reported_total_liabilities_is_untouched():
    """323410.KS (KakaoBank) reaches TCL/TL = 247.5 under FMP's populating convention, and
    SLF.TO (Sun Life) 18.9.  A factor-10 or factor-100 cut would refuse their rows -- which is
    why that one limb carries 500 and not 10.  The empty band is 247.5 -> 957.7, so 500 clears
    the bank by 2x and still catches ALAQU.PA's six rows, which a cut at 1,000 missed."""
    df = _frame([{'source': '323410.KS', 'date': '2023-04-01',
                  'totalCurrentLiabilities': 2.571016e13, 'totalLiabilities': 1.03896e11,
                  'totalAssets': 5.052698e13, 'totalStockholdersEquity': 5.87806e12}])
    _, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 0, (
        'a real bank was refused at TCL/TL = 247.5; the limb must clear it.')
    #  ALAQU.PA at 957.7-999.8 must be caught -- the cut at 1,000 sat ON that cluster and
    #  fired on 1 row while its comment claimed 7.
    alaqu = _frame([{'source': 'ALAQU.PA', 'date': '2016-04-01',
                     'totalCurrentLiabilities': 4764730.0, 'totalLiabilities': 4975.0}])
    _, rep2 = npol.refuse_impossible_cells(alaqu, verbose=False)
    assert 'current_liabilities_within_liabilities' in set(rep2['relation'])


def test_a_sustained_level_change_is_not_a_spike():
    """SOFI's SPAC merger: totalAssets 4.66e5 -> 8.56e9 -> 8.06e8.  The level MOVES and STAYS,
    so it is not a spike and must not be refused."""
    df = _frame([
        {'source': 'SOFI', 'date': '2020-07-01', 'totalAssets': 4.66179e5},
        {'source': 'SOFI', 'date': '2020-10-01', 'totalAssets': 8.563499e9},
        {'source': 'SOFI', 'date': '2021-01-01', 'totalAssets': 8.058174e8},
    ])
    _, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 0


def test_a_near_zero_earnings_quarter_is_not_a_spike():
    """BMV.L earns GBP 24 between two GBP 1-3M quarters, VIVK's CFO passes through 35.

    A FLOW legitimately passes near zero.  That is why the spike field set is restricted to
    balance-sheet STOCKS plus an UP-ONLY limb on SG&A -- applying it to netIncome or operating
    cash flow over-reaches immediately and measurably.
    """
    assert 'netIncome' not in npol.SCALE_SPIKE_FIELDS
    assert 'netCashProvidedByOperatingActivities' not in npol.SCALE_SPIKE_FIELDS
    assert 'ebitda' not in npol.SCALE_SPIKE_FIELDS
    df = _frame([
        {'source': 'BMV.L', 'date': '2024-04-01', 'netIncome': 9.22e5},
        {'source': 'BMV.L', 'date': '2024-10-01', 'netIncome': 24.142},
        {'source': 'BMV.L', 'date': '2025-04-01', 'netIncome': 2.622263e6},
    ])
    _, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 0


def test_sga_down_spike_is_allowed_up_spike_is_not():
    """A shell reporting almost no SG&A for a quarter is odd; one reporting 1,000x its own
    neighbours while everything else continues normally is not a quarter a going concern has."""
    assert npol.SCALE_SPIKE_FIELDS['sellingGeneralAndAdministrativeExpenses'] == 'up'
    down = _frame([
        {'source': 'S', 'date': '2024-01-01',
         'sellingGeneralAndAdministrativeExpenses': 2.1e6},
        {'source': 'S', 'date': '2024-04-01',
         'sellingGeneralAndAdministrativeExpenses': 6.1e2},
        {'source': 'S', 'date': '2024-07-01',
         'sellingGeneralAndAdministrativeExpenses': 6.15e5},
    ])
    _, rep = npol.refuse_impossible_cells(down, verbose=False)
    assert len(rep) == 0


# --------------------------------------------------------------------------- #
#  IT ABSTAINS -- it never ejects                                              #
# --------------------------------------------------------------------------- #
def test_it_never_removes_a_row_or_a_source():
    """The CEO ruled ABSTAIN.  `data_quality.check_price_sanity` was the obvious home and is
    the wrong one: a False there makes `filter_invalid_data` delete every row AT OR BEFORE the
    flagged one, so JHX -- whose corrupt cell is its NEWEST row -- would lose its entire
    history."""
    df = _frame([{'source': 'JHX', 'date': '2026-04-01', 'totalAssets': 1.3493e4,
                  'totalLiabilities': 6.9528e9, 'totalStockholdersEquity': 6.5406e9,
                  'revenue': 1.4746e9},
                 {'source': 'JHX', 'date': '2026-01-01', 'totalAssets': 1.36886e10,
                  'totalLiabilities': 7.2631e9, 'totalStockholdersEquity': 6.4255e9,
                  'revenue': 1.4039e9}])
    out, _ = npol.refuse_impossible_cells(df, verbose=False)
    assert len(out) == len(df)
    assert list(out['source']) == list(df['source'])
    #  fields the relation does not name are untouched -- this is a CELL refusal
    assert out.loc[0, 'revenue'] == 1.4746e9


def test_a_refused_primary_field_does_NOT_eject_the_source():
    """THE BLOCKING DEFECT THIS FILE ORIGINALLY MISSED (review S1, 2026-08-14).

    `test_it_never_removes_a_row_or_a_source` above promises exactly this property IN ITS NAME
    and asserts it only about `refuse_impossible_cells` IN ISOLATION -- which is why it passed
    while the change ejected JHX and SZZL from the universe.  `totalStockholdersEquity` is in
    `PRIMARY_PRESENT`, whose limb is `isna()` on the newest row, and `primary_eject` removes
    the WHOLE SOURCE.  Blanking equity therefore deleted the $10bn company the guard exists to
    protect.  This test closes the door one along, against the real coupling.
    """
    df = _frame([
        {'source': 'JHX', 'date': '2026-01-01', 'price': 40.0, 'marketCap': 1.0e10,
         'weightedAverageShsOut': 2.5e8, 'netIncome': 2.85e7,
         'netCashProvidedByOperatingActivities': 1.0e8,
         'totalStockholdersEquity': 6.4255e9, 'totalAssets': 1.36886e10,
         'totalLiabilities': 7.2631e9, 'revenue': 1.4039e9},
        {'source': 'JHX', 'date': '2026-04-01', 'price': 40.0, 'marketCap': 1.0e10,
         'weightedAverageShsOut': 2.5e8, 'netIncome': 1.043e8,
         'netCashProvidedByOperatingActivities': 1.0e8,
         'totalStockholdersEquity': 6.5406e9, 'totalAssets': 1.3493e4,
         'totalLiabilities': 6.9528e9, 'revenue': 1.4746e9},
    ])
    assert npol.primary_eject(df, verbose=False).empty, 'fixture must not eject before the guard'
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert np.isnan(out.loc[1, 'totalStockholdersEquity']), 'the cell must still be refused'
    ej = npol.primary_eject(out, verbose=False)
    assert ej.empty, (
        'the input-sanity guard EJECTED the source through primary_eject: %s. A refused cell '
        'is a deliberate abstention, not an absent primary input.'
        % ej[['source', 'field', 'limb']].to_dict('records'))


def test_the_refusal_stamp_is_carried_on_the_row_and_is_idempotent():
    df = _frame([
        {'source': 'X', 'date': '2026-01-01', 'totalAssets': 1.0e10,
         'totalLiabilities': 5.0e9, 'totalStockholdersEquity': 5.0e9},
        {'source': 'X', 'date': '2026-04-01', 'totalAssets': 1.0e4,
         'totalLiabilities': 5.0e9, 'totalStockholdersEquity': 5.0e9},
    ])
    out, _ = npol.refuse_impossible_cells(df, verbose=False)
    stamp = str(out.loc[1, npol.SANITY_REFUSED_COLUMN])
    assert set(stamp.split('|')) == {'totalAssets', 'totalLiabilities',
                                     'totalStockholdersEquity'}
    assert not str(out.loc[0, npol.SANITY_REFUSED_COLUMN] or '')
    #  running the guard twice must refuse nothing further -- filter_invalid_data runs TWICE
    _, rep2 = npol.refuse_impossible_cells(out, verbose=False)
    assert len(rep2) == 0


def test_a_frame_without_the_stamp_column_behaves_exactly_as_before():
    """Every pre-section-5 panel and every external caller keeps its previous verdict."""
    df = _frame([{'source': 'Y', 'date': '2026-01-01', 'price': 10.0, 'marketCap': 1.0e9,
                  'weightedAverageShsOut': 1.0e8, 'netIncome': 1.0e7,
                  'netCashProvidedByOperatingActivities': 1.0e7,
                  'totalStockholdersEquity': np.nan, 'totalAssets': 1.0e9,
                  'revenue': 1.0e8}])
    assert npol.SANITY_REFUSED_COLUMN not in df.columns
    ej = npol.primary_eject(df, verbose=False)
    assert len(ej) == 1 and ej.iloc[0]['field'] == 'totalStockholdersEquity', (
        'a genuinely ABSENT primary input must still eject -- the exemption is for REFUSED '
        'cells only, not a blanket amnesty on NaN.')


def test_the_evidence_csv_logs_the_original_value_not_a_value_it_already_blanked():
    """Review S4: 29 of 326 records logged NaN because the value was read from the frame the
    loop was mutating.  ALAQU.PA-shaped: one row failing two relations that share a field."""
    df = _frame([{'source': 'ALAQU.PA', 'date': '2016-04-01', 'totalAssets': 7213.0,
                  'totalCurrentAssets': 6185856.0, 'totalLiabilities': 4975.0,
                  'totalStockholdersEquity': 2237810.0,
                  'propertyPlantEquipmentNet': 182518.0}])
    _, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) > 0
    assert rep['value'].notna().all(), (
        'the CSV logged NaN as an ORIGINAL value for %d record(s); its stated purpose is that '
        'a refusal can be argued with.'
        % int(rep['value'].isna().sum()))
    ta = rep[rep['field'] == 'totalAssets']['value'].unique()
    assert list(ta) == [7213.0]


def test_a_duplicated_index_does_not_raise_and_preserves_the_caller_index():
    """Review S9: `float(ratio.loc[idx])` raised TypeError on duplicate labels -- in the fetch
    the per-ticker guard swallowed it and the abstention silently did not happen."""
    df = _frame([
        {'source': 'S', 'date': '2025-04-01', 'totalCurrentLiabilities': 1.63956e11},
        {'source': 'S', 'date': '2025-07-01', 'totalCurrentLiabilities': 2.52e8},
        {'source': 'S', 'date': '2025-10-01', 'totalCurrentLiabilities': 1.47099e11},
    ])
    df.index = [7, 7, 7]
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 1
    assert list(out.index) == [7, 7, 7], 'the caller must not get a reindexed frame back'


def test_absent_columns_skip_the_relation_rather_than_failing_it():
    """A ragged payload must not be reported as corruption."""
    df = _frame([{'source': 'X', 'date': '2026-01-01', 'totalAssets': 1.0e9}])
    out, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 0 and out.loc[0, 'totalAssets'] == 1.0e9


def test_empty_and_none_frames_are_safe():
    out, rep = npol.refuse_impossible_cells(pd.DataFrame([]), verbose=False)
    assert len(rep) == 0
    out, rep = npol.refuse_impossible_cells(None, verbose=False)
    assert out is None and len(rep) == 0


def test_orientation_does_not_change_the_answer():
    """This repo carries both newest-first and oldest-first frames; the spike rule sorts."""
    rows = [
        {'source': 'S', 'date': '2025-04-01', 'totalCurrentLiabilities': 1.63956e11},
        {'source': 'S', 'date': '2025-07-01', 'totalCurrentLiabilities': 2.52e8},
        {'source': 'S', 'date': '2025-10-01', 'totalCurrentLiabilities': 1.47099e11},
    ]
    _, a = npol.refuse_impossible_cells(_frame(rows), verbose=False)
    _, b = npol.refuse_impossible_cells(_frame(rows[::-1]), verbose=False)
    assert len(a) == len(b) == 1
    assert a.iloc[0]['value'] == b.iloc[0]['value'] == 2.52e8


# --------------------------------------------------------------------------- #
#  THE TABLE ITSELF                                                            #
# --------------------------------------------------------------------------- #
def test_only_the_identity_is_two_sided():
    two = {n for n, _, _, _, ts, _ in npol.IMPOSSIBLE_RELATIONS if ts}
    assert two == {'balance_sheet_identity'}, (
        'a CONTAINMENT relation must not be two-sided: current assets being far SMALLER than '
        'total assets is the normal case.')


def test_containment_relations_do_not_fire_on_the_normal_case():
    df = _frame([{'source': 'N', 'date': '2026-01-01', 'totalAssets': 1.0e9,
                  'totalCurrentAssets': 1.0e5, 'propertyPlantEquipmentNet': 1.0e3,
                  'totalCurrentLiabilities': 1.0e3, 'totalLiabilities': 5.0e8,
                  'totalStockholdersEquity': 5.0e8}])
    _, rep = npol.refuse_impossible_cells(df, verbose=False)
    assert len(rep) == 0
