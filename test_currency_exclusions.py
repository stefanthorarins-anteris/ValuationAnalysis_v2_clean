"""Reporting-currency exclusions: ARS leaves the universe, and the reconciliation holds.

OFFLINE.  The rule reads one column of the panel; nothing here touches a network.

THE CASE (CEO ruling, 2026-08-08).  ARS reporters used to be KEPT and merely abstained on
for FX (`fx_rates.ABSTAIN_CURRENCIES`).  That was ruled insufficient: abstention fixes the
CURRENCY, while the STATEMENTS are broken -- `BMA`'s `totalAssets` = 2.325e16 sitting
between 2.056e13 and 2.420e13 (961x) and `revenue` = 1.201e9 between 1.443e12 and 1.907e12
(1,588x), with `marketCap` continuous through both -- and every metric downstream of those
statements is contaminated, not just the size band.  ARS reporters are now EXCLUDED.

MEASURED THROUGH THIS CODE on the real 2026-08-07 CUR3K panel (61,007 rows, 2,613 sources):
2 sources / 48 rows removed (`BMA` 24, `CRESY` 24), panel 2,613 -> 2,611, and the universe
identity `resolved == panel + failed + removed + residual` balances at residual 0.

THE SECOND HALF OF THIS FILE is about that identity.  The old removed-source counter counted
every source with ANY removed row, so a PARTIAL removal -- rows dropped, name survives --
was subtracted from the universe while still sitting in the panel, driving the residual
NEGATIVE.  Measured on the same panel: old **108** vs true **84**, residual **-24**, and the
"UNIVERSE DOES NOT RECONCILE" banner fires on a run that reconciles perfectly.  A
reconciliation that cries wolf is worse than none.

**THE MISCOUNT IS NOT A QUARANTINE ARTEFACT** (corrected 2026-08-08, reviewer F-2).  Of the
24 partially-removed sources, **23 are ordinary `data_before_corruption` PASS-3 trims** and
only **1** is the `058820.KQ` quarantine.  PASS 3 has behaved this way since it was written,
so the counter has been wrong on every panel carrying any partial trim; 817df52 added the
24th case, it did not cause the class.  What hid it until now was the accumulate-never-assign
bug fixed on 2026-08-07, which destroyed the removal frame before anything could count it.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import carveOut as co
import currency_exclusions as cx
import data_quality as dq
import fx_rates as fx


def _panel(currencies, source='BMA', n=12, start='2020-06-30'):
    """One source's panel, with `currencies` broadcast or given per row."""
    ends = pd.date_range(start, periods=n, freq='QE')
    if isinstance(currencies, str):
        currencies = [currencies] * n
    return pd.DataFrame({
        'source': [source] * n,
        'periodEndDate': ends,
        'date': [pd.Timestamp(d.year, ((d.month - 1) // 3) * 3 + 1, 1) for d in ends],
        'reportedCurrency': currencies,
        'revenue': np.linspace(4.18e10, 1.91e12, n),
        'totalAssets': np.linspace(6.25e11, 2.42e13, n),
        'marketCap': np.linspace(1.27e11, 7.29e12, n),
        'price': np.linspace(1994.0, 1.14e5, n),
        'weightedAverageShsOut': [6.4e8] * n,
    })


def _clean_panel(source='AAPL', n=12):
    p = _panel('USD', source=source, n=n)
    p['revenue'] = np.linspace(9.0e10, 1.2e11, n)
    p['totalAssets'] = np.linspace(3.2e11, 3.6e11, n)
    p['marketCap'] = np.linspace(2.0e12, 3.0e12, n)
    p['price'] = np.linspace(120.0, 190.0, n)
    return p


# --------------------------------------------------------------------------- #
#  The rule itself                                                            #
# --------------------------------------------------------------------------- #
def test_the_rule_is_NAMED_DATED_and_carries_its_EVIDENCE():
    """Same bar as vendor_contamination.QUARANTINE_RULES: a name that leaves the universe
    without an artifact stating WHY is the defect this whole line of work removes."""
    assert set(cx.EXCLUDED_CURRENCIES) == {'ARS'}
    rule = cx.EXCLUDED_CURRENCIES['ARS']
    assert rule.currency == 'ARS'
    assert rule.added == '2026-08-08'
    assert 'IAS 29' in rule.reason
    for token in ('2.325e16', '961x', '1,588x', 'marketCap continuous through both',
                  'CRESY'):
        assert token in rule.evidence, token
    label = rule.label()
    assert label.startswith('currency_excluded [ARS:')
    assert '2.325e16' in label, 'the evidence must travel in the removal reason itself'


def test_the_REASON_does_not_assert_of_one_NAME_what_was_measured_on_ANOTHER():
    """Reviewer F-7.  The reason string is stamped onto EVERY excluded source's rows, so it
    must be a statement about the CURRENCY.  An earlier draft said 'plus a vendor-side scale
    defect', which is true of BMA and false of CRESY -- and CRESY's rows carried it."""
    rule = cx.EXCLUDED_CURRENCIES['ARS']
    assert 'defect' not in rule.reason.split('this vendor is')[0], \
        'the reason must not attribute a defect to the name being removed'
    #  The per-name specifics live in the EVIDENCE, and the evidence says out loud that
    #  CRESY is not damaged.
    assert 'CRESY shows NO defect' in rule.evidence
    assert '7.5x nominal rise' in rule.evidence and '45.6x' in rule.evidence


def test_it_is_CURRENCY_scoped_and_NOT_a_ticker_list():
    """CEO's explicit choice: the defect is the hyperinflation accounting regime, so a
    NEW ARS reporter that never existed when the rule was written is excluded too.  A
    ticker list would admit it silently."""
    src = open(os.path.join(_HERE, 'currency_exclusions.py'), encoding='utf-8').read()
    assert "'BMA'" not in src.split('"""', 2)[-1], \
        'no ticker may appear in the executable half of the rule'
    hits = cx.excluded_sources(_panel('ARS', source='SOME.NEW.ARS.NAME'))
    assert set(hits) == {'SOME.NEW.ARS.NAME'}


def test_it_excludes_the_WHOLE_SOURCE_not_just_the_ARS_rows():
    """Source-scoped by design.  Removing only the ARS-labelled rows would leave a
    truncated series that still gets scored -- and a name that CHANGED reporting currency
    mid-history is where restatement noise is WORST, not a case for keeping half."""
    mixed = _panel(['USD'] * 6 + ['ARS'] * 6, source='BMA')
    mask, records = cx.exclusion_records(mixed)
    assert mask.all(), 'every row of the source leaves, including the non-ARS rows'
    assert len(records) == 12


def test_a_NON_excluded_currency_in_the_same_frame_is_untouched():
    frame = pd.concat([_panel('ARS', source='BMA'), _clean_panel('AAPL')],
                      ignore_index=True)
    mask, _rec = cx.exclusion_records(frame)
    assert set(frame.loc[mask, 'source']) == {'BMA'}
    assert set(frame.loc[~mask, 'source']) == {'AAPL'}


@pytest.mark.parametrize('vendor_string', ['ARS', 'ars', ' ARS ', 'Ars'])
def test_vendor_CASE_and_WHITESPACE_cannot_defeat_the_rule(vendor_string):
    """A rule that misses because of a trailing space is a rule that silently does not
    exist, and `reportedCurrency` is a vendor string we do not control."""
    assert set(cx.excluded_sources(_panel(vendor_string))) == {'BMA'}


def test_a_panel_with_NO_currency_column_reports_the_MISS_rather_than_a_clean_zero():
    """Every panel fetched before `reportedCurrency` was folded into ingest has no such
    column -- including the 2026-01-08 NA1_EU1 panel, which CONTAINS BMA and CRESY.  The
    rule cannot fire there, and saying '0 excluded' would be a false negative."""
    no_ccy = _panel('ARS').drop(columns=['reportedCurrency'])
    ok, note = cx.applicable(no_ccy)
    assert not ok
    assert 'reportedCurrency' in note and 'CANNOT be applied' in note
    assert cx.excluded_sources(no_ccy) == {}


def test_an_unusable_frame_never_raises():
    for frame in (None, pd.DataFrame(), pd.DataFrame({'x': [1]})):
        ok, _note = cx.applicable(frame)
        assert not ok
        assert cx.excluded_sources(frame) == {}
        mask, records = cx.exclusion_records(frame)
        assert not mask.any() and records == []


# --------------------------------------------------------------------------- #
#  Wired into data_quality -- the part that survives a re-fetch               #
# --------------------------------------------------------------------------- #
def test_the_filter_REMOVES_the_source_and_LOGS_it_with_the_EVIDENCE(capsys):
    frame = pd.concat([_panel('ARS', source='BMA'), _clean_panel('AAPL')],
                      ignore_index=True)
    clean, removed = dq.filter_invalid_data(frame, min_periods_required=4, verbose=True)
    out = capsys.readouterr().out
    assert set(clean['source']) == {'AAPL'}
    c = removed[removed['removal_reason'].str.startswith('currency_excluded')]
    assert len(c) == 12
    assert c['removal_reason'].str.contains('961x').all()
    assert 'REPORTING-CURRENCY EXCLUSION: removed 1 source(s) entirely' in out
    assert 'BMA' in out


def test_the_removal_reaches_the_TRANSPARENCY_frame_not_just_the_panel():
    """`removed_df` is what drives removed_data_quality_*.csv (repo root since 2026-08-10) -- which ships
    with output/ -- plus the BoMetric_df propagation and the reconciliation counters.
    Deleting rows while reporting nothing removed is this project's signature defect."""
    frame = _panel('ARS')
    _clean, removed = dq.filter_invalid_data(frame, min_periods_required=4, verbose=False)
    assert {'source', 'date', 'removal_reason'} <= set(removed.columns)
    assert set(removed['source']) == {'BMA'}


def test_it_propagates_to_BoMetric_df_so_STAGE_1_stops_scoring_it(capsys):
    """Stage-1 scores BoMetric_df, and an excluded name left there would go on feeding the
    cross-sectional medians every OTHER company is scored against."""
    frame = pd.concat([_panel('ARS', source='BMA'), _clean_panel('AAPL')],
                      ignore_index=True)
    bm = frame[['source', 'date', 'revenue']].copy()
    out = dq.apply_data_quality_filter({'cdx_df': frame, 'BoMetric_df': bm},
                                       verbose=False, save_log=False)
    capsys.readouterr()
    assert set(out['BoMetric_df']['source']) == {'AAPL'}


def test_it_is_IDEMPOTENT_because_the_filter_runs_TWICE_per_run(capsys):
    frame = pd.concat([_panel('ARS', source='BMA'), _clean_panel('AAPL')],
                      ignore_index=True)
    once, rem1 = dq.filter_invalid_data(frame, min_periods_required=4, verbose=False)
    twice, rem2 = dq.filter_invalid_data(once, min_periods_required=4, verbose=False)
    capsys.readouterr()
    assert len(twice) == len(once)
    assert len(rem2) == 0
    assert len(rem1) == 12


def test_the_exclusion_runs_FIRST_so_no_row_is_logged_under_TWO_reasons():
    """THE REAL REASON FOR THE ORDERING, asserted as BEHAVIOUR (reviewer F-3).

    The rationale first written here was PASS 0's, copied: "an ARS row would be the adjacent
    preceding row for the market-cap step check".  That mechanism CANNOT operate -- PASS 1
    keys both baselines per source and 0b removes the source entirely, so no ARS row is ever
    adjacent to another name's row.

    What the ordering actually buys: an excluded source that ALSO trips an arithmetic check
    would otherwise have the SAME rows recorded twice, once as `currency_excluded` and once
    as `data_before_corruption` -- inflating the row count and making the transparency CSV
    give two different answers for why one name left.  This fixture is that case: an ARS
    source carrying a genuine 1000x market-cap break."""
    bad = _panel('ARS', source='BMA')
    bad.loc[6, 'marketCap'] = bad.loc[6, 'marketCap'] * 1000.0
    frame = pd.concat([bad, _clean_panel('AAPL')], ignore_index=True)
    _clean, removed = dq.filter_invalid_data(frame, min_periods_required=4, verbose=False)

    bma = removed[removed['source'] == 'BMA']
    assert len(bma) == 12, 'every row of the excluded source, logged exactly once'
    assert bma['removal_reason'].str.startswith('currency_excluded').all()
    assert not bma['removal_reason'].str.contains('data_before_corruption').any()
    assert not bma.duplicated(subset=['source', 'date']).any(), 'no row logged twice'


def test_a_failure_COMPUTING_the_exclusion_is_LOUD_but_never_fatal(monkeypatch, capsys):
    """RENAMED (reviewer F-12): this patches `__import__`, so it raises INSIDE `fn` -- the
    limb that genuinely is non-fatal.  The old name claimed the whole pass was never fatal,
    which stopped being true when F-6 moved the mutation and the printing outside the
    handler on purpose.  Test the property you actually assert.

    Warn-and-continue on a ~12-hour run, and it must be impossible to read the output as
    currency-filtered."""
    import builtins
    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name == 'currency_exclusions':
            raise ImportError('simulated')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', _boom)
    clean, _rem = dq.filter_invalid_data(_panel('ARS'), min_periods_required=4,
                                         verbose=False)
    out = capsys.readouterr().out
    assert set(clean['source']) == {'BMA'}, 'the run must continue with the rows present'
    assert 'REPORTING-CURRENCY EXCLUSION DID NOT RUN' in out
    assert 'currency-filtered' in out
    assert 'STILL IN THIS PANEL' in out


def test_a_RAISING_PRINT_is_DELIBERATELY_FATAL_and_takes_NO_rows_with_it(monkeypatch):
    """THE FAITHFUL F-6 REPRODUCTION, pinned as the chosen behaviour.

    A print that raises -- a Windows console-codepage failure -- is the real-world trigger,
    because the verbose printing was the ONLY code between the mutation and the `except`.
    Under the OLD shape this produced `clean=8, removed=0`: ten rows gone, no record, run
    continues.  Under the current shape it propagates OUT of the filter and no frame is
    returned at all.

    That is the trade F-6 chose and it is not an oversight: a crash loses the run and
    nothing else, while the old behaviour lost rows and told nobody.  Re-wrapping the
    post-guard region to make this non-fatal again would walk the fix straight back, so
    this test exists to make that regression loud."""
    import builtins
    real_print = builtins.print

    def _print_explodes(*a, **k):
        text = ' '.join(str(x) for x in a)
        if 'REPORTING-CURRENCY EXCLUSION: removed' in text:
            raise UnicodeEncodeError('charmap', 'x', 0, 1, 'simulated codepage failure')
        return real_print(*a, **k)

    monkeypatch.setattr(builtins, 'print', _print_explodes)
    with pytest.raises(UnicodeEncodeError):
        dq.filter_invalid_data(_panel('ARS'), min_periods_required=4, verbose=True)


def test_the_filter_ANNOUNCES_that_it_could_not_run_on_a_currency_less_panel(capsys):
    dq.filter_invalid_data(_panel('ARS').drop(columns=['reportedCurrency']),
                           min_periods_required=4, verbose=True)
    out = capsys.readouterr().out
    assert 'REPORTING-CURRENCY EXCLUSION: NOT APPLIED' in out


# --------------------------------------------------------------------------- #
#  F-4: one stray vendor label must not read like a genuine reporter          #
# --------------------------------------------------------------------------- #
def test_a_ONE_ROW_stray_label_is_DISTINGUISHABLE_in_the_record_from_a_real_reporter():
    """Measured on the 2026-08-07 CUR3K panel: 44 sources carry more than one reporting
    currency and FIVE have exactly ONE minority-currency row (NUAG.TO, QRC.TO, TFII,
    TFII.TO, TRX.TO).  A single stray `ARS` on any of them deletes a 24-quarter North
    American history under an IAS 29 reason.  The removal STAYS -- a name whose currency
    label we cannot trust is not scoreable -- but 1/24 and 24/24 must never produce the
    same record."""
    stray = _panel(['USD'] * 23 + ['ARS'], source='TFII', n=24)
    genuine = _panel('ARS', source='BMA', n=24)
    _c1, r1 = dq.filter_invalid_data(stray, min_periods_required=4, verbose=False)
    _c2, r2 = dq.filter_invalid_data(genuine, min_periods_required=4, verbose=False)
    s1 = r1['removal_reason'].iloc[0]
    s2 = r2['removal_reason'].iloc[0]
    assert 'ARS on 1/24 rows' in s1 and 'MINORITY LABEL' in s1
    assert 'ARS on 24/24 rows' in s2 and 'MINORITY LABEL' not in s2
    assert s1 != s2, 'the two cases must not produce identical evidence'
    assert cx.is_minority_label(1, 24) and not cx.is_minority_label(24, 24)


def test_the_minority_case_is_SHOUTED_on_the_console_too(capsys):
    dq.filter_invalid_data(_panel(['USD'] * 23 + ['ARS'], source='TFII', n=24),
                           min_periods_required=4, verbose=True)
    assert 'MINORITY LABEL' in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  F-5: "could not apply" ships as evidence, not just as a console line       #
# --------------------------------------------------------------------------- #
def test_the_COULD_NOT_APPLY_fact_reaches_an_ARTIFACT(tmp_path, monkeypatch):
    """Three `baseline_tools` callers pass `verbose=False`, so a console-only miss has no
    witness at all on those paths.  'Applied' was evidenced and 'could not apply' was not --
    the wrong asymmetry."""
    monkeypatch.chdir(tmp_path)
    no_ccy = _panel('ARS').drop(columns=['reportedCurrency'])
    dq.apply_data_quality_filter({'cdx_df': no_ccy}, verbose=False, save_log=True)
    #  REPO ROOT since 2026-08-10 (CEO): `output/` did not reach Drive on the 2026-08-10
    #  run while every root-level artifact did, so the evidence CSVs moved.  The assertion
    #  moves WITH the ruling and stays strict about the file existing.
    files = list(tmp_path.glob('CurrencyExclusionStatus_*.csv'))
    assert len(files) == 1
    rows = pd.read_csv(files[0])
    assert list(rows['status']) == ['NOT_APPLIED']
    assert 'reportedCurrency' in rows['note'].iloc[0]
    assert rows['watched_currencies'].iloc[0] == 'ARS'


def test_the_status_file_APPENDS_so_the_IDEMPOTENT_second_pass_cannot_erase_the_first(
        tmp_path, monkeypatch):
    """The filter runs TWICE per pipeline run and pass 2 correctly removes nothing.  An
    overwriting writer would replace pass 1's 'EXCLUDED BMA' with pass 2's 'no match' --
    the accumulate-never-assign defect of 2026-08-07, rebuilt in a new file."""
    monkeypatch.chdir(tmp_path)
    d = {'cdx_df': pd.concat([_panel('ARS', source='BMA'), _clean_panel('AAPL')],
                             ignore_index=True)}
    d = dq.apply_data_quality_filter(d, verbose=False, save_log=True)
    d = dq.apply_data_quality_filter(d, verbose=False, save_log=True)
    rows = pd.read_csv(list(tmp_path.glob('CurrencyExclusionStatus_*.csv'))[0])
    assert list(rows['status']) == ['EXCLUDED', 'APPLIED_NO_MATCH']
    assert rows['source'].iloc[0] == 'BMA'


def test_the_removal_CSV_carries_a_RUN_IDENTIFIER(tmp_path, monkeypatch):
    """F-9: `output/` now receives removal records from the pipeline AND from a standalone
    backtest, and a bare timestamp cannot tell them apart."""
    monkeypatch.chdir(tmp_path)
    d = {'cdx_df': _panel('ARS'), 'universe': 'stock_CUR3K',
         'universe_fingerprint': 'abc123'}
    dq.apply_data_quality_filter(d, verbose=False, save_log=True)
    f = list(tmp_path.glob('removed_data_quality_*.csv'))[0]
    assert pd.read_csv(f)['run_id'].unique().tolist() == ['stock_CUR3K@abc123']
    assert dq.run_identifier({}) == 'unknown-unstamped-run'


# --------------------------------------------------------------------------- #
#  F-6: rows and their record are removed together or not at all              #
# --------------------------------------------------------------------------- #
def test_a_FAILING_pass_removes_NOTHING_and_records_nothing(capsys):
    """The old shape mutated the frame and THEN could raise, clearing the record -- rows
    gone, nothing saying so.  Now `fn` only computes; the drop happens after the record is
    in hand, so a failure leaves the frame untouched."""
    frame = _panel('ARS', source='BMA')

    def _explodes(_df):
        raise RuntimeError('simulated pass failure')

    out, records, sources = dq.guarded_row_pass(frame, _explodes, ['!!! BOOM (%(err)s)'])
    err = capsys.readouterr().out
    assert out is frame and len(out) == len(frame), 'no rows may leave on failure'
    assert records == [] and sources == []
    assert 'BOOM' in err and 'RuntimeError' in err, 'and it must be LOUD'


def test_rows_and_their_RECORD_leave_TOGETHER_or_not_at_all():
    """The invariant, stated directly: whatever the pass does, the number of rows that
    left the frame equals the number of records handed back.

    NOTE this drives the seam through the REAL `cx.exclusion_records`, so it exercises the
    CALLEE.  The seam's own enforcement is tested below with a lying `fn`."""
    frame = pd.concat([_panel('ARS', source='BMA'), _clean_panel('AAPL')],
                      ignore_index=True)

    def _real(df):
        return cx.exclusion_records(df)

    out, records, sources = dq.guarded_row_pass(frame, _real, ['!!! unused'])
    assert len(frame) - len(out) == len(records) > 0
    assert sources == ['BMA']


def test_the_SEAM_REFUSES_a_pass_that_masks_rows_it_did_not_RECORD(capsys):
    """F-11.  The seam used to drop `mask.sum()` rows and return whatever `fn` handed back,
    UNCHECKED -- so an `fn` returning `(one_true_mask, [])` delivered F-6's exact harm
    THROUGH THE SEAM BUILT TO CLOSE IT.  Neither shipped `fn` does that, which is precisely
    why it needs a test that does: this drives the seam, not a callee."""
    frame = _clean_panel('AAPL', n=6)
    liar = pd.Series([True] + [False] * 5, index=frame.index)

    out, records, sources = dq.guarded_row_pass(frame, lambda _d: (liar, []),
                                                ['!!! CONTEXT LINE'])
    err = capsys.readouterr().out
    assert len(out) == len(frame), 'NOTHING may be removed by an unrecorded pass'
    assert records == [] and sources == []
    assert 'DID NOT RECORD WHAT IT REMOVED -- PASS REFUSED' in err
    assert 'rows masked for removal : 1' in err and 'removal records returned: 0' in err
    assert 'CONTEXT LINE' in err, 'the caller banner must still say WHICH pass'


def test_the_SEAM_ALSO_REFUSES_a_record_for_a_removal_that_never_HAPPENED(capsys):
    """The mirror direction.  A record with no masked row is a removal artifact for a name
    that is still in the panel -- it corrupts the reconciliation just as surely, and by
    inflating rather than deflating."""
    frame = _clean_panel('AAPL', n=6)
    none_masked = pd.Series(False, index=frame.index)
    phantom = [{'source': 'AAPL', 'date': None, 'price': 1.0, 'marketCap': 1.0,
                'removal_reason': 'invented'}]

    out, records, sources = dq.guarded_row_pass(frame, lambda _d: (none_masked, phantom),
                                                ['!!! CONTEXT LINE'])
    assert capsys.readouterr().out.count('PASS REFUSED') == 1
    assert len(out) == len(frame) and records == [] and sources == []


def test_a_HONEST_pass_is_untouched_by_the_completeness_check():
    """The check must not cost the normal path anything: one record per masked row passes,
    and a pass that masks nothing and records nothing is still a clean no-op."""
    frame = _clean_panel('AAPL', n=6)
    two = pd.Series([True, True] + [False] * 4, index=frame.index)
    recs = [{'source': 'AAPL', 'removal_reason': 'x'} for _ in range(2)]
    out, records, sources = dq.guarded_row_pass(frame, lambda _d: (two, recs), ['!!! x'])
    assert len(out) == 4 and len(records) == 2 and sources == ['AAPL']


def test_a_pass_that_matches_nothing_is_a_clean_NO_OP():
    frame = _clean_panel('AAPL')
    out, records, sources = dq.guarded_row_pass(frame, lambda d: cx.exclusion_records(d),
                                                ['!!! unused'])
    assert out is frame and records == [] and sources == []


def test_BOTH_row_removing_passes_go_through_the_SAME_guarded_seam():
    """Reviewer asked for the fix in both places rather than duplicated care in one.  The
    quarantine and the currency exclusion now share one atomic applicator, so there is no
    second ordering for a future editor to get wrong."""
    import inspect
    src = inspect.getsource(dq.filter_invalid_data)
    assert src.count('guarded_row_pass(') == 2


# --------------------------------------------------------------------------- #
#  The two rules are NOT redundant                                            #
# --------------------------------------------------------------------------- #
def test_the_FX_ABSTENTION_survives_as_the_BACKSTOP():
    """The exclusion runs inside the quality filter, i.e. on the PIPELINE path.
    baseline_tools/ and every hand-run script reach carveOut's conversions WITHOUT it, and
    there the abstention is the only thing between a correct ARS rate and a
    confidently-measured '$4.9B bank' built on numbers wrong by 1000x.  Deleting either
    rule alone leaves a live hole."""
    assert 'ARS' in fx.ABSTAIN_CURRENCIES
    assert 'ARS' not in co.FX_TO_USD
    assert 'ARS' in cx.EXCLUDED_CURRENCIES


# --------------------------------------------------------------------------- #
#  The consumer the rules were written for                                    #
# --------------------------------------------------------------------------- #
class _StopAfterLoad(Exception):
    """Sentinel: run_all has reached the scenario loop, which is all these tests need."""


def _run_all_capturing(monkeypatch, tmp_path, dmdic_in, **kw):
    """Drive `run_all` and return the dmdic it holds when it reaches `run_scenario`.

    BEHAVIOURAL, not source-inspection (reviewer F-10).  The heavy tail is cut off by a
    sentinel raised from a stubbed `run_scenario`, so the test exercises the real load /
    filter / hand-off path and nothing else."""
    import backtest_unified as bt
    import utils as _utils
    seen = {}

    def _fake_scenario(dmdic, *a, **k):
        seen['dmdic'] = dmdic
        raise _StopAfterLoad()

    monkeypatch.setattr(bt, 'run_scenario', _fake_scenario)
    monkeypatch.setattr(_utils, 'loadWrapper', lambda *a, **k: dmdic_in)
    #  NEVER let the suite read the real key file (repo rule M1): `run_all`'s load path
    #  calls getDataFetchConfiguration purely to default `loadfname`, and that reads
    #  `fmpAPIkey.txt` at import-of-config time.
    monkeypatch.setattr(bt.cf, 'getDataFetchConfiguration', lambda *a, **k: {})
    monkeypatch.chdir(tmp_path)          # any CSV the filter writes lands in tmp
    with pytest.raises(_StopAfterLoad):
        bt.run_all(buy_years=[2021], eval_years_list=[1], verbose=False,
                   save_results=False, **kw)
    return seen['dmdic']


def _backtest_fixture():
    cdx = pd.concat([_panel('ARS', source='BMA', n=24, start='2018-03-31'),
                     _clean_panel('AAPL', n=24)], ignore_index=True)
    cdx['date'] = pd.date_range('2018-01-01', periods=24, freq='QS').tolist() * 2
    return {'cdx_df': cdx, 'BoMetric_df': cdx[['source', 'date', 'revenue']].copy()}


def test_the_BACKTEST_load_path_actually_REMOVES_the_excluded_name(monkeypatch, tmp_path):
    """`vendor_contamination`'s own justification is "Stage-1 is untouched; the BACKTEST IS
    hit, since backtest_unified defaults to --buy_years 2020,2021,2022" -- and until
    2026-08-08 that entry point loaded the saved panel RAW and applied neither rule.  A
    data-side rule that does not reach the consumer it was written for is not a fix.

    Asserted on the FRAME the scenario actually scores, not on where a call sits in the
    source text."""
    got = _run_all_capturing(monkeypatch, tmp_path, _backtest_fixture(),
                             loadfname='irrelevant.pickle')
    assert set(got['cdx_df']['source']) == {'AAPL'}, 'BMA must be gone before scoring'
    assert set(got['BoMetric_df']['source']) == {'AAPL'}


def test_the_backtest_does_NOT_refilter_a_panel_handed_in_by_the_pipeline(monkeypatch,
                                                                          tmp_path):
    """Sbocker filters before calling run_all on both its paths.  Re-filtering in-pipeline
    would be idempotent but would double the console record of every removal, which is how
    a reconciliation figure gets misread.

    Proven by handing in a dmdic that STILL CONTAINS an excludable name: if run_all filtered
    it, BMA would vanish.  It must survive, because filtering an in-pipeline frame is the
    caller's job and already done."""
    handed_in = _backtest_fixture()
    got = _run_all_capturing(monkeypatch, tmp_path, {'unused': True}, dmdic=handed_in)
    assert 'BMA' in set(got['cdx_df']['source']), \
        'the in-pipeline frame must be passed through untouched'


# --------------------------------------------------------------------------- #
#  The universe reconciliation                                                #
# --------------------------------------------------------------------------- #
def _dmdic(cdx, resolved_symbols, failed):
    return {
        'cdx_df': cdx,
        'BoMetric_df': cdx[['source', 'date', 'revenue']].copy(),
        'Tickers_df': pd.DataFrame({'symbol': list(resolved_symbols)}),
        'tickersfailed': list(failed),
    }


def test_an_EXCLUDED_source_is_counted_as_a_universe_EXIT_and_the_identity_BALANCES(capsys):
    import Sbocker as sb
    cdx = pd.concat([_panel('ARS', source='BMA'), _clean_panel('AAPL')],
                    ignore_index=True)
    d = _dmdic(cdx, ['BMA', 'AAPL', 'GONE'], ['GONE'])
    d = dq.apply_data_quality_filter(d, verbose=False, save_log=False)
    assert d['n_dq_removed_sources'] == 1
    assert d['dq_removed_source_list'] == ['BMA']
    residual = sb.print_universe_reconciliation(d, {'tickersfailed': ['GONE']},
                                                verbose=False)
    capsys.readouterr()
    assert residual == 0


def test_a_PARTIAL_removal_is_NOT_a_universe_exit(capsys):
    """Counting a partially-trimmed source as a removal subtracts a name that is still in
    the panel.  On the 2026-08-07 CUR3K panel that is residual -24 across 24 such sources
    (23 PASS-3 `data_before_corruption` trims + 1 quarantine), i.e. the 'UNIVERSE DOES NOT
    RECONCILE' banner firing on a run that reconciles perfectly.  The quarantine is used as
    the fixture here because it is the cleanest single-source case, NOT because it is the
    cause -- see the module docstring."""
    import Sbocker as sb
    import vendor_contamination as vc
    cmg = _clean_panel('058820.KQ', n=18)
    cmg['periodEndDate'] = pd.date_range('2020-06-30', periods=18, freq='QE')
    cmg['date'] = [pd.Timestamp(d.year, ((d.month - 1) // 3) * 3 + 1, 1)
                   for d in cmg['periodEndDate']]
    assert vc.quarantine_mask(cmg).sum() == 10, 'fixture must hit the real rule'

    d = _dmdic(cmg, ['058820.KQ', 'GONE'], ['GONE'])
    d = dq.apply_data_quality_filter(d, verbose=False, save_log=False)
    assert '058820.KQ' in set(d['cdx_df']['source']), 'the name survives the trim'
    assert d['n_dq_removed_sources'] == 0, 'a survivor is not a universe exit'
    assert d['n_dq_partially_removed_sources'] == 1
    assert d['dq_partially_removed_source_list'] == ['058820.KQ']
    residual = sb.print_universe_reconciliation(d, {'tickersfailed': ['GONE']},
                                                verbose=False)
    capsys.readouterr()
    assert residual == 0, 'the old counter made this -1'


def test_the_partial_count_is_REPORTED_not_swallowed(capsys):
    """Not a universe exit, but still a real removal with real evidence -- it gets its own
    counter and its own line rather than disappearing to make the sum work."""
    import Sbocker as sb
    cmg = _clean_panel('058820.KQ', n=18)
    cmg['periodEndDate'] = pd.date_range('2020-06-30', periods=18, freq='QE')
    cmg['date'] = [pd.Timestamp(d.year, ((d.month - 1) // 3) * 3 + 1, 1)
                   for d in cmg['periodEndDate']]
    d = _dmdic(cmg, ['058820.KQ', 'GONE'], ['GONE'])
    d = dq.apply_data_quality_filter(d, verbose=True, save_log=False)
    sb.print_universe_reconciliation(d, {'tickersfailed': ['GONE']}, verbose=True)
    out = capsys.readouterr().out
    assert 'partially trimmed and KEPT' in out
    assert 'PARTIALLY trimmed and still in the panel' in out
