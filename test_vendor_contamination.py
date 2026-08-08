"""Vendor contamination: the 058820.KQ quarantine, and the detector that found it.

OFFLINE.  The detector is panel-internal by design (zero API calls), so nothing here needs
a network layer at all.

THE CASE.  `058820.KQ` (CMG Pharmaceutical, KOSDAQ) carries CHIPOTLE's income statement and
balance sheet for 2020-03-31 -> 2022-09-30, labelled KRW, matching `CMG` to the dollar, then
SNAPPING to genuine KRW at 2022-12-31 (~13x scale break inside one name's history, which
reads as spectacular growth at the boundary).  `marketCap` is unaffected and runs continuous
in real KRW throughout; `cik` is zeroed rather than copied.  It is LIVE AT THE API, so a
re-fetch re-ingests it -- which is why the quarantine is a data-side rule and the detector
is a standing check, not a one-off scrub.

MEASURED THROUGH THIS CODE on the real 2026-08-07 CUR3K panel (61,007 rows, 2,613 sources):
the quarantine removes exactly 10 of 058820.KQ's 24 rows and nothing else; the detector
finds 425 source pairs sharing >= 3 (date, revenue, totalAssets) triples, of which THREE
are name-mismatches -- two are the CMG case (against `CMG` and against `0HXW.L`, Chipotle's
LSE line) and one is `ALTA.PA`/`AREIT.PA`, a legitimate parent/subsidiary that used to
clear only because the legal form `sca` was counted as a shared distinctive token (F-7).
That is the false-positive profile the design is built around: the raw matches are
dominated by legitimate cross-listings, and the name comparison is what separates them --
imperfectly, which is why this is a REPORT and never an action.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import data_quality as dq
import vendor_contamination as vc


#  The real contaminated series, in the panel's own two date conventions.
_CHIPOTLE_REV = [1.364738e9, 1.601414e9, 1.607710e9, 1.741575e9, 1.892538e9,
                 1.952315e9, 1.960633e9, 2.020539e9, 2.213339e9, 2.220175e9]
_CHIPOTLE_TA = [5.370129e9, 5.631640e9, 5.982896e9, 6.149059e9, 6.320454e9,
                6.627567e9, 6.652958e9, 6.467257e9, 6.545336e9, 6.817437e9]
_GENUINE_REV = [2.179433e10, 2.295395e10, 2.543304e10, 2.353718e10, 2.198250e10,
                2.281368e10, 2.485378e10, 2.471393e10]
_GENUINE_TA = [1.693591e11, 1.793435e11, 1.829819e11, 1.806591e11, 1.868518e11,
               1.838859e11, 2.572594e11, 3.047244e11]


def _cmg_panel():
    """058820.KQ as it really arrives: 10 contaminated quarters then 8 genuine ones,
    with BOTH `date` (quarter start, post-setDatesToQuarterly) and `periodEndDate`."""
    ends = pd.date_range('2020-06-30', periods=18, freq='QE')
    rev = _CHIPOTLE_REV + _GENUINE_REV
    ta = _CHIPOTLE_TA + _GENUINE_TA
    return pd.DataFrame({
        'source': ['058820.KQ'] * 18,
        'periodEndDate': ends,
        'date': [pd.Timestamp(d.year, ((d.month - 1) // 3) * 3 + 1, 1) for d in ends],
        'revenue': rev,
        'totalAssets': ta,
        #  marketCap is CONTINUOUS in real KRW across the boundary -- the property that
        #  makes every market-cap sanity check blind to this.
        'marketCap': np.linspace(1.12e11, 6.07e10, 18),
        'weightedAverageShsOut': np.linspace(2.82e7, 2.79e7, 18),
        'price': np.linspace(3900, 2100, 18),
        'reportedCurrency': ['KRW'] * 18,
    })


# --------------------------------------------------------------------------- #
#  The rule itself                                                            #
# --------------------------------------------------------------------------- #
def test_the_rule_is_NAMED_DATED_and_carries_its_EVIDENCE():
    """Not a magic constant buried in a filter: someone reading it in a year has to be
    able to decide whether it still applies without re-deriving the whole case."""
    rules = [r for r in vc.QUARANTINE_RULES if r.source == '058820.KQ']
    assert len(rules) == 1
    r = rules[0]
    assert (str(r.start.date()), str(r.end.date())) == ('2020-03-31', '2022-09-30')
    assert 'Chipotle' in r.reason or 'CMG' in r.reason
    assert r.added == '2026-08-08'
    for must in ('marketCap', 'cik', 'COMPANY NAME', 'backtest'):
        assert must in r.evidence, must
    assert 'vendor_contamination' in r.label() and '058820.KQ' in r.label()


def test_it_quarantines_the_CONTAMINATED_rows_and_ONLY_those():
    panel = _cmg_panel()
    mask = vc.quarantine_mask(panel)
    assert int(mask.sum()) == 10
    #  every removed row carries Chipotle's numbers; every kept row carries the real ones
    assert sorted(panel.loc[mask, 'revenue']) == sorted(_CHIPOTLE_REV)
    assert panel.loc[~mask, 'revenue'].min() > 1e10


def test_the_boundary_quarter_that_STARTS_the_genuine_series_is_KEPT():
    """Off-by-one here would delete a real quarter (invisible loss) or keep a fake one."""
    panel = _cmg_panel()
    mask = vc.quarantine_mask(panel)
    kept = panel[~mask].sort_values('periodEndDate').iloc[0]
    assert str(kept['periodEndDate'].date()) == '2022-12-31'
    assert kept['revenue'] == pytest.approx(2.179433e10)


def test_the_window_matches_under_BOTH_date_conventions():
    """A row has two dates -- `periodEndDate` (vendor, 2022-09-30) and `date` (the same row
    after setDatesToQuarterly, 2022-07-01).  The rule is stated in VENDOR dates, so matching
    `date` naively would be off by a whole quarter.  Both must select the same 10 rows, and
    it must not matter which column a given frame happens to carry."""
    panel = _cmg_panel()
    both = int(vc.quarantine_mask(panel).sum())
    only_date = int(vc.quarantine_mask(panel.drop(columns=['periodEndDate'])).sum())
    only_ped = int(vc.quarantine_mask(panel.drop(columns=['date'])).sum())
    assert both == only_date == only_ped == 10


def test_a_DIFFERENT_source_in_the_same_window_is_untouched():
    other = _cmg_panel().assign(source='000660.KS')
    assert not vc.quarantine_mask(other).any()


def test_an_unusable_frame_never_raises():
    """It sits on the critical path of a ~12-hour run."""
    for frame in (None, pd.DataFrame(), pd.DataFrame({'x': [1]})):
        assert not vc.quarantine_mask(frame).any()


# --------------------------------------------------------------------------- #
#  Wired into data_quality -- the part that makes it survive a re-fetch       #
# --------------------------------------------------------------------------- #
def test_the_filter_REMOVES_the_rows_and_LOGS_them_with_a_reason(capsys):
    clean, removed = dq.filter_invalid_data(_cmg_panel(), min_periods_required=4,
                                            verbose=False)
    capsys.readouterr()
    assert len(clean) == 8
    assert set(clean['source']) == {'058820.KQ'}
    q = removed[removed['removal_reason'].str.contains('vendor_contamination')]
    assert len(q) == 10
    assert q['removal_reason'].str.contains('058820.KQ').all()
    assert q['removal_reason'].str.contains('Chipotle').all()


def test_the_removal_reaches_the_TRANSPARENCY_frame_not_just_the_panel():
    """Removing rows while reporting nothing removed is this project's signature defect
    (the accumulate-never-assign bug of 2026-08-07).  The quarantine must not reintroduce
    it: the rows have to be in `removed_df`, because that frame drives the CSV, the
    BoMetric_df propagation and the universe-reconciliation counters."""
    _clean, removed = dq.filter_invalid_data(_cmg_panel(), min_periods_required=4,
                                             verbose=False)
    assert {'source', 'date', 'removal_reason'} <= set(removed.columns)
    assert len(removed) >= 10


def test_it_propagates_to_BoMetric_df_by_source_and_date(capsys):
    """Stage-1 scores BoMetric_df.  A quarantine that cleaned only cdx_df would leave the
    contaminated quarters scoring -- the exact half-fix audit H-1 found."""
    panel = _cmg_panel()
    bm = panel[['source', 'date', 'revenue']].copy()
    dmdic = {'cdx_df': panel, 'BoMetric_df': bm}
    out = dq.apply_data_quality_filter(dmdic, verbose=False, save_log=False)
    capsys.readouterr()
    assert len(out['BoMetric_df']) == 8
    assert out['BoMetric_df']['revenue'].min() > 1e10


def test_it_is_IDEMPOTENT_because_the_filter_runs_TWICE_per_run(capsys):
    panel = _cmg_panel()
    once, rem1 = dq.filter_invalid_data(panel, min_periods_required=4, verbose=False)
    twice, rem2 = dq.filter_invalid_data(once, min_periods_required=4, verbose=False)
    capsys.readouterr()
    assert len(twice) == len(once) == 8
    assert len(rem2) == 0


def test_the_quarantine_runs_BEFORE_the_arithmetic_checks():
    """Order matters: a contaminated row is the ADJACENT PRECEDING ROW for the market-cap
    step check, i.e. corrupt data acting as the baseline that judges real data."""
    import inspect
    src = inspect.getsource(dq.filter_invalid_data)
    assert src.index('PASS 0') < src.index('PASS 1')
    assert src.index('vendor_contamination') < src.index('check_price_sanity')


def test_a_broken_quarantine_is_LOUD_but_never_fatal(monkeypatch, capsys):
    """Warn-and-continue, like every other guard on this path -- but it must be impossible
    to read the output as quarantined."""
    import builtins
    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name == 'vendor_contamination':
            raise ImportError('simulated')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', _boom)
    clean, _rem = dq.filter_invalid_data(_cmg_panel(), min_periods_required=4,
                                         verbose=False)
    out = capsys.readouterr().out
    assert len(clean) == 18, 'the run must continue with the rows still present'
    assert 'QUARANTINE DID NOT RUN' in out
    assert 'DO NOT treat this output as quarantined' in out


def test_the_existing_checks_could_NOT_have_caught_it():
    """Stated as an executable fact so the rule cannot later be deleted as redundant.  The
    13x break is an order of magnitude under _MCAP_BREAK_RATIO, and it is in the wrong
    field anyway -- marketCap runs continuous straight through the window."""
    panel = _cmg_panel().sort_values('date').reset_index(drop=True)
    prev = None
    for _i, row in panel.iterrows():
        ok, why = dq.check_price_sanity(
            row, prev_row_mcap=(None if prev is None else prev['marketCap']),
            prev_row_shares=(None if prev is None else prev['weightedAverageShsOut']))
        assert ok, (row['periodEndDate'], why)
        prev = row
    caps = panel['marketCap']
    assert (caps.max() / caps.min()) < dq._MCAP_BREAK_RATIO


# --------------------------------------------------------------------------- #
#  The detector                                                               #
# --------------------------------------------------------------------------- #
def _two_issuer_panel(n_shared=3, name_b='Chipotle Mexican Grill, Inc.'):
    """One clean issuer plus a second carrying the SAME (date, revenue, totalAssets)
    triples for `n_shared` quarters."""
    dates = pd.date_range('2020-03-31', periods=8, freq='QE')
    a = pd.DataFrame({'source': ['058820.KQ'] * 8, 'date': dates,
                      'revenue': _CHIPOTLE_REV[:8], 'totalAssets': _CHIPOTLE_TA[:8]})
    b = a.copy()
    b['source'] = 'CMG'
    #  only the first n_shared quarters actually coincide
    b.loc[b.index[n_shared:], 'revenue'] = np.arange(1, 9 - n_shared) * 1e7
    b.loc[b.index[n_shared:], 'totalAssets'] = np.arange(1, 9 - n_shared) * 2e7
    names = {'058820.KQ': 'CMG Pharmaceutical Co., Ltd.', 'CMG': name_b}
    return pd.concat([a, b], ignore_index=True), names


def test_it_finds_the_founding_case_and_calls_it_a_NAME_MISMATCH():
    panel, names = _two_issuer_panel(n_shared=8)
    got = vc.detect_shared_fundamentals(panel, names=names)
    assert len(got) == 1
    row = got.iloc[0]
    assert {row['source_a'], row['source_b']} == {'058820.KQ', 'CMG'}
    assert row['n_shared'] == 8
    assert row['verdict'] == 'NAME_MISMATCH'
    assert bool(row['name_match']) is False


def test_a_LEGITIMATE_cross_listing_is_matched_and_NOT_flagged():
    """The discriminator, and the reason the detector is a report rather than a filter:
    cross-listings share fundamentals BY CONSTRUCTION and dominate the raw matches."""
    panel, names = _two_issuer_panel(n_shared=8,
                                     name_b='CMG Pharmaceutical Co., Ltd. (KDR)')
    got = vc.detect_shared_fundamentals(panel, names=names)
    assert got.iloc[0]['verdict'] == 'name_match'
    assert 'pharmaceutical' in got.iloc[0]['shared_tokens']


@pytest.mark.parametrize('n,found', [(1, False), (2, False), (3, True), (8, True)])
def test_the_threshold_is_THREE_shared_triples(n, found):
    """Three independent multi-digit numbers agreeing on three independent dates is not a
    coincidence; one or two can be."""
    panel, names = _two_issuer_panel(n_shared=n)
    assert (len(vc.detect_shared_fundamentals(panel, names=names)) > 0) is found


def test_generic_legal_forms_do_NOT_count_as_a_name_match():
    """'Corteva, Inc.' and 'EIDP, Inc.' must not match on 'inc' -- the comparison is on the
    DISTINCTIVE part of the name or it discriminates nothing."""
    assert vc.normalise_name('Chipotle Mexican Grill, Inc.') == {'chipotle', 'mexican',
                                                                'grill'}
    assert vc.normalise_name('CMG Pharmaceutical Co., Ltd.') == {'cmg', 'pharmaceutical'}
    assert not (vc.normalise_name('Corteva, Inc.') & vc.normalise_name('EIDP, Inc.'))


def test_SCA_is_a_legal_form_and_cannot_clear_a_pair():
    """F-7 (reviewer): `sca` was missing from the generic set and was the SOLE reason
    ALTA.PA/AREIT.PA cleared.  A legal form vouching for a match is the failure; that the
    pair is genuinely related is a matter for the human reading the report."""
    assert 'sca' in vc._GENERIC_NAME_TOKENS
    assert vc.normalise_name('Altarea SCA') == {'altarea'}
    assert not (vc.normalise_name('Altarea SCA') & vc.normalise_name('Altareit SCA'))


def test_ACCENTS_are_FOLDED_not_split_on():
    """F-7 (reviewer): the tokeniser split on `[^0-9a-z]`, i.e. on every non-ASCII letter,
    so an accented "Societe Generale" shattered into {'soci','rale'} -- and the fragment
    'soci' then appeared in 23 different names on the panel, any one of which could clear a
    pair as a 'shared distinctive token'.  On a European universe that makes the name
    check, the ONLY thing separating contamination from cross-listings, silently
    unreliable exactly where the universe is densest."""
    accented = 'Société Générale'
    assert vc.normalise_name(accented) == {'societe', 'generale'}
    assert vc.normalise_name(accented) == vc.normalise_name('Societe Generale')
    assert 'soci' not in vc.normalise_name(accented)
    assert vc.normalise_name('L’Oréal S.A.') == {'oreal'}
    #  and an accented name still matches its unaccented twin, which is the point
    assert vc.normalise_name('Nestlé S.A.') & vc.normalise_name('Nestle SA')


def test_an_UNREADABLE_name_is_reported_as_UNKNOWN_not_as_a_mismatch():
    """A name we cannot parse is not evidence of contamination.  Real case on the CUR3K
    panel: `0RJ6.L` / `LOUP.PA`, both 'L.d.c. S.a.', which normalises to nothing."""
    panel, names = _two_issuer_panel(n_shared=8, name_b='L.d.c. S.a.')
    names['058820.KQ'] = 'L.d.c. S.a.'
    got = vc.detect_shared_fundamentals(panel, names=names)
    assert got.iloc[0]['verdict'] == 'name_unknown'
    assert got.iloc[0]['name_match'] is None


def test_NaN_and_all_zero_rows_cannot_manufacture_a_match():
    """Shell filings zero-zero on both fields would otherwise match every other shell."""
    dates = pd.date_range('2020-03-31', periods=6, freq='QE')
    panel = pd.concat([
        pd.DataFrame({'source': ['A'] * 6, 'date': dates, 'revenue': [0.0] * 6,
                      'totalAssets': [0.0] * 6}),
        pd.DataFrame({'source': ['B'] * 6, 'date': dates, 'revenue': [0.0] * 6,
                      'totalAssets': [0.0] * 6}),
        pd.DataFrame({'source': ['C'] * 6, 'date': dates, 'revenue': [np.nan] * 6,
                      'totalAssets': [np.nan] * 6}),
        pd.DataFrame({'source': ['D'] * 6, 'date': dates, 'revenue': [np.nan] * 6,
                      'totalAssets': [np.nan] * 6}),
    ], ignore_index=True)
    assert len(vc.detect_shared_fundamentals(panel)) == 0


def test_a_degenerate_triple_shared_by_MANY_sources_is_dropped():
    """A value shared by 50 issuers is a placeholder, not an identity -- and pairing it up
    is quadratic for no signal."""
    dates = pd.date_range('2020-03-31', periods=4, freq='QE')
    frames = [pd.DataFrame({'source': ['S%02d' % i] * 4, 'date': dates,
                            'revenue': [1e6] * 4, 'totalAssets': [2e6] * 4})
              for i in range(vc.MAX_SOURCES_PER_TRIPLE + 5)]
    assert len(vc.detect_shared_fundamentals(pd.concat(frames, ignore_index=True))) == 0


def test_the_detector_makes_ZERO_api_calls():
    """Stated as an executable fact: it is panel-internal, which is why it can run on
    every fetch."""
    import inspect
    src = inspect.getsource(vc)
    for banned in ('requests', 'safe_json_list', 'safe_http_get', 'urlopen', 'apikey'):
        assert banned not in src, banned


def test_the_stage_writes_its_evidence_EVEN_WHEN_CLEAN(tmp_path):
    """"the check ran and found nothing" and "the check did not run" are different facts,
    and only one of them is fine."""
    dates = pd.date_range('2020-03-31', periods=6, freq='QE')
    clean = pd.DataFrame({'source': ['A'] * 6, 'date': dates,
                          'revenue': np.arange(1, 7) * 1e6,
                          'totalAssets': np.arange(1, 7) * 3e6})
    vc.run_detector_stage(clean, names={'A': 'Alpha Inc.'}, run_date='2026-08-08',
                          outdir=str(tmp_path), verbose=False)
    path = tmp_path / 'VendorContaminationFlags_2026-08-08.csv'
    assert path.exists()
    assert len(pd.read_csv(path)) == 0


def test_the_stage_shouts_when_it_finds_a_mismatch(tmp_path, capsys):
    panel, names = _two_issuer_panel(n_shared=8)
    vc.run_detector_stage(panel, names=names, run_date='2026-08-08',
                          outdir=str(tmp_path), verbose=True)
    out = capsys.readouterr().out
    assert 'VENDOR CONTAMINATION SUSPECTED' in out
    assert '058820.KQ' in out
    #  and it must say plainly that it did NOT act
    assert 'REPORTED, NOT removed' in out
    assert 'QUARANTINE_RULES' in out


def test_the_stage_never_raises_on_a_junk_frame(tmp_path):
    for frame in (None, pd.DataFrame(), pd.DataFrame({'x': [1]})):
        assert len(vc.run_detector_stage(frame, outdir=str(tmp_path), verbose=False)) == 0


def test_the_REJECTED_scale_break_detector_is_recorded_as_rejected():
    """It fires on 119 of 2,613 sources, is dominated by de-SPACs/IPOs/first-full-balance-
    sheet events, and ranks 058820.KQ only 24th.  Recorded in the module so it is not
    rebuilt by the next person who notices the 13x jump."""
    doc = vc.__doc__
    assert 'REJECTED' in doc and 'scale-break' in doc
    assert 'de-SPAC' in doc and '24th' in doc


# --------------------------------------------------------------------------- #
#  Wiring                                                                     #
# --------------------------------------------------------------------------- #
def test_Sbocker_runs_the_detector_on_the_RAW_panel_before_the_quality_filter():
    """A detector that only ever sees post-filter data cannot report what the filter
    already removed -- the population most worth knowing about."""
    import inspect
    import Sbocker
    src = inspect.getsource(Sbocker.main)
    assert 'run_detector_stage(' in src
    i_det = src.index('run_detector_stage(')
    i_dq = src.index('dq.apply_data_quality_filter(')
    assert i_det < i_dq
