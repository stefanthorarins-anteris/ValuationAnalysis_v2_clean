"""Dated, expiring exclusions -- the safety properties, not the happy path.

3,692 names were being excluded in production by a file nobody could read, so the tests that
matter here are the ones that say WHAT CANNOT HAPPEN: the legacy file cannot be loaded, an
entry with no expiry cannot become permanent by accident, an expired entry cannot be applied,
and the change cannot alter a run on its own.
"""
import os
from datetime import date, timedelta

import pytest

import exclusions as x


TODAY = date(2026, 8, 14)


def _write(tmp_path, lines):
    p = tmp_path / 'ExclusionList_test.csv'
    p.write_text('\n'.join(lines) + '\n')
    return str(p)


# --------------------------------------------------------------------------- #
#  (a) THE LEGACY 3,692-NAME FILE IS UNREACHABLE                               #
# --------------------------------------------------------------------------- #
def test_legacy_bare_ticker_row_is_refused_whole(tmp_path):
    """A headerless row of tickers -- the pre-2026-08 format -- applies ZERO names.

    This is the single most important property in the module.  The old loader took
    `csv.reader(f)` and used `templist[0]` as the whole list, which is precisely how a
    February-2023 file was still removing 3,692 names in January 2026.
    """
    p = _write(tmp_path, ['BSD.PA,ELCO.L,BKT,BSP.DE,ALWEC.PA,DNP,JGH'])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == [], (
        'a bare ticker row was ACCEPTED and applied %d name(s). The legacy format carries no '
        'date, no reason and no expiry, so it cannot be evaluated and must apply nothing.'
        % len(v.applied))
    assert len(v.by_status('malformed')) == 1
    assert 'FILE REFUSED WHOLE' in v.by_status('malformed')[0].note


def test_the_real_legacy_file_in_the_repo_applies_nothing():
    """Not a synthetic: the actual 2023 file still sitting in the repo root."""
    p = 'ManualEliminationTickersList_fmp_2023-02-14.csv'
    if not os.path.exists(p):
        pytest.skip('%s not present in this tree' % p)
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == [], (
        '%s applied %d names. It is the file that produced `manelim3692`; under the dated '
        'schema it must apply none.' % (p, len(v.applied)))


# --------------------------------------------------------------------------- #
#  (b) EXPIRY IS REAL, IN BOTH DIRECTIONS                                      #
# --------------------------------------------------------------------------- #
def test_expired_entry_is_not_applied_and_is_reported(tmp_path):
    p = _write(tmp_path, [
        ','.join(x.EXCLUSION_HEADER),
        'AAA,transient_fetch,fetch failed,2026-07-01,2026-07-15,run-observed',
        'BBB,transient_fetch,fetch failed,2026-08-10,2026-08-24,run-observed',
    ])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == ['BBB']
    assert [e.ticker for e in v.by_status('expired')] == ['AAA']
    assert 'AAA' in v.report(verbose=False) and 'EXPIRED' in v.report(verbose=False)


def test_short_history_name_ages_off_by_itself(tmp_path):
    """THE CASE THE OLD RULE COULD NOT EXPRESS.

    `newmanelimtckrs = manualelim + (tickersfailed - lenfail)` SUBTRACTED the
    history-length failures precisely because, with no expiry, listing them would ban a
    two-year-old company forever.  With an expiry the name can be listed AND age off.
    """
    added = TODAY - timedelta(days=100)
    exp = x.default_expiry('short_history', added)
    assert exp == added + timedelta(days=90)
    p = _write(tmp_path, [
        ','.join(x.EXCLUSION_HEADER),
        'NEWCO,short_history,failed the history-length gate,%s,%s,run-observed'
        % (added.strftime(x.DATE_FMT), exp.strftime(x.DATE_FMT)),
    ])
    assert x.load_exclusions(p, as_of=TODAY, verbose=False).applied == []
    assert x.load_exclusions(p, as_of=added + timedelta(days=30),
                             verbose=False).applied == ['NEWCO']


def test_blank_expiry_is_refused_except_for_duplicate(tmp_path):
    """"Forever" must be REACHED, never DEFAULTED INTO by leaving a field empty."""
    p = _write(tmp_path, [
        ','.join(x.EXCLUSION_HEADER),
        'DUPE,duplicate,second listing of one issuer,2026-08-14,,DedupSurvivorReport',
        'SNEAK,vendor_bad,bad data,2026-08-14,,someone left it blank',
    ])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == ['DUPE']
    bad = v.by_status('malformed')
    assert [e.ticker for e in bad] == ['SNEAK']
    assert 'blank expiry' in bad[0].note


def test_transient_fetch_has_the_shortest_life_and_duplicate_the_longest():
    """The expiry table is an argument about the conditions, so assert its ordering."""
    assert x.CATEGORIES['duplicate'][0] is None
    assert (x.CATEGORIES['transient_fetch'][0]
            < x.CATEGORIES['short_history'][0]
            < x.CATEGORIES['vendor_bad'][0]
            < x.CATEGORIES['ceo'][0])


# --------------------------------------------------------------------------- #
#  (c) A MALFORMED OR REASONLESS ENTRY IS REFUSED, NOT APPLIED                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('row,why', [
    ('AAA,not_a_category,x,2026-08-14,2026-12-01,e', 'unknown category'),
    ('AAA,ceo,,2026-08-14,2026-12-01,e', 'no reason'),
    ('AAA,ceo,x,not-a-date,2026-12-01,e', 'unparseable added'),
    ('AAA,ceo,x,2026-08-14,nonsense,e', 'unparseable expires'),
    (',ceo,x,2026-08-14,2026-12-01,e', 'no ticker'),
])
def test_malformed_entries_are_refused(tmp_path, row, why):
    p = _write(tmp_path, [','.join(x.EXCLUSION_HEADER), row])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == [], 'a %s entry was applied; refusing is the safe direction' % why
    assert len(v.by_status('malformed')) == 1


def test_missing_file_is_empty_not_an_error():
    v = x.load_exclusions('no_such_exclusion_file_at_all.csv', as_of=TODAY, verbose=False)
    assert v.applied == [] and v.path is None


# --------------------------------------------------------------------------- #
#  RECONCILIATION AND THE WRITER/LOADER SCHEMA                                 #
# --------------------------------------------------------------------------- #
def test_reconcile_raises_when_an_exclusion_has_no_live_entry(tmp_path):
    p = _write(tmp_path, [','.join(x.EXCLUSION_HEADER),
                          'AAA,ceo,because,2026-08-14,2027-08-14,e'])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert x.reconcile(v.applied, v) is True
    with pytest.raises(AssertionError):
        x.reconcile(v.applied + ['GHOST'], v)


def test_writer_and_loader_share_one_schema(tmp_path):
    """They did not before: the writer emitted a headerless ticker row and the loader read
    `templist[0]`, so nothing carried a date, a reason or an expiry."""
    entries = x.propose_from_run(['AAA', 'BBB', 'CCC'], ['CCC'])
    p = str(tmp_path / 'out.csv')
    x.write_exclusions(p, entries)
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == ['AAA', 'BBB', 'CCC']
    cats = {e.ticker: e.category for e in v.by_status('live')}
    assert cats == {'AAA': 'transient_fetch', 'BBB': 'transient_fetch',
                    'CCC': 'short_history'}, (
        'the `- lenfail` subtraction is gone: a history-length failure must be RECORDED as '
        '`short_history` with an expiry, not omitted from the list entirely.')


def test_an_expired_entry_does_not_shadow_a_re_observation(tmp_path):
    """THE EXPIRY WAS ONE-SHOT (review S2, 2026-08-14).

    `have` was built over ALL entries including expired ones, so once a machine-proposed entry
    aged off, the same failure re-observed on every later run was silently dropped and the name
    could never be listed again.  That contradicts this module's headline argument -- that a
    short-history name "can go ON the list and AGE OFF" -- because it aged off and could not
    come back, however many quarters it kept failing.
    """
    expired = x.Entry('ACME', 'transient_fetch', 'fetch failed on this run', '2026-07-15',
                      '2026-07-29', 'run-observed', 'expired', 'expired 2026-07-29')
    merged = x.merge_entries([expired], x.propose_from_run(['ACME'], []))
    live = [e for e in merged if e.status == 'live']
    assert [e.ticker for e in live] == ['ACME'], (
        'the re-observation was dropped: %s' % [(e.ticker, e.status) for e in merged])
    #  and the superseded machine-authored row is collapsed rather than accumulating forever
    assert len(merged) == 1
    #  round-trip: the refreshed entry is LIVE on the next load
    p = str(tmp_path / 'rt.csv')
    x.write_exclusions(p, merged)
    assert x.load_exclusions(p, as_of=TODAY, verbose=False).applied == ['ACME']


def test_an_expired_HAND_EDITED_entry_is_kept_as_evidence_when_superseded():
    """A human's record of what they did and when is not the machine's to tidy away."""
    human = x.Entry('ACME', 'transient_fetch', 'CEO looked at this', '2026-01-01',
                    '2026-07-29', 'by hand', 'expired', '')
    merged = x.merge_entries([human], x.propose_from_run(['ACME'], []))
    assert len(merged) == 2
    assert sum(1 for e in merged if e.status == 'live') == 1
    assert any(e.evidence == 'by hand' for e in merged)


def test_hand_edited_entry_is_not_clobbered_by_a_machine_re_observation():
    """The CEO edits this file by hand; a re-observation must not extend his expiry."""
    human = x.Entry('AAA', 'transient_fetch', 'CEO looked at this', '2026-01-01',
                    '2026-12-31', 'by hand', 'live', '')
    merged = x.merge_entries([human], x.propose_from_run(['AAA'], []))
    assert len(merged) == 1 and merged[0].expires == '2026-12-31'
    assert merged[0].reason == 'CEO looked at this'


# --------------------------------------------------------------------------- #
#  THE CHANGE ALONE CANNOT ALTER A RUN                                         #
# --------------------------------------------------------------------------- #
def test_default_configuration_excludes_nothing():
    """Three independent reasons, each asserted, because one of them failing silently is
    exactly how the 3,692-name list stayed live for three years."""
    import configuration as cf
    assert not os.path.exists(x.DEFAULT_EXCLUSION_FILE), (
        '%r exists in this tree, so the default is no longer guaranteed empty.'
        % x.DEFAULT_EXCLUSION_FILE)
    #  1. the default filename does not exist -> empty verdict
    assert x.load_exclusions(x.DEFAULT_EXCLUSION_FILE, verbose=False).applied == []
    #  2. `-manelimtickers` still defaults to OFF
    ns = cf.build_parser().parse_args([]) if hasattr(cf, 'build_parser') else None
    if ns is not None:
        assert getattr(ns, 'manelimtickers', None) in (None, False)
    #  3. even a populated list is only reachable via BOTH the flag and a schema file
    assert x.DEFAULT_EXCLUSION_FILE != 'ManualEliminationTickersList_fmp_2023-02-14.csv'

# --------------------------------------------------------------------------- #
#  ROTATION -- the retest slice (CEO, 2026-08-24)                              #
# --------------------------------------------------------------------------- #
def _rotation_list(tmp_path, n=300, category='transient_fetch', added=None, expires=None):
    added = added or (TODAY - timedelta(days=1))
    expires = expires or (TODAY + timedelta(days=400))
    rows = ['ticker,category,reason,added,expires,evidence']
    for i in range(n):
        rows.append('T%04d,%s,fetch failed on this run,%s,%s,run-observed'
                    % (i, category, added.strftime(x.DATE_FMT), expires.strftime(x.DATE_FMT)))
    return _write(tmp_path, rows)


def test_EVERY_listed_name_is_retested_within_RETEST_SLOTS_runs(tmp_path):
    """*** THE PROMISE ROTATION MAKES, AND THE ONLY ONE THAT MATTERS. ***  An armed list with
    no rotation is a permanent ban; the whole argument for arming it is that every name comes
    back round within a BOUNDED number of runs.  So walk every cycle and assert the union of
    the held-out slices covers the list -- not that each slice is roughly 10%, which is a
    property of the hash and not a promise to anybody."""
    path = _rotation_list(tmp_path, n=300)
    seen = set()
    for cycle in range(x.RETEST_SLOTS):
        v = x.load_exclusions(path, as_of=TODAY, verbose=False, cycle=cycle)
        seen |= set(v.held_out)
    live = {e.ticker for e in x.load_exclusions(path, as_of=TODAY, verbose=False,
                                                rotate=False).entries if e.status == 'live'}
    assert live, 'fixture produced no live entries'
    assert live - seen == set(), (
        '%d name(s) were never held out in %d cycles, so an armed list bans them forever: %s'
        % (len(live - seen), x.RETEST_SLOTS, sorted(live - seen)[:10]))


def test_a_held_out_name_is_NOT_applied_so_the_run_actually_refetches_it(tmp_path):
    """The hold-out has to reach `applied`, or rotation is a log line that changes nothing.
    `applied` is the ONLY thing the pipeline filters on (see the ExclusionVerdict docstring),
    so this is the assertion that connects the mechanism to the behaviour."""
    path = _rotation_list(tmp_path, n=300)
    v = x.load_exclusions(path, as_of=TODAY, verbose=False, cycle=0)
    assert v.held_out, 'cycle 0 held nobody out -- the fixture cannot exercise this'
    for tk in v.held_out:
        assert tk not in v.applied, (
            '%r was held out for re-test and STILL applied -- the name is excluded, so the '
            'run never re-fetches it and the hold-out is cosmetic' % tk)


def test_the_slot_is_STABLE_across_processes_and_across_list_growth():
    """Two properties, one test, because both are what make "within 10 runs" true of a list
    that changes between runs:
      * `hash()` is salted per process (PYTHONHASHSEED), so a slot built on it would differ
        between two runs of the SAME list -- a name could then be skipped indefinitely;
      * a positional slice (`sorted(...)[i::10]`) reshuffles every later name's slot whenever
        a name is inserted, so the bound would reset on every list change."""
    known = {t: x.retest_slot(t) for t in ('AAPL', '007700.KS', 'SHEL.L', '0QQF.L')}
    assert known == {t: x.retest_slot(t) for t in known}, 'retest_slot is not deterministic'
    #  A hash of the ticker cannot depend on what else is in the list -- there is no list
    #  argument -- so growth-stability is structural.  Pin it anyway, because the obvious
    #  "improvement" is to pass the list in.
    import inspect
    assert 'entries' not in inspect.signature(x.retest_slot).parameters, (
        'retest_slot now sees the list, so a name inserted anywhere can move another name to '
        'a different slot and the within-N-runs bound stops holding')


def test_the_cycle_counter_ADVANCES_through_the_written_file(tmp_path):
    """Rotation only walks the slots if the counter moves.  It rides in a `#` comment, which
    the loader already skips, so this also pins that a counter cannot make a valid file
    malformed (safety lock (a) checks EXCLUSION_HEADER on the first non-comment line)."""
    path = _rotation_list(tmp_path, n=50)
    seen_cycles = []
    for _ in range(x.RETEST_SLOTS + 2):
        v = x.load_exclusions(path, as_of=TODAY, verbose=False)
        assert not v.by_status('malformed'), (
            'the cycle comment made the file malformed: %s'
            % [e.note for e in v.by_status('malformed')][:2])
        seen_cycles.append(v.cycle)
        x.write_exclusions(path, v.entries, cycle=v.cycle)
    assert seen_cycles[:x.RETEST_SLOTS] == list(range(x.RETEST_SLOTS)), seen_cycles
    assert seen_cycles[x.RETEST_SLOTS] == 0, 'the counter did not wrap'


def test_the_3_MONTH_BLANKET_holds_out_a_name_rotation_would_not(tmp_path):
    """The CEO's backstop.  A machine entry live for more than BLANKET_RETEST_DAYS is
    re-fetched whatever the slot arithmetic says -- so a slot that never comes up, a counter
    that stops advancing, or a hand-rewritten file cannot produce a permanent ban."""
    old = TODAY - timedelta(days=x.BLANKET_RETEST_DAYS + 1)
    path = _rotation_list(tmp_path, n=300, added=old)
    #  Pick a name whose slot is NOT this cycle, so rotation is provably not what catches it.
    v0 = x.load_exclusions(path, as_of=TODAY, verbose=False, cycle=0)
    victims = [t for t in v0.held_out if v0.held_out[t] == 'blanket']
    assert victims, 'no name was caught by the blanket -- the fixture is not exercising it'
    for t in victims:
        assert x.retest_slot(t) != 0, 'this name was in the rotation slice anyway'
        assert t not in v0.applied


def test_the_blanket_does_NOT_reach_a_HAND_ADDED_permanent_entry(tmp_path):
    """A `duplicate` line is a structural claim with no clock BY DESIGN, and the blanket must
    not quietly convert the CEO's "forever" into 90 days.  Only `run-observed` entries are
    machine claims about a fetch, so only those are re-probed."""
    old = (TODAY - timedelta(days=x.BLANKET_RETEST_DAYS + 30)).strftime(x.DATE_FMT)
    path = _write(tmp_path, [
        'ticker,category,reason,added,expires,evidence',
        'BRK-B,duplicate,second listing of Berkshire Hathaway,%s,,DedupSurvivorReport.csv' % old,
    ])
    #  Force the cycle to the slot this ticker is NOT in, so rotation cannot be the cause.
    cycle = (x.retest_slot('BRK-B') + 1) % x.RETEST_SLOTS
    v = x.load_exclusions(path, as_of=TODAY, verbose=False, cycle=cycle)
    assert v.held_out == {}, (
        'the 90-day blanket re-probed a hand-added permanent entry (%s) -- it must only reach '
        'run-observed machine claims' % v.held_out)
    assert v.applied == ['BRK-B']


def test_short_history_is_NOT_spent_on_a_rotation_slot(tmp_path):
    """`short_history` recovers by AGING, deterministically -- 328 of the 710 rows on the
    2026-08-22 list.  A rotation slot spent on a name that provably cannot have recovered yet
    is a wasted fetch, so this cohort is scheduled, not sampled.  Its category expiry (90d)
    is what re-probes it."""
    path = _rotation_list(tmp_path, n=300, category='short_history')
    for cycle in range(x.RETEST_SLOTS):
        v = x.load_exclusions(path, as_of=TODAY, verbose=False, cycle=cycle)
        assert all(r != 'rotation' for r in v.held_out.values()), (
            'a short_history name was held out by ROTATION on cycle %d: %s'
            % (cycle, {k: r for k, r in v.held_out.items() if r == 'rotation'}))


def test_the_computed_short_history_retry_date_is_LATER_than_the_flat_expiry():
    """The CEO's refinement (a): `16 - n` more quarters is COMPUTABLE, so a name holding 4
    periods should not be re-probed as if it needed one.  The date may only push a retry OUT,
    never pull it in -- pulling it in would spend calls earlier than the flat policy does."""
    added = TODAY
    flat = x.default_expiry('short_history', added)
    near = x.scheduled_retry_date(added, x.HISTORY_GATE_PERIODS - 1)   # needs 1 more quarter
    far = x.scheduled_retry_date(added, 4)                             # needs 12 more
    assert near is not None and far is not None
    assert far > flat, (far, flat)
    assert far > near
    #  And the fallback is honest: with no period count there is no computed date at all.
    assert x.scheduled_retry_date(added, None) is None


def test_propose_from_run_uses_the_computed_date_ONLY_when_the_count_is_supplied():
    """Production does not supply the period count today (`getData_fmp` appends the bare
    ticker to `lenfail`), so this pins BOTH arms: with the map the expiry is computed, and
    without it the flat 90-day category expiry is used and nothing silently pretends
    otherwise."""
    added = TODAY
    flat = x.propose_from_run(['AAA'], ['AAA'], added=added)[0]
    assert flat.expires == x.default_expiry('short_history', added).strftime(x.DATE_FMT)
    comp = x.propose_from_run(['AAA'], ['AAA'], added=added, lenfail_periods={'AAA': 4})[0]
    assert comp.expires == x.scheduled_retry_date(added, 4).strftime(x.DATE_FMT)
    assert comp.expires > flat.expires


def test_a_name_can_only_be_listed_AFTER_it_has_actually_been_fetched():
    """*** THE GUARD THE BRIEF ASKED FOR BY NAME: the list must not be able to skip a name it
    has never tried. ***  It is STRUCTURAL, not a runtime check: `propose_from_run` is the only
    machine author of entries and it can only emit names that are in `tickersfailed`, i.e.
    names this run fetched and this run saw fail.

    WHAT THIS CANNOT DETECT: a HAND-ADDED entry.  A `ceo` or `duplicate` line is a human
    judgement about a name the human names, and nothing here asks whether the machine ever
    tried it -- by design, but it does mean "every applied name was fetched at least once" is
    true of the machine-authored half only."""
    proposed = x.propose_from_run(['AAA', 'BBB'], ['BBB'], added=TODAY)
    assert {e.ticker for e in proposed} == {'AAA', 'BBB'}
    assert all(e.evidence == 'run-observed' for e in proposed)
    #  A name that did not fail cannot be proposed, however it is passed in.
    assert x.propose_from_run([], ['CCC'], added=TODAY) == []


def test_ROTATION_IS_ON_BY_DEFAULT(tmp_path):
    """The safe default, not the convenient one.  `rotate=False` produces a verdict whose
    `cycle` is None, and `report()` says so in as many words -- because an armed list read
    without rotation is a permanent ban and must never be the thing you get by forgetting an
    argument."""
    path = _rotation_list(tmp_path, n=300)
    assert x.load_exclusions(path, as_of=TODAY, verbose=False).cycle is not None
    v = x.load_exclusions(path, as_of=TODAY, verbose=False, rotate=False)
    assert v.cycle is None and v.held_out == {}
    assert 'ROTATION: NOT APPLIED' in v.report(verbose=False)


def test_the_applied_count_EXCLUDES_the_held_out_slice(tmp_path):
    """A held-out name is re-fetched, so counting it as an exclusion would overstate the
    filter by the size of the slice -- in the report the CEO reads, every run."""
    path = _rotation_list(tmp_path, n=300)
    v = x.load_exclusions(path, as_of=TODAY, verbose=False, cycle=0)
    live = sum(1 for e in v.entries if e.status == 'live')
    assert sum(v.counts_by_category().values()) == len(v.applied) == live - len(v.held_out)

def test_the_rotation_banner_does_NOT_promise_a_RUN_bound_for_the_SCHEDULED_cohort(tmp_path):
    """*** THE BANNER LIED TO THE OPERATOR, AND IT IS THE ONLY THING THEY SEE. ***  It read
    "Every listed name is retried within 10 runs" -- false for `short_history`, which
    rotation deliberately skips and which is 328 of the 710 rows on the 2026-08-22 list, 46%
    of the very list the banner describes.  A promise the mechanism does not make is exactly
    how an armed list becomes a permanent ban nobody audits."""
    rows = ['ticker,category,reason,added,expires,evidence']
    added = (TODAY - timedelta(days=1)).strftime(x.DATE_FMT)
    for i in range(20):
        rows.append('R%03d,transient_fetch,fetch failed on this run,%s,%s,run-observed'
                    % (i, added, (TODAY + timedelta(days=10)).strftime(x.DATE_FMT)))
    for i in range(30):
        rows.append('S%03d,short_history,failed the history-length gate,%s,%s,run-observed'
                    % (i, added, (TODAY + timedelta(days=80)).strftime(x.DATE_FMT)))
    path = _write(tmp_path, rows)
    text = x.load_exclusions(path, as_of=TODAY, verbose=False).report(verbose=False)
    assert 'Every listed name is retried within' not in text, (
        'the banner is back to promising a uniform run bound it does not deliver')
    assert '20 ROTATED name(s): retried within %d runs' % x.RETEST_SLOTS in text
    assert '30 SCHEDULED name(s)' in text and 'NOT within %d runs' % x.RETEST_SLOTS in text


def test_select_retest_has_NO_unreachable_scheduled_arm(tmp_path):
    """The deleted branch fired on `expires <= as_of`, which `_classify` already calls
    `'expired'` and which this loop already skips -- unreachable on every input, and MORE
    unreachable once `lenfail_periods` pushes expiries out.  Asserted over a fixture that
    spans the whole space: expired, live-and-near, live-and-far, rotated and scheduled."""
    rows = ['ticker,category,reason,added,expires,evidence']
    for i, (cat, days) in enumerate([('short_history', -5), ('short_history', 1),
                                     ('short_history', 1200), ('transient_fetch', -5),
                                     ('transient_fetch', 3), ('transient_fetch', 900)] * 20):
        rows.append('T%03d,%s,r,%s,%s,run-observed'
                    % (i, cat, (TODAY - timedelta(days=1)).strftime(x.DATE_FMT),
                       (TODAY + timedelta(days=days)).strftime(x.DATE_FMT)))
    path = _write(tmp_path, rows)
    seen = set()
    for cycle in range(x.RETEST_SLOTS):
        seen |= set(x.load_exclusions(path, as_of=TODAY, verbose=False,
                                      cycle=cycle).held_out.values())
    assert seen <= {'rotation', 'blanket'}, (
        'select_retest produced a hold-out reason outside {rotation, blanket}: %s' % seen)
    assert 'scheduled' not in seen


def test_a_SCHEDULED_name_with_a_MULTI_YEAR_expiry_is_still_reached_by_the_blanket(tmp_path):
    """The consequence of deleting the arm, pinned: a `short_history` name whose COMPUTED
    retry date is ~3 years out is skipped by rotation and is not yet expired -- so the
    3-month blanket is the only thing that re-probes it, and it must."""
    old = (TODAY - timedelta(days=x.BLANKET_RETEST_DAYS + 1)).strftime(x.DATE_FMT)
    far = (TODAY + timedelta(days=1100)).strftime(x.DATE_FMT)
    path = _write(tmp_path, [
        'ticker,category,reason,added,expires,evidence',
        'AAA,short_history,4 of 16 periods,%s,%s,run-observed' % (old, far)])
    v = x.load_exclusions(path, as_of=TODAY, verbose=False,
                          cycle=(x.retest_slot('AAA') + 1) % x.RETEST_SLOTS)
    assert v.held_out == {'AAA': 'blanket'}, v.held_out
    assert v.applied == []
