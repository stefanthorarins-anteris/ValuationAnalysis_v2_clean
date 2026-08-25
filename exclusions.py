"""exclusions.py -- DATED, EXPIRING exclusions.  One module, so the whole rule is
auditable in one place (the same reason `nan_policy` is one module).

THE CEO'S DESIGN, in his words: dupes and names that structurally cannot meet the data
requirements belong on such a list, and *"of course, that list should be ignored and
rewritten periodically."*  Every entry therefore carries a CATEGORY, a REASON, the date it
was ADDED and the date it EXPIRES.  Dupes and known-bad vendor data persist; short-history
names age off automatically as they qualify.

WHAT THIS REPLACES, AND WHY IT COULD NOT SIMPLY BE RE-WIRED
-----------------------------------------------------------
`ManualEliminationTickersList_<ds>_<date>.csv` was ONE CSV ROW OF BARE TICKERS -- no date,
no reason, no expiry.  It was written every run by `utils.writeManElimToFile` from

    newmanelimtckrs = manualelimtickers + (tickersfailed - lenfail)

and, since the 2026-07-19 provenance fix, read back by NOTHING: `configuration` loads it
only under `-manelimtickers 1` and `Sbocker` then blanked it outright
(`manualelim_applied = []`).  Production ran `manelim3692` on 2025-12-09 / 2026-01-08 and
`manelim0` on every 2026-08 run.

TWO PROPERTIES OF THE OLD ACCUMULATION RULE ARE **NOT** CARRIED FORWARD, deliberately:

  1. IT SUBTRACTED `lenfail`.  The names that failed the history-length gate -- exactly the
     cohort the CEO named -- were excluded FROM the list on purpose, because a two-year-old
     company qualifies in two years and must not be banned forever.  That reasoning is
     sound and it is what a DATED EXPIRY reconciles: a short-history name can now go ON the
     list and AGE OFF, which neither the old design nor a naive re-wire achieves.
  2. IT BANKED `tickersfailed` WITH NO EXPIRY AND NO REASON.  A name whose fetch failed once
     because the vendor was slow was banned permanently, indistinguishably from a confirmed
     duplicate.  Transient failures are now their own category with the SHORTEST expiry in
     the table.

SAFETY -- THE PART THAT MATTERS MOST.  3,692 names were being excluded in production, so
re-enabling loading against a wrong list would silently remove thousands of names from the
universe.  Three properties, each testable:

  (a) THE LEGACY FILE CANNOT BE LOADED.  `ManualEliminationTickersList_fmp_2023-02-14.csv`
      is a bare ticker row with no header.  `load_exclusions` requires the schema header
      and REFUSES the whole file otherwise, reporting it as malformed and applying ZERO
      names.  The 3,692-name list is therefore unreachable by construction, not by
      convention.
  (b) EMPTY AND OFF BY DEFAULT, AND THE LOOP STILL SHIPS **INERT**.  A missing file is an
      empty list, not an error; `configuration` still defaults `-manelimtickers` to 0; AND the
      run WRITES `ExclusionList_<ds>_<date>.csv` while the loader READS
      `DEFAULT_EXCLUSION_FILE` = `ExclusionList_fmp.csv`, so what a run accumulates is never
      what the next run applies.  Arming it is a deliberate human act: review the dated file,
      copy it to `ExclusionList_fmp.csv`, pass `-manelimtickers 1`.  Stated plainly because an
      earlier draft of this docstring described the accumulation as a working loop -- it is
      not, and calling shipped-inert machinery "working" is the same class of error as calling
      a 2023 list "current".  *** THE ROTATION MACHINERY ADDED 2026-08-24 DOES NOT ARM IT
      EITHER.  Rotation is what makes arming SAFE; the two locks above are untouched, so
      nothing in that change can alter a run on its own. ***
  (d) AN ARMED LIST IS NEVER A PERMANENT BAN, because `load_exclusions` rotates by default.
      Every run HOLDS OUT a slice and re-fetches it.  THE BOUND IS NOT THE SAME FOR EVERY
      NAME, and saying otherwise was the banner's own defect until 2026-08-24:
        * categories IN the rotation -- `transient_fetch`, `duplicate`, `vendor_bad`, `ceo`
          (382 of the 710 rows on 2026-08-22) -- are re-fetched within `RETEST_SLOTS` RUNS;
        * `SCHEDULED_CATEGORIES` (`short_history`, 328 rows) is deliberately NOT rotated,
          because it recovers by AGING and a slot spent on it buys an answer already known.
          It is bounded on a DAY clock instead: `BLANKET_RETEST_DAYS`, or its own expiry.
      Both bounds are real and both are printed by `verdict.report()` every run.  See the
      ROTATION block below.

WHAT AN ARMED RUN DOES DIFFERENTLY, IN ONE PLACE
------------------------------------------------
Read this before arming; it is the whole behavioural delta.
  * `get_tickers` receives `manualelimtickers = verdict.applied` and never fetches those
    names.  On the 2026-08-22 list that is ~669 of 710 names -- roughly 3,300 API calls a run
    NOT made.  The evidence that this is safe: 407 of CUR3K's 411 fails re-failed on CUR6K in
    the SAME category and all 407 had also failed on 08-19; only 4 recovered, all `.KQ`.
  * `verdict.held_out` -- this run's slice -- IS fetched, exactly as if it were not listed.
    ~38 names a run on that list, ~190 calls, ~5 minutes.  If a held-out name fails again it
    is re-observed and re-listed; if it now succeeds it is simply not re-proposed.  A recovery
    therefore needs no threshold, no audit and no decision -- it needs one fetch.
  * The universe shrinks by `len(verdict.applied)`, and `Sbocker.print_universe_reconciliation`
    has to account for exactly that many names.
  * NOTHING IS EXCLUDED THAT WAS NEVER FETCHED.  A machine entry is only ever created by
    `propose_from_run` from the run's own `tickersfailed`, and the write happens after the
    fetch while the read happens before it -- so a name cannot be excluded on the run that
    first observed it.
  * The steady state on today's list is NOT "excluded forever": `transient_fetch` expires at
    14 days, so the whole 382-name cohort is re-probed once a fortnight regardless of
    rotation, and `short_history` ages off at 90 days.  Rotation's marginal value is therefore
    largest on the LONG-LIVED categories (`duplicate` never, `vendor_bad` 180d, `ceo` 365d),
    which today's 100%-machine-authored list contains none of.  Said here because the saving
    is easy to over-read: arming saves the calls, rotation makes the long-lived categories
    safe, and they are not the same thing.
  (c) IT RECONCILES AND IT SHOUTS.  `verdict.report()` prints the applied count broken down
      by category, every expired entry it IGNORED, and every malformed line -- and
      `reconcile` asserts that every applied ticker is accounted for by exactly one LIVE
      entry.  An exclusion that no live entry explains is a raise, not a silent drop.

THE FORMAT IS A CSV WITH A HEADER, one entry per line:

    ticker,category,reason,added,expires,evidence
    058820.KQ,vendor_bad,FMP serves Chipotle's statements,2026-08-14,2027-02-10,vendor_contamination
    BRK-B,duplicate,second listing of Berkshire Hathaway,2026-08-14,,DedupSurvivorReport_2026-08-13.csv

Human-editable (the CEO adds and removes names by hand; a bad entry is ONE VISIBLE LINE and
is removed by deleting it) and machine-readable.  `expires` blank means NEVER, and is
permitted ONLY for the categories whose default expiry is None -- a blank expiry on any
other category is a malformed entry and is refused, so "forever" can never be reached by
leaving a field empty.
"""

import csv
import hashlib
import os
from collections import namedtuple
from datetime import datetime, timedelta

DATE_FMT = '%Y-%m-%d'

EXCLUSION_HEADER = ['ticker', 'category', 'reason', 'added', 'expires', 'evidence']

#  The default `-manelimfilename`.  It DOES NOT EXIST in a fresh tree, and a missing file is
#  an EMPTY verdict -- so the default can only ever exclude zero names.  The old default
#  named the 2023 bare-ticker file.
DEFAULT_EXCLUSION_FILE = 'ExclusionList_fmp.csv'

#  CATEGORY -> (default lifetime in days, justification).
#  THE LIFETIME IS ARGUED FROM WHAT THE CONDITION IS, not chosen for roundness.  The
#  question each answers is "how long until this fact could plausibly have changed?".
CATEGORIES = {
    #  A second listing of the same issuer is a STRUCTURAL fact about the security, not a
    #  fact about the data: it does not become false by waiting.  So no clock -- the only
    #  way off this category is a human deleting the line, which is the correct review
    #  mechanism for a permanent claim.  `dedup_issuer` owns the automatic case; this is
    #  the manual override for what it cannot see.
    'duplicate': (None, 'structural: a second listing of one issuer does not expire'),
    #  Vendor data known bad -- e.g. `058820.KQ` served Chipotle's statements for nine
    #  quarters.  This is a SERVER-SIDE fact that FMP can repair at any time without telling
    #  us, and the name underneath is usually a real company.  Six months forces a re-probe
    #  instead of a permanent ban on a real issuer; the CMG case was itself re-probed on
    #  2026-08-08 and was still live, which is exactly the check this expiry schedules.
    'vendor_bad': (180, 'server-side and repairable by the vendor: re-probe every 2 quarters'),
    #  The cohort the OLD RULE SUBTRACTED.  A short-history name gains one period per
    #  reporting quarter, so the earliest date on which the answer can differ is one quarter
    #  away.  90 days re-tests it exactly when re-testing can change the outcome, and costs
    #  one retry per quarter rather than a permanent ban (the old design) or a retry every
    #  run (the current blank-list behaviour).
    'short_history': (90, 'gains one period per quarter: re-test on the quarterly cadence'),
    #  A once-off fetch failure -- throttling, a timeout, a ragged payload.  It carries
    #  almost no information about the name, so it earns the SHORTEST life in the table:
    #  long enough that one bad night does not re-cost the same failures on an immediate
    #  re-run, short enough that a slow vendor evening cannot ban a name for a quarter.
    #  THE OLD WRITER BANKED THIS CLASS WITH NO EXPIRY AT ALL.
    #  *** THE NAME IS A MISNOMER FOR 100% OF THE TESTABLE POPULATION, AND THE 14 DAYS ARE
    #  DELIBERATELY LEFT ALONE (2026-08-24).  407 of CUR3K's 411 fails re-failed on CUR6K in
    #  the same category, all 407 having also failed on 08-19: these are PERSISTENT, not
    #  transient.  Lengthening the life is what would turn arming into the ~1,900-call-a-run
    #  saving the CEO is after, and rotation is what would make a longer life safe -- but that
    #  is a policy change to the number of names fetched per run and it is the CEO's to make,
    #  not something to slip in beside a mechanism.  RAISED, NOT TAKEN. ***
    'transient_fetch': (14, 'says almost nothing about the name: expire fast'),
    #  The CEO's own hand-added names.  A year, not forever, so a judgement made once is
    #  re-made deliberately rather than inherited silently -- which is how the 2023 list
    #  came to be applied in 2026.
    'ceo': (365, 'a human judgement should be re-made, not inherited'),
}

#  Categories whose entries may carry a BLANK `expires`.  Derived from CATEGORIES so the two
#  cannot drift apart.
NEVER_EXPIRES = frozenset(k for k, (d, _) in CATEGORIES.items() if d is None)


#  ###################################################################################
#  ##                 ROTATION -- RETEST A SLICE EVERY RUN (CEO, 2026-08-24)          ##
#  ###################################################################################
#  THE PROBLEM ARMING SOLVES, AND THE PROBLEM ARMING CREATES.  The list is computed every
#  run (710 rows on 2026-08-22) and thrown away, because the writer emits
#  `ExclusionList_fmp_<date>.csv` and the loader reads `ExclusionList_fmp.csv`.  Arming it
#  saves ~1,900 wasted calls a run -- 407 of CUR3K's 411 fails re-failed on CUR6K in the same
#  category and all 407 had also failed on 08-19; only 4 recovered, all `.KQ`.  But an armed
#  list that never re-probes is a permanent ban dressed as a cache, and 0.97% of these names
#  DO recover.
#
#  WHY ROTATION AND NOT THE 1%-RANDOM-AUDIT.  The audit (sample 1%, refetch everything if any
#  sampled name is valid) fires 6.6% of the time at the observed 0.97% recovery rate -- about
#  one run in fifteen, each costing a 3,550-call / ~48-minute refetch -- and still misses the
#  individual recoveries it exists to catch, because 7 names cannot resolve a 1% effect.
#  Rotation retests a FIXED 10% SLICE every run: ~71 names x 5 calls = ~355 calls (~5 min),
#  every name retried within a BOUNDED 10 runs, and no threshold logic anywhere.
#
#  THE SLICE IS DETERMINISTIC, NOT RANDOM.  `retest_slot` hashes the ticker into one of
#  `RETEST_SLOTS` buckets and the run holds out the bucket matching this run's cycle counter.
#  A hash (not `sorted()[i::10]`) so the slice does not reshuffle when the list grows: adding
#  a name moves only that name, and a name's slot never changes.  MD5 (not `hash()`) because
#  Python's string hash is salted per process -- `PYTHONHASHSEED` would make the slice differ
#  between two runs of the same list, which is exactly the property this must not have.
#
#  THE CYCLE COUNTER LIVES IN THE FILE, AS A COMMENT.  `load_exclusions` already skips lines
#  starting with `#`, so the counter rides in the artifact it belongs to without changing
#  `EXCLUSION_HEADER` (a schema change would make every existing file malformed, which is the
#  one thing safety lock (a) must keep working).  A file with no counter starts at 0.
RETEST_SLOTS = 10
CYCLE_COMMENT_PREFIX = '# retest_cycle:'

#  THE CEO'S 3-MONTH BLANKET BACKSTOP.  No MACHINE-AUTHORED entry may be applied for more
#  than this long without the name being re-fetched, whatever the rotation logic thinks.  It
#  costs nothing (the name is re-fetched, not deleted) and it catches anything the slot
#  arithmetic gets wrong -- a slot that never comes up, a cycle counter that stops advancing,
#  a list rewritten by hand.
#
#  *** IT IS APPLIED TO `run-observed` ENTRIES ONLY.  A hand-added `duplicate` line is a
#  permanent structural claim with no clock by design, and a blanket that re-probed it would
#  quietly convert the CEO's "forever" into "90 days" -- the mirror of the defect this module
#  was built to remove. ***
BLANKET_RETEST_DAYS = 90

#  Categories that recover DETERMINISTICALLY BY AGING and therefore do not need a random slot.
#  `short_history` is 328 of the 710 rows on 2026-08-22 (46%): a name that failed the
#  `< HISTORY_GATE_PERIODS` gate with `n` periods needs `HISTORY_GATE_PERIODS - n` more, which
#  is COMPUTABLE -- there is no point spending a rotation slot on a name that provably cannot
#  have recovered yet.  Excluding them from rotation leaves rotation to cover the ~380
#  genuinely-persistent names.
SCHEDULED_CATEGORIES = frozenset(['short_history'])
HISTORY_GATE_PERIODS = 16
#  Days per reporting period, for the computed short-history retry date.  365.25/4.
DAYS_PER_PERIOD = 91.3125


def retest_slot(ticker, slots=RETEST_SLOTS):
    """Which rotation slice `ticker` belongs to: a stable integer in [0, slots).

    STABLE ACROSS PROCESSES AND ACROSS LIST GROWTH.  MD5 of the ticker, not `hash()` (salted
    per process by PYTHONHASHSEED) and not a position in a sorted list (every insertion would
    reshuffle every later name's slot, so "every name within 10 runs" would stop being true
    the moment the list changed).
    """
    h = hashlib.md5(str(ticker).encode('utf-8')).hexdigest()
    return int(h[:8], 16) % int(slots)


def scheduled_retry_date(added, n_periods, gate=HISTORY_GATE_PERIODS):
    """When a `short_history` name can FIRST have enough history, or None if unknowable.

    A name that failed the gate holding `n_periods` of the `gate` required needs
    `gate - n_periods` more reporting periods, and no re-fetch before then can change the
    answer.  That is arithmetic, not a guess, which is why this cohort gets a date rather than
    a slot.

    *** `n_periods` IS NOT AVAILABLE FROM THE PIPELINE TODAY, AND THAT IS RECORDED RATHER
    THAN PAPERED OVER.  `getData_fmp.getFsData_fmp` appends the TICKER to `lenfail` and
    discards the count it had in hand at the moment it made the verdict, so
    `propose_from_run` receives a bare list and this function is reached with `n_periods =
    None` on every production run today.  Wiring it is a one-line change in the FETCH path
    (`lenfail.append(ticker)` -> also record the period count), which was deliberately not
    made on the eve of a run.  Until it is, `short_history` keeps its 90-day category expiry,
    which is the same quarterly cadence by a cruder route. ***

    THE BLANKET CAPS THIS ANYWAY, and the two CEO refinements genuinely conflict here: a name
    holding 4 periods needs 12 more (~3 years), and `BLANKET_RETEST_DAYS` re-probes it every
    90 days regardless.  So the computed date can only ever RAISE the cadence from "every 10
    runs" to "every quarter" -- it cannot push a retry past the blanket.  That is the honest
    resolution and it is cheap (5 calls per name per quarter); it is not the full saving the
    computed date could give on its own.

    WHAT THIS DATE ACTUALLY CHANGES, stated precisely because the first version of this note
    implied it fed a hold-out arm in `select_retest` and it does not (that arm was dead code
    and is gone).  It changes ONE thing: the `expires` stamped on the entry, hence WHEN the
    entry ages off and the name stops being excluded.  Later expiry = fewer pointless
    re-fetches of a name that provably cannot have qualified yet.  The BLANKET still holds
    the name out at 90 days either way, so this can never delay a probe -- only remove one.
    """
    if n_periods is None:
        return None
    try:
        need = int(gate) - int(n_periods)
    except (TypeError, ValueError):
        return None
    if need <= 0:
        #  It already qualifies -- the failure was not really the length gate.  No date.
        return None
    return added + timedelta(days=int(round(need * DAYS_PER_PERIOD)))


def read_cycle(path):
    """This run's rotation cycle, read from the file's `# retest_cycle:` comment.

    Returns 0 for a missing file, a file with no counter, or an unparseable one -- a
    rotation that starts at slot 0 is always correct, just not continuous with the last run.
    """
    try:
        if not path or not os.path.exists(path):
            return 0
        with open(path, 'r', newline='') as f:
            for line in f:
                t = line.strip()
                if t.startswith(CYCLE_COMMENT_PREFIX):
                    return int(t[len(CYCLE_COMMENT_PREFIX):].strip()) % RETEST_SLOTS
    except (ValueError, OSError):
        return 0
    return 0


def select_retest(entries, cycle, as_of, slots=RETEST_SLOTS,
                  blanket_days=BLANKET_RETEST_DAYS):
    """{ticker: reason} for every LIVE entry this run must HOLD OUT and re-fetch.

    Held out, not deleted: the entry stays in the file with its dates and its reason.  If the
    name fails again this run, `propose_from_run` re-observes it and `merge_entries` refreshes
    it; if it now succeeds, it is simply not re-proposed and ages off normally.  So a recovery
    needs no threshold, no audit and no decision -- it needs one fetch.

    TWO reasons, in the order they are checked (a third, `scheduled`, was written and was
    unreachable -- see the note where it used to be):
      `rotation`  slot == cycle, for a category NOT in SCHEDULED_CATEGORIES.  ~1/`slots` of
                  the eligible entries, every run.
      `blanket`   a `run-observed` entry live for more than `blanket_days`.  The CEO's
                  3-month backstop, and the ONLY hold-out a SCHEDULED_CATEGORIES entry can
                  get -- rotation skips that cohort by design.

    *** SO THE COVERAGE PROMISE IS NOT UNIFORM, AND `report()` NOW SAYS SO. ***  A rotated
    name is retried within `slots` RUNS; a scheduled name is retried on a DAY clock -- at
    `blanket_days`, or when its own expiry ages the entry off, whichever comes first.  On the
    2026-08-22 list that is 382 names on the run clock and 328 on the day clock.

    *** IT CANNOT SKIP A NAME IT HAS NEVER TRIED, AND THAT IS STRUCTURAL RATHER THAN
    CHECKED.  A machine entry is only ever created by `propose_from_run` from the run's own
    `tickersfailed`, i.e. from names that WERE fetched and DID fail on that run; and the write
    happens after the fetch while the read happens before it, so a name cannot be excluded on
    the same run it was first observed.  A hand-added entry is a human judgement about a name
    the human names, and is not a claim that the machine tried it.  `verdict.report()` prints
    the applied/held-out split by category so the two populations stay visible. ***
    """
    out = {}
    for e in entries:
        if e.status != 'live':
            continue
        machine = (e.evidence == 'run-observed')
        if e.category not in SCHEDULED_CATEGORIES and retest_slot(e.ticker, slots) == int(cycle) % int(slots):
            out[e.ticker] = 'rotation'
            continue
        if machine and blanket_days:
            try:
                added = _parse_date(e.added)
            except ValueError:
                added = None
            if added is not None and (as_of - added).days > int(blanket_days):
                out[e.ticker] = 'blanket'
                continue
        #  *** THERE IS NO THIRD `'scheduled'` ARM, AND THE ONE THAT WAS HERE WAS DEAD CODE
        #  (reviewer, 2026-08-24). ***  It fired on `expires <= as_of` -- but `_classify`
        #  marks exactly that condition `'expired'`, and this loop skips every entry that is
        #  not `'live'`, so the branch was unreachable on every input.  The comment defending
        #  it ("it is here so the mechanism exists the moment the fetch path records the
        #  period count") was FALSE IN THE OPPOSITE DIRECTION: supplying `lenfail_periods`
        #  pushes `expires` LATER (~3 years for a 4-period name), which keeps the entry live
        #  for longer and makes the arm deader still.
        #  NOTHING IS LOST BY DELETING IT.  A `short_history` name is re-probed by two
        #  mechanisms that do work: its own EXPIRY ages the entry off (that IS the scheduled
        #  retest -- the name simply stops being excluded), and until then the BLANKET arm
        #  above holds it out at `BLANKET_RETEST_DAYS` whatever its expiry says.  What
        #  `SCHEDULED_CATEGORIES` still does is keep it out of the ROTATION slice, which is
        #  the only thing it was ever needed for.
    return out


Entry = namedtuple('Entry', EXCLUSION_HEADER + ['status', 'note'])


class ExclusionVerdict(object):
    """What a list said, what was applied, and what was ignored -- all three, always.

    `applied`  : tickers whose entry is LIVE as of the run date.  This is the ONLY thing the
                 pipeline may filter on.
    `entries`  : every parsed entry, each stamped `status` in {live, expired, malformed}.
    `path`     : the file it came from, or None when no file was read.
    """

    def __init__(self, entries, path=None, as_of=None, held_out=None, cycle=None):
        self.entries = list(entries)
        self.path = path
        self.as_of = as_of
        #  {ticker: reason} for names a LIVE entry covers that this run RE-FETCHES anyway --
        #  the rotation slice, the 3-month blanket and the computed short-history schedule.
        #  See `select_retest`.  Empty dict means "rotation ran and held nobody out"; None is
        #  not used, so a reader never has to distinguish it from "rotation did not run".
        self.held_out = dict(held_out or {})
        self.cycle = cycle
        _live = {e.ticker for e in self.entries if e.status == 'live'}
        self.applied = sorted(_live - set(self.held_out))
        #  THE IDENTITY THAT MAKES THE HOLD-OUT AUDITABLE: every live name is either applied
        #  or held out, never neither and never both.  A name in neither set would be silently
        #  dropped from the universe with nothing recording it -- the exact invisible failure
        #  this module exists to prevent, reached by a new route.
        assert set(self.applied) | set(self.held_out) == _live, (
            'exclusions: %d live name(s) are in neither the applied nor the held-out set'
            % len(_live - set(self.applied) - set(self.held_out)))

    def by_status(self, status):
        return [e for e in self.entries if e.status == status]

    def counts_by_category(self):
        #  Counts what was APPLIED, not what is live -- a held-out name is re-fetched, so
        #  counting it as an exclusion would overstate the filter by the size of the slice.
        applied = set(self.applied)
        out = {}
        for e in self.by_status('live'):
            if e.ticker in applied:
                out[e.category] = out.get(e.category, 0) + 1
        return out

    def report(self, verbose=True):
        """Say LOUDLY how many names were excluded and why.  Returns the printed text.

        An exclusion list is the one component whose failure mode is INVISIBLE -- names it
        removes simply are not there to be missed -- so silence is not an option even when
        it applied nothing.
        """
        lines = []
        if self.path is None:
            lines.append('EXCLUSIONS: no list configured -- 0 name(s) excluded.')
        else:
            lines.append('EXCLUSIONS: %r as of %s -- APPLIED %d name(s), IGNORED %d expired, '
                         'REFUSED %d malformed.'
                         % (self.path, self.as_of.strftime(DATE_FMT),
                            len(self.applied), len(self.by_status('expired')),
                            len(self.by_status('malformed'))))
            for cat in sorted(self.counts_by_category()):
                lines.append('    %-16s %5d applied   (%s)'
                             % (cat, self.counts_by_category()[cat], CATEGORIES[cat][1]))
            #  THE HOLD-OUT IS THE PART A READER MUST NOT HAVE TO INFER.  It is the only
            #  thing standing between an armed list and a permanent ban, so it says how many
            #  names this run RE-FETCHES despite holding a live entry for them, and why.
            if self.cycle is not None:
                #  *** THE BANNER USED TO CLAIM "Every listed name is retried within N runs",
                #  AND THAT IS FALSE FOR 46% OF THE LIST (reviewer, 2026-08-24). ***  Rotation
                #  DELIBERATELY skips SCHEDULED_CATEGORIES -- 328 of the 710 rows on the
                #  2026-08-22 list are `short_history` -- so the run-bounded promise holds for
                #  the rotated cohort ONLY.  The scheduled cohort is covered on a DAY clock
                #  (the blanket, and its own expiry), not a RUN clock, and an operator reading
                #  a promise the mechanism does not make is exactly how an armed list becomes
                #  a permanent ban nobody audits.  Both cohorts are now counted, by name.
                _sched_live = sum(1 for e in self.entries
                                  if e.status == 'live' and e.category in SCHEDULED_CATEGORIES)
                _rot_live = sum(1 for e in self.entries
                                if e.status == 'live'
                                and e.category not in SCHEDULED_CATEGORIES)
                lines.append('    ROTATION: cycle %d of %d -- %d name(s) HELD OUT and '
                             're-fetched this run (%s).'
                             % (self.cycle, RETEST_SLOTS, len(self.held_out),
                                ', '.join('%s=%d' % (r, sum(1 for v in self.held_out.values()
                                                            if v == r))
                                          for r in ('rotation', 'blanket')
                                          if any(v == r for v in self.held_out.values()))
                                or 'none'))
                lines.append('        %d ROTATED name(s): retried within %d runs.'
                             % (_rot_live, RETEST_SLOTS))
                lines.append('        %d SCHEDULED name(s) (%s): NOT in the rotation slice -- '
                             'retried on a DAY clock, at the %d-day blanket or at their own '
                             'expiry, whichever is sooner. NOT within %d runs.'
                             % (_sched_live, ', '.join(sorted(SCHEDULED_CATEGORIES)),
                                BLANKET_RETEST_DAYS, RETEST_SLOTS))
                for tk in sorted(self.held_out)[:40]:
                    lines.append('        HELD OUT, RE-FETCHED: %-14s (%s)'
                                 % (tk, self.held_out[tk]))
                if len(self.held_out) > 40:
                    lines.append('        ... and %d more' % (len(self.held_out) - 40))
            else:
                lines.append('    ROTATION: NOT APPLIED -- every live entry was applied '
                             'without a re-test slice. An armed list with no rotation is a '
                             'PERMANENT ban on every name in it.')
            for e in self.by_status('expired'):
                lines.append('    EXPIRED, NOT APPLIED: %-14s %-16s expired %s (%s)'
                             % (e.ticker, e.category, e.expires, e.reason))
            for e in self.by_status('malformed'):
                lines.append('    MALFORMED, REFUSED : %-14s %s' % (e.ticker or '<blank>', e.note))
        text = '\n'.join(lines)
        if verbose:
            print(text, flush=True)
        return text


def _parse_date(s):
    s = (s or '').strip()
    if not s:
        return None
    return datetime.strptime(s, DATE_FMT).date()


def default_expiry(category, added):
    """`added` + the category's lifetime, or None where the category never expires."""
    days = CATEGORIES[category][0]
    if days is None:
        return None
    return added + timedelta(days=int(days))


def _classify(row, as_of):
    """One raw dict -> an `Entry` with its status decided.  Never raises on bad input.

    A malformed entry is REFUSED (never applied) and REPORTED -- it is not silently dropped
    and it is emphatically not applied on the benefit of the doubt.  An exclusion is a
    deletion from the universe, so the safe direction when the entry cannot be read is to
    keep the name.
    """
    tk = (row.get('ticker') or '').strip()
    cat = (row.get('category') or '').strip()
    reason = (row.get('reason') or '').strip()
    added_s = (row.get('added') or '').strip()
    expires_s = (row.get('expires') or '').strip()
    ev = (row.get('evidence') or '').strip()
    mk = lambda status, note: Entry(tk, cat, reason, added_s, expires_s, ev, status, note)

    if not tk:
        return mk('malformed', 'no ticker')
    if cat not in CATEGORIES:
        return mk('malformed', 'unknown category %r (known: %s)'
                  % (cat, ', '.join(sorted(CATEGORIES))))
    if not reason:
        #  A reason is MANDATORY.  The whole defect being repaired is a list of bare tickers
        #  that nobody could audit; an entry with no reason recreates it one line at a time.
        return mk('malformed', 'no reason given')
    try:
        added = _parse_date(added_s)
    except ValueError:
        return mk('malformed', 'unparseable added date %r (want YYYY-MM-DD)' % added_s)
    if added is None:
        return mk('malformed', 'no added date')
    try:
        expires = _parse_date(expires_s)
    except ValueError:
        return mk('malformed', 'unparseable expires date %r (want YYYY-MM-DD)' % expires_s)
    if expires is None and cat not in NEVER_EXPIRES:
        #  "Forever" must be REACHED, never DEFAULTED INTO by leaving a field empty -- that
        #  is the mechanism that made a 2023 list still authoritative in 2026.
        return mk('malformed',
                  'blank expiry is only allowed for category/ies %s; %r must state one'
                  % (', '.join(sorted(NEVER_EXPIRES)) or '(none)', cat))
    if expires is not None and expires <= as_of:
        return mk('expired', 'expired %s' % expires_s)
    return mk('live', '')


def load_exclusions(path, as_of=None, verbose=True, rotate=True, cycle=None):
    """Read an exclusion list and decide what is LIVE, and of that, what is APPLIED.

    Returns an `ExclusionVerdict`.  Never raises on a bad file: a missing path is an empty
    verdict, and a file without the schema header is REFUSED WHOLE (see (a) in the module
    docstring -- this is what makes the legacy 3,692-name bare-ticker file unreachable).

    `rotate` -- run the retest slice (CEO, 2026-08-24).  DEFAULT ON, and it is the safe
    default rather than the convenient one: with `rotate=False` an armed list is a PERMANENT
    ban on every name in it, which is the failure mode this module's whole docstring is about.
    An offline reader that wants the raw live set (a report, a diff) passes False and gets a
    verdict whose `cycle` is None, which `report()` says out loud.

    `cycle` -- override this run's rotation cycle.  Default reads it from the file's own
    `# retest_cycle:` comment (see `read_cycle`), so the counter travels with the artifact.
    """
    as_of = as_of or datetime.today().date()
    if not path or not os.path.exists(path):
        v = ExclusionVerdict([], path=None, as_of=as_of)
        if verbose:
            v.report()
        return v
    with open(path, 'r', newline='') as f:
        rows = list(csv.reader(f))
    #  Drop comment and blank lines so the file can carry its own notes.
    rows = [r for r in rows if r and not str(r[0]).lstrip().startswith('#')]
    if not rows or [c.strip() for c in rows[0]] != EXCLUSION_HEADER:
        found = ','.join(rows[0][:6]) if rows else '<empty file>'
        entry = Entry('', '', '', '', '', '', 'malformed',
                      'FILE REFUSED WHOLE: first non-comment line must be the header %r, '
                      'found %r. A bare ticker row (the pre-2026-08 '
                      'ManualEliminationTickersList format) carries no date, no reason and '
                      'no expiry, so it cannot be evaluated and 0 names are applied.'
                      % (','.join(EXCLUSION_HEADER), found))
        v = ExclusionVerdict([entry], path=path, as_of=as_of)
        if verbose:
            v.report()
        return v
    entries = [_classify(dict(zip(EXCLUSION_HEADER, r + [''] * 6)), as_of)
               for r in rows[1:]]
    _cycle = None
    _held = {}
    if rotate:
        _cycle = read_cycle(path) if cycle is None else int(cycle) % RETEST_SLOTS
        _held = select_retest(entries, _cycle, as_of)
    v = ExclusionVerdict(entries, path=path, as_of=as_of, held_out=_held, cycle=_cycle)
    if verbose:
        v.report()
    return v


def reconcile(applied, verdict):
    """Every applied ticker is accounted for by exactly one LIVE entry -- or raise.

    The universe-reconciliation identity `Sbocker.print_universe_reconciliation` prints, one
    layer down: an exclusion that no live entry explains is the exact shape of the silent
    drop this project has already been bitten by twice.  Raising is correct here because it
    can only fire on a programming error (the applied set is DERIVED from the verdict), not
    on bad user data -- bad user data is already handled as `malformed`.
    """
    live = {}
    for e in verdict.by_status('live'):
        live[e.ticker] = live.get(e.ticker, 0) + 1
    missing = [t for t in applied if t not in live]
    dupes = sorted(t for t, n in live.items() if n > 1)
    if missing:
        raise AssertionError(
            'exclusions.reconcile: %d applied ticker(s) have NO live entry explaining them: '
            '%s. Every excluded name must be accounted for by a live, unexpired entry.'
            % (len(missing), missing[:20]))
    if dupes:
        #  Not fatal -- two entries for one name is a human editing artefact, and the name is
        #  excluded either way -- but it must not pass unremarked, because deleting ONE of
        #  the two lines then looks like it did nothing.
        print('EXCLUSIONS WARNING: %d ticker(s) carry more than one live entry: %s. '
              'Removing one line will NOT re-admit the name.' % (len(dupes), dupes[:20]),
              flush=True)
    return True


def merge_entries(existing, new):
    """Hand-edited LIVE entries win; an EXPIRED entry never blocks a re-observation.

    THE CEO EDITS THIS FILE BY HAND, so the writer must never clobber his lines.  Keyed on
    (ticker, category): a LIVE existing entry is kept VERBATIM -- its reason, its dates, its
    evidence -- and a machine-proposed entry is added only where no live entry holds that key.
    A machine re-observation therefore cannot quietly extend a human's expiry, and a human
    deletion is not undone on the next run unless the machine observes the condition again.

    THE KEY IS BUILT OVER *LIVE* ENTRIES ONLY, AND THAT ONE WORD IS THE WHOLE FIX (review S2).
    It used to be built over ALL entries, expired ones included -- so the first time a
    machine-proposed entry aged off, the same failure re-observed on every subsequent run was
    silently dropped and the name could never be listed again.  That made the expiry ONE-SHOT
    and directly contradicted this module's headline argument, that a short-history name "can
    now go ON the list and AGE OFF": it aged off and could not come back, however many quarters
    it kept failing.  Latent at the time (nothing loads a schema-conforming file yet), but it
    was latent in the exact feature being shipped.

    EXPIRED MACHINE-AUTHORED DUPLICATES ARE COLLAPSED.  An expired entry is kept as EVIDENCE
    (see `write_exclusions`), but a name that fails every run for a year should leave one
    historical row, not fifty, so a superseded `run-observed` entry with the same key is
    dropped when a fresh observation replaces it.  HAND-EDITED entries are never collapsed --
    a human's record of what they did and when is not the machine's to tidy away.
    """
    out = list(existing)
    live_keys = {(e.ticker, e.category) for e in out if e.status == 'live'}
    for e in new:
        key = (e.ticker, e.category)
        if key in live_keys:
            continue
        out = [p for p in out
               if not (p.status != 'live' and (p.ticker, p.category) == key
                       and p.evidence == 'run-observed')]
        out.append(e)
        live_keys.add(key)
    return out


def propose_from_run(tickersfailed, lenfail, added=None, lenfail_periods=None):
    """The RUN's own observations, as dated entries -- the replacement for

        newmanelimtckrs = manualelimtickers + (tickersfailed - lenfail)

    The subtraction is GONE and its motive is served by the expiry instead:
      * `lenfail` (failed the history-length gate)  -> `short_history`,   90 days
      * every other fetch failure                   -> `transient_fetch`, 14 days
    so a short-history name is now RECORDED (with the reason that it was short) and ages off
    on the quarterly cadence at which the answer can actually change, rather than being
    omitted so that nothing remembers why it keeps failing.
    """
    added = added or datetime.today().date()
    lenset = {str(t) for t in (lenfail or [])}
    periods = {str(k): v for k, v in (lenfail_periods or {}).items()}
    out = []
    for t in sorted({str(x) for x in (tickersfailed or [])}):
        cat = 'short_history' if t in lenset else 'transient_fetch'
        reason = ('failed the history-length gate on this run'
                  if cat == 'short_history' else 'fetch failed on this run')
        exp = default_expiry(cat, added)
        #  A COMPUTED RETRY DATE WHERE ONE EXISTS (CEO refinement, 2026-08-24).  A name that
        #  failed the length gate holding `n` of HISTORY_GATE_PERIODS needs `gate - n` more
        #  reporting periods and CANNOT recover before then -- so re-testing it earlier is a
        #  call spent on an answer that is already known.  The date is used only when it is
        #  LATER than the category default: this may push a retry out, never pull it in.
        if cat == 'short_history':
            _sched = scheduled_retry_date(added, periods.get(t))
            if _sched is not None and (exp is None or _sched > exp):
                reason = ('failed the history-length gate on this run with %s of %d period(s); '
                          'the earliest date it can qualify is computable'
                          % (periods.get(t), HISTORY_GATE_PERIODS))
                exp = _sched
        out.append(Entry(t, cat, reason, added.strftime(DATE_FMT),
                         exp.strftime(DATE_FMT) if exp else '',
                         'run-observed', 'live', ''))
    #  SAY WHETHER THE COMPUTED SCHEDULE WAS AVAILABLE AT ALL.  Without `lenfail_periods` this
    #  cohort falls back to the flat 90-day category expiry, which is a DIFFERENT (cruder)
    #  policy -- and a silent fallback here would look exactly like a run where every name
    #  happened to need one more quarter.
    if lenset and not periods:
        print('EXCLUSIONS: %d short_history name(s) proposed with the FLAT 90-day expiry -- '
              'no per-name period count was supplied, so the computed retry date could not be '
              'used. Wiring it is a one-line change in getData_fmp (record the period count '
              'beside `lenfail.append(ticker)`).' % len(lenset), flush=True)
    return out


def write_exclusions(path, entries, cycle=None):
    """Write the schema.  Expired entries are KEPT, not deleted.

    An expired entry is EVIDENCE -- it records that the name was excluded, why, and when it
    stopped being excluded.  Deleting it on write would make the list forget its own history
    and would make a recurring failure look like a first occurrence every time.  The loader
    ignores them; the file remembers them.

    `cycle` -- the rotation counter THIS run used.  Written as a `# retest_cycle: N+1`
    comment, i.e. ADVANCED, so the next run holds out the next slice and the cycle walks all
    `RETEST_SLOTS` slots.  A comment rather than a column because `EXCLUSION_HEADER` is what
    safety lock (a) checks: adding a column would make every existing file malformed.  Passing
    None writes no counter, and a file with no counter starts the next run at slot 0 -- always
    correct, merely not continuous.
    """
    with open(path, 'w', newline='') as f:
        if cycle is not None:
            f.write('%s %d\n' % (CYCLE_COMMENT_PREFIX, (int(cycle) + 1) % RETEST_SLOTS))
        w = csv.writer(f)
        w.writerow(EXCLUSION_HEADER)
        for e in entries:
            w.writerow([e.ticker, e.category, e.reason, e.added, e.expires, e.evidence])
    return path
