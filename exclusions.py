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
  (b) EMPTY AND OFF BY DEFAULT, AND THE LOOP SHIPS **INERT**.  A missing file is an empty
      list, not an error; `configuration` still defaults `-manelimtickers` to 0; AND the run
      WRITES `ExclusionList_<ds>_<date>.csv` while the loader READS
      `DEFAULT_EXCLUSION_FILE` = `ExclusionList_fmp.csv`, so what a run accumulates is never
      what the next run applies.  Arming it is a deliberate human act: review the dated file,
      copy it to `ExclusionList_fmp.csv`, pass `-manelimtickers 1`.  Stated plainly because an
      earlier draft of this docstring described the accumulation as a working loop -- it is
      not, and calling shipped-inert machinery "working" is the same class of error as calling
      a 2023 list "current".
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
    'transient_fetch': (14, 'says almost nothing about the name: expire fast'),
    #  The CEO's own hand-added names.  A year, not forever, so a judgement made once is
    #  re-made deliberately rather than inherited silently -- which is how the 2023 list
    #  came to be applied in 2026.
    'ceo': (365, 'a human judgement should be re-made, not inherited'),
}

#  Categories whose entries may carry a BLANK `expires`.  Derived from CATEGORIES so the two
#  cannot drift apart.
NEVER_EXPIRES = frozenset(k for k, (d, _) in CATEGORIES.items() if d is None)


Entry = namedtuple('Entry', EXCLUSION_HEADER + ['status', 'note'])


class ExclusionVerdict(object):
    """What a list said, what was applied, and what was ignored -- all three, always.

    `applied`  : tickers whose entry is LIVE as of the run date.  This is the ONLY thing the
                 pipeline may filter on.
    `entries`  : every parsed entry, each stamped `status` in {live, expired, malformed}.
    `path`     : the file it came from, or None when no file was read.
    """

    def __init__(self, entries, path=None, as_of=None):
        self.entries = list(entries)
        self.path = path
        self.as_of = as_of
        self.applied = sorted({e.ticker for e in self.entries if e.status == 'live'})

    def by_status(self, status):
        return [e for e in self.entries if e.status == status]

    def counts_by_category(self):
        out = {}
        for e in self.by_status('live'):
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


def load_exclusions(path, as_of=None, verbose=True):
    """Read an exclusion list and decide what is LIVE as of `as_of` (default: today).

    Returns an `ExclusionVerdict`.  Never raises on a bad file: a missing path is an empty
    verdict, and a file without the schema header is REFUSED WHOLE (see (a) in the module
    docstring -- this is what makes the legacy 3,692-name bare-ticker file unreachable).
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
    v = ExclusionVerdict(entries, path=path, as_of=as_of)
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


def propose_from_run(tickersfailed, lenfail, added=None):
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
    out = []
    for t in sorted({str(x) for x in (tickersfailed or [])}):
        cat = 'short_history' if t in lenset else 'transient_fetch'
        reason = ('failed the history-length gate on this run'
                  if cat == 'short_history' else 'fetch failed on this run')
        exp = default_expiry(cat, added)
        out.append(Entry(t, cat, reason, added.strftime(DATE_FMT),
                         exp.strftime(DATE_FMT) if exp else '',
                         'run-observed', 'live', ''))
    return out


def write_exclusions(path, entries):
    """Write the schema.  Expired entries are KEPT, not deleted.

    An expired entry is EVIDENCE -- it records that the name was excluded, why, and when it
    stopped being excluded.  Deleting it on write would make the list forget its own history
    and would make a recurring failure look like a first occurrence every time.  The loader
    ignores them; the file remembers them.
    """
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(EXCLUSION_HEADER)
        for e in entries:
            w.writerow([e.ticker, e.category, e.reason, e.added, e.expires, e.evidence])
    return path
