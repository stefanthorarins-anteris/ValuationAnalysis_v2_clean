"""Per-source REPORTING FREQUENCY classification, and the rows-per-year it implies.

WHY THIS EXISTS
---------------
FMP serves every fundamentals row as a "quarter", but ~14% of the universe (mostly LSE)
reports SEMI-ANNUALLY, and FMP labels those halves Q2/Q4 while the flow figures inside
them cover SIX months.  Every row-window assumption in this pipeline was written as
"4 rows = 1 year": TTM sums, YoY growth (pct_change(-4)), Piotroski's prior-year row,
the Beneish/Montier YoY shifts, the head(n) scoring windows, the >=16-row history gate.
For a semi-annual filer all of those are wrong by a factor of two -- a "TTM" spans 24
months, a "YoY" growth spans 2 years, and a 4-row scoring window covers 2 years of
calendar time against a quarterly peer's 1.

This module is the ONE place that decides, per source, whether a row is a quarter or a
half.  Callers ask for `rows_per_year` and parameterise their window with it; nothing
forks into a duplicated semi-annual code path.

It also owns the OTHER half of "how to read a per-source panel": the ROW-ORDER contract
(`to_newest_first` / `assert_newest_first`, at the bottom of this file).  Row order and
window length are the same kind of fact -- both decide WHICH rows a metric sees -- and both
had been remembered at each call site instead of stated at a boundary, which is how
`moatIdentifier` came to read a name's OLDEST rows and two log blocks came to print the
oldest three under the label "first 3".

CLASSIFICATION, in priority order
---------------------------------
1. THE `period` FIELD (primary, authoritative).  Captured at ingest since 2026-07-19
   (audit H-2 / fix 14) and present in cdx_df from the next full fetch onward.  A
   quarterly filer shows Q1 and/or Q3; a semi-annual filer shows only Q2/Q4 (FMP's
   labelling of H1/H2) or literal H1/H2.
2. DATE CADENCE (fallback, for every pickle saved before that field existed).  The
   median gap between a source's consecutive period-end dates: ~92 days quarterly,
   ~183 days semi-annual.  This is the method the fmp-specialist audit used.

CONSERVATIVE BIAS (deliberate, and the reason for the UNKNOWN state)
-------------------------------------------------------------------
Misclassifying a QUARTERLY reporter as semi-annual is the expensive error: it halves
every window on a name whose rows really are quarters, corrupting its TTM sums and YoY
growth outright.  Misclassifying a semi-annual reporter as quarterly merely leaves
today's known defect in place for that name.  So the thresholds are asymmetric, the
ambiguous band resolves to UNKNOWN, and UNKNOWN is treated as QUARTERLY -- i.e. exactly
today's behaviour, no change, counted and logged rather than guessed.
"""

import numpy as np
import pandas as pd

QUARTERLY = 'quarterly'
SEMIANNUAL = 'semiannual'
UNKNOWN = 'unknown'

ROWS_PER_YEAR = {QUARTERLY: 4, SEMIANNUAL: 2}
DEFAULT_ROWS_PER_YEAR = 4          # UNKNOWN -> quarterly path (unchanged behaviour)

# --- cadence thresholds (days between consecutive period ends) -----------------------
# A quarter is ~91.3 days, a half ~182.6.  The gap between the bands is enormous, so
# both cut-offs sit far from either mode and the ambiguous middle is left UNKNOWN.
CADENCE_SEMIANNUAL_MIN_DAYS = 150.0   # >= this -> semi-annual
CADENCE_QUARTERLY_MAX_DAYS = 120.0    # <= this -> quarterly
CADENCE_MIN_GAPS = 3                  # need >=3 gaps (4 rows) before trusting a median

# --- the fetch HISTORY GATE, in CALENDAR years --------------------------------------
# `failTests.testForAPIFaults_fmp` rejects a ticker with less than FETCH_HISTORY_YEARS of
# history, expressed as `rows_per_year(freq) * FETCH_HISTORY_YEARS` rows -- 16 quarterly OR
# 8 semi-annual, i.e. the same calendar span either way.  It lives HERE, beside the other
# per-frequency window constants, because CLASSIFY_RECENT_DAYS is deliberately the SAME
# span (see its note) and the two must not drift: the classifier should look at the same
# history the gate demands.  The gate used to spell the 4 out as a bare literal.
FETCH_HISTORY_YEARS = 4

# --- `period`-label thresholds ------------------------------------------------------
# A quarterly filer with a very short history could show only Q2/Q4 by chance; require a
# real sample before concluding semi-annual from labels alone.
PERIOD_MIN_LABELLED_ROWS = 4

_QUARTERLY_ONLY_LABELS = {'Q1', 'Q3'}
_HALF_LABELS = {'H1', 'H2', 'S1', 'S2'}
_SEMI_COMPATIBLE_LABELS = {'Q2', 'Q4', 'H1', 'H2', 'S1', 'S2'}


def _norm_label(x):
    return x.strip().upper() if isinstance(x, str) else ''


def classify_from_period(period_values):
    """Frequency from a source's FISCAL `period` labels.  UNKNOWN when they cannot decide.

    CONTRACT -- DO NOT "IMPROVE" THIS BY BACK-FILLING `period` FROM `date`.
    A missing `period` column means NO SIGNAL, and the caller must fall back to date
    cadence.  It must NEVER be reconstructed from the date column, because
    utils.setDatesToQuarterly snaps every period end onto a CALENDAR-quarter start: a
    semi-annual filer's June/December halves become "Q2"/"Q4" but its fiscal halves ending
    e.g. March/September become "Q1"/"Q3".  Deriving labels that way would classify ~279
    of the 1,118 genuinely semi-annual sources on the 2026-07-17 panel as QUARTERLY --
    inverting the classification on exactly the names this module exists to protect.
    FMP's real field is FISCAL and only ever stamps a half as Q2/Q4 (verified live on
    SMIN.L and TAM.L, 2026-07-25).
    """
    labels = {_norm_label(v) for v in (period_values if period_values is not None else [])}
    labels.discard('')
    if not labels:
        return UNKNOWN
    if labels & _QUARTERLY_ONLY_LABELS:
        return QUARTERLY                      # Q1 or Q3 present -> a real quarterly filer
    if labels & _HALF_LABELS:
        return SEMIANNUAL                     # explicit halves
    n_labelled = sum(1 for v in period_values if _norm_label(v))
    if labels <= _SEMI_COMPATIBLE_LABELS and n_labelled >= PERIOD_MIN_LABELLED_ROWS:
        return SEMIANNUAL                     # only Q2/Q4 across a real sample
    return UNKNOWN                            # FY-only, or too short to tell


def classify_from_cadence(dates):
    """Frequency from the median gap between consecutive period ends.  UNKNOWN in the
    ambiguous band, or when there are too few usable gaps."""
    d = pd.to_datetime(pd.Series(list(dates)), errors='coerce').dropna()
    if len(d) < CADENCE_MIN_GAPS + 1:
        return UNKNOWN
    gaps = d.sort_values().diff().dt.days.dropna()
    gaps = gaps[gaps > 0]
    if len(gaps) < CADENCE_MIN_GAPS:
        return UNKNOWN
    med = float(gaps.median())
    if med >= CADENCE_SEMIANNUAL_MIN_DAYS:
        return SEMIANNUAL
    if med <= CADENCE_QUARTERLY_MAX_DAYS:
        return QUARTERLY
    return UNKNOWN


# =========================================================================== #
#  WHICH SIGNAL WINS WHEN `period` AND CADENCE DISAGREE -- ONE FLIP-ABLE FLAG   #
# =========================================================================== #
# SETTLED (fmp-specialist live check on SMIN.L + TAM.L, 2026-07-25): PERIOD WINS.
# FMP's `period` is a FISCAL label -- a semi-annual filer's halves are only ever stamped
# Q2/Q4, never Q1/Q3 (zero Q1/Q3 across both test names; gaps uniformly ~182-184d).  So
# `period` is authoritative wherever it exists and needs no tie-break against cadence.
#
# The review's alarm -- "279 of the 1,118 cadence-semiannual sources carry Q1/Q3 period
# stamps" -- was OUR OWN artifact, not FMP's: saved cdx_df pickles carry NO `period`
# column at all, and utils.setDatesToQuarterly snaps every date to a calendar-quarter
# start, so those "stamps" were CALENDAR labels derived from the date column and
# orthogonal to FMP's fiscal field.  See the CONTRACT note on classify_from_period.
#
# The constant stays as a single documented seam (if FMP's convention ever changes, this
# is the one line to flip), and the conflict logging below stays as a STANDING GUARD --
# on real fetched data it should report ~zero, so any future count is a real signal.
#
#   'period'   -- period wins outright  [CURRENT, and the verified-correct choice]
#   'cadence'  -- cadence wins outright
#   'unknown'  -- a disagreement resolves to UNKNOWN -> the quarterly path, counted
CLASSIFIER_PRIORITY = 'period'


# =========================================================================== #
#  RECENCY: classify on the RECENT history, not the whole fetched panel        #
# =========================================================================== #
# THE DEFECT (fix, 2026-07-31).  Both signals were WHOLE-HISTORY reductions:
#   * classify_from_period is a SET UNION -- a single `Q1` or `Q3` label ANYWHERE in the
#     fetched history stamps the source QUARTERLY for its entire panel;
#   * classify_from_cadence is a MEDIAN GAP over the whole window.
# Neither has a notion of "now", so the verdict's meaning is a function of FETCH DEPTH.  At
# `-nrperiods 24` the window is ~6 years; at `-nrperiods 80` it is ~20.
#
# That matters because reporting frequency is not a fixed property of an issuer -- it CHANGES.
# UK/LSE issuers largely dropped quarterly reporting after the FCA removed the
# interim-management-statement requirement (~2014-15).  So on a deep fetch a currently
# SEMI-ANNUAL LSE filer with a pre-2015 quarterly era carries Q1/Q3 labels from that era, the
# set union sees them, and it is stamped QUARTERLY -- silently reverting the ENTIRE semi-annual
# fix wave for that name: Graham's EPS_ttm sums 4 halves = 24 months, Piotroski's prior-period
# lag becomes 2 years, the Stage-1 flow factors run at 1.0 instead of 0.5, and every
# scale_window call goes unscaled.  ~14% of the universe reports semi-annually, so this is the
# cohort the whole module exists to protect.
#
# THE WINDOW, and why 4 years.  It is bounded on BOTH sides, and the bounds nearly meet:
#   LOWER BOUND -- the window must contain enough rows for the SEMI-ANNUAL verdicts to be
#   reachable at all, and semi-annual is the sparse side (2 rows/year).  The label path needs
#   PERIOD_MIN_LABELLED_ROWS = 4 all-Q2/Q4 rows = 2 years; the cadence path needs
#   CADENCE_MIN_GAPS = 3 gaps = 4 rows = ~1.5 years.  A window under ~2 years makes SEMIANNUAL
#   UNREACHABLE, and UNKNOWN is treated as quarterly -- i.e. too short a window reverts the fix
#   just as thoroughly as too long a one.
#   UPPER BOUND -- the window must not span a regime change.  The FCA change is ~11 years back
#   from 2026, but the bound that matters is the GENERAL one: whatever the next convention
#   change is, a shorter window clears it sooner.
# 4 years sits with ~2x margin over the lower bound (a semi-annual filer gets ~8 rows against a
# 4-row floor and ~7 gaps against a 3-gap floor; a quarterly filer gets ~16 rows and ~8
# separate Q1/Q3 sightings, so a missing quarter cannot flip it) and ~3x clearance under the
# upper one.  It is also exactly the calendar span the pipeline's own history gate already
# demands (FETCH_HISTORY_YEARS, consumed by failTests.testForAPIFaults_fmp as
# rows_per_year x FETCH_HISTORY_YEARS rows), so the classifier now looks at the same span the
# fetch gate requires rather than at whatever depth was fetched.
#
# ANCHORED TO THE SOURCE'S OWN NEWEST ROW, NOT TO TODAY.  A dead/delisted name in the
# survivorship-clean dead-merged universe, and every point-in-time `as_of` reproduction, has a
# newest row that is YEARS old; a window anchored to the wall clock would select ZERO rows for
# them and hand every such name back as UNKNOWN -> quarterly, reverting the fix across the
# entire offline PIT path.  "The most recent 4 years OF THIS SOURCE'S HISTORY" is
# fetch-depth-independent AND as-of-independent.
CLASSIFY_RECENT_DAYS = 1461                  # 4.0 years (4 x 365.25), see above

# FLOOR, so recency can never DESTROY a signal it was meant to sharpen.  If the 4-year window
# holds fewer than this many rows (a sparse or gappy source), fall back to the most recent
# CLASSIFY_RECENT_MIN_ROWS rows regardless of their dates.  8 rows = 4 semi-annual years, i.e.
# 2x the PERIOD_MIN_LABELLED_ROWS floor -- enough for either verdict to be reachable.  Without
# this floor a semi-annual filer with one missing half in the last four years would drop to
# UNKNOWN -> the quarterly path, which is the very outcome the fix exists to prevent.  The
# fallback is still BOUNDED (8 rows, not the whole panel), so the fetch-depth dependence is
# gone either way.
CLASSIFY_RECENT_MIN_ROWS = 8


def _recent_slice(dates, period_values, days=CLASSIFY_RECENT_DAYS,
                  min_rows=CLASSIFY_RECENT_MIN_ROWS):
    """(dates, period_values) restricted to the source's most RECENT `days` of history.

    `dates` and `period_values` must be POSITIONALLY aligned (all three call sites take both
    off the same per-source frame, so they are).  Returns the inputs UNCHANGED -- i.e. today's
    whole-history behaviour -- whenever the restriction cannot be made safely:
      * `dates` is None (a caller with labels but no dates has nothing to be recent about);
      * no date parses (an all-NaT column carries no recency information);
      * the two inputs disagree in length (alignment cannot be assumed, so do not guess).
    Falling back to the OLD behaviour on every unsafe input is deliberate: this function must
    never be the reason a source loses its classification.
    """
    if dates is None:
        return dates, period_values
    d = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True), errors='coerce')
    if d.notna().sum() == 0:
        return dates, period_values
    pv = None if period_values is None else list(period_values)
    if pv is not None and len(pv) != len(d):
        return dates, period_values
    newest = d.max()
    # Compared as a DAY DIFFERENCE rather than `newest - Timedelta(days)`: constructing a
    # Timedelta from a large `days` raises OutOfBoundsTimedelta, so an over-wide window (a
    # caller disabling the cap, or a test sweeping the constant) would blow up instead of
    # degrading to whole-history.  The subtraction of two real dates cannot overflow.
    age_days = (newest - d).dt.days
    keep = age_days.notna() & (age_days <= int(days))
    if int(keep.sum()) < int(min_rows):
        # Sparse/gappy: take the most recent `min_rows` DATED rows instead of the calendar
        # window.  Ranking by date (not by row order) so this does not depend on whether the
        # caller's frame is newest-first or oldest-first -- the two call orders differ.
        order = d.rank(method='first', ascending=False)          # 1 = newest; NaT -> NaN
        keep = order.notna() & (order <= int(min_rows))
        if int(keep.sum()) == 0:
            return dates, period_values
    idx = list(keep[keep].index)
    return d.iloc[idx], (None if pv is None else [pv[i] for i in idx])


def classify_source(dates=None, period_values=None, conflicts=None, source=None,
                    recent_days=CLASSIFY_RECENT_DAYS):
    """Frequency for ONE source, decided on its RECENT history (see CLASSIFY_RECENT_DAYS).

    THE ONE PLACE THE RECENCY WINDOW IS APPLIED.  All three production call sites funnel
    through here -- the ingest stamp (getData_fmp.fillPreReqdf), the fetch history gate
    (failTests) and the whole-panel map (frequency_by_source) -- so the window cannot be
    applied at two of three sites, which is this project's signature defect.  The two
    primitives (classify_from_period / classify_from_cadence) stay PURE whole-input reductions:
    they are the "what do these labels/dates say" layer and are called directly by tests.

    `conflicts`: optional list.  When `period` and cadence BOTH resolve and DISAGREE, a
    record (source, by_period, by_cadence) is appended -- that is how the disagreement
    becomes a counted, loggable event rather than a silent tie-break.  BOTH verdicts are taken
    on the SAME recent window, so a conflict is now a genuine disagreement between the two
    signals rather than an artifact of them reading different spans.
    """
    dates, period_values = _recent_slice(dates, period_values, days=recent_days)
    by_period = classify_from_period(period_values) if period_values is not None else UNKNOWN
    by_cadence = classify_from_cadence(dates) if dates is not None else UNKNOWN

    if by_period != UNKNOWN and by_cadence != UNKNOWN and by_period != by_cadence:
        if conflicts is not None:
            conflicts.append((source, by_period, by_cadence))
        if CLASSIFIER_PRIORITY == 'cadence':
            return by_cadence
        if CLASSIFIER_PRIORITY == 'unknown':
            return UNKNOWN
        return by_period                      # 'period' -- current, unchanged
    if by_period != UNKNOWN:
        return by_period
    return by_cadence


# =========================================================================== #
#  ONE SOURCE OF TRUTH: the ingest-time verdict, CARRIED IN THE DATA           #
# =========================================================================== #
# Column written ONCE per ticker by getData_fmp.fillPreReqdf and read by every downstream
# consumer.  Before this (review item 9) three sites decided independently:
#   A  the Graham EPS_ttm window, classifying from the RAW period-end dates;
#   B  the flow factors / diff windows / tail trim / ROE hurdle, classifying from the
#      SNAPPED quarter-start dates (utils.setDatesToQuarterly had already run);
#   C  Stage-1, on a third path.
# A and B can DISAGREE, because snapping moves a period end by up to ~92 days: a
# semi-annual filer whose halves end on the first day of one quarter and the last day of
# the next-but-one has RAW gaps ~182d (semi-annual) but SNAPPED gaps ~91d (quarterly).
# 282 sources on the 07-17 panel carry colliding snapped quarters, so this is a live
# surface, not a hypothetical.
#
# The fix is to classify ONCE, as early as the data allows -- at ingest, on the RAW dates
# and the authoritative `period` field, before anything is snapped -- and have every other
# site READ that verdict rather than re-derive it.  `frequency_by_source` therefore prefers
# this column over `period` over cadence.
FREQ_COLUMN = 'reportingFrequency'
_VALID_FREQ = (QUARTERLY, SEMIANNUAL, UNKNOWN)

# =========================================================================== #
#  THE CONFLICT WATCHDOG, CARRIED IN THE DATA TOO                              #
# =========================================================================== #
# WHY THIS COLUMN EXISTS (fix, 2026-07-31).  The "STANDING GUARD" below reported NOTHING, by
# construction, and would have reported nothing on the deep-history fetch:
#   * `classify_source` only records a conflict when handed a `conflicts` list, and the INGEST
#     site -- the ONLY place that holds both the RAW period-end dates and the authoritative
#     `period` field -- called it withOUT `conflicts` and withOUT `source`;
#   * the universe-wide banner site (postBo.py) DOES pass a list, but `frequency_by_source`
#     SHORT-CIRCUITS on the stored FREQ_COLUMN and so never calls `classify_source` at all.
# Those two facts compose: the moment the ingest verdict started being stored (which is the
# right design), the only site that could see a conflict stopped looking and the only site
# still looking could no longer see one.  Reproduced 2026-07-31: a frame carrying a genuine
# period-vs-cadence conflict AND a stored verdict emits no banner and no CSV; drop the stored
# column and both appear.  So "zero conflicts reported" meant "nothing looked", not "none
# exist" -- and the ship-gate called this the single most important post-fetch diagnostic.
#
# The fix has to survive the short-circuit, so the conflict is STAMPED INTO THE FRAME beside
# the verdict itself: detected once at ingest (where both signals exist and are un-snapped),
# carried in the panel, and re-emitted by `frequency_by_source` on the stored-column path.
# That makes the diagnostic work at BOTH sites, and makes it survive the pickle -- postBo runs
# off a saved panel, so an in-memory-only accumulator would go dark again the moment the fetch
# and the ranking are separate invocations.
#
# ENCODING: '' (or NaN) = no conflict; otherwise 'by_period|by_cadence'.  A plain string so it
# rides through forceNumOnDf's passthrough list and pickles without a custom dtype.
FREQ_CONFLICT_COLUMN = 'reportingFrequencyConflict'
_CONFLICT_SEP = '|'


def encode_conflict(conflicts):
    """'by_period|by_cadence' for the FIRST record in `conflicts`, else '' (no conflict).

    `classify_source` appends at most one record per source, so the first is the only one.
    Taking the first rather than asserting len<=1 keeps this a diagnostic that cannot itself
    break a 12-hour fetch."""
    if not conflicts:
        return ''
    _src, by_period, by_cadence = conflicts[0]
    return '%s%s%s' % (by_period, _CONFLICT_SEP, by_cadence)


def decode_conflict(value):
    """(by_period, by_cadence) from a stamped value, or None when there is no conflict."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or _CONFLICT_SEP not in v:
        return None
    by_period, _sep, by_cadence = v.partition(_CONFLICT_SEP)
    by_period, by_cadence = by_period.strip(), by_cadence.strip()
    if by_period in _VALID_FREQ and by_cadence in _VALID_FREQ:
        return (by_period, by_cadence)
    return None


def frequency_by_source(cdx_df, verbose=False, csv=False):
    """{source: frequency} for a whole panel, computed ONCE.

    PRIORITY: the stored FREQ_COLUMN (the ingest-time verdict, the single source of truth)
    -> the `period` field -> date cadence.  Sources that cannot be decided come back
    UNKNOWN, which every caller treats as quarterly.

    Conflicts are collected from BOTH paths -- freshly, for sources classified here, and by
    decoding FREQ_CONFLICT_COLUMN for sources served from the stored verdict -- so the
    watchdog reports on a panel that carries the ingest verdict (see FREQ_CONFLICT_COLUMN).

    `csv` -- write ReportingFrequencyConflicts_<date>.csv.  Defaults to FALSE and is enabled
    at exactly ONE site: postBo's UNIVERSE-WIDE read.  There are four `verbose=True` callers
    and they run on DIFFERENT pools (the full universe, the top-100, Stage-1, the forensics);
    if each wrote the shared filename, the last one to run -- typically a narrow pool with
    nothing to report -- would CLOBBER the universe-wide list with a shorter or empty one.
    The banner still prints at every site; only the artifact is single-writer.
    """
    out = {}
    if cdx_df is None or 'source' not in getattr(cdx_df, 'columns', []):
        return out
    has_freq = FREQ_COLUMN in cdx_df.columns
    has_period = 'period' in cdx_df.columns
    has_conflict = FREQ_CONFLICT_COLUMN in cdx_df.columns
    cols = ['source'] + (['date'] if 'date' in cdx_df.columns else []) \
                      + (['period'] if has_period else []) \
                      + ([FREQ_COLUMN] if has_freq else []) \
                      + ([FREQ_CONFLICT_COLUMN] if has_conflict else [])
    sub = cdx_df[cols]
    conflicts = []
    n_from_col = 0
    for src, grp in sub.groupby('source', sort=False):
        stored = ''
        if has_freq:
            # One verdict per source by construction; take the first non-empty and, if the
            # frame somehow carries two, fall through rather than silently pick one.
            vals = {str(v).strip() for v in grp[FREQ_COLUMN]
                    if isinstance(v, str) and str(v).strip() in _VALID_FREQ}
            if len(vals) == 1:
                stored = vals.pop()
        if stored:
            out[src] = stored
            n_from_col += 1
            # THE SHORT-CIRCUIT MUST NOT SWALLOW THE DIAGNOSTIC.  `classify_source` is not
            # called on this path, so a conflict can only be reported by decoding the one the
            # ingest site already found and stamped (FREQ_CONFLICT_COLUMN).  Without this the
            # banner site sees an empty list on every panel that carries an ingest verdict --
            # i.e. on all of them from the next fetch onward.
            if has_conflict:
                for _v in grp[FREQ_CONFLICT_COLUMN]:
                    _dec = decode_conflict(_v)
                    if _dec is not None:
                        conflicts.append((src, _dec[0], _dec[1]))
                        break
            continue
        out[src] = classify_source(
            dates=grp['date'] if 'date' in grp.columns else None,
            period_values=list(grp['period']) if has_period else None,
            conflicts=conflicts, source=src)
    if verbose:
        src_of_truth = ('stored %s (ingest verdict)' % FREQ_COLUMN if n_from_col
                        else ('period' if has_period else 'cadence'))
        if n_from_col and n_from_col < len(out):
            src_of_truth += (' for %d of %d sources; the rest fell back to %s'
                             % (n_from_col, len(out),
                                'period' if has_period else 'cadence'))
        print(describe_counts(out, source_of_truth=src_of_truth), flush=True)
    # n_examined is what turns "no banner" into a POSITIVE statement.  Passing the source
    # count means a zero result reads "the watchdog RAN over N sources and found none",
    # which is a different fact from "nothing looked" -- the fact that was indistinguishable
    # before this fix and that made the diagnostic worthless.
    log_conflicts(conflicts, verbose=verbose, csv=csv, n_examined=len(out),
                  detected_via=('stamped ingest column + fresh classification'
                                if has_conflict else 'fresh classification only'))
    return out


def log_conflicts(conflicts, verbose=True, csv=True, n_examined=None,
                  detected_via=None, label=''):
    """LOUD, counted, per-source report of every `period`-vs-cadence disagreement.

    A disagreement means the two independent signals for a source's reporting frequency
    do not agree, so ONE of them is wrong for that name and the tie-break
    (CLASSIFIER_PRIORITY) is silently deciding how it gets scored.  That must never be
    invisible: 279 such names exist on the 2026-07-17 panel, and 25% of the semi-annual
    cohort flipping path between two runs would otherwise look like ordinary churn.

    ZERO IS REPORTED AS A RESULT, NOT AS SILENCE (fix, 2026-07-31).  This used to `return`
    immediately on an empty list, which made "0 conflicts" and "nothing ever looked"
    indistinguishable in the run log -- and for the whole period the ingest verdict was being
    stored, it was ALWAYS the latter (see FREQ_CONFLICT_COLUMN).  With `n_examined` supplied
    the empty case prints an affirmative one-liner and still writes the (header-only) CSV, so
    the ARTIFACT'S ABSENCE now means the watchdog did not run.  That is the whole point of a
    standing guard: a guard that is silent when healthy cannot be distinguished from a guard
    that is dead.
    """
    if not conflicts:
        if verbose:
            print('REPORTING-FREQUENCY CONFLICT CHECK%s: 0 conflict(s)%s -- the watchdog RAN'
                  '%s. (`period` and date cadence agree wherever both resolve.)'
                  % (' [%s]' % label if label else '',
                     '' if n_examined is None else ' across %d source(s) examined' % n_examined,
                     '' if not detected_via else ' via %s' % detected_via), flush=True)
        if csv:
            _write_conflict_csv([], verbose=verbose)
        return conflicts
    if verbose:
        print('!' * 78, flush=True)
        print('!!! REPORTING-FREQUENCY CONFLICT%s: `period` and date cadence DISAGREE on '
              '%d source(s)%s.'
              % (' [%s]' % label if label else '', len(conflicts),
                 '' if n_examined is None else ' of %d examined' % n_examined), flush=True)
        print('!!! Tie-break in force: CLASSIFIER_PRIORITY = %r (reporting_period.py) -- '
              'one constant, flip it when the FMP `period`-semantics question is settled.'
              % CLASSIFIER_PRIORITY, flush=True)
        n_pq = sum(1 for _s, p, c in conflicts if p == QUARTERLY and c == SEMIANNUAL)
        n_ps = sum(1 for _s, p, c in conflicts if p == SEMIANNUAL and c == QUARTERLY)
        print('!!!   period=quarterly / cadence=semiannual : %d  <-- the reclassification '
              'risk (a real half-yearly filer scored as quarterly)' % n_pq, flush=True)
        print('!!!   period=semiannual / cadence=quarterly : %d' % n_ps, flush=True)
        print('!!!   sources: %s'
              % ', '.join(str(s) for s, _p, _c in conflicts[:40])
              + (' ... (+%d more, see the CSV)' % (len(conflicts) - 40)
                 if len(conflicts) > 40 else ''), flush=True)
        print('!' * 78, flush=True)
    if csv:
        _write_conflict_csv(conflicts, verbose=verbose)
    return conflicts


def _write_conflict_csv(conflicts, verbose=True):
    """Write ReportingFrequencyConflicts_<date>.csv -- ALWAYS, even when empty.

    Header-only on zero conflicts, deliberately: the file's PRESENCE is the evidence that the
    check ran, so its absence is now a real signal rather than the ambiguity it used to be.
    Best-effort and fully swallowed -- a diagnostic must never be able to abort a 12-hour
    fetch, which is why this is the one place that touches the filesystem."""
    try:
        fn = ('ReportingFrequencyConflicts_%s.csv'
              % pd.Timestamp.today().strftime('%Y-%m-%d'))
        pd.DataFrame(list(conflicts),
                     columns=['source', 'by_period', 'by_cadence']).to_csv(fn, index=False)
        if verbose:
            print('  frequency-conflict list written to: %s (%d row(s))'
                  % (fn, len(conflicts)), flush=True)
    except Exception as _e:
        if verbose:
            print('  WARNING: could not write frequency-conflict list (%s)' % _e, flush=True)


def rows_per_year_by_source(cdx_df, verbose=False):
    """{source: rows_per_year} -- 2 for semi-annual, 4 for quarterly AND for unknown."""
    return {s: ROWS_PER_YEAR.get(f, DEFAULT_ROWS_PER_YEAR)
            for s, f in frequency_by_source(cdx_df, verbose=verbose).items()}


def rows_per_year(freq_or_map, source=None):
    """rows_per_year from a frequency string, or from a {source: freq} map + source.

    Anything unrecognised (including None and UNKNOWN) yields DEFAULT_ROWS_PER_YEAR, so a
    caller that has no classification available behaves exactly as it does today.
    """
    if isinstance(freq_or_map, dict):
        if source is None:
            return DEFAULT_ROWS_PER_YEAR
        val = freq_or_map.get(source, DEFAULT_ROWS_PER_YEAR)
        if isinstance(val, str):
            return ROWS_PER_YEAR.get(val, DEFAULT_ROWS_PER_YEAR)
        try:
            return int(val) if int(val) in (2, 4) else DEFAULT_ROWS_PER_YEAR
        except (TypeError, ValueError):
            return DEFAULT_ROWS_PER_YEAR
    return ROWS_PER_YEAR.get(freq_or_map, DEFAULT_ROWS_PER_YEAR)


def scale_window(n, rpy, minimum=1):
    """Scale a row-count window written for QUARTERLY data to `rpy` rows per year so it
    spans the SAME CALENDAR TIME.  rpy=4 returns n unchanged (bit-identical); rpy=2
    halves it, rounded to the nearest row and never below `minimum` (>= 1 row).

    CONTRACT: the result is NEVER LARGER than `n`.  A `minimum=2` floor used to violate
    exactly that (review H1, 2026-07-25): scale_window(1, rpy=2) returned 2 -- a window
    TWICE the quarterly one, on the metric family with the smallest window in the
    pipeline.  Production runs `fsMAnumber = 1` (its default in
    configuration.getDataFetchConfiguration), so every
    semi-annual name's calc_diff ran `rolling(2).mean()` -- a 12-month smoothing -- where
    a quarterly name ran `rolling(1)`, i.e. no smoothing at all.  That silently altered
    all 18 d* columns (44.5% of Stage-1 summed weight) for 14.4% of the universe.

    Rounding is HALF-UP (floor(x + 0.5)) rather than Python's bankers' rounding, so
    n=1, rpy=2 -> 1 row instead of 0, and n=3, rpy=2 -> 2 rather than 1.  A 0.5-row
    window is not representable, and rounding it DOWN to 0 would be worse than rounding
    up to 1: 1 row is the shortest window that exists, and for a semi-annual filer one
    row already covers twice the calendar time of a quarterly row.
    """
    # n <= 0 is degenerate -- a zero-length window is not a window.  Clamp it in BOTH
    # branches so the contract is uniform (review R-N6: the floor used to sit OUTSIDE the
    # cap, so n=0 returned 1 on the scaled path but 0 on the rpy=4 path -- two different
    # answers to the same degenerate input).
    n = int(n)
    if n <= 0:
        return max(1, int(minimum))
    if rpy is None or int(rpy) == DEFAULT_ROWS_PER_YEAR:
        return n
    exact = n * (int(rpy) / float(DEFAULT_ROWS_PER_YEAR))
    scaled = int(np.floor(exact + 0.5))
    # Cap FIRST, then floor -- and the floor itself can never exceed n, so the invariant
    # 1 <= result <= n holds for every input.
    return min(n, max(min(int(minimum), n), scaled))


# =========================================================================== #
#  FLOW-SCALE CORRECTION (valuation-specialist annualization ruling, 2026-07-25)
# =========================================================================== #
# A semi-annual row's FLOW covers six months, a quarterly row's three.  Any metric that
# divides a FLOW by a STOCK (or a stock by a flow) therefore reads ~2x (or ~1/2x) on a
# semi-annual filer purely from the reporting convention.  Two different corrections are
# needed, and WHICH ONE depends on whether the test has an ABSOLUTE threshold:
#
#  * per_quarter_factor  -- for SCALE-FREE tests: a cross-sectional MEAN test
#    (metvec - median) or a z-scored Stage-2 metric is invariant to multiplying EVERY
#    row by the same constant, so only the RELATIVE semi-annual/quarterly ratio matters.
#    Normalising to a common PER-QUARTER basis (x1 quarterly, x0.5 semi-annual) fixes the
#    comparability and leaves the quarterly path BIT-IDENTICAL.  Equivalent to true
#    annualisation up to the global constant 4, which such tests cannot see.
#
#  * annualize_factor    -- for tests with an ABSOLUTE threshold (a unity test against
#    1.0, the moat comparators against 0.10/0.15/0.75, Altman's fixed coefficients).
#    Here the basis is load-bearing: netDebt/EBITDA must be compared against a YEAR of
#    EBITDA, not a quarter.  True annualisation (x rows_per_year) is required, and it is
#    NOT a no-op for quarterly names -- that is intended, not an accident.
def per_quarter_factor(rpy):
    """Factor putting a source's per-period flow on a common PER-QUARTER basis.
    1.0 for quarterly (exact no-op), 0.5 for semi-annual."""
    return float(int(rpy)) / float(DEFAULT_ROWS_PER_YEAR)


def annualize_factor(rpy):
    """Factor converting a source's per-period flow to a FULL-YEAR flow: x4 quarterly,
    x2 semi-annual.  Use only where a threshold is absolute (see the note above)."""
    return float(int(rpy))


# Stage-1 metric keys whose ratio must be flow-corrected, and how.
#   'flow_num'  : flow / stock  -> multiply the ratio (earningsYield, salesToMarketCap)
#   'flow_den'  : stock / flow  -> DIVIDE the ratio (netDebtToEBITDA)
# The MODE ('per_quarter' vs 'annualize') follows the test's threshold type, per above.
#
# THE LEG IS A PROPERTY OF THE RATIO'S ORIENTATION, SO INVERTING A METRIC MOVES IT.  When
# `pfcfRatio` (price/FCF, flow in the DENOMINATOR) was inverted to `freeCashFlowToMarketCap`
# (FCF/marketCap, flow in the NUMERATOR) on 2026-08-04, this entry had to move from 'flow_den'
# to 'flow_num' with it.  Leaving the old leg would have applied the semi-annual correction
# BACKWARDS -- x2.0 where x0.5 is needed, a 4x error on every semi-annual name, silently,
# because the key still resolved and still returned a plausible factor.  If you ever inverse a
# Stage-1 ratio, check this table in the same edit.
STAGE1_FLOW_CORRECTION = {
    # mean tests: SINCE C-12 (2026-08-06) these are tested against ABSOLUTE STORED BARS
    # (meanBars.BARS), no longer against the pooled median -- so the old "scale-free, any common
    # basis will do" justification is GONE, yet the mode is UNCHANGED and must stay 'per_quarter':
    # the bars are STATED ON THE PER-QUARTER BASIS THIS FACTOR PRODUCES.  Switching either yield
    # to 'annualize' against an unchanged bar would silently QUADRUPLE the threshold.
    'earningsYield':            ('flow_num', 'per_quarter'),
    'freeCashFlowToMarketCap':  ('flow_num', 'per_quarter'),   # was pfcfRatio / 'flow_den'
    'salesToMarketCap':         ('flow_num', 'per_quarter'),   # w=0, corrected for consistency
    # unity test against an ABSOLUTE 1.0 -> must be a real year of EBITDA
    'netDebtToEBITDA':          ('flow_den', 'annualize'),
}


def stage1_flow_factor(key, rpy):
    """Multiplicative factor to apply to a Stage-1 ratio for this source's frequency.
    1.0 when the key needs no correction (or the source is quarterly and the mode is
    per-quarter)."""
    spec = STAGE1_FLOW_CORRECTION.get(key)
    if spec is None:
        return 1.0
    leg, mode = spec
    f = per_quarter_factor(rpy) if mode == 'per_quarter' else annualize_factor(rpy)
    return f if leg == 'flow_num' else 1.0 / f


# =========================================================================== #
#  ROW ORDER: ONE CONTRACT, STATED AT THE BOUNDARY                             #
# =========================================================================== #
# WHY THIS SECTION EXISTS.  Every recency window in this pipeline is positional --
# `head(n)`, `iloc[0]`, `iloc[lag]`, `pct_change(-1)`, `shift(-rpy)` -- so a frame's ROW
# ORDER is as load-bearing as its window LENGTH, and it was being REMEMBERED at each call
# site rather than stated once.  Two shipped defects came from exactly that:
#   * `postBo.moatIdentifier` read each name's OLDEST rows, because cdx_df arrives
#     OLDEST-first (data_quality sorts ascending; 7,752 of 7,752 sources) and nothing
#     re-sorted before `head(n)`.  Median window lag from the newest filing was 1.00 year
#     for a quarterly filer and 7.00 years for a semi-annual one once `rpy` scaling
#     shortened the window; moatScore differed on 50.2% of 400 real names, by up to +-7
#     points on an 0-11 scale.
#   * two `postBoWrapper` log blocks printed the OLDEST three rows under the label
#     "first 3" -- ~2020 rows presented as the newest data in the line the CEO reads to
#     judge whether the fetch picked up the current quarter.
#
# THE TWO POLICIES ARE BOTH CORRECT, FOR DIFFERENT CALLERS -- so the policy is a REQUIRED
# argument with NO DEFAULT.  That is the whole design decision here: a single permissive
# helper everywhere would be wrong, and a single strict helper everywhere was wrong (review
# L4 -- it let a LOG LINE abort Stage-2 and every deliverable of a 12-hour fetch).
#   ON_BAD_DATE_RAISE  -- FORENSICS.  `detectManipulation` / `forensicFlags` back the
#       Beneish/Montier year-over-year shifts off this order; an unparseable date there
#       must FAIL LOUDLY rather than sort as NaT and silently mis-pair two years.
#   ON_BAD_DATE_COERCE -- DIAGNOSTICS.  A print-only sample must never be able to cost the
#       run, so it coerces (NaT sorted LAST) and degrades to "possibly mis-ordered".
#
# There is exactly ONE strict sort implementation in the pipeline and it is
# `detectManipulation._toNewestFirst`; the strict branch below DELEGATES to it rather than
# re-implementing it, so the two can never drift.  The import is deliberately LAZY:
# detectManipulation imports this module, so a top-level import would be circular.
NEWEST_FIRST = 'newest_first'          # row 0 = the MOST RECENT period
OLDEST_FIRST = 'oldest_first'          # row 0 = the OLDEST period (how cdx_df is stored)

ON_BAD_DATE_RAISE = 'raise'
ON_BAD_DATE_COERCE = 'coerce'
_BAD_DATE_POLICIES = (ON_BAD_DATE_RAISE, ON_BAD_DATE_COERCE)

#  Attribute the row-order contract is stamped under, so a frame can DECLARE its order
#  instead of a reader having to know it.  Measured on pandas 2.3.1: `df.attrs` is PRESERVED
#  by copy / pickle / head / boolean mask / reset_index / set_index / assign / drop / apply,
#  and DROPPED by merge and by concat whose inputs' attrs differ.  So an absent stamp means
#  "not declared" -- never "oldest-first".  The stamp is a convenience for checking, not the
#  guarantee; the guarantee is the sort at the boundary.
ROW_ORDER_ATTR = 'va_row_order'


def to_newest_first(df, on_bad_date, date_col='date'):
    """A per-source frame re-sorted NEWEST-FIRST (row 0 = most recent period).

    `on_bad_date` is REQUIRED and has no default -- see the section note.  Pass
    ON_BAD_DATE_RAISE for anything a wrong pairing would corrupt (the forensic YoY
    shifts), ON_BAD_DATE_COERCE for anything that only prints.

    Returns a NEW frame with a reset index; the input is never mutated.  The result is
    stamped with its order (ROW_ORDER_ATTR) so a downstream reader can assert it rather
    than assume it.
    """
    if on_bad_date not in _BAD_DATE_POLICIES:
        raise ValueError(
            'to_newest_first: on_bad_date must be %r (forensics -- an unparseable date '
            'MUST fail) or %r (diagnostics -- a print must never abort the run), got %r. '
            'There is deliberately NO default: the two callers need different answers.'
            % (ON_BAD_DATE_RAISE, ON_BAD_DATE_COERCE, on_bad_date))
    if on_bad_date == ON_BAD_DATE_RAISE:
        # ONE strict implementation for the whole pipeline (see the section note).
        from detectManipulation import _toNewestFirst
        out = _toNewestFirst(df) if date_col == 'date' else (
            df.sort_values(date_col, key=lambda s: pd.to_datetime(s), ascending=False)
              .reset_index(drop=True))
    else:
        d = df.copy()
        _k = '_rp_order_key'
        d[_k] = pd.to_datetime(d[date_col], errors='coerce')
        out = (d.sort_values(_k, ascending=False, na_position='last')
                .drop(columns=[_k]).reset_index(drop=True))
    return stamp_row_order(out, NEWEST_FIRST)


def stamp_row_order(df, order):
    """Declare `df`'s row order on the frame itself.  Returns the same frame."""
    if order not in (NEWEST_FIRST, OLDEST_FIRST):
        raise ValueError('stamp_row_order: unknown order %r' % (order,))
    try:
        df.attrs[ROW_ORDER_ATTR] = order
    except Exception:                   # a stamp must never be able to break a caller
        pass
    return df


def row_order_of(df):
    """The declared row order, or None when the frame does not declare one.

    None means UNDECLARED -- never "oldest-first".  `attrs` is dropped by concat/merge, so
    a missing stamp is uninformative and a caller must not read a verdict into it.
    """
    try:
        v = df.attrs.get(ROW_ORDER_ATTR)
    except Exception:
        return None
    return v if v in (NEWEST_FIRST, OLDEST_FIRST) else None


def newest_first_violations(df, date_col='date', by=None):
    """How many rows sit OUT OF newest-first order -- a CHECK, not a re-sort.

    Non-mutating and non-raising by construction: it coerces for its own comparison and
    ignores unparseable dates, because its job is to report on the frame it was handed,
    not to gate it.  `by` groups the check (e.g. 'source') so a multi-source panel is
    checked per source.  Returns (n_violations, n_compared).
    """
    try:
        if df is None or date_col not in getattr(df, 'columns', []):
            return 0, 0
        d = pd.to_datetime(df[date_col], errors='coerce')
        groups = ([d] if by is None or by not in df.columns
                  else [g for _k, g in d.groupby(df[by], sort=False)])
        bad = cmp = 0
        for g in groups:
            g = g.dropna()
            if len(g) < 2:
                continue
            cmp += len(g) - 1
            bad += int((g.diff().dropna() > pd.Timedelta(0)).sum())
        return bad, cmp
    except Exception:
        return 0, 0


def assert_newest_first(df, label, date_col='date', by=None, verbose=True):
    """Report (never raise) when a frame that is SUPPOSED to be newest-first is not.

    Deliberately a REPORT and not an exception: this is called on the production scoring
    path, and the pipeline's own standing rule is that a check on a print/diagnostic path
    must not be able to cost a 12-hour run.  The loud line is what converts "row order is
    remembered" into "row order is observed".  Returns True when the frame is in order.
    """
    bad, cmp = newest_first_violations(df, date_col=date_col, by=by)
    if bad and verbose:
        print('!' * 78, flush=True)
        print('!!! ROW ORDER: %s is NOT newest-first -- %d of %d adjacent row pair(s) go '
              'FORWARD in time. Every head(n)/iloc[0] window on this frame is reading the '
              'WRONG END (this is the moatIdentifier / "first 3" defect class).'
              % (label, bad, cmp), flush=True)
        print('!' * 78, flush=True)
    return bad == 0


def describe_counts(freq_map, source_of_truth='cadence'):
    """One-line classification summary for the run log."""
    n_q = sum(1 for v in freq_map.values() if v == QUARTERLY)
    n_s = sum(1 for v in freq_map.values() if v == SEMIANNUAL)
    n_u = sum(1 for v in freq_map.values() if v == UNKNOWN)
    tot = max(1, len(freq_map))
    return ("REPORTING FREQUENCY (via %s): quarterly=%d (%.1f%%), semiannual=%d (%.1f%%), "
            "unknown=%d (%.1f%%) -> unknown is scored on the QUARTERLY path (unchanged)"
            % (source_of_truth, n_q, 100.0 * n_q / tot, n_s, 100.0 * n_s / tot,
               n_u, 100.0 * n_u / tot))
