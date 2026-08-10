"""adhoc_penalty.py  --  THE AD-HOC PENALTY BUCKET: A SOFT VETO FOR DATA PROBLEMS
(CEO, 2026-08-10)

WHAT IT IS
----------
A per-source accumulator of small penalty POINTS, each one recording a place where the
pipeline COULD NOT JUDGE a name because the data was missing or internally inconsistent.
The points are converted to a Stage-2 score penalty by ONE fixed weight and SUBTRACTED
from `AggScore` BEFORE the sort/rank, so a name we could not fully evaluate ranks below an
otherwise identical name we could.

    penalty (AggScore units) = -WEIGHT * (total points for this source)

THE CEO'S DESIGN, IN HIS WORDS: *"carry ad-hoc points into stage 2 based on these type of
things. So if a company gets past this interest-coverage veto because of data problems, we
should set some small minus number in the ad-hoc scoring bucket, which is then added (which
in effect lowers the score) before sorting/ranking takes place. And this can be like a
soft-veto system to catch data problems that might not even be the companies fault."*

WHY A SOFT VETO AND NOT A HARDER GATE.  `stage1_veto` is ABSOLUTE -- one persistent red flag
removes the name from the deliverable outright -- so it can only act where the evidence is
sufficient to make that call, and it ABSTAINS otherwise.  Abstention is the hole: 372 of the
1,388 general names on the 2026-08-10 panel passed `uInterestCoverage` UNCHECKED, and 303 of
those had between one and seven usable rows out of eight, i.e. the field is computable on
them and the vendor simply did not report it consistently.  Those names were previously
INDISTINGUISHABLE from names that passed the flag on a full window.  They now carry points.

THE WEIGHT IS FIXED AT 0.01 AND IS NOT PROPORTIONAL -- THIS IS A RULING, NOT A DEFAULT
-------------------------------------------------------------------------------------
CEO, 2026-08-10: *"we will add to this bucket every time data is missing and thus scaling is
there... We don't only have to think about weights, we have to think about the amount of
lowering we are doing. Just set the weight to 0.01. We will do the scaling in the amount
mostly."*

So the SCALING LIVES IN THE AMOUNT.  Each distinct data-gap event adds points; the weight
only converts points to score.  A name with five separate data problems is penalised five
times over and nobody touches the weight.  DO NOT make the weight proportional to anything,
and do not "tune" it in place of adding or removing a contributing check -- the check is the
decision, the weight is a unit conversion.

IT IS NOT A MEMBER OF THE STAGE-2 WEIGHT VECTOR, AND MUST NEVER BECOME ONE
--------------------------------------------------------------------------
`scoringWeights.DEPLOYED` enforces `Sigma|w| = 1.000000` at import, and every published
AggScore range and the presentation chip rest on it.  This bucket is applied OUTSIDE that
normalised weighted-z sum: `postBoRank.postBoScoreRanking` writes it into its own column
(`PENALTY_COLUMN`) after the metric columns have been weighted, and `getAggScore` sums it in
along with them.  Consequences that follow, all deliberate:

  * the metric weights are untouched, so `Sigma|w|` is still exactly 1 (asserted by test);
  * the penalty is in ABSOLUTE AggScore units, not in sigmas of anything -- it does not
    move when the pool's dispersion moves, which is what makes 0.01 mean one thing;
  * it is EXCLUDED from `rankOfRanks_diag` (`postBoRank.ROR_EXCLUDE`), which is an
    equal-weight METRIC lens; a penalty is not a metric and ranking it there would give a
    0.01 nudge the same say as a whole metric column.

SCALE, MEASURED AGAINST THE SCORE GEOMETRY RATHER THAN ASSERTED.  On the shipped 2026-08-10
general top-100 the median-to-rank-20 AggScore distance is **0.3327**.  So one point (0.01) is
3.0% of the distance from the middle of the pool to the shortlist boundary -- decisive for a
near-tie, never decisive on its own -- and the largest bucket a single Stage-1 veto flag can
produce (seven refused rows of eight) is 0.07, 21% of it.  That is the intended shape of a
SOFT veto: it moves a name, it does not remove one.
MEASURED CONSEQUENCE on that same top-100, under the ROW-LEVEL corroborator: 34 of the 100
carry a penalty (total -0.94 AggScore, worst -0.08, mean -0.028), 60 of the 100 change
position, the largest single move is 10 places, and the TOP-20 deliverable changes by ONE name
(`WSE` out, `EIN.DE` in).

MONOTONICITY IS THE PROPERTY TO PROTECT.  The charge must never fall as the data gets worse.
The first version of this bucket broke that -- it read the SHAPE of the refusals and gave a
FULLY refused window a free pass -- so the charge climbed to -0.07 at seven missing rows of
eight and dropped to 0.00 at eight of eight.  `test_stage1_veto` now walks all eight lengths
and asserts the charge strictly increases.

EVERY CONTRIBUTION IS SELF-DESCRIBING, WHICH IS HALF THE POINT
--------------------------------------------------------------
A contribution carries `(source, check, reason, points)`.  `check` names the mechanism that
raised it and `reason` says what was wrong with THIS source, so a penalised name can be
explained from the shipped CSV without re-running anything.  This is the ONE place a future
"we could not judge this" case goes: add a `check`, not a new gate.  A second bespoke gate
per data defect is how a pipeline acquires ten thresholds nobody can reconcile.

WHAT IT IS NOT.  It is not `stage1_veto` (that ejects, this ranks), it is not
`postBo.psbrfilter` (an inert -1.5 cutoff on `z x w` columns), and it is not a metric: it
has no sign convention to get wrong, no z-score, no pool dependence.  Points are always
POSITIVE and the penalty is always NEGATIVE.
"""

import pandas as pd


#  THE FIXED WEIGHT (CEO, 2026-08-10).  Points -> AggScore.  NOT proportional to anything;
#  see the module docstring for why the scaling lives in the AMOUNT instead.
WEIGHT = 0.01

#  The Stage-2 column the penalty travels in.  It is NOT a metric key and must never appear
#  in `scoringWeights.METRIC_KEYS`; `postBoRank` adds it after weighting and excludes it from
#  the rank-of-ranks diagnostic.
PENALTY_COLUMN = 'adhocPenalty'

#  Columns of the itemised evidence frame, fixed here so the CSV schema is one thing.
ITEM_COLUMNS = ('pool', 'source', 'check', 'reason', 'points', 'penalty')


class PenaltyBook:
    """An accumulator of ad-hoc penalty points, itemised per (source, check).

    Deliberately a plain object passed by the caller rather than a module-level global.
    `postBoScoreRanking` runs ONCE PER POOL (general + five carve-out cohorts) and each pool
    scores a different name set, so a run-scoped global would have to be filtered per pool at
    every read -- and a filter that is forgotten reads as "no data problems here".  One book
    per run, handed to each pool explicitly, cannot be forgotten silently: a pool given no
    book scores a penalty of exactly 0 and SAYS SO in its banner.
    """

    def __init__(self):
        self._items = []
        #  Places the bucket COULD NOT LOOK.  Kept apart from `_items` because they carry ZERO
        #  points by definition and `add` refuses a zero contribution -- but they must still
        #  reach the shipped CSV, or a run whose corroborator was unreachable is indeceivably
        #  identical to a clean one (reviewer R3).
        self._unmeasured = []

    def add(self, source, check, reason, points, pool=None):
        """Record `points` of penalty against `source`, attributed to `check`.

        `points` must be > 0 -- a zero-point contribution is not a finding, it is noise in
        the evidence file, and a NEGATIVE one would be a REWARD for missing data, which is
        the defect this whole layer exists to remove.  Both are refused loudly rather than
        clamped, because a caller computing a negative amount has a sign error the caller
        needs to see.
        """
        pts = float(points)
        if not (pts > 0):
            raise ValueError(
                'adhoc_penalty: refusing a contribution of %r points for %r/%r. Points are a '
                'severity COUNT and must be > 0 -- a negative amount would REWARD missing '
                'data, and a zero one is not a finding.' % (points, source, check))
        self._items.append({'pool': pool, 'source': source, 'check': str(check),
                            'reason': str(reason), 'points': pts,
                            'penalty': -WEIGHT * pts})
        return self

    def declare_unmeasured(self, check, reason, pool=None):
        """Record a place the bucket COULD NOT JUDGE -- zero points, but not silence.

        THE FAILURE THIS EXISTS FOR: `adhoc_penalty_uncorroborated` reached stdout and the
        saved `resdic`, and NOT the dated CSV -- which is the artifact a reader actually opens.
        So a run whose corroborating column was unreachable produced a CSV byte-indistinguishable
        from a run on a clean panel.  "Charged nothing" and "could not look" are different
        facts, and only one of them is good news.
        """
        self._unmeasured.append({'pool': pool, 'source': '', 'check': str(check),
                                 'reason': str(reason), 'points': 0.0, 'penalty': 0.0})
        return self

    @property
    def unmeasured(self):
        return list(self._unmeasured)

    def __len__(self):
        return len(self._items)

    @property
    def sources(self):
        return {it['source'] for it in self._items}

    def points_by_source(self):
        """{source: total points}.  Only sources with a contribution appear."""
        out = {}
        for it in self._items:
            out[it['source']] = out.get(it['source'], 0.0) + it['points']
        return out

    def penalty_by_source(self):
        """{source: the NEGATIVE AggScore penalty} = -WEIGHT * points.

        Negative by construction: a caller that ADDS this to a score lowers it, which is the
        CEO's stated shape ("added (which in effect lowers the score)").  There is no branch
        anywhere that flips the sign, so a penalty can never become a bonus.
        """
        return {s: -WEIGHT * p for s, p in self.points_by_source().items()}

    def penalty_series(self, sources):
        """A float Series over `sources` (0.0 where a source has no contribution).

        Positionally aligned to the sequence given, so the caller can assign it straight
        onto a frame's column without an index join -- the Stage-2 frames are row-aligned by
        construction and an index join is precisely what has silently mis-paired frames in
        this pipeline before.
        """
        pen = self.penalty_by_source()
        return pd.Series([float(pen.get(s, 0.0)) for s in sources], dtype='float64')

    def itemised(self, pool=None):
        """The contributions as a DataFrame, one row per (source, check).

        `pool` filters to one pool's contributions; None returns everything.  Sorted by
        total points DESCENDING per source so the worst-affected names read first, then by
        source and check for a stable, diffable file.
        """
        items = self._items if pool is None else [i for i in self._items
                                                  if i['pool'] == pool]
        if not items:
            return pd.DataFrame(columns=list(ITEM_COLUMNS))
        df = pd.DataFrame(items)[list(ITEM_COLUMNS)]
        tot = df.groupby('source')['points'].transform('sum')
        return (df.assign(_tot=tot)
                  .sort_values(['_tot', 'source', 'check'],
                               ascending=[False, True, True], kind='mergesort')
                  .drop(columns='_tot').reset_index(drop=True))

    def summary(self, pool=None):
        """{source: total points} plus a compact reason string -- for a log line."""
        df = self.itemised(pool)
        if df.empty:
            return {}
        return {s: (float(g['points'].sum()), '; '.join(g['check'].tolist()))
                for s, g in df.groupby('source')}


def write_evidence_csv(book, path=None, run_date=None):
    """Ship the bucket as evidence: one dated CSV naming each penalised source, its total
    points and penalty, and the itemised reasons.

    WRITTEN AT THE REPO ROOT, not into `output/` (CEO, 2026-08-10): the root-level artifacts
    from the 2026-08-10 run all reached Drive and `output/` did not, so a new evidence file
    placed in `output/` would be a record nobody receives.  See `Sbocker.allowlist_patterns`,
    which carries the matching top-level glob.

    Returns the path written, or None (never raises -- an evidence file must not be able to
    cost a scored run).
    """
    try:
        if path is None:
            stamp = run_date or pd.Timestamp.today().strftime('%Y-%m-%d')
            path = 'AdHocPenaltyBucket_%s.csv' % stamp
        df = book.itemised()
        unmeasured = pd.DataFrame(book.unmeasured)
        #  The per-source TOTAL rides in the same file rather than in a second one: the CEO
        #  reads "which names were penalised and by how much" and "why" together, and two
        #  files is how the second one goes unread.  A `section` column separates them, the
        #  same idiom as MissingDataFillReport.
        if df.empty:
            out = pd.DataFrame(columns=list(ITEM_COLUMNS) + ['section'])
        else:
            tot = (df.groupby('source', as_index=False)
                     .agg(points=('points', 'sum'), penalty=('penalty', 'sum'),
                          check=('check', lambda s: '%d check(s)' % len(s)),
                          reason=('reason', lambda s: ' | '.join(s))))
            tot['pool'] = ''
            out = pd.concat([tot.assign(section='per_source')[
                                 list(ITEM_COLUMNS) + ['section']],
                             df.assign(section='per_item')],
                            ignore_index=True, sort=False)
        #  THE THIRD SECTION, AND IT IS WRITTEN EVEN WHEN THE OTHER TWO ARE EMPTY (reviewer
        #  R3).  It goes FIRST so a reader meets the caveat before the numbers it qualifies:
        #  a bucket total means one thing on a fully-measured run and another on a run where
        #  a corroborator could not be reached.
        if len(unmeasured):
            out = pd.concat([unmeasured.assign(section='NOT_MEASURED')[
                                 list(ITEM_COLUMNS) + ['section']], out],
                            ignore_index=True, sort=False)
        out.to_csv(path, index=False)
        return path
    except Exception as _e:
        print('WARNING: could not write the ad-hoc penalty evidence CSV (%s: %s)'
              % (type(_e).__name__, _e), flush=True)
        return None
