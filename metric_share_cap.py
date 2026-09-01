"""metric_share_cap.py  --  THE SINGLE-METRIC SHARE CAP (CEO, 2026-08-31)

WHAT IT IS
----------
A per-name truncation applied to the WEIGHTED metric contributions (`z x w`) after the
weighting step and before `getAggScore`, so that no single metric may account for more
than `CAP` of a name's total ABSOLUTE contribution.  A name whose score is dominated by
one column has that column's contribution cut back and KEEPS COMPETING on the reduced
score; it is never removed.

    contribution_ic = z_ic * w_c            (signed; this IS the AggScore addend)
    base_i          = sum_c |contribution_ic|
    share_ic        = |contribution_ic| / base_i
    ... and after this module runs, max_c share_ic <= CAP for every name i.

THE CEO'S RULING, AND WHAT IT RESTS ON -- READ THIS BEFORE MOVING ANY NUMBER HERE
---------------------------------------------------------------------------------
*** THIS CHANGE IS NOT EVIDENCE-DRIVEN, AND NO BACKTEST CAN VALIDATE OR REFUTE IT. ***

A dedicated analysis (`design/losers-vs-beaters-2026-08-31.md`) concluded: CHANGE NO
WEIGHTS ON THIS EVIDENCE.  42 measured pick-rows in a single macro regime, a null AUC band
of [0.28, 0.72], nothing surviving Benjamini-Hochberg, and the flagged configuration barely
present in the graded sample at all (the shipped top-20's median `earnYield` share is +27.9%
against the backtest sample's -2.8%).  The CEO overrode that conclusion DELIBERATELY, on
MECHANISM rather than statistics:

    a high earnings yield on a cyclical at the top of a freight cycle is peak-cycle
    earnings capitalised as if permanent -- the textbook value trap -- and the backtest
    cannot see it, because the sample contains almost none of that configuration.

So the beat-rate CANNOT be read as evidence for or against this cap, now or after any
future run.  If a later beat-rate moves, that movement is about the panel, not about this.
Do not let a report cite a beat-rate delta as a verdict on the cap, in either direction,
and do not "validate" the cap by backtesting it -- the sample that would have to carry the
test is the sample the analysis already showed is not there.  What this rests on is the
mechanism argument above.  Argue with THAT if you want to change it.

WHY A CAP ON THE CONTRIBUTION AND NOT AN EJECTION
-------------------------------------------------
"Cap any single metric's share of a name's score" was offered to, and ruled on by, the CEO
as a cap on the CONTRIBUTION.  The alternative reading -- eject any name whose top metric
exceeds the cap -- deletes names outright and was NOT what was put to him: on the
2026-08-31 panel it would have removed 13 of the 97 ranked names and 4 of the top-20 for a
property none of them can control.  Truncation keeps the name in the ranking on a score
that no longer rests on one column, which is the stated intent.

THE CAP IS A FIXED POINT, NOT A ONE-PASS CUT, AND THAT IS THE WHOLE DIFFERENCE
------------------------------------------------------------------------------
Truncating a contribution SHRINKS the base it is a share of.  A naive one-pass cut -- set
the offending contribution to `CAP * base_before` -- therefore leaves that metric ABOVE
`CAP` of the base that actually ships: on this panel HAFN's `earnYield` would read 0.2500
of the old base and 0.2650 of the new one.  A guard whose own post-condition is violated by
its own output is the exact failure mode this project has filed fifteen times (a test that
pins the defect it covers, a guard blind to the defect beneath it), so it is not built that
way.

The fixed point is closed-form.  With `S` the set of capped metrics (|S| = k) and `R` the
sum of the ABSOLUTE contributions of the metrics NOT in S:

    every capped metric lands at   c = CAP * R / (1 - k*CAP)
    the post-cap base is           T = R / (1 - k*CAP)
    so every capped metric's post-cap share is EXACTLY CAP, by construction.

`k` is found by scanning k = 1, 2, ... over the absolute contributions sorted descending
and taking the first k for which the resulting `c` sits strictly below the k-th largest and
at or above the (k+1)-th -- the standard water-filling consistency test.  `k` is bounded by
`_k_max(cap)` -- the largest k with `1 - k*CAP > 0`, i.e. 3 at CAP = 0.25 -- because beyond
that the denominator is non-positive.  (An earlier version of this sentence said the bound
was `floor(1/CAP) - 1` "which is asserted".  Both halves were wrong: at CAP = 0.3 the code
allows k = 3 and 3 is legitimate -- three metrics at 30% each is 90% with a 10% tail -- and
there is no assertion, only a loop bound.)  `_assert_post_condition` then re-derives the
shares from the returned frame and
raises if any exceeds the cap: the module checks its own output rather than trusting the
algebra above.

A THIN TAIL CASCADES, AND THAT IS THE FIXED POINT RATHER THAN A DEFECT.  `k` is often
larger than "the number of metrics that were over the cap to begin with": truncating the
dominant metric shrinks the total, which can push the SECOND metric over the cap even
though it started below it.  A row of [0.9, 0.05, 0.03, 0.01, 0.01] resolves at k = 3, not
k = 1, and the name's total falls from 1.00 to 0.08.  That reads as violent and is correct:
a name that is 90% one metric carries almost no other information, so once that metric is
refused the deciding vote there is almost nothing left to score it on.  NOT REACHED ON THE
2026-08-31 PANEL -- all 13 bound names there resolve at k = 1, the worst share being 0.4572
(`JEN.DE`).

*** THAT IS A FACT ABOUT ONE PANEL, NOT A PROPERTY OF THE CAP.  The earlier wording here
("NOT REACHED ON REAL DATA") stated it as the latter, and that is the sentence that stops
anyone looking for the effect in a run log. ***  An independent review (2026-08-31)
simulated the SHIPPED weight vector and the real `squash(z, k=3)` over 500 panels x 97
names: k >= 2 lands on 0.02-0.09% of names under dense draws, and on 1.77% -- about 1.7
names per 97-name panel -- once 20% of cells are imputed to z = 0.  That last one is the
LIVE regime: `IMPUTED_EXCLUDE_AT = 0.20` (postBoRank) ships names with up to a fifth of
their weight imputed, and `fillna(0)` makes an imputed cell contribute EXACTLY zero, which
thins the tail the fixed point needs.  Worst case there erased 62% of a name's absolute
mass and moved its AggScore by 0.303 -- the size of the entire median->rank-20 distance
this docstring quotes elsewhere (~0.31).  (The simulation drew metrics independently; real
metrics correlate and correlation spreads the base, so 1.77% is probably an over-estimate.
It is not an over-estimate of zero.)

So a cascade IS reachable, and the log must make it visible: `format_report` prints
`n_capped` on every hit line, because one metric name beside a single share pair reads as a
one-metric trim whether it was one metric or three.  `test_a_thin_tail_CASCADES_...` pins
the behaviour so a future editor does not "fix" it into a one-pass cut.

THE SHARE IS OF THE ABSOLUTE CONTRIBUTION.  THAT IS A DECISION AND IT CHANGES THE BITE
--------------------------------------------------------------------------------------
Metric contributions are SIGNED (`CycleHeat` is negative by design, and any metric is
negative for a below-median name), so "share of a name's score" has two readings and they
are not close:

  (i)  |c| / sum|c|   -- share of the total absolute contribution.  BUILT.
  (ii)   c / AggScore -- share of the signed total.  NOT BUILT, and not buildable.

(ii) is what the 2026-08-31 run review reported ("earnYield contributes 0.314 of TNK's 1.006
AggScore = 31%") and it is what the CEO was shown when he set 25%.  It cannot be the rule,
and the reason is measured, not stylistic: on the 2026-08-31 general pool **37 of the 97
ranked names have AggScore <= 0**, so (ii) is a share of a negative or near-zero number.
It goes negative (RMV.L's `earnYield` reads -6.3%, META's -7.7%, LNTH's -11.4%), and for a
name scoring near zero it diverges.  A cap defined on (ii) would do something arbitrary on
38% of the pool.  (i) is bounded in [0, 1] for every name WHOSE BASE IS POSITIVE AND
FINITE, which is why the small/negative-total question, raised as a hazard when this was
specified, does not arise here.

IT IS NEITHER BOUNDED NOR DEFINED FOR FOUR DEGENERATE ROWS.  An earlier version of this
paragraph said the all-zero row was "the only degenerate case left".  It was wrong on three
counts, and every one of the ones it missed is reachable:

*** THE RULE FOR ALL FOUR: THE ROW SHIPS UNCAPPED AND IS NAMED IN THE LOG.  Nothing is
zeroed, nothing is ejected, and nothing raises.  A name this module cannot cap keeps the
score it would have had if this module did not exist -- which fabricates nothing, promotes
nothing and hides nothing -- and `format_report` says so under a block as loud as the one
that reports the names it did cap.  See `_assert_post_condition` for why NOT raising is the
right answer, and for the reasoning error that briefly made it the wrong one. ***

  ALL-ZERO (base = 0).  There is nothing to be a share of.  `status = 'all_zero'`, every
  share reported as NaN.

  INFEASIBLE (at most `_k_max(cap)` non-zero contributions -- at CAP = 0.25 that is three or
  fewer).  With n non-zero metrics the smallest achievable maximum share is 1/n, so when the
  scan can cap EVERY non-zero metric there is no tail left to be a share of and the fixed
  point degenerates to c = 0.  Committing that would not be capping the name; it would be
  DELETING its score and putting 0.0 in its place -- and on a pool where 37 of 97 names
  score at or below zero, 0.0 is a large PROMOTION invented by the guard that exists to stop
  one metric deciding a name.  So the truncation is not committed: `status = 'infeasible'`.

  WOULD-ERASE (feasible by count, but the fixed point would leave less than `_MASS_FLOOR` of
  the name's absolute mass).  THE COUNT WAS THE WRONG AXIS, AND THE FIRST VERSION OF THIS
  GUARD USED ONLY THE COUNT.  `[0.30, 0.10, 0.05, 1e-9]` has four non-zero metrics, so it
  passes any count test, and the cap still truncated all four to 1e-9: AggScore 0.450 ->
  4e-09, reported `status = 'ok'`, `share_after = 0.2500`, post-condition clean.  That is the
  same invented promotion as the infeasible case, one epsilon outside the case the count
  could see.  The harm was never "too few metrics", it was "the cap replaced a real score
  with a fabricated zero", so the condition is now stated on the MASS.
  `status = 'would_erase'`.

  NON-FINITE (any NaN or inf contribution).  The share is undefined for the entire row --
  for an `inf` as much as for a NaN -- so the row is skipped.  That is a real hole: a name
  at 90% on one metric passes straight through it.  It is COUNTED and printed rather than
  left silent.  It WAS silent: `n_capped = 0`, `share_before = -1.0`, and no log line at
  all, which reads identically to "the cap looked at this name and nothing bound".  That is
  precisely the confusion `format_report`'s own docstring says it exists to prevent,
  reproduced at the per-name level.

MEASURED, NOT HYPOTHETICAL -- BUT MEASURED OFF-UNIVERSE, AND THAT MATTERS.  Running this
module over the saved panels: `postRank_2026-08-11_fmp_stock_CUR3K` has TWO of 100 names
with fewer than four non-zero weighted contributions -- `STRT` (93% of |w| imputed to zero)
and `PET.TO` (95%) -- and `postRank_2026-08-07_..._CUR3K` has `STRT` again.  `STRT` appearing
on both says its sparsity is a property of that issuer's data coverage rather than of one
night's fetch.  The pre-fix cap set those names to an AggScore of exactly 0.0000 -- for
`PET.TO`, -0.0464 -> 0.0000, vaulting it over the 40 names in that pool scoring at or below
zero -- and the post-condition passed both without a word.

*** BOTH PANELS ARE CUR3K.  THE LIVE UNIVERSE IS CUR6K (4,941 sources).  An earlier version
of this paragraph called CUR3K "the CURRENT universe"; it is not, and every sparse-name and
mass-floor number in this file is therefore drawn from a DIFFERENT and SMALLER universe than
the one that ships.  Read them as a LOWER BOUND on the shape's incidence -- evidence that it
occurs and roughly what it looks like -- and NOT as a prediction of tonight's rate, which
nobody here has measured.  A larger universe reaches further down the coverage distribution,
so the honest expectation is more of these names, not fewer. ***

*** DO NOT WRITE THAT THE IMPUTATION LADDER PROTECTS AGAINST THIS.  IT DOES NOT, AND A
PREVIOUS VERSION OF THIS DOCSTRING SAID IT DID. ***  `imputation_ladder` is CALLED at
`postBoRank.py:218`, but that call only COMPUTES the exclusion set; the REMOVAL happens at
`postBoRank.py:340-343`, on `postRank`, after `getAggScore` at `:305` -- eighty-three lines
AFTER the cap runs at `:257`.  The frame this module sees is the full pre-dedup,
pre-exclusion pool, heavily-imputed names included.  The ladder cannot intercept anything
here, on any path, whether or not its fill report succeeded.

*** THE CONSEQUENCE, WHICH THE CEO HAS NOT YET SEEN AND MUST: 25% MEANS SOMETHING WEAKER ON
(i) THAN IT MEANT ON THE (ii) DISTRIBUTION HE CHOSE IT AGAINST.  Measured on the 2026-08-31
top-20: on (ii) 18 of 20 names have a metric at or above 25% (93 of 97 pool-wide); on (i)
only 4 of 20 do (13 of 97).  The cap level that reproduces on (i) the bite 25% has on (ii)
is roughly 0.16-0.20, not 0.25.  This is a LEVEL question for the CEO, not a licence for
anyone here to re-pick the number -- 0.25 is what he set and 0.25 is what ships. ***

THERE IS NO RENORMALISATION, AND THAT IS THE POINT
--------------------------------------------------
A capped name's AggScore FALLS by exactly the truncated amount.  Rescaling the name back to
its pre-cap total would reallocate weight inside the name and leave the total unchanged --
and since the total is what ranks, that would make the cap a no-op on the deliverable.  So
the score of a capped name is on a slightly compressed scale relative to an uncapped one's,
and that compression IS the penalty.

`Sigma|w| = 1.000000` IS UNTOUCHED.  This module never sees the weight vector; it acts on
CONTRIBUTIONS, exactly as `adhoc_penalty` does, and for the same reason (see that module's
"IT IS NOT A MEMBER OF THE STAGE-2 WEIGHT VECTOR").  `scoringWeights._validate()` still
asserts the vector sums to 1 at import.  What a reader must NOT conclude from that is that
every published AggScore is still a unit-weight sum: for the 11-13 capped names on this
panel the effective weight actually summed is below 1, and any consumer quoting an AggScore
RANGE across names should read the `n_capped` column rather than assume a common scale.

THE CAP IS SYMMETRIC, WHICH FORGIVES AS WELL AS PENALISES -- MEASURED, NOT ASSUMED
----------------------------------------------------------------------------------
The rule is stated on |contribution|, so it truncates a large NEGATIVE contribution too: a
name that is catastrophically bad on one metric has that penalty cut back and RISES.  On
the 2026-08-31 panel that is the single largest effect the cap has anywhere in the pool --
`BOSS.DE` climbs 15 places (rank 58 -> 43) because its `freeCashFlowPerShareGrowth` sits at
-38.1% of its absolute contribution, and `JEN.DE` climbs 9 on the same mechanism (-45.7%).
Four of the thirteen names the cap touches are helped by it, not penalised.

This is the honest reading of "no single metric decides a name" and it is what ships, but it
is NOT what the cap was aimed at, so it is stated here rather than discovered later.  The
alternative -- truncating only POSITIVE contributions, so the cap can stop a metric carrying
a name UP but never rescue one it drags DOWN -- was measured on the same panel: it binds on
10 names instead of 13, produces the SAME top-20, and its largest move is 4 places instead
of 15.  That is a CEO decision, not a developer's, and it is a three-line change here if he
wants it.

WHAT THIS DOES *NOT* FIX -- AND IT IS THE FINDING THAT PROMPTED THE WHOLE EXERCISE
----------------------------------------------------------------------------------
*** THE CAP DOES NOT CLEAR THE k-PROPERTY GUARD, AND MARGINALLY WORSENS IT. ***

The k-property (`postBoRank.single_column_reach_check`) asks whether one column can carry a
MEDIAN name into the top-20 -- an ABSOLUTE-MAGNITUDE question:

    max_c |w_c| x max_i |zeta_ic|   <   AggScore(rank 20) - AggScore(median)

A PROPORTIONAL cap does not bound an ABSOLUTE magnitude.  On 2026-08-31 the largest single
`earnYield` contribution anywhere in the pool is TNK's 0.3141 -- and TNK's `earnYield` share
is only 21.0%, because TNK's total absolute contribution is large.  The cap never touches
it.  The guard reads:

    shipped        reach 0.3141 vs distance 0.3115 = 1.01x  VIOLATES
    cap only       reach 0.3141 vs distance 0.3094 = 1.02x  VIOLATES  (slightly WORSE:
                   the cap compresses the top of the pool, lowering rank-20 more than the
                   median, so the distance the reach is measured against SHRINKS)
    weight shift   reach 0.2983 vs distance 0.3192 = 0.93x  OK
    both           reach 0.2983 vs distance 0.3140 = 0.95x  OK

So it is the `earnYield` weight cut, not the cap, that clears the guard -- and only by a 5-7%
margin on one panel, which is thin enough that it can flip back on the next.  If the k
property is the thing to fix, the instrument is a bound on |w_c| x max|zeta_ic| (a weight cut
or a tighter squash), not this.  Do not let this module be cited as the k-property's remedy.

SCOPE: THE GENERAL POOL ONLY, AND THE REASON IS STRUCTURAL
-----------------------------------------------------------
`postBoScoreRanking` runs once per pool, so this could apply to the five carve-out cohorts
too.  It deliberately does not.  The cohort vectors CONCENTRATE WEIGHT BY DESIGN -- FIN-1
puts 0.275 on `bVpRatio` and 0.275 on `tbVpRatio`, and `scoringWeights` section D.2 argues
that concentration as the cohort's thesis -- so a 25% cap would bind on essentially every
member of those cohorts, for a reason that has nothing to do with a value trap and
everything to do with a designed vector.  That is the "cap that binds on something
structural rather than on the shape it is aimed at" failure, and it is avoided by scoping.
The CEO's ruling was about the shipped top-20, which is the general pool.  Widening this to
a cohort is a design decision that has to re-argue that cohort's weight concentration first.
"""
import numpy as np
import pandas as pd

#  THE CAP.  CEO, 2026-08-31, set by him after being shown the top-20 share distribution.
#  Read the "(i) vs (ii)" section above before moving it: the number was chosen against the
#  signed-AggScore distribution and is applied to the absolute-contribution one, where it
#  bites on 4 of the top-20 rather than the ~18 that distribution showed.
CAP = 0.25

#  Only the general pool is capped -- see "SCOPE" above.  A pool label not in here is scored
#  exactly as it was before this module existed.
CAPPED_POOLS = frozenset({'general'})

#  Absolute tolerance for the post-condition check.  The fixed point is exact in real
#  arithmetic; this covers float64 round-off in the base sum only.
_SHARE_TOL = 1e-9



def _k_max(cap):
    """The largest number of metrics the fixed point may cap at once.

    THE SINGLE SOURCE OF TRUTH FOR THIS BOUND, and it is one function because it was two.
    `_cap_value` bounded its scan with `floor(1/cap - 1e-12)` while the feasibility test
    asked `n * cap >= 1 - 1e-12`, and at `cap = 1/3` the two disagreed: `3 * cap` is
    `0.9999999999999998` so a three-metric row was admitted as feasible, while `1/cap` is
    `3.0000000000000004` so the `- 1e-12` pulled the scan's bound down to 2 and the k = 3
    solution was unreachable.  The row then shipped uncapped and the post-condition raised
    "this is a defect in `_cap_value`" at a reader who had done nothing wrong.

    The bound is what the algebra requires and nothing more: `c = cap * R / (1 - k*cap)`
    needs `1 - k*cap > 0`, i.e. k strictly below `1/cap`.  Deriving both callers from this
    one expression is what makes them incapable of disagreeing.  The two OLD spellings agreed
    numerically everywhere measured (500+ cap levels x n in 1..39, zero divergent pairs), so
    this is hygiene against a FUTURE re-level rather than a live defect -- and note that it is
    a different thing from the exact-tie knife-edge, which IS live at CAP = 0.25 and is
    documented in `_assert_post_condition` -- but the module's own remedy text invites a
    re-level, so the trap was laid for a change we expect somebody to make.
    """
    return int(np.floor(1.0 / cap - 1e-12))


def _cap_value(abs_row, cap):
    """The fixed-point cap level for one name, or None if no metric exceeds `cap`.

    `abs_row` is the row's ABSOLUTE contributions (any order).  Returns the value every
    over-cap metric is truncated to, chosen so that after truncation each of them is
    EXACTLY `cap` of the new total.  See the module docstring for the derivation.
    """
    s = np.sort(abs_row)[::-1]
    if not np.isfinite(s).all() or s.sum() <= 0.0:
        return None
    #  k is bounded a priori; without this the scan could propose a denominator <= 0.
    k_max = _k_max(cap)
    for k in range(1, min(len(s), k_max) + 1):
        denom = 1.0 - k * cap
        if denom <= 0.0:                      # unreachable given k_max; kept as a hard stop
            break
        c = cap * s[k:].sum() / denom
        below = s[k] if k < len(s) else -np.inf
        #  consistency: the k-th largest must still be ABOVE the cap value (so it really is
        #  capped) and the (k+1)-th must be at or BELOW it (so it really is not).
        if c < s[k - 1] and c >= below:
            return float(c)
    return None


#  --- PER-NAME OUTCOMES -------------------------------------------------------------- #
#  Every name in the pool leaves `apply_share_cap` with exactly one of these in the
#  report's `status` column.  The point is that "the cap ran and nothing bound on this
#  name" and "the cap could not run on this name" are DIFFERENT values rather than the same
#  `n_capped = 0` -- which is the per-name form of the confusion `format_report` exists to
#  prevent at the panel level.
#
#  THE THREE UNCAPPABLE STATUSES ALL BEHAVE IDENTICALLY: the row is returned EXACTLY as it
#  arrived and is named in the run log.  They are separate values only so the log can say
#  WHICH kind of row it was, because the three have different remedies.
STATUS_OK = 'ok'                    # a share was defined; capped or not, it now obeys the cap
STATUS_ALL_ZERO = 'all_zero'        # base = 0; nothing to be a share of
STATUS_INFEASIBLE = 'infeasible'    # <= _k_max(cap) non-zero metrics; no tail to be a share of
STATUS_WOULD_ERASE = 'would_erase'  # feasible, but the fixed point would erase the name's mass
STATUS_NON_FINITE = 'non_finite'    # a NaN/inf contribution; the share is undefined
UNCAPPABLE = (STATUS_INFEASIBLE, STATUS_WOULD_ERASE, STATUS_NON_FINITE)

#  THE FRACTION OF ITS ABSOLUTE MASS A NAME MUST KEEP for the truncation to be committed.
#
#  THIS IS A FABRICATION DETECTOR, NOT A SCORING LEVEL, AND THE DISTINCTION IS THE WHOLE
#  JUSTIFICATION FOR PICKING A NUMBER HERE.  `CAP` is the CEO's; this is a developer guard,
#  and it is positioned so that it CANNOT change any outcome the module documents as correct:
#    * the cascade this file pins as correct and deliberate -- [0.9, 0.05, 0.03, 0.01, 0.01]
#      resolving at k = 3 -- leaves 8% of the mass, EIGHT TIMES this floor, so it is
#      untouched;
#    * across the five saved panels the worst surviving fraction any real name reaches is
#      0.56, FIFTY-SIX TIMES this floor;
#    * the shapes it exists to catch are three to eight ORDERS of magnitude below it
#      ([0.30, 0.10, 0.05, 1e-3] leaves 0.0089; the 1e-9 version leaves 8.9e-09).
#
#  SAID PLAINLY BECAUSE IT IS THE WEAKEST POINT IN THIS FILE: the band between this floor and
#  the pinned 8% cascade is a genuine judgment call that nobody has ruled on, and 0.01 is not
#  derived from anything -- it is chosen an order of magnitude clear of the nearest documented
#  behaviour on one side and orders of magnitude clear of the observed pathologies on the
#  other.  If this ever fires on a real name, the answer is to look at the NAME, not to lower
#  the floor.
#
#  AND IT MOVES THE HAZARD RATHER THAN REMOVING IT -- recorded, not argued with (third review,
#  2026-08-31).  A name at 99.0% on one metric is still FEASIBLE and still capped, down to
#  about 1.3% of its mass, which is above this floor and therefore committed.  The floor buys
#  roughly two orders of magnitude, not immunity: it converts "the cap can fabricate a zero"
#  into "the cap can fabricate a hundredth".  Whether THAT is acceptable is the same level
#  question as the 1%-to-8% band, and it is the CEO's, not this module's.  Measured margin on
#  the CUR3K panels: the thinnest surviving mass on a committed row is 0.8400 and 0.8058 --
#  84x and 81x this floor -- so nothing real is anywhere near it today.
_MASS_FLOOR = 0.01


def apply_share_cap(contrib, cap=CAP, sources=None):
    """Truncate every metric contribution that exceeds `cap` of its name's absolute total.

    `contrib` is a DataFrame of SIGNED weighted contributions (`z x w`), one column per
    metric, one row per name -- i.e. exactly `postBoRank`'s `temp_normpsmdf_weighted`, and
    NOTHING ELSE.  It must not contain the ad-hoc penalty column (which is not a metric and
    is not capped -- see the module docstring) or any identifier column.

    `sources` is the pool's ticker list in the SAME ROW ORDER as `contrib`.  Pass it: it
    becomes the report's `source` column, and `source` is the only thing the report can be
    joined on downstream (`postRank` is re-indexed by `getAggScore`, so position does not
    survive).  A report built without it cannot be joined at all.

    Returns `(capped, report)`:
      capped  a NEW DataFrame, same shape/columns/index, with over-cap contributions
              truncated in place and their SIGN preserved.
      report  one row per name, always the full pool: `source` (when given), `n_capped`
              (0 where nothing bound), `cap_value`, `share_before`, `share_after`,
              `metric_before`, `contrib_before` (that metric's SIGNED pre-cap contribution --
              the log's RAISED label is derived from ITS sign, not from `agg_delta`),
              `base_before`, `base_after`, `status`, and the `agg_delta` this cap costs.

    A share that is not defined is NaN, never a sentinel.  `share_before = -1.0` used to
    stand for "undefined" here; it printed to the run log as if it were a share, shipped
    into `rankdic['share_cap_report']`, and -- being below the cap as a number -- walked
    straight past the post-condition that exists to catch exactly that row.

    RAISES ALMOST NEVER, AND THE EXCEPTION IS NAMED RATHER THAN GLOSSED.  A row the cap
    cannot honour ships UNCAPPED with a `status` naming why, and `format_report` prints it.
    The one data condition that DOES still raise is a row whose contributions tie EXACTLY at
    the fixed point, where `_cap_value` finds no consistent k and the post-condition then
    fires.  That is measured-zero on this pipeline, not impossible -- see
    `_assert_post_condition`, which sets out the scope and why the raise is kept.
    """
    cols = list(contrib.columns)
    C = contrib.to_numpy(dtype='float64', copy=True)
    C0 = contrib.to_numpy(dtype='float64')          # the pre-cap frame, kept for agg_delta
    A = np.abs(C)
    #  `errstate` because a base sum can OVERFLOW on pathological input, and numpy would
    #  print "overflow encountered in reduce" to stderr beside a log block that explains the
    #  row in prose.  The inf is the intended signal -- it is what routes the row to
    #  STATUS_NON_FINITE -- so only the noise is suppressed, not the condition.
    with np.errstate(over='ignore', invalid='ignore'):
        base_before = A.sum(axis=1)
    n_capped = np.zeros(len(C), dtype='int64')
    cap_value = np.full(len(C), np.nan)
    status = np.array([STATUS_OK] * len(C), dtype=object)
    k_max = _k_max(cap)

    for i in range(len(C)):
        if not np.isfinite(A[i]).all() or not np.isfinite(base_before[i]):
            #  UNCAPPED, and that is a hole -- so it is recorded and printed, not skipped in
            #  silence.  A share cannot be formed at all when one addend of the base is NaN
            #  or infinite; guessing one would be worse than declaring the row un-assessed.
            #
            #  THE BASE IS TESTED SEPARATELY FROM THE CELLS, and it is not redundant: every
            #  cell can be finite while their SUM overflows to inf (five contributions of
            #  1e308 do it).  Such a row used to pass this check, keep `status = 'ok'`, and
            #  then trip the post-condition -- a data condition surfacing as an algebra
            #  assertion, which is the shape this module is trying to stop making.
            status[i] = STATUS_NON_FINITE
            continue
        if base_before[i] <= 0.0:
            status[i] = STATUS_ALL_ZERO
            continue
        if int((A[i] > 0.0).sum()) <= k_max:
            #  INFEASIBLE: the scan could cap every non-zero metric, leaving no tail, so the
            #  fixed point is c = 0 and committing it would ANNIHILATE the row.  Derived from
            #  `_k_max` rather than from a second tolerance of its own -- see that function.
            status[i] = STATUS_INFEASIBLE
            continue
        c = _cap_value(A[i], cap)
        if c is None:
            continue
        over = A[i] > c
        #  THE TRUNCATION IS NOT COMMITTED UNTIL THE MASS IS CHECKED.  Counting metrics
        #  catches the row with an EMPTY tail; it does not catch the row with a NEGLIGIBLE
        #  one, and both end with a fabricated ~0 AggScore.  Computed on a candidate so the
        #  row can be left exactly as it arrived.
        cand_base = float(np.where(over, c, A[i]).sum())
        if cand_base < _MASS_FLOOR * base_before[i]:
            status[i] = STATUS_WOULD_ERASE
            cap_value[i] = c            # what it WOULD have been, for the log
            continue
        n_capped[i] = int(over.sum())
        cap_value[i] = c
        C[i, over] = np.sign(C[i, over]) * c

    capped = pd.DataFrame(C, index=contrib.index, columns=cols)
    A_after = np.abs(C)
    with np.errstate(over='ignore', invalid='ignore'):
        base_after = A_after.sum(axis=1)

    #  A share exists only where the base is positive AND finite.  Everywhere else it is
    #  NaN -- see the docstring above on why the old -1.0 was not a smaller version of this.
    defined_before = np.isfinite(base_before) & (base_before > 0.0)
    defined_after = np.isfinite(base_after) & (base_after > 0.0)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        sh_before = A / np.where(defined_before, base_before, np.nan)[:, None]
        sh_after = A_after / np.where(defined_after, base_after, np.nan)[:, None]
        #  `errstate` covers the subtraction too: on a non-finite row this is inf - inf,
        #  which numpy reports as "invalid value encountered in subtract" -- an unexplained
        #  warning printed to stderr right beside the log block that has just explained the
        #  skip in prose.  The NaN result is correct and intended; only the noise is not.
        agg_delta = C.sum(axis=1) - C0.sum(axis=1)

    share_before = np.full(len(C), np.nan)
    share_after = np.full(len(C), np.nan)
    contrib_before = np.full(len(C), np.nan)
    metric_before = [None] * len(C)
    for i in range(len(C)):
        if defined_before[i]:
            #  the DOMINANT metric by |contribution|.  `None` where there is no dominant
            #  metric to name: the old code fell back to `cols[0]` -- `nanargmax` over an
            #  all-(-1.0) row returns index 0 -- and printed the first column of the frame
            #  as if it were the name's biggest driver.
            j = int(np.argmax(sh_before[i]))
            metric_before[i] = cols[j]
            contrib_before[i] = float(C0[i, j])
            share_before[i] = float(sh_before[i, j])
        if defined_after[i]:
            share_after[i] = float(np.nanmax(sh_after[i]))

    data = {
        'n_capped': n_capped,
        'cap_value': cap_value,
        'metric_before': metric_before,
        'contrib_before': contrib_before,
        'share_before': share_before,
        'share_after': share_after,
        'agg_delta': agg_delta,
        'base_before': base_before,
        'base_after': base_after,
        'status': status,
    }
    if sources is not None:
        srcs = list(sources)
        if len(srcs) != len(C):
            raise ValueError('metric_share_cap: `sources` has %d entries for %d rows -- it '
                             'must be the pool in the SAME row order as `contrib`, or the '
                             'report is joined to the wrong names.' % (len(srcs), len(C)))
        data = dict([('source', srcs)] + list(data.items()))
    report = pd.DataFrame(data, index=contrib.index)

    _assert_post_condition(report, cap)
    return capped, report


def _assert_post_condition(report, cap):
    """The module checks its OWN output -- on the rows it CLAIMED to handle.

    This is not ceremony.  The one-pass form of this cap (truncate to `cap x base_before`)
    LOOKS right and leaves the capped metric above `cap` of the base that ships -- a guard
    whose own post-condition its own output violates.  Asserting the post-condition is what
    makes that class of error impossible to ship silently.

    *** THE FIRST VERSION CHECKED ONLY THE OVERSHOOT AND WAS STRUCTURALLY BLIND TO THE
    COLLAPSE -- the worse of the two.  A row the cap zeroed reported `share_after = -1.0`,
    and `-1.0 > 0.25` is False, so a report in which EVERY name had been annihilated passed
    this function in silence. ***

    WHAT RAISES AND WHAT DOES NOT, AND WHY THE LINE IS THERE
    -------------------------------------------------------
    ALMOST no DATA condition raises, and the exception is real.  A row the cap cannot be
    applied to -- too few non-zero metrics, a fixed point that would erase the name's mass, a
    non-finite contribution -- ships UNCAPPED, carries a `status`, and is named in the run
    log.

    *** THE HONEST QUALIFIER, BECAUSE TWO EARLIER VERSIONS OF THIS FILE SAID "NEVER RAISES ON
    A DATA CONDITION" AND THAT WAS FALSE.  An EXACT TIE between two contributions at the fixed
    point is a data condition, and it raises: `_cap_value`'s consistency test is `c < s[k-1]
    and c >= s[k]`, and when the fixed point lands ON a tied contribution both comparisons can
    fail by a single ulp, leaving no consistent k for a row that genuinely needs capping.  A
    third review reproduced it AT THE SHIPPED CAP = 0.25 -- `[0.78, 0.72, 0.65, 0.63, 0.09,
    0.07]` gives `c1 = 0.71999999999999986` against `s[1] = 0.71999999999999997`, one ulp
    short -- so the earlier claim that this was "harmless at CAP = 0.25" was ALSO wrong.  What
    is true is narrower and worth stating exactly:

      * the trigger is ONE cause, fully characterised: across 1,189,851 over-cap rows on
        decimal grids, every one of the 13,917 failures had a relative gap to the tie
        boundary of at most 2.675e-16 (one to two ulps), median 0.0;
      * it needs EXACT ties, which decimal grids manufacture and float pipelines do not:
        zero raises in 2,000,000 continuous-float rows, zero in 196,000 jittered real rows,
        zero in 200 real panel rows -- where the closest any name came to the boundary was
        2.6e-5, about 1e11 ulps away;
      * the winsorizer that could have manufactured ties by clipping many z-scores to the
        same +-3 was REMOVED on 2026-08-03.

    So the rate on this pipeline is measured zero, the hazard is understood, and the fix is
    NOT to re-tolerance the water-filling comparisons -- three independent reviews now agree
    that trading a measured-zero hazard for a fresh one in the only piece of algebra
    everything else rests on is the worse deal.  The fix is that this text no longer claims
    the case away. ***

    An ALGEBRA condition does raise: if the cap WAS applied and its own post-condition does not hold, the module
    is broken and must not hand back a ranking.

    *** THE SECOND VERSION OF THIS FUNCTION RAISED ON THE INFEASIBLE ROW, AND THAT WAS
    WRONG.  The argument for it was that the three available answers were "zero the row"
    (fabricates a promotion), "ship it uncapped silently" (hides a one-metric name) and
    "refuse" -- and that refusing was the only honest one.  THE TRICHOTOMY WAS FALSE: the
    fourth answer, ship-uncapped-AND-SAY-SO, was already implemented eight lines up for the
    non-finite row.  It fabricates nothing and hides nothing, and it costs no list.
    Refusing costs the entire list: `postBo.py:697` does not exception-guard the general-pool
    call, so the AssertionError propagates out of Sbocker and the run dies with no postRank,
    no AggScoreTop CSV and no top-20.  The condition is deterministic on a panel, so
    re-running from the saved metric pickle hits the identical assertion -- recovery needs a
    code edit at whatever hour someone notices.  And it FIRES: on the saved
    `postRank_2026-08-11_..._CUR3K` panel (`STRT`, `PET.TO`) and the 2026-08-07 one (`STRT`),
    both from the current CUR3K universe.  The belief that the imputation ladder would
    intercept those names first was simply false -- see the module docstring.  Trading a
    silent mis-rank of one name for a dead run and no shortlist at all is a worse deal, and
    it was made on an argument that had not been checked against the control flow. ***

    So: the four uncappable statuses are outcomes, not errors.  The three checks below fire
    only on rows with `status == 'ok'` -- rows the module asserts it HAS handled.  Scoping
    them that way is also what stops an `inf` contribution being reported as "had a defined
    share before the cap and NONE after it", which it never had.
    """
    handled = report[report['status'] == STATUS_OK]

    #  INVARIANT 1: A COMMITTED TRUNCATION MUST NEVER DESTROY THE NAME.  A handled row has
    #  to leave with a base that is still positive and still finite -- `base_after == 0` is
    #  the annihilation the pre-fix module shipped in silence, and a non-finite base is a
    #  name about which no statement can be made at all.
    #
    #  ONE CHECK, NOT TWO, AND IT IS A BACKSTOP.  This was two checks; the second tested
    #  `base_before > 0 and share_after is NaN`, which no input could reach -- and because
    #  `inf > 0` is True it fired on an infinite contribution with the message "had a defined
    #  share before the cap and NONE after it", about a row that never had one.  Merged and
    #  scoped to handled rows, both problems go.
    #
    #  It is UNREACHABLE while the `would_erase` check stands in front of it, and that is the
    #  point of a backstop: it is what fails loudly if the mass floor is ever weakened or
    #  removed.  `test_the_collapse_guard_FIRES_from_apply_share_cap` stands the floor down
    #  and drives this through the REAL entry point, rather than hand-building a frame and
    #  calling this function directly -- which is what the previous version of that test did,
    #  and is the shape this file's own docstring names as the house defect.
    destroyed = handled[~(np.isfinite(handled['base_after']) & (handled['base_after'] > 0))]
    if len(destroyed):
        raise AssertionError(
            'metric_share_cap: %d name(s) were CAPPED and left with a base that is zero or '
            'non-finite -- the truncation destroyed the name rather than compressing it, '
            'which puts an AggScore of 0.0 (or nothing at all) where a real score was. This '
            'is an ALGEBRA failure, not a data condition: an uncappable row is supposed to '
            'be declined before it reaches here. Offenders: %s'
            % (len(destroyed), list(destroyed.index[:10])))

    #  THE OVERSHOOT.  This one DOES raise on a row the module tried to cap and failed,
    #  because that is the module being broken rather than the data being awkward -- and
    #  because it is the only thing that detects a cap replaced by a pass-through, which is
    #  a defect no amount of logging would surface.
    bad = handled[handled['share_after'] > cap + _SHARE_TOL]
    if len(bad):
        raise AssertionError(
            'metric_share_cap: %d name(s) left the cap ABOVE it (worst %.6f > %.6f) and '
            'shipped UNCAPPED. `_cap_value` found no consistent k for a row that needed one. '
            'Two known causes: (a) the fixed-point search is broken or has been replaced by '
            'a pass-through -- the usual case, and the reason this raises; (b) the row has '
            'two contributions that tie EXACTLY at the fixed point, where the strict/'
            'non-strict comparisons in `_cap_value` can both fail by one ulp. (b) IS '
            'reachable at CAP = 0.25 -- but only with EXACT ties, which need a decimal grid: '
            'it has a measured rate of ZERO over 2,000,000 continuous-float rows, 196,000 '
            'jittered real rows and 200 real panel rows, so on live data suspect (a) first '
            'and check whether the row you are looking at has two identical contributions. '
            'Offenders: %s'
            % (len(bad), float(bad['share_after'].max()), cap, list(bad.index[:10])))


def format_report(report, pool_label, cap=CAP, top_n=20):
    """The run-log block.  Says what bound, on which metric, ON HOW MANY METRICS, and what
    it cost -- and says so even when NOTHING bound, because "the cap ran and bound on
    nobody" and "the cap did not run" read identically to anyone scanning a log, and only
    one of them is a fact about the panel.

    IT ALSO NAMES EVERY ROW THE CAP COULD NOT BE APPLIED TO.  Those rows ship UNCAPPED, so
    this block is the ONLY place their existence is visible to a reader; the module
    deliberately does not raise on them (see `_assert_post_condition`), which makes the
    loudness of this block the whole of the mitigation.

    `source` is read off the report itself, so the log and the shipped frame name the same
    thing.  Falls back to the index only for a report built without `sources`."""
    n = len(report)
    hit = report[report['n_capped'] > 0]
    zeroed = report[report['status'] == STATUS_ALL_ZERO]
    lines = ['SINGLE-METRIC SHARE CAP [%s]: cap %.0f%% of a name\'s ABSOLUTE contribution; '
             'BOUND on %d of %d name(s)%s'
             % (pool_label, 100 * cap, len(hit), n,
                ' -- NOTHING was truncated on this panel' if not len(hit) else '')]

    def _src(idx):
        if 'source' in report.columns:
            v = report['source'].loc[idx]
            return v if isinstance(v, str) else str(v)
        return str(idx)

    if len(hit):
        lines.append('    (share is |z x w| / sum|z x w|, NOT of the signed AggScore -- see '
                     'metric_share_cap; the two differ by a factor of ~1.5 on this pool)')
        #  `n_capped` is printed on EVERY line and it is not decoration: the share pair
        #  shown is the DOMINANT metric's only, so without the count a three-metric cascade
        #  (which erases most of a name's mass) is indistinguishable from a one-metric trim,
        #  and the AggScore delta beside it cannot be reconciled with the single pair.
        #  No rank is printed and that is deliberate: this runs BEFORE `getAggScore`, so no
        #  ranking exists yet.  A position printed here would be the scoring frame's row
        #  order, which is not a rank and would be read as one.
        for idx, row in hit.sort_values('share_before', ascending=False).head(top_n).iterrows():
            #  THE LABEL IS DERIVED FROM THE DOMINANT METRIC'S OWN SIGN, not from the sign
            #  of `agg_delta`.  `agg_delta` is the sum of the signed changes across ALL
            #  capped metrics, so on a k>1 row with mixed signs it can be positive while the
            #  dominant contribution is positive too -- and the old label then told the
            #  reader that a POSITIVE dominant metric "was NEGATIVE".  The mechanism claim
            #  and the direction of the move are now two separate statements, because they
            #  are two separate facts.
            dom = row['contrib_before']
            rose = row['agg_delta'] > 0
            if dom < 0 and rose:
                note = '  <-- RAISED: its dominant contribution was NEGATIVE and was truncated'
            elif dom < 0:
                note = '  <-- dominant contribution NEGATIVE (net move still down: k > 1)'
            elif rose:
                note = ('  <-- RAISED although its dominant contribution was POSITIVE: k > 1 '
                        'with mixed signs, so the truncated negatives outweigh it')
            else:
                note = ''
            lines.append('    %-12s  %-26s %.4f -> %.4f  n_capped=%d   AggScore %+.4f%s'
                         % (_src(idx), row['metric_before'],
                            row['share_before'], row['share_after'],
                            int(row['n_capped']), row['agg_delta'], note))
        if len(hit) > top_n:
            lines.append('    ... and %d more (full per-name detail in '
                         "rankdic['share_cap_report'])" % (len(hit) - top_n))
        n_multi = int((hit['n_capped'] > 1).sum())
        lines.append('    total AggScore moved: %+.4f over %d name(s); %d had a NEGATIVE '
                     'dominant contribution (the cap is symmetric by ruling, see the module '
                     'docstring) and %d rose on net; %d bound on MORE THAN ONE metric (a '
                     'cascade -- the share pair above is the dominant metric only)'
                     % (float(hit['agg_delta'].sum()), len(hit),
                        int((hit['contrib_before'] < 0).sum()),
                        int((hit['agg_delta'] > 0).sum()), n_multi))

    #  --- THE UNCAPPABLE ROWS.  THIS BLOCK IS THE MITIGATION, NOT A FOOTNOTE ------------
    #  These names ship with the score they would have had if this module did not exist.
    #  Nothing else in the run says so, because nothing raises: if this block is quiet or
    #  gets truncated away, the module is back to hiding exactly what it was built to expose.
    _WHY = {
        STATUS_INFEASIBLE: ('at most %d non-zero metric contribution(s), so NO truncation '
                            'can put every metric at or below %.0f%% -- the fixed point '
                            'would zero the whole row and invent an AggScore of 0.0'
                            % (_k_max(cap), 100 * cap)),
        STATUS_WOULD_ERASE: ('the fixed point would leave under %.0f%% of the name\'s '
                             'absolute mass -- that is erasing the score, not compressing it'
                             % (100 * _MASS_FLOOR)),
        STATUS_NON_FINITE: ('a non-finite (NaN/inf) contribution makes the share undefined '
                            'for the whole row'),
    }
    for st in UNCAPPABLE:
        rows = report[report['status'] == st]
        if not len(rows):
            continue
        #  EVERY name, with NO `[:20]`.  This block is the whole of the mitigation for a
        #  row that ships uncapped, and a mitigation that silently truncates itself is the
        #  same defect one level up: the twenty-first name would ship undisclosed while the
        #  log still read as a complete disclosure.  A long line is the correct cost.
        lines.append('    *** %d name(s) NOT CAPPED and shipped with their UNCAPPED score '
                     '[%s]: %s. These names were NOT assessed against the cap and may be '
                     'dominated by a single metric. ALL %d are named here, and in %s: %s'
                     % (len(rows), st, _WHY[st], len(rows),
                        SHARE_CAP_CSV % '<date>',
                        ', '.join(_src(i) for i in rows.index)))
    if len(zeroed):
        lines.append('    %d name(s) had NO contribution at all (every metric exactly zero); '
                     'there is no share to cap and their AggScore is 0.0 on their own '
                     'account, not on this module\'s: %s'
                     % (len(zeroed), ', '.join(_src(i) for i in zeroed.index[:20])
                        + (' ...' if len(zeroed) > 20 else '')))
        #  This list MAY abbreviate where the uncappable one may not: an all-zero row is not
        #  a name the cap failed to assess, it is a name with nothing to assess, and its
        #  AggScore is 0.0 whether or not this module exists.  The COUNT above is exact and
        #  the full list is in the evidence CSV.
        if len(zeroed) > 20:
            lines.append('        (%d of those %d names are not printed above; every one of '
                         'them is in %s)'
                         % (len(zeroed) - 20, len(zeroed), SHARE_CAP_CSV % '<date>'))
    return '\n'.join(lines)


#  THE SHIPPED EVIDENCE FILE.  Dated, at the repo root, one row per name.
#
#  WHY IT EXISTS (third review, 2026-08-31).  Until this writer the cap disclosed itself in
#  exactly two places: `rankdic['share_cap_report']`, which no consumer in this repo reads,
#  and stdout.  Every comparable layer in this pipeline emits a dated CSV --
#  AdHocPenaltyBucket, MissingDataFillReport, DedupSurvivorReport, DollarVolumeFloor,
#  Stage1VetoEjections -- and the cap did not.  That is not a cosmetic gap.  The ENTIRE
#  justification for shipping an uncappable name uncapped is "and we said so", and a
#  disclosure that lives only in a console log is a disclosure nobody is obliged to have
#  read.  This file is what makes that sentence true.
#
#  *** TWO ONE-LINE ADDITIONS OUTSIDE THIS MODULE ARE STILL NEEDED FOR IT TO REACH THE CEO,
#  AND NEITHER FILE WAS IN SCOPE FOR THE CHANGE THAT ADDED THIS WRITER:
#      Sbocker.allowlist_patterns  needs  'ShareCapReport_*.csv'   (or the file is written
#          and never transferred to Drive -- written-but-unshipped is the same as unwritten)
#      conftest._EVIDENCE_GLOBS    needs  'ShareCapReport_*.csv'   (RULE E, so a test that
#          forgets to redirect the path cannot drop one in the repo root)
#  Until the first lands, this artifact exists only on the run machine. ***
SHARE_CAP_CSV = 'ShareCapReport_%s.csv'

#  Header once per process, then append: `postBoScoreRanking` runs ONCE PER POOL, and a
#  per-call overwrite would leave only the last one.  Same idiom, and same reason, as
#  `postBoRank._write_missing_csv`.  Only the general pool is capped today, so the append
#  matters the moment `CAPPED_POOLS` grows -- which is a decision, not an accident.
_CSV_STARTED = set()

#  Column order is the READING order, not the frame order: who, then what happened to them,
#  then the numbers behind it.  `pool` leads because the file carries every capped pool.
CSV_COLUMNS = ('pool', 'source', 'status', 'n_capped', 'metric_before', 'contrib_before',
               'share_before', 'share_after', 'cap_value', 'base_before', 'base_after',
               'agg_delta')


def _evidence_dir():
    """Where the run's evidence CSVs go.  `transfer_utils.EVIDENCE_DIR` is the house answer
    and is anchored to the module's own directory rather than to the CWD.  Falling back to
    the CWD if that import fails keeps an evidence writer from being able to break a scored
    run over a missing helper."""
    try:
        import transfer_utils as _tu
        return _tu.EVIDENCE_DIR
    except Exception:
        return '.'


def write_evidence_csv(report, pool_label, path=None, run_date=None):
    """Append this pool's share-cap report to the dated evidence CSV.

    Returns the path written, or None.  NEVER RAISES -- an evidence file must not be able to
    cost a scored run, the same contract `adhoc_penalty.write_evidence_csv` carries and for
    the same reason.  A failure PRINTS, because a silent failure here would recreate exactly
    the gap this writer closes.

    EVERY NAME IN THE POOL IS WRITTEN, not only the ones the cap bound on.  "The cap bound on
    nobody" and "the cap did not run" have to stay distinguishable in the artifact as well as
    in the log, and the `status` column is what makes an uncappable name findable by a reader
    who never saw stdout.
    """
    try:
        if report is None or not len(report):
            return None
        import os
        out = report.copy()
        out.insert(0, 'pool', pool_label)
        if 'source' not in out.columns:
            out.insert(1, 'source', [str(i) for i in out.index])
        cols = [c for c in CSV_COLUMNS if c in out.columns]
        cols += [c for c in out.columns if c not in cols]
        out = out[cols]
        if path is None:
            stamp = run_date or pd.Timestamp.today().strftime('%Y-%m-%d')
            d = _evidence_dir()
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, SHARE_CAP_CSV % stamp)
        first = path not in _CSV_STARTED
        out.to_csv(path, index=False, mode='w' if first else 'a', header=first)
        _CSV_STARTED.add(path)
        return path
    except Exception as _e:
        print('WARNING: could not write the share-cap evidence CSV (%s: %s). The cap still '
              'ran and the run log above is unaffected -- but the per-name disclosure did '
              'NOT reach an artifact.' % (type(_e).__name__, _e), flush=True)
        return None
