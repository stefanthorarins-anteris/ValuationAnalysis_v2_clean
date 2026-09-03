"""metric_share_cap.py  --  THE SINGLE-METRIC SHARE CAP (CEO, 2026-08-31)

WHAT IT IS
----------
A per-name truncation applied to the WEIGHTED metric contributions (`z x w`) after the
weighting step and before `getAggScore`, so that no metric which is HELPING a name may
account for more than `CAP` of that name's total ABSOLUTE contribution.  A name whose
score is carried by one column has that column's contribution cut back and KEEPS
COMPETING on the reduced score; it is never removed.

    contribution_ic = z_ic * w_c            (signed; this IS the AggScore addend)
    base_i          = sum_c |contribution_ic|          (BOTH signs -- see DENOMINATOR)
    share_ic        = contribution_ic / base_i
    ... and after this module runs, max share_ic <= CAP for every name i, where the max
    runs over the POSITIVE contributions ONLY.

A NEGATIVE contribution is never truncated, at any magnitude: the cap may PENALISE and
may never RESCUE.  CEO ruling, 2026-09-01 -- see POSITIVE-ONLY below, which is the single
most important section in this file for anyone about to change the algebra.

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

POSITIVE-ONLY: THE CAP MAY PENALISE, IT MAY NEVER RESCUE  (CEO RULING, 2026-09-01)
----------------------------------------------------------------------------------
*** THIS SECTION USED TO DEFEND THE OPPOSITE BEHAVIOUR, AND THE DEFENCE WAS OVERRULED.  It
argued that truncating a large NEGATIVE contribution too was "the honest reading of no
single metric decides a name".  The CEO's answer: the cap was set to stop one metric
carrying a name UP, and forgiving a catastrophic score on one metric is not a side effect of
that, it is the opposite of it.  The reading is not available any more.  If you want to
re-open it, argue with the ruling -- do not restore the behaviour because the algebra is
tidier on |contribution|. ***

WHAT THE SYMMETRIC RULE DID, AND WHY IT IS THE WRONG SIGN.  The rule was stated on
|contribution|, so it truncated a large negative contribution as readily as a large positive
one: a name that was catastrophically bad on one metric had that penalty cut back and ROSE.
That was not a corner case; on both measured panels it was the cap's single largest effect
anywhere in the pool.

    2026-08-31 panel   13 bound, of which 4 RAISED.  `BOSS.DE` climbed 15 places
                       (rank 58 -> 43) on a `freeCashFlowPerShareGrowth` sitting at -38.1%
                       of its absolute contribution; `JEN.DE` climbed 9 on the same
                       mechanism (-45.7%).
    2026-09-01 panel   11 bound, of which 4 RAISED -- `JEN.DE` +0.1301 (rank 82 -> 72),
                       `BOSS.DE` +0.1053 (58 -> 45), `OII` +0.0597, `KFY` +0.0019.
                       (`ShareCapReport_2026-09-01.csv`, reproducible from that file: all 11
                       bound rows resolve at k = 1, and the 4 raised rows are exactly the 4
                       whose dominant contribution is negative.)

A fresh-eyes review on 2026-09-03 reached the same finding independently and put the size of
it plainly: +0.1301 is LARGER THAN THE RANK-12-TO-RANK-20 SPREAD of the shipped list, and it
is the wrong sign for an objective that is worried about losers.

THE MEASUREMENT THAT MOTIVATED THE RULING, KEPT.  Positive-only was measured on the 08-31
panel before it was chosen: it bound on 10 names instead of 13, produced the SAME top-20,
and its largest move was 4 places instead of 15.  So the instrument gets STRICTLY less
violent and the deliverable does not move -- which is what made this a cheap ruling to make.
*** NOT REPRODUCED HERE, AND SAID PLAINLY BECAUSE IT IS THE ONE NUMBER IN THIS SECTION
NOBODY CAN RE-DERIVE: neither the 08-31 contribution panel nor an 08-31 ShareCapReport is on
disk, and rank moves need an AggScore vector that no saved artifact for either date carries.
The 08-31 line above is therefore INHERITED from the note that recorded it, not verified. ***

WHAT IS REPRODUCED, on the two artifacts that do exist (2026-09-03):

    2026-09-01 (from `ShareCapReport_2026-09-01.csv`, exact -- see below)
                       bound 11 -> 7, RAISED 4 -> 0, total AggScore moved
                       -0.0966 -> -0.3936.  The 7 that still bind are unchanged, cell for
                       cell: same `cap_value`, same `agg_delta`.
    2026-08-11 CUR3K panel (`postRank_2026-08-11_fmp_stock_CUR3K.pickle`, re-run both ways)
                       see `test_the_saved_panel_...` -- RAISED = 0 under the new rule and
                       non-zero under the old one, which is what makes the panel a
                       regression rather than an illustration.

WHY THE 09-01 RECONSTRUCTION IS EXACT AND NOT AN APPROXIMATION, since a report is not a
panel and normally could not settle this.  Every bound row on that panel has `n_capped = 1`,
and for a k = 1 row the fixed point satisfies `c = CAP * T` with `T < base_before`, so the
consistency test `s[1] <= c` forces the SECOND-largest contribution below `CAP *
base_before`.  Only the dominant metric is therefore over the cap as a share of the original
base.  Two consequences follow with no further information:
  * dominant POSITIVE -> the candidate set is the same single metric, `R` is the same, so `c`
    and `agg_delta` are IDENTICAL under either rule;
  * dominant NEGATIVE -> there is no candidate over the cap at all, so nothing binds and
    `agg_delta` becomes exactly 0.
That is why "bound 11 -> 7, raised 4 -> 0" is a derivation and not an estimate.  It does NOT
extend to a k > 1 panel, and it does not give ranks.

THE DENOMINATOR STAYS `sum_c |contribution_c|`, OVER BOTH SIGNS -- AND THAT WAS A REAL FORK
---------------------------------------------------------------------------------------------
The ruling fixes WHICH contributions may be truncated.  It does not, by itself, say what the
share is a share OF, and the two candidates give different names:

  (A) base = sum over ALL metrics of |c|          BUILT.  A negative contribution is
      excluded from the CANDIDATES but stays in the DENOMINATOR at full magnitude.
  (B) base = sum over the POSITIVE metrics only   NOT BUILT.

(B) is rejected on three grounds and the first is decisive:

  1. IT CHANGES WHAT 0.25 MEANS, AGAIN, AND NOBODY RULED ON THAT.  Dropping the negatives
     from the denominator SHRINKS it, so every positive share RISES and the cap bites harder
     and on more names -- and hardest on names that carry big penalties, since those have the
     smallest positive base.  This file already documents (see the "(i) vs (ii)" section) that
     0.25 means something WEAKER on the absolute-share distribution than on the signed one the
     CEO was shown when he picked it.  Moving the denominator a second time, in the
     tightening direction, would re-level the CEO's number by a developer's choice of
     denominator.  0.25 is his; the denominator must not be a back door to changing it.
  2. IT IS INCOHERENT AS A RULE.  Under (B) a name's positive metrics are truncated MORE
     because it has a big penalty somewhere else.  "No single metric may carry a name up" does
     not imply "a name with bad news elsewhere may be carried up less" -- that is a second,
     unargued penalty riding on the first.
  3. IT REINTRODUCES THE DEGENERACY THE (i)-vs-(ii) SECTION REJECTED (ii) FOR.  A name with no
     positive contribution has a positive base of ZERO, and a name with one small positive has
     a base near zero, so the share diverges.  On the 2026-08-31 general pool 37 of 97 names
     score at or below zero, so this is a populated region, not a corner.  (A) is bounded in
     [0, 1] for every name whose absolute base is positive and finite, exactly as before.

A fourth fact, recorded as a fact and NOT used as an argument: (A)'s consequence on the 09-01
panel is exactly derivable from the shipped evidence CSV (above), and (B)'s is not derivable
from anything on this disk, because the positive base of a row is not in the report.  That is
a convenience, not a reason, and it must not be cited as one.

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


def _truncation_mask(row, c):
    """The cells the cap is PERMITTED to truncate: contributions that are HELPING the name
    and sit above the fixed point `c`.

    ONE LINE, AND IT IS A FUNCTION FOR TWO REASONS.

    (1) IT IS THE CEO'S 2026-09-01 RULING, so it gets a name and a single home rather than
    being an inline comparison a reviewer scrolls past.  "The cap may penalise, never
    rescue" is `row > c` instead of `abs(row) > c`, and that one character is the entire
    behavioural difference between this module and the one that RAISED four names on the
    2026-09-01 panel.

    (2) IT MAKES INVARIANT 3 CERTIFIABLE.  `c` is always positive, so no negative
    contribution can satisfy `row > c` and the no-rescue invariant in
    `_assert_post_condition` is UNREACHABLE through the shipped code -- which is what a
    guard should be, and also what makes it impossible to watch fail.  A guard nobody has
    seen fire is a guard nobody knows works, so the only honest way to certify it is to
    substitute the pre-ruling absolute mask here and drive the real entry point.
    `test_mutation_the_SYMMETRIC_mask_is_REJECTED_by_the_no_rescue_invariant` does exactly
    that; it is the reason this is not inlined.
    """
    return row > c


def _cap_value(row, cap):
    """The fixed-point cap level for one name, or None if no POSITIVE contribution exceeds
    `cap` of the name's absolute total.

    `row` is the row's SIGNED contributions (any order).  IT USED TO BE THE ABSOLUTE ROW,
    and the change of argument IS the CEO's 2026-09-01 positive-only ruling: the sign of a
    contribution now decides whether it is a truncation CANDIDATE, so this function cannot
    be given a row that has already thrown the sign away.

    THE ALGEBRA IS UNCHANGED; ONLY THE CANDIDATE SET SHRANK.  `base` is still the FULL
    absolute total -- every metric, both signs -- and `R` is still "the absolute mass that
    is NOT truncated".  What changed is that the negatives are now permanently inside `R`
    instead of being eligible for `S`:

        base = sum_c |contribution_c|                    (unchanged -- see the DENOMINATOR
                                                          section of the module docstring)
        p    = the POSITIVE contributions, descending    (the only candidates)
        R    = sum|negatives| + sum p[k:]                the absolute mass NOT truncated
        c    = cap * R / (1 - k*cap)
        T    = R / (1 - k*cap)                           so c / T == cap exactly

    `R` IS SUMMED DIRECTLY AND MUST NOT BE COMPUTED AS `base - sum(p[:k])`.  It is the same
    quantity in exact arithmetic and a DIFFERENT one in float64, because the subtraction
    cancels the whole tail away: on `[0.30, 0.10, 0.05, 1e-15]` the head sums to
    `0.45000000000000001` against a base of `0.45000000000000001`, so the subtraction returns
    0.0 (or a negative ulp) and the k = 3 fixed point is lost -- the row then finds no
    consistent k, ships UNCAPPED, and the `would_erase` guard that is supposed to decline it
    never runs.  The pre-2026-09-01 code summed `s[k:]` directly and was correct for exactly
    this reason; writing the subtraction back in is a silent regression that only the
    1e-15 case in `test_a_FEASIBLE_row_whose_cap_would_ERASE_it_ships_UNCAPPED` catches.

    A CONSEQUENCE WORTH STATING, because it is why the infeasibility test in
    `apply_share_cap` had to change too: a single negative contribution of any size puts a
    permanent floor under `R`, so a row that WAS infeasible under the symmetric rule (too few
    non-zero metrics, fixed point degenerating to c = 0) can be perfectly feasible now.
    `[+0.30, -0.30, -0.30]` is the case: three non-zero metrics, declined as infeasible by
    the old count test, and under this rule R = 0.60, c = 0.20, post-cap share exactly 0.25
    with 80% of the mass surviving.  Deciding feasibility on the raw non-zero COUNT would now
    decline a row the cap can honour perfectly well.
    """
    a = np.abs(row)
    if not np.isfinite(a).all():
        return None
    base = float(a.sum())
    if not np.isfinite(base) or base <= 0.0:
        return None
    #  THE CANDIDATES.  `row > 0.0` and not `a > 0.0`: this is the whole ruling in one
    #  comparison.  A negative contribution is never selected, at any magnitude.
    p = np.sort(row[row > 0.0])[::-1]
    if len(p) == 0:
        return None                           # nothing the cap is permitted to touch
    #  The negatives are the part of `R` that no k can ever remove.  Held once, outside the
    #  scan, because it does not depend on k.
    neg_mass = float(a[row < 0.0].sum())
    #  k is bounded a priori; without this the scan could propose a denominator <= 0.
    k_max = _k_max(cap)
    for k in range(1, min(len(p), k_max) + 1):
        denom = 1.0 - k * cap
        if denom <= 0.0:                      # unreachable given k_max; kept as a hard stop
            break
        #  SUMMED, NOT SUBTRACTED -- see the docstring.  `base - p[:k].sum()` is the same
        #  number in real arithmetic and loses the entire tail to cancellation in float64.
        R = neg_mass + float(p[k:].sum())
        c = cap * R / denom
        below = p[k] if k < len(p) else -np.inf
        #  consistency: the k-th largest POSITIVE must still be ABOVE the cap value (so it
        #  really is capped) and the (k+1)-th must be at or BELOW it (so it really is not).
        #  The comparisons run over the positive subsequence only, so the exact-tie hazard
        #  documented in `_assert_post_condition` is unchanged in character and strictly
        #  narrower in reach: a tie between a positive and a negative can no longer trigger
        #  it, because the negative is not in this sequence.
        if c < p[k - 1] and c >= below:
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
#
#  #####################################################################################
#  ##  RE-EXAMINED UNDER POSITIVE-ONLY (2026-09-03).  THE FLOOR IS RETAINED AND ITS      ##
#  ##  MOTIVATING CASE IS DEAD.  BOTH HALVES OF THAT MATTER.                             ##
#  #####################################################################################
#
#  THE CASE IT WAS BUILT FOR IS NOW UNREACHABLE, AND THE OLD RATIONALE MUST NOT BE READ AS
#  STILL LIVE.  Every sentence above and in the `would_erase` / `infeasible` sections was
#  written about ONE harm: the cap replacing a real NEGATIVE score with a fabricated 0.0,
#  which on a pool where 37 of 97 names score at or below zero is a large PROMOTION invented
#  by the guard.  Positive-only kills that harm outright, and not by degree -- a committed
#  truncation now only ever replaces a POSITIVE contribution with a smaller positive one, so
#  `agg_delta <= 0` identically (INVARIANT 3 in `_assert_post_condition` asserts it).  The
#  cap can no longer move ANY name upward, so it cannot fabricate a promotion, so the
#  "annihilation is a promotion" argument that justified this number is spent.  THE CEO FOUND
#  THIS FLOOR HARD TO UNDERSTAND, and that is why: its rationale depended entirely on the
#  symmetric behaviour he has now ruled out.
#
#  THE MECHANISM, HOWEVER, IS STILL REACHABLE -- with a changed sign and a much narrower
#  domain.  `base_after >= sum |negative contributions|`, because a negative is never
#  truncated, so this floor can only fire on a name whose absolute mass is at least 99%
#  POSITIVE and whose positive tail is degenerate.  `[+0.30, -0.001, 0, 0, 0]` is the shape:
#  R = 0.001, c = 3.3e-4, surviving mass 0.44% -- declined.  So does the all-positive
#  `[0.30, 0.10, 0.05, 1e-9]`, which is the row the COUNT test could not see and which is
#  why this condition is stated on the mass in the first place.
#
#  WHY IT IS KEPT ANYWAY, in one sentence: what such a row would ship is not a penalty of a
#  chosen size but a score of ~0 whose magnitude is set by the epsilon in the tail -- move
#  that tail from 1e-9 to 1e-3 and the surviving mass moves six orders of magnitude, on a
#  name whose business did not change.  The instrument the CEO authorised is "truncate a
#  helping metric to 25% of the base"; collapsing a +0.45 score to 4e-09 is not that
#  instrument, it is the fixed point degenerating.  A fabricated extreme is refused in
#  EITHER direction, and that is the whole of the argument now.
#
#  *** THE PART THAT IS THE CEO'S CALL AND IS NOT MINE, FLAGGED RATHER THAN DECIDED: the
#  harm this floor now prevents is a large DEMOTION, and he has just ruled that he wants the
#  cap to penalise.  A perfectly coherent answer is "let it collapse -- I asked for
#  penalties, and a name that is 99% one metric deserves nothing".  I have NOT taken that
#  answer, because the size of the demotion is arbitrary rather than chosen, but the
#  disagreement is a level question for him and the floor is one line to remove.  What is NOT
#  available is leaving the floor in place with its old promotion rationale unexamined; that
#  is the state this block exists to end. ***
#
#  INVARIANT 1 of `_assert_post_condition` (`base_after > 0` on a handled row) remains
#  UNREACHABLE either way, and the reason has changed: `c = 0` requires `R = 0`, which is now
#  the joint infeasibility condition and is declined before the fixed point is committed.
#  Removing this floor would therefore NOT surface as a raise -- it would ship a silent
#  4e-09.  Stated because the tempting argument for retention ("the backstop would catch it")
#  is false, and a reviewer should not be given a reason that does not hold.
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
              `metric_before`, `contrib_before` (that metric's SIGNED pre-cap contribution),
              `metric_capped` / `pos_share_before` / `pos_share_after` (the largest HELPING
              contribution -- the only kind the cap may act on, and the pair the
              post-condition is stated on), `base_before`, `base_after`, `status`, and the
              `agg_delta` this cap costs.  `agg_delta` is <= 0 for every row by ruling.

    `metric_before` and `metric_capped` DIFFER exactly when a name's biggest driver is a
    PENALTY.  That is not an edge case to be tidied away -- it is the population the
    2026-09-01 ruling protects, and `format_report` says so on the name's own line.

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
        n_pos = int((C[i] > 0.0).sum())
        n_neg = int((C[i] < 0.0).sum())
        if n_pos == 0:
            #  NOTHING THE CAP IS PERMITTED TO TOUCH, and it is a plain no-bind rather than
            #  an uncappable row: every contribution this name has is a penalty, and under
            #  the positive-only ruling a penalty is never truncated at any magnitude.  So
            #  the name ships unchanged with `status = 'ok'`, `n_capped = 0` -- which is the
            #  truth ("the cap ran and bound on nothing here"), not a decline.
            #
            #  IT IS COUNTED IN THE LOG ANYWAY (`format_report`'s positive-only block),
            #  because THIS IS THE POPULATION THE OLD SYMMETRIC CAP RESCUED.  A name that is
            #  100% penalties was exactly the shape whose dominant negative got truncated and
            #  whose score ROSE, so a reader comparing this run against a pre-ruling one needs
            #  to be able to see how many such names there were.  `share_before` still reports
            #  its dominant metric's share, so a single-metric-dominated penalty name remains
            #  visible in the artifact -- it is simply no longer acted on.
            continue
        if n_neg == 0 and n_pos <= k_max:
            #  INFEASIBLE, AND THE CONDITION IS NOW A JOINT ONE (positive-only, 2026-09-01).
            #  It used to be `n_nonzero <= k_max` on the ABSOLUTE row.  That test is WRONG
            #  under this rule and would decline rows the cap can honour: a negative
            #  contribution can never be truncated, so it stays inside `R` permanently and
            #  puts a floor under the fixed point.  The degenerate c = 0 therefore needs BOTH
            #  halves -- no negative anywhere to hold `R` up, AND few enough positives that
            #  the scan can truncate all of them.  `[+0.30, -0.30, -0.30]` is the row that
            #  makes the difference concrete: declined by the old count test, and resolved at
            #  c = 0.20 with 80% of its mass surviving under this one.  See `_cap_value`.
            #
            #  WHAT COMMITTING c = 0 WOULD DO HAS ALSO CHANGED SIGN, and the new reading is
            #  the honest one: such a row is by construction ALL-POSITIVE, so annihilating it
            #  is a large DEMOTION rather than the invented promotion the symmetric rule
            #  produced.  It is still declined -- see `_MASS_FLOOR` for why a fabricated
            #  extreme is refused in either direction -- but a reader must not carry the old
            #  "0.0 is a promotion" rationale over to this branch.
            status[i] = STATUS_INFEASIBLE
            continue
        c = _cap_value(C[i], cap)
        if c is None:
            continue
        #  THE RULING, VIA THE ONE FUNCTION THAT ENCODES IT.  Was `A[i] > c` on the
        #  absolute row, which is what truncated catastrophic penalties and RAISED four names
        #  on the 2026-09-01 panel.  See `_truncation_mask` for why it is a named function
        #  and not an inline comparison.
        over = _truncation_mask(C[i], c)
        #  THE TRUNCATION IS NOT COMMITTED UNTIL THE MASS IS CHECKED.  Counting metrics
        #  catches the row with an EMPTY tail; it does not catch the row with a NEGLIGIBLE
        #  one, and both end with a fabricated AggScore.  Computed on a candidate so the
        #  row can be left exactly as it arrived.
        cand_base = float(np.where(over, c, A[i]).sum())
        if cand_base < _MASS_FLOOR * base_before[i]:
            status[i] = STATUS_WOULD_ERASE
            cap_value[i] = c            # what it WOULD have been, for the log
            continue
        n_capped[i] = int(over.sum())
        cap_value[i] = c
        #  No `np.sign` any more, and its absence is a statement: everything in `over` is
        #  positive by construction, so the truncated value IS `c`.  The old
        #  `np.sign(C[i, over]) * c` existed only to send a truncated negative back to `-c`.
        C[i, over] = c

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
    #  --- THE POSITIVE SIDE: the quantity the cap now ACTS ON -------------------------- #
    #  `share_before`/`share_after` are the DOMINANT metric by |contribution| and their
    #  meaning is deliberately unchanged -- they answer "what drives this name", which is
    #  still the right question for a reader and is what the value-level tests pin.
    #
    #  BUT SINCE 2026-09-01 THEY ARE NO LONGER THE POST-CONDITION'S SUBJECT, and conflating
    #  the two would be a guard blind to its own rule.  Under positive-only, a name's
    #  dominant metric can be a NEGATIVE one that the cap is forbidden to touch, so
    #  `share_after` may legitimately sit ABOVE the cap -- and on a bound row it can even
    #  RISE (the truncation shrinks the base while the untouched negative keeps its
    #  magnitude, so its share of a smaller total is larger).  Printing that pair as the
    #  cap's before/after would read as the cap having INCREASED a concentration.
    #
    #  So the positive-side pair is carried explicitly: `metric_capped` is the largest
    #  HELPING contribution -- the candidate that actually binds -- and
    #  `pos_share_before`/`pos_share_after` are its share of the absolute base. THESE are
    #  what `_assert_post_condition` checks and what `format_report` prints on a hit line.
    pos_share_before = np.full(len(C), np.nan)
    pos_share_after = np.full(len(C), np.nan)
    metric_capped = [None] * len(C)
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
            #  ...and the largest POSITIVE one, which is a DIFFERENT column whenever the
            #  name's biggest driver is a penalty.  `None`/NaN for a name with no positive
            #  contribution at all: there is no candidate, so there is no share to report,
            #  and a zero here would read as "its best metric contributes nothing".
            pos = C0[i] > 0.0
            if pos.any():
                jp = int(np.argmax(np.where(pos, C0[i], -np.inf)))
                metric_capped[i] = cols[jp]
                pos_share_before[i] = float(C0[i, jp] / base_before[i])
        if defined_after[i]:
            share_after[i] = float(np.nanmax(sh_after[i]))
            pos_a = C[i] > 0.0
            if pos_a.any():
                pos_share_after[i] = float(np.max(C[i][pos_a]) / base_after[i])

    data = {
        'n_capped': n_capped,
        'cap_value': cap_value,
        'metric_before': metric_before,
        'contrib_before': contrib_before,
        'share_before': share_before,
        'share_after': share_after,
        'metric_capped': metric_capped,
        'pos_share_before': pos_share_before,
        'pos_share_after': pos_share_after,
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

    #  ORDER MATTERS HERE, AND IT WAS CHOSEN AFTER WATCHING BOTH FIRE.  The rescue check
    #  runs BEFORE the overshoot check because the mutation that matters trips BOTH: restoring
    #  the pre-ruling absolute mask makes a truncated negative land at `+c`, which both
    #  rescues the name AND leaves the positive share fractionally over the cap (0.3125 vs
    #  0.25 on the row used in `test_mutation_the_SYMMETRIC_mask_is_REJECTED_...`).  With the
    #  overshoot first, the reader gets "the fixed-point search is broken" -- a generic
    #  algebra complaint pointing at `_cap_value`, which is innocent -- instead of "the cap
    #  RAISED a name", which names the actual defect and the ruling it breaks.  Diagnosis
    #  quality is the whole value of an assertion that only ever fires on a code change.
    #
    #  The two mutations already in the suite are unaffected, and that was checked rather
    #  than assumed: a pass-through cap and a one-pass cut both leave `agg_delta <= 0`
    #  (they truncate positives, or nothing at all), so neither reaches this check and both
    #  still report the overshoot they are written against.

    #  INVARIANT 3: THE CAP MAY PENALISE AND MAY NEVER RESCUE.  This is the CEO's 2026-09-01
    #  ruling asserted as algebra rather than trusted to the candidate mask above it.
    #
    #  WHY IT IS AN ASSERTION AND NOT A LOG LINE.  Every committed truncation replaces a
    #  POSITIVE contribution `x` with a smaller positive `c`, so `agg_delta` is a sum of
    #  strictly negative terms and cannot come out positive in exact arithmetic.  A positive
    #  `agg_delta` therefore means the candidate selection has been broken -- most plausibly
    #  by somebody restoring `A[i] > c` or `np.sign(...) * c` in the loop, which is precisely
    #  the reversion this invariant exists to catch and precisely the shape a reviewer would
    #  read straight past.  That is an ALGEBRA failure, which this module raises on, not a
    #  data condition, which it does not.
    #
    #  THE TOLERANCE IS RELATIVE TO THE NAME'S OWN MASS, deliberately.  `agg_delta` is a
    #  difference of two float64 row sums, so cancellation gives it an error that scales with
    #  `base_before`, not with 1.0 -- an absolute `1e-9` would be far too tight on a
    #  large-base name and far too loose on a tiny one.  Scaled by the base, the shipped
    #  panels sit ~13 orders of magnitude inside it while a genuine sign reversion (the
    #  smallest of the four 2026-09-01 raises was KFY at +0.0019 on a base of ~0.30, i.e.
    #  ~6e-3 relative) is ~10 orders of magnitude OUTSIDE it.  The gap is wide enough that no
    #  choice inside it changes any verdict.
    _rescue_tol = _SHARE_TOL * np.maximum(1.0, handled['base_before'].to_numpy(dtype='float64'))
    rescued = handled[handled['agg_delta'].to_numpy(dtype='float64') > _rescue_tol]
    if len(rescued):
        raise AssertionError(
            'metric_share_cap: %d name(s) were RAISED by the cap (worst agg_delta %+.6f) -- '
            'the cap rescued a name instead of penalising it, which the CEO ruled out on '
            '2026-09-01. A committed truncation only ever replaces a POSITIVE contribution '
            'with a smaller one, so a positive agg_delta is an ALGEBRA failure and not a data '
            'condition: check whether the candidate mask in `apply_share_cap` has been '
            'reverted to the absolute row (`A[i] > c`), or whether `_cap_value` is again '
            'being handed |contributions| instead of signed ones. Offenders: %s'
            % (len(rescued), float(rescued['agg_delta'].max()),
               list(rescued.index[:10])))



    #  THE OVERSHOOT.  This one DOES raise on a row the module tried to cap and failed,
    #  because that is the module being broken rather than the data being awkward -- and
    #  because it is the only thing that detects a cap replaced by a pass-through, which is
    #  a defect no amount of logging would surface.
    #
    #  *** STATED ON `pos_share_after`, NOT ON `share_after`, SINCE THE 2026-09-01 RULING.
    #  `share_after` is the DOMINANT metric's share of the absolute base, and under
    #  positive-only a dominant NEGATIVE contribution is one the cap is forbidden to touch --
    #  so `share_after > cap` is now a legitimate outcome and asserting on it would raise on
    #  correct behaviour.  Worse, it would raise for the exact population the ruling exists
    #  to protect: the four names the symmetric cap RAISED on the 2026-09-01 panel are all
    #  dominant-negative rows.  The post-condition the module actually establishes is
    #  narrower and is the one that ships: no contribution that HELPS a name may exceed `cap`
    #  of its absolute total. ***
    bad = handled[handled['pos_share_after'] > cap + _SHARE_TOL]
    if len(bad):
        raise AssertionError(
            'metric_share_cap: %d name(s) left the cap with a POSITIVE contribution ABOVE it '
            '(worst %.6f > %.6f) and shipped UNCAPPED. `_cap_value` found no consistent k for '
            'a row that needed one. '
            'Two known causes: (a) the fixed-point search is broken or has been replaced by '
            'a pass-through -- the usual case, and the reason this raises; (b) the row has '
            'two contributions that tie EXACTLY at the fixed point, where the strict/'
            'non-strict comparisons in `_cap_value` can both fail by one ulp. (b) IS '
            'reachable at CAP = 0.25 -- but only with EXACT ties, which need a decimal grid: '
            'it has a measured rate of ZERO over 2,000,000 continuous-float rows, 196,000 '
            'jittered real rows and 200 real panel rows, so on live data suspect (a) first '
            'and check whether the row you are looking at has two identical contributions. '
            '(Positive-only NARROWS (b): a tie between a positive and a negative can no '
            'longer trigger it, because a negative is not a candidate.) '
            'Offenders: %s'
            % (len(bad), float(bad['pos_share_after'].max()), cap, list(bad.index[:10])))


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
    lines = ['SINGLE-METRIC SHARE CAP [%s]: cap %.0f%% of a name\'s ABSOLUTE contribution, '
             'POSITIVE contributions ONLY (CEO 2026-09-01: the cap may penalise, never '
             'rescue); BOUND on %d of %d name(s)%s'
             % (pool_label, 100 * cap, len(hit), n,
                ' -- NOTHING was truncated on this panel' if not len(hit) else '')]

    def _src(idx):
        if 'source' in report.columns:
            v = report['source'].loc[idx]
            return v if isinstance(v, str) else str(v)
        return str(idx)

    if len(hit):
        lines.append('    (share is (z x w) / sum|z x w|, NOT of the signed AggScore -- '
                     'see metric_share_cap; the two differ by a factor of ~1.5 on this '
                     'pool. The pair below is the largest HELPING contribution, which is '
                     'what the cap acts on; a name\'s biggest DRIVER can be a penalty, '
                     'and where it is, the line says so and the penalty was left at full '
                     'value.)')
        #  `n_capped` is printed on EVERY line and it is not decoration: the share pair
        #  shown is the DOMINANT metric's only, so without the count a three-metric cascade
        #  (which erases most of a name's mass) is indistinguishable from a one-metric trim,
        #  and the AggScore delta beside it cannot be reconciled with the single pair.
        #  No rank is printed and that is deliberate: this runs BEFORE `getAggScore`, so no
        #  ranking exists yet.  A position printed here would be the scoring frame's row
        #  order, which is not a rank and would be read as one.
        for idx, row in hit.sort_values('pos_share_before',
                                        ascending=False).head(top_n).iterrows():
            #  *** THE `RAISED` LABELS ARE GONE, AND THEIR ABSENCE IS THE 2026-09-01
            #  RULING.  There were three of them and each described something the
            #  SYMMETRIC cap really did: a truncated negative dominant metric, or a net
            #  rise on a mixed-sign k > 1 row.  Under positive-only NONE of those states
            #  is reachable -- every committed truncation lowers a POSITIVE contribution,
            #  so `agg_delta <= 0` on every row, asserted as INVARIANT 3 in
            #  `_assert_post_condition`.  Keeping the branches would leave dead prose that
            #  a future reader would take as evidence the cap can still rescue.  What
            #  replaces them is the note below, which reports the thing that IS now true
            #  and is NOT self-evident: the name's biggest driver was a penalty, and the
            #  cap deliberately did not touch it. ***
            dom_is_penalty = (row['contrib_before'] < 0
                              and row['metric_capped'] != row['metric_before'])
            note = ''
            if dom_is_penalty:
                note = ('  <-- its biggest driver is the PENALTY %s at %.4f of |base|, '
                        'LEFT AT FULL VALUE (positive-only)'
                        % (row['metric_before'], row['share_before']))
            lines.append('    %-12s  %-26s %.4f -> %.4f  n_capped=%d   AggScore %+.4f%s'
                         % (_src(idx), row['metric_capped'],
                            row['pos_share_before'], row['pos_share_after'],
                            int(row['n_capped']), row['agg_delta'], note))
        if len(hit) > top_n:
            lines.append('    ... and %d more (full per-name detail in '
                         "rankdic['share_cap_report'])" % (len(hit) - top_n))
        n_multi = int((hit['n_capped'] > 1).sum())
        lines.append('    total AggScore moved: %+.4f over %d name(s), and EVERY move is '
                     'DOWN or zero BY RULING (%d rose -- any number here other than 0 is a '
                     'defect, see INVARIANT 3); %d had a NEGATIVE dominant contribution '
                     'that was LEFT UNTOUCHED; %d bound on MORE THAN ONE metric (a cascade '
                     '-- the share pair above is the largest HELPING metric only)'
                     % (float(hit['agg_delta'].sum()), len(hit),
                        int((hit['agg_delta'] > 0).sum()),
                        int((hit['contrib_before'] < 0).sum()), n_multi))

    #  --- WHAT THE SYMMETRIC CAP WOULD HAVE RESCUED, NAMED ----------------------------- #
    #  THE POPULATION THIS BLOCK EXISTS FOR is the one the 2026-09-01 ruling took out of
    #  the cap's reach: a name whose biggest driver is a PENALTY over the cap.  Under the
    #  symmetric rule that penalty was truncated and the name's score ROSE -- 4 of the 13
    #  bound names on 2026-08-31 and 4 of the 11 on 2026-09-01, and the largest single
    #  effect the cap had anywhere on either panel.  They now ship with the penalty at
    #  full value, which makes them INVISIBLE in the block above (nothing bound on them)
    #  at exactly the moment a reader comparing this run against a pre-ruling one most
    #  needs to see them.
    #
    #  NAMED AND NOT MERELY COUNTED, and the reason is a disclosure that would otherwise
    #  have been LOST rather than merely changed.  `PET.TO` on the saved 2026-08-11 panel
    #  is the case that forced this: two non-zero contributions, both negative, so the old
    #  count-based feasibility test declined it as `infeasible` and the UNCAPPABLE block
    #  below printed its name.  Under positive-only there is simply nothing to act on, so
    #  it is an ordinary `status = 'ok'` row -- its score is byte-identical either way, but
    #  it is a name sitting at 75% of its base on ONE penalty metric and shipping uncapped,
    #  which is exactly the fact the uncappable block existed to surface.  A change of
    #  status must not quietly cost a name its line in the log.
    #
    #  THE FILTER IS `share_before > cap`, deliberately, and it is what keeps this list
    #  short and decision-relevant: every name in it is one the symmetric rule WOULD have
    #  bound on and raised.  A penalty-dominated name UNDER the cap was never touched by
    #  either rule and is not news.
    _unreached = report[(report['status'] == STATUS_OK) & (report['n_capped'] == 0)
                        & (report['contrib_before'] < 0)
                        & (report['share_before'] > cap)]
    if len(_unreached):
        _npos = int(_unreached['pos_share_before'].isna().sum())
        lines.append('    *** %d name(s) are DOMINATED BY A PENALTY over %.0f%% and were '
                     'DELIBERATELY NOT CAPPED (positive-only, CEO 2026-09-01): the '
                     'pre-ruling symmetric cap would have truncated that penalty and '
                     'RAISED these names. %d of them have NO positive contribution at '
                     'all. This is the ruling working, not a finding -- but they ship '
                     'single-metric-dominated, so ALL %d are named here and in %s: %s'
                     % (len(_unreached), 100 * cap, _npos, len(_unreached),
                        SHARE_CAP_CSV % '<date>',
                        ', '.join(_src(i) for i in _unreached.index)))

    #  --- THE UNCAPPABLE ROWS.  THIS BLOCK IS THE MITIGATION, NOT A FOOTNOTE ------------
    #  These names ship with the score they would have had if this module did not exist.
    #  Nothing else in the run says so, because nothing raises: if this block is quiet or
    #  gets truncated away, the module is back to hiding exactly what it was built to expose.
    _WHY = {
        STATUS_INFEASIBLE: ('at most %d contribution(s), ALL OF THEM POSITIVE, so there '
                            'is no untruncatable tail left to be a share of -- the fixed '
                            'point would zero the whole row and invent an AggScore of '
                            '0.0. Positive-only makes this a JOINT condition: one '
                            'negative contribution can never be truncated, so it holds '
                            'the tail up and the row becomes feasible'
                            % _k_max(cap)),
        STATUS_WOULD_ERASE: ('the fixed point would leave under %.0f%% of the name\'s '
                             'absolute mass -- that is erasing the score, not compressing '
                             'it. Since the positive-only ruling such a row is almost '
                             'entirely POSITIVE mass, so what is declined here is a '
                             'fabricated maximal DEMOTION, not the fabricated promotion '
                             'the symmetric rule produced'
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
#  *** THE TWO ONE-LINE ADDITIONS OUTSIDE THIS MODULE HAVE BOTH LANDED (verified on disk,
#  2026-09-03).  THIS NOTE PREVIOUSLY SAID THEY WERE STILL NEEDED AND THAT THE ARTIFACT
#  "exists only on the run machine", WHICH IS NOW FALSE PROSE OF EXACTLY THE KIND THIS FILE
#  KEEPS A TEST FOR:
#      Sbocker.allowlist_patterns  has  'ShareCapReport_*.csv'  at `Sbocker.py:161`, so the
#          file is transferred rather than written-and-stranded
#      conftest._EVIDENCE_GLOBS    has  'ShareCapReport_*.csv'  at `conftest.py:260` (RULE E,
#          so a test that forgets to redirect the path cannot drop one in the repo root)
#  Both are still NAMED here, because `test_the_evidence_CSV_records_what_it_needs_from_
#  OUTSIDE_this_module` reads this comment: the dependency is real and must stay visible even
#  though it is currently satisfied -- if either line is deleted, the artifact silently stops
#  reaching the CEO and nothing else in this module would notice. ***
SHARE_CAP_CSV = 'ShareCapReport_%s.csv'

#  Header once per process, then append: `postBoScoreRanking` runs ONCE PER POOL, and a
#  per-call overwrite would leave only the last one.  Same idiom, and same reason, as
#  `postBoRank._write_missing_csv`.  Only the general pool is capped today, so the append
#  matters the moment `CAPPED_POOLS` grows -- which is a decision, not an accident.
_CSV_STARTED = set()

#  Column order is the READING order, not the frame order: who, then what happened to them,
#  then the numbers behind it.  `pool` leads because the file carries every capped pool.
CSV_COLUMNS = ('pool', 'source', 'status', 'n_capped',
               #  the name's biggest DRIVER (either sign) -- what the name rests on
               'metric_before', 'contrib_before', 'share_before', 'share_after',
               #  the largest HELPING contribution -- what the cap is allowed to act on, and
               #  the pair the post-condition is stated on.  A row where `metric_capped`
               #  differs from `metric_before` is a name whose dominant metric is a PENALTY
               #  the cap deliberately left alone (CEO ruling, 2026-09-01).
               'metric_capped', 'pos_share_before', 'pos_share_after',
               'cap_value', 'base_before', 'base_after', 'agg_delta')


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
