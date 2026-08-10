"""stage1_veto.py  --  THE STAGE-1 RED-FLAG VETO LAYER  (CEO, 2026-08-05)

WHAT IT IS, AND HOW IT DIFFERS FROM THE GATE IT SITS ON
-------------------------------------------------------
Stage-1 proper is a WEIGHTED PASS-RATE SUM: every criterion contributes `w x (passes/8)`, so a
name that fails one criterion outright loses that criterion's weight and can still make the
top-100 on the strength of the others.  For most criteria that is the right shape -- they are
QUALITY readings and they trade off.

Five of them are not.  They are SOLVENCY / EARNINGS-REALITY conditions where failing
persistently is not a lower score, it is a disqualification:

    returnOnAssets     > 0      is it profitable on its asset base at all
    CFOlessEarnings    > 0      is the profit cash rather than an accrual
    uCurrentRatio      > 1      can it pay its current liabilities
    netDebtToEBITDA    (rule)   is the leverage serviceable      (three-branch, calcMetrics)
    uInterestCoverage  > 1      does operating profit cover the interest bill

`k >= 1` ejects: ONE persistent red flag is enough.  That is the CEO's decision and it is the
whole design -- a veto that needed two flags would be a second scoring layer wearing a
disqualification's name.

FAIL IS DEFINED AS `PASSED <= 1 OF THE NEWEST 8 ROWS`, NOT `0 OF 8`
-------------------------------------------------------------------
Deliberate, and the reason is a data property rather than a leniency: a single bad vendor print
-- one restated quarter, one mis-signed line -- must not be able to eject a name on its own.
Requiring `0 of 8` would make the veto maximally sensitive to exactly the noise this repo keeps
finding in the feed.  `<= 1 of 8` still means "essentially never passes over two years" while
tolerating one bad row.  It is a THRESHOLD, so it is stated here once and named
`FAIL_MAX_PASSES`; do not inline it.

...AND THAT DEFINITION IS ONLY APPLIED ON 8 ROWS OF *EVIDENCE*  (register C-15, CEO 2026-08-06)
------------------------------------------------------------------------------------------------
`FAIL_MAX_PASSES = 1` is an ABSOLUTE count, so on its own it is nonsense on a short window: a
source with ONE row that PASSED that row scores `passes = 1 <= 1` and would fail -- on ALL FIVE
flags at once -- so a 100% pass rate would be read as a persistent red flag and the name ejected.
A source must never be ejected BECAUSE its window is short.

A PROPORTIONAL rule (fail when `passes/rows <= 1/8`) keeps a verdict on every window length, but
it is the wrong reading of this threshold and it does not fix the real problem.  `<= 1` is not a
rate -- its stated justification is an ABSOLUTE allowance for ONE bad vendor print, which does
not scale down.  Worse, proportional would still eject a 2-row source that failed both rows: 6
months of data used to make a claim worded as "essentially never passes over TWO YEARS".  The
defect is not the arithmetic, it is EVIDENCE SUFFICIENCY.  So the rule is a floor -- but a
PER-FLAG one, on EVIDENCE rather than on rows:

    a flag may FAIL only when it has `WINDOW_ROWS` rows of COUNTABLE EVIDENCE and passed at
    most `FAIL_MAX_PASSES` of them.  Otherwise it ABSTAINS -- it neither fails nor passes,
    and it is excluded from the `k` count.

WHY PER-FLAG, AND WHAT THE OLD PER-SOURCE FLOOR GOT WRONG.  The previous version counted ROWS,
not rows of evidence, and the module said so and flagged it as an unresolved tension: "a source
with a full 8-row window whose `uInterestCoverage` is NaN in 7 of them can still fail on 1 row of
real evidence".  That tension is unresolvable with a BLANKET rule, because the two candidate
blanket rules are each right on some flags and catastrophic on others -- and this was MEASURED,
not argued:

    with `EJECT_MIN_FLAGS = 1`, `uInterestCoverage` un-abstained fails 2,856 sources (36.8% of
    the universe); abstained it fails 1,188 (15.3%).  1,668 sources -- 21.5% OF THE UNIVERSE --
    would be EJECTED OUTRIGHT FOR HAVING NO DEBT.

That is the largest defect in this layer as it was coded, and it is a pure artifact: FMP reports
`interestExpense` as 0 for a debt-free name, `calcMetrics`'s `interest_expense_positive` guard
correctly refuses the row (a debt-free company does not HAVE a coverage ratio), NaN is not a pass
-- and the veto then read "cannot service its interest" off a company with no interest to service.

So the rule is stated PER FIELD, because "what does a refused row MEAN" is a property of the
field and of nothing else:

  * ADVERSE / MOOT field -- EVERY ROW COUNTS, and a refused row counts as a NON-PASS.  This is
    exactly the current behaviour, so four of the five flags are BIT-IDENTICAL to before.
  * BENIGN field -- only ADMISSIBLE rows count, toward both the evidence floor and the passes.

`MIN_WINDOW_ROWS` FALLS OUT AS A SPECIAL CASE and is DELETED rather than kept alongside.  A
5-row source has at most 5 rows of evidence on every flag, so every flag abstains and it cannot
be ejected -- today's behaviour DERIVED from the evidence rule instead of stated separately
beside it.  Two rules that happen to agree are a rule and a coincidence waiting to diverge.

WHAT THIS COSTS, STATED PLAINLY.  A source with 7 rows of unbroken red flags is NOT ejected.
That is a chosen FALSE NEGATIVE, and the asymmetry is why: ejection is absolute and removes a
name from the deliverable outright, while a NON-ejected bad name still has to survive the whole
weighted Stage-1 score, the carve-out and the top-100 cut.  Wrongly ejecting on thin evidence is
the more expensive error for a disqualification layer.
ABSTAINING IS A DIFFERENT DECISION FROM FAILING, AND IT OPENS ITS OWN HOLE: a short-history
source is UN-VETOABLE on that flag, so a young company can be hard to disqualify at all.  That is
related to but NOT a resolution of register C-7 (unequal history length across sources) -- C-7 is
about whether unequal history biases the SCORE, and it stays OPEN.

ABSTENTIONS ARE COUNTED PER FLAG, NOT PER SOURCE, and the change of unit is the point: a source
that abstained on `uInterestCoverage` because it is DEBT-FREE and a source that abstained on
everything because its history is short are different facts, and a per-source count cannot tell
them apart -- it would report "1,668 names not fully evaluated" and hide that they are all one
flag -- though NOT, as this paragraph once implied, all for one benign reason: see the
JUDGE-AND-PENALISE section below.  A veto that silently declined to evaluate a fifth of a pool
would otherwise be indistinguishable from one that found it clean.  See
design/stage1-veto-decisions.md.

BEHIND A FLAG -- NOW ON, AND WHAT CARRIES THE VISIBILITY INSTEAD  (CEO, 2026-08-07)
-----------------------------------------------------------------------------------
`ENABLED = True`.  The CEO turned the layer ON deliberately on 2026-08-07, for the GENERAL POOL
ONLY (`VETO_POOLS`), on the evidence measured below: it ejects 58.4% of the general pool but
moves only 5 of the top 100.  Register C-9 (the veto had never been evaluated on real data)
closes on that measurement.

THE DEFAULT USED TO BE `False`, and the reason it was is worth keeping because the flip does NOT
delete the requirement, it MOVES it.  Default-OFF bought two things: an OFFLINE A/B from saved
pickles without touching a shipped run -- still true, `enabled=`/`pools=` override per call and
never mutate module state -- and, the real one, that *nothing ships into the gate silently*: with
the default OFF, turning it on was a visible single-line event.  DEFAULT-ON GIVES THAT UP, so
the visibility is now carried by the ARTIFACT rather than by the default: `postBo` stamps
`stage1_veto` into `RunProvenance-<date>_<ds>_<filter>.json` -- enabled, pools, the three
parameters and the per-pool in/ejected/out counts -- from the RUN'S OWN reports, not from this
module's constants.  A run therefore states which veto regime produced its top-100, which is the
property that actually mattered; a silent flip is now detectable from the deliverables alone,
which it never was under default-OFF.  IF YOU EVER MOVE THIS FLAG AGAIN, CHECK THAT STAMP STILL
FIRES -- it, not the default, is what stops two runs with different pools looking identical.

Override per call by passing `enabled=` to `apply_veto` (an explicit argument always wins over
the module default, so a research script never has to mutate global state), or by assigning the
module attribute (`import stage1_veto as sv; sv.ENABLED = False`).

WHERE IT RUNS -- THE GENERAL POOL ONLY  (CEO, 2026-08-07; supersedes the all-six-pools rule)
--------------------------------------------------------------------------------------------
BEFORE the `head(100)` cut, on the GENERAL POOL ONLY.  It GATES the pool and the survivors are
then ranked -- placement before the cut is what makes it a veto rather than a shortlist trim.
`VETO_POOLS` names the pools it may run on; on any other pool `apply_veto` is a NO-OP that
returns its input unchanged and reports `applies=False` with the reason.  Per-pool reports are
still emitted for every pool and ejection counts are LOGGED, because a veto that silently
removed a third of a cohort would be indistinguishable from one that removed nobody -- and
"out of scope here" must be distinguishable from "found nothing".

THE PREVIOUS RULE, AND WHY IT WAS OVERRULED -- KEPT, NOT DELETED.  This module used to run on
all six pools and defended it in these words: *"applied to the general pool alone it would say
a red flag matters less in a cohort.  A red flag is a red flag."*  That principle is CORRECT IN
INTENT -- a disqualification must not be softened because a name sits in a cohort -- and it was
WRONG ABOUT WHAT THE FLAGS MEASURE in cohorts where two of them are STRUCTURALLY UNDEFINED.
`uCurrentRatio > 1` and `netDebtToEBITDA` (rule) presuppose a working-capital balance sheet and
serviceable-from-EBITDA leverage.  A REIT carries mortgage debt at 5-8x EBITDA BY DESIGN and
holds essentially no current assets; a bank's balance sheet is its business, not its financing.
On those cohorts the flags are not a strict reading of a red flag, THEY ARE THE WRONG QUESTION --
so the old argument never applied, because there was no red flag to be strict about.

MEASURED, on the 2026-08-07 run with the veto evaluated OFFLINE against the saved pickle:

    pool                in    ejected      %
    general           1545        902   58.4
    REIT                49         47   95.9   <-- the tell
    BalanceSheetFin    121         87   71.9
    Mining             204        105   51.5
    FinManager          42         19   45.2
    InvestmentVehicle   19         11   57.9

95.9% of REITs is not the veto finding bad REITs; it is the veto reporting that REITs are REITs.
It is ALSO NOT A THRESHOLD PROBLEM -- no level of `netDebtToEBITDA` makes that flag a solvency
reading on a mortgage vehicle -- so do not "fix" it by loosening a bar per cohort.

THAT OPEN ISSUE IS NOW CLOSED -- PER-COHORT FLAG SETS  (CEO, 2026-08-07, same day)
-----------------------------------------------------------------------------------
The resolution the paragraph above asked for exists: `POOL_FLAGS` gives each cohort the flags
that MEAN something on its balance sheet, and `VETO_POOLS` is now
`('general', 'REIT', 'Mining', 'FinManager', 'BalanceSheetFin')`.  Three of the five change a
cohort's list; `InvestmentVehicle` alone stays out of scope.

THE FIX IS MOSTLY SUBTRACTIVE, WHICH IS THE FINDING.  Removing the structurally-undefined flags
ALONE took REIT from 97.0% ejected to 23.9% and FinManager from 44.2% to 3.8%.  The cohorts were
never the problem; the flag set was.  Everything added back is a unity or sign test on a field
that is defined there -- no cohort-specific THRESHOLD exists anywhere in this module.

THE FIVE STRUCTURALLY-UNDEFINED FLAGS, WITH THEIR MEASURED FALSE POSITIVES.  The register above
named two.  There are five, and the three that were missing are missing because nobody had
looked at WHICH names the flag ejected:

  1. `uCurrentRatio` on the leveraged-vehicle and bank cohorts -- presupposes a working-capital
     balance sheet.  A REIT holds essentially no current assets.
  2. `netDebtToEBITDA` on the same -- a REIT carries mortgage debt at 5-8x EBITDA BY DESIGN; a
     bank's balance sheet is its business, not its financing.
  3. `returnOnAssets` ON REIT -- ejects Icade, Piedmont, Macerich, Centerspace.  A US REIT
     depreciates APPRECIATING buildings and an IFRS REIT books revaluation LOSSES, so negative
     net income is routine and is not a solvency event.  (It is NOT undefined on the banks or on
     FinManager, where it is kept and is the whole of the BalanceSheetFin set.)
  4. `CFOlessEarnings` ON REIT -- ejects CTP N.V., NEPI Rockcastle, Mainstreet Equity.  Their net
     income CONTAINS unrealised revaluation gains, so CFO < NI is arithmetic, not accrual abuse.
  5. `CFO > 0` ON INVESTMENTVEHICLE -- ejects 5 of 15, INCLUDING 3 OF THE TOP-25.  Under ASC 946
     an investment company's PORTFOLIO PURCHASES are an OPERATING cash flow, so CFO measures
     which way the portfolio is moving, not solvency: a fund deploying capital reads as a fund
     burning cash.  This is why `InvestmentVehicle` gets no flag set rather than a smaller one.

THE PROVENANCE OF THE ONE FLAG THAT IS NOT A PURE UNITY TEST.  `cashRunwayOneYear` is
`cash + CFO x rpy > 0` -- can the company fund a year at its current burn.  The HORIZON is not a
chosen level: IAS 1.25 and ASC 205-40 both require management to assess going concern over AT
LEAST TWELVE MONTHS, so twelve months is the statutory horizon and the flag inherits it.  `rpy`
(4 quarterly, 2 semi-annual) is what makes it twelve months for a semi-annual filer too, rather
than twelve months for one filer and six for another.

WHAT THE COHORT FLAGS COST TODAY, STATED PLAINLY: FinManager ejects 2 of 52 and one of them
(`PREVA.AS`) IS IN THE CURRENT TOP-25 -- this change moves a shipped list.  BalanceSheetFin
ejects 3 of 125 (two shells and MBIA in runoff), REIT 2 of 67, Mining 22 of 218 (the DESIGNED
three-flag set -- see the 2026-08-08 revert note in `POOL_FLAGS['Mining']`; this figure is a
PREDICTION and is not verifiable on any saved panel, because no saved panel carries those three
columns); none of those
three touch a top-25.

THE FOUR NEW PANEL COLUMNS NOW EXIST, ON THEIR OWN CHANNEL -- AND NO SAVED PANEL CARRIES THEM.
`reitEbitdaInterestCoverage`, `producerEbitdaPositive`, `cashRunwayOneYear` and `equityPositive`
are declared in `createDicts.BoMetric_veto_dict`, computed by `calcMetrics.calc_veto` and written
by `getData_fmp.build_bometric_rows` at FETCH time.  THEY ARE NOT IN `BoMetric_special_dict` AND
MUST NOT BE MOVED THERE: every entry in the five SCORING dicts carries a `Tier` and a `Sign` and
`calcScore.calcByTier` scores it, so four veto columns there would silently add FOUR WEIGHTED
STAGE-1 CRITERIA TO EVERY POOL, general included.  `createDicts._assert_veto_never_scored` checks
the two key sets are disjoint at import.

They are built from `ebitda` and `cashAndCashEquivalents`, which were only CAPTURED from the
2026-08-05 preReq change onward, so they exist on a panel fetched after that change and ON NO
EARLIER ONE -- NOTHING HERE IS BACKTESTABLE.  On today's panel REIT and Mining therefore report
`applies=False` with `missing_columns` set and are UN-VETOED until the next full fetch; FinManager
and BalanceSheetFin use only existing columns and are live now.  That asymmetry is REPORTED per
pool, not inferred -- see `_STALE_PANEL_NOT_APPLICABLE`.  When a raw input is absent the builder
OMITS the column rather than emitting an all-NaN one, deliberately: an all-NaN column is PRESENT,
so it would pass `missing_columns`, abstain on every row and report the cohort as gated-and-clean.

ON THE GENERAL POOL THE CHANGE IS SMALL: the veto moves only 5 of the top 100 (95% overlap) --
ejecting UHS, PEY.TO, SBH, 215200.KQ, TCL-A.TO and promoting 000270.KS, ATH.TO, DRW3.DE,
HCO.PA, LEGH.  A defensible cleanup, not a rewrite of the deliverable.

JUDGE ON AVAILABLE ROWS **AND** PENALISE THE GAP  (CEO, 2026-08-10)
-------------------------------------------------------------------
The evidence floor above is no longer ALL-OR-NOTHING, and the abstention is no longer FREE.
Two changes, one ruling:

  1. A flag with at least `PARTIAL_MIN_EVIDENCE_ROWS` (6) rows of countable evidence is
     JUDGED on the rows it has.  Below that it still abstains entirely -- a name with one
     usable row must not be ejected on that row.  The number is argued at its definition.
  2. EVERY data gap this layer meets is charged to the AD-HOC PENALTY BUCKET
     (`adhoc_penalty`) -- a SOFT veto that lowers the name's Stage-2 `AggScore` before the
     sort instead of removing it.  Two checks: a short PANEL (fewer than eight rows at all,
     charged once per source) and REFUSED ROWS within the rows that exist (charged per flag).

WHAT IS CHARGED IS DECIDED PER **ROW**, AGAINST A RAW INPUT (reviewer S1, 2026-08-10).  A
refused row is charged iff the row's own corroborating field says the refusal cannot be
structural: a zero `interestExpense` on a row reporting positive `totalDebt`, or an ABSENT
revenue figure behind the pre-production reading.  See `REFUSAL_CORROBORATOR` for the table,
the per-field rulings, and the measurement that forced this shape.

THE RULE THIS REPLACED WAS NON-MONOTONE AND WAS LIVE IN THE SHIPPED 100.  It read the SHAPE
of the refusals instead -- charging a PARTIALLY refused window and giving a FULLY refused one
a free pass on the reading that it meant a debt-free company.  So the charge climbed to -0.07
at seven missing rows of eight and fell to 0.00 at eight of eight: THE WORST DATA PAID LEAST.
`AAPL` sat in the free-pass set with a median `totalDebt` of $97.5bn.  C-15 is still
preserved, but by the operand rather than by the window: a name with no debt is still not
charged for having no coverage ratio, and a name with debt no longer escapes by having ALL its
rows broken instead of some.  See also `FIELD_EVIDENCE['uInterestCoverage']`, which carries the
correction of the "i.e. DEBT-FREE names" claim that used to justify the abstention ruling.

WHAT IT IS NOT -- `psbrfilter`
------------------------------
`postBo.psbrfilter` (postBo.py) is a DIFFERENT mechanism and is deliberately untouched: it is a
-1.5 cutoff on Stage-2 `z x w` columns, so its threshold means a different bar per metric, and it
is inert (stored in resdic, wired to nothing) pending its own soundness review.  This layer is
not a revival of it and must not be merged with it.
"""

import numpy as np
import pandas as pd

#  --- THE FLAG.  ON since 2026-08-07 (CEO), general pool only.  See the module docstring for
#  what replaced default-OFF as the "nothing ships into the gate silently" guarantee: the
#  `stage1_veto` block in RunProvenance-*.json, written from the run's own per-pool reports.
ENABLED = True

#  --- WHICH POOLS IT MAY RUN ON (CEO, 2026-08-07, SECOND RULING) -------------------------
#  FIVE OF THE SIX POOLS.  The previous ruling (general alone) was correct about the DEFECT and
#  wrong about the REMEDY: the two structurally-undefined flags are the problem, not the idea of
#  vetoing a cohort.  So the fix is SUBTRACTIVE -- each cohort carries the flags that MEAN
#  something on its balance sheet, and carries no flag that does not.  Removing the two undefined
#  flags ALONE took REIT from 97.0% ejected to 23.9% and FinManager from 44.2% to 3.8%, which is
#  the measurement that shows the cohorts were never the problem.
#  Still a SET rather than a boolean, for the same reason as before: `apply_veto` reports
#  `applies=False` BY NAME, so "out of scope" and "found nothing" never look the same.
#  `InvestmentVehicle` is the ONE pool still out of scope -- see `NOT_APPLICABLE_REASONS`.
VETO_POOLS = ('general', 'REIT', 'Mining', 'FinManager', 'BalanceSheetFin')

#  --- THE THREE PARAMETERS, EACH A STATED DECISION --------------------------------------
#  Rows of the newest-first per-source window a flag is evaluated over.  8 is Stage-1's own
#  scoring window (`calcScore.calcByTier`'s head(n)); using a different one would mean the veto
#  and the score disagreed about what "recently" means.
WINDOW_ROWS = 8
#  A flag FAILS when it passed at most this many of `WINDOW_ROWS` rows of EVIDENCE.  1, not 0 --
#  see the docstring.
FAIL_MAX_PASSES = 1

#  --- THE PARTIAL-WINDOW FLOOR (CEO, 2026-08-10) ----------------------------------------
#  "JUDGE ON AVAILABLE ROWS **AND** PENALISE THE GAP."  Until this ruling the evidence floor
#  was all-or-nothing at `WINDOW_ROWS`: a flag with seven usable rows of eight ABSTAINED, and
#  the name passed it unchecked.  Now a flag with at least `PARTIAL_MIN_EVIDENCE_ROWS` rows is
#  JUDGED on the rows it has, and the missing rows go to the ad-hoc penalty bucket instead of
#  being forgotten (`adhoc_penalty`).  Below the floor it still abstains ENTIRELY -- and the
#  gap is still bucketed.
#
#  WHY 6, AND NOT 1 / 2 / 8 -- the number is a decision, so it is argued rather than picked:
#    * `FAIL_MAX_PASSES` is an ABSOLUTE allowance for ONE bad vendor print, so the pass RATE a
#      failure claims depends on the window: <=1 of 8 is <=12.5%, <=1 of 6 is <=16.7%, <=1 of 2
#      is <=50%.  6 keeps the failure claim within ~1.33x of the full-window claim, i.e. the
#      SAME KIND of statement.  At 2 rows it is a different statement wearing the same words,
#      which is exactly the objection the C-15 docstring above raises against the proportional
#      rule and it is not weakened by this ruling.
#    * 6 quarterly rows is 18 MONTHS.  The failure wording is "essentially never passes"; over
#      18 months that is still a PERSISTENCE claim.  Over 2 rows (6 months) it is not.
#    * EJECTION IS ABSOLUTE and removes a name from the deliverable, so the asymmetry the
#      docstring states -- wrongly ejecting on thin evidence is the more expensive error --
#      argues for a HIGH floor, and the bucket now carries the cost of the abstention that a
#      high floor leaves behind.  That is the trade the ruling makes: the thin-evidence names
#      are no longer un-penalised, so the floor no longer has to do all the work.
#  MEASURED on the 2026-08-10 CUR3K panel, general pool (1,388 in / 799 ejected today), as the
#  NEW ejections a floor of `m` would add: m=7 -> 10, m=6 -> 12, m=5 -> 14, m=4 -> 15,
#  m=3 -> 22, m=2 -> 27, m=1 -> 40.  All of them `uInterestCoverage`; no other flag has a
#  partial window on this panel.  So 6 buys 12 ejections of the 303 mixed-window names -- a
#  conservative reading of the ruling, and the rest are handled by the bucket rather than by
#  the gate.
PARTIAL_MIN_EVIDENCE_ROWS = 6

#  NO `MIN_WINDOW_ROWS` -- DELETED AT C-15, DELIBERATELY.  The per-source row floor is now a
#  DERIVED consequence of the per-flag evidence floor (a 5-row source has <= 5 evidence rows on
#  every flag, so every flag abstains).  Do not re-add it: a second rule that merely agrees with
#  the first is the pair that silently diverges later.
#  Failed flags needed to eject.  1 -- one persistent red flag is a disqualification.
EJECT_MIN_FLAGS = 1

#  --- THE BUCKET CONTRIBUTIONS THIS LAYER RAISES (CEO, 2026-08-10) -----------------------
#  Named so the evidence CSV's `check` column is a fixed vocabulary rather than an f-string,
#  and so a reader can grep for the mechanism that penalised a name.
#  TWO checks, not three, and the split is by CAUSE rather than by consequence: a source that
#  has fewer than eight rows AT ALL is a different data problem from a source with eight rows
#  on which the vendor refused some.  Whether the flag was then JUDGED on the partial window or
#  ABSTAINED is carried in the `reason` text, not in a third check name -- the points already
#  separate them (an abstention has a bigger gap by construction, so it costs more).
CHECK_SHORT_PANEL = 'stage1_veto:short_panel'
CHECK_REFUSED_ROWS = 'stage1_veto:refused_rows'
#  Not a charge -- a declared HOLE in the measurement.  Carries 0 points and rides the CSV's
#  `NOT_MEASURED` section, so an unreachable corroborator is visible in the artifact a reader
#  opens rather than only on stdout (reviewer R3).
CHECK_UNCORROBORATED = 'stage1_veto:corroborator_unavailable'

#  Reserved key in `_evaluate`'s `gaps` -- NOT a source.  Carries the flags whose corroborating
#  raw column could not be reached, so `apply_veto` can report a MISSING MEASUREMENT instead of
#  presenting "charged nothing" as "found nothing".  A ticker cannot collide with it: no symbol
#  contains a space.
_UNCORROBORATED_KEY = '__uncorroborated flags__'

#  Prefix the corroborating raw columns are joined under, so they can never collide with (or be
#  silently suffixed beside) a veto column of the same name.  See `_corroborator_frame`.
_CORROBORATOR_PREFIX = '__corroborator__'

#  --- THE FIVE FLAGS: BoMetric_df column -> the row-level pass condition -----------------
#  Each condition is stated on the COLUMN AS THE PANEL CARRIES IT, so no arithmetic is
#  re-implemented here and no threshold is restated:
#    * `returnOnAssets` is the base column (netIncome/totalAssets since I-5), tested `> 0`.
#    * `CFOlessEarnings` is CFO - netIncome, tested `> 0`.
#    * `uCurrentRatio` is the RATIO itself (the `u` prefix is the column's name, not a
#      transform), so the unity bar is `> 1`.
#    * `netDebtToEBITDA` is the THREE-BRANCH VERDICT column, already positive-means-pass -- the
#      rule lives in `calcMetrics.net_debt_three_branch` and is NOT duplicated here.
#    * `uInterestCoverage` is operatingIncome/interestExpense, so the bar is `> 1`.
#  NaN is NOT a pass in any of them (`> x` is False for NaN), which matches
#  `calcScore.calcByTier`'s NaN-scores-as-a-fail ruling -- the veto and the gate read a missing
#  row the same way, deliberately.
FLAGS = {
    'returnOnAssets':    lambda s: s > 0,
    'CFOlessEarnings':   lambda s: s > 0,
    'uCurrentRatio':     lambda s: s > 1,
    'netDebtToEBITDA':   lambda s: s > 0,
    'uInterestCoverage': lambda s: s > 1,
}

#  --- THE PER-POOL FLAG SETS (CEO, 2026-08-07) ------------------------------------------
#  A pool's entry is the flag set the veto is CLAIMING IS DEFINED on that balance sheet, and
#  nothing else.  `general` IS `FLAGS` -- the same object, not a copy, so the general pool can
#  never drift from the five-flag set the docstring above describes.
#
#  THE CHANGE IS MOSTLY SUBTRACTIVE, AND THAT IS THE FINDING.  Dropping `uCurrentRatio` and
#  `netDebtToEBITDA` -- the two flags the previous ruling identified as structurally undefined --
#  is on its own most of the correction: REIT 97.0% -> 23.9% ejected, FinManager 44.2% -> 3.8%.
#  The added flags are the small half: they restore a SOLVENCY reading to cohorts that would
#  otherwise be gated on profitability and accruals alone.
#
#  EVERY BAR HERE IS A UNITY TEST OR A SIGN TEST.  No percentile, no chosen level.  A cohort-
#  specific THRESHOLD would be the "re-tuned copy of this one" the previous ruling forbade; a
#  cohort-specific FIELD is what it asked for.
#
#  SIGN-SAFETY IS ENFORCED IN THE COLUMN, NOT HERE.  Each ratio flag's denominator is restricted
#  to `> 0` by the admissibility gate that BUILDS its panel column, exactly as
#  `interest_expense_positive` does for `uInterestCoverage`; an inadmissible row arrives here as
#  NaN.  So no condition below can invert, and none of them re-states a domain.
POOL_FLAGS = {
    'general': FLAGS,

    #  REIT.  Gone: `uCurrentRatio` (a REIT holds essentially no current assets) and
    #  `netDebtToEBITDA` (mortgage debt at 5-8x EBITDA is the DESIGN, not a red flag).
    #  ALSO GONE, and this is the part the earlier docstring did not know -- both were MEASURED
    #  to be structurally undefined here, see the five-flag register below:
    #    `returnOnAssets`  -- a US REIT depreciates appreciating buildings and an IFRS REIT books
    #                         revaluation losses, so negative net income is routine, not a
    #                         solvency event.  Ejected Icade, Piedmont, Macerich, Centerspace.
    #    `CFOlessEarnings` -- net income CONTAINS unrealised revaluation gains, so CFO < NI is
    #                         arithmetic rather than accrual abuse.  Ejected CTP N.V.,
    #                         NEPI Rockcastle, Mainstreet Equity.
    #  WHAT REPLACES THEM is the one solvency question a rent-collecting leveraged vehicle DOES
    #  answer: does the rent cover the interest bill.  2 of 67 fail (0.65x and -1.29x), 18
    #  abstain, 0 of the top-25 move.
    'REIT': {
        'reitEbitdaInterestCoverage': lambda s: s > 1,
    },

    #  MINING -- THE THREE DESIGNED FLAGS, AND ONLY THOSE (CEO, 2026-08-08).
    #
    #  These three were the specialist's design: they PARTITION the cohort by construction.
    #  `producerEbitdaPositive` asks whether a PRODUCING miner earns money at its own cost
    #  curve and ABSTAINS on a pre-revenue explorer (there is no cost curve to be wrong
    #  about); `cashRunwayOneYear` is the flag that judges exactly those explorers; and
    #  `equityPositive` is the balance-sheet floor under both.  Predicted union 22 of 218,
    #  0 of the top-25.
    #
    #  REVERTED 2026-08-08.  `returnOnAssets`, `CFOlessEarnings` and `uInterestCoverage`
    #  were added here by a dispatch brief and were NEVER PART OF THE DESIGN.  MEASURED on
    #  the 2026-08-07 CUR3K panel (277 Basic-Materials sources), those three ALONE eject
    #  **89 of 277 = 32.1%** -- `returnOnAssets` 80, `uInterestCoverage` 45,
    #  `CFOlessEarnings` 2 (overlapping) -- against a design that predicted 22 of 218
    #  (10.1%) for the WHOLE set.  So the three additions accounted for the entire
    #  overshoot, and they do it for a reason the design already anticipated: an
    #  exploration-stage miner has no earnings and no interest cover BY DEFINITION, so
    #  `returnOnAssets > 0` and `uInterestCoverage > 1` are structurally undefined on the
    #  half of this cohort that is pre-production -- exactly the failure mode the
    #  five-flag register documents for REITs, reproduced in a new cohort.
    #
    #  IT IS BACKTESTABLE, AND IT HAS NOW BEEN BACKTESTED (corrected 2026-08-09).
    #  This block used to say "NOT BACKTESTABLE ... NO SAVED PANEL CARRIES THEM ... the
    #  22-of-218 prediction therefore CANNOT be verified offline".  THAT WAS FALSE, and it
    #  conflated two different things: the DERIVED columns are indeed absent from
    #  `BoMetric_df` (checked: all four veto keys absent from both `BoMetric_df` and
    #  `cdx_df` on the 2026-08-07 CUR3K panel), but every RAW INPUT they are built from is
    #  present on that panel's `cdx_df` -- `ebitda`, `cashAndCashEquivalents`,
    #  `netCashProvidedByOperatingActivities`, `totalStockholdersEquity`, `revenue` and
    #  `interestExpense`, all six verified.  A missing derived column is a rebuild, not a
    #  re-fetch.  The claim survived because nobody tried it.
    #
    #  MEASURED 2026-08-09 by rebuilding the columns with the PRODUCTION formulas
    #  (`calcMetrics.calc_veto`, with each key's declared `Guard`, `rpy` per source from
    #  `reporting_period.rows_per_year_by_source`) over the EXACT row set the saved panel
    #  kept, then running this module's own `_evaluate`:
    #
    #    MINING   30 of 277 Basic-Materials sources = 10.8%   against a design that
    #             predicted 10.1%.  Per flag: `producerEbitdaPositive` 14,
    #             `cashRunwayOneYear` 11, `equityPositive` 6 (overlapping).  63 sources
    #             carry at least one abstention -- the pre-revenue explorers the design
    #             says `producerEbitdaPositive` should refuse.
    #    REIT      3 of 76 Real-Estate sources = 3.9%, all on
    #             `reitEbitdaInterestCoverage`; 21 sources abstain.
    #
    #  So the designed set lands where it was designed to land, and the 32.1% overshoot was
    #  entirely the three reverted additions -- that is now a MEASUREMENT rather than an
    #  inference from the arithmetic.
    #
    #  WHAT IS STILL TRUE, AND MUST NOT BE READ AS WEAKENED BY THE ABOVE: a LIVE run on a
    #  panel built before the 2026-08-05 preReq change still reports Mining `applies=False`
    #  with `missing_columns` set, because the live path reads the derived columns off
    #  `BoMetric_df` and does not rebuild them.  Do not read a clean Mining report before
    #  the next full fetch as the veto passing.  The offline rebuild above is a BACKTEST,
    #  not the shipped path.
    #
    #  OPEN, FLAGGED NOT FIXED: `createDicts.BoMetric_veto_dict` states per-flag counts
    #  "of 218" (43 abstain / 11 / 7 / 4).  That denominator matches no cohort measured
    #  here -- Basic Materials on this panel is 277 -- so those numbers are either from a
    #  different population or from the design rather than a measurement.  They are left
    #  alone rather than silently restated, because guessing at their provenance is how a
    #  number acquires a label it does not own.
    'Mining': {
        'producerEbitdaPositive': lambda s: s > 0,
        'cashRunwayOneYear':      lambda s: s > 0,
        'equityPositive':         lambda s: s > 0,
    },

    #  FINMANAGER -- fee-earning asset managers.  NO NEW COLUMN AND NO NEW EVIDENCE RULING: this
    #  entry is three existing panel columns with their existing rulings.  A manager earns fees on
    #  someone else's assets, so `returnOnAssets` and the accrual test read normally; the two
    #  undefined flags are dropped.  Ejects 2 of 52, and one of them (`PREVA.AS`) is in TODAY'S
    #  top-25 -- this is the cohort whose shipped list the change actually moves.
    'FinManager': {
        'returnOnAssets':    lambda s: s > 0,
        'CFOlessEarnings':   lambda s: s > 0,
        'uInterestCoverage': lambda s: s > 1,
    },

    #  BALANCESHEETFIN -- banks and insurers, where the balance sheet IS the business.  ONE FLAG,
    #  deliberately: a bank that cannot earn a positive return on its own asset base is failing at
    #  the only thing its asset base is for.  3 of 125 -- two shells and MBIA in runoff.
    #
    #  NO COVERAGE OR CAPITAL-ADEQUACY FLAG HERE, AND THE NUMBER IS RECORDED SO NOBODY RE-DERIVES
    #  IT: `ebitda / interestExpense > 1` fails 40 OF 125 on this cohort, and the failing set is
    #  RBC, TD, Scotiabank, BMO, CIBC, National Bank, US Bancorp, Huntington, Citizens, ABN AMRO,
    #  ICICI.  Interest expense is a bank's COST OF GOODS, not its debt service, so the ratio is
    #  measuring the wrong thing -- the same class of error as `netDebtToEBITDA` on a REIT, and it
    #  is exactly why the REIT flag is NOT copied here.
    'BalanceSheetFin': {
        'returnOnAssets': lambda s: s > 0,
    },
}

#  --- WHAT A REFUSED ROW MEANS, PER FIELD (register C-15, CEO 2026-08-06) ---------------
#  The evidence floor above needs to know, for each flag, whether a REFUSED row (NaN in the
#  panel column -- a guard refusal or a non-computable input) is EVIDENCE or a GAP.  That is
#  a property of the field, so it is ruled on per field, each ruling SOURCE-VERIFIED against
#  the NaN channel that actually produces the refusal.  `counts` is today's behaviour.
#
#    'counts'        every row counts toward the evidence floor, and a refused row counts as
#                    a NON-PASS.  Use when refusal is itself adverse (ADVERSE), or when it is
#                    so rare / so gated upstream that the distinction cannot bite (MOOT).
#    'not_evidence'  only ADMISSIBLE (non-refused) rows count -- toward the floor AND toward
#                    the passes.  Use when refusal has NO adverse reading (BENIGN).
#
#  ONLY ONE FLAG IS BENIGN.  This is not a general softening of the veto; it is one field
#  where the refusal was measurably reading the opposite of the truth.
FIELD_EVIDENCE = {
    #  MOOT -> treated as adverse.  Inputs are 0.00% NaN, so the ONLY channel into a refusal
    #  is `totalAssets <= 0` (97 rows) -- a degenerate balance sheet, which is adverse on any
    #  reading.  Ruled explicitly rather than left to the default so it is a decision.
    'returnOnAssets':    'counts',
    #  MOOT.  Gated at source: `failTests.py:96,158-172` rejects a ticker outright if any
    #  statement is empty or short, so the residual is 0.44% of rows across 4 sources.  Too
    #  small to move a verdict either way; ruled 'counts' to stay bit-identical.
    'CFOlessEarnings':   'counts',
    #  MOOT -> adverse.  Never absent in the panel.
    'uCurrentRatio':     'counts',
    #  ADVERSE, and this one is load-bearing rather than incidental.  All three inputs are
    #  0.00% NaN, so a refusal is NEVER missing data: `net_debt_three_branch` falls through to
    #  NaN exactly when EBITDA <= 0 AND the name is not net cash -- which IS the adverse
    #  condition.  Abstaining here would let "loses money at the EBITDA line and carries net
    #  debt" dodge the leverage flag entirely, i.e. the flag would go quiet precisely on the
    #  names it exists for.
    'netDebtToEBITDA':   'counts',
    #  BENIGN -> abstain.  THE ONE THAT CHANGES, and the measured defect behind C-15: of the
    #  19.74pp of refused rows, 19.01pp are `interestExpense == 0`, which the
    #  `interest_expense_positive` guard refuses because a company with no interest bill does
    #  not HAVE a coverage ratio.  Counting that as a non-pass ejected 1,668 sources (21.5% of
    #  the universe).  The residual refusals -- 0.70% negative reported interest expense (the
    #  ratio inverts sign) and 0.03% genuinely NaN -- have no adverse reading either, so they
    #  abstain too rather than being split out.
    #
    #  ***"i.e. DEBT-FREE NAMES" WAS WRONG FOR THE MAJORITY, AND IS CORRECTED HERE***
    #  (CEO, 2026-08-10).  This comment used to justify the ruling as "19.01pp are
    #  `interestExpense == 0`, i.e. DEBT-FREE names" and the log line repeated it.  MEASURED on
    #  the 2026-08-10 CUR3K panel over each source's newest EIGHT rows:
    #      interestExpense == 0 on 14.67% of rows (20,952 rows, 2,619 sources)
    #      160 sources are zero on EVERY row      -- consistent with debt-free
    #      550 sources are MIXED: zero on some rows and POSITIVE on others
    #  A company is not debt-free in five quarters and indebted in three, so on those 550 the
    #  zero is FMP writing 0 where it has no value.  `CUBE` reads 0 for five consecutive
    #  quarters and then 29-31M; `EQR` reads 0 on three rows, and on one of them
    #  `ebitda == operatingIncome` EXACTLY -- a second broken field on the same row.  In the
    #  general pool the split is just as stark: of the 372 names abstaining on this flag, 69
    #  have NO computable row (the debt-free reading survives for those) and 303 have between
    #  one and seven (131 of them at seven of eight).
    #
    #  THE RULING ITSELF STANDS -- 'not_evidence' is still right, because a refused row still
    #  has no ADVERSE reading and counting it as a non-pass would still eject names for a
    #  vendor's blank.  What changed is that the abstention is no longer FREE: a MIXED window
    #  is charged to the ad-hoc penalty bucket in `_evaluate`, and a window with six or more
    #  computable rows is now JUDGED on them (`PARTIAL_MIN_EVIDENCE_ROWS`).
    #  A debt-free name is NOT thereby unvetted on leverage: `netDebtToEBITDA`'s net-cash
    #  branch still evaluates it, on an explicit operand condition.
    'uInterestCoverage': 'not_evidence',

    #  --- the cohort columns (CEO, 2026-08-07) ------------------------------------------
    #  BENIGN -> abstain.  SAME RULING AND SAME REASONING AS `uInterestCoverage`, on the same
    #  refusal channel: the column's admissibility gate is `interestExpense > 0`, so a refused row
    #  is a REIT with no interest expense, and "cannot cover its interest" is not a reading of a
    #  name with no interest to cover.  18 of 67 abstain.
    'reitEbitdaInterestCoverage': 'not_evidence',
    #  BENIGN -> abstain.  The gate is `revenue > 0`, so a refused row is a PRE-PRODUCTION miner.
    #  Zero revenue means the cost-curve question this flag asks does not exist yet -- it is not a
    #  miner failing to make money on its ore, it is a miner with no ore sold.  Counting it as a
    #  non-pass would eject the entire exploration half of the cohort for being explorers, which
    #  is the `uInterestCoverage`-on-debt-free defect in a new field.  43 of 218 abstain, and they
    #  are precisely the names `cashRunwayOneYear` then judges.
    'producerEbitdaPositive':     'not_evidence',
    #  ADVERSE / MOOT -> counts.  Both operands (`cash`, CFO) are 0.00% NaN in the panel, so a
    #  refusal is never missing data; there is no benign channel into one.
    'cashRunwayOneYear':          'counts',
    #  MOOT -> counts.  `totalStockholdersEquity` is never absent, and a degenerate one is adverse
    #  on any reading.  Same shape as `returnOnAssets`.
    'equityPositive':             'counts',
}

#  --- WHY A POOL IS OUT OF SCOPE, PER POOL (CEO, 2026-08-07) ----------------------------
#  ONE REASON PER POOL, because there is no longer a shared one.  The old single string asserted
#  the REIT/bank rationale for EVERY out-of-scope pool; after this change REIT and the banks are
#  IN scope, so that text was left describing cohorts it no longer applied to -- and it never
#  applied to the pool that is actually still out of scope.
NOT_APPLICABLE_REASONS = {
    #  THE ONLY POOL STILL OUT OF SCOPE, and NOT for the structurally-undefined reason.
    #  MEASURED: n = 15, and NOTHING in it fails the statutory asset-coverage test -- so there is
    #  no ejection for a veto to make and a flag set would be apparatus with no work to do.
    #  SEPARATELY, the obvious candidate flag is structurally undefined here: `CFO > 0` ejects
    #  5 of the 15 INCLUDING 3 OF THE TOP-25, because under ASC 946 an investment company's
    #  PORTFOLIO PURCHASES are an OPERATING cash flow -- so CFO measures whether the fund was
    #  buying or selling, not whether it is solvent.  A fund deploying capital reads as a fund
    #  burning cash.
    'InvestmentVehicle': (
        'pool %r is not in VETO_POOLS %s: n = 15 and nothing in it fails the statutory '
        'asset-coverage test, so there is no ejection to make. The obvious candidate flag is '
        'also structurally undefined here -- under ASC 946 portfolio purchases are an OPERATING '
        'cash flow, so `CFO > 0` ejects 5 of 15 (3 of them in the top-25) for DEPLOYING CAPITAL. '
        'This cohort is NOT vetoed and is NOT thereby certified clean.'),
}

#  Fallback for a pool nobody has ruled on (a cohort added to `carveOut` without a flag set).
#  It states IGNORANCE rather than borrowing another cohort's rationale -- the failure the
#  per-pool split exists to stop.
_DEFAULT_NOT_APPLICABLE = (
    'pool %r is not in VETO_POOLS %s and has no flag set ruled on for it. NOT a finding about '
    'this cohort: no red-flag set has been established as DEFINED on its balance sheet, so the '
    'veto declines to evaluate it. This cohort is NOT vetoed and is NOT thereby certified clean.')

#  Emitted when the PANEL predates a pool's flag set -- see `apply_veto`.  A distinct reason and
#  a distinct report key, because "we chose not to gate this cohort" and "this panel cannot carry
#  the gate" are different facts and only one of them is fixed by re-fetching.
_STALE_PANEL_NOT_APPLICABLE = (
    'pool %r IS in VETO_POOLS but the panel is missing %d of its veto column(s): %s. The panel '
    'was built by an older metric set that does not compute them, so the cohort CANNOT be vetoed '
    'on the flag set ruled for it. Vetoing on the SUBSET that is present would silently ship a '
    'weaker gate under the same name, so the veto declines the pool entirely. RE-FETCH to enable '
    'it. This cohort is NOT vetoed and is NOT thereby certified clean.')


#  EVERY FLAG IN EVERY POOL HAS AN EVIDENCE RULING, CHECKED AT IMPORT.  Without this a cohort
#  flag added without a `FIELD_EVIDENCE` entry raises a KeyError deep inside `_evaluate`, which
#  `postBo`'s guard would turn into an entirely un-vetoed run -- a missing RULING would present
#  as a missing VETO.  Import-time and loud, so it cannot reach a fetch.
_unruled = sorted({c for f in POOL_FLAGS.values() for c in f} - set(FIELD_EVIDENCE))
if _unruled:
    raise KeyError(
        'stage1_veto: %d veto flag(s) have no FIELD_EVIDENCE ruling: %s. What a REFUSED row means '
        'is a property of the field and must be RULED, not defaulted -- see the `uInterestCoverage` '
        'measurement for what a wrong default costs.' % (len(_unruled), _unruled))
del _unruled


#  --- IS A REFUSED ROW A DATA PROBLEM, OR A FACT ABOUT THE COMPANY?  (reviewer S1,
#  --- 2026-08-10) ------------------------------------------------------------------------
#  THE DEFECT THIS REPLACES, AND IT WAS LIVE IN THE SHIPPED 100.  The first version of the
#  ad-hoc bucket charged a PARTIALLY refused window and gave a FULLY refused one a free pass,
#  on the reading that a fully-refused `uInterestCoverage` window means a DEBT-FREE company.
#  That made the penalty NON-MONOTONE in exactly the wrong direction: it climbed to -0.07 at
#  seven missing rows of eight and then fell to 0.00 at eight of eight, so THE WORST DATA PAID
#  LEAST.
#
#  MEASURED on the 2026-08-10 CUR3K panel, general pool: 69 sources have a fully-refused
#  `uInterestCoverage` window, and **62 of them carry POSITIVE `totalDebt`** -- 492 rows of
#  real debt reported alongside `interestExpense == 0`.  `AAPL` is in that set with a MEDIAN
#  `totalDebt` of $97.5bn and a zero interest expense on all eight rows.  Four of the 69 are in
#  the shipped top-100 (`BKE` $372M, `BZ` $220M, `QLYS` $53M, and `DPAM.PA` which is genuinely
#  debt-free) and the first three paid nothing while 21 names in the same 100 paid -0.01 for a
#  single missing row.
#
#  THE ARGUMENT THAT KILLS THE FREE PASS IS THE ONE THE MIXED-WINDOW RULE ALREADY MADE.  That
#  rule says a company is not debt-free in five quarters and indebted in three, so a mixed
#  window is the vendor rather than the balance sheet.  The identical logic refutes the
#  zero-window reading: AAPL with $97.5bn of debt and eight zero rows is not a debt-free
#  company, it is eight missing rows.  The correction had been applied HALF-WAY.
#
#  SO THE TEST MOVES FROM THE WINDOW TO THE ROW, and it reads a RAW INPUT rather than
#  inferring from the shape of the refusals: **a refused row is charged iff the row's own
#  corroborating field says the refusal cannot be structural.**  That closes both directions at
#  once -- it charges the zero-window names that carry debt, and it STOPS charging refused rows
#  in a mixed window whose own `totalDebt` is 0 (74 rows across 20 general names, which were a
#  genuinely debt-free quarter being billed as a data gap).
#
#  `column` is read from `cdx_df`, joined to the veto window on (source, date).  A field with
#  NO corroborator is ruled explicitly as None WITH ITS REASON -- absence must be a decision.
#  PREFER AN OPERAND THAT IS **INDEPENDENT** OF THE REFUSAL CHANNEL; WHERE YOU CANNOT,
#  ENUMERATE EVERY OUTCOME OF THE SHARED ONE.  (reviewer R1, 2026-08-10 -- this generalises to
#  every corroborator anyone adds later, so it is stated here rather than in one entry.)
#    * `totalDebt` is INDEPENDENT of `interestExpense <= 0`, the channel it corroborates, so it
#      can distinguish every case and a single `> 0` test is complete.
#    * `revenue` IS `producerEbitdaPositive`'s OWN GUARD INPUT (`revenue > 0`), so it can only
#      ever witness ONE SIDE of that guard: every refused row is NaN, zero, or negative BY
#      CONSTRUCTION.  A predicate that names only one of those three silently lets the others
#      through -- which is exactly the defect corrected below.
_CORROBORATE_POSITIVE = 'positive'      # charge the row iff column > 0
#  Charge iff the column is NaN OR NEGATIVE -- i.e. every refused-row outcome EXCEPT an exact
#  zero.  The three outcomes, ruled one at a time:
#    NaN      -> no revenue FIGURE at all.  Missing data; charge.
#    0        -> the PRE-PRODUCTION fact this flag abstains for.  Do NOT charge; charging it
#                would bill the exploration half of the cohort for being explorers, which is
#                the C-15 defect re-committed inside the bucket.
#    negative -> contra-revenue / a restatement.  NOT a pre-production fact -- a company with
#                no operations does not report NEGATIVE sales -- so it is adverse or broken
#                either way; charge.
#  MEASURED (reviewer R1): `078130.KQ` runs 16.7bn, 15.0bn, 15.8bn, 18.4bn, 18.7bn, **-1.62bn**,
#  20.1bn, 12.8bn -- one negative quarter inside a fully-producing history; `OCI.AS` has two
#  interior negatives among positives.  An earlier version of this table tested `missing` only,
#  which de-charged both of them.
_CORROBORATE_MISSING_OR_NEGATIVE = 'missing_or_negative'

REFUSAL_CORROBORATOR = {
    #  The refusal channel is `interest_expense_positive` (`interestExpense <= 0`).  DEBT is
    #  what makes an interest bill non-optional, so a row reporting positive `totalDebt` and
    #  zero interest expense is a data problem on its face; a row with no debt genuinely has
    #  no coverage ratio and is not charged.  This is the field the whole finding is about.
    'uInterestCoverage': ('totalDebt', _CORROBORATE_POSITIVE),
    #  SAME CHANNEL, SAME RULING.  A REIT with mortgage debt and a zero interest expense is
    #  the identical defect; a REIT with no debt is the identical non-defect.
    'reitEbitdaInterestCoverage': ('totalDebt', _CORROBORATE_POSITIVE),
    #  The refusal channel is `revenue > 0`, and `revenue` is that guard's OWN INPUT -- see
    #  the independence note above -- so every outcome must be named.  ZERO is the
    #  pre-production fact and is NOT charged; NaN (408 of the 61,144 panel rows, 0.67%) and
    #  NEGATIVE (370 rows) both are.
    'producerEbitdaPositive': ('revenue', _CORROBORATE_MISSING_OR_NEGATIVE),
    #  --- the `counts` fields: NO CORROBORATOR, AND THAT IS NOT AN OMISSION -----------------
    #  On a `counts` field every row is evidence and a refused row is already scored as a
    #  NON-PASS, so the flag has ALREADY priced it.  Charging the bucket as well would bill the
    #  same row twice -- once through the veto and once through the score.
    'returnOnAssets': None,
    'CFOlessEarnings': None,
    'uCurrentRatio': None,
    'netDebtToEBITDA': None,
    'cashRunwayOneYear': None,
    'equityPositive': None,
}

#  EVERY FLAG HAS A CORROBORATOR RULING, CHECKED AT IMPORT -- the same guard, and for the same
#  reason, as the `FIELD_EVIDENCE` check above: a cohort flag added without one would silently
#  charge nothing, and "no data problems here" is the answer a missing ruling gives.
_uncorroborated = sorted({c for f in POOL_FLAGS.values() for c in f}
                         - set(REFUSAL_CORROBORATOR))
if _uncorroborated:
    raise KeyError(
        'stage1_veto: %d veto flag(s) have no REFUSAL_CORROBORATOR ruling: %s. Whether a '
        'refused row is a DATA PROBLEM or a FACT ABOUT THE COMPANY is a property of the field '
        'and must be RULED -- `None` with a reason is a valid ruling, a missing entry is not.'
        % (len(_uncorroborated), _uncorroborated))
del _uncorroborated

#  Emitted when the bucket CANNOT judge a refusal because the corroborating column is not in
#  reach.  A distinct, reported state: "we charged nothing" and "we could not tell" must not
#  look the same, which is the failure the free pass was.
_NO_CORROBORATOR_PANEL = (
    'the ad-hoc penalty bucket could not judge refused rows on %s: the corroborating column '
    '%r is not available (no cdx panel was passed, or it lacks the column). Refusals on that '
    'flag are NOT charged this run -- that is a MISSING MEASUREMENT, not a finding that the '
    'data is clean.')


def failed_flags(bm_df, flags=None):
    """{source: sorted list of FAILED flag names} over every source in `bm_df`.

    `flags` is the flag set to evaluate, defaulting to the GENERAL pool's `FLAGS`.  The default
    is the general set and not "whatever the caller meant" deliberately: a cohort's set is only
    ever reached through `apply_veto`, which looks it up from `pool_label`, so a flag set can
    never be applied to a pool it was not ruled for.

    `bm_df` is a Stage-1 BoMetric panel (many sources).  Rows are taken NEWEST-FIRST per source
    -- the same contract `calcScore.simpleScore_fromDict` enforces -- and re-sorted here rather
    than assumed, because nothing on the live path guarantees the ingestion order.

    A source with too little EVIDENCE on a flag simply does not carry that flag here: an
    abstention is not a failure.  A source that abstained on all five maps to an EMPTY list.
    Use `_evaluate` if you also need to know WHICH flags abstained -- an empty list here means
    "clean" and "not evaluated" alike, which is precisely why the report carries the
    abstentions separately, and per flag.
    """
    return _evaluate(bm_df, flags)[0]


def missing_columns(bm_df, flags):
    """The flag columns `flags` needs that `bm_df` does not carry, in declaration order."""
    return [c for c in flags if c not in bm_df.columns]


def _flag_evidence(win, col, cond, refusal):
    """(n_evidence, passes) for one flag on one source's window.

    `refusal == 'counts'`   -> every row is evidence and a refused (NaN) row is a NON-PASS,
                               since `> x` is False for NaN.  n_evidence == len(win), so this
                               branch reproduces the old per-source row floor EXACTLY.
    `refusal == 'not_evidence'` -> only admissible rows count, on BOTH sides.  Restricting the
                               pass count too is redundant arithmetically (NaN never passes)
                               but not redundant as code: it states that the passes and the
                               floor are counted over the SAME set, which is the invariant a
                               later edit could otherwise break silently.
    """
    vals = pd.to_numeric(win[col], errors='coerce')
    if refusal == 'not_evidence':
        vals = vals[vals.notna()]
    return len(vals), int(cond(vals).sum())


def _corroborator_frame(bm_df, cdx_df, flags):
    """`bm_df` with each flag's corroborating raw column joined on (source, date).

    Returns (frame, unavailable) -- `unavailable` names the flags whose corroborator could NOT
    be reached, so the caller can REPORT that rather than silently charge nothing.

    JOINED ON (source, date) AND NOT ASSUMED POSITIONAL.  `getData_fmp` drops the oldest `rpy`
    rows of each source's metric frame, so `bm_df` and `cdx_df` have DIFFERENT row counts per
    source and a positional pairing would read one quarter's debt against another quarter's
    refusal.  VERIFIED on the 2026-08-10 panel: all 52,092 distinct (source, date) keys in
    `BoMetric_df` are present in `cdx_df`, so the join loses nothing.  `cdx_df` is de-duplicated
    on the key first -- restated quarters appear twice there and would otherwise multiply the
    veto window's rows.
    """
    wanted = {}
    for col in flags:
        spec = REFUSAL_CORROBORATOR.get(col)
        if spec is not None:
            wanted[col] = spec
    if not wanted:
        return bm_df, []
    cols = {c for c, _p in wanted.values()}
    if cdx_df is None:
        return bm_df, sorted(wanted)
    #  `or []` on an Index raises (`The truth value of a Index is ambiguous`), so the default
    #  is supplied to getattr rather than by falsy-coalescing the result.
    have = set(getattr(cdx_df, 'columns', []))
    #  BOTH SIDES OF THE JOIN ARE GUARDED, not just the right one (reviewer R5).  A `bm_df`
    #  without `date` used to reach `merge(on=['source','date'])` and raise `KeyError` INSIDE
    #  `_evaluate` -- and `postBo` wraps the whole veto in ONE `except`, so a missing column on
    #  a DIAGNOSTIC input would have degraded the run to ENTIRELY UN-VETOED.  That is the
    #  per-layer-versus-per-pool degradation failure this module already fixed once for stale
    #  panels (`_STALE_PANEL_NOT_APPLICABLE`); the bucket must degrade the BUCKET, never the veto.
    left_have = set(getattr(bm_df, 'columns', []))
    if not {'date', 'source'} <= have or not {'date', 'source'} <= left_have:
        return bm_df, sorted(wanted)
    missing_cols = cols - have
    unavailable = sorted(c for c, (col, _p) in wanted.items() if col in missing_cols)
    usable = sorted(cols & have)
    if not usable:
        return bm_df, sorted(wanted)
    right = cdx_df[['source', 'date'] + usable].copy()
    right['date'] = pd.to_datetime(right['date'], errors='coerce')
    #  `keep='first'` IS A DECISION, DECLARED (reviewer R6).  A restated quarter appears
    #  twice in `cdx_df` and the two rows can carry DIFFERENT `totalDebt`; the frame arrives
    #  newest-first per source, so "first" is the LATEST-INGESTED restatement -- the same
    #  canonical-restatement choice `stage2_metrics.prepare_eps_series` makes, and the one that
    #  matches what the rest of the pipeline scores.  It is decisive on only 2-3 rows of this
    #  panel, which is why it is declared rather than instrumented.
    right = right.drop_duplicates(['source', 'date'], keep='first')
    #  RENAMED TO A RESERVED PREFIX, not merged under the raw name.  If `bm_df` ever carried a
    #  column of the same name, a plain merge would either overwrite it or silently suffix it
    #  -- and a suffixed corroborator is one `_row_is_a_data_problem` would fail to FIND,
    #  charging nothing and reporting nothing.  A reserved name makes "not joined" detectable
    #  instead of invisible, which is the whole class of defect this finding is about.
    right = right.rename(columns={c: _CORROBORATOR_PREFIX + c for c in usable})
    out = bm_df.merge(right, on=['source', 'date'], how='left')
    return out, unavailable


def _row_is_a_data_problem(win, col):
    """Boolean Series over `win`: is a REFUSED row on `col` a data problem rather than a fact
    about the company?  All-False when the flag has no corroborator or the column is absent.

    THE WHOLE FINDING S1 IS IN THIS FUNCTION.  See `REFUSAL_CORROBORATOR` for the measurement
    that forced it: the previous rule read the SHAPE of the refusals (a fully-refused window
    was assumed structural) and therefore charged the worst data the least.  This reads a RAW
    INPUT on the SAME ROW instead, so the answer does not depend on how many of the source's
    other rows happened to compute.
    """
    spec = REFUSAL_CORROBORATOR.get(col)
    if spec is None:
        return pd.Series(False, index=win.index)
    column, predicate = spec
    joined = _CORROBORATOR_PREFIX + column
    if joined not in win.columns:
        #  NOT silently False: reaching here means the join did not happen for this flag, and
        #  `_corroborator_frame` will have named it in `unavailable` so `apply_veto` reports a
        #  MISSING MEASUREMENT.  The all-False return is the safe arithmetic; the REPORT is
        #  what stops it being read as "no data problems".
        return pd.Series(False, index=win.index)
    vals = pd.to_numeric(win[joined], errors='coerce')
    if predicate == _CORROBORATE_POSITIVE:
        return vals > 0
    if predicate == _CORROBORATE_MISSING_OR_NEGATIVE:
        #  Every refused-row outcome EXCEPT an exact zero.  Written as the UNION of the two
        #  charged cases rather than as `~(vals == 0)`, because the negation form would also
        #  charge a row the guard never refused, and it hides which outcomes were ruled on.
        return vals.isna() | (vals < 0)
    raise ValueError('stage1_veto: %r declares unknown corroborator predicate %r'
                     % (col, predicate))


def _evaluate(bm_df, flags=None, cdx_df=None):
    """(failed, abstained, gaps) -- `{source: [failed flag names]}`,
    `{source: {flag: n_evidence}}` for every (source, flag) pair that ABSTAINED for want of
    evidence, and `{source: [(check, reason, points), ...]}` -- the AD-HOC PENALTY BUCKET
    contributions this panel earns (CEO, 2026-08-10).  Split out from `failed_flags` so
    `apply_veto` can REPORT the abstentions and raise the bucket without a second pass over
    the panel; the public single-value contract of `failed_flags` is unchanged.

    THE ABSTENTION SET IS UNCHANGED IN MEANING but no longer unchanged in SIZE: a flag with
    at least `PARTIAL_MIN_EVIDENCE_ROWS` rows of evidence is now JUDGED on them (CEO ruling,
    "judge on available rows AND penalise the gap"), so it appears in neither `abstained` nor
    -- unless it actually failed -- in `failed`.

    `cdx_df` -- the RAW fundamentals panel, used ONLY to decide whether a refused row is a data
    problem (`REFUSAL_CORROBORATOR`).  IT REACHES NO VERDICT: `failed` and `abstained` are
    computed from `bm_df` alone and are bit-identical with or without it.  Absent, the bucket
    declines to charge refusals and `apply_veto` says so -- see `_NO_CORROBORATOR_PANEL`.
    """
    if flags is None:
        flags = FLAGS
    missing = missing_columns(bm_df, flags)
    if missing:
        raise KeyError(
            'stage1_veto: the panel is missing %d veto column(s): %s. These are Stage-1 '
            'criterion columns, so a panel that lacks them was built by an older metric set '
            'and CANNOT be vetoed -- re-fetch, or run with the veto off. Scoring the veto on a '
            'subset of its flags would silently weaken it.' % (len(missing), missing))

    df = bm_df
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(['source', 'date'], ascending=[True, False])

    df, _uncorroborated = _corroborator_frame(df, cdx_df, flags)
    #  Carried out on the returned `gaps` under a reserved key, so `apply_veto` can report the
    #  degradation without a second inspection of the panel.
    out, abstained, gaps = {}, {}, {}
    if _uncorroborated:
        gaps[_UNCORROBORATED_KEY] = _uncorroborated
    for source, grp in df.groupby('source', sort=False):
        win = grp.head(WINDOW_ROWS)
        n_rows = len(win)
        bad = []
        #  --- BUCKET CONTRIBUTION 1: THE SOURCE SIMPLY HAS FEWER THAN EIGHT ROWS ---------
        #  Raised ONCE PER SOURCE, not once per flag.  A 5-row source has at most 5 evidence
        #  rows on EVERY flag, so charging the same three missing rows five times over would
        #  make a short history five times worse than one broken field -- a scaling nobody
        #  decided.  One panel-level gap, one contribution.
        if n_rows < WINDOW_ROWS:
            gaps.setdefault(source, []).append((
                CHECK_SHORT_PANEL,
                'panel carries %d of the %d newest-first rows the veto window needs, so every '
                'flag is evaluated (or refused) on a short history'
                % (n_rows, WINDOW_ROWS),
                float(WINDOW_ROWS - n_rows)))
        for col, cond in flags.items():
            n_ev, passes = _flag_evidence(win, col, cond, FIELD_EVIDENCE[col])
            #  THE EVIDENCE FLOOR (C-15), NOW IN TWO STEPS (CEO, 2026-08-10).
            #  `FAIL_MAX_PASSES` is an ABSOLUTE count, so it is only meaningful over a window
            #  long enough for "passed at most one" to still mean "essentially never": on 1 row
            #  of evidence a flag that PASSED that row scores `passes = 1 <= 1` and would fail
            #  -- a 100% pass rate read as a persistent red flag.  So:
            #    n_ev >= PARTIAL_MIN_EVIDENCE_ROWS -> JUDGE on the rows that exist;
            #    below it                          -> ABSTAIN entirely, as before.
            #  EITHER WAY the missing rows go to the bucket below FOR ASSESSMENT, which is the
            #  half of the ruling that stops an abstention being automatically free.  Whether
            #  they are actually CHARGED is a separate question answered per row by the
            #  corroborator -- an abstention on a genuinely debt-free name still costs nothing.
            if n_ev >= PARTIAL_MIN_EVIDENCE_ROWS:
                if passes <= FAIL_MAX_PASSES:
                    bad.append(col)
            else:
                abstained.setdefault(source, {})[col] = n_ev

            #  --- BUCKET CONTRIBUTION 2: REFUSED ROWS THAT THE ROW'S OWN RAW DATA CONTRADICTS
            #  PER ROW, AGAINST A RAW INPUT -- NOT per window, and NOT from the shape of the
            #  refusals (reviewer S1, 2026-08-10).  The `n_ev == 0` free pass this replaces is
            #  documented at `REFUSAL_CORROBORATOR`, together with the measurement that killed
            #  it: it made the penalty NON-MONOTONE, climbing to -0.07 at seven missing rows of
            #  eight and falling to 0.00 at eight of eight, so the WORST data paid LEAST.
            #
            #  A row is charged iff it is REFUSED **and** its corroborator says the refusal
            #  cannot be structural -- `totalDebt > 0` against a zero interest expense, or an
            #  ABSENT revenue figure against the pre-production reading.  Both directions close
            #  at once: a fully-refused window on a name carrying debt is now charged, and a
            #  refused row in a mixed window whose own `totalDebt` is 0 is NOT (a genuinely
            #  debt-free quarter was being billed as a data gap -- 74 rows across 20 general
            #  names on this panel).
            #
            #  A `counts` field reaches here with no corroborator and charges nothing, which is
            #  deliberate: its refused rows are ALREADY scored as non-passes by the flag itself,
            #  so charging the bucket too would bill the same row twice.
            if n_rows:
                refused = pd.to_numeric(win[col], errors='coerce').isna()
                chargeable = int((refused & _row_is_a_data_problem(win, col)).sum())
                if chargeable:
                    _spec = REFUSAL_CORROBORATOR.get(col)
                    gaps.setdefault(source, []).append((
                        CHECK_REFUSED_ROWS,
                        '%s: %d of %d available rows are not computable AND report %s on the '
                        'SAME row, so the refusal cannot mean the flag is inapplicable to this '
                        'company -- it is missing data. Flag was %s.'
                        % (col, chargeable, n_rows,
                           ('a positive `%s`' % _spec[0])
                           if _spec[1] == _CORROBORATE_POSITIVE
                           else ('no `%s` figure at all, or a NEGATIVE one' % _spec[0]),
                           ('JUDGED on the %d row(s) that compute' % n_ev)
                           if n_ev >= PARTIAL_MIN_EVIDENCE_ROWS
                           else ('ABSTAINED -- %d row(s) is under the %d-row partial-window '
                                 'floor' % (n_ev, PARTIAL_MIN_EVIDENCE_ROWS))),
                        float(chargeable)))
        out[source] = sorted(bad)
    return out, abstained, gaps


def pool_flags(pool_label):
    """The flag set ruled for `pool_label` -- `FLAGS` for any pool without its own entry.

    The default is the general five-flag set so that a pool named in `VETO_POOLS` without a
    `POOL_FLAGS` entry is gated by SOMETHING rather than by an empty dict, which would eject
    nobody and report a clean cohort.  A pool with no ruling belongs OUT of `VETO_POOLS`, not in
    it with no flags.
    """
    return POOL_FLAGS.get(pool_label, FLAGS)


def apply_veto(scores_df, bm_df, pool_label='general', enabled=None, verbose=True,
               pools=None, penalty_book=None, cdx_df=None):
    """`scores_df` with vetoed sources removed.  Returns (kept, report).

    `scores_df` is a Stage-1 score frame carrying a `source` column (`BoScore_df` or a carve-out
    cohort's slice of it); `bm_df` is the panel those scores were computed from.

    BIT-IDENTICAL WHEN OFF: with `enabled` falsy the input frame is returned UNCHANGED (the same
    object, not a copy) and `report['enabled']` is False.  The module default is ON since
    2026-08-07 (CEO); an explicit `enabled=` argument overrides it so a research script never
    mutates globals.

    PER-POOL FLAG SETS: the flags evaluated are `POOL_FLAGS[pool_label]`, so a cohort is gated on
    the fields that MEAN something on its balance sheet.  When `pool_label` is not in `VETO_POOLS`
    the input frame is returned UNCHANGED, with `report['applies'] = False` and
    `report['not_applicable_reason']` stating why FOR THAT POOL.  The report is still emitted, so
    a cohort the veto DECLINED TO GATE is distinguishable from one it gated and found clean.
    `pools=` overrides `VETO_POOLS` for an offline A/B (the per-cohort ejection rates in the
    docstring were measured that way) without mutating module state, exactly as `enabled=` does.

    A STALE PANEL DECLINES ONE POOL, NOT THE WHOLE LAYER.  A pool whose flag columns the panel
    does not carry returns `applies=False` with `report['missing_columns']` set, rather than
    raising -- the cohort flag sets landed in code before the fetch that computes their columns,
    so on today's panel the cohorts that need NEW columns must degrade to un-vetoed while the
    ones that need only EXISTING columns still run.  Raising would have taken every pool down
    together through `postBo`'s single guard.

    `cdx_df` -- the RAW fundamentals panel.  Used ONLY by the ad-hoc penalty bucket, to decide
    whether a refused row is a DATA PROBLEM or a fact about the company (see
    `REFUSAL_CORROBORATOR`).  IT REACHES NO VERDICT -- the ejection set is bit-identical with or
    without it -- but WITHOUT it the bucket cannot judge a refusal, declines to charge one, and
    REPORTS that as a missing measurement rather than letting it look like a clean panel.

    `penalty_book` -- an `adhoc_penalty.PenaltyBook` (CEO, 2026-08-10).  When given, every
    data gap this layer meets is CHARGED to it, per source and per check, for THIS POOL's
    members only.  The bucket is a SOFT veto: it never removes a name here, it lowers that
    name's Stage-2 score before the sort.  Optional purely so the existing callers and the
    offline A/B harnesses keep working unchanged -- a call with no book behaves exactly as
    before, and the report still carries `adhoc_penalty_points` so a reader can see what the
    bucket WOULD have charged.
    """
    if enabled is None:
        enabled = ENABLED
    if pools is None:
        pools = VETO_POOLS
    flags = pool_flags(pool_label)
    applies = pool_label in pools
    report = {'pool': pool_label, 'enabled': bool(enabled), 'applies': bool(applies),
              'flags': sorted(flags), 'missing_columns': [],
              'not_applicable_reason': None, 'n_in': len(scores_df),
              'n_ejected': 0, 'n_out': len(scores_df), 'by_flag': {}, 'ejected': [],
              #  ABSTENTIONS, PER FLAG (C-15) -- `{flag: number of sources in THIS pool that
              #  abstained on it}`.  Per FLAG and not per SOURCE deliberately: the dominant
              #  abstention is `uInterestCoverage`, and a per-source count
              #  would report those as "not evaluated" alongside genuinely short-history names
              #  and hide that they are one flag.  Reported, not hidden: "found clean" and
              #  "never evaluated" are different facts.
              #  THEY ARE **NOT** ALL ONE BENIGN CAUSE, and an earlier version of this comment
              #  said they were (corrected 2026-08-10, reviewer R2): 206 of the 222 names still
              #  abstaining on a coverage flag on this panel carry POSITIVE median `totalDebt`.
              #  The bucket charges those; the abstention count alone does not distinguish
              #  them, which is why both numbers are reported side by side.
              'n_short_window': {}, 'short_window': {},
              #  THE AD-HOC PENALTY BUCKET this pool raised (CEO, 2026-08-10).  Reported per
              #  pool for the same reason the abstentions are: a pool that raised no points
              #  and a pool the layer never looked at must not read the same.
              'adhoc_penalty_points': {},
              #  Flags whose refusals the bucket COULD NOT judge for want of the corroborating
              #  raw column (reviewer S1).  Non-empty means "NOT MEASURED", never "clean".
              'adhoc_penalty_uncorroborated': []}
    if not enabled:
        return scores_df, report
    if not applies:
        #  NOT A SILENT SKIP.  The reason travels in the report and (when verbose) in the log,
        #  because the failure mode here is a reader seeing `n_ejected = 0` on a cohort and
        #  concluding the cohort is clean.
        report['not_applicable_reason'] = (
            NOT_APPLICABLE_REASONS.get(pool_label, _DEFAULT_NOT_APPLICABLE)
            % (pool_label, list(pools)))
        if verbose:
            print('STAGE-1 VETO [%s]: NOT APPLIED -- %s'
                  % (pool_label, report['not_applicable_reason']), flush=True)
        return scores_df, report

    #  THE PANEL-CAPABILITY CHECK, and it is deliberately BEFORE `_evaluate` rather than a caught
    #  KeyError.  A cohort flag set can name a column that only a LATER fetch computes, and the
    #  honest response is to decline THAT pool loudly -- not to fail the call and let `postBo`'s
    #  single `except` turn one stale column into an entirely un-vetoed run.
    absent = missing_columns(bm_df, flags)
    if absent:
        report['applies'] = False
        report['missing_columns'] = absent
        report['not_applicable_reason'] = (
            _STALE_PANEL_NOT_APPLICABLE % (pool_label, len(absent), absent))
        if verbose:
            print('STAGE-1 VETO [%s]: NOT APPLIED -- %s'
                  % (pool_label, report['not_applicable_reason']), flush=True)
        return scores_df, report

    bad, abstained, gaps = _evaluate(bm_df, flags, cdx_df=cdx_df)
    _uncorroborated = gaps.pop(_UNCORROBORATED_KEY, [])
    ejected = [s for s in scores_df['source']
               if len(bad.get(s, [])) >= EJECT_MIN_FLAGS]

    #  --- RAISE THE AD-HOC PENALTY BUCKET (CEO, 2026-08-10) ------------------------------
    #  RESTRICTED TO THIS POOL'S MEMBERS, exactly as the abstention report is: `_evaluate`
    #  walks the WHOLE panel (it has to -- the window is per source), so charging every gap it
    #  found would give the general pool the cohorts' data problems as well.
    #  CHARGED TO EJECTED NAMES TOO, deliberately: a name can be ejected on one flag and have
    #  a data gap on another, and the bucket is written to the evidence CSV for every name it
    #  touched.  Only the SURVIVORS reach Stage-2, so an ejected name's points cost it nothing
    #  -- it is already gone -- but the record of the gap is what the CSV is for.
    #  REACHES THE SHIPPED CSV, not just the report (reviewer R3).  Declared BEFORE the
    #  charges so the caveat is written even on a pool that charged nothing.
    if penalty_book is not None:
        for _f in _uncorroborated:
            penalty_book.declare_unmeasured(
                CHECK_UNCORROBORATED,
                _NO_CORROBORATOR_PANEL % (_f, (REFUSAL_CORROBORATOR.get(_f)
                                               or ('<none>', ''))[0]),
                pool=pool_label)
    _pool_members = list(scores_df['source'])
    adhoc_points = {}
    for s in _pool_members:
        for check, reason, points in gaps.get(s, ()):
            adhoc_points[s] = adhoc_points.get(s, 0.0) + points
            if penalty_book is not None:
                penalty_book.add(s, check, reason, points, pool=pool_label)
    by_flag = {}
    for s in ejected:
        for f in bad[s]:
            by_flag[f] = by_flag.get(f, 0) + 1
    #  Restricted to this pool's members, so the six per-pool reports sum to something meaningful
    #  instead of each repeating the whole panel's abstentions.
    pool_short = {s: abstained[s] for s in scores_df['source'] if s in abstained}
    n_short_by_flag = {}
    for _flags in pool_short.values():
        for f in _flags:
            n_short_by_flag[f] = n_short_by_flag.get(f, 0) + 1

    kept = scores_df[~scores_df['source'].isin(set(ejected))]
    report.update(n_ejected=len(ejected), n_out=len(kept),
                  by_flag=dict(sorted(by_flag.items())), ejected=sorted(ejected),
                  n_short_window=dict(sorted(n_short_by_flag.items())),
                  short_window={s: dict(sorted(f.items()))
                                for s, f in sorted(pool_short.items())},
                  adhoc_penalty_points=dict(sorted(adhoc_points.items())),
                  adhoc_penalty_uncorroborated=list(_uncorroborated))
    if verbose:
        #  PER-POOL LOGGING IS PART OF THE DESIGN, not debug output: a veto that removed most of
        #  a cohort and one that removed nobody are indistinguishable without it.
        #  THE FLAG COUNT IS THIS POOL'S, not a hard-coded 5: the cohorts carry 1, 3 and 6 flags,
        #  and a line reading "k>=1 of 5" on a one-flag cohort would misreport the gate that ran.
        print('STAGE-1 VETO [%s]: %d -> %d names (%d ejected, k>=%d of %d flags %s failed at '
              '<=%d passes over a window of at least %d EVIDENCE rows, out of %d). '
              'Ejections by flag: %s'
              % (pool_label, report['n_in'], report['n_out'], report['n_ejected'],
                 EJECT_MIN_FLAGS, len(flags), sorted(flags), FAIL_MAX_PASSES,
                 PARTIAL_MIN_EVIDENCE_ROWS, WINDOW_ROWS,
                 report['by_flag'] or '{}'), flush=True)
        if n_short_by_flag:
            #  PER FLAG.  A count of names would say "1,668 not evaluated" and conceal that
            #  every one is `uInterestCoverage` on a debt-free balance sheet -- the exact
            #  confusion between "found clean" and "declined to look" this line exists to stop.
            print('STAGE-1 VETO [%s]: ABSTENTIONS by flag (under %d rows of countable '
                  'evidence, so the flag could not fail -- these names passed THAT flag '
                  'UNCHECKED): %s. Whether an abstention is ALSO a data problem is decided '
                  'PER ROW against a raw operand (`REFUSAL_CORROBORATOR`), not from how many '
                  'of the window refused: on this panel 206 of the 222 names still abstaining '
                  'on a coverage flag carry POSITIVE median `totalDebt`, so most of them are '
                  'missing rows rather than debt-free companies. See the bucket line below '
                  'for what was charged.'
                  % (pool_label, PARTIAL_MIN_EVIDENCE_ROWS,
                     dict(sorted(n_short_by_flag.items()))),
                  flush=True)
        for _f in _uncorroborated:
            #  LOUD, and BEFORE the bucket line: a reader who sees a small bucket must be able
            #  to tell "few data problems" from "we could not look".
            print('STAGE-1 VETO [%s]: %s'
                  % (pool_label,
                     _NO_CORROBORATOR_PANEL
                     % (_f, (REFUSAL_CORROBORATOR.get(_f) or ('<none>', ''))[0])), flush=True)
        if adhoc_points:
            #  THE SOFT VETO'S OWN LINE.  Without it the bucket would be the one gate in this
            #  file whose action is invisible from the run -- and a penalty nobody can see is
            #  indistinguishable from no penalty, which is the failure the per-pool logging
            #  above exists to stop.
            _worst = sorted(adhoc_points.items(), key=lambda kv: -kv[1])[:8]
            print('STAGE-1 VETO [%s]: AD-HOC PENALTY BUCKET -- %d name(s) charged %.0f point(s) '
                  'in total (a SOFT veto: lowers the Stage-2 score before the sort, ejects '
                  'nobody). Worst: %s'
                  % (pool_label, len(adhoc_points), sum(adhoc_points.values()),
                     ', '.join('%s %.0f' % (s, p) for s, p in _worst)), flush=True)
    return kept, report
