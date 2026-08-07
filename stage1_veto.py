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
flag and all for one benign reason.  A veto that silently declined to evaluate a fifth of a pool
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

OPEN ISSUE, so the next reader does not re-derive this: the carve-out cohorts are now UNVETOED
on solvency entirely.  The right resolution is a per-cohort red-flag set stated on fields that
MEAN something there (e.g. interest coverage and FFO-based leverage for REITs, capital adequacy
for banks), not a re-tuned copy of this one.  Until that exists, cohorts are gated by the
weighted Stage-1 score alone -- which is the pre-veto status quo, not a regression.

ON THE GENERAL POOL THE CHANGE IS SMALL: the veto moves only 5 of the top 100 (95% overlap) --
ejecting UHS, PEY.TO, SBH, 215200.KQ, TCL-A.TO and promoting 000270.KS, ATH.TO, DRW3.DE,
HCO.PA, LEGH.  A defensible cleanup, not a rewrite of the deliverable.

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

#  --- WHICH POOLS IT MAY RUN ON (CEO, 2026-08-07) ---------------------------------------
#  THE GENERAL POOL ALONE.  Two of the five flags (`uCurrentRatio`, `netDebtToEBITDA`) are
#  STRUCTURALLY UNDEFINED on the leveraged-vehicle and bank cohorts, so applying them there
#  asks the wrong question rather than asking a strict one -- see the module docstring for
#  the measured per-cohort ejection rates (REIT 95.9%) that overruled the old all-pools rule.
#  A SET, not a boolean, so adding a cohort later is a one-line decision WITH the flag set it
#  is claiming to be defined on; and so `apply_veto` can report `applies=False` by name rather
#  than the caller silently not calling it -- "out of scope" and "found nothing" must not look
#  the same in the report.
VETO_POOLS = ('general',)

#  --- THE THREE PARAMETERS, EACH A STATED DECISION --------------------------------------
#  Rows of the newest-first per-source window a flag is evaluated over.  8 is Stage-1's own
#  scoring window (`calcScore.calcByTier`'s head(n)); using a different one would mean the veto
#  and the score disagreed about what "recently" means.
WINDOW_ROWS = 8
#  A flag FAILS when it passed at most this many of `WINDOW_ROWS` rows of EVIDENCE.  1, not 0 --
#  see the docstring.
FAIL_MAX_PASSES = 1
#  NO `MIN_WINDOW_ROWS` -- DELETED AT C-15, DELIBERATELY.  The per-source row floor is now a
#  DERIVED consequence of the per-flag evidence floor (a 5-row source has <= 5 evidence rows on
#  every flag, so every flag abstains).  Do not re-add it: a second rule that merely agrees with
#  the first is the pair that silently diverges later.
#  Failed flags needed to eject.  1 -- one persistent red flag is a disqualification.
EJECT_MIN_FLAGS = 1

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
    #  19.74pp of refused rows, 19.01pp are `interestExpense == 0`, i.e. DEBT-FREE names that
    #  the `interest_expense_positive` guard refuses because a debt-free company does not HAVE
    #  a coverage ratio.  Counting that as a non-pass ejected 1,668 sources (21.5% of the
    #  universe) FOR HAVING NO DEBT.  The residual refusals -- 0.70% negative reported interest
    #  expense (the ratio inverts sign) and 0.03% genuinely NaN -- have no adverse reading
    #  either, so they abstain too rather than being split out.
    #  A debt-free name is NOT thereby unvetted on leverage: `netDebtToEBITDA`'s net-cash
    #  branch still evaluates it, on an explicit operand condition.
    'uInterestCoverage': 'not_evidence',
}


def failed_flags(bm_df):
    """{source: sorted list of FAILED flag names} over every source in `bm_df`.

    `bm_df` is a Stage-1 BoMetric panel (many sources).  Rows are taken NEWEST-FIRST per source
    -- the same contract `calcScore.simpleScore_fromDict` enforces -- and re-sorted here rather
    than assumed, because nothing on the live path guarantees the ingestion order.

    A source with too little EVIDENCE on a flag simply does not carry that flag here: an
    abstention is not a failure.  A source that abstained on all five maps to an EMPTY list.
    Use `_evaluate` if you also need to know WHICH flags abstained -- an empty list here means
    "clean" and "not evaluated" alike, which is precisely why the report carries the
    abstentions separately, and per flag.
    """
    return _evaluate(bm_df)[0]


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


def _evaluate(bm_df):
    """(failed, abstained) -- `{source: [failed flag names]}` and
    `{source: {flag: n_evidence}}` for every (source, flag) pair that ABSTAINED for want of
    evidence.  Split out from `failed_flags` so `apply_veto` can REPORT the abstentions
    without a second pass over the panel; the public single-value contract of `failed_flags`
    is unchanged.
    """
    missing = [c for c in FLAGS if c not in bm_df.columns]
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

    out, abstained = {}, {}
    for source, grp in df.groupby('source', sort=False):
        win = grp.head(WINDOW_ROWS)
        bad = []
        for col, cond in FLAGS.items():
            n_ev, passes = _flag_evidence(win, col, cond, FIELD_EVIDENCE[col])
            #  THE EVIDENCE FLOOR (C-15).  `FAIL_MAX_PASSES` is an ABSOLUTE count, so it is
            #  only meaningful over a full window: on 1 row of evidence a flag that PASSED
            #  that row scores `passes = 1 <= 1` and would fail -- a 100% pass rate read as a
            #  persistent red flag.  Under the floor the flag ABSTAINS: it is not a failure
            #  and it is not a pass, and it does not reach the `k` count either way.
            if n_ev < WINDOW_ROWS:
                abstained.setdefault(source, {})[col] = n_ev
            elif passes <= FAIL_MAX_PASSES:
                bad.append(col)
        out[source] = sorted(bad)
    return out, abstained


def apply_veto(scores_df, bm_df, pool_label='general', enabled=None, verbose=True,
               pools=None):
    """`scores_df` with vetoed sources removed.  Returns (kept, report).

    `scores_df` is a Stage-1 score frame carrying a `source` column (`BoScore_df` or a carve-out
    cohort's slice of it); `bm_df` is the panel those scores were computed from.

    BIT-IDENTICAL WHEN OFF: with `enabled` falsy the input frame is returned UNCHANGED (the same
    object, not a copy) and `report['enabled']` is False.  The module default is ON since
    2026-08-07 (CEO); an explicit `enabled=` argument overrides it so a research script never
    mutates globals.

    GENERAL POOL ONLY: when `pool_label` is not in `VETO_POOLS` the input frame is likewise
    returned UNCHANGED, with `report['applies'] = False` and `report['not_applicable_reason']`
    stating why.  The report is still emitted, so a cohort the veto DECLINED TO GATE is
    distinguishable from one it gated and found clean.  `pools=` overrides `VETO_POOLS` for an
    offline A/B (the per-cohort ejection rates in the docstring were measured that way) without
    mutating module state, exactly as `enabled=` does.
    """
    if enabled is None:
        enabled = ENABLED
    if pools is None:
        pools = VETO_POOLS
    applies = pool_label in pools
    report = {'pool': pool_label, 'enabled': bool(enabled), 'applies': bool(applies),
              'not_applicable_reason': None, 'n_in': len(scores_df),
              'n_ejected': 0, 'n_out': len(scores_df), 'by_flag': {}, 'ejected': [],
              #  ABSTENTIONS, PER FLAG (C-15) -- `{flag: number of sources in THIS pool that
              #  abstained on it}`.  Per FLAG and not per SOURCE deliberately: the dominant
              #  abstention is `uInterestCoverage` on DEBT-FREE names, and a per-source count
              #  would report those as "not evaluated" alongside genuinely short-history names
              #  and hide that they are one flag with one benign cause.  Reported, not hidden:
              #  "found clean" and "never evaluated" are different facts.
              'n_short_window': {}, 'short_window': {}}
    if not enabled:
        return scores_df, report
    if not applies:
        #  NOT A SILENT SKIP.  The reason travels in the report and (when verbose) in the log,
        #  because the failure mode here is a reader seeing `n_ejected = 0` on a cohort and
        #  concluding the cohort is clean.
        report['not_applicable_reason'] = (
            'pool %r is not in VETO_POOLS %s: `uCurrentRatio` and `netDebtToEBITDA` are '
            'structurally undefined on the carve-out cohorts (a REIT carries mortgage debt at '
            '5-8x EBITDA by design and holds no current assets), so the veto asks the wrong '
            'question there rather than a strict one. Measured: 95.9%% of REITs ejected. This '
            'cohort is NOT vetoed and is NOT thereby certified clean.'
            % (pool_label, list(pools)))
        if verbose:
            print('STAGE-1 VETO [%s]: NOT APPLIED -- %s'
                  % (pool_label, report['not_applicable_reason']), flush=True)
        return scores_df, report

    bad, abstained = _evaluate(bm_df)
    ejected = [s for s in scores_df['source']
               if len(bad.get(s, [])) >= EJECT_MIN_FLAGS]
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
                                for s, f in sorted(pool_short.items())})
    if verbose:
        #  PER-POOL LOGGING IS PART OF THE DESIGN, not debug output: a veto that removed most of
        #  a cohort and one that removed nobody are indistinguishable without it.
        print('STAGE-1 VETO [%s]: %d -> %d names (%d ejected, k>=%d of 5 flags failed at '
              '<=%d of %d EVIDENCE rows). Ejections by flag: %s'
              % (pool_label, report['n_in'], report['n_out'], report['n_ejected'],
                 EJECT_MIN_FLAGS, FAIL_MAX_PASSES, WINDOW_ROWS,
                 report['by_flag'] or '{}'), flush=True)
        if n_short_by_flag:
            #  PER FLAG.  A count of names would say "1,668 not evaluated" and conceal that
            #  every one is `uInterestCoverage` on a debt-free balance sheet -- the exact
            #  confusion between "found clean" and "declined to look" this line exists to stop.
            print('STAGE-1 VETO [%s]: ABSTENTIONS by flag (under %d rows of countable evidence, '
                  'so the flag could not fail -- these names passed THAT flag UNCHECKED): %s. '
                  'For `uInterestCoverage` the usual cause is a DEBT-FREE name (no interest '
                  'expense, so no coverage ratio exists), not missing data.'
                  % (pool_label, WINDOW_ROWS, dict(sorted(n_short_by_flag.items()))),
                  flush=True)
    return kept, report
