"""scoringWeights.py  --  THE SINGLE SOURCE OF TRUTH for every scoring weight.

WHY THIS FILE EXISTS (single-source refactor, 2026-08-02)
--------------------------------------------------------
The same 21 weight numbers used to be written out as literals in five places:
`createDicts.getPostDict`, `createDicts.getPostDict_legacy`,
`baseline_tools/tune_run.MU_GENERAL`, `baseline_tools/new_scorer_bench.W_THEORY`, and
`carveOut.COHORT_WEIGHTS` (five more vectors on the same key set).  Changing a weight
therefore meant editing several files and HOPING they agreed -- and one of them was
already wrong (see PRE-EXISTING DISAGREEMENT below).  Everything now derives from the
tables in this module, and `test_scoring_weights_single_source.py` fails if any consumer
drifts.

TO CHANGE A WEIGHT (SINCE E-2, 2026-08-04): you no longer edit a number -- you edit the
BLOCK MODEL in section B (a block budget, a metric's block/tier, or the P sub-block split)
and every one of the six vectors recomputes.  `MU_GENERAL` (the tuner prior) and
`W_THEORY` (the research bench) follow automatically, as before; the Sigma|w| = 1
invariant below will tell you immediately if the vector no longer normalises.

WHAT IS *NOT* HERE.  Stage-1 tier weights (Tier S/A/B/C/D -> 1/0.75/0.5/0.3/0.1, anything
else -> 0) live in `calcScore.calcByTier` and are deliberately NOT mirrored here -- a
mirrored copy would be exactly the defect this module exists to remove.  The Stage-1
criterion registry (Tiers included) stays in `createDicts.getDicts`.

PRE-EXISTING DISAGREEMENT, REPORTED NOT RECONCILED (2026-08-02).
`baseline_tools/model_vs_metric.WEIGHTS` is a SIXTH copy whose comment claims it is
"from createDicts.getPostDict (the AggScore weights)".  It is not: all 18 of its values
are the pre-2026-07-14 LEGACY vector (`LEGACY` below), not the deployed one, and its
excluded-metric note quotes the legacy 0.35/0.25/0.1 as well.  It is consumed by
`new_scorer_bench.old_composite` and `real_ic.py` as the "OLD" comparison arm, where the
legacy numbers may well be what those benches intend -- so the numbers are LEFT EXACTLY
AS THEY ARE and only flagged.  Deciding whether that arm should be legacy or deployed is
a research call for the CEO, not a refactor.


A.  STAGE-2 CANONICAL STRUCTURE
-------------------------------
`POSTBM_EQMET` and `POSTNEW_KEYS` fix the SHAPE of `createDicts.getPostDict()`: which
metric keys exist, in what order (postBoRank builds `postScoreMetric_df`'s columns from
`.keys()`, so order is load-bearing), and which underlying `eqMet` column each
benchmark metric reads.  Only the NUMBERS live in the vectors below.
"""

# --- A. structure -----------------------------------------------------------
# metric key -> the BoMetric/cdx column it is computed from ("eqMet").
POSTBM_EQMET = {
    'RoA':                     'returnOnAssets',
    'earnYield':               'earningsYield',
    'grahamNumberToPrice':     'grahamNumberToPrice',
    'bVpRatio':                'pbRatio',
    'revenueGrowth':           'revenue',
    'incomeQuality':           'incomeQuality',
    'returnOnEquity':          'returnOnEquity',
    'returnOnCapitalEmployed': 'returnOnCapitalEmployed',
    'currentRatio':            'currentRatio',
    'grossProfitMargin':       'grossProfitMargin',
}

# the "new" Stage-2 metrics -- computed in postBoRank rather than read from an eqMet.
#
# EVERYTHING AFTER `CycleHeat` IS APPENDED, NOT INSERTED.  `shareCountChange` and
# `longTermDebtChange` are the two Piotroski components extracted as standalone metrics for
# the FIN-1 vector (E-2, 2026-08-04; see the C-block / FIN-1 notes in section B).
# `interestCoverage` and `navPerShareGrowth` are the 2026-08-06 additions (S-block Tier 1 in
# every non-exempt vector, and FIN-1's R-block Tier 1 respectively).  They go at the END
# because this tuple's ORDER is the column order of `postScoreMetric_df`, and appending
# leaves every existing column exactly where it was.
#
# `interestCoverage` LIVES HERE AND NOT IN `POSTBM_EQMET` for two reasons, both structural:
# there is no `interestCoverage` COLUMN on cdx_df to be an eqMet (Stage-1 builds
# `uInterestCoverage` in calcMetrics from operatingIncome / interestExpense, which is a
# criterion column, not a fundamentals field), and inserting a key into POSTBM_EQMET would
# move every column that follows it.
POSTNEW_KEYS = (
    'freeCashFlowYield',
    'freeCashFlowPerShareGrowth',
    'DcfToPrice',
    'marketCapRevQuants',
    'Altman-Z',
    'Piotroski',
    'tbVpRatio',
    'BoScore',
    'EPStoEPSmean',
    'priceGrowth',
    'CycleHeat',
    'shareCountChange',
    'longTermDebtChange',
    'interestCoverage',
    'navPerShareGrowth',
)

# all 23 keys in EMISSION order (postBm block, then postNew block).
METRIC_KEYS = tuple(POSTBM_EQMET) + POSTNEW_KEYS


# =========================================================================== #
# --- B.  THE BLOCK MODEL -- THE AUTHORITY (issue E-2, 2026-08-04) ----------- #
# =========================================================================== #
# WHAT CHANGED, AND IT CHANGES EVERY SCORE BY DESIGN.  Until now `DEPLOYED` was 21
# hand-entered decimals and the five cohorts were 105 more.  From E-2 there are no
# hand-entered weights at all: a weight is COMPUTED from
#
#       w = W_block  x  sigma_subblock  x  tau_tier / n_members_in_tier
#
# and the six vectors (general + five carve-out cohorts) are six evaluations of that one
# formula.  Provenance:  projects/investment-filter/design/weighting-strategy.md sections
# 14 and 15 (systems-designer), on CEO decisions of 2026-08-04.
#
# WHY DERIVED RATHER THAN TRANSCRIBED -- this is not tidiness, it is the tripwire:
#   * SIX of the eighteen general weights are REPEATING decimals (0.23/3, 0.26/12,
#     0.086x2/3, 0.086/3, 0.05x2/3, 0.05/3).  Hand-entering the 4-dp display values gives
#     Sigma|w| = 0.9999 and `_validate()` below halts EVERY import.  The designer flagged
#     exactly this ("compute w in float; do not transcribe decimals").
#   * DEFECT DEMOTION IS A RULE, not a number.  `grahamNumberToPrice` sits at Tier 3 only
#     while audit D3 is unfixed; `incomeQuality` returned to Tier 1 the moment D2 was fixed
#     (2026-08-01, `(CFO - NI)/totalAssets`).  Under the old table that restoration was a
#     re-negotiation; here it is a one-character edit to a tier and the whole vector
#     re-derives -- "restored by rule, not by re-negotiation".
#   * The COHORTS derive from the general vector by Rule PROP (below), so the next time the
#     CEO moves a general budget the five cohort vectors move with it instead of silently
#     drifting.  105 unarguable numbers become one table of blocks and tiers.
#
# WHAT IS *NOT* CLAIMED, restated from the design so nobody upgrades it in transit: these
# budgets are a HAND-REASONED THEORY PRIOR, not a fitted estimate.  ~1.3 independent
# 36-month windows exist and 380 of 380 refit candidates fell inside one standard error, so
# no re-fit is possible and none is proposed.  The improvement is that the numbers are
# arguable, not that they are calibrated.

# --- B.1  the two structural constants -------------------------------------- #
# tau: the 3:2:1 directness ladder, normalised over the tiers ACTUALLY OCCUPIED.
#   Tier 1 = primary carrier (states the block's question with no proxy step)
#   Tier 2 = proxy or guard (one substitution, or guards a specific Tier-1 failure)
#   Tier 3 = weak or defective form (a different question that correlates, or the right
#            question in a form known to be wrong -- this is where defect demotion lands)
# Equal split WITHIN a tier: members of one tier are by construction doing the same job at
# the same directness, so averaging them is noise reduction.  A LONE Tier-2 metric
# therefore takes the whole Tier-2 budget -- deliberate ("tiers get budgets, metrics do
# not"), and it is why `freeCashFlowPerShareGrowth` out-weights `RoA`/`ROCE` inside D.
TIER_TAU = {1: 3.0, 2: 2.0, 3: 1.0}

# P is the only block with sub-blocks: P-E (earnings cheapness, the thesis) and P-A (asset
# cheapness, the reading that SURVIVES a discredited E).  Frozen at 0.65/0.35 to stay
# identified against tau -- do not unfreeze one to chase a magnitude in the other.
SIGMA_P = {'E': 0.65, 'A': 0.35}

# --- B.2  the general block budgets -- THE CEO'S DECISION ------------------- #
# CEO 2026-08-04: cheapness (P) pinned at 0.26; the `C` holding pen carved at 0.05; the
# remaining loss spread across R/N/S/M so their proportions hold -- and DURABILITY HELD
# EQUAL TO CHEAPNESS at 0.26 (weighting-strategy section 15.5 variant (b)).  Holding D
# equal taxes R/N/S/M by a further ~6% and buys one thing worth having: `earnYield` stays
# the largest single weight in the vector THROUGH the coming D3 fix (see B.6).
#
#   P CHEAP     is it cheap
#   R REAL      is the E cash rather than an accrual artifact
#   N NORMAL    is the E at a cycle peak
#   D DURABLE   can the business defend the E
#   S SURVIVES  can the equity survive
#   M MECHANISM why is it mispriced
#   C           NOT a seventh question -- a HOLDING PEN for metrics whose conditioning
#               design (issue E-3) is PARKED.  It dissolves when E-3 unparks and its 0.05
#               returns to the six blocks pro rata.  Do not let a reader infer a
#               seven-question taxonomy from this vector.
#
# THE S RE-BUDGET (CEO, 2026-08-06).  `interestCoverage` joins S as its FIRST EVER Tier-1
# member, and the CEO raised W_S from 0.0860 to **0.120** to pay for it.  0.120 is a
# DELIBERATE UNDER-FUND of the harmless-to-the-incumbents level: holding `currentRatio` and
# `Altman-Z` at their present weights while inserting a Tier-1 member would have needed
# W_S = 0.1720 (3:2:1 over the same tau ladder), and the CEO capped at 0.120 instead, so
# both existing members lose ~30% (currentRatio 0.0573 -> 0.0400, Altman-Z 0.0287 ->
# 0.0200).  That is the decision, not a rounding consequence: the solvency block gets a
# direct instrument and the two proxies it had are demoted to pay for most of it.
#
# THE OTHER 0.034 IS TAKEN PROPORTIONALLY FROM THE SIX NON-S BLOCKS, and proportionally is
# the whole point -- a flat or hand-picked take would have re-opened every budget argument
# in section 15.  Scaling the non-S blocks by (1 - W_S_new)/(1 - W_S_old) = 0.880/0.914
# preserves EVERY ratio among non-S metrics, and in particular preserves the CEO's
# deliberate cheapness = durability equality (P and D stay exactly equal at 0.250328) and
# therefore Rule PROP's 1:1 residual split for the cohorts.  Nothing else about the block
# model moves.
#
# WRITTEN AS A DERIVATION, NOT AS SEVEN DECIMALS, for the reason in the section header:
# 0.26 * 0.880/0.914 is a non-terminating decimal, and transcribing its display value is
# how Sigma|w| becomes 0.999999 and `_validate()` halts every import.  `_PRE_S_BUDGETS` is
# PROVENANCE -- the budgets as the CEO decided them on 2026-08-04, before the S re-budget --
# and must not be "updated"; the live table is what the scaling below produces.
_PRE_S_BUDGETS = {
    'P': 0.26,
    'R': 0.1634,
    'N': 0.1290,
    'D': 0.26,
    'S': 0.0860,
    'M': 0.0516,
    'C': 0.0500,
}
W_S = 0.120
_S_SCALE = (1.0 - W_S) / (1.0 - _PRE_S_BUDGETS['S'])
_POST_S_BUDGETS = {b: (W_S if b == 'S' else w * _S_SCALE)
                   for b, w in _PRE_S_BUDGETS.items()}

# THE N RE-BUDGET (CEO, 2026-08-10): *"slightly increase the weight in stage 2, but just
# slightly."*  `CycleHeat` is N's Tier-1 member against `EPStoEPSmean` at Tier 2, so the 3:2
# tau split gives it exactly 0.6 of the block: it read -0.074521 and the CEO asked for
# roughly -0.085.
#
# WRITTEN AS THE TARGET WEIGHT, NOT AS A BUDGET, and that is the point of doing it here rather
# than by nudging a decimal until it lands: the number the CEO named is CycleHeat's, so
# CycleHeat's is the number in the file and the block budget is DERIVED from it.  0.085 * 5/3
# is non-terminating, so transcribing its display value is how `Sigma|w|` becomes 0.999999 and
# `_validate()` halts every import (section header, and the same trap the S re-budget records).
#
# THE OTHER BLOCKS PAY FOR IT PROPORTIONALLY -- the SAME mechanism as the S re-budget, for the
# same reason: a flat or hand-picked take would re-open every budget argument in section 15.
# Scaling the six non-N blocks by (1 - W_N_new)/(1 - W_N_old) preserves EVERY ratio among
# non-N metrics, and in particular the CEO's deliberate cheapness = durability equality (P and
# D stay exactly equal), and therefore Rule PROP's 1:1 residual split for the cohorts.
#
# THE COST, PRICED RATHER THAN IMPLIED -- ALL 19 LIVE WEIGHTS, BEFORE -> AFTER.  Every non-N
# weight is multiplied by exactly 0.98005779865 (a 1.99% cut); the two N members rise by
# 1.14062.  This is the whole invoice for the CEO's "just slightly":
#     earnYield                   0.122035 -> 0.119601        RoA                       0.062582 -> 0.061334
#     incomeQuality               0.094393 -> 0.092511        returnOnCapitalEmployed   0.062582 -> 0.061334
#     freeCashFlowPerShareGrowth  0.083443 -> 0.081779        freeCashFlowYield         0.062929 -> 0.061674
#     bVpRatio                    0.065711 -> 0.064401        interestCoverage          0.060000 -> 0.058803
#     marketCapRevQuants          0.049681 -> 0.048690        grahamNumberToPrice       0.040678 -> 0.039867
#     currentRatio                0.040000 -> 0.039202        Piotroski                 0.032093 -> 0.031453
#     tbVpRatio                   0.021904 -> 0.021467        returnOnEquity            0.020861 -> 0.020445
#     grossProfitMargin           0.020861 -> 0.020445        Altman-Z                  0.020000 -> 0.019601
#     revenueGrowth               0.016047 -> 0.015727
#     EPStoEPSmean                0.049681 -> 0.056667   (RISES -- N's Tier-2 member)
#     CycleHeat                  -0.074521 -> -0.085000   (the change being bought)
# The three DELIBERATELY_ZEROED metrics and the two FIN-1-only extractions stay at 0.000000.
#
# THE THESIS MARGIN SURVIVES AND ITS RATIOS ARE UNCHANGED: `earnYield` is still the largest
# single |w| (0.119601 vs `incomeQuality` 0.092511 = 1.293x, exactly as before), because a
# PROPORTIONAL take cannot move a ratio between two non-N metrics.  Against CycleHeat's new
# 0.085000 it holds at 1.407x -- down from 1.638x, and that narrowing IS the purchase.
#
# SIGN IS NOT TOUCHED HERE AND CANNOT BE.  `CycleHeat` carries its sign on the ASSIGNMENT
# (`('N', '-', 1, -1)`), not on the budget, and `_block_vector` multiplies the derived
# magnitude by it -- so raising a BUDGET can only make the penalty larger, never invert it.
# That is the property to check if this number is ever moved again: hot late-cycle must stay
# WORSE, and the test asserts `DEPLOYED['CycleHeat'] < 0`.
CYCLEHEAT_TARGET_W = 0.085
#  CycleHeat = W_N * tau_1 / (tau_1 + tau_2) = W_N * 3/5, so W_N = target * 5/3.
W_N = CYCLEHEAT_TARGET_W * (TIER_TAU[1] + TIER_TAU[2]) / TIER_TAU[1]
_N_SCALE = (1.0 - W_N) / (1.0 - _POST_S_BUDGETS['N'])
GENERAL_BUDGETS = {b: (W_N if b == 'N' else w * _N_SCALE)
                   for b, w in _POST_S_BUDGETS.items()}

# --- B.3  metric -> (block, sub-block, tier, sign) -------------------------- #
# THE GENERAL POOL'S ASSIGNMENT.  A metric ABSENT from this table is OUT OF THE SCHEME and
# scores 0 -- that is how `DcfToPrice` / `BoScore` / `priceGrowth` stay zeroed, and it is
# why absence here is a stronger statement than a 0.000 literal was: there is no number to
# nudge.
#
# `sign` lives on the ASSIGNMENT, not in the metric: a negative-weight metric's column
# holds the quantity as measured (hot EPS, dilution, added leverage) and the weight decides
# that more of it is worse.  Three metrics are negative; all three are read this way.
#
# The tier REASONS are in weighting-strategy section 4.4.  The three that will look wrong
# to a reader and are not:
#   * `incomeQuality` T1 and `freeCashFlowYield` T2, not the reverse -- R's question is
#     about ACCRUALS, and `freeCashFlowYield` answers it through a PRICE-BEARING
#     substitution.  If R's primary carrier were price-bearing, R would partly re-buy the
#     thesis, which is the one thing the P/R separation exists to prevent.
#   * `grahamNumberToPrice` T3 -- DEFECT DEMOTION for audit D3 (`.head(16).mean()` skips
#     NaN, so a name is scored on exactly its profitable quarters; partial on 27 of 100).
#     Promote it back to Tier 2 when D3 is fixed and read B.6 first.
#   * `grossProfitMargin` T3 -- not a defect but a STATED CONFIGURATION: sector
#     neutralisation is CEO-ratified OFF, so the column ranks industries first.  Tier 2 if
#     it is ever turned ON.
GENERAL_ASSIGNMENT = {
    'earnYield':                  ('P', 'E', 1, +1),
    'grahamNumberToPrice':        ('P', 'E', 3, +1),   # T2 once D3 is fixed
    'bVpRatio':                   ('P', 'A', 1, +1),
    'tbVpRatio':                  ('P', 'A', 3, +1),   # a guard on a Tier-1 metric
    'incomeQuality':              ('R', '-', 1, +1),   # D2 FIXED 2026-08-01 -> back to T1
    'freeCashFlowYield':          ('R', '-', 2, +1),
    'CycleHeat':                  ('N', '-', 1, -1),   # NEGATIVE: penalises a hot late cycle
    'EPStoEPSmean':               ('N', '-', 2, +1),
    'RoA':                        ('D', '-', 1, +1),
    'returnOnCapitalEmployed':    ('D', '-', 1, +1),
    'freeCashFlowPerShareGrowth': ('D', '-', 2, +1),
    'grossProfitMargin':          ('D', '-', 3, +1),   # T2 if neutralisation is turned ON
    'returnOnEquity':             ('D', '-', 3, +1),
    # S GAINED ITS FIRST TIER-1 MEMBER on 2026-08-06 -- the carriage note that used to sit on
    # `currentRatio` ("S has NO Tier-1 member") is RETIRED, not merely edited.  Interest
    # coverage states S's question -- can the equity survive its own debt service -- with no
    # proxy step, which is what Tier 1 means; `currentRatio` is a liquidity proxy for it and
    # `Altman-Z` is a 1968 discriminant fitted on US manufacturers, i.e. the right question in
    # a form known to be off-domain for most of this universe.  Their demotion to 2 and 3 is
    # therefore a re-reading of the SAME ladder, not a penalty.
    'interestCoverage':           ('S', '-', 1, +1),
    'currentRatio':               ('S', '-', 2, +1),
    'Altman-Z':                   ('S', '-', 3, +1),
    'marketCapRevQuants':         ('M', '-', 2, +1),   # single member -> tier label inert
    'Piotroski':                  ('C', '-', 2, +1),   # the CEO's 2/3 of the holding pen
    'revenueGrowth':              ('C', '-', 3, +1),   # ... and its 1/3
}
# NOTE ON `C`: the CEO's stated 2/3 - 1/3 split needed NO new mechanism.  It is
# arithmetically identical to the standard tau rule with tiers {2, 3} occupied (2:1), so `C`
# is one more block and not a seventh code path.  A third metric parked here is handled by
# the rule already.

# --- B.4  the derivation ---------------------------------------------------- #
def _block_vector(budgets, assignment):
    """{metric: w} from block budgets + (block, sub, tier, sign) assignments.

    Three rules, in the order they apply:

      1. SUB-BLOCK SCALING (rule 5).  A sub-block's share of its block is scaled by
         (best occupied tier's tau)/3 -- x1 with a Tier-1 member, x2/3 if its best is
         Tier 2, x1/3 if all its members are Tier 3 -- then renormalised across sibling
         sub-blocks.  Inert everywhere in the general pool; it does the real work in REIT,
         where BOTH P-E members are Tier 3 and P-A therefore takes 62% of the cheapness
         budget BY RULE rather than by hand-zeroing `earnYield`.
      2. TIER SPLIT.  The sub-block budget divides across OCCUPIED tiers in tau ratio.
      3. EQUAL WITHIN TIER.

    A block listed in `budgets` with NO member in `assignment` contributes nothing: its
    budget is HELD AND UNSPENT (Rule UNM / Rule OOD with zero members).  That is why the
    returned vector's Sigma|w| is the cohort's SPENDABLE weight and not necessarily 1 --
    the shortfall is the cohort's unpriced risk and is reported, never absorbed silently
    (see COHORT_UNPRICED_RISK).
    """
    members = {}
    for metric, (block, sub, tier, sign) in assignment.items():
        if tier not in TIER_TAU:
            raise RuntimeError('scoringWeights: %r declares tier %r; tiers are %s'
                               % (metric, tier, sorted(TIER_TAU)))
        if block not in budgets:
            raise RuntimeError('scoringWeights: %r is assigned to block %r, which has no '
                               'budget in this vector (blocks: %s)'
                               % (metric, block, sorted(budgets)))
        members.setdefault((block, sub), []).append((metric, tier, sign))

    out = {}
    for block, w_block in budgets.items():
        subs = sorted({sub for b, sub in members if b == block})
        if not subs:
            continue                      # held and unspent -- see the docstring
        # 1. sub-block scaling, renormalised across the sub-blocks that HAVE members
        scaled = {}
        for sub in subs:
            best = min(tier for _m, tier, _s in members[(block, sub)])
            declared = SIGMA_P[sub] if sub in SIGMA_P else 1.0
            scaled[sub] = declared * TIER_TAU[best] / TIER_TAU[1]
        share_total = sum(scaled.values())
        for sub in subs:
            w_sub = w_block * scaled[sub] / share_total
            # 2. + 3. tau over occupied tiers, equal within tier
            by_tier = {}
            for metric, tier, sign in members[(block, sub)]:
                by_tier.setdefault(tier, []).append((metric, sign))
            tau_total = sum(TIER_TAU[t] for t in by_tier)
            for tier, group in by_tier.items():
                per_metric = w_sub * TIER_TAU[tier] / tau_total / len(group)
                for metric, sign in group:
                    out[metric] = sign * per_metric
    return out


# --- B.5  the DEPLOYED vector ----------------------------------------------- #
# 18 non-zero weights over the 23 canonical keys.  Sigma|w| = 1.000000 exactly (asserted at
# import by `_validate()`, and the exact-float identity is asserted by
# test_e2_weight_vector).  The five zeros are the three DELIBERATELY_ZEROED metrics plus
# the two FIN-1-only extractions, which carry no general-pool weight by design.
DEPLOYED = {k: 0.0 for k in METRIC_KEYS}
DEPLOYED.update(_block_vector(GENERAL_BUDGETS, GENERAL_ASSIGNMENT))

# --- B.6  what a future editor needs to know before touching a tier --------- #
# THE THESIS MARGIN ("Bound 2-prime"): `earnYield` must be the LARGEST SINGLE |w| in the
# general vector -- the legibility property behind "P/E is essentially the final boss".  It
# currently reads 0.12675 vs `incomeQuality` 0.09804 = 1.29x, and it has ALREADY BROKEN
# ONCE under a budget change, so test_e2_weight_vector asserts it rather than trusting it.
#
# THE ONE FORWARD TRAP.  Fixing audit D3 promotes `grahamNumberToPrice` to P-E Tier 2, P-E
# then splits 3:2 instead of 3:1, and `earnYield` falls from 0.4875*W_P to 0.39*W_P =
# 0.1014.  Against `incomeQuality` = 0.6*W_R = 0.098 that still holds -- by 0.0034.  It
# holds ONLY because D was held equal to P: at the proportional alternative
# (W_R = 0.1748) `incomeQuality` would be 0.10488 and the D3 fix would break the margin
# unless W_P went to 0.27.  So: fix D3, then RE-RUN the margin assertion; do not fix D3 and
# a budget in the same change.
#
# MINING IS DECLARED EXEMPT from the thesis margin, and the reason is not "it fails" -- it
# is that the failure is NOT RECOVERABLE at any admissible budget.  Post-D3 Mining would
# need W_P > 1.538 * W_N, or W_N cut below its cohort deviation, and cutting W_N abandons the
# cycle-block deviation that is the entire purpose of the Mining cohort.  On stated
# mechanism: for a miner the trailing E *is* the commodity price, so the cycle question is
# genuinely co-dominant with the thesis.  Same footing as FIN-3's ROE exemption.
# (Recorded consequence, per the designer: two declared exemptions out of six is the point
# at which this stops being a property of the DESIGN and becomes a property of the GENERAL
# vector only.)
#
# *** THE EXEMPTION NOW BINDS.  IT IS ABOUT TODAY, NOT ABOUT POST-D3.  (2026-08-10) ***
# This block used to close with "Mining's margin currently PASSES at 1.17x ... the exemption
# is about what happens after D3, not about today".  BOTH HALVES OF THAT ARE NOW FALSE, and
# they are corrected rather than deleted because the reversal is the informative part.
# The CEO's N re-budget (W_N 0.124201 -> 0.141667, so `CycleHeat` reads the -0.085 he asked
# for) reaches the cohorts through Rule PROP, and Mining's N RATIO IS ABOVE THE ANCHOR'S by
# design -- so Mining's N rose further than the general vector's, from 0.1822 to 0.2078.
# MEASURED on the shipped vector: `CycleHeat` 0.124667 against `earnYield` 0.120861, a ratio
# of **0.9695**.  The cycle metric is now the LARGEST SINGLE WEIGHT in the Mining cohort and
# the thesis is second, which re-ranks the Mining side-list.
# That is the declared exemption doing exactly what it was declared for -- but it is now an
# OBSERVATION, not a hypothetical, and `test_e2_weight_vector` pins it as such.  If the cycle
# is not wanted as Mining's largest weight, the lever is Mining's N ratio in `_cohort_ratios`,
# NOT the general budget: lowering W_N would give back the -0.085 the CEO asked for.
THESIS_METRIC = 'earnYield'
THESIS_MARGIN_EXEMPT_COHORTS = frozenset({
    # cohort -> the exemption is declared, with its mechanism, in the notes above / below
    'Mining',            # cycle co-dominant with the thesis; unrecoverable post-D3
    'REIT',             # `earnYield` is Tier 3 there by rule -- exempt by construction
    'InvestmentVehicle',  # P-E is out of domain: there is no operating E to be the boss
    'BalanceSheetFin',   # ROE is the industry's own yardstick on regulated scarce capital
})

# The three DELIBERATELY-ZEROED metrics, named so downstream guards can key off the
# INTENT rather than off `w == 0` (postBoRank._assert_offline_dcf_is_score_neutral and
# the research benches both rest on these being zero).
# Their VALUES are pinned by `test_scoring_weights_single_source.py`, NOT by the
# import-time `_validate()` below -- on purpose.  A re-weighting that resurrects one of
# them is a DESIGN decision, and blocking `import scoringWeights` is the wrong way to
# raise it: it would turn a weight edit into a dead pipeline.  A failing test (plus
# postBoRank's own SystemExit refusal for DcfToPrice) says the same thing without
# holding the run hostage.
DELIBERATELY_ZEROED = ('DcfToPrice', 'BoScore', 'priceGrowth')


# --- C. the LEGACY vector -- A/B ONLY, NOT DECISIONAL -----------------------
# Pre-2026-07-14 double-counted DEFAULT weights (the certified 30.0% target-cell
# baseline).  Retained for A/B against the promoted mu theory prior now decisional in
# DEPLOYED; NOT decisional.  SAME key set / order / eqMet mapping as DEPLOYED (enforced
# below), so swapping it in reproduces the pre-promotion picks and nothing else.
#
# NOT a delta from DEPLOYED: all 21 of the pre-E-2 values differ, so a delta table would be
# a second full vector wearing a disguise.  It is a named vector on the CANONICAL key set
# instead -- which is what actually prevents silent drift (a metric added to DEPLOYED and
# forgotten here is what would break, and _validate() catches exactly that).
#
# DcfToPrice = 0.35 here vs 0.000 deployed is a REAL, DELIBERATE difference, not a bug;
# `test_scoring_weights_single_source` pins it so nobody "tidies" it away.
# Integer literals are preserved as ints (2, 1) because that is what the vector has
# always emitted.
#
# THE TWO E-2 METRICS ARE 0 HERE, and the zero is the honest value, not a placeholder: this
# vector IS the pre-2026-07-14 pipeline, which had no such columns.  They are listed only
# because `_validate()` requires every named vector to cover the canonical key set exactly
# -- a MISSING key does not raise downstream, it scores that metric at weight 1.0.
LEGACY = {
    'RoA':                        2,
    'earnYield':                  2,
    'grahamNumberToPrice':        1,
    'bVpRatio':                   0.25,
    'revenueGrowth':              1,
    'incomeQuality':              1,
    'returnOnEquity':             1,
    'returnOnCapitalEmployed':    1,
    'currentRatio':               0.35,
    'grossProfitMargin':          0.75,
    'freeCashFlowYield':          2,
    'freeCashFlowPerShareGrowth': 1.5,
    'DcfToPrice':                 0.35,
    'marketCapRevQuants':         0.25,
    'Altman-Z':                   0.5,
    'Piotroski':                  0.75,
    'tbVpRatio':                  0.5,
    'BoScore':                    0.1,
    'EPStoEPSmean':               0.5,
    'priceGrowth':                0.5,
    'CycleHeat':                 -0.5,   # Negative weight penalizes hot late-cycle stocks
    'shareCountChange':           0,     # did not exist pre-2026-07-14 -- see the note above
    'longTermDebtChange':         0,     # did not exist pre-2026-07-14 -- see the note above
    'interestCoverage':           0,     # did not exist pre-2026-07-14 -- see the note above
    'navPerShareGrowth':          0,     # did not exist pre-2026-07-14 -- see the note above
}


# =========================================================================== #
# --- D.  the five COHORT vectors -- DERIVED, not hand-set (E-2, 2026-08-04) - #
# =========================================================================== #
# Threaded into postBoScoreRanking(weight_override=...) per cohort by postBo; the
# general/main pool uses NO override (DEPLOYED).  weight 0 -> metric dropped from
# AggScore (constant/neutral in rankOfRanks); does not change cohort membership.
#
# Cohort LABELS are single-sourced in carveOut (REIT_SECTOR-derived 'REIT'/'Mining' and
# the FIN1_VEHICLE / FIN2_MANAGER / FIN3_BALSHEET constants).  They are spelled out as
# literals here only because carveOut imports THIS module, so importing it back would be
# a cycle; carveOut asserts at import that the two agree.
#
# WHAT REPLACED THE 105 HAND-SET NUMBERS.  Each cohort is now (i) a set of per-block RATIOS
# to the general budget, argued once per cohort, (ii) an in-domain/tier DELTA against
# GENERAL_ASSIGNMENT, and (iii) the same `_block_vector` derivation.  Three rules govern
# the cohort budgets:
#
#   Rule PROP  cohort block budget = general budget x the cohort's stated ratio; the
#              RESIDUAL goes to P and D in the GENERAL vector's own P:D ratio.  That ratio
#              is currently 1:1 (the CEO held durability equal to cheapness), so the
#              residual splits evenly -- and Rule PROP tracks whatever P:D he sets later,
#              which is why this question does not recur.
#   P-floor    a cohort's W_P may not fall below the GENERAL W_P (0.26).  It tracks the general
#              decision by construction rather than being a second number to maintain.
#              *** IT DOES NOT CURRENTLY BIND ANYWHERE, and saying otherwise was wrong. ***
#              The design introduced it to stop Mining's cycle block overtaking the thesis, and
#              at the old proportional anchor it did exactly that (Rule PROP alone landed Mining
#              at P = 0.2256).  Under the contamination override we ship (see Mining's note in
#              D.1) P takes the residual and lands at 0.2726, comfortably above the floor -- and
#              every other cohort clears it too.  It is retained as a SAFETY RAIL against a
#              future ratio edit, not as a rule doing live work.  Kept rather than retired
#              because it is one comparison and it raises LOUDLY at import; but do not describe
#              it as load-bearing, and do not let a test docstring claim it is.
#   Rules OOD / UNM  a block whose QUESTION does not apply is OOD: W_B = 0, and the residual
#              rule redistributes it.  A block whose question APPLIES but has no instrument
#              is UNM: the budget is HELD AND LEFT UNSPENT, and the shortfall is reported as
#              that cohort's unpriced risk (COHORT_UNPRICED_RISK).  The distinction is
#              deliberate and it is the whole point -- redistributing an uncarried block
#              silently converts "we cannot measure this" into "this does not matter".
#
# --- WHAT THIS CHANGES vs the pre-E-2 cohort vectors, flagged not buried ------
# `BoScore` NOW GOES TO 0 IN ALL FIVE COHORTS.  It previously carried 0.1 raw (~0.7-3.0%
# normalised) in every cohort while being 0.000 in the general vector, and that was recorded
# as a KNOWN OPEN ISSUE the CEO had not ruled on.  E-2 resolves it as a CONSEQUENCE rather
# than a tidy-up: `BoScore` is a composite of the other metrics, so it belongs to no block,
# and a metric with no block has no weight under this scheme.  It is also ARITHMETICALLY
# FORCED -- the cohort vectors renormalise by Sigma|w|, so leaving 0.1 on BoScore would
# rescale every designed weight (FIN-1's `bVpRatio` would land at 0.344 instead of the
# designed value) and the published cohort numbers would not be the ones that were argued.
# `priceGrowth` and `DcfToPrice` stay 0 in every cohort, as before, for the reasons below:
#   priceGrowth  the ONE Stage-2 metric with an acknowledged, UNCORRECTED semi-annual 2x
#                LEVEL bias (a 6-month price move scored against a quarterly 3-month one);
#                the window scaling cannot fix a level.
#   DcfToPrice   computed from a LIVE DCF call using the CURRENT market price, while every
#                other metric is as-of period end -- a basis mix inside the cohort score and
#                the one channel that can import lookahead, which is exactly why the PIT
#                reproduction drops it (stage2_pit.DROP_METRICS).

# --- D.1  the ratio anchor -------------------------------------------------- #
# The cohort DEVIATIONS were argued at the section-14.5.3 anchor, NOT at today's budgets.
# Only the RATIO cohort/general propagates (Rule PROP), so these two tables are PROVENANCE:
# they record the numbers the domain arguments were made in, and the live budgets are
# computed from the ratio.  Do not "update" them to today's values -- that would silently
# turn a ratio into a level.
_RATIO_ANCHOR_GENERAL = {'P': 0.25, 'R': 0.19, 'N': 0.15, 'D': 0.25, 'S': 0.10, 'M': 0.06}
_RATIO_ANCHOR_COHORT = {
    # Mining -- the cycle question nearly becomes the thesis.  N rises hardest of any
    # cohort (a miner's trailing E *is* the commodity price).  D is the ONE cohort where the
    # residual rule is overridden: two of D's three tiers are price-CONTAMINATED for a miner
    # (`grossProfitMargin` is a price observation, ROE/ROCE read the commodity cycle), so the
    # marginal budget would buy more of a contaminated instrument.
    #
    # WHICH OVERRIDE -- A DESIGN CHOICE, NOT AN ARITHMETIC NECESSITY (corrected 2026-08-04;
    # an earlier version of this comment claimed the latter and was WRONG).  At the equal-D
    # anchor there are THREE readings, and TWO of them normalise:
    #   (i)  D pinned at its 0.72 ratio, P takes the residual -> P 0.2726 / D 0.1872, Sigma = 1
    #   (ii) P floored at the general 0.26, D absorbs         -> P 0.2600 / D 0.1998, Sigma = 1
    #   (iii) both pinned                                     -> Sigma = 0.9874, ruled out
    # (i) and (ii) COINCIDE at the old proportional anchor, so reproducing section 15.7's table
    # cannot discriminate between them -- and section 15.7 states (ii)'s mechanism.
    # WE SHIP (i), on substance: the 0.72 ratio is not a leftover, it ENCODES the contamination
    # discount on Mining's durability instruments, and (ii) would silently dilute that discount
    # to 0.7685 -- weakening a stated design claim as a side effect of an anchor change nobody
    # made for that reason.  Under (i) the contamination discount is preserved exactly.
    # Impact is contained either way: Mining is a side-list and never the emitted shortlist.
    'Mining':            {'R': 0.20, 'N': 0.22, 'S': 0.10, 'M': 0.05, 'D': 0.18},
    # REIT -- R at its highest anywhere: when GAAP E is depreciation-dominated and unusable,
    # the cash-confirmation question is what is left.  N lower (property cycles are slower).
    'REIT':              {'R': 0.23, 'N': 0.12, 'S': 0.10, 'M': 0.05},
    # FIN-2 managers/brokers/platforms -- near-general; small trims on R and N only.
    'FinManager':        {'R': 0.18, 'N': 0.14, 'S': 0.10, 'M': 0.06},
    # FIN-3 banks/lenders/insurers -- R collapses (a bank's earnings are misstated through
    # PROVISIONING ADEQUACY, not accruals-vs-cash, and no instrument for that exists in
    # either stage: carriage note).  N rises for the CREDIT cycle -- a low P/E on
    # pre-normalisation provisions is the canonical bank value trap.  S rises because capital
    # adequacy is the dominant risk here, and is then held UNSPENT because the pipeline
    # cannot see it.
    'BalanceSheetFin':   {'R': 0.08, 'N': 0.16, 'S': 0.14, 'M': 0.04},
}

# --- D.2  per-cohort deltas ------------------------------------------------- #
# `ood`    metrics whose QUESTION or INSTRUMENT does not apply in this cohort -> dropped
#          from the assignment, so their block's budget flows to the surviving members (or,
#          if none survive, is held unspent).
# `retier` a metric whose DIRECTNESS differs in this cohort.
# `add`    a metric that exists ONLY in this cohort's vector.
# `c_ood`  the `C` holding pen is out of domain -> W_C = 0, redistributed by the residual.
_COHORT_DELTAS = {
    'Mining': dict(ood=(), retier={}, add={}, c_ood=False),

    'REIT': dict(
        # THE S BUDGET IS UNPARKED (CEO, 2026-08-06), AND DELIBERATELY SO -- not as a side
        # effect of `interestCoverage` landing in GENERAL_ASSIGNMENT.
        #
        # WHAT THIS NOTE USED TO SAY, kept so the condition and its discharge read together:
        # "Rule UNM with ZERO members: `Altman-Z` and `currentRatio` are the wrong instruments
        # for a business that runs negative working capital BY DESIGN, and zeroing them in the
        # old vector removed the SOLVENCY QUESTION from the most leverage-sensitive cohort in
        # the set.  The budget is held and reported unspent UNTIL NET-DEBT/EBITDA OR INTEREST
        # COVERAGE EXISTS."
        #
        # THE CONDITION IS MET, BY `interestCoverage` -- the second of the two named
        # instruments, added to Stage-1 on 2026-08-05 (createDicts, Tier B) and promoted to a
        # Stage-2 S-block Tier-1 metric on 2026-08-06.  (`netDebtToEBITDA` also exists now, as
        # a Stage-1 three-branch `special`, but it is NOT a Stage-2 column, so it is not what
        # discharges this.)  `Altman-Z` and `currentRatio` STAY out of domain for exactly the
        # reason above -- nothing about them improved -- so REIT's S block is a lone Tier-1
        # member and takes the whole cohort S budget (0.120), and REIT's reported unpriced
        # risk goes 8.6% -> 0%.
        #
        # AND THE HONEST LIMIT, on the same footing as FIN-1's in D.3: interest coverage
        # instruments the refinancing question, it does not fully ANSWER it.  A REIT's real
        # solvency risk is the maturity WALL and the LTV covenant, neither of which a
        # flow-coverage ratio can see; a name that comfortably covers today's interest bill
        # can still fail to roll its debt.  0% unpriced means "the block now has an
        # instrument", not "REIT leverage risk is priced".
        ood=('Altman-Z', 'currentRatio'),
        # GAAP net income for a property company bears almost no relation to distributable
        # cash, so BOTH P-E members are Tier 3 and sub-block scaling (rule 5) fires: P-A
        # takes ~62% of the cheapness budget BY RULE.  That reproduces what the old vector
        # achieved by zeroing `earnYield` outright, without the collateral damage -- a
        # REIT's earnings multiple is a poor instrument, not a meaningless one.
        retier={'earnYield': 3},
        add={}, c_ood=False),

    'InvestmentVehicle': dict(   # FIN-1
        # A closed-end fund or BDC *is* its NAV: earnings are mark-to-market noise and cash
        # flow is portfolio churn.  N and D are Rule OOD (not badly measured -- INAPPLICABLE)
        # and P-E is OOD too (there is no operating E), so sigma_P-A becomes 1.00 by rule.
        # The `C` block is OOD on COMPONENT-level grounds, and this is the sharpest finding
        # in the design: 7 of Piotroski's 9 components are undefined or degenerate here, and
        # an undefined component does not go MISSING -- it scores 0, i.e. a FAILED test.
        # (Verified in stage2_metrics.piotroski: a NaN input makes every comparison False.)
        # So the composite is not merely uninformative in FIN-1, it is SYSTEMATICALLY
        # PUNITIVE against every member of the cohort, which is worse than absent.
        ood=('earnYield', 'grahamNumberToPrice',          # P-E: no operating E
             'incomeQuality', 'freeCashFlowYield',        # R: portfolio churn, not accruals
             'CycleHeat', 'EPStoEPSmean',                 # N: mark-to-market noise
             'RoA', 'returnOnCapitalEmployed', 'returnOnEquity',
             'grossProfitMargin', 'freeCashFlowPerShareGrowth',   # D: no operating business
             'currentRatio', 'Altman-Z',                  # S: no working-capital cycle
             # `interestCoverage` IS OOD HERE FOR THE SAME REASON THE WHOLE D BLOCK IS
             # (2026-08-06): it is operatingIncome / interestExpense, and a closed-end fund
             # or BDC has NO OPERATING INCOME -- its P&L is mark-to-market portfolio revaluation
             # plus a management fee.  The numerator is the same quantity FIN-1 already declares
             # inapplicable eleven lines up, so admitting it into S here would contradict that
             # ruling.  FIN-1's S question is answered by `longTermDebtChange` (leverage
             # DIRECTION against the regulatory asset-coverage limit), which stays this block's
             # sole member and keeps the whole 0.15.
             'interestCoverage'),
        # `tbVpRatio` CO-PRIMARY with `bVpRatio`.  For an investment vehicle the
        # book-to-tangible-book wedge is not a goodwill GUARD on the thesis, it IS the
        # thesis's failure mode: goodwill from acquired management contracts, capitalised
        # deal costs and non-portfolio intangibles are precisely the assets that are NOT the
        # fund's NAV.  This promotion took the largest single FIN-1 column 0.597 -> 0.393 IN
        # ISOLATION; the two added metrics below then take it to **0.275**, which is what
        # actually ships.  (0.393 is the design's published figure for the promotion alone and
        # is asserted separately by test_e2_weight_vector, so both numbers are real -- but the
        # SHIPPED concentration is 0.275 and this comment used to stop at 0.393.)
        #
        # AND THE MEASURED CAVEAT, because it changes what 0.275 means.  On the live FIN-1 pool
        # rho(bVpRatio, tbVpRatio) = 0.991 and tbV/bV is EXACTLY 1.0000 on 5 of 7 members -- a
        # fund carries no goodwill, so the wedge this promotion is built on is ~zero and the
        # co-primary split is COSMETIC for most of the cohort.  The price-to-NAV axis therefore
        # carries 0.55, not 0.275.  Still a real improvement (that axis held ~0.90 before), and
        # the improvement is the three OTHER columns, not the split -- but do not quote 0.275 as
        # the cohort's concentration without this sentence attached.  The design's own
        # [MEASUREMENT GATE] asked for exactly this measurement and its stated consequence is a
        # price-to-NAV / discount-persistence METRIC REQUEST, not a further weight change.
        retier={'tbVpRatio': 1},
        # THE TWO EXTRACTED PIOTROSKI COMPONENTS -- see the block note below -- plus
        # `navPerShareGrowth`, added 2026-08-06 as R's TIER-1 CARRIER (see D.4).  R's 0.15 now
        # splits 3:2 over the two occupied tiers: navPerShareGrowth +0.090, shareCountChange
        # -0.060 (it previously held R alone at 0.150).  NOTHING IN P MOVES -- `bVpRatio` and
        # `tbVpRatio` keep 0.275 each, which is the point: this is the minimal change that
        # instruments R's question properly without re-opening the co-primary split that the
        # E-2 design argued and measured.
        add={'navPerShareGrowth':  ('R', '-', 1, +1),
             'shareCountChange':   ('R', '-', 2, -1),
             'longTermDebtChange': ('S', '-', 2, -1)},
        c_ood=True),

    'FinManager': dict(ood=(), retier={}, add={}, c_ood=False),   # FIN-2

    'BalanceSheetFin': dict(   # FIN-3
        ood=('returnOnCapitalEmployed',      # capital employed is not a meaningful denominator
             'grossProfitMargin',            # no gross margin exists
             'freeCashFlowPerShareGrowth', 'freeCashFlowYield',   # bank FCF is not a quantity
             'Altman-Z', 'currentRatio',     # 1968 US manufacturers; deposits are not debt
             # `interestCoverage` IS OOD HERE TOO, AND THIS IS A DECISION MADE ON PURPOSE
             # (2026-08-06) rather than a metric quietly riding into a cohort.  For a bank or
             # insurer interest expense is a COST OF GOODS, not a financing charge: paying
             # depositors and policyholders IS the business, so operatingIncome /
             # interestExpense is a spread-margin reading dressed as a solvency ratio, and a
             # bank with a fat net interest margin would score as "safe" on it.  FIN-3's S
             # question is CAPITAL ADEQUACY (CET1 / risk-weighted assets), the pipeline still
             # cannot see it, and Rule UNM therefore still holds this cohort's S budget
             # UNSPENT -- now at 0.168 rather than 0.1204, because the S ratio propagates the
             # raised general budget.  Unparking REIT (CEO, 2026-08-06) was scoped to REIT.
             'interestCoverage'),
        # ROE is the industry's own yardstick on regulated scarce capital.  This is FIN-3's
        # DECLARED thesis-margin exemption: ROE ends up above `earnYield` on purpose.
        retier={'returnOnEquity': 1, 'RoA': 2},
        add={},
        # `C` OOD here too, on the same component-level grounds: dGrossMargin is undefined,
        # dCurrentRatio is out of domain, the CFO-based points are unusable (a bank's CFO is
        # dominated by balance-sheet flows) and dAssetTurnover is meaningless -- 4-5 of 9.
        c_ood=True),
}

# --- D.3  THE TWO EXTRACTED PIOTROSKI COMPONENTS (FIN-1 only) --------------- #
# WHY THEY EXIST.  FIN-1's R block ("is the NAV real, or an artifact?") is the only failure
# mode a closed-end fund has, and it had NO INSTRUMENT -- the design carried it as Rule UNM
# with 15% of the cohort's budget reported unpriced.  FIN-1's S block was in the same
# position.  Two of Piotroski's nine points ARE meaningful for an investment vehicle and
# were locked inside a composite that is 7/9 undefined there:
#   `shareCountChange`   -> R.  Issuing shares BELOW net asset value is the canonical BDC
#                           red flag: total NAV rises while NAV PER SHARE falls, so a name
#                           can look cheap on `bVpRatio` while bleeding per-share value.
#                           That is an ARTIFACT test on the P-A thesis, which is exactly
#                           R's question -- so it goes in R, not in a new block.
#   `longTermDebtChange` -> S.  A vehicle's survival risk is leverage: BDCs run near a
#                           regulatory asset-coverage limit, and the direction of the
#                           leverage ratio is the one solvency signal the pipeline can see.
# Both take a NEGATIVE weight (more dilution / more leverage = worse), like `CycleHeat`.
# With a single member each, the tier label is inert -- it is recorded for when a second
# instrument arrives.
#
# WHAT THIS IS *NOT*, and it is the trap the project's own standing premise names.  These are
# NOT a "fraction of computable Piotroski tests passed" -- that would REWARD a company for
# having fewer tests apply to it, and missing data must never reward by default.  Each is a
# standalone CONTINUOUS quantity whose SIGN is the Piotroski point it came from, and an
# undefined input yields NaN, which the normalisation path imputes to the column MEDIAN
# (neither a pass nor a fail).  See stage2_metrics.share_count_change /
# long_term_debt_change for the per-input NaN rules.
#
# HONEST LIMITS, recorded rather than papered over:
#   * Unparking R and S takes FIN-1's reported unpriced risk from 30% to 0%, and that
#     overstates what two binary-derived columns buy.  They INSTRUMENT the two blocks; they
#     do not fully ANSWER them.  The proper instruments -- price-to-NAV / discount
#     persistence for R, asset-coverage ratio for S -- are still absent.
#   * `longTermDebtChange` is a CHANGE measure, so an unlevered vehicle that stays unlevered
#     reads 0 while a levered one that deleverages reads better than it.  That is the
#     Piotroski component's own semantics, faithfully extracted; a LEVEL term would be a new
#     metric, not an extraction.
#   * Neither column is in `reviewReference.PLAYBOOK_METRICS` or the presentation deck, so a
#     FIN-1 name's review page does not yet explain 30% of its score.  Both are
#     currency-invariant and therefore admissible there; wiring them is a separate change.


# --- D.4  `navPerShareGrowth` -- FIN-1's R-block Tier 1 (2026-08-06) -------- #
# WHAT THE CEO ASKED FOR AND WHAT THIS IS INSTEAD, stated first because the substitution is
# the substantive part of the change and a reader must not mistake this column for the thing
# it replaced.  The ask was a PRICE-TO-NAV instrument for FIN-1.  A genuine one is NOT
# COMPUTABLE here: no endpoint this pipeline fetches carries a fund-published NAV, and the
# only surrogate is GAAP book equity -- which equals NAV only under US investment-company
# accounting (ASC 946) and is an approximation everywhere else.  Worse, that surrogate is
# ALREADY IN THE VECTOR as `bVpRatio`, and a discount-PERSISTENCE column measures rho = +0.806
# against it: a price-to-NAV metric built this way would be `bVpRatio` wearing a new name and
# would concentrate the P-A thesis rather than instrument R.
#
# SO THE COLUMN MEASURES A DIFFERENT QUESTION -- not the discount LEVEL (already carried) but
# whether the NAV IS REAL.  A fund whose stated NAV is an accounting artifact has one
# observable signature: NAV PER SHARE THAT FAILS TO COMPOUND.  That is exactly R's question
# ("is the quantity the thesis rests on an artifact?") applied to the P-A thesis, and it sits
# at Tier 1 because it states that question with no proxy step, where `shareCountChange`
# (Tier 2) guards ONE specific way it fails -- issuance below NAV.
#
# IT IS A PROXY AND IS LABELLED ONE EVERYWHERE (docstring, reviewReference.METRIC_BASIS): the
# NAV leg is `bookValuePerShare`, exact for ASC-946 vehicles and approximate for the rest of
# the cohort.  Do not let it read as a fund-published NAV in any artifact.
#
# MEASURED ON THE LIVE FIN-1 COHORT (88 names): computable on 87 of 88, and rho = -0.277
# against `bVpRatio` -- a genuinely independent axis, which is the whole reason it is worth a
# Tier-1 slot that a price-to-NAV rebuild would not have earned.
#
# FIN-1 ONLY, and that is a scope decision rather than a claim that the quantity is
# meaningless elsewhere: for an operating company book value per share is a residual, not the
# thesis, so its growth rate would be a weak durability reading competing with instruments D
# already has.  It is absent from GENERAL_ASSIGNMENT, so it scores 0.000 in the general pool
# and in the other four cohorts by construction.


def _cohort_budgets(label):
    """Rule PROP + P-floor + OOD/UNM -> this cohort's seven block budgets (Sigma = 1)."""
    if label == 'InvestmentVehicle':
        # FIN-1 is CONSTRUCTED DIRECTLY, not propagated: nothing in it derives from the
        # general anchor (three blocks are OOD and P-E is OOD inside a fourth), so there is
        # no ratio to carry forward.  W_P = 0.55 deliberately BREAKS the W_P < 1/2
        # value-trap bound -- correctly, because that bound's derivation assumes a
        # meaningful qualifier set and in FIN-1 the qualifier QUESTIONS do not apply.
        return {'P': 0.55, 'R': 0.15, 'N': 0.0, 'D': 0.0, 'S': 0.15, 'M': 0.15, 'C': 0.0}

    delta = _COHORT_DELTAS[label]
    ratios = _RATIO_ANCHOR_COHORT[label]
    budgets = {b: GENERAL_BUDGETS[b] * ratios[b] / _RATIO_ANCHOR_GENERAL[b]
               for b in ratios}
    budgets['C'] = 0.0 if delta['c_ood'] else GENERAL_BUDGETS['C']
    residual = 1.0 - sum(budgets.values())
    if 'D' in budgets:
        # the contamination override (Mining): D is pinned by its ratio and P takes the
        # whole residual.  The P-floor is then a CHECK, not an adjustment.
        budgets['P'] = residual
    else:
        # Rule PROP proper: split the residual between P and D in the general P:D ratio.
        pd_ratio = GENERAL_BUDGETS['P'] / GENERAL_BUDGETS['D']
        budgets['P'] = residual * pd_ratio / (1.0 + pd_ratio)
        budgets['D'] = residual / (1.0 + pd_ratio)
    if budgets['P'] < GENERAL_BUDGETS['P'] - 1e-12:
        raise RuntimeError(
            'scoringWeights: cohort %r lands W_P = %.6f, BELOW the P-floor (the general '
            'W_P = %.6f). The floor exists to stop a cohort deviation inverting the '
            'thesis -- re-argue the cohort ratios rather than shaving P.'
            % (label, budgets['P'], GENERAL_BUDGETS['P']))
    return budgets


def _cohort_assignment(label):
    """GENERAL_ASSIGNMENT with this cohort's in-domain / tier / added-metric deltas."""
    delta = _COHORT_DELTAS[label]
    out = {m: spec for m, spec in GENERAL_ASSIGNMENT.items() if m not in delta['ood']}
    if delta['c_ood']:
        out = {m: spec for m, spec in out.items() if spec[0] != 'C'}
    for metric, tier in delta['retier'].items():
        block, sub, _old, sign = out[metric]
        out[metric] = (block, sub, tier, sign)
    out.update(delta['add'])
    return out


COHORT_LABELS = ('Mining', 'REIT', 'InvestmentVehicle', 'FinManager', 'BalanceSheetFin')

#  IN-BUDGET cohort weights: each metric at its share of the cohort's SEVEN-BLOCK budget, so
#  Sigma|w| here is the cohort's SPENDABLE weight (< 1 wherever a block is held unspent).
#  `COHORT_WEIGHTS` below renormalises to 1 for scoring; the gap is the unpriced risk.  The
#  name `_RAW` is kept because every consumer and test already reads it.
COHORT_WEIGHTS_RAW = {
    label: {k: 0.0 for k in METRIC_KEYS} for label in COHORT_LABELS}
for _label in COHORT_LABELS:
    COHORT_WEIGHTS_RAW[_label].update(
        _block_vector(_cohort_budgets(_label), _cohort_assignment(_label)))
del _label


# --- E. derivations ---------------------------------------------------------
def normalise(vector):
    """Scale a weight vector so Sigma|w| = 1.  Signs preserved; an all-zero vector is
    returned unchanged (there is nothing to normalise to)."""
    tot = sum(abs(float(v)) for v in vector.values())
    if tot <= 0:
        return dict(vector)
    return {k: float(v) / tot for k, v in vector.items()}


def sum_abs(vector):
    return sum(abs(float(v)) for v in vector.values())


def deployed_weights():
    """A FRESH copy of the canonical flat {metric: w} vector, in emission order.

    This is what every consumer should call instead of writing the numbers out.
    A copy, so a caller that mutates its own prior cannot corrupt the canon."""
    return {k: DEPLOYED[k] for k in METRIC_KEYS}


# No `legacy_weights()` counterpart on purpose: the legacy vector's only consumers want the
# getPostDict SHAPE, which `createDicts.getPostDict_legacy()` already gives them, and an
# unused second accessor is one more thing to keep honest.  Read `LEGACY` directly if you
# want the flat form.


# --- COHORT VECTORS NORMALISED TO Sigma|w| = 1 (domain review S7, 2026-07-26) ---
# The general vector sums to exactly 1.000; Mining summed to ~14.35, so cohort AggScores
# were ~14x the general scale while shipping SIDE BY SIDE with general values in three CSV
# families, against a presentation chip whose stated empirical range is general-only.
# Normalising is RANK-INVARIANT within a cohort (dividing every weight by one positive
# constant scales the score, never reorders it -- asserted in the proof), so this changes
# no cohort's ordering and only puts the numbers on a comparable scale.
#
# (The "~14.35" above is the sum as it stood WHEN S7 WAS WRITTEN -- the cohort vectors were
# then RELATIVE numbers on an arbitrary scale.  Since E-2 they are absolute shares of a
# seven-block budget, so `COHORT_WEIGHTS_RAW` sums to the cohort's SPENDABLE weight, and this
# normalisation is exactly the design's stated "renormalise the spendable weights to 1".
# Kept verbatim rather than edited so the S7 ruling still reads as it was made.)
COHORT_WEIGHTS = {label: normalise(vec) for label, vec in COHORT_WEIGHTS_RAW.items()}

#  THE UNPRICED-RISK REPORT -- Rule UNM's other half, and the reason the renormalisation
#  above is not the whole story.  A block whose question APPLIES but has no instrument keeps
#  its budget and spends it on nothing; renormalising then raises every SPENDABLE share by
#  1/spendable, which is exactly how "we cannot measure this" turns into "this does not
#  matter" if nobody prints the residue.  So it is a first-class derived number, reported per
#  cohort at the point of use (postBo's carve-out banner) rather than left implicit.
#  REIT 8.6% (leverage / refinancing risk: no net-debt-to-EBITDA, no interest coverage) and
#  FIN-3 12.04% (capital adequacy: the pipeline cannot see it) are the two live ones.
COHORT_UNPRICED_RISK = {label: 1.0 - sum_abs(vec)
                        for label, vec in COHORT_WEIGHTS_RAW.items()}


# --- F. structural invariants, checked AT IMPORT ----------------------------
# These check STRUCTURE (key sets, normalisation), never a value judgement, so a
# deliberate re-weighting never trips them -- but a metric added to DEPLOYED and
# forgotten in LEGACY or a cohort does, LOUDLY, at import rather than as a silent
# `weight_series.get(col, 1)` default of 1.0 deep inside Stage-2
# (postBoRank.postBoScoreRanking's weighting loop, and
# tune_run.FastPitContext._finish) -- which is the actual failure mode this replaces:
# a MISSING key does not raise, it scores that metric at weight 1.0 against a vector
# whose other weights are ~0.05.
_SUM_TOL = 1e-12


def _validate():
    canon = set(METRIC_KEYS)
    if len(METRIC_KEYS) != len(canon):
        raise RuntimeError('scoringWeights: duplicate key in METRIC_KEYS')
    named = [('DEPLOYED', DEPLOYED), ('LEGACY', LEGACY)]
    named += [('COHORT_WEIGHTS_RAW[%r]' % k, v) for k, v in COHORT_WEIGHTS_RAW.items()]
    for name, vec in named:
        missing = sorted(canon - set(vec))
        extra = sorted(set(vec) - canon)
        if missing or extra:
            raise RuntimeError(
                'scoringWeights: %s has drifted off the canonical metric key set. '
                'missing=%s extra=%s. Every weight vector must cover EXACTLY the %d '
                'keys in METRIC_KEYS -- a missing key silently scores that metric at '
                'weight 1.0 downstream.' % (name, missing, extra, len(METRIC_KEYS)))
    tot = sum_abs(DEPLOYED)
    if abs(tot - 1.0) > _SUM_TOL:
        raise RuntimeError(
            'scoringWeights: DEPLOYED no longer normalises -- Sigma|w| = %.17g, '
            'expected 1.0. Re-weighting is fine, but the vector must still sum to 1 '
            '(every published AggScore range and the presentation chip assume it).'
            % tot)
    for z in DELIBERATELY_ZEROED:
        if z not in canon:
            raise RuntimeError('scoringWeights: DELIBERATELY_ZEROED names %r, which is '
                               'not a metric key' % z)


_validate()
