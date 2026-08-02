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

TO CHANGE A WEIGHT: edit `DEPLOYED` here and NOTHING ELSE.  `MU_GENERAL` (the tuner
prior) and `W_THEORY` (the research bench) recompute automatically; the Sigma|w| = 1
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
)

# all 21 keys in EMISSION order (postBm block, then postNew block).
METRIC_KEYS = tuple(POSTBM_EQMET) + POSTNEW_KEYS


# --- B. the DEPLOYED vector -- THE AUTHORITY --------------------------------
# DECISIONAL weights. Promoted 2026-07-14 (MD directive, valuation-specialist
# theory prior) to the mu THEORY-PRIOR vector that produced the certified 38.5%
# target-cell beat-rate (top-20, 36mo, pooled buy2021+buy2022) -- up from the 30.0%
# baseline under the legacy double-counted defaults. This is exactly the LOCKED
# effective weights (GP=0.100 primary variant) mapped onto the getPostDict keys and
# normalized Sigma=1. ONLY the weight VALUES changed vs the legacy vector -- every
# metric key / eqMet / scoring path / ordering is identical, so the as_of=None
# machinery invariant holds: only the picks move.
# Three metrics zeroed (DcfToPrice / BoScore / priceGrowth -- two drops + the
# priceGrowth bug); CycleHeat stays NEGATIVE (late-cycle penalty). Legacy defaults
# preserved in LEGACY for A/B -- NOT deleted.
#
# `# legacy N` on each line is the value the SAME metric carried in LEGACY, kept
# in-line so the promotion is readable without diffing two tables.
DEPLOYED = {
    'RoA':                        0.060,    # legacy 2
    'earnYield':                  0.0605,   # legacy 2
    'grahamNumberToPrice':        0.033,    # legacy 1
    'bVpRatio':                   0.033,    # legacy 0.25
    'revenueGrowth':              0.027,    # legacy 1
    # incomeQuality: the KEY and the WEIGHT are unchanged, but the
    # QUANTITY changed on 2026-08-01 (audit D2, CEO-approved).  It is no
    # longer FMP's CFO/NI ratio -- which inverts for loss-makers and
    # explodes as NI->0 -- but (CFO - netIncome)/totalAssets, the
    # sign-safe scale-free form (stage2_metrics.income_quality_accruals).
    # Stage-1 got this treatment in July (CFOlessEarnings in
    # createDicts.getDicts BoMetric_special_dict); Stage-2 was missed until now.
    # WEIGHT PROVENANCE, recorded rather than quietly fixed: 0.072 was
    # FITTED AGAINST THE RATIO, so it is a weight INHERITED by a
    # different quantity.  Re-fitting is a separate exercise and is NOT
    # authorised; this is a known, accepted consequence of correcting the
    # metric in isolation.  Sign VERIFIED to still be high-is-good:
    # spearman(new, old) = +0.32 on the healthy NI>0/CFO>0 pool, so
    # +0.072 keeps its meaning where the old metric was not inverted.
    'incomeQuality':              0.072,    # legacy 1
    'returnOnEquity':             0.030,    # legacy 1
    'returnOnCapitalEmployed':    0.060,    # legacy 1
    'currentRatio':               0.038,    # legacy 0.35
    'grossProfitMargin':          0.100,    # legacy 0.75
    'freeCashFlowYield':          0.0605,   # legacy 2
    'freeCashFlowPerShareGrowth': 0.043,    # legacy 1.5
    'DcfToPrice':                 0.000,    # legacy 0.35 -- DROPPED (BoDCF broken / no PIT DCF)
    'marketCapRevQuants':         0.080,    # legacy 0.25
    'Altman-Z':                   0.062,    # legacy 0.5
    'Piotroski':                  0.072,    # legacy 0.75
    'tbVpRatio':                  0.033,    # legacy 0.5
    'BoScore':                    0.000,    # legacy 0.1 -- DROPPED
    'EPStoEPSmean':               0.056,    # legacy 0.5
    'priceGrowth':                0.000,    # legacy 0.5 -- DROPPED (sign/seasonality bug)
    'CycleHeat':                 -0.080,    # legacy -0.5 -- NEGATIVE: penalizes hot late-cycle stocks
}

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
# NOT a delta from DEPLOYED: all 21 values differ, so a delta table would be a second
# full vector wearing a disguise.  It is a named vector on the CANONICAL key set instead
# -- which is what actually prevents silent drift (a metric added to DEPLOYED and
# forgotten here is what would break, and _validate() catches exactly that).
#
# DcfToPrice = 0.35 here vs 0.000 deployed is a REAL, DELIBERATE difference, not a bug;
# `test_scoring_weights_single_source` pins it so nobody "tidies" it away.
# Integer literals are preserved as ints (2, 1) because that is what the vector has
# always emitted.
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
}


# --- D. per-cohort vectors (CEO-approved, valuation-specialist proposal) -----
# Threaded into postBoScoreRanking(weight_override=...) per cohort by postBo; the
# general/main pool uses NO override (DEPLOYED).  weight 0 -> metric dropped from
# AggScore (constant/neutral in rankOfRanks); does not change cohort membership.
#
# Cohort LABELS are single-sourced in carveOut (REIT_SECTOR-derived 'REIT'/'Mining' and
# the FIN1_VEHICLE / FIN2_MANAGER / FIN3_BALSHEET constants).  They are spelled out as
# literals here only because carveOut imports THIS module, so importing it back would be
# a cycle; carveOut asserts at import that the two agree.
#
# --- priceGrowth / DcfToPrice ZEROED IN EVERY COHORT (domain review S5, 2026-07-26) ---
# Both were 0.000 in the GENERAL vector and NON-ZERO here -- priceGrowth 0.25 (Mining,
# FIN-1) / 0.5 (REIT, FIN-2, FIN-3), DcfToPrice 0.25-0.35 in every cohort -- so the
# stage2_metrics comments that justify leaving their known defects alone "because w=0.000"
# were false on all five cohort paths.  What they were carrying:
#   priceGrowth  the ONE Stage-2 metric with an acknowledged, UNCORRECTED semi-annual 2x
#                LEVEL bias (a 6-month price move scored against a quarterly 3-month one);
#                the window scaling cannot fix a level.
#   DcfToPrice   computed from a LIVE DCF call using the CURRENT market price, while every
#                other metric is as-of period end -- a basis mix inside the cohort score and
#                the one channel that can import lookahead, which is exactly why the PIT
#                reproduction drops it (stage2_pit.DROP_METRICS).
# Zeroed rather than corrected: correcting priceGrowth needs a metric-definition decision
# and DcfToPrice needs a point-in-time DCF that does not exist.
#
# KNOWN OPEN ISSUE, PRESERVED DELIBERATELY: `BoScore` carries 0.1 in all five cohorts
# while it is 0.000 in DEPLOYED (dropped there).  The CEO has NOT ruled on this, so it is
# kept EXACTLY as-is and pinned by a test -- do not "fix" it to 0 as part of any other
# change.
COHORT_WEIGHTS_RAW = {
    'Mining': {
        'earnYield': 0.5, 'RoA': 1.0, 'returnOnEquity': 0.5, 'returnOnCapitalEmployed': 0.5,
        'grahamNumberToPrice': 0.25, 'bVpRatio': 0.75, 'tbVpRatio': 1.0, 'freeCashFlowYield': 1.5,
        'freeCashFlowPerShareGrowth': 0.5, 'revenueGrowth': 0.5, 'incomeQuality': 1.25,
        'grossProfitMargin': 0.25, 'Altman-Z': 1.0, 'Piotroski': 0.75, 'currentRatio': 0.75,
        'DcfToPrice': 0, 'EPStoEPSmean': 1.0, 'CycleHeat': -1.5, 'priceGrowth': 0,
        'marketCapRevQuants': 0.25, 'BoScore': 0.1,
    },
    'REIT': {
        'earnYield': 0, 'RoA': 0.5, 'returnOnEquity': 0.5, 'returnOnCapitalEmployed': 0.25,
        'grahamNumberToPrice': 0, 'bVpRatio': 0.5, 'tbVpRatio': 0.5, 'freeCashFlowYield': 1.0,
        'freeCashFlowPerShareGrowth': 0.75, 'revenueGrowth': 1.0, 'incomeQuality': 1.25,
        'grossProfitMargin': 0, 'Altman-Z': 0, 'Piotroski': 0.25, 'currentRatio': 0,
        'DcfToPrice': 0, 'EPStoEPSmean': 0, 'CycleHeat': -0.25, 'priceGrowth': 0,
        'marketCapRevQuants': 0.25, 'BoScore': 0.1,
    },
    'InvestmentVehicle': {   # FIN-1 Investment Vehicles
        'earnYield': 0, 'RoA': 0, 'returnOnEquity': 0.25, 'returnOnCapitalEmployed': 0,
        'grahamNumberToPrice': 0, 'bVpRatio': 2.0, 'tbVpRatio': 1.0, 'freeCashFlowYield': 0,
        'freeCashFlowPerShareGrowth': 0, 'revenueGrowth': 0, 'incomeQuality': 0,
        'grossProfitMargin': 0, 'Altman-Z': 0, 'Piotroski': 0, 'currentRatio': 0,
        'DcfToPrice': 0, 'EPStoEPSmean': 0, 'CycleHeat': 0, 'priceGrowth': 0,
        'marketCapRevQuants': 0, 'BoScore': 0.1,
    },
    'FinManager': {   # FIN-2 Managers / Brokers / Platforms
        'earnYield': 1.5, 'RoA': 0.5, 'returnOnEquity': 2.0, 'returnOnCapitalEmployed': 1.0,
        'grahamNumberToPrice': 0.25, 'bVpRatio': 0.25, 'tbVpRatio': 0.25, 'freeCashFlowYield': 2.0,
        'freeCashFlowPerShareGrowth': 1.5, 'revenueGrowth': 1.5, 'incomeQuality': 1.0,
        'grossProfitMargin': 0, 'Altman-Z': 0.25, 'Piotroski': 0.5, 'currentRatio': 0.25,
        'DcfToPrice': 0, 'EPStoEPSmean': 0.5, 'CycleHeat': -0.5, 'priceGrowth': 0,
        'marketCapRevQuants': 0.25, 'BoScore': 0.1,
    },
    'BalanceSheetFin': {  # FIN-3 Balance-Sheet Financials (banks / lenders / insurers)
        'earnYield': 1.0, 'RoA': 0.5, 'returnOnEquity': 2.0, 'returnOnCapitalEmployed': 0,
        'grahamNumberToPrice': 0.75, 'bVpRatio': 1.5, 'tbVpRatio': 1.0, 'freeCashFlowYield': 0,
        'freeCashFlowPerShareGrowth': 0, 'revenueGrowth': 0.75, 'incomeQuality': 0.25,
        'grossProfitMargin': 0, 'Altman-Z': 0, 'Piotroski': 0.25, 'currentRatio': 0,
        'DcfToPrice': 0, 'EPStoEPSmean': 1.0, 'CycleHeat': -1.0, 'priceGrowth': 0,
        'marketCapRevQuants': 0.25, 'BoScore': 0.1,
    },
}


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
# (The "~14.35" above is the sum as it stood WHEN S7 WAS WRITTEN, i.e. before S5 zeroed
# priceGrowth / DcfToPrice in every cohort.  Today's raw sums are Mining 13.85, REIT 7.10,
# InvestmentVehicle 3.35, FinManager 14.10, BalanceSheetFin 10.35 -- pinned as a checksum
# in test_scoring_weights_single_source.  Kept verbatim rather than edited so the ruling
# still reads as it was made.)
COHORT_WEIGHTS = {label: normalise(vec) for label, vec in COHORT_WEIGHTS_RAW.items()}


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
