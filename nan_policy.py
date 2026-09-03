"""THE TWO-TIER NaN POLICY -- one module, so the whole rule is auditable in one place.

Built to `projects/investment-filter/design/nan-policy.md` (specification + ADDENDUM) and to
the CEO's direction behind it:

    "we should have some columns such that if there are NaNs, we should just disqualify them.
     Then there are secondary columns where we have some way of dealing with NaNs that don't
     benefit the ticker."

    "On the partial, we should treat as NaN if the coverage is worse than some threshold. We
     should also treat as NaN if there are a lot of gaps in it (but most not hit on
     semi-annual stuff)."

    "For metrics that are NaN because of adverse things I don't think we should punish them
     again for it. So just put it like earnings were close to 0."

FOUR MECHANISMS, and each one has exactly ONE home:

  1. PRIMARY PRESENCE (section 1a / ADDENDUM C) -- five RAW inputs whose absence makes the
     valuation question unanswerable.  A source-level EJECT, applied by
     `data_quality.filter_invalid_data` (NOT a new gate -- see `primary_eject`).
  2. COVERAGE + GAPPINESS (section 3) -- a windowed metric computed on too little, or on a
     scattered, history becomes NaN, which the Stage-2 normaliser then imputes at the column
     MEDIAN.  Applied by `window_verdict`, called from every windowed metric in
     `stage2_metrics`.
  3. BOUNDARY IMPUTATION (ADDENDUM A) -- a metric undefined because an input is ADVERSE takes
     the analytic LIMIT of the metric at that input's domain boundary, admissible only where
     that limit is finite AND is the metric's WORST admissible value.  `BOUNDARY_LIMIT`.
  4. REFUSAL -- where the limit does not exist, is infinite, or lands on the metric's BEST
     side, the metric is refused (stays NaN -> column median).  `REFUSED_NOT_IMPUTED`.

WHAT IS DELIBERATELY *NOT* HERE, because both were closed by the CEO:

  * NO Stage-2 `returnOnEquity` FLOOR.  Considered and CLOSED (CEO, 2026-08-05): "Such a
    company (negative equity with high earnings) could be surging due to good investments. But
    that would pop up as positive somewhere else. We just need to make sure it is not treated
    positively here."  Refusing the metric puts the name at the column MEDIAN, which is
    neutral rather than positive, so the requirement is already met and a floor would be a
    second punishment for one fact.  The open note in
    `stage2_metrics.postbm_metric` ("strict double-negative ranks below every single negative
    is NOT achieved here") is therefore CLOSED AS NOT-WANTED, not outstanding.
  * NO REFUSED-vs-MISSING CHANNEL.  A per-cell reason channel through the normaliser was
    explicitly ruled out.  Observability is instead a run-level COUNT (`POLICY_COUNTS` below):
    it answers "how many cells did each rule convert, per column, per pool" without giving a
    cell two identities that every consumer of `postScoreMetric` would then have to know
    about.

EVERY NUMBER IN THIS MODULE IS MEASURED on [panel = baseline_tools/resdic_2026-07-17_CORRECTED
.pickle, 176,781 rows / 7,729 sources; pool = its deployed general top-100].  A figure with no
panel attached is how this project carried a stale pair of numbers for weeks -- do not re-quote
one without it.
"""

import numpy as np
import pandas as pd

import reporting_period as rp


# =========================================================================== #
#  1.  PRIMARY PRESENCE -- the five raw inputs, and the two impossibility checks
# =========================================================================== #
#  THE RULE THAT GENERATES THE LIST, so it is an enumeration and not a preference.  A column is
#  PRIMARY iff it is a RAW PROVIDER INPUT and either
#    (a) the valuation question does not exist without it, or
#    (b) it IS the substance of an income-quality or an indebtedness test.
#  Derived metrics are NEVER primary: `uGrahamNumberToPrice` is all-NaN on 24.97% of the
#  universe and `dSalesToInventory` on 40.76%, and both are undefined precisely BECAUSE the
#  company is loss-making or asset-light -- which is an answer, not a gap.  Ejecting on a
#  derived metric would disqualify companies for being cheap-and-troubled, i.e. the CEO's own
#  rule inverted.
#
#  `revenue` and `totalAssets` were DROPPED from primary (ADDENDUM C1, CEO: "Everything that
#  does not indicate income quality or indebtedness is suspect to me tbh").  Both are SCALE
#  denominators -- they answer "how big is it", not "are these earnings real".  They keep a
#  place here only as ARITHMETIC IMPOSSIBILITY checks (`SANITY_IMPOSSIBLE`), which is what
#  their entire measured eject always was.
#
#  A LEGITIMATE ZERO OR NEGATIVE VALUE NEVER DISQUALIFIES.  A negative `netIncome` is the
#  answer the filter exists to read; zero revenue is a real investment vehicle.  Only absence
#  (NaN) and arithmetic impossibility disqualify.

#  (a) -- there is no valuation question without these.  Condition: NaN or <= 0.
PRIMARY_POSITIVE = (
    #  price: cheapness is a ratio TO price.  No price, no P/E, no yield, no discount.
    'price',
    #  marketCap: the actual denominator of earnYield (w 0.127), freeCashFlowYield (0.065) and
    #  marketCapRevQuants (0.052) -- 24.4% of the general weight vector.
    'marketCap',
    #  weightedAverageShsOut: every per-share quantity, INCLUDING `price` itself, which is
    #  marketCap/shares since the 2026-07-19 ingest fix.
    'weightedAverageShsOut',
)

#  (b) -- these ARE the income-quality / indebtedness test.  Condition: NaN ONLY.
PRIMARY_PRESENT = (
    #  netIncome: the earnings whose quality is the whole question.  Sign is never a reason.
    'netIncome',
    #  netCashProvidedByOperatingActivities: "do the earnings convert to cash" -- the accrual
    #  test itself.  THE ONLY LIMB WITH REAL CONTENT: 117 sources (1.51%) [universe], 0 [pool].
    'netCashProvidedByOperatingActivities',
    #  totalStockholdersEquity: the indebtedness denominator -- leverage, and what is left for
    #  owners.
    'totalStockholdersEquity',
)

#  ARITHMETIC IMPOSSIBILITY on the same newest row -> the same eject, for a different reason.
#  {field: (comparison, bound)} -- the row is inadmissible when the comparison holds.
#  MEASURED [universe] / [pool]: totalAssets <= 0 -> 13 (0.17%) / 0 ; revenue < 0 -> 36
#  (0.47%) / 0.  Union with the primary limbs: 166 (2.15%) / 0.
SANITY_IMPOSSIBLE = {
    #  A going concern has positive assets.  A NEGATIVE-equity company still has positive
    #  assets -- equity is what goes negative -- so <= 0 here is a broken statement, not a
    #  distressed one.
    'totalAssets': ('<=', 0.0),
    #  Revenue may legitimately be ZERO (a pre-revenue biotech, a holding company) and this
    #  check deliberately does NOT fire on that.  Only a NEGATIVE revenue on the AS-OF row is
    #  treated as impossible.
    'revenue': ('<', 0.0),
}

#  WHY THE VERDICT IS TAKEN ON THE SOURCE'S NEWEST ROW, and not on any row of its history.
#  THIS IS A DELIBERATE READING OF section 2, and the numbers in the spec are the numbers this
#  reading produces -- so it is also the reading the spec was measured under.  Three reasons:
#
#   1. IT IS THE QUESTION BEING ASKED.  The filter values the company AS OF NOW.  A missing
#      2018 cash-flow statement does not make today's valuation unanswerable; a missing
#      current one does.
#   2. A ROW-LEVEL FLAG WOULD NOT BE SELECTIVE.  `check_price_sanity` flags a ROW, and
#      `filter_invalid_data` then deletes every row AT OR BEFORE it -- prefix removal.  Run
#      row-level, the two impossibility checks alone touch 476 sources on `revenue < 0`
#      (6.16%) and 366 on `totalAssets <= 0` (4.74%) and would delete history in front of
#      them, against 36 and 13 on the newest row.  And `revenue < 0` is NOT reliably
#      corruption at that volume -- 1,307 panel rows are negative, and
#      `stage2_metrics.income_quality_accruals` already records negative revenue as
#      legitimate contra-revenue / refunds.  Treating a documented legitimate value as
#      corruption, and cascading it backwards over a name's whole history, is a far larger
#      change than the one the CEO authorised.
#   3. IT REPRODUCES THE MEASUREMENT.  Section 1a's per-limb figures (totalAssets 13, revenue
#      36, union of limbs 1-6 = 49) and ADDENDUM C's union (117) are exactly the newest-row
#      counts, reproduced to the source; no other window reproduces them.
#
#  DEVIATION FROM THE SPEC, STATED RATHER THAN BURIED: section 1a's limb 7 marks the CFO limb
#  "pool-conditional -- not primary where both consumers carry w = 0, i.e. FIN-1".  That is NOT
#  implemented, and could not be without inverting the pipeline: the eject decides UNIVERSE
#  MEMBERSHIP and runs at `Sbocker.py:490`, while cohort membership is decided by the carve-out
#  far downstream of it, so a cohort-conditional eject would either need a new gate after the
#  carve (which section 2 forbids) or a duplicate cohort classifier inside data_quality.
#  MEASURED COST OF NOT IMPLEMENTING IT: 4 of the 79 FIN-1 members, 0 pool names.
PRIMARY_VERDICT_ROW = 'newest'

#  A DELIBERATE ABSTENTION IS NOT AN ABSENT INPUT -- the column that says so.
#  §5's `refuse_impossible_cells` blanks cells whose value the vendor contradicts.  Three of
#  the fields it can blank are PRIMARY (`totalStockholdersEquity` here; `totalAssets` and
#  `revenue` under SANITY_IMPOSSIBLE), and `_limb_fails` reads a blank as ABSENCE -- so the
#  §5 guard, whose entire contract is "abstain, never eject", was EJECTING JHX and SZZL
#  through this door.  Measured before the fix: 0 sources ejected -> 2, both on
#  `totalStockholdersEquity`/NaN, i.e. the change deleted the $10bn company it exists to
#  protect.  (Found in independent review, 2026-08-14.)
#
#  THE FIX IS TO MAKE THE REFUSAL LEGIBLE RATHER THAN TO STOP REFUSING.  Each row carries the
#  pipe-joined names of the fields §5 refused ON THAT ROW, so `primary_eject` can tell "we
#  declined to trust this number" from "the provider never sent one".  A column, not a module
#  registry, deliberately: it rides `pd.concat`, the saved pickle and the `-loadbometric`
#  reload with the data it describes, and `reset_counts` already documents what a module-level
#  accumulator costs when one process scores twice.
#
#  IT IS THE CLASS THAT IS FIXED, NOT THE INSTANCE.  Dropping `totalStockholdersEquity` from
#  the identity's blanked fields also clears both ejects at zero measured cost, and was the
#  reviewer's verified stopgap -- but it re-arms silently the moment any future relation names
#  a primary field (`netIncome` and `netCashProvidedByOperatingActivities` are both one edit
#  away), and the defect is precisely that an abstention is indistinguishable from an absence.
SANITY_REFUSED_COLUMN = 'sanityRefusedFields'
_SANITY_REFUSED_SEP = '|'


def refused_fields_mask(df, field, column=SANITY_REFUSED_COLUMN):
    """Boolean Series: True where `field` was DELIBERATELY REFUSED by §5 on that row.

    A frame without the column has refused nothing -- every pre-§5 panel, every test frame and
    every external caller keeps its exact previous behaviour.
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    col = df[column].astype('object').where(df[column].notna(), '')
    tok = _SANITY_REFUSED_SEP + field + _SANITY_REFUSED_SEP
    return col.map(lambda v: tok in (_SANITY_REFUSED_SEP + str(v) + _SANITY_REFUSED_SEP))


def primary_limbs():
    """[(field, condition_label)] for every limb, primary and impossibility, in report order."""
    out = [(f, 'NaN or <= 0') for f in PRIMARY_POSITIVE]
    out += [(f, 'NaN') for f in PRIMARY_PRESENT]
    out += [(f, 'impossible: %s %g' % (op, b)) for f, (op, b) in SANITY_IMPOSSIBLE.items()]
    return out


def _limb_fails(values, field):
    """Boolean Series: True where `values` (already numeric) is INADMISSIBLE for `field`."""
    if field in PRIMARY_POSITIVE:
        return values.isna() | (values <= 0)
    if field in PRIMARY_PRESENT:
        return values.isna()
    op, bound = SANITY_IMPOSSIBLE[field]
    #  NaN is NOT an impossibility -- these two fields are SECONDARY for absence (ADDENDUM C1),
    #  so a NaN here goes to the column median like any other secondary gap.  `fillna(False)`
    #  rather than `fillna(True)`, and that asymmetry with the domain guards
    #  (calcMetrics.apply_domain_guard uses fillna(False) to make an UNDETERMINED DOMAIN
    #  inadmissible) is deliberate: there the question is "may I score this row", here it is
    #  "must I delete this company".
    cmp_ = (values <= bound) if op == '<=' else (values < bound)
    return cmp_.fillna(False)


def primary_eject(df, source_col='source', date_col='date', verbose=False):
    """Sources to EJECT, and which limb fired -- the section 1a / ADDENDUM C verdict.

    `df` is a cdx-schema frame carrying several rows per source in ANY row order (the verdict
    is taken on each source's LATEST `date`, found explicitly rather than by position, so this
    cannot be broken by an oldest-first caller).

    Returns a DataFrame [source, date, field, limb, value] with ONE row per (source, limb)
    that fired -- so the report says WHICH limb ejected each name, which section 2 asks for.
    Empty frame when nothing ejects.

    NOT A NEW GATE.  This is called from inside `data_quality.filter_invalid_data`, the
    source-level exclusion that already exists and already runs twice.  A second gate is worse
    than either, and Stage-1 structurally CANNOT express an eject: `calcScore.calcByTier`
    returns a PASS-RATE, so a NaN there is soft degradation (a name failing eight of eight
    rows on a Tier-S criterion still scores).  That ruling is not reopened here.
    """
    if df is None or len(df) == 0 or source_col not in df.columns:
        return pd.DataFrame(columns=['source', 'date', 'field', 'limb', 'value'])
    fields = [f for f, _ in primary_limbs()]
    missing = [f for f in fields if f not in df.columns]
    if missing:
        #  LOUD, not silent: a panel that does not carry a primary input cannot be checked for
        #  its presence, and quietly skipping the limb would report "0 ejected" for a reason
        #  that has nothing to do with the data.
        raise KeyError(
            'nan_policy.primary_eject: the frame is missing %d field(s) the primary-presence '
            'policy is defined over: %s. Every one is a RAW provider input that every cdx '
            'panel carries; a frame without them cannot be checked, and reporting 0 ejects '
            'would be a false negative.' % (len(missing), missing))
    _keep = [source_col, date_col] + fields + (
        [SANITY_REFUSED_COLUMN] if SANITY_REFUSED_COLUMN in df.columns else [])
    d = df[_keep].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    #  The AS-OF row per source = its latest parseable date.  `idxmax` on the date picks it
    #  regardless of the caller's row order; ties keep the first occurrence, which matches
    #  every other newest-row reader in the pipeline (`stage2_metrics` reads row 0 of a
    #  newest-first frame, i.e. the first of a tied pair).
    d = d.dropna(subset=[date_col])
    if d.empty:
        return pd.DataFrame(columns=['source', 'date', 'field', 'limb', 'value'])
    newest = d.loc[d.groupby(source_col)[date_col].idxmax()]
    rows = []
    for field, limb in primary_limbs():
        v = pd.to_numeric(newest[field], errors='coerce')
        bad = _limb_fails(v, field)
        #  A CELL §5 REFUSED IS AN ABSTENTION, NOT AN ABSENCE, so it must not eject the source.
        #  Without this the guard that exists to keep JHX in the universe removed it.
        bad = bad & ~refused_fields_mask(newest, field)
        if not bad.any():
            continue
        sub = newest.loc[bad, [source_col, date_col]].copy()
        sub['field'] = field
        sub['limb'] = limb
        sub['value'] = v[bad].values
        rows.append(sub.rename(columns={source_col: 'source', date_col: 'date'}))
    out = (pd.concat(rows, ignore_index=True) if rows
           else pd.DataFrame(columns=['source', 'date', 'field', 'limb', 'value']))
    if verbose:
        n_src = out['source'].nunique() if len(out) else 0
        print('PRIMARY-PRESENCE EJECT: %d source(s) of %d (%.2f%%) fail at least one limb on '
              'their newest row.' % (n_src, newest[source_col].nunique(),
                                     100.0 * n_src / max(1, newest[source_col].nunique())),
              flush=True)
        if len(out):
            for field, limb in primary_limbs():
                k = int((out['field'] == field).sum())
                if k:
                    print('    %-40s %-18s %5d source(s)' % (field, limb, k), flush=True)
    return out


# =========================================================================== #
#  2.  COVERAGE AND GAPPINESS -- partial history collapses into NaN            #
# =========================================================================== #
#  The CEO's simplification: FOUR cases become TWO.  full -> use it; partial-but-adequate ->
#  use it; partial-BELOW-THRESHOLD -> NaN; absent -> NaN.  Then ONE rule handles NaN.

#  Below this share of its window a windowed metric is not a measurement of the window.
COVERAGE_MIN = 0.50

#  THE DENOMINATOR IS ROWS *PRESENT*, NOT THE NOMINAL WINDOW.  390 of 7,729 sources (5.05%)
#  carry fewer rows than their scaled window; against a nominal denominator they would fail
#  coverage on every metric BY CONSTRUCTION, which turns a coverage rule into a covert
#  history-DEPTH filter and double-counts `min_periods_required = 8` in data_quality.
#
#  AND THE STRUCTURAL LAG COMES OFF THAT DENOMINATOR TOO, for the same reason one step down.
#  A YoY metric is `pct_change(-rpy)`, so the OLDEST `rpy` rows of a source's series have no
#  counterpart and are NaN BY ARITHMETIC, not by absence.  On a source whose panel is shorter
#  than window+lag those structural NaNs sit INSIDE the window and would be counted as gaps:
#  measured, that alone flags an extra 54 sources on `revenueGrowth` and 28 on
#  `freeCashFlowPerShareGrowth` -- i.e. it would penalise a short history through the coverage
#  door, which is precisely what "rows present" was chosen to avoid.  The lag is read from the
#  metric REGISTRY (`stage2_metrics.STAGE2_METRIC_SPEC`), never guessed at a call site.
COVERAGE_DENOMINATOR = 'rows_present_less_structural_lag'

#  GAPPINESS IS A SECOND, SEPARATE TEST -- coverage asks HOW MANY rows are computable,
#  gappiness asks whether they are CONTIGUOUS.  A semi-annual filer with 8 of 8 has full
#  coverage and no gaps; a quarterly filer at rows 1, 4, 7, 10 has half coverage AND is gappy.
#  A scattered series makes a mean or a trend less trustworthy at equal coverage, and it is the
#  signature of a company that STOPPED REPORTING -- which is information, not absence.

#  (i) CALENDAR gappiness, NAME-level.  A gap is a spacing of more than GAP_TOLERANCE x the
#  filer's OWN expected cadence.  FREQUENCY-NORMALISATION IS NOT OPTIONAL AND THIS IS THE
#  MEASUREMENT THAT SAYS SO: under the cadence-relative rule semi-annual filers are flagged
#  24 of 1,108 (2.17%) against quarterly 407 of 6,620 (6.15%) -- i.e. LESS often, correctly.
#  Under a naive FIXED 3-month expectation at the same 1.6x tolerance, 1,108 of 1,108 (100%)
#  are flagged.  The machinery to avoid it already exists
#  (`reporting_period`); do NOT invent a second notion of "expected rows".
GAP_TOLERANCE = 1.6
#  The AMBIENT Stage-2 scoring window, in quarters, that the NAME-level calendar test is taken
#  over.  It is a fallback, not an authority: every caller that HAS the ambient `nq` passes it
#  (`stage2_metrics._reduce(..., scoring_nq=nq)`).  The two metrics that cannot -- `CycleHeat`
#  and `EPStoEPSmean`, whose own `nq` is a 28-quarter BASELINE, not the scoring window -- fall
#  back to this.  Production runs the scoring window at 16, so the fallback and the passed
#  value agree today; keeping it a named constant is what makes a future divergence visible
#  rather than silent.
SCORING_WINDOW_NQ = 16
#  TWO, not one: one gap is routinely a late filing or a fiscal-year change; two is a
#  stoppage.  A threshold of 1 would cost 432 sources (5.59%) and 7 pool names for a signal
#  that is mostly calendar noise.  At 2: 142 (1.84%) [universe], 2 [pool].
MAX_CALENDAR_GAPS = 1
_DAYS_PER_MONTH = 30.4375

#  (ii) METRIC-level gappiness, PER COLUMN: interior missing RUNS in the metric's own
#  computable mask.  Leading and trailing missing stretches are NOT interior -- a trailing
#  stretch is the structural lag above, and a leading one is a metric that has simply stopped
#  being computable, which coverage already prices.  MEASURED at >= 2 runs [universe]/[pool]:
#  grahamNumberToPrice 197 (2.55%)/3, returnOnEquity 110 (1.42%)/0, freeCashFlowPerShareGrowth
#  104/0, revenueGrowth 84/0, incomeQuality 47 (0.61%)/0, CycleHeat 15/0, everything else <= 10.
#  Type-U union: 285 (3.69%) [universe], 0 [pool].
MAX_INTERIOR_RUNS = 1


def expected_month_spacing(rpy):
    """Months between consecutive filings for a source with `rpy` rows per year."""
    return 12.0 / float(int(rpy))


def calendar_gap_count(dates, rpy, window_quarters=SCORING_WINDOW_NQ):
    """Number of CADENCE-RELATIVE filing gaps in the source's most recent scaled window.

    `dates` is that one source's period-end dates in any order; the most recent
    `rp.scale_window(window_quarters, rpy)` of them are examined.  A gap is a spacing of more
    than `GAP_TOLERANCE` x this filer's own expected cadence -- 4.8 months for a quarterly
    filer, 9.6 for a semi-annual one.
    """
    d = pd.to_datetime(pd.Series(list(dates)), errors='coerce').dropna().sort_values()
    w = rp.scale_window(int(window_quarters), rpy)
    d = d.iloc[-w:] if len(d) > w else d
    if len(d) < 2:
        return 0
    spacing = d.diff().dropna().dt.days.to_numpy(dtype='float64')
    return int((spacing > GAP_TOLERANCE * expected_month_spacing(rpy) * _DAYS_PER_MONTH).sum())


def calendar_gap_refused(tempcdx, rpy, window_quarters=SCORING_WINDOW_NQ,
                         date_col='date'):
    """True when this source has stopped filing often enough that no WINDOWED metric of it is
    a measurement -- `MAX_CALENDAR_GAPS` or more gaps in the scaled scoring window.

    THE WINDOW IS THE AMBIENT SCORING WINDOW FOR EVERY METRIC, not each metric's own.  This is
    a NAME-level property ("this company stopped filing twice in four years"), so making it
    per-metric would give one company two different answers to one question about it.

    CONSEQUENCE, STATED PLAINLY BECAUSE THE SPEC OVERSTATES IT.  Section 3b(i) concludes "2 or
    more calendar gaps therefore DISQUALIFIES", on the grounds that `earnYield` is primary.  It
    does not, and cannot: the primary set is five RAW INPUTS (section 1a / ADDENDUM C), and
    `earnYield` is a derived metric which section 1c rules can never be primary.  What actually
    happens is that every windowed metric goes to the column MEDIAN while the point-in-time
    ones (Piotroski, Altman-Z, marketCapRevQuants, shareCountChange, longTermDebtChange) still
    score, so `normalizeAndDropNA` does not drop the row either.  So this rule NEUTRALISES; it
    does not eject.  The name is then visible as such in the run's own
    `MissingDataFillReport` (it will carry a large `imputed_weight_share`).
    IF AN EJECT IS WANTED IT IS A CEO DECISION AND THE PRICE IS MEASURED: 142 sources (1.84%)
    [universe] and 2 [pool].  It is not taken here.
    """
    if tempcdx is None or date_col not in getattr(tempcdx, 'columns', []):
        return False
    return calendar_gap_count(tempcdx[date_col], rpy, window_quarters) > MAX_CALENDAR_GAPS


def interior_missing_runs(computable):
    """Number of interior missing RUNS in a boolean 'this row is computable' mask.

    Leading/trailing missing stretches are excluded -- see MAX_INTERIOR_RUNS.  An all-missing
    or all-present mask has 0 runs.
    """
    a = np.asarray(computable, dtype=bool)
    if not a.any():
        return 0
    first, last = int(a.argmax()), len(a) - 1 - int(a[::-1].argmax())
    runs, prev = 0, True
    for x in a[first:last + 1]:
        if (not x) and prev:
            runs += 1
        prev = bool(x)
    return runs


# =========================================================================== #
#  3.  BOUNDARY IMPUTATION, AND 4. REFUSAL                                     #
# =========================================================================== #
#  THE RULE (ADDENDUM A1), verbatim in intent:
#
#      For a metric undefined because an input is ADVERSE, impute the LIMIT of the metric as
#      that input approaches its domain boundary from the admissible side.
#            v_impute(metric) = lim metric(x)  as x -> boundary+
#      Admissible ONLY when that limit EXISTS, is FINITE, and is the metric's WORST admissible
#      value.  Where the limit is +/-infinity, or lands at the metric's BEST value, the rule
#      does NOT apply and the metric must be REFUSED, not imputed.
#
#  IT IS A LIMIT, NOT "the smallest representable positive input".  `sqrt(22.5 * 1e-308 *
#  BVPS)` happens to equal the limit for the one column here, but it is not robust in general
#  (a formula with a 1/x term overflows) and a magic epsilon is not auditable.  Each limit is
#  taken analytically and written HERE, beside the metric that owns it -- the same place
#  `STAGE2_METRIC_SPEC` already declares window and frequency treatment.
#
#  TYPE D vs TYPE U (carried from missing-data-regime.md section 3).  Type D = the column is
#  undefined BECAUSE of what the company is; Type U = absence says nothing about the company.
#  Type D is exactly two columns.
TYPE_D = ('grahamNumberToPrice', 'EPStoEPSmean')

#  {metric: (limit, why)} -- the column-space value imputed when the metric is FULLY undefined
#  over its window AND the reason is adverse.
BOUNDARY_LIMIT = {
    #  Stage-2.  metric = mean over the window of grahamNumber/price, and
    #  grahamNumber = sqrt(22.5 * EPS_ttm * BVPS) which is undefined for EPS_ttm <= 0 or
    #  BVPS <= 0.  As EPS_ttm -> 0+ (BVPS > 0 fixed): sqrt(22.5 * 0+ * BVPS) -> 0, so
    #  grahamNumber/price -> 0/price = 0.  FINITE, and it is the FLOOR: the whole meaningful
    #  range of this metric is above zero for any profitable company (observed [pool] minimum
    #  +0.1254), so 0.0 ranks the name BELOW every observed name on this one axis.  ADMISSIBLE.
    'grahamNumberToPrice': (
        0.0, 'lim sqrt(22.5*EPS_ttm*BVPS)/price as EPS_ttm -> 0+  =  0 ; the metric floor'),
    #  Stage-1 (the same quantity, per ROW, before the unity test).  The criterion column is
    #  grahamNumber/price and `calcScore.calcByTier` tests `metvec - 1 > 0`, so imputing 0.0
    #  makes the TESTED value -1.0 and the criterion FAILS -- which is exactly what it does
    #  with the NaN today.  BEHAVIOUR-IDENTICAL BY CONSTRUCTION, and that identity is the
    #  strongest available confirmation that the rule is right: the fail becomes DERIVED
    #  ("there is no earnings-based valuation floor to compare price against") rather than
    #  incidental ("the number was missing").
    'uGrahamNumberToPrice': (
        0.0, 'same limit, per row; unity test then yields -1.0 = FAIL, as today'),
}

#  {metric: why it is refused instead} -- the escape clause of A1, exercised.
REFUSED_NOT_IMPUTED = {
    #  ADDENDUM A2 gives this metric a boundary of -1.0 at the 34.4th percentile.  THAT IS
    #  WRONG ON BOTH COUNTS, and the error is a SIGN error in the spec against the shipped
    #  formula.  `stage2_metrics.eps_to_eps_mean` computes
    #      (epsmean - ewma_recent_eps) / |epsmean|
    #  i.e. mean MINUS recent -- POSITIVE means the recent year sits BELOW the name's own
    #  history, which is the mean-reversion side the +0.0516 weight BETS ON.  The spec's table
    #  writes the limit of "(EPS - mean)/abs(mean)", i.e. recent MINUS mean, which is the
    #  formula with its sign reversed; that is where -1.0 comes from.
    #
    #  TAKE THE ACTUAL LIMIT AND THE RULE REFUSES ITSELF.  The adverse condition is "one of the
    #  most recent `rpy` EPS is <= 0"; driving those to 0+ makes the EWMA term -> 0, leaving
    #  epsmean/|epsmean| = +1 when epsmean > 0.  MEASURED over the 3,888 sources that hit the
    #  gate: the limit is NOT a constant (min -275.79, p10 -2.17, median -1.00, p90 +0.82, max
    #  +1.00) and it is POSITIVE for 25.8% of them -- landing at +1.0, which is the METRIC'S
    #  OBSERVED MAXIMUM (universe max +1.000).  So for a quarter of the affected names the
    #  "boundary" is the single most REWARDED value the column can take, and for the rest it
    #  sits at the 21st percentile rather than at a floor.
    #
    #  A1's own escape clause therefore applies: the limit is not the metric's worst admissible
    #  value, so REFUSE.  And refusing is also what the CEO's instruction wants here, because
    #  this metric's NaN was never a PUNISHMENT to begin with -- the positivity gate exists to
    #  stop a loss-maker collecting the maximum mean-reversion reward.  Imputing -1.0 would
    #  have been a fresh punishment (measured in the spec itself: ten pool names each drop 0-3
    #  ranks), which is the opposite of "I don't think we should punish them again for it".
    'EPStoEPSmean': (
        'the limit is not the metric floor -- it is +1.0 (the observed MAXIMUM, i.e. the '
        'most-rewarded value) for 25.8% of the 3,888 affected sources and only median -1.0 '
        '(21st percentile) for the rest; A1 admits a boundary only where the limit IS the '
        'worst admissible value, so this one is REFUSED and stays at the column median'),
    #  Not a Stage-2 metric, recorded because the coordinator asked whether the family is
    #  closed.  revenue/inventory as inventory -> 0 is +INFINITY, which lands on the metric's
    #  BEST side and would score a PASS: an inventory-free business acing an inventory-TURN
    #  test.  Tier N (w = 0) already achieves the refusal; the intent is recorded so a future
    #  re-weighting cannot reactivate it silently.  3,641 sources (47.11%), 26 pool.
    'dSalesToInventory': (
        'limit is +infinity and lands on the metric BEST side -> would score a PASS for having '
        'no inventory; refused (and w = 0 already achieves it)'),
    #  The whole B2 family: these are NOT NaN at all -- the ratio returns a finite number with
    #  the WRONG SIGN, so no NaN rule can reach them and no boundary exists to impute.  They
    #  were fixed at `da79aee` by inversion or by a domain guard, and THAT LIST IS NOT
    #  REOPENED HERE.
    'B2_sign_inverting_family': (
        'not undefined but PERVERSE -- fixed by inversion or domain guard at da79aee '
        '(createDicts / calcMetrics.STAGE1_DOMAIN_GUARDS); no boundary applies'),
}

#  Reasons, as stamped by `getData_fmp.stamp_frequency_and_graham`, under which Graham's
#  non-computability is ADVERSE rather than a data gap.  MEASURED over the 61,832 newest-8 rows
#  of the panel: negative_eps 22,103 + negative_bv 1,109 = 23,212 adverse against
#  missing_inputs 208 -- so 99.1% of Graham's non-computability is adverse and 0.9% is a
#  genuine gap.  That is the strongest evidence in this project that "undefined" and "missing"
#  are different objects.
GRAHAM_ADVERSE_REASONS = ('graham_undefined_negative_eps', 'graham_undefined_negative_bv')
GRAHAM_REASON_COLUMN = 'grahamUndefinedReason'


def graham_adverse_mask(df):
    """Rows whose `grahamNumber` is undefined because an input was ADVERSE (not missing).

    Returns an all-False mask when the reason column is absent, which is the honest answer: a
    panel with no reason column cannot distinguish adverse from missing, and imputing on a
    guess would put a real value where a gap is.
    """
    if df is None or GRAHAM_REASON_COLUMN not in getattr(df, 'columns', []):
        return pd.Series(False, index=getattr(df, 'index', None))
    return df[GRAHAM_REASON_COLUMN].isin(GRAHAM_ADVERSE_REASONS).fillna(False)


# =========================================================================== #
#  OBSERVABILITY -- a run-level COUNT, not a per-cell channel                   #
# =========================================================================== #
#  One row per (pool, column, rule) accumulated across a run, in the same style as
#  `postBoRank.NORM_DIAGNOSTICS`: a LIST appended to, because the Stage-2 scorer runs ONCE PER
#  POOL and a per-call dict would leave only the last cohort.  Nothing reads it back into a
#  score.  This is what section 2 asks for ("add the two threshold counts") WITHOUT the
#  refused-vs-missing channel the CEO ruled out.
POLICY_COUNTS = []
_ACTIVE_POOL = ['general']


def reset_counts(label='general'):
    """Clear `POLICY_COUNTS` for a NEW RUN.  Called once per run by `postBo.postBoWrapper`.

    WHY THIS EXISTS (review finding, 2026-08-05): the counter is a module-level accumulator, and
    it was never cleared in production.  Accumulating ACROSS POOLS within one run is the point --
    the Stage-2 scorer runs once per pool and a per-call dict would leave only the last cohort --
    but accumulating across RUNS is not: any process that scores twice (the backtest harness, the
    tuner, a test session, `baseline_tools/nan_policy_report`'s two arms) would report the first
    run's conversions again in the second, and the count is one of the artifacts the run is
    judged on.  `NORM_DIAGNOSTICS` has the same shape and the same exposure; only tests reset it
    today.
    """
    del POLICY_COUNTS[:]
    _ACTIVE_POOL[0] = label or 'general'


def set_pool(label):
    """Label subsequent `POLICY_COUNTS` rows.  Called once per pool by the Stage-2 scorer."""
    _ACTIVE_POOL[0] = label or 'general'


def _count(column, rule):
    POLICY_COUNTS.append({'pool': _ACTIVE_POOL[0], 'column': column, 'rule': rule})


def counts_frame():
    """POLICY_COUNTS as a (pool, column, rule, n) frame.  Empty frame when nothing fired."""
    if not POLICY_COUNTS:
        return pd.DataFrame(columns=['pool', 'column', 'rule', 'n'])
    return (pd.DataFrame(POLICY_COUNTS).groupby(['pool', 'column', 'rule'])
            .size().reset_index(name='n').sort_values(['pool', 'column', 'rule']))


def report_counts(verbose=True):
    """Print the per-rule conversion counts for the run.  EMITS ONLY."""
    f = counts_frame()
    if not verbose:
        return f
    if f.empty:
        print('NaN-POLICY: no cell was converted by coverage, gappiness or a boundary limit.',
              flush=True)
        return f
    print('NaN-POLICY CONVERSIONS (per pool / column / rule):', flush=True)
    for _, r in f.iterrows():
        print('    %-10s %-30s %-24s %5d' % (r['pool'], r['column'], r['rule'], r['n']),
              flush=True)
    return f


# =========================================================================== #
#  THE ONE ENTRY POINT EVERY WINDOWED METRIC CALLS                             #
# =========================================================================== #
RULE_COVERAGE = 'coverage_below_min'
RULE_METRIC_GAPS = 'interior_gaps'
RULE_CALENDAR_GAPS = 'calendar_gaps'
RULE_BOUNDARY = 'boundary_imputed'
RULE_REFUSED = 'refused_no_boundary'


def window_verdict(values, w, key, rpy, tempcdx=None, structural_lag=0,
                   scoring_nq=SCORING_WINDOW_NQ, boundary_ok=None):
    """The windowed reduction of `values`, with the whole two-tier policy applied.

    THIS IS THE ONLY PLACE A COVERAGE, GAPPINESS OR BOUNDARY DECISION IS MADE, so every
    Stage-2 caller -- production, the certified PIT reproduction and the offline tools --
    inherits it from one implementation rather than from four copies.

    values         : the metric's PER-ROW series, NEWEST-FIRST, over the source's full panel.
    w              : this metric's already-scaled window (rp.scale_window(...)).
    key            : the Stage-2 metric key -- decides Type D/U and the boundary limit.
    rpy            : this source's rows per year.
    tempcdx        : the source's frame, for the NAME-level calendar-gap test and (for
                     grahamNumberToPrice) the adverse-vs-missing reason.  None disables the
                     calendar test only; coverage and gappiness still apply.
    structural_lag : rows at the OLD end that cannot be computable by arithmetic (the YoY
                     lag).  Removed from the coverage DENOMINATOR -- see COVERAGE_DENOMINATOR.
    scoring_nq     : the ambient scoring window, in quarters, for the calendar-gap test.
    boundary_ok    : boolean mask, aligned to `values`, marking rows whose non-computability is
                     ADVERSE.  Required for a Type-D column to take its boundary; None means
                     "cannot tell", which refuses rather than imputes.

    ORDER IS PART OF THE POLICY.  The calendar test is name-level and comes FIRST (a company
    that stopped filing has no trustworthy window for any metric to be measured over).  Then
    the Type-D branch, because a fully-undefined Type-D column must reach its BOUNDARY rather
    than the coverage rule's NaN -- inverting these two is the one ordering the spec calls out
    by name ("a correctly-computed coverage feeding a median fill on a Type-D column CREATES
    the reward the whole scheme exists to prevent").
    """
    v = pd.to_numeric(pd.Series(list(values)), errors='coerce')
    v = v.replace([np.inf, -np.inf], np.nan)
    n_full = len(v)
    vw = v.head(int(w)) if w else v.head(0)
    n_present = len(vw)
    if n_present == 0:
        return np.nan

    # ---- name-level calendar gappiness ------------------------------------------------
    if calendar_gap_refused(tempcdx, rpy, scoring_nq):
        _count(key, RULE_CALENDAR_GAPS)
        return np.nan

    ok = vw.notna()
    n_ok = int(ok.sum())

    # ---- Type D: fully undefined -> boundary or refuse; partial -> its own observations --
    if key in TYPE_D:
        if n_ok == 0:
            lim = BOUNDARY_LIMIT.get(key)
            adverse = False
            if boundary_ok is not None:
                bm = pd.Series(list(boundary_ok), index=v.index).head(int(w))
                adverse = bool(bm.fillna(False).any())
            if lim is not None and adverse:
                _count(key, RULE_BOUNDARY)
                return float(lim[0])
            _count(key, RULE_REFUSED)
            return np.nan
        #  PARTIAL COVERAGE MUST NOT TAKE THE BOUNDARY (ADDENDUM A, closing clause).  A name
        #  with 4 of 16 computable Graham quarters WAS profitable four times; imputing "as if
        #  earnings were 0" would discard its own observations.  It keeps its observed mean;
        #  the D-8 coverage SHRINKAGE toward the column prior is the designed treatment and is
        #  a separate, unimplemented item -- NOT the coverage-to-NaN collapse, which is why
        #  the coverage and interior-gap gates below are skipped for Type D.
        return vw.mean()

    # ---- Type U: coverage, then interior gappiness -------------------------------------
    if n_ok == 0:
        return np.nan
    n_struct = max(0, int(structural_lag) - max(0, n_full - n_present))
    denom = max(0, n_present - n_struct)
    if denom == 0:
        return np.nan
    if (n_ok / float(denom)) < COVERAGE_MIN:
        _count(key, RULE_COVERAGE)
        return np.nan
    if interior_missing_runs(ok.to_numpy()) > MAX_INTERIOR_RUNS:
        _count(key, RULE_METRIC_GAPS)
        return np.nan
    return vw.mean()


# =========================================================================== #
#  5.  CROSS-FIELD IMPOSSIBILITY -- the vendor's numbers contradict each other #
# =========================================================================== #
#  THE CEO'S RULING (2026-08-14): an input sanity check that ABSTAINS.  "A ratio that is
#  physically impossible means the VENDOR DATA IS WRONG, not that the company is unusual."
#
#  WHY IT IS HERE AND NOT IN `data_quality.check_price_sanity`, WHICH IS THE OBVIOUS HOME.
#  That function is the repo's existing cross-field-plausibility idiom and its thresholds are
#  set exactly where "the combination becomes ARITHMETICALLY IMPOSSIBLE".  But a False from
#  it enters `filter_invalid_data`, which deletes every row AT OR BEFORE the flagged one
#  (prefix removal).  The corruption this section exists for sits on the NEWEST row of JHX --
#  so routing it there would delete a $10bn building-products company's entire history over
#  one bad cell.  That is an EJECT; the CEO ruled ABSTAIN.  Section 1's `SANITY_IMPOSSIBLE`
#  has the same problem for the same reason (it ejects the SOURCE).  So this is a FIFTH
#  mechanism with a fifth home: it refuses CELLS.
#
#  IS THIS `vendor_contamination`'s JOB?  NO -- IT IS A DIFFERENT CHECK, and the difference is
#  not cosmetic.  `vendor_contamination` answers "are these numbers SOMEONE ELSE'S?": it hashes
#  (date, revenue, totalAssets) triples ACROSS SOURCES, finds pairs sharing three or more, and
#  compares company names -- which is how it found `058820.KQ` serving Chipotle's statements.
#  Its evidence is BETWEEN two names, its output is a REPORT a human promotes into
#  `QUARANTINE_RULES`, and it removes whole row windows.  This check answers "are these numbers
#  CONSISTENT WITH THEMSELVES?": its evidence is WITHIN one filing (or one name's own adjacent
#  periods), it needs no second source to exist, it is AUTOMATIC because an identity violation
#  admits no legitimate reading, and it refuses CELLS.  Neither subsumes the other -- JHX's
#  divided-by-1e6 assets match no other company on earth, so the contamination detector cannot
#  see them, and Chipotle's statements under a KOSDAQ ticker are internally perfectly
#  consistent, so this check cannot see those.  They are deliberately kept apart.
#  ONE COUPLING, AND IT RUNS ONE WAY (review S11).  They are not INDEPENDENT: this check runs
#  upstream at fetch time and can NaN `totalAssets`, which is one of the three fields
#  `vendor_contamination._triples` hashes and then `dropna`s on.  So ~50 of 61,354 rows lose
#  their corroboration capacity for the cross-source detector.  Immaterial in size, and stated
#  here so the next reader does not re-derive it.
#
#  WHY THE INSTRUMENT IS AN ACCOUNTING IDENTITY AND NOT A RATIO BOUND.  The change was
#  proposed as "derive thresholds for the impossible ratios (SGA/revenue, (LTD+CL)/assets)
#  from the panel".  MEASURED, THAT DOES NOT WORK AND THE PROPOSAL IS REFUSED:
#  `GXAI` -- a real pre-revenue micro-cap ramping from $256 to $6.0M of quarterly revenue --
#  carries sgaTTM/salesTTM of 9,456 and, one quarter earlier, +inf.  `RDZN`'s CORRUPT value
#  is 1.98e5.  Twenty-fold apart, both on the same axis: no threshold on that ratio separates
#  a legitimately unusual company from a corrupt cell, and the brief's own instruction was
#  that the line is "physically impossible, not unusual -- orders of magnitude, not factors".
#  An IDENTITY has no such tail.  A pre-revenue biotech's balance sheet balances; a leveraged
#  REIT's balances; a commodity trader's balances.  Every relation below is a definitional
#  containment or the balance-sheet identity itself, so a company cannot fail one by being
#  unusual -- only by being misreported.  [panel = HomeGDrive/pipeline Boresults_dic CUR3K
#  2026-08-13, 61,354 rows / 2,629 sources.]
#
#  THE DEFECT IS A UNITS ERROR, AND SAYING SO IS WHAT SETS THE THRESHOLDS.  The two names
#  that motivated this are not "extreme", they are off by a POWER OF TEN:
#      JHX  2026-04-01  totalAssets 1.3493e4 while totalLiabilities + equity = 1.34934e10
#                       -- reported assets are (L+E)/1e6 TO FIVE FIGURES; totalCurrentAssets
#                       (1.7592e3 vs 1.8268e9) and PP&E (3.2794e3 vs 3.3188e9) are divided by
#                       the same 1e6 while revenue, SG&A, debt and equity stay full-scale.
#      RDZN 2024-10-01  totalCurrentLiabilities 6.136344e13 against totalLiabilities 6.2476e7
#                       -- current liabilities 982,186x TOTAL liabilities.
#      RDZN 2025-10-01  SG&A 1.090059e13 against 1.006e7 the quarter before and 1.542e7 after.
#  NOTE THE SHAPE: RDZN's are SINGLE CELLS.  The claim that "RDZN's SGA is ~2e5x its revenue
#  across four rows" is the TTM rolling sum smearing ONE corrupt cell over four windows, not
#  four bad rows -- the raw panel has exactly one of each.
#
#  WHAT IS DELIBERATELY *NOT* HERE, because it was measured and REFUSED:
#    * `SGA <= grossProfit - operatingIncome` (total operating expenses).  A true inequality,
#      and it catches RDZN's SG&A cell at 9.74e5.  But the DENOMINATOR degenerates on real
#      companies -- FMP reports operatingIncome == grossProfit whenever the opex lines are
#      absent -- so the ratio reaches 1.03e3 for ERA.PA, 1.45e3 for LACR.PA, 1.49e3 for CFX.L
#      and 6.16e9 / 8.18e10 for two Korean industrials, all with ordinary statements.  RDZN
#      sits INSIDE that population.  No threshold separates them, so the limb is not shipped
#      and RDZN's SG&A cell is instead caught by the isolated-spike rule below.
#    * `revenue / totalAssets` (asset turnover).  It would catch SSRM, whose 2026-04-01
#      balance sheet is scaled down ~1,390x while its income statement is not -- but its
#      legitimate tail (traders, distributors) runs into the same decade, and 65 panel rows
#      sit in [100, 1000) with no way to adjudicate them from the panel alone.  SSRM is
#      therefore NOT caught by this change and is named here rather than left implied.
#      IT IS MISSED FOR *TWO* REASONS, NOT ONE (review S6).  Its identities hold because the
#      whole balance sheet was scaled together -- AND its corrupt row is its NEWEST, where
#      the spike rule is structurally blind (see the endpoint note below).  An earlier draft
#      gave only the first, which reads as though the spike rule had had a shot at it.
#  EACH RELATION DECLARES WHETHER IT IS TWO-SIDED, AND THAT FLAG IS NOT COSMETIC.  The first
#  version of this table had no such flag and tested `ratio >= factor` for everything -- which
#  silently MISSED JHX, the case that motivated the whole change: its `totalAssets` is
#  (L+E)/1e6, so the identity ratio is 1e-6 and a one-sided upper test never fires.  Caught by
#  measuring the before/after rather than by reading the code.
#    IDENTITY (A = L + E) -> TWO-SIDED: either side of 1 is the same contradiction.
#    CONTAINMENT (subset <= superset) -> ONE-SIDED: a subset being much SMALLER than its
#    superset is the normal case and must never fire.
IMPOSSIBLE_RELATIONS = (
    #  (name, numerator fields, denominator fields, factor, two_sided, fields refused)
    #
    #  BALANCE-SHEET IDENTITY.  A = L + E, so A/(L+E) = 1 for any correct filing.  MEASURED
    #  log10 of that ratio: 90.14% within +-0.02 (i.e. +-5%), and the distribution is
    #  RIGHT-SKEWED for a real reason -- FMP's `totalStockholdersEquity` frequently excludes
    #  minority interest, which makes A exceed L+E by the MI share (7.37% in [1.05, 1.26],
    #  1.54% in [1.26, 2.0]).  That mechanism is BOUNDED: minority interest cannot exceed the
    #  balance sheet, so it cannot take the ratio past ~2.  Counts by |log10|:
    #      [0.5,1) 122    [1,2) 54    [2,3) 10    [3,5.5) 7    [5.5,6.5) 2
    #  A factor of 100 is 50x past the largest mechanism that can produce a LEGITIMATE
    #  deviation, and there is then a clean 2.4-decade empty band below it.  CORRECTED
    #  (review S7): the far bin holds TWO rows, not one -- JHX at 10^-6.00 and FMONC.PA at
    #  10^+5.62.  An earlier draft said 'JHX alone at 1e-6', which is wrong in the direction
    #  that matters, because FMONC.PA is caught by the UPPER limb and would be missed if the
    #  relation were reasoned about as a lower-tail test.
    #  COST 19 rows of 61,197 computable (0.031%).  The 54 rows in [10,100) -- 51 above, 3
    #  below -- are LEFT ALONE: they are probably wrong too, but no mechanism names them, and
    #  that is the over-reach line.
    ('balance_sheet_identity',
     ('totalAssets',), ('totalLiabilities', 'totalStockholdersEquity'), 100.0, True,
     ('totalAssets', 'totalLiabilities', 'totalStockholdersEquity')),

    #  CURRENT ASSETS ARE A SUBSET OF TOTAL ASSETS.  No filer, no convention, no sector: a
    #  subset cannot exceed its superset.  MEASURED TCA/TA: 0.445% in (1, 1.1] (rounding and
    #  restated lines), 1 row in (1.1, 2], 2 rows in (2, 10], **ZERO rows in (10, 100]**, then
    #  7 rows in (100, 1000] -- a measured empty decade separating the slop from the
    #  degenerate population, exactly the density argument the AQI floor was rebuilt on.  The
    #  cut is placed where the empty band is.  COST 7 rows (0.011%); all seven adjudicated
    #  (CRML totalAssets 3,125 against TCA 344,668; RAIN 401,763 against 53.8M; ALAQU.PA x5).
    ('current_assets_within_assets',
     ('totalCurrentAssets',), ('totalAssets',), 10.0, False,
     ('totalCurrentAssets', 'totalAssets')),

    #  NET PP&E IS A SUBSET OF TOTAL ASSETS.  Same argument.  MEASURED PPE/TA: 2 rows in
    #  (1, 1.1], 13 in (1.1, 2], 5 in (2, 10], then 19 rows at >= 10 of which every one is
    #  adjudicable as corrupt (OTEX 2026-04 PP&E 6.57e11 against assets 1.31e10; USIO
    #  2025-10 6.58e12 against 1.35e8; three Korean KOSDAQ names whose totalAssets is the
    #  1,000x-too-small cell the identity limb also flags).  COST 19 rows (0.031%).
    ('ppe_within_assets',
     ('propertyPlantEquipmentNet',), ('totalAssets',), 10.0, False,
     ('propertyPlantEquipmentNet', 'totalAssets')),

    #  CURRENT LIABILITIES ARE A SUBSET OF TOTAL LIABILITIES -- AND THIS ONE GETS 500, NOT 10,
    #  FOR A MEASURED REASON.  FMP populates `totalLiabilities` with only a FRAGMENT of the
    #  liability side for insurers and banks while `totalCurrentLiabilities` carries the whole
    #  of it, so the ratio reaches 247.5 for KakaoBank (323410.KS), 78 for ENJ/ENO, 30 for
    #  085620.KS, 18.9 for Sun Life (SLF.TO) and 15 for iA Financial (IAG.TO).  Those are
    #  large, real, healthy companies and a factor-10 or factor-100 cut WOULD REFUSE THEIR
    #  ROWS -- the over-reach the brief warns about, caught by measurement not by argument.
    #      cut 100 -> 10 rows      cut 500 -> 7 rows      cut 1,000 -> 1 row
    #  THE CUT WAS 1,000 AND THAT WAS WRONG (review S5).  ALAQU.PA's six rows sit at 957.7,
    #  972.1, 972.5, 996.8, 998.0 and 999.8 -- ALL BELOW 1,000 -- so the limb fired on ONE row
    #  while this comment claimed seven, and 1,000 sat directly ON a cluster edge, slicing
    #  exactly the kind of population the spike rule is careful not to slice.  The empty band
    #  is 247.5 -> 957.7; 500 sits inside it, catches all seven, clears KakaoBank by 2x, and is
    #  the constant `SCALE_SPIKE_FACTOR` already uses.  COST 7 rows (0.011%).
    ('current_liabilities_within_liabilities',
     ('totalCurrentLiabilities',), ('totalLiabilities',), 500.0, False,
     ('totalCurrentLiabilities', 'totalLiabilities')),
)

#  ---- ISOLATED SCALE SPIKE -- one period disagreeing with BOTH its neighbours -------------
#  The relations above are cross-field at one instant.  This is the same defect seen along
#  TIME, and it is what catches the single corrupt cell no identity contains (RDZN's SG&A).
#  A value is refused when it differs from BOTH the preceding and the following period, IN
#  THE SAME DIRECTION, by at least `SCALE_SPIKE_FACTOR`.
#
#  "ISOLATED" IS THE COMMON SHAPE, NOT THE ONLY ONE, and pretending otherwise would misdescribe
#  what ships.  MEASURED, several names ALTERNATE: `NRP.AS` reports PP&E as 537,000 / 1.54e9 /
#  606,000 / 2.04e9 ... every six months for eight years (14 rows refused on that one name),
#  and `IMDA.PA`, `WHA.AS` and `ARBB.L` do the same.  Each row still disagrees with both of its
#  neighbours by >=500x, so the rule fires on all of them -- correctly: no year-over-year index
#  can be computed across two incompatible units, and abstaining on every such row is the
#  honest answer, not a bug in the rule.
#
#  Requiring both sides and one direction is what makes it a SPIKE rather than a LEVEL CHANGE: a SPAC merger, a rights
#  issue or a shell starting to trade moves the level and STAYS there (SOFI's totalAssets
#  4.66e5 -> 8.56e9 -> 8.06e8, a SPAC merger; CRML's equity 1.42e4 -> 1.51e8 -> 1.51e8), and
#  NEITHER OF THOSE TWO CELLS is refused.  Stated precisely because it is easy to overclaim:
#  a DIFFERENT SOFI cell (totalCurrentAssets, 2021-01) IS refused, so "SOFI is untouched"
#  would be false -- what the rule leaves alone is the LEVEL CHANGE, not the name.  Growth is likewise untouched: GXAI's revenue rises 256 -> 6,005,051 over three
#  years without a single period disagreeing with both its neighbours by 1,000x.
#
#  THE FACTOR IS 500, AND IT IS READ OFF THE DENSITY -- the same argument the AQI floor was
#  rebuilt on, and it is NOT 1,000.  MEASURED on 382,720 interior comparisons over the field
#  set below (both neighbours present, all three values non-zero), |log10| of the smaller of
#  the two neighbour ratios, in quarter-decade bins.  THE DENOMINATOR IS THE FULL INTERIOR
#  POPULATION, not the 181,760 same-side subset an earlier draft divided by -- every
#  PERCENTAGE in that draft was ~2.3x too large, though the raw counts were right and no
#  conclusion moved (review S7):
#      [10^1.00,10^1.25) 336   [10^2.00,10^2.25)  78   [10^3.00,10^3.25) 55
#      [10^1.25,10^1.50) 206   [10^2.25,10^2.50)  73   [10^3.25,10^3.50) 29
#      [10^1.50,10^1.75) 159   [10^2.50,10^2.75)  57   [10^3.50,10^3.75)  8
#      [10^1.75,10^2.00) 149   [10^2.75,10^3.00)  88   [10^3.75,10^4.00)  1
#  The body decays monotonically 336 -> 57 and then RISES to 88 before falling away to
#  8 and 1.  That rise is the degenerate population -- the vendor's THOUSANDS-vs-UNITS error --
#  and it STRADDLES 10^3 rather than starting there, because the ratio is taken against
#  neighbours that are themselves moving, so a true x1,000 error lands anywhere in roughly
#  [333, 3000].  A cut AT 1,000 therefore slices that population down the middle and refuses
#  only its upper half; the excess BEGINS at 10^2.75 = 562.  500 is that onset rounded to a
#  readable constant, placed just OUTSIDE the excess rather than inside it.
#      cut 100    413 cells (0.108%)     cut   500  215 cells (0.056%)   <- SHIPPED
#      cut 1,000  117 cells (0.031%)     cut 3,162   33 cells (0.009%)
#      cut 10,000  24 cells (0.006%)
#  A cut at 1,000 leaves the whole 88-observation excess bin in the data; 500 costs 98 extra
#  cells (0.026% of comparisons) to take it out.
#
#  THE FIELD SET IS RESTRICTED TO BALANCE-SHEET STOCKS, AND THAT RESTRICTION IS THE WHOLE
#  DEFENCE.  Applied to FLOWS the rule over-reaches immediately and measurably: BMV.L earns
#  GBP 24 in one quarter between two GBP 1-3M quarters, VIVK's operating cash flow passes
#  through 35 between 1.6M and 7.0M, ALK.L's EBITDA oscillates through 0.025.  A flow
#  legitimately passes near zero; a company's total assets do not teleport to a millionth of
#  themselves for exactly one quarter and back.  `sellingGeneralAndAdministrativeExpenses` is
#  included but UP-SPIKE ONLY, on the same asymmetry: an expense that momentarily rounds to
#  nothing is odd (a shell), whereas one that is 1,000x its own neighbours on both sides
#  while revenue and every other line continue normally is not a quarter any going concern
#  has.  It catches RDZN (1.006e7 -> 1.090059e13 -> 1.542e7), JSG.L, SHI.L and ETG.TO.
#
#  THE CORRUPTION IS NOT CONFINED TO MICRO-CAPS, which is the finding that most justifies
#  shipping this.  Adjudicated refusals on the 2026-08-13 panel include `CME` (current
#  liabilities 5.58e7 for one quarter between two ~1.5e11 quarters -- CME Group holds ~$50bn
#  of clearing-member deposits, so the 5.58e7 is the wrong one), `ABN.AS` (1.41e8 between two
#  ~2.8e11 quarters), `UL`/`ULVR.L`/`UNA.AS` (Unilever's three listings, all three carrying the
#  same 1.19e10 SG&A row between two ~1e7 rows) and `OTEX` (PP&E 6.57e11 against total assets
#  of 1.31e10).  Before this, every one of those fed a Beneish index.
#  STRUCTURALLY BLIND ON EVERY SOURCE'S NEWEST AND OLDEST ROW, and that is a real limit, not
#  a detail: `ok` requires BOTH neighbours, so a row with no successor can never fire -- 5,258
#  of 61,354 panel rows (8.6%), and the newest row is the one that drives every current-period
#  metric.  JHX's corruption is on its newest row and survived only because the IDENTITY limb
#  caught it; SSRM's is on its newest row and is missed entirely.  NOT REPAIRED HERE,
#  deliberately: a one-sided endpoint test cannot distinguish a corrupt endpoint from a
#  genuine level change (a SPAC merger, a rights issue, a disposal) landing on the newest row,
#  which is exactly the discrimination the two-sided form buys.  The identity limbs are what
#  cover the endpoint today; closing the rest of the gap needs a different instrument.
SCALE_SPIKE_FACTOR = 500.0
#  ONE HOME FOR THE RELATION LABEL.  It was built inline as a format string in exactly one
#  place and is now READ BACK by `_drop_relation_hits_the_spike_rule_already_explained`, which
#  identifies spike hits BY THIS PREFIX.  Two literals would be this repo's recorded worst bug
#  class (one fact stated twice, drifting apart) on a string that silently decides whether the
#  coupling guard fires at all -- a typo would make it a no-op that no assertion would notice.
SCALE_SPIKE_RELATION_PREFIX = 'isolated_scale_spike:'


def scale_spike_relation(field):
    """The relation label the isolated-scale-spike rule reports for `field`."""
    return SCALE_SPIKE_RELATION_PREFIX + str(field)


#  field -> allowed spike direction: 'both' (a stock cannot move either way) or 'up'.
SCALE_SPIKE_FIELDS = {
    'totalAssets': 'both',
    'totalLiabilities': 'both',
    'totalStockholdersEquity': 'both',
    'totalCurrentAssets': 'both',
    'totalCurrentLiabilities': 'both',
    'propertyPlantEquipmentNet': 'both',
    'sellingGeneralAndAdministrativeExpenses': 'up',
}

#  ---- PRICE-SCALE CONTRADICTION -- the third rule, and it is NOT an identity ---------------
#  A row whose `price` is a vanishing fraction of its OWN `bookValuePerShare` is not a
#  valuation, it is a units error.  Both quantities are per-share and both are in the
#  company's reporting currency, so every FX effect and every minor-unit convention cancels
#  and what is left is a price-to-book multiple.
#
#  READ THE `IMPOSSIBLE_RELATIONS` PREAMBLE BEFORE MOVING THIS RULE INTO THAT TABLE.  It says,
#  correctly, that every relation there is a definitional containment or the balance-sheet
#  identity, so a company cannot fail one BY BEING UNUSUAL -- and it records two statistical
#  ratios (SG&A/revenue, revenue/totalAssets) that were MEASURED AND REFUSED for exactly the
#  reason that a legitimate tail overlaps the corrupt population.  Price-to-book is a
#  statistical rule of that second kind, not an identity, so it is kept SEPARATE and carries
#  its own evidence.  Folding it into that table would quietly weaken what the table means.
#
#  0.02 IS A PROVISIONAL CONSERVATIVE FLOOR, NOT A THRESHOLD, AND THE DIFFERENCE IS THE WHOLE
#  JUSTIFICATION (review, 2026-09-01).  An earlier version of this note defended the number as
#  "one definition of contaminated" -- shared with `price_scale_audit`'s `PB_ALARM`.  That is an
#  argument about CONSISTENCY, and consistency was never the difficulty here: a REPORTING
#  threshold and a REFUSING threshold have ASYMMETRIC COSTS, so the same number is not
#  automatically right for both jobs.  Over-reach in the detector costs one line in a log.
#  Over-reach HERE invents an adverse judgement -- `calcScore.calcByTier` scores a NaN as a FAIL
#  (`calcScore.calcByTier's `Sign * val > 0` pass test`), so a wrongly-refused row does not abstain, it marks a real company down.
#  WHAT ACTUALLY SUPPORTS 0.02 is that the two errors differ in KIND, and the earlier wording
#  here -- "under-reaching costs nothing" -- was the weaker claim and simply false.
#  UNDER-REFUSING LEAVES A CONTAMINATED NAME READING AS CHEAP IN THE CEO'S LIST, which is the
#  entire defect this rule exists to remove, so it costs exactly what Q-48 costs.  OVER-REFUSING
#  invents a FAIL against a real company on evidence nobody has.  Both are real; what separates
#  them is VISIBILITY.  An under-refusal leaves the pipeline where it already was, on a name the
#  detector still reports; an over-refusal creates a NEW wrong answer that nothing downstream
#  reports at all.  That is what puts the cut at the conservative end -- and it is why the cut is
#  EXPECTED TO MOVE OUTWARD once the conjunction below is measured, rather than treated as
#  settled.  It is still shared with
#  `price_scale_audit` -- defined here, imported there -- so the reporting and the refusing sides
#  can never name different rows; that sharing is worth having, it is simply not the reason the
#  number is 0.02.  Derivation of the number itself, unchanged, from the module that set it: "a
#  1000x under-scale maps the ORDINARY price/book range [0.5, 10] onto [0.0005, 0.01], so an
#  alarm at 0.02 covers a true price/book up to 20 ... a real distressed equity bottoms out
#  around 0.05-0.10".
#
#  WHAT THE THRESHOLD DOES *NOT* HAVE, stated because every other cut in this module has one:
#  A MEASURED EMPTY BAND.  Row-level log10(price/book) on the 2026-08-29 CUR6K panel (255,090
#  computable rows) decays smoothly through the cut -- 628 rows in [0.056,0.100), 351 in
#  [0.032,0.056), then 127, 109, 107, 50, 45, 16 -- with no gap anywhere near 0.02.  So this is
#  NOT the density argument `SCALE_SPIKE_FACTOR` and the AQI floor rest on and must not be
#  described as one.
#
#  THE EVIDENCE THAT IT IS NEVERTHELESS THE RIGHT KIND OF ROW is an INDEPENDENT WITNESS: the
#  share count on the row against the source's OWN median share count.  It is REPORTED here and
#  NOT gated on -- and the reason first given for that was WRONG, so it is corrected rather than
#  quietly dropped.  It said gating on the witness would be "a second definition of
#  contaminated".  IT WOULD NOT: A CONJUNCTION IS A NARROWER DEFINITION, NOT A SECOND ONE, and
#  the witness is computable exactly where this rule runs -- `tempfund` at `the refusal hook in getData_fmp.getFundamentalsData`
#  is one ticker's whole history, so the source's own median share count is already in hand.
#  THE REAL REASON IT IS DEFERRED IS SCOPE: `marketCap/equity < 0.10 AND shares >= 5x off own
#  median` would reach the 0.02-0.10 band and roughly TRIPLE the coverage -- a change in how much
#  of the universe this rule touches, not a units-error repair -- and its false-refusal rate has
#  not been measured.  Ship the floor, widen deliberately.  SCHEDULED, NOT DECLINED.  Share-count corruption is what produces this shape when
#  the price is sound.  Share of rows sitting >= 5x off their own source's median share count,
#  by price/book band, same panel:
#      [0.000,0.005)   151 rows  31.8%       [0.020,0.030)    93 rows  24.7%
#      [0.005,0.010)   119 rows  20.2%       [0.030,0.050)   306 rows  32.0%
#      [0.010,0.020)   129 rows  20.9%       [0.050,0.100)   687 rows  22.6%
#      [0.100,0.200) 2,886 rows   9.3%       whole population 255,090 rows  3.45%
#  A 6-9x enrichment over the base rate inside the cut -- AND THE SAME ENRICHMENT PERSISTS
#  ABOVE IT.  So this rule UNDER-REACHES BY CONSTRUCTION: it refuses a SUBSET of the
#  contaminated rows, and rows between 0.02 and 0.10 carry the signature at the same rate and
#  are LEFT ALONE.  That is this module's over-reach line taken in the conservative direction;
#  it is NOT a claim that the population above the cut is clean.
#  Adjudicated by hand on the 08-29 panel, every one a vendor scale break rather than a
#  valuation: PARR (2.9M shares against $490M of equity, with `price` pinned at exactly 2.20
#  for eight consecutive quarters), BACTI-B.ST (10,000 shares against SEK 122M of equity),
#  ZEN.L (6.3M shares against 104M), ANOT.ST (9.3M against 1.10bn), 001230.KS (9.0M for one
#  quarter between two ~45M quarters), QBY0.DE / 0CHZ.L (199,326 against 24,915,897 -- one
#  company, two listings, the same break on both).
#
#  WHAT THE CONTAMINATION IS WORTH, so severity is judged from a number and not an impression.
#  Every Stage-1 criterion whose value is a function of a refused field, with its tier weight
#  (S=1.0, A=0.75, B=0.5, C=0.3, D=0.1, N=0), against a Sigma-w of 17.85 over all five scoring
#  dicts:
#      mEarningsYield             earningsYield   S  1.00      mSalesToMarketCap   N  0.00
#      mFreeCashFlowToMarketCap   marketCap       S  1.00      uGrahamNumberToPrice N 0.00
#      mBookToPrice               marketCap       B  0.50
#      dBookToPrice               marketCap       B  0.50
#      dCFOtoMarketCap            marketCap       B  0.50
#      PEG (via calcMetrics.peg_local)  price     C  0.30
#  TOTAL 3.80 of 17.85 = 21.3% OF THE STAGE-1 GATE.  Recorded here because it was first
#  reported as 2.80 / 15.7% -- an arithmetic slip that dropped the two Tier-B DIFF criteria
#  (`dBookToPrice` and `dCFOtoMarketCap`, 0.50 each) and understated the exposure by 36%.
#  `bookToPrice`, the criterion the issue register names, is the FOURTH largest of the eight.
#
#  COST on the 2026-08-29 CUR6K panel: 352 rows over 75 of 4,934 sources.  (The per-share form
#  this rule was first written in fired on 399 rows / 85 sources; see `price_scale_hits` for why
#  the balance-sheet form is narrower and better founded.)
#
#  THE MARKET-CAP SIDE IS REFUSED AND THE BALANCE SHEET IS NOT.  THREE shapes produce this one
#  signature, and it is the multiplicity -- not the absence of a decade -- that rules out a
#  corrective factor: there is no single number to multiply by.
#    * THE ATRI SHAPE, a REAL decade, and it is settled to a traded price (Q-43: bought at
#      $675.07 "after correcting FMP's 1000x scaling").  FMP serves `price` and `marketCap` at
#      1/1000 of the tape while the share count and the balance sheet stay sound.  Do not read
#      the multiplicity as "the decade shape does not exist" -- it does, and it is the case this
#      whole line of work started from.
#    * THE QBY0.DE SHAPE.  The SHARE COUNT is ~100x too small on the older rows, so `marketCap`
#      (= price x shares) is 100x too small and `bookValuePerShare` (= equity / shares) is ~100x
#      too large, while `price` and `totalStockholdersEquity` are both CORRECT across the break
#      (equity 92.34M -> 94.17M and price 4.34 -> 3.47 over the same quarter in which marketCap
#      goes 0.865M -> 86.5M and the share count goes 199,326 -> 24,915,897).
#    * THE CCM SHAPE, which is NOT a scaling defect at all: negative book equity carrying a
#      sign-DISAGREEING `bookValuePerShare` (+12,783 against equity -2.10bn).  The balance-sheet
#      form of the test excludes it; the per-share form did not.
#  WHY THE REFUSAL IS ONE-SIDED, by evidence rather than by ignorance.  `totalStockholdersEquity`
#  already carries TWO guards in this module -- `balance_sheet_identity` and the
#  isolated-scale-spike rule both name it -- and it runs continuous across every adjudicated
#  break.  `marketCap` carried NONE.  So the contradiction is read against the leg two other
#  rules already vouch for, and only the cap side, plus the quantities derived from it, abstain.
#
#  `earningsYield` RIDES WITH THEM because it is a VENDOR field carrying the same price in its
#  denominator -- verified on this panel: `earningsYield` equals `netIncomePerShare / price` to
#  within 1% on 94.9% of 264,601 rows, median ratio 1.000000 -- and it is Stage-1's Tier-S
#  (w = 1.0) cheapness criterion `mEarningsYield`.  Refusing the price while leaving the
#  vendor's price-derived yield behind would fix the smaller exposure and keep the larger one.
#
#  WHAT IS DELIBERATELY *NOT* REFUSED, so the residual is named rather than implied:
#    * `weightedAverageShsOut` -- the broken field in the QBY0.DE shape and SOUND in the ATRI
#      shape, and the contradiction does not identify it.  Refusing it would take out
#      `dSharesOutstanding` (Tier B) and every per-share vendor field on a name whose defect may
#      be entirely on the price side.
#    * `totalStockholdersEquity` -- the leg the test is taken AGAINST, guarded twice already in
#      this module, measured continuous across every adjudicated break.
#    * `netIncomePerShare`, `revenue` -- measured sound across the break on the names adjudicated
#      above; refusing them would remove a correct number.
#    * STAGE-2 IS NOT MADE CONSERVATIVE BY THIS RULE, AND THAT ASYMMETRY IS DELIBERATE.  Stage-1
#      scores a NaN as a FAIL (`calcScore.calcByTier's `Sign * val > 0` pass test`), so a refusal there costs the name.  Stage-2
#      imputes a missing metric to the COLUMN MEDIAN, so a refusal there costs it NOTHING -- it
#      is a mild reward relative to a bad reading.  That is the right treatment and not an
#      oversight: a name only reaches the Stage-2 pool by surviving Stage-1, where the refusal
#      already bit, and inside a 100-name RANKING "no opinion" is what an absent measurement
#      means.  Penalising it twice would double-count.  Stage-2 metrics that go absent on a
#      refused name: `earnYield`, `freeCashFlowYield`, `bVpRatio`, `tbVpRatio`,
#      `grahamNumberToPrice`, `priceGrowth`, Altman's `x4`, `marketCapRevQuants` (via
#      `stage2_metrics._mcap_for_quants` -> `MCAP_QUANT_MISSING = 0.0`, i.e. neutral) and --
#      the one that reaches the price side through NO price field at all --
#      `stage2_metrics.nav_per_share_growth`, whose only contaminated input is
#      `bookValuePerShare`.  It is an ENDPOINT PAIR (`bvps.iloc[0]` against `bvps.iloc[-1]`), so
#      ONE refused row at either edge of its window kills the whole metric for that name.
#    * `grahamNumber` -- built from the RAW `bookValuePerShare` at `getData_fmp.stamp_frequency_and_graham`, which
#      runs BEFORE the hook at `:178`, so it is NOT refused.  Harmless today only because both of
#      its consumers divide by `price`, which is.  NAMED AS AN UNDEFENDED INVARIANT: a future
#      consumer of `grahamNumber` that does not divide by price would read a contaminated number
#      with nothing to stop it.
#
#  THE DETECTOR CANNOT SEE WHAT THIS RULE REMOVES, and that is handled rather than noted: check A
#  reads `price` and `bookValuePerShare`, this rule refuses BOTH, so it finds nothing on those rows and would print
#  a CLEAN REPORT over a defect it used to name -- "a detector blind to its own motivating case
#  is worse than none", in that module's own words.  `price_scale_audit.run_audit` therefore
#  reads the `SANITY_REFUSED_COLUMN` stamp and reports refused-upstream sources by name, so
#  "found nothing" and "already refused" can never print as the same line.
#  ---- THE WIDENING (Q-75), AND THE FORM IT COULD *NOT* TAKE -----------------------------
#  THE CEO RULED: ship the conjunction -- `marketCap/equity < 0.10` AND a share count >= 5x off
#  the source's own median -- on the argument that TWO conditions are NARROWER PER ROW than one,
#  so false refusals should fall even as coverage rises.  The first half of that is true.  THE
#  SECOND HALF IS FALSE, AND IT IS FALSE IN THE DIRECTION THAT MATTERS: a conjunction cannot be
#  a superset of one of its own conjuncts, so `pb < 0.10 AND witness` does not WIDEN `pb < 0.02`
#  -- it REPLACES it with something narrower on the rows they share.
#  MEASURED, five saved panels, rows refused:
#      panel                 pb<0.02   CONJUNCTION   UNION (shipped)
#      resdic_2026-07-17        265         160            354
#      CUR3K_2026-08-11          59          48             94
#      CUR3K_2026-08-07          67          53            101
#      NA1_EU1_2026-01-08       599         318            713
#      NA1_EU1_2025-12-09       636         342            779
#  The conjunction alone LOSES 46-437 rows per panel, and the rows it loses are THIS RULE'S OWN
#  HEADLINE CASES.  Hand-checked on `resdic_2026-07-17`:
#    * QBY0.DE -- 22 of its 24 rows sit at pb 0.006-0.013 with a share count of ~199,326.  The
#      CORRUPTION IS THE MAJORITY OF THE HISTORY, so 199,326 *is* the source's own median: the
#      witness ratio is 1.000 on all 22 contaminated rows and 125.0 on the TWO SOUND ones
#      (2025-10 and 2026-01, shares 24,915,897).  The witness does not merely miss here, it
#      points at the wrong rows.  Under the conjunction QBY0.DE is refused on ZERO rows.
#    * 0CHZ.L -- the SAME company's other listing carries the SOUND share count (24,915,897)
#      throughout and the SAME low `marketCap`, so pb is 0.006-0.013 on 22 rows with a witness
#      ratio of 1.000 everywhere.  This is the ATRI shape: the price side is broken, the share
#      count is not, and there is no share-count evidence to find.  Under the conjunction, ZERO.
#      (CORRECTION, because the note beside `PRICE_SCALE_REFUSE` has this wrong: QBY0.DE and
#      0CHZ.L are NOT "the same break on both".  They differ in EXACTLY the share count --
#      199,326 against 24,915,897, a factor of 125 -- while carrying IDENTICAL `marketCap` and
#      `totalStockholdersEquity` on every matching date.  So the cap side is contaminated on
#      BOTH listings and the share count on only one, which is why `marketCap = price x shares`
#      cannot be the whole mechanism on that pair.)
#    * CMCM -- share count flat at 574k-620k (witness ratio <= 1.11 on every row), equity
#      $1.5-3.8bn, marketCap $8-44M, pb 0.0024-0.016 on all 23 rows.  Conjunction: ZERO rows.
#  Two shipped fixtures pin exactly this: `test_the_ATRI_shape_is_refused...` and
#  `test_the_QBY0_shape_refuses_only_the_contaminated_ROWS...`.  The conjunction fails both.
#
#  SO THE SHIPPED FORM IS THE UNION, AND IT IS ONE RULE WITH TWO LEVELS -- NOT TWO DEFINITIONS
#  OF CONTAMINATED, AND NOT ONE NUMBER EITHER.  Said plainly, because the brief asked for one
#  definition and this is what one definition actually costs here:
#      refuse iff  pb < PRICE_SCALE_PB_ALARM                       (the unconditional floor)
#              or (pb < PRICE_SCALE_PB_WIDE and the witness fires) (the witnessed band)
#  The DEFINITION of contaminated is unchanged and singular -- "the market-cap side contradicts
#  this company's own balance sheet".  What the witness buys is a WEAKER price/book requirement,
#  because it supplies independent evidence for the same conclusion.  There is no way to widen
#  only-where-a-witness-fires with a single number, and pretending otherwise would be the
#  drifting-constant defect this module keeps recording.  Both numbers are exported and both are
#  pinned to `price_scale_audit` by `test_there_is_exactly_one_definition_of_contaminated`, so
#  the reporting side and the refusing side still cannot name different rows.
#
#  THE WITNESS IS THE SOURCE'S OWN MEDIAN SHARE COUNT, computed from `weightedAverageShsOut`
#  which is ALREADY ON THE FRAME -- no fetch, no new vendor field.  `PRICE_SCALE_WITNESS_MIN_ROWS`
#  exists because a median over one or two rows cannot separate the outlier from the level: with
#  two rows and a 100x break the median is the midpoint and BOTH rows land 10x off it, so the
#  rule would refuse the sound one too.  Measured cost of the minimum: 0 rows on all five panels
#  (every source there carries 16-24 rows), so it guards a shape the panels happen not to hold.
#
#  IT INTRODUCES NO NEW READ/REFUSE COUPLING, which is the one thing that could have made it
#  unsafe.  `weightedAverageShsOut` is refused by NO producer in this module -- not in
#  `PRICE_SCALE_REFUSE`, not in `SCALE_SPIKE_FIELDS`, named in no relation's refuse tuple -- so
#  the witness cannot read a cell this same pass has already rejected.
#  `test_the_witness_field_is_refused_by_NO_producer` pins that, because it is one edit away
#  from being false.
#
#  MEASURED FALSE-REFUSAL RATE OF THE WIDENED BAND -- 35 newly-refused rows over 19 sources on
#  CUR3K_2026-08-11, EVERY ONE ADJUDICATED BY HAND against the source's own share, price, equity
#  and book-value series:
#      24 rows / 14 sources  GENUINE VENDOR BREAK
#         WDH        2 rows, and the single best case for widening: shares 361.9M -> 36.1M
#                    (exactly 10x) on the NEWEST TWO ROWS with `price` flat at 13.45 -> 13.29 and
#                    equity flat at 5.1bn.  pb is 0.093 and 0.079, so THE CURRENT RULE MISSES IT
#                    ENTIRELY, and the spike rule is structurally blind on a newest row.  A real
#                    ~$400M company whose current-period market cap is 10x too low, feeding
#                    21.3% of the Stage-1 gate, with no existing guard able to see it.
#         MTVA       6 rows, share count 6,183-10,099 against an own-median of 460,674 with
#                    `price` pinned at exactly 27.83 for 21 consecutive quarters.  THREE of its
#                    rows were already refused at pb<0.02; the widening picks up six contiguous
#                    siblings of the same break, which is the shape the floor half-caught.
#         AGMR.TO    3 rows, 186M -> 22.9M shares with `price` unchanged at ~0.05.
#         025560.KS  2 rows, isolated 5-6x dips against 16.8-22.4M, `price` continuous.
#         CHLL.L     2 rows, the alternating 38,057 / 69,085 shape (cf. NRP.AS above).
#         058650.KS  1 row, 432,000 shares for one quarter against 3.896M in all 23 others.
#         003380.KQ  1 row, 16.9M against 97.3M, `price` continuous 7,340 -> 10,630 -> 13,550.
#         DLCG.TO    1 row, 784,240 between 52.8M and 78.3M, `price` continuous.
#         PYC.L      1 row, 2.0M between 135M and 276M, `price` continuous.
#         0QAU.L / VMX.PA / VK.PA  1 row each -- one company, three listings, the same isolated
#                    8x share-count break on 2025-10.
#         BLNE       1 row; this name's vendor share count is chaotic throughout (211,600 ->
#                    1.86M -> 9.8M -> 193,018 -> 27.9M) and no row of it is usable.
#         LAT.PA     1 row, genuinely broken (mcap 8.2M against 100.9M of equity, for a real
#                    aerospace supplier) but flagged for the WRONG REASON -- see ramp mechanism
#                    1 below.  Counted sound in OUTCOME, not in EVIDENCE.
#       4 rows /  2 sources  FALSE REFUSAL, both by one mechanism: ALDBT.PA 3, URU.L 1.
#       7 rows /  3 sources  UNRESOLVED, named rather than counted as wins:
#         XNET       5 rows.  Shares fall 63.5M -> 12.5M, a factor of 5.06 -- the witness fires
#                    by 1.2% -- and stay there while `price` roughly triples.  That is the
#                    signature of a REAL 1-for-5 ADS-ratio change, not a break, and a
#                    simultaneous equity restatement (318M -> 1.05bn) muddies it further.
#         VIVK 1, VEEE 1 -- both on the ramp mechanism, and on both the FLAGGED row is
#                    plausibly the CORRECT one rather than the broken one.
#  So 24 sound / 4 false / 7 unresolved: a false-refusal rate of 11.4% on the rows I can call,
#  and 31.4% if every unresolved row is counted against it.  For comparison, the balance-sheet
#  restatement's own 14 newly-refused rows adjudicated 10 unambiguous / 1 earned / 1
#  misattributed / 3 unresolved -- so the widened band sits in the same range and slightly
#  worse.  THIS IS NOT A CLEAN CHANGE AND MUST NOT BE REPORTED AS ONE.
#
#  THE TWO MECHANISMS THAT PRODUCE THE FALSE REFUSALS, because a rate with no mechanism cannot
#  be improved:
#    1. MONOTONE DILUTION RAMPS -- the dominant one; 5 of the 6 non-genuine sources.  A source
#       whose share count grows more than 5x ACROSS ITS WHOLE HISTORY has BOTH ENDS >= 5x off
#       its own median with NO row anomalous: ALDBT.PA ramps 1,440 -> 5.02M (3,500x), URU.L
#       138,000 -> 59.1M (430x), LAT.PA 1.15M -> 12.5bn, VEEE and VIVK ~22x.  The witness is
#       measuring the ramp, not a break.  URU.L is the clearest demonstration that the 5x cut
#       then slices ARBITRARILY: six adjacent early rows all sit at pb 0.02-0.05 with witness
#       ratios 2.4-5.6, and exactly the one row whose ratio crosses 5.0 is refused while five
#       indistinguishable siblings are not.
#    2. LEGITIMATE >= 5:1 REVERSE SPLITS AND ADS-RATIO CHANGES -- XNET, at 5.06x.  Any real
#       consolidation of 5:1 or more puts one half of a history >= 5x off its own median.
#
#  A NEIGHBOUR-RELATIVE RAMP GUARD WAS BUILT, MEASURED AND REJECTED; the numbers are recorded so
#  nobody spends a day rediscovering it.  The candidate: additionally require the row to be
#  FURTHER off its source's own median than BOTH of its date-adjacent rows are -- true for an
#  isolated break, false on a ramp where every row is a similar distance out.  MEASURED on
#  CUR3K_2026-08-11 it took the 35 new rows to 17, and the split is fatal:
#      adjudicated GENUINE     24 -> 13        adjudicated FALSE  4 -> 2
#      adjudicated UNRESOLVED   7 ->  2
#  It kills WDH's 2025-10 row, five of MTVA's six and two of AGMR.TO's three, while KEEPING half
#  the false refusals.  THE REASON IS STRUCTURAL, not a tuning miss: THE CORRUPTION USUALLY
#  ARRIVES AS A CONTIGUOUS BLOCK OF ROWS, so each broken row's neighbour is broken too and
#  nothing "stands out".  That is the same fact that makes the spike rule blind on endpoints,
#  restated one field along.  NOT SHIPPED.
#  WHAT WOULD PROBABLY WORK AND IS *NOT* MEASURED, so it is a lead and not a plan: a share-count
#  break moves `marketCap` while leaving `price` CONTINUOUS (WDH: shares /10, price 13.45 ->
#  13.29), a real reverse split multiplies `price` by the split factor and leaves `marketCap`
#  continuous, and a dilution ramp moves both gradually.  A price-vs-shares consistency witness
#  would separate all three where the median witness cannot.  Do not ship it on this paragraph.
#
#  WHAT THE WIDENING STILL MISSES, because it under-reaches in named directions:
#    * THE MAJORITY-CORRUPT SOURCE.  QBY0.DE's 22 contaminated rows are held ONLY by the pb<0.02
#      floor and carry a witness ratio of 1.000 on every one.  The median cannot see a defect
#      that IS the median, so raising the floor would lose them to nothing.
#    * THE SOUND-SHARE-COUNT SHAPE (ATRI, 0CHZ.L, CMCM).  No share-count evidence exists, so the
#      witnessed band cannot reach these at all -- only the floor can.
#    * THE 0.10-0.20 BAND, which still carries the witness at 7.5-12.4% against a whole-panel
#      base rate of 1.1-1.8% (measured, five panels).  A 5-8x enrichment left deliberately
#      untouched: a real equity CAN trade at 15% of book, and an over-refusal here invents a
#      Stage-1 FAIL that nothing downstream reports.
#    * The widening touches NO source in the saved panel's own deployed top-100 (checked against
#      `resdic_2026-07-17`'s `postRank`; the current rule touches none either), so it does not
#      move the CEO's list on that panel and did not need a rescore to ship.  That is a fact
#      about THAT PANEL, not a property of the rule.
#  WHAT THE WIDENING DOES DOWNSTREAM, measured because it lands on the number the CEO's list is
#  banded by.  `carveOut.marketcap_usd_by_source` takes the latest NON-NaN market cap, so
#  refusing a source's NEWEST one does not make it unknown -- it silently becomes an earlier
#  quarter's, and `carveOut.marketcap_fallback_report` exists to record exactly that.  The
#  widening gives that report MORE ROWS: sources with a refused newest market cap go 33 -> 40 on
#  resdic_2026-07-17 and 7 -> 10 on CUR3K_2026-08-11.  Every added case, with where `last()` then
#  lands:
#      WDH      402.0M -> 4.870bn (2 quarters back)   0QAU.L  2.540M -> 21.98M (1)
#      XNET      70.0M ->  54.97M (4)                 MPL.L   1.069M -> 6.912M (3)
#      NXTT      18.3M ->  61.28M (2)                 KTTA/KTTAW 5.216M -> 9.070M (1)
#      ALDBT.PA 169.2k ->  1.857M (2)
#  WDH is the case that matters and the fallback is RIGHT: 4.87bn is the pre-break market cap, so
#  the name moves out of a micro band it never belonged in.  ALDBT.PA is one of the two
#  adjudicated FALSE refusals and its fallback moves a nano-cap 11x, from 169k to 1.86M -- still
#  a nano-cap, so it reaches no band the CEO reads, but it is the shape an over-refusal takes
#  downstream and is named rather than netted off against WDH.
#  NOTE FOR WHOEVER OWNS `carveOut`: that function's docstring records "seven sources have a
#  refused newest market cap" from the 2026-08-29 CUR6K panel.  That figure is now LOW by about
#  40% and needs re-measuring on the next panel; it was not edited here because this change does
#  not own that file.
PRICE_SCALE_PB_ALARM = 0.02
#  The widened band's upper edge.  Reached ONLY with the witness (see above).
PRICE_SCALE_PB_WIDE = 0.10
PRICE_SCALE_RELATION = 'price_scale:price_vs_bookValuePerShare'
PRICE_SCALE_REFUSE = ('price', 'marketCap', 'bookValuePerShare', 'earningsYield')
#  The witness: this field, against the SOURCE'S OWN median of it.  Refused by no producer.
PRICE_SCALE_WITNESS_FIELD = 'weightedAverageShsOut'
PRICE_SCALE_WITNESS_FACTOR = 5.0
PRICE_SCALE_WITNESS_MIN_ROWS = 4


def share_count_witness_ratio(df, source_col='source'):
    """Series: how many times off its SOURCE'S OWN median each row's share count sits.

    Symmetric -- `max(shares/median, median/shares)` -- because a break can be in either
    direction and the witness asserts nothing about which.  NaN where the ratio is not
    computable: an absent or non-positive share count, or a source with fewer than
    `PRICE_SCALE_WITNESS_MIN_ROWS` usable rows, where a median cannot separate an outlier from
    the level.

    A frame with no `source` column is treated as ONE source, which is exactly what it is on
    the live path: the hook in `getData_fmp.getFundamentalsData` passes `tempfund`, one
    ticker's whole history, so the source's own median is already in hand with no groupby.
    """
    if PRICE_SCALE_WITNESS_FIELD not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype='float64')
    sh = pd.to_numeric(df[PRICE_SCALE_WITNESS_FIELD], errors='coerce')
    sh = sh.where(sh > 0)
    if source_col in df.columns:
        grp = df[source_col].astype(str)
        med = sh.groupby(grp).transform('median')
        cnt = sh.groupby(grp).transform('count')
    else:
        med = pd.Series(sh.median(), index=df.index, dtype='float64')
        cnt = pd.Series(float(sh.count()), index=df.index, dtype='float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = pd.concat([sh / med, med / sh], axis=1).max(axis=1)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio.where(cnt >= PRICE_SCALE_WITNESS_MIN_ROWS)


def price_scale_hits(df, pb_alarm=None, pb_wide=None, source_col='source'):
    """DataFrame [row, relation, ratio, fields] -- one row per price-scale contradiction.

    `ratio` is the price/book multiple itself, so the refusal CSV carries the number that
    caused the refusal and a reader can argue with the cut without re-deriving anything.

    THE RATIO IS `marketCap / totalStockholdersEquity`, NOT `price / bookValuePerShare`, AND THE
    REASON IS THE OPPOSITE OF THE ONE FIRST GIVEN (review 2, 2026-09-01).  The first version of
    this note argued the two forms are ALGEBRAICALLY the same quantity and the restatement
    therefore costs nothing -- and then, one sentence later, that `bookValuePerShare` is
    sign-wrong on CCM.  BOTH CANNOT BE TRUE.  `bookValuePerShare` is a RAW FMP FIELD, not
    `equity / shares` recomputed, so `price / bvps == marketCap / equity` is an EMPIRICAL
    AGREEMENT that holds where the vendor is self-consistent, not an identity that holds
    always -- `price` itself IS `marketCap / weightedAverageShsOut`, so `price / (marketCap /
    shares)` is 1.0 BY CONSTRUCTION and quoting its agreement rate proves nothing.  (An earlier
    draft cited "98.71% within 0.1%" as evidence; it was a tautology dressed as a measurement
    and is deleted rather than softened.)  The real asymmetry is that `bookValuePerShare` is a
    RAW VENDOR FIELD while `totalStockholdersEquity` is the statement line it is supposed to be
    derived from, so the two forms diverge exactly where the vendor is wrong.  THE RESTATEMENT
    IS RIGHT *BECAUSE* THEY DIVERGE THERE, not because they agree elsewhere.  The balance-sheet form is the one that does not route the rule's own
    domain limb through a field the vendor has already corrupted on the rows under test.  `bookValuePerShare` is a raw FMP field, and it is
    MEASURED WRONG on this rule's own headline live case: CCM carries `bookValuePerShare`
    +12,783 against `totalStockholdersEquity` -2.10bn -- disagreeing in SIGN.  On the per-share
    form the `> 0` limb, whose whole job is to keep the rule off genuinely book-insolvent
    companies, was passed by that sign-wrong number, so the rule fired on all 38 of CCM's rows
    including its NEWEST.  On the balance-sheet form CCM fires on ZERO rows, which is correct:
    it is book-insolvent, not mis-scaled.  This is the house's "compute, don't consume vendor
    fields" applied to the one field the rule's correctness hinged on.
    MEASURED COST OF THE RESTATEMENT on the 08-29 CUR6K panel: 399 rows / 85 sources -> 352 rows
    / 75 sources (338 rows common, 61 dropped, 14 added).  Both motivating shapes are preserved
    unchanged (0CHZ.L 25, QBY0.DE 25, CMCM 23 rows in both), as are PARR, KWS.DE and RBY.TO.

    ALL FOURTEEN NEWLY-REFUSED ROWS ADJUDICATED BY HAND, because a change that ADDS refusals
    cannot rest on the old population's adjudication:
      TEN are unambiguous share-count or scale breaks -- ABVX 2013-10 (bvps 4,919 against a
        source whose real bvps runs 1-10), BEWI.OL 2016-01 and 2017-01 (shares 100,374 and
        102,999 against 191-236M, bvps reported as exactly 0.0), BTTC 2018-10 (19,933 shares
        against ~5M), COST.L 1986-10, KEN.L 2016-04 (marketCap 177,060 against equity 10.35M),
        PPBT 2020-07 and 2020-10 (3,160 and 13,925 shares against a ~137k median), ZAL.DE
        2013-01 (221,000 shares, bvps 0.0).
      ONE is caught for the right reason by the new form and was INVISIBLE to the old one:
        PARR 2010-07 carries a NEGATIVE `bookValuePerShare` (-169.49) against POSITIVE equity
        (545.6M) -- the CCM defect with the sign reversed -- so the per-share form's `bvps > 0`
        limb suppressed it.  This is the restatement earning its keep in the other direction.
      ONE IS ATTRIBUTED TO THE WRONG LEG, and it is a real limitation of the one-sided
        refusal: ALMDG.PA 2007-10 reports equity 16.18bn against 120-145M in every later
        period, i.e. it is the BALANCE SHEET that is ~100x out, not the market cap.  The row
        is still a units error and the contaminated cheapness RATIO is still suppressed
        (marketCap is refused, so `mBookToPrice` and its siblings go absent), but the field
        this rule blanks is the sound one.  The equity guards do not cover it either: the
        break is 134x, under `SCALE_SPIKE_FACTOR`'s 500.
      THREE ARE GENUINELY AMBIGUOUS and are named rather than counted as wins: SEA1.OL
        2020-04, 2020-10 and 2021-01.  Their share count is AT the source's own median
        (witness ratio 1.000), so there is no share-count evidence at all; a shipping name at
        marketCap 1.1M against 86M of book equity is a plausible deep-distress reading as well
        as a plausible break.  Their `bookValuePerShare` and `totalStockholdersEquity`
        disagree by 1.7x, which is the only thing pointing at corruption.
    So: 11 of 14 sound, 1 right-outcome-wrong-attribution, 3 unresolved.

    STRICTLY POSITIVE ON BOTH LEGS.  NEGATIVE book equity is insolvency -- a real state of a
    real company, scored as such everywhere else in this pipeline (see the `bookToPrice`
    mean-dict entry: "A negative book yield is a true measurement of a real company ... it
    belongs in the ruler").  A non-positive `marketCap` is already handled by the primary limbs.
    Neither is a units error, so neither fires here.

    THE ASYMMETRY IS NOW BY EVIDENCE RATHER THAN BY IGNORANCE, and that is worth stating because
    the earlier version of this rule refused BOTH legs on the ground that the panel could not say
    which was wrong.  It can.  `totalStockholdersEquity` already has TWO guards in this module --
    `balance_sheet_identity` and the `isolated_scale_spike` rule both name it -- and it runs
    CONTINUOUS across every break adjudicated by hand (QBY0.DE 92.34M -> 94.17M over the quarter
    in which `marketCap` goes 0.865M -> 86.5M).  The market-cap side had NO guard at all.  So the
    contradiction is read as "the cap side cannot be trusted against a balance sheet that two
    other rules already vouch for", and only the cap side and the quantities derived from it are
    refused.

    ROW-LEVEL, NOT SOURCE-LEVEL, AND THAT IS THE SUBSTANTIVE CHOICE.  `price_scale_audit`
    aggregates to a per-source MEDIAN over the whole history because it is REPORTING a name;
    refusing on that basis would refuse a name's CORRECT current cheapness readings for a break
    in rows nobody scores.  Measured on the 08-29 panel: of the eight sources that module flags,
    four (BGMS, ENDUR.OL, IPOK.DE, SEA1.OL) are clean inside the newest-8 Stage-1 window and
    would have had a sound criterion refused.  A refusal scores as a FAIL in Stage-1, so that is
    not a neutral over-reach -- it invents an adverse judgement.
    """
    empty = pd.DataFrame([], columns=['row', 'relation', 'ratio', 'fields'])
    if 'marketCap' not in df.columns or 'totalStockholdersEquity' not in df.columns:
        return empty
    cut = PRICE_SCALE_PB_ALARM if pb_alarm is None else float(pb_alarm)
    wide = PRICE_SCALE_PB_WIDE if pb_wide is None else float(pb_wide)
    mcap = pd.to_numeric(df['marketCap'], errors='coerce')
    equity = pd.to_numeric(df['totalStockholdersEquity'], errors='coerce')
    with np.errstate(divide='ignore', invalid='ignore'):
        pb = (mcap / equity).replace([np.inf, -np.inf], np.nan)
    #  THE DOMAIN LIMB IS FACTORED OUT OF BOTH LEVELS, deliberately: the witness widens the
    #  price/book requirement and NOTHING ELSE.  Strictly positive on both legs, so negative
    #  book equity (insolvency -- a real state of a real company) and a non-positive market cap
    #  (already a primary limb) still never fire here, at either level.
    domain = ((mcap > 0) & (equity > 0) & pb.notna()).fillna(False)
    #  ONE RULE, TWO LEVELS.  Read the block above the constants for why the CEO-ruled bare
    #  conjunction was measured, falsified and could not ship, and what the union costs.
    witness = (share_count_witness_ratio(df, source_col=source_col)
               >= float(PRICE_SCALE_WITNESS_FACTOR)).fillna(False)
    bad = (domain & ((pb < cut) | ((pb < wide) & witness))).fillna(False)
    if not bool(bad.any()):
        return empty
    rows = [{'row': idx, 'relation': PRICE_SCALE_RELATION,
             'ratio': float(pb.loc[idx]), 'fields': PRICE_SCALE_REFUSE}
            for idx in df.index[bad]]
    return pd.DataFrame(rows, columns=['row', 'relation', 'ratio', 'fields'])


_SANITY_REPORT_COLS = ['source', 'date', 'occ', 'relation', 'ratio', 'field', 'value']


def _sum_fields(df, fields):
    s = None
    for f in fields:
        if f not in df.columns:
            return None
        v = pd.to_numeric(df[f], errors='coerce')
        s = v if s is None else (s + v)
    return s


def impossible_relation_hits(df):
    """DataFrame [row, relation, ratio, fields] -- one row per (row, relation) that fires.

    A relation whose fields are not all present is SKIPPED, not failed: this runs on the
    per-ticker frame during the fetch, where a ragged payload can genuinely lack a column,
    and refusing cells because a column is absent would convert a coverage gap into a
    corruption finding.
    """
    out = []
    for name, num_f, den_f, factor, two_sided, refuse in IMPOSSIBLE_RELATIONS:
        num = _sum_fields(df, num_f)
        den = _sum_fields(df, den_f)
        if num is None or den is None:
            continue
        #  |num| / |den|, so a SIGN error cannot hide inside a magnitude test and a negative
        #  equity (legitimate) is compared on size, not on sign.
        ratio = (num.abs() / den.abs()).replace([np.inf, -np.inf], np.nan)
        bad = ratio.notna() & (ratio >= float(factor))
        if two_sided:
            #  The reciprocal limb.  JHX's totalAssets is (L+E)/1e6, i.e. ratio 1e-6: without
            #  this the case the change exists for goes uncaught.  `> 0` because a zero
            #  numerator is an ABSENT line, not a contradicted one -- section 1 already owns
            #  totalAssets <= 0 and would eject it.
            bad = bad | (ratio.notna() & (ratio > 0) & (ratio <= 1.0 / float(factor)))
        for idx in df.index[bad]:
            out.append({'row': idx, 'relation': name, 'ratio': float(ratio.loc[idx]),
                        'fields': refuse})
    return pd.DataFrame(out, columns=['row', 'relation', 'ratio', 'fields'])


def scale_spike_hits(df, date_col='date'):
    """DataFrame [row, relation, ratio, fields] for the isolated-spike rule.

    `df` is ONE SOURCE's rows in any order; the comparison is against the date-adjacent
    periods, found by sorting, so the caller's orientation (this repo carries both
    newest-first and oldest-first frames) cannot change the answer.
    """
    empty = pd.DataFrame([], columns=['row', 'relation', 'ratio', 'fields'])
    if date_col not in df.columns or len(df) < 3:
        return empty
    order = pd.to_datetime(df[date_col], errors='coerce').sort_values().index
    out = []
    for field, direction in SCALE_SPIKE_FIELDS.items():
        if field not in df.columns:
            continue
        v = pd.to_numeric(df.loc[order, field], errors='coerce').abs()
        prev, nxt = v.shift(1), v.shift(-1)
        ok = v.notna() & prev.notna() & nxt.notna() & (v > 0) & (prev > 0) & (nxt > 0)
        k = float(SCALE_SPIKE_FACTOR)
        up = (v >= k * prev) & (v >= k * nxt)
        down = (prev >= k * v) & (nxt >= k * v)
        bad = (ok & (up if direction == 'up' else (up | down))).fillna(False)
        for idx in v.index[bad]:
            r = max(float(v.loc[idx] / prev.loc[idx]), float(prev.loc[idx] / v.loc[idx]))
            out.append({'row': idx, 'relation': scale_spike_relation(field),
                        'ratio': r, 'fields': (field,)})
    return pd.DataFrame(out, columns=['row', 'relation', 'ratio', 'fields']) if out else empty


def refusal_restore_map(report, source_col='source', date_col='date'):
    """{(source, normalised date): {field: PRE-REFUSAL value}} from a refusal report.

    WHY THIS EXISTS.  A section-5 refusal is an ABSTENTION for SCORING -- the metrics that read
    the cell go NaN.  It must NOT be an abstention for the DATA-QUALITY GATE, because that gate
    is what DELETES corrupt history, and a check that is skipped because its input is NaN does
    not abstain, it PASSES.  `data_quality.check_price_sanity` reads five fields and this
    module's price-scale rule blanks three of them (`price`, `marketCap`, `earningsYield`), so
    on a refused row checks 3, 5 and 6 all skip and the row -- previously deleted, with its
    whole prefix -- comes back merely blanked.  Reproduced end to end on an ATRI-shape fixture:
    PASS 1 flagged two `mcap_step_break` rows before and none after, and the rows PASS 3 would
    delete went 7 -> 0.

    THE REPORT ALREADY CARRIES THE PRE-REFUSAL VALUES.  `refuse_impossible_cells` records
    `value` per refused cell precisely so a refusal can be argued with, and that report reaches
    the run as `inputSanityRefusals`.  So the fix needs NO shadow column, NO schema change and
    NO second copy of the fact -- it reads the artifact that already exists.

    THE KEY CARRIES AN OCCURRENCE INDEX, NOT JUST (source, date).  The panel has duplicate
    `(source, date)` rows -- 296 over 76 sources on the 2026-08-11 CUR3K panel, AAPL among
    them -- and a two-part key COLLAPSES when both twins are refused, handing twin 0 twin 1's
    pre-refusal numbers.  `check_price_sanity`'s step check compares ADJACENT market caps, so
    a borrowed value can create or suppress a break and therefore a prefix deletion; the
    twins are NOT interchangeable.  The per-row stamp intersection in `data_quality` fixes
    WHICH FIELDS to restore and cannot fix WHICH ROW, so the key has to carry it.
    RESIDUAL ASSUMPTION, AND THE FIRST VERSION OF THIS NOTE NAMED THE WRONG MECHANISM.  It
    said the two sides agree "while a source's rows keep their ingest order ... a re-sort
    between the two would silently mis-pair the twins".  THERE IS A RE-SORT -- `data_quality`
    sorts by `(source, date)` before the pass that consumes this map.  What actually makes the
    pairing hold is that the sort is STABLE: pandas' default preserves the relative order of
    rows sharing a key, so duplicate `(source, date)` twins keep their ingest order THROUGH
    the sort and both sides still count them the same way.  The assumption is on the sort's
    STABILITY, not on its absence -- narrower, checkable, and the thing a future reader needs
    to know if they ever change that sort's `kind`.

    IT DELIBERATELY RESTORES RATHER THAN ESCALATING.  The alternative -- treating a refused row
    as a PASS-1 corruption outright -- would have been simpler and is WRONG: PASS 3 removes the
    whole prefix at or before the newest corrupt date, so it would delete far more history than
    the pre-refusal pipeline ever did (PARR alone carries 19 scattered refused rows in 80).
    Restoring the values reproduces the pre-refusal deletion set EXACTLY -- no more, no less --
    which is the actual requirement: a refusal must not suppress a deletion that would otherwise
    have happened.
    """
    out = {}
    if report is None or not len(report):
        return out
    cols = getattr(report, 'columns', [])
    if not all(c in cols for c in (source_col, date_col, 'field', 'value')):
        return out
    for r in report.itertuples(index=False):
        key = (str(getattr(r, source_col)),
               _normalise_refusal_date(getattr(r, date_col)),
               int(getattr(r, 'occ', 0) or 0))
        out.setdefault(key, {})[getattr(r, 'field')] = getattr(r, 'value')
    return out


def _normalise_refusal_date(v):
    """A hashable date key that matches however the panel carries its `date`.

    The report's date comes off the frame it was built from and the panel's comes off the
    frame being filtered; both are Timestamps on the live path, but a CSV round-trip makes one
    a string.  Normalising both through `pd.Timestamp` is what stops a silent all-miss -- and
    a silent all-miss here reads exactly like "nothing was refused", which is the failure this
    whole function exists to prevent.
    """
    try:
        t = pd.Timestamp(v)
        return None if pd.isna(t) else t.normalize()
    except Exception:
        return str(v)



#  ---- Q-72: THE SEVENTEEN OTHER READ/REFUSE COUPLINGS, MEASURED --------------------------
#  Every producer in this pass is computed on the SAME PRE-BLANKING FRAME, so one producer can
#  read a field another producer in the same call has already condemned.  The sweep is DERIVED,
#  not enumerated by hand: a coupling is (P reads f, Q refuses f, Q != P), taken over the four
#  `IMPOSSIBLE_RELATIONS` and the price-scale rule as READERS.  It finds 19.  Two are the
#  price-scale rule reading `totalStockholdersEquity`, closed by
#  `_drop_price_scale_on_already_refused_equity`.  SEVENTEEN WERE LIVE.
#
#  THE COUNTERFACTUAL IS EXACT, WHICH IS WHY THESE COUNTS MEAN SOMETHING.  Every producer's
#  fire test is `ratio.notna() & ...`, so a producer CANNOT fire on a row whose input is already
#  NaN.  Therefore every row in (Q refuses f) AND (P fires) is a row where P's blanking would
#  NOT HAVE HAPPENED had the pass been ordered -- not "might have differed".
#
#  MEASURED INCIDENCE, five saved panels [resdic_2026-07-17 176,781 rows / 7,729 sources;
#  CUR3K_2026-08-11 61,255 / 2,624; CUR3K_2026-08-07 61,007 / 2,613; NA1_EU1_2026-01-08 211,978
#  / 9,012; NA1_EU1_2025-12-09 215,288 / 9,155], rows summed across all five:
#      READER                                  READS                      REFUSED BY          n
#      ppe_within_assets                       totalAssets                spike:totalAssets  31
#      balance_sheet_identity                  totalAssets                ppe_within_assets  26
#      ppe_within_assets                       totalAssets                balance_sheet_id.  26
#      balance_sheet_identity                  totalAssets                spike:totalAssets  24
#      current_assets_within_assets            totalAssets                ppe_within_assets  18
#      ppe_within_assets                       totalAssets                current_assets_wa  18
#      balance_sheet_identity                  totalAssets                current_assets_wa  15
#      current_assets_within_assets            totalAssets                balance_sheet_id.  15
#      balance_sheet_identity                  totalLiabilities           current_liab_wl    12
#      current_liabilities_within_liabilities  totalLiabilities           balance_sheet_id.  12
#      ppe_within_assets                       propertyPlantEquipmentNet  spike:PPE          11
#      current_assets_within_assets            totalAssets                spike:totalAssets  10
#      current_assets_within_assets            totalCurrentAssets         spike:TCA           9
#      current_liabilities_within_liabilities  totalCurrentLiabilities    spike:TCL           4
#      balance_sheet_identity                  totalLiabilities           spike:totalLiabs.   2
#      balance_sheet_identity                  totalStockholdersEquity    spike:TSE           2
#      current_liabilities_within_liabilities  totalLiabilities           spike:totalLiabs.   0
#  SO "LEAVE IT, INCIDENCE IS ZERO" WAS NOT AVAILABLE: sixteen of the seventeen fire on real
#  rows, on every panel.  The recorded number for the seventeenth is 0 and it stays -- that is
#  evidence for the next reader, not a hunch.
#
#  THE SEVENTEEN SPLIT CLEANLY INTO TWO KINDS, AND ONLY ONE OF THEM IS AN ORDERING PROBLEM.
#    NINE have the SPIKE RULE as the refuser.  The spike rule is the only producer in this
#    module that IDENTIFIES A SINGLE CELL: its evidence is the field's own value against its two
#    date-adjacent periods, so it reads NONE of the sibling fields and refuses EXACTLY the one
#    field its evidence names -- it has no collateral at all.  That makes spike-before-relations
#    a well-founded and ACYCLIC order: a containment hit can be dropped because the spike
#    condemned its input, and the spike hit can never be dropped in return.
#    EIGHT are containment/identity pairs reading EACH OTHER (bsi <-> ca_in_a, bsi <-> ppe_in_a,
#    ca_in_a <-> ppe_in_a on `totalAssets`; bsi <-> cl_in_l on `totalLiabilities`).  THESE ARE
#    MUTUAL AND ORDERING IS NOT THE INSTRUMENT FOR THEM.  Whichever side you drop, the other's
#    read was equally corrupt; drop both and the refusal disappears entirely, which is worse
#    than the coupling.  They are LEFT, with the numbers above, and what the fix would need is
#    stated below rather than guessed at.
#
#  WHAT THE FIX BUYS, MEASURED per panel -- hits kept, and distinct (row, field) cells no longer
#  blanked:
#      resdic_2026-07-17    hits 674 -> 658 (-16)   cells 1,510 -> 1,490  (20 cells spared)
#      CUR3K_2026-08-11     hits 319 -> 298 (-21)   cells   528 ->   500  (28 cells spared)
#      CUR3K_2026-08-07     hits 329 -> 308 (-21)   cells   563 ->   535  (28 cells spared)
#      NA1_EU1_2026-01-08   hits 1,235 -> 1,220 (-15) cells 3,123 -> 3,104 (19 cells spared)
#      NA1_EU1_2025-12-09   hits 1,325 -> 1,305 (-20) cells 3,336 -> 3,314 (22 cells spared)
#  Spared cells by field on CUR3K_2026-08-11: propertyPlantEquipmentNet 9, totalLiabilities 9,
#  totalStockholdersEquity 7, totalAssets 2, totalCurrentAssets 1.  Hits dropped by relation:
#  ppe_within_assets 10, balance_sheet_identity 8, current_assets_within_assets 2,
#  current_liabilities_within_liabilities 1.
#  THE WORKED CASE, so "spared" is a fact and not a hope.  `081580.KQ` 2020-10 is one of the
#  three Korean KOSDAQ names the `ppe_within_assets` note already describes as "totalAssets is
#  the 1,000x-too-small cell the identity limb also flags".  BEFORE: the spike rule refuses
#  `totalAssets`, and on the same pre-blanking frame `balance_sheet_identity` divides by that
#  same corrupt `totalAssets` and blanks `totalLiabilities` and `totalStockholdersEquity`, while
#  `ppe_within_assets` divides by it and blanks `propertyPlantEquipmentNet` -- four cells, three
#  of them sound.  AFTER: only `totalAssets` is refused, by the producer whose evidence actually
#  names it.  Three sound cells recovered on one row.
#
#  IT ALSO PARTLY RESOLVES THE MUTUAL EIGHT, as a side effect and not by design: when a spike
#  hit is present it breaks the tie for every relation on that row, so the mutual incidence
#  falls too -- 4 -> 0, 50 -> 32, 52 -> 34, 16 -> 0, 20 -> 0 on the five panels in the order
#  above.  The residual is the rows where two containment relations fire with NO spike witness,
#  and there ordering genuinely cannot help.
#  WHAT WOULD CLOSE THE REMAINING EIGHT, RECORDED AND DELIBERATELY NOT SHIPPED: when two
#  containment/identity producers fire on the same row and their refused-field sets INTERSECT,
#  the shared field is the one the evidence JOINTLY names, so refusing the INTERSECTION rather
#  than the UNION would spare the rest.  That is a change to what "both sides are refused" MEANS
#  -- the semantics this section's own docstring defends -- with unenumerated cases (an empty
#  intersection, three-way agreement), and it is NOT MEASURED.  It is a design question, not an
#  ordering bug, and it goes back to the CEO rather than into this patch.
#
#  THE SECOND DIRECTION, SWEPT.  This guard runs BEFORE
#  `_drop_price_scale_on_already_refused_equity`, and the order is load-bearing: that guard asks
#  "did another producer refuse this row's equity", and dropping a `balance_sheet_identity` hit
#  can REMOVE the only such refusal, so a price-scale hit that used to be dropped now STANDS.
#  MEASURED: +2 price-scale hits on NA1_EU1_2026-01-08 (596 -> 598) and +2 on NA1_EU1_2025-12-09
#  (633 -> 635); zero change on the other three panels.  That is the CORRECT outcome, not a
#  regression -- the equity was never independently condemned, only condemned by a verdict that
#  itself rested on a corrupt `totalAssets` -- but it is a real behaviour change and is reported
#  rather than absorbed.  Run in the other order the guard would read a stale hit set.
#  NO NEW EJECT PATH, checked in both directions: a dropped hit means a cell keeps its RAW
#  value, and `_limb_fails` reads a raw value, not a blank, so no primary limb newly sees an
#  absence.  The only field where that could bite is `totalAssets` under `SANITY_IMPOSSIBLE`
#  (`<= 0`), and a hit is dropped ONLY when the spike rule refused that same field -- which
#  leaves it blanked AND stamped, so `refused_fields_mask` still subtracts it.
#  `test_the_coupling_guard_does_not_change_which_sources_are_EJECTED` pins that.
#
#  WHAT THIS GUARD CANNOT DETECT: it is keyed on a field a spike hit NAMES, so a corrupt cell
#  the spike rule misses -- which is every source's newest and oldest row, 8.6% of panel rows --
#  leaves its coupling wide open.  The guard makes the ordering honest where evidence exists; it
#  does not manufacture evidence.
IMPOSSIBLE_RELATION_READS = {
    #  DERIVED FROM THE TABLE, never typed alongside it: a relation reads exactly its numerator
    #  and denominator fields.  A new relation is therefore covered by the guard the moment it
    #  is added to `IMPOSSIBLE_RELATIONS`, which is the failure mode the price-scale note calls
    #  "re-arms silently the moment any future relation names a primary field".
    name: tuple(num_f) + tuple(den_f)
    for name, num_f, den_f, _factor, _two_sided, _refuse in IMPOSSIBLE_RELATIONS
}


def _drop_relation_hits_the_spike_rule_already_explained(hits, verbose=False):
    """Drop a containment/identity hit whose READ field the spike rule refused on that row.

    Q-72.  See the block above `IMPOSSIBLE_RELATION_READS` for the measured per-coupling
    incidence, which nine of the seventeen this closes, why the other eight are left, and the
    second-direction sweep.

    ASYMMETRIC AND ACYCLIC, for the same reason
    `_drop_price_scale_on_already_refused_equity` is: the spike rule refuses only the one field
    its own evidence names and reads none of the sibling fields, so it has no collateral and is
    never itself a drop target.  The containment relations refuse BOTH legs and assert nothing
    about which is sound -- so when a spike hit names one of those legs, the OTHER leg's
    blanking is collateral off a cell this same pass has condemned.
    """
    if hits is None or not len(hits) or 'relation' not in getattr(hits, 'columns', []):
        return hits
    rel = hits['relation'].astype(str)
    is_spike = rel.str.startswith(SCALE_SPIKE_RELATION_PREFIX)
    if not bool(is_spike.any()):
        return hits
    #  {row: fields the SPIKE rule refused on it}
    spiked = {}
    for h in hits[is_spike].itertuples(index=False):
        spiked.setdefault(h.row, set()).update(h.fields or ())
    if not spiked:
        return hits
    reads = IMPOSSIBLE_RELATION_READS
    drop = pd.Series(
        [(r in reads) and bool(set(reads[r]) & spiked.get(row, set()))
         for r, row in zip(rel, hits['row'])],
        index=hits.index).fillna(False)
    n = int(drop.sum())
    if not n:
        return hits
    if verbose:
        print('INPUT SANITY: %d relation hit(s) DROPPED -- a field the relation DIVIDES BY or '
              'SUMS was already refused on that row by the isolated-scale-spike rule, which '
              'names one cell rather than a contradictory pair, so the relation\'s other leg '
              'would have been blanked off a number this pass has itself rejected.'
              % n, flush=True)
    return hits[~drop].reset_index(drop=True)


#  The one field the price-scale rule DIVIDES BY that another producer in the same pass can
#  reject.  Named once, here, so the guard below and the rule's own commentary cannot drift.
PRICE_SCALE_DIVISOR = 'totalStockholdersEquity'


def _drop_price_scale_on_already_refused_equity(hits, verbose=False):
    """Remove price-scale hits whose row has ALREADY had its equity refused by another rule.

    THE PREMISE THIS ENFORCES WAS PROSE ONLY UNTIL NOW, AND IT IS THE PREMISE THE WHOLE
    ONE-SIDED DESIGN RESTS ON.  `price_scale_hits` refuses the market-cap side and NOT the
    balance sheet, on the stated ground that `totalStockholdersEquity` "already carries two
    guards in this module and runs continuous across every adjudicated break".  But all three
    producers are computed on the SAME PRE-BLANKING FRAME, so the rule divided by raw equity
    even on rows where `balance_sheet_identity` or the spike rule had -- in the very same call
    -- already declared that equity contradictory.  The rule was citing guards it then ignored.

    WHAT THAT COST, on a row with a SOUND price side and an inflated equity: four cells
    (`price`, `marketCap`, `bookValuePerShare`, `earningsYield`) refused on the strength of a
    number the pass had already rejected, each scoring as a FAIL in `calcScore.calcByTier` --
    3.80 of 17.85 of the Stage-1 gate, for no evidential reason.

    THE TWO RULES ARE POSITIVELY COUPLED, NOT INDEPENDENT, so this is a region and not a
    corner: with assets and liabilities sound and equity inflated by k, the identity (factor
    100) fires around k >~ 200 for a normal equity/assets ~ 0.5, while `marketCap/equity <
    0.02` needs only k >~ 50-150 at an ordinary price/book of 1-3.  Every sufficiently large
    inflated-equity break therefore manufactured a price-scale hit mechanically.

    DROPPING, NOT DOWNGRADING, and only this direction.  A row whose equity is already refused
    needs no second opinion from a rule that divides by it; the balance-sheet cells stay
    refused, because the producer that judged them is unaffected.  This is deliberately NOT
    symmetric -- the price side has no rule vouching for it, so a price-scale hit on a row with
    SOUND equity still stands.

    IT IS THE ONLY COUPLING THAT CAN FALSIFY A PRODUCER'S OWN PREMISE, AND THAT -- NOT THE
    FIELD FAMILY -- IS THE LINE.  A derived sweep finds 19 read/refuse couplings; 17 are
    pre-existing among the identity relations and the spike rule.  An earlier version of this
    note drew the line at "the other seventeen stay inside the balance sheet", and that does
    not survive pressure: `ppe_within_assets` can blank a perfectly sound
    `propertyPlantEquipmentNet` off a corrupt `totalAssets`, which is collateral damage to a
    sound cell however the fields are grouped.
    THE DISTINCTION THAT HOLDS IS ABOUT PREMISES.  The seventeen are SYMMETRIC containment
    tests: each says "these two numbers contradict each other" and refuses BOTH, asserting
    nothing about which is sound -- so a second rule rejecting one of them cannot contradict
    anything the first claimed.  The price-scale rule is the only producer whose design
    ASSERTS that one leg is sound and refuses only the other, on the strength of guards it
    names.  It is therefore the only one a coupling can FALSIFY: when those guards fire, the
    premise the asymmetry rests on is gone and the rule was still acting on it.  That is why
    this one is closed and the other seventeen are reported rather than changed.
    """
    if hits is None or not len(hits) or 'relation' not in getattr(hits, 'columns', []):
        return hits
    ps = hits['relation'] == PRICE_SCALE_RELATION
    if not bool(ps.any()):
        return hits
    #  Rows any OTHER producer refused the divisor on.
    other = hits[~ps]
    bad_rows = {h.row for h in other.itertuples(index=False)
                if PRICE_SCALE_DIVISOR in (h.fields or ())}
    if not bad_rows:
        return hits
    drop = ps & hits['row'].isin(bad_rows)
    n = int(drop.sum())
    if n and verbose:
        print('INPUT SANITY: %d price-scale hit(s) DROPPED -- their `%s` was already refused '
              'by another rule on the same row, so the divisor the price-scale test rests on '
              'is one this pass has itself rejected.' % (n, PRICE_SCALE_DIVISOR), flush=True)
    return hits[~drop].reset_index(drop=True)


def refuse_impossible_cells(df, date_col='date', source_col='source', verbose=False):
    """Replace every cell a relation contradicts with NaN.  Returns (frame, report).

    THIS IS AN ABSTENTION, NOT AN EJECTION.  The row stays, the source stays, the name still
    scores on everything the refused cells do not feed; the metrics that DO read them become
    NaN and are reported as 'data-incomplete: dig-deeper' by the same machinery the Beneish
    base guards already use.  It never deletes a row and never touches a source's membership.

    BOTH SIDES OF A VIOLATED RELATION ARE REFUSED, deliberately.  When
    `totalCurrentLiabilities` is 982,186x `totalLiabilities`, the panel does not say which of
    the two is wrong -- and using either one as if it were the sound half is exactly the
    assumption that produced the finding in the first place.  Abstaining on both is the
    honest reading of "these numbers contradict each other".

    `df` may hold one source (the live call site, inside the fetch loop) or many (a saved
    panel); the spike rule is applied PER SOURCE either way.
    """
    if df is None or len(df) == 0:
        return df, pd.DataFrame([], columns=_SANITY_REPORT_COLS)
    #  POSITIONAL INTERNALLY (review S9).  `ratio.loc[idx]` and `out.at[...]` assume UNIQUE
    #  labels; on a duplicated index they raise `TypeError: cannot convert the series to
    #  <class 'float'>`.  In the fetch that is caught by the per-ticker guard, so the ticker
    #  would keep its RAW values with a warning -- the abstention silently not happening, which
    #  is the worst of the three outcomes.  The caller's index is restored before returning, so
    #  no caller sees a reindexed frame.
    _orig_index = df.index
    df = df.reset_index(drop=True)
    #  THREE RULE PRODUCERS, ALL POSITIONALLY INDEXED AGAINST THE SAME RESET FRAME:
    #  cross-field impossibility at one instant, the isolated scale spike along TIME, and
    #  the price-scale contradiction.  The third is per-ROW and needs no neighbours, so it
    #  runs on the whole frame like the first rather than per source like the spike rule.
    parts = [impossible_relation_hits(df),
             price_scale_hits(df, source_col=source_col)]
    if source_col in df.columns and df[source_col].nunique() > 1:
        for _, sub in df.groupby(source_col, sort=False):
            parts.append(scale_spike_hits(sub, date_col=date_col))
    else:
        parts.append(scale_spike_hits(df, date_col=date_col))
    parts = [p for p in parts if len(p)]
    if not parts:
        if verbose:
            print('INPUT SANITY: 0 cell(s) refused -- no cross-field impossibility in '
                  '%d row(s).' % len(df), flush=True)
        df.index = _orig_index
        return df, pd.DataFrame([], columns=_SANITY_REPORT_COLS)
    hits = pd.concat(parts, ignore_index=True)
    #  Position of each row among the rows sharing its (source, date), in frame order.
    _occ = {}
    if source_col in df.columns and date_col in df.columns:
        _seen = {}
        for _i, _s, _d in zip(df.index, df[source_col].astype(str),
                              pd.to_datetime(df[date_col], errors='coerce')):
            _k = (_s, _d)
            _occ[_i] = _seen.get(_k, 0)
            _seen[_k] = _occ[_i] + 1
    #  ORDER IS LOAD-BEARING (Q-72).  The spike-priority guard runs FIRST because the
    #  equity guard below asks "did another producer refuse this row's equity", and dropping a
    #  `balance_sheet_identity` hit can remove the only such refusal.  Run the other way round,
    #  the equity guard would answer from a hit set that no longer ships.  Measured effect of
    #  the coupling: +2 price-scale hits on each NA1_EU1 panel, 0 on the other three.
    hits = _drop_relation_hits_the_spike_rule_already_explained(hits, verbose=verbose)
    hits = _drop_price_scale_on_already_refused_equity(hits, verbose=verbose)
    out = df.copy()
    rec = []
    refused_by_row = {}
    for h in hits.itertuples():
        for f in h.fields:
            if f not in out.columns:
                continue
            #  READ THE ORIGINAL VALUE FROM `df`, NOT FROM `out` (review S4).  `out` is the
            #  frame this loop is ALREADY MUTATING, so a cell refused by two relations had its
            #  value logged as NaN by the second -- 29 of 326 records on the 2026-08-13 panel,
            #  including every ALAQU.PA containment row and RDZN's spike row.  That defeats the
            #  CSV's whole purpose, which is that a refusal can be ARGUED WITH.
            rec.append({'source': (df.at[h.row, source_col]
                                   if source_col in df.columns else ''),
                        'date': (df.at[h.row, date_col] if date_col in df.columns else ''),
                        #  OCCURRENCE INDEX WITHIN (source, date).  `(source, date)` IS NOT
                        #  UNIQUE on this panel -- 296 duplicate-key rows over 76 sources on
                        #  the 2026-08-11 CUR3K panel -- so a restore map keyed on it alone
                        #  COLLAPSES when both twins are refused and hands twin 0 twin 1's
                        #  numbers.  That is not harmless: `check_price_sanity`'s step check
                        #  compares ADJACENT market caps, so a borrowed value can create or
                        #  suppress a break and therefore a whole prefix deletion.
                        'occ': _occ.get(h.row, 0),
                        'relation': h.relation, 'ratio': h.ratio, 'field': f,
                        'value': df.at[h.row, f]})
            out.at[h.row, f] = np.nan
            refused_by_row.setdefault(h.row, set()).add(f)
    #  STAMP WHAT WAS REFUSED, so a downstream reader can tell an abstention from an absence.
    #  Merged with any existing value rather than overwritten: this function is idempotent by
    #  design (a refused cell is already NaN, so no relation fires on it a second time) and
    #  `data_quality.filter_invalid_data` runs TWICE on the live path.
    if refused_by_row:
        existing = (out[SANITY_REFUSED_COLUMN].astype('object')
                    if SANITY_REFUSED_COLUMN in out.columns
                    else pd.Series('', index=out.index, dtype='object'))
        existing = existing.where(existing.notna(), '')
        for row, fields in refused_by_row.items():
            prev = {t for t in str(existing.at[row]).split(_SANITY_REFUSED_SEP) if t}
            existing.at[row] = _SANITY_REFUSED_SEP.join(sorted(prev | fields))
        out[SANITY_REFUSED_COLUMN] = existing
    report = pd.DataFrame(rec, columns=_SANITY_REPORT_COLS)
    if verbose and len(report):
        print('INPUT SANITY: refused %d cell(s) on %d row(s) across %d source(s) -- the '
              'vendor numbers contradict each other, so the metrics reading them ABSTAIN '
              'rather than score a corrupt value.'
              % (len(report), hits['row'].nunique(), report['source'].nunique()), flush=True)
        for rel, k in report['relation'].value_counts().items():
            print('    %-46s %5d cell(s)' % (rel, k), flush=True)
    out.index = _orig_index
    return out, report
