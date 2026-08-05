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
    d = df[[source_col, date_col] + fields].copy()
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
