import pandas as pd
import numpy as np

import reporting_period as rp

# `rpy` = this source's rows per year (4 quarterly / 2 semi-annual, reporting_period).
# It scales the moving-average window so it spans the same CALENDAR time, and it is the
# divisor for any ANNUAL rate expressed per period.  rpy defaults to 4 -> unchanged.

# =========================================================================== #
#  STAGE-1 DOMAIN GUARDS (sign-inversion fix, 2026-08-04)                      #
# =========================================================================== #
# A ratio whose ADVERSE quantity sits in the DENOMINATOR does not fail -- it INVERTS SIGN and
# scores as the best possible value.  Where the ratio can be rewritten in yield form that is
# the better fix and no guard is needed (see createDicts's module comment); where it cannot,
# the out-of-domain rows are REFUSED here.
#
# A guard is a predicate on the RAW statement frame returning the ADMISSIBLE rows.  Refused
# rows become NaN, and `calcScore.calcByTier` already scores NaN as a fail (`Sign * NaN > 0`
# is False), so no new scoring state is introduced -- which is the reason this is a mask and
# not a new branch in the scorer.
#
# WHY REFUSAL RATHER THAN A TRANSFORM OF THE PERVERSE VALUE.  Three candidates were
# considered -- negate the value, impute the column's floor, or refuse:
#   * NEGATION IS NOT SIGN-SAFE.  It flips the value across zero, which lands on the FAIL side
#     for a Sign +1 (higher-is-better) criterion but on the PASS side for a Sign -1 one.
#     MEASURED on the head(8) window: negating `returnOnEquity`'s 2,426 double-negative rows
#     takes them from 2,255 passes to 2 (correct), while negating `uNetDebtToEBITDA`'s 9,200
#     takes them from 5,280 passes to 9,200 -- i.e. it makes every one of them PASS.  A single
#     blanket transform cannot serve both signs.
#   * A BOUNDARY/FLOOR IMPUTATION DOES NOT APPLY.  The project's boundary rule imputes the
#     LIMIT of the metric as the input approaches its domain edge, and admits it only where
#     that limit is FINITE.  For every ratio here the limit is +/-infinity as the denominator
#     -> 0, so that rule's own escape clause applies: REFUSE, do not impute.
#   * AND AT STAGE-1 THE CHOICE IS MOOT ANYWAY.  calcByTier returns `w if pass else 0` -- a
#     BINARY outcome, with no ranking for a magnitude to inform.  Any treatment that lands on
#     the fail side is behaviour-IDENTICAL.  So refusal is the cheapest correct option and adds
#     no tuned constant.  (The choice would only become live in Stage-2, which RANKS; the one
#     Stage-2 metric in this family is `returnOnEquity` -- see stage2_metrics.postbm_metric.)
#
# A NaN in a guard's own input makes the row INADMISSIBLE: an undetermined domain is not a
# licence to score.  That is `fillna(False)` in apply_domain_guard, not an accident.
STAGE1_DOMAIN_GUARDS = {
    # Book equity must be POSITIVE for any ratio that divides by it, or leverage and return
    # ratios inverse-scale and change sign.  `> 0` and not `>= 0`: zero equity is division by
    # zero, and the pre-existing ruling on it (createDicts freeCashFlowToEquity) is that NaN
    # describes a technically-insolvent balance sheet better than either infinity.
    'equity_positive':
        lambda df: pd.to_numeric(df['totalStockholdersEquity'], errors='coerce') > 0,
    # EBITDA PROXY = operatingIncome + depreciationAndAmortization.  FMP publishes
    # `netDebtToEBITDA` but not the EBITDA behind it, so the guard reconstructs the sign from
    # the two statement lines it does publish.  A PROXY, stated as one: near zero the proxy and
    # the true EBITDA can straddle, so individual rows at the boundary may be classified
    # differently than FMP would.  The guard's DIRECTION does not depend on the proxy.
    'ebitda_positive':
        lambda df: (pd.to_numeric(df['operatingIncome'], errors='coerce')
                    + pd.to_numeric(df['depreciationAndAmortization'], errors='coerce')) > 0,
    # A tax RATE's admissible domain is >= 0.  Negative means the denominator (pre-tax income)
    # was non-positive or the expense was a credit; either way "tax efficiency" is not what the
    # number is measuring.  `>= 0` and not `> 0`: a zero rate is a real, defined answer (no tax
    # paid on positive income), and 9,326 head(8) rows are exactly zero -- refusing those would
    # be a much larger and unrelated change.
    'tax_rate_nonnegative':
        lambda df: pd.to_numeric(df['effectiveTaxRate'], errors='coerce') >= 0,
    # PEG needs BOTH periods of its growth leg -- see _peg_growth_defined.
    'peg_growth_defined': lambda df: _peg_growth_defined(df),
}


#  EPS BASIS FOR THE PEG GUARD.  The vendor's PEG is built from `eps` (income available to
#  COMMON); `netIncomePerShare` is a near-perfect proxy but NOT identical -- measured sign
#  agreement 92.8%, median absolute error 2.5%.  `eps` / `epsdiluted` are now captured at
#  ingest (createDicts.preReq_dict 'inc') but are ABSENT from every saved panel, so the guard
#  reads the PROXY and says so.  SWITCHING TO `eps` MUST BE A DELIBERATE EDIT, not an
#  `eps if present else proxy` fallback: a silent basis change on the first fetch that carries
#  the column would move the guard's boundary with nothing in the run to say it had.
_PEG_EPS_FIELD = 'netIncomePerShare'


def _peg_growth_defined(df):
    """Rows where PEG's growth leg is DEFINED: `eps_t > 0` AND `eps_{t-1} > 0`.

    THE VENDOR'S FORMULA, established arithmetically (no vendor docs exist), matched to all
    printed digits on nine deliberately-seasonal quarters:

        PEG = [ price / (4 * eps_t) ]  /  [ 100 * (eps_t / eps_{t-1} - 1) ]

    So BOTH legs are SINGLE-PERIOD (the PE leg is one quarter annualised x4, not a trailing
    sum) and the growth leg is SEQUENTIAL quarter-over-quarter.  Both operands of that growth
    ratio can cross zero, which gives PEG FOUR sign states and only ONE of them defined.
    MEASURED on the 61,472-row head(8) window of the 7,729-source panel:

        eps_t > 0, eps_{t-1} > 0   34,398 rows  pass 0.3671   DEFINED -- the real criterion
        eps_t < 0, eps_{t-1} > 0    5,129 rows  pass 0.8830   FALSE PASS: PE < 0 and growth < 0
                                                              cancel into a positive PEG
        eps_t < 0, eps_{t-1} < 0   16,902 rows  pass 0.4218   FALSE PASS: same cancellation
        eps_t > 0, eps_{t-1} < 0    5,040 rows  pass 0.0006   TURNAROUND: the growth ratio
                                                              flips sign, PEG < 0, and the
                                                              criterion FAILS the company

    WHY THE GUARD IS TWO-SIDED AND NOT JUST `eps_t > 0`.  One-sided would state a domain that
    is not the metric's: PEG is a ratio to a GROWTH RATE, and a growth rate computed across a
    sign change is not a growth rate.  A turnaround is undefined, not bad.
    BUT BE PRECISE ABOUT WHAT THAT BUYS, because the honest number is small: at Stage-1 a
    refused row is scored by `calcByTier` as `w if Sign*NaN > 0 else 0` = 0, i.e. STILL A FAIL.
    So the two-sided guard does NOT convert the 5,037 wrongly-failed turnaround rows into
    passes -- it makes the REASON honest and removes the 3 that were passing on the sign flip.
    Measured difference between the two guard shapes: 11,664 vs 11,661 passes removed, i.e. 3.
    THE TURNAROUND DEFECT IS THEREFORE RECORDED, NOT FIXED HERE.  Fixing it needs either a
    change to Stage-1's NaN-is-a-fail rule (deliberately out of scope) or a PEG computed
    locally on a basis that is defined across a sign change.  Do not read this guard as having
    recovered those rows.

    ROW ORDER IS LOAD-BEARING: `df` is ONE source, NEWEST-FIRST, so `shift(-1)` is one period
    OLDER -- the same convention `calc_diff` uses.  The oldest row's predecessor is NaN and
    therefore inadmissible, which is correct (no prior period, no growth rate) and costs
    nothing: `build_bometric_rows` trims the oldest `rpy` rows anyway.
    """
    e = pd.to_numeric(df[_PEG_EPS_FIELD], errors='coerce')
    return (e > 0) & (e.shift(-1) > 0)


def apply_domain_guard(df, values, guard):
    """`values` with every row the named guard rejects replaced by NaN.

    `df` is the RAW statement frame the ratio was built from (positional index, newest-first);
    `values` is the ratio as `calc_simpleRatio` returns it -- a list, positionally aligned with
    `df`.  Returns a list, so the caller's downstream handling is unchanged.

    ORDER MATTERS AND IS THE CALLER'S RESPONSIBILITY: the guard must be applied to the LEVEL
    BEFORE `calc_diff` takes the difference.  Guarding after the diff would leave a diff
    computed ACROSS an out-of-domain row -- a change measured against a meaningless base.
    Guarding before makes the diff NaN whenever EITHER leg is inadmissible, which is the
    honest reading: there is no defined change in a quantity that was undefined.
    """
    if guard not in STAGE1_DOMAIN_GUARDS:
        raise KeyError(
            "calcMetrics.apply_domain_guard: no guard named %r (known: %r). A `Guard` key was "
            "added to a createDicts metric entry without a predicate here -- add it, or drop "
            "the key. Silently ignoring an unknown guard would leave the criterion scoring "
            "its perverse rows with nothing to say so." % (guard, sorted(STAGE1_DOMAIN_GUARDS)))
    admissible = STAGE1_DOMAIN_GUARDS[guard](df)
    v = pd.to_numeric(pd.Series(list(values)), errors='coerce').to_numpy(dtype='float64')
    ok = np.asarray(admissible.fillna(False), dtype=bool)
    if len(ok) != len(v):
        raise ValueError(
            "calcMetrics.apply_domain_guard: guard %r produced %d flags for %d values. The "
            "guard reads the raw frame and the values are positionally aligned to it, so a "
            "length mismatch means the caller passed a different frame than the ratio was "
            "built from." % (guard, len(ok), len(v)))
    return np.where(ok, v, np.nan).tolist()


#maybe check if denom is 0?
def calc_simpleRatio(df,strUp,strDn):
#    res = pd.DataFrame()
#    res[resultString] = df[strUp] / df[strDn]
    if strDn == 'Identity':
        tmpres = df[strUp]
    else:
        tmpres = df[strUp]/df[strDn]

    return tmpres.tolist()
#    return res

def calc_compRatio(df,strUp,strDn,metstr,n,rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """DEAD CODE -- NO CALLER, AND IT WOULD RAISE IF THERE WERE ONE (flagged 2026-08-02).

    Repo-wide there is no call site: Stage-1 builds its rolling means via
    calc_simpleRatio + calc_diff (getData_fmp.build_bometric_rows).  And the final line is
    `res.tolist()` on a DataFrame, which has no `.tolist()` -- so the first caller would get
    an AttributeError, not a metric.  NOT removed here: deletion is outside the mandate of the
    pass that found it (behaviour-preserving refactor), and it is harmless while unreachable.
    Do not adopt it without fixing the return.
    """
    res = pd.DataFrame()
    if strDn == 'Identity':
        res[metstr] = df[strUp]
    else:
        res[metstr] = df[strUp] / df[strDn]

    res = res.iloc[::-1]
    res[metstr] = res[metstr].rolling(rp.scale_window(n, rpy)).mean()
    res = res.iloc[::-1]

    return res.tolist()

def calc_diff(df,metstr,n,rpy=rp.DEFAULT_ROWS_PER_YEAR):
    res = pd.DataFrame()

    dstr = "d" + metstr[0].upper() + metstr[1:]
    # shift(-1) is ONE REPORTING PERIOD, already frequency-relative -- it needs no
    # rescaling.  Only the smoothing window does, so the average covers the same
    # calendar span for a semi-annual filer as for a quarterly one.
    res[dstr] = df[metstr] - df[metstr].shift(-1)
    res = res.iloc[::-1]
    res[dstr] = res[dstr].rolling(rp.scale_window(n, rpy)).mean()
    res = res.iloc[::-1]

    return res

#  Stage-1 keys this function knows how to compute.  It MUST stay in step with
#  createDicts.getDicts()'s BoMetric_special_dict, which is the dict the only caller
#  (getData_fmp.build_bometric_rows) iterates.
#
#  WHY IT IS ENFORCED RATHER THAN TRUSTED (2026-08-02).  Before this, an unrecognised
#  `metstr` fell through every branch and returned an EMPTY DataFrame, which the caller
#  assigned straight into `tempMetric_df[key1]` -- so a metric added to the dict without a
#  branch here became an all-NaN column that then scored as pool-neutral, silently.  That is
#  the same silent-default shape as an unregistered Stage-2 metric, and it is closed the same
#  way: fail loudly, naming the key.
_SPECIAL_KEYS = ('CFOlessEarnings', 'PEG', 'returnOnEquity',
                 'capitalExpenditureCoverageRatio')


def calc_special(df,metstr,n,rpy=rp.DEFAULT_ROWS_PER_YEAR,guard=None):
    """`guard`: optional STAGE1_DOMAIN_GUARDS name, applied to the computed column.

    The special criteria are FORMULAS rather than Upper/Lower ratio specs, so they never pass
    through build_bometric_rows's ratio loop where the declared `Guard` is applied.  The caller
    forwards it here instead, and the guard is applied to the FINISHED column -- which is the
    same point in the computation (these formulas have no diff stage of their own)."""
    if metstr not in _SPECIAL_KEYS:
        raise KeyError(
            "calcMetrics.calc_special has no formula for %r (known: %r). It used to return an "
            "EMPTY frame for an unknown key, which the caller wrote into BoMetric_df as an "
            "all-NaN column that scored as pool-neutral -- add the branch, or remove the key "
            "from createDicts.BoMetric_special_dict." % (metstr, list(_SPECIAL_KEYS)))
    res = pd.DataFrame()
    #if str == 'dInvPEG':
    #    Fix. Needs to be Earnings per share. And needs to be higher than unity (annualized)
    #    temp = pd.DataFrame()
    #    temp['ep'] = df['netIncome']/df['price']
    #    temp['dep'] = temp['ep'] - temp['ep'].shift(-1)
    #    temp['de'] = df['netIncome'] - df['netIncome'].shift(-1)
    #    res[str] = temp['dep']*temp['de']
    #    res = res.iloc[::-1]
    #    res[str] = res[str].rolling(n).mean()
    #    res = res.iloc[::-1]
    if metstr == 'CFOlessEarnings':
        # Sign-SAFE accrual test: CFO - netIncome (a DIFFERENCE, so no sign-flipping
        # denominator).  Replaces the uIncomeQuality = CFO/NI > 1 unity test -- see
        # createDicts.BoMetric_special_dict for the measured inversion it fixes.
        # No window/rpy scaling: both legs are the SAME period's flow, so the ratio of
        # their difference to zero is frequency-invariant (a sign test on a difference of
        # two same-period flows). Sign +1, tested as `metvec > 0`.
        res[metstr] = (pd.to_numeric(df['netCashProvidedByOperatingActivities'],
                                     errors='coerce')
                       - pd.to_numeric(df['netIncome'], errors='coerce'))
    elif metstr == 'PEG':
        res[metstr] = np.where(df['priceEarningsToGrowthRatio'] != 0, 1 / df['priceEarningsToGrowthRatio'], 0) - 1
    #elif str == 'CFOlessEarnings':
    #    res[metstr] = df['netCashProvidedByOperatingActivities'] - df['netIncome']
    #    res = res.iloc[::-1]
    #    res[metstr] = res[metstr].rolling(n).mean()
    #    res = res.iloc[::-1]
    elif metstr == 'returnOnEquity':
        #res[metstr] = df['netIncome']/df['totalStockholdersEquity'] - 0.12/4
        # 0.12 is an ANNUAL 12% ROE hurdle spread over the periods in a year, so the
        # divisor is rows-per-year -- NOT a fixed 4.  A semi-annual filer's row covers
        # six months and was being compared against a 3-month hurdle, i.e. a bar half
        # as high as intended.  (Row-based site NOT on the audit's list -- found in the
        # 2026-07-25 sweep.)
        #
        # THE `equity_positive` GUARD (declared in createDicts.BoMetric_special_dict) is what
        # stops a NEGATIVE-equity, NEGATIVE-income company clearing this 12% hurdle on a
        # positive ROE built from two negatives.  It is applied below, from the declaration --
        # NOT hard-coded here -- so the domain condition sits beside the metric.
        res[metstr] = df['returnOnEquity'] - 0.12/float(rpy)
    # An 'EPStoEPSmean' branch used to sit here and was UNREACHABLE: 'EPStoEPSmean' is a
    # STAGE-2 metric key (createDicts postNewRankingDict, computed by
    # stage2_metrics.eps_to_eps_mean) and has never been in BoMetric_special_dict, which is
    # the only dict this function is called with. Removed 2026-08-02 -- verified against
    # getDicts(): the special dict holds exactly _SPECIAL_KEYS. It also computed a DIFFERENT
    # quantity from the Stage-2 metric of the same name (a raw EPS-minus-mean level, not the
    # dimensionless deviation), so leaving it in place invited exactly the Stage-1/Stage-2
    # same-name-different-basis confusion that the accruals divergence came from. Nothing
    # referenced it, so no stored artifact changes.
    elif metstr == 'capitalExpenditureCoverageRatio':
        tempce2cr = df[metstr]
        ce2cr = -tempce2cr.fillna(0)
        res[metstr] = ce2cr - 2

    if guard is not None:
        res[metstr] = apply_domain_guard(df, res[metstr].tolist(), guard)

    return res