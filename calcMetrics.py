import pandas as pd
import numpy as np

import reporting_period as rp

# `rpy` = this source's rows per year (4 quarterly / 2 semi-annual, reporting_period).
# It scales the moving-average window so it spans the same CALENDAR time, and it is the
# divisor for any ANNUAL rate expressed per period.  rpy defaults to 4 -> unchanged.

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


def calc_special(df,metstr,n,rpy=rp.DEFAULT_ROWS_PER_YEAR):
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

    return res