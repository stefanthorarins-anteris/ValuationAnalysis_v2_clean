## Description: we read preReq_dict and put into a new dataframe, relVars. We read calcFromfs_dict and for each key we
#               lookup upper and lower and calculate the ratio at each time, adding the result into a new dataframe we
#               call BoPrep_df. If it is in the diff dict we calculate the 4 quarter
#               difference as well. If it is in the mean, we add the sum of the ratio to a dataframe which
#               continuously holds the total sum of that ratio, as well as updating a counter of tickers processed. We
#               also update the mean. We then loop over the "provided" dictionaries and add them to BoPref_df

from collections import defaultdict
## Dict for fundamental calculation
# Sales to inventory should probably be S' - I' > 0, not (S/I)' > 0

import macroConditions as mcond
import scoringWeights as sw          # SINGLE SOURCE OF TRUTH for every scoring weight


# --- KNOWN DUPLICATE STAGE-1 CRITERION (made explicit, 2026-08-02) ------------
# `EPS` and `netIncomePerShare` in BoMetric_diff_dict are BYTE-IDENTICAL criterion
# specs -- same Upper ('netIncomePerShare'), same Lower ('Identity'), same Sign (+1) --
# differing ONLY in Tier: 'B' (w = 0.5) vs 'N' (w = 0).  So the panel carries two
# byte-identical columns, dEPS and dNetIncomePerShare (verified `.equals()` True on the
# 2026-07-17 panel, 148,172 rows, zero NaN mismatches), and the Tier-N one is scored at
# weight 0.
#
# MEASURED, not assumed (2026-08-02): calcScore.calcByTier maps any tier outside
# S/A/B/C/D to w = 0, so the criterion returns exactly 0.0 for every name (max = min =
# 0.0 over 400 names, no NaN), and deleting the registry entry outright leaves Stage-1
# BITWISE identical on the saved panel (identical source order, max |score diff| = 0.0).
# So Tier N genuinely contributes nothing TODAY.
#
# IT IS STILL NOT DELETED, deliberately.  A registry entry is cheap; the risk is one
# tier edit away.  If a future re-weighting promotes `netIncomePerShare` off Tier N, the
# SAME criterion would silently score TWICE -- 0.5 + its new weight on one quantity --
# and nothing in the pipeline would say so.  Keeping the entry and declaring the pair
# here means `test_scoring_weights_single_source.py` fails the moment both carry weight,
# which is a decision prompt rather than a silent double-count.
#
# Removing one entry is ALSO not free of side effects, which is the other reason to
# leave it: it drops a column from BoMetric_df (so an older saved panel and a newer one
# no longer have the same schema, and calcScore's schema gate is column-EXACT on what it
# needs), and it changes the Stage-1 NaN-accounting readout's per-name criterion COUNT
# and name list (the `_nan_acct` readout in calcScore.simpleScore_fromDict, fed by
# calcByTier's `nan_sink`) even though the summed tier weight it reports is
# unaffected.  Neither touches a score, but both are visible output.
#
# Each entry is (BoMetric_diff_dict key that CARRIES the weight, the inert TWIN).
DUPLICATE_DIFF_CRITERIA = (('EPS', 'netIncomePerShare'),)


def getDicts():
    # sub dicts
    # Possible entries later:
    #   'netOperatingAssets': {'Upper': '?', 'Lower': '?', 'Tier': 'B'}
    #   'dSalesToEmployees': {'Upper': 'revenue', 'Lower': 'Employees', 'Tier': 'B'},
    #   Defensive Internal Ratio

    preReq_dict = {'bs': ['totalAssets', 'longTermDebt', 'inventory', 'totalStockholdersEquity', 'totalLiabilities',
                          'totalCurrentAssets', 'totalCurrentLiabilities','propertyPlantEquipmentNet', 'otherCurrentAssets'],
                   'inc': ['netIncome', 'grossProfit', 'revenue', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'depreciationAndAmortization',
                           'sellingGeneralAndAdministrativeExpenses', 'operatingIncome','interestExpense',
                           # reportedCurrency: the statement's reporting currency (USD/SEK/EUR/...).
                           # Captured (was discarded at ingest) so marketCap -- stored in this same
                           # reporting currency -- can be converted to USD for market-cap banding
                           # (carveOut.marketcap_usd_series). A string column; rides through unused by
                           # every ratio calc. Populates on the next full fetch; absent on saved pickles.
                           'reportedCurrency',
                           # --- REPORTING-PERIOD PROVENANCE (audit C-1 / H-2, 2026-07-19) ---
                           # All four are ALREADY in the paid v3/income-statement response and were
                           # simply discarded at ingest.  Like reportedCurrency they ride through
                           # every ratio calculation unused (they never reach BoMetric_df, whose
                           # columns come from the metric dicts, not from preReq_dict) and populate
                           # from the NEXT full fetch; they are absent on saved pickles.
                           #
                           # period       'Q1'..'Q4' / 'FY'.  THE missing field behind the biggest
                           #              open data defect: FMP labels a SEMI-ANNUAL filer's H1/H2 as
                           #              Q2/Q4 carrying true 6-MONTH flows, and with no period field
                           #              nothing downstream can tell a 3-month flow from a 6-month
                           #              one -- so those names are scored on ~2x flows against
                           #              quarterly peers, and iloc[4] is 2 years back for them (see
                           #              stage2_metrics.piotroski).  Capturing it is the
                           #              precondition for annualising-or-excluding them.
                           # calendarYear FMP's FISCAL-year label -- NOT the calendar year of the
                           #              period-end date.  Verified: TAM.L's 2026-03-31 row carries
                           #              calendarYear=2025.  Never use it as a calendar anchor or to
                           #              derive a date; it disambiguates 52/53-week drift only.
                           # fillingDate  filing date.  Real for SEC filers (30-51d after period end)
                           #              but a PLACEHOLDER equal to the period end for ~50% of rows
                           #              (mostly non-US), so a point-in-time slice must use it only
                           #              where acceptedDate > period end and fall back to a fixed lag
                           #              otherwise -- do not treat it as availability truth blindly.
                           # acceptedDate timestamp the filing was accepted; the discriminator for the
                           #              fillingDate-placeholder test above.
                           'period', 'calendarYear', 'fillingDate', 'acceptedDate'],
                   'cf': ['freeCashFlow', 'netCashProvidedByOperatingActivities','netCashUsedProvidedByFinancingActivities',
                          'dividendsPaid'],
                   'km': ['netIncomePerShare', 'pbRatio', 'earningsYield', 'pfcfRatio', 'grahamNumber', 'grahamNetNet',
                          'marketCap', 'returnOnTangibleAssets', 'incomeQuality', 'bookValuePerShare', 'netDebtToEBITDA',
                          'daysSalesOutstanding', 'capexPerShare', 'tangibleBookValuePerShare',
                          'dividendYield', 'payoutRatio'],
                   'fr': ['returnOnEquity', 'debtEquityRatio', 'currentRatio', 'grossProfitMargin','netProfitMargin',
                          'effectiveTaxRate', 'returnOnCapitalEmployed', 'returnOnAssets', 'priceEarningsToGrowthRatio',
                          'daysOfInventoryOutstanding','capitalExpenditureCoverageRatio'],
                   'other': ['price']
                   }


    #n is > 0; d is difference > 0; m is larger than the mean; u is larger than unity
    BoMetric_Calc_dict =   {'currentRatio':                 {'Operation': ['u', 'd'],   'Sign': 1},
                            'returnOnAssets':               {'Operation': ['n','d'],    'Sign': 1},
                            'debtEquityRatio':              {'Operation': ['m'],        'Sign': -1},
                            'effectiveTaxRate':             {'Operation': ['d'],        'Sign': -1},
                            'returnOnCapitalEmployed':      {'Operation': ['d'],        'Sign': 1},
                            'grossProfitMargin':            {'Operation': ['d','m'],    'Sign': 1},
                            'pfcfRatio':                    {'Operation': ['m'],        'Sign': -1},
                            'earningsYield':                {'Operation': ['m'],        'Sign': 1},
                            'returnOnTangibleAssets':       {'Operation': ['d'],        'Sign': 1},
                            'pbRatio':                      {'Operation': ['m', 'd'],   'Sign': -1},
                            'grahamNetNet':                 {'Operation': ['n'],        'Sign': 1},
                            'netIncomePerShare':            {'Operation': ['d'],        'Sign': 1},
                            'grossProfitToAssets':          {'Operation': ['d'],        'Sign': 1},
                            'grahamNumberToPrice':          {'Operation': ['u'],        'Sign': 1},
                            'salesToAssets':                {'Operation': ['d'],        'Sign': 1},
                            # INVERTED to longTermDebt/totalAssets (ruling Q1.2, 2026-07-26) --
                            # see the BoMetric_diff_dict entry for the full reasoning.
                            'assetsToLongTermLiabilities':  {'Operation': ['d'],        'Sign': -1},
                            'salesToMarketCap':             {'Operation': ['m'],        'Sign': 1},
                            'CFO':                          {'Operation': ['n'],        'Sign': 1},
                            'salesToInventory':             {'Operation': ['d'],        'Sign': 1},
                            'grossProfit':                  {'Operation': ['d'],        'Sign': 1},
                            'freeCashFlowToEquity':         {'Operation': ['d'],        'Sign': 1},
                            'CFOtoMarketCap':               {'Operation': ['d'],        'Sign': 1},
                            # incomeQuality's UNITY test is REPLACED by the sign-safe
                            # difference CFOlessEarnings (see BoMetric_special_dict).

                            'revenue':                      {'Operation': ['d'],        'Sign': 1},
                            'sharesOutstanding':            {'Operation': ['d'],        'Sign': -1},
                            'EPS':                          {'Operation': ['d'],        'Sign': 1},
                            'EquityToAssets':               {'Operation': ['m'],        'Sign': 1},
                            'netDebtToEBITDA':              {'Operation': ['u'],        'Sign': -1},
                            'netProfitMargin':              {'Operation': ['m'],        'Sign': 1},
                          }

    BoMetric_base_dict =    {
        'returnOnAssets':   {'Upper': 'netIncome',                              'Lower': 'totalAssets', 'Tier': 'S', 'Sign': 1},
        'grahamNetNet':     {'Upper': 'grahamNetNet',                           'Lower': 'Identity', 'Tier': 'N', 'Sign': 1},
        'CFO':              {'Upper': 'netCashProvidedByOperatingActivities',   'Lower': 'Identity', 'Tier': 'S', 'Sign': 1}
        }

    BoMetric_diff_dict =    {
        'returnOnTangibleAssets':       {'Upper': 'returnOnTangibleAssets',                 'Lower': 'Identity',                'Tier': 'B', 'Sign': 1},
        'returnOnAssets':               {'Upper': 'returnOnAssets',                         'Lower': 'Identity',                'Tier': 'S', 'Sign': 1},
        'grossProfitToAssets':          {'Upper': 'grossProfit',                            'Lower': 'totalAssets',             'Tier': 'A', 'Sign': 1},
        # NOT inverted (ruling Q1.2, 2026-07-26).  inventory == 0 on 41.44% of rows, and the
        # limit of revenue/inventory as inventory -> 0 IS the good side -- so this is the same
        # defect class as assetsToLongTermLiabilities.  Left alone for two reasons: Tier 'N'
        # means w = 0, so it cannot move a score; and for a services firm inventory == 0 means
        # the ratio is NOT APPLICABLE rather than infinitely good, so NaN is the more honest
        # reading than a synthetic best-case. Revisit if this metric is ever given weight.
        'salesToInventory':             {'Upper': 'revenue',                                'Lower': 'inventory',               'Tier': 'N', 'Sign': 1},
        'salesToAssets':                {'Upper': 'revenue',                                'Lower': 'totalAssets',             'Tier': 'N', 'Sign': 1},
        'grossProfitMargin':            {'Upper': 'grossProfitMargin',                      'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        'effectiveTaxRate':             {'Upper': 'effectiveTaxRate',                       'Lower': 'Identity',                'Tier': 'C', 'Sign': -1},
        'currentRatio':                 {'Upper': 'currentRatio',                           'Lower': 'Identity',                'Tier': 'B', 'Sign': 1},
        # INVERTED: longTermDebt/totalAssets, Sign -1 (ruling Q1.2, 2026-07-26).
        # It was totalAssets/longTermDebt with Sign +1.  longTermDebt == 0 on 17,824 of
        # 177,350 panel rows (10.05%) -- DEBT-FREE firms -- and TA/0 is +-inf, which
        # forceNumOnDf turns into NaN, which calcByTier scores as a FAIL.  So a debt-free
        # balance sheet FAILED a leverage-SAFETY test, and it failed it for having no debt.
        # The general rule: if the limit of the ratio as the denominator -> 0 is the GOOD
        # side, then inf -> NaN -> fail is backwards.  Inverting puts the zero-able quantity
        # in the NUMERATOR, so debt-free = 0.0 = best and is DEFINED; NaN rows drop from
        # 11.05% to 0.35% (only genuinely-corrupt totalAssets <= 0).  Sign flips to -1
        # because the metric direction reverses: falling leverage is now the good side.
        # WHAT THIS DID AND DID NOT FIX -- measured, because the difference matters:
        #   FIXED     the ratio is now DEFINED for a debt-free firm (0.0 instead of NaN):
        #             0.0% of debt-free rows had a value before, 96.6% do now.  So the
        #             criterion no longer fails them on a MISSING-DATA technicality, and the
        #             direction is now economically right (falling leverage scores).
        #   NOT FIXED they still mostly FAIL, for a different and legitimate reason: this is
        #             a DIFF test, and a permanently debt-free firm has d(0) = 0, which is
        #             not > 0.  85.7% of debt-free rows with a computable diff still fail
        #             (85.6% of them have a zero diff).  Only the 14.3% that actually
        #             DE-levered in the period pass.
        # That is true of EVERY diff test here -- "no change" is not "improvement" -- and is
        # not specific to this metric.  Do not describe the inversion as "debt-free firms no
        # longer fail a leverage test": they largely still do, on a defensible basis rather
        # than on missingness.
        'assetsToLongTermLiabilities':  {'Upper': 'longTermDebt',                           'Lower': 'totalAssets',             'Tier': 'B', 'Sign': -1},
        'grossProfit':                  {'Upper': 'grossProfit',                            'Lower': 'Identity',                'Tier': 'N', 'Sign': 1},
        'returnOnCapitalEmployed':      {'Upper': 'returnOnCapitalEmployed',                'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        # NOT inverted (ruling Q1.2, 2026-07-26).  totalStockholdersEquity == 0 on only 39 of
        # 177,350 rows (0.02%), and -- decisively -- the zero-denominator rule does NOT apply:
        # the limit of FCF/equity as equity -> 0 is +inf when FCF > 0 and -inf when FCF < 0,
        # so there is no single 'good side' to point at.  Zero book equity is also an
        # economically degenerate state (technically insolvent), which NaN describes better
        # than either infinity.
        'freeCashFlowToEquity':         {'Upper': 'freeCashFlow',                           'Lower': 'totalStockholdersEquity', 'Tier': 'B', 'Sign': 1},
        'CFOtoMarketCap':               {'Upper': 'netCashProvidedByOperatingActivities',   'Lower': 'marketCap',               'Tier': 'B', 'Sign': 1},
        # DUPLICATE OF 'EPS' BELOW -- byte-identical spec, differing only in Tier ('N',
        # w = 0, so inert today).  See DUPLICATE_DIFF_CRITERIA at the top of this module
        # before giving this a tier: promoting it double-counts the same quantity.
        'netIncomePerShare':            {'Upper': 'netIncomePerShare',                      'Lower': 'Identity',                'Tier': 'N', 'Sign': 1},
        'pbRatio':                      {'Upper': 'pbRatio',                                'Lower': 'Identity',                'Tier': 'B', 'Sign': -1},
        'revenue':                      {'Upper': 'revenue',                                'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        'sharesOutstanding':            {'Upper': 'weightedAverageShsOut',                  'Lower': 'Identity',                'Tier': 'B', 'Sign': -1},
        # THE WEIGHT-BEARING half of the duplicate pair (the other is
        # 'netIncomePerShare' above, Tier 'N').  See DUPLICATE_DIFF_CRITERIA.
        'EPS':                          {'Upper': 'netIncomePerShare',                      'Lower': 'Identity',                'Tier': 'B', 'Sign': 1}
                             }

    BoMetric_mean_dict =    {
        'pbRatio':              {'Upper': 'pbRatio',                    'Lower': 'Identity',    'Tier': 'B', 'Sign': -1},
        'salesToMarketCap':     {'Upper': 'revenue',                    'Lower': 'marketCap',   'Tier': 'N', 'Sign': 1},
        'earningsYield':        {'Upper': 'earningsYield',              'Lower': 'Identity',    'Tier': 'S', 'Sign': 1},
        'debtEquityRatio':      {'Upper': 'debtEquityRatio',            'Lower': 'Identity',    'Tier': 'C', 'Sign': -1},
        'pfcfRatio':            {'Upper': 'pfcfRatio',                  'Lower': 'Identity',    'Tier': 'S', 'Sign': -1},
        'EquityToAssets':       {'Upper': 'totalStockholdersEquity',    'Lower': 'totalAssets', 'Tier': 'D', 'Sign': 1},
        'grossProfitMargin':    {'Upper': 'grossProfitMargin',          'Lower': 'Identity',    'Tier': 'B', 'Sign': 1},
        'netProfitMargin':      {'Upper': 'netProfitMargin',            'Lower': 'Identity',    'Tier': 'C', 'Sign': 1},
                             }

    BoMetric_unity_dict =    {
        'currentRatio':         {'Upper': 'currentRatio',       'Lower': 'Identity',    'Tier': 'S', 'Sign': 1},
        'grahamNumberToPrice':  {'Upper': 'grahamNumber',       'Lower': 'price',       'Tier': 'S', 'Sign': 1},
        'netDebtToEBITDA':      {'Upper': 'netDebtToEBITDA',    'Lower': 'Identity',    'Tier': 'A', 'Sign': -1}
                             }

    BoMetric_special_dict ={
        # CFO - netIncome > 0 -- the sign-SAFE replacement for the old
        # uIncomeQuality unity test (domain review S1, fixed 2026-07-26).
        #
        # It used to test FMP's incomeQuality (= CFO/NI) against 1.0.  That is a RATIO whose
        # DENOMINATOR CHANGES SIGN, so the test inverts for loss-makers.  Measured on the
        # 2026-07-17 panel (175,503 usable rows, Tier S, w=1.0):
        #     NI>0 CFO>0 healthy                        pass 0.742
        #     NI>0 CFO<0 profit but no cash             pass 0.016  (correctly fails)
        #     NI<0 CFO>0 loss but CASH-GENERATIVE, the GOOD case   pass 0.034
        #     NI<0 CFO<0 loss AND burning cash          pass 0.316  <- ~10x the good case
        # 12,241 rows -- 14.6% of ALL passes of this w=1.0 criterion -- were companies
        # losing money AND burning cash.  The difference form fixes the inversion outright:
        # the good case goes to 1.000 and profit-without-cash to 0.000.
        #
        # This is exactly the form Piotroski p4 already uses (stage2_metrics.piotroski:
        # `CFO > netIncome`), so the pipeline now states the accrual test one way.
        # SEMANTICS, stated so the remaining 0.672 on loss-and-burning is not read as a
        # miss: this is an EARNINGS-QUALITY (accruals) test, not a profitability test -- a
        # loss-maker whose cash outflow is smaller than its accounting loss genuinely has
        # better earnings quality.  Profitability is tested separately and at Tier S by
        # returnOnAssets and the CFO>0 base test.
        'CFOlessEarnings':                  {'Tier': 'S', 'Sign': 1},
        'PEG':                              {'Tier': 'C', 'Sign': 1},
        'returnOnEquity':                   {'Tier': 'C', 'Sign': 1},
        'capitalExpenditureCoverageRatio':  {'Tier': 'C', 'Sign': 1},
                            }

    return preReq_dict, BoMetric_Calc_dict , BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict,BoMetric_special_dict

def getBaseMeanDiffUnitySpecialDicts():
    preReq_dict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict, BoMetric_special_dict = getDicts()

    return BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict, BoMetric_special_dict
def _buildPostDicts(weights):
    """Assemble the two Stage-2 ranking dicts from a flat {metric: w} vector.

    THE SHAPE IS THE CONTRACT.  postBoRank consumes these as
    `pd.DataFrame(columns=postBmRankingDict.keys())`, so the key ORDER is load-bearing;
    it is taken from scoringWeights.POSTBM_EQMET / POSTNEW_KEYS, which is the same
    order these dicts have always been written in.  Fresh inner dicts are built on
    every call so a caller that mutates the result cannot reach the canon.
    """
    postBmRankingDict = {k: {'eqMet': eq, 'w': weights[k]}
                         for k, eq in sw.POSTBM_EQMET.items()}
    postNewRankingDict = {k: {'w': weights[k]} for k in sw.POSTNEW_KEYS}
    return postBmRankingDict, postNewRankingDict


def getPostDict(macroAdj=1):
    """DECISIONAL Stage-2 weights -- the deployed mu theory prior.

    THE NUMBERS AND THEIR FULL PROVENANCE LIVE IN `scoringWeights.DEPLOYED`, which is
    the SINGLE SOURCE OF TRUTH for every scoring weight in the repo (single-source
    refactor, 2026-08-02).  Read that file to change a weight or to find out where
    0.072 on incomeQuality came from, which three metrics are deliberately zeroed and
    why, and why CycleHeat is negative -- all of it moved there VERBATIM, not
    summarised, so there is exactly one place to look.

    Nothing about this function's OUTPUT changed in the move: same two dicts, same keys
    in the same order, same eqMet mapping, same 'w' values and types.  Only the source
    of the numbers moved.  `getPostDict_legacy()` still holds the pre-2026-07-14
    vector for A/B.  `macroAdj` is unused, as before.
    """
    return _buildPostDicts(sw.DEPLOYED)


def getPostDict_legacy(macroAdj=1):
    """Pre-2026-07-14 double-counted DEFAULT weights (the certified 30.0% target-cell
    baseline). Retained for A/B against the promoted mu theory prior now decisional in
    getPostDict(); NOT decisional. Identical keys/eqMet/ordering to getPostDict -- only
    the 'w' values differ, so swapping this in reproduces the pre-promotion picks.

    Values in `scoringWeights.LEGACY` (same single source; the two vectors are held on
    ONE canonical key set so neither can silently drift off the other).  Note it keeps
    DcfToPrice = 0.35 where the deployed vector has 0.000 -- a real, deliberate
    difference, pinned by test_scoring_weights_single_source.py."""
    return _buildPostDicts(sw.LEGACY)

def getMetricDicts():
    preReqDict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict,BoMetric_unity_dict, BoMetric_special_dict = getDicts()

    return BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict
def dictCheckValid():
    dictCheckbool = True
    preReq_dict = getPreReqDict()
    BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict = getMetricDicts()
    testdict = {**BoMetric_base_dict,**BoMetric_mean_dict,**BoMetric_diff_dict,**BoMetric_unity_dict,}
    matchVec = []
    for baseMet in testdict.keys():
        for entry in testdict[baseMet]:
            matchVec = []
            if entry == 'Upper' or entry == 'Lower':
                x = testdict[baseMet][entry]
                for fdl in preReq_dict:
                    if x in preReq_dict[fdl] or x == 'Identity':
                        matchVec.append(True)
                    else:
                        matchVec.append(False)
                if not any(matchVec):
                    print(x)
                    dictCheckbool = False

    # Check for duplicates
    bigPreReqList = []
    for key in preReq_dict:
        bigPreReqList = bigPreReqList + preReq_dict[key]

    if len(bigPreReqList) > len(list(set(bigPreReqList))):
        print('preReq_dict has duplicates')
        dictCheckbool = False
    # Check for unnecessary elements in preReq_dict


    return dictCheckbool

def getPreReqDict():
    #preReqDict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict,BoMetric_unity_dict, BoMetric_special_dict = getDicts()
    dictList = getDicts()

    return dictList[0]