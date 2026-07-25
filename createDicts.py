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
                           # calendarYear the fiscal year FMP assigns; disambiguates the 52/53-week
                           #              drift that produces duplicate quarter stamps.
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
                            'assetsToLongTermLiabilities':  {'Operation': ['d'],        'Sign': -1},
                            'salesToMarketCap':             {'Operation': ['m'],        'Sign': 1},
                            'CFO':                          {'Operation': ['n'],        'Sign': 1},
                            'salesToInventory':             {'Operation': ['d'],        'Sign': 1},
                            'grossProfit':                  {'Operation': ['d'],        'Sign': 1},
                            'freeCashFlowToEquity':         {'Operation': ['d'],        'Sign': 1},
                            'CFOtoMarketCap':               {'Operation': ['d'],        'Sign': 1},
                            'incomeQuality':                {'Operation': ['u'],        'Sign': 1},
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
        'salesToInventory':             {'Upper': 'revenue',                                'Lower': 'inventory',               'Tier': 'N', 'Sign': 1},
        'salesToAssets':                {'Upper': 'revenue',                                'Lower': 'totalAssets',             'Tier': 'N', 'Sign': 1},
        'grossProfitMargin':            {'Upper': 'grossProfitMargin',                      'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        'effectiveTaxRate':             {'Upper': 'effectiveTaxRate',                       'Lower': 'Identity',                'Tier': 'C', 'Sign': -1},
        'currentRatio':                 {'Upper': 'currentRatio',                           'Lower': 'Identity',                'Tier': 'B', 'Sign': 1},
        'assetsToLongTermLiabilities':  {'Upper': 'totalAssets',                            'Lower': 'longTermDebt',            'Tier': 'B', 'Sign': 1},
        'grossProfit':                  {'Upper': 'grossProfit',                            'Lower': 'Identity',                'Tier': 'N', 'Sign': 1},
        'returnOnCapitalEmployed':      {'Upper': 'returnOnCapitalEmployed',                'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        'freeCashFlowToEquity':         {'Upper': 'freeCashFlow',                           'Lower': 'totalStockholdersEquity', 'Tier': 'B', 'Sign': 1},
        'CFOtoMarketCap':               {'Upper': 'netCashProvidedByOperatingActivities',   'Lower': 'marketCap',               'Tier': 'B', 'Sign': 1},
        'netIncomePerShare':            {'Upper': 'netIncomePerShare',                      'Lower': 'Identity',                'Tier': 'N', 'Sign': 1},
        'pbRatio':                      {'Upper': 'pbRatio',                                'Lower': 'Identity',                'Tier': 'B', 'Sign': -1},
        'revenue':                      {'Upper': 'revenue',                                'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        'sharesOutstanding':            {'Upper': 'weightedAverageShsOut',                  'Lower': 'Identity',                'Tier': 'B', 'Sign': -1},
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
        'incomeQuality':        {'Upper': 'incomeQuality',      'Lower': 'Identity',    'Tier': 'S', 'Sign': 1},
        'netDebtToEBITDA':      {'Upper': 'netDebtToEBITDA',    'Lower': 'Identity',    'Tier': 'A', 'Sign': -1}
                             }

    BoMetric_special_dict ={
        'PEG':                              {'Tier': 'C', 'Sign': 1},
        'returnOnEquity':                   {'Tier': 'C', 'Sign': 1},
        'capitalExpenditureCoverageRatio':  {'Tier': 'C', 'Sign': 1},
                            }

    return preReq_dict, BoMetric_Calc_dict , BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict,BoMetric_special_dict

def getBaseMeanDiffUnitySpecialDicts():
    preReq_dict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict, BoMetric_special_dict = getDicts()

    return BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict, BoMetric_special_dict
def getPostDict(macroAdj=1):
    # DECISIONAL weights. Promoted 2026-07-14 (MD directive, valuation-specialist
    # theory prior) to the mu THEORY-PRIOR vector that produced the certified 38.5%
    # target-cell beat-rate (top-20, 36mo, pooled buy2021+buy2022) -- up from the 30.0%
    # baseline under the legacy double-counted defaults. This is exactly
    # tune_run.MU_GENERAL (GP=0.100 primary variant): the LOCKED effective weights
    # mapped onto these getPostDict keys and normalized Sigma=1. ONLY the weight VALUES
    # changed vs the legacy vector -- every metric key / eqMet / scoring path / ordering
    # is identical, so the as_of=None machinery invariant holds: only the picks move.
    # Three metrics zeroed (DcfToPrice / BoScore / priceGrowth -- two drops + the
    # priceGrowth bug); CycleHeat stays NEGATIVE (late-cycle penalty). Legacy defaults
    # preserved in getPostDict_legacy() for A/B -- NOT deleted.
    postBmRankingDict = {'RoA':                 {'eqMet': 'returnOnAssets',         'w': 0.060},   # legacy 2
                         'earnYield':           {'eqMet': 'earningsYield',          'w': 0.0605},  # legacy 2
                         'grahamNumberToPrice': {'eqMet': 'grahamNumberToPrice',    'w': 0.033},   # legacy 1
                         'bVpRatio':            {'eqMet': 'pbRatio',                'w': 0.033},   # legacy 0.25
                         'revenueGrowth':       {'eqMet': 'revenue',                'w': 0.027},   # legacy 1
                         'incomeQuality':       {'eqMet': 'incomeQuality',          'w': 0.072},   # legacy 1
                         'returnOnEquity':      {'eqMet': 'returnOnEquity',         'w': 0.030},   # legacy 1
                         'returnOnCapitalEmployed': {'eqMet': 'returnOnCapitalEmployed', 'w': 0.060},  # legacy 1
                         'currentRatio':        {'eqMet': 'currentRatio',           'w': 0.038},   # legacy 0.35
                         'grossProfitMargin':   {'eqMet': 'grossProfitMargin',      'w': 0.100}    # legacy 0.75
                         }

    postNewRankingDict =    {'freeCashFlowYield':           {'w': 0.0605},  # legacy 2
                             'freeCashFlowPerShareGrowth':  {'w': 0.043},   # legacy 1.5
                             'DcfToPrice':                  {'w': 0.000},   # legacy 0.35 -- DROPPED (BoDCF broken / no PIT DCF)
                             'marketCapRevQuants':          {'w': 0.080},   # legacy 0.25
                             'Altman-Z':                    {'w': 0.062},   # legacy 0.5
                             'Piotroski':                   {'w': 0.072},   # legacy 0.75
                             'tbVpRatio':                   {'w': 0.033},   # legacy 0.5
                             'BoScore':                     {'w': 0.000},   # legacy 0.1 -- DROPPED
                             'EPStoEPSmean':                {'w': 0.056},   # legacy 0.5
                             'priceGrowth':                 {'w': 0.000},   # legacy 0.5 -- DROPPED (sign/seasonality bug)
                             'CycleHeat':                   {'w': -0.080}   # legacy -0.5 -- NEGATIVE: penalizes hot late-cycle stocks
                             }

    return postBmRankingDict, postNewRankingDict


def getPostDict_legacy(macroAdj=1):
    """Pre-2026-07-14 double-counted DEFAULT weights (the certified 30.0% target-cell
    baseline). Retained for A/B against the promoted mu theory prior now decisional in
    getPostDict(); NOT decisional. Identical keys/eqMet/ordering to getPostDict -- only
    the 'w' values differ, so swapping this in reproduces the pre-promotion picks."""
    postBmRankingDict = {'RoA':                 {'eqMet': 'returnOnAssets',         'w': 2},
                         'earnYield':           {'eqMet': 'earningsYield',          'w': 2},
                         'grahamNumberToPrice': {'eqMet': 'grahamNumberToPrice',    'w': 1},
                         'bVpRatio':            {'eqMet': 'pbRatio',                'w': 0.25},
                         'revenueGrowth':       {'eqMet': 'revenue',                'w': 1},
                         'incomeQuality':       {'eqMet': 'incomeQuality',          'w': 1},
                         'returnOnEquity':      {'eqMet': 'returnOnEquity',         'w': 1},
                         'returnOnCapitalEmployed': {'eqMet': 'returnOnCapitalEmployed', 'w': 1},
                         'currentRatio':        {'eqMet': 'currentRatio',           'w': 0.35},
                         'grossProfitMargin':   {'eqMet': 'grossProfitMargin',      'w': 0.75}
                         }

    postNewRankingDict =    {'freeCashFlowYield':           {'w': 2},
                             'freeCashFlowPerShareGrowth':  {'w': 1.5},
                             'DcfToPrice':                  {'w': 0.35},
                             'marketCapRevQuants':          {'w': 0.25},
                             'Altman-Z':                    {'w': 0.5},
                             'Piotroski':                   {'w': 0.75},
                             'tbVpRatio':                   {'w': 0.5},
                             'BoScore':                     {'w': 0.1},
                             'EPStoEPSmean':                {'w': 0.5},
                             'priceGrowth':                 {'w': 0.5},
                             'CycleHeat':                   {'w': -0.5}  # Negative weight penalizes hot late-cycle stocks
                             }

    return postBmRankingDict, postNewRankingDict

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