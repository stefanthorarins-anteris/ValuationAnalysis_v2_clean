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
                           'sellingGeneralAndAdministrativeExpenses', 'operatingIncome','interestExpense'],
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
    #'SalePerEmployee': {'w': 0.5},
    #'FCFperShare': {'w': 0.2},
    # calculate FCFyield (>5-10%), Profit Margin over sales (>15%)
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