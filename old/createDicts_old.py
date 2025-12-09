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
    preReq_dict = {'bs': ['totalAssets', 'longTermDebt', 'inventory', 'totalStockholdersEquity', 'totalLiabilities'],
                   'inc': ['netIncome', 'grossProfit', 'revenue'],
                   'cf': ['freeCashFlow', 'netCashProvidedByOperatingActivities'],
                   'km': ['netIncomePerShare', 'pbRatio', 'earningsYield', 'pfcfRatio', 'grahamNumber', 'grahamNetNet',
                          'marketCap', 'returnOnTangibleAssets', 'incomeQuality', 'bookValuePerShare'],
                   'fr': ['returnOnEquity', 'debtEquityRatio', 'currentRatio', 'grossProfitMargin', 'grossProfitMargin',
                          'effectiveTaxRate', 'returnOnCapitalEmployed', 'returnOnAssets'],
                   'other': ['price', 'shares']
                   }

    # Req_dic = {}
    # sub dicts
    # Possible entries later:
    #   'netOperatingAssets': {'Upper': '?', 'Lower': '?', 'Tier': 'B'}
    #   'dEquity': {'Upper': 'Equity', 'Lower': 'Identity', 'Tier': 'B'},
    #   'earningsPerShare': {'Upper': 'netIncome', 'Lower': 'totalAssets', 'Tier': 'B', 'Sign': 1}
    #   'dSalesToEmployees': {'Upper': 'revenue', 'Lower': 'Employees', 'Tier': 'B'},
    #   'dRoExdEquity': {'Tier': 'C', 'Sign': 1},
    #   bookValuePerShare >

    BoMetric_special_dict = {'dInvPEG': {'Tier': 'N', 'Sign': 1},
                             'CFOlessEarnings': {'Tier': 'A', 'Sign': 1},
                             'returnOnEquity': {'Tier': 'B', 'Sign': 1}
                             }

    # n is > 0; d is difference > 0; m is larger than the mean; u is larger than unity

    BoMetric_provided_fr_dict = {'currentRatio': {'Operation': ['u', 'd'], 'Sign': 1},
                                 'returnOnAssets': {'Operation': ['n', 'd'], 'Sign': 1},
                                 'debtEquityRatio': {'Operation': ['m'], 'Sign': -1},
                                 'effectiveTaxRate': {'Operation': ['d'], 'Sign': -1},
                                 'returnOnEquity': {'Operation': ['d'], 'Sign': 1},
                                 'returnOnCapitalEmployed': {'Operation': ['d'], 'Sign': 1},
                                 'grossProfitMargin': {'Operation': ['d'], 'Sign': 1},
                                 }

    BoMetric_provided_km_dict = {'pfcfRatio': {'Operation': ['m'], 'Sign': -1},
                                 'earningsYield': {'Operation': ['m'], 'Sign': 1},
                                 'returnOnTangibleAssets': {'Operation': ['d'], 'Sign': 1},
                                 'pbRatio': {'Operation': ['m', 'd'], 'Sign': -1},
                                 'grahamNetNet': {'Operation': ['n'], 'Sign': 1},
                                 'netIncomePerShare': {'Operation': ['d'], 'Sign': 1}
                                 }

    calcFromfs_dict = {'grossProfitToAssets': {'Operation': ['d'], 'Sign': 1},
                       'grahamNumberToPrice': {'Operation': ['u'], 'Sign': 1},
                       'salesToAssets': {'Operation': ['d'], 'Sign': 1},
                       'assetsToLongTermLiabilities': {'Operation': ['d'], 'Sign': -1},
                       'salesToMarketCap': {'Operation': ['m'], 'Sign': 1},
                       'CFO': {'Operation': ['n'], 'Sign': 1},
                       'salesToInventory': {'Operation': ['d'], 'Sign': 1},
                       'grossProfit': {'Operation': ['d'], 'Sign': 1},
                       'freeCashFlowToEquity': {'Operation': ['d'], 'Sign': 1},
                       'CFOtoMarketCap': {'Operation': ['d'], 'Sign': 1},
                       }

    BoMetric_base_dict = {'returnOnAssets': {'Upper': 'netIncome', 'Lower': 'totalAssets', 'Tier': 'S', 'Sign': 1},
                          'grahamNetNet': {'Upper': 'grahamNetNet', 'Lower': 'Identity', 'Tier': 'N', 'Sign': 1},
                          'CFO': {'Upper': 'netCashProvidedByOperatingActivities', 'Lower': 'Identity', 'Tier': 'S',
                                  'Sign': 1}
                          }

    BoMetric_diff_dict = {
        'returnOnTangibleAssets': {'Upper': 'netIncome', 'Lower': 'tangibleAssets', 'Tier': 'B', 'Sign': 1},
        'returnOnAssets': {'Upper': 'netIncome', 'Lower': 'totalAssets', 'Tier': 'S', 'Sign': 1},
        'grossProfitToAssets': {'Upper': 'grossProfit', 'Lower': 'totalAssets', 'Tier': 'A', 'Sign': 1},
        'salesToInventory': {'Upper': 'revenue', 'Lower': 'inventory', 'Tier': 'N', 'Sign': 1},
        'salesToAssets': {'Upper': 'revenue', 'Lower': 'totalAssets', 'Tier': 'N', 'Sign': 1},
        'grossProfitMargin': {'Upper': 'grossProfit', 'Lower': 'Revenue', 'Tier': 'A', 'Sign': 1},
        'effectiveTaxRate': {'Upper': 'ProvisionForIncomeTaxes', 'Lower': 'IncomeBeforeTax', 'Tier': 'C', 'Sign': -1},
        'currentRatio': {'Upper': 'currentAssets', 'Lower': 'currentLiabilities', 'Tier': 'B', 'Sign': 1},
        'assetsToLongTermLiabilities': {'Upper': 'totalAssets', 'Lower': 'longTermDebt', 'Tier': 'C', 'Sign': 1},
        'grossProfit': {'Upper': 'grossProfit', 'Lower': 'Identity', 'Tier': 'N', 'Sign': 1},
        'returnOnCapitalEmployed': {'Upper': 'netIncome', 'Lower': 'ownersEquity', 'Tier': 'B', 'Sign': 1},
        'freeCashFlowToEquity': {'Upper': 'freeCashFlow', 'Lower': 'totalStockholdersEquity', 'Tier': 'C', 'Sign': 1},
        'CFOtoMarketCap': {'Upper': 'netCashProvidedByOperatingActivities', 'Lower': 'marketCap', 'Tier': 'D',
                           'Sign': 1},
        'netIncomePerShare': {'Upper': 'netIncome', 'Lower': 'shares', 'Tier': 'N', 'Sign': 1},
        'pbRatio': {'Upper': 'marketCap', 'Lower': 'bookValue', 'Tier': 'B', 'Sign': -1}
        }

    BoMetric_mean_dict = {'pbRatio': {'Upper': 'marketCap', 'Lower': 'bookValue', 'Tier': 'A', 'Sign': -1},
                          'salesToMarketCap': {'Upper': 'revenue', 'Lower': 'marketCap', 'Tier': 'N', 'Sign': 1},
                          'earningsYield': {'Upper': 'netIncome', 'Lower': 'price', 'Tier': 'S', 'Sign': 1},
                          'debtEquityRatio': {'Upper': 'totalLiabilities', 'Lower': 'totalAssets', 'Tier': 'C',
                                              'Sign': -1},
                          'pfcfRatio': {'Upper': 'marketCap', 'Lower': 'freeCashFlow', 'Tier': 'S', 'Sign': -1},
                          }

    BoMetric_unity_dict = {
        'currentRatio': {'Upper': 'currentAssets', 'Lower': 'currentLiabilities', 'Tier': 'S', 'Sign': 1},
        'grahamNumberToPrice': {'Upper': 'grahamNumber', 'Lower': 'price', 'Tier': 'S',
                                'Sign': 1}
        }

    return preReq_dict, calcFromfs_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict, BoMetric_provided_fr_dict, BoMetric_provided_km_dict, BoMetric_special_dict


def getBaseMeanDiffUnitySpecialDicts():
    preReq_dict, calcFromfs_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_provided_fr_dict, BoMetric_provided_km_dict, BoMetric_special_dict = getDicts()

    return BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_special_dict, BoMetric_unity_dict


def getPostDict(macroAdj=1):
    # ADD NEW SHARES METRIC. WITH A MINUS
    postBmRankingDict = {'RoA': {'eqMet': 'returnOnAssets', 'w': 20},
                         'PE': {'eqMet': 'earningsYield', 'w': 40},
                         'grahamNumberToPrice': {'eqMet': 'grahamNumberToPrice', 'w': 1},
                         'grahamNetNet': {'eqMet': 'returnOnAssets', 'w': 10},
                         'pbRatio': {'eqMet': 'pbRatio', 'w': -1.5},
                         }

    postNewRankingDict = {'FCFperShare': {'w': 0.2},
                          'freeCashFlowYield': {'w': 100},
                          'DcfToPrice': {'w': 0.017},
                          'currentRatio': {'eqMet': 'currentRatio',
                                           'w': mcond.macroCondRankWeight('currentRatio', 'LHH') * 0.5},
                          'marketCapRevQuants': {'eqMet': 'marketCap', 'w': 1},
                          'SalePerEmployee': {'w': 0.5},
                          'Altman-Z': {'w': 0.17},
                          'Piotroski': {'w': 0.15},
                          }

    return postBmRankingDict, postNewRankingDict


