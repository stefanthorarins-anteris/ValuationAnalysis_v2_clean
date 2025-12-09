from datetime import datetime
import pandas as pd
import requests
def testForAPIFaults_fmp(failcodes,compyear,ticker,period,limit,baseurl,api_key):
    failbool = False
    whyfail = 'None'
    #baseurl = 'https://financialmodelingprep.com/api/'
    #resp_km = requests.get(f'{baseurl}v3/key-metrics/{ticker}?period={period}&limit={limit}&apikey={api_key}')
    #resp_fr = requests.get(f'{baseurl}v3/ratios/{ticker}?period={period}&limit={limit}&apikey={api_key}')
    #resp_inc = requests.get(f'{baseurl}v3/income-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
    #resp_bs = requests.get(f'{baseurl}v3/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
    #resp_cf = requests.get(f'{baseurl}v3/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
    calldic = {'km': 'key-metrics', 'fr': 'ratios','inc': 'income-statement', 'bs': 'balance-sheet-statement',
               'cf': 'cash-flow-statement'}
    #respstatcodes = [resp_km.status_code, resp_fr.status_code, resp_inc.status_code, resp_bs.status_code,
    #                 resp_cf.status_code]
    resplist = []
    respfail = False
    fsdfdic = {}
    for key in calldic.keys():
        resp = requests.get(f'{baseurl}v3/{calldic[key]}/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        if resp.status_code in failcodes:
            respfail = True
            failbool = True
            whyfail = 'failcode'
            break
        else:
            resplist.append(resp.json())
            fsdfdic[key] = pd.DataFrame(resp.json())

    if respfail == False:
        if any([not lst for lst in resplist]):
            failbool = True
            whyfail = 'emptyfail'
        else:
            tempdf = pd.DataFrame(resplist[2])
            if 'date' in tempdf.columns:
                strdate = tempdf['date'].iloc[0]
                if compyear > datetime.strptime(strdate, '%Y-%m-%d').year:
                    failbool = True
                    whyfail = 'datefail'
                else:
                    lentest = [len(resp) for resp in resplist]
                    if period == 'quarter':
                        if any(j < 16 for j in lentest):
                            failbool = True
                            whyfail = 'lenfail'
                    elif period == 'annual':
                        if any(j < 4 for j in lentest):
                            failbool = True
                            whyfail = 'lenfail'
                    else:
                        raise Exception('Why is period not either quarter or annual?')
            else:
                raise Exception(f'No column, in dataframe from API, with the name: date')

    outdic = {}
    if failbool == False:
        outdic = fsdfdic

    return failbool, whyfail, outdic
