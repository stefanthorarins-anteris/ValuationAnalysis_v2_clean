import pandas as pd
import createDicts as cdic
import os
import csv
from datetime import datetime

def loadWrapper(type,loaddic):
    if type == 'metric':
        lbmfn = loaddic['loadBoMetricfname']
        if loaddic['loadBoMetric']:
            xdic = pd.read_pickle(lbmfn)
        else:
            xdic = initBoMetric_fromDict()
    elif type == 'results':
        lbrfn = loaddic['loadBoResultsfname']
        if loaddic['loadBoResults']:
            xdic = pd.read_pickle(lbrfn)
    else:
        raise Exception('Illegal type in loading. Only metric and results allowed')

    xdic #metricdic or resdic
    return xdic

def saveWrapper(type,savedata):
    fidag = datetime.today().strftime('%Y-%m-%d')
    tf = savedata['tickerfilter']
    sf = savedata['sectorfilter']
    lentdf = savedata['BoMetric_df']['source'].nunique()
    nrmanelim = len(savedata['manualelimtickers'])
    lentfail = len(savedata['tickersfailed'])
    ds = savedata['datasource']
    fname_bmdf = f'Bo{type}_dic-{ds}_{tf}_{sf}_{fidag}_len{lentdf}_manelim{nrmanelim}_fails{lentfail}.pickle'
    pd.to_pickle(savedata, fname_bmdf)

def initBoMetric_fromDict():
    preReq_dict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict,BoMetric_special_dict = cdic.getDicts()
    BMdfcollist = ['date']
    complist = ['date']
    cdxcollist = ['date']

    for key in BoMetric_Calc_dict:
        ops = BoMetric_Calc_dict[key]['Operation']
        for o in ops:
            if o == 'n':
                coln = key
            elif o == 'm':
                coln = "m" + key[0].upper() + key[1:]
            elif o == 'u':
                coln = "u" + key[0].upper() + key[1:]
            elif o == 'd':
                coln = "d" + key[0].upper() + key[1:]
            BMdfcollist.append(coln)

    for key1 in BoMetric_special_dict:
        BMdfcollist.append(key1)

    for key in preReq_dict:
        cdxcollist.extend(preReq_dict[key])

    BoTckr_count = pd.DataFrame({string: [0] for string in BMdfcollist[1:]})
    #BMdfcollist = ['date'] + BMdfcollist
    #cdxcollist = ['date'] + BMdfcollist
    BoMetric_sum = pd.DataFrame(columns=BMdfcollist)
    BMdfcollist.append('source')
    cdxcollist.append('source')
    BoMetric_df = pd.DataFrame(columns=BMdfcollist)
    cdx_df = pd.DataFrame(columns=cdxcollist)

    #period = 'quarter'
    #limit = 8
    #temp_resp = requests.get(f'{baseurl}/key-metrics/AAPL?period={period}&limit={limit + 4}&apikey={api_key}')
    #if temp_resp.status_code in range(400, 600):
    #    raise Exception('Something wrong with API, I suppose')
    #else:
    #    temp_resp_df = pd.DataFrame(temp_resp.json())

    #BoMetric_df['date'] = temp_resp_df['date']
    #BoMetric_sum['date'] = temp_resp_df['date']
    #cdx_df['date'] = temp_resp_df['date']

    metricdic = {'BoMetric_df': BoMetric_df, 'BoMetric_sum': BoMetric_sum,
                 'BoTckr_count': BoTckr_count, 'cdx_df': cdx_df}
    return metricdic

def writeManElimToFile(dmdic,manualelimtickers):
    tfilter = dmdic['tickerfilter']
    ds = dmdic['datasource']
    fidag = datetime.today().strftime('%Y-%m-%d')
    mefn = f'ManualEliminationTickersList_{ds}_{fidag}.csv'
    with open(mefn, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(manualelimtickers)

def get_lastIndexRead(lastindex_fn):
    allowedfn = ['lastIndexOfRead_fmp_stock_US1.txt','lastIndexOfRead_fmp_stock_NA1_EU1.txt', 'lastIndexOfRead_fmp_stock_NA1.txt', 'lastIndexOfRead_fmp_stock_WW1_TV.txt']
    if lastindex_fn in allowedfn:
        if not os.path.exists(lastindex_fn):
            with open(lastindex_fn, "w") as file:
                file.write('%d' % 0)
                startindex = 0
                print('File didnt exist, but filename is allowed. I created the file and set the starting index to 0')
        else:
            with open(lastindex_fn) as f:
                lines_list = f.readlines()
                startindex = int(lines_list[0])
    else:
        raise Exception('Not Implemented')

    return startindex

def write_lastIndexRead(lastindex_fn, currentIndex = 0):
    with open(f'{lastindex_fn}', 'w') as f:
        f.write('%d' % currentIndex)

    return None


def setDatesToQuarterly(df):
    df['date'] = pd.PeriodIndex(df.date, freq='Q').to_timestamp()

    return df