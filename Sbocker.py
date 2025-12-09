import sys
import configuration as cf
import utils as utils
import getData_gen as gdg
import getData_fmp as gdf
import calcScore as csf
import postBo as pb
import detectManipulation as dm
def main():
    import sys
    import configuration as cf
    import utils as utils
    import getData_gen as gdg
    import getData_fmp as gdf
    import calcScore as csf
    import postBo as pb
    import detectManipulation as dm
    import portfolio as pf
    #import warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    args = sys.argv[1:]

    # Assign parameters
    configdic = cf.getDataFetchConfiguration(args)
    loadBoMetricbool = configdic['loadBoMetric']
    loadBoResultbool = configdic['loadBoResults']
    saveBoMetricbool = configdic['saveBoMetric']
    saveBoResultbool = configdic['saveBoResults']
    # for test
    if 'portfoliotestyear' not in configdic.keys():
        portfoliotestyear = -1
    else:
        portfoliotestyear = configdic['portfoliotestyear']

    #configdic['nrTaT'] = 50
    #loadBoMetricbool = 1
    #loadBoResultbool = 1

    # Initialize? Metric, Results and set manual eliminition of tickers list
    loadmetricdic = {'loadBoMetric': loadBoMetricbool, 'loadBoMetricfname': configdic['loadBoMetricfname']}
    datandmetricdic = utils.loadWrapper('metric', loadmetricdic)

    # Either load or get fundamental data from API, as well as the averages
    if not loadBoMetricbool:
        # Assign variables and get Tickers info and dataframe
        datasource, api_key, tickerfilter = configdic['datasource'], configdic['api_key'],  configdic['tickerfilter']
        manualelimtickers, baseurl = configdic['manualelimtickers'], configdic['baseurl']
        manualelimtickers = []
        Tickers_df = gdg.get_tickers(datasource, baseurl, api_key, manualelimtickers, tickerfilter,
                                     sfilt ='all', mcapf = -1, fn = '')
        # Assign variables and get financial data and calculate relevant metrics
        cdx_df, BoMetric_df, nrTaT = datandmetricdic['cdx_df'], datandmetricdic['BoMetric_df'], configdic['nrTaT']
        getfunddic = gdf.get_fundamentals_fmp(Tickers_df, cdx_df, BoMetric_df, baseurl, api_key, configdic['compyear'],
                                              configdic['fsMAnumber'], configdic['nrTaT'], configdic['startindex'],
                                              configdic['period'], configdic['nrperiods'])
        newmanelimtckrs = list(set(manualelimtickers + list(set(getfunddic['tickersfailed']) - set(getfunddic['lenfail']))))
        datandmetricdic.update(getfunddic)
        datandmetricdic['manualelimtickers'] = newmanelimtckrs

        lenhcy = len(datandmetricdic['hasCurrentYear'])
        if lenhcy > 0 and lenhcy < 3/4 * (len(Tickers_df) - len(datandmetricdic['tickersfailed'])):
            datandmetricdic['BoMetric_df'] = datandmetricdic['BoMetric_df'].iloc[1:,:]
            datandmetricdic['cdx_df'] = datandmetricdic['cdx_df'].iloc[1:,:]

        meandic = csf.getAves2(getfunddic['BoMetric_df'])
        # Note that **getfunddic should overwrite key-value combinations in datandmetricdic
        datandmetricdic = {**datandmetricdic, **{'Tickers_df': Tickers_df}, **getfunddic, **meandic, **configdic}

        #write to info to file
        utils.write_lastIndexRead(configdic['lastindex_fn'], getfunddic['cind'])
        utils.writeManElimToFile(datandmetricdic,newmanelimtckrs)
        # Save results if saveBoMetric == 1
        if saveBoMetricbool:
            utils.saveWrapper('metric', datandmetricdic)
    else:
        loadmetricdic = {'loadBoMetric': loadBoMetricbool, 'loadBoMetricfname': configdic['loadBoMetricfname']}
        datandmetricdic = utils.loadWrapper('metric', loadmetricdic)

    if portfoliotestyear > 0:
        datandmetricdic = pf.portfolioBacktestWrapper(portfoliotestyear,datandmetricdic)

    else:
        if not loadBoResultbool:
            resdic = pb.postBoWrapper(datandmetricdic)
            resdic = {**resdic, **datandmetricdic}

            # save results according to boolean. Note that saveBoResults = 0 if loadBoResults = 1
            if saveBoResultbool:
                utils.saveWrapper('results',resdic)
        else:
            loadresdic = {'loadBoResults': loadBoResultbool, 'loadBoResultsfname': configdic['loadBoResultsfname']}
            resdic = utils.loadWrapper('results', loadresdic)

    resdic = pb.findHighestOfEachSector(resdic)

    moatdf = pb.moatIdentifier(resdic['BoScore_df']['source'],resdic['cdx_df'])
    resdic.update({'moatdf': moatdf})

    detmandic = dm.detectManipulationWrapper(resdic)
    resdic = {**resdic, **detmandic}

    print(resdic['postRank'].head(50))

    pb.writeResWrapper(resdic)

    return None

if __name__ == '__main__':
    main()



