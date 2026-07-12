from datetime import datetime
import csv
import utils

def getDataFetchConfiguration(args):
    # Assign tickerfilter
    tickerfilterlist = ['stock_NA1', 'stock_US1', 'stock_WW1_TV','stock_NA1_EU1', 'stock_US1_EU1', 'stock_US1_EU2']
    if '-tickerfilter' in args:
        it = args.index("-tickerfilter")
        tickerfilter = args[it+1]
        if tickerfilter not in tickerfilterlist:
            raise Exception('-tickerfilter argument not valid')
    else:
        tickerfilter = tickerfilterlist[3]

    # Assign datasource
    datasourcelist = ['fmp']
    if '-datasource' in args:
        id = args.index("-datasource")
        datasource = args[id+1]
        if datasource not in datasourcelist:
            raise Exception('-datasource argument not valid')
    else:
        datasource = datasourcelist[0]
    # Getting the associated API baseurl and setting the api_key, for the datasource 'fmp'
    if datasource == 'fmp':
        api_key_fname = 'fmpAPIkey.txt'
        api_key = open('fmpAPIkey.txt', 'r').read()
        baseurl = "https://financialmodelingprep.com/api/"

    # Assign filtering on market cap band to filter
    if '-mcapAbove' in args:
        print('-mcapAbove not yet implemented. Will be ignored')
        ima = args.index('-mcapAbove')
        mcapUL = int(args[ima+1])
    else:
        mcapUL = -1
    if '-mcapBelow' in args:
        print('-mcapBelow not yet implemented. Will be ignored')
        imb = args.index('-mcapBelow')
        mcapLL = int(args[imb+1])
    else:
        mcapLL = -1

    mcapUL, mcapLL = [-1,-1]

    #sectorlist = ['all', 'Basic Materials', 'Healthcare', 'Financial Services', 'Energy', 'Consumer Cyclical',
    # 'Biotechnology', 'Consumer Defensive', 'Pharmaceuticals', 'Industrials', 'Communication Services', 'Technology',
    # 'Real Estate', 'Utilities', 'Media', 'Hotels, Restaurants & Leisure', 'Food Products', 'Machinery',
    # 'Electrical Equipment', 'Commercial Services & Supplies', 'Semiconductors', 'Construction',
    # 'Textiles, Apparel & Luxury Goods', 'Metals & Mining', 'Retail', 'Logistics & Transportation', 'Road & Rail',
    # 'Chemicals', 'Professional Services', 'Insurance', 'Airlines', 'Aerospace & Defense', 'Telecommunication',
    # 'Services', 'Consumer Goods', 'Trading Companies & Distributors', 'Banking', 'Consumer products', 'Packaging',
    # 'Conglomerates']
    sectorlist = ['all', 'Unspecified', 'Basic Materials', 'Healthcare', 'Financial Services',
                  'Energy', 'Consumer Cyclical', 'Consumer Defensive', 'Industrials',
                  'Communication Services', 'Technology', 'Real Estate', 'Utilities']
    if '-sectorfilter' in args:
        print('Limited implementation of sector filter')
        isf = args.index('-sectorfilter')
        sectorfilter = args[imb+1]
        if sectorfilter not in sectorlist:
            raise Exception('-sectorfilterr argument not valid')
    else:
        sectorfilter = 'all'


    #Assign period of the data
    periodlist = ['quarter', 'annual']
    if '-period' in args:
        ip = args.index('-period')
        period = args[ip + 1]
        if period not in periodlist:
            raise Exception('-period argument is not valid')
    else:
        period = 'quarter'

    #Assign number of periods to fetch
    if '-nrperiods' in args:
        inp = args.index('-nrperiods')
        nrperiods = int(args[inp + 1])
    else:
        nrperiods = 6 * 4

    #nr of Tickers at a Time
    if '-nrTaT' in args:
        itat = args.index('-nrTaT')
        nrTaT = int(args[itat+1])
    else:
        nrTaT = -1

    # Get comparison year (default last year)
    if '-compyear' in args:
        ic = args.index("-compyear")
        compyearstr = args[id+1]
        if compyearstr == 'lastYear':
            compyear = datetime.now().year - 1
        elif compyearstr == 'thisYear':
            compyear = datetime.now().year
        else:
            raise Exception('compyear argument is not valid')
    else:
        compyear = datetime.now().year - 1

    # Set how many datapoints used in moving average of the entries of the financial statments fetched
    if '-fsMAnumber' in args:
        ima = args.index('-fsMAnumber')
        fsMAnumber = int(args[ima + 1])
    else:
        fsMAnumber = 1

    # Set number of periods used in averaging when calculating score for each metric
    if '-nrScorePeriods' in args:
        insp = args.index('-nrScorePeriods')
        nrScorePeriods = int(args[insp + 1])
    else:
        nrScorePeriods = 8

    # number of stocks in top list and the presentation, respectively
    if '-ntopagg' in args:
        inta = args.index('-ntopagg')
        ntopagg = int(args[inta + 1])
    else:
        ntopagg =  100

    if '-ntopxlsx' in args:
        intx = args.index('-ntopxlsx')
        ntopxlsx = int(args[intx + 1])
    else:
        ntopxlsx = 20

    if '-newOnly' in args:
        newOnly = 1
    else:
        newOnly = 0

    # Assign values to saving and loading bools
    if '-savebometric' in args:
        isbm = args.index('-savebometric')
        saveBoMetric = 1 if int(args[isbm+1]) > 0 else 0
    else:
        saveBoMetric = 1

    # Assign booleans on saving and loading
    if '-saveboresults' in args:
        isbr = args.index('-saveboresults')
        saveBoResults = 1 if int(args[isbr+1]) > 0 else 0
    else:
        saveBoResults = 1

    if '-loadbometric' in args:
        ilbm = args.index('-loadbometric')
        loadBoMetric = 1 if int(args[ilbm+1]) > 0 else 0
    else:
        loadBoMetric = 0

    if '-loadboresults' in args:
        ilbr = args.index('-loadboresults')
        loadBoResults = 1 if int(args[ilbr+1]) > 0 else 0
    else:
        loadBoResults = 0

    if loadBoMetric:
        if saveBoMetric:
            print('Since loadBoMetric is set to unity, saveBoMetric is disabled')
            saveBoMetric = 0
    if loadBoResults:
        if saveBoResults:
            print('Since loadBoResults is set to unity, saveBoResults is disabled')
            saveBoResults =  0

    # Set boolean that determines whether symbol changes are affecting fetched data
    if '-symbolChangeRestock' in args:
        ischr = args.index('-symbolChangeRestock')
        symbchRestock = 1 if int(args[ischr+1]) > 0 else 0
    else:
        symbchRestock = 0

    # Assign loading filenames of Metrics, Results and elimination list of Tickers
    if '-bometricfilename' in args:
        ibmfn = args.index('-bometricfilename')
        loadBoMetricfname = args[ibmfn+1]
    else:
        loadBoMetricfname = 'Bometric_dic-fmp_stock_NA1_EU1_all_2023-03-16_len6728_manelim3692_fails6729.pickle'

    if '-boresultsfilename' in args:
        ibrfn = args.index('-boresultsfilename')
        loadBoResultsfname = args[ibrfn+1]
    else:
        loadBoResultsfname = 'Boresults_dic-fmp_stock_NA1_EU1_all_2023-03-16_len6728_manelim3692_fails6729.pickle'

    # Assign boolean and filename to manual elimination of ticker symbols before fetching data
    if '-manelimtickers' in args:
        imet = args.index('-manelimtickers')
        manelimtickersbool = args[imet + 1]
    else:
        manelimtickersbool = 0

    if '-manelimfilename'in args:
        imefn = args.index('-manelimfilename')
        manelimtick_fname_toget = args[ibmfn + 1]
    else:
        manelimtickersbool = 1
        manelimtick_fname_toget = 'ManualEliminationTickersList_fmp_2023-02-14.csv'

    # Point-in-time as-of date D (design 2026-07-12 restructure).  Default None =
    # today / live run (reproduces current behaviour bit-for-bit).  Pass an ISO date
    # (YYYY-MM-DD) to run the pipeline as-of that past date (survivorship-safe PIT
    # universe + availability-date metric slice).  Tonight's full deep-fetch is a
    # LIVE run -> omit this flag (as_of stays None).
    if '-asof' in args:
        iao = args.index('-asof')
        # LOW-B fix: bounds-check so `-asof` as the FINAL arg (no date) raises a
        # clear error instead of an opaque IndexError.
        if iao + 1 >= len(args):
            raise Exception('-asof requires a date argument (YYYY-MM-DD)')
        as_of = args[iao + 1]
        # MEDIUM-B guard (review addendum 2): the -asof path is only PARTIALLY
        # point-in-time.  simpleScore_fromDict applies the row-level availability
        # slice (L1/L4), but the cross-sectional baseline bm_ave / getAves2 (L2) and
        # the per-ticker means (L3) are still computed over the FULL panel, and
        # DCF/beta (L5/L6) are not substituted.  So a -asof run STILL embeds L2/L3
        # lookahead and must NOT be treated as clean PIT.  Warn loudly so a
        # partial-PIT run is never mistaken for a clean one.  (Tonight is as_of=None
        # -> this never fires on the live run; the guard is wired for when -asof is
        # used later.)
        import warnings as _w
        _w.warn(
            "PARTIAL-PIT: -asof applies ONLY the row-level availability slice "
            "(L1/L4). The cross-sectional baseline (L2, getAves2/bm_ave), the "
            "per-ticker means (L3), and DCF/beta substitution (L5/L6) are NOT yet "
            "point-in-time -- this run STILL embeds L2/L3 lookahead. Do NOT treat "
            "its output as clean point-in-time.")
    else:
        as_of = None

    # -ingest_delisted (default OFF): gate for the survivorship / delisted-entity
    # ingestion (delisted_ingest.run_ingest).  When OFF the ingestion module is
    # never imported and the live path is untouched / bit-for-bit.  Turn ON for the
    # full survivorship deep-fetch.  Optional -delisted_max_pages bounds the
    # registry pagination guard.
    ingest_delisted = 1 if '-ingest_delisted' in args else 0
    if '-delisted_max_pages' in args:
        idmp = args.index('-delisted_max_pages')
        delisted_max_pages = int(args[idmp + 1])
    else:
        delisted_max_pages = 500
    startfromlastindex = 1 if '-startfromlastindex' in args else 0

    if '-portfolioTest' in args:
        ipt = args.index('-portfolioTest')
        portfoliotestyear = args[ipt +1]
    else:
        portfoliotestyear = -1

    # Unified backtesting parameters
    if '-runbacktest' in args:
        runbacktest = 1
    else:
        runbacktest = 0
    
    if '-backtest_buy_years' in args:
        iby = args.index('-backtest_buy_years')
        backtest_buy_years = [int(y) for y in args[iby + 1].split(',')]
    else:
        backtest_buy_years = None  # Will use defaults in backtest_unified
    
    if '-backtest_eval_years' in args:
        iey = args.index('-backtest_eval_years')
        backtest_eval_years = [int(y) for y in args[iey + 1].split(',')]
    else:
        backtest_eval_years = None
    
    if '-backtest_topn' in args:
        itn = args.index('-backtest_topn')
        backtest_topn = int(args[itn + 1])
    else:
        backtest_topn = 100

    # Transfer directory for end-of-run copy to Google-Drive-synced folder
    # Default None (off). When set, the pipeline copies the output allowlist there at
    # end-of-run, after ALL outputs are written and after ingestion completes.
    if '-transfer_dir' in args:
        itd = args.index('-transfer_dir')
        if itd + 1 >= len(args):
            raise Exception('-transfer_dir requires a directory path argument')
        transfer_dir = args[itd + 1]
    else:
        transfer_dir = None

    # Skip loading manual elimination CSV when loading metrics (it's already in the pickle file)
    if loadBoMetric:
        manualelimtickers = []
    elif manelimtickersbool:
        with open(manelimtick_fname_toget, 'r') as file:
            reader = csv.reader(file)
            templist = list(reader)
            manualelimtickers = templist[0]
    else:
        manualelimtickers = []

    # Inform of consistency
    if loadBoMetric or loadBoResults:
        print('Note that loading might overwrite other arguments.')

    lastindex_fn = 'lastIndexOfRead_' + datasource + '_' + tickerfilter + '.txt'
    if '-startfromlastindex' in args:
        startindex = utils.get_lastIndexRead(lastindex_fn)
    else:
        startindex = 0

    # get the starting index for getting data for fundamentals


    configdic = {'tickerfilter': tickerfilter, 'datasource': datasource, 'baseurl': baseurl, 'api_key': api_key,
                 'period': period, 'nrperiods': nrperiods, 'nrTaT': nrTaT, 'compyear': compyear, 'newOnly': newOnly,
                 'fsMAnumber': fsMAnumber, 'startindex': startindex, 'mcapUL': mcapUL, 'mcapLL': mcapLL,
                 'saveBoMetric': saveBoMetric, 'saveBoResults': saveBoResults, 'loadBoMetric': loadBoMetric,
                 'loadBoResults': loadBoResults, 'symbchRestock': symbchRestock, 'loadBoMetricfname': loadBoMetricfname,
                 'loadBoResultsfname': loadBoResultsfname, 'manualelimtickers': manualelimtickers,
                 'lastindex_fn': lastindex_fn, 'nrScorePeriods': nrScorePeriods, 'ntopagg': ntopagg,
                 'ntopxlsx': ntopxlsx, 'sectorfilter': sectorfilter, 'portfoliotestyear': portfoliotestyear,
                 'sectorlist': sectorlist,
                 'runbacktest': runbacktest, 'backtest_buy_years': backtest_buy_years,
                 'backtest_eval_years': backtest_eval_years, 'backtest_topn': backtest_topn,
                 'as_of': as_of, 'ingest_delisted': ingest_delisted,
                 'delisted_max_pages': delisted_max_pages,
                 'startfromlastindex': startfromlastindex, 'transfer_dir': transfer_dir}

    return configdic

