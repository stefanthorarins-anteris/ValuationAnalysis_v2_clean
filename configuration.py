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
        nrperiods = args[inp + 1]
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
        isbm = args.index('-savebobetric')
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

    if '-portfolioTest' in args:
        ipt = args.index('-portfolioTest')
        portfoliotestyear = args[ipt +1]
    else:
        portfoliotestyear = -1

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
                 'sectorlist': sectorlist}

    return configdic

