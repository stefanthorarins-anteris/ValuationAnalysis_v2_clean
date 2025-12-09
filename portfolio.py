import pandas as pd
import sys
import configuration as cf
import utils as utils
import getData_gen as gdg
import getData_fmp as gdf
import calcScore as csf
import postBo as pb
import requests
import gains as gains
from datetime import datetime

args = sys.argv[1:]
def simulatePortfolio(portfoliofn,api_key):
    baseurl = 'https://financialmodelingprep.com/api/'
    #buydata must contain [ticker, buydate, buyprice, volume and sector]
    portfolio, pfdividends = createPortfolio()
    portdic = pd.read_pickle(portfoliofn)
    histprices = fillPriceVec(portfolio,api_key)
    histdivs = fillDivsVec(portfolio,api_key)

def createPortfolio(stocksbought=''):
    portfolio = pd.DataFrame(columns=['symbol','buydate','buyprice','volume','currentprice','sector','totalCashflow','gain'])
    pfdividends = pd.DataFrame(columns=['symbol','date','dividend'])
    # Allow calling with an empty value or a dataframe of bought stocks
    if isinstance(stocksbought, pd.DataFrame) and not stocksbought.empty:
        portfolio = addToPortfolio(portfolio, stocksbought)

    return portfolio, pfdividends

def addToPortfolio(portfolio,stocksBought):
    if ['symbol','buydate','buyprice','volume','sector'] not in stocksBought.columns:
        raise Exception('incorrect format on input dataframe')
    for ticker in stocksBought['symbol'].unique():
        tempdf = stocksBought[stocksBought['symbol'] == ticker]
        if len(tempdf) == 1:
            stock = {'symbol': ticker, 'buydate': tempdf.buydate, 'buyprice': tempdf.buyprice,
                     'volume': tempdf.volume,'currentprice': tempdf.buyprice,
                     'sector': tempdf.sector, 'gain': 0}
            portfolio.append(stock)
        else:
            stock = pd.DataFrame(columns=['symbol','buydate','buyprice','volume'])
            for i in range(0,len(tempdf)-1):
                tempdf_i = tempdf[i,:]
                tmpstock = {'symbol': ticker, 'buydate': tempdf_i.buydate, 'buyprice': tempdf_i.buyprice,
                     'volume': tempdf_i.volume,'currentprice': tempdf_i.buyprice,
                     'sector': tempdf_i.sector, 'gain': 0}

            pd.concat([portfolio,tmpstock])

    return portfolio

def fillPriceVec(portfolio,api_key,baseurl):
    histprices = pd.DataFrame(columns=['date', 'symbol', 'price'])
    histprices['symbol'] = portfolio.symbol
    for symbol in portfolio.symbol.unique():
        tempdf = pd.DataFrame(columns=['date', 'symbol', 'price'])
        tempres = requests.get(f'{baseurl}v3/discounted-cash-flow/{symbol}?apikey={api_key}')
        tempdf = utils.setDatesToQuarterly(pd.DataFrame.from_dict(tempres.json()))
        histprices = pd.concat([histprices, tempdf])

def fillDivsVec(portfolio,api_key,baseurl):
    histprices = pd.DataFrame(columns=['date', 'symbol', 'dividend'])
    histprices['symbol'] = portfolio.symbol
    for symbol in portfolio.symbol.unique():
        tempdf = pd.DataFrame(columns=['date', 'symbol', 'dividend'])
        tempres = requests.get(f'{baseurl}v3/historical-price-full/stock_dividend/{symbol}?apikey={api_key}')
        tempdf = utils.setDatesToQuarterly(setNaNtoZero(pd.DataFrame.from_dict(tempres.json())))
        histprices = pd.concat([histprices, utils.setDatesToQuarterly(tempdf)])

def updateCurrentPrice(portfolio,histprices,currentdate):
    for ticker in portfolio['symbol'].unique():
        cp = getCurrentPrice(ticker,histprices)
        portfolio[portfolio['symbol']==ticker]['currentprice'] = cp

    return portfolio

def getCurrentPrice(ticker,histprices,currentdate):
    tp = histprices.loc[histprices['symbol'] == ticker]
    cp = tp[tp['date'] == currentdate]
    return cp

def calculateGain(portfolio,histprices,histdivs,currentdate):
    #firstdate = getDatesFromPortFolio(portfolio,'min')
    #currentprices = getCurrentPrice(portfolio,currentdate)
    #divpersharetot = getDividends(portfolio,histdivs,firstdate,currentdate)
    #portfolio['gain'] = (currentprices - portfolio['buyprice'] + divpersharetot)/portfolio['buyprice']

    return portfolio

def setNaNtoZero(df):

    return None

def fillInQuarters(df):

    return None

def stocksBought(symbols,pfty_dt,baseurl,api_key):
    sbdf = pd.DataFrame(columns=['symbol','buydate','buyprice','volume','sector'])
    sbdf_orig = sbdf.copy()
    for symb in symbols:
        tempdf = sbdf_orig
        #call api for price at date pfty_dt, buy 1 stock of it.
        #concat the row to sbdf
        #TTT

    return sbdf


def createStocksBought(symbols, pfty_dt, baseurl, api_key, default_volume=1):
    """Create a DataFrame of stocks to buy on `pfty_dt` for the given `symbols`.

    For each symbol this fetches historical prices (via `gains.getHistPrices`) and picks a price
    near `pfty_dt` (using `gains.getPrice`). If price lookup fails the symbol is skipped.
    Returns DataFrame with columns ['symbol','buydate','buyprice','volume','sector'].
    """
    rows = []
    for s in symbols:
        try:
            hist = gains.getHistPrices(s, api_key, baseurl)
            price = gains.getPrice(s, hist, pfty_dt)
            # try to get sector from profile endpoint as best-effort
            sector = ''
            prof = gdg.safe_get(f"{baseurl}v3/profile/{s}?apikey={api_key}")
            if prof and isinstance(prof, list) and len(prof) > 0:
                sector = prof[0].get('sector', '')
            elif prof and isinstance(prof, dict):
                sector = prof.get('sector', '')

            if price is None or (isinstance(price, float) and pd.isna(price)):
                # skip symbols with no price available at the target date
                continue

            rows.append({'symbol': s, 'buydate': pd.to_datetime(pfty_dt), 'buyprice': float(price),
                         'volume': int(default_volume), 'sector': sector})
        except Exception:
            # non-fatal; skip symbol
            continue

    if rows:
        return pd.DataFrame(rows, columns=['symbol','buydate','buyprice','volume','sector'])
    else:
        return pd.DataFrame(columns=['symbol','buydate','buyprice','volume','sector'])



def portfolioBacktestWrapper(portfoliotestfirstyear,portfoliotestlastyear, datandmetricdic,resdic):
    #test
    portfoliotestfirstyear = 2017
    portfoliotestlastyear = 2021
    #test
    tempbmdf    = datandmetricdic['BoMetric_df']
    tempbmav    = datandmetricdic['BoMetric_ave']
    #tempbmda    = datandmetricdic['BoMetric_dateAve']
    tempcdx_df  = datandmetricdic['cdx_df']
    tempn       = datandmetricdic['nrScorePeriods']
    symbols = resdic['postRank']
    backtest_df = pd.DataFrame(columns=['date','1yearGain','2yearGain','3yearGain','5yearGain','10yearGain'])
    for pfyear in range(portfoliotestfirstyear,portfoliotestlastyear):
        tempdmdic = datandmetricdic
        pfty_dt = datetime.strptime(f'{pfyear}-12-31',"%Y-%m-%d")
        newbmdf = tempbmdf.loc[(tempbmdf['date'] <= pfty_dt)]
        #newbmda = tempbmda.loc[(tempbmda['date'] <= pfty_dt)]
        newcdx_df =  tempcdx_df.loc[(tempcdx_df['date'] <= pfty_dt)]
        tempdmdic['BoMetric_df'] = newbmdf
        #tempdmdic['BoMetric_ave'] = newbmav
        #tempdmdic['BoMetric_dateAve'] = newbmda
        tempdmdic['cdx_df'] = newcdx_df

        tempresdic = pb.postBoWrapper(tempdmdic)

        stocksBought = createStocksBought(symbols,pfty_dt,tempdmdic['baseurl'],tempdmdic['api_key'])
        testPortfolio = createPortfolio(stocksBought)

        calculateGain()
