
def getBuySellPrice(ticker,cdx,beta=0.15,MoS=0.33:
    delta = getHistoricGrowth(ticker,cdx)
    TVM = getTerminalValueMultiplier(ticker,cdx)
    futureCashFlow = getFutureCashFlow(ticker,cdx,beta,delta,TVM)
    NVA = getDiscountedNetAssets(ticker,cdx)
    intrinsicValue = futureCashFlow + NVA
    BuyPrice = intrinsicValue*(1-MoS)
    SellPrice = intrinsicValue*1.1
    BuyPriceToPrice = BuyPrice/tprice
    SellPriceToPrice = SellPrice/tprice

    writeIVGtoXLS(ticker,cdx,intrinsicValue,BuyPrice,SellPrice,BuyPriceToPrice,SellPriceToPrice)

    return BuyPrice, SellPrice, BuyPriceToPrice, SellPriceToPrice

def getHistoricGrowth(ticker,cdx):
    # 1. Get revenue growth
    # 2. Get earnings growth
    # 3. Get free cash flow growth
    # 4. Get netAsset growth
    # 5. Get BookValue growth
    # 6. Calculate weighted average
    return None

def getTerminalValueMultiplier(ticker,cdx):
    # 1. Check historic FCF/P and P/E
    # 2. Discount each value by the growth in the year before (subjective)
    # 3. (Perform other corrections?)
    # 4. Take the mean

    return None

def getFutureCashFlow(ticker,cdx,beta,delta, deltaEQ,TVM):
    # 1. Calculate 3 years of beta discounted cash flow at delta
    # 2. Calculate 5 years of beta discounted cash flow at falling delta to deltaEQ
    # 3. Calculate 2 years of beta discounted cash flow at deltaEQ
    # 4. Calculate one time beta discounted cash flow at TVM*FCF_10
    # 5. Sum up

    return None

def getDiscountedNetAssets(ticker,cdx):
    # 1. Get Net assets from fmp
    # 2. Assume it will depricated by 10%
    # 3. return it

    return None

def writeIVGtoXLS(ticker,cdx,intrinsicValue,BuyPrice,SellPrice,BuyPriceToPrice,SellPriceToPrice):

    return None