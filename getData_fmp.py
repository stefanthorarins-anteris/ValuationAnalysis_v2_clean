import pandas as pd
import requests
import calcMetrics as cm
import json
from tqdm import tqdm
import numpy as np
import warnings
import createDicts as cdic
import getData_gen as gdg
import failTests as ft
import utils as utils
from datetime import datetime


def get_fundamentals_fmp(Tickers_df, cdx_df, BoMetric_df, baseurl,
                         api_key,compyear, n=1, nrTaT=-1, startindex=0,period='quarter',limit=44):
    print('Fetching financial data from FMP and calculating relevant metrics.')
    if not isinstance(Tickers_df, pd.DataFrame):
        raise Exception('provide a DataFrame')
    if period == 'quarter' and limit < 16:
        raise Exception('Number of periods, if periods are quarters, must be larger than 16')
    tickersfailed = []
    lenfail = []
    datefail = []
    pricefail = []
    pricefailESN = []
    emptyfail = []
    hasCurrentYear = []
    if nrTaT < 0 and startindex == 0:
        pbar = tqdm(total=len(Tickers_df))
    elif nrTaT < 0 and startindex > 0:
        pbar = tqdm(total=len(Tickers_df)-startindex)
    else:
        total = min(nrTaT,len(Tickers_df)-startindex)
        pbar = tqdm(total=total)
    cntr = 0
    Tickers_df = Tickers_df.iloc[startindex: ,:]
    preReq_dict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, \
        BoMetric_unity_dict, BoMetric_special_dict = cdic.getDicts()
    for row in Tickers_df.itertuples():
        cntr = cntr + 1
        ticker = row.symbol

        km, fr, inc, bs, cf, tickersfailed, lenfail, datefail, emptyfail = getFsData_fmp(ticker, period, limit,baseurl,
                                                                                         api_key, compyear, tickersfailed,
                                                                                         lenfail, datefail,emptyfail)
        if not (isinstance(km, int) and km == -37707):
            tempfund, tempMetric_df = initTempMets(BoMetric_df.columns, cdx_df.columns,
                                                                                   bs['date'], ticker)

            tempfund, hcy = fillPreReqdf(tempfund, preReq_dict, bs, inc, cf, km, fr)
            tempMetric_df = utils.setDatesToQuarterly(tempMetric_df)
            if hcy == 1:
                hasCurrentYear.append(ticker)

            if not gdg.checkIfValidFS(tempfund):
                tickersfailed.append(ticker)
                pricefail.append(ticker)
                pricefailESN.append(row.exchangeShortName)
            else:
                tempdf = pd.DataFrame()
                tempdf['date'] = tempfund['date']
                # need to lag denominator for Assets, Investment and such [determined before t]

                ratioOpCalcDicts = {**BoMetric_base_dict, **BoMetric_mean_dict, **BoMetric_unity_dict, **BoMetric_diff_dict}
                for key in ratioOpCalcDicts:
                    restr = key
                    strUp = ratioOpCalcDicts[key]['Upper']
                    strDn = ratioOpCalcDicts[key]['Lower']
                    tf = cm.calc_simpleRatio(tempfund, strUp, strDn)
                    if key in BoMetric_base_dict:
                        tempMetric_df[restr] = tf
                    if key in BoMetric_mean_dict:
                        mrestr = "m" + restr[0].upper() + restr[1:]
                        tempMetric_df[mrestr] = tf
                    if key in BoMetric_unity_dict:
                        urestr = "u" + restr[0].upper() + restr[1:]
                        tempMetric_df[urestr] = tf
                    if key in BoMetric_diff_dict:
                        tempdf['forDiff'] = tf
                        tf = cm.calc_diff(tempdf,'forDiff',n)
                        drestr = "d" + restr[0].upper() + restr[1:]
                        tempMetric_df[drestr] = tf

                for key1 in BoMetric_special_dict.keys():
                    tf = cm.calc_special(tempfund, key1, n)
                    tempMetric_df[key1] = tf

                tempMetric_df_trimmed = tempMetric_df.drop(tempMetric_df.tail(4).index)

                # align schemas (preserve all columns) before concatenation to avoid losing columns
                cols_union = BoMetric_df.columns.union(tempMetric_df_trimmed.columns)
                BoMetric_df = BoMetric_df.reindex(columns=cols_union)
                tempMetric_df_trimmed = tempMetric_df_trimmed.reindex(columns=cols_union)
                # perform concat while suppressing the specific FutureWarning about
                # concatenation with empty / all-NA entries (make this local only)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
                    )
                    BoMetric_df = pd.concat([BoMetric_df, tempMetric_df_trimmed], ignore_index=True)
                # align schemas for cdx as well
                cols_union_cdx = cdx_df.columns.union(tempfund.columns)
                cdx_df = cdx_df.reindex(columns=cols_union_cdx)
                tempfund = tempfund.reindex(columns=cols_union_cdx)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
                    )
                    cdx_df = pd.concat([cdx_df, tempfund], ignore_index=True)

        if nrTaT > 0 and cntr == nrTaT:
            break
        elif len(tickersfailed) > (cntr + 1)*20:
            break
        pbar.update(n=1)
    pbar.close()

    #BoMetric_df = utils.setDatesToQuarterly(BoMetric_df)
    BoMetric_df, cdx_df = gdg.fixAfterGetData(BoMetric_df, cdx_df)

    # Materialize a USD market-cap column from the just-captured reportedCurrency, so the
    # saved cdx_df is self-describing for market-cap banding (carveOut.MCAP_BANDS). This
    # is the SAME shared FX path the band selection / grading use, so nothing can drift.
    # Guarded + best-effort: never blocks the fetch. No-op (all-NaN) on data lacking
    # reportedCurrency -- consumers then read every name as unknown-mcap (never misbanded).
    try:
        import carveOut as _co
        cdx_df['marketCap_usd'] = _co.marketcap_usd_series(cdx_df).values
    except Exception as _e:
        print(f'WARNING: marketCap_usd materialization skipped ({type(_e).__name__}: {_e})')

    resfunddic = {'BoMetric_df':BoMetric_df,
                  'cdx_df': cdx_df, 'tickersfailed': tickersfailed, 'lenfail': lenfail, 'pricefail': pricefail,
                  'datefail': datefail, 'emptyfail': emptyfail, 'cind': cntr, 'hasCurrentYear': hasCurrentYear}
    return resfunddic

def _align_statements_by_date(bs, inc, cf, km, fr):
    """R-E cross-statement date-join (design R-E, getData_fmp.py:48).

    The old fillPreReqdf assigned bs/inc/cf/km/fr columns by RangeIndex POSITION.
    Deep history makes ragged statement lengths (a missing/extra quarter in one
    statement) more likely, and positional assignment then MISPAIRS periods across
    statements, corrupting every cross-statement ratio.  This re-aligns each
    statement to the balance-sheet reference dates by ACTUAL date.

    IDENTITY GUARANTEE (bit-for-bit, behaviour-preserving on the common case): when
    every statement carries the SAME, duplicate-free, identically-ordered date
    vector as bs -- which is the well-formed case for essentially all tickers at the
    current 24-quarter depth -- reindexing to bs['date'] returns each row in the same
    position as the old positional code, so tempfund is byte-identical.  The join
    only DIFFERS when statements are ragged (the bug it fixes).

    SAFETY FALLBACK: if any statement (incl. bs) has duplicate or unusable dates,
    date alignment is ambiguous -> fall back to the original POSITIONAL behaviour for
    that ticker (never worse than today, and the row count is never changed).

    Returns (aligned_dict, used_date_join: bool).  aligned_dict maps
    'bs'/'inc'/'cf'/'km'/'fr' -> a frame reindexed to bs['date'] (or the raw frame on
    fallback).
    """
    stmts = {'bs': bs, 'inc': inc, 'cf': cf, 'km': km, 'fr': fr}
    ref_dates = pd.to_datetime(bs['date'], errors='coerce')
    # unusable if bs dates are non-unique or all-NaT -> positional fallback
    if ref_dates.isna().all() or ref_dates.duplicated().any():
        return stmts, False
    aligned = {}
    for name, sdf in stmts.items():
        if 'date' not in sdf.columns:
            return stmts, False
        s = sdf.copy()
        s_dates = pd.to_datetime(s['date'], errors='coerce')
        if s_dates.duplicated().any():
            # ambiguous mapping for this statement -> positional fallback (whole ticker)
            return stmts, False
        s.index = s_dates
        aligned[name] = s.reindex(ref_dates)
    return aligned, True


def fillPreReqdf(tempfund,preReq_dict,bs,inc,cf,km,fr):
    hcybool = False
    aligned, used_join = _align_statements_by_date(bs, inc, cf, km, fr)
    for key1 in preReq_dict:
        for i in preReq_dict[key1]:
            if key1 == 'bs':
                tempfund[i] = aligned['bs'][i].values if used_join else bs[i]
            elif key1 == 'inc':
                tempfund[i] = aligned['inc'][i].values if used_join else inc[i]
            elif key1 == 'cf':
                tempfund[i] = aligned['cf'][i].values if used_join else cf[i]
            elif key1 == 'km':
                tempfund[i] = aligned['km'][i].values if used_join else km[i]
            elif key1 == 'fr':
                tempfund[i] = aligned['fr'][i].values if used_join else fr[i]
            else:
                #tempfund['shares'] = inc['revenue'] / km['revenuePerShare']
                # PRICE = marketCap / weightedAverageShsOut  (audit C-2 fix, 2026-07-19).
                #
                # It used to be derived as quarterly-PE * quarterly-EPS:
                #   fr['priceEarningsRatio'] * (inc['netIncome'] / inc['weightedAverageShsOut'])
                # but FMP's QUARTERLY priceEarningsRatio is ANNUALISED (price / TTM-ish EPS)
                # while the EPS factor here is a SINGLE quarter, so the product came out at
                # ~1/4 of the real share price.  Proven on the 2026-07-17 panel:
                # marketCap / (price * shares) had median 3.99992 and 69% of all 176,193
                # usable rows inside +-1% of exactly 4.0.
                #
                # The damage was NOT the uniform scale (z-scored metrics are invariant to
                # it) but the Stage-1 Tier-S UNITY test grahamNumber/price > 1, which is
                # scale-SENSITIVE: on the same panel it passed 70.6% of rows on the divided
                # price vs 13.4% on marketCap/shares -- a weight-1.0 criterion that had
                # degenerated into "almost everyone passes".
                #
                # marketCap and weightedAverageShsOut are both already fetched, are in the
                # company's own reporting currency (so the ratio is currency-consistent) and
                # are both as-of the statement period end (verified: 24 distinct marketCap
                # values per source, no lookahead), which is exactly the as-of convention
                # every price-based metric here assumes.
                #
                # +-inf (shares == 0) is normalised to NaN: an undefined price must read as
                # MISSING to checkIfValidFS, which runs on tempfund before forceNumOnDf's
                # inf->NaN sweep.
                if used_join:
                    _km, _inc = aligned['km'], aligned['inc']
                    _price = (_km['marketCap'].values
                              / _inc['weightedAverageShsOut'].values)
                else:
                    _price = km['marketCap'] / inc['weightedAverageShsOut']
                tempfund['price'] = pd.Series(_price).replace([np.inf, -np.inf], np.nan).values

    # GRAHAM NUMBER, computed in-pipeline (review H2 fix, 2026-07-25).
    #
    # FMP's quarterly `grahamNumber` is sqrt(22.5 * EPS_QUARTERLY * BVPS), i.e. HALF the
    # published sqrt(22.5 * EPS_ANNUAL * BVPS) -- proven on the 2026-07-17 panel:
    #   median( FMP graham / sqrt(22.5 * netIncomePerShare_q * bookValuePerShare) )
    #     = 1.0000 with 79.5% of 110,264 rows inside 1%, versus 0.5000 against the
    #     4x-EPS (annualised) form.
    # With `price` fixed to the real share price, the weight-1.0 Tier-S UNITY test
    # `grahamNumber/price > 1` therefore went from 2x too LOOSE (70.6% pass on the old
    # divided price) to 2x too STRICT (13.4%); calibrated is ~42-43%.  Rescaling FMP's
    # field would work numerically, but computing the number outright is the honest fix
    # and removes the dependency on an undocumented FMP convention that could change.
    #
    # EPS_ttm = netIncome_ttm / weightedAverageShsOut(current row), NOT the sum of four
    # quarterly netIncomePerShare values: each quarter's per-share figure uses its OWN
    # share count, so summing them mixes share bases, whereas one TTM earnings total over
    # the current share count is a single consistent basis -- and it is the SAME basis as
    # `price` (marketCap/weightedAverageShsOut) that this ratio is compared against.
    #
    # TTM sums are taken over the SAME set of rows for both inputs (the ttm_aligned_sums
    # convention): a row's TTM is NaN unless all 4 of its trailing quarters are present,
    # so a gap yields "not computable" rather than a 3-quarter sum masquerading as a year.
    #
    # Graham is UNDEFINED for negative earnings or negative book value (the sqrt has no
    # real root and the screen is a value floor for profitable, asset-backed firms), so
    # EPS_ttm <= 0 or BVPS <= 0 -> NaN.  NaN, not 0: Stage-1 scores NaN as a FAIL of this
    # criterion, which is the correct reading of "no Graham floor exists here"; a 0 would
    # be a real computed value that happens to fail.
    #
    # SEMI-ANNUAL CAVEAT (audit C-1): FMP labels an H1/H2 filer's halves as Q2/Q4, so a
    # 4-row trailing window is TWENTY-FOUR months for those names and this EPS_ttm is
    # correspondingly ~2x. That cannot be resolved until `period` is captured and read
    # (fix 14 captures it; nothing consumes it yet). Revisit with the Piotroski lag.
    _ni = pd.to_numeric(tempfund.get('netIncome'), errors='coerce')
    _sh = pd.to_numeric(tempfund.get('weightedAverageShsOut'), errors='coerce')
    _bvps = pd.to_numeric(tempfund.get('bookValuePerShare'), errors='coerce')
    # tempfund is NEWEST-FIRST here (raw FMP order), so a forward-looking rolling sum on
    # the reversed series gives each row the sum of ITSELF plus the 3 older quarters.
    _pair = pd.concat([_ni, _sh], axis=1)
    _pair = _pair.where(_pair.notna().all(axis=1))       # aligned rows only
    _ni_ttm = _pair.iloc[::-1, 0].rolling(4).sum().iloc[::-1]
    _eps_ttm = _ni_ttm / _sh
    _graham = np.sqrt(22.5 * _eps_ttm.where(_eps_ttm > 0)
                      * _bvps.where(_bvps > 0))
    tempfund['grahamNumber'] = _graham.replace([np.inf, -np.inf], np.nan).values

    # Keep the RAW fiscal period-end date BEFORE setDatesToQuarterly overwrites `date`
    # with a quarter-start stamp (audit H-2 fix, 2026-07-19).  `date` is deliberately left
    # exactly as it is -- every downstream consumer (the cross-statement date join, the
    # forensic YoY shifts, CycleHeat's restatement tie-break, data_quality's row matching)
    # keys off the quarterly stamp -- so this is ADDITIVE.  The quarter stamp is lossy in
    # the two ways that matter: 52/53-week fiscal drift collapses two different period ends
    # onto ONE quarter (282 sources carry duplicate quarters), and a fiscal year that does
    # not align to calendar quarters cannot be recovered from it afterwards.
    tempfund['periodEndDate'] = tempfund['date'].values
    tempfund = utils.setDatesToQuarterly(tempfund)
    if tempfund['date'].iloc[0].year == datetime.today().year:
        hcybool = True

    return tempfund, hcybool

def getFsData_fmp(ticker, period, limit, baseurl, api_key,compyear, tickersfailed, lenfail,datefail,emptyfail,
                  dead_path=False, http_get=None):
    """Fetch the 5 statement endpoints for one ticker and apply the gates.

    dead_path : forwarded to testForAPIFaults_fmp -- on the DELISTED-ENTITY
        ingestion path it BYPASSES the datefail gate (F-A) and RELAXES the >=16q
        lenfail gate (F-B) so dead names are not silently dropped.  Default False
        keeps the live path bit-for-bit.
    http_get : optional injected HTTP getter for offline testing.  When None (the
        LIVE overnight fetch path), it defaults to gdg.safe_http_get -- a bounded
        (timeout + retry/backoff) getter -- so a single stalled/hung FMP endpoint
        cannot hang the ~12h run.  This is BEHAVIOUR-PRESERVING on the happy path:
        safe_http_get returns the SAME requests.Response for a healthy 200 (only a
        10s timeout is added, which a healthy endpoint never trips), so parsed
        fundamentals are byte-identical to the old bare requests.get.  On a
        persistent timeout/connection-error or a retryable 5xx/429 that survives
        retries, safe_http_get hands back a FAILING Response (status_code in the
        400-599 failcodes, or a _FailedResponse(599)); the existing failcode gate
        then records the ticker as a normal fetch failure (tickersfailed) and the
        loop CONTINUES to the next ticker -- it never raises/aborts the run.
    """
    if http_get is None:
        http_get = gdg.safe_http_get
    failcodes = list(range(400, 600))
    failbool, whyfail, outdic = ft.testForAPIFaults_fmp(failcodes,compyear,ticker,period,limit,baseurl,api_key,
                                                        dead_path=dead_path, http_get=http_get)
    if failbool:
        tickersfailed.append(ticker)
        if whyfail == 'datefail':
            datefail.append(ticker)
        elif whyfail == 'lenfail':
            lenfail.append(ticker)
        elif whyfail == 'emptyfail':
            emptyfail.append(ticker)
        km, fr, inc, bs, cf = -37707, -1, -1, -1, -1
    else:
        km = outdic['km'] #pd.DataFrame.from_dict(resp_km.json())
        fr = outdic['fr'] #pd.DataFrame.from_dict(resp_fr.json())
        inc = outdic['inc'] #pd.DataFrame.from_dict(resp_inc.json())
        bs = outdic['bs'] #pd.DataFrame.from_dict(resp_bs.json())
        cf = outdic['cf'] #pd.DataFrame.from_dict(resp_cf.json())

    return km, fr, inc, bs, cf, tickersfailed, lenfail, datefail, emptyfail

def symbchRestock(tckrs_df,baseurl,period,limit,api_key,compyear,timdir='old2new'):
    symbch_df = pd.DataFrame(requests.get(f'https://financialmodelingprep.com/api/v4/symbol_change?apikey={api_key}').json())
    if timdir == 'new2old':
        int = list(set(symbch_df.newSymbol) & set(tckrs_df.symbol))
    else:
        int = list(set(symbch_df.oldSymbol) & set(tckrs_df.symbol))

    succbool_lvl2 = []
    failcodes = list(range(400, 600))
    failers = []
    failers_lvl2 = []
    succnotadded = []
    succ_lvl2 = []
    failstosucc = []

    print(f'Starting symbol restock: {timdir}')
    pbar = tqdm(total=len(int))
    for ticker in int:
        failbool_lvl2_agg = False
        resp_km = requests.get(f'{baseurl}/key-metrics/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_fr = requests.get(f'{baseurl}/ratios/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_inc = requests.get(f'{baseurl}/income-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_bs = requests.get(f'{baseurl}/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_cf = requests.get(f'{baseurl}/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        respstatcodes = [resp_km.status_code, resp_fr.status_code, resp_inc.status_code, resp_bs.status_code,
                         resp_cf.status_code]
        failbool, whyfail = ft.testForAPIFaults(respstatcodes, failcodes,compyear, resp_km, resp_fr, resp_inc, resp_bs, resp_cf)
        if failbool:
            failers.append(ticker)
            if timdir == 'new2old':
                nt_df = symbch_df[symbch_df['newSymbol'] == ticker]['oldSymbol']
            else:
                nt_df = symbch_df[symbch_df['oldSymbol'] == ticker]['newSymbol']
            for i in range(0, len(nt_df)):
                ticker_next = nt_df.iloc[i]
                resp_km = requests.get(f'{baseurl}/key-metrics/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_fr = requests.get(f'{baseurl}/ratios/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_inc = requests.get(f'{baseurl}/income-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_bs = requests.get(f'{baseurl}/balance-sheet-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_cf = requests.get(f'{baseurl}/cash-flow-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                respstatcodes = [resp_km.status_code, resp_fr.status_code, resp_inc.status_code, resp_bs.status_code,resp_cf.status_code]
                failbool_lvl2, whyfail = ft.testForAPIFaults(respstatcodes, failcodes,compyear, resp_km, resp_fr, resp_inc, resp_bs,resp_cf)
                if failbool_lvl2:
                    failbool_lvl2_agg = True
                    failers_lvl2.append(ticker_next)
                elif ticker_next not in list(tckrs_df['symbol']):
                    tckrs_df.loc[tckrs_df['symbol'] == ticker, 'symbol'] = ticker_next
                    failstosucc.append(ticker_next)
                else:
                    succnotadded.append(ticker_next)
            if failbool_lvl2_agg == False:
                succ_lvl2.append(ticker)
        pbar.update(n=1)
    pbar.close()
    fullfail = list(set(failers + failers_lvl2))
    tckrs_df_new = tckrs_df[~tckrs_df['symbol'].isin(fullfail)].reset_index(drop=True)

    if len(failers) > 0:
        pcfixed = len(succ_lvl2)/len(failers)
        pcnotadded = len(succnotadded)/len(failers)

    return tckrs_df_new, failers, failers_lvl2, succ_lvl2, succnotadded, failstosucc, pcfixed, pcnotadded

def initTempMets(dfcols,cdxcols,datevec,ticker):
    tempMetric_df = pd.DataFrame(columns=dfcols)
    tempfund = pd.DataFrame(columns=cdxcols)
    tempfund['date'] = datevec
    tempfund['source'] = ticker
    tempMetric_df['date'] = datevec
    tempMetric_df['source'] = ticker

    return tempfund, tempMetric_df