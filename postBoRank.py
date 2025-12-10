import createDicts as cdic
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def postBoScoreRanking(bmtop,bstop,cdxtop,baseurl,api_key,period='quarter',nq=16):
    import sys
    print('Ranking the top 100 stocks, according to BoScore.')
    sys.stdout.flush()  # Ensure output is printed before progress bar
    
    # Diagnostic: Check input dataframes BEFORE any calculations
    print("\n" + "="*60, flush=True)
    print("DIAGNOSTIC: Input dataframes check (BEFORE calculations)", flush=True)
    print("="*60, flush=True)
    
    # Check bmtop (BoMetric top 100)
    if bmtop.empty:
        print("ERROR: bmtop (BoMetric top 100) is EMPTY!", flush=True)
    else:
        print(f"bmtop shape: {bmtop.shape} (rows, columns)", flush=True)
        print(f"bmtop unique sources: {bmtop['source'].nunique() if 'source' in bmtop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in bmtop.columns:
            print(f"bmtop sample sources: {list(bmtop['source'].head(3).values)}", flush=True)
        print(f"bmtop columns (first 10): {list(bmtop.columns[:10])}", flush=True)
        # Check for NaN in numeric columns
        numeric_cols = bmtop.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            nan_pct = (bmtop[numeric_cols].isna().sum() / len(bmtop) * 100).round(1)
            print(f"bmtop NaN percentage in numeric columns (first 5): {dict(nan_pct.head(5))}", flush=True)
    
    # Check bstop (BoScore top 100)
    if bstop.empty:
        print("ERROR: bstop (BoScore top 100) is EMPTY!", flush=True)
    else:
        print(f"\nbstop shape: {bstop.shape} (rows, columns)", flush=True)
        print(f"bstop unique sources: {bstop['source'].nunique() if 'source' in bstop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in bstop.columns:
            print(f"bstop sample sources: {list(bstop['source'].head(3).values)}", flush=True)
        if 'score' in bstop.columns:
            print(f"bstop score stats: min={bstop['score'].min():.4f}, max={bstop['score'].max():.4f}, mean={bstop['score'].mean():.4f}", flush=True)
    
    # Check cdxtop (cdx top 100) - this is critical as many metrics depend on it
    if cdxtop.empty:
        print("\nERROR: cdxtop (cdx top 100) is EMPTY!", flush=True)
    else:
        print(f"\ncdxtop shape: {cdxtop.shape} (rows, columns)", flush=True)
        print(f"cdxtop unique sources: {cdxtop['source'].nunique() if 'source' in cdxtop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in cdxtop.columns:
            print(f"cdxtop sample sources: {list(cdxtop['source'].head(3).values)}", flush=True)
        print(f"cdxtop columns (first 10): {list(cdxtop.columns[:10])}", flush=True)
        # Check for required columns that will be used in calculations
        # Including columns needed for Altman-Z and Piotroski calculations
        required_cols = ['freeCashFlow', 'weightedAverageShsOut', 'marketCap', 'grahamNumber', 'price', 
                         'tangibleBookValuePerShare', 'totalAssets', 'totalLiabilities', 'totalCurrentAssets',
                         'totalCurrentLiabilities', 'totalStockholdersEquity', 'operatingIncome', 'revenue',
                         'netIncome', 'netCashProvidedByOperatingActivities', 'longTermDebt', 'currentRatio',
                         'grossProfitMargin']
        missing_cols = [col for col in required_cols if col not in cdxtop.columns]
        if missing_cols:
            print(f"WARNING: cdxtop missing required columns: {missing_cols}", flush=True)
        else:
            print(f"cdxtop has all required columns: {required_cols}", flush=True)
        # Check for NaN in key columns
        key_cols = [col for col in required_cols if col in cdxtop.columns]
        if key_cols:
            nan_pct = (cdxtop[key_cols].isna().sum() / len(cdxtop) * 100).round(1)
            print(f"cdxtop NaN percentage in key columns: {dict(nan_pct)}", flush=True)
    
    print("="*60 + "\n", flush=True)
    sys.stdout.flush()  # Final flush before starting calculations
    
    #test
    #bmtop = BoM_dftop100
    #bstop = BoS_dftop100
    #cdxtop = cdx_dftop100
    #period='quarter'
    #nq = 12
    #baseurl = configdic['baseurl']
    #api_key = configdic['api_key']
    #test
    postBmRankingDict, postNewRankingDict = cdic.getPostDict()
    postScoreMetric_df = pd.DataFrame()
    postScoreMetric_df['source'] = bstop['source']
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postBmRankingDict.keys())], axis=1)
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postNewRankingDict.keys())], axis = 1)
    #postScoreMetric_df.drop(labels=['SalePerEmployee'],axis=1,inplace=True)
    #postScoreMetric_df['BoScore'] = bstop['score']*0.2
    postRanking_df = pd.DataFrame()
    weight_df = pd.DataFrame()
    # Build a stable weight mapping from the post dictionaries so we always have a weight for each metric
    postBmRankingDict_local, postNewRankingDict_local = cdic.getPostDict()
    weight_series = {**{k: postBmRankingDict_local[k]['w'] for k in postBmRankingDict_local},
                     **{k: postNewRankingDict_local[k]['w'] for k in postNewRankingDict_local}}
    #weight_df['BoScore'] = pd.Series(0.2)
    weightzerobool = False
    mcapAve = cdxtop.marketCap.mean()
    cdxtop['mcapQuants'] = (-1)*((pd.qcut(cdxtop['marketCap'], 4).cat.codes/(3) - 0.5))
    #mcapAve = 70310173993
    tempcntr = 0

    # Note: Bulk endpoints require higher subscription tier, using individual API calls only
    dcf_bulk_dict = {}
    scores_bulk_dict = {}

    pbar = tqdm(total=len(bstop['source'].unique()))
    for ticker in bstop['source']:
        tempcdx = cdxtop.loc[cdxtop['source'] == ticker]
        tempfcf = tempcdx.freeCashFlow
        tempshares = tempcdx.weightedAverageShsOut
        tempmcap = tempcdx.marketCap
        tempmcapQuants = tempcdx.mcapQuants.iloc[0]
        #tempcr = tempcdx.currentRatio

        #resp_fr = requests.get(f'{baseurl}v3/ratios/{ticker}?period={period}&limit={nq}&apikey={api_key}')
        #resp_km = requests.get(f'{baseurl}v3/key-metrics/{ticker}?period={period}&limit={nq}&apikey={api_key}')
        
        # DCF data (used for DcfToPrice metric)
        dcf_from_bulk = ticker in dcf_bulk_dict
        if dcf_from_bulk:
            dcf_data = [dcf_bulk_dict[ticker]]
            resp_dcf_status = "bulk"
        else:
            # Fallback to individual API call
            resp_dcf = requests.get(f'{baseurl}v3/discounted-cash-flow/{ticker}?apikey={api_key}')
            resp_dcf_status = resp_dcf.status_code
            try:
                dcf_data = resp_dcf.json() if resp_dcf.status_code == 200 else []
            except:
                dcf_data = []
        
        # Note: Altman-Z and Piotroski are now calculated from fundamentals (no API needed)
        # Financial growth still needs individual calls (no bulk endpoint available)
        resp_fg = requests.get(f'{baseurl}v3/financial-growth/{ticker}?limit={nq}&apikey={api_key}')
        # Note: Using price data from cdxtop dataframe instead of API call
        
        # Convert bulk data (already in dict format) or API response to DataFrame
        dcf = pd.DataFrame.from_dict(dcf_data) if dcf_data and isinstance(dcf_data, list) else pd.DataFrame()
        
        # Normalize DCF column names - bulk CSV might have different names than JSON API
        if not dcf.empty:
            # Map common variations to standard names
            column_mapping = {}
            for col in dcf.columns:
                col_lower = col.lower().replace(' ', '').replace('_', '')
                if col_lower == 'stockprice' or col_lower == 'stock_price':
                    column_mapping[col] = 'Stock Price'
                elif col_lower == 'dcf':
                    column_mapping[col] = 'dcf'
            if column_mapping:
                dcf = dcf.rename(columns=column_mapping)
        
        # Diagnostic for first ticker only
        if tempcntr == 0:
            print(f"\nDIAGNOSTIC: First ticker ({ticker}) data:", flush=True)
            print(f"  DCF source: {'bulk' if dcf_from_bulk else 'individual'}, status: {resp_dcf_status}, empty: {dcf.empty}, shape: {dcf.shape if not dcf.empty else 'N/A'}", flush=True)
            if not dcf.empty:
                print(f"  DCF columns: {list(dcf.columns)}", flush=True)
            print(f"  tempcdx (fundamentals) empty: {tempcdx.empty}, shape: {tempcdx.shape if not tempcdx.empty else 'N/A'}", flush=True)
            if not tempcdx.empty:
                print(f"  tempcdx columns: {list(tempcdx.columns[:5])}...", flush=True)
                print(f"  tempcdx sample values (first row):", flush=True)
                print(f"    freeCashFlow: {tempcdx['freeCashFlow'].iloc[0] if 'freeCashFlow' in tempcdx.columns else 'N/A'}", flush=True)
                print(f"    marketCap: {tempcdx['marketCap'].iloc[0] if 'marketCap' in tempcdx.columns else 'N/A'}", flush=True)
                print(f"    operatingIncome: {tempcdx['operatingIncome'].iloc[0] if 'operatingIncome' in tempcdx.columns else 'N/A'}", flush=True)
            if not dcf_from_bulk and resp_dcf_status != 200:
                print(f"  DCF error: {resp_dcf.text[:100] if 'resp_dcf' in locals() else 'N/A'}", flush=True)
            print(f"  Note: Altman-Z and Piotroski calculated from tempcdx fundamentals", flush=True)
        
        # Calculate metrics using tempcdx fundamentals and bstop scores
        for key1 in postBmRankingDict.keys():
            met = postBmRankingDict[key1]['eqMet']
            weight = postBmRankingDict[key1]['w']
            temp = bmtop[bmtop['source']==ticker].head(nq)
            if key1 == 'grahamNumberToPrice':
                tempgnprat = tempcdx['grahamNumber'] / tempcdx['price']
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempgnprat.head(nq).mean()
            elif key1 == 'bVpRatio':
                tempbvtop = 1/tempcdx[met]
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempbvtop.head(nq).mean()
            elif key1 == 'revenueGrowth':
                revPCTgr = tempcdx[met].pct_change(-4, fill_method=None)
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = revPCTgr.head(nq).mean()
            else:
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempcdx[met].head(nq).mean()
            weight_df[key1] = pd.Series(weight)

        for key2 in postNewRankingDict.keys():
            weightzerobool = False
            #if key2 == 'FCFperShare':
            #    weight = postNewRankingDict[key2]['w']
            #    weight_df[key2] = pd.Series(weight)
            #    #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = fr['freeCashFlowPerShare'].head(nq).mean()*weight
            #    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (tempfcf/tempshares).head(nq).mean()*weight

            if key2 == 'freeCashFlowYield':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = km['freeCashFlowYield'].head(nq).mean()*weight
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (tempfcf / tempmcap).head(nq).mean()
                #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = km.freeCashFlowYield.head(nq).mean() * weight

            if key2 == 'freeCashFlowPerShareGrowth':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                fcfps = tempfcf/tempshares
                fcfpsgr = fcfps.pct_change(-4, fill_method=None)
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = fcfpsgr.head(nq).mean()
                #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = km.freeCashFlowYield.head(nq).mean() * weight

            if key2 == 'EPStoEPSmean':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                eps = tempcdx['netIncome'] / tempcdx['weightedAverageShsOut']
                epsmean = eps.mean()
                a = 0.4
                tw = a*(1+(1-a) + (1-a)**2 + (1-a)**3)
                if all(eps.iloc[0:4] > 0):
                    epstoepsmean = epsmean - (a/tw)*(eps.iloc[0] + eps.iloc[1]*(1-a) +
                                                     eps.iloc[2]*(1-a)**2 + eps.iloc[3]*(1-a)**3)
                else:
                    epstoepsmean = 0

                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = epstoepsmean

            # Metrics that need dcf - only calculate if dcf is available
            if key2 == 'DcfToPrice':
                if not dcf.empty:
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    # DCF endpoint returns 'Stock Price' not 'price'
                    # Handle both JSON API format ('Stock Price') and CSV bulk format variations
                    price_col = None
                    if 'Stock Price' in dcf.columns:
                        price_col = 'Stock Price'
                    elif 'StockPrice' in dcf.columns:
                        price_col = 'StockPrice'
                    elif 'stock_price' in dcf.columns:
                        price_col = 'stock_price'
                    
                    if 'dcf' in dcf.columns and price_col:
                        temp = dcf['dcf'].head(nq).mean()
                        temp2 = dcf[price_col].iloc[0] if len(dcf) > 0 else None
                        if temp2 is not None and temp2 != 0:
                            postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (temp/temp2)
                        else:
                            postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan
                    else:
                        # Diagnostic: log missing columns for first ticker
                        if tempcntr == 0:
                            print(f"  WARNING: DCF missing required columns. Available: {list(dcf.columns)}, need: ['dcf', price_col]", flush=True)
                        postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan
                else:
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan

            #if key2 == 'currentRatio':
            #    weight = postNewRankingDict[key2]['w']
            #    weight_df[key2] = pd.Series(weight)
            #    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = tempcr.head(nq).mean()*weight

            if key2 == 'marketCapRevQuants':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (tempmcap.head(nq).mean()/mcapAve - 1)*weight
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = tempmcapQuants

            if key2 == 'tbVpRatio':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                tbtp = tempcdx['tangibleBookValuePerShare']/tempcdx['price']
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] =\
                    tbtp.head(nq).mean()

            # Calculate Altman-Z Score from available data (no API needed)
            # Z = 1.2×(WC/TA) + 1.4×(RE/TA) + 3.3×(EBIT/TA) + 0.6×(MVE/TL) + 1.0×(Sales/TA)
            if key2 == 'Altman-Z':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                try:
                    if len(tempcdx) >= 1:
                        # Use most recent data
                        curr = tempcdx.iloc[0]
                        ta = curr['totalAssets']
                        tl = curr['totalLiabilities']
                        
                        if ta > 0 and tl > 0:
                            # Working Capital / Total Assets
                            wc = curr['totalCurrentAssets'] - curr['totalCurrentLiabilities']
                            x1 = wc / ta
                            
                            # Retained Earnings / Total Assets (approximated using equity)
                            # RE is typically the largest component of stockholders' equity
                            re = curr['totalStockholdersEquity']
                            x2 = re / ta
                            
                            # EBIT / Total Assets
                            ebit = curr['operatingIncome']
                            x3 = ebit / ta
                            
                            # Market Value of Equity / Total Liabilities
                            mve = curr['marketCap']
                            x4 = mve / tl
                            
                            # Sales / Total Assets
                            sales = curr['revenue']
                            x5 = sales / ta
                            
                            altman_z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
                            postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = altman_z
                        else:
                            postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan
                    else:
                        postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan
                except Exception:
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan

            # Calculate Piotroski F-Score from available data (no API needed)
            # 9 binary criteria, each worth 1 point
            if key2 == 'Piotroski':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                try:
                    if len(tempcdx) >= 2:
                        curr = tempcdx.iloc[0]  # Most recent
                        prev = tempcdx.iloc[1]  # Previous period
                        
                        ta_curr = curr['totalAssets']
                        ta_prev = prev['totalAssets']
                        
                        if ta_curr > 0 and ta_prev > 0:
                            # Profitability (4 points)
                            # 1. ROA > 0
                            p1 = 1 if curr['netIncome'] / ta_curr > 0 else 0
                            # 2. Operating Cash Flow > 0
                            p2 = 1 if curr['netCashProvidedByOperatingActivities'] > 0 else 0
                            # 3. Change in ROA > 0
                            roa_curr = curr['netIncome'] / ta_curr
                            roa_prev = prev['netIncome'] / ta_prev
                            p3 = 1 if roa_curr > roa_prev else 0
                            # 4. Accrual: Cash Flow > Net Income
                            p4 = 1 if curr['netCashProvidedByOperatingActivities'] > curr['netIncome'] else 0
                            
                            # Leverage/Liquidity (3 points)
                            # 5. Decrease in Long-term Debt ratio
                            ltd_ratio_curr = curr['longTermDebt'] / ta_curr
                            ltd_ratio_prev = prev['longTermDebt'] / ta_prev
                            p5 = 1 if ltd_ratio_curr < ltd_ratio_prev else 0
                            # 6. Increase in Current Ratio
                            p6 = 1 if curr['currentRatio'] > prev['currentRatio'] else 0
                            # 7. No new shares issued
                            p7 = 1 if curr['weightedAverageShsOut'] <= prev['weightedAverageShsOut'] else 0
                            
                            # Operating Efficiency (2 points)
                            # 8. Higher Gross Margin
                            p8 = 1 if curr['grossProfitMargin'] > prev['grossProfitMargin'] else 0
                            # 9. Higher Asset Turnover
                            at_curr = curr['revenue'] / ta_curr
                            at_prev = prev['revenue'] / ta_prev
                            p9 = 1 if at_curr > at_prev else 0
                            
                            piotroski = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                            postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = piotroski
                        else:
                            postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan
                    else:
                        postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan
                except Exception:
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan

            if key2 == 'BoScore':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = bstop.loc[
                    bstop['source'] == ticker, 'score']

            # Metrics that need dcf - only calculate if dcf is available
            if key2 == 'priceGrowth':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                # Calculate priceGrowth from price data in cdxtop dataframe (no API call needed)
                if 'price' in tempcdx.columns and not tempcdx['price'].empty:
                    # Calculate percentage change (negative because we want growth, not decline)
                    # pct_change(-1) calculates change from previous period (going backwards in time)
                    price_growth = -tempcdx['price'].pct_change(-1, fill_method=None).head(nq).mean()
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = price_growth
                else:
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = np.nan

        #    if key2 == 'SalePerEmployee':
        #        #temp = emp['employeeCount'].head(nq)
        #        #temp2 = inc['revenue'].head(nq)
        #        #spe = temp2/temp
        #        #dspe = spe - spe.shift(-1)

        #        #if any(np.isinf(dspe)) or any(np.isnan(dspe)):
        #            #weightzerobool = True

        #        #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = dspe.mean()

        # Diagnostic: Check if any metrics were calculated for first ticker
        if tempcntr == 0:
            sample_metrics = ['RoA', 'earnYield', 'grahamNumberToPrice', 'freeCashFlowYield', 'BoScore', 'Altman-Z', 'Piotroski']
            print(f"\nDIAGNOSTIC: Sample metric values after calculation for {ticker}:", flush=True)
            for metric in sample_metrics:
                if metric in postScoreMetric_df.columns:
                    val = postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, metric].values
                    if len(val) > 0 and not pd.isna(val[0]):
                        print(f"  {metric}: {val[0]}", flush=True)
                    else:
                        print(f"  {metric}: NOT CALCULATED (NaN)", flush=True)
                else:
                    print(f"  {metric}: COLUMN NOT FOUND", flush=True)
            print(f"  Note: Altman-Z and Piotroski are now calculated from fundamentals (no API needed)", flush=True)

        tempcntr  = tempcntr + 1
        pbar.update(n=1)

    #normalize
    testpsm_df = postScoreMetric_df
    
    # Diagnostic check: Print statistics about postScoreMetric_df before normalization
    print("\n" + "="*60)
    print("DIAGNOSTIC: postScoreMetric_df statistics before normalizeAndDropNA")
    print("="*60)
    print(f"DataFrame shape: {postScoreMetric_df.shape} (rows, columns)")
    print(f"Total rows: {len(postScoreMetric_df)}")
    print(f"Columns: {list(postScoreMetric_df.columns)}")
    
    if not postScoreMetric_df.empty:
        metric_cols = [col for col in postScoreMetric_df.columns if col != 'source']
        print(f"\nMetric columns (excluding 'source'): {len(metric_cols)}")
        
        if len(metric_cols) > 0:
            # Convert to numeric for analysis
            numeric_df = postScoreMetric_df[metric_cols].apply(pd.to_numeric, errors='coerce')
            
            # NaN statistics per column
            print("\nNaN statistics per column:")
            nan_counts = numeric_df.isna().sum()
            nan_pct = (nan_counts / len(numeric_df) * 100).round(2)
            for col in metric_cols:
                print(f"  {col}: {nan_counts[col]} NaN ({nan_pct[col]}%)")
            
            # Overall statistics
            total_cells = len(numeric_df) * len(metric_cols)
            total_nan = numeric_df.isna().sum().sum()
            print(f"\nOverall: {total_nan}/{total_cells} NaN values ({total_nan/total_cells*100:.2f}%)")
            
            # Rows with all NaN
            rows_all_nan = (numeric_df.isna().sum(axis=1) == len(metric_cols)).sum()
            print(f"Rows with ALL metrics NaN: {rows_all_nan}/{len(numeric_df)} ({rows_all_nan/len(numeric_df)*100:.2f}%)")
            
            # Rows with at least one valid metric
            rows_some_valid = (numeric_df.isna().sum(axis=1) < len(metric_cols)).sum()
            print(f"Rows with at least one valid metric: {rows_some_valid}/{len(numeric_df)} ({rows_some_valid/len(numeric_df)*100:.2f}%)")
            
            # Mean, min, max for columns with valid data
            print("\nColumn statistics (for non-NaN values):")
            for col in metric_cols:
                col_data = numeric_df[col].dropna()
                if len(col_data) > 0:
                    print(f"  {col}: mean={col_data.mean():.4f}, min={col_data.min():.4f}, max={col_data.max():.4f}, count={len(col_data)}")
                else:
                    print(f"  {col}: ALL NaN (no valid values)")
        else:
            print("WARNING: No metric columns found!")
    else:
        print("WARNING: DataFrame is empty!")
    
    print("="*60 + "\n")
    
    postScoreMetric_df, outlierlist = normalizeAndDropNA(postScoreMetric_df)

    temp_normpsmdf_weighted = postScoreMetric_df.drop('source', axis=1)
    # Apply weights using the stable weight_series mapping; if a weight is missing, default to 1
    for col in temp_normpsmdf_weighted.columns:
        w = weight_series.get(col, 1)
        temp_normpsmdf_weighted[col] = postScoreMetric_df[col].values * w
    #psmdf_normalized = pd.concat([postScoreMetric_df[postScoreMetric_df.columns.difference(tempnum.columns)], temp_normpsmdf_weighted], axis=1)
    psmdf_normalized = pd.concat(
        [postScoreMetric_df[postScoreMetric_df.columns.difference(temp_normpsmdf_weighted.columns)], temp_normpsmdf_weighted], axis=1)


    postRank = getAggScore(psmdf_normalized)

    tmpcorr = np.corrcoef(list(postRank['BoScore'].values), list(postRank['AggScore'].values))
    BoAggCorr = tmpcorr[0,1]

    postRank = getRankOfRanks(postRank)
    plotRank = postRank
    plotRank['rankOfRanks'] = plotRank['rankOfRanks']/10
    plotRank['AggScore'] = plotRank['AggScore']/10
    mlist = list(set(plotRank.columns) - set(['source']))
    plotRank = postBoRankingPassFilter(plotRank,mlist,5,5)

    #finalPostRank_df = getFinalRank(postRank)

    #roror = getRankOfRankOfRanks(finalPostRank_df)

    pbar.close()
    #rankdic = {'finalBoRank_df': finalPostRank_df, 'postRank': postRank, 'postRankOfRanks': postRankOfRanks,
    #           'psmdf_normalized': psmdf_normalized, 'BoAggCorr': BoAggCorr, 'outlierlist': outlierlist,
    #           'roror': roror}
    rankdic = {'postRank': postRank, 'postScoreMetric': postScoreMetric_df,
               'psmdf_normalized': psmdf_normalized, 'BoAggCorr': BoAggCorr, 'outlierlist': outlierlist}

    return rankdic

def normalizeAndDropNA(df):
    df.reset_index(inplace=True, drop=True)
    
    # Check if dataframe is empty or has no metric columns
    if df.empty:
        print("Warning: Input dataframe is empty.")
        return df, []
    
    # Replace inf values with nan (modern approach without inplace)
    metric_cols = [col for col in df.columns if col != 'source']
    
    if len(metric_cols) == 0:
        print("Warning: No metric columns found (only 'source' column present).")
        return df, []
    
    df_clean = df.copy()
    # Suppress the FutureWarning about downcasting in replace()
    with pd.option_context('future.no_silent_downcasting', True):
        for col in metric_cols:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows only if ALL metric columns are NaN (completely invalid rows)
    # This is less aggressive than dropping rows with ANY NaN, preserving more data
    nan_counts = df_clean[metric_cols].isna().sum(axis=1)
    dropmask = nan_counts < len(metric_cols)  # Keep rows with at least one valid metric
    outlierlist = list(df_clean['source'][~dropmask].copy())
    dfnona = df_clean[dropmask].copy()
    
    # Guard: if all rows have NaN, return empty df with warning
    if dfnona.empty:
        print(f"Warning: All {len(df)} rows dropped due to NaN values (all metric columns were NaN). Returning empty dataframe.")
        return dfnona, list(df['source'])
    
    tempnum = dfnona.drop('source',axis=1).apply(pd.to_numeric, errors='coerce')
    # calculate the mean and standard deviation of each column (NaN values are skipped by default)
    colmeans = tempnum.mean()
    colstds = tempnum.std()
    # Handle division by zero: if std is 0 or NaN, set normalized values to 0
    colstds = colstds.replace(0, np.nan).fillna(1)  # Avoid division by zero
    # subtract the mean and divide by the standard deviation
    temp_normpsmdf = (tempnum - colmeans) / colstds
    # Fill remaining NaN values with 0 (for columns that were all NaN)
    temp_normpsmdf = temp_normpsmdf.fillna(0)
    dfnona[temp_normpsmdf.columns] = temp_normpsmdf
    mask = abs(temp_normpsmdf) > 4
    to_keep = (~mask).all(axis=1)  # Keep rows where ALL columns are within 4 std (stricter than original)
    dfnonanorm = dfnona[to_keep].copy()
    outlierlist = list(set(outlierlist + list(dfnona['source'][~to_keep])))
    
    # Guard: if filtering removed all rows, keep at least the top 20% (avoid empty result)
    if dfnonanorm.empty and len(dfnona) > 0:
        print(f"Warning: Outlier filtering (>4 std) dropped all {len(dfnona)} rows. Keeping top 20% by row count.")
        keep_count = max(1, len(dfnona) // 5)
        dfnonanorm = dfnona.head(keep_count).copy()

    return dfnonanorm, outlierlist

def getAggScore(df):
    #df['AggScore'] = np.nan
    cts = list(set(df.columns) - set(['source']))
    df['AggScore'] = df[cts].sum(axis=1)
    postRank = df
    postRank.sort_values(by='AggScore',ascending=False,inplace=True)
    postRank.reset_index(drop=True,inplace=True)

    #postRank = pd.DataFrame(columns=['source','AggScore'])
    #postRank['source'] = df['source']
    #postRank['BoScore'] = df['BoScore']
    #for i in range(0,len(postRank['source'])):
    #    rl = list(df.iloc[i,:])
    #    postRank.loc[postRank['source'] == rl[0], 'AggScore'] = sum(rl[1:])

    #postRank.dropna(inplace=True)
    #postRank.sort_values(by='AggScore',ascending=False,inplace=True)
    #postRank.reset_index(drop=True,inplace=True)

    return postRank

def getRankOfRanks(df):
    postRankOfRanks = pd.DataFrame()
    for col in df.columns:
        if col not in ['source']:
            postRankOfRanks[col + 'rank'] = df[col].rank(ascending=False,method='dense')

    cts = list(set(postRankOfRanks.columns) - set(['source']))
    df['rankOfRanks'] = postRankOfRanks[cts].sum(1).rank(ascending=True,method='dense')

    return df

def getRankOfRankOfRanks(df):
    roror = pd.DataFrame()
    roror['source'] = df['source']
    for col in df.columns:
        if col in ['rankOfRanks', 'AggScore']:
            roror[col + 'ror'] = df[col].rank(ascending=False,method='dense')

    roror['rankOfRanksOfRanks'] = roror.sum(1)
    roror['rankOfRanksOfRanks'] = roror['rankOfRanksOfRanks'].rank(ascending=True,method='dense')
    roror.sort_values(by='rankOfRanksOfRanks',inplace=True)

    return roror

def getFinalRank(pr_df,pror_df):
    tmpfpr_df = pd.DataFrame(columns=['source','AggScore'])
    tmpfpr_df['source'] = pr_df['source']
    tmpfpr_df['AggScore'] = pr_df['AggScore']
    finalPostRank_df = tmpfpr_df.merge(pror_df[['source','rankOfRanks','BoScorerank']], on='source',how='inner')

    return finalPostRank_df

def postBoRankingPassFilter(df,mlist,lco,hco):
    pf = df[~df[df.columns.intersection(mlist)].lt(lco).any(axis=1)]
    pf = pf[~pf[pf.columns.intersection(mlist)].gt(hco).any(axis=1)]
    pf.reset_index(inplace=True, drop=True)

    return pf