import sys
import configuration as cf
import utils as utils
import getData_gen as gdg
import getData_fmp as gdf
import calcScore as csf
import postBo as pb
import detectManipulation as dm
import shutil
import os
import glob
from pathlib import Path

def transfer_outputs_to_drive(transfer_dir, configdic, verbose=True):
    """
    Copy run output artifacts to a Google-Drive-synced folder at end-of-run.

    Args:
        transfer_dir: str, path to target directory (None/empty = skip transfer)
        configdic: dict, the configuration dictionary (contains datasource, tickerfilter, ingest_delisted flags)
        verbose: bool, log what was copied

    Returns:
        dict with transfer_result={'status': 'success'/'skipped'/'warning',
                                    'copied_files': N,
                                    'total_size_mb': X,
                                    'message': str}
    """
    result = {
        'status': 'skipped',
        'copied_files': 0,
        'total_size_mb': 0.0,
        'message': 'transfer_dir not set; transfer disabled',
        'files_list': []
    }

    if not transfer_dir:
        if verbose:
            print("[TRANSFER] Skipped (transfer_dir not set)")
        return result

    # DENYLIST: patterns that must NEVER be copied
    denylist_patterns = ['*key*', '*pem', 'fmpAPIkey.txt']

    # ALLOWLIST: explicit patterns to copy
    allowlist_patterns = [
        'Bometric_dic-*.pickle',
        'Boresults_dic-*.pickle',
        'postRank_*.pickle',
        'AggScoreTop*.csv',
        'PresentationTop*.xlsx',
        'ForensicFlagsTop*.csv',
        'real_prices.csv'
    ]

    # Always include run_logs and price_data directories
    allowlist_dirs = [
        'run_logs',
        'baseline_tools/price_data'
    ]

    # If -ingest_delisted ran, also include delisted outputs
    if configdic.get('ingest_delisted'):
        allowlist_dirs.append('delisted_out')

    # Check if transfer_dir path exists; if parent exists, try to create it
    transfer_path = Path(transfer_dir)
    parent_path = transfer_path.parent

    if not parent_path.exists():
        if verbose:
            print(f"[TRANSFER] WARNING: Parent directory does not exist: {parent_path}")
            print(f"[TRANSFER]   (Drive not mounted or path invalid? Skipping transfer.)")
        result['status'] = 'warning'
        result['message'] = f'Parent directory does not exist; skipping transfer'
        return result

    # Try to create the target directory
    try:
        transfer_path.mkdir(parents=False, exist_ok=True)
        if verbose:
            print(f"[TRANSFER] Target directory created/exists: {transfer_path}")
    except Exception as e:
        if verbose:
            print(f"[TRANSFER] WARNING: Could not create target directory: {e}")
        result['status'] = 'warning'
        result['message'] = f'Could not create target directory: {e}'
        return result

    # Helper to check if a file matches the denylist
    def is_denied(filename):
        fname_lower = filename.lower()
        for pattern in denylist_patterns:
            if '*' in pattern:
                # Simple wildcard matching
                prefix = pattern.replace('*', '')
                if prefix in fname_lower:
                    return True
            else:
                if fname_lower == pattern.lower():
                    return True
        return False

    # Copy files matching allowlist patterns
    copied_files = []
    total_size = 0

    for pattern in allowlist_patterns:
        matched_files = glob.glob(pattern)
        for fpath in matched_files:
            if is_denied(fpath):
                if verbose:
                    print(f"[TRANSFER] DENIED (denylist): {fpath}")
                continue

            try:
                dest = transfer_path / Path(fpath).name
                shutil.copy2(fpath, str(dest))
                size = os.path.getsize(str(dest)) / (1024 * 1024)  # MB
                total_size += size
                copied_files.append(fpath)
                if verbose:
                    print(f"[TRANSFER] Copied: {fpath} ({size:.2f} MB)")
            except Exception as e:
                if verbose:
                    print(f"[TRANSFER] ERROR copying {fpath}: {e}")

    # Copy directories (with contents into matching subfolder)
    for dirpat in allowlist_dirs:
        if not os.path.isdir(dirpat):
            if verbose:
                print(f"[TRANSFER] Skipped (not found): {dirpat}/")
            continue

        # Check for denylist in directory contents
        has_denied = False
        for root, dirs, files in os.walk(dirpat):
            for fname in files:
                if is_denied(fname):
                    if verbose:
                        print(f"[TRANSFER] DENIED (denylist): {os.path.join(root, fname)}")
                    has_denied = True

        if has_denied:
            continue

        # Copy the directory
        try:
            dest_dir = transfer_path / dirpat
            if dest_dir.exists():
                shutil.rmtree(str(dest_dir))

            shutil.copytree(dirpat, str(dest_dir))

            # Calculate total size
            for root, dirs, files in os.walk(str(dest_dir)):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    size = os.path.getsize(fpath) / (1024 * 1024)  # MB
                    total_size += size

            copied_files.append(dirpat)
            if verbose:
                print(f"[TRANSFER] Copied directory: {dirpat}/")
        except Exception as e:
            if verbose:
                print(f"[TRANSFER] ERROR copying directory {dirpat}: {e}")

    # Assert that the key file was NOT copied
    key_file_in_target = transfer_path / 'fmpAPIkey.txt'
    if key_file_in_target.exists():
        if verbose:
            print(f"[TRANSFER] CRITICAL ERROR: fmpAPIkey.txt was copied! Removing it.")
        try:
            key_file_in_target.unlink()
        except Exception as e:
            if verbose:
                print(f"[TRANSFER] ERROR removing key file: {e}")
            result['status'] = 'error'
            result['message'] = 'Key file was mistakenly copied and could not be removed'
            return result

    result['status'] = 'success'
    result['copied_files'] = len(copied_files)
    result['total_size_mb'] = total_size
    result['files_list'] = copied_files
    result['message'] = f"Transferred {len(copied_files)} items ({total_size:.2f} MB) to {transfer_dir}"

    if verbose:
        print(f"[TRANSFER] Success: {len(copied_files)} items, {total_size:.2f} MB total")
        print(f"[TRANSFER] Destination: {transfer_dir}")

    return result

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
    import backtest_unified as bt
    import data_quality as dq
    #import warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    args = sys.argv[1:]

    # Assign parameters
    configdic = cf.getDataFetchConfiguration(args)
    # Point-in-time as-of date D (default None = today / live; reproduces current
    # behaviour bit-for-bit).  Threaded into the universe build and the scorer.
    as_of = configdic.get('as_of', None)
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
                                     sfilt ='all', mcapf = -1, fn = '', as_of=as_of)
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

        # Apply data quality filter to freshly fetched data
        datandmetricdic = dq.apply_data_quality_filter(datandmetricdic, verbose=True, save_log=True)

        #write to info to file
        utils.write_lastIndexRead(configdic['lastindex_fn'], getfunddic['cind'])
        utils.writeManElimToFile(datandmetricdic,newmanelimtckrs)
        # Save results if saveBoMetric == 1
        if saveBoMetricbool:
            utils.saveWrapper('metric', datandmetricdic)
    else:
        loadmetricdic = {'loadBoMetric': loadBoMetricbool, 'loadBoMetricfname': configdic['loadBoMetricfname']}
        datandmetricdic = utils.loadWrapper('metric', loadmetricdic)

    # Apply data quality filter (remove corrupted/invalid price data)
    # This must happen BEFORE any scoring to prevent garbage from affecting calculations
    datandmetricdic = dq.apply_data_quality_filter(datandmetricdic, verbose=True, save_log=True)

    if portfoliotestyear > 0:
        datandmetricdic = pf.portfolioBacktestWrapper(portfoliotestyear,datandmetricdic)

    else:
        if not loadBoResultbool:
            resdic = pb.postBoWrapper(datandmetricdic, as_of=as_of)
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

    # Merge moatScore into postRank
    if 'postRank' in resdic and not moatdf.empty:
        moat_merge = moatdf[['source', 'moatScore']].copy()
        resdic['postRank'] = resdic['postRank'].merge(moat_merge, on='source', how='left')

    detmandic = dm.detectManipulationWrapper(resdic)
    resdic = {**resdic, **detmandic}

    print(resdic['postRank'].head(50))

    pb.writeResWrapper(resdic)
    
    # Save postRank to pickle for backtesting
    from datetime import datetime
    postrank_fname = f"postRank_{datetime.today().strftime('%Y-%m-%d')}_{configdic['datasource']}_{configdic['tickerfilter']}.pickle"
    postrank_data = {
        'postRank': resdic['postRank'],
        'cdx_df': resdic['cdx_df'],
        'moatdf': moatdf,
        'date_created': datetime.today().strftime('%Y-%m-%d'),
        'datasource': configdic['datasource'],
        'tickerfilter': configdic['tickerfilter']
    }
    import pandas as pd
    pd.to_pickle(postrank_data, postrank_fname)
    print(f"PostRank saved to: {postrank_fname}")

    # ---- Delisted-entity (survivorship) ingestion -- GATED, default OFF ----
    # ACQUIRES survivorship data (registry + dead fundamentals + dead prices) and
    # stores it for later point-in-time analysis.  It runs AFTER the live results
    # are saved above, so the live top-20 is delivered first and is NEVER affected
    # by (or delayed-into) the ingestion.  When -ingest_delisted is OFF the module
    # is not even imported -> the live path is bit-for-bit unchanged.
    if configdic.get('ingest_delisted'):
        import delisted_ingest as di
        # reuse the already-built live universe symbols when available (avoids a
        # duplicate universe fetch); else run_ingest builds it itself.
        live_syms = None
        try:
            live_syms = list(resdic['Tickers_df']['symbol'])
        except Exception:
            live_syms = None
        di.run_ingest(configdic, live_symbols=live_syms)

    # Optional: Run unified backtesting if flag is set
    run_backtest = configdic.get('runbacktest', 0)
    if run_backtest:
        print("\n" + "="*70)
        print("RUNNING UNIFIED BACKTESTING")
        print("="*70)
        bt.run_all(
            dmdic=resdic,
            buy_years=configdic.get('backtest_buy_years', None),
            eval_years_list=configdic.get('backtest_eval_years', None),
            topn=configdic.get('backtest_topn', 100),
            verbose=True,
            save_results=True
        )

    # ---- End-of-run transfer to Google-Drive-synced folder (GATED, default OFF) ----
    # When -transfer_dir is set, copy the output allowlist there after ALL outputs
    # are written and AFTER ingestion completes. Gracefully skip if path doesn't exist.
    transfer_dir = configdic.get('transfer_dir')
    if transfer_dir:
        print("\n" + "="*70)
        print("END-OF-RUN TRANSFER TO GOOGLE DRIVE")
        print("="*70)
        transfer_result = transfer_outputs_to_drive(transfer_dir, configdic, verbose=True)
        print(f"Transfer result: {transfer_result['message']}")
        # Log the transfer result into a summary line if desired
    else:
        transfer_dir = None  # Explicitly set for consistency

    return None

if __name__ == '__main__':
    main()



