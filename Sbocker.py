"""
Sbocker.py -- the pipeline ENTRY POINT / orchestrator for the ValuationAnalysis
filter.  Full AS-IS + AS-INTENDED map: design/pipeline-reference.md.

LIVE / DECISIONAL stage order (drives the emitted top-20):
  1. fetch fundamentals (getData_fmp)      -> raw FMP data
  2. data-quality prune (data_quality)     -> cleaned cdx_df / BoMetric_df
  3. Stage-1 BoScore   (calcScore)          -> top-100 pool
  4. carve partition + issuer-dedup (carveOut) -> per-cohort pools
  5. Stage-2 AggScore  (postBo -> postBoRank.postBoScoreRanking, mu weights)
                                            -> general top-20 + 5 cohort side-lists
  6. forensic / moat / manipulation decoration (forensicFlags, detectManipulation)
  7. emit postRank pickle + presentation + append-only pick-log (pick_log)

Stage-2 metric formulas live ONCE in stage2_metrics.py, shared with the offline
reproduction (baseline_tools/stage2_pit.py) so the live scorer and the
validation gate can never silently drift apart.

Gated-OFF by default (hand-run offline harness, NOT part of the automatic run):
delisted-ingest, in-run backtest, Drive transfer, and the whole
baseline_tools/ validation / beat-rate / tuner stack.

PHASE-B SEAM (not built here): wiring the offline validation/analysis INTO this
run belongs after the emission stages -- keep that boundary clean.
"""

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
import transfer_utils as tu

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

    # DENYLIST: patterns that must NEVER be copied.  Sourced from transfer_utils so
    # the end-of-run path and the incremental per-phase path share ONE denylist.
    denylist_patterns = tu.DENYLIST_PATTERNS

    # ALLOWLIST: explicit patterns to copy
    allowlist_patterns = [
        'Bometric_dic-*.pickle',
        'Boresults_dic-*.pickle',
        'postRank_*.pickle',
        'AggScoreTop*.csv',
        'PresentationTop*.xlsx',
        'ForensicFlagsTop*.csv',
        'real_prices.csv',
        'SideList_*.csv',
        'RawMetricsTop100*.csv',
        'CohortMetricStats*.csv'
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

    # Denylist check reused from transfer_utils (single source of truth).
    is_denied = tu.is_denied

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
                copied_files.append(fpath)
                # Size tally is BEST-EFFORT: a stat error must never crash the run
                # nor mislabel a copy that already succeeded (graceful-continue).
                try:
                    size = os.path.getsize(str(dest)) / (1024 * 1024)  # MB
                    total_size += size
                    if verbose:
                        print(f"[TRANSFER] Copied: {fpath} ({size:.2f} MB)")
                except Exception as se:
                    if verbose:
                        print(f"[TRANSFER] Copied: {fpath} (size tally failed: {se})")
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
            copied_files.append(dirpat)
            if verbose:
                print(f"[TRANSFER] Copied directory: {dirpat}/")

            # Size tally is BEST-EFFORT (see file loop): a file vanishing
            # mid-walk or any stat error must never crash the run nor un-count a
            # directory that already copied successfully.
            try:
                for root, dirs, files in os.walk(str(dest_dir)):
                    for fname in files:
                        try:
                            total_size += os.path.getsize(os.path.join(root, fname)) / (1024 * 1024)  # MB
                        except Exception:
                            pass
            except Exception as se:
                if verbose:
                    print(f"[TRANSFER] WARNING: size tally failed for {dirpat}/: {se}")
        except Exception as e:
            if verbose:
                print(f"[TRANSFER] ERROR copying directory {dirpat}: {e}")

    # Assert that the key file was NOT copied (shared post-copy safety net).
    if not tu.assert_no_key_file(transfer_path, verbose=verbose):
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

def print_transfer_launch_status(configdic, verbose=True):
    """LOUD launch-time banner (printed BEFORE the long fetch begins) stating
    whether the Drive transfer is ON or OFF, the resolved target, and whether
    that target exists and is writable.

    If transfer is ON but the target is NOT usable (Drive unmounted / bad path /
    not writable), emit a LOUD warning RIGHT HERE so the operator can fix the
    mount before spending ~12h -- this is the highest-value guard against the
    silent-miss that lost last night's outputs.  Never raises.

    Returns {'enabled': bool, 'ok': bool, 'detail': str}.
    """
    line = "=" * 70
    transfer_dir = configdic.get('transfer_dir')
    disabled_reason = configdic.get('transfer_disabled_reason')

    if not transfer_dir:
        reason = disabled_reason or 'no target resolved'
        if verbose:
            print("\n" + line)
            print("  DRIVE TRANSFER: OFF (disabled)")
            print(f"  reason : {reason}")
            print(line + "\n")
        return {'enabled': False, 'ok': True, 'detail': reason}

    probe = tu.probe_transfer_target(transfer_dir)
    if verbose:
        print("\n" + line)
        print("  DRIVE TRANSFER: ON")
        print(f"  target : {transfer_dir}")
        print(f"  exists : {probe['exists']}   writable: {probe['writable']}")
        print(f"  detail : {probe['detail']}")
        print(line)
        if not probe['ok']:
            print("!" * 70)
            print("  !!! WARNING: transfer is ON but the target is NOT usable !!!")
            print("  !!! Outputs will NOT reach the Drive unless you fix this NOW !!!")
            print(f"  !!! ({probe['detail']}) !!!")
            print("!" * 70)
        print("")
    return {'enabled': True, 'ok': probe['ok'], 'detail': probe['detail']}


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
    import run_logger as rl
    #import warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    args = sys.argv[1:]

    # Assign parameters
    configdic = cf.getDataFetchConfiguration(args)

    # ---- START RUN LOGGING ----
    # Install stdout/stderr tee to log file with api_key scrubbing for file writes only.
    # Restores original streams in the finally block below.
    api_key = configdic.get('api_key', None)
    log_path, log_file = rl.start_run_logging(api_key=api_key)

    # ---- LOUD Drive-transfer launch status (BEFORE the ~12h fetch) ----------
    # Surface ON/OFF, resolved target, and target usability NOW so an unmounted
    # Drive or bad path is caught at launch, not discovered 12h later.
    print_transfer_launch_status(configdic, verbose=True)

    try:
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
            # DELIBERATELY NOT APPLIED (CEO decision, 2026-07-19).  The
            # ManualEliminationTickersList is a MACHINE-ACCUMULATED FETCH-FAILURE log
            # (it is rebuilt below as manualelim + tickersfailed - lenfail), NOT CEO
            # curation -- it contains ordinary companies whose fetch happened to fail
            # once.  Blanking it deliberately RETRIES names whose coverage may have
            # improved, which is the wanted behaviour.  Kept as an explicit, named
            # variable so the provenance stamp can state what actually ran.
            manualelim_loaded = list(manualelimtickers)
            manualelim_applied = []
            manualelimtickers = manualelim_applied
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

            # Note that **getfunddic should overwrite key-value combinations in datandmetricdic
            datandmetricdic = {**datandmetricdic, **{'Tickers_df': Tickers_df}, **getfunddic, **configdic}

            # PROVENANCE (audit C-3 fix, 2026-07-19).  configdic is spread LAST above, so
            # its 'manualelimtickers' (the list LOADED from disk) overwrote the
            # newly-accumulated list assigned at :330 -- and utils.saveWrapper stamps the
            # filename from that key.  The 2026-07-17 run therefore shipped
            # `...len7752_manelim3692_fails2075.pickle`, asserting a 3,692-name filter that
            # NEVER RAN, while the pickle carried the stale 2023 list instead of the 759
            # names this run actually accumulated.  Re-assert AFTER the spread so no merge
            # order can silence it, and separate the two meanings that were colliding in
            # one key:
            #   manualelim_applied -> what filtered THIS run's universe (what the filename
            #                         must state); [] under the CEO decision above
            #   manualelimtickers  -> the accumulated fetch-failure log to carry FORWARD
            datandmetricdic['manualelim_applied'] = manualelim_applied
            datandmetricdic['manualelim_loaded'] = manualelim_loaded
            datandmetricdic['manualelimtickers'] = newmanelimtckrs
            print('MANUAL-ELIM PROVENANCE: loaded %d name(s) from %r, APPLIED %d; '
                  'accumulated %d fetch-failure name(s) for the next run. Filename will '
                  'stamp manelim%d.'
                  % (len(manualelim_loaded), configdic.get('manualelimtick_fname_toget', 'n/a'),
                     len(manualelim_applied), len(newmanelimtckrs), len(manualelim_applied)),
                  flush=True)

            # Apply data quality filter to freshly fetched data
            datandmetricdic = dq.apply_data_quality_filter(datandmetricdic, verbose=True, save_log=True)

            # bm_ave AFTER the data-quality filter (audit H-1 fix, 2026-07-19).
            # getAves2 used to run on getfunddic['BoMetric_df'] BEFORE the filter, so the
            # cross-sectional MEDIAN that every 'm'-prefixed (mean-relative) Stage-1 test
            # compares each company against was computed over the corrupt-era rows the
            # filter exists to remove.  Recomputing it on the filtered frame (which now
            # also has the row-level removals propagated, see data_quality) moves 26 of the
            # 36 medians by more than 1% on the 2026-07-17 data -- several by 20-190% --
            # so this changes Stage-1 baselines and therefore the ranking, BY DESIGN.
            meandic = csf.getAves2(datandmetricdic['BoMetric_df'])
            datandmetricdic = {**datandmetricdic, **meandic}

            #write to info to file
            utils.write_lastIndexRead(configdic['lastindex_fn'], getfunddic['cind'])
            utils.writeManElimToFile(datandmetricdic,newmanelimtckrs)
            # Save results if saveBoMetric == 1
            if saveBoMetricbool:
                metric_fname = utils.saveWrapper('metric', datandmetricdic)
                # PHASE 1 boundary: sync the freshly-written metric pickle to Drive so a
                # later crash still leaves phase-1 output on Drive.  No-op if transfer_dir
                # unset; never raises.
                tu.copy_artifacts_to_transfer_dir(
                    configdic.get('transfer_dir'), [metric_fname], verbose=True)
        else:
            loadmetricdic = {'loadBoMetric': loadBoMetricbool, 'loadBoMetricfname': configdic['loadBoMetricfname']}
            datandmetricdic = utils.loadWrapper('metric', loadmetricdic)

        # Apply data quality filter (remove corrupted/invalid price data)
        # This must happen BEFORE any scoring to prevent garbage from affecting calculations
        _bm_rows_pre2 = len(datandmetricdic.get('BoMetric_df', []))
        datandmetricdic = dq.apply_data_quality_filter(datandmetricdic, verbose=True, save_log=True)

        # SELF-HEAL the H-1 ordering risk (review M2).  getAves2 runs after the FIRST
        # data-quality pass; this is the SECOND, so if it ever removed anything, bm_ave
        # would once again describe a frame that is no longer the one being scored -- the
        # exact defect fix 13 closed.  The step check is built to be idempotent (its
        # market-cap step is bounded to one reporting period, so a pair created BY a
        # removal cannot fire) and measures 0 removals on the real panel across three
        # passes -- but "measured idempotent" is not "guaranteed idempotent", so recover
        # rather than trust it, and say so loudly if it ever fires.
        _bm_rows_post2 = len(datandmetricdic.get('BoMetric_df', []))
        if _bm_rows_post2 != _bm_rows_pre2:
            print("!" * 78, flush=True)
            print("!!! data-quality pass 2 removed %d BoMetric row(s) -- RECOMPUTING "
                  "bm_ave so the Stage-1 baselines match the frame actually scored "
                  "(H-1 guard)." % (_bm_rows_pre2 - _bm_rows_post2), flush=True)
            print("!" * 78, flush=True)
            datandmetricdic = {**datandmetricdic,
                               **csf.getAves2(datandmetricdic['BoMetric_df'])}

        if portfoliotestyear > 0:
            datandmetricdic = pf.portfolioBacktestWrapper(portfoliotestyear,datandmetricdic)

        else:
            if not loadBoResultbool:
                resdic = pb.postBoWrapper(datandmetricdic, as_of=as_of)
                resdic = {**resdic, **datandmetricdic}

                # save results according to boolean. Note that saveBoResults = 0 if loadBoResults = 1
                if saveBoResultbool:
                    results_fname = utils.saveWrapper('results',resdic)
                    # PHASE 2 boundary: sync the results pickle to Drive incrementally.
                    tu.copy_artifacts_to_transfer_dir(
                        configdic.get('transfer_dir'), [results_fname], verbose=True)
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

        deliverable_fnames = pb.writeResWrapper(resdic)

        # PHASE 3 boundary (deliverables): the human-readable top-N deliverables
        # (AggScoreTop*.csv, PresentationTop*.xlsx, ForensicFlagsTop*.csv) are written
        # by writeResWrapper ABOVE, before the postRank pickle and the (optional,
        # multi-hour) delisted ingestion below.  Sync them to Drive as soon as they are
        # written so an ingestion-phase crash can't lose them.  Same helper + denylist as
        # the other phase copies; no-op when transfer_dir is unset; never raises.
        tu.copy_artifacts_to_transfer_dir(
            configdic.get('transfer_dir'), deliverable_fnames, verbose=True)

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

        # PHASE 3 boundary: sync the postRank pickle to Drive incrementally, so the
        # ranked output is on Drive before the (optional, long) delisted ingestion runs.
        tu.copy_artifacts_to_transfer_dir(
            configdic.get('transfer_dir'), [postrank_fname], verbose=True)

        # ---- PROSPECTIVE PICK-LOG stage (append-only forward track record) --------
        # Append this run's GENERAL top-N + the five cohort side-lists as NEW, immutable
        # rows to pick_log.csv -- one row per (run, list, stock), stamped with the run's
        # as_of date BEFORE any outcome exists (survivorship-free, un-gameable). Runs AFTER
        # the deliverables are emitted (writeResWrapper above) so it logs exactly the frames
        # that shipped. Fully isolated + guarded: a failure logs LOUDLY but never crashes the
        # run, and it only writes the LOCAL file (no auto-commit/push -- public repo, per CEO).
        # The import itself is inside this guard too: run_pick_log_stage is self-guarded and
        # never raises, so this outer try only catches an IMPORT failure -- degrading it
        # loud-but-safe rather than letting a pick-log module problem crash the deliverable.
        try:
            import pick_log as plog
            plog.run_pick_log_stage(resdic, as_of=as_of)
        except Exception:
            import traceback as _tb
            _pl_banner = ("\n" + "!" * 78 + "\n"
                          "!!! PICK-LOG STAGE COULD NOT BE IMPORTED/STARTED -- RUN CONTINUES !!!\n"
                          "!!! The forward pick-log was NOT written this run; deliverables above  !!!\n"
                          "!!! are UNAFFECTED. Investigate the pick_log module import.            !!!\n"
                          + "!" * 78 + "\n")
            print(_pl_banner, file=sys.stderr, flush=True)
            _tb.print_exc(file=sys.stderr)
            print(_pl_banner, flush=True)

        # ---- POST-PICK ANALYSIS SUITE (strictly additive, guarded) -----------------
        # Promote the offline baseline_tools/ diagnostics into pipeline stages so a single
        # overnight run also emits the analysis.  Runs AFTER the pick-log (picks + pickle +
        # pick-log already written above => pick path is DONE and UNAFFECTED) and BEFORE the
        # optional delisted ingestion.  Each analysis is a SEPARATELY-guarded stage inside
        # run_analysis_suite; a stage failure banners loudly + never crashes the run or the
        # picks.  The outer try here only guards the IMPORT (baseline_tools is a sys.path dir,
        # not a package), mirroring the pick-log stage's import guard.  NO commit/push; heavy
        # ESTIMATION sub-block is OFF unless -run_estimation 1.
        try:
            _bt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_tools')
            if _bt_dir not in sys.path:
                sys.path.insert(0, _bt_dir)
            import pipeline_analysis as _pa
            _pa.run_analysis_suite(resdic, configdic)
        except Exception:
            import traceback as _tb
            _pa_banner = ("\n" + "!" * 78 + "\n"
                          "!!! ANALYSIS SUITE COULD NOT BE IMPORTED/STARTED -- RUN CONTINUES !!!\n"
                          "!!! The post-pick analysis readouts were NOT produced this run;       !!!\n"
                          "!!! the deliverables + pick-log above are UNAFFECTED. Investigate the  !!!\n"
                          "!!! pipeline_analysis import.                                          !!!\n"
                          + "!" * 78 + "\n")
            print(_pa_banner, file=sys.stderr, flush=True)
            _tb.print_exc(file=sys.stderr)
            print(_pa_banner, flush=True)

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

        # ---- End-of-run transfer to Google-Drive-synced folder (OPT-OUT, default ON) ----
        # Transfer runs by default; disabled only by -no_transfer / -transfer_dir none.
        # Copy the output allowlist after ALL outputs are written and AFTER ingestion.
        # If transfer was ON but the copy did NOT happen (target absent/unwritable at
        # copy time), emit a LOUD warning -- never the old quiet skip.
        transfer_dir = configdic.get('transfer_dir')
        disabled_reason = configdic.get('transfer_disabled_reason')
        if transfer_dir:
            print("\n" + "="*70)
            print("END-OF-RUN TRANSFER TO GOOGLE DRIVE")
            print("="*70)
            transfer_result = transfer_outputs_to_drive(transfer_dir, configdic, verbose=True)
            print(f"Transfer result: {transfer_result['message']}")
            if transfer_result['status'] != 'success':
                print("\n" + "!"*70)
                print("!!! DRIVE TRANSFER DID NOT COMPLETE -- OUTPUTS DID NOT REACH THE DRIVE !!!")
                print(f"!!! status = {transfer_result['status']}")
                print(f"!!! detail = {transfer_result['message']}")
                print(f"!!! target = {transfer_dir}")
                print("!!! ACTION: copy the run outputs to the Drive MANUALLY.")
                print("!"*70 + "\n")
        else:
            # Explicit disabled case -- never a silent skip.
            print(f"\nDrive transfer DISABLED by {disabled_reason or '-transfer_dir none'}")

    except BaseException:
        # ---- EXCEPTION HANDLING: write traceback to log before closing ----
        # Capture the full traceback and write it to the log file so a mid-run
        # crash (including KeyboardInterrupt on a hung run) leaves a complete log
        # ending in the error. Then re-raise so the interpreter's default handler
        # prints it to console. Catches BaseException (not just Exception) to handle
        # KeyboardInterrupt (Ctrl-C) and abnormal exits (exit codes != 0).
        import traceback as _tb_module
        import sys as _sys_module

        # Guard: SystemExit (intentional exit, any code) should NOT write a failure banner.
        # Only write the banner for unhandled exceptions (KeyboardInterrupt, etc).
        _exc_info = _sys_module.exc_info()
        _is_system_exit = isinstance(_exc_info[1], SystemExit)

        # Write traceback to log UNLESS it's a sys.exit (intentional exit).
        if not _is_system_exit:
            _tb_text = _tb_module.format_exc()
            # Scrub the traceback (may contain apikey=... in exception message)
            _tb_text_scrubbed = rl.scrub(_tb_text, api_key)
            try:
                log_file.write("\n" + "=" * 78 + "\n")
                log_file.write("UNHANDLED EXCEPTION:\n")
                log_file.write("=" * 78 + "\n")
                log_file.write(_tb_text_scrubbed)
                log_file.write("=" * 78 + "\n")
                log_file.flush()
            except Exception:
                # If I/O error during traceback write, preserve the original exception
                # (traceback is already in the log before this write, so it's not lost).
                pass

        # Close log and restore streams, then re-raise so exception propagates.
        # Guard the close against I/O errors so the original exception surfaces.
        try:
            rl.end_run_logging(log_path, log_file)
        except Exception:
            # If close fails, don't replace the original exception.
            pass
        raise
    finally:
        # ---- CLOSE RUN LOGGING (success case) ----
        # If we get here without an exception, close the log normally.
        # (Exception case is handled in the except block above.)
        try:
            rl.end_run_logging(log_path, log_file)
        except Exception:
            # Log file already closed in except block; ignore
            pass

if __name__ == '__main__':
    main()



