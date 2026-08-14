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

    # ALLOWLIST: explicit patterns to copy.
    #
    # DELIVERABLES *AND* EVIDENCE (widened 2026-08-07 -- this is the common cause).
    # This list was built around what the analyst READS (the ranked deliverables) and
    # never around what would let anyone CHECK them.  The consequence, measured on the
    # 2026-08-07 CUR3K run: a run that behaves correctly and a run that silently drops
    # 3% of its universe produce BYTE-IDENTICAL transferred artifact sets.  Four
    # separate silent failures that run (82 names dropped with no fail bucket, an
    # unwritten CurrencyFloorFlips, an eight-month-old sector map, an unbuilt profile
    # map) were all invisible for the SAME reason: their evidence never left the
    # machine.  Everything inside the pickle reconciled exactly; everything outside it
    # was simply absent.  So the rule for this list is now: if it is the ONLY record of
    # a decision the pipeline made about the universe, it ships.
    allowlist_patterns = [
        # --- deliverables (what the analyst reads) ---
        'Bometric_dic-*.pickle',
        'Boresults_dic-*.pickle',
        'postRank_*.pickle',
        'AggScoreTop*.csv',
        'PresentationTop*.xlsx',
        'ForensicFlagsTop*.csv',
        'real_prices.csv',
        'SideList_*.csv',
        'MarketCapBand_*.csv',
        'RawMetricsTop100*.csv',
        'CohortMetricStats*.csv',
        # --- evidence (what makes the deliverables checkable) ---
        # Each of these is the SOLE on-disk record of a decision that removed,
        # re-priced or re-classified members of the universe.  Without them a
        # reconciliation gap can only be found by re-running the pipeline.
        'CurrencyFloorFlips_*.csv',
        'ExcludedShareClasses_*.csv',
        'MissingDataFillReport_*.csv',
        'ReportingFrequencyConflicts_*.csv',
        'RunProvenance-*.json',
        'pick_log*.csv',
        #  --- THE EVIDENCE CSVs THAT MOVED OUT OF `output/` (CEO, 2026-08-10) ------------
        #  THE CASE THAT DECIDED IT.  On the 2026-08-10 CUR3K run every ROOT-LEVEL artifact
        #  reached Drive and `output/` AND `logs/` DID NOT: no DedupSurvivorReport, no
        #  FxRates, no CurrencyExclusionStatus, no DelistedPrune and no run log for that date
        #  ever arrived, and Drive's copies of those groups stop at 08-08 / 08-07.  The dedup
        #  breakdown was recoverable only because a copy happened to be inside a pickle.
        #
        #  THIS INVERTS THE REASONING RECORDED BELOW, WHICH IS WHY THE FILES MOVED RATHER
        #  THAN JUST GAINING PATTERNS.  The `MeanBarCalibration` note further down declines a
        #  top-level glob for a file living in `output/` on the grounds that it would be a
        #  DEAD GLOB -- `glob.glob` resolves from CWD, so a pattern cannot reach into a
        #  subdirectory -- and the `DedupSurvivorReport` note declines one on the grounds
        #  that `output/` "already ships whole, so the evidence cannot be lost to a pattern
        #  that stops matching after a filename change".  BOTH LEGS FLIP once the file is at
        #  root: the glob is live because root IS where glob looks, and "ships whole" is
        #  precisely the guarantee that failed.  A directory that ships whole and does not
        #  arrive ships nothing.
        #
        #  The writers now resolve their directory from `transfer_utils.EVIDENCE_DIR`, so
        #  these six patterns and those six writers have one place to disagree, not two.
        #  `output/` STAYS in `allowlist_dirs` below -- the historical files already in it
        #  must still travel if it ever does, and nothing is removed by this change.
        'DedupSurvivorReport_*.csv',
        'FxRates_*.csv',
        'FxRatesHistorical_*.csv',
        'CurrencyExclusionStatus_*.csv',
        'DelistedPrune_*.csv',
        'VendorContaminationFlags_*.csv',
        'removed_data_quality_*.csv',
        #  The AD-HOC PENALTY BUCKET (CEO, 2026-08-10) -- the itemised record of every name
        #  whose Stage-2 score was lowered for a data problem, and why.  It is the SOLE
        #  on-disk record of that decision, so it ships by the rule stated above, and it is
        #  written at root for the reason stated above.
        'AdHocPenaltyBucket_*.csv',
        #  The STAGE-1 VETO EJECTION LIST (CEO, 2026-08-13) -- the names the veto removed
        #  from each pool and the flags that removed them.  It ships by exactly the rule
        #  stated above: it is the SOLE on-disk record of a decision that removed members of
        #  the universe (the largest such decision the pipeline makes), and `report['ejected']`
        #  reaches neither the RunProvenance sidecar (which carries counts, not names) nor the
        #  postRank pickle.  Written at root for the reason stated above.
        'Stage1VetoEjections_*.csv',
        #  The mean-bar calibration (added 2026-08-09).  It is the run's own watchdog on the
        #  Stage-1 bars -- WRITTEN ALWAYS, even with no breach, because its PRESENCE is the
        #  evidence the check ran -- and it matched no pattern here, so it reached the other
        #  machine on no run.
        #  WHY A PATTERN AND NOT A MOVE INTO output/, which is the opposite of the call made
        #  for DedupSurvivorReport below.  Two reasons, both specific to this file:
        #    1. `meanBars._prior_streaks` READS THE PRIOR CALIBRATION CSVs FROM THE SAME
        #       DIRECTORY to chain the breach-streak hysteresis.  Moving the writer to
        #       output/ would strand the existing root-level history (which is git-TRACKED)
        #       outside the search path and silently restart every streak at zero -- a
        #       transfer fix that breaks a correctness mechanism is not a fix.
        #    2. output/ is .gitignore'd.  This file is tracked, so moving it TRADES the git
        #       channel for the Drive channel; a pattern ADDS the Drive channel and keeps
        #       both.
        #  The objection recorded below -- that a top-level pattern would be a DEAD GLOB --
        #  is about a file living in output/. This one is written to CWD by
        #  `meanBars.emit_calibration(directory='.')`, which is where `glob.glob` looks, so
        #  the pattern genuinely matches. Checked against the on-disk name
        #  (`MeanBarCalibration-2026-08-07_stock_CUR3K.csv`), not against the format string.
        #  SINCE 2026-08-10 THIS FILE IS NO LONGER THE EXCEPTION -- the other evidence CSVs
        #  moved to root beside it (see the block above) -- but reason 1 above is still the
        #  reason this one CANNOT move anywhere else: `meanBars._prior_streaks` reads the
        #  prior calibration CSVs back from the directory it writes to, so relocating the
        #  writer would strand the git-tracked root-level history and silently restart every
        #  breach streak at zero.  It stays at root, and it stays at root for its own reason.
        'MeanBarCalibration-*.csv',
        #  The RAW vendor delisted list (added 2026-08-09).  The record of what the prune
        #  REMOVED is `DelistedPrune_<date>.csv` (moved to the repo root 2026-08-10) -- this is
        #  the other half: what the VENDOR said on the day.  It ships because it is NOT
        #  REPRODUCIBLE: `v3/delisted-companies?page=0` is a moving window, so a later call
        #  cannot recover the list this run pruned against, and without it a disputed
        #  removal can only be argued from our own derived file.  Top-level and dated, so
        #  the glob genuinely matches (unlike a pattern aimed into output/ -- see below).
        'delisted_tickers_*.csv',
        # --- the profile maps (added 2026-08-08) ---
        # THE SAME EVIDENCE GAP 84abd40 CLOSED EVERYWHERE ELSE, missed here because these
        # are INPUTS rather than run outputs -- but they are inputs that DECIDE WHICH
        # TICKER SURVIVES, and they are rebuilt by the run itself, so nothing else on the
        # transfer side records their contents.  `sectorsdic` is the same class of object
        # (it gates the cohort split and an eight-month-old copy of it was one of the four
        # silent failures above) and `industrydic` feeds the concentration split.  Checked,
        # not assumed: none of the four was matched by any pattern in this list, and none is
        # caught by the denylist.
        #
        # A CLAIM THAT USED TO SIT HERE, CORRECTED TO WHAT IS ACTUALLY EVIDENCED (2026-08-09).
        # This block asserted that on the 2026-08-08 run "volavgdic went 0 -> 3,127 entries,
        # the volume term decided 12 dedup groups".  MEASURED on the only artifact from that
        # date on THIS machine, `output/DedupSurvivorReport_2026-08-08.csv`: 248 rows,
        # `decided_by` = canonicity 94, punctuation 93, alphabetical 31, shares 29,
        # marketCap 1 -- **ZERO** decided by either volume term -- with `dropped_volAvg` and
        # `survivor_volAvg` null and `*_volAvg_asof` = 'not-captured' on all 248 rows.  This
        # machine also holds NO `volavgdic_fmp_*.pickle` and no `isindic_fmp_*.pickle` at all.
        # THE 3,127 IS NOT REFUTED, IT IS UNVERIFIED HERE.  The likely reading is that it came
        # from the CEO's separate FETCH machine (which did build the map) while the CSV above
        # is from a local run with no map -- two different runs on one date.  That cannot be
        # settled from this machine, so it is recorded as unverified rather than asserted or
        # deleted.  WHAT THE LOCAL EVIDENCE DOES ESTABLISH is the argument this list needs and
        # is enough on its own: with no map, 31 groups fell to the ALPHABET on 2026-08-08 --
        # the failure mode the maps exist to remove -- and nothing on the transfer side
        # recorded whether a map was present, which is precisely why these four now ship.
        #
        # `sectorsdic_fmp.pickle` is UNDATED in its filename -- the only one of the four
        # that is -- so on the receiving side it is distinguishable only by mtime; the
        # copy is shutil.copy2, which preserves it.
        'volavgdic_fmp_*.pickle',
        'isindic_fmp_*.pickle',
        'sectorsdic_fmp.pickle',
        'industrydic_fmp_*.pickle',
    ]

    # Directory artifacts.
    # `logs/` is the console tee written by run_logger.start_run_logging on EVERY run
    # (run_logger.py:101-107).  It holds the DQ removal banner, the floor banner, the
    # per-exchange counts and every `WARNING:` emitted by every warn-and-continue
    # `except` in the pipeline -- i.e. the only place a swallowed failure is recorded.
    # It was written correctly on every run and shipped on none of them.  It is
    # .gitignore'd (line 39), so Drive is the ONLY channel it can travel by.
    # `output/` holds the per-decision detail CSVs (e.g. removed_data_quality_*.csv,
    # which names the 82 dropped sources and the reason for each).
    #
    # `output/` CARRIED `DedupSurvivorReport_*.csv` and `DelistedPrune_<date>.csv` UNTIL
    # 2026-08-10, AND THAT IS THE POINT: it carried them, and they did not travel.
    #
    # The arguments this block used to make were: the dedup report is "deliberately written
    # INTO `output/` rather than given its own top-level pattern here: this directory already
    # ships whole, so the evidence cannot be lost to a pattern that stops matching after a
    # filename change"; and the delisted-prune record likewise, because "a top-level pattern
    # (which glob.glob() resolves from CWD and would therefore match nothing) is not needed
    # and would be decorative".  Both were sound conditional on `output/` arriving.  On the
    # 2026-08-10 run it did not, and neither did `logs/`, so BOTH files were lost for that
    # date while every root-level artifact from the same run was fine.  A pattern that stops
    # matching is a risk you can test for; a directory that silently does not copy is one
    # nobody was checking.  The CEO's ruling is therefore to move the writers to root, which
    # is what the pattern block above records, and both files now have live top-level globs.
    #
    # WHAT THEY ARE, kept because it is the reason they ship at all.  The dedup report is the
    # per-dropped-line record of WHICH TICKER SURVIVED each issuer group, which term decided
    # it, and the raw volAvg readings and as-of dates behind that decision -- the sole on-disk
    # record of the survivor pick.  The delisted-prune record is the named list of every
    # symbol the prune removed, with its reason and the page-0 limitation stated IN the file;
    # that prune runs UPSTREAM of Tickers_df, so `resolved` is already post-prune and the
    # end-of-run reconciliation's residual of 0 proves nothing about it.
    #
    # `output/` REMAINS in this list, unchanged: the historical files already inside it (and
    # the backtest output folders, which are not evidence CSVs and did not move) must still
    # travel whenever the directory copy does succeed.  Nothing was removed from the manifest
    # by this change -- only added.
    #
    # `run_logs/` is DELIBERATELY NOT unconditional.  Its only writer is
    # delisted_ingest.py (run_logging.RunLogger is constructed nowhere else), so on a
    # normal run the directory is empty and listing it would make the new end-of-run
    # reconciliation report a MISSING artifact on 100% of runs -- alarm fatigue, which
    # would defeat the entire point of the reconciliation.  It is added below, next to
    # delisted_out, only when the run that writes it actually ran.
    allowlist_dirs = [
        'logs',
        'output',
        'baseline_tools/price_data'
    ]

    # If -ingest_delisted ran, also include delisted outputs AND run_logs/ -- that
    # ingestion is the only thing that writes run_logs/ (see note above).
    if configdic.get('ingest_delisted'):
        allowlist_dirs.append('delisted_out')
        allowlist_dirs.append('run_logs')

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

    # THE ONE MISCONFIGURATION THE pipeline/non-pipeline SPLIT MAKES POSSIBLE (CEO,
    # 2026-08-11).  The destination is being restructured into a `pipeline/` folder the
    # program owns and a `non-pipeline/` folder it must never touch, with the run
    # pointed at the `pipeline` leaf.  Nothing in the transfer logic needs to know
    # that -- it writes wherever it is pointed -- which is exactly why passing the
    # PARENT by mistake would silently work: run artifacts would land beside the manual
    # files and the destination-side denylist check would start reporting on them.
    # Refused, not warned about, and the message names the directory that was meant.
    misconfig = tu.looks_like_transfer_parent(transfer_path)
    if misconfig:
        if verbose:
            print(f"[TRANSFER] REFUSED: {misconfig}")
        result['status'] = 'error'
        result['message'] = f'Refused to transfer: {misconfig}'
        return result

    # Try to create the target directory
    already_existed = transfer_path.exists()
    try:
        transfer_path.mkdir(parents=False, exist_ok=True)
        if verbose:
            print(f"[TRANSFER] Target directory created/exists: {transfer_path}")
        # A LABEL FOR WHOEVER OPENS THE FOLDER, written ONCE -- when the pipeline
        # itself creates the directory -- and never rewritten.  It says what this
        # folder is, that runs add to and overwrite it, that nothing here is ever
        # deleted by us, and where manual files belong instead.
        if not already_existed:
            tu.write_pipeline_readme(transfer_path, verbose=verbose)
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

    # A FILE THAT FAILED TO COPY IS NOT A SUCCESS EITHER (review S4, 2026-08-11).
    # This loop's `except` printed and continued with no effect on the status, exactly
    # as the directory loop's did.  It is not a theoretical twin: THREE allowlist
    # patterns resolve to UNDATED names that are overwritten every run --
    # `real_prices.csv`, `sectorsdic_fmp.pickle`, `pick_log.csv` -- so when the
    # destination refuses the overwrite (the same WinError-5 class that broke the
    # directories), the destination keeps the STALE file, the reconciliation finds a
    # non-empty file of the right name and reports the group COMPLETE, and the status
    # stays 'success'.  Reviewer reproduced all three.  That is byte-for-byte the
    # signature of the bug this change exists to fix, and "an eight-month-old sector
    # map" is one of the four silent failures the 2026-08-07 post-mortem quoted in the
    # comments above.  The reconciliation cannot see it -- only the copier knows.
    file_failures = []

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
                file_failures.append(f"{fpath} ({e})")
                if verbose:
                    print(f"[TRANSFER] ERROR copying {fpath}: {e}")

    # ---- Copy directories (with contents into matching subfolder) -----------------
    #
    # THE 2026-08-08..08-11 PRODUCTION FAILURE AND ITS THREE FIXES.  For four
    # consecutive runs every root-level FILE above reached the Drive and NOT ONE of
    # these directories did.  The CEO's console:
    #
    #   ERROR copying directory logs: [WinError 5] Access is denied:
    #                                 "E:\drive\valuationTransfer\logs"
    #
    # -- and the path in that error is the DESTINATION, so the call that failed was
    # the `shutil.rmtree(dest_dir)` this loop did BEFORE copying, not the copy.
    # Google Drive's virtual filesystem refuses the directory delete.  Cost: no
    # `output/`, no `logs/`, no `run_logs/` at the destination on any of those runs,
    # i.e. no FX evidence, no dedup detail and no run log available for offline
    # analysis -- which is why six evidence CSVs were moved to the repo root on
    # 2026-08-10 as a workaround, and why a file had to be copied across by hand.
    #
    # (1) THE DELETE IS GONE.  `dirs_exist_ok=True` (Python >= 3.8; this repo runs
    #     3.13) merges into the directory that is already there, so the operation
    #     Drive refuses is never issued.
    #
    #     THE CONSEQUENCE, STATED SO IT READS AS A DECISION AND NOT AN OVERSIGHT
    #     (CEO, 2026-08-11): without the delete, a file deleted LOCALLY now persists
    #     at the destination.  That is CORRECT here.  The transfer dir is an OUTBOX,
    #     not a mirror -- its job is to get this run's artifacts onto a machine that
    #     can read them, not to be a faithful reflection of this machine.  A stale
    #     artifact at the destination is recoverable (it is dated in its own
    #     filename, and the reconciliation below states what THIS run put there); an
    #     artifact deleted from the destination is not.  Given the evidence-loss this
    #     path has already caused, keeping too much is the safe error.  If a true
    #     mirror is ever wanted it needs an explicit prune step that can survive a
    #     filesystem which refuses deletes -- it is NOT recoverable by restoring the
    #     rmtree, which is the thing that broke.
    #
    # (2) DENY THE FILE, NOT THE TREE.  This loop used to walk the tree, and a single
    #     file matching the denylist set `has_denied` and `continue`d past the WHOLE
    #     directory -- silently.  FMP's own endpoint family is called `key-metrics`,
    #     so one cached response named after it would have cost the entire `output/`
    #     tree, and nothing in the output would have said so.  `copytree(ignore=...)`
    #     drops the denied FILE and ships its siblings.  The safety property is
    #     unchanged and is now enforced per-file at every depth: no denied file may
    #     reach the destination.  `transfer_utils.is_denied` stays the single source
    #     of truth -- the matching is NOT reimplemented here.  Independently audited
    #     at the Windows copy primitive (review, 2026-08-11): 7 denied files planted at
    #     5 depths, 5 files written, ZERO denied ones -- not even transiently.
    #
    # (3) A DIRECTORY FAILURE IS NO LONGER A SUCCESS.  Failures are recorded in
    #     `dir_failures` and fail the overall status (see the result block below).
    #     Warn-and-continue still holds -- a 12-hour run must not die on a Drive
    #     hiccup -- but "non-fatal" was implemented as "invisible": three quiet ERROR
    #     lines scrolled past and the tail of the run reported a clean transfer.
    dir_failures = []

    def _ignore_denied(src_dir, names):
        """`shutil.copytree` ignore-callable: drop DENIED FILES, keep everything
        else.  Directories are never dropped for their own name -- `is_denied`
        matches a BASENAME and its contract is that a file is denied for its own
        name, never because a parent directory happens to contain 'key'/'pem'.
        Sub-directory CONTENTS are still checked: copytree calls this for every
        directory it descends into."""
        drop = set()
        for name in names:
            full = os.path.join(src_dir, name)
            if os.path.isdir(full):
                continue
            if is_denied(name):
                drop.add(name)
                if verbose:
                    print(f"[TRANSFER] DENIED (denylist): {full}")
        return drop

    for dirpat in allowlist_dirs:
        if not os.path.isdir(dirpat):
            if verbose:
                print(f"[TRANSFER] Skipped (not found): {dirpat}/")
            continue

        # Copy the directory
        try:
            dest_dir = transfer_path / dirpat
            shutil.copytree(dirpat, str(dest_dir),
                            ignore=_ignore_denied, dirs_exist_ok=True)
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
            dir_failures.append(f"{dirpat}/ ({e})")
            if verbose:
                print(f"[TRANSFER] ERROR copying directory {dirpat}: {e}")

    # ---- DESTINATION-SIDE CHECK, RUN UNCONDITIONALLY, DELETES NOTHING -------------
    # ONE CHECK OVER THE WHOLE TRANSFER ROOT, OUTSIDE EVERY `try` (review S1/S2).  An
    # earlier version called this inside the copy loop's `try`, one line after
    # `copytree`, so ANY copy failure jumped to the `except` and it never ran for that
    # directory -- weakest in exactly the scenario it was added for.  Here it cannot
    # be skipped, and it reaches `run_logs/` (only allowlisted under
    # -ingest_delisted), the `baseline_tools/` parent the copy itself creates, and
    # top-level names `assert_no_key_file` never knew (it checks the exact filename
    # `fmpAPIkey.txt` only).
    #
    # IT REPORTS; IT DOES NOT DELETE (CEO, 2026-08-11: "Please don't go deleting my
    # API key").  It used to unlink what it found, as compensation for dropping the
    # rmtree.  Both legs of that fell over: (a) the thing being compensated for never
    # happens -- the copy direction never writes a denied file, instrumented at the
    # Windows copy primitive with 7 denied files planted at 5 depths and 0 written --
    # so anything found here was put there BY A HUMAN; and (b) `*key*` is a substring
    # pattern, so the measured deletions on realistic filenames included
    # `Turkey_exposure_2026.csv`, `Keystone_Corp_thesis.docx` and
    # `hockey_stick_screen.csv`, with no Recycle Bin.  The destination holds the CEO's
    # investment research.  A credential-shaped file on a synced folder is an INCIDENT
    # TO REPORT -- deleting it does not unpublish what Drive already uploaded, it only
    # removes the evidence that it was there.
    denied_at_destination = tu.find_denied_files(transfer_path, verbose=verbose)

    # Report (never remove) a key file at the destination top level -- the shared
    # safety net, kept because the incremental copier uses it too.  Redundant with the
    # recursive check above and deliberately so: two independent detectors, no deletes.
    key_net_clean = tu.assert_no_key_file(transfer_path, verbose=verbose)

    result['copied_files'] = len(copied_files)
    result['total_size_mb'] = total_size
    result['files_list'] = copied_files
    result['dir_failures'] = dir_failures
    result['file_failures'] = file_failures
    result['denied_at_destination'] = denied_at_destination
    result['message'] = f"Transferred {len(copied_files)} items ({total_size:.2f} MB) to {transfer_dir}"

    # A COPY THAT FAILED IS NOT A SUCCESS (defect 3).  This assignment used to be
    # unconditional, eleven lines below an `except` that printed and continued -- so a
    # total directory failure reported `status = 'success'` and the caller's loud
    # "DRIVE TRANSFER DID NOT COMPLETE" banner (Sbocker.main) never fired.
    # Precedence: an unremovable secret outranks a failed copy outranks success.
    if denied_at_destination or not key_net_clean:
        result['status'] = 'error'
        detail = ', '.join(str(p) for p in denied_at_destination) or 'fmpAPIkey.txt'
        result['message'] += (f" -- DENYLISTED FILE(S) PRESENT AT THE DESTINATION "
                              f"(NOT removed -- remove by hand): {detail}")
    elif dir_failures or file_failures:
        result['status'] = 'warning'
        if dir_failures:
            result['message'] += f" -- DIRECTORY COPY FAILED for: {', '.join(dir_failures)}"
        if file_failures:
            result['message'] += f" -- FILE COPY FAILED for: {', '.join(file_failures)}"
    else:
        result['status'] = 'success'

    # ---- POST-TRANSFER RECONCILIATION ---------------------------------------
    # Every copy failure here is warn-and-continue by design (transfer_utils
    # contract 2), which for a long time also made it INVISIBLE: on 2026-08-07 the
    # run_logs/ group reached the Drive with nothing in it and the console looked
    # identical to a clean transfer.  Re-derive what each listed artifact group
    # should have put at the destination and check what actually IS there, so the
    # tail of a run states whether the transfer was COMPLETE.  The groups are built
    # from the SAME allowlists the copy loops walked, so the two cannot drift.
    # Directory groups pass their dirpat as the dest subdir because this function
    # copies to `transfer_path / dirpat` (not to the basename).
    recon_groups = [(pat, glob.glob(pat)) for pat in allowlist_patterns]
    recon_groups += [(d + '/', [d], d) for d in allowlist_dirs]
    reconciliation = tu.reconcile_transfer(transfer_dir, recon_groups, verbose=verbose)
    result['reconciliation'] = reconciliation
    result['complete'] = reconciliation.get('complete', False)
    result['message'] += f" -- {reconciliation.get('summary', 'not reconciled')}"

    # ---- THE RECONCILIATION NOW REACHES THE STATUS (defect 3b) --------------------
    # This block was written after the 2026-08-07 incident (run_logs/ arrived empty and
    # the console was indistinguishable from a clean transfer), it WORKED, and nothing
    # ever consumed its verdict: `complete` was set on the result and read by nobody,
    # so the copier kept grading its own homework.  Wiring it in closes the CLASS, not
    # just the rmtree instance -- it catches a copy that leaves the destination short
    # WITHOUT raising, which no `except` can.
    #
    # ONLY `incomplete` FAILS THE STATUS, NOT `complete`, AND THE DIFFERENCE MATTERS:
    #   * `incomplete` -- the artifact existed locally and did not fully land.  That is
    #     a COPY failure, i.e. this function's fault, and it must fail the status.
    #   * `missing`    -- nothing was produced locally, so nothing could copy.  In
    #     reconcile_transfer's own words: "the WRITER for these did not run.  Fixing
    #     the copier will not help."
    # `complete` is False when EITHER is non-empty, and on a healthy machine 8 of the
    # 31 pattern groups legitimately produce nothing (measured 2026-08-11:
    # real_prices.csv, FxRates_*, FxRatesHistorical_*, CurrencyExclusionStatus_*,
    # DelistedPrune_*, VendorContaminationFlags_*, removed_data_quality_*,
    # AdHocPenaltyBucket_*).  Failing the status on `complete` would therefore fire the
    # loud banner on EVERY run -- alarm fatigue, which is the mechanism that hid this
    # failure in the first place, so it would be the same bug with a new coat.  The
    # never-written groups are NOT swallowed: the caller reports them separately and
    # loudly, and the per-group reconciliation table above names every one of them.
    partial = reconciliation.get('incomplete') or []
    if partial and result['status'] == 'success':
        result['status'] = 'warning'
        result['message'] += (f" -- INCOMPLETE AT DESTINATION: {', '.join(partial)}")

    # THE TARGET VANISHED BETWEEN THE COPY AND THE CHECK (review R2-3, pre-existing).
    # If the Drive goes away mid-run, `reconcile_transfer` can verify NOTHING, and it
    # returns `missing == []` and `incomplete == []` -- so the gate above cannot see
    # it and the run reported `success` after nothing reached the Drive.  This is a
    # DIFFERENT flag from `complete` on purpose: the alarm-fatigue argument that
    # settled the missing-vs-incomplete question does not apply, because an
    # unavailable target fires on EXACTLY ZERO healthy runs.
    if reconciliation.get('unreconciled') and result['status'] == 'success':
        result['status'] = 'warning'
        result['message'] += " -- TARGET UNAVAILABLE AT RECONCILIATION TIME"

    if verbose:
        # The word here follows the STATUS.  It used to say "Success" unconditionally,
        # directly underneath the ERROR lines of a directory that had just failed.
        verdict = 'Success' if result['status'] == 'success' else result['status'].upper()
        print(f"[TRANSFER] {verdict}: {len(copied_files)} items, {total_size:.2f} MB total")
        print(f"[TRANSFER] Destination: {transfer_dir}")

    return result


def report_transfer_outcome(result, transfer_dir, emit=print):
    """Print the end-of-run verdict on the Drive transfer.  Returns True if the LOUD
    banner fired.

    EXTRACTED FROM `main` (review, 2026-08-11) SO IT CAN BE TESTED AT ALL.  It lived
    inline in a ~1000-line `main` that cannot be invoked offline, so the only coverage
    it could have was an AST grep over its own source -- and that grep was satisfied by
    a COMMENT: neutering the block by replacing `result.get('reconciliation')` with
    `{}` left the whole suite green.  A reporting path whose entire job is to be
    noticed deserves better than a string search, and `emit` makes it a pure function
    of the result dict.

    TWO REGISTERS, DELIBERATELY, because the two conditions mean different things and
    one banner for both would make the banner meaningless:
      * a copy failure, an unremovable secret, or a group SHORT at the destination
        -> our fault, the operator must act now -> the loud banner;
      * a group NEVER PRODUCED locally -> the writer did not run, re-copying will not
        help, and it fires on a perfectly healthy run for the 8 of 31 pattern groups
        this machine legitimately never writes -> a quiet, named NOTE.
    """
    result = result if isinstance(result, dict) else {}
    recon = result.get('reconciliation')
    recon = recon if isinstance(recon, dict) else {}
    never_written = recon.get('missing') or []
    status = result.get('status')

    def _names(seq):
        # Every producer path builds these from f-strings, but the reporter is the
        # tail of a 12-hour run: it must not be the thing that raises.
        return ', '.join(str(x) for x in (seq or []))

    emit(f"Transfer result: {result.get('message')}")
    emit(f"Transfer reconciled complete: {result.get('complete')}")

    loud = status != 'success'
    if loud:
        emit("\n" + "!" * 70)
        emit("!!! DRIVE TRANSFER DID NOT COMPLETE -- OUTPUTS DID NOT REACH THE DRIVE !!!")
        emit(f"!!! status = {status}")
        emit(f"!!! detail = {result.get('message')}")
        emit(f"!!! target = {transfer_dir}")
        if result.get('dir_failures'):
            emit(f"!!! directories that FAILED to copy: {_names(result['dir_failures'])}")
        if result.get('file_failures'):
            emit(f"!!! files that FAILED to copy: {_names(result['file_failures'])}")
        if recon.get('incomplete'):
            emit(f"!!! present locally but SHORT at the destination: "
                 f"{_names(recon['incomplete'])}")
        if recon.get('unreconciled'):
            emit("!!! the transfer target was UNAVAILABLE when the run tried to verify "
                 "it -- NOTHING can be confirmed to have reached the Drive.")
        if result.get('denied_at_destination'):
            # A SECRET problem, not an evidence problem -- and the run does NOT delete
            # it: on a synced folder the upload may already have happened, and the
            # destination holds the CEO's own research, which we do not touch.
            emit(f"!!! DENYLISTED FILE(S) PRESENT ON THE DRIVE -- NOT removed by the "
                 f"run; check and remove by hand: "
                 f"{_names(result['denied_at_destination'])}")
        emit("!!! ACTION: copy the run outputs to the Drive MANUALLY.")
        emit("!" * 70 + "\n")

    if never_written:
        emit("\n" + "-" * 70)
        emit("NOTE: listed artifacts that were NEVER PRODUCED LOCALLY (so nothing "
             "could be copied):")
        emit(f"      {_names(never_written)}")
        emit("      -> the WRITER did not run; re-copying will not help.  Expected "
             "for artifacts this run does not emit.")
        emit("-" * 70 + "\n")

    return loud


def print_universe_reconciliation(datandmetricdic, getfunddic, verbose=True):
    """END-OF-FETCH RECONCILIATION: does the universe add up?

        resolved == panel_sources + fetch_failures + filter_removals + residual

    Every name that entered the run must leave it through exactly one door.  When
    the residual is non-zero, names disappeared without any channel recording it --
    which is precisely what happened on 2026-08-07 (3140 resolved, 445 failed, 2613
    in the panel, 82 unexplained, nothing said so).  This makes that self-detecting.

    Reports; never raises and never aborts.  A multi-hour fetch must not die on a
    bookkeeping check -- but it must not hide one either.  Returns the residual, or
    None if the inputs could not be resolved.
    """
    try:
        tdf = datandmetricdic.get('Tickers_df')
        resolved = int(len(tdf)) if tdf is not None else None

        failed_raw = (getfunddic or {}).get('tickersfailed') or []
        n_failed = len(set(failed_raw))

        cdx = datandmetricdic.get('cdx_df')
        n_panel = None
        if cdx is not None and len(cdx) > 0:
            src_col = next((c for c in ('source', 'symbol', 'ticker', 'source_id')
                            if c in cdx.columns), None)
            n_panel = int(cdx[src_col].nunique()) if src_col else int(len(cdx))

        # ENTIRELY-removed sources only.  A PARTIAL removal (the 058820.KQ quarantine
        # window, a pre-corruption row trim) leaves the name in the panel, so counting it
        # here would subtract it twice and manufacture a negative residual -- see the
        # counter's derivation in data_quality.apply_data_quality_filter.  The partial
        # count travels beside it in `n_dq_partially_removed_sources` and is reported
        # below, because "trimmed but kept" is a fact worth seeing, just not a universe
        # exit.
        n_removed = datandmetricdic.get('n_dq_removed_sources')
        n_partial = datandmetricdic.get('n_dq_partially_removed_sources')

        if resolved is None or n_panel is None:
            if verbose:
                print('[reconcile] SKIPPED -- universe/panel not both available '
                      f'(resolved={resolved}, panel={n_panel})', flush=True)
            return None

        accounted = n_panel + n_failed + (n_removed or 0)
        residual = resolved - accounted

        datandmetricdic['reconcile_resolved'] = resolved
        datandmetricdic['reconcile_panel_sources'] = n_panel
        datandmetricdic['reconcile_fetch_failures'] = n_failed
        datandmetricdic['reconcile_filter_removals'] = n_removed
        datandmetricdic['reconcile_residual'] = residual

        datandmetricdic['reconcile_partial_removals'] = n_partial

        if verbose:
            print(f'[reconcile] universe {resolved} = panel {n_panel} + fetch-failed '
                  f'{n_failed} + filter-removed {n_removed} + residual {residual}'
                  f'  (plus {n_partial} source(s) partially trimmed and KEPT -- not a '
                  f'universe exit, so not in the identity)',
                  flush=True)
            if residual != 0:
                bar = '!' * 78
                print('\n' + bar)
                print('!!! UNIVERSE DOES NOT RECONCILE -- %d name(s) UNACCOUNTED FOR' % residual)
                print('!!!   resolved into the run : %d' % resolved)
                print('!!!   in the scored panel   : %d' % n_panel)
                print('!!!   failed the fetch      : %d' % n_failed)
                print('!!!   removed by filters    : %s' % n_removed)
                print('!!!   RESIDUAL              : %d  <-- no channel recorded these' % residual)
                print('!!! A non-zero residual means names left the universe without any')
                print('!!! artifact naming them or the reason.  Do NOT read the ranked')
                print('!!! output as covering the stated universe until this is explained.')
                print('!!! Check output/removed_*.csv and the logs/ tee for swallowed')
                print('!!! WARNINGs before trusting this run.')
                print(bar + '\n', flush=True)
        return residual
    except Exception as e:
        if verbose:
            print(f'[reconcile] WARNING: reconciliation check failed safely: {e}', flush=True)
        return None


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


_UNIVERSE_KEYS = ('universe', 'universe_label', 'universe_fingerprint',
                  'universe_exchanges', 'universe_symbols', 'universe_every_exchange',
                  'universe_expected_count', 'universe_definition_changed',
                  'universe_previous_exchanges', 'universe_codes_verified',
                  'universe_note')


def universe_provenance_for_run(configdic, loaded=None, verbose=True):
    """The universe stamp THIS run's artifacts should carry -- honest on the load path.

    ON THE FETCH PATH the answer is simply the active `-tickerfilter`'s definition: the
    data was just pulled for it.

    ON THE LOAD PATH IT IS NOT (2026-08-03).  `-loadbometric` re-scores a panel that was
    fetched for whatever universe built it, and the CURRENT `-tickerfilter` may be
    something else entirely.  Stamping the current filter onto loaded data would
    manufacture provenance -- the precise failure the stamp exists to prevent, committed
    by the stamp itself.  So:

      * loaded panel CARRIES a stamp -> that stamp wins, and a disagreement with the
        active `-tickerfilter` is announced LOUDLY (the artifact FILENAME will be built
        from the active filter, so the two would otherwise silently contradict);
      * loaded panel carries NO stamp (anything built before 2026-08-02) -> stamp it
        `unknown-unstamped-panel` and say so. An honest "unknown" is worth more than a
        confident wrong answer, because a reader comparing two runs can at least see
        that one basis is unestablished.
    """
    import universes as un
    fetch_stamp = {k: configdic.get(k) for k in _UNIVERSE_KEYS}
    if loaded is None:
        return fetch_stamp

    loaded_fp = (loaded or {}).get('universe_fingerprint')
    active = configdic.get('universe')
    bang = '!' * 78
    if loaded_fp:
        stamp = {k: loaded.get(k) for k in _UNIVERSE_KEYS}
        # COMPARE FINGERPRINTS, NOT NAMES (fixed 2026-08-03).  This originally tested
        # `loaded['universe'] != active`, i.e. the NAME -- which made it blind to the one
        # case the fingerprint was introduced for: two panels both called
        # `stock_NA1_EU1`, one from either side of the 2026-08-02 restoration, differing
        # by 1,046 members. Same name, different universe, and the banner said nothing.
        # The name is still reported (it is what the output FILENAMES are built from), but
        # the DISAGREEMENT is decided on the fingerprint.
        active_fp = configdic.get('universe_fingerprint')
        if verbose and (loaded_fp != active_fp or loaded.get('universe') != active):
            msg = '\n'.join([
                '', bang,
                '!!! UNIVERSE MISMATCH: the LOADED panel and the active -tickerfilter '
                'disagree.',
                '!!!   loaded panel : %s  (fingerprint %s)'
                % (loaded.get('universe'), loaded_fp),
                '!!!   -tickerfilter: %s  (fingerprint %s)' % (active, active_fp),
                '!!!   differ by     : %s'
                % ('DEFINITION (same name, different membership -- e.g. either side of '
                   'the 2026-08-02 restoration)' if loaded.get('universe') == active
                   else 'NAME' if loaded_fp == active_fp else 'NAME and DEFINITION'),
                '!!! The DATA is the loaded panel, so the loaded stamp is what the '
                'artifacts',
                '!!! carry -- but the output FILENAMES are built from -tickerfilter, so '
                'they',
                '!!! will name a universe this run did not actually score. Pass the '
                'matching',
                '!!! -tickerfilter, or read the fingerprint rather than the filename.',
                bang, ''])
            print(msg, file=sys.stderr, flush=True)
            print(msg, flush=True)
        return stamp

    if verbose:
        msg = '\n'.join([
            '', bang,
            '!!! LOADED PANEL CARRIES NO UNIVERSE STAMP (built before 2026-08-02).',
            '!!! Its membership CANNOT be established from the artifact, so this run is',
            '!!! stamped universe_fingerprint=unknown-unstamped-panel rather than being',
            '!!! attributed to the active -tickerfilter (%s), which would be a guess'
            % active,
            '!!! dressed as provenance. Anything compared against this run is comparing',
            '!!! against an UNESTABLISHED basis -- re-fetch for anything decisional.',
            bang, ''])
        print(msg, file=sys.stderr, flush=True)
        print(msg, flush=True)
    stamp = {k: None for k in _UNIVERSE_KEYS}
    stamp['universe'] = loaded.get('tickerfilter') or 'unknown'
    stamp['universe_fingerprint'] = 'unknown-unstamped-panel'
    stamp['universe_note'] = ('panel predates universe stamping (2026-08-02); membership '
                              'not establishable from the artifact')
    return stamp


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

    # ---- LIVE FX (ONE v3/quotes/forex call), BEFORE the ~12h fetch ----------
    # `carveOut.FX_TO_USD` was an UNDATED hardcoded snapshot: ~7% median drift, 13
    # currencies past 10%, and on the 2026-08-07 CUR3K panel ELEVEN wrong universe-
    # membership decisions at the $25M floor plus 32 names in the wrong band.  One call
    # (~0.006% of a run's ~16,300) replaces it with dated rates that are validated PER
    # RATE -- the endpoint serves 125 dead pairs out of 1,550 behind HTTP 200, so a
    # response-level check would be worthless.
    #
    # AT LAUNCH, NOT AT USE, deliberately: a broken FX feed is then visible before the
    # 12 hours are spent rather than after, and a launch-time quote is at most ~12h old
    # when the conversions run -- an order of magnitude inside the 7-day staleness bar.
    #
    # A FAILURE HERE DOES NOT FALL BACK TO THE CONSTANTS AND DOES NOT STOP THE RUN.  It
    # leaves every reportedCurrency unresolvable, which is the already-built degradation:
    # the floor KEEPS every name and prints its NOT-ENFORCED banner, the bands are
    # skipped, the size tilt scores NEUTRAL.  See fx_rates.py and carveOut's FX source
    # state for why a stale rate is treated as the same kind of wrong number as a
    # missing currency.
    try:
        import fx_rates as fxr
        fxr.install_for_run(configdic.get('baseurl'), configdic.get('api_key'),
                            verbose=True)
    except Exception:
        import traceback as _tb
        import carveOut as _co_fx
        _co_fx.mark_fx_unavailable('fx_rates.install_for_run raised at launch')
        _tb.print_exc(file=sys.stderr)
        print('\n' + '!' * 78 + '\n'
              '!!! FX INSTALL FAILED AT LAUNCH -- run PROCEEDS with NO usable FX.       !!!\n'
              '!!! Every reportedCurrency is UNKNOWN: the $25M floor is NOT enforced,   !!!\n'
              '!!! the market-cap bands are skipped, and the size tilt is NEUTRAL.      !!!\n'
              '!!! It does NOT fall back to the old hardcoded constants -- deliberate.  !!!\n'
              + '!' * 78 + '\n', flush=True)

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
                                         sfilt ='all', mcapf = -1, fn = '', as_of=as_of,
                                         force_rebuild_maps=configdic.get('force_rebuild_maps'))
            # Assign variables and get financial data and calculate relevant metrics
            cdx_df, BoMetric_df, nrTaT = datandmetricdic['cdx_df'], datandmetricdic['BoMetric_df'], configdic['nrTaT']
            #  `configdic['fsMAnumber']` USED TO SIT BETWEEN compyear AND nrTaT HERE.  The
            #  smoothing it fed was deleted 2026-08-14 (CEO) after being proven inert at its
            #  production value of 1 -- see calcMetrics.calc_diff -- so the argument is gone
            #  and the four after it shift up one position.
            getfunddic = gdf.get_fundamentals_fmp(Tickers_df, cdx_df, BoMetric_df, baseurl, api_key, configdic['compyear'],
                                                  configdic['nrTaT'], configdic['startindex'],
                                                  configdic['period'], configdic['nrperiods'])
            newmanelimtckrs = list(set(manualelimtickers + list(set(getfunddic['tickersfailed']) - set(getfunddic['lenfail']))))
            datandmetricdic.update(getfunddic)
            datandmetricdic['manualelimtickers'] = newmanelimtckrs

            # REMOVED (review S9, 2026-07-26): a `hasCurrentYear` trim used to sit here --
            #     if lenhcy > 0 and lenhcy < 3/4 * (...):
            #         BoMetric_df = BoMetric_df.iloc[1:, :]; cdx_df = cdx_df.iloc[1:, :]
            # It was a SILENT NO-OP and had been since it was written: `getfunddic` still
            # holds the UNTRIMMED frames and is spread LAST into datandmetricdic a few lines
            # below, so the trim was discarded on every run.  That is the identical
            # "spread-last overwrites" defect the manual-elim provenance fix (C-3) documents
            # immediately beneath -- the fix landed on `manualelimtickers` and missed these
            # two lines.  Deleted rather than repaired: dropping the globally-first row of
            # two whole panels is not a defensible response to "fewer than 3/4 of names have
            # a current-year statement" (it removes ONE ticker's oldest row, chosen by
            # ingestion order), and nothing downstream depends on it having happened.
            # `hasCurrentYear` itself is still captured and still returned for diagnostics.

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

            # UNIVERSE PROVENANCE (2026-08-02).  The universe_* keys arrive via configdic
            # (configuration.getDataFetchConfiguration -> universes.provenance) and reach
            # the saved pickle through the configdic spread above.  Re-asserted and
            # ANNOUNCED here for the same reason the manual-elim stamp is: the artifact
            # filename carries only the universe NAME, and four names now denote a
            # different membership than they did before this date, so the definition
            # FINGERPRINT is the only thing that lets a later comparison be checked
            # rather than assumed.  Failing to state it is how this project ended up with
            # two irreconcilable beat-rate figures.
            datandmetricdic['universe_resolved_members'] = int(len(Tickers_df))
            datandmetricdic.update(universe_provenance_for_run(configdic))
            print('UNIVERSE PROVENANCE: %s (fingerprint %s), %d members resolved; '
                  'definition changed 2026-08-02: %s. Stamped into the saved artifact.'
                  % (configdic.get('universe', 'n/a'),
                     configdic.get('universe_fingerprint', 'n/a'),
                     len(Tickers_df),
                     configdic.get('universe_definition_changed', 'n/a')),
                  flush=True)

            # ---- VENDOR-CONTAMINATION DETECTOR (post-fetch, ZERO API calls) ----
            # Hash every (date, revenue, totalAssets) triple in the freshly fetched panel,
            # find sources sharing >= 3 of them, and flag the pairs whose company names do
            # NOT match.  This is the check that found `058820.KQ` serving CHIPOTLE's
            # statements under a KOSDAQ ticker -- and FMP still serves those rows today, so
            # this has to run on every fetch, not once.
            #
            # RUNS ON THE RAW PANEL, BEFORE the quality filter: the filter's job is to
            # remove rows, and a detector that only ever sees post-filter data cannot
            # report what the filter already took out -- which is the population most
            # worth knowing about.  Its output is a REPORT, never an action: legitimate
            # cross-listings and renames are indistinguishable from contamination to a
            # machine (Corteva/EIDP, NOW/DNOW, MicroStrategy/Strategy), so a human
            # promotes a finding into vendor_contamination.QUARANTINE_RULES.
            # Fully guarded -- a detector must never cost a 12-hour fetch.
            try:
                import vendor_contamination as _vc
                _vc_names = {}
                try:
                    _vc_names = dict(zip(Tickers_df['symbol'], Tickers_df['name']))
                except Exception:
                    pass
                _vc.run_detector_stage(datandmetricdic.get('cdx_df'), names=_vc_names,
                                       verbose=True)
            except Exception:
                import traceback as _tb
                _tb.print_exc(file=sys.stderr)
                print('WARNING: vendor-contamination detector did not run; the fetch and '
                      'every deliverable are UNAFFECTED, but this run has no record of '
                      'the shared-fundamentals check.', flush=True)

            # Apply data quality filter to freshly fetched data
            datandmetricdic = dq.apply_data_quality_filter(datandmetricdic, verbose=True, save_log=True)

            # UNIVERSE RECONCILIATION (2026-08-07).  Every name that entered must be
            # accounted for by exactly one of: it is in the panel, it failed the fetch,
            # or a filter removed it.  On the 2026-08-07 CUR3K run 3140 resolved, 445
            # failed and 2613 landed in the panel -- an 82-name residual that NOTHING
            # reported, because the only filter that could explain it had just
            # overwritten its own record (data_quality.py, now fixed).  Printing this
            # identity every run is what makes that whole class of silent drop
            # self-detecting: a residual of zero is a one-line confirmation, and a
            # non-zero residual is a banner naming the exact shortfall.
            print_universe_reconciliation(datandmetricdic, getfunddic, verbose=True)

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
            # BASIS CHECK ON THE LOAD PATH (review S10, 2026-07-26).  -loadbometric on a
            # pre-2026-07-19 pickle silently mixes bases: its `price` is the old
            # quarterly-PE derivation (~1/4 of the real price) and its `grahamNumber` is
            # FMP's quarterly one, so Stage-1 scores the OLD basis while Stage-2 applies
            # today's per-quarter corrections to the same cdx_df.  Verified on the 07-17
            # pickle: marketCap/(price*shares) median 3.9999, and uGrahamNumberToPrice
            # therefore runs ~2x loose.  Nothing announced it.  One loud banner does.
            utils.check_panel_basis(datandmetricdic,
                                    configdic['loadBoMetricfname'])
            # AND THE BoMetric HALF OF THE SAME HAZARD (2026-08-05).  The check above measures a
            # cdx-level ratio; the NaN-policy change altered two Stage-1 CRITERION COLUMNS
            # instead, and renamed nothing -- so calcScore's column-exact schema gate passes a
            # stale panel silently.  See utils.check_bometric_basis for the detector and why an
            # exact 0.0 is a reliable one.
            utils.check_bometric_basis(datandmetricdic,
                                       configdic['loadBoMetricfname'])

            # UNIVERSE PROVENANCE ON THE LOAD PATH (2026-08-03).  The fetch-path stamp
            # sits inside `if not loadBoMetricbool:`, and the configdic spread that
            # carries the universe_* keys is fetch-only too -- so a `-loadbometric` run
            # emitted its deliverables with NO universe provenance at all, which is the
            # half of the stamp that mattered most (re-scoring a saved panel is exactly
            # when two runs get compared). Stamped HERE, after the reload, because the
            # reload replaces `datandmetricdic` wholesale and would discard an earlier
            # stamp. Resolved from the LOADED panel, never from the active
            # -tickerfilter -- see universe_provenance_for_run.
            _loaded_stamp = universe_provenance_for_run(configdic, loaded=datandmetricdic)
            datandmetricdic.update(_loaded_stamp)
            print('UNIVERSE PROVENANCE (load path): %s (fingerprint %s) -- taken from '
                  'the LOADED panel, not from -tickerfilter (%s).'
                  % (_loaded_stamp.get('universe'),
                     _loaded_stamp.get('universe_fingerprint'),
                     configdic.get('universe')), flush=True)

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

        # UNIVERSE STAMP ON `resdic`, ON EVERY PATH (2026-08-03).  `resdic` is what
        # `writeResWrapper` turns into the CSV/XLSX deliverables and what the postRank
        # pickle is built from, so it is the one object that must always carry the stamp.
        # It arrives via the datandmetricdic merge on the compute path, but the
        # -loadboresults path replaces resdic wholesale, so normalise here rather than
        # trusting one branch. An already-present stamp WINS (it describes the data);
        # only a missing one is resolved.
        if not resdic.get('universe_fingerprint'):
            resdic.update(universe_provenance_for_run(configdic, loaded=resdic))

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
        # UNIVERSE STAMP ON THE postRank PICKLE (2026-08-03).  This is the artifact the
        # BACKTEST reads, i.e. the one most likely to be compared across runs -- and it
        # carried only `tickerfilter`, a NAME whose meaning changed on 2026-08-02. Two
        # postRank pickles could therefore claim the same universe while describing
        # different pools, which is exactly how this project acquired two irreconcilable
        # beat-rate figures. Taken from `resdic`, NOT re-derived from configdic: resdic's
        # stamp already accounts for the load paths (where the data's universe is the
        # LOADED panel's, not the active -tickerfilter's), so all three artifacts agree
        # by construction rather than by two call sites happening to match.
        postrank_data.update({k: resdic.get(k) for k in _UNIVERSE_KEYS})
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
            # HONOUR THE RECONCILIATION, NOT ONLY THE STATUS (2026-08-11).  This used
            # to read `status` alone, and `status` used to be set to 'success'
            # unconditionally -- so on the 2026-08-08..08-11 runs, where EVERY
            # directory failed to copy, the banner never fired and the tail of a
            # 12-hour log reported a clean transfer.  The reporting itself lives in
            # `report_transfer_outcome` so it is testable without invoking `main`.
            report_transfer_outcome(transfer_result, transfer_dir)
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



