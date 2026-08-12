"""
Incremental / crash-resilient artifact transfer for the ValuationAnalysis pipeline
(investment-filter restructure, 2026-07-13).

WHY THIS EXISTS
---------------
The pipeline used to copy its outputs to the Google-Drive-synced transfer dir ONLY
at end-of-run (Sbocker.transfer_outputs_to_drive).  A crash mid-run therefore lost
the Drive copy of everything and left no manifest.  This module factors the copy
mechanics OUT of that end-of-run function so that BOTH the end-of-run full transfer
AND the new per-phase incremental copies share ONE denylist and ONE key-file safety
net -- the denylist is defined here exactly once and reused everywhere.

SAFETY CONTRACT (all four are load-bearing)
-------------------------------------------
  1. The FMP API key (fmpAPIkey.txt / *key* / *.pem) is NEVER copied.  Enforced by
     the denylist on every path AND by a post-copy assertion that removes the key
     file if it somehow lands in the destination.  Explicit named artifacts only --
     this helper NEVER blanket-copies a tree.
  2. A copy failure NEVER propagates.  A multi-hour fetch must not die because a
     Google-Drive copy hiccuped -- every failure is a warn-and-continue.
  3. Idempotent: re-copying / overwriting an already-copied artifact is fine.
  4. transfer_dir falsy -> strict no-op (existing behaviour byte-identical).

NON-FATAL IS NOT SILENT (added 2026-08-07)
------------------------------------------
Contract 2 above is right -- but for a long time it was implemented as *silent*:
a listed artifact that produced nothing looked, from the console, exactly like a
listed artifact that transferred perfectly.  The 2026-08-07 CUR3K run copied every
artifact EXCEPT run_logs/ and nobody could tell from the run output.  `reconcile_
transfer` closes that: after the copy, it re-derives what SHOULD be at the
destination and checks what actually IS, then prints a per-group verdict loudly.
Warn-and-continue still holds -- the run does not die -- but the operator reading
the tail of a run can now tell whether the transfer was COMPLETE.
"""
import fnmatch
import os
import shutil
from pathlib import Path

# Single source of truth for the denylist.  transfer_outputs_to_drive imports this
# same constant so the incremental path and the end-of-run path can never diverge.
DENYLIST_PATTERNS = ['*key*', '*pem', 'fmpAPIkey.txt']

# --- WHERE THE PIPELINE'S EVIDENCE CSVs ARE WRITTEN (CEO, 2026-08-10) --------------------
# THE REPO ROOT, i.e. the run's CWD.  It was `output/` for five of them, and the 2026-08-10
# run is the case that decided it: EVERY root-level artifact from that run reached Drive, and
# `output/` and `logs/` DID NOT -- no DedupSurvivorReport, no FxRates, no
# CurrencyExclusionStatus, no DelistedPrune, no run log for 08-10 ever arrived.  Drive's copies
# of those groups stop at 08-08 / 08-07.  The dedup breakdown for 08-10 was recoverable at all
# only because it happened to be inside a pickle.
#
# THE ARGUMENT THAT USED TO POINT THE OTHER WAY IS NOT WRONG, IT IS SPENT.  `Sbocker` declined
# a top-level glob for `DedupSurvivorReport` on the grounds that "output/ already ships whole,
# so the evidence cannot be lost to a pattern that stops matching after a filename change" and
# that "a top-level pattern would be a DEAD GLOB" for a file living in `output/`.  Both legs
# INVERT once the file is at root: the glob is live because that is where `glob.glob` looks,
# and "ships whole" is exactly the property that failed.  A directory that ships whole and does
# not arrive ships nothing.
#
# ONE NAMED CONSTANT rather than seven string literals, so the next move is one edit and so no
# writer can drift from the transfer manifest by accident.  It lives HERE, beside the denylist,
# because this module is already the single source of truth for what travels.
#
# THE ONE FILE THAT WAS ALREADY AT ROOT AND MUST STAY THERE: `MeanBarCalibration-*.csv`.
# `meanBars._prior_streaks` READS the prior calibration CSVs back from the directory it writes
# to, to chain the breach-streak hysteresis, and it is git-TRACKED -- moving it would strand
# that history and silently restart every streak at zero.  It is unaffected by this change.
EVIDENCE_DIR = '.'

# Top-level key filenames the post-copy safety net actively removes if present.
KEY_FILENAMES = ('fmpAPIkey.txt',)


def is_denied(filename):
    """True if the BASENAME of <filename> matches any denylist pattern
    (case-insensitive).  Matching on the basename (not the full path) means a file
    is denied for its OWN name, never merely because a parent directory happens to
    contain 'key'/'pem'.

    REAL GLOB SEMANTICS (fixed 2026-08-11).  This used to do
    `prefix = pattern.replace('*', ''); if prefix in fname_lower`, i.e. it turned
    EVERY pattern into a SUBSTRING test -- so `'*pem'`, which as a glob means "ends
    with pem" (a `.pem` certificate), actually meant "contains pem" and denied
    `Pemex_bond_notes.md` and `PEMBINA_pipeline_dcf.xlsx`.  That is a plain bug
    against the patterns' own meaning, and it is not cosmetic: this same predicate
    governs the COPY direction, so a research file whose name merely contained "pem"
    sitting in `output/` was SILENTLY NOT TRANSFERRED -- a transfer gap of exactly
    the class this module's 2026-08-11 work exists to end.  `fnmatch` gives the
    patterns the meaning they are written in.

    WHAT THIS DELIBERATELY DOES NOT CHANGE: `'*key*'` has wildcards on BOTH sides, so
    it is a substring test BY DESIGN and still denies `Turkey_exposure_2026.csv`,
    `monkey_basket.csv` and `hockey_stick_screen.csv`.  That is a real remaining
    transfer gap; narrowing it is a denylist-policy decision (it is also what catches
    FMP's `key-metrics_*` cache files), not a bug fix, and is deliberately NOT made
    here.  The credential match is unchanged: `fmpAPIkey.txt` still denies by exact
    name AND by `*key*`, and `*.pem` still denies."""
    fname_lower = os.path.basename(str(filename).rstrip('/\\')).lower()
    for pattern in DENYLIST_PATTERNS:
        if fnmatch.fnmatchcase(fname_lower, pattern.lower()):
            return True
    return False


def assert_no_key_file(transfer_path, verbose=True):
    """Post-copy safety net: REPORT whether a key file is sitting at the top level of
    <transfer_path>.  Returns True if the destination is clean.

    REPORT-ONLY SINCE 2026-08-11 (CEO: "Please don't go deleting my API key").  This
    used to `unlink()` what it found.  It no longer deletes anything, for two reasons
    that both point the same way:

      * THE COPY DIRECTION NEVER WRITES A DENIED FILE -- independently instrumented at
        the Windows copy primitive (7 denied files planted at 5 depths: 5 files
        written, 0 denied, none even transiently).  So a key file AT the destination
        did not get there by us; it was put there by a human, and deleting a human's
        file was never this module's business.
      * ON A SYNCED FOLDER, DELETING IS NOT CONTAINMENT.  `os.remove` does not
        unpublish what Drive already uploaded, and there is no Recycle Bin on this
        path.  A credential-shaped file on a synced destination is an INCIDENT TO
        REPORT, not a mess to tidy -- and quietly tidying it is precisely how it would
        stop being reported.

    The caller escalates: finding one fails the run's transfer status loudly."""
    transfer_path = Path(transfer_path)
    clean = True
    for name in KEY_FILENAMES:
        target = transfer_path / name
        try:
            if target.exists():
                clean = False
                if verbose:
                    print(f"[TRANSFER] CRITICAL: {name} found in transfer dir -- "
                          f"NOT removed (report-only); remove it by hand.")
        except Exception as e:
            clean = False
            if verbose:
                print(f"[TRANSFER] ERROR checking for key file {name}: {e}")
    return clean


def find_denied_files(dest_root, verbose=True):
    """RECURSIVE destination-side CHECK: return the paths of every file under
    <dest_root> whose own basename matches the denylist.  **Deletes nothing.**

    WHY IT ONLY REPORTS (CEO, 2026-08-11: "Please don't go deleting my API key").
    The first version of this function DELETED what it found, as compensation for
    dropping the `shutil.rmtree` of the destination that Google Drive refuses.  Two
    findings killed that:

      * THE THING IT COMPENSATED FOR IS NOT NEEDED.  The copy direction never writes
        a denied file -- instrumented at the Windows copy primitive, 7 denied files
        planted at 5 depths, 5 written, 0 denied, none even transiently.  A denied
        file at the destination can therefore only have been put there BY A HUMAN.
      * DELETING WAS DESTROYING RESEARCH.  `'*key*'` is a substring pattern, so the
        measured deletions on realistic filenames included `Turkey_exposure_2026.csv`,
        `Keystone_Corp_thesis.docx`, `hockey_stick_screen.csv` and (before the
        `is_denied` glob fix) `Pemex_bond_notes.md` and `PEMBINA_pipeline_dcf.xlsx` --
        with NO Windows Recycle Bin on this path (`os.remove` -> `DeleteFileW`).  The
        destination holds the CEO's investment research.  That is unacceptable, and
        narrowing the predicate would not have made deleting a human's files our
        business in the first place.

    So this is a DETECTOR.  A credential-shaped file on a cloud-synced folder is an
    incident to report -- deleting it does not unpublish what Drive already uploaded,
    it only removes the evidence that it was there.  The caller escalates: any hit
    fails the run's transfer status loudly and names every path.

    `is_denied` is the single source of truth for what counts as denied; the matching
    is not restated here.  NEVER raises, and NEVER writes."""
    found = []
    try:
        root = Path(dest_root)
        if not root.exists():
            return found
        for dirpath, _dirs, files in os.walk(str(root)):
            for fname in files:
                if not is_denied(fname):
                    continue
                full = os.path.join(dirpath, fname)
                found.append(full)
                if verbose:
                    print(f"[TRANSFER] CRITICAL: denylisted file present at the "
                          f"destination (NOT removed): {full}")
    except Exception as e:
        if verbose:
            print(f"[TRANSFER] WARNING: destination denylist check failed: {e}")
    return found


# --- THE PIPELINE / NON-PIPELINE SPLIT (CEO, 2026-08-11) -----------------------------
# The transfer destination is being restructured into
#     <drive>\valuationTransfer\pipeline\       <- the program writes ONLY here
#     <drive>\valuationTransfer\non-pipeline\   <- the CEO's manual drop zone, never touched
# and the run is pointed at the `pipeline` leaf.  The transfer logic needs no change for
# that -- it already writes wherever `-transfer_dir` points -- but the new shape makes
# exactly ONE new misconfiguration possible: passing the PARENT.  That would put run
# artifacts beside the manual files and bring the manual files inside the destination-side
# denylist check, so it is refused rather than warned about.
NON_PIPELINE_DIRNAME = 'non-pipeline'
PIPELINE_DIRNAME = 'pipeline'


def looks_like_transfer_parent(transfer_dir):
    """If <transfer_dir> CONTAINS a `non-pipeline` directory it is the PARENT of the
    split, not the pipeline leaf.  Returns a specific operator-facing message naming
    the directory that should have been passed, or None if the target looks right.

    Detection is by the presence of the sibling, not by the path's spelling: the CEO
    creates `non-pipeline/`, so its presence one level down is the fact that
    distinguishes the parent from the leaf.  NEVER raises."""
    try:
        if not transfer_dir:
            return None
        path = Path(transfer_dir)
        if not (path / NON_PIPELINE_DIRNAME).is_dir():
            return None
        return (f"target looks like the PARENT of the pipeline/non-pipeline split: "
                f"it contains a '{NON_PIPELINE_DIRNAME}/' directory. Point "
                f"-transfer_dir at '{path / PIPELINE_DIRNAME}' instead -- writing run "
                f"artifacts here would mix them with the manual drop zone.")
    except Exception:
        # A target we cannot even inspect is not our business to veto.
        return None


_README_NAME = 'README-pipeline-managed.txt'
_README_TEXT = """This folder is written by the ValuationAnalysis pipeline.

  * Every run ADDS files here and OVERWRITES files of the same name.
  * Nothing here is ever DELETED by the pipeline -- it is an OUTBOX, not a mirror, so
    a file that no longer exists on the run machine still stays here.
  * Put manual files in the sibling '{non_pipeline}' folder instead. The pipeline never
    reads, writes or touches that folder.
  * If a file whose name looks like a credential (matching {patterns}) turns up in
    here, the run REPORTS it loudly and does NOT delete it -- remove it by hand.

This file is a label, not a lock: it is written once, when the pipeline first creates
this folder, and is never rewritten.
"""


def write_pipeline_readme(transfer_path, verbose=True):
    """Drop a one-time label into a transfer directory the pipeline just created.
    Returns the path written, or None (already present / could not write).

    NOT rewritten on every run: it is a note to a human opening the folder, and
    rewriting it would be one more thing a run does to a synced directory for no
    reason.  NEVER raises -- a missing label must never affect a 12-hour run."""
    try:
        target = Path(transfer_path) / _README_NAME
        if target.exists():
            return None
        text = _README_TEXT.format(non_pipeline=NON_PIPELINE_DIRNAME,
                                   patterns=', '.join(DENYLIST_PATTERNS))
        with open(target, 'w', encoding='utf-8') as fh:
            fh.write(text)
        if verbose:
            print(f"[TRANSFER] Wrote {_README_NAME} into {transfer_path}")
        return str(target)
    except Exception as e:
        if verbose:
            print(f"[TRANSFER] WARNING: could not write {_README_NAME}: {e}")
        return None


def probe_transfer_target(transfer_dir):
    """Non-destructive launch-time probe of the transfer target, used to surface
    an unusable Drive path BEFORE the ~12h fetch begins (rather than discovering
    it only at end-of-run).  Returns a dict:
        {'ok': bool, 'exists': bool, 'writable': bool, 'detail': str}
    'ok' is True only when the target's parent exists, the target dir is
    creatable, AND a probe file can be written+deleted there.  Mirrors the
    parent-must-exist contract of _ensure_transfer_dir / transfer_outputs_to_drive
    so the launch verdict matches what the actual copy will do.  NEVER raises."""
    info = {'ok': False, 'exists': False, 'writable': False, 'detail': ''}
    try:
        if not transfer_dir:
            info['detail'] = 'no target resolved'
            return info
        transfer_path = Path(transfer_dir)
        if not transfer_path.is_absolute():
            # Refuse to create a stray junk dir under the run's cwd.
            info['detail'] = (f'target is not an absolute path ({transfer_dir}) '
                              f'-- refusing to probe/create under cwd')
            return info
        parent = transfer_path.parent
        if not parent.exists():
            info['detail'] = f'parent dir missing ({parent}) -- Drive not mounted?'
            return info
        # Catch the pipeline/non-pipeline mix-up BEFORE the ~12h fetch, which is the
        # entire reason this probe exists.
        misconfig = looks_like_transfer_parent(transfer_path)
        if misconfig:
            info['detail'] = misconfig
            return info
        try:
            transfer_path.mkdir(parents=False, exist_ok=True)
            info['exists'] = True
        except Exception as e:
            info['detail'] = f'cannot create target dir: {e}'
            return info
        probe = transfer_path / '.transfer_write_probe'
        try:
            with open(probe, 'w') as fh:
                fh.write('probe')
            probe.unlink()
            info['writable'] = True
            info['ok'] = True
            info['detail'] = 'target exists and is writable'
        except Exception as e:
            info['detail'] = f'target exists but is NOT writable: {e}'
        return info
    except Exception as e:
        info['detail'] = f'probe error: {e}'
        return info


def expected_dest_names(sources, subdir=None):
    """Given local source paths, return the destination-RELATIVE paths the copy
    should have produced.  Files land at <transfer>/<basename>; a named directory
    lands at <transfer>/<dirname>/<relpath>, so a directory's expectation is its
    FILES (an empty dir expects nothing -- which is exactly how a never-written
    artifact reports as EMPTY rather than as a healthy copy).

    <subdir> overrides the destination sub-path for directory sources.  It exists
    because the two copiers DISAGREE on where a directory lands: Sbocker's
    end-of-run path uses `transfer/<dirpat>` (keeping 'baseline_tools/price_data'
    nested) while copy_artifacts_to_transfer_dir uses `transfer/<basename>`.  The
    caller states which convention applies rather than this helper guessing."""
    out = []
    for src in sources or []:
        try:
            src = str(src)
            if os.path.isdir(src):
                base = os.path.basename(src.rstrip('/\\'))
                for root, _dirs, files in os.walk(src):
                    for fn in files:
                        if is_denied(fn):
                            continue
                        rel = os.path.relpath(os.path.join(root, fn), src)
                        out.append(os.path.join(subdir or base, rel))
            elif os.path.isfile(src):
                if not is_denied(os.path.basename(src)):
                    out.append(os.path.basename(src))
        except Exception:
            # A source we cannot even inspect contributes no expectation; the group
            # will read as EMPTY, which is the honest (and loud) outcome.
            continue
    return out


def reconcile_transfer(transfer_dir, groups, verbose=True):
    """Post-transfer reconciliation: for every artifact GROUP the run claims to
    transfer, did anything actually land at the destination?

    WHY: warn-and-continue (contract 2) means no copy failure is fatal -- but for a
    long time it also meant no copy failure was VISIBLE.  A group that produced
    nothing locally, a group whose copy raised and was swallowed, and a group that
    transferred perfectly all printed the same thing: effectively nothing.  This
    function makes the difference impossible to miss from the tail of a run.

    Parameters
    ----------
    transfer_dir : str
    groups : ordered list of (group_name, [local source paths]) or
             (group_name, [local source paths], dest_subdir) -- the SAME allowlist
             the copy loop walked, so the two can never disagree.

    Returns {'complete': bool, 'groups_total', 'groups_complete', 'missing': [...],
             'incomplete': [...], 'detail': {group: (landed, expected)}, 'summary': str}

    NEVER raises.  Verifies against the DESTINATION (not against a copy-loop
    counter), so a swallowed exception cannot report itself as a success.
    """
    res = {'complete': True, 'groups_total': 0, 'groups_complete': 0,
           'missing': [], 'incomplete': [], 'detail': {}, 'summary': '',
           'unreconciled': False}
    try:
        transfer_path = Path(transfer_dir) if transfer_dir else None
        if transfer_path is None or not transfer_path.exists():
            res['complete'] = False
            # THE TARGET VANISHED BETWEEN THE COPY AND THIS CHECK (2026-08-11).  This
            # branch leaves `missing` AND `incomplete` both empty, so a status gate
            # built on those two -- which is the right gate, because 8 of the 31
            # pattern groups are legitimately empty on a healthy run -- cannot see it,
            # and the run reported `success` after nothing reached the Drive.  Same
            # signature as the bug this module's 2026-08-11 work exists to close.
            # A separate flag, because this one fires on EXACTLY ZERO healthy runs:
            # there is no alarm-fatigue cost, so the reasoning that settled the
            # missing-vs-incomplete question does not extend here.
            res['unreconciled'] = True
            res['summary'] = 'transfer target unavailable -- nothing could be reconciled'
            if verbose:
                print(f"[TRANSFER] RECONCILE: target unavailable ({transfer_dir})")
            return res

        for entry in groups:
            name, sources = entry[0], entry[1]
            subdir = entry[2] if len(entry) > 2 else None
            res['groups_total'] += 1
            expected = expected_dest_names(sources, subdir=subdir)
            landed = 0
            for rel in expected:
                try:
                    dest = transfer_path / rel
                    if dest.exists() and dest.stat().st_size > 0:
                        landed += 1
                except Exception:
                    pass
            res['detail'][name] = (landed, len(expected))
            if len(expected) == 0:
                # Nothing was produced locally -> nothing could transfer.  This is
                # the run_logs case: the copy "succeeded" over an empty dir.
                res['missing'].append(name)
                res['complete'] = False
            elif landed < len(expected):
                res['incomplete'].append(name)
                res['complete'] = False
            else:
                res['groups_complete'] += 1

        res['summary'] = (f"transferred {res['groups_complete']}/{res['groups_total']} "
                          f"artifact groups")
        if res['missing']:
            res['summary'] += f"; NOTHING PRODUCED: {', '.join(res['missing'])}"
        if res['incomplete']:
            res['summary'] += f"; PARTIAL: {', '.join(res['incomplete'])}"

        if verbose:
            print("\n" + "-" * 70)
            print(f"[TRANSFER] RECONCILIATION: {res['summary']}")
            for name, (landed, exp) in res['detail'].items():
                mark = 'OK     ' if (exp and landed == exp) else ('EMPTY  ' if not exp else 'PARTIAL')
                print(f"[TRANSFER]   {mark} {name}: {landed}/{exp} files at destination")
            if not res['complete']:
                print("!" * 70)
                print("!!! TRANSFER INCOMPLETE -- one or more listed artifacts did NOT reach the Drive.")
                if res['missing']:
                    print(f"!!! NOTHING PRODUCED LOCALLY (so nothing to copy): {', '.join(res['missing'])}")
                    print("!!!   -> the WRITER for these did not run.  Fixing the copier will not help.")
                if res['incomplete']:
                    print(f"!!! PARTIALLY COPIED (copy failed and was swallowed): {', '.join(res['incomplete'])}")
                    print("!!!   -> re-copy these MANUALLY before relying on the Drive snapshot.")
                print("!" * 70)
            print("-" * 70 + "\n")
    except Exception as e:
        res['complete'] = False
        res['summary'] = f'reconciliation error: {e}'
        if verbose:
            print(f"[TRANSFER] WARNING: reconciliation failed safely: {e}")
    return res


def _ensure_transfer_dir(transfer_dir, verbose=True):
    """Resolve/create the transfer dir.  Returns a Path on success, or None on any
    failure (warn + continue -- never raises).  Mirrors the end-of-run function's
    parent-must-exist guard so an unmounted Drive is a graceful skip, not a crash."""
    try:
        transfer_path = Path(transfer_dir)
        parent = transfer_path.parent
        if not parent.exists():
            if verbose:
                print(f"[TRANSFER] WARNING: parent dir missing ({parent}); skipping copy.")
            return None
        transfer_path.mkdir(parents=False, exist_ok=True)
        return transfer_path
    except Exception as e:
        if verbose:
            print(f"[TRANSFER] WARNING: could not prepare transfer dir: {e}")
        return None


def copy_artifacts_to_transfer_dir(transfer_dir, artifacts, verbose=True):
    """Copy an EXPLICIT set of just-written artifacts (files or named dirs) to
    <transfer_dir>, honoring the denylist.  This is the incremental crash-resilience
    helper called at each phase boundary.

    Contract (see module docstring): NEVER raises; transfer_dir falsy -> no-op;
    denylist holds on every path; missing sources are skipped (not an error);
    idempotent overwrite.

    Returns a small result dict {status, copied, denied, errors, files}.
    """
    result = {'status': 'skipped', 'copied': 0, 'denied': 0, 'errors': 0, 'files': [],
              'refused': [], 'key_file_at_destination': False}
    if not transfer_dir:
        return result
    try:
        transfer_path = _ensure_transfer_dir(transfer_dir, verbose=verbose)
        if transfer_path is None:
            result['status'] = 'warning'
            return result

        for artifact in artifacts:
            if not artifact:
                continue
            try:
                src = str(artifact)
                base = os.path.basename(src.rstrip('/\\'))

                # DENYLIST FIRST -- a secret is never copied under any path.
                if is_denied(base):
                    result['denied'] += 1
                    if verbose:
                        print(f"[TRANSFER] DENIED (denylist): {src}")
                    continue

                if os.path.isdir(src):
                    # DIRECTORIES ARE NOT SUPPORTED HERE (closed by DELETION, CEO
                    # approved 2026-08-11).  This branch used to do
                    # rmtree-then-copytree with WHOLE-TREE denial -- both of the
                    # defects that cost four production runs every `logs/`,
                    # `output/` and `run_logs/` on the Drive, still sitting here in
                    # the incremental copier.  It is DORMANT: all six call sites
                    # (Sbocker 972/1046/1086/1115, delisted_ingest._flush_boundary at
                    # 958/977/999/1037) pass file paths only, verified twice
                    # independently.  So rather than repair a path nothing uses -- and
                    # carry two copies of the same subtle fix -- the input is refused.
                    # A directory here is a programming error, and it is now a LOUD
                    # one instead of a silent evidence-loss the day someone tries it.
                    result['errors'] += 1
                    result['refused'].append(src)
                    if verbose:
                        print(f"[TRANSFER] ERROR: directories are not supported by "
                              f"copy_artifacts_to_transfer_dir -- refusing {src!r}. "
                              f"Pass explicit file paths, or use "
                              f"Sbocker.transfer_outputs_to_drive for directories.")
                    continue
                elif os.path.isfile(src):
                    dest = transfer_path / base
                    shutil.copy2(src, str(dest))
                    result['copied'] += 1
                    result['files'].append(src)
                    if verbose:
                        print(f"[TRANSFER] Copied: {src} -> {dest}")
                else:
                    # Not yet on disk -> nothing to copy (not an error: a phase may
                    # legitimately not have produced this artifact).
                    if verbose:
                        print(f"[TRANSFER] Skipped (not found): {src}")
            except Exception as e:
                result['errors'] += 1
                if verbose:
                    print(f"[TRANSFER] WARNING: failed to copy {artifact}: {e}")
                # continue -- a single copy hiccup must never kill the run.

        # Safety net: REPORT (never delete -- see assert_no_key_file) a key file at
        # the destination top level.  A copy hiccup stays 'success' as it always has
        # (contract 2: warn and continue); a refused DIRECTORY or a key file sitting
        # on the synced folder is an error, because neither is something a run should
        # be allowed to scroll past.
        result['key_file_at_destination'] = not assert_no_key_file(transfer_path,
                                                                  verbose=verbose)
        result['status'] = ('error' if (result['refused']
                                        or result['key_file_at_destination'])
                            else 'success')
    except Exception as e:
        # Absolute backstop -- the helper must NEVER propagate to the run.
        result['status'] = 'error'
        if verbose:
            print(f"[TRANSFER] WARNING: incremental copy aborted safely: {e}")
    return result
