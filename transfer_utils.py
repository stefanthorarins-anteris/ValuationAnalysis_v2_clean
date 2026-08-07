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
import os
import shutil
from pathlib import Path

# Single source of truth for the denylist.  transfer_outputs_to_drive imports this
# same constant so the incremental path and the end-of-run path can never diverge.
DENYLIST_PATTERNS = ['*key*', '*pem', 'fmpAPIkey.txt']

# Top-level key filenames the post-copy safety net actively removes if present.
KEY_FILENAMES = ('fmpAPIkey.txt',)


def is_denied(filename):
    """True if the BASENAME of <filename> matches any denylist pattern
    (case-insensitive).  Matching on the basename (not the full path) means a file
    is denied for its OWN name, never merely because a parent directory happens to
    contain 'key'/'pem'."""
    fname_lower = os.path.basename(str(filename).rstrip('/\\')).lower()
    for pattern in DENYLIST_PATTERNS:
        if '*' in pattern:
            prefix = pattern.replace('*', '')
            if prefix in fname_lower:
                return True
        else:
            if fname_lower == pattern.lower():
                return True
    return False


def assert_no_key_file(transfer_path, verbose=True):
    """Post-copy safety net: if a key file somehow reached <transfer_path>, delete
    it.  Returns True if the destination is clean (no key file present)."""
    transfer_path = Path(transfer_path)
    clean = True
    for name in KEY_FILENAMES:
        target = transfer_path / name
        try:
            if target.exists():
                clean = False
                if verbose:
                    print(f"[TRANSFER] CRITICAL: {name} found in transfer dir -- removing it.")
                target.unlink()
                clean = True
        except Exception as e:
            clean = False
            if verbose:
                print(f"[TRANSFER] ERROR removing key file {name}: {e}")
    return clean


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
           'missing': [], 'incomplete': [], 'detail': {}, 'summary': ''}
    try:
        transfer_path = Path(transfer_dir) if transfer_dir else None
        if transfer_path is None or not transfer_path.exists():
            res['complete'] = False
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
    result = {'status': 'skipped', 'copied': 0, 'denied': 0, 'errors': 0, 'files': []}
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
                    # Named-directory copy: skip the WHOLE dir if it contains any
                    # denied file (matches end-of-run behaviour).
                    denied_inside = False
                    for root, _dirs, files in os.walk(src):
                        for fn in files:
                            if is_denied(fn):
                                denied_inside = True
                                result['denied'] += 1
                                if verbose:
                                    print(f"[TRANSFER] DENIED (denylist): {os.path.join(root, fn)}")
                    if denied_inside:
                        continue
                    dest_dir = transfer_path / base
                    if dest_dir.exists():
                        shutil.rmtree(str(dest_dir))
                    shutil.copytree(src, str(dest_dir))
                    result['copied'] += 1
                    result['files'].append(src)
                    if verbose:
                        print(f"[TRANSFER] Copied dir: {src}/ -> {dest_dir}")
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

        # Safety net: ensure no key file slipped into the destination top level.
        assert_no_key_file(transfer_path, verbose=verbose)
        result['status'] = 'success'
    except Exception as e:
        # Absolute backstop -- the helper must NEVER propagate to the run.
        result['status'] = 'error'
        if verbose:
            print(f"[TRANSFER] WARNING: incremental copy aborted safely: {e}")
    return result
