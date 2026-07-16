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
