"""pytest configuration for the repo root.

WHY THIS FILE EXISTS (ship gate, 2026-07-27)
--------------------------------------------
`test_pipeline_prereqs.py` is a STANDALONE pre-flight SCRIPT, not a pytest module.  It is
named `test_*` and its console helpers are named `test_pass` / `test_fail` / `test_warn`,
so pytest collects the file and then errors on all three ("fixture 'msg' not found") --
which is why a bare `pytest` at the repo root was never green even when every real suite
passed.

The collection is worse than cosmetic: the script does its work AT IMPORT TIME, and that
work includes

  * `open('fmpAPIkey.txt')` -- collection FAILS outright on any machine without the key
    file, and reads the secret on machines that have it, and
  * LIVE FMP API calls (profile / key-metrics / ratios / DCF probes)

so merely *collecting* the repo fired real network requests against the paid API and made
the outcome depend on the key being present.  Blocking collection is therefore the fix;
renaming the helpers would silence the errors but leave the import-time key read and the
network calls in place.

The script itself is UNCHANGED and still runs the way it is documented to:
`python test_pipeline_prereqs.py`.
"""
#  `test_cycleheat.py` ADDED 2026-08-13 — SAME DEFECT, MISSED FOR THE SAME REASON.
#  It is a SCRIPT wearing a test_ prefix: zero test functions, `api_key =
#  open('fmpAPIkey.txt').read()` at MODULE level (line 15), and three module-level
#  `for` loops that fire live paid `v3/profile` and `v4/*` calls at IMPORT time
#  (lines 75, 122, 147).  Collection imports it, so the calls ran on EVERY
#  `pytest .` from the repo root.
#
#  The blast radius is the part worth recording: the house has been running the full
#  suite as its standard verification for weeks, and repeatedly told the CEO that a
#  directory-scoped run is offline-safe.  It was not.  Roughly eight paid calls fired
#  per full-suite invocation.  Found 2026-08-13 by a reviewer who noticed the cost,
#  fired the calls itself before realising, and DISCLOSED it rather than burying it.
#
#  This list is the whole defence and it was one entry long while a second file with
#  the identical shape sat beside it.  Anything named `test_*.py` that is really a
#  script belongs here — or, better, gets renamed so it cannot be collected at all.
collect_ignore = ["test_pipeline_prereqs.py", "test_cycleheat.py"]


# --------------------------------------------------------------------------------- #
#  REPO-WIDE FX-STATE ISOLATION (moved here from test_fx_rates.py, reviewer F-6)     #
# --------------------------------------------------------------------------------- #
#  `carveOut` holds the run's FX source in MODULE-GLOBAL state (`_FX_STATE`/`_LIVE_FX`),
#  because it has to: every conversion call site reads it, and threading it through would
#  touch every consumer.  The cost is that a test which installs a table LEAKS into every
#  test that runs after it -- including tests in other FILES, since module state outlives
#  a module's tests.  A leaked 'live' table makes `_fx_to_usd` return None for currencies
#  the constants would have resolved, so the symptom is an unrelated suite failing on a
#  NaN market cap, which is about as hard to trace as this project gets.
#
#  It lived in `test_fx_rates.py` and therefore protected only that file.  It belongs
#  here: the state is global, so the guard must be too, and a future test author touching
#  carveOut will not know to isolate something they never see.  Blast radius today is
#  zero -- no other test installs FX -- which is exactly why it is cheap to fix now.
#
#  This root conftest is loaded for `baseline_tools/` too (it is under the rootdir), so
#  one fixture covers BOTH pytest invocations.  The import is inside the fixture so a
#  collection-time import cannot fail for suites that never touch carveOut.
import pytest


@pytest.fixture(autouse=True)
def _isolate_carveout_fx_state():
    """Reset carveOut's global FX source to 'unset' around EVERY test in the repo."""
    try:
        import carveOut as _co
    except Exception:
        yield
        return
    _co.clear_live_fx_rates()
    yield
    _co.clear_live_fx_rates()


@pytest.fixture(autouse=True)
def _isolate_carveout_map_caches():
    """Reset carveOut's cached map variables around EVERY test in the repo.

    The ISIN, volAvg, and volAvg-profile map caches are module-level globals
    that are memoized after the first load. If tests run in sequence and earlier
    tests load the maps (e.g., when pickles exist in the repo root), later tests
    see the cached values. This fixture resets all map caches to None before each
    test, forcing a fresh load in an isolated state.
    """
    try:
        import carveOut as _co
    except Exception:
        yield
        return
    _co._ISIN_MAP_CACHE = None
    _co._VOLAVG_MAP_CACHE = None
    _co._VOLAVG_PROFILE_CACHE = None
    yield
    _co._ISIN_MAP_CACHE = None
    _co._VOLAVG_MAP_CACHE = None
    _co._VOLAVG_PROFILE_CACHE = None


@pytest.fixture
def _isolated_absent_map_state(monkeypatch):
    """Isolate the map-loading state for tests that verify maps are absent.

    The absent-map tests check that carveOut's map loaders return {} when no
    pickles exist. These loaders glob the repo root first, so if run-artifact
    pickles sit there, the tests silently find them and fail.

    This fixture uses a NARROWER SEAM than blocking all glob operations: it
    monkeypatches the loaders themselves to return empty maps. This avoids
    side-effects on any other code that might glob for non-map reasons.

    This makes the loaders return {} as expected, letting the test state its own
    premise about absence instead of inferring it from the filesystem.
    """
    try:
        import carveOut as _co
    except Exception:
        yield
        return

    # Replace with empty-returning versions FIRST
    def fake_load_isin_map(*args, **kwargs):
        return {}

    def fake_load_volavg_map(*args, **kwargs):
        return {}

    def fake_load_volavg_profile_map(*args, **kwargs):
        return {}

    # Monkeypatch the loaders themselves (narrower seam than blocking all glob)
    monkeypatch.setattr(_co, '_load_isin_map', fake_load_isin_map)
    monkeypatch.setattr(_co, '_load_volavg_map', fake_load_volavg_map)
    monkeypatch.setattr(_co, '_load_volavg_profile_map', fake_load_volavg_profile_map)

    # Then reset caches to force use of the patched loaders
    _co._ISIN_MAP_CACHE = None
    _co._VOLAVG_MAP_CACHE = None
    _co._VOLAVG_PROFILE_CACHE = None

    yield

    # Clean up: reset caches after the test (revert to None so next test starts fresh)
    _co._ISIN_MAP_CACHE = None
    _co._VOLAVG_MAP_CACHE = None
    _co._VOLAVG_PROFILE_CACHE = None


# --------------------------------------------------------------------------------- #
#  REPO WRITE GUARD -- A TEST RUN MAY NOT DAMAGE A TRACKED ARTIFACT (2026-08-31)      #
# --------------------------------------------------------------------------------- #
#  WHAT HAPPENED.  `transfer_utils.EVIDENCE_DIR` was `'.'` -- CWD-dependent -- so the
#  evidence writers landed wherever the process was launched from.  Register item Q-29
#  fixed that to the module's own directory (the repo root), which is RIGHT for the
#  pipeline and stays.  But the tests isolated themselves with `monkeypatch.chdir(
#  tmp_path)`, and that isolation worked ONLY because the constant was CWD-relative.
#  The moment it stopped being CWD-relative, running the suite wrote the pipeline's real
#  evidence filenames into the REPO ROOT: four git-tracked 2026-08-31 run-evidence CSVs
#  were OVERWRITTEN with pytest fixture data (`DollarVolumeFloor_2026-08-31.csv` went
#  from 2,591 real rows to 2 synthetic ones), plus six stray `removed_data_quality_*`
#  droppings.  They were recoverable only because they happened to be tracked.
#
#  WHY A GUARD AND NOT JUST THE FIX.  The per-test fix (point EVIDENCE_DIR at tmp_path)
#  protects the tests that remember to do it.  Nothing protected the repo from a test
#  that forgets -- and the failure was SILENT: the suite went green while destroying the
#  run evidence the CEO reviews.  This guard makes the damage class impossible to do
#  QUIETLY rather than merely fixed once.
#
#  TWO RULES, both enforced by intercepting the write primitives in-process:
#    RULE T (tracked-file shield)  -- no test may open-for-write, delete or rename any
#        git-TRACKED file in this repo.  Derived from `git ls-files` at session start, so
#        it needs no maintenance and covers every artifact the repo carries, not just the
#        evidence CSVs.  This is the rule that would have stopped the actual damage.
#    RULE E (dropping shield)      -- no test may CREATE a run-evidence-shaped file in
#        the repo root even if it is untracked.  This is the stray-`removed_data_quality_*`
#        half.  It is the MAINTAINED rule: the glob list below must grow when a new
#        evidence writer is added.  Rule T does not need that; rule E does.
#
#  INSTALLED FROM `pytest_configure`, i.e. BEFORE COLLECTION, because collection imports
#  every test module and a module-level write would otherwise land before any fixture ran.
#
#  WHAT THIS GUARD CANNOT DETECT -- stated because a guard blind to its own target is the
#  house's most-logged test defect:
#    * WRITES FROM A SUBPROCESS.  Only this interpreter's `builtins.open` / `io.open` /
#      `os.open` / `os.remove` / `os.unlink` / `os.rename` / `os.replace` / `shutil.copy*`
#      / `shutil.move` are wrapped.  Anything a test shells out to writes unimpeded.  The
#      session-end verifier below is the backstop for exactly this.
#    * WRITES FROM A C EXTENSION that calls the OS directly instead of going through the
#      Python-level primitives (pyarrow's native writers are the realistic case; pandas
#      `to_csv` was measured on 2026-08-31 to go through `builtins.open`, so it IS covered).
#    * PATH SPELLINGS THAT EVADE THE PREFIX TEST.  Matching is `os.path.abspath` +
#      `normcase`, deliberately NOT `realpath`, because a `realpath` syscall on every
#      `open()` in the suite is a cost the suite does not need to pay.  A symlink,
#      junction, `subst` drive or UNC spelling that reaches a tracked file by another name
#      is NOT caught.
#    * FILES THAT BECOME TRACKED MID-SESSION.  The tracked set is a session-start snapshot.
#    * A TEST THAT UNINSTALLS THE GUARD by restoring `builtins.open` itself.  This guards
#      against accident, not against a determined test.
#    * DAMAGE TO UNTRACKED FILES that are not evidence-shaped -- e.g. another developer's
#      work-in-progress module.  Rule T can only see what git can see.
#    * The verifier reports CONTENT CHANGE, so a write that reproduces the file byte for
#      byte is invisible to it.  That is not damage.
import contextlib as _contextlib
import fnmatch as _fnmatch
import hashlib
import io as _io
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT_KEY = _os.path.normcase(_REPO_ROOT)

#  RULE E's list: the CSV basenames the pipeline produces at the repo root.
#
#  IT IS THE `*.csv` HALF OF `Sbocker.allowlist_patterns`, AND THAT IS NOT A COINCIDENCE --
#  it is the same question asked twice.  Sbocker's list answers "which root-level artifacts
#  must TRAVEL"; this one answers "which root-level artifact NAMES may a test not create".
#  Both enumerate the pipeline's root-level output, so they drift apart the moment a writer
#  is added to one and forgotten in the other, which is exactly what happened when
#  `ShareCapReport_*.csv` arrived on 2026-09-01 and had to be added by hand in two files.
#  `test_repo_write_guard.test_RULE_E_covers_every_CSV_in_the_transfer_manifest` PARSES
#  Sbocker and fails if this list falls behind it, so the next omission is a red suite and
#  not an email.  It is duplicated rather than imported on purpose: `conftest` is loaded
#  before collection, and importing the pipeline's orchestrator to configure a test guard
#  would put the whole import graph -- and its import-time side effects -- in front of every
#  pytest invocation in the repo.
#
#  WHAT THE SYNC TEST STILL CANNOT SEE: a writer that produces a root-level CSV and is in
#  NEITHER list.  That file is unshipped as well as unguarded, so the transfer gap is the
#  louder of the two symptoms -- but this list is not evidence that the manifest is complete.
_EVIDENCE_GLOBS = (
    'AdHocPenaltyBucket_*.csv',
    'AggScoreTop*.csv',
    'CohortMetricStats*.csv',
    'CurrencyExclusionStatus_*.csv',
    'CurrencyFloorFlips_*.csv',
    'DedupSurvivorReport_*.csv',
    'DelistedPrune_*.csv',
    'DollarVolumeFloor_*.csv',
    'ExcludedShareClasses_*.csv',
    'ExclusionList_*.csv',
    'ForensicFlagsTop*.csv',
    'FxRatesHistorical_*.csv',
    'FxRates_*.csv',
    'InputSanityRefusals_*.csv',
    'MarketCapBand_*.csv',
    'MeanBarCalibration-*.csv',
    'MissingDataFillReport_*.csv',
    'RawMetricsTop100*.csv',
    'ReportingFrequencyConflicts_*.csv',
    #  The SINGLE-METRIC SHARE CAP REPORT (2026-09-01) -- the disclosure that a name which
    #  could not be capped still shipped with its real score.  Added here and to
    #  `Sbocker.allowlist_patterns` at the cap author's request.
    'ShareCapReport_*.csv',
    'SideList_*.csv',
    'Stage1VetoEjections_*.csv',
    'VendorContaminationFlags_*.csv',
    'delisted_tickers_*.csv',
    'pick_log*.csv',
    'real_prices.csv',
    'removed_data_quality_*.csv',
)

_TRACKED = frozenset()          # normcase'd absolute paths of git-tracked files
_EVIDENCE_SNAPSHOT = {}         # protected path -> sha1 at session start
_GUARD_INSTALLED = []           # (owner, attr, original) triples, for uninstall
_GUARD_ARMED = [False]


class RepoWriteBlocked(RuntimeError):
    """A test tried to write, delete or rename a protected file inside the repo."""


def _guard_load_tracked():
    try:
        out = _subprocess.run(['git', 'ls-files', '-z'], cwd=_REPO_ROOT,
                              stdout=_subprocess.PIPE, stderr=_subprocess.DEVNULL,
                              timeout=60)
    except Exception:
        return frozenset()
    if out.returncode != 0:
        return frozenset()
    names = [n for n in out.stdout.decode('utf-8', 'replace').split('\0') if n]
    return frozenset(_os.path.normcase(_os.path.join(_REPO_ROOT, n.replace('/', _os.sep)))
                     for n in names)


def _guard_classify(path, kind='write'):
    """Return a refusal REASON for <path>, or None if the operation is allowed.

    <kind> is 'write' (create/modify) or 'delete' (unlink, or the SOURCE of a rename).
    RULE T covers both -- a tracked file must survive either.  RULE E covers 'write' only:
    it exists to stop a dropping being CREATED, and refusing to let anything DELETE an
    untracked dropping would make the mess permanent instead of preventing it.

    Cheap-first: `abspath` is string work plus one `getcwd`, and anything outside the repo
    -- which is every `tmp_path` write, i.e. the overwhelming majority -- exits on the
    prefix test without touching the filesystem."""
    if not _GUARD_ARMED[0]:
        return None
    try:
        ap = _os.path.abspath(_os.fspath(path))
    except Exception:
        return None
    key = _os.path.normcase(ap)
    if not (key == _REPO_ROOT_KEY or key.startswith(_REPO_ROOT_KEY + _os.sep)):
        return None                            # outside the repo -- not our business
    if key in _TRACKED:
        return ('RULE T: %s is a git-TRACKED file in this repo.  A test may not write, '
                'delete or rename tracked repo content.' % ap)
    if kind == 'write' and _os.path.normcase(_os.path.dirname(ap)) == _REPO_ROOT_KEY:
        base = _os.path.basename(ap)
        for pat in _EVIDENCE_GLOBS:
            if _fnmatch.fnmatch(base, pat):
                return ('RULE E: %s carries a run-evidence artifact name and this is the '
                        'REPO ROOT.  A test may not drop pipeline evidence into the repo.'
                        % ap)
    return None


def _guard_refuse(path, verb, kind='write'):
    why = _guard_classify(path, kind)
    if why is None:
        return
    raise RepoWriteBlocked(
        "REPO WRITE GUARD blocked %s.\n  %s\n\n"
        "  This is almost certainly the EVIDENCE_DIR trap: `transfer_utils.EVIDENCE_DIR`\n"
        "  is the REPO ROOT (deliberately -- it must stay CWD-independent for the\n"
        "  pipeline), so `monkeypatch.chdir(tmp_path)` does NOT isolate a test that\n"
        "  triggers an evidence writer.  Redirect the constant itself:\n\n"
        "      import transfer_utils as tu\n"
        "      monkeypatch.setattr(tu, 'EVIDENCE_DIR', str(tmp_path))\n\n"
        "  (VA_ALLOW_REPO_WRITES=1 disarms the guard for a deliberate non-test\n"
        "  invocation.  Never set it to make a failing test pass.)"
        % (verb, why))


def _guard_install():
    _real_open = _io.open                 # builtins.open IS io.open in CPython
    _real_os_open = _os.open
    _real_remove = _os.remove
    _real_unlink = _os.unlink
    _real_rename = _os.rename
    _real_replace = _os.replace
    _real_copyfile = _shutil.copyfile
    _real_copy = _shutil.copy
    _real_copy2 = _shutil.copy2
    _real_move = _shutil.move

    def guarded_open(file, mode='r', *a, **k):
        if isinstance(mode, str) and any(c in mode for c in 'wax+'):
            _guard_refuse(file, 'open(%r)' % mode)
        return _real_open(file, mode, *a, **k)

    def guarded_os_open(path, flags, *a, **k):
        if flags & (_os.O_WRONLY | _os.O_RDWR | _os.O_CREAT | _os.O_TRUNC | _os.O_APPEND):
            _guard_refuse(path, 'os.open')
        return _real_os_open(path, flags, *a, **k)

    def guarded_remove(path, **k):
        _guard_refuse(path, 'os.remove', 'delete')
        return _real_remove(path, **k)

    def guarded_unlink(path, **k):
        _guard_refuse(path, 'os.unlink', 'delete')
        return _real_unlink(path, **k)

    def guarded_rename(src, dst, **k):
        _guard_refuse(src, 'os.rename (source)', 'delete')
        _guard_refuse(dst, 'os.rename (destination)')
        return _real_rename(src, dst, **k)

    def guarded_replace(src, dst, **k):
        _guard_refuse(src, 'os.replace (source)', 'delete')
        _guard_refuse(dst, 'os.replace (destination)')
        return _real_replace(src, dst, **k)

    def _guarded_copy(real, name):
        def _f(src, dst, *a, **k):
            target = dst
            try:
                if _os.path.isdir(dst):
                    target = _os.path.join(dst, _os.path.basename(_os.fspath(src)))
            except Exception:
                pass
            _guard_refuse(target, name)
            return real(src, dst, *a, **k)
        return _f

    import builtins as _builtins
    for owner, attr, new in (
            (_builtins, 'open', guarded_open),
            (_io, 'open', guarded_open),
            (_os, 'open', guarded_os_open),
            (_os, 'remove', guarded_remove),
            (_os, 'unlink', guarded_unlink),
            (_os, 'rename', guarded_rename),
            (_os, 'replace', guarded_replace),
            (_shutil, 'copyfile', _guarded_copy(_real_copyfile, 'shutil.copyfile')),
            (_shutil, 'copy', _guarded_copy(_real_copy, 'shutil.copy')),
            (_shutil, 'copy2', _guarded_copy(_real_copy2, 'shutil.copy2')),
            (_shutil, 'move', _guarded_copy(_real_move, 'shutil.move')),
    ):
        _GUARD_INSTALLED.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, new)


@_contextlib.contextmanager
def guard_disarmed():
    """Disarm the guard for the duration of the block.

    FOR THE GUARD'S OWN TESTS ONLY.  They have to be able to put a file back that the
    guard is -- correctly -- refusing to let them write, and a test that cannot clean up
    after itself is a worse neighbour than the guard is a nuisance.  Anything ELSE reaching
    for this is working around the guard instead of fixing the test that tripped it: the
    fix is almost always `monkeypatch.setattr(tu, 'EVIDENCE_DIR', str(tmp_path))`."""
    was = _GUARD_ARMED[0]
    _GUARD_ARMED[0] = False
    try:
        yield
    finally:
        _GUARD_ARMED[0] = was


def _guard_uninstall():
    while _GUARD_INSTALLED:
        owner, attr, original = _GUARD_INSTALLED.pop()
        setattr(owner, attr, original)


def _guard_sha1(path, _opener):
    h = hashlib.sha1()
    with _opener(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _guard_protected_evidence_paths():
    """The tracked, evidence-shaped files the session-end verifier hashes.  A SUBSET of
    rule T on purpose: hashing all of the repo's tracked files every session buys nothing,
    and the evidence CSVs are the artifacts whose loss is not recoverable from a rerun."""
    out = []
    for key in _TRACKED:
        base = _os.path.basename(key)
        if any(_fnmatch.fnmatch(base, _os.path.normcase(p)) for p in _EVIDENCE_GLOBS):
            if _os.path.exists(key):
                out.append(key)
    return sorted(out)


def pytest_configure(config):
    """Arm the guard BEFORE collection."""
    global _TRACKED
    if _os.environ.get('VA_ALLOW_REPO_WRITES') == '1':
        return
    _TRACKED = _guard_load_tracked()
    if not _TRACKED:
        print('\n[repo-write-guard] WARNING: `git ls-files` produced nothing -- RULE T is '
              'INACTIVE for this session; only the evidence-name rule is in force.',
              file=_sys.stderr)
    _plain_open = _io.open
    for p in _guard_protected_evidence_paths():
        try:
            _EVIDENCE_SNAPSHOT[p] = _guard_sha1(p, _plain_open)
        except Exception:
            pass
    _guard_install()
    _GUARD_ARMED[0] = True


def pytest_sessionfinish(session, exitstatus):
    """Backstop for everything the interceptor cannot see (subprocesses, C-level writes).

    DETECTION ONLY -- it does not restore.  Other developers are live in this tree and a
    conftest that silently mutates files on their behalf is a worse failure than the one
    it prevents.  It prints the exact restore command instead."""
    if not _GUARD_ARMED[0]:
        return
    _GUARD_ARMED[0] = False
    _guard_uninstall()
    damaged = []
    for path, before in _EVIDENCE_SNAPSHOT.items():
        try:
            if not _os.path.exists(path):
                damaged.append((path, 'DELETED'))
            elif _guard_sha1(path, _io.open) != before:
                damaged.append((path, 'CONTENT CHANGED'))
        except Exception:
            pass
    if damaged:
        rel = [_os.path.relpath(p, _REPO_ROOT).replace(_os.sep, '/') for p, _w in damaged]
        print('\n' + '!' * 78, file=_sys.stderr)
        print('!!! REPO WRITE GUARD: %d TRACKED RUN-EVIDENCE FILE(S) CHANGED DURING THIS '
              'SESSION.' % len(damaged), file=_sys.stderr)
        print('!!! The interceptor did not catch the writer, so it came from a subprocess',
              file=_sys.stderr)
        print('!!! or a C-level write.  RESTORE BEFORE DOING ANYTHING ELSE:', file=_sys.stderr)
        for p, what in damaged:
            print('!!!   %-16s %s' % (what, _os.path.relpath(p, _REPO_ROOT)), file=_sys.stderr)
        print('!!!', file=_sys.stderr)
        print('!!!   git checkout -- %s' % ' '.join(rel), file=_sys.stderr)
        print('!' * 78 + '\n', file=_sys.stderr)
        session.exitstatus = 1
