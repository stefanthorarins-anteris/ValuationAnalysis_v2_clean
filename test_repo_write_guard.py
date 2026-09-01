"""The repo write guard's own tests -- see the long note in `conftest.py`.

WHOLLY OFFLINE.  Nothing here touches the network, and nothing here can damage the repo
even if the guard it tests is completely broken.  That property is not incidental: a test
for a "you may not overwrite this file" guard is, by its nature, one bug away from being
the thing that overwrites the file.  Three devices keep it safe:

  * the REAL-REPO tests aim at writers that this repo documents as NEVER-RAISING and
    assert on the EFFECT (the tracked bytes are unchanged), and they snapshot and restore
    those bytes in a `finally` regardless of the outcome;
  * the destructive primitives (truncate, delete, rename, `to_csv`) are exercised against
    a SIMULATED repo root inside `tmp_path`, by pointing the guard's own `_REPO_ROOT_KEY`
    and `_TRACKED` at it.  Same code path, no real file at risk;
  * the one real-repo `open` is in APPEND mode and is never written to, so it cannot
    change a byte even if the guard lets it through.

WHAT THESE TESTS CANNOT TELL YOU.  They prove the guard refuses the writes it is given.
They say nothing about the writes it never sees -- subprocesses, C-level writers, and the
other blind spots enumerated in `conftest.py`.  `test_the_session_end_verifier_...` below
covers the detection backstop's LOGIC but cannot manufacture a real C-level write, so the
backstop's real-world coverage is argued, not measured.
"""
import io
import os
import shutil

import pandas as pd
import pytest

import conftest as _guard
import currency_exclusions as cx
import data_quality as dq

REPO = _guard._REPO_ROOT


def _a_tracked_evidence_csv():
    """A real, tracked, evidence-shaped CSV that exists right now."""
    paths = _guard._guard_protected_evidence_paths()
    if not paths:
        pytest.skip('no tracked evidence CSV in this checkout')
    return paths[0]


def _simulate_repo_root_at(monkeypatch, root, tracked=()):
    """Point the guard's repo-root and tracked-set at <root> so the destructive primitives
    can be exercised for real without a real file at risk.  This is the guard's OWN
    classification data, not a stub of the guard -- `guarded_open` and friends are the
    installed ones and run unmodified."""
    key = os.path.normcase(str(root))
    monkeypatch.setattr(_guard, '_REPO_ROOT_KEY', key)
    monkeypatch.setattr(_guard, '_TRACKED',
                        frozenset(os.path.normcase(str(p)) for p in tracked))


# --------------------------------------------------------------------------- #
#  0.  the guard is actually installed                                        #
# --------------------------------------------------------------------------- #
def test_the_guard_is_INSTALLED_and_ARMED():
    """The meta-test.  Every other test here is vacuous if the interceptor never went in,
    and `pytest_configure` failing silently is a realistic way for that to happen."""
    import builtins
    assert _guard._GUARD_ARMED[0], 'the guard is not armed -- is VA_ALLOW_REPO_WRITES set?'
    assert builtins.open.__name__ == 'guarded_open', (
        'builtins.open is not the guarded wrapper: %r' % builtins.open)
    assert io.open.__name__ == 'guarded_open'
    assert os.remove.__name__ == 'guarded_remove'
    assert os.replace.__name__ == 'guarded_replace'
    assert _guard._TRACKED, 'the tracked-file set is empty -- RULE T is inactive'


# --------------------------------------------------------------------------- #
#  1.  RULE T -- tracked files                                                #
# --------------------------------------------------------------------------- #
def test_RULE_T_refuses_an_append_open_of_a_REAL_tracked_evidence_csv():
    """Against the real repo, in APPEND mode and writing nothing, so a broken guard costs
    a modified mtime and not a byte of content."""
    target = _a_tracked_evidence_csv()
    before = _guard._guard_sha1(target, io.open)
    with pytest.raises(_guard.RepoWriteBlocked) as ei:
        fh = open(target, 'a')
        fh.close()
    assert 'RULE T' in str(ei.value)
    assert 'EVIDENCE_DIR' in str(ei.value), 'the message must name the trap it exists for'
    assert _guard._guard_sha1(target, io.open) == before


def test_RULE_T_refuses_TRUNCATE_DELETE_RENAME_and_COPY(tmp_path, monkeypatch):
    """The destructive primitives, against a simulated repo root.  `open('w')` is the one
    that actually happened on 2026-08-31; the others are the obvious neighbours."""
    victim = tmp_path / 'DollarVolumeFloor_2026-08-31.csv'
    victim.write_text('real,evidence\n1,2\n')                # written BEFORE the guard aims here
    _simulate_repo_root_at(monkeypatch, tmp_path, tracked=[victim])
    donor = tmp_path.parent / 'donor.csv'
    donor.write_text('x\n')

    for verb, call in (
            ("open('w')", lambda: open(victim, 'w')),
            ("open('a')", lambda: open(victim, 'a')),
            ("open('r+')", lambda: open(victim, 'r+')),
            ('os.remove', lambda: os.remove(victim)),
            ('os.unlink', lambda: os.unlink(victim)),
            ('os.replace', lambda: os.replace(donor, victim)),
            ('os.rename', lambda: os.rename(victim, tmp_path / 'moved.csv')),
            ('shutil.copyfile', lambda: shutil.copyfile(donor, victim)),
            ('shutil.copy2', lambda: shutil.copy2(donor, victim)),
            ('pandas.to_csv', lambda: pd.DataFrame({'a': [1]}).to_csv(victim, index=False)),
    ):
        with pytest.raises(_guard.RepoWriteBlocked):
            call()
        assert victim.read_text() == 'real,evidence\n1,2\n', (
            '%s got through the guard and changed the file' % verb)


def test_RULE_T_refuses_a_COPY_INTO_a_directory_by_basename(tmp_path, monkeypatch):
    """`shutil.copy(src, a_directory)` names the destination only implicitly.  A guard that
    checked `dst` verbatim would wave it through, which is the whole reason the wrapper
    re-derives the target basename."""
    victim = tmp_path / 'DelistedPrune_2026-08-31.csv'
    victim.write_text('kept\n')
    _simulate_repo_root_at(monkeypatch, tmp_path, tracked=[victim])
    donor_dir = tmp_path.parent / 'donor_dir'
    donor_dir.mkdir(exist_ok=True)
    donor = donor_dir / 'DelistedPrune_2026-08-31.csv'
    donor.write_text('junk\n')
    with pytest.raises(_guard.RepoWriteBlocked):
        shutil.copy(donor, tmp_path)
    assert victim.read_text() == 'kept\n'


# --------------------------------------------------------------------------- #
#  2.  RULE E -- untracked droppings that WEAR an evidence name               #
# --------------------------------------------------------------------------- #
def test_RULE_E_refuses_a_NEW_evidence_shaped_file_in_the_repo_root(tmp_path, monkeypatch):
    """The six stray `removed_data_quality_*.csv` of 2026-08-31.  Nothing was overwritten,
    so RULE T would not have fired; the file simply did not belong in the repo."""
    _simulate_repo_root_at(monkeypatch, tmp_path, tracked=[])
    dropping = tmp_path / 'removed_data_quality_20260831_235959.csv'
    with pytest.raises(_guard.RepoWriteBlocked) as ei:
        open(dropping, 'w')
    assert 'RULE E' in str(ei.value)
    assert not dropping.exists()


def test_RULE_E_is_SCOPED_TO_THE_ROOT_and_does_not_police_subdirectories(
        tmp_path, monkeypatch):
    """A deliberate limit, stated so it is not mistaken for a hole: the evidence writers
    write to the ROOT, and extending the name rule to every subdirectory would start
    refusing a test's own `tmp_path/output/FxRatesHistorical_*.csv` scaffolding -- which
    `test_fx_rates` legitimately builds."""
    _simulate_repo_root_at(monkeypatch, tmp_path, tracked=[])
    sub = tmp_path / 'output'
    sub.mkdir()
    inner = sub / 'FxRatesHistorical_2019-01-01_2026-08-08.csv'
    inner.write_text('currency,date,rate\n')                 # must NOT raise
    assert inner.exists()


def test_RULE_E_covers_every_CSV_in_the_transfer_manifest():
    """RULE E is the guard's ONE hand-maintained list, so it is the guard's one place to
    rot.  It rotted immediately: `ShareCapReport_*.csv` was added to the pipeline on
    2026-09-01 and had to be pasted into two files by hand, with nothing to catch the
    omission except the author remembering to mention it.

    `Sbocker.allowlist_patterns` asks the same question for a different reason -- which
    root-level artifacts must TRAVEL -- and it is maintained, because a missing pattern
    there means the CEO never receives the file.  Pinning this list to that one means the
    next writer is caught by a red suite instead of by an email.

    PARSED, not imported: importing `Sbocker` to check a test guard would drag the whole
    pipeline import graph in front of every pytest invocation in this repo.

    WHAT IT CANNOT DETECT: a root-level writer missing from BOTH lists.  This proves the
    two agree, not that either is complete."""
    import ast
    with io.open(os.path.join(REPO, 'Sbocker.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    manifest = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'allowlist_patterns'
                for t in node.targets):
            manifest = [e.value for e in node.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    assert manifest, 'could not find Sbocker.allowlist_patterns -- has it been renamed?'
    csv_patterns = {p for p in manifest if p.endswith('.csv')}
    assert csv_patterns, 'the manifest carries no CSV patterns -- the parse is wrong'
    missing = sorted(csv_patterns - set(_guard._EVIDENCE_GLOBS))
    assert not missing, (
        'Sbocker ships these root-level CSVs and RULE E does not guard their names, so a '
        'test that forgets to redirect EVIDENCE_DIR can drop one in the repo root: %s\n'
        'Add them to `_EVIDENCE_GLOBS` in conftest.py.' % missing)


# --------------------------------------------------------------------------- #
#  3.  the guard must not become the new problem                              #
# --------------------------------------------------------------------------- #
def test_the_guard_ALLOWS_ordinary_writes_outside_the_repo(tmp_path):
    """The false-positive check.  Every isolated test in the suite writes into `tmp_path`;
    if the guard touched those it would be worse than the bug."""
    p = tmp_path / 'DollarVolumeFloor_2026-08-31.csv'        # evidence NAME, outside the repo
    pd.DataFrame({'a': [1, 2]}).to_csv(p, index=False)
    assert len(pd.read_csv(p)) == 2
    os.remove(p)


def test_the_guard_ALLOWS_READING_a_tracked_file():
    target = _a_tracked_evidence_csv()
    with open(target, 'rb') as fh:
        assert fh.read(1) != b''


def test_the_guard_ALLOWS_an_untracked_non_evidence_write_inside_the_repo():
    """`__pycache__`, `.pytest_cache` and the like.  Blocking these would break collection
    itself, so it is worth pinning that they are allowed."""
    probe = os.path.join(REPO, '__pycache__', '_repo_write_guard_probe.tmp')
    os.makedirs(os.path.dirname(probe), exist_ok=True)
    try:
        with open(probe, 'w') as fh:
            fh.write('ok')
        assert os.path.exists(probe)
    finally:
        if os.path.exists(probe):
            os.remove(probe)


# --------------------------------------------------------------------------- #
#  4.  END TO END, THROUGH THE REAL WRITERS, ON THE REAL REPO                 #
# --------------------------------------------------------------------------- #
#  These are the ones that matter.  Each reproduces the EXACT mistake of 2026-08-31 -- a
#  test that isolates itself with `monkeypatch.chdir(tmp_path)` and nothing else -- and
#  asserts the repo survives it.
def test_the_REAL_currency_exclusion_writer_cannot_clobber_the_TRACKED_status_csv(
        tmp_path, monkeypatch, capsys):
    """`cx.write_status` is documented "Never raises", so the guard's refusal arrives as a
    swallowed warning and a `None` return -- which is the RIGHT shape: a test that forgot
    to isolate itself now writes nothing instead of destroying the run's evidence.

    THE RUN DATE IS PASSED EXPLICITLY, not taken from the clock.  Aiming this at "today"
    made it a RULE T test only on days when a run had already produced today's file; the
    moment the date rolled it quietly demoted itself to a RULE E test and stopped covering
    the clobber it is named for.  It now aims at a file that IS tracked, whatever day it is."""
    tracked = [p for p in _guard._guard_protected_evidence_paths()
               if os.path.basename(p).startswith('currencyexclusionstatus_')]
    if not tracked:
        pytest.skip('no tracked CurrencyExclusionStatus_*.csv in this checkout')
    target = sorted(tracked)[-1]
    run_date = os.path.basename(target)[len('currencyexclusionstatus_'):-len('.csv')]
    before = io.open(target, 'rb').read()
    try:
        monkeypatch.chdir(tmp_path)          # the isolation that USED to be sufficient
        got = cx.write_status([{'status': 'EXCLUDED', 'currency': 'ARS', 'source': 'BMA',
                                'note': 'guard test'}], run_date=run_date)
        assert got is None, 'the writer reported success -- it wrote somewhere'
        out = capsys.readouterr().out
        assert 'RepoWriteBlocked' in out, (
            'the refusal must be VISIBLE, not merely effective: %r' % out[-400:])
        assert io.open(target, 'rb').read() == before, 'the tracked status CSV changed'
    finally:
        #  The guard is (correctly) refusing to let this test write that path, so the
        #  restore has to step outside it.  This is the ONLY sanctioned use of the hatch.
        with _guard.guard_disarmed():
            if io.open(target, 'rb').read() != before:
                with io.open(target, 'wb') as fh:
                    fh.write(before)


def test_the_REAL_removed_data_writer_cannot_DROP_a_stray_csv_in_the_repo_root(
        tmp_path, monkeypatch):
    """`dq.save_removed_data` does NOT swallow, so here the refusal is the exception
    itself.  This is the six-droppings half of 2026-08-31."""
    def _strays():
        return {n for n in os.listdir(REPO)
                if n.startswith('removed_data_quality_') and n.endswith('.csv')}

    before = _strays()
    try:
        monkeypatch.chdir(tmp_path)          # the isolation that USED to be sufficient
        frame = pd.DataFrame({'source': ['BMA'], 'reason': ['guard test']})
        with pytest.raises(_guard.RepoWriteBlocked) as ei:
            dq.save_removed_data(frame, run_id='guard-test')
        assert 'RULE E' in str(ei.value)
        assert _strays() == before, 'the repo root gained a dropping'
    finally:
        #  Belt and braces: if the guard were broken this test would itself be a dropping
        #  factory, so it cleans up whatever it made regardless of the assertions.
        with _guard.guard_disarmed():
            for name in _strays() - before:
                os.remove(os.path.join(REPO, name))


# --------------------------------------------------------------------------- #
#  5.  the session-end verifier's logic                                       #
# --------------------------------------------------------------------------- #
def test_the_session_end_verifier_DETECTS_a_content_change_and_a_DELETION(tmp_path):
    """The backstop for the writes the interceptor cannot see.  Only its comparison logic
    is testable here: a real subprocess or C-level write cannot be manufactured inside a
    test that is itself forbidden to damage the repo, so the backstop's coverage of those
    cases is ARGUED, NOT MEASURED."""
    kept = tmp_path / 'kept.csv'
    kept.write_text('a\n')
    changed = tmp_path / 'changed.csv'
    changed.write_text('a\n')
    gone = tmp_path / 'gone.csv'
    gone.write_text('a\n')
    snap = {str(p): _guard._guard_sha1(str(p), io.open) for p in (kept, changed, gone)}

    changed.write_text('b\n')
    os.remove(gone)

    damaged = []
    for path, sha in snap.items():
        if not os.path.exists(path):
            damaged.append((path, 'DELETED'))
        elif _guard._guard_sha1(path, io.open) != sha:
            damaged.append((path, 'CONTENT CHANGED'))
    assert sorted(damaged) == sorted([(str(changed), 'CONTENT CHANGED'),
                                      (str(gone), 'DELETED')])
