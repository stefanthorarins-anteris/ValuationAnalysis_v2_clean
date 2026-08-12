"""
THE DIRECTORY HALF OF THE DRIVE TRANSFER  (2026-08-11)

WHAT THIS FILE IS FOR.  `Sbocker.transfer_outputs_to_drive` copies two kinds of thing:
root-level FILES (glob'd from the allowlist patterns) and named DIRECTORIES
(`logs/`, `output/`, `baseline_tools/price_data`).  For at least four consecutive
production runs EVERY file arrived at the Drive and NO directory did, and the run's
own tail still said the transfer succeeded.  The CEO's console from the 2026-08-11
CUR3K run:

    ERROR copying directory logs: [WinError 5] Access is denied: "E:\\drive\\valuationTransfer\\logs"

-- the path named in the error is the DESTINATION, so the call that failed was the
`shutil.rmtree(dest_dir)` the loop did before copying, not the copy.  Google Drive's
virtual filesystem refuses the directory delete.  Consequence: no `output/`, no
`logs/`, no `run_logs/` at the destination on any of those runs, which is why the FX
evidence, the dedup detail and every run log were unavailable for offline analysis,
and why six evidence CSVs were moved to the repo root on 2026-08-10 as a workaround.

The tests below pin the three defects that produced that, plus the safety property
that must survive fixing them.  Every one of the first four FAILED against the
implementation as it stood on 2026-08-11; they are written against BEHAVIOUR at the
destination, never against the copy loop's own bookkeeping, because "the copy loop
thinks it copied" is precisely the claim that was false.

NO NETWORK, NO KEY FILE, NO REAL DRIVE: every test runs in a tmp_path and chdirs
into it, because the function resolves its allowlist with `glob.glob()` from CWD.
"""
import os
import shutil

import pytest

import Sbocker
import transfer_utils as tu


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write(path, text="x"):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        fh.write(text)
    return str(path)


def _run_dir(tmp_path, monkeypatch):
    """A synthetic run CWD + a Drive-like transfer target whose PARENT exists but
    whose leaf does not (mirrors a mounted Drive folder; the function mkdirs the
    leaf with parents=False)."""
    run = tmp_path / "run"
    run.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    transfer_dir = str(drive / "valuationTransfer")
    monkeypatch.chdir(run)
    return run, transfer_dir


def _files_under(root):
    """Every file at or under <root>, as destination-relative posix paths."""
    out = set()
    for dirpath, _dirs, files in os.walk(str(root)):
        for fn in files:
            rel = os.path.relpath(os.path.join(dirpath, fn), str(root))
            out.add(rel.replace(os.sep, "/"))
    return out


def _populate(run):
    """The two directories that matter on a real run: the console tee and the
    per-decision evidence CSVs.  `run_logs/` is deliberately absent -- it only
    exists when -ingest_delisted ran."""
    _write(run / "logs" / "run_2026-08-11.log", "banner\n")
    _write(run / "logs" / "nested" / "phase2.log", "more\n")
    _write(run / "output" / "DedupSurvivorReport_2026-08-11.csv", "a,b\n")
    _write(run / "output" / "FxRates_2026-08-11.csv", "ccy,rate\n")


# --------------------------------------------------------------------------- #
# 1. THE FAILING OPERATION IS THE DELETE, NOT THE COPY
# --------------------------------------------------------------------------- #
def test_a_destination_that_refuses_rmtree_still_receives_the_directory(tmp_path, monkeypatch):
    """The production failure, reproduced.  A destination directory already exists
    (it does, on every run after the first) and the filesystem refuses to delete it.

    Before the fix the loop called `shutil.rmtree(dest)` first, so WinError 5 landed
    in the `except` and the whole directory was skipped -- every file in it lost for
    that run.  After the fix there is no delete to refuse: `copytree(...,
    dirs_exist_ok=True)` merges into the directory that is already there.

    `rmtree` is patched to raise the EXACT error the CEO's console showed, so this
    test fails if anybody reintroduces a delete-then-copy under any name."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)

    # The destination already exists and holds last run's copy -- the state every
    # run after the first one meets.
    _write(os.path.join(transfer_dir, "logs", "run_2026-08-10.log"), "yesterday\n")

    def _refuse(*a, **k):
        raise PermissionError(13, "Access is denied", os.path.join(transfer_dir, "logs"))

    monkeypatch.setattr(shutil, "rmtree", _refuse)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    landed = _files_under(transfer_dir)
    assert "logs/run_2026-08-11.log" in landed, (
        "the run's console tee did NOT reach the Drive -- this is the production "
        "failure of 2026-08-08..08-11 (%r)" % sorted(landed))
    assert "logs/nested/phase2.log" in landed, "the copy must still be recursive"
    assert "output/DedupSurvivorReport_2026-08-11.csv" in landed
    assert "output/FxRates_2026-08-11.csv" in landed
    assert result["status"] == "success", result


def test_the_outbox_keeps_what_the_local_run_no_longer_has(tmp_path, monkeypatch):
    """THE DELIBERATE CONSEQUENCE OF DROPPING THE DELETE, pinned so it reads as a
    decision and not as an oversight.  The transfer dir is an OUTBOX, not a mirror:
    a file that exists at the destination and no longer exists locally STAYS.  A
    stale artifact at the destination is recoverable; a deleted one is not, and the
    delete is also the operation Drive refuses."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    kept = os.path.join(transfer_dir, "output", "DedupSurvivorReport_2026-08-08.csv")
    _write(kept, "an older run's evidence, no longer on this machine\n")

    Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    assert os.path.exists(kept), (
        "the outbox must not lose history the local machine has already discarded")
    assert "output/FxRates_2026-08-11.csv" in _files_under(transfer_dir)


# --------------------------------------------------------------------------- #
# 2. DENY THE FILE, NOT THE TREE
# --------------------------------------------------------------------------- #
def test_a_denied_file_is_skipped_while_its_siblings_still_ship(tmp_path, monkeypatch):
    """One denied file used to drop the ENTIRE directory: the loop walked the tree,
    set `has_denied` and `continue`d past the whole thing, silently.  FMP's own
    endpoint family is called `key-metrics`, so a cache file named after it is a live
    hazard, not a hypothetical -- it would have silently cost the whole of `output/`.

    The safety property is unchanged and is asserted here too: the denied file must
    NOT be at the destination."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    _write(run / "output" / "key-metrics_cache.json", "{}")          # *key*
    _write(run / "logs" / "nested" / "server.pem", "SECRET")          # *pem
    _write(run / "logs" / "fmpAPIkey.txt", "SECRET")                  # exact

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    landed = _files_under(transfer_dir)
    assert "output/DedupSurvivorReport_2026-08-11.csv" in landed, (
        "a denied SIBLING must not drop the directory (%r)" % sorted(landed))
    assert "logs/run_2026-08-11.log" in landed
    assert "logs/nested/phase2.log" in landed

    # ...and the safety guarantee, restated at the destination:
    assert "output/key-metrics_cache.json" not in landed
    assert "logs/nested/server.pem" not in landed
    assert "logs/fmpAPIkey.txt" not in landed
    for rel in landed:
        assert not tu.is_denied(rel), "DENIED FILE REACHED THE DRIVE: %s" % rel
    assert result["status"] == "success", result


def test_no_denied_file_reaches_the_destination_at_any_depth(tmp_path, monkeypatch):
    """The load-bearing safety property of this whole module, checked per-file at
    every depth rather than per-directory.  `transfer_utils.is_denied` stays the
    single source of truth -- the assertion below re-uses it rather than restating
    the patterns."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    for rel in ("output/fmpAPIkey.txt",
                "output/a/backup_key_2026.txt",
                "output/a/b/private.pem",
                "logs/nested/API_KEY_dump.txt",
                "baseline_tools/price_data/keyring.json"):
        _write(run / rel, "SECRET")
    _write(run / "baseline_tools" / "price_data" / "prices_2024.csv", "d,p\n")

    Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    landed = _files_under(transfer_dir)
    assert landed, "nothing copied at all -- the test proves nothing"
    for rel in landed:
        assert not tu.is_denied(rel), "DENIED FILE REACHED THE DRIVE: %s" % rel
    assert "SECRET" not in "".join(landed)
    # the non-denied siblings at those same depths DID ship:
    assert "baseline_tools/price_data/prices_2024.csv" in landed
    assert "logs/nested/phase2.log" in landed


def test_the_denied_file_is_never_WRITTEN_not_merely_deleted_afterwards(tmp_path, monkeypatch):
    """THE DISTINCTION THAT ABSENCE-AT-THE-END CANNOT MAKE, and it is the one that
    matters for a secret.  The destination is a GOOGLE-DRIVE-SYNCED folder: a key
    file that is written there and deleted a second later may already have been
    uploaded.  So "not present when the run finished" is NOT the guarantee -- "never
    written" is.

    Found by mutation testing (2026-08-11): with the copy's denylist disabled
    entirely, every other test in this file still passed, because the destination
    check deleted the evidence afterwards.  Defence in depth is right, but it must not
    be able to MASK the first line of defence.  The destination check no longer
    deletes anything at all, which makes this test the ONLY thing standing between a
    disabled copy denylist and a key file on the Drive.

    THE ROOT-LEVEL PLANT (`SideList_monkey_...`) closes the review's R2-5: the
    root-file loop's `if is_denied(fpath): continue` had no test of its own, and its
    allowlist patterns are wildcards that CAN collide (`SideList_*.csv`,
    `pick_log*.csv`)."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    _write(run / "output" / "fmpAPIkey.txt", "SECRET")                      # top level of the dir
    _write(run / "output" / "a" / "b" / "deep_key_backup.txt", "SECRET")    # two levels down
    _write(run / "logs" / "nested" / "server.pem", "SECRET")                # one level down
    # ROOT LEVEL, and it MATCHES an allowlist glob (`SideList_*.csv`) -- the file loop
    # must deny it on the way past, or it is written to a synced folder:
    _write(run / "SideList_monkey_Top100-2026-08-11.csv", "SECRET")

    found_at_dest = []
    _real_find = tu.find_denied_files

    def _spy(dest_root, verbose=True):
        hits = _real_find(dest_root, verbose=verbose)
        found_at_dest.extend(hits)
        return hits

    monkeypatch.setattr(tu, "find_denied_files", _spy)

    Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    assert found_at_dest == [], (
        "a denied file was WRITTEN to the Drive-synced destination -- and nothing "
        "removes it, so it is simply there: %r" % found_at_dest)
    landed = _files_under(transfer_dir)
    assert "SideList_monkey_Top100-2026-08-11.csv" not in landed
    assert "output/DedupSurvivorReport_2026-08-11.csv" in landed, "siblings must still ship"
    assert "logs/nested/phase2.log" in landed


def test_a_directory_named_like_a_secret_still_ships_its_non_denied_contents(tmp_path, monkeypatch):
    """THE CONTRACT `_ignore_denied` IS WRITTEN TO, pinned.  `is_denied` matches a
    BASENAME and its documented contract is that a file is denied for its OWN name,
    never because a parent directory happens to contain 'key'/'pem' -- so the ignore
    callable skips directories.  Dropping that `isdir` guard passed all eleven of the
    earlier tests (reviewer mutation M12) while silently reinstating defect (2) for
    any directory whose name matches: `output/api_keys/` would take its whole subtree
    with it, which is exactly the whole-tree denial this change removed."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    _write(run / "output" / "api_keys" / "report.csv", "a,b\n")
    _write(run / "output" / "api_keys" / "fmpAPIkey.txt", "SECRET")   # denied by ITS OWN name
    _write(run / "logs" / "pem_certs" / "expiry.log", "ok\n")

    Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    landed = _files_under(transfer_dir)
    assert "output/api_keys/report.csv" in landed, (
        "a directory was denied for its OWN name, taking its innocent contents with it "
        "-- that is the whole-tree denial this change removed (%r)" % sorted(landed))
    assert "logs/pem_certs/expiry.log" in landed
    assert "output/api_keys/fmpAPIkey.txt" not in landed, "the FILE is still denied"
    for rel in landed:
        assert not tu.is_denied(rel), rel


def test_a_denied_file_at_the_destination_is_REPORTED_and_NOT_DELETED(tmp_path, monkeypatch):
    """THE DESIGN THE CEO SET (2026-08-11): "Please don't go deleting my API key."

    This function used to DELETE what it found here, as compensation for dropping the
    `rmtree`.  Both legs of that argument fell over.  (a) The thing it compensated for
    never happens: the copy direction never writes a denied file, so anything found at
    the destination was put there BY A HUMAN.  (b) `*key*` is a substring pattern, so
    the measured deletions on realistic filenames included `Turkey_exposure_2026.csv`,
    `Keystone_Corp_thesis.docx` and `hockey_stick_screen.csv` -- with NO Recycle Bin --
    and the destination holds the CEO's investment research.

    So it detects and reports.  Deleting would not be containment anyway: on a synced
    folder the upload may already have happened, and unlinking removes the evidence
    that the file was ever there rather than the file's exposure.

    THE FOUR EXTRA PLANTS also pin the round-1 S2 scope finding: while the check ran
    per copied directory, all four of these went unnoticed on a `status='success'`
    run.  Checking the transfer ROOT once catches them."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    planted = _write(os.path.join(transfer_dir, "output", "old_apikey_backup.txt"), "SECRET")
    planted_deep = _write(os.path.join(transfer_dir, "logs", "nested", "leaked.pem"), "SECRET")
    # --- S2: the four the per-directory version could never reach ---
    #  run_logs/ is only allowlisted when -ingest_delisted ran, so it is not copied here:
    outside_dir = _write(os.path.join(transfer_dir, "run_logs", "api_key_dump.txt"), "SECRET")
    #  the copy creates `baseline_tools/`; the per-dir version only ever saw `price_data`:
    parent_dir = _write(os.path.join(transfer_dir, "baseline_tools", "leaked_key.txt"), "SECRET")
    #  and two top-level names `assert_no_key_file` does not know (it checks the exact
    #  filename `fmpAPIkey.txt` only):
    top_wild = _write(os.path.join(transfer_dir, "backup_key_2026.txt"), "SECRET")
    top_pem = _write(os.path.join(transfer_dir, "server.pem"), "SECRET")
    #  and a REAL one, which the run must also refuse to touch:
    top_key = _write(os.path.join(transfer_dir, "fmpAPIkey.txt"), "SECRET")

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    # (a) NOTHING IS DELETED -- every single one is still there:
    for label, path in (("inside a copied directory", planted),
                        ("two levels down", planted_deep),
                        ("a directory not copied on this run", outside_dir),
                        ("the parent directory the copy itself creates", parent_dir),
                        ("a top-level *key* name", top_wild),
                        ("a top-level *pem name", top_pem),
                        ("the real key file", top_key)):
        assert os.path.exists(path), f"the run DELETED a file at the destination ({label}): {path}"

    # (b) ...and every single one is REPORTED, by path, with the run failed:
    assert result["status"] == "error", result
    for path in (planted, planted_deep, outside_dir, parent_dir, top_wild, top_pem, top_key):
        assert path in result["denied_at_destination"], (
            "unreported denylisted file at the destination: %s" % path)
    loud, text = _emitted(result)
    assert loud is True
    assert "NOT removed" in text
    assert "old_apikey_backup.txt" in text and "api_key_dump.txt" in text


def test_the_destination_check_runs_even_when_the_directory_copy_FAILED(tmp_path, monkeypatch):
    """ROUND-1 S1, THE WORST FINDING OF THAT REVIEW, still pinned now that the check
    only reports.  It used to sit inside the copy loop's `try`, one line after
    `copytree`, so any copy failure jumped past it -- and a refused overwrite is the
    plausible Drive residual, meaning it was skipped in exactly the scenario it was
    added for.  If Drive really refuses overwrites, `output/` raises every run and the
    check would never have run, ever."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    planted = _write(os.path.join(transfer_dir, "output", "old_apikey_backup.txt"), "SECRET")

    _real_copytree = shutil.copytree

    def _fail_only_output(src, dst, *a, **k):
        if os.path.basename(str(src).rstrip("/\\")) == "output":
            raise PermissionError(13, "Access is denied", str(dst))
        return _real_copytree(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "copytree", _fail_only_output)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    # recon-blind assertions first: the reconciliation can also raise the status, so
    # `!= success` alone would not prove the copy-failure gate fired (round-2 review).
    assert any("output" in f for f in result["dir_failures"])
    assert result["denied_at_destination"] == [planted], (
        "the destination check was skipped because the copy failed -- which is the one "
        "case it exists for: %r" % result["denied_at_destination"])
    assert os.path.exists(planted), "and it must still not delete anything"
    assert result["status"] == "error", result


def test_a_secret_a_failed_copy_and_a_short_group_together_yield_ERROR(tmp_path, monkeypatch):
    """THE PRECEDENCE CHAIN, WHICH NO SINGLE TEST DROVE (round-2 review R2-4: three
    mutations survived -- inverting the precedence, removing the `== 'success'` guard
    so the reconciliation can lower an error to a warning, and dropping the key-net
    leg -- purely because no test had two conditions at once).  All three conditions
    fire here, and a denylisted file on the Drive must outrank both."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    planted = _write(os.path.join(transfer_dir, "fmpAPIkey.txt"), "SECRET")

    _real_copytree = shutil.copytree

    def _fail_only_output(src, dst, *a, **k):
        if os.path.basename(str(src).rstrip("/\\")) == "output":
            raise PermissionError(13, "Access is denied", str(dst))
        return _real_copytree(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "copytree", _fail_only_output)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    assert result["dir_failures"], "the copy failure must actually have happened"
    assert result["reconciliation"]["incomplete"], "the short group must actually exist"
    assert result["denied_at_destination"] == [planted]
    assert result["status"] == "error", (
        "a denylisted file on the Drive must outrank a copy failure and a short group, "
        "and nothing may lower it back to a warning: %r" % result["status"])
    loud, text = _emitted(result)
    assert loud is True
    assert "fmpAPIkey.txt" in text and "output" in text


def test_the_key_file_net_ALONE_fails_the_run_when_the_recursive_check_is_blind(
        tmp_path, monkeypatch):
    """THE SECOND DETECTOR, TESTED ON ITS OWN.  `find_denied_files` and
    `assert_no_key_file` both catch a top-level `fmpAPIkey.txt` -- deliberately, two
    independent detectors -- and that redundancy means neither leg of
    `if denied_at_destination or not key_net_clean` can be mutated away visibly while
    both are working.  `find_denied_files` CAN legitimately come back empty (its
    `os.walk` swallows errors by design, and `Path.exists()` swallows `OSError`), which
    is exactly the case where the cheap top-level check is the only thing left."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    planted = _write(os.path.join(transfer_dir, "fmpAPIkey.txt"), "SECRET")

    monkeypatch.setattr(tu, "find_denied_files", lambda root, verbose=True: [])

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    assert result["denied_at_destination"] == [], "the recursive check is blind here"
    assert result["status"] == "error", (
        "with the recursive check blind, the top-level key-file net is the only "
        "detector left and it must still fail the run: %r" % result["status"])
    assert os.path.exists(planted), "and it still must not be deleted"
    loud, text = _emitted(result)
    assert loud is True


# --------------------------------------------------------------------------- #
# 3. A DIRECTORY FAILURE IS NOT A SUCCESS
# --------------------------------------------------------------------------- #
def test_a_directory_that_raised_is_not_reported_as_success(tmp_path, monkeypatch):
    """The reporting defect that let the first one run for four runs: the `except`
    printed three quiet ERROR lines into a 12-hour log and eleven lines later
    `result['status'] = 'success'` ran unconditionally, so the caller's loud
    "DRIVE TRANSFER DID NOT COMPLETE" banner never fired."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)

    def _boom(src, dst, **k):
        raise PermissionError(13, "Access is denied", str(dst))

    monkeypatch.setattr(shutil, "copytree", _boom)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    # RECON-BLIND ASSERTION FIRST.  The round-2 review measured that the two
    # assertions this test used to lead with (`status != success`, `"logs" in message`)
    # are BOTH masked: the reconciliation independently raises the status and its own
    # `-- INCOMPLETE AT DESTINATION: logs/` supplies the literal.  They read as
    # coverage they do not provide, so the load-bearing check is `dir_failures`.
    assert result.get("dir_failures"), (
        "the copy failure was not recorded at all: %r" % result)
    assert any("logs" in f for f in result["dir_failures"]), result["dir_failures"]
    assert result["status"] != "success", (
        "every directory failed to copy and the run called it a success: %r" % result)


def test_a_directory_that_raised_fails_the_status_even_when_the_DESTINATION_looks_fine(
        tmp_path, monkeypatch):
    """ISOLATING DEFECT 3 FROM DEFECT 3b.  Found by mutation testing (2026-08-11):
    reverting the status fix alone did NOT fail the test above, because the
    reconciliation caught the same failure by a different route.  Two independent
    guards is the right design -- but a test suite that cannot tell them apart will
    let one of them be deleted silently.

    `baseline_tools/price_data` is STATIC: the same files every run.  So when its
    copy fails on run N, last run's identical copy is still sitting at the
    destination and the reconciliation -- which checks the destination, by name --
    reports it perfectly complete.  Only the recorded copy failure knows the run did
    not do what it said."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    _write(run / "baseline_tools" / "price_data" / "prices_2024.csv", "d,p\n")
    # last run's copy of the same static file, already at the destination:
    _write(os.path.join(transfer_dir, "baseline_tools", "price_data", "prices_2024.csv"), "d,p\n")

    _real_copytree = shutil.copytree

    def _fail_only_price_data(src, dst, *a, **k):
        # *a matters: shutil's own recursion into a sub-directory re-enters
        # `shutil.copytree` with SIX positional arguments, so a fake that only
        # accepts (src, dst, **k) makes every nested directory fail too -- which
        # silently turned this test into a copy of the one below.
        if "price_data" in str(src):
            raise PermissionError(13, "Access is denied", str(dst))
        return _real_copytree(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "copytree", _fail_only_price_data)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    # the reconciliation is happy -- yesterday's identical file is right there:
    assert result["reconciliation"]["detail"]["baseline_tools/price_data/"] == (1, 1)
    assert "baseline_tools/price_data/" not in result["reconciliation"]["incomplete"]
    # ...and the run must STILL not call itself a success:
    assert result["status"] != "success", (
        "a directory copy raised and the destination happened to look right, so the "
        "run reported success: %r" % result)
    assert any("price_data" in f for f in result["dir_failures"]), result


def test_a_root_FILE_that_failed_to_copy_is_not_reported_as_success(tmp_path, monkeypatch):
    """S4, END TO END.  The root-file loop's `except` printed and continued with no
    effect on the status, exactly as the directory loop's did -- and it is not a
    theoretical twin.  THREE allowlist patterns resolve to UNDATED names overwritten
    every run (`real_prices.csv`, `sectorsdic_fmp.pickle`, `pick_log.csv`), so a
    refused overwrite leaves LAST run's file at the destination; the reconciliation
    then finds a non-empty file of exactly the right name and reports the group
    COMPLETE.  Nothing except the copier can know.  An eight-month-old sector map is
    literally one of the four silent failures of 2026-08-07.

    BOTH stale copies are planted deliberately.  With only one planted, the OTHER
    group reads as `incomplete` and the reconciliation flips the status by itself --
    which is defence in depth doing its job, but it masks the gate under test and let
    mutation M17 survive.  Here the destination looks perfect and only `file_failures`
    knows anything went wrong."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    _write(run / "sectorsdic_fmp.pickle", "THIS RUN\n")
    _write(run / "pick_log.csv", "ticker,date\n")
    # last run's copies, already at the destination and STALE:
    stale = _write(os.path.join(transfer_dir, "sectorsdic_fmp.pickle"), "EIGHT MONTHS OLD\n")
    _write(os.path.join(transfer_dir, "pick_log.csv"), "an older run's picks\n")

    _real_copy2 = shutil.copy2

    def _refuse_the_undated(src, dst, *a, **k):
        if os.path.basename(str(src)) in ("sectorsdic_fmp.pickle", "pick_log.csv"):
            raise PermissionError(13, "Access is denied", str(dst))
        return _real_copy2(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "copy2", _refuse_the_undated)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    # the reconciliation is satisfied -- last run's files are right there, non-empty:
    assert result["reconciliation"]["detail"]["sectorsdic_fmp.pickle"] == (1, 1)
    assert result["reconciliation"]["detail"]["pick_log*.csv"] == (1, 1)
    assert result["reconciliation"]["incomplete"] == [], (
        "the reconciliation must be BLIND here, or this test is not exercising the "
        "file-failure gate: %r" % result["reconciliation"]["incomplete"])
    assert open(stale).read() == "EIGHT MONTHS OLD\n", "the destination kept the stale copy"
    # ...and the run must not call itself a success:
    assert result["status"] != "success", (
        "a root artifact failed to copy, the destination kept a stale version, and the "
        "run reported success -- the identical signature to the bug being fixed: %r"
        % result["status"])
    assert len(result["file_failures"]) == 2, result["file_failures"]
    loud, text = _emitted(result)
    assert loud is True and "sectorsdic_fmp.pickle" in text


def test_a_SILENTLY_partial_directory_copy_is_not_reported_as_success(tmp_path, monkeypatch):
    """CLOSING THE CLASS, NOT THE INSTANCE.  A copy that raises is now caught -- but
    the failure mode this function has actually shipped twice is a copy that raises
    NOTHING and still leaves the destination short (2026-08-07: `run_logs/` arrived
    empty and the console was indistinguishable from a clean transfer).

    `reconcile_transfer` already re-derives what should be at the destination and
    computes `result['complete']`; it was written for exactly this and nobody
    consumed it.  Here `copytree` succeeds while copying only one file, so nothing
    raises and only the reconciliation can tell."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)

    def _half(src, dst, **k):
        os.makedirs(str(dst), exist_ok=True)
        for dirpath, _d, files in os.walk(str(src)):
            for fn in sorted(files):
                rel = os.path.relpath(os.path.join(dirpath, fn), str(src))
                target = os.path.join(str(dst), rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(os.path.join(dirpath, fn), target)
                return str(dst)          # copy exactly ONE file, then claim success
        return str(dst)

    monkeypatch.setattr(shutil, "copytree", _half)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    assert result["complete"] is False, result.get("reconciliation")
    assert "logs/" in (result["reconciliation"]["incomplete"]), result["reconciliation"]
    assert result["status"] != "success", (
        "a directory that landed half-copied WITHOUT raising was reported as a clean "
        "transfer -- the reconciliation saw it and the status ignored it: %r" % result)


def test_a_group_never_written_locally_does_NOT_flip_the_status(tmp_path, monkeypatch):
    """THE DELIBERATE LIMIT ON THE ABOVE, so the loud banner keeps meaning something.

    `reconcile_transfer` reports two different things in one flag.  `incomplete` =
    it existed locally and did not land -> a COPY failure, which is this function's
    fault and must fail the status.  `missing` = nothing was produced locally ->
    the WRITER did not run, and (in reconcile_transfer's own words) "fixing the
    copier will not help".

    Eight of the 31 allowlist pattern groups match nothing on a healthy machine
    (measured 2026-08-11: real_prices.csv, FxRates_*, FxRatesHistorical_*,
    CurrencyExclusionStatus_*, DelistedPrune_*, VendorContaminationFlags_*,
    removed_data_quality_*, AdHocPenaltyBucket_*), so `complete` is False on a
    PERFECTLY HEALTHY run.  Gating "DRIVE TRANSFER DID NOT COMPLETE" on `complete`
    alone would fire it every single run -- alarm fatigue, which is the exact
    mechanism that hid the real failure.  So: `missing` is reported separately and
    loudly by the caller, and only `incomplete` fails the status."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    assert result["status"] == "success", result
    assert result["complete"] is False, "no root artifacts exist, so nothing could land"
    assert result["reconciliation"]["missing"], "the never-written groups must still be named"
    assert not result["reconciliation"]["incomplete"], result["reconciliation"]
    # ...and the directories that DID exist reconciled clean:
    assert result["reconciliation"]["detail"]["logs/"] == (2, 2), result["reconciliation"]["detail"]
    assert result["reconciliation"]["detail"]["output/"] == (2, 2)


# --------------------------------------------------------------------------- #
# 4. THE REPORTING BLOCK ITSELF
#    Behavioural, not an AST grep.  The first version of this test string-searched
#    the caller's source, and one of its three assertions was satisfied BY A COMMENT:
#    neutering the block (`recon = {}`) left the whole suite green.  A reporting path
#    whose entire job is to be noticed has to be tested by what it EMITS, which is
#    why `report_transfer_outcome(result, transfer_dir, emit=...)` exists.
# --------------------------------------------------------------------------- #
def _emitted(result, transfer_dir="E:/drive/valuationTransfer"):
    lines = []
    loud = Sbocker.report_transfer_outcome(result, transfer_dir, emit=lines.append)
    return loud, "\n".join(str(x) for x in lines)


def test_the_report_is_QUIET_on_a_clean_transfer():
    loud, text = _emitted({'status': 'success', 'message': 'ok', 'complete': True,
                           'reconciliation': {'missing': [], 'incomplete': []}})
    assert loud is False
    assert "DID NOT COMPLETE" not in text
    assert "NEVER PRODUCED LOCALLY" not in text


def test_the_report_is_LOUD_on_a_directory_failure_and_names_it():
    loud, text = _emitted({'status': 'warning', 'message': 'boom', 'complete': False,
                           'dir_failures': ['logs/ (WinError 5)'],
                           'reconciliation': {'missing': [], 'incomplete': []}})
    assert loud is True
    assert "DRIVE TRANSFER DID NOT COMPLETE" in text
    assert "logs/ (WinError 5)" in text, "the banner must name the directory that failed"
    assert "copy the run outputs to the Drive MANUALLY" in text


def test_the_report_is_LOUD_on_a_FILE_failure_and_names_it():
    """The S4 half: three allowlist patterns are UNDATED and overwritten every run,
    so a refused overwrite leaves a stale file the reconciliation reports as
    complete.  Only the copier knows, so only the copier can say so."""
    loud, text = _emitted({'status': 'warning', 'message': 'boom', 'complete': True,
                           'file_failures': ['sectorsdic_fmp.pickle (WinError 5)'],
                           'reconciliation': {'missing': [], 'incomplete': []}})
    assert loud is True
    assert "sectorsdic_fmp.pickle" in text


def test_the_report_names_a_group_that_is_SHORT_at_the_destination():
    loud, text = _emitted({'status': 'warning', 'message': 'boom', 'complete': False,
                           'reconciliation': {'missing': [], 'incomplete': ['output/']}})
    assert loud is True
    assert "SHORT at the destination" in text and "output/" in text


def test_the_report_screams_about_a_denied_file_on_the_drive_and_says_it_was_NOT_removed():
    """The run does not delete it (CEO, 2026-08-11), so the report must say so
    explicitly -- otherwise the operator reads 'reported' as 'handled' and the file
    stays on a synced folder indefinitely."""
    loud, text = _emitted({'status': 'error', 'message': 'boom', 'complete': False,
                           'denied_at_destination': ['E:/drive/x/output/old_key.txt'],
                           'reconciliation': {'missing': [], 'incomplete': []}})
    assert loud is True
    assert "DENYLISTED FILE(S) PRESENT ON THE DRIVE" in text
    assert "NOT removed" in text and "remove by hand" in text
    assert "old_key.txt" in text


def test_a_NEVER_PRODUCED_group_gets_the_quiet_note_and_NOT_the_loud_banner():
    """The deliberate split, at the reporting layer.  8 of the 31 pattern groups
    legitimately produce nothing on a healthy machine; putting them behind the loud
    banner would fire it every run, which is the alarm-fatigue mechanism that hid the
    original failure."""
    loud, text = _emitted({'status': 'success', 'message': 'ok', 'complete': False,
                           'reconciliation': {'missing': ['real_prices.csv',
                                                          'FxRates_*.csv'],
                                              'incomplete': []}})
    assert loud is False, "a never-written group must NOT fire the loud banner"
    assert "NEVER PRODUCED LOCALLY" in text
    assert "real_prices.csv" in text and "FxRates_*.csv" in text
    assert "the WRITER did not run" in text


def test_the_report_always_states_the_reconciliation_verdict():
    """`complete` was computed on every run since 2026-08-07 and read by nobody.  It
    is now printed whatever the outcome -- this is the assertion a comment cannot
    satisfy."""
    for status, complete in (('success', True), ('success', False), ('warning', False)):
        _loud, text = _emitted({'status': status, 'message': 'm', 'complete': complete,
                                'reconciliation': {'missing': [], 'incomplete': []}})
        assert f"Transfer reconciled complete: {complete}" in text, (status, text)


def test_the_report_survives_a_result_dict_that_is_missing_everything():
    """It runs at the tail of a 12-hour run; it must never be the thing that raises.
    The round-2 fuzz found the two shapes that DID raise -- a non-dict
    `reconciliation`, and joining a list holding a non-str -- neither reachable from
    the real producer, both now guarded, so the docstring's absolute is now true."""
    for r in ({}, None, {'status': 'error'}, "not a dict", 7,
              {'status': 'warning', 'reconciliation': 'not a dict'},
              {'status': 'warning', 'dir_failures': [None, 3],
               'denied_at_destination': [object()]}):
        loud, text = _emitted(r)
        assert isinstance(text, str)


def test_the_report_says_the_target_was_unavailable_when_nothing_could_be_verified():
    """R2-3's reporting half: `missing` and `incomplete` are both empty on this
    branch, so without its own line the operator sees a clean-looking tail."""
    loud, text = _emitted({'status': 'warning', 'message': 'm', 'complete': False,
                           'reconciliation': {'missing': [], 'incomplete': [],
                                              'unreconciled': True}})
    assert loud is True
    assert "UNAVAILABLE" in text


# --------------------------------------------------------------------------- #
# 5. THE DENYLIST PREDICATE ITSELF
#    `is_denied` governs BOTH directions -- what is refused at the destination and
#    what is refused on the way out -- so a false positive here is a silent transfer
#    gap, which is the class of bug this whole change exists to end.
# --------------------------------------------------------------------------- #
def test_the_credential_patterns_still_deny():
    """Non-negotiable: the fix to the glob semantics must not loosen the credential
    match by even one name."""
    for name in ("fmpAPIkey.txt", "FMPAPIKEY.TXT", "backup_key_2026.txt",
                 "server.pem", "PRIVATE.PEM", "api_key_dump.txt",
                 "key-metrics_cache.json", "output/a/b/deep_key_backup.txt"):
        assert tu.is_denied(name), "credential-shaped name no longer denied: %r" % name


def test_pem_means_ENDS_with_pem_not_CONTAINS_pem():
    """`'*pem'` was implemented as `'pem' in name` -- every pattern was turned into a
    substring test by `pattern.replace('*','')`.  As a glob it means ENDS WITH, and
    the difference is not academic: `is_denied` also decides what SHIPS, so a research
    file named `Pemex_bond_notes.md` sitting in `output/` was silently not
    transferred -- the exact failure class this change exists to end."""
    for name in ("Pemex_bond_notes.md", "PEMBINA_pipeline_dcf.xlsx",
                 "pemex.csv", "compensation_screen.csv", "temperature_hedge.csv"):
        assert not tu.is_denied(name), "still denied by the substring bug: %r" % name
    assert tu.is_denied("wildcard.pem"), "a real .pem must still deny"


def test_key_remains_a_SUBSTRING_pattern_BY_DESIGN_and_this_is_a_known_gap():
    """DOCUMENTING A LIMIT, NOT ASSERTING A VIRTUE.  `'*key*'` has wildcards on BOTH
    sides, so under correct glob semantics it is still a substring test and still
    denies ordinary analyst filenames.  These files will NOT transfer.  That is a
    denylist-POLICY question (the same pattern is what catches FMP's `key-metrics_*`
    caches), not a bug to fix quietly here -- so it is pinned, visibly, as the thing
    someone must decide rather than discover."""
    for name in ("Turkey_exposure_2026.csv", "monkey_basket.csv",
                 "hockey_stick_screen.csv", "Keystone_Corp_thesis.docx"):
        assert tu.is_denied(name), (
            "if this now passes, the denylist policy was changed -- intentionally? %r" % name)
    # ...and the files that were only ever collateral of the BUG are free again:
    assert not tu.is_denied("Turkiye_exposure.csv")


def test_a_pem_named_research_file_now_TRANSFERS(tmp_path, monkeypatch):
    """The end-to-end consequence of the glob fix, at the copy layer rather than the
    predicate: the file reaches the destination."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    _write(run / "output" / "Pemex_bond_notes.md", "thesis\n")
    _write(run / "output" / "PEMBINA_pipeline_dcf.xlsx", "model\n")
    _write(run / "output" / "server.pem", "SECRET")

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    landed = _files_under(transfer_dir)
    assert "output/Pemex_bond_notes.md" in landed, sorted(landed)
    assert "output/PEMBINA_pipeline_dcf.xlsx" in landed
    assert "output/server.pem" not in landed, "the real .pem must still be refused"
    assert result["status"] == "success", result


# --------------------------------------------------------------------------- #
# 6. THE pipeline / non-pipeline FOLDER SPLIT (CEO, 2026-08-11)
# --------------------------------------------------------------------------- #
def test_pointing_at_the_PARENT_of_the_split_is_REFUSED_and_names_the_right_folder(
        tmp_path, monkeypatch):
    """The one misconfiguration the new folder structure makes possible.  Nothing in
    the transfer logic needs to know about the split -- it writes wherever it is
    pointed -- which is exactly why passing the PARENT would silently work: run
    artifacts would land beside the CEO's manual files, and the destination-side
    denylist check would start reporting on his research."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)
    os.makedirs(os.path.join(transfer_dir, "non-pipeline"))
    os.makedirs(os.path.join(transfer_dir, "pipeline"))
    manual = _write(os.path.join(transfer_dir, "non-pipeline", "my_notes.md"), "mine\n")

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    assert result["status"] == "error", result
    assert "non-pipeline" in result["message"]
    assert os.path.join("pipeline") in result["message"], (
        "the refusal must name the directory that WAS meant: %r" % result["message"])
    # nothing was written, and the manual drop zone is untouched:
    assert "logs" not in os.listdir(transfer_dir)
    assert open(manual).read() == "mine\n"
    loud, text = _emitted(result)
    assert loud is True


def test_the_pipeline_LEAF_is_accepted_normally(tmp_path, monkeypatch):
    """The other half: a `pipeline` leaf with a `non-pipeline` SIBLING (not a child)
    is the correct target and must transfer normally.  Detection is by the presence of
    a `non-pipeline` CHILD, so a sibling must not trip it."""
    run = tmp_path / "run"
    run.mkdir()
    root = tmp_path / "drive" / "valuationTransfer"
    (root / "non-pipeline").mkdir(parents=True)
    monkeypatch.chdir(run)
    _populate(run)

    result = Sbocker.transfer_outputs_to_drive(str(root / "pipeline"), {}, verbose=False)

    assert result["status"] == "success", result
    assert "logs/run_2026-08-11.log" in _files_under(root / "pipeline")
    assert tu.looks_like_transfer_parent(root) is not None, "the parent IS detectable"
    assert tu.looks_like_transfer_parent(root / "pipeline") is None


def test_a_README_labels_the_folder_ONCE_and_is_never_rewritten(tmp_path, monkeypatch):
    """A LABEL, NOT A LICENCE.  It tells whoever opens the folder that runs add to and
    overwrite it, that the pipeline deletes nothing, and where manual files belong.
    Written when the pipeline CREATES the directory and never again -- an edited note
    must survive the next run."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)

    Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    readme = os.path.join(transfer_dir, "README-pipeline-managed.txt")
    assert os.path.exists(readme), sorted(os.listdir(transfer_dir))
    body = open(readme, encoding="utf-8").read()
    assert "non-pipeline" in body and "OVERWRITES" in body
    assert "never DELETED" in body or "never" in body

    with open(readme, "w", encoding="utf-8") as fh:
        fh.write("the operator edited this\n")
    Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)
    assert open(readme, encoding="utf-8").read() == "the operator edited this\n", (
        "the README was rewritten on a later run")

    # AND THE WRITER'S OWN GUARD, DIRECTLY.  Two guards protect this -- the caller's
    # `already_existed` gate and the writer's `exists()` check -- and each masks the
    # other from an end-to-end test, so mutating either one alone was invisible.
    # Testing the writer on its own separates them.
    assert tu.write_pipeline_readme(transfer_dir, verbose=False) is None, (
        "the writer must decline when a README is already there")
    assert open(readme, encoding="utf-8").read() == "the operator edited this\n"


def test_no_README_is_written_into_a_folder_the_pipeline_did_NOT_create(tmp_path, monkeypatch):
    """THE OTHER HALF OF THE SAME GUARD PAIR.  The label is written when the PIPELINE
    creates the directory (CEO's wording) and not otherwise: a folder the CEO made by
    hand is his, and dropping a file into it uninvited is the small version of exactly
    the presumption this round removed from the destination check."""
    run = tmp_path / "run"
    run.mkdir()
    root = tmp_path / "drive"
    root.mkdir()
    made_by_hand = root / "valuationTransfer"
    made_by_hand.mkdir()                     # the operator created it, not us
    monkeypatch.chdir(run)
    _populate(run)

    result = Sbocker.transfer_outputs_to_drive(str(made_by_hand), {}, verbose=False)

    assert result["status"] == "success", result
    assert "logs/run_2026-08-11.log" in _files_under(made_by_hand), "it still transfers"
    assert not (made_by_hand / "README-pipeline-managed.txt").exists(), (
        "the run wrote a file into a directory it did not create")


def test_a_target_that_vanishes_before_the_reconciliation_is_not_a_success(tmp_path, monkeypatch):
    """R2-3, END TO END.  Drive present at mkdir and copy time, gone by the time the
    run tries to verify: `reconcile_transfer` can check nothing, so `missing` AND
    `incomplete` are both empty and the status gate built on those two cannot see it.
    The run reported `success` after nothing reached the Drive -- the same signature as
    the bug this whole change exists to close.  Failing on it costs nothing, because it
    fires on exactly zero healthy runs."""
    run, transfer_dir = _run_dir(tmp_path, monkeypatch)
    _populate(run)

    _real_reconcile = tu.reconcile_transfer

    def _target_gone(td, groups, verbose=True):
        shutil.rmtree(str(td), ignore_errors=True)      # the Drive unmounts, mid-run
        return _real_reconcile(td, groups, verbose=verbose)

    monkeypatch.setattr(tu, "reconcile_transfer", _target_gone)

    result = Sbocker.transfer_outputs_to_drive(transfer_dir, {}, verbose=False)

    assert result["reconciliation"]["unreconciled"] is True
    assert result["reconciliation"]["missing"] == []
    assert result["reconciliation"]["incomplete"] == [], (
        "if this is non-empty the test is not exercising the unreconciled gate")
    assert result["status"] != "success", (
        "nothing could be verified to have reached the Drive and the run said success")
    loud, text = _emitted(result)
    assert loud is True and "UNAVAILABLE" in text


# --------------------------------------------------------------------------- #
# 7. THE INCREMENTAL COPIER'S DIRECTORY PATH, CLOSED BY DELETION (CEO approved)
# --------------------------------------------------------------------------- #
def test_the_incremental_copier_REFUSES_a_directory_loudly(tmp_path):
    """`copy_artifacts_to_transfer_dir` carried BOTH defects that cost four production
    runs every `logs/`, `output/` and `run_logs/` on the Drive -- rmtree-then-copytree
    and whole-tree denial -- in a branch that is dormant (all six call sites pass file
    paths only).  Rather than maintain a second copy of the same subtle fix, the input
    is refused: a directory here is a programming error and is now a LOUD one instead
    of a silent evidence loss the day someone tries it."""
    src = tmp_path / "src"
    (src / "logs").mkdir(parents=True)
    _write(src / "logs" / "run.log", "x")
    good = _write(src / "postRank.pickle", "x")
    parent = tmp_path / "drive"
    parent.mkdir()
    transfer_dir = str(parent / "run_out")

    r = tu.copy_artifacts_to_transfer_dir(transfer_dir, [str(src / "logs"), good],
                                          verbose=False)

    assert r["status"] == "error", r
    assert r["refused"] == [str(src / "logs")], r["refused"]
    assert "logs" not in os.listdir(transfer_dir), "the directory must not be copied"
    # ...and the FILE alongside it still copied -- refusing an input is not aborting:
    assert "postRank.pickle" in os.listdir(transfer_dir)
    assert r["copied"] == 1, r
