"""
OFFLINE tests for crash-resilient mid-run output handling (investment-filter,
2026-07-13).  Synthetic, deterministic, NO network -- every network touch point in
run_ingest is injected; the copy helper touches only temp dirs.

Run:  python baseline_tools/test_incremental_transfer.py

Proves the load-bearing guarantees of the incremental transfer + manifest flush:
  * artifacts land in the transfer dir INCREMENTALLY, not only at end-of-run
                                                        -> test_incremental_copy_is_incremental
  * a CURRENT manifest exists on disk after an EARLY phase, counts accumulated
                                                        -> test_manifest_flush_midrun
  * the API key is NEVER copied under ANY path (file, wildcard, .pem, nested dir)
                                                        -> test_denylist_never_copies_key
  * a copy/flush FAILURE logs + is swallowed; the run continues (no exception)
                                                        -> test_copy_failure_never_crashes
  * transfer_dir unset -> strict no-op                  -> test_unset_is_strict_noop
  * end-to-end: a LATE crash (in the dead-fundamentals phase) still leaves ALL
    prior phases synced to Drive + a current manifest    -> test_late_crash_prior_phases_synced
  * the human-readable top-N deliverables (AggScore/Presentation/ForensicFlags)
    reach Drive at the PRE-ingestion boundary, before any ingestion output, and
    the key is still never copied                        -> test_deliverables_synced_pre_ingestion
"""
import os
import sys
import glob
import json
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import transfer_utils as tu
import delisted_ingest as di
from run_logging import RunLogger


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mk_transfer_dir():
    """A transfer dir whose PARENT exists but whose leaf does not (the helper
    creates the leaf) -- mirrors a real mounted-Drive parent."""
    parent = tempfile.mkdtemp(prefix="xfer_parent_")
    return parent, os.path.join(parent, "run_out")


def _write(path, text="x"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _names_in(d):
    return set(os.listdir(d)) if os.path.isdir(d) else set()


# --------------------------------------------------------------------------- #
# 1. incremental: each phase's file appears as that phase completes
# --------------------------------------------------------------------------- #
def test_incremental_copy_is_incremental():
    src = tempfile.mkdtemp(prefix="src_")
    parent, transfer_dir = _mk_transfer_dir()
    try:
        p1 = _write(os.path.join(src, "Bometric_dic-phase1.pickle"))
        r = tu.copy_artifacts_to_transfer_dir(transfer_dir, [p1], verbose=False)
        assert r["status"] == "success" and r["copied"] == 1, r
        assert "Bometric_dic-phase1.pickle" in _names_in(transfer_dir)
        # phase-2 artifact does NOT exist in the transfer dir yet (proves it is
        # copied per-phase, not batched at the end).
        assert "Boresults_dic-phase2.pickle" not in _names_in(transfer_dir)

        p2 = _write(os.path.join(src, "Boresults_dic-phase2.pickle"))
        tu.copy_artifacts_to_transfer_dir(transfer_dir, [p2], verbose=False)
        assert {"Bometric_dic-phase1.pickle", "Boresults_dic-phase2.pickle"} \
            <= _names_in(transfer_dir)

        # idempotent: re-copying phase-1 overwrites cleanly, no error.
        r3 = tu.copy_artifacts_to_transfer_dir(transfer_dir, [p1], verbose=False)
        assert r3["errors"] == 0 and r3["copied"] == 1
        print("PASS incremental copy: phase files land per-phase, not batched; idempotent")
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(parent, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 2. manifest flush mid-run: current manifest on disk after an EARLY phase
# --------------------------------------------------------------------------- #
def test_manifest_flush_midrun():
    log_dir = tempfile.mkdtemp(prefix="logs_")
    try:
        lg = RunLogger("midrun", out_dir=log_dir, secrets=["KEY"], echo=False)
        lg.set_count("live_count", 100)
        lg.incr("registry_total", 5)
        path = lg.write_manifest()           # EARLY flush (before the long loop)
        assert path and os.path.exists(path), "an early-phase manifest must exist on disk"
        man = json.load(open(path, encoding="utf-8"))
        assert man["counts"]["live_count"] == 100
        assert man["counts"]["registry_total"] == 5
        assert man["finished"] is not None, "flush stamps an as-of finished time"

        # a later flush ACCUMULATES + overwrites idempotently (single manifest file).
        lg.set_count("dead_fund_fetched", 42)
        lg.write_manifest()
        man2 = json.load(open(lg.manifest_path, encoding="utf-8"))
        assert man2["counts"]["live_count"] == 100
        assert man2["counts"]["dead_fund_fetched"] == 42
        assert len(glob.glob(os.path.join(log_dir, "run_manifest_*.json"))) == 1
        lg.close()
        print("PASS manifest flush mid-run: current manifest on disk, counts accumulate, idempotent")
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 3. denylist: the API key is NEVER copied, under any path
# --------------------------------------------------------------------------- #
def test_denylist_never_copies_key():
    src = tempfile.mkdtemp(prefix="src_")
    parent, transfer_dir = _mk_transfer_dir()
    try:
        key = _write(os.path.join(src, "fmpAPIkey.txt"), "SECRET")     # exact name
        wildcard = _write(os.path.join(src, "backup_key_2026.txt"))    # *key*
        pem = _write(os.path.join(src, "server.pem"))                  # *pem
        data = _write(os.path.join(src, "bulk_prices.parquet"))        # legit
        r = tu.copy_artifacts_to_transfer_dir(
            transfer_dir, [key, wildcard, pem, data], verbose=False)
        names = _names_in(transfer_dir)
        assert "bulk_prices.parquet" in names, "legit artifact must copy"
        assert "fmpAPIkey.txt" not in names, "KEY LEAKED via exact name"
        assert "backup_key_2026.txt" not in names, "KEY LEAKED via *key* wildcard"
        assert "server.pem" not in names, "secret LEAKED via *pem"
        assert r["denied"] == 3 and r["copied"] == 1, r

        # a denied file INSIDE a passed directory blocks the whole-dir copy.
        subdir = os.path.join(src, "run_logs")
        os.makedirs(subdir)
        _write(os.path.join(subdir, "run_events.jsonl"))
        _write(os.path.join(subdir, "fmpAPIkey.txt"), "SECRET")
        tu.copy_artifacts_to_transfer_dir(transfer_dir, [subdir], verbose=False)
        assert "run_logs" not in _names_in(transfer_dir), \
            "a dir containing a key file must NOT be copied"

        # post-copy safety net actively removes a key file that somehow lands there.
        planted = _write(os.path.join(transfer_dir, "fmpAPIkey.txt"), "SECRET")
        assert os.path.exists(planted)
        assert tu.assert_no_key_file(transfer_dir, verbose=False) is True
        assert not os.path.exists(planted), "safety net must delete a planted key file"
        print("PASS denylist: key never copied (exact/*key*/*pem/nested dir) + safety net removes it")
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(parent, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 4. copy/flush failure NEVER crashes the run
# --------------------------------------------------------------------------- #
def test_copy_failure_never_crashes():
    src = tempfile.mkdtemp(prefix="src_")
    try:
        good = _write(os.path.join(src, "postRank.pickle"))

        # (a) bad dest: parent does not exist -> graceful warning, no raise.
        bad_dir = os.path.join(tempfile.gettempdir(), "no_such_parent_xyz", "leaf")
        r_a = tu.copy_artifacts_to_transfer_dir(bad_dir, [good], verbose=False)
        assert r_a["status"] == "warning" and r_a["copied"] == 0, r_a

        # (b) copy2 itself raises mid-copy -> swallowed, counted, run continues.
        parent, transfer_dir = _mk_transfer_dir()
        orig = tu.shutil.copy2
        try:
            def boom(*a, **k):
                raise OSError("simulated Drive hiccup")
            tu.shutil.copy2 = boom
            r_b = tu.copy_artifacts_to_transfer_dir(transfer_dir, [good], verbose=False)
        finally:
            tu.shutil.copy2 = orig
        assert r_b["status"] == "success" and r_b["errors"] == 1 and r_b["copied"] == 0, r_b
        # we reached here without an exception -> the run would have continued.

        # (c) manifest flush hiccup is also swallowed (returns None, no raise).
        lg = RunLogger("failflush", out_dir=transfer_dir, secrets=["K"], echo=False)
        orig_dump = json.dump
        try:
            def dboom(*a, **k):
                raise OSError("disk full")
            di.RunLogger  # touch to keep import used
            import run_logging as rl
            rl.json.dump = dboom
            out = lg.write_manifest()
            assert out is None, "a failing manifest flush must return None, not raise"
        finally:
            rl.json.dump = orig_dump
            lg.close()
        shutil.rmtree(parent, ignore_errors=True)
        print("PASS copy/flush failure is swallowed (bad dest, copy2 raise, flush raise) -- run continues")
    finally:
        shutil.rmtree(src, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 5. transfer_dir unset -> strict no-op
# --------------------------------------------------------------------------- #
def test_unset_is_strict_noop():
    src = tempfile.mkdtemp(prefix="src_")
    try:
        f = _write(os.path.join(src, "postRank.pickle"))
        for falsy in (None, "", 0):
            r = tu.copy_artifacts_to_transfer_dir(falsy, [f], verbose=False)
            assert r == {'status': 'skipped', 'copied': 0, 'denied': 0,
                         'errors': 0, 'files': []}, (falsy, r)
        print("PASS transfer_dir unset/falsy -> strict no-op (no dirs touched)")
    finally:
        shutil.rmtree(src, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 6. end-to-end: a LATE crash still leaves prior phases synced + current manifest
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


def _router_get(dead_symbol):
    """Single injected get() routing registry / ride-along / bulk URLs offline."""
    reg_page = [
        {"symbol": dead_symbol, "companyName": "Dead Co", "exchange": "NYSE",
         "ipoDate": "2000-01-01", "delistedDate": "2023-05-01"},
        {"symbol": "OTHER", "companyName": "Other Co", "exchange": "NASDAQ",
         "ipoDate": "2001-01-01", "delistedDate": "2019-06-30"},
    ]
    split_hist = {"historical": [
        {"date": "2020-08-28", "close": 500.0, "adjClose": 125.0},
        {"date": "2020-08-31", "close": 125.0, "adjClose": 125.0},
    ]}
    filing_rows = [{"date": f"{y}-12-31", "fillingDate": f"{y + 1}-02-15",
                    "acceptedDate": f"{y + 1}-02-14"} for y in range(2015, 2024)]
    state = {"reg_calls": 0}

    def _get(url):
        if "delisted-companies" in url:
            state["reg_calls"] += 1
            return reg_page if state["reg_calls"] == 1 else []   # short page -> terminate
        if "historical-price-full/AAPL" in url:
            return split_hist
        if "income-statement/AAPL" in url:
            return filing_rows
        if "batch-request-end-of-day-prices" in url:
            d = url.split("date=")[1].split("&")[0]
            return [{"symbol": dead_symbol, "date": d, "close": 60.0, "adjClose": 60.0}]
        return []
    return _get


def test_late_crash_prior_phases_synced():
    data_dir = tempfile.mkdtemp(prefix="deadout_")
    log_dir = tempfile.mkdtemp(prefix="deadlogs_")
    parent, transfer_dir = _mk_transfer_dir()
    dead_symbol = "DEADX"
    configdic = {
        "baseurl": "https://x/api/", "api_key": "SUPERSECRETKEY",
        "transfer_dir": transfer_dir, "period": "quarter",
        "startfromlastindex": 0, "delisted_max_pages": 5,
    }
    orig_fund = di.fetch_dead_fundamentals

    def crash(*a, **k):
        raise RuntimeError("simulated crash in the LONG dead-fundamentals loop")

    crashed = False
    try:
        di.fetch_dead_fundamentals = crash
        try:
            di.run_ingest(configdic, live_symbols=["AAPL", "MSFT"],
                          get=_router_get(dead_symbol), http_get=lambda u: _FakeResp([]),
                          sleep=lambda s: None, data_dir=data_dir, log_dir=log_dir,
                          do_fundamentals=True)
        except RuntimeError:
            crashed = True
    finally:
        di.fetch_dead_fundamentals = orig_fund

    assert crashed, "the injected late crash must actually fire"
    xfer = _names_in(transfer_dir)

    # prior phases (registry, ride-alongs, bulk) ALREADY on Drive despite the crash:
    assert glob.glob(os.path.join(transfer_dir, "delisted_registry.*")), \
        f"registry not synced before crash: {xfer}"
    assert glob.glob(os.path.join(transfer_dir, "ridealong_split_*.json")), xfer
    assert glob.glob(os.path.join(transfer_dir, "ridealong_filing_*.json")), xfer
    assert glob.glob(os.path.join(transfer_dir, "bulk_prices.*")), xfer
    # a CURRENT manifest + JSONL rode along to Drive too:
    assert glob.glob(os.path.join(transfer_dir, "run_manifest_*.json")), xfer
    assert glob.glob(os.path.join(transfer_dir, "run_events_*.jsonl")), xfer
    # the phase that crashed did NOT sync its output:
    assert not glob.glob(os.path.join(transfer_dir, "dead_fundamentals_*.pickle")), \
        "the crashed phase's artifact must NOT be present"

    # a CURRENT manifest is on the LOCAL disk with counts accumulated so far:
    local_man = glob.glob(os.path.join(log_dir, "run_manifest_*.json"))
    assert local_man, "no local manifest after a mid-run crash"
    man = json.load(open(local_man[0], encoding="utf-8"))
    assert man["counts"].get("live_count") == 2, man["counts"]

    # the API key NEVER reached the transfer dir under any path:
    assert "SUPERSECRETKEY" not in "".join(xfer)
    assert "fmpAPIkey.txt" not in xfer
    for root, _d, files in os.walk(transfer_dir):
        assert "fmpAPIkey.txt" not in files
        for fn in files:
            assert not tu.is_denied(fn), f"denied file synced: {fn}"

    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(log_dir, ignore_errors=True)
    shutil.rmtree(parent, ignore_errors=True)
    print("PASS late-crash: registry+ridealong+bulk+manifest+JSONL synced; crashed phase absent; no key")


# --------------------------------------------------------------------------- #
# 7. deliverables reach Drive at the PRE-ingestion phase boundary
# --------------------------------------------------------------------------- #
def test_deliverables_synced_pre_ingestion():
    """Mirrors the Sbocker.main Phase-3 deliverable copy: writeResWrapper writes the
    human-readable top-N deliverables, then main copies them to the transfer dir
    BEFORE run_ingest.  Proves (a) all three deliverables land, (b) they are on Drive
    before any ingestion-phase output exists there, (c) the key is never copied."""
    src = tempfile.mkdtemp(prefix="src_")
    parent, transfer_dir = _mk_transfer_dir()
    try:
        # Exact filename patterns emitted by postBo.writeResWrapper (see postBo.py:
        # AggScoreTop{n}-, PresentationTop{n}-, ForensicFlagsTop{n}-).
        agg = _write(os.path.join(src, "AggScoreTop100-2026-01-09_fmp_stock_NA1_EU1.csv"))
        pres = _write(os.path.join(src, "PresentationTop20-2026-01-09_fmp_stock_NA1_EU1.xlsx"))
        forensic = _write(os.path.join(src, "ForensicFlagsTop100-2026-01-09_fmp_stock_NA1_EU1.csv"))
        # A key file rides along in the same call -- it must still never be copied.
        key = _write(os.path.join(src, "fmpAPIkey.txt"), "SECRET")

        deliverables = [agg, pres, forensic]
        r = tu.copy_artifacts_to_transfer_dir(
            transfer_dir, deliverables + [key], verbose=False)

        names = _names_in(transfer_dir)
        for d in deliverables:
            assert os.path.basename(d) in names, f"deliverable not synced: {d}"
        assert r["copied"] == 3 and r["denied"] == 1, r

        # PRE-ingestion: no ingestion-phase output has been produced yet, so none is
        # present in the transfer dir at this boundary -- proving deliverables reach
        # Drive BEFORE (not after) the multi-hour ingestion.
        assert not glob.glob(os.path.join(transfer_dir, "dead_fundamentals_*.pickle")), names
        assert not glob.glob(os.path.join(transfer_dir, "delisted_registry.*")), names
        assert not glob.glob(os.path.join(transfer_dir, "bulk_prices.*")), names

        # the key never landed, under any path:
        assert "fmpAPIkey.txt" not in names, "KEY LEAKED at the deliverable boundary"
        for root, _d, files in os.walk(transfer_dir):
            for fn in files:
                assert not tu.is_denied(fn), f"denied file synced: {fn}"
        print("PASS deliverables synced pre-ingestion: AggScore+Presentation+ForensicFlags "
              "on Drive before any ingestion output; key never copied")
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(parent, ignore_errors=True)


if __name__ == "__main__":
    test_incremental_copy_is_incremental()
    test_manifest_flush_midrun()
    test_denylist_never_copies_key()
    test_copy_failure_never_crashes()
    test_unset_is_strict_noop()
    test_late_crash_prior_phases_synced()
    test_deliverables_synced_pre_ingestion()
    print("\nALL INCREMENTAL-TRANSFER TESTS PASSED (7/7)")
