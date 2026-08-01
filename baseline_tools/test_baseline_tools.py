"""
Offline tests for the investment-filter baseline tools.

Runs on SYNTHETIC fixtures (fast, deterministic) plus an optional smoke test
against a real saved pickle if one is present. NO network, ever.

Run:  python baseline_tools/test_baseline_tools.py
"""

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import universe_dedup as ud
import benchmark_loader as bl
import beat_rate as br


def test_dedup_collapses_preferred_and_keeps_distinct():
    # GSL / GSL-PB share issuer fundamentals; ACME is a distinct company.
    df = pd.DataFrame({
        "source": ["GSL", "GSL-PB", "ACME"],
        "date": ["2025-09-30"] * 3,
        "revenue": [1000.0, 1000.0, 2500.0],
        "netIncome": [200.0, 200.0, 300.0],
        "totalAssets": [5000.0, 5000.0, 9000.0],
        "weightedAverageShsOut": [50.0, 50.0, 80.0],
    })
    kept, dropped, _ = ud.dedup_universe(df)
    assert "GSL" in kept and "ACME" in kept
    assert dropped == {"GSL-PB": "GSL"}, dropped
    print("PASS dedup preferred/distinct")


def test_dedup_dualclass_toggle():
    df = pd.DataFrame({
        "source": ["TCL-A", "TCL-B"],
        "date": ["2025-09-30"] * 2,
        "revenue": [500.0, 500.0],
        "netIncome": [40.0, 40.0],
        "totalAssets": [2000.0, 2000.0],
        "weightedAverageShsOut": [30.0, 30.0],
    })
    kept_c, dropped_c, _ = ud.dedup_universe(df, collapse_dual_class=True)
    assert len(kept_c) == 1 and len(dropped_c) == 1
    kept_k, dropped_k, _ = ud.dedup_universe(df, collapse_dual_class=False)
    assert len(kept_k) == 2 and len(dropped_k) == 0
    print("PASS dedup dual-class toggle")


def test_benchmark_window_return():
    s = pd.Series(
        [100.0, 110.0, 121.0, 133.1],
        index=pd.to_datetime(["2019-12-31", "2020-12-31",
                              "2021-12-31", "2022-12-31"]),
    )
    # 2019-12-31 -> 2022-12-31 : 133.1/100 - 1 = 0.331
    r = bl.window_return(s, "2019-12-31", "2022-12-31")
    assert abs(r - 0.331) < 1e-9, r
    # nearest-prior semantics: a weekend eval date maps back to 2022-12-31
    r2 = bl.window_return(s, "2019-12-31", "2023-01-01")
    assert abs(r2 - 0.331) < 1e-9, r2
    print("PASS benchmark window_return")


def _write_bundle(tmp):
    sel = pd.DataFrame({
        "window_id": ["buy2019"] * 3,
        "buy_date": ["2019-12-31"] * 3,
        "eval_date": ["2022-12-31"] * 3,
        "source": ["WIN", "LOSE", "GONE"],
        "stage1_rank": [1, 2, 3],
        "stage2_rank": [1, 2, 3],
        "BoScore": [1.0, 0.9, 0.8],
        "AggScore": [3.0, 2.0, 1.0],
        "is_top20": [True, True, True],
        "currency": ["USD", "USD", "SEK"],
        "non_usd_flag": [False, False, True],
        "exchange_suffix": ["", "", ".ST"],
    })
    prices = pd.DataFrame({
        "window_id": ["buy2019"] * 5,
        "source": ["WIN", "WIN", "LOSE", "LOSE", "GONE"],
        "leg": ["buy", "eval", "buy", "eval", "buy"],
        "date_actual": ["2019-12-31", "2022-12-30",
                        "2019-12-31", "2022-12-30", "2019-12-31"],
        "adjClose": [100.0, 200.0, 100.0, 105.0, 50.0],  # GONE has no eval leg
    })
    # Benchmark: +33.1% over the window.
    bench = pd.DataFrame({
        "date": ["2019-12-31", "2022-12-30"],
        "level": [100.0, 133.1],
    })
    sel.to_csv(os.path.join(tmp, "selections.csv"), index=False)
    prices.to_csv(os.path.join(tmp, "prices.csv"), index=False)
    bench.to_csv(os.path.join(tmp, "benchmark.csv"), index=False)
    import json
    manifest = {k: "test" for k in __import__("bundle_spec").MANIFEST_KEYS}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f)


def test_beat_rate_and_missing_policy():
    import bundle_spec
    with tempfile.TemporaryDirectory() as tmp:
        _write_bundle(tmp)
        ok, problems = bundle_spec.validate_bundle(tmp)
        assert ok, problems

        # WIN: +100% vs bench +33.1% -> excess ~66.9pp >= 10pp -> beat
        # LOSE: +5% vs +33.1% -> excess -28pp -> not beat
        # GONE: no eval price -> missing policy
        res = br.compute_beat_rate(tmp, threshold=0.10, missing="fail")
        # denominator: WIN,LOSE,GONE all have buy prices -> 3 evaluated
        # beats: WIN only -> 1/3
        assert abs(res["pooled_beat_rate"] - 1.0 / 3.0) < 1e-9, res["pooled_beat_rate"]

        res_drop = br.compute_beat_rate(tmp, threshold=0.10, missing="drop")
        # GONE dropped -> denominator 2, beats WIN -> 1/2
        assert abs(res_drop["pooled_beat_rate"] - 0.5) < 1e-9, res_drop["pooled_beat_rate"]

        # USD-only excludes GONE(SEK): WIN,LOSE -> 1/2
        assert abs(res["usd_only_beat_rate"] - 0.5) < 1e-9, res["usd_only_beat_rate"]
        assert res["n_non_usd"] == 1
    print("PASS beat_rate + missing policy + non-USD split + bundle validate")


def test_real_pickle_smoke_optional():
    """If a Boresults pickle is present, confirm dedup runs and collapses
    known duplicate lines on the real top-100. Skipped if absent."""
    import glob
    import pytest
    cands = glob.glob("Boresults_dic-*2026-01-09*.pickle")
    if not cands:
        # EXPLICIT SKIP, NOT a bare `return` (audit C4, fixed 2026-07-31).  Under pytest a
        # `return` from a test body is a PASS, so this reported GREEN having asserted NOTHING
        # whenever the artifact was absent -- and absent is the normal state on the machine
        # that runs the fetch.  A ship gate that turns itself off silently is worse than one
        # that is missing, because the green is read as coverage.
        pytest.skip("no Boresults 2026-01-09 pickle here -- real-pickle smoke NOT run")
    d = pd.read_pickle(cands[0])
    cdx = d["cdx_df"]
    top100 = list(d["BoS_dftop100"]["source"])
    sub = cdx[cdx["source"].isin(top100)]
    kept, dropped, _ = ud.dedup_universe(sub)
    assert 0 < len(kept) < 100, (len(kept), len(dropped))
    if "GSL-PB" in top100 and "GSL" in top100:
        assert dropped.get("GSL-PB") == "GSL", dropped.get("GSL-PB")
    print(f"PASS real-pickle smoke (kept {len(kept)}, dropped {len(dropped)})")


if __name__ == "__main__":
    import pytest as _pytest
    _n_skipped = 0
    for _fn in (test_dedup_collapses_preferred_and_keeps_distinct,
                test_dedup_dualclass_toggle,
                test_benchmark_window_return,
                test_beat_rate_and_missing_policy,
                test_real_pickle_smoke_optional):
        # pytest's Skipped does NOT derive from Exception, so a plain call would abort the
        # script rather than skip one check.  Catch it explicitly and COUNT it, so script mode
        # reports the same honest "not everything ran" the pytest run now does.
        try:
            _fn()
        except _pytest.skip.Exception as _s:
            _n_skipped += 1
            print(f"SKIP {_fn.__name__}: {_s}")
    print("\nALL OFFLINE TESTS PASSED"
          + (f" ({_n_skipped} SKIPPED -- NOT a pass)" if _n_skipped else ""))
