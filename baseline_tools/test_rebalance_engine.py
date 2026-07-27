"""
Self-checks for the Strategy-B rebalance engine.

HARD GATE: Strategy A == rebalance-engine at k == horizon, tx_cost_bps == 0, must
reproduce the CERTIFIED 36mo top-20 pooled-clean per-name beat-rate = 30.0% (12/40)
over buy2021 + buy2022 -- proving the engine is a faithful superset of the certified
harness.  The certified 30% baseline is the LEGACY default weights, so this gate scores
with getPostDict_legacy() (getPostDict()'s live default is now the promoted mu theory
prior).  Also unit-tests the schedule / turnover / tx-drag / UnevaluableK plumbing with
a stub ranker (no scorer, fast).

Run:  python baseline_tools/test_rebalance_engine.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import pytest

import returns_core as rc
import rebalance_engine as reb

_HOME = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
PICKLE = os.path.join(
    _HOME, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-07-13_len7879_manelim3692_fails1966.pickle")
DEAD = os.path.join(_HOME, "delisted_out", "dead_fundamentals_20260713_104350.pickle")
REGISTRY = os.path.join(_HOME, "delisted_out", "delisted_registry.csv")
PRICES = os.path.join(_HERE, "price_data", "real_prices.csv")
PRICES_2025 = os.path.join(_HERE, "price_data", "real_prices_2025.csv")

CERT_WINDOWS = [("buy2021", "2021-12-31"), ("buy2022", "2022-12-30")]
CERT_PER_WINDOW = {"2021-12-31": (0.25, 20), "2022-12-30": (0.35, 20)}
CERT_POOLED = 12 / 40   # 30.0%


# --------------------------------------------------------------------------- #
#  Unit tests -- stub ranker (fast, no scorer)                                #
# --------------------------------------------------------------------------- #
def _stub_price_source():
    anchors = list(rc.DEFAULT_ANCHORS)
    lut = {}
    rng = np.random.default_rng(7)
    for t in [f"S{i}" for i in range(60)] + ["URTH"]:
        lvl = 100.0
        for a in anchors:
            lut[(t, a)] = lvl
            lvl *= float(1.0 + rng.normal(0.05, 0.2)); lvl = max(lvl, 1.0)
    ps = rc.PriceSource.__new__(rc.PriceSource)
    ps.anchors = anchors; ps._idx = {a: i for i, a in enumerate(anchors)}; ps._lut = lut
    return ps


def test_schedule():
    anchors = list(rc.DEFAULT_ANCHORS)
    # k == horizon -> single sub-period [buy, eval]
    s = reb.rebalance_schedule("2021-12-31", 36, 36, anchors)
    assert s == ["2021-12-31", "2024-12-31"], s
    # k == 12 over 36mo -> 3 sub-periods
    s = reb.rebalance_schedule("2021-12-31", 36, 12, anchors)
    assert s == ["2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31"], s
    # 6mo / 3mo unevaluable on the annual grid
    for k in (6, 3):
        try:
            reb.rebalance_schedule("2021-12-31", 36, k, anchors)
            assert False, f"k={k} should be UnevaluableK on annual grid"
        except reb.UnevaluableK:
            pass
    print("  [ok] schedule + UnevaluableK (annual grid: 3mo/6mo raise)")


def test_frictionless_equals_txzero():
    ps = _stub_price_source()
    uni = [f"S{i}" for i in range(60)]

    def rank_fn(as_of):
        # deterministic but as-of-dependent ordering -> real turnover between periods
        r = (hash(as_of) & 0xFFFF)
        return uni[r % 60:] + uni[:r % 60]

    res0 = reb.evaluate_strategy("2021-12-31", 36, 12, rank_fn, ps, N=20,
                                 tx_cost_bps=0.0)
    resx = reb.evaluate_strategy("2021-12-31", 36, 12, rank_fn, ps, N=20,
                                 tx_cost_bps=50.0)
    # at tx=0, net == frictionless exactly
    assert abs(res0["basket_return_net"] - res0["basket_return_frictionless"]) < 1e-12
    # tx>0 must DRAG the net return below frictionless when turnover>0
    assert resx["turnover_oneway_total"] > 0
    assert resx["basket_return_net"] < resx["basket_return_frictionless"] - 1e-9
    # frictionless is identical regardless of tx setting
    assert abs(res0["basket_return_frictionless"]
               - resx["basket_return_frictionless"]) < 1e-12
    print("  [ok] tx=0 net==frictionless; tx>0 drags net below frictionless")


def test_beatrate_via_primitive():
    """window_beat_rate must equal a direct rc.beat_rate on the same single-period
    returns table (k==H), proving we route through the certified primitive."""
    ps = _stub_price_source()
    uni = [f"S{i}" for i in range(60)]
    rank_fn = lambda as_of: uni
    res = reb.evaluate_strategy("2021-12-31", 36, 36, rank_fn, ps, N=20,
                                tx_cost_bps=0.0)
    rate_engine, n_engine = reb.window_beat_rate(res, use_net=False)
    # direct certified path over the same top-20
    rdf = rc.compute_returns(uni[:20], "2021-12-31", "2024-12-31", ps)
    bench = rc.benchmark_return(ps, "2021-12-31", "2024-12-31")
    rate_direct, n_direct = rc.beat_rate(rdf, bench, threshold=0.10, missing="fail")
    assert abs(rate_engine - rate_direct) < 1e-12, (rate_engine, rate_direct)
    assert n_engine == n_direct, (n_engine, n_direct)
    print(f"  [ok] window_beat_rate == direct rc.beat_rate ({rate_engine:.3f})")


# --------------------------------------------------------------------------- #
#  HARD GATE -- real ranker reproduces the certified 30%                      #
# --------------------------------------------------------------------------- #
def test_certified_reproduction():
    for p in (PICKLE, DEAD, REGISTRY, PRICES, PRICES_2025):
        if not os.path.exists(p):
            print(f"  [SKIP] certified reproduction -- missing input: {p}")
            return
    import dead_merge as dm
    import stage2_pit as s2
    import createDicts as cdic

    print("  loading pickle + dead-merge (this is the slow part) ...")
    dmdic = pd.read_pickle(PICKLE)
    dead = pd.read_pickle(DEAD)
    registry = dm.load_registry(REGISTRY)
    # SKIP, do not fail, when the on-disk panel predates the current metric set.
    # Since 2026-07-26/27 three gates refuse an old-generation panel BY DESIGN -- the
    # Stage-1 schema gate (a missing criterion column), the price-basis check, and the
    # dead/live merge-content check.  A "certified 30% reproduction" measured on the
    # 2026-07-13 panel is therefore no longer assertable: its live rows carry the OLD
    # price/Graham basis and lack the new Stage-1 columns.  The test stays valuable the
    # moment a panel is fetched with the current code, so it is skipped WITH THE REASON
    # rather than deleted, weakened, or left red.
    try:
        merged, _ = dm.merge_dead_into_dmdic(dmdic, dead, registry, as_of="2018-12-31")
    except SystemExit as e:
        # pytest.skip, not a bare `return`: a test that asserts NOTHING must not report
        # as PASSED (that is how an unasserted gate hides in a green summary).
        pytest.skip("certified reproduction -- dead/live merge refused (panel predates "
                    "the current metric set): %s" % str(e)[:140])
    ps = rc.PriceSource(PRICES, supp_csv=PRICES_2025)

    # The certified 30% baseline was measured under the LEGACY (pre-2026-07-14) default
    # weights.  getPostDict()'s decisional default is now the promoted mu theory prior,
    # so pin this reproduction gate to getPostDict_legacy() -- it must still assert the
    # 30% BASELINE, unchanged.  (Reads pool_after_norm, which is undeduped regardless of
    # the new issuer-dedup, so the 12/40 count is apples-to-apples with certification.)
    _leg_bm, _leg_new = cdic.getPostDict_legacy()
    legacy_weights = {**{k: _leg_bm[k]["w"] for k in _leg_bm},
                      **{k: _leg_new[k]["w"] for k in _leg_new}}

    uni_cache = {}

    def rank_fn(as_of):
        if as_of not in uni_cache:
            uni_cache[as_of] = dm.pit_universe(dmdic, registry, as_of=as_of)
        res = s2.reproduce_pit_top(merged, as_of, universe_override=uni_cache[as_of],
                                   weight_override=legacy_weights)
        return [] if res is None else res["pool_after_norm"]

    results = []
    try:
        _probe = rank_fn(CERT_WINDOWS[0][1])
    except KeyError as e:
        if "OLDER version of the metric set" in str(e):
            pytest.skip("certified reproduction -- Stage-1 schema gate: %s" % str(e)[:140])
        raise
    for wid, buy in CERT_WINDOWS:
        res = reb.evaluate_strategy(buy, 36, 36, rank_fn, ps, N=20, tx_cost_bps=0.0)
        rate, n = reb.window_beat_rate(res, use_net=False)
        exp_rate, exp_n = CERT_PER_WINDOW[buy]
        print(f"  {wid}: engine A(k=H,tx=0) beat-rate={rate:.3f} n={n}  "
              f"(certified {exp_rate:.3f} / {exp_n})")
        assert n == exp_n, f"{wid} n mismatch: {n} != {exp_n}"
        assert abs(rate - exp_rate) < 1e-9, f"{wid} rate mismatch: {rate} != {exp_rate}"
        results.append(res)

    pooled, tot = reb.pooled_beat_rate(results, use_net=False)
    print(f"  POOLED A(k=H,tx=0): {pooled:.4f} ({tot}) vs certified {CERT_POOLED:.4f} (40)")
    assert tot == 40, tot
    assert abs(pooled - CERT_POOLED) < 1e-9, (pooled, CERT_POOLED)
    print("  [ok] DEGENERATE-A reproduces the certified 30.0% (12/40)")


if __name__ == "__main__":
    print("Rebalance-engine self-checks")
    print("- unit tests (stub ranker):")
    test_schedule()
    test_frictionless_equals_txzero()
    test_beatrate_via_primitive()
    print("- HARD GATE (real ranker):")
    test_certified_reproduction()
    print("ALL SELF-CHECKS PASSED")
