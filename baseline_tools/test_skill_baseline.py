"""
Tests for skill_baseline.py -- oracle-best-N + random-N baseline.

Two layers:
  * UNIT tests on the pure helpers with hand-built fixtures (fast, no data): the three
    corrections (universe-match guard, dead-name parity, mean-return contamination),
    determinism, and the oracle/random math.
  * INTEGRATION test on the real local pickle (skipped if the data is absent): asserts
    determinism end-to-end and the oracle >= filter >= random ordering.

No network.  Prices only via returns_core.PriceSource.  No api_key ever printed.
Run:  pytest baseline_tools/test_skill_baseline.py -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import returns_core as rc
import skill_baseline as sb


# --------------------------------------------------------------------------- #
#  A tiny in-memory PriceSource: two anchors, hand-set prices.                #
# --------------------------------------------------------------------------- #
B, E = "2021-12-31", "2024-12-31"   # real date anchors (benchmark_series parses them)


class FakePrices:
    """anchors buy=B, eval=E.  price map {(ticker, anchor): px}; missing -> None."""
    def __init__(self, prices, anchors=(B, E)):
        self.anchors = list(anchors)
        self._idx = {a: i for i, a in enumerate(self.anchors)}
        self._lut = dict(prices)

    def price(self, t, a):
        return self._lut.get((t, a))

    def last_before(self, t, a):
        j = self._idx.get(a)
        if j is None:
            return None
        for k in range(j - 1, -1, -1):
            aa = self.anchors[k]
            if (t, aa) in self._lut:
                return aa, self._lut[(t, aa)]
        return None

    def benchmark_series(self, symbol="URTH"):
        rows = [(a, self._lut[(symbol, a)]) for a in self.anchors if (symbol, a) in self._lut]
        return pd.Series({pd.Timestamp(a): lvl for a, lvl in rows}).sort_index()


# --------------------------------------------------------------------------- #
#  clean_windows                                                              #
# --------------------------------------------------------------------------- #
def test_clean_windows_36mo():
    w = sb.clean_windows(36)
    assert w == [("2021-12-31", "2024-12-31"), ("2022-12-30", "2025-12-31")]


def test_clean_windows_12mo():
    w = sb.clean_windows(12)
    assert w == [("2021-12-31", "2022-12-30"), ("2022-12-30", "2023-12-29"),
                 ("2023-12-29", "2024-12-31"), ("2024-12-31", "2025-12-31")]


def test_clean_windows_excludes_pre_2021():
    # No buy anchor before 2021 appears (universe-degenerate) at any cadence.
    for cad in (12, 36):
        buys = [b for b, _ in sb.clean_windows(cad)]
        assert all(b >= "2021" for b in buys)


# --------------------------------------------------------------------------- #
#  CORRECTION #3: mean-return contamination (drop + winsorize)                #
# --------------------------------------------------------------------------- #
def test_contamination_dropped_from_mean():
    # A near-zero-buy-price artifact of +403,382% (=4033.82) must be dropped; a real
    # basket around +20% must be preserved.
    rets = [0.10, 0.20, 0.30, 4033.82]
    mean_dropped, mean_winsor, n_contam, contam = sb._clean_and_winsor(
        rets, contam_cap=sb.DEFAULT_CONTAM_RETURN_CAP, winsor=sb.DEFAULT_WINSOR)
    assert n_contam == 1
    assert abs(contam[0] - 4033.82) < 1e-9
    assert abs(mean_dropped - 0.20) < 1e-9          # (0.1+0.2+0.3)/3, artifact gone
    assert mean_winsor < 1.0                         # not blown up by the artifact


def test_beat_rate_immune_to_contamination():
    # The +403,382% name is still just ONE beater on the count metric -> beat-rate is not
    # distorted the way the mean is.  (Verified via returns_core directly.)
    rdf = pd.DataFrame({
        "ticker": ["a", "b", "c", "artifact"],
        "buy_adjClose": [100.0, 100.0, 100.0, 0.001],
        "eval_adjClose": [130.0, 90.0, 125.0, 40.34],
        "terminal_adjClose": [130.0, 90.0, 125.0, 40.34],
        "total_return": [0.30, -0.10, 0.25, 40339.0],
        "total_return_floor": [0.30, -0.10, 0.25, 40339.0],
        "terminal_flag": [False, False, False, False],
        "status": ["ok", "ok", "ok", "ok"],
    })
    rate, n = rc.beat_rate(rdf, benchmark_ret=0.05, threshold=0.10, missing="fail")
    # beaters: a (+.30>=.15), c (+.25>=.15), artifact -> 3 of 4
    assert n == 4
    assert abs(rate - 0.75) < 1e-9


# --------------------------------------------------------------------------- #
#  CORRECTION #2: dead-name parity (symmetric missing='fail', variable denom)  #
# --------------------------------------------------------------------------- #
def test_dead_name_parity_missing_fail_symmetric():
    ps = FakePrices({
        ("URTH", B): 100.0, ("URTH", E): 110.0,   # bench +10%
        ("win", B): 100.0, ("win", E): 130.0,     # +30% -> beats by +20pp
        ("lose", B): 100.0, ("lose", E): 105.0,   # +5%  -> misses
        ("dead", B): 100.0,                         # no eval -> terminal, missing=fail
        ("no_buy", E): 50.0,                        # no buy leg -> excluded (both sides)
    })
    names = ["win", "lose", "dead", "no_buy"]
    rdf = rc.compute_returns(names, B, E, ps)
    bench = rc.benchmark_return(ps, B, E)
    # missing='fail': denom excludes no_buy (variable denom = 3), 'dead' counts as a miss.
    flags, rets = sb._name_flags_and_returns(rdf, bench, threshold=0.10, missing="fail")
    assert len(flags) == 3                 # win, lose, dead  (no_buy excluded)
    assert flags.tolist() == [True, False, False]
    rate, n = rc.beat_rate(rdf, bench, threshold=0.10, missing="fail")
    assert n == 3 and abs(rate - 1 / 3) < 1e-9


# --------------------------------------------------------------------------- #
#  Determinism of the random Monte-Carlo (same seed -> identical samples)      #
# --------------------------------------------------------------------------- #
def _toy_ctx():
    """One window, a 'universe' rung of 12 names with a spread of returns."""
    ps_map = {("URTH", B): 100.0, ("URTH", E): 110.0}
    names = [f"n{i}" for i in range(12)]
    # returns from -20% to +90% in 10pp steps
    for i, nm in enumerate(names):
        ps_map[(nm, B)] = 100.0
        ps_map[(nm, E)] = 100.0 * (1 + (-0.20 + 0.10 * i))
    ps = FakePrices(ps_map)
    sets = {"universe": names, "top200": names, "top100": names}
    # a fake filter_res: top20 = first 4, pool = all 12
    res = {"top20": names[:4], "pool_after_norm": names, "stage1_top100": names}
    ctx = [(B, E, res, sets)]
    return ctx, ps


def test_random_rung_deterministic():
    ctx, ps = _toy_ctx()
    kw = dict(pick_n=4, n_draws=200, threshold=0.10, missing="fail",
              contam_cap=sb.DEFAULT_CONTAM_RETURN_CAP, winsor=sb.DEFAULT_WINSOR)
    a = sb._random_rung("universe", ctx, ps, rng=np.random.default_rng(0), **kw)
    b = sb._random_rung("universe", ctx, ps, rng=np.random.default_rng(0), **kw)
    assert a["beat_samples"] == b["beat_samples"]
    assert a["dollar_dropped_samples"] == b["dollar_dropped_samples"]
    # different seed -> (almost surely) different draw sequence
    c = sb._random_rung("universe", ctx, ps, rng=np.random.default_rng(1), **kw)
    assert a["beat_samples"] != c["beat_samples"]


def test_oracle_is_ceiling_over_filter_and_random():
    ctx, ps = _toy_ctx()
    # oracle-best-4 from the 12-name pool = the 4 highest returns; on this toy set
    # (returns up to +90%) all 4 beat bench+10pp -> beat-rate 100%.
    orc = sb._oracle_metrics(ctx, ps, oracle_ns=[4], threshold=0.10, missing="fail",
                             contam_cap=sb.DEFAULT_CONTAM_RETURN_CAP,
                             winsor=sb.DEFAULT_WINSOR)
    assert abs(orc[4]["beat_rate"] - 1.0) < 1e-9
    # oracle dollar ceiling = mean of the 4 highest returns (0.90,0.80,0.70,0.60)=0.75
    assert abs(orc[4]["dollar_return"]["mean_dropped"] - 0.75) < 1e-9
    # random-4 mean beat-rate must be <= the oracle ceiling
    rnd = sb._random_rung("universe", ctx, ps, pick_n=4, n_draws=500, threshold=0.10,
                          missing="fail", rng=np.random.default_rng(0),
                          contam_cap=sb.DEFAULT_CONTAM_RETURN_CAP, winsor=sb.DEFAULT_WINSOR)
    assert rnd["beat_rate"]["mean"] <= orc[4]["beat_rate"] + 1e-9


def test_short_history_eps_guard_is_behavior_preserving():
    # The parallel stage2_metrics refactor now guards <4-row frames at the source, so the
    # opt-in guard must be TRANSPARENT: it returns 0 for a <4-row frame (no crash) and hands
    # >=4-row frames to the original untouched, and it restores the attribute on exit.
    import stage2_metrics as sm
    orig = sm.eps_to_eps_mean
    short = pd.DataFrame({"netIncome": [10.0, 10.0], "weightedAverageShsOut": [5.0, 5.0],
                          "source": ["x", "x"]})
    full = pd.DataFrame({"netIncome": [10.0] * 4, "weightedAverageShsOut": [5.0] * 4})
    with sb.short_history_eps_guard(enabled=True):
        assert sm.eps_to_eps_mean(short) == 0
        assert sm.eps_to_eps_mean(full) == orig(full)   # >=4 rows unchanged
    assert sm.eps_to_eps_mean is orig                   # restored on exit


def test_percentile_of():
    samples = [0.0, 0.1, 0.2, 0.3, 0.4]
    assert sb._percentile_of(0.25, samples) == pytest.approx(60.0)
    assert sb._percentile_of(-1.0, samples) == pytest.approx(0.0)
    assert sb._percentile_of(1.0, samples) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
#  INTEGRATION -- real local data (skipped if absent)                         #
# --------------------------------------------------------------------------- #
_P = sb._default_paths()
_HAVE_DATA = all(os.path.exists(_P[k]) for k in ("pickle", "dead", "registry", "prices"))


@pytest.mark.skipif(not _HAVE_DATA, reason="local pickle/dead/registry/prices absent")
def test_integration_determinism_and_ordering():
    dmdic, merged, registry, ps = sb.load_inputs(
        _P["pickle"], _P["dead"], _P["registry"], _P["prices"], _P["prices_2025"])
    kw = dict(cadence_months=36, pick_n=20, oracle_ns=(3, 20), n_draws=300, seed=0)
    r1 = sb.run_skill_baseline(dmdic, merged, registry, ps, **kw)
    r2 = sb.run_skill_baseline(dmdic, merged, registry, ps, **kw)
    # determinism: identical random samples for the same seed
    assert (r1["random"]["universe"]["beat_samples"]
            == r2["random"]["universe"]["beat_samples"])
    # ordering: oracle >= filter >= random at the common N
    assert r1["sanity"]["ordering_holds"] is True
    # no contaminated raw universe mean is surfaced as the ladder dollar figure
    # (it is the dropped mean by construction); universe rung should be finite.
    uni_dollar = dict(r1["ladder"]["dollar_return"])["universe"]
    assert uni_dollar == uni_dollar and abs(uni_dollar) < sb.DEFAULT_CONTAM_RETURN_CAP
