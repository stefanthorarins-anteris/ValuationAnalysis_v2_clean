"""Unit tests for the ticker-agnostic returns primitive (returns_core).

Uses a tiny in-memory fake price source -- no pickle, no CSV, no network -- so the
primitive's logic (ok / terminal / no_buy, primary vs floor, derived views) is pinned
independent of the data.  Plus one smoke test against the real PriceSource CSV.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import returns_core as rc


class FakePS:
    """Anchor grid A0<A1<A2<A3; prices given as {(ticker, anchor): price}."""
    def __init__(self, prices, anchors=("A0", "A1", "A2", "A3")):
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
            p = self._lut.get((t, self.anchors[k]))
            if p is not None:
                return self.anchors[k], p
        return None


def test_ok_return():
    ps = FakePS({("X", "A0"): 100.0, ("X", "A2"): 150.0})
    r = rc.compute_returns(["X"], "A0", "A2", ps).iloc[0]
    assert r["status"] == "ok"
    assert not r["terminal_flag"]
    assert math.isclose(r["total_return"], 0.5)
    assert math.isclose(r["total_return_floor"], 0.5)  # equal when eval present


def test_terminal_uses_last_before_eval():
    # eval (A3) missing; last available before A3 is A2 = 120 -> primary = 0.2
    ps = FakePS({("X", "A0"): 100.0, ("X", "A1"): 110.0, ("X", "A2"): 120.0})
    r = rc.compute_returns(["X"], "A0", "A3", ps).iloc[0]
    assert r["status"] == "terminal"
    assert r["terminal_flag"]
    assert math.isclose(r["terminal_adjClose"], 120.0)
    assert math.isclose(r["total_return"], 0.2)      # primary: 120/100 - 1
    assert math.isclose(r["total_return_floor"], -1.0)  # floor: -100%


def test_terminal_falls_back_to_buy_when_only_buy():
    ps = FakePS({("X", "A0"): 100.0})  # only buy leg exists
    r = rc.compute_returns(["X"], "A0", "A3", ps).iloc[0]
    assert r["status"] == "terminal"
    assert math.isclose(r["total_return"], 0.0)      # buy/buy - 1
    assert math.isclose(r["total_return_floor"], -1.0)


def test_no_buy_excluded():
    ps = FakePS({("X", "A2"): 150.0})  # no buy leg
    df = rc.compute_returns(["X"], "A0", "A2", ps)
    assert df.iloc[0]["status"] == "no_buy"
    assert np.isnan(df.iloc[0]["total_return"])
    assert len(rc.included(df)) == 0
    assert np.isnan(rc.average_return(df))


def test_input_order_preserved():
    ps = FakePS({("A", "A0"): 10.0, ("A", "A1"): 11.0,
                 ("B", "A0"): 20.0, ("B", "A1"): 30.0})
    df = rc.compute_returns(["B", "A"], "A0", "A1", ps)
    assert list(df["ticker"]) == ["B", "A"]  # order preserved for top-N slicing


def test_derived_views():
    ps = FakePS({("A", "A0"): 100.0, ("A", "A1"): 120.0,   # +0.20
                 ("B", "A0"): 100.0, ("B", "A1"): 80.0,    # -0.20
                 ("C", "A2"): 5.0})                        # no buy -> excluded
    df = rc.compute_returns(["A", "B", "C"], "A0", "A1", ps)
    assert math.isclose(rc.average_return(df), 0.0)        # (0.2 - 0.2)/2
    c = rc.counts(df)
    assert (c["n_included"], c["n_terminal"], c["n_no_buy"]) == (2, 0, 1)
    # excess vs a 5% benchmark
    assert math.isclose(rc.excess_return(df, 0.05), -0.05)
    # beat-rate: A beats a -10% bench by >=10pp (0.2-(-0.1)=0.30>=0.10 -> True),
    # B does not (-0.2-(-0.1)=-0.10). -> 1/2
    br, n = rc.beat_rate(df, -0.10, threshold=0.10)
    assert n == 2 and math.isclose(br, 0.5)


def test_floor_average():
    ps = FakePS({("A", "A0"): 100.0, ("A", "A1"): 120.0, ("A", "A2"): 130.0,
                 ("B", "A0"): 100.0, ("B", "A1"): 90.0})  # B: eval A3 missing
    df = rc.compute_returns(["A", "B"], "A0", "A3", ps)
    # A: A3 missing -> last_before = A2 (130) -> primary +0.30; floor -1.0
    # B: A3 missing -> last_before = A1 (90)  -> primary -0.10; floor -1.0
    assert math.isclose(rc.average_return(df, floor=False), (0.30 + -0.10) / 2)
    assert math.isclose(rc.average_return(df, floor=True), -1.0)


def test_smoke_real_pricesource():
    import pytest
    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "price_data", "real_prices.csv")
    if not os.path.exists(csv):
        # EXPLICIT SKIP, NOT a bare `return` (audit C4, fixed 2026-07-31).  Under pytest a
        # `return` is a PASS, so this reported green having asserted nothing whenever the price
        # CSV was absent -- the normal state on the machine that runs the fetch.
        pytest.skip("no price_data/real_prices.csv here -- real-pricesource smoke NOT run")
    ps = rc.PriceSource(csv)
    # URTH benchmark must resolve and be positive over a 12mo anchor window.
    b = rc.benchmark_return(ps, "2021-12-31", "2022-12-30")
    assert b == b  # not NaN
    # a well-known survivor should have a real return
    df = rc.compute_returns(["AAPL"], "2021-12-31", "2022-12-30", ps)
    assert df.iloc[0]["status"] in ("ok", "terminal", "no_buy")


if __name__ == "__main__":
    import traceback
    import pytest as _pytest
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    skips = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        # Skipped derives from BaseException, so `except Exception` below would NOT catch it and
        # the script would abort.  It is also NOT a pass, so it gets its own counter.
        except _pytest.skip.Exception as _s:
            skips += 1
            print(f"SKIP {fn.__name__}: {_s}")
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-fails-skips}/{len(fns)} passed"
          + (f", {skips} SKIPPED (not a pass)" if skips else ""))
    sys.exit(1 if fails else 0)
