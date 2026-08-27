"""Offline regression tests for the URTH-benchmark-in-price-source wiring + the
real_ic None-eval guard.  Pure Python, sub-second, NO network, NO real price CSVs
mutated (test 1 writes to a pytest tmp path only).

These pin the two bugs that killed the vs-MSCI-World readout on a fresh-fetch machine:
  1. the bulk-fetch allow-list dropped the benchmark ETF (URTH) because it is not a
     name in the stock universe  -> PriceSource.benchmark_series raised 'URTH absent';
  2. real_ic.ic_table divided a missing eval-leg (None) by the buy leg -> TypeError.

AND, since 2026-08-27, THE ALLOW-LIST ITSELF.  Bug 1 was fixed by force-keeping URTH past a
filter that was still wrong for every OTHER name it dropped -- dead companies, whole venues --
which is how the run machine's grid came to price 18 of 40 picks.  The filter is gone from
`run_price_fetch_stage`; test 1b is the general statement, and test 1 is now the special case
of it that happens to have its own history.
"""
import os
import sys

import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_analysis as pa
import returns_core as rc
import real_ic as ric
import model_vs_metric as mvm


# --------------------------------------------------------------------------- #
# Test 1: the fresh bulk-by-date fetch KEEPS the benchmark symbol even though it
#         is absent from tonight's stock universe.  (Verifies the fetch fix
#         without any network -- the market getter is faked.)                  #
# --------------------------------------------------------------------------- #
def test_bulk_fetch_keeps_benchmark(monkeypatch, tmp_path):
    import delisted_ingest as di
    import fetch_prices as fp

    #  PAYLOAD FLOOR LOWERED FOR THIS TEST, DELIBERATELY (2026-08-22).  The fetch path now
    #  rejects a body below `fetch_prices.MIN_PAYLOAD_ROWS` (20,000) because three saved
    #  pulls contained truncated bodies that `if rows:` accepted.  This test's fake market is
    #  five rows, so the floor correctly refuses it and nothing is written -- which has
    #  nothing to do with what the test is about.  THIS TEST IS ABOUT THE ALLOW-LIST: that
    #  the benchmark ETF survives a filter built from the stock universe.  Lowering the floor
    #  keeps the subject intact rather than fabricating 20,000 fake rows to tunnel through a
    #  guard this test is not examining.  (Its subject WAS "the benchmark survives a filter
    #  built from the stock universe"; there is no filter any more, so it is now simply "the
    #  benchmark reaches the grid" -- see test 1b for the rule that replaced the exception.)
    #  The floor has its own coverage in
    #  test_fetch_prices.py and test_pipeline_fetch_guards.py, including that it is measured
    #  on the PAYLOAD rather than on the kept rows -- which is the interaction that could
    #  otherwise hide here.
    monkeypatch.setattr(fp, "MIN_PAYLOAD_ROWS", 1)

    # Stock universe = three ordinary names; the benchmark ETF is deliberately NOT here.
    universe = ["AAA", "BBB", "CCC"]
    assert rc.BENCHMARK_SYMBOL not in universe  # premise: the old filter DID drop it

    resdic = {
        "Tickers_df": pd.DataFrame({"symbol": universe}),
        "cdx_df": pd.DataFrame({"source": universe}),
    }
    configdic = {"api_key": "TESTKEY", "baseurl": "https://example.invalid/api/"}

    # Fake bulk EOD market: the universe names + the benchmark ETF + an off-universe
    # name.  Mirrors safe_get_bulk_csv's list-of-dict return shape.
    market = [{"symbol": s, "adjClose": "100.0"} for s in universe]
    market += [{"symbol": rc.BENCHMARK_SYMBOL, "adjClose": "69.48"},
               {"symbol": "ZZZ_OFFUNIVERSE", "adjClose": "1.0"}]
    monkeypatch.setattr(di, "safe_get_bulk_csv", lambda url, **kw: list(market))

    # Redirect the OUTPUT price CSVs to a tmp path so the real (gitignored) files are
    # never touched; nonexistent tmp paths force the fetch branch (need_main/need_supp).
    main_csv = tmp_path / "real_prices.csv"
    supp_csv = tmp_path / "real_prices_2025.csv"
    monkeypatch.setattr(pa, "_PRICES_CSV", str(main_csv))
    monkeypatch.setattr(pa, "_PRICES_2025_CSV", str(supp_csv))
    # Keep it cheap: one main anchor year.
    monkeypatch.setattr(pa, "_MAIN_PRICE_YEARS", [2018])

    out = pa.run_price_fetch_stage(resdic, configdic, log=lambda *a: None)
    assert out["main"] == str(main_csv) and os.path.exists(main_csv)

    written = pd.read_csv(main_csv)
    syms = set(written["symbol"])
    assert rc.BENCHMARK_SYMBOL in syms, "FIX REGRESSED: benchmark dropped by allow-list"
    assert set(universe) <= syms                      # universe names kept
    #  THIS ASSERTION USED TO READ `not in`, and flipping it IS the 2026-08-27 change rather
    #  than collateral damage.  It pinned the allow-list: a name we had already paid for in
    #  the bulk body was DISCARDED before writing because it was not in tonight's universe.
    #  Every delisted name a survivorship-clean backtest needs is off-universe by
    #  construction, so the old assertion was pinning the coverage defect in place.
    assert "ZZZ_OFFUNIVERSE" in syms                  # off-universe kept: no allow-list


# --------------------------------------------------------------------------- #
# Test 1b: THE RULE, not the exception.  The stage writes what the bulk body    #
#          returned; nothing is dropped for being absent from tonight's         #
#          universe.  FAILS AGAINST HEAD ca71e05, where the allow-list kept     #
#          only Tickers_df/cdx_df members plus the hardcoded benchmark.         #
# --------------------------------------------------------------------------- #
def test_bulk_fetch_KEEPS_names_outside_tonights_universe(monkeypatch, tmp_path):
    """The three classes the allow-list silently deleted, one representative each.

    Chosen because each is a MEASURED loss on the run machine's grid, not a hypothetical:
      * DEAD  -- a company delisted between the anchor and tonight.  It cannot be in
                 tonight's universe, and it is exactly what a survivorship-clean backtest
                 needs a price for.
      * VENUE -- a `.PA` name.  Seven venues (.PA/.KS/.OL/.KQ/.BR/.AS/.LS) were priced at
                 ZERO of the 8 anchors because the 2026-07-17 universe did not reach them.
      * CLASS -- a preferred line, which the universe carve-out removes but which the
                 grader still meets in an old pick list.

    A test that checked only URTH keeps passing under a filter that drops all three -- which
    is precisely how the original benchmark fix shipped while the defect stayed.
    """
    import delisted_ingest as di
    import fetch_prices as fp

    monkeypatch.setattr(fp, "MIN_PAYLOAD_ROWS", 1)      # same reason as test 1

    universe = ["AAA", "BBB", "CCC"]
    off_universe = ["ATRI", "MC.PA", "PBFX-P"]
    assert not (set(off_universe) & set(universe))      # premise: none is scored tonight

    resdic = {
        "Tickers_df": pd.DataFrame({"symbol": universe}),
        "cdx_df": pd.DataFrame({"source": universe}),
    }
    configdic = {"api_key": "TESTKEY", "baseurl": "https://example.invalid/api/"}

    market = [{"symbol": s, "adjClose": "100.0"} for s in universe + off_universe]
    market += [{"symbol": rc.BENCHMARK_SYMBOL, "adjClose": "69.48"}]
    monkeypatch.setattr(di, "safe_get_bulk_csv", lambda url, **kw: list(market))

    main_csv = tmp_path / "real_prices.csv"
    monkeypatch.setattr(pa, "_PRICES_CSV", str(main_csv))
    monkeypatch.setattr(pa, "_PRICES_2025_CSV", str(tmp_path / "real_prices_2025.csv"))
    monkeypatch.setattr(pa, "_MAIN_PRICE_YEARS", [2018])

    pa.run_price_fetch_stage(resdic, configdic, log=lambda *a: None)

    syms = set(pd.read_csv(main_csv)["symbol"].astype(str))
    missing = [s for s in off_universe if s not in syms]
    assert not missing, ("the pipeline fetch is narrowing its own grid again -- these were in "
                         "the paid-for body and were not written: %s" % missing)
    assert set(universe) <= syms                        # and the populations that already survived
    assert rc.BENCHMARK_SYMBOL in syms


# --------------------------------------------------------------------------- #
# Test 1c: a non-price is STILL dropped.  The widening is about WHICH SYMBOLS   #
#          we keep, not about accepting junk rows -- pinned so nobody reads     #
#          "no allow-list" as "no row is ever dropped".                         #
# --------------------------------------------------------------------------- #
def test_a_null_price_is_still_dropped(monkeypatch, tmp_path):
    import delisted_ingest as di
    import fetch_prices as fp

    monkeypatch.setattr(fp, "MIN_PAYLOAD_ROWS", 1)
    resdic = {"Tickers_df": pd.DataFrame({"symbol": ["AAA"]}),
              "cdx_df": pd.DataFrame({"source": ["AAA"]})}
    market = [{"symbol": "AAA", "adjClose": "100.0"},
              {"symbol": "NULLPX", "adjClose": "null"},
              {"symbol": "EMPTYPX", "adjClose": ""},
              {"symbol": "", "adjClose": "5.0"}]
    monkeypatch.setattr(di, "safe_get_bulk_csv", lambda url, **kw: list(market))

    main_csv = tmp_path / "real_prices.csv"
    monkeypatch.setattr(pa, "_PRICES_CSV", str(main_csv))
    monkeypatch.setattr(pa, "_PRICES_2025_CSV", str(tmp_path / "real_prices_2025.csv"))
    monkeypatch.setattr(pa, "_MAIN_PRICE_YEARS", [2018])

    pa.run_price_fetch_stage(
        resdic, {"api_key": "TESTKEY", "baseurl": "https://example.invalid/api/"},
        log=lambda *a: None)

    syms = set(pd.read_csv(main_csv)["symbol"].astype(str))
    assert syms == {"AAA"}, syms


# --------------------------------------------------------------------------- #
# Test 2: PriceSource.benchmark_series resolves (non-empty, no raise) when the   #
#         benchmark rows are present on the anchor grid.                         #
# --------------------------------------------------------------------------- #
def test_benchmark_series_resolves(tmp_path):
    anchors = rc.DEFAULT_ANCHORS[:4]
    rows = [{"date_requested": a, "symbol": rc.BENCHMARK_SYMBOL, "adjClose": 100.0 + i}
            for i, a in enumerate(anchors)]
    rows += [{"date_requested": anchors[0], "symbol": "AAA", "adjClose": 10.0}]
    csv = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    ps = rc.PriceSource(str(csv), anchors=anchors)
    s = ps.benchmark_series()
    assert not s.empty
    assert len(s) == len(anchors)
    assert list(s.index) == [pd.Timestamp(a) for a in anchors]


# --------------------------------------------------------------------------- #
# Test 3: real_ic.ic_table REFUSES a missing eval-date price column.             #
#                                                                               #
# THIS TEST USED TO ASSERT THE EXACT OPPOSITE, and that is worth recording.  It  #
# read "does not raise when the eval-date price column is missing; the real IC   #
# is simply omitted", and it asserted `tbl["IC_real"].isna().all()` as the       #
# CORRECT outcome.  It therefore PINNED THE DEFECT: on the 08-20 and 08-22 runs  #
# every anchor was missing (the module's hardcoded dates were not columns of the #
# grid it was handed), so the whole table came back NaN -- passing this test --   #
# and the stage printed "COMPOSITE IC_real=+nan vs best single (RoA)             #
# IC_real=+nan -> smoking gun DOES NOT hold".  A silently-omitted leg is         #
# indistinguishable downstream from a leg that measured nothing, which is why    #
# the leniency had to go rather than be documented.                             #
# --------------------------------------------------------------------------- #
def test_ic_table_REFUSES_a_missing_eval_column():
    sources = [f"S{i}" for i in range(5)]
    buy, ev = "2018-12-31", "2020-12-28"
    panel = pd.DataFrame({"source": sources, "date": pd.Timestamp(buy)})
    for mm in mvm.METRICS:
        panel[mm] = np.arange(len(sources), dtype=float)
    panel["_price"] = np.arange(1, len(sources) + 1, dtype=float)

    # `real` has the BUY column but NOT the eval column.
    real = pd.DataFrame({buy: np.arange(1, len(sources) + 1, dtype=float)}, index=sources)
    assert ev not in real.columns

    with pytest.raises(KeyError) as e:
        ric.ic_table(panel, real, [(buy, ev)], "t")
    assert ev in str(e.value)
    # the message must point at the fix, not just at the symptom
    assert "resolve_anchors" in str(e.value)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
