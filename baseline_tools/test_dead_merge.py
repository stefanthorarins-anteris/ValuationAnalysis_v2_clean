"""
Offline tests for dead_merge (no network -- synthetic entity + guarded real slice).

Covers the wiring that was implemented + the review's must-fixes:
  * live invariant, both cheap (as_of=None) and REAL (override=None -> top-20
    bit-identical to the pre-merge scoring path, on the real pickle if present)
  * per-entity build -> live schema, quarter-START dated, source == entity_id
  * collision handling (prefer_live skips a source already live)
  * INF-SCRUB PARITY: a zero-denominator inf in a dead row is scrubbed to NaN exactly
    like a live row (gdg.forceNumOnDf), so it cannot corrupt the Stage-1 pool median
  * exchange-matched universe (dead side restricted to the NA1 baseline set)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import createDicts as cdic
import getData_gen as gdg
import dead_merge as dm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_PICKLE = os.path.join(
    REPO, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-01-09_len8106_manelim3692_fails1725.pickle")
DEAD_PICKLE = r"C:/Users/stefanthorarinsson/Documents/HomeGDrive/delisted_out/dead_fundamentals_20260713_104350.pickle"
REGISTRY = r"C:/Users/stefanthorarinsson/Documents/HomeGDrive/delisted_out/delisted_registry.csv"


def _synthetic_entity(symbol="ZZZ", nq=12, start="2015-03-31", zero_mcap_row=None,
                      int_zero_peg=False):
    """A minimal but VALID dead entity: km/fr/inc/bs/cf with every preReq field, real
    period-END dates, priceEarningsRatio present.  `zero_mcap_row` forces marketCap=0
    at that (newest-first) row -> salesToMarketCap = revenue/0 = inf, to exercise the
    inf-scrub (a zero-denominator case correlated with distressed/delisting names).
    `int_zero_peg` stores priceEarningsToGrowthRatio as an INT64 column of exact 0
    (the PMA/MSW crash case: calc_special does 1/PEG before np.where masks it)."""
    preReq = cdic.getDicts()[0]
    dates = pd.date_range(start, periods=nq, freq="QE").strftime("%Y-%m-%d").tolist()
    dates = dates[::-1]  # newest-first, as FMP serves
    rng = np.random.default_rng(0)

    def frame(fields, extra=None):
        df = pd.DataFrame({"date": dates, "symbol": symbol})
        for f in fields:
            df[f] = rng.uniform(1.0, 100.0, size=nq)
        if extra:
            for k, v in extra.items():
                df[k] = v
        return df

    inc = frame(preReq["inc"], extra={
        "fillingDate": dates, "acceptedDate": dates, "filing_date_source": "fixed_lag"})
    inc["netIncome"] = rng.uniform(5.0, 50.0, size=nq)
    inc["weightedAverageShsOut"] = rng.uniform(10.0, 100.0, size=nq)
    bs = frame(preReq["bs"])
    cf = frame(preReq["cf"])
    km = frame(preReq["km"])
    if zero_mcap_row is not None:
        km.loc[zero_mcap_row, "marketCap"] = 0.0
    fr = frame(preReq["fr"], extra={"priceEarningsRatio": rng.uniform(5.0, 25.0, size=nq)})
    if int_zero_peg:
        fr["priceEarningsToGrowthRatio"] = np.zeros(nq, dtype="int64")  # exact int64 0
    return {"km": km, "fr": fr, "inc": inc, "bs": bs, "cf": cf,
            "symbol": symbol, "filing_date_source": "fixed_lag", "short_history": False}


def _mini_dmdic():
    """A tiny survivor dmdic with cdx_df/BoMetric_df columns matching production."""
    preReq = cdic.getDicts()[0]
    cdx_cols = ["date", "source"] + sorted({f for v in preReq.values() for f in v
                                            if f != "price"}) + ["price"]
    bm_cols = ["date", "source", "returnOnAssets", "grahamNetNet", "CFO", "mSalesToMarketCap"]
    cdx = pd.DataFrame(columns=cdx_cols)
    cdx.loc[0] = 0
    cdx["source"] = "LIVEA"
    cdx["date"] = "2019-01-01"
    bm = pd.DataFrame(columns=bm_cols)
    bm.loc[0] = 0
    bm["source"] = "LIVEA"
    bm["date"] = "2019-01-01"
    return {"cdx_df": cdx, "BoMetric_df": bm,
            "Tickers_df": pd.DataFrame({"symbol": ["LIVEA"],
                                        "exchangeShortName": ["NYSE"]})}


def _registry(symbol="ZZZ", entity_id="ZZZ", exchange="NASDAQ",
              ipo="2010-01-01", delisted="2022-06-30"):
    r = pd.DataFrame({"symbol": [symbol], "entity_id": [entity_id],
                      "exchange": [exchange], "ipoDate": [ipo], "delistedDate": [delisted]})
    r["ipoDate"] = pd.to_datetime(r["ipoDate"])
    r["delistedDate"] = pd.to_datetime(r["delistedDate"])
    return r


def test_live_invariant_as_of_none():
    dmdic = _mini_dmdic()
    out, stats = dm.merge_dead_into_dmdic(dmdic, {"ZZZ": _synthetic_entity()},
                                          _registry(), as_of=None)
    assert out is dmdic
    assert stats["merged"] is False


def test_build_produces_live_schema_quarter_start():
    dmdic = _mini_dmdic()
    cdx_cols = list(dmdic["cdx_df"].columns)
    bm_cols = list(dmdic["BoMetric_df"].columns)
    cdx_dead, bm_dead = dm.dead_to_scoring_frames(
        {"ZZZ": _synthetic_entity()}, _registry(), cdx_cols, bm_cols,
        live_sources={"LIVEA"})
    assert list(cdx_dead.columns) == cdx_cols
    assert not cdx_dead.empty
    assert set(cdx_dead["source"]) == {"ZZZ"}
    md = pd.to_datetime(cdx_dead["date"]).dt.day.unique().tolist()
    assert md == [1]  # quarter-START stamp
    assert cdx_dead["price"].notna().any()


def test_collision_prefer_live_skips():
    dmdic = _mini_dmdic()
    cdx_cols = list(dmdic["cdx_df"].columns)
    bm_cols = list(dmdic["BoMetric_df"].columns)
    reg = _registry(symbol="LIVEA", entity_id="LIVEA")
    cdx_dead, _ = dm.dead_to_scoring_frames(
        {"LIVEA": _synthetic_entity(symbol="LIVEA")}, reg, cdx_cols, bm_cols,
        live_sources={"LIVEA"}, collision="prefer_live")
    assert cdx_dead.empty
    assert cdx_dead.attrs["build_stats"]["skipped_collision"] == 1


def test_inf_scrub_parity_with_live():
    """A zero-denominator inf in a dead row must end up NaN, exactly as forceNumOnDf
    (the live post-ingest scrub) turns inf->NaN."""
    dmdic = _mini_dmdic()
    cdx_cols = list(dmdic["cdx_df"].columns)
    bm_cols = list(dmdic["BoMetric_df"].columns)
    ent = _synthetic_entity(zero_mcap_row=0)  # newest quarter has marketCap=0
    _, bm_dead = dm.dead_to_scoring_frames({"ZZZ": ent}, _registry(),
                                           cdx_cols, bm_cols, live_sources=set())
    assert not bm_dead.empty
    num = bm_dead.drop(columns=["date", "source"]).to_numpy(dtype="float64", na_value=np.nan)
    assert not np.isinf(num).any(), \
        "dead frames must be inf-free after fixAfterGetData (live parity)"
    assert bm_dead["mSalesToMarketCap"].isna().any(), \
        "the zero-denominator cell (revenue/0) must be scrubbed to NaN, not survive as inf"
    # parity anchor: the SAME function on a live-like inf frame yields NaN
    live_like = pd.DataFrame({"date": ["2020-01-01"], "source": ["X"],
                              "returnOnAssets": [np.inf]})
    assert gdg.forceNumOnDf(live_like)["returnOnAssets"].isna().all()


def test_floatify_int_and_object_zero_to_float():
    """_floatify coerces BOTH an int64-zero column (PMA/MSW) AND an object column with
    an embedded python int 0 (CDIX) to float64 -- the two real crash cases."""
    df = pd.DataFrame({"date": ["2020-01-01", "2020-04-01"], "symbol": ["X", "X"],
                       "peg_int": np.zeros(2, dtype="int64"),
                       "peg_obj": pd.Series([0.5, 0], dtype="object")})  # python int 0
    assert df["peg_obj"].dtype == object
    out = dm._floatify(df)
    assert str(out["peg_int"].dtype) == "float64" and out["peg_int"].iloc[0] == 0.0
    assert str(out["peg_obj"].dtype) == "float64" and out["peg_obj"].iloc[1] == 0.0
    assert out["date"].dtype == object  # identifier columns untouched


def test_int_zero_peg_builds_without_crash():
    """A dead entity with int64-zero priceEarningsToGrowthRatio (PMA/MSW) must build
    cleanly: int 0 -> 0.0 -> 1/0.0 = inf -> masked by calc_special's np.where."""
    dmdic = _mini_dmdic()
    cdx_cols = list(dmdic["cdx_df"].columns)
    bm_cols = ["date", "source", "PEG"]  # PEG is the calc_special metric at risk
    ent = _synthetic_entity(int_zero_peg=True)
    cdx_dead, bm_dead = dm.dead_to_scoring_frames({"ZZZ": ent}, _registry(),
                                                  cdx_cols, bm_cols, live_sources=set())
    assert not bm_dead.empty and "PEG" in bm_dead.columns
    # the raw statement dtype is float post-build (matches live representation)
    assert str(cdx_dead["priceEarningsToGrowthRatio"].dtype) == "float64"


def test_pit_universe_exchange_matched():
    """Dead names on non-NA1 exchanges must be EXCLUDED; the merged universe = NA1
    live survivors UNION NA1 dead-alive, nothing else."""
    dmdic = _mini_dmdic()  # live survivor LIVEA on NYSE
    reg = pd.DataFrame({
        "symbol": ["DEADNA1", "DEADLSE"], "entity_id": ["DEADNA1", "DEADLSE"],
        "exchange": ["NASDAQ", "LSE"],
        "ipoDate": pd.to_datetime(["2010-01-01", "2010-01-01"]),
        "delistedDate": pd.to_datetime(["2022-06-30", "2022-06-30"])})
    uni = set(dm.pit_universe(dmdic, reg, as_of="2020-12-31"))
    assert "LIVEA" in uni          # NA1 live survivor kept
    assert "DEADNA1" in uni        # NA1 dead-alive included
    assert "DEADLSE" not in uni    # non-NA1 dead EXCLUDED (exchange-matched)


def _merge_content_case(bm_extra_cols, live_values):
    """Merge one synthetic dead entity into a live frame whose BoMetric_df carries
    `bm_extra_cols`, with `live_values` written on the single live row.  Returns the
    SystemExit message, or None when the gate allowed the merge."""
    dmdic = _mini_dmdic()
    bm = dmdic["BoMetric_df"]
    for c in bm_extra_cols:
        bm[c] = live_values.get(c, 0.0)
    dmdic["BoMetric_df"] = bm
    reg = pd.DataFrame({
        "symbol": ["ZZZ"], "entity_id": ["ZZZ"], "exchange": ["NASDAQ"],
        "ipoDate": pd.to_datetime(["2010-01-01"]),
        "delistedDate": pd.to_datetime(["2022-06-30"])})
    try:
        dm.merge_dead_into_dmdic(dmdic, {"ZZZ": _synthetic_entity()}, reg,
                                 as_of="2020-12-31")
    except SystemExit as e:
        return str(e)
    return None


def test_merge_content_gate_scope_is_the_criterion_set():
    """The merge-content gate must refuse on a column the SCORER READS and must NOT
    refuse on a retired one.

    `uIncomeQuality` was retired 2026-07-26 (replaced by `CFOlessEarnings`) but still
    sits in older panels; because the dead side is built by TODAY's code it comes out
    all-NaN there, which the first version of this gate reported as a generation
    mismatch and refused on -- blocking a merge that would have scored FINE, since
    nothing reads the column.  Pin both directions so the scope cannot silently widen
    back to 'every column in the frame'."""
    _b, _m, _d, _u, _s = cdic.getBaseMeanDiffUnitySpecialDicts()
    crit = set(list(_b) + ['m' + k[0].upper() + k[1:] for k in _m]
               + ['d' + k[0].upper() + k[1:] for k in _d]
               + ['u' + k[0].upper() + k[1:] for k in _u] + list(_s))
    assert "uIncomeQuality" not in crit      # retired -> out of scope
    assert "mSalesToMarketCap" in crit       # scored -> in scope

    # (a) RETIRED column all-NaN on the dead side ONLY -> must NOT refuse.
    assert _merge_content_case(["uIncomeQuality"], {"uIncomeQuality": 1.0}) is None

    # (b) SCORED criterion all-NaN on the LIVE side -> MUST refuse, and name it.
    msg = _merge_content_case(["mSalesToMarketCap"], {"mSalesToMarketCap": np.nan})
    assert msg is not None and "mSalesToMarketCap" in msg
    assert "uIncomeQuality" not in msg

    # (c) explicit override proceeds on the same known-invalid basis.
    os.environ["ALLOW_MERGE_CONTENT_MISMATCH"] = "1"
    try:
        assert _merge_content_case(["mSalesToMarketCap"],
                                   {"mSalesToMarketCap": np.nan}) is None
    finally:
        del os.environ["ALLOW_MERGE_CONTENT_MISMATCH"]


@pytest.mark.skipif(not (os.path.exists(REAL_PICKLE) and os.path.exists(DEAD_PICKLE)
                         and os.path.exists(REGISTRY)),
                    reason="real pickles not present (dev-only)")
def test_real_override_none_top20_bit_identical():
    """On the REAL pickle: merging dead names and scoring with universe_override=None
    must reproduce the pre-merge top-20 EXACTLY (dead rows filtered by na1_symbols)."""
    import pickle
    import stage2_pit as s2
    dmdic = pd.read_pickle(REAL_PICKLE)
    with open(DEAD_PICKLE, "rb") as f:
        dead = pickle.load(f)
    reg = dm.load_registry(REGISTRY)
    D = "2020-12-31"
    # SKIP, do not fail, when the real pickle predates the current metric set.  Since
    # 2026-07-26 Stage-1 refuses a panel missing a criterion column (calcScore's schema
    # gate) -- the same posture as the price-basis refusal -- so "bit-identical top-20 on
    # the 07-13 panel" is no longer assertable BY DESIGN.  The test stays valuable for a
    # freshly-fetched panel, so it is skipped with the reason rather than deleted or
    # weakened.
    try:
        base = s2.reproduce_pit_top(dmdic, D, na1_only=True)
    except KeyError as e:
        if "OLDER version of the metric set" in str(e):
            pytest.skip("real pickle predates the current metric set (Stage-1 schema gate): "
                        + str(e)[:120])
        raise
    merged, _ = dm.merge_dead_into_dmdic(dmdic, dead, reg, as_of=D,
                                         entities=list(dead.keys())[:40])
    got = s2.reproduce_pit_top(merged, D, na1_only=True, universe_override=None)
    assert got["top20"] == base["top20"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
