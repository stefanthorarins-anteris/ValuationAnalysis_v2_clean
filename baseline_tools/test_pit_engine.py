"""
Offline tests for the point-in-time / survivorship-safe engine (Build Task #1 +
the buildable build-blockers).  SYNTHETIC fixtures, deterministic, NO network.

Run:  python baseline_tools/test_pit_engine.py

Covers the design s12 reviewer checklist entry points:
  (1) as_of=None leaves data untouched (live invariant)      -> test_as_of_none_identity
  (2) PIT availability-date slice, off-by-one at D           -> test_availability_slice_L1
  (4) fresh DCF: (1+k)**t discounting, per-share, NaN rules   -> test_dcf_*
  (5) beta window / entity-bounded prices                    -> (verify_dcf_beta.beta_recovery)
  (3) entity-life slicing + split at both thresholds (BBBY)  -> test_entity_split_*
  MED#7 NaN-not-0                                            -> test_med7_nan_propagates
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import dcf as dcf_mod
import pit_slice as ps
import entity_id as eid
import score_missing as sm
import dcf_to_price as dtp
import universe_pit as up


# --------------------------------------------------------------------------- #
def _panel(n=16, fcf0=100e6, growth=0.02, shares=100e6, start="2019-03-31"):
    dates = pd.date_range(start, periods=n, freq="QE")
    fcf = fcf0 * (1 + growth) ** np.arange(n)
    rev = 5 * fcf
    return pd.DataFrame({"date": dates, "freeCashFlow": fcf, "revenue": rev,
                         "weightedAverageShsOut": shares})


def test_as_of_none_identity():
    p = _panel()
    out = ps.slice_panel_as_of(p, D=None)
    assert out is p, "as_of=None must return the SAME object untouched"
    print("PASS as_of=None identity (live invariant)")


def test_availability_slice_L1():
    # quarter-START stamp 2020-01-01 represents Q1-2020; period_end=2020-03-31;
    # +90d availability ~ 2020-06-29.  A D just before that must NOT admit the row.
    p = pd.DataFrame({
        "date": pd.to_datetime(["2019-10-01", "2020-01-01"]),
        "freeCashFlow": [1.0, 2.0], "revenue": [5.0, 6.0],
        "weightedAverageShsOut": [10.0, 10.0],
    })
    avail = ps.availability_date(p)
    # 2020-01-01 (Q1) end_time = 2020-03-31, +90d = ~2020-06-29
    assert avail.iloc[1] > pd.Timestamp("2020-06-01"), avail.iloc[1]
    # D in April 2020: naive date<=D would admit the Q1 row 2mo early; availability
    # slice must drop it.
    kept = ps.slice_panel_as_of(p, D="2020-04-15")
    assert len(kept) == 1 and kept["date"].iloc[0] == pd.Timestamp("2019-10-01")
    # D after availability: row admitted
    kept2 = ps.slice_panel_as_of(p, D="2020-07-15")
    assert len(kept2) == 2
    print("PASS availability-date slice (L1, no ~5mo lookahead)")


def test_dcf_discounting_correct():
    """Confirm (1+k)**t discounting, not k**t (the old BoDCF bug), by hand-check on a
    FLAT-FCF, zero-growth-clamped case."""
    # flat FCF, flat revenue -> g0 clamped; use enough history.
    p = _panel(n=20, fcf0=100e6, growth=0.0, shares=100e6)
    # revenue flat -> CAGR 0 -> within clamp
    val, info = dcf_mod.fair_value_per_share(p, D=None, beta=1.0, rf=0.04)
    assert info["reason"] == "ok", info
    # k_e = 0.04 + 1.0*0.05 = 0.09 ; fcf0 = 100e6 ; g fades 0 -> 0.025.
    # Recompute independently with (1+k)**t and compare.
    k = 0.09
    g0 = 0.0
    gt = dcf_mod.G_TERMINAL
    H = dcf_mod.HORIZON
    fcf0 = 4 * 100e6   # base FCF0 = MEDIAN of last-8 TTM (4-quarter sum) values
    gpath = [g0 + (gt - g0) * (t - 1) / (H - 1) for t in range(1, H + 1)]
    pv, fcf_t = 0.0, fcf0
    for t in range(1, H + 1):
        fcf_t *= (1 + gpath[t - 1])
        pv += fcf_t / (1 + k) ** t
    tv = fcf_t * (1 + gt) / (k - gt)
    pv += tv / (1 + k) ** H
    expected = pv / 100e6
    assert abs(val - expected) < 1e-6, (val, expected)
    # sanity: discounting by k**t (the bug) would give a WILDLY different (larger)
    # number -> confirm we are nowhere near it.
    pv_bug, fcf_t = 0.0, fcf0
    for t in range(1, H + 1):
        fcf_t *= (1 + gpath[t - 1])
        pv_bug += fcf_t / k ** t
    bug_val = pv_bug / 100e6
    assert abs(val - bug_val) > abs(val), "must not match the WACC**t bug"
    print(f"PASS DCF (1+k)**t discounting (value/sh={val:.4f}, TV frac={info['tv_fraction']:.2f})")


def test_dcf_nan_rules():
    # negative FCF0 -> NaN
    p_neg = _panel(n=16, fcf0=-50e6, growth=0.0)
    v, i = dcf_mod.fair_value_per_share(p_neg, beta=1.0)
    assert np.isnan(v) and i["reason"] == "nonpositive_fcf0", i
    # short history -> NaN
    p_short = _panel(n=8)
    v, i = dcf_mod.fair_value_per_share(p_short, beta=1.0)
    assert np.isnan(v) and i["reason"] == "insufficient_history", i
    # financial sector -> NaN
    v, i = dcf_mod.fair_value_per_share(_panel(), beta=1.0, sector="Banking")
    assert np.isnan(v) and i["reason"] == "excluded_sector", i
    # no beta -> NaN (never the old 1.0 fallback on the DCF path)
    v, i = dcf_mod.fair_value_per_share(_panel(), beta=np.nan)
    assert np.isnan(v) and i["reason"] == "no_beta", i
    # g_terminal >= k_e (tiny rf, tiny beta) -> NaN
    v, i = dcf_mod.fair_value_per_share(_panel(), beta=0.0, rf=0.0)
    assert np.isnan(v) and i["reason"] == "g_terminal_ge_wacc", i
    # k_e barely EXCEEDS g_terminal (near-singular TV) -> NaN, contained at source
    # (LOW-3).  beta=-0.24, rf=0.04 -> k_e=0.028, spread=0.003 < MIN_KE_SPREAD=0.01.
    v, i = dcf_mod.fair_value_per_share(_panel(), beta=-0.24, rf=0.04)
    assert np.isnan(v) and i["reason"] == "ke_spread_too_small", i
    print("PASS DCF NaN rules (FCF0<=0, short, financial, no-beta, g>=k, k~=g)")


def test_entity_split_recycled_ticker():
    """BBBY-style: a dead entity + a same-symbol successor with a corroborated price
    gap MUST split; the live occupant keeps the bare symbol (as_of=None unchanged)."""
    records = [
        {"symbol": "BBBY", "ipoDate": "1992-06-01", "delistedDate": "2023-05-01",
         "companyName": "Bed Bath & Beyond", "is_live": False},
        {"symbol": "BBBY", "ipoDate": "2024-02-01", "delistedDate": None,
         "companyName": "New Beyond Inc", "is_live": True,
         "has_price_gap": True},
    ]
    merged = eid.assign_entity_ids(records, mode="merge")
    ids = {r["companyName"]: r["entity_id"] for r in merged}
    assert ids["New Beyond Inc"] == "BBBY", ids   # live keeps bare symbol
    assert ids["Bed Bath & Beyond"] == "BBBY_2", ids
    # overlapping lifespans -> MERGE (administrative duplicate), never split
    overlap = [
        {"symbol": "DUP", "ipoDate": "2000-01-01", "delistedDate": "2020-01-01",
         "companyName": "Dup Co", "is_live": False},
        {"symbol": "DUP", "ipoDate": "2010-01-01", "delistedDate": None,
         "companyName": "Dup Co", "is_live": True},
    ]
    mm = eid.assign_entity_ids(overlap, mode="merge")
    assert len({r["entity_id"] for r in mm}) == 1, "overlap must merge"
    print("PASS entity split (recycled ticker) + merge-on-overlap")


def test_entity_split_death_count_band():
    """F4: non-overlap WITHOUT a corroborant -> merge-first keeps 1 entity (scoring),
    split-first makes 2 (death-count floor).  Different precision for the two uses."""
    recs = [
        {"symbol": "X", "ipoDate": "2005-01-01", "delistedDate": "2018-01-01",
         "companyName": "X Corp", "is_live": False},
        {"symbol": "X", "ipoDate": "2019-01-01", "delistedDate": None,
         "companyName": "X Corp", "is_live": True},   # same name, no price gap flag
    ]
    n_merge = len({r["entity_id"] for r in eid.assign_entity_ids(recs, mode="merge")})
    n_split = len({r["entity_id"] for r in eid.assign_entity_ids(recs, mode="split")})
    assert n_merge == 1 and n_split == 2, (n_merge, n_split)
    print("PASS F4 death-rate band (merge=ceiling, split=floor)")


def test_alive_as_of():
    e = {"ipoDate": "2010-01-01", "delistedDate": "2020-01-01"}
    assert eid.alive_as_of(e, "2015-06-01")
    assert not eid.alive_as_of(e, "2005-01-01")   # not yet IPO'd
    assert not eid.alive_as_of(e, "2021-01-01")   # already dead
    live = {"ipoDate": "2010-01-01", "delistedDate": None}
    assert eid.alive_as_of(live, "2026-01-01")
    # LOW-1: unknown ipoDate is NOT assumed alive at an arbitrary D by default
    # (precision-first, no lookahead); only when the caller opts in explicitly.
    unk = {"ipoDate": None, "delistedDate": "2020-01-01"}
    assert not eid.alive_as_of(unk, "2015-01-01")
    assert eid.alive_as_of(unk, "2015-01-01", unknown_ipo_alive=True)
    assert not eid.alive_as_of(unk, "2021-01-01", unknown_ipo_alive=True)  # dead by D
    print("PASS alive_as_of predicate (+ unknown-ipoDate tightening, LOW-1)")


def test_med7_nan_propagates():
    """MED#7: a missing metric must NOT be neutralised to 0/mean.  reliability_shrink
    keeps it missing and SHRINKS the composite; legacy_neutralize reproduces the old
    bug (fills 0)."""
    z = pd.DataFrame({
        "source": ["A", "B"],
        "m1": [1.0, 1.0],
        "m2": [1.0, np.nan],   # B is missing m2
        "m3": [1.0, 1.0],
    })
    shrunk = sm.reliability_shrink(z, weights={"m1": 1, "m2": 1, "m3": 1})
    legacy = sm.legacy_neutralize(z)
    # legacy: B's m2 -> 0, sum = 2.0 ; A = 3.0
    assert abs(legacy.iloc[1] - 2.0) < 1e-9, legacy.iloc[1]
    # shrunk: B present-sum = 2.0, d=1/3, phi=1-(0.5)*(1/3)**2=0.9444, S=1.888..
    assert shrunk.iloc[1] < 2.0, "missing must SHRINK, not just drop to present-sum"
    assert shrunk.iloc[0] == 3.0 * (1.0 - 0.5 * 0.0), "full name unshrunk"
    # MEDIUM-3: legacy_neutralize must apply per-metric weights to faithfully
    # reproduce the live AggScore (postBoRank.py:528-535).  Unweighted != live.
    weights = {"m1": 1.0, "m2": 0.35, "m3": 1.0}
    legacy_w = sm.legacy_neutralize(z, weights=weights)
    # A: 1*1 + 1*0.35 + 1*1 = 2.35 ; B (m2 filled 0): 1*1 + 0*0.35 + 1*1 = 2.0
    assert abs(legacy_w.iloc[0] - 2.35) < 1e-9, legacy_w.iloc[0]
    assert abs(legacy_w.iloc[1] - 2.0) < 1e-9, legacy_w.iloc[1]
    # weighting genuinely changes A vs the unweighted sum (3.0) -> not equivalent
    assert abs(legacy_w.iloc[0] - legacy.iloc[0]) > 1e-9, "weights must matter"
    print(f"PASS MED#7 NaN propagates (legacy B={legacy.iloc[1]:.3f} vs "
          f"shrunk B={shrunk.iloc[1]:.3f}); legacy weighted A={legacy_w.iloc[0]:.3f}")


def test_build_universe_union():
    """MEDIUM-1: build_universe must UNION live survivors with the registry entities
    alive_as_of D.  A delisted-only registry must NOT drop the survivors (reverse
    survivorship), and a name dead-before-D must NOT appear at D."""
    # (1) as_of=None -> bare live symbols, untouched
    assert up.build_universe(["MSFT", "AAPL"], as_of=None) == ["AAPL", "MSFT"]

    # (2) as_of=D with a DELISTED-ONLY registry (survivors live only in live_symbols):
    #     LATECO delisted AFTER D  -> alive at D (a survivor of the historical universe);
    #     DEADCO delisted BEFORE D -> excluded.
    live = ["AAPL", "MSFT"]
    registry = pd.DataFrame([
        {"entity_id": "DEADCO", "symbol": "DEADCO",
         "ipoDate": "2000-01-01", "delistedDate": "2020-01-01"},   # dead before D
        {"entity_id": "LATECO", "symbol": "LATECO",
         "ipoDate": "2005-01-01", "delistedDate": "2024-06-01"},   # delisted after D
    ])
    u = up.build_universe(live, registry=registry, as_of="2023-01-27")
    assert set(u) == {"AAPL", "MSFT", "LATECO"}, u   # survivors kept, dead-before-D dropped
    assert "DEADCO" not in u

    # (3) a live occupant that positively IPO'd AFTER D is excluded (no lookahead)
    live_df = pd.DataFrame([{"symbol": "NEWLY", "ipoDate": "2025-01-01"}])
    assert up.build_universe(live_df, as_of="2023-01-27") == []
    print("PASS build_universe union (MEDIUM-1: survivors kept, dead-before-D + "
          "IPO-after-D excluded)")


def test_dcf_to_price_nan_and_basis():
    # NaN numerator -> NaN ratio (never 0)
    assert np.isnan(dtp.dcf_to_price(np.nan, 10.0))
    # non-positive price -> NaN
    assert np.isnan(dtp.dcf_to_price(20.0, 0.0))
    # ok
    assert abs(dtp.dcf_to_price(20.0, 10.0) - 2.0) < 1e-9
    # basis selection: raw close preferred for the denominator
    px = pd.DataFrame({"date": ["2023-01-01", "2023-06-01"],
                       "close": [10.0, 12.0], "adjClose": [5.0, 6.0]})
    assert dtp.as_of_price(px, D="2023-06-30", basis="raw") == 12.0
    assert dtp.as_of_price(px, D="2023-06-30", basis="adjclose") == 6.0
    print("PASS DcfToPrice NaN + split-basis (S3)")


def _synthetic_bometric(n_q=12):
    """A tiny deterministic BoMetric_df-like frame for two symbols, newest-first,
    with the columns calcScore.simpleScore_fromDict reads via createDicts."""
    import createDicts as cdic
    _, calc, base, mean, diff, unity, special = cdic.getDicts()
    # collect every metric-column name simpleScore will index
    cols = set()
    for k in base:
        cols.add(k)
    for k in mean:
        cols.add("m" + k[0].upper() + k[1:])
    for k in diff:
        cols.add("d" + k[0].upper() + k[1:])
    for k in unity:
        cols.add("u" + k[0].upper() + k[1:])
    for k in special:
        cols.add(k)
    dates = pd.date_range("2020-03-31", periods=n_q, freq="QE")[::-1]  # newest first
    rows = []
    for si, sym in enumerate(["AAA", "BBB"]):
        for qi, dt in enumerate(dates):
            r = {"date": dt, "source": sym}
            for j, c in enumerate(sorted(cols)):
                # deterministic, symbol- and quarter-varying, both signs present
                r[c] = float(((si + 1) * (j + 1) - qi) % 7 - 3)
            rows.append(r)
    return pd.DataFrame(rows)


def test_as_of_none_scoring_invariant():
    """as_of=None through simpleScore_fromDict must be a strict no-op: identical to
    calling it with no as_of at all (the live default), on real synthetic data.
    Guards the plumbing from ever leaking a change onto the live path."""
    import calcScore as cs
    import createDicts as cdic
    bm = _synthetic_bometric()
    _, calc, base, mean, diff, unity, special = cdic.getDicts()
    # build a BoMetric_ave the mean/unity tiers can subtract (median over panel)
    num = bm.drop(columns=["source"]).select_dtypes(include=[float, int])
    ave = num.median(numeric_only=True)
    da = pd.DataFrame()
    a = cs.simpleScore_fromDict(bm.copy(), ave, da, 8)               # live default
    b = cs.simpleScore_fromDict(bm.copy(), ave, da, 8, as_of=None)   # explicit None
    a = a.sort_values("source").reset_index(drop=True)
    b = b.sort_values("source").reset_index(drop=True)
    assert list(a["source"]) == list(b["source"])
    assert np.array_equal(a["score"].astype(float).to_numpy(),
                          b["score"].astype(float).to_numpy())
    print("PASS as_of=None scoring invariant (explicit None == live default)")


def test_fillprereq_date_join_identity_and_ragged():
    """R-E date-join: (A) on aligned, equal-length, dup-free statements it is
    BIT-FOR-BIT identical to the old positional assignment; (B) on a ragged
    statement (a missing middle quarter) it pairs by DATE (no positional shift);
    (C) duplicate dates fall back to positional (never worse than today)."""
    import getData_fmp as gdf
    import createDicts as cdic
    preReq, *_ = cdic.getDicts()
    dates = pd.date_range("2020-03-31", periods=8, freq="QE")

    def _stmt(keys, dts, scale):
        d = {"date": list(dts)}
        for j, k in enumerate(keys):
            d[k] = [float((j + 1) * scale + i) for i in range(len(dts))]
        return pd.DataFrame(d)

    bs = _stmt(preReq["bs"], dates, 10)
    inc = _stmt(preReq["inc"], dates, 20)
    cf = _stmt(preReq["cf"], dates, 30)
    km = _stmt(preReq["km"], dates, 40)
    fr = _stmt(preReq["fr"] + ["priceEarningsRatio"], dates, 50)

    # (A) identity on aligned data: date-join value == positional, for ALL FIVE
    # statements (LOW-A: the prior test only asserted bs).
    aligned, used = gdf._align_statements_by_date(bs, inc, cf, km, fr)
    assert used is True
    for nm, raw in (("bs", bs), ("inc", inc), ("cf", cf), ("km", km), ("fr", fr)):
        for i in preReq.get(nm, []):
            assert np.array_equal(aligned[nm][i].to_numpy(), raw[i].to_numpy()), (nm, i)

    # (A2) fillPreReqdf-level tempfund identity: date-join path == positional path on
    # well-formed data (hardens the safety-net invariant beyond the column check).
    tf_cols = []
    for v in preReq.values():
        tf_cols += list(v)
    tf_cols += ["date", "source", "price"]
    tf_join, _ = gdf.fillPreReqdf(pd.DataFrame({"date": dates, "source": "T"}).reindex(
        columns=list(dict.fromkeys(tf_cols))), preReq, bs, inc, cf, km, fr)
    # recompute the same via a positional (dup-date) fallback frame and compare the
    # shared prereq columns
    # `grahamNumber` is EXCLUDED from the pass-through identity check: since 2026-07-25 it
    # is RECOMPUTED in-pipeline as sqrt(22.5 * EPS_ttm * bookValuePerShare) rather than
    # taken from FMP's km payload (review H2 -- FMP's quarterly value is half the published
    # one).  Asserting pass-through for it was an OBSOLETE assertion that aborted this test
    # BEFORE its ragged-statement and duplicate-date assertions ran, which is how the price
    # length-assignment fetch-killer (ship-gate B1) reached review with no live test
    # coverage of fillPreReqdf's fallback path.  It is asserted separately below.
    _RECOMPUTED = {"grahamNumber"}
    for nm in ("bs", "inc", "cf", "km", "fr"):
        for i in preReq.get(nm, []):
            if i in _RECOMPUTED:
                continue
            assert np.array_equal(
                pd.to_numeric(tf_join[i], errors="coerce").to_numpy(),
                pd.to_numeric({"bs": bs, "inc": inc, "cf": cf, "km": km, "fr": fr}[nm][i],
                              errors="coerce").to_numpy()), (nm, i)

    # (A2b) grahamNumber is COMPUTED, not passed through: it must differ from FMP's km
    # column and must equal sqrt(22.5 * EPS_ttm * BVPS) built from the same frame.
    _g_out = pd.to_numeric(tf_join["grahamNumber"], errors="coerce").to_numpy()
    _g_fmp = pd.to_numeric(km["grahamNumber"], errors="coerce").to_numpy()
    assert not np.array_equal(_g_out, _g_fmp),         "grahamNumber must be recomputed in-pipeline, not passed through from FMP"
    _ni = pd.to_numeric(tf_join["netIncome"], errors="coerce")
    _sh = pd.to_numeric(tf_join["weightedAverageShsOut"], errors="coerce")
    _bv = pd.to_numeric(tf_join["bookValuePerShare"], errors="coerce")
    _eps = _ni.iloc[::-1].rolling(4).sum().iloc[::-1] / _sh
    _exp = np.sqrt(22.5 * _eps.where(_eps > 0) * _bv.where(_bv > 0)).to_numpy()
    assert np.allclose(_g_out, _exp, equal_nan=True), (_g_out[:4], _exp[:4])

    # (A3) REORDERED dates (LOW-A): a statement whose date rows are shuffled relative
    # to bs must still pair BY DATE (reindex), not by position -> identical values in
    # bs's date order.
    inc_shuf = inc.sample(frac=1.0, random_state=1).reset_index(drop=True)
    aligned_s, used_s = gdf._align_statements_by_date(bs, inc_shuf, cf, km, fr)
    assert used_s is True
    col0 = preReq["inc"][0]
    assert np.array_equal(aligned_s["inc"][col0].to_numpy(), inc[col0].to_numpy()), \
        "reordered statement must pair by date, not position"

    # (B) ragged: drop inc's 3rd quarter (index 2).  Positional would shift every
    # later row up by one; the date-join must instead pair by date -> value at the
    # dropped date is NaN, all others unchanged.
    inc_ragged = inc.drop(index=2).reset_index(drop=True)
    aligned2, used2 = gdf._align_statements_by_date(bs, inc_ragged, cf, km, fr)
    assert used2 is True
    col = preReq["inc"][0]
    got = aligned2["inc"][col].to_numpy()
    # date at position 2 (dates[2]) was dropped -> NaN there; positional value inc[col][2]
    assert np.isnan(got[2]), got
    # the row that positional would have mis-slid into position 2 is inc's original
    # index-3 value; the date-join keeps it at position 3 (its true date)
    assert got[3] == inc[col].to_numpy()[3], (got[3], inc[col].to_numpy()[3])

    # (C) duplicate dates -> positional fallback
    bs_dup = bs.copy()
    bs_dup.loc[1, "date"] = bs_dup.loc[0, "date"]
    aligned3, used3 = gdf._align_statements_by_date(bs_dup, inc, cf, km, fr)
    assert used3 is False and aligned3["bs"] is bs_dup
    print("PASS R-E date-join (identity on aligned, date-pairs ragged, dup->fallback)")


def test_parallel_candidate_scores():
    """Item 3: the new scorer runs as a NON-DECISIONAL parallel candidate at an
    as_of date, alongside the (decisional) old scorer; picks come from the OLD
    score, and making the new one decisional is refused."""
    import numpy as _np
    sys.path.insert(0, os.path.join(REPO, "baseline_tools"))
    import new_scorer_bench as nsb
    rng = _np.random.default_rng(7)
    dates = pd.date_range("2018-03-31", periods=24, freq="QE")
    raw_cols = ["netIncome", "weightedAverageShsOut", "freeCashFlow",
                "returnOnAssets", "earningsYield", "grahamNumber", "price",
                "pbRatio", "revenue", "incomeQuality", "returnOnEquity",
                "returnOnCapitalEmployed", "currentRatio", "grossProfitMargin",
                "marketCap", "tangibleBookValuePerShare", "totalAssets",
                "totalLiabilities", "totalCurrentAssets", "totalCurrentLiabilities",
                "totalStockholdersEquity", "operatingIncome", "longTermDebt",
                "netCashProvidedByOperatingActivities",
                #  2026-08-06: the raw legs of `interestCoverage` and `navPerShareGrowth`.
                #  `rng.normal(base*100, 20)` is drawn strictly positive below, so both clear
                #  their guards (interestExpense > 0, bookValuePerShare > 0) and the channels
                #  are actually EXERCISED here rather than being all-NaN and silently absent
                #  from the composite.
                "interestExpense", "bookValuePerShare"]
    rows = []
    for si in range(8):
        base = 1.0 + si
        for dt in dates:
            r = {"date": dt, "source": f"S{si}"}
            for c in raw_cols:
                r[c] = float(abs(rng.normal(base * 100, 20)) + 1)
            rows.append(r)
    cdx = pd.DataFrame(rows)
    panel = nsb.augmented_panel(cdx)
    D = "2023-12-31"
    out = nsb.parallel_candidate_scores(panel, D, topn=3)
    # both scores present, side by side
    assert "S_decisional" in out and "S_candidate_new" in out
    # picks are exactly the top-3 by the DECISIONAL score, never the candidate
    picks = set(out.index[out["is_pick"]])
    top3_dec = set(out["S_decisional"].dropna().sort_values(ascending=False).head(3).index)
    assert picks == top3_dec, (picks, top3_dec)
    assert out.attrs["decisional"] == "old"
    # refuses to make the candidate decisional
    try:
        nsb.parallel_candidate_scores(panel, D, decisional="new")
        raise AssertionError("should have refused decisional='new'")
    except ValueError:
        pass
    print("PASS parallel candidate scores (old decisional, new candidate-only, "
          "switch refused)")


if __name__ == "__main__":
    test_as_of_none_identity()
    test_availability_slice_L1()
    test_dcf_discounting_correct()
    test_dcf_nan_rules()
    test_entity_split_recycled_ticker()
    test_entity_split_death_count_band()
    test_alive_as_of()
    test_med7_nan_propagates()
    test_build_universe_union()
    test_dcf_to_price_nan_and_basis()
    test_as_of_none_scoring_invariant()
    test_fillprereq_date_join_identity_and_ragged()
    test_parallel_candidate_scores()
    print("\nALL PIT-ENGINE TESTS PASSED")
