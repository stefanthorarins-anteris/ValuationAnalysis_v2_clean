"""
The GATED GAP-FILL route `real+derived` (offline; no network).

THE INSTRUCTION IT IMPLEMENTS (CEO, verbatim): "Wire it in where real is empty, but we should
probably adjust if there is a systemic bias in the derived price vs the real price."

WHAT THIS ROUTE IS NOT.  `CompositePriceSource` ('derived+real') prefers the DERIVED leg
wherever the derived leg exists, reassigning ~2,000 names the real file prices perfectly well.
`GapFillPriceSource` ('real+derived') never touches a name the real leg can price.  The two
must not be confused, so `test_gapfill_is_NOT_the_derived_preferred_composite` pins the
difference directly.

THE MEASURED OUTCOME, on the run machine's grid + the 08-22 CUR6K panel (2026-08-22):
1,187 names gap-filled, ~120 REFUSED.  Per venue, cleared / refused:
    .PA 426/13 (97.0%)   .KS 323/0 (100%)   .KQ 126/0 (100%)   .LS 28/0 (100%)
    .BR  82/5  (94.3%)   .AS  78/9 (89.7%)  .OL 117/70 (62.6%)  .L 0/10 (0%)
Every refusal reason but four is `currency_mismatch`.

THE BIAS QUESTION HAS A STRUCTURAL ANSWER, AND IT IS THE ONE THING TO READ BEFORE TRUSTING
ANY NUMBER OFF THIS ROUTE.  The derived-vs-real bias is measurable only on the OVERLAP -- names
BOTH legs price.  This route uses the derived leg ONLY where the real leg is empty, i.e.
exactly OFF that overlap.  The measured population and the used population are DISJOINT, so
NO correction is applied and none could be justified from the table.
`test_the_bias_table_CANNOT_see_the_venues_the_route_actually_uses` pins that as a property of
the code rather than leaving it as a caveat in prose.

WHAT THESE TESTS CANNOT DETECT.  They pin ROUTING -- which leg serves which ticker, and that a
refusal is published rather than substituted.  They say nothing about whether a derived level
on Korea is CORRECT, because nothing in this repo can: there is no Korean real price to compare
against and no measurable KRW venue anywhere to extrapolate from.  A green run here is not
evidence the filled prices are right.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import derived_prices as dpx
import price_grid_audit as pga
import returns_core as rc

ANCH = ["2020-12-31", "2021-12-31", "2022-12-30", "2023-12-29"]


# --------------------------------------------------------------------------- #
#  Fixtures                                                                   #
# --------------------------------------------------------------------------- #
def _panel(rows):
    """rows: (source, periodEndDate, marketCap, shares, dividendsPaid, currency)."""
    df = pd.DataFrame(rows, columns=["source", "periodEndDate", "marketCap",
                                     "weightedAverageShsOut", "dividendsPaid",
                                     "reportedCurrency"])
    df["price"] = df["marketCap"] / df["weightedAverageShsOut"]
    return df


def _quarters(sym, ccy, caps, shares=1000.0):
    """One row per anchor year-end, no dividends -- so the level IS the price."""
    return [(sym, "%s-12-31" % (2020 + i), c, shares, 0.0, ccy)
            for i, c in enumerate(caps)]


def _real(tmp_path, priced):
    """priced = {symbol: {anchor: adjClose}}.  Writes the on-disk price schema."""
    rows = []
    for sym, per in priced.items():
        for a, px in per.items():
            rows.append({"date_requested": a, "date_actual": a,
                         "symbol": sym, "adjClose": px})
    p = tmp_path / "real_prices.csv"
    pd.DataFrame(rows, columns=["date_requested", "date_actual", "symbol",
                                "adjClose"]).to_csv(p, index=False)
    return rc.PriceSource(str(p), anchors=ANCH)


#  US name the real leg prices; a Korean name it cannot (KRW reporter on a KRW line, so the
#  currency guard CLEARS); an Oslo name it cannot (USD reporter on a NOK line, so the guard
#  REFUSES) -- the three cases the route has to distinguish.
PANEL_ROWS = (_quarters("AAPL", "USD", [100.0, 110.0, 120.0, 130.0])
              + _quarters("005930.KS", "KRW", [50.0, 60.0, 70.0, 80.0])
              + _quarters("EQNR.OL", "USD", [20.0, 25.0, 30.0, 35.0]))
UNIVERSE = ["AAPL", "005930.KS", "EQNR.OL"]


def _gapfill(tmp_path, **derived_kw):
    real = _real(tmp_path, {"AAPL": {a: 10.0 * (i + 1) for i, a in enumerate(ANCH)}})
    derived = dpx.DerivedPriceSource(_panel(PANEL_ROWS), anchors=ANCH,
                                     benchmark_source=real, **derived_kw)
    return dpx.GapFillPriceSource(real, derived), real, derived


# --------------------------------------------------------------------------- #
#  BEHAVIOURAL -- the routing                                                 #
# --------------------------------------------------------------------------- #
def test_real_wins_wherever_the_real_leg_has_a_price(tmp_path):
    """The route must be INERT on every name the real file already prices -- that is the
    whole difference from the derived-preferred composite."""
    gf, real, _d = _gapfill(tmp_path)
    for a in ANCH:
        assert gf.price("AAPL", a) == real.price("AAPL", a)
    assert "AAPL" not in gf._gapfilled


def test_the_derived_leg_fills_ONLY_where_the_real_leg_is_empty(tmp_path):
    gf, _r, derived = _gapfill(tmp_path)
    assert gf._gapfilled == {"005930.KS"}
    for a in ANCH:
        assert gf.price("005930.KS", a) == derived.price("005930.KS", a)
        assert gf.price("005930.KS", a) is not None


def test_a_currency_mismatched_real_empty_name_is_REFUSED_not_substituted(tmp_path):
    """Oslo: a USD reporter on a NOK-listed line.  The guard refuses, and the route publishes
    the refusal rather than handing back a number that would carry the FX move as return."""
    gf, _r, _d = _gapfill(tmp_path)
    for a in ANCH:
        assert gf.price("EQNR.OL", a) is None, "a refusal must not yield a price"
    rep = gf.assignment_report(UNIVERSE).set_index("ticker")
    assert rep.loc["EQNR.OL", "leg"] == "REFUSED"
    assert rep.loc["EQNR.OL", "reason"] == "currency_mismatch"
    assert rep.loc["AAPL", "leg"] == "real"
    assert rep.loc["005930.KS", "leg"] == "derived_fill"


def test_a_refused_name_resolves_to_no_buy_not_to_the_minus_100_floor(tmp_path):
    """The consequence downstream: a refusal is an EXCLUSION, which is the harmless failure
    mode.  It must not become a -100% total loss."""
    gf, _r, _d = _gapfill(tmp_path)
    rdf = rc.compute_returns(UNIVERSE, ANCH[0], ANCH[-1], gf)
    by = dict(zip(rdf["ticker"], rdf["status"]))
    assert by["EQNR.OL"] == "no_buy"
    assert by["AAPL"] == "ok" and by["005930.KS"] == "ok"
    assert not (rdf["total_return_floor"] == -1.0).any()


def test_no_window_can_mix_legs(tmp_path):
    """A real buy leg with a derived eval leg DOUBLE-COUNTS dividends (measured IC residual
    +0.0554 mixed vs +0.0028 consistent).  Here it is impossible by construction: the
    gap-filled set and the real-priceable set are disjoint SETS, not a per-anchor choice.

    Built so the naive rule would mix: PARTIAL has a real price at the FIRST anchor only and a
    derived level at every anchor.  A per-anchor "use whichever leg has a price" rule takes
    real at the buy leg and derived at the eval leg; the shipped rule keeps it on real.
    """
    real = _real(tmp_path, {"AAPL": {a: 10.0 for a in ANCH},
                            "PARTIAL": {ANCH[0]: 7.0}})
    rows = PANEL_ROWS + _quarters("PARTIAL", "USD", [7.0, 8.0, 9.0, 10.0])
    derived = dpx.DerivedPriceSource(_panel(rows), anchors=ANCH, benchmark_source=real)
    gf = dpx.GapFillPriceSource(real, derived)
    assert "PARTIAL" not in gf._gapfilled
    assert gf._gapfilled & gf._real_tickers == set()
    assert gf.price("PARTIAL", ANCH[0]) == 7.0
    assert gf.price("PARTIAL", ANCH[-1]) is None, (
        "the eval leg came from the derived source -- that is a mixed window")


def test_gapfill_is_NOT_the_derived_preferred_composite(tmp_path):
    """The two composites must disagree on exactly the shared names, and the gap-fill must be
    the one that leaves them alone."""
    real = _real(tmp_path, {"AAPL": {a: 10.0 for a in ANCH}})
    derived = dpx.DerivedPriceSource(_panel(PANEL_ROWS), anchors=ANCH,
                                     benchmark_source=real)
    gf = dpx.GapFillPriceSource(real, derived)
    comp = dpx.CompositePriceSource(derived, real)
    assert comp.price("AAPL", ANCH[0]) == derived.price("AAPL", ANCH[0])
    assert gf.price("AAPL", ANCH[0]) == real.price("AAPL", ANCH[0])
    assert comp.price("AAPL", ANCH[0]) != gf.price("AAPL", ANCH[0])


def test_the_benchmark_always_comes_from_the_real_leg(tmp_path):
    """URTH is an ETF and files no statements, so it can never be derived."""
    real = _real(tmp_path, {"AAPL": {a: 10.0 for a in ANCH},
                            rc.BENCHMARK_SYMBOL: {a: 100.0 + i
                                                  for i, a in enumerate(ANCH)}})
    derived = dpx.DerivedPriceSource(_panel(PANEL_ROWS), anchors=ANCH,
                                     benchmark_source=real)
    gf = dpx.GapFillPriceSource(real, derived)
    assert list(gf.benchmark_series()) == list(real.benchmark_series())


# --------------------------------------------------------------------------- #
#  BEHAVIOURAL -- the published refusal                                       #
# --------------------------------------------------------------------------- #
def test_per_venue_counts_reports_clear_and_refuse(tmp_path):
    gf, _r, _d = _gapfill(tmp_path)
    pv = gf.per_venue_counts(UNIVERSE).set_index("venue")
    assert pv.loc[".KS", "derived_fill"] == 1
    assert pv.loc[".KS", "REFUSED"] == 0
    assert pv.loc[".KS", "pct_cleared_of_real_empty"] == 100.0
    assert pv.loc[".OL", "REFUSED"] == 1
    assert pv.loc[".OL", "pct_cleared_of_real_empty"] == 0.0
    #  a venue the real leg covers is INERT: nothing to clear, nothing to refuse
    assert pv.loc["(none)", "real"] == 1
    assert pv.loc["(none)", "n_real_empty"] == 0
    assert np.isnan(pv.loc["(none)", "pct_cleared_of_real_empty"])


def test_refusal_reasons_names_the_venue_and_the_reason(tmp_path):
    gf, _r, _d = _gapfill(tmp_path)
    ref = gf.refusal_reasons(UNIVERSE)
    assert list(ref.itertuples(index=False, name=None)) == [(".OL", "currency_mismatch", 1)]


def test_a_name_absent_from_the_panel_entirely_is_refused_as_not_in_panel(tmp_path):
    gf, _r, _d = _gapfill(tmp_path)
    rep = gf.assignment_report(["GHOST.PA"]).set_index("ticker")
    assert rep.loc["GHOST.PA", "leg"] == "REFUSED"
    assert rep.loc["GHOST.PA", "reason"] == "not_in_panel"


@pytest.mark.parametrize("sym,expected", [
    ("AAPL", None),               # priceable
    ("005930.KS", None),          # priceable
    ("EQNR.OL", "currency_mismatch"),
    ("NOPE.PA", "not_in_panel"),
])
def test_drop_reason(tmp_path, sym, expected):
    _gf, _r, derived = _gapfill(tmp_path)
    assert derived.drop_reason(sym) == expected


def test_drop_reason_counts_is_a_census_not_a_sample(tmp_path):
    _gf, _r, derived = _gapfill(tmp_path)
    assert derived.drop_reason_counts() == {"currency_mismatch": 1}


# --------------------------------------------------------------------------- #
#  MUTATION -- the currency guard is load-bearing, and stays ON               #
# --------------------------------------------------------------------------- #
def test_the_currency_guard_is_ON_by_default():
    assert dpx.REQUIRE_LISTING_CURRENCY_MATCH is True


def test_MUTATION_with_the_currency_guard_OFF_the_refusal_becomes_a_filled_number(tmp_path):
    """Reopening the defect on purpose.  This is the move the route must NOT make to widen
    coverage: the guard fixed a measured -0.1234 mean log gap on the 22.2% mismatched
    subpopulation, and Oslo is that subpopulation.  If this test failed, the refusal test
    above would be testing the fixture rather than the guard.
    """
    gf, _r, _d = _gapfill(tmp_path, require_listing_currency_match=False)
    assert "EQNR.OL" in gf._gapfilled
    assert gf.price("EQNR.OL", ANCH[0]) is not None
    assert gf.refusal_reasons(UNIVERSE).empty


# --------------------------------------------------------------------------- #
#  STRUCTURAL -- wiring, defaults, and the audit's honesty                    #
# --------------------------------------------------------------------------- #
def test_real_is_still_the_default_route_and_the_new_one_is_registered():
    assert dpx.PRICE_ROUTES[0] == "real"
    assert "real+derived" in dpx.PRICE_ROUTES
    import inspect
    sig = inspect.signature(dpx.build_price_source)
    assert "route" in sig.parameters and sig.parameters["route"].default is inspect._empty, \
        "route must be explicit at every call site -- no implicit second route"


def test_build_price_source_returns_the_gapfill_for_the_new_route(tmp_path):
    real_csv = tmp_path / "real_prices.csv"
    pd.DataFrame([{"date_requested": a, "date_actual": a, "symbol": "AAPL",
                   "adjClose": 10.0} for a in ANCH]).to_csv(real_csv, index=False)
    src = dpx.build_price_source("real+derived", prices_csv=str(real_csv),
                                 panel=_panel(PANEL_ROWS), anchors=ANCH)
    assert isinstance(src, dpx.GapFillPriceSource)
    assert isinstance(dpx.build_price_source("derived+real", prices_csv=str(real_csv),
                                             panel=_panel(PANEL_ROWS), anchors=ANCH),
                      dpx.CompositePriceSource)
    assert isinstance(dpx.build_price_source("real", prices_csv=str(real_csv),
                                             anchors=ANCH), rc.PriceSource)


def test_the_gapfill_diagnostics_state_the_extrapolation_limit(tmp_path):
    """The limit must travel with the object, not live only in a report someone may not
    print."""
    gf, _r, _d = _gapfill(tmp_path)
    d = gf.diagnostics()
    assert d["route"] == "real+derived gap-fill"
    assert d["n_tickers_gapfilled"] == 1
    assert "EXTRAPOLATION" in d["bias_measurability"]
    assert "disjoint" in d["leg_selection"]


def test_the_audit_banner_stops_claiming_EVERY_number_on_a_second_route():
    """B4-5.  The banner's "EVERY number the analysis stages below print is computed on this
    grid" goes FALSE the moment a second route is wired in.  It must say so."""
    rep = {'path': 'p', 'mtime': 'm', 'age_days': 1, 'n_overlap': 1, 'n_panel': 2,
           'overlap_frac': 0.5, 'findings': ['x']}
    real_banner = pga.banner(rep)
    assert 'EVERY number the analysis stages below print' in real_banner

    #  KWARG-TOLERANT so the CONTENT assertion runs.  A bare `banner(rep,
    #  price_route=...)` raises TypeError against a banner that takes no route, which proves
    #  the signature moved and says nothing about the claim.  What matters is that the
    #  unconditional "EVERY number" sentence is GONE on a second route -- against a banner
    #  with no route parameter that sentence is still there, and this fails on it.
    try:
        other = pga.banner(rep, price_route='real+derived')
    except TypeError:
        other = pga.banner(rep)
    assert 'EVERY number the analysis stages below print' not in other, (
        "the banner still claims every number below is computed on this grid, which is "
        "false the moment a second price route is wired in")
    assert 'real+derived' in other
    assert 'THIS AUDIT COVERS THE REAL GRID ONLY' in other


def test_STRUCTURAL_pipeline_analysis_reads_ONE_price_route_key_for_both_uses():
    """The audit's banner and the price source must not be able to disagree about which
    route is running."""
    import inspect
    import pipeline_analysis as pa
    src = inspect.getsource(pa)
    assert src.count('configdic.get("price_route"') == 2, \
        'the audit stage and _build_price_source must read the same key'
    assert 'price_route=str(configdic.get("price_route", "real")' in src


# --------------------------------------------------------------------------- #
#  THE BIAS TABLE -- and what it structurally cannot reach                     #
# --------------------------------------------------------------------------- #
def test_gap_stats_puts_mean_and_the_tails_BEFORE_the_median():
    """Column order is the guard here.  A median over a mostly currency-matched population is
    0.0000 by construction, and that exact vacuity let the -0.1234 currency defect ship on
    this module -- so the median is named `median_ref_only` and comes last."""
    import derived_price_validate as dpv
    st = dpv._gap_stats([-1.0, 0.0, 0.0, 0.0, 1.0])
    keys = list(st)
    assert keys.index("mean") < keys.index("median_ref_only")
    assert keys.index("p05") < keys.index("median_ref_only")
    assert keys.index("p95") < keys.index("median_ref_only")
    assert st["median_ref_only"] == 0.0 and st["mean"] == 0.0
    #  and the tails are what carry the information the median discards
    assert st["p05"] < 0 < st["p95"]


def test_gap_stats_sees_a_bias_the_median_is_blind_to():
    """The concrete failure: a fifth of the population at -0.12 and the rest at 0."""
    import derived_price_validate as dpv
    vals = [0.0] * 80 + [-0.1234] * 20
    st = dpv._gap_stats(vals)
    assert st["median_ref_only"] == 0.0
    assert st["mean"] == pytest.approx(-0.0247, abs=5e-4)
    assert st["p05"] == pytest.approx(-0.1234, abs=1e-4)


def test_the_bias_table_CANNOT_see_the_venues_the_route_actually_uses(tmp_path,
                                                                     monkeypatch):
    """THE STRUCTURAL LIMIT, pinned as code behaviour rather than left as prose.

    A cell reaches the bias table only if the REAL leg prices the name.  The gap-fill route
    uses the derived leg only where the real leg is EMPTY.  So the venue that gets filled
    (`.KS` here) can never appear in the table, and the venue that appears (`(none)`) is never
    filled.  The two populations are disjoint, which is why any correction carried from one to
    the other is an extrapolation.
    """
    import derived_price_validate as dpv
    real_csv = tmp_path / "real_prices.csv"
    pd.DataFrame([{"date_requested": a, "date_actual": a, "symbol": "AAPL",
                   "adjClose": 10.0 * (i + 1)}
                  for i, a in enumerate(ANCH)]).to_csv(real_csv, index=False)
    monkeypatch.setattr(dpv, "PAIRS", [(ANCH[0], ANCH[-1])])
    monkeypatch.setattr(dpv, "PRICES", str(real_csv))
    monkeypatch.setattr(dpv, "PRICES_2025", str(tmp_path / "nope.csv"))
    monkeypatch.setattr(dpv, "load_panel", lambda _p: _panel(PANEL_ROWS))
    monkeypatch.setattr(rc, "DEFAULT_ANCHORS", ANCH)

    bias = dpv.venue_currency_bias_table("ignored", min_n=1)
    um = dpv.unmeasurable_venue_table("ignored")
    assert set(bias["venue"]) == {"(none)"}, \
        "a venue the real leg cannot price must not appear in a bias table"
    assert ".KS" not in set(bias["venue"])
    #  ... and the unmeasurable table is where it DOES appear, labelled
    assert ".KS" in set(um["venue"])
    row = um.set_index("venue").loc[".KS"]
    assert row["n_real_priced"] == 0
    assert row["listing_ccy"] == "KRW"
    assert row["measurable_same_ccy_venues"] == "-- NONE --", \
        "no measurable KRW venue exists, so nothing can be extrapolated into Korea"
    assert "UNMEASURABLE" in row["bias"]
