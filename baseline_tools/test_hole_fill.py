"""
INTERIOR-HOLE imputation from the derived leg's GROWTH RATIO (offline; no network).

WHAT IT IS FOR (CEO, 2026-08-22): a name priced at the buy anchor and missing at the eval
anchor reads -100% and is scored a miss, even when a LATER price proves it was alive.  For
an interior hole that is not a punishment for missing data, it is a false reading that
sandbags our own track record.

WHY IT IS A RATIO AND NOT A PRICE -- THE REFUTATION, PINNED.  "Fill the hole with the
derived price" cannot be done literally: the two legs are not on one scale.  `adjClose` is
BACK-adjusted, the derived level accumulates FORWARD.  Measured on the 18,108 (ticker,
anchor) cells both legs price, `derived_level / adjClose` runs p1 0.0100 / p50 1.1690 /
p99 9.4600, min 0.0037, max 338.53, with only 36.06% inside +/-10% of 1.0.  So splicing a
LEVEL manufactures a return of up to ~338x at the splice.
`test_the_NAIVE_level_splice_would_manufacture_a_return` builds that case and shows the
shipped construction getting it right, so nobody can "simplify" this class back into the
bug.

THE CORRECTNESS PROOF IS test_the_fill_is_RETURN_PRESERVING_across_the_hole.  If any basis
mixing had crept in, that identity would fail.  It is asserted numerically rather than
argued in prose.
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
import returns_core as rc

ANCH = ["2020-12-31", "2021-12-31", "2022-12-30", "2023-12-29"]
P, H, N = ANCH[0], ANCH[1], ANCH[2]      # prev-real, hole, next-real


# --------------------------------------------------------------------------- #
#  Fixtures                                                                   #
# --------------------------------------------------------------------------- #
def _real(tmp_path, priced, name="real_prices.csv"):
    rows = [{"date_requested": a, "date_actual": a, "symbol": s, "adjClose": px}
            for s, per in priced.items() for a, px in per.items()]
    p = tmp_path / name
    pd.DataFrame(rows, columns=["date_requested", "date_actual", "symbol",
                                "adjClose"]).to_csv(p, index=False)
    return rc.PriceSource(str(p), anchors=ANCH)


def _panel(levels, ccy="USD", shares=1000.0):
    """levels = {symbol: {anchor: derived LEVEL}}.  No dividends, so level == price."""
    rows = []
    for sym, per in levels.items():
        for a, lvl in per.items():
            rows.append((sym, f"{a[:4]}-12-31", lvl * shares, shares, 0.0, ccy))
    df = pd.DataFrame(rows, columns=["source", "periodEndDate", "marketCap",
                                     "weightedAverageShsOut", "dividendsPaid",
                                     "reportedCurrency"])
    df["price"] = df["marketCap"] / df["weightedAverageShsOut"]
    return df


def _derived(levels, real=None, **kw):
    return dpx.DerivedPriceSource(_panel(levels), anchors=ANCH,
                                  benchmark_source=real, **kw)


#  HOLEY: real prices at P and N, a hole at H.  The derived leg is on a deliberately
#  DIFFERENT scale (x100) so a level splice would be visibly absurd and only a ratio works.
REAL_HOLEY = {"HOLEY": {P: 10.0, N: 12.0, ANCH[3]: 13.0}}
DERIVED_HOLEY = {"HOLEY": {P: 1000.0, H: 1100.0, N: 1200.0, ANCH[3]: 1300.0}}
G_PH = 1100.0 / 1000.0        # the derived growth over (P, H) = 1.10


def _src(tmp_path, real_prices=None, derived_levels=None):
    real = _real(tmp_path, real_prices or REAL_HOLEY)
    der = _derived(derived_levels or DERIVED_HOLEY, real=real)
    return dpx.HoleFilledPriceSource(real, der), real, der


# --------------------------------------------------------------------------- #
#  THE REFUTATION                                                             #
# --------------------------------------------------------------------------- #
def test_the_NAIVE_level_splice_would_manufacture_a_return(tmp_path):
    """"Fill the hole with the derived price" taken literally, against the shipped rule.

    The derived leg here is on a x100 scale -- which is not a contrived number: measured,
    `derived_level / adjClose` spans 0.0037 to 338.53 on real data and only 36% of cells sit
    within +/-10% of 1.0.  A level splice reads +10,900% over (P, H); the ratio construction
    reads +10%, which is what the derived leg actually says happened.
    """
    hf, real, der = _src(tmp_path)
    naive = der.price("HOLEY", H) / real.price("HOLEY", P) - 1.0
    assert naive == pytest.approx(109.0)                      # +10,900%
    shipped = hf.price("HOLEY", H) / real.price("HOLEY", P) - 1.0
    assert shipped == pytest.approx(G_PH - 1.0)               # +10%
    assert abs(naive) > 100 * abs(shipped)


# --------------------------------------------------------------------------- #
#  THE THREE PROPERTIES                                                       #
# --------------------------------------------------------------------------- #
def test_property_1_the_rescued_window_reads_the_DERIVED_total_return(tmp_path):
    hf, _real, _der = _src(tmp_path)
    rdf = rc.compute_returns(["HOLEY"], P, H, hf)
    row = rdf.iloc[0]
    assert row["status"] == "ok", "the whole point: it is no longer terminal"
    assert row["total_return"] == pytest.approx(G_PH - 1.0)
    assert row["total_return_floor"] == pytest.approx(G_PH - 1.0)
    assert bool(row["terminal_flag"]) is False


def test_property_2_the_fill_is_RETURN_PRESERVING_across_the_hole(tmp_path):
    """THE CORRECTNESS PROOF.  (1+r(P->H)) * (1+r(H->N)) == 1 + r(P->N), identically.

    Any basis mixing -- a derived level meeting a real level inside one ratio -- breaks this
    identity, so asserting it numerically is stronger than the algebra in the docstring.
    It also means the fill can never INJECT return: it only decides how the outer return is
    split between the two sub-periods.
    """
    hf, _real, _der = _src(tmp_path)
    r_ph = rc.compute_returns(["HOLEY"], P, H, hf).iloc[0]["total_return"]
    r_hn = rc.compute_returns(["HOLEY"], H, N, hf).iloc[0]["total_return"]
    r_pn = rc.compute_returns(["HOLEY"], P, N, hf).iloc[0]["total_return"]
    assert (1 + r_ph) * (1 + r_hn) == pytest.approx(1 + r_pn, rel=1e-12)


@pytest.mark.parametrize("g_scale", [0.01, 0.5, 1.001, 2.0, 50.0])
def test_property_2_holds_for_ANY_derived_growth(tmp_path, g_scale):
    """The identity is structural, not a property of the fixture numbers -- so it must hold
    however wrong the derived leg is.  That is exactly what bounds the damage."""
    lv = {"HOLEY": dict(DERIVED_HOLEY["HOLEY"])}
    lv["HOLEY"][H] = lv["HOLEY"][P] * g_scale
    hf, _real, _der = _src(tmp_path, derived_levels=lv)
    r_ph = rc.compute_returns(["HOLEY"], P, H, hf).iloc[0]["total_return"]
    r_hn = rc.compute_returns(["HOLEY"], H, N, hf).iloc[0]["total_return"]
    r_pn = rc.compute_returns(["HOLEY"], P, N, hf).iloc[0]["total_return"]
    assert r_ph == pytest.approx(g_scale - 1.0)
    assert (1 + r_ph) * (1 + r_hn) == pytest.approx(1 + r_pn, rel=1e-12)


def test_a_window_SPANNING_the_hole_is_bit_for_bit_unchanged(tmp_path):
    """The corollary that matters operationally: no certified number that spans a hole can
    move, so this cannot quietly re-price the existing sample."""
    hf, real, _der = _src(tmp_path)
    for buy, ev in ((P, N), (P, ANCH[3]), (N, ANCH[3])):
        a = rc.compute_returns(["HOLEY"], buy, ev, real).iloc[0]
        b = rc.compute_returns(["HOLEY"], buy, ev, hf).iloc[0]
        assert a["total_return"] == b["total_return"]
        assert a["status"] == b["status"]


# --------------------------------------------------------------------------- #
#  WHAT IT MUST NOT TOUCH -- the no_buy / terminal distinction                 #
# --------------------------------------------------------------------------- #
def test_a_name_with_NO_real_price_stays_no_buy(tmp_path):
    """No prior real anchor to scale from, so there is nothing to impute against.  The
    derived leg prices it perfectly well -- and is still not used here.  Handing those names
    a price is GapFillPriceSource's job, deliberately kept separate."""
    real = _real(tmp_path, REAL_HOLEY)
    der = _derived({**DERIVED_HOLEY, "KOREA.KS": {a: 100.0 for a in ANCH}}, real=real,
                   require_listing_currency_match=False)
    hf = dpx.HoleFilledPriceSource(real, der)
    assert hf.price("KOREA.KS", P) is None
    rdf = rc.compute_returns(["KOREA.KS"], P, N, hf)
    assert rdf.iloc[0]["status"] == "no_buy"
    #  scoped to KOREA.KS: HOLEY is in the same fixture and IS legitimately imputed
    assert not any(t == "KOREA.KS" for (t, _a) in hf._imputed)


def test_a_TRAILING_gap_stays_terminal(tmp_path):
    """The series genuinely ends: no later real price, so it is not an interior hole.  This
    is the distinction `terminal` exists to express and the fill must not erase it."""
    real_px = {"DEADCO": {P: 10.0, H: 11.0}}          # nothing at N or later
    der_lv = {"DEADCO": {a: 1000.0 * (1 + 0.5 * i) for i, a in enumerate(ANCH)}}
    hf, _real, _der = _src(tmp_path, real_prices=real_px, derived_levels=der_lv)
    assert hf.price("DEADCO", N) is None
    rdf = rc.compute_returns(["DEADCO"], P, N, hf)
    assert rdf.iloc[0]["status"] == "terminal"
    assert rdf.iloc[0]["total_return_floor"] == -1.0
    assert not hf._imputed


def test_a_LEADING_gap_is_not_filled(tmp_path):
    """Before the first real price the name was not listed; imputing there would invent a
    buy leg for a company that had not IPO'd."""
    real_px = {"LATECO": {N: 12.0, ANCH[3]: 13.0}}
    der_lv = {"LATECO": {a: 1000.0 + 100.0 * i for i, a in enumerate(ANCH)}}
    hf, _real, _der = _src(tmp_path, real_prices=real_px, derived_levels=der_lv)
    assert hf.price("LATECO", P) is None
    assert hf.price("LATECO", H) is None
    assert rc.compute_returns(["LATECO"], P, N, hf).iloc[0]["status"] == "no_buy"


def test_an_EXACTLY_repeated_derived_level_is_refused_by_the_backfill_guard(tmp_path):
    """g == 1.0 exactly is not a flat price, it is the vendor carrying a value forward.
    `DerivedPriceSource._guard_repeated_price` drops the whole run, so the hole is refused
    rather than imputed at 0%.  Discovered by parametrising property 2 at g_scale=1.0 and
    getting NaN -- worth its own test now that the interaction is understood."""
    lv = {"HOLEY": dict(DERIVED_HOLEY["HOLEY"])}
    lv["HOLEY"][H] = lv["HOLEY"][P]                 # bit-identical to the previous period
    hf, _real, _der = _src(tmp_path, derived_levels=lv)
    assert hf.price("HOLEY", H) is None
    assert rc.compute_returns(["HOLEY"], P, H, hf).iloc[0]["status"] == "terminal"


def test_a_hole_priced_off_the_SAME_FILING_as_its_base_is_refused(tmp_path):
    """THE BUG THIS FOUND.  The derived anchor rule is "newest period end <= Dec-31", so a
    name that skipped a filing is priced at BOTH anchors off the same statement: g comes out
    exactly 1.0 and the hole would be imputed with a fabricated 0% return -- which looks
    identical to a real measurement.  Refused on the IDENTITY of the filing, so there is no
    threshold.  Before the fix this returned 10.0 where it should return None.
    """
    der_lv = {"HOLEY": {P: 1000.0, N: 1200.0}}      # no filing at all in the hole year
    hf, _real, der = _src(tmp_path, derived_levels=der_lv)
    assert der.price("HOLEY", H) is not None, "premise: the derived leg CARRIES FORWARD here"
    assert der.picked_period_end("HOLEY", H) == der.picked_period_end("HOLEY", P)
    assert hf.price("HOLEY", H) is None
    assert hf.refusal_counts() == {"same_filing_as_base_anchor": 1}
    assert rc.compute_returns(["HOLEY"], P, H, hf).iloc[0]["status"] == "terminal"


def test_a_hole_the_derived_leg_cannot_bridge_is_REFUSED_and_counted(tmp_path):
    """It must price BOTH the hole and the anchor being scaled from.  A refusal leaves the
    hole exactly as it was -- it does not fall back to something invented."""
    #  The derived leg must have NO pick at the base anchor either -- its first filing is
    #  AFTER P, so there is nothing to scale from.  (A missing filing at the HOLE is a
    #  different case: the anchor rule carries the previous one forward, which is
    #  test_a_hole_priced_off_the_SAME_FILING_as_its_base_is_refused.)
    der_lv = {"HOLEY": {N: 1200.0, ANCH[3]: 1300.0}}
    hf, _real, der = _src(tmp_path, derived_levels=der_lv)
    assert der.price("HOLEY", P) is None
    assert hf.price("HOLEY", H) is None
    assert hf.refusal_counts() == {"derived_leg_cannot_bridge": 1}
    assert rc.compute_returns(["HOLEY"], P, H, hf).iloc[0]["status"] == "terminal"


def test_a_multi_anchor_hole_run_is_filled_from_the_SAME_prior_real_anchor(tmp_path):
    """Two consecutive holes both scale off P, so the composition identity still closes over
    the whole run rather than compounding an imputed value onto an imputed value."""
    real_px = {"GAPPY": {P: 10.0, ANCH[3]: 20.0}}
    der_lv = {"GAPPY": {P: 1000.0, H: 1100.0, N: 1500.0, ANCH[3]: 2000.0}}
    hf, _real, _der = _src(tmp_path, real_prices=real_px, derived_levels=der_lv)
    assert hf._imputed[("GAPPY", H)]["from_anchor"] == P
    assert hf._imputed[("GAPPY", N)]["from_anchor"] == P
    assert hf.price("GAPPY", H) == pytest.approx(10.0 * 1.10)
    assert hf.price("GAPPY", N) == pytest.approx(10.0 * 1.50)
    r1 = rc.compute_returns(["GAPPY"], P, N, hf).iloc[0]["total_return"]
    r2 = rc.compute_returns(["GAPPY"], N, ANCH[3], hf).iloc[0]["total_return"]
    r_all = rc.compute_returns(["GAPPY"], P, ANCH[3], hf).iloc[0]["total_return"]
    assert (1 + r1) * (1 + r2) == pytest.approx(1 + r_all, rel=1e-12)


# --------------------------------------------------------------------------- #
#  DISJOINTNESS -- the argument that had to be re-derived, not assumed         #
# --------------------------------------------------------------------------- #
def test_hole_filled_and_gap_filled_populations_are_DISJOINT(tmp_path):
    """Hole-fill needs >= 2 real anchors; gap-fill needs ZERO.  So no ticker can be served by
    both, per-ticker exclusivity survives, and no window can mix measurement bases."""
    real = _real(tmp_path, REAL_HOLEY)
    der = _derived({**DERIVED_HOLEY, "KOREA.KS": {a: 100.0 + i for i, a in enumerate(ANCH)}},
                   real=real, require_listing_currency_match=False)
    hf = dpx.HoleFilledPriceSource(real, der)
    gf = dpx.GapFillPriceSource(hf, der)
    hole_tickers = {t for (t, _a) in hf._imputed}
    assert hole_tickers == {"HOLEY"}
    assert gf._gapfilled == {"KOREA.KS"}
    assert hole_tickers & gf._gapfilled == set()


def test_the_composed_source_serves_each_ticker_from_ONE_basis(tmp_path):
    """End to end: HOLEY is real-scale throughout (its hole imputed by ratio) and KOREA.KS is
    derived throughout.  Neither has a window that straddles two bases."""
    real = _real(tmp_path, REAL_HOLEY)
    der = _derived({**DERIVED_HOLEY, "KOREA.KS": {a: 100.0 * (1 + 0.1 * i)
                                                  for i, a in enumerate(ANCH)}},
                   real=real, require_listing_currency_match=False)
    gf = dpx.GapFillPriceSource(dpx.HoleFilledPriceSource(real, der), der)
    #  HOLEY: the spanning window is untouched real; the rescued window is the derived ratio
    assert gf.price("HOLEY", P) == 10.0
    assert gf.price("HOLEY", H) == pytest.approx(11.0)
    assert gf.price("HOLEY", N) == 12.0
    #  KOREA.KS: entirely the derived leg
    assert gf.price("KOREA.KS", P) == pytest.approx(100.0)
    r = rc.compute_returns(["KOREA.KS"], P, N, gf).iloc[0]["total_return"]
    assert r == pytest.approx(100.0 * 1.2 / 100.0 - 1.0)


# --------------------------------------------------------------------------- #
#  WIRING -- default OFF, and refused where it has no meaning                  #
# --------------------------------------------------------------------------- #
def test_the_fill_is_OFF_by_default(tmp_path):
    import inspect
    sig = inspect.signature(dpx.build_price_source)
    assert sig.parameters["fill_interior_holes"].default is False


def _csv(tmp_path):
    rows = [{"date_requested": a, "date_actual": a, "symbol": s, "adjClose": px}
            for s, per in REAL_HOLEY.items() for a, px in per.items()]
    p = tmp_path / "real_prices.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_build_price_source_applies_the_fill_on_the_real_route(tmp_path):
    plain = dpx.build_price_source("real", prices_csv=_csv(tmp_path), anchors=ANCH)
    filled = dpx.build_price_source("real", prices_csv=_csv(tmp_path), anchors=ANCH,
                                    panel=_panel(DERIVED_HOLEY), fill_interior_holes=True)
    assert isinstance(plain, rc.PriceSource)
    assert isinstance(filled, dpx.HoleFilledPriceSource)
    assert plain.price("HOLEY", H) is None
    assert filled.price("HOLEY", H) == pytest.approx(11.0)


def test_build_price_source_hole_fills_BEFORE_gap_filling(tmp_path):
    src = dpx.build_price_source("real+derived", prices_csv=_csv(tmp_path), anchors=ANCH,
                                 panel=_panel(DERIVED_HOLEY), fill_interior_holes=True)
    assert isinstance(src, dpx.GapFillPriceSource)
    assert isinstance(src.real, dpx.HoleFilledPriceSource)
    assert src.price("HOLEY", H) == pytest.approx(11.0)


@pytest.mark.parametrize("route", ["derived", "derived+real"])
def test_the_flag_is_REFUSED_on_a_derived_preferred_route(tmp_path, route):
    """Refused rather than silently dropped: a caller must not be able to believe holes were
    filled when the route makes that meaningless."""
    with pytest.raises(ValueError, match="REAL-preferred"):
        dpx.build_price_source(route, prices_csv=_csv(tmp_path), anchors=ANCH,
                               panel=_panel(DERIVED_HOLEY), fill_interior_holes=True)


def test_the_benchmark_is_never_hole_filled(tmp_path):
    """URTH is an ETF, files no statements, and must keep coming off the real leg."""
    real = _real(tmp_path, {**REAL_HOLEY,
                            rc.BENCHMARK_SYMBOL: {a: 100.0 + i for i, a in enumerate(ANCH)}})
    hf = dpx.HoleFilledPriceSource(real, _derived(DERIVED_HOLEY, real=real))
    assert list(hf.benchmark_series()) == list(real.benchmark_series())
    assert not any(t == rc.BENCHMARK_SYMBOL for (t, _a) in hf._imputed)


def test_diagnostics_state_the_basis_and_the_invariant(tmp_path):
    hf, _real, _der = _src(tmp_path)
    d = hf.diagnostics()
    assert d["n_holes_filled"] == 1
    assert "RATIO" in d["basis"]
    assert "return-preserving" in d["invariant"]
    assert "no_buy" in d["untouched"] and "TRAILING" in d["untouched"]


def test_imputation_report_is_per_venue_with_the_g_distribution(tmp_path):
    hf, _real, _der = _src(tmp_path)
    rep = hf.imputation_report()
    assert list(rep["venue"]) == ["(none)"]
    assert int(rep["filled"].iloc[0]) == 1
    assert float(rep["g_median"].iloc[0]) == pytest.approx(G_PH)


def test_is_imputed_marks_only_the_imputed_cell(tmp_path):
    hf, _real, _der = _src(tmp_path)
    assert hf.is_imputed("HOLEY", H) is True
    assert hf.is_imputed("HOLEY", P) is False
    assert hf.is_imputed("HOLEY", N) is False


# --------------------------------------------------------------------------- #
#  MUTATION                                                                   #
# --------------------------------------------------------------------------- #
def test_MUTATION_without_the_fill_the_rescued_window_is_terminal_again(tmp_path):
    """Sensitivity control for property 1: on the bare real leg the same window reads -100%
    and is scored a miss.  If this passed with the fill on, the tests above would be
    measuring the fixture."""
    real = _real(tmp_path, REAL_HOLEY)
    rdf = rc.compute_returns(["HOLEY"], P, H, real)
    assert rdf.iloc[0]["status"] == "terminal"
    assert rdf.iloc[0]["total_return_floor"] == -1.0
    rate, n = rc.beat_rate(rdf, 0.0, threshold=0.10, missing="fail")
    assert (rate, n) == (0.0, 1)


# --------------------------------------------------------------------------- #
#  PIPELINE WIRING -- default OFF, because it moves measured numbers           #
# --------------------------------------------------------------------------- #
def test_the_pipeline_knob_is_OFF_by_default_and_returns_the_plain_real_leg(tmp_path,
                                                                           monkeypatch):
    """`configdic['fill_interior_holes']` absent must give the byte-identical real leg, so no
    shipped number can move by accident."""
    import pipeline_analysis as pa
    csv = _csv(tmp_path)
    monkeypatch.setattr(pa, "_PRICES_CSV", csv)
    monkeypatch.setattr(pa, "_PRICES_2025_CSV", str(tmp_path / "absent.csv"))
    ps = pa._build_price_source(lambda *a: None, {})
    assert isinstance(ps, rc.PriceSource)
    assert not isinstance(ps, dpx.HoleFilledPriceSource)
    assert ps.price("HOLEY", H) is None


@pytest.mark.parametrize("flag", [0, "0", "false", None])
def test_falsy_config_values_all_leave_the_fill_OFF(tmp_path, monkeypatch, flag):
    import pipeline_analysis as pa
    csv = _csv(tmp_path)
    monkeypatch.setattr(pa, "_PRICES_CSV", csv)
    monkeypatch.setattr(pa, "_PRICES_2025_CSV", str(tmp_path / "absent.csv"))
    cfg = {} if flag is None else {"fill_interior_holes": flag}
    assert isinstance(pa._build_price_source(lambda *a: None, cfg), rc.PriceSource)


def test_the_pipeline_knob_reads_the_SAME_name_the_source_takes():
    """One spelling, so a config key cannot drift from the parameter it feeds."""
    import inspect
    import pipeline_analysis as pa
    src = inspect.getsource(pa._build_price_source)
    assert "fill_interior_holes" in src
    assert "fill_interior_holes=fill_holes" in src
