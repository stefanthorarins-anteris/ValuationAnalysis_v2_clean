"""THE 1000x VENDOR PRICE-SCALE DETECTOR fires on the name that motivated it.

WHY THAT IS THE FIRST TEST.  The detector the issue register proposed -- synthetic price
(`marketCap/weightedAverageShsOut`) against the grid -- does NOT fire on ATRI, because FMP
scales `marketCap` by the same 1/1000 and the ratio cancels to 1.04.  A guard blind to its own
motivating case is worse than no guard, because it reads as an all-clear.  So the first thing
pinned here is that the shipped check catches the ATRI shape, and the second is that the
rejected check demonstrably does not -- kept as an executable record of WHY the design changed,
so nobody re-proposes it from the register text.

WHAT THESE TESTS CANNOT SEE, stated plainly:
  * They cannot tell a real deep-value company from a scaled one.  The check is a HEURISTIC on
    a valuation ratio; a genuine liquidation at 0.015x book is indistinguishable here and will
    be flagged.  That is why the stage reports and never gates.
  * They cannot detect a scaling defect applied CONSISTENTLY to the balance sheet as well as
    the price -- if equity were scaled by the same 1/1000, price/book returns to normal and
    both checks go quiet.  Nothing on disk would reveal that; only `vwap` (which the grid does
    not carry) or an external price source would.
  * The fixtures are hand-built frames.  A change that stops `cdx_df` carrying
    `bookValuePerShare` would make the audit silently report zero names, and only
    `test_the_audit_says_so_when_it_cannot_run` covers that -- by asserting it complains
    rather than returning a clean empty result.

OFFLINE.  No network, no pickle, no price fetch.
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

import price_scale_audit as psa


def _panel(rows):
    """rows: (source, marketCap, shares, equity, revenue, currency) per quarter."""
    out = []
    for i, (src, mcap, sh, eq, rev, cur) in enumerate(rows):
        out.append({"source": src,
                    "date": pd.Timestamp("2021-03-31") + pd.DateOffset(months=3 * (i % 8)),
                    "price": mcap / sh,
                    "bookValuePerShare": eq / sh,
                    "revenue": rev,
                    "weightedAverageShsOut": sh,
                    "reportedCurrency": cur})
    return pd.DataFrame(out)


def _repeat(src, mcap, sh, eq, rev, cur="USD", n=8):
    return [(src, mcap, sh, eq, rev, cur)] * n


#  ATRI's REAL saved numbers, from `delisted_out/dead_fundamentals_20260713_104350.pickle`
#  (2021-12-31 quarter): marketCap 1,269,524.90 -- itself scaled 1/1000 -- against 1,801,000
#  shares and a balance sheet FMP serves unscaled.  Equity is reconstructed from the
#  `bookValuePerShare` the same file reports (135.65 at the 2024 quarters; ~123 here).
_ATRI = _repeat("ATRI", 1269524.90, 1801000, 1801000 * 123.0, 48773000 * 4)


def test_the_shipped_check_FIRES_on_ATRI_the_name_that_motivated_it():
    out = psa.check_fundamentals_internal(_panel(_ATRI))
    row = out[out["source"] == "ATRI"]
    assert len(row) == 1, "ATRI was not flagged at all"
    assert row.iloc[0]["severity"] == "ALARM"
    assert row.iloc[0]["price_over_book"] < 0.02
    #  and the corroboration that makes it readable: one decade makes it ordinary
    assert 0.5 < row.iloc[0]["pb_x1000"] < 50


def test_the_REJECTED_check_does_NOT_fire_on_ATRI_which_is_why_it_was_rejected(tmp_path):
    """Executable record of the measurement that killed the register's proposal.  FMP applies
    the 1/1000 to `marketCap` too, so synthetic-vs-grid cancels to ~1.0 on the broken name."""
    grid = pd.DataFrame([{"date_requested": d, "symbol": "ATRI", "adjClose": 0.67507}
                         for d in ("2021-12-31", "2022-12-30", "2023-12-29")])
    gpath = os.path.join(str(tmp_path), "real_prices.csv")
    grid.to_csv(gpath, index=False)
    panel = _panel(_ATRI)
    panel["date"] = pd.Timestamp("2021-12-31")
    out, _ = psa.check_synthetic_vs_grid(panel, gpath)
    assert "ATRI" not in set(out["source"] if len(out) else []), (
        "synthetic-vs-grid flagged ATRI -- if this now passes, re-read the docstring: it "
        "would mean marketCap is no longer scaled and the rejected check has become viable")


def test_a_normal_company_is_not_flagged():
    """price/book 2.0 -- the live panel's median is 1.91."""
    out = psa.check_fundamentals_internal(_panel(_repeat("GOOD", 2000.0, 1000, 1000.0, 5000)))
    assert not len(out)


def test_a_single_quarter_of_noise_is_not_a_scaling_defect():
    """The defect is persistent by construction -- it is how the vendor stores the series.
    One stale marketCap must not raise an alarm."""
    rows = _repeat("BLIP", 2000.0, 1000, 1000.0, 5000, n=7) + \
        [("BLIP", 1.0, 1000, 1000.0, 5000, "USD")]
    out = psa.check_fundamentals_internal(_panel(rows))
    assert not len(out), "a one-quarter outlier moved the median"


def test_the_check_is_currency_free_and_the_pence_venues_cannot_confound_it():
    """Both legs are per-share quantities in the REPORTING currency, so a GBp-quoting London
    listing -- 227 of the 249 decade-scale disagreements the grid-based check produces -- is
    structurally invisible to this one.  That is the whole reason it is check A."""
    lon = _repeat("PENCE.L", 2000.0, 1000, 1000.0, 5000, cur="GBP")
    krw = _repeat("ADR", 2000.0, 1000, 1000.0, 5000, cur="KRW")
    out = psa.check_fundamentals_internal(_panel(lon + krw))
    assert not len(out)


# --------------------------------------------------------------------------- #
#  Check B: classify, do not alarm                                            #
# --------------------------------------------------------------------------- #
def _grid(tmp_path, rows):
    p = os.path.join(str(tmp_path), "real_prices.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


_ANCHORS = ("2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31")


def _panel_at_anchors(source, price, cur="USD"):
    return pd.DataFrame([{"source": source, "date": pd.Timestamp(a), "price": price,
                          "bookValuePerShare": price / 2.0, "revenue": 1000.0,
                          "weightedAverageShsOut": 100.0, "reportedCurrency": cur}
                         for a in _ANCHORS])


def test_a_pence_venue_is_classified_MINOR_UNIT_not_alarmed(tmp_path):
    """The LSE quotes in pence against GBP reporting: an exact 1/100 on hundreds of names.
    Alarming on that would bury the one finding that matters under 227 non-findings."""
    g = _grid(tmp_path, [{"date_requested": a, "symbol": "X.L", "adjClose": 100.0}
                         for a in _ANCHORS])
    out, _ = psa.check_synthetic_vs_grid(_panel_at_anchors("X.L", 1.0, "GBP"), g)
    assert list(out["verdict"]) == ["MINOR_UNIT"]


def test_an_FX_rate_is_classified_CURRENCY_MISMATCH_not_a_scaling_defect(tmp_path):
    """BSAC's CLP/USD median ratio is 1071.65 on the real grid -- 7.2% off 1000.  A loose
    'near 1000x' rule would have called a Chilean bank a vendor defect.  The tolerance is set
    against that measurement, so this test is the one that pins it."""
    g = _grid(tmp_path, [{"date_requested": a, "symbol": "BSAC", "adjClose": 1.0}
                         for a in _ANCHORS])
    out, _ = psa.check_synthetic_vs_grid(_panel_at_anchors("BSAC", 1071.65, "CLP"), g)
    assert list(out["verdict"]) == ["CURRENCY_MISMATCH"]


def test_an_unexplained_exact_decade_IS_raised(tmp_path):
    """The case check B exists for: the grid is scaled but `marketCap` is not, on a venue with
    no minor-unit convention."""
    g = _grid(tmp_path, [{"date_requested": a, "symbol": "ZZZ", "adjClose": 0.5}
                         for a in _ANCHORS])
    out, _ = psa.check_synthetic_vs_grid(_panel_at_anchors("ZZZ", 500.0, "USD"), g)
    assert list(out["verdict"]) == ["SCALING_SUSPECT"]


def test_a_10x_disagreement_is_below_the_floor(tmp_path):
    """10x is within reach of ordinary marketCap staleness plus dividend back-adjustment over
    six years; only >= 100x is treated as a units question."""
    g = _grid(tmp_path, [{"date_requested": a, "symbol": "ZZZ", "adjClose": 5.0}
                         for a in _ANCHORS])
    out, _ = psa.check_synthetic_vs_grid(_panel_at_anchors("ZZZ", 50.0, "USD"), g)
    assert not len(out)


# --------------------------------------------------------------------------- #
#  The stage reports and NEVER mutates                                        #
# --------------------------------------------------------------------------- #
def test_the_audit_changes_nothing_it_reads():
    """Q-38 stayed OPEN rather than being parked because its detector reaches names that are
    live in scoring.  The boundary is that it reads and prints; nothing downstream consumes
    it and no price is corrected."""
    panel = _panel(_ATRI + _repeat("GOOD", 2000.0, 1000, 1000.0, 5000))
    before = panel.copy(deep=True)
    psa.run_audit(panel, prices_csv=None, log=lambda *a: None)
    pd.testing.assert_frame_equal(panel, before)


def test_the_stage_is_wired_in_and_does_not_gate():
    import inspect
    import pipeline_analysis as pa
    src = inspect.getsource(pa.run_post_pick_analysis_suite) \
        if hasattr(pa, "run_post_pick_analysis_suite") else inspect.getsource(pa)
    assert "_audit_price_scale_stage" in src
    stage = inspect.getsource(pa._audit_price_scale_stage)
    for forbidden in ("adjClose =", "= psa.correct", "drop(", "raise RuntimeError"):
        assert forbidden not in stage, f"the stage looks like it does more than report: {forbidden}"


def test_the_audit_says_so_when_it_cannot_run():
    """Silence must never be the output of a missing input -- that is how a guard becomes
    decoration.  An empty or column-less panel must produce a complaint, not a clean zero."""
    said = []
    psa.run_audit(pd.DataFrame(), log=said.append)
    assert any("did not run" in s for s in said)


def test_the_report_names_its_own_blind_spots_every_run():
    """A detector that prints '0 flagged' and nothing else reads as an all-clear for a
    question it never asked."""
    said = []
    psa.run_audit(_panel(_repeat("GOOD", 2000.0, 1000, 1000.0, 5000)),
                  prices_csv=None, log=said.append)
    text = "\n".join(said)
    assert "BLIND SPOT" in text.upper()
    assert "marketCap" in text, "the reason check B misses ATRI is not stated"
    assert "absent from the panel" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
