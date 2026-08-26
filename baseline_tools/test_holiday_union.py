"""
The GENERALISED per-venue holiday union in `returns_core.PriceSource` (offline).

WHAT WENT WRONG.  `_merge_supplementary` did prefer-anchor / fill-from-neighbour for ONE
anchor, 2025-12-31, filling from 2025-12-30 "because exchanges closed on 2025-12-31
(.DE/.T/.KS ...) only report on 2025-12-30".  December 31 is a venue holiday every year.
Measured on the repo-local dev grid, `.DE`/`.ST`/`.IC` are unpriceable at 2018-12-31,
2019-12-31, 2021-12-31 AND 2024-12-31, and `.KS`/`.KQ` additionally at 2023-12-29.  The
union was right and applied at one anchor out of eight.

THE TRAP THIS MUST NOT REPEAT.  `fetch_prices.run_bulk`'s step-back is GLOBAL -- it asks
"did this date return any rows at all" -- so ~34,000 US rows make a body non-empty, the
step-back never fires, and Paris/Korea/Oslo are absent with nothing complaining.  A union
with that shape would be useless, so `test_the_union_is_PER_VENUE_not_global` is the
load-bearing test in this file: a fill for `.DE` must happen even when the anchor body is
full of US names.

WHAT THESE TESTS DO NOT CLAIM.  The fill needs a neighbouring body to fill FROM, and
neither grid on disk has one at the damaged anchors -- verified separately, the lookup
table is bit-identical to HEAD's on both grids.  These tests prove the MECHANISM on
synthetic grids that DO carry a neighbour body.  They say nothing about whether any real
number moves, and today none does.
"""
import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import returns_core as rc

COLS = ["date_requested", "date_actual", "symbol", "adjClose"]


def _csv(tmp_path, rows, name="grid.csv"):
    """rows = [(date_requested, date_actual, symbol, adjClose), ...]

    NOTE THE FIXTURE RULE THAT MAKES THESE TESTS DISCRIMINATE.  A neighbour-day row must
    carry its OWN `date_requested`, not the anchor's.  A row whose `date_requested` IS the
    anchor is taken by the ANCHOR layer whatever its `date_actual` says, so a fixture built
    that way passes against a PriceSource with no fill layer at all -- measured: three of
    the venue tests below did exactly that and were testing the certified selection instead
    of the union.  A separately-fetched neighbour-day body lands under its own date anyway,
    so the corrected shape is also the realistic one.
    """
    p = tmp_path / name
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    return str(p)


# --------------------------------------------------------------------------- #
#  BEHAVIOURAL -- the defect, at an anchor that is not 2025                    #
# --------------------------------------------------------------------------- #
def test_BEHAVIOURAL_a_venue_shut_on_dec31_is_priced_from_the_neighbour_at_an_EARLY_anchor(
        tmp_path):
    """The exact dev-grid defect: `.DE`/`.ST` report on 2018-12-28 and not on 2018-12-31.

    Under the old rule this anchor had no union at all (the union was 2025-only), so both
    names were unpriceable and every backtest leg through 2018 silently dropped them.
    """
    rows = [
        ("2018-12-31", "2018-12-31", "AAPL", 100.0),      # US traded on the 31st
        ("2018-12-28", "2018-12-28", "SAP.DE", 90.0),     # XETRA shut on the 31st
        ("2018-12-28", "2018-12-28", "VOLV-B.ST", 80.0),  # Stockholm shut on the 31st
    ]
    ps = rc.PriceSource(_csv(tmp_path, rows), anchors=["2018-12-31"])
    assert ps.price("AAPL", "2018-12-31") == 100.0
    assert ps.price("SAP.DE", "2018-12-31") == 90.0
    assert ps.price("VOLV-B.ST", "2018-12-31") == 80.0


def test_the_union_is_PER_VENUE_not_global(tmp_path):
    """THE LOAD-BEARING ONE.  A full US body at the anchor must not suppress the `.DE` fill.

    This is the shape `run_bulk`'s step-back got wrong.  Built so that a global rule --
    "the anchor returned rows, therefore nothing to fill" -- fails it and a per-symbol rule
    passes: 50 US names price AT the anchor while the two European names do not.
    """
    rows = [("2021-12-31", "2021-12-31", f"US{i}", 10.0 + i) for i in range(50)]
    rows += [("2021-12-30", "2021-12-30", "SAP.DE", 90.0),
             ("2021-12-30", "2021-12-30", "VOLV-B.ST", 80.0)]
    ps = rc.PriceSource(_csv(tmp_path, rows), anchors=["2021-12-31"])
    assert ps.price("US0", "2021-12-31") == 10.0
    assert ps.price("SAP.DE", "2021-12-31") == 90.0, \
        "a global 'the body was non-empty' rule would leave this unpriced"
    assert ps.price("VOLV-B.ST", "2021-12-31") == 80.0


def test_the_union_applies_at_EVERY_anchor_not_just_one(tmp_path):
    """Six anchors, the same venue shut on Dec-31 at each.  The old rule fixed the last."""
    anchors = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31",
               "2022-12-30", "2025-12-31"]
    neighbour = {"2018-12-31": "2018-12-28", "2019-12-31": "2019-12-30",
                 "2020-12-31": "2020-12-30", "2021-12-31": "2021-12-30",
                 "2022-12-30": "2022-12-29", "2025-12-31": "2025-12-30"}
    rows = []
    for a in anchors:
        rows.append((a, a, "AAPL", 100.0))
        rows.append((neighbour[a], neighbour[a], "SAP.DE", 90.0))
    ps = rc.PriceSource(_csv(tmp_path, rows), anchors=anchors)
    for a in anchors:
        assert ps.price("SAP.DE", a) == 90.0, f"no fill at anchor {a}"


def test_the_2025_special_case_it_replaces_still_behaves_identically(tmp_path):
    """Regression on the rule that was hardcoded: PREFER the 2025-12-31 close, FILL from
    2025-12-30 only for symbols with no 2025-12-31 row.

    The supplementary file carries its own `date_requested`, exactly as
    `real_prices_2025.csv` does, so this is the shipped shape and not a reconstruction.
    """
    main = _csv(tmp_path, [("2024-12-31", "2024-12-31", "AAPL", 50.0)], "main.csv")
    supp = _csv(tmp_path, [
        ("2025-12-31", "2025-12-31", "AAPL", 100.0),   # trades on the 31st -> preferred
        ("2025-12-30", "2025-12-30", "AAPL", 99.0),    # must NOT win
        ("2025-12-30", "2025-12-30", "SAP.DE", 90.0),  # only on the 30th -> filled
    ], "supp.csv")
    ps = rc.PriceSource(main, anchors=["2024-12-31", "2025-12-31"], supp_csv=supp)
    assert ps.price("AAPL", "2025-12-31") == 100.0
    assert ps.price("SAP.DE", "2025-12-31") == 90.0
    assert ps.price("AAPL", "2024-12-31") == 50.0


def test_the_fill_is_ADD_ONLY_and_never_overwrites_an_anchor_price(tmp_path):
    """A price present at the anchor is the anchor's price, full stop.  This is what keeps
    the change monotone: coverage can only rise and no existing leg can move."""
    rows = [("2021-12-31", "2021-12-31", "AAPL", 100.0),
            ("2021-12-30", "2021-12-30", "AAPL", 111.0)]
    ps = rc.PriceSource(_csv(tmp_path, rows), anchors=["2021-12-31"])
    assert ps.price("AAPL", "2021-12-31") == 100.0


def test_the_fill_NEVER_reaches_FORWARD_of_the_anchor(tmp_path):
    """A `date_actual` after the anchor is look-ahead and is not eligible, even when it is
    the only row the symbol has.

    NOTE the `date_requested` here is NOT an anchor, and that is deliberate throughout the
    fill tests below.  A row whose `date_requested` IS the anchor is taken by the ANCHOR
    layer whatever its `date_actual` says, so such a fixture tests the certified selection
    rather than the fill.  (That also means the anchor layer would consume a forward-dated
    row if one existed -- RAISED, NOT FIXED: it is the axis C1 certified, `run_bulk` only
    ever steps backwards, and all four price files on disk carry zero rows with
    date_actual > date_requested, checked 2026-08-22.)
    """
    rows = [("2023-01-03", "2023-01-03", "SAP.DE", 90.0)]
    ps = rc.PriceSource(_csv(tmp_path, rows), anchors=["2022-12-30"])
    assert ps.price("SAP.DE", "2022-12-30") is None


def test_the_fill_window_is_bounded(tmp_path):
    """Outside the window the name stays unpriced -- an arbitrarily stale price is not a
    substitute for the anchor close."""
    rows = [("2021-12-17", "2021-12-17", "SAP.DE", 90.0)]   # 14 days back
    grid = _csv(tmp_path, rows)
    assert rc.PriceSource(grid, anchors=["2021-12-31"]).price("SAP.DE", "2021-12-31") is None
    wide = rc.PriceSource(grid, anchors=["2021-12-31"], fill_window_days=14)
    assert wide.price("SAP.DE", "2021-12-31") == 90.0


def test_the_fill_takes_the_day_CLOSEST_to_the_anchor(tmp_path):
    """Two eligible days in the window -> the newer one, because it is the better estimate
    of the anchor close."""
    rows = [("2021-12-28", "2021-12-28", "SAP.DE", 88.0),
            ("2021-12-30", "2021-12-30", "SAP.DE", 90.0)]
    ps = rc.PriceSource(_csv(tmp_path, rows), anchors=["2021-12-31"])
    assert ps.price("SAP.DE", "2021-12-31") == 90.0


def test_the_fill_reaches_across_files(tmp_path):
    """The neighbour body may live in a different file from the anchor body -- which is the
    2025 case, and the reason `supp_csv` accepts an iterable now."""
    main = _csv(tmp_path, [("2021-12-31", "2021-12-31", "AAPL", 100.0)], "main.csv")
    s1 = _csv(tmp_path, [("2021-12-30", "2021-12-30", "SAP.DE", 90.0)], "s1.csv")
    s2 = _csv(tmp_path, [("2021-12-30", "2021-12-30", "VOLV-B.ST", 80.0)], "s2.csv")
    ps = rc.PriceSource(main, anchors=["2021-12-31"], supp_csv=[s1, s2])
    assert ps.price("SAP.DE", "2021-12-31") == 90.0
    assert ps.price("VOLV-B.ST", "2021-12-31") == 80.0


def test_fill_report_states_what_the_fill_did(tmp_path):
    rows = [("2021-12-31", "2021-12-31", "AAPL", 100.0),
            ("2021-12-29", "2021-12-29", "SAP.DE", 90.0)]
    rep = rc.PriceSource(_csv(tmp_path, rows), anchors=["2021-12-31"]).fill_report()
    assert list(rep["anchor"]) == ["2021-12-31"]
    assert int(rep["n_priced"].iloc[0]) == 2
    assert int(rep["n_filled"].iloc[0]) == 1
    assert int(rep["max_fill_lag_days"].iloc[0]) == 2


def test_a_price_csv_without_date_actual_still_loads(tmp_path):
    """Degrade, do not crash: such rows cannot serve as fill sources, which is correct."""
    p = tmp_path / "legacy.csv"
    pd.DataFrame([{"date_requested": "2021-12-31", "symbol": "AAPL", "adjClose": 100.0}]
                 ).to_csv(p, index=False)
    ps = rc.PriceSource(str(p), anchors=["2021-12-31"])
    assert ps.price("AAPL", "2021-12-31") == 100.0
    assert int(ps.fill_report()["n_filled"].iloc[0]) == 0


def test_a_price_csv_missing_a_required_column_fails_loudly(tmp_path):
    p = tmp_path / "broken.csv"
    pd.DataFrame([{"date_requested": "2021-12-31", "symbol": "AAPL"}]).to_csv(p, index=False)
    with pytest.raises(KeyError, match="adjClose"):
        rc.PriceSource(str(p), anchors=["2021-12-31"])


# --------------------------------------------------------------------------- #
#  MUTATION -- reopen the defect and prove the tests above see it              #
# --------------------------------------------------------------------------- #
def test_MUTATION_with_the_fill_disabled_the_shut_venue_is_unpriced_again(tmp_path):
    """`fill_window_days=0` is the pre-generalisation behaviour (minus the 2025 special
    case).  If this passed WITH the fill on, the tests above would be testing the fixture.
    """
    rows = [("2018-12-31", "2018-12-31", "AAPL", 100.0),
            ("2018-12-31", "2018-12-28", "SAP.DE", 90.0)]
    ps = rc.PriceSource(_csv(tmp_path, rows), anchors=["2018-12-31"], fill_window_days=0)
    assert ps.price("AAPL", "2018-12-31") == 100.0
    assert ps.price("SAP.DE", "2018-12-31") == 90.0, (
        "with the fill off this must come from the ANCHOR layer -- the row's "
        "date_requested IS the anchor, so it was always reachable")


def test_MUTATION_the_fill_is_what_prices_a_name_whose_date_requested_is_NOT_an_anchor(
        tmp_path):
    """The sharper mutation.  The row above is reachable by the anchor layer because its
    `date_requested` happens to be the anchor.  A row pulled under its OWN date -- which is
    how a separate neighbour-day fetch lands, and how `real_prices_2025.csv` is written --
    is reachable ONLY by the fill.
    """
    rows = [("2018-12-31", "2018-12-31", "AAPL", 100.0),
            ("2018-12-28", "2018-12-28", "SAP.DE", 90.0)]
    grid = _csv(tmp_path, rows)
    on = rc.PriceSource(grid, anchors=["2018-12-31"])
    #  THE ASSERTION FIRST, and the fill-off control only where the knob exists.  Building
    #  the `fill_window_days=0` twin up front made this test die with a TypeError against a
    #  PriceSource that has no such parameter -- proving only that the signature moved,
    #  while the sentence it is actually testing ("without the fill layer this row is
    #  unreachable") never got evaluated.  This is the ONE test that shows B2's fill is
    #  load-bearing rather than a restatement of the anchor layer, so it must fail on
    #  content.
    assert on.price("SAP.DE", "2018-12-31") == 90.0, (
        "a neighbour-day row pulled under its OWN date is unreachable -- the anchor layer "
        "alone cannot see it, so there is no per-anchor holiday union")
    try:
        off = rc.PriceSource(grid, anchors=["2018-12-31"], fill_window_days=0)
    except TypeError:
        pytest.skip("no fill_window_days knob to switch off")
    assert off.price("SAP.DE", "2018-12-31") is None


# --------------------------------------------------------------------------- #
#  STRUCTURAL                                                                 #
# --------------------------------------------------------------------------- #
def test_STRUCTURAL_the_window_cannot_reach_a_neighbouring_anchors_body():
    """A fill that reached the previous anchor would price a name a full year early and
    read as a flat return.  The margin is ~360 days, but assert it rather than trust it --
    the anchor list is edited by hand."""
    anchors = [pd.Timestamp(a) for a in rc.DEFAULT_ANCHORS]
    gaps = [(b - a).days for a, b in zip(anchors, anchors[1:])]
    assert min(gaps) > 10 * rc.DEFAULT_FILL_WINDOW_DAYS, \
        f"fill window {rc.DEFAULT_FILL_WINDOW_DAYS}d is not safely inside the anchor " \
        f"spacing (min gap {min(gaps)}d)"


def test_STRUCTURAL_the_hardcoded_2025_only_union_is_gone():
    """`_merge_supplementary` with its `anchor='2025-12-31'` / `fill_from='2025-12-30'`
    defaults was the defect.  A generalised fill plus a surviving special case would be two
    rules disagreeing, so assert the old one is not still there."""
    assert not hasattr(rc.PriceSource, "_merge_supplementary")
    import inspect
    src = inspect.getsource(rc.PriceSource)
    assert "2025-12-30" not in src, "a 2025 literal is still steering the union"
