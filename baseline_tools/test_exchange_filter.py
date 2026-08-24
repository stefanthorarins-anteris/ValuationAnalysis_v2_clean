"""
`exchange_filter` threaded through `pit_universe` -- and what an UNPRICEABLE name resolves
to once the scope opens (offline; no network, no pipeline run).

WHAT WENT WRONG.  `dead_merge.pit_universe` has always taken an `exchange_filter`, and
`None` means the NA1 set (NYSE/NASDAQ/TSX).  Both PIT callers --
`depth_horizon_grid.rank_all_anchors` and `skill_baseline.window_sets` -- passed NOTHING.
So the entire PIT backtest ran NYSE/NASDAQ/TSX-only while the deployed filter ranks and
picks from KOSPI, KOSDAQ, LSE, XETRA, PAR, STO, OSL and BRU.  A backtest scoped narrower
than the thing it grades is a different experiment, not a conservative one.  Measured on
the 08-22 CUR6K panel: 1,767 live sources under NA1 against 4,954 with no restriction.

THE SAFETY QUESTION THIS FILE ANSWERS.  The price refetch is deferred, so opening the scope
brings in 1,277 names on the seven wholly-unpriced venues.  A name with NO price anywhere
resolves to `status='no_buy'` and every derived view EXCLUDES it -- it can NOT reach the
-1.0 total-loss floor, because that needs a buy leg PRESENT and only the eval leg missing.
Verified on the run machine's real grid (2026-08-22): 1,277 of 1,277 `no_buy`, 0 at the
floor, `included()` empty, `beat_rate` n=0.  The tests below pin the MECHANISM behind that.

WHAT THE MECHANISM DOES NOT PROTECT AGAINST, and it is the real hazard: a name priced at the
buy anchor and missing at the eval anchor IS floored at -100% and counted as not-beating,
whether it died or the grid simply has a hole.  On the run machine's grid 29 `.L` and 5
`.DE` names are exactly that for the 2021->2024 window, and 18 of the 29 are priced again at
2025-12-31 -- provably alive.  Those venues arrive with the widened scope.  See
`price_grid_audit`'s interior-hole finding; this file pins that the two cases are genuinely
distinguished by `status`.
"""
import ast
import inspect
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

import dead_merge as dm
import returns_core as rc


# --------------------------------------------------------------------------- #
#  Fixtures                                                                   #
# --------------------------------------------------------------------------- #
#  Two US names, one Korean, one Oslo, one Paris -- the shape of the real question.
_TICKERS = [
    ("AAPL", "NASDAQ"), ("XOM", "NYSE"), ("SHOP.TO", "TSX"),
    ("005930.KS", "KSC"), ("EQNR.OL", "OSL"), ("AIR.PA", "PAR"),
]


def _dmdic():
    return {
        "Tickers_df": pd.DataFrame([{"symbol": s, "exchangeShortName": e}
                                    for s, e in _TICKERS]),
        "cdx_df": pd.DataFrame([{"source": s} for s, _e in _TICKERS]),
    }


NA1_ONLY = ["AAPL", "SHOP.TO", "XOM"]
EVERYTHING = sorted(s for s, _e in _TICKERS)


# --------------------------------------------------------------------------- #
#  BEHAVIOURAL -- the scope                                                   #
# --------------------------------------------------------------------------- #
def test_the_DEFAULT_is_unchanged_and_is_still_NA1_only():
    """The whole change must be inert unless a caller asks for something else."""
    assert dm.pit_universe(_dmdic(), pd.DataFrame(), as_of=None) == NA1_ONLY


def test_ALL_EXCHANGES_opens_the_scope_to_the_universe_actually_scored():
    got = dm.pit_universe(_dmdic(), pd.DataFrame(), as_of=None,
                          exchange_filter=dm.ALL_EXCHANGES)
    assert got == EVERYTHING
    for sym in ("005930.KS", "EQNR.OL", "AIR.PA"):
        assert sym in got, f"{sym} still excluded -- the sentinel did not resolve"


def test_an_explicit_exchange_list_is_honoured_on_both_sides():
    got = dm.pit_universe(_dmdic(), pd.DataFrame(), as_of=None,
                          exchange_filter=("NASDAQ", "KSC"))
    assert got == ["005930.KS", "AAPL"]


def test_the_scope_is_intersected_with_the_actual_cdx_sources():
    """A name on Tickers_df with no cdx row is not scoring-meaningful and must not appear --
    the pre-existing invariant, which the ALL branch has to preserve too."""
    d = _dmdic()
    d["Tickers_df"] = pd.concat([d["Tickers_df"],
                                 pd.DataFrame([{"symbol": "GHOST.OL",
                                                "exchangeShortName": "OSL"}])],
                                ignore_index=True)
    got = dm.pit_universe(d, pd.DataFrame(), as_of=None,
                          exchange_filter=dm.ALL_EXCHANGES)
    assert "GHOST.OL" not in got
    assert got == EVERYTHING


#  `expected` is a SENTINEL STRING rather than `dm.ALL_EXCHANGES` at decoration time.  A
#  parametrize list that touches a new module attribute makes the whole FILE fail to
#  COLLECT against code that lacks it, so not one test in it can report -- a collection
#  error is the most tautological failure there is, and it hides the real ones behind it.
#  Resolved inside the test body instead, where a missing attribute fails ONE test.
_ALL = "<ALL_EXCHANGES>"


@pytest.mark.parametrize("spec,expected", [
    (None, None),
    ("", None),
    ("na1", None),
    ("NA1", None),
    ("all", _ALL),
    ("ALL", _ALL),
    ("NYSE,NASDAQ,LSE", ("NYSE", "NASDAQ", "LSE")),
    (" NYSE , KSC ", ("NYSE", "KSC")),
    (",,", None),
    (["NYSE", "OSL"], ("NYSE", "OSL")),
])
def test_resolve_exchange_filter(spec, expected):
    if expected is _ALL:
        expected = dm.ALL_EXCHANGES
    assert dm.resolve_exchange_filter(spec) == expected


def test_resolve_does_NOT_case_fold_exchange_names():
    """exchangeShortName is vendor data.  Upper-casing 'KSC' happens to be harmless and
    upper-casing a mixed-case venue would silently match nothing, so nothing is folded."""
    assert dm.resolve_exchange_filter("ksc") == ("ksc",)


# --------------------------------------------------------------------------- #
#  BEHAVIOURAL -- what an unpriceable name resolves to                        #
# --------------------------------------------------------------------------- #
class _Grid:
    """A minimal price source: {(ticker, anchor): price}."""

    def __init__(self, anchors, lut):
        self.anchors = list(anchors)
        self._idx = {a: i for i, a in enumerate(self.anchors)}
        self._lut = dict(lut)

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


ANCHORS = ["2021-12-31", "2024-12-31"]


def test_a_name_with_NO_price_anywhere_is_no_buy_and_cannot_reach_the_minus_100_floor():
    """The Korea/Oslo/Paris case.  `no_buy` -> NaN, excluded from every derived view."""
    grid = _Grid(ANCHORS, {("AAPL", "2021-12-31"): 100.0, ("AAPL", "2024-12-31"): 150.0})
    rdf = rc.compute_returns(["AAPL", "005930.KS", "EQNR.OL"], *ANCHORS,
                             price_source=grid)
    by = dict(zip(rdf["ticker"], rdf["status"]))
    assert by == {"AAPL": "ok", "005930.KS": "no_buy", "EQNR.OL": "no_buy"}
    unpriced = rdf[rdf["ticker"] != "AAPL"]
    assert unpriced["total_return"].isna().all()
    assert unpriced["total_return_floor"].isna().all()
    assert not (unpriced["total_return_floor"] == -1.0).any()
    assert list(rc.included(rdf)["ticker"]) == ["AAPL"]


def test_the_DISCRIMINATING_control_a_missing_EVAL_leg_DOES_reach_the_floor():
    """The same fixture CAN produce -1.0, which is what makes the assertion above mean
    something rather than being a property of the harness.

    It is also the live hazard: `.L`/`.DE` names arrive with the widened scope holding a buy
    leg and no eval leg, and this is what happens to them.
    """
    grid = _Grid(ANCHORS, {("DEADCO", "2021-12-31"): 100.0})
    rdf = rc.compute_returns(["DEADCO"], *ANCHORS, price_source=grid)
    row = rdf.iloc[0]
    assert row["status"] == "terminal"
    assert row["total_return_floor"] == -1.0
    assert bool(row["terminal_flag"]) is True
    #  and missing='fail' -- the DEFAULT the headline beat-rate uses -- scores it a miss
    rate, n = rc.beat_rate(rdf, 0.0, threshold=0.10, missing="fail")
    assert (rate, n) == (0.0, 1)


def test_an_all_unpriceable_set_contributes_NOTHING_rather_than_a_zero():
    """n=0 and NaN, not 0.0 -- an average over an empty set must not read as a bad result."""
    grid = _Grid(ANCHORS, {})
    rdf = rc.compute_returns(["005930.KS", "EQNR.OL"], *ANCHORS, price_source=grid)
    rate, n = rc.beat_rate(rdf, 0.05)
    assert n == 0 and np.isnan(rate)
    assert np.isnan(rc.average_return(rdf))


# --------------------------------------------------------------------------- #
#  STRUCTURAL -- the threading itself, which is the whole defect              #
# --------------------------------------------------------------------------- #
def _pit_universe_calls(fn):
    tree = ast.parse(inspect.getsource(fn))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = getattr(f, "attr", None) or getattr(f, "id", None)
        if name == "pit_universe":
            out.append(n)
    return out


@pytest.mark.parametrize("modname,funcname", [
    ("depth_horizon_grid", "rank_all_anchors"),
    ("skill_baseline", "window_sets"),
])
def test_STRUCTURAL_both_PIT_callers_pass_exchange_filter_through(modname, funcname):
    """THE defect, pinned.  Both call sites used to call `pit_universe` with no override, so
    the NA1 default applied silently.  A future edit that drops the keyword again fails
    here."""
    mod = __import__(modname)
    calls = _pit_universe_calls(getattr(mod, funcname))
    assert len(calls) == 1, f"expected one pit_universe call in {modname}.{funcname}"
    kw = {k.arg for k in calls[0].keywords}
    assert "exchange_filter" in kw, (
        f"{modname}.{funcname} calls pit_universe WITHOUT exchange_filter -- the PIT "
        "backtest is silently NA1-only again")


@pytest.mark.parametrize("modname,funcname", [
    ("depth_horizon_grid", "rank_all_anchors"),
    ("depth_horizon_grid", "run_in_pipeline"),
    ("skill_baseline", "window_sets"),
    ("skill_baseline", "run_skill_baseline"),
])
def test_STRUCTURAL_the_parameter_exists_and_defaults_to_the_old_behaviour(modname,
                                                                          funcname):
    """Threaded, and threaded with the OLD default -- so nothing that already ran moves."""
    mod = __import__(modname)
    sig = inspect.signature(getattr(mod, funcname))
    assert "exchange_filter" in sig.parameters
    assert sig.parameters["exchange_filter"].default is None


def test_STRUCTURAL_pipeline_analysis_exposes_the_scope_as_a_config_knob():
    """The automatic run needs a way to ask for it, and the key must default to NA1."""
    import pipeline_analysis as pa
    src = inspect.getsource(pa)
    assert "pit_exchange_filter" in src
    assert "resolve_exchange_filter" in src
