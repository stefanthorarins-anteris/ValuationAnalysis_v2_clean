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
import depth_horizon_grid as dhg


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
#  THESE THREE PINNED THE RETIRED RULE.  They asserted the frozen 2021+ window lists and,
#  by name, that "no buy anchor before 2021 appears (universe-degenerate)" -- which is the
#  `-nrperiods 80`-lifted history-cap rationale Q-28 removed.  With buy2020 promoted on
#  SURVIVORSHIP evidence they failed, and the failing assertion was the retired rule itself,
#  not a defect in the promotion.  Rewritten to assert the INVARIANTS they were standing in
#  for; the frozen lists are gone because restating a set that moves is the defect Q-28 is
#  about, one layer down in a test.
def test_clean_windows_follows_the_ONE_definition_of_the_graded_set():
    """`skill_baseline` runs beside the beat-rate table and the pair is printed as differing
    only by the carve, so its anchor set must BE `dhg.CLEAN_BUY_IDS`, not a copy of it."""
    assert sb.CLEAN_BUY_ANCHORS == [b for w, b in dhg.BUY_ANCHORS if w in dhg.CLEAN_BUY_IDS]
    for cad in (12, 36):
        buys = [b for b, _ in sb.clean_windows(cad)]
        assert buys == sorted(buys), "windows must come out in anchor order"
        assert set(buys) <= set(sb.CLEAN_BUY_ANCHORS)


def test_clean_windows_emits_one_window_per_anchor_that_HAS_an_eval_leg():
    """The real content of the old frozen lists: a clean anchor is graded exactly when its
    buy+cadence lands on an anchor that exists on the grid, and is dropped silently when it
    does not (the latest anchors have no eval leg yet)."""
    anchors = list(rc.DEFAULT_ANCHORS)
    idx = {a: i for i, a in enumerate(anchors)}
    for cad in (12, 36):
        expected = []
        for buy in sb.CLEAN_BUY_ANCHORS:
            i = idx.get(buy)
            if i is None:
                continue
            j = i + cad // 12
            if j < len(anchors):
                expected.append((buy, anchors[j]))
        assert sb.clean_windows(cad, anchors) == expected


def test_no_EXCLUDED_anchor_leaks_into_the_graded_set_at_any_cadence():
    """What `..._excludes_pre_2021` was actually protecting, stated as the thing that is
    still true: an anchor held out in `ANCHOR_EXCLUSION_REASONS` must never be graded.
    'pre-2021' was a proxy for that and stopped being one when buy2020 was promoted."""
    excluded = {b for w, b in dhg.BUY_ANCHORS if w in dhg.ANCHOR_EXCLUSION_REASONS}
    assert excluded, "fixture assumes at least one anchor is held out"
    for cad in (12, 36):
        buys = {b for b, _ in sb.clean_windows(cad)}
        assert not (buys & excluded), f"an excluded anchor is being graded: {buys & excluded}"


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
    # SKIP, do not fail, when the on-disk panel predates the current metric set -- the
    # same posture as test_rebalance_engine::test_certified_reproduction.  Since
    # 2026-07-26/27 the Stage-1 schema gate refuses a panel missing a criterion column
    # (here `CFOlessEarnings`, which replaced the retired uIncomeQuality unity test) and
    # the dead/live merge-content gate refuses a generation-mixed merge.  Both refusals
    # are CORRECT on the 2026-07-13/17 panels; weakening either to keep this test green
    # would defeat the gate it is tripping.  Skips with the reason, and starts asserting
    # again the moment a panel is fetched with the current code.
    try:
        dmdic, merged, registry, ps = sb.load_inputs(
            _P["pickle"], _P["dead"], _P["registry"], _P["prices"], _P["prices_2025"])
    except SystemExit as e:
        pytest.skip("dead/live merge refused -- panel predates the current metric "
                    "set: %s" % str(e)[:160])
    kw = dict(cadence_months=36, pick_n=20, oracle_ns=(3, 20), n_draws=300, seed=0)
    try:
        r1 = sb.run_skill_baseline(dmdic, merged, registry, ps, **kw)
    except KeyError as e:
        if "Stage-1 criterion column" not in str(e):
            raise
        pytest.skip("Stage-1 schema gate refused the panel (older metric set): %s"
                    % str(e)[:160])
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


# --------------------------------------------------------------------------- #
#  THE MEASUREMENT BASIS -- 25.9% must not sit beside 25.0% on a different one #
# --------------------------------------------------------------------------- #
#  THE DEFECT.  `skill_baseline.window_sets` called `reproduce_pit_top` with no
#  `apply_stage1_veto`, and `stage2_pit` defaults it False, while `dhg.rank_all_anchors`
#  defaults it True and stamps VETOED.  So the 2026-08-31 run printed a VETOED beat-rate of
#  25.0% and this module's UN-VETOED filter top-20 of 25.9% as a matched pair, under a
#  `pipeline_analysis` sentence reading "the intended difference between the two is the
#  carve".  The ANCHOR-SET half of that same sentence had already been found false and fixed;
#  the veto half was still open, and this report was the one of the three the stamping batch
#  never reached, so it carried no basis at all.
def _veto_probe(monkeypatch):
    """Records what `window_sets` asks of the veto and of `reproduce_pit_top`."""
    import sys
    import types
    import dead_merge as dm
    import stage2_pit as s2
    seen = {}

    monkeypatch.setattr(sb.dm, "pit_universe", lambda *a, **k: ["A", "B", "C"])

    bo = pd.DataFrame({"source": ["A", "B", "C"], "score": [3.0, 2.0, 1.0]})
    monkeypatch.setattr(sb.s2, "stage1_boscore", lambda *a, **k: bo.copy())

    def _fake_veto(scores_df, bm_df, pool_label=None, cdx_df=None, **kw):
        seen["veto_called"] = True
        seen["veto_pool"] = pool_label
        seen["veto_got_cdx"] = cdx_df is not None
        kept = scores_df[scores_df["source"] != "B"].reset_index(drop=True)
        return kept, {"enabled": True, "applies": True, "n_ejected": 1}

    sv = types.ModuleType("stage1_veto")
    sv.apply_veto = _fake_veto
    monkeypatch.setitem(sys.modules, "stage1_veto", sv)

    def _fake_reproduce(merged, buy, **kw):
        seen["reproduce_veto"] = kw.get("apply_stage1_veto")
        top100 = ["A", "C"] if kw.get("apply_stage1_veto") else ["A", "B", "C"]
        return {"top20": top100[:2], "pool_after_norm": top100,
                "stage1_top100": top100,
                "basis": ("VETOED (stage-1 solvency gate applied, 1 ejected)"
                          if kw.get("apply_stage1_veto") else "un-vetoed")}

    monkeypatch.setattr(sb.s2, "reproduce_pit_top", _fake_reproduce)

    merged = {"BoMetric_df": pd.DataFrame({"source": ["A", "B", "C"],
                                           "date": ["2020-01-01"] * 3}),
              "cdx_df": pd.DataFrame({"source": ["A", "B", "C"],
                                      "date": ["2020-01-01"] * 3})}
    return seen, merged


def test_window_sets_applies_the_veto_BY_DEFAULT(monkeypatch):
    """The default is the fix.  A parameter defaulting False would have left every existing
    caller -- including the pipeline's -- producing the same mismatched pair."""
    seen, merged = _veto_probe(monkeypatch)
    sets, res, _uni = sb.window_sets({}, merged, None, "2020-12-31")
    assert seen.get("veto_called") is True
    assert seen.get("reproduce_veto") is True
    assert seen.get("veto_pool") == "general", "the PIT path must gate the pool production gates"
    assert seen.get("veto_got_cdx") is True, (
        "the ad-hoc penalty bucket's corroborator panel was not passed, so a refusal cannot "
        "be judged -- the same call shape stage2_pit makes")
    assert res["basis"].startswith("VETOED")


def test_the_veto_reaches_the_RUNGS_as_well_as_the_filter(monkeypatch):
    """NOT a detail.  Correction #1 asserts the derived Stage-1 top-100 equals
    `reproduce_pit_top`'s, and an ejection PROMOTES the next name -- so vetoing one side only
    trips that assertion and kills the whole stage.  It is also the right comparator: the
    random floor must be drawn from the pool the filter actually picks from."""
    seen, merged = _veto_probe(monkeypatch)
    sets, res, _uni = sb.window_sets({}, merged, None, "2020-12-31")
    assert "B" not in sets["universe"], "the ejected name is still in the random-draw universe"
    assert set(sets["top100"]) == set(res["stage1_top100"]), (
        "the consistency guard would have fired: rungs and filter on different bases")


def test_the_UN_vetoed_basis_stays_REPRODUCIBLE(monkeypatch):
    """Every skill figure published before 2026-08-31 was un-vetoed.  The escape hatch has to
    work, or the archive becomes unreproducible -- and the report stamps which one it ran."""
    seen, merged = _veto_probe(monkeypatch)
    sets, res, _uni = sb.window_sets({}, merged, None, "2020-12-31",
                                     apply_stage1_veto=False)
    assert seen.get("veto_called") is None, "the veto ran on the explicitly un-vetoed path"
    assert seen.get("reproduce_veto") is False
    assert "B" in sets["universe"]
    assert res["basis"] == "un-vetoed"


def test_the_report_CARRIES_ITS_BASIS():
    """The half the 2026-08-27 stamping batch missed.  A number with no basis on it is what
    let 25.9% be read against 25.0% for four days."""
    import basis_stamp as bstamp
    stamped = _report_skeleton("VETOED (stage-1 solvency gate applied, 1125 ejected)")
    text = sb.format_report(stamped)
    assert "MEASUREMENT BASIS" in text
    assert "VETOED" in text
    assert "Do NOT compare a" in text, "the standing warning is not on the report"
    #  and carve-OFF -- the one difference from the beat-rate table that remains
    assert "carve: OFF" in text


def test_an_UNSTAMPED_result_says_UNSTAMPED_rather_than_saying_nothing():
    """Silence is what the defect looked like.  An absent basis must read as 'I do not know',
    never as a default basis and never as blank."""
    import basis_stamp as bstamp
    bare = _report_skeleton(None)
    bare["basis"] = None
    text = sb.format_report(bare)
    assert bstamp.UNSTAMPED in text


def _report_skeleton(basis):
    """The smallest `run_skill_baseline`-shaped dict `format_report` will render."""
    import basis_stamp as bstamp
    per = {"2020-12-31": {"basis": basis}}
    return {
        "basis": bstamp.of(per),
        "per_anchor_basis": per,
        "config": {"stage1_veto": basis is not None, "cadence_months": 36,
                   "windows": [("2020-12-31", "2023-12-29")], "pick_n": 20,
                   "n_draws": 10, "seed": 0, "threshold": 0.10, "missing": "fail",
                   "exchange_filter": "na1 (NYSE/NASDAQ/TSX)"},
        "filter": {"beat_rate": {"pooled": 0.25, "n": 56},
                   "dollar_return": {"mean_dropped": 0.4, "mean_winsor": 0.35}},
        "oracle": {20: {"beat_rate": 0.9, "n": 20,
                        "dollar_return": {"mean_dropped": 2.0}}},
        "random": {"universe": {"beat_rate": {"mean": 0.1, "p5": 0.0, "p95": 0.2},
                                "filter_beat_percentile": 90.0, "n_contam_in_rung": 0}},
        "ladder": {"beat_rate": [("universe", 0.1), ("filter_top20", 0.25)],
                   "dollar_return": [("universe", 0.1), ("filter_top20", 0.4)]},
        "sanity": {"pick_n": 20, "oracle_beat": 0.9, "filter_beat": 0.25,
                   "random_beat_universe": 0.1, "ordering_holds": True},
    }


def test_the_pipeline_and_the_grid_ask_for_THE_SAME_BASIS():
    """The property the two printed numbers rest on, asserted where it can rot.

    `dhg.rank_all_anchors` defaults the veto ON because it is the grading instrument; this
    module now does too; and the suite's own call site states it rather than inheriting it.
    Any one of those three drifting re-opens the gap."""
    import inspect
    import sys as _sys
    import depth_horizon_grid as dhg
    assert inspect.signature(dhg.rank_all_anchors).parameters["stage1_veto"].default is True
    assert inspect.signature(sb.run_skill_baseline).parameters["stage1_veto"].default is True
    assert inspect.signature(sb.window_sets).parameters["apply_stage1_veto"].default is True

    import pipeline_analysis as pa
    src = inspect.getsource(pa.run_analysis_suite)
    i = src.index("run_skill_baseline(")
    assert "stage1_veto=True" in src[i:i + 400], (
        "the suite's skill-baseline call no longer states its basis explicitly")


def test_the_beat_rate_header_no_longer_ASSERTS_another_stages_basis():
    """The sentence itself.  "the intended difference between the two is the carve" was a
    claim about a stage this function cannot see, made in prose, and it was false for four
    days.  It must point at the stamps instead."""
    import inspect
    import pipeline_analysis as pa
    src = inspect.getsource(pa.beat_rate_vs_urth)
    assert "intended difference between the two is the carve" not in src
    assert "stamp" in src.lower()
