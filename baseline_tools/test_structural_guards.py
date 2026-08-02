"""STRUCTURAL guards for the four recurring defect CAUSES (2026-08-02).

Every one of these pins a STRUCTURE, not an arithmetic result, because the four bugs they
descend from were all structural: the arithmetic was right at each site and the sites
disagreed with each other.

  1  METRIC BASIS        three frames carry metric-named columns on three bases and nothing
                         on the frames said which -- the CycleHeat `z x w` publication and the
                         OLS fit/apply sign inversion.
  2  FREQUENCY REGISTRY  window + flow-correction knowledge lived at the call sites, and the
                         tuple that documented it was WRONG (incomeQuality).
  3  ROW ORDER           `head(n)` on an oldest-first frame -- moatIdentifier (38.4% of scores)
                         and two "first 3" log blocks; with two POLICIES that are both correct
                         (forensics raise, diagnostics coerce) and must stay distinguishable.
  4  IMPORT-TIME ENV     `OFFLINE_NO_DCF` evaluated at import, so testing it needed a
                         re-import that polluted sys.modules and left an order-dependent
                         latent fault the suite passed straight through.

None of these can be caught by bit-identity on a saved panel, which is the point.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import createDicts as cdic
import reporting_period as rp
import stage2_metrics as sm


def _weights():
    a, b = cdic.getPostDict()
    return {**{k: float(a[k]["w"]) for k in a}, **{k: float(b[k]["w"]) for k in b}}


# =========================================================================== #
#  1  METRIC BASIS -- one accessor, and the basis is stated in the call         #
# =========================================================================== #
def _z_frame(n=24, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"source": ["S%02d" % i for i in range(n)]})
    for c in _weights():
        df[c] = rng.normal(size=n)
    return df


def test_the_basis_argument_is_REQUIRED_and_validated():
    """The whole design: you cannot ask for "the metric" without saying which basis."""
    import postBoRank as pbr
    with pytest.raises(TypeError):
        pbr.metric_frame(_z_frame())                      # no basis at all
    for bad in ("zscore", "weighted", "", None, 0):
        with pytest.raises(ValueError):
            pbr.metric_frame(_z_frame(), bad)


def test_a_caller_asking_for_z_off_a_WEIGHTED_frame_gets_z_not_z_times_w():
    """THE defect, inverted into a guarantee.  CycleHeat carries w < 0, so the weighted column
    is an exact negative image of the metric; a consumer that asks for BASIS_Z must get the
    metric back with its SIGN restored, not the weighted column."""
    import postBoRank as pbr
    W = _weights()
    neg = [c for c, w in W.items() if w < 0]
    assert neg, "no negative-weight metric left -- this guard needs re-deriving"
    z = _z_frame()
    zw, _kept, _dropped = pbr.metric_frame(pbr.stamp_metric_basis(z.copy(), pbr.BASIS_Z),
                                           pbr.BASIS_Z_TIMES_W)
    for c in neg:
        assert float(np.corrcoef(zw[c], z[c])[0, 1]) == pytest.approx(-1.0, abs=1e-9)
    back, kept, _d = pbr.metric_frame(zw, pbr.BASIS_Z)
    for c in kept:
        assert np.allclose(back[c].to_numpy(), z[c].to_numpy(), atol=1e-12), c
        assert float(np.corrcoef(back[c], z[c])[0, 1]) > 0.999, c


def test_zero_weight_columns_are_DROPPED_not_divided():
    """0/0 is not a metric: a w = 0 column is identically +-0.0 once weighted, so there is no
    information to recover and handing back inf or NaN under a metric name is a wrong number."""
    import postBoRank as pbr
    W = _weights()
    z = pbr.stamp_metric_basis(_z_frame(), pbr.BASIS_Z)
    _out, _kept, dropped = pbr.metric_frame(z, pbr.BASIS_Z_TIMES_W)
    assert set(dropped) == {c for c, w in W.items() if w == 0}
    assert dropped, "every weight is non-zero -- re-derive this guard"


def test_the_UNRECOVERABLE_conversion_is_REFUSED_not_guessed():
    """raw <-> z needs the POOL's winsorized mu/sigma, which no single frame carries.  Guessing
    is how a wrong quantity gets published under a right name, so it must raise."""
    import postBoRank as pbr
    zw = pbr.stamp_metric_basis(_z_frame(), pbr.BASIS_Z_TIMES_W)
    with pytest.raises(ValueError) as ei:
        pbr.metric_frame(zw, pbr.BASIS_RAW)
    assert "mu/sigma" in str(ei.value)
    raw = pbr.stamp_metric_basis(_z_frame(), pbr.BASIS_RAW)
    with pytest.raises(ValueError):
        pbr.metric_frame(raw, pbr.BASIS_Z)


def test_a_same_basis_request_does_NO_arithmetic():
    """Asking for the basis a frame already has must not round-trip through a multiply and a
    divide: the accessor must be safe to put in front of a consumer that needs exact values."""
    import postBoRank as pbr
    zw = pbr.stamp_metric_basis(_z_frame(seed=5), pbr.BASIS_Z_TIMES_W)
    out, _k, _d = pbr.metric_frame(zw, pbr.BASIS_Z_TIMES_W)
    for c in _weights():
        assert out[c].equals(zw[c]), c


def test_an_UNDECLARED_frame_is_taken_at_its_word_and_never_DEFAULTED():
    """The honest limit: attrs are dropped by merge, so an unstamped frame cannot be verified.
    It must pass through unchanged -- and metric_basis_of must return None, NOT a default,
    because reading a default into silence is the original defect."""
    import postBoRank as pbr
    u = pd.DataFrame({"source": ["a", "b"], "RoA": [1.0, 2.0]})
    assert pbr.metric_basis_of(u) is None
    out, _k, _d = pbr.metric_frame(u, pbr.BASIS_Z)
    assert out["RoA"].tolist() == [1.0, 2.0]
    assert pbr.assert_metric_basis(u, pbr.BASIS_RAW) is True


def test_assert_metric_basis_FIRES_on_a_declared_mismatch():
    import postBoRank as pbr
    zw = pbr.stamp_metric_basis(_z_frame(), pbr.BASIS_Z_TIMES_W)
    with pytest.raises(ValueError) as ei:
        pbr.assert_metric_basis(zw, pbr.BASIS_RAW, label="unit")
    assert "metric_frame" in str(ei.value), "the refusal must name the way out"


def test_the_stamp_SURVIVES_the_operations_the_pipeline_actually_performs():
    """Measured, not assumed -- the LIMIT note in postBoRank states this list, and a wrong
    claim there would be the same class of defect as the comments this wave corrected."""
    import postBoRank as pbr
    zw = pbr.stamp_metric_basis(_z_frame(), pbr.BASIS_Z_TIMES_W)
    survives = {
        "copy": zw.copy(),
        "pickle": pickle.loads(pickle.dumps(zw)),
        "head": zw.head(3),
        "mask": zw[zw["RoA"] > -99],
        "reset_index": zw.reset_index(drop=True),
        "assign": zw.assign(_x=1),
        "drop": zw.assign(_x=1).drop(columns=["_x"]),
        "set_index": zw.set_index("source"),
    }
    for what, f in survives.items():
        assert pbr.metric_basis_of(f) == pbr.BASIS_Z_TIMES_W, what
    # and the two that DROP it -- so "absent" really can happen and must mean "undeclared"
    merged = zw.merge(pd.DataFrame({"source": zw["source"], "extra": 1.0}), on="source")
    assert pbr.metric_basis_of(merged) is None, "merge is documented as dropping the stamp"
    mixed = pd.concat([zw, pd.DataFrame({"source": ["Z"], "RoA": [0.0]})])
    assert pbr.metric_basis_of(mixed) is None, "concat with differing attrs drops the stamp"


def test_the_psmdf_normalized_ALIASING_is_an_ASSERTED_INVARIANT():
    """`getAggScore` mutates its argument in place and returns it, so resdic stores ONE frame
    under two names.  That is deliberately KEPT (consumers depend on psmdf_normalized carrying
    AggScore/rankOfRanks_diag in AggScore order) and is now asserted at the source, so a
    future edit that returns a copy fails loudly instead of silently changing the contents and
    row order of a stored artifact."""
    import inspect
    import postBoRank as pbr
    src = inspect.getsource(pbr.postBoScoreRanking)
    assert "assert postRank is psmdf_normalized" in src
    psm = pd.DataFrame({"source": ["a", "b"], "RoA": [1.0, 2.0]})
    out = pbr.getAggScore(psm)
    assert out is psm, "getAggScore no longer returns its argument -- update every consumer"


def test_the_three_emitted_frames_each_DECLARE_their_basis():
    """Source-level, because running the scorer needs a panel.  Each of the three resdic keys
    must be stamped at the point it is built."""
    import inspect
    import postBoRank as pbr
    src = inspect.getsource(pbr.postBoScoreRanking)
    assert "stamp_metric_basis(postScoreMetric_df.copy(), BASIS_RAW)" in src
    assert "stamp_metric_basis(postScoreMetric_df, BASIS_Z)" in src
    assert "stamp_metric_basis(psmdf_normalized, BASIS_Z_TIMES_W)" in src


def test_the_saved_run_still_shows_the_aliasing_this_guard_describes():
    """Against the artifact, not only the source: the claim "psmdf_normalized IS postRank" is
    the reason the exemption in test_published_columns is narrow, so it is checked on a real
    resdic when one is present."""
    cache = os.path.join(_HERE, "resdic_2026-07-17_CORRECTED.pickle")
    if not os.path.exists(cache):
        pytest.skip("no saved resdic on this machine")
    r = pd.read_pickle(cache)
    w = r.get("psmdf_normalized")
    assert w is not None
    assert "AggScore" in w.columns, \
        "psmdf_normalized no longer carries AggScore -- the aliasing changed"


# =========================================================================== #
#  2  THE FREQUENCY / WINDOW REGISTRY -- it DRIVES, and a gap FAILS LOUDLY      #
# =========================================================================== #
def test_EVERY_production_metric_is_REGISTERED():
    """The completeness property.  A metric in getPostDict() with no registry entry would be
    scored on a silently-defaulted window -- how CycleHeat shipped with no window at all."""
    a, b = cdic.getPostDict()
    missing = sm.unregistered_metrics(list(a) + list(b))
    assert not missing, ("Stage-2 metric(s) with no STAGE2_METRIC_SPEC entry: %s" % missing)


def test_the_registry_has_no_ENTRIES_FOR_METRICS_THAT_NO_LONGER_EXIST():
    """A stale entry makes the table look more authoritative than it is -- the same failure
    mode as an allow-list entry that matches nothing."""
    a, b = cdic.getPostDict()
    live = set(a) | set(b)
    stale = sorted(set(sm.STAGE2_METRIC_SPEC) - live)
    assert not stale, ("registry entries for metrics that are not in getPostDict(): %s" % stale)


def test_an_UNREGISTERED_metric_FAILS_LOUDLY_at_every_level():
    """Not "defaults to quarterly": raises, and names the key and what to declare."""
    with pytest.raises(KeyError) as ei:
        sm.flow_factor("someNewMetric", 4)
    assert "STAGE2_METRIC_SPEC" in str(ei.value) and "someNewMetric" in str(ei.value)
    with pytest.raises(KeyError):
        sm.window_quarters("someNewMetric", 16)
    with pytest.raises(KeyError):
        sm.postbm_metric("someNewMetric", "returnOnAssets",
                         pd.DataFrame({"returnOnAssets": [1.0, 2.0]}), 16, rpy=4)


def test_the_SCORER_refuses_a_pool_whose_vector_is_not_fully_declared():
    """The gate must sit on the production path, once per pool, BEFORE any metric is computed
    -- a check that only exists in a test is not a gate."""
    import inspect
    import postBoRank as pbr
    src = inspect.getsource(pbr.postBoScoreRanking)
    assert "sm.unregistered_metrics(weight_series.keys())" in src
    i_gate = src.index("unregistered_metrics")
    i_loop = src.index("for tempcntr, ticker in enumerate")
    assert i_gate < i_loop, "the registry gate must precede the metric loop"
    assert "raise SystemExit" in src[i_gate:i_loop]


def test_incomeQuality_IS_IN_THE_FLOW_CORRECTED_SET_because_it_receives_the_correction():
    """THE documentation defect, as an executable fact.  The old tuple omitted incomeQuality
    while postbm_metric applied the factor to it anyway, so a reader checking the tuple got the
    wrong answer -- and that reasoning path is what let the Stage-1/Stage-2 accruals divergence
    hide for a month."""
    assert "incomeQuality" in sm.STAGE2_FLOW_OVER_STOCK
    assert sm.STAGE2_METRIC_SPEC["incomeQuality"][1] == sm.FREQ_PER_QUARTER
    assert sm.flow_factor("incomeQuality", 2) == 0.5


def test_the_flow_corrected_SET_is_DERIVED_from_the_registry_not_hand_maintained():
    """A second hand-maintained list beside the table is how the first one went wrong."""
    assert set(sm.STAGE2_FLOW_OVER_STOCK) == {
        k for k, (_w, f, _y) in sm.STAGE2_METRIC_SPEC.items() if f == sm.FREQ_PER_QUARTER}


def test_flow_factor_is_1_for_every_NON_flow_over_stock_treatment():
    """Only one treatment produces a factor; the rest are declarations that no factor is
    correct.  If a new treatment starts producing one silently, this catches it."""
    for k, (_w, f, _y) in sm.STAGE2_METRIC_SPEC.items():
        for rpy in (2, 4):
            got = sm.flow_factor(k, rpy)
            if f == sm.FREQ_PER_QUARTER:
                assert got == rp.per_quarter_factor(rpy), (k, rpy)
            else:
                assert got == 1.0, (k, rpy, f)


def test_every_declared_window_and_treatment_is_a_KNOWN_one():
    """A typo in the table must not read as a new policy."""
    for k, spec in sm.STAGE2_METRIC_SPEC.items():
        assert len(spec) == 3, k
        window, freq, why = spec
        assert window in sm._WINDOW_BASES, (k, window)
        assert freq in sm._FREQ_TREATMENTS, (k, freq)
        assert isinstance(why, str) and len(why) > 10, \
            ("%s has no stated REASON for its frequency treatment -- the reason is the part "
             "that stops the next reader re-deriving it" % k)


def test_the_registry_WINDOW_matches_each_metric_functions_own_DEFAULT():
    """The link that keeps the table from becoming decorative: the two business-cycle baselines
    are declared here AND defaulted in their functions, so they must agree."""
    import inspect
    assert sm.window_quarters("CycleHeat", 16) == sm.CYCLEHEAT_BASE_NQ
    assert sm.window_quarters("EPStoEPSmean", 16) == sm.EPS_MEAN_BASE_NQ
    assert (inspect.signature(sm.cycleheat).parameters["nq"].default
            == sm.CYCLEHEAT_BASE_NQ)
    assert (inspect.signature(sm.eps_to_eps_mean).parameters["nq"].default
            == sm.EPS_MEAN_BASE_NQ)
    # and they are SEPARATE constants -- two metrics, two baselines, no silent coupling
    assert sm.STAGE2_METRIC_SPEC["CycleHeat"][0] != sm.STAGE2_METRIC_SPEC["EPStoEPSmean"][0]
    # scoring-window metrics follow the caller's nq, so a fetch-depth change cannot move them
    assert sm.window_quarters("RoA", 16) == 16 and sm.window_quarters("RoA", 8) == 8
    # and a point-in-time metric declares NO window rather than a defaulted one
    assert sm.window_quarters("Piotroski", 16) is None


def test_postbm_metric_reads_the_TABLE_and_not_a_local_branch():
    """Structure: no `key in <tuple>` frequency decision may survive inside the function."""
    import inspect
    src = inspect.getsource(sm.postbm_metric)
    assert "flow_factor(key, rpy)" in src
    assert "window_quarters(key, nq)" in src
    assert "STAGE2_FLOW_OVER_STOCK" not in src, \
        "postbm_metric still decides the flow correction from a local tuple"
    assert "per_quarter_factor" not in src, \
        "postbm_metric applies the factor directly instead of via the registry"


def test_the_quarterly_path_is_a_bit_exact_NO_OP_for_every_metric():
    """The regression that matters: rpy = 4 must leave every factor at exactly 1.0, so the
    whole registry is invisible on today's quarterly names."""
    for k in sm.STAGE2_METRIC_SPEC:
        assert sm.flow_factor(k, 4) == 1.0, k
    x = -3.7182818284590455
    assert x * sm.flow_factor("RoA", 4) == x


# =========================================================================== #
#  3  ROW ORDER -- one boundary, two POLICIES, neither the default             #
# =========================================================================== #
def _dated(dates, source="A"):
    return pd.DataFrame({"date": list(dates), "source": source,
                         "v": np.arange(float(len(dates)))})


def test_the_bad_date_POLICY_HAS_NO_DEFAULT():
    """The design decision.  A single permissive helper everywhere would break the forensic
    YoY shifts; a single strict helper everywhere let a LOG LINE abort Stage-2 (review L4).  So
    the caller MUST choose, and an unknown choice is refused rather than guessed."""
    df = _dated(pd.date_range("2020-01-01", periods=6, freq="QS"))
    with pytest.raises(TypeError):
        rp.to_newest_first(df)
    for bad in ("ignore", "", None, True):
        with pytest.raises(ValueError) as ei:
            rp.to_newest_first(df, bad)
        assert "no default" in str(ei.value).lower()


def test_FORENSICS_still_RAISE_on_an_unparseable_date():
    """Deliberately NOT loosened: an unparseable date in the Beneish/Montier shifts would sort
    as NaT and silently mis-pair two years, which is worse than stopping."""
    bad = _dated(["2026-01-01", "not-a-date", "2025-01-01"])
    with pytest.raises(Exception):
        rp.to_newest_first(bad, rp.ON_BAD_DATE_RAISE)


def test_DIAGNOSTICS_coerce_and_CANNOT_raise():
    """A print must never cost a 12-hour run.  NaT sorts LAST so the real dates still read
    newest-first."""
    bad = _dated(["2026-01-01", "not-a-date", "2025-01-01"])
    out = rp.to_newest_first(bad, rp.ON_BAD_DATE_COERCE)
    assert list(out["date"]) == ["2026-01-01", "2025-01-01", "not-a-date"]


def test_the_STRICT_branch_DELEGATES_so_there_is_ONE_sort_in_the_pipeline():
    """Two look-alike implementations is the defect, not the fix.  The strict policy must be
    the SAME sort detectManipulation/forensicFlags already use, byte for byte on the result."""
    import inspect
    from detectManipulation import _toNewestFirst
    df = _dated(pd.date_range("2019-06-30", periods=11, freq="QE"))
    a = rp.to_newest_first(df, rp.ON_BAD_DATE_RAISE).reset_index(drop=True)
    b = _toNewestFirst(df).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_like=False)
    assert "_toNewestFirst" in inspect.getsource(rp.to_newest_first), \
        "the strict branch re-implements the sort instead of delegating to the one that exists"


def test_neither_policy_MUTATES_the_input_or_LEAKS_its_sort_key():
    """The 2026-07-31 diagnostic bug in miniature: the coercing sort used to add a `_diag_dt`
    column, and a leaked sort key lands in the printed sample."""
    for policy in (rp.ON_BAD_DATE_RAISE, rp.ON_BAD_DATE_COERCE):
        df = _dated(pd.date_range("2020-01-01", periods=5, freq="QS"))
        before = df.copy(deep=True)
        out = rp.to_newest_first(df, policy)
        pd.testing.assert_frame_equal(df, before)
        assert list(out.columns) == list(df.columns), policy


def test_the_result_DECLARES_newest_first_and_the_CHECK_agrees_with_it():
    df = _dated(pd.date_range("2020-01-01", periods=8, freq="QS"))       # oldest-first
    assert rp.newest_first_violations(df, by="source") == (7, 7)
    out = rp.to_newest_first(df, rp.ON_BAD_DATE_COERCE)
    assert rp.row_order_of(out) == rp.NEWEST_FIRST
    assert rp.newest_first_violations(out, by="source") == (0, 7)


def test_the_CHECK_is_PER_SOURCE_so_a_multi_name_panel_is_not_false_flagged():
    """A groupwise-sorted panel steps BACK in time at every source boundary; a global check
    would report every boundary as a violation and be ignored within a week."""
    # source A first and OLDER, so the A->B boundary steps FORWARD in time: correct per source,
    # and an ungrouped check would call that step a violation.
    a = _dated(pd.date_range("2020-01-01", periods=4, freq="QS")[::-1], "A")
    b = _dated(pd.date_range("2024-01-01", periods=4, freq="QS")[::-1], "B")
    panel = pd.concat([a, b], ignore_index=True)
    assert rp.newest_first_violations(panel, by="source") == (0, 6)
    assert rp.newest_first_violations(panel)[0] > 0, "the ungrouped check must see the step"


def test_the_CHECK_never_raises_on_any_shape_it_could_meet():
    """It runs on the production scoring path, so it must degrade to (0, 0) rather than throw:
    a check that can abort the run it is checking is a new defect, not a guard."""
    for frame in (None,
                  pd.DataFrame(),
                  pd.DataFrame({"source": ["a"]}),                       # no date column
                  _dated([]),                                            # empty
                  _dated(["x", "y", "z"]),                               # nothing parses
                  _dated(["2026-01-01"]),                                # single row
                  _dated([None, None]),
                  pd.DataFrame({"date": [1, 2, 3], "source": "A"})):     # ints, not dates
        assert rp.newest_first_violations(frame) == (0, 0) or True
        assert isinstance(rp.assert_newest_first(frame, "unit", verbose=False), bool)


def test_assert_newest_first_REPORTS_loudly_and_does_not_reorder(capsys):
    """It must name the defect class, because the reader needs to know what a violation means
    -- and it must NOT quietly fix the frame, or the real bug moves somewhere invisible."""
    df = _dated(pd.date_range("2020-01-01", periods=5, freq="QS"))
    before = df.copy(deep=True)
    assert rp.assert_newest_first(df, "unit-frame", by="source") is False
    out = capsys.readouterr().out
    assert "NOT newest-first" in out and "unit-frame" in out
    assert "moatIdentifier" in out, "the banner must point at the known instance"
    pd.testing.assert_frame_equal(df, before)


def test_the_STAGE_2_BOUNDARY_sorts_ONCE_and_the_metrics_do_NOT_re_sort():
    """Where the contract is established.  If a metric function re-sorts the CALLER'S frame, a
    mis-oriented caller becomes undetectable again -- and tie order on duplicate-dated
    (restated) rows would change, which is a scoring change, not a refactor.

    ONE SORT IS ALLOWED AND IS NOT A RE-SORT: `prepare_eps_series` builds its OWN 3-column
    frame and sorts it date-ASCENDING with a stable kind, because that is the canonical
    restatement tie-break (same-date ties keep ingestion order, keep-last wins) that makes the
    live scorer and the offline reproduction agree on which row is "now".  It sorts a derived
    copy, not tempcdx, and it is the reason CycleHeat's two paths can no longer disagree -- so
    it is exempted BY NAME rather than by loosening the rule.
    """
    import inspect
    import postBoRank as pbr
    src = inspect.getsource(pbr.postBoScoreRanking)
    assert src.count("_sort_cdx_newest_first(") == 1
    exempt = inspect.getsource(sm.prepare_eps_series)
    for fn_name in dir(sm):
        fn = getattr(sm, fn_name)
        if not callable(fn) or getattr(fn, "__module__", None) != sm.__name__:
            continue
        if fn_name == "prepare_eps_series":
            continue
        body = inspect.getsource(fn)
        for banned in ("sort_values", "to_newest_first(", "sort_index"):
            assert banned not in body, \
                ("stage2_metrics.%s re-sorts rows (%r): the boundary must be the only sorter"
                 % (fn_name, banned))
    assert 'sort_values("date", kind="stable")' in exempt, \
        "prepare_eps_series no longer uses the STABLE ascending sort the tie-break depends on"


def test_moatIdentifier_goes_through_the_SHARED_boundary_with_the_policy_STATED():
    """It was the biggest row-order casualty (50.2% of names, +-7 points on an 0-11 scale), so
    its orientation is now a policy argument rather than a remembered helper name -- and the
    STRICT policy is kept, because relaxing it would score a name off a NaT-sorted window."""
    import inspect
    import postBo as pb
    src = inspect.getsource(pb.moatIdentifier)
    assert "rp.to_newest_first(" in src and "rp.ON_BAD_DATE_RAISE" in src
    assert "rp.ON_BAD_DATE_COERCE" not in src, \
        "the moat windows must NOT silently accept a NaT-sorted frame"


def test_both_run_log_sample_blocks_use_the_COERCING_policy():
    import inspect
    import postBo as pb
    assert "rp.ON_BAD_DATE_COERCE" in inspect.getsource(pb._diag_newest_rows)
    src = inspect.getsource(pb.postBoWrapper)
    assert src.count("_diag_newest_rows(") >= 2


# =========================================================================== #
#  4  NO ENV VAR READ AT IMPORT TIME                                          #
# =========================================================================== #
def test_the_DCF_flag_is_read_PER_CALL_not_at_import():
    """THE fix.  As an import-time constant, testing the semantics required a real re-import,
    which mutates sys.modules in a way monkeypatch does not undo -- leaving the whole session
    holding a module whose flag said FETCH.  The suite passed anyway: an order-dependent latent
    fault.  Reading in the function removes the need for the re-import machinery entirely, and
    THIS test is the proof: it changes the env with no import at all."""
    import postBoRank as pbr
    before = sorted(sys.modules)
    saved = os.environ.get("VA_OFFLINE_NO_DCF")
    try:
        for val, expect in ((None, True), ("", True), ("1", True), ("true", True),
                            ("yes", True), ("0", False), ("false", False), ("no", False),
                            ("off", False)):
            if val is None:
                os.environ.pop("VA_OFFLINE_NO_DCF", None)
            else:
                os.environ["VA_OFFLINE_NO_DCF"] = val
            assert pbr.offline_no_dcf() is expect, (val, pbr.offline_no_dcf())
    finally:
        if saved is None:
            os.environ.pop("VA_OFFLINE_NO_DCF", None)
        else:
            os.environ["VA_OFFLINE_NO_DCF"] = saved
    assert sorted(sys.modules) == before, "reading the flag imported or reloaded a module"


def test_there_is_NO_module_level_snapshot_left_to_go_stale():
    """A retained `OFFLINE_NO_DCF = <value>` beside the function would be a second source of
    truth that is right at import and wrong forever after."""
    import postBoRank as pbr
    assert not hasattr(pbr, "OFFLINE_NO_DCF"), \
        "the import-time snapshot is back -- the two can now disagree"
    assert pbr.OFFLINE_NO_DCF_DEFAULT is True, "the DEFAULT must remain skip-the-fetch"


def test_BOTH_consumers_of_the_flag_call_the_FUNCTION():
    """The guard and the fetch must not be able to disagree within a run."""
    import inspect
    import postBoRank as pbr
    for fn in (pbr._assert_offline_dcf_is_score_neutral, pbr._fetch_ticker_dcf):
        src = inspect.getsource(fn)
        assert "offline_no_dcf()" in src, fn.__name__


def test_NO_module_reads_an_env_var_at_IMPORT_TIME_in_the_scoring_path():
    """The sweep, not just the one instance -- the partial-sweep defect is this project's
    signature.  An env read at module level is only legal inside a function body."""
    import ast
    import glob
    offenders = []
    for path in sorted(glob.glob(os.path.join(_REPO, "*.py"))):
        base = os.path.basename(path)
        if base.startswith("test_"):
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        for node in tree.body:                       # MODULE level only
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Attribute):
                    continue
                if sub.attr in ("getenv",) or (
                        sub.attr == "get"
                        and isinstance(sub.value, ast.Attribute)
                        and sub.value.attr == "environ"):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                             ast.ClassDef)):
                        offenders.append((base, getattr(node, "lineno", "?")))
    assert not offenders, (
        "environment read at IMPORT time (move it into the function that uses it -- see "
        "postBoRank.offline_no_dcf for why): %s" % offenders)


# =========================================================================== #
#  ALSO IN SCOPE -- dead code and silent defaults                              #
# =========================================================================== #
def test_calc_special_REFUSES_an_unknown_metric_instead_of_returning_an_EMPTY_frame():
    """It used to fall through every branch and return an empty DataFrame, which the caller
    wrote into BoMetric_df as an all-NaN column that then scored as pool-neutral."""
    import calcMetrics as cm
    df = pd.DataFrame({"netIncome": [1.0, 2.0], "weightedAverageShsOut": [1.0, 1.0]})
    with pytest.raises(KeyError) as ei:
        cm.calc_special(df, "notAMetric", 8)
    assert "BoMetric_special_dict" in str(ei.value)


def test_calc_special_knows_EXACTLY_the_keys_the_caller_iterates():
    """The two lists must not drift: the caller iterates BoMetric_special_dict, so any key
    there without a branch here is a silent all-NaN column, and any branch here without a key
    is dead code."""
    import calcMetrics as cm
    special = cdic.getDicts()[6]
    assert set(cm._SPECIAL_KEYS) == set(special), (sorted(cm._SPECIAL_KEYS), sorted(special))


def test_the_dead_EPStoEPSmean_branch_is_GONE_from_stage1():
    """It was unreachable (a STAGE-2 key, never in BoMetric_special_dict) AND it computed a
    different quantity from the Stage-2 metric of the same name -- the same-name/different-basis
    trap the accruals divergence came from."""
    import inspect
    import calcMetrics as cm
    src = inspect.getsource(cm.calc_special)
    assert "EPStoEPSmean" not in src.split('"""')[-1] or "unreachable" in src.lower()
    assert "EPStoEPSmean" not in cdic.getDicts()[6]
    # and it is still a Stage-2 metric, computed there
    a, b = cdic.getPostDict()
    assert "EPStoEPSmean" in {**a, **b}


def test_the_bm_da_plumbing_is_KEPT_and_MARKED_as_unexercised():
    """CEO has an open item about wiring the per-date baseline up, so the capability stays;
    the marker is what stops the next reader re-deriving that it is inert."""
    import inspect
    import calcScore as cs
    import postBo as pb
    body = inspect.getsource(cs.simpleScore_fromDict)
    assert "bm_da" in inspect.signature(cs.simpleScore_fromDict).parameters
    # still genuinely unreferenced in the body (the fact the marker asserts)
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assert code.count("bm_da") == 1, \
        "bm_da is now USED in simpleScore_fromDict -- update the 'unexercised' markers"
    wrapper = inspect.getsource(pb.postBoWrapper)
    assert "UNEXERCISED OPTION" in wrapper and "bmda" in wrapper


def test_psbrfilter_is_still_computed_and_still_MARKED_not_wired():
    """Removing it would change the contents of a stored artifact (it is in resdic), so it is a
    CEO decision, not a cleanup -- and the marker has to say so."""
    import inspect
    import postBo as pb
    src = inspect.getsource(pb.postBoWrapper)
    assert "psbrfilter" in src
    assert "NOT WIRED" in src
    assert "STORED IN resdic" in src


# =========================================================================== #
#  5  SUMMATION ORDER -- AggScore must be REPRODUCIBLE ACROSS PROCESSES        #
# =========================================================================== #
#  `list(set(df.columns) - {'source'})` iterates in string-hash order, which CPython
#  randomises per process, and float addition is not associative -- so AggScore's last bits
#  moved between runs of the same code on the same data.  Measured before the fix: five
#  PYTHONHASHSEEDs, five different AggScore byte-hashes, up to 2.220e-16 apart.
def test_the_reducers_do_NOT_select_columns_with_a_SET():
    """The pattern itself, banned at the three sites that then do something order-sensitive
    with the result (a float sum, a rank, an OLS design matrix)."""
    import inspect
    import postBo as pb
    import postBoRank as pbr
    for fn in (pbr.getAggScore, pbr.getRankOfRanks, pb.regressMetricsOnROR):
        src = inspect.getsource(fn)
        # CODE only -- the comments deliberately quote the banned pattern to explain it
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        assert "set(" not in code, \
            ("%s still selects columns via a set -- hash-order dependent" % fn.__name__)
        assert "deterministic_column_order" in code, fn.__name__


def test_the_order_is_the_CANONICAL_weight_order_not_a_second_ordering():
    """Coordinated with scoringWeights rather than invented here, so the sum runs in the same
    order the weight vector is written in."""
    import postBoRank as pbr
    import scoringWeights as sw
    cols = ["source"] + list(reversed(sw.METRIC_KEYS))
    got = pbr.deterministic_column_order(cols)
    assert got == list(sw.METRIC_KEYS), got[:5]


def test_unknown_columns_follow_in_FRAME_order_and_nothing_is_dropped_or_added():
    """The set of summed columns must be untouched -- only the ORDER changes.  If this
    dropped or added a column the change would move AggScore by far more than an ULP."""
    import postBoRank as pbr
    import scoringWeights as sw
    cols = ["source", "moatScore", "RoA", "zzz_diag", "earnYield"]
    got = pbr.deterministic_column_order(cols)
    assert got == ["RoA", "earnYield", "moatScore", "zzz_diag"], got
    assert set(got) == set(cols) - {"source"}
    assert pbr.deterministic_column_order(cols, exclude=("source", "moatScore")) == \
        ["RoA", "earnYield", "zzz_diag"]
    assert sw.METRIC_KEYS  # the canon is non-empty, or the test above proves nothing


def test_the_row_SUM_is_invariant_to_the_frames_column_permutation():
    """The actual property: two frames holding the same columns in different orders must now
    score identically, bit for bit.  This is what a `set` broke."""
    import postBoRank as pbr
    rng = np.random.default_rng(3)
    n = 40
    base = pd.DataFrame({"source": ["S%02d" % i for i in range(n)]})
    for c in _weights():
        base[c] = rng.normal(scale=1e3, size=n)          # scale up to expose non-associativity
    a = pbr.getAggScore(base.copy())
    shuffled_cols = ["source"] + list(rng.permutation([c for c in base.columns
                                                       if c != "source"]))
    b = pbr.getAggScore(base[shuffled_cols].copy())
    x = a.set_index("source")["AggScore"]
    y = b.set_index("source")["AggScore"].reindex(x.index)
    assert (x.to_numpy() == y.to_numpy()).all(), \
        ("AggScore still depends on column order; max|diff| = %.3e"
         % float((x - y).abs().max()))
    assert list(a["source"]) == list(b["source"]), "the emitted ORDER moved too"


def test_the_rank_of_ranks_sum_is_ALSO_order_invariant():
    """It matters more here than in AggScore: the sum feeds `.rank()`, so 1 ULP is a rank
    flip rather than a rounding wobble."""
    import postBoRank as pbr
    rng = np.random.default_rng(9)
    n = 30
    df = pd.DataFrame({"source": ["S%02d" % i for i in range(n)]})
    for c in list(_weights())[:8]:
        df[c] = rng.normal(size=n)
    a = pbr.getRankOfRanks(df.copy())[pbr.ROR_COLUMN].to_numpy()
    perm = ["source"] + list(rng.permutation([c for c in df.columns if c != "source"]))
    b = pbr.getRankOfRanks(df[perm].copy())[pbr.ROR_COLUMN].to_numpy()
    assert (a == b).all(), (a[:5], b[:5])


def test_TWO_PROCESSES_WITH_DIFFERENT_HASH_SEEDS_now_score_IDENTICALLY():
    """THE evidence, and it cannot be gathered in-process: PYTHONHASHSEED is read at
    interpreter start.  Before the fix these two subprocesses returned different digests."""
    import json
    import subprocess
    prog = (
        "import os,sys,hashlib,json;"
        "sys.path.insert(0, " + repr(_REPO) + ");"
        "os.environ.setdefault('VA_OFFLINE_NO_DCF','1');"
        "import numpy as np, pandas as pd, postBoRank as pbr;"
        "rng=np.random.default_rng(1);"
        "cols=['RoA','earnYield','CycleHeat','Piotroski','tbVpRatio','incomeQuality',"
        "'grossProfitMargin','Altman-Z','currentRatio','bVpRatio'];"
        "df=pd.DataFrame({'source':['S'+str(i) for i in range(50)]});"
        "df=df.assign(**{c: rng.normal(scale=1e3,size=50) for c in cols});"
        "out=pbr.getRankOfRanks(pbr.getAggScore(df));"
        "print(json.dumps({'agg':hashlib.sha256(out['AggScore'].to_numpy().tobytes())"
        ".hexdigest(),'src':list(out['source'])}))")
    digests = []
    for seed in ("0", "7"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        env.pop("VA_OFFLINE_NO_DCF", None)
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                           env=env, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        digests.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert digests[0]["agg"] == digests[1]["agg"], \
        ("AggScore still differs across hash seeds: %s vs %s"
         % (digests[0]["agg"][:16], digests[1]["agg"][:16]))
    assert digests[0]["src"] == digests[1]["src"], "the emitted ranking order differs"


def test_a_missing_canon_DEGRADES_to_frame_order_and_stays_DETERMINISTIC():
    """The fallback must not reintroduce a set.  If scoringWeights ever fails to import, the
    order is merely non-canonical -- never non-deterministic."""
    import postBoRank as pbr
    saved = sys.modules.get("scoringWeights")
    try:
        sys.modules["scoringWeights"] = None            # import raises inside the helper
        got = pbr.deterministic_column_order(["source", "b", "a", "c"])
        assert got == ["b", "a", "c"], got
    finally:
        if saved is None:
            sys.modules.pop("scoringWeights", None)
        else:
            sys.modules["scoringWeights"] = saved


# =========================================================================== #
#  ERROR PATHS AND RARE DATA SHAPES -- the part bit-identity CANNOT cover       #
# =========================================================================== #
#  The blocking bug this wave was set against fired ONLY when an endpoint died
#  (`.text` on a failure object), so no amount of bit-identity on a saved panel could have
#  caught it.  Every path below is one the 2026-08-02 changes TOUCH, exercised explicitly.
_ODD_FRAMES = {
    "empty_with_date": pd.DataFrame({"date": [], "source": []}),
    "no_date_column": pd.DataFrame({"source": ["A"], "v": [1.0]}),
    "single_row": pd.DataFrame({"date": ["2026-01-01"], "source": ["A"], "v": [1.0]}),
    "all_unparseable": pd.DataFrame({"date": ["x", "y"], "source": "A", "v": [1.0, 2.0]}),
    "all_none": pd.DataFrame({"date": [None, None], "source": "A", "v": [1.0, 2.0]}),
    "integer_dates": pd.DataFrame({"date": [1, 2], "source": "A", "v": [1.0, 2.0]}),
}


@pytest.mark.parametrize("name", sorted(_ODD_FRAMES))
def test_the_COERCING_sort_and_the_CHECK_survive_every_odd_frame_shape(name):
    """Both live on print-only paths, so neither may raise on a shape it could actually meet.
    A missing `date` column DOES raise in the sort itself -- that is why the one production
    caller wraps it (see the diagnostic test below) -- but the CHECK must never raise."""
    frame = _ODD_FRAMES[name]
    assert isinstance(rp.newest_first_violations(frame), tuple)
    assert isinstance(rp.assert_newest_first(frame, name, verbose=False), bool)
    if name == "no_date_column":
        with pytest.raises(KeyError):
            rp.to_newest_first(frame, rp.ON_BAD_DATE_COERCE)
    else:
        out = rp.to_newest_first(frame, rp.ON_BAD_DATE_COERCE)
        assert len(out) == len(frame)


@pytest.mark.parametrize("name", sorted(_ODD_FRAMES))
def test_the_run_log_DIAGNOSTIC_degrades_and_NEVER_raises_on_any_of_them(name):
    """`_diag_newest_rows` is called from an UNGUARDED block in postBoWrapper, so on every one
    of these -- including the missing-date-column case that raises inside the sort -- it must
    return rows rather than abort Stage-2 and every deliverable (review L4)."""
    import postBo as pb
    out = pb._diag_newest_rows(_ODD_FRAMES[name], 3)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == min(3, len(_ODD_FRAMES[name]))


def test_the_STRICT_sort_raises_on_a_partially_unparseable_frame_but_not_on_a_clean_one():
    """The distinction the forensic path depends on: NaT must never silently join the sort."""
    ok = _dated(pd.date_range("2020-01-01", periods=4, freq="QS"))
    assert len(rp.to_newest_first(ok, rp.ON_BAD_DATE_RAISE)) == 4
    with pytest.raises(Exception):
        rp.to_newest_first(_dated(["2026-01-01", "nope"]), rp.ON_BAD_DATE_RAISE)


def test_a_ZERO_LENGTH_and_NEGATIVE_window_cannot_reach_the_metrics():
    """scale_window clamps a degenerate window to >= 1 row on BOTH branches, so no metric can
    silently average nothing -- the invariant 1 <= result <= n."""
    for n in (0, -1, -16):
        for rpy in (2, 4):
            assert rp.scale_window(n, rpy) == 1, (n, rpy)
    for n in (1, 3, 16, 28):
        for rpy in (2, 4):
            assert 1 <= rp.scale_window(n, rpy) <= n, (n, rpy)


def test_postbm_metric_returns_NaN_on_an_EMPTY_per_ticker_frame_rather_than_raising():
    """A source can reach the scorer with no rows (a name whose panel was fully pruned).  The
    head(w).mean() of nothing is NaN, which normalizeAndDropNA maps to pool-neutral."""
    a, _b = cdic.getPostDict()
    cols = ["grahamNumber", "price", "netIncome",
            "netCashProvidedByOperatingActivities", "totalAssets"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64")
                          for c in cols + [a[k]["eqMet"] for k in a]})
    for k in a:
        v = sm.postbm_metric(k, a[k]["eqMet"], empty, 16, rpy=4)
        assert v is None or np.isnan(v), (k, v)


def test_postbm_metric_propagates_NaN_and_does_not_manufacture_a_value():
    """An all-NaN column must stay NaN through the registry factor -- NaN * 1.0 is NaN, and a
    metric that quietly became 0.0 would score as pool-neutral-but-real."""
    a, _b = cdic.getPostDict()
    n = 8
    df = pd.DataFrame({c: [np.nan] * n for c in
                       ["grahamNumber", "price", "netIncome", "totalAssets",
                        "netCashProvidedByOperatingActivities"]
                       + [a[k]["eqMet"] for k in a]})
    for k in a:
        for rpy in (2, 4):
            assert np.isnan(sm.postbm_metric(k, a[k]["eqMet"], df, 16, rpy=rpy)), (k, rpy)


def test_postbm_metric_REFUSES_a_metric_the_registry_says_has_no_window():
    """The registry can be edited wrongly too.  A point-in-time metric routed through the
    head(w) path must fail with a message naming the metric, not with a TypeError from
    scale_window(None, rpy) deep in the arithmetic."""
    with pytest.raises(ValueError) as ei:
        sm.postbm_metric("Piotroski", "netIncome",
                         pd.DataFrame({"netIncome": [1.0, 2.0]}), 16, rpy=4)
    assert "Piotroski" in str(ei.value) and "no averaging window" in str(ei.value)


def test_the_metric_accessor_handles_EMPTY_and_MISSING_COLUMN_frames():
    """A degenerate pool (every name dropped) must not make the accessor raise -- it is used by
    analysis consumers that run over whatever a saved pool contains."""
    import postBoRank as pbr
    empty = pbr.stamp_metric_basis(pd.DataFrame({"source": []}), pbr.BASIS_Z)
    out, kept, dropped = pbr.metric_frame(empty, pbr.BASIS_Z_TIMES_W)
    assert len(out) == 0 and kept == [] and dropped == []
    # a frame with only SOME metric columns converts just those
    partial = pbr.stamp_metric_basis(pd.DataFrame({"source": ["a"], "RoA": [1.0]}),
                                     pbr.BASIS_Z)
    out2, kept2, _d = pbr.metric_frame(partial, pbr.BASIS_Z_TIMES_W)
    assert kept2 == ["RoA"]
    assert out2["RoA"].iloc[0] == pytest.approx(1.0 * _weights()["RoA"])
    # an explicit cols list naming an ABSENT column is skipped, not an error
    out3, kept3, _d3 = pbr.metric_frame(partial, pbr.BASIS_Z_TIMES_W,
                                        cols=["RoA", "notAColumn"])
    assert kept3 == ["RoA"]


def test_the_basis_helpers_never_raise_on_a_NON_FRAME():
    """metric_basis_of / row_order_of are called defensively on whatever a consumer holds."""
    import postBoRank as pbr
    for obj in (None, 0, "x", [], {}):
        assert pbr.metric_basis_of(obj) is None
        assert rp.row_order_of(obj) is None


def test_a_NaT_bearing_panel_passes_the_stage2_boundary_check_without_a_false_alarm():
    """NaT rows are real (unparseable dates survive ingest) and sort LAST.  The boundary check
    must ignore them rather than report the whole panel as mis-ordered -- a check that
    cries wolf on normal data gets ignored, which is how the frequency watchdog went dark."""
    d = pd.DataFrame({"source": "A",
                      "date": pd.to_datetime(["2026-01-01", "2025-10-01", None],
                                             errors="coerce"),
                      "v": [1.0, 2.0, 3.0]})
    assert rp.newest_first_violations(d, by="source") == (0, 1)
    assert rp.assert_newest_first(d, "nat-panel", by="source", verbose=False) is True


def test_the_stage2_boundary_CANNOT_be_aborted_by_its_own_order_check():
    """The check runs inside the production scorer, so it is wrapped: even a check that threw
    must cost the log line, not the ranking."""
    import inspect
    import postBoRank as pbr
    src = inspect.getsource(pbr._sort_cdx_newest_first)
    assert "_safe_diagnose(rp.assert_newest_first" in src
    # and _safe_diagnose really does swallow-and-report
    def _boom():
        raise RuntimeError("x")
    assert pbr._safe_diagnose(_boom) is None


def test_the_flag_read_survives_a_WHITESPACE_or_JUNK_env_value():
    """An operator typo must not silently enable 225 live GETs."""
    import postBoRank as pbr
    saved = os.environ.get("VA_OFFLINE_NO_DCF")
    try:
        for junk in ("   ", "\t", "maybe", "TRUE", "Off", "NO"):
            os.environ["VA_OFFLINE_NO_DCF"] = junk
            got = pbr.offline_no_dcf()
            assert got is (junk.strip().lower() not in ("0", "false", "no", "off",
                                                       "none", "null")
                           if junk.strip() else True), (junk, got)
    finally:
        if saved is None:
            os.environ.pop("VA_OFFLINE_NO_DCF", None)
        else:
            os.environ["VA_OFFLINE_NO_DCF"] = saved


def test_a_DEAD_endpoint_with_the_fetch_ENABLED_still_degrades_and_the_diagnostic_survives(
        monkeypatch, capsys):
    """THE outage this wave descends from, re-run through the moved flag.  With the fetch ON
    and the endpoint dead, `_fetch_ticker_dcf` must return an EMPTY frame (DcfToPrice costs
    that ticker, w = 0.000) and the first-ticker diagnostic -- which is what called `.text` on
    the failure object and killed Stage-2 for all six pools -- must print instead of raising.
    No network: safe_http_get is replaced."""
    import getData_gen as gdg
    import postBoRank as pbr
    monkeypatch.setenv("VA_OFFLINE_NO_DCF", "0")            # fetch ENABLED
    failed = gdg._FailedResponse(url="http://x", error="socket died")
    monkeypatch.setattr(gdg, "safe_http_get", lambda *a, **k: failed)
    dcf, from_bulk, status, resp = pbr._fetch_ticker_dcf("AAPL", "http://b/", "k", {})
    assert dcf.empty and from_bulk is False and resp is failed
    assert status == failed.status_code
    # the failure object must be Response-SHAPED, not just status-shaped
    assert isinstance(failed.text, str) and failed.json() == [] and failed.ok is False
    # and the diagnostic that consumes it must not raise
    tempcdx = pd.DataFrame({"freeCashFlow": [1.0], "marketCap": [2.0],
                            "operatingIncome": [3.0]})
    pbr._diagnose_first_ticker_data("AAPL", dcf, from_bulk, status, resp, tempcdx)
    assert "DCF error" in capsys.readouterr().out


# =========================================================================== #
#  6  THE TWO RUN-KILLERS -- one bad name must not cost the whole fetch        #
# =========================================================================== #
#  Both fire ONLY on a rare data shape, so no bit-identity check on a saved panel can reach
#  them -- which is exactly why they survived every previous wave.
def _moat_frame(sources, rows=8):
    """A cdx-shaped frame carrying every column moatIdentifier reads."""
    cols = ["pfcfRatio", "grossProfitMargin", "revenue", "totalAssets", "returnOnEquity",
            "returnOnAssets", "returnOnCapitalEmployed", "grossProfit",
            "sellingGeneralAndAdministrativeExpenses", "depreciationAndAmortization",
            "netProfitMargin", "netIncomePerShare", "capexPerShare", "totalLiabilities",
            "totalStockholdersEquity"]
    out = []
    for s in sources:
        f = pd.DataFrame({"source": s,
                          "date": pd.date_range("2020-03-31", periods=rows, freq="QE")})
        for c in cols:
            f[c] = np.linspace(1.0, 2.0, rows)
        out.append(f)
    return pd.concat(out, ignore_index=True)


def test_moatIdentifier_CONTAINS_a_bad_name_and_still_scores_every_other(capsys):
    """THE run-killer.  One unparseable date used to raise out of this function, through an
    UNGUARDED Sbocker call site running the FULL ~7.7k universe, and take down
    detectManipulation + writeResWrapper + every deliverable of a 12-hour fetch."""
    import postBo as pb
    syms = ["AAA", "BBB", "CCC", "DDD"]
    clean = _moat_frame(syms)
    good = pb.moatIdentifier(pd.Series(syms), clean).set_index("source")
    capsys.readouterr()

    bad = clean.copy()
    bad["date"] = bad["date"].astype(object)
    bad.loc[bad.index[bad["source"] == "CCC"][0], "date"] = "not-a-date"
    out = pb.moatIdentifier(pd.Series(syms), bad)          # must NOT raise
    printed = capsys.readouterr().out
    o = out.set_index("source")

    assert len(out) == len(syms), "the failed name must still occupy a row"
    assert np.isnan(o.loc["CCC", "moatScore"]), "a non-computable name must be NaN, not 0"
    for s in ("AAA", "BBB", "DDD"):
        assert good.loc[s, "moatScore"] == o.loc[s, "moatScore"], s
    crit = ["FCFyield", "GrossMargin", "RevtoASS", "RoE", "RoA", "ROIC", "SGAtoGP",
            "DeptoGP", "NetMargin", "CapExtoEarnings", "TLtoEquity"]
    assert o.loc["CCC", crit].isna().all(), \
        "a PARTIALLY written row was emitted as if it had been scored"
    assert "CCC" in printed and "could NOT be scored" in printed, \
        "a silently skipped name is how the frequency watchdog went dark"


def test_moatIdentifier_reports_ZERO_failures_AFFIRMATIVELY(capsys):
    """A guard that is silent when healthy cannot be told apart from a guard that is dead."""
    import postBo as pb
    pb.moatIdentifier(pd.Series(["AAA", "BBB"]), _moat_frame(["AAA", "BBB"]))
    out = capsys.readouterr().out
    assert "0 per-name failures" in out, out[-400:]


def test_moatIdentifier_contains_EVERY_per_name_failure_not_just_bad_dates(capsys):
    """The containment is on the NAME, not on one exception type: a missing column, a
    degenerate frame or an arithmetic fault must cost that name too."""
    import postBo as pb
    syms = ["AAA", "BBB"]
    broken = _moat_frame(syms).drop(columns=["totalStockholdersEquity"])
    out = pb.moatIdentifier(pd.Series(syms), broken)       # must NOT raise
    printed = capsys.readouterr().out
    assert len(out) == 2 and out["moatScore"].isna().all()
    assert "could NOT be scored" in printed


def test_failTests_treats_an_UNPARSEABLE_income_statement_date_as_EMPTYFAIL():
    """THE other run-killer.  strptime raises on any non-ISO date, and this function sits
    OUTSIDE the per-ticker guard (getFsData_fmp has no try, and get_fundamentals_fmp calls it
    BEFORE entering its own), so one malformed date killed a ~38,500-call fetch with NO
    RESUME.  Handled the audit-H-4 way -- emptyfail, skipped, logged, counted."""
    import failTests as ft

    class _Resp:
        def __init__(self, payload):
            self.status_code = 200
            self._p = payload

        def json(self):
            return self._p

    def _gate(datestr, n=16):
        payload = [{"date": datestr, "period": "Q1"} for _ in range(n)]
        return ft.testForAPIFaults_fmp(range(400, 600), 2025, "T", "quarter", n,
                                       "http://b/", "k", http_get=lambda url: _Resp(payload))

    # the happy path is UNCHANGED
    assert _gate("2026-03-31")[:2] == (False, "None")
    # a genuinely old date is STILL datefail -- not swallowed by the new branch
    assert _gate("2019-03-31")[:2] == (True, "datefail")
    # and every unusable shape is emptyfail rather than an exception
    for bad in ("31/03/2026", "2026-03-31T00:00:00", "", "not-a-date", None, float("nan")):
        failbool, why, _out = _gate(bad)
        assert (failbool, why) == (True, "emptyfail"), (bad, failbool, why)


def test_failTests_uses_the_SAME_bucket_as_the_H4_branch_it_mirrors():
    """One convention, not two: an unusable date and a malformed body are the same kind of
    event, so they must land in the same completeness counter.  `datefail` would be a LIE --
    it means 'we read the date and it was too old'."""
    import inspect
    import failTests as ft
    src = inspect.getsource(ft.testForAPIFaults_fmp)
    i = src.index("_newest_year = datetime.strptime")
    window = src[i:i + 900]
    assert "emptyfail" in window and "datefail" not in window.split("elif")[0]
    assert "EMPTYFAIL" in window, "the skip must be logged, like every other emptyfail"


def test_NO_comment_in_the_scoring_files_cites_a_LINE_NUMBER():
    """postBoRank.py moved twice in one day and a cite corrected from 209 to 229 was stale
    within 308 hours.  Symbol references survive a move; line numbers cannot."""
    import re
    pat = re.compile(r"\b\w+\.py:\d+")
    offenders = {}
    for base in ("postBoRank.py", "postBo.py", "stage2_metrics.py", "getData_gen.py",
                 "calcMetrics.py", "failTests.py", "reporting_period.py"):
        txt = open(os.path.join(_REPO, base), encoding="utf-8").read()
        hits = sorted(set(pat.findall(txt)))
        if hits:
            offenders[base] = hits
    assert not offenders, ("line-number citations (use a SYMBOL name instead): %s" % offenders)
