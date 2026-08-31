"""The DELIVERABLE stages must survive a post-fetch network failure.

THE DEFECT (fixed 2026-07-31).  The fetch loop is hardened (gdg.safe_http_get: 10s timeout, 3
retries, exponential backoff).  The stages that produce the actual deliverables were not:
  * postBoRank._fetch_ticker_dcf ran a bare requests.get with NO timeout, and its `try` covered
    only `.json()` -- not the GET;
  * postBo.writeBoAggToCSV made ~4-5 bare `requests.get(...).json()` calls x 100 names with NO
    try/except anywhere in the loop body;
  * postBo.createPresentation made ~7 more x 20 names;
  * and both were called UNGUARDED from writeResWrapper.
So a throttled 200 carrying an HTML body -> JSONDecodeError -> no AggScore CSV, no XLSX, no
forensic CSV, no side-lists, no band CSVs, no pick-log.  With no timeout, a hung socket stalls
an UNATTENDED run indefinitely.  Probability is ELEVATED precisely because these ~500-700 calls
fire immediately after 12+ hours of sustained API load.

Every test here is OFFLINE: the HTTP layer is injected (`_get=`) or monkeypatched.  Nothing in
this file may make a real network call.
"""

import ast
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import getData_gen as gdg


# --------------------------------------------------------------------------- #
#  Fake responses                                                             #
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status_code=200, payload=None, raises=None):
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises is not None:
            raise self._raises
        return self._payload


def _throttled_html():
    """THE failure mode: a 200 whose body is HTML, so `.json()` raises."""
    return _Resp(200, raises=ValueError("Expecting value: line 1 column 1 (char 0)"))


# --------------------------------------------------------------------------- #
#  safe_json_list -- the shared helper                                        #
# --------------------------------------------------------------------------- #
def test_a_throttled_200_with_an_HTML_body_yields_an_empty_list_not_an_exception():
    """The exact case that would have cost the run.  Bare `.json()` raised here."""
    got = gdg.safe_json_list("http://x", _get=lambda *a, **k: _throttled_html(),
                             sleep=lambda _s: None, verbose=False)
    assert got == []


def test_a_connection_error_on_every_attempt_yields_an_empty_list():
    import requests

    def _boom(*_a, **_k):
        raise requests.ConnectionError("socket died")

    got = gdg.safe_json_list("http://x", _get=_boom, sleep=lambda _s: None, verbose=False)
    assert got == []


def test_a_429_is_RETRIED_with_backoff_and_then_degrades():
    calls, slept = [], []

    def _get(*_a, **_k):
        calls.append(1)
        return _Resp(429, payload=[])

    got = gdg.safe_json_list("http://x", _get=_get, sleep=slept.append, verbose=False)
    assert got == []
    assert len(calls) == 3, "must retry, not give up on the first 429"
    assert slept and all(s > 0 for s in slept), slept
    assert slept == sorted(slept), "backoff must be non-decreasing (exponential)"


def test_a_transient_429_followed_by_success_RECOVERS():
    """The point of retrying: the value is not lost to one throttle."""
    seq = [_Resp(429), _Resp(200, payload=[{"a": 1}])]

    def _get(*_a, **_k):
        return seq.pop(0)

    got = gdg.safe_json_list("http://x", _get=_get, sleep=lambda _s: None, verbose=False)
    assert got == [{"a": 1}]


def test_a_TIMEOUT_is_always_passed_to_the_transport():
    """With no timeout a hung socket stalls an unattended overnight run FOREVER -- the worst
    failure mode of the set, and the one a retry cannot save you from."""
    seen = {}

    def _get(url, params=None, headers=None, timeout=None):
        seen["timeout"] = timeout
        return _Resp(200, payload=[])

    gdg.safe_json_list("http://x", _get=_get, sleep=lambda _s: None, verbose=False)
    assert seen["timeout"] is not None and seen["timeout"] > 0, seen


def test_an_ERROR_MESSAGE_dict_body_becomes_empty_and_a_record_dict_is_wrapped():
    err = gdg.safe_json_list("http://x", verbose=False, sleep=lambda _s: None,
                             _get=lambda *a, **k: _Resp(200, {"Error Message": "limit"}))
    assert err == []
    one = gdg.safe_json_list("http://x", verbose=False, sleep=lambda _s: None,
                             _get=lambda *a, **k: _Resp(200, {"symbol": "AAPL"}))
    assert one == [{"symbol": "AAPL"}]


def test_a_healthy_response_is_returned_UNCHANGED():
    """The no-op guarantee: hardening must not alter the data on the happy path."""
    payload = [{"currentRatio": 1.5}, {"currentRatio": 1.4}]
    got = gdg.safe_json_list("http://x", verbose=False,
                             _get=lambda *a, **k: _Resp(200, payload))
    assert got == payload


def test_a_non_200_degrades_quietly():
    for code in (401, 403, 404):
        assert gdg.safe_json_list("http://x", verbose=False, sleep=lambda _s: None,
                                  _get=lambda *a, **k: _Resp(code, [])) == []


# --------------------------------------------------------------------------- #
#  The call sites: no bare requests.get survives in the post-fetch stages     #
# --------------------------------------------------------------------------- #
def _live_call_lines(path, func_name):
    """Source lines of `func_name` that make a NON-hardened GET (comments excluded)."""
    import inspect
    import importlib
    mod = importlib.import_module(path)
    src = inspect.getsource(getattr(mod, func_name))
    bad = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            continue
        if "requests.get(" in s:
            bad.append(s)
    return bad


@pytest.mark.parametrize("module,func", [
    ("postBo", "writeBoAggToCSV"),
    ("postBo", "createPresentation"),
    ("postBoRank", "_fetch_ticker_dcf"),
])
def test_no_bare_requests_get_remains_in_the_post_fetch_stages(module, func):
    """The SWEEP.  ~500-700 calls lived across these three functions; a fix applied to two of
    the three would leave the run just as losable, which is this project's signature defect."""
    bad = _live_call_lines(module, func)
    assert not bad, "%s.%s still makes un-hardened GET(s): %s" % (module, func, bad)


def test_the_hardened_helpers_are_actually_the_ones_used():
    """A negative assertion is not enough -- prove the replacement is the hardened path."""
    import inspect
    import postBo
    import postBoRank
    a = inspect.getsource(postBo.writeBoAggToCSV)
    b = inspect.getsource(postBo.createPresentation)
    c = inspect.getsource(postBoRank._fetch_ticker_dcf)
    assert a.count("gdg.safe_json_list(") >= 4, a.count("gdg.safe_json_list(")
    assert b.count("gdg.safe_json_list(") >= 7, b.count("gdg.safe_json_list(")
    assert "gdg.safe_http_get(" in c


def test_fetch_ticker_dcf_returns_an_empty_frame_instead_of_raising(monkeypatch):
    """A dead DCF endpoint must cost DcfToPrice (weight 0.000 in the live vector) for one
    ticker, never the Stage-2 scorer."""
    import postBoRank as pbr
    import requests

    def _boom(*_a, **_k):
        raise requests.ConnectionError("socket died")

    monkeypatch.setattr(gdg.time, "sleep", lambda _s: None, raising=False)
    monkeypatch.setattr(requests, "get", _boom)
    dcf, from_bulk, status, resp = pbr._fetch_ticker_dcf("AAPL", "http://b/", "k", {})
    assert isinstance(dcf, pd.DataFrame) and dcf.empty
    assert from_bulk is False
    assert status != 200


def test_fetch_ticker_dcf_survives_a_throttled_html_body(monkeypatch):
    import postBoRank as pbr
    import requests
    monkeypatch.setattr(gdg.time, "sleep", lambda _s: None, raising=False)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _throttled_html())
    dcf, _fb, _st, _r = pbr._fetch_ticker_dcf("AAPL", "http://b/", "k", {})
    assert dcf.empty


# --------------------------------------------------------------------------- #
#  B1 -- the defect THIS WAVE INTRODUCED, and the gap that let it through      #
# --------------------------------------------------------------------------- #
#  The original suite tested `_fetch_ticker_dcf` and never the DIAGNOSTIC that consumes its 4th
#  return value, so it could not see that routing the fetch through safe_http_get now hands a
#  `_FailedResponse` to code doing `resp_dcf.text`.  Reproduced:
#      status: 599  resp type: _FailedResponse  has .text: False
#      AttributeError: '_FailedResponse' object has no attribute 'text'
#  It fires in exactly the dead/hung-endpoint case the hardening exists for, on the FIRST ticker
#  of each of the 6 pools, and kills Stage-2 -> no postRank -> no CSV/XLSX/side-lists/band
#  CSVs/pick-log.  These tests drive the CONSUMER, not the fetcher.
def _failed_response():
    import requests

    def _boom(*_a, **_k):
        raise requests.ConnectionError("socket died")

    return gdg.safe_http_get("http://x", _get=_boom, sleep=lambda _s: None)


def test_the_failed_response_shim_is_RESPONSE_SHAPED():
    """`.text` is the second-most-used Response attribute after `.status_code`.  The shim owning
    it is what generalises the fix: any OTHER consumer reaching for `.text` is now safe too,
    whereas a getattr at one call site would have fixed only that site."""
    r = _failed_response()
    assert isinstance(r, gdg._FailedResponse)
    assert r.status_code == 599
    assert isinstance(r.text, str) and r.text, r.text
    assert isinstance(r.content, bytes)
    assert r.ok is False
    assert r.json() == []
    # and it must be sliceable exactly like a real body, since that is how it is consumed
    assert isinstance(r.text[:100], str)


def test_the_dcf_diagnostic_does_NOT_raise_on_a_FailedResponse():
    """THE regression test for B1 -- the one the original suite was missing."""
    import postBoRank as pbr
    resp = _failed_response()
    tempcdx = pd.DataFrame({"freeCashFlow": [1.0], "marketCap": [2.0],
                            "operatingIncome": [3.0]})
    pbr._diagnose_first_ticker_data("AAPL", pd.DataFrame(), False,
                                    resp.status_code, resp, tempcdx)


@pytest.mark.parametrize("resp", [None, "not-a-response", 42, object()])
def test_the_dcf_diagnostic_tolerates_ANY_response_shaped_object(resp):
    """A diagnostic must not depend on the exact type it happens to be handed."""
    import postBoRank as pbr
    pbr._diagnose_first_ticker_data("AAPL", pd.DataFrame(), False, 599, resp,
                                    pd.DataFrame({"freeCashFlow": [1.0]}))


def test_a_BROKEN_diagnostic_can_never_abort_stage2():
    """The structural half of B1.  Fixing the one attribute leaves the structure that turned a
    print bug into total loss: three unguarded diagnostic calls inside a postBoScoreRanking that
    postBoWrapper does not wrap.  `_safe_diagnose` reports loudly and continues."""
    import postBoRank as pbr

    def _explodes(*_a, **_k):
        raise RuntimeError("diagnostic is broken")

    out = pbr._safe_diagnose(_explodes, 1, 2, three=3)
    assert out is None
    # and it really does return the value on the happy path
    assert pbr._safe_diagnose(lambda a, b: a + b, 2, 3) == 5


def test_every_diagnostic_call_in_stage2_is_routed_through_the_guard():
    """The sweep: three call sites, and a fix applied to two of three is this project's
    signature defect."""
    import ast
    import inspect
    import postBoRank as pbr
    src = inspect.getsource(pbr.postBoScoreRanking)
    tree = ast.parse(src.strip())
    direct = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id.startswith("_diagnose"):
            direct.append(n.func.id)
    assert not direct, "diagnostic(s) called without _safe_diagnose: %s" % direct
    guarded = [ast.unparse(n.args[0]) for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_safe_diagnose" and n.args]
    # SEVEN sites now.  `_diagnose_inputs` was missed on the first pass and this assertion
    # caught it; `missing_data_fill_report` was added 2026-08-01, `single_column_reach_check`
    # 2026-08-03 (the E-1 k-property check) and `npol.report_counts` 2026-08-05 (the NaN-policy
    # per-rule conversion counts), all guarded for the same reason -- they are emits-only code
    # sitting on the critical path.  Enumerated explicitly so an eighth diagnostic added later
    # must be declared here rather than slipping in unguarded.
    assert sorted(guarded) == ["_diagnose_first_ticker_data",
                               "_diagnose_first_ticker_metrics",
                               "_diagnose_inputs",
                               "_diagnose_pre_normalize",
                               "missing_data_fill_report",
                               "npol.report_counts",
                               "single_column_reach_check"], guarded


# --------------------------------------------------------------------------- #
#  B2 -- the last unguarded JSON parse in the ~38,500-call FETCH path          #
# --------------------------------------------------------------------------- #
def test_a_throttled_html_200_in_the_FETCH_path_skips_the_TICKER_not_the_RUN():
    """The wave hardened ~700 post-fetch calls against exactly this body-shape while leaving the
    ~38,500-call fetch path bare -- and the fetch path has NO RESUME (write_lastIndexRead runs
    only after get_fundamentals_fmp returns), so a crash at hour 11 loses 11 hours."""
    import failTests as ft
    failbool, whyfail, _out = ft.testForAPIFaults_fmp(
        list(range(400, 600)), 2025, "AAPL", "quarter", 24, "http://b/", "k",
        http_get=lambda *a, **k: _throttled_html())
    assert failbool is True
    assert whyfail == "emptyfail", whyfail


def test_the_throttled_ticker_is_COUNTED_and_the_loop_continues():
    """It must land in the completeness buckets, not vanish -- and `km` must come back as the
    -37707 sentinel the fetch loop checks before processing a ticker."""
    import getData_fmp as gdf
    tf, lf, dfl, ef = [], [], [], []
    km, _fr, _inc, _bs, _cf, tf, lf, dfl, ef = gdf.getFsData_fmp(
        "AAPL", "quarter", 24, "http://b/", "k", 2025, tf, lf, dfl, ef,
        http_get=lambda *a, **k: _throttled_html())
    assert km == -37707, km
    assert tf == ["AAPL"] and ef == ["AAPL"]
    assert dfl == [] and lf == [], "must not be misreported as datefail/lenfail"


def test_it_is_emptyfail_NOT_failcode():
    """`failcode` means a definitive HTTP status.  Conflating an unparseable 200 body with one
    would misreport the cause in the run's completeness artifacts."""
    import failTests as ft
    _fb, why_body, _o = ft.testForAPIFaults_fmp(
        list(range(400, 600)), 2025, "T", "quarter", 24, "http://b/", "k",
        http_get=lambda *a, **k: _throttled_html())
    _fb2, why_status, _o2 = ft.testForAPIFaults_fmp(
        list(range(400, 600)), 2025, "T", "quarter", 24, "http://b/", "k",
        http_get=lambda *a, **k: _Resp(503, []))
    assert why_body == "emptyfail"
    assert why_status == "failcode"


def test_a_HEALTHY_fetch_response_is_still_parsed_UNCHANGED():
    """The no-op guarantee on the happy path -- this is the 38,500-call path, so a behaviour
    change here would be far worse than the defect."""
    import failTests as ft
    rows = [{"date": "2026-03-31", "revenue": 1.0}] * 24
    failbool, whyfail, out = ft.testForAPIFaults_fmp(
        list(range(400, 600)), 2020, "T", "quarter", 24, "http://b/", "k",
        http_get=lambda *a, **k: _Resp(200, rows))
    assert failbool is False, whyfail
    assert set(out) == {"km", "fr", "inc", "bs", "cf"}
    assert len(out["inc"]) == 24


def _contains(parent, target):
    return any(node is target for node in ast.walk(parent))


def test_no_unguarded_resp_json_remains_in_the_fetch_path():
    """The sweep over the fetch path's own JSON parsing."""
    import inspect
    import failTests as ft
    tree = ast.parse(inspect.getsource(ft.testForAPIFaults_fmp).strip())
    unguarded = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Attribute) and n.func.attr == "json":
            # every .json() must sit inside a Try
            protected = any(isinstance(a, ast.Try) and _contains(a, n)
                            for a in ast.walk(tree))
            if not protected:
                unguarded.append(n.lineno)
    assert not unguarded, "unguarded resp.json() at relative line(s) %s" % unguarded


# --------------------------------------------------------------------------- #
#  Stage-level: one failed deliverable must not take the others down          #
# --------------------------------------------------------------------------- #
def test_writeResWrapper_guards_BOTH_api_heavy_deliverable_stages():
    """The loss the brief is actually about: writeBoAggToCSV and createPresentation were called
    UNGUARDED, so a failure in either also cost the forensic CSV, the XLSX, the side-lists, the
    band CSVs and the pick-log -- every artifact of a 12-hour fetch."""
    import inspect
    import postBo
    src = inspect.getsource(postBo.writeResWrapper)
    for call in ("flag_df = writeBoAggToCSV(", "createPresentation("):
        i = src.index(call)
        before = src[max(0, i - 900):i]
        assert "try:" in before, "%s is not inside a try/except" % call
    # and each has its own handler, so one failing does not skip the other
    assert src.count("except Exception as _e:") >= 4, src.count("except Exception as _e:")


def test_the_aggscore_guard_does_not_null_out_flag_df():
    """flag_df must keep its pre-call value on failure: writeBoAggToCSV RETURNS the reconciled
    table, and rebinding it to None would cascade into the forensic CSV and the XLSX."""
    import inspect
    import postBo
    src = inspect.getsource(postBo.writeResWrapper)
    i = src.index("except Exception as _e:", src.index("flag_df = writeBoAggToCSV("))
    handler = src[i:i + 500]
    assert "flag_df = None" not in handler
    assert "flag_df =" not in handler.split("print(")[0].replace(
        "except Exception as _e:", "")


# --------------------------------------------------------------------------- #
#  Item 5 -- the three raise paths a healthy 200 can still carry               #
# --------------------------------------------------------------------------- #
#  My original comment claimed "every consumer already guards on len(...) == 0 -> 'NaN'".  That
#  was FALSE and it was the premise for skipping the per-row guard.  None of these three is a
#  FAILED call -- each is a healthy 200 with an awkward VALUE, which safe_json_list cannot help
#  with, so they need value-level handling.
def test_fmt4_absorbs_the_null_that_FMP_really_sends():
    """`"{:.4f}".format(None)` raises TypeError, and FMP returns `"beta": null` for plenty of
    non-US listings."""
    import postBo
    assert postBo._fmt4(None) == 'NaN'
    assert postBo._fmt4(float('nan')) == 'NaN'
    assert postBo._fmt4('1.5') == 'NaN', "a string must not be silently formatted"
    assert postBo._fmt4(True) == 'NaN', "bool is an int subclass but is not a price"
    assert postBo._fmt4(1.23456) == '1.2346'
    assert postBo._fmt4(2) == '2.0000'


def test_price_and_beta_no_longer_raise_on_a_null_profile_field():
    """The exact defect: these two checked key PRESENCE but not None, while the
    grahamNumberToPrice block a few lines above always checked None explicitly."""
    import inspect
    import postBo
    src = inspect.getsource(postBo.writeBoAggToCSV)
    for field, vec in (("'price'", "priceVec"), ("'beta'", "betaVec")):
        i = src.index("%s.append(_fmt4(" % vec)
        assert i > 0, "%s no longer routed through _fmt4" % vec
    assert '"{:.4f}".format(temp_resp_pr[0]' not in src, "a raw .format on a profile field remains"


def test_one_mean_score_absorbs_absent_and_DUPLICATED_sources():
    """`.item()` raises ValueError on a multi-row selection -- issuer clones are a known
    phenomenon in this pipeline -- and KeyError when the column is missing."""
    import postBo
    m = pd.DataFrame({'source': ['AAA'], 'M_Score_mean': [1.5]})
    assert postBo._one_mean_score(m, 'AAA', 'M_Score_mean') == 1.5
    assert postBo._one_mean_score(m, 'ZZZ', 'M_Score_mean') is None       # absent
    dup = pd.concat([m, m], ignore_index=True)
    assert postBo._one_mean_score(dup, 'AAA', 'M_Score_mean') is None     # duplicated
    nan = pd.DataFrame({'source': ['AAA'], 'M_Score_mean': [np.nan]})
    assert postBo._one_mean_score(nan, 'AAA', 'M_Score_mean') is None     # NaN
    assert postBo._one_mean_score(m, 'AAA', 'NoSuchColumn') is None       # missing column
    assert postBo._one_mean_score(None, 'AAA', 'M_Score_mean') is None    # missing frame


def test_gross_margin_validates_ALL_FOUR_rows_before_summing():
    """NOT named in the review -- found applying its lesson to the rest of the loop.  The sum
    spans rows [0..3] but only row [0] was type-checked, so a None in rows 1-3 raised TypeError
    on `float + None`."""
    import inspect
    import postBo
    src = inspect.getsource(postBo.writeBoAggToCSV)
    assert "for i in range(4)" in src
    assert "temp_resp_fr[0]['grossProfitMargin'] + temp_resp_fr[1]" not in src, \
        "the unchecked four-row sum survives"


def test_the_row_guard_PADS_and_so_cannot_ragged_the_vectors():
    """The reviewer's correction: only the NAIVE guard is defeated by the parallel per-row
    vectors.  A `finally` that pads to a per-row target length cannot ragged them -- which is
    the whole reason the guard is workable.  Asserted structurally, since driving the real loop
    needs a full resdic.

    THE COUNT MOVED 12 -> 11 ON 2026-08-13, and the DIRECTION is the point.  `ccyVec` (the new
    `priceCurrency` column, register N-3) joined the tuple, and `dyVec` / `GNtPVec` LEFT it:
    `dividendYield` and `GrahamNumberToPrice` are now computed from the run's own panel, so
    they need no vendor response and must not sit inside a guard that nulls a row when one
    fails -- a throttled call on one name would otherwise blank a number we already held.
    The literal stays a literal on purpose: a new per-row vector that is NOT added to
    `_row_vectors` is invisible to the pad, would ragged on the first degraded row, and would
    take the whole CSV with it.  The number is a speed bump for whoever adds the next one.
    """
    import inspect
    import postBo
    src = inspect.getsource(postBo.writeBoAggToCSV)
    assert "_row_vectors = (" in src
    assert "assert len(_row_vectors) == 11" in src, "the vector count must be pinned"
    #  AND THE PIN MUST DESCRIBE THE REAL TUPLE, not just exist: bumping the literal without
    #  adding the vector would satisfy the line above and still ragged the frame.
    _tup = src[src.index("_row_vectors = ("):]
    _tup = _tup[:_tup.index(")")]
    assert len([v for v in _tup.split("(")[1].split(",") if v.strip()]) == 11
    assert "ccyVec" in _tup, "the priceCurrency vector must be covered by the pad guard"
    assert "dyVec" not in _tup and "GNtPVec" not in _tup, (
        "dividendYield / GrahamNumberToPrice are computed from the panel; putting them back "
        "in the row guard would let a failed vendor call blank a value we already hold")
    assert "finally:" in src
    i_fin = src.index("finally:")
    tail = src[i_fin:i_fin + 400]
    assert "del _v[_want:]" in tail, "a partial append must be discarded"
    assert "_v.append('NaN')" in tail, "vectors must be padded to length"
    # the guard must wrap the body, i.e. precede the first append
    assert src.index("try:") < src.index("crVec.append")
    assert "_rows_degraded" in src and "AGGSCORE-CSV DEGRADED-ROW SUMMARY" in src


def test_the_padding_logic_itself_keeps_twelve_vectors_aligned():
    """The arithmetic of the guard, exercised directly on stand-in vectors: a row that fails
    part-way must leave every vector at exactly the target length."""
    vecs = [[] for _ in range(12)]
    for row_i in range(1, 6):
        want = row_i
        try:
            vecs[0].append('a')                 # simulate a partial row
            vecs[1].append('b')
            if row_i == 3:
                raise RuntimeError("mid-row failure")
            for v in vecs[2:]:
                v.append('c')
        except RuntimeError:
            pass
        finally:
            for v in vecs:
                del v[want:]
                while len(v) < want:
                    v.append('NaN')
        assert {len(v) for v in vecs} == {want}, (row_i, [len(v) for v in vecs])
    assert vecs[5] == ['c', 'c', 'NaN', 'c', 'c'], vecs[5]


def test_the_presentation_skips_a_PAGE_when_its_core_statements_are_empty():
    """km / fr / cf are consumed UNGUARDED (km.earningsYield, fr['symbol'], cf.freeCashFlow), so
    an empty response for any of them is an AttributeError/KeyError.  Hardening the transport
    does NOT fix that -- a genuinely empty FMP response gives the same empty frame -- so the
    page must be skipped explicitly."""
    import inspect
    import postBo
    src = inspect.getsource(postBo.createPresentation)
    assert "if km.empty or fr.empty or cf.empty:" in src
    i_guard = src.index("if km.empty or fr.empty or cf.empty:")
    i_use = src.index("symb_df['Symbol'] = fr['symbol']")
    assert i_guard < i_use, "the skip must precede the first unguarded consumer"
    assert "continue" in src[i_guard:i_use]
    assert "_pages_skipped" in src, "skipped pages must be counted and reported"
    assert "PRESENTATION SKIPPED-PAGE SUMMARY" in src


def test_a_skipped_page_is_REPORTED_not_silent():
    """A silently missing deck page is worse than a loud one: the CEO reads the deck at 3am and
    would have no way to tell a skipped name from a name that never ranked."""
    import inspect
    import postBo
    src = inspect.getsource(postBo.createPresentation)
    i = src.index("_pages_skipped.append(symb)")
    assert "WARNING" in src[max(0, i - 500):i]


# --------------------------------------------------------------------------- #
#  D1 -- 225 live GETs per run feeding a w=0.000 metric                        #
# --------------------------------------------------------------------------- #
#  `postBoScoreRanking` fires one discounted-cash-flow GET per ticker so `DcfToPrice` can be
#  computed and then MULTIPLIED BY ZERO.  And it runs ONCE PER POOL, not once per run: the
#  general pool (head(100)) plus five carve-out cohorts (head(25) each) = 225 GETs, verified
#  against the shipped 2026-07-17 resdic.  The escape hatch existed and was well-built; it was
#  simply off by default.  Default flipped 2026-07-31.
@pytest.fixture
def reimport_pbr(monkeypatch):
    """Set VA_OFFLINE_NO_DCF and hand back postBoRank.  NO RE-IMPORT ANY MORE (2026-08-02).

    THE NAME IS KEPT ONLY SO THE D1 TESTS BELOW READ AS ONE BLOCK; there is nothing left to
    re-import.  `OFFLINE_NO_DCF` used to be evaluated at IMPORT time, so testing the env
    semantics required a real `importlib.import_module` -- which mutates `sys.modules`, and
    monkeypatch does NOT restore that.  A test that re-imported with `=0` therefore left the
    WHOLE SESSION holding a postBoRank whose flag said FETCH even after the env var was
    restored: verified, the stale module is the same object and keeps the wrong value.  Any
    later test, or test FILE, importing postBoRank silently got it.  The suite passed anyway,
    which is what made it an ORDER-DEPENDENT LATENT FAULT rather than a visible failure -- and
    the reason the flag is now read per call (postBoRank.offline_no_dcf).

    So this fixture no longer touches sys.modules at all; it sets the env var (monkeypatch
    restores it) and resets the once-per-process banner flag.  That the tests below still pass
    with no re-import IS the evidence the import-time read is gone.
    """
    import postBoRank as mod

    def _do(env):
        if env is None:
            monkeypatch.delenv('VA_OFFLINE_NO_DCF', raising=False)
        else:
            monkeypatch.setenv('VA_OFFLINE_NO_DCF', env)
        monkeypatch.setattr(mod, '_DCF_BANNER_SHOWN', False)
        return mod

    yield _do


def test_the_deployed_DcfToPrice_weight_really_IS_zero():
    """THE PRECONDITION.  Skipping the fetch is score-neutral ONLY because this is 0.  Asserted
    first and separately, because everything else about D1 rests on it -- and the LEGACY vector
    carries 0.35, so this is not a formality."""
    import createDicts as cdic
    postBm, postNew = cdic.getPostDict()
    w = float({**postBm, **postNew}['DcfToPrice']['w'])
    assert w == 0.0, "DcfToPrice now carries w=%r -- D1's premise is void" % w
    # and in every per-cohort override too, since Stage-2 runs once per cohort
    import carveOut as co
    for label, wov in co.COHORT_WEIGHTS.items():
        assert float(wov.get('DcfToPrice', 0) or 0) == 0.0, (label, wov.get('DcfToPrice'))


def test_the_DCF_fetch_is_SKIPPED_BY_DEFAULT(reimport_pbr):
    """The fix: unset env must mean no call."""
    pbr = reimport_pbr(None)
    assert pbr.offline_no_dcf() is True
    assert pbr.OFFLINE_NO_DCF_DEFAULT is True


def test_NO_network_call_is_made_on_the_default_path(reimport_pbr, monkeypatch):
    """Behavioural, not just flag-level: the GET must not happen.  A live call here would raise."""
    pbr = reimport_pbr(None)

    def _explode(*_a, **_k):
        raise AssertionError("a LIVE DCF GET fired on the default path")

    monkeypatch.setattr(gdg, 'safe_http_get', _explode)
    dcf, from_bulk, status, resp = pbr._fetch_ticker_dcf("AAPL", "http://b/", "k", {})
    assert dcf.empty and from_bulk is False
    assert status == "offline-skipped"
    assert resp is None


@pytest.mark.parametrize("env,skip", [
    (None, True), ('', True), ('1', True), ('true', True), ('yes', True),
    ('0', False), ('false', False), ('no', False), ('off', False),
])
def test_the_escape_hatch_is_a_TRUTH_test_not_a_presence_test(env, skip, reimport_pbr):
    """The audit-C2 footgun in reverse.  With the default now ON, an operator writing
    `VA_OFFLINE_NO_DCF=0` to re-enable the live fetch must actually GET the live fetch -- a
    presence test would ignore them and silently keep skipping."""
    pbr = reimport_pbr(env)
    assert pbr.offline_no_dcf() is skip, (env, pbr.offline_no_dcf())


def test_the_six_offline_tools_still_get_the_skip(reimport_pbr):
    """Backwards compatibility: six baseline_tools scripts do
    `os.environ.setdefault('VA_OFFLINE_NO_DCF', '1')`, which must keep meaning SKIP."""
    pbr = reimport_pbr('1')
    assert pbr.offline_no_dcf() is True


def test_default_off_is_CONDITIONAL_on_the_weight_being_zero(reimport_pbr, monkeypatch):
    """THE POINT OF THE GUARD, and the thing the coordinator asked to be pinned.  If someone
    later gives DcfToPrice a weight, skipping the fetch must REFUSE LOUDLY -- never score the
    metric as blank.  A guard that fails silently would make this change a scoring defect."""
    import createDicts as cdic
    pbr = reimport_pbr(None)
    monkeypatch.setattr(cdic, 'getPostDict',
                        lambda *a, **k: ({}, {'DcfToPrice': {'w': 0.35}}))
    with pytest.raises(SystemExit) as ei:
        pbr._assert_offline_dcf_is_score_neutral()
    msg = str(ei.value)
    assert 'DcfToPrice' in msg and '0.35' in msg, msg
    assert 'REFUSING' in msg, msg
    assert 'VA_OFFLINE_NO_DCF=0' in msg, "the refusal must name the way out"


def test_the_guard_also_covers_the_PER_COHORT_weight_overrides(reimport_pbr):
    """Stage-2 runs once per cohort with a `weight_override`, so a cohort could give DcfToPrice a
    weight without the main vector changing.  The original guard read only getPostDict and would
    have missed it -- a real hole, and flipping the default is what made it load-bearing."""
    pbr = reimport_pbr(None)
    with pytest.raises(SystemExit) as ei:
        pbr._assert_offline_dcf_is_score_neutral({'DcfToPrice': 0.5})
    assert 'weight_override' in str(ei.value), str(ei.value)
    # and a zero override is fine
    pbr._assert_offline_dcf_is_score_neutral({'DcfToPrice': 0.0})


def test_the_guard_is_passed_THIS_POOLS_weights_at_the_call_site():
    """The wiring, not just the helper: a widened guard that is never handed the override is
    decoration."""
    import inspect
    import postBoRank as pbr
    src = inspect.getsource(pbr.postBoScoreRanking)
    assert '_assert_offline_dcf_is_score_neutral(weight_override)' in src


def test_the_guard_does_NOT_fire_when_the_live_fetch_is_ENABLED(reimport_pbr, monkeypatch):
    """With the fetch on, DcfToPrice's weight is irrelevant -- the metric is really computed."""
    import createDicts as cdic
    pbr = reimport_pbr('0')
    monkeypatch.setattr(cdic, 'getPostDict',
                        lambda *a, **k: ({}, {'DcfToPrice': {'w': 0.35}}))
    pbr._assert_offline_dcf_is_score_neutral()          # must not raise


def test_the_banner_prints_ONCE_per_process_not_once_per_pool(reimport_pbr, capsys):
    """Stage-2 runs 6 times; a banner repeated 6 times trains the reader to skip it."""
    pbr = reimport_pbr(None)
    for _ in range(6):
        pbr._assert_offline_dcf_is_score_neutral()
    out = capsys.readouterr().out
    assert out.count('DCF fetch SKIPPED') == 1, out.count('DCF fetch SKIPPED')


def test_the_banner_states_the_SCOPE_it_actually_has_AND_what_that_costs(reimport_pbr, capsys):
    """The banner must describe the run a reader is about to get, and price it.

    THE OLD SHAPE, KEPT AS THE LESSON.  It said "this gates STAGE-2 only" and named
    `writeBoAggToCSV` / `createPresentation` as paths the flag does NOT touch.  Every word of
    that was true, it printed on every run, and the 2026-08-31 run fired ~97 live
    `discounted-cash-flow` GETs out of `writeBoAggToCSV` immediately underneath it.  Nobody
    costed it for four months, because the note gave a reader nothing to cost it WITH.

    The scope is still narrow -- the CEO ruled on 2026-08-31 to KEEP those two display
    columns -- so what this now pins is that the line carries the two numbers that turn a
    disclaimer into a decision: how many calls, and what they buy in the score."""
    pbr = reimport_pbr(None)
    pbr._assert_offline_dcf_is_score_neutral()
    out = capsys.readouterr().out
    assert 'SCOPE' in out
    assert 'writeBoAggToCSV' in out and 'createPresentation' in out, (
        'the banner does not name the paths that still fetch')
    assert '117' in out, 'the banner does not say what the un-gated paths COST'
    assert '0.000' in out, (
        'the banner does not say what they buy in the score -- the fact that makes the '
        'trade legible')
    assert '2026-08-31' in out, 'the banner does not record that this is a decision'
    #  and it must not claim the display cells go blank -- they do not
    assert 'reads empty/NaN in any' not in out


def _dcf_endpoint_callers():
    """{module: [(enclosing FunctionDef, its source)]} for every live DCF endpoint call.

    AST, and the ENCLOSING FUNCTION rather than the file: the question is what the code that
    fires the call says about itself, and a mention of the flag somewhere else in a 2,900-line
    module answers nothing."""
    import ast
    import os
    out = {}
    for name in sorted(os.listdir(_HERE)):
        if not name.endswith('.py') or name.startswith('test_'):
            continue
        path = os.path.join(_HERE, name)
        with open(path, encoding='utf-8') as fh:
            src = fh.read()
        if 'v3/discounted-cash-flow' not in src:
            continue
        tree = ast.parse(src)
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if 'v3/discounted-cash-flow' not in node.value:
                continue
            fn = parents.get(node)
            while fn is not None and not isinstance(fn, (ast.FunctionDef,
                                                         ast.AsyncFunctionDef)):
                fn = parents.get(fn)
            hits.append((fn, ast.get_source_segment(src, fn) if fn is not None else ''))
        if hits:
            out[name[:-3]] = hits
    return out


#  EVERY live DCF call site in the repo, and what the flag does about it.  This registry IS
#  the scope claim the `postBoRank` banner makes; the tests below hold the code to it in both
#  directions, so the banner cannot drift from the code and a FOURTH copy of the fetch cannot
#  appear without somebody writing down which side of the line it is on.
#
#  'gated'   -- the enclosing function consults `offline_no_dcf()` and skips.
#  'kept'    -- it fires unconditionally, BY CEO RULING (2026-08-31), and the enclosing
#               function must say so with the cost and the weight beside it.
DCF_CALL_SITES = {
    'postBoRank._fetch_ticker_dcf': 'gated',
    'postBo.writeBoAggToCSV': 'kept',
    'postBo.createPresentation': 'kept',
}


def test_the_set_of_live_DCF_call_sites_is_EXACTLY_the_one_we_wrote_down():
    """A FOURTH copy of this fetch must not be able to appear quietly.

    The defect was never one bad line: it was that `VA_OFFLINE_NO_DCF` was read in one place
    while THREE functions fired the endpoint, and nobody could see that from either end.
    Naming the known offenders is what let the third one exist, so this asks the repo instead
    of a list -- and EQUALITY, so a site that disappears fails too rather than leaving a stale
    entry that reads as coverage."""
    found = {'%s.%s' % (mod, getattr(fn, 'name', '<module level>'))
             for mod, hits in _dcf_endpoint_callers().items() for fn, _src in hits}
    assert found == set(DCF_CALL_SITES), (
        'unregistered DCF call sites: %s; registered but gone: %s.  Every live '
        'discounted-cash-flow fetch must be recorded in DCF_CALL_SITES as gated by '
        'offline_no_dcf() or as deliberately kept.'
        % (sorted(found - set(DCF_CALL_SITES)), sorted(set(DCF_CALL_SITES) - found)))


def test_the_GATED_call_site_really_consults_the_flag():
    """The one path the flag does gate. If this stops being true the banner's core claim --
    that Stage-2 skips ~225 GETs -- is false, and the score-neutrality guard beside it is
    guarding nothing."""
    for mod, hits in _dcf_endpoint_callers().items():
        for fn, src in hits:
            where = '%s.%s' % (mod, getattr(fn, 'name', '?'))
            if DCF_CALL_SITES.get(where) != 'gated':
                continue
            assert 'offline_no_dcf' in (src or ''), (
                '%s is registered as gated but does not consult offline_no_dcf()' % where)


def test_the_KEPT_call_sites_carry_the_DECISION_that_keeps_them():
    """THE REPLACEMENT FOR A TEST THAT PINNED THE BUG.

    This slot used to hold `test_the_OTHER_DCF_call_sites_are_NOT_gated_by_this_flag`, which
    asserted the defect verbatim and told anyone who fixed it to "update the scope claim".
    It was instance fifteen of a test asserting the thing it covered, and it is not coming
    back: what is pinned now is not that the calls are un-gated but that the code SAYS WHY,
    with the two numbers that make it a decision rather than an oversight.

    Without the weight in particular, the next reader finds ~117 paid calls feeding a display
    column, reads it as the oversight it looked like on 2026-08-31, and gates it again."""
    for mod, hits in _dcf_endpoint_callers().items():
        for fn, src in hits:
            where = '%s.%s' % (mod, getattr(fn, 'name', '?'))
            if DCF_CALL_SITES.get(where) != 'kept':
                continue
            body = src or ''
            assert 'VA_OFFLINE_NO_DCF' in body, (
                '%s does not say that the flag does not gate it' % where)
            assert 'CEO, 2026-08-31' in body or 'CEO ruling' in body, (
                '%s does not record WHOSE decision this is or when' % where)
            assert '0.000' in body, (
                '%s does not state the scored weight -- the fact that makes ~117 paid calls '
                'a legible trade instead of a bug' % where)
            assert '117' in body, (
                '%s does not state what the decision costs per run' % where)


def test_the_sweep_can_SEE_the_call_sites_it_is_about():
    """A guard blind to its own target is the failure this repo keeps producing, so the
    sweep's reach is asserted rather than assumed: it must find all three functions that fire
    the endpoint, by name."""
    found = {'%s.%s' % (mod, getattr(fn, 'name', '?'))
             for mod, hits in _dcf_endpoint_callers().items() for fn, _src in hits}
    for must in ('postBo.writeBoAggToCSV', 'postBo.createPresentation',
                 'postBoRank._fetch_ticker_dcf'):
        assert must in found, 'the sweep does not reach %s: %s' % (must, sorted(found))


def test_the_saved_run_confirms_SIX_pools_ie_225_calls():
    """The count, from the artifact rather than from arithmetic alone: 100 general + 5x25."""
    cache = os.path.join(_HERE, 'baseline_tools', 'resdic_2026-07-17_CORRECTED.pickle')
    if not os.path.exists(cache):
        pytest.skip('no saved resdic on this machine -- call count NOT re-derived here')
    r = pd.read_pickle(cache)
    sl = r.get('carveout_sidelists') or {}
    per_cohort = {k: (len(v['postRank']) if v and 'postRank' in v else 0) for k, v in sl.items()}
    assert len(per_cohort) == 5, per_cohort
    total = len(r['postRank']) + sum(per_cohort.values())
    assert total == 225, (total, per_cohort)


# --------------------------------------------------------------------------- #
#  Missing-data fill calibration -- EMITS ONLY, must change no score           #
# --------------------------------------------------------------------------- #
def _fill_fixture():
    """A pool with known NaNs, one NEGATIVE-weight column, and one zero-weight column."""
    n = 20
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "source": ["S%02d" % i for i in range(n)],
        "earnYield": rng.normal(0.05, 0.02, n),
        "CycleHeat": rng.normal(0.0, 1.0, n),          # negative weight
        "DcfToPrice": rng.normal(1.0, 0.2, n),         # ZERO weight -> excluded
    })
    df.loc[0:2, "earnYield"] = np.nan                  # 3 imputed cells
    df.loc[5, "CycleHeat"] = np.nan                    # 1 imputed cell
    W = pd.Series({"earnYield": 0.0605, "CycleHeat": -0.080, "DcfToPrice": 0.0})
    return df, W


def _score(df, W, with_diag, raw=None):
    import postBoRank as pbr
    norm, _o = pbr.normalizeAndDropNA(df.copy(), weight_series=W)
    if with_diag:
        pbr.missing_data_fill_report(raw if raw is not None else df.copy(), norm, W,
                                     pool="t", csv=False, verbose=False)
    wt = norm.drop("source", axis=1)
    for c in wt.columns:
        wt[c] = norm[c].values * W.get(c, 1)
    psm = pd.concat([norm[norm.columns.difference(wt.columns)], wt], axis=1)
    return pbr.getAggScore(psm).set_index("source")["AggScore"]


def test_the_fill_report_SURVIVES_the_index_reset_that_silenced_it_for_general(capsys):
    """*** THE DEFECT: THE GENERAL POOL HAD NO IMPUTATION AUDIT AT ALL. ***

    `MissingDataFillReport_2026-08-10.csv` carries the five carve-out cohorts and NOTHING for
    `general` -- the one pool that produces the deliverable.  The call was made; it RAISED and
    the guard swallowed it into a one-line WARNING that scrolled past in a 12-hour run.

    THE CAUSE IS AN INDEX MISMATCH.  `normalizeAndDropNA` starts with
    `df.reset_index(inplace=True, drop=True)`, which mutates the CALLER'S frame, while
    `postScoreMetric_raw` is snapshotted BEFORE that call and keeps its original index -- on
    the general pool 0..104 with gaps, inherited from `BoS_dftop100`.  `zc[~imputed]` then
    indexes a 0..99 Series with a boolean labelled 0..104 and pandas raises
    `IndexingError: Unalignable boolean Series`.  The cohorts survived only because their
    frames happened to arrive 0-based, which is what made it look pool-specific.

    Reproduced here EXACTLY (a raw frame with a gappy index, a normalised frame without), so
    the fix is pinned against the shape that broke rather than against a tidy fixture.
    """
    import postBoRank as pbr
    df, W = _fill_fixture()
    raw = df.copy()
    raw.index = range(0, 5 * len(raw), 5)          # the gappy, non-0-based general index
    norm, _o = pbr.normalizeAndDropNA(df.copy(), weight_series=W)
    col, name = pbr.missing_data_fill_report(raw, norm, W, pool="general", csv=False,
                                             verbose=False)
    out = capsys.readouterr().out
    assert col is not None and name is not None, (
        'the general pool STILL produces no fill report: %s' % out)
    assert 'skipped for pool=general' not in out
    assert set(col['pool']) == {'general'} and len(col) == 2
    #  the numbers must be the ones the aligned frames give, not merely non-empty
    assert int(col.set_index('column').loc['earnYield', 'n_imputed']) == 3
    assert int(col.set_index('column').loc['CycleHeat', 'n_imputed']) == 1

    #  AND A GENUINELY MIS-ALIGNED PAIR MUST **RAISE**, NOT WARN (reviewer S4).  The guard
    #  originally sat INSIDE the function's own `try`, so its `except Exception` caught it and
    #  returned `(None, None)` after one stdout line -- byte-for-byte the signature of the
    #  swallowed `IndexingError` above.  A guard that fails the same way as the defect it
    #  guards against is not a guard, so it now sits outside the `try` and propagates to
    #  `_safe_diagnose`, which is what keeps a raise from costing the run.
    bad = raw.iloc[:-1].copy()
    with pytest.raises(ValueError, match='row-aligned'):
        pbr.missing_data_fill_report(bad, norm, W, pool="general", csv=False, verbose=False)


def test_the_fill_report_leaves_AggScore_BIT_IDENTICAL(capsys):
    """THE property that makes this safe to add on the night of a 12-hour fetch.  A diagnostic
    that perturbs the score is not a diagnostic."""
    df, W = _fill_fixture()
    off = _score(df, W, False)
    on = _score(df, W, True, raw=df.copy())
    capsys.readouterr()
    assert (off.values == on.values).all(), (off - on).abs().max()
    assert (list(off.sort_values(ascending=False).index)
            == list(on.sort_values(ascending=False).index))


def test_it_reads_a_NEGATIVE_weight_column_the_right_way_round(capsys):
    """The trap the brief names: a fill ABOVE a column's median is an ADVANTAGE only if the
    weight is POSITIVE.  CycleHeat is w < 0, so the same percentile is a PENALTY -- the imputed
    name is scored as if it were hot."""
    import postBoRank as pbr
    df, W = _fill_fixture()
    norm, _o = pbr.normalizeAndDropNA(df.copy(), weight_series=W)
    col, _name = pbr.missing_data_fill_report(df.copy(), norm, W, pool="t", csv=False,
                                             verbose=False)
    capsys.readouterr()
    ch = col[col["column"] == "CycleHeat"].iloc[0]
    assert ch["weight"] < 0
    pct = ch["fill_percentile_in_observed_z"]
    # fill_effect must carry the SIGN, so its sense is opposite to the naive percentile
    assert np.sign(ch["fill_effect"]) == -np.sign(pct - 0.5), (pct, ch["fill_effect"])
    if pct > 0.5:
        assert "PENALTY" in ch["fill_reading"], (pct, ch["fill_reading"])
    # and a POSITIVE-weight column reads the naive way
    ey = col[col["column"] == "earnYield"].iloc[0]
    assert np.sign(ey["fill_effect"]) == np.sign(
        ey["fill_percentile_in_observed_z"] - 0.5)


def test_it_counts_imputed_cells_per_column_and_per_name(capsys):
    import postBoRank as pbr
    df, W = _fill_fixture()
    norm, _o = pbr.normalizeAndDropNA(df.copy(), weight_series=W)
    col, name = pbr.missing_data_fill_report(df.copy(), norm, W, pool="t", csv=False,
                                             verbose=False)
    capsys.readouterr()
    assert int(col.loc[col["column"] == "earnYield", "n_imputed"].iloc[0]) == 3
    assert int(col.loc[col["column"] == "CycleHeat", "n_imputed"].iloc[0]) == 1
    assert int((name["n_imputed_cols"] > 0).sum()) == 4          # S00..S02 + S05
    assert int(name["n_imputed_cols"].sum()) == 4
    # the per-name weight share makes "scored largely on fills" visible rather than inferred
    s05 = name[name["source"] == "S05"].iloc[0]
    # the reported share is rounded to 4dp for the CSV, so compare at that resolution
    assert s05["imputed_weight_share"] == pytest.approx(
        0.080 / (0.0605 + 0.080), abs=1e-4)
    assert "CycleHeat" in s05["imputed_cols"]


def test_ZERO_weight_columns_are_EXCLUDED_from_the_report(capsys):
    """A w=0 column cannot move the score, so a fill in it is not a calibration fact."""
    import postBoRank as pbr
    df, W = _fill_fixture()
    norm, _o = pbr.normalizeAndDropNA(df.copy(), weight_series=W)
    col, name = pbr.missing_data_fill_report(df.copy(), norm, W, pool="t", csv=False,
                                             verbose=False)
    capsys.readouterr()
    assert "DcfToPrice" not in set(col["column"])
    assert int(name["n_weighted_cols"].iloc[0]) == 2


def test_it_runs_for_the_GENERAL_pool_AND_all_five_COHORTS():
    """The cohorts are the unmeasured gap: a cohort concentrating most of its weight on two
    columns makes one fill far more consequential per cell than the same fill in the general
    pool.  Both call sites must pass a pool label."""
    import inspect
    import postBo
    import postBoRank as pbr
    src = inspect.getsource(pbr.postBoScoreRanking)
    assert "missing_data_fill_report" in src
    assert "pool=(pool_label or 'general')" in src
    ps = inspect.getsource(postBo.postBoWrapper)
    assert "pool_label='general'" in ps
    assert "pool_label=label" in ps, "the carve-out cohorts are not labelled"
    import carveOut as co
    assert len(co.COHORT_WEIGHTS) == 5, len(co.COHORT_WEIGHTS)


def test_the_report_is_GUARDED_and_routed_through_safe_diagnose():
    """Emits-only code on the critical path must not be able to abort Stage-2."""
    import inspect
    import postBoRank as pbr
    src = inspect.getsource(pbr.postBoScoreRanking)
    assert "_safe_diagnose(missing_data_fill_report" in src
    # and the function swallows its own failures too
    assert "except Exception" in inspect.getsource(pbr.missing_data_fill_report)
    assert pbr.missing_data_fill_report(None, None, None, csv=False, verbose=False) == (None, None)


def test_the_CSV_APPENDS_across_pools_rather_than_clobbering(tmp_path, monkeypatch, capsys):
    """postBoScoreRanking runs six times.  A per-call overwrite would leave only the last
    cohort -- the same single-writer clobber the frequency-conflict CSV hit."""
    import glob as _glob
    import postBoRank as pbr
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pbr, "_MISSING_CSV_STARTED", set())
    df, W = _fill_fixture()
    norm, _o = pbr.normalizeAndDropNA(df.copy(), weight_series=W)
    for pool in ("general", "REIT", "Mining"):
        pbr.missing_data_fill_report(df.copy(), norm, W, pool=pool, csv=True, verbose=False)
    capsys.readouterr()
    files = _glob.glob(str(tmp_path / "MissingDataFillReport_*.csv"))
    assert len(files) == 1, files
    out = pd.read_csv(files[0])
    assert set(out["pool"]) == {"general", "REIT", "Mining"}, set(out["pool"])
    assert set(out["section"]) == {"per_column", "per_name"}


# --------------------------------------------------------------------------- #
#  The fetch loop's own hardening is NOT weakened                             #
# --------------------------------------------------------------------------- #
def test_safe_json_list_is_built_ON_safe_http_get_not_beside_it():
    """One transport, one retry policy.  A second hand-rolled retry loop would drift.

    Checked on the CODE only -- the docstring names `requests.get` when describing the defect,
    and a substring scan over the whole source would trip on the explanation rather than on a
    real call (which is exactly what it did on first run)."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(gdg.safe_json_list).strip())
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "safe_http_get" in calls, calls
    assert "requests.get" not in calls, calls
