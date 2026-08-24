"""
Payload-acceptance self-checks for the real-price fetch (OFFLINE; no network, ever).

WHAT WENT WRONG.  `run_bulk`'s only completeness test was `if rows:`, and three bodies in
the saved pulls satisfy it while being unusable:

    2020-12-31   9,901 rows   (repull of the SAME date: 55,438)
    2022-12-30  15,441 rows   (repull of the SAME date: 76,690)
    2024-12-28   3,589 rows   (a SATURDAY; 93.8% USD-suffixed crypto pairs, zero FX)

The two truncations are VENUE-STRUCTURED, not byte-truncated and not throttled -- the
9,901-row body is 98% foreign-suffixed against 48.7% in its own repull.  So these tests
pin a MAGNITUDE rule and nothing about the endpoint's internals; there is deliberately no
retry-the-same-date test, because there is deliberately no retry-the-same-date code.

WHAT THESE TESTS CANNOT SEE, STATED SO NOBODY MISREADS A GREEN RUN.  Every assertion here
is about the SIZE of a body.  A body missing one whole small venue (`.PA` is ~569 names of
~60,000) is 99% of full size, passes every test in this file, and is exactly the defect
that left seven venues unpriceable at every anchor of the run machine's grid.  A green
suite here does NOT mean a written body is complete.

NO NETWORK.  Every test replaces `fetch_prices.fetch_bulk_for_date` with a table lookup AND
poisons `requests.get`, so a code path that reached HTTP would fail the test rather than
spend a call.  `MUTATION` tests re-open the defect on purpose to prove the guard tests are
sensitive to the guard rather than to the fixture.
"""
import os
import sys
from datetime import date, datetime

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fetch_prices as fp

#  The real counts, from the five saved pulls.  Used as fixture sizes so the tests are
#  exercised at the magnitudes that actually occurred rather than at toy numbers.
TRUNCATED_2020 = 9901
REPULL_2020 = 55438
TRUNCATED_2022 = 15441
REPULL_2022 = 76690
SATURDAY_2024 = 3589
GOOD_BODIES = [45662, 51646, 55438, 58838, 59965, 60152, 62189, 64490, 71985, 73758, 76690]


# --------------------------------------------------------------------------- #
#  Harness                                                                    #
# --------------------------------------------------------------------------- #
def _body(n, prefix="S"):
    """A synthetic bulk payload of `n` rows in the shape `_extract` consumes."""
    return [{"symbol": f"{prefix}{i}", "adjClose": 10.0 + i} for i in range(n)]


@pytest.fixture
def offline(monkeypatch):
    """Install a date -> row-count table as the fetch, and poison the HTTP client.

    Returns a `call(sizes)` factory; the returned object records every date actually
    requested, so a test can assert on what was NOT asked for (the weekend case) as well
    as on what came back.
    """
    def boom(*a, **k):                      # pragma: no cover - must never run
        raise AssertionError("a test reached requests.get -- this suite is offline")
    monkeypatch.setattr(fp.requests, "get", boom)
    monkeypatch.setattr(fp.time, "sleep", lambda *_a, **_k: None)

    class Recorder:
        def __init__(self, sizes):
            self.sizes = sizes
            self.requested = []

        def fetch(self, api_key, date_str, timeout=30):
            assert api_key == "NOT-A-KEY", "the fake fetch got an unexpected key"
            self.requested.append(date_str)
            return _body(self.sizes.get(date_str, 0), prefix=date_str + "-")

    def install(sizes):
        r = Recorder(sizes)
        monkeypatch.setattr(fp, "fetch_bulk_for_date", r.fetch)
        return r
    return install


def _run(tmp_path, sizes, anchors, offline, max_lookback=4, symbols_filter=None):
    rec = offline(sizes)
    out = tmp_path / "prices.csv"
    result = fp.run_bulk("NOT-A-KEY", anchors, max_lookback, symbols_filter, str(out))
    #  ARITY-TOLERANT UNPACK, and it is not defensive noise.  `run_bulk` returned
    #  (calls, written) before the floor was added and returns (calls, written, refused)
    #  after.  A hard 3-tuple unpack made EVERY behavioural test below die with a TypeError
    #  on this line -- before a single assertion ran -- when measured against the
    #  pre-change code.  That is a tautological failure: it proves the signature moved, not
    #  that the guard works, and counting it as a kill inflates the score.  Tolerating the
    #  arity makes each test fail on the ASSERTION it is actually about
    #  ("assert [] == ['2020-12-31']"), which is the only outcome worth counting.
    calls, written = result[0], result[1]
    refused = result[2] if len(result) > 2 else []
    df = pd.read_csv(out) if os.path.getsize(out) else pd.DataFrame()
    return dict(rec=rec, calls=calls, written=written, refused=refused, df=df,
                out=str(out))


# --------------------------------------------------------------------------- #
#  BEHAVIOURAL -- the three bodies that actually shipped                      #
# --------------------------------------------------------------------------- #
def test_BEHAVIOURAL_the_9901_row_2020_truncation_is_rejected_and_the_step_back_is_written(
        tmp_path, offline):
    """The exact 2020-12-31 case: `if rows:` took the 9,901-row body; the floor does not.

    2020-12-31 is a Thursday and 2020-12-30 a Wednesday, so the step-back stays on
    weekdays and the weekend guard is not what is doing the work here.
    """
    anchors = [date(2020, 12, 31)]
    r = _run(tmp_path, {"2020-12-31": TRUNCATED_2020,
                        "2020-12-30": REPULL_2020}, anchors, offline)
    assert r["rec"].requested == ["2020-12-31", "2020-12-30"]
    assert set(r["df"]["date_actual"]) == {"2020-12-30"}
    assert r["written"] == REPULL_2020
    assert r["refused"] == []
    #  and the truncated body's rows are nowhere in the file
    assert not r["df"]["symbol"].astype(str).str.startswith("2020-12-31-").any()


def test_BEHAVIOURAL_the_15441_row_2022_truncation_is_rejected_too(tmp_path, offline):
    anchors = [date(2022, 12, 30)]
    r = _run(tmp_path, {"2022-12-30": TRUNCATED_2022,
                        "2022-12-29": REPULL_2022}, anchors, offline)
    assert set(r["df"]["date_actual"]) == {"2022-12-29"}
    assert r["written"] == REPULL_2022


def test_BEHAVIOURAL_the_2024_12_28_saturday_is_never_even_requested(tmp_path, offline):
    """The Saturday body was accepted by `if rows:`.  Now no call is spent on it at all.

    2024-12-31 is a Tuesday, so a 4-day lookback walks 12-31, 12-30 (Mon), 12-29 (Sun),
    12-28 (Sat), 12-27 (Fri).  The two weekend candidates must be absent from the
    request log entirely -- not requested-and-rejected.
    """
    anchors = [date(2024, 12, 31)]
    sizes = {"2024-12-31": 0, "2024-12-30": 0,
             "2024-12-29": 99999,        # a Sunday: would be taken if requested
             "2024-12-28": SATURDAY_2024,
             "2024-12-27": 62189}
    r = _run(tmp_path, sizes, anchors, offline)
    assert "2024-12-28" not in r["rec"].requested
    assert "2024-12-29" not in r["rec"].requested
    assert r["rec"].requested == ["2024-12-31", "2024-12-30", "2024-12-27"]
    assert set(r["df"]["date_actual"]) == {"2024-12-27"}


def test_BEHAVIOURAL_a_body_above_the_absolute_floor_but_below_half_the_median_is_refused(
        tmp_path, offline):
    """The relative rule's own job: 25,000 rows clears MIN_PAYLOAD_ROWS and still is not a
    trading day against a run whose other bodies median ~60,000.

    This is the case the absolute backstop cannot see, so it is the test that proves the
    two rules are not redundant.
    """
    anchors = [date(2018, 12, 31), date(2019, 12, 31), date(2020, 12, 31),
               date(2021, 12, 31), date(2022, 12, 30)]
    sizes = {"2018-12-31": 45662, "2019-12-31": 51646,
             "2020-12-31": 25000,                       # short, but > 20,000
             "2021-12-31": 60152, "2022-12-30": 76690}
    r = _run(tmp_path, sizes, anchors, offline)
    assert [x["date_str"] for x in r["refused"]] == ["2020-12-31"]
    assert r["refused"][0]["n_payload"] == 25000
    assert "2020-12-31" not in set(r["df"]["date_requested"])
    #  every OTHER anchor is present and untouched
    assert set(r["df"]["date_requested"]) == {"2018-12-31", "2019-12-31",
                                              "2021-12-31", "2022-12-30"}
    assert r["written"] == 45662 + 51646 + 60152 + 76690


def test_BEHAVIOURAL_a_healthy_run_refuses_nothing_and_writes_every_anchor(
        tmp_path, offline):
    """The false-positive side.  The eleven observed good bodies span 45,662..76,690, a
    1.68x spread; the rule must not fire anywhere inside that.

    Anchors are built through `nearest_weekday_on_or_before`, the same pre-mapping
    `build_anchor_dates` applies, so no anchor is itself a weekend -- an earlier draft of
    this test used raw Dec-31 dates and 2022-12-31/2023-12-31 are a Saturday and a Sunday.
    """
    anchors = [fp.nearest_weekday_on_or_before(datetime(2018 + i, 12, 31)).date()
               for i in range(len(GOOD_BODIES))]
    sizes = {a.strftime("%Y-%m-%d"): n for a, n in zip(anchors, GOOD_BODIES)}
    r = _run(tmp_path, sizes, anchors, offline)
    assert r["refused"] == []
    assert len(set(r["df"]["date_requested"])) == len(anchors)
    assert r["written"] == sum(GOOD_BODIES)
    assert r["calls"] == len(anchors)          # one call per anchor, no lookback spent


def test_BEHAVIOURAL_the_floor_is_measured_on_the_payload_not_on_the_kept_rows(
        tmp_path, offline):
    """`--symbols-file` can legitimately cut a full body to a handful of rows.  The
    completeness question is about what the endpoint returned.

    The run-machine grid is exactly this shape: ~5,000-9,000 rows per anchor after a symbol
    filter, off full-size payloads.  A floor applied after the filter would refuse the
    entire grid.
    """
    anchors = [date(2021, 12, 31)]
    keep = {"2021-12-31-7"}
    r = _run(tmp_path, {"2021-12-31": 60152}, anchors, offline, symbols_filter=keep)
    assert r["refused"] == []
    assert r["written"] == 1


def test_BEHAVIOURAL_a_single_date_run_still_rejects_a_truncated_body(tmp_path, offline):
    """The hole the absolute backstop exists for: with one anchor there is no median to
    compare against, so the relative rule is vacuous by construction."""
    anchors = [date(2020, 12, 31)]
    r = _run(tmp_path, {"2020-12-31": TRUNCATED_2020}, anchors, offline)
    assert r["df"].empty
    assert r["written"] == 0
    #  refused-by-median is EMPTY -- it was the in-line absolute floor that caught it, and
    #  saying so keeps the two mechanisms distinguishable in the report.
    assert r["refused"] == []


def test_BEHAVIOURAL_leave_one_out_median_survives_a_two_anchor_run(tmp_path, offline):
    """A pooled median over {25,000, 60,152} is 42,576 and neither body is below half of
    it, so the truncation would pass.  Leave-one-out judges 25,000 against 60,152."""
    anchors = [date(2020, 12, 31), date(2021, 12, 31)]
    r = _run(tmp_path, {"2020-12-31": 25000, "2021-12-31": 60152}, anchors, offline)
    assert [x["date_str"] for x in r["refused"]] == ["2020-12-31"]


# --------------------------------------------------------------------------- #
#  MUTATION -- reopen each defect and prove the tests above see it            #
# --------------------------------------------------------------------------- #
def test_MUTATION_with_the_absolute_floor_disabled_the_truncated_body_is_written_again(
        tmp_path, offline, monkeypatch):
    """Sensitivity check on the 2020 test.  With MIN_PAYLOAD_ROWS back at the old
    `if rows:` behaviour (1 row is enough), the 9,901-row body is taken and the
    step-back never happens -- i.e. the assertion above is testing the guard."""
    monkeypatch.setattr(fp, "MIN_PAYLOAD_ROWS", 1)
    anchors = [date(2020, 12, 31)]
    r = _run(tmp_path, {"2020-12-31": TRUNCATED_2020,
                        "2020-12-30": REPULL_2020}, anchors, offline)
    assert r["rec"].requested == ["2020-12-31"]
    assert set(r["df"]["date_actual"]) == {"2020-12-31"}
    assert r["written"] == TRUNCATED_2020


def test_MUTATION_with_the_median_fraction_at_zero_the_short_body_is_written_again(
        tmp_path, offline, monkeypatch):
    """Sensitivity check on the relative-floor test.  `run_bulk` must read the constant at
    call time, not have baked it in at import."""
    monkeypatch.setattr(fp, "SHORT_BODY_MEDIAN_FRACTION", 0.0)
    anchors = [date(2018, 12, 31), date(2019, 12, 31), date(2020, 12, 31),
               date(2021, 12, 31), date(2022, 12, 30)]
    sizes = {"2018-12-31": 45662, "2019-12-31": 51646, "2020-12-31": 25000,
             "2021-12-31": 60152, "2022-12-30": 76690}
    r = _run(tmp_path, sizes, anchors, offline)
    assert r["refused"] == []
    assert "2020-12-31" in set(r["df"]["date_requested"])


def test_MUTATION_with_the_weekend_guard_disabled_the_saturday_body_is_taken_again(
        tmp_path, offline, monkeypatch):
    """Sensitivity check on the Saturday test."""
    monkeypatch.setattr(fp, "is_weekend", lambda d: False)
    anchors = [date(2024, 12, 31)]
    sizes = {"2024-12-31": 0, "2024-12-30": 0, "2024-12-29": 99999,
             "2024-12-28": SATURDAY_2024, "2024-12-27": 62189}
    r = _run(tmp_path, sizes, anchors, offline)
    assert "2024-12-29" in r["rec"].requested
    assert set(r["df"]["date_actual"]) == {"2024-12-29"}


# --------------------------------------------------------------------------- #
#  UNIT -- the two predicates, stated directly                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d,expected", [
    (date(2024, 12, 28), True),    # the Saturday that shipped
    (date(2024, 12, 29), True),    # Sunday
    (date(2024, 12, 27), False),   # Friday
    (date(2024, 12, 30), False),   # Monday
    (date(2018, 12, 31), False),   # Monday -- the holiday-damaged anchor is NOT a weekend
])
def test_is_weekend(d, expected):
    assert fp.is_weekend(d) is expected


def test_median_of_handles_the_empty_and_even_cases():
    assert fp.median_of([]) is None
    assert fp.median_of([60152]) == 60152
    assert fp.median_of([25000, 60152]) == 42576.0
    assert fp.median_of(GOOD_BODIES) == 60152


def test_short_against_median_is_inert_without_a_median():
    """A one-anchor run must not be able to refuse itself."""
    assert fp.short_against_median(9901, None) is False
    assert fp.short_against_median(9901, 0) is False
    assert fp.short_against_median(9901, 60152) is True
    assert fp.short_against_median(45662, 60152) is False


def test_the_observed_populations_are_separated_by_the_shipped_constants():
    """The derivation, asserted rather than left in a comment: every observed bad body is
    refused and every observed good body is kept, by the numbers that shipped."""
    med = fp.median_of(GOOD_BODIES)
    for bad in (SATURDAY_2024, TRUNCATED_2020, TRUNCATED_2022):
        assert bad < fp.MIN_PAYLOAD_ROWS
    for good in GOOD_BODIES:
        assert good >= fp.MIN_PAYLOAD_ROWS
        assert not fp.short_against_median(good, med)


# --------------------------------------------------------------------------- #
#  STRUCTURAL -- the wiring, so a future edit cannot orphan the constants     #
# --------------------------------------------------------------------------- #
def _names_in(fn):
    import ast
    import inspect
    return {n.id for n in ast.walk(ast.parse(inspect.getsource(fn)))
            if isinstance(n, ast.Name)}


def test_STRUCTURAL_run_bulk_consults_both_acceptance_tests_and_the_weekend_guard():
    """A guard that is defined but no longer called is the failure this pins.  Read off the
    AST rather than the text so a mention in a comment does not satisfy it."""
    names = _names_in(fp.run_bulk)
    assert "is_weekend" in names
    assert "body_is_acceptable" in names
    assert "refusals_against_median" in names


def test_STRUCTURAL_the_acceptance_helpers_are_the_ONE_definition_of_the_rule():
    """The floor lives in two callers now -- this script and
    `pipeline_analysis._fetch_bulk_scrubbed` -- so the rule itself must live in ONE place.
    Two copies of a correctness guard kept in lockstep by comment is the drift this project
    has already paid for once (see dead_merge.NA1_EXCHANGES)."""
    assert "MIN_PAYLOAD_ROWS" in _names_in(fp.body_is_acceptable)
    inner = _names_in(fp.refusals_against_median)
    assert "median_of" in inner and "short_against_median" in inner


def test_STRUCTURAL_the_pipeline_stage_uses_THOSE_helpers_and_not_a_bare_if_rows():
    """FIX 1, pinned.  `pipeline_analysis._fetch_bulk_scrubbed` is what the pipeline calls;
    `run_bulk` is called from nowhere in it.  The floor was therefore guarding a path
    production does not take, and every test above passed regardless."""
    import ast
    import inspect
    import pipeline_analysis as pa
    src = inspect.getsource(pa._fetch_bulk_scrubbed)
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    for helper in ("body_is_acceptable", "refusals_against_median", "is_weekend"):
        assert helper in names, (
            f"the PIPELINE fetch path does not consult {helper} -- B1 is guarding "
            "fetch_prices.run_bulk, which the pipeline never calls")
    #  CHECKED ON THE AST, NOT THE TEXT.  The function's own docstring quotes `if rows:`
    #  while explaining the defect, so a substring scan flags the sentence that documents
    #  the fix -- exactly the trap `test_the_audit_module_contains_no_network_surface`
    #  already had to dodge in test_price_grid_audit.
    tree = ast.parse(src)
    bare = [n for n in ast.walk(tree)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
            and n.test.id == "rows"]
    assert not bare, "the bare `if rows:` presence check is back on the real path"


def test_STRUCTURAL_no_retry_of_the_same_date_was_added():
    """The evidence contradicts a throttle (the short bodies are venue-clustered, not
    empty or errored), so a sleep-and-retry-the-same-date loop would be a mechanism
    nothing supports.  Asserted because it is the tempting wrong fix."""
    import inspect
    src = inspect.getsource(fp.run_bulk)
    #  the ONE sleep is the pre-existing politeness pause between step-back calls
    assert src.count("time.sleep") == 1
    assert "range(max_lookback + 1)" in src
