"""
Payload + per-venue acceptance ON THE PATH THE PIPELINE ACTUALLY TAKES (OFFLINE, no network).

WHAT WENT WRONG, AND IT IS THE WHOLE REASON THIS FILE EXISTS.  `fetch_prices.run_bulk` grew a
payload floor and a weekend guard.  `run_price_fetch_stage` does not call `run_bulk` -- it
calls `pipeline_analysis._fetch_bulk_scrubbed`, which still read `if rows:` -- and the stage
docstring said "via fetch_prices.run_bulk", which was false.  So every test in
`test_fetch_prices.py` passed while production ran unguarded.  A guard on an unexecuted code
path is worse than no guard: it reads as coverage.

THE PER-VENUE TEST USES THE PREVIOUS GRID, ANCHOR-MATCHED, and the two tests that matter most
here are the ones about what that choice buys and what it does not:

  * `test_a_genuine_year_end_venue_holiday_is_NOT_a_false_positive` -- a WITHIN-RUN reference
    ("a venue seen at another anchor must appear at this one") flags `.DE`/`.ST`/`.IC` at
    2018-12-31, when those exchanges were legitimately shut.  Matching the previous grid at
    the SAME anchor makes the reference carry each venue's own calendar, so the expectation
    simply does not exist there.
  * `test_a_venue_the_REFERENCE_also_lacks_stays_silent` -- the cost of an external
    reference.  The run machine's grid holds only ['(none)', '.DE', '.IC', '.L', '.ST',
    '.TO'], so the seven venues at the centre of this whole investigation (1,421 names)
    carry no expectation and their absence is NOT detected.  This test can DEFEND a venue
    set; it cannot BOOTSTRAP one.  Asserted, so nobody reads a green fetch as "all venues
    present".

NO NETWORK.  `delisted_ingest.safe_get_bulk_csv` is replaced by a table lookup and the log is
captured, so a path that reached HTTP would fail rather than spend a call.
"""
import os
import sys
from datetime import date

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fetch_prices as fp
import pipeline_analysis as pa

KEY = "NOT-A-REAL-KEY"
BASE = "https://example.invalid/api/"

#  The real counts, so the guards are exercised at the magnitudes that actually occurred.
TRUNCATED_2020 = 9901
REPULL_2020 = 55438
SATURDAY_2024 = 3589


def _sym(tag, venue, i):
    """The one place a fixture symbol is spelled, so a test cannot guess it wrong.

    An earlier draft built names as f"{tag}{venue}{i}{suffix}", which for `.DE` produced
    `a.DE0.DE` -- and the test that filtered on `a.DE0` silently kept one row instead of two
    and asserted the wrong number.  One helper, used by both the payload and the allow-list.
    """
    return f"{tag}{i}" if venue == "(none)" else f"{tag}{i}{venue}"


def _body(counts, tag="x"):
    """A synthetic bulk payload from ABSOLUTE per-venue counts.

    `counts` is an int (US-only) or {venue: n}.  Absolute, deliberately: a first draft took
    proportional weights, so a fixture asking for "3 .DE names out of 50,000" actually
    produced 187 and the partial-venue-loss test passed for the wrong reason -- it never
    tripped the guard at all.
    """
    if isinstance(counts, int):
        counts = {"(none)": counts}
    rows = []
    for venue, k in counts.items():
        rows.extend({"symbol": _sym(tag, venue, i), "adjClose": 10.0 + i}
                    for i in range(k))
    return rows


@pytest.fixture
def offline(monkeypatch):
    """Install a date -> payload table as the HTTP layer; record requested dates and log."""
    import delisted_ingest as di

    class Rec:
        def __init__(self, bodies):
            self.bodies = bodies
            self.requested = []
            self.log_lines = []

        def get(self, url):
            #  the date is the only thing this fake needs, and taking it off the URL proves
            #  the caller really did build a per-date request
            ds = url.split("date=")[1].split("&")[0]
            self.requested.append(ds)
            return self.bodies.get(ds, [])

        def log(self, *a):
            self.log_lines.append(" ".join(str(x) for x in a))

    def install(bodies):
        r = Rec(bodies)
        monkeypatch.setattr(di, "safe_get_bulk_csv", r.get)
        return r
    return install


def _grid(tmp_path, name, per_anchor):
    """per_anchor = {date_requested: [symbols]}.  Writes the on-disk grid schema."""
    rows = [{"date_requested": a, "date_actual": a, "symbol": s, "adjClose": 1.0}
            for a, syms in per_anchor.items() for s in syms]
    p = tmp_path / name
    pd.DataFrame(rows, columns=["date_requested", "date_actual", "symbol",
                                "adjClose"]).to_csv(p, index=False)
    return str(p)


def _fetch(tmp_path, bodies, anchors, offline, reference_paths=None, symbols_filter=None,
           companion_days=0):
    """COMPANION DAYS DEFAULT TO 0 HERE, and that is a deliberate scoping choice rather than a
    guard being switched off.

    Every test in this file is about ANCHOR-BODY ACCEPTANCE -- the absolute payload floor, the
    per-venue shortfall test, the deferred relative-median floor.  The companion pull added in
    2026-08-27 fetches ADDITIONAL preceding weekdays under their own `date_requested`, which
    is orthogonal to all three: it changes what else lands in the file, not which anchor body
    is accepted.  Leaving it on would have made five of these tests fail on their fixture
    rather than on their subject (a step-back date supplied as a fallback is also a companion
    candidate), which measures the harness, not the guard.

    So the companion behaviour is held FIXED here and tested EXPLICITLY in its own section at
    the bottom of this file, where the assertions are about companions.
    """
    rec = offline(bodies)
    out = tmp_path / "out.csv"
    #  ARITY- AND SIGNATURE-TOLERANT.  Before this change `_fetch_bulk_scrubbed` took no
    #  `reference_paths` and returned (calls, written); a hard call would raise TypeError on
    #  this line and every behavioural test below would die before asserting anything --
    #  a failure that proves only that the signature moved.  Tolerating both makes each test
    #  fail on its own assertion instead, which is the only countable outcome.
    kw = {}
    try:
        import inspect
        params = inspect.signature(pa._fetch_bulk_scrubbed).parameters
        if "reference_paths" in params:
            kw["reference_paths"] = reference_paths
        if "companion_days" in params:
            kw["companion_days"] = companion_days
    except (TypeError, ValueError):     # pragma: no cover - unintrospectable callable
        kw["reference_paths"] = reference_paths
    result = pa._fetch_bulk_scrubbed(BASE, KEY, anchors, symbols_filter, str(out),
                                     rec.log, **kw)
    calls, written = result[0], result[1]
    refused = result[2] if len(result) > 2 else []
    vf = result[3] if len(result) > 3 else []
    df = pd.read_csv(out) if os.path.getsize(out) else pd.DataFrame()
    return dict(rec=rec, calls=calls, written=written, refused=refused, venue=vf, df=df)


# --------------------------------------------------------------------------- #
#  BEHAVIOURAL -- the three bodies that shipped, now on the REAL path          #
# --------------------------------------------------------------------------- #
def test_BEHAVIOURAL_the_9901_row_truncation_is_rejected_on_the_PIPELINE_path(tmp_path,
                                                                             offline):
    r = _fetch(tmp_path, {"2020-12-31": _body(TRUNCATED_2020),
                          "2020-12-30": _body(REPULL_2020)},
               [date(2020, 12, 31)], offline)
    assert r["rec"].requested == ["2020-12-31", "2020-12-30"]
    assert set(r["df"]["date_actual"]) == {"2020-12-30"}
    assert r["written"] == REPULL_2020


def test_BEHAVIOURAL_the_saturday_is_never_requested_on_the_PIPELINE_path(tmp_path,
                                                                         offline):
    bodies = {"2024-12-31": [], "2024-12-30": [],
              "2024-12-29": _body(99999),          # a Sunday: taken if requested
              "2024-12-28": _body(SATURDAY_2024),
              "2024-12-27": _body(62189)}
    r = _fetch(tmp_path, bodies, [date(2024, 12, 31)], offline)
    assert "2024-12-28" not in r["rec"].requested
    assert "2024-12-29" not in r["rec"].requested
    assert set(r["df"]["date_actual"]) == {"2024-12-27"}


def test_BEHAVIOURAL_the_deferred_median_floor_refuses_on_the_PIPELINE_path(tmp_path,
                                                                           offline):
    anchors = [date(2018, 12, 31), date(2019, 12, 31), date(2020, 12, 31),
               date(2021, 12, 31), date(2022, 12, 30)]
    sizes = {"2018-12-31": 45662, "2019-12-31": 51646,
             "2020-12-31": 25000,                    # short, but above the absolute floor
             "2021-12-31": 60152, "2022-12-30": 76690}
    r = _fetch(tmp_path, {d: _body(n) for d, n in sizes.items()}, anchors, offline)
    assert [x["anchor"] for x in r["refused"]] == ["2020-12-31"]
    assert "2020-12-31" not in set(r["df"]["date_requested"])
    assert r["written"] == 45662 + 51646 + 60152 + 76690


def test_BEHAVIOURAL_a_single_anchor_fetch_still_rejects_a_truncated_body(tmp_path,
                                                                         offline):
    """The SUPP leg is exactly this: one anchor, so the median rule is vacuous and only the
    absolute floor can fire."""
    r = _fetch(tmp_path, {"2025-12-31": _body(TRUNCATED_2020)},
               [date(2025, 12, 31)], offline)
    assert r["df"].empty and r["written"] == 0
    assert r["refused"] == []          # caught in-line, not by the median rule


def test_BEHAVIOURAL_a_healthy_fetch_refuses_nothing(tmp_path, offline):
    anchors = [fp.nearest_weekday_on_or_before(pd.Timestamp(y, 12, 31)).date()
               for y in (2019, 2020, 2021, 2022)]
    sizes = [51646, 58838, 60152, 64490]
    r = _fetch(tmp_path, {a.strftime("%Y-%m-%d"): _body(n)
                          for a, n in zip(anchors, sizes)}, anchors, offline)
    assert r["refused"] == [] and r["venue"] == []
    assert r["calls"] == len(anchors)
    assert r["written"] == sum(sizes)


# --------------------------------------------------------------------------- #
#  BEHAVIOURAL -- the per-venue completeness test (FIX 3)                      #
# --------------------------------------------------------------------------- #
#  The 15,441-row 2022-12-30 body was ~half UK+Canada with Germany and Sweden absent
#  ENTIRELY, at a size the absolute floor does not catch once the vendor universe grows.
#  That is the shape below.
#  Absolute counts.  Both bodies clear the absolute floor and sit within 1.25x of each
#  other, so no payload rule can tell them apart -- only the venue comparison can.
FULL = {"(none)": 48000, ".DE": 8000, ".ST": 8000, ".L": 8000, ".TO": 8000}
NO_DE_ST = {"(none)": 48000, ".L": 8000, ".TO": 8000}


def test_a_venue_clustered_truncation_is_stepped_past_using_the_previous_grid(tmp_path,
                                                                            offline):
    """THE FIX-3 CASE.  Both bodies clear the absolute floor and are within a factor of 1.25
    of each other, so neither payload rule can see the difference -- only the venue
    comparison can."""
    ref = _grid(tmp_path, "ref.csv",
                {"2022-12-30": [f"r{i}.DE" for i in range(200)]
                                + [f"s{i}.ST" for i in range(200)]
                                + [f"u{i}" for i in range(800)]})
    bodies = {"2022-12-30": _body(NO_DE_ST, tag="bad"),
              "2022-12-29": _body(FULL, tag="good")}
    r = _fetch(tmp_path, bodies, [date(2022, 12, 30)], offline, reference_paths=[ref])
    assert set(r["df"]["date_actual"]) == {"2022-12-29"}
    assert len(r["venue"]) == 1
    lost = {v for v, _n, _ref in r["venue"][0]["shortfall"]}
    assert lost == {".DE", ".ST"}
    assert r["venue"][0]["anchor"] == "2022-12-30"


def test_a_genuine_year_end_venue_holiday_is_NOT_a_false_positive(tmp_path, offline):
    """THE OBJECTION TO A WITHIN-RUN REFERENCE, answered.  `.DE`/`.ST` are legitimately zero
    at 2018-12-31 and present at 2019-12-31.  A within-run rule ("seen at another anchor, so
    expected here") rejects the 2018 body forever.  Anchor-matching against the previous grid
    means 2018 carries NO expectation for them, because the reference has none either."""
    ref = _grid(tmp_path, "ref.csv", {
        "2018-12-31": [f"u{i}" for i in range(800)],                       # no .DE/.ST
        "2019-12-31": [f"u{i}" for i in range(800)]
                      + [f"r{i}.DE" for i in range(200)]
                      + [f"s{i}.ST" for i in range(200)]})
    bodies = {"2018-12-31": _body(NO_DE_ST, tag="a"),
              "2019-12-31": _body(FULL, tag="b")}
    r = _fetch(tmp_path, bodies, [date(2018, 12, 31), date(2019, 12, 31)], offline,
               reference_paths=[ref])
    assert r["venue"] == [], "a legitimate year-end venue holiday was flagged"
    assert set(r["df"]["date_requested"]) == {"2018-12-31", "2019-12-31"}
    assert r["calls"] == 2, "no lookback step was wasted on a false positive"


def test_a_venue_the_REFERENCE_also_lacks_stays_silent(tmp_path, offline):
    """THE STATED BLINDNESS, asserted rather than promised.  The run machine's grid has no
    `.PA`/`.KS`/`.OL` rows at any anchor, so a new body missing them raises nothing.  This
    test DEFENDS a venue set; it cannot BOOTSTRAP one."""
    ref = _grid(tmp_path, "ref.csv",
                {"2021-12-31": [f"u{i}" for i in range(800)]})   # US only, like the run grid
    bodies = {"2021-12-31": _body(50000, tag="a")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31)], offline, reference_paths=[ref])
    assert r["venue"] == []
    assert r["written"] > 0


def test_with_NO_previous_grid_the_venue_test_is_inert_and_says_so(tmp_path, offline):
    """The pipeline's own common case: the stage fetches only when the file is ABSENT."""
    bodies = {"2021-12-31": _body(NO_DE_ST, tag="a")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31)], offline,
               reference_paths=[str(tmp_path / "nope.csv")])
    assert r["venue"] == []
    assert any("INERT" in line for line in r["rec"].log_lines), \
        "an inert completeness test must announce itself, not pass silently"


def test_a_small_reference_venue_does_not_trip_the_test(tmp_path, offline):
    """Below VENUE_MIN_REFERENCE_ROWS, "absent" is about a handful of names rather than about
    the fetch, so it must not reject an anchor."""
    tiny = pa.VENUE_MIN_REFERENCE_ROWS - 1
    ref = _grid(tmp_path, "ref.csv",
                {"2021-12-31": [f"u{i}" for i in range(800)]
                                + [f"t{i}.IC" for i in range(tiny)]})
    r = _fetch(tmp_path, {"2021-12-31": _body(50000, tag="a")},
               [date(2021, 12, 31)], offline, reference_paths=[ref])
    assert r["venue"] == []


def test_a_partial_venue_loss_is_caught_not_only_a_total_one(tmp_path, offline):
    """Presence alone would miss a body that keeps 3 of 200 `.DE` names, which is why the
    test is a SHARE of the reference count and not a set membership."""
    ref = _grid(tmp_path, "ref.csv",
                {"2021-12-31": [f"u{i}" for i in range(800)]
                                + [f"r{i}.DE" for i in range(200)]})
    bodies = {"2021-12-31": _body({"(none)": 48000, ".DE": 3}, tag="a"),
              "2021-12-30": _body({"(none)": 48000, ".DE": 200}, tag="b")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31)], offline, reference_paths=[ref])
    #  ASSERTED, not indexed.  Subscripting r["venue"][0] straight away turned "the guard
    #  did not fire" into an IndexError, which reads as a broken test rather than as a
    #  detected defect.  Same reasoning as the arity-tolerant unpack in `_fetch`.
    assert r["venue"], "the partial venue loss (3 of 200 .DE names) was NOT detected"
    assert [v for v, _n, _r in r["venue"][0]["shortfall"]] == [".DE"]
    assert r["venue"][0]["shortfall"][0][1:] == (3, 200)
    assert set(r["df"]["date_actual"]) == {"2021-12-30"}


def test_the_reference_is_compared_AFTER_the_local_symbol_filter(tmp_path, offline):
    """Like-for-like: the reference grid was itself written through the allow-list, so
    comparing a PRE-filter body against a POST-filter reference would report a shortfall on
    every venue and reject every anchor forever.

    Built so the naive version fails: the reference has 20 `.DE` rows (above the min-rows
    floor) and the allow-list keeps 20 of the body's 8,000 `.DE` names.  Comparing the raw
    payload would read 8,000-vs-20 as fine, but comparing the KEPT rows reads 20-vs-20 --
    and a version that compared the wrong side of the filter would drift the other way as
    soon as the allow-list narrowed.
    """
    ref = _grid(tmp_path, "ref.csv",
                {"2021-12-31": [f"u{i}" for i in range(40)]
                                + [f"r{i}.DE" for i in range(20)]})
    keep = ({_sym("a", "(none)", i) for i in range(40)}
            | {_sym("a", ".DE", i) for i in range(20)})
    r = _fetch(tmp_path, {"2021-12-31": _body(FULL, tag="a")},
               [date(2021, 12, 31)], offline, reference_paths=[ref],
               symbols_filter=keep)
    assert r["venue"] == []
    assert r["written"] == 60


def test_an_unreadable_reference_is_skipped_not_fatal(tmp_path, offline):
    bad = tmp_path / "bad.csv"
    bad.write_text("this is not a price grid\n", encoding="utf-8")
    r = _fetch(tmp_path, {"2021-12-31": _body(FULL, tag="a")},
               [date(2021, 12, 31)], offline, reference_paths=[str(bad)])
    assert r["written"] > 0
    assert any("unreadable" in line for line in r["rec"].log_lines)


# --------------------------------------------------------------------------- #
#  MUTATION -- reopen each guard on the real path                              #
# --------------------------------------------------------------------------- #
def test_MUTATION_with_the_absolute_floor_disabled_the_truncation_is_written_again(
        tmp_path, offline, monkeypatch):
    monkeypatch.setattr(fp, "MIN_PAYLOAD_ROWS", 1)
    r = _fetch(tmp_path, {"2020-12-31": _body(TRUNCATED_2020),
                          "2020-12-30": _body(REPULL_2020)},
               [date(2020, 12, 31)], offline)
    assert r["rec"].requested == ["2020-12-31"]
    assert r["written"] == TRUNCATED_2020


def test_MUTATION_with_the_weekend_guard_disabled_the_saturday_is_taken_again(
        tmp_path, offline, monkeypatch):
    monkeypatch.setattr(fp, "is_weekend", lambda d: False)
    bodies = {"2024-12-31": [], "2024-12-30": [], "2024-12-29": _body(99999),
              "2024-12-28": _body(SATURDAY_2024), "2024-12-27": _body(62189)}
    r = _fetch(tmp_path, bodies, [date(2024, 12, 31)], offline)
    assert "2024-12-29" in r["rec"].requested
    assert set(r["df"]["date_actual"]) == {"2024-12-29"}


def test_MUTATION_with_the_venue_share_at_zero_the_clustered_truncation_ships(
        tmp_path, offline, monkeypatch):
    """Sensitivity check on the Fix-3 test: at share 0 no venue can ever read as absent."""
    monkeypatch.setattr(pa, "VENUE_MIN_SURVIVING_SHARE", 0.0)
    ref = _grid(tmp_path, "ref.csv",
                {"2022-12-30": [f"r{i}.DE" for i in range(200)]
                                + [f"s{i}.ST" for i in range(200)]
                                + [f"u{i}" for i in range(800)]})
    bodies = {"2022-12-30": _body(NO_DE_ST, tag="bad"),
              "2022-12-29": _body(FULL, tag="good")}
    r = _fetch(tmp_path, bodies, [date(2022, 12, 30)], offline, reference_paths=[ref])
    assert r["venue"] == []
    assert set(r["df"]["date_actual"]) == {"2022-12-30"}


# --------------------------------------------------------------------------- #
#  The key must not reach the log                                              #
# --------------------------------------------------------------------------- #
def test_the_api_key_never_appears_in_the_stage_log(tmp_path, offline):
    """The URL carries the key; the log lines must not.  `safe_get_bulk_csv` scrubs its own
    errors, but the per-call log line is written by this module."""
    r = _fetch(tmp_path, {"2021-12-31": _body(FULL, tag="a")},
               [date(2021, 12, 31)], offline)
    assert r["rec"].log_lines
    for line in r["rec"].log_lines:
        assert KEY not in line
        assert "apikey" not in line.lower()


def test_reference_paths_resolution_order(tmp_path):
    """Operator-named first, then the file about to be overwritten, then a sibling .bak."""
    out = str(tmp_path / "real_prices.csv")
    assert pa._reference_paths({}, out) == [None, out, out + ".bak"]
    assert pa._reference_paths({"price_grid_reference": "/x/y.csv"}, out)[0] == "/x/y.csv"


@pytest.mark.parametrize("off", [0, "0", "false", "False"])
def test_the_venue_check_has_an_off_switch(off):
    """It is the one guard here that can leave an anchor with NO body -- a venue legitimately
    delisted since the reference reads as a shortfall -- so it must be switchable without a
    code edit."""
    assert pa._reference_paths({"price_grid_venue_check": off}, "/x/real_prices.csv") == []


def test_the_venue_check_is_ON_by_default_and_stays_on_for_unrelated_config():
    assert pa._reference_paths({}, "/x/p.csv") != []
    assert pa._reference_paths({"price_grid_venue_check": 1}, "/x/p.csv") != []
    assert pa._reference_paths({"something_else": 0}, "/x/p.csv") != []


def test_a_truncated_reference_makes_the_test_WEAKER_never_spuriously_stricter(tmp_path,
                                                                              offline):
    """The property that makes it safe to point this at the file being overwritten -- which
    may be the corrupted one.  The test fires only when the NEW body has FEWER rows for a
    venue than the reference, so a thin reference degrades toward silence."""
    thin_ref = _grid(tmp_path, "thin.csv", {"2021-12-31": [f"u{i}" for i in range(20)]})
    r = _fetch(tmp_path, {"2021-12-31": _body(FULL, tag="a")},
               [date(2021, 12, 31)], offline, reference_paths=[thin_ref])
    assert r["venue"] == [], "a truncated reference rejected a FULL body"
    assert r["written"] > 0


# --------------------------------------------------------------------------- #
#  COMPANION DAYS -- the HOLIDAY half of the coverage problem                  #
#                                                                             #
#  Removing the save-side allow-list recovers venues that were IN the anchor   #
#  body and thrown away.  It does nothing for venues that shut BEFORE the      #
#  calendar year-end and are therefore absent from that body: measured on the  #
#  already-unfiltered dev grid, `.KS` / `.KQ` / `.OL` (711 names) are zero at  #
#  2021-12-31 and 2024-12-31, BOTH legs of the buy2021 clean window.  These    #
#  pin the fetch-side fix for that half.                                       #
# --------------------------------------------------------------------------- #
def test_a_companion_day_is_pulled_and_written_under_ITS_OWN_date(tmp_path, offline):
    """The whole mechanism in one assertion.  The companion must NOT be written under the
    anchor: `PriceSource`s anchor layer selects on `date_requested`, so an anchor-labelled
    companion would silently compete to BE the anchor price instead of being a fill source
    for names the anchor lacks."""
    bodies = {"2021-12-31": _body(FULL, tag="anch"),
              "2021-12-30": _body(FULL, tag="comp")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31)], offline, companion_days=1)
    assert r["calls"] == 2
    got = set(zip(r["df"]["date_requested"], r["df"]["date_actual"]))
    assert ("2021-12-31", "2021-12-31") in got
    assert ("2021-12-30", "2021-12-30") in got
    #  and NEVER the anchor wearing the companion date
    assert ("2021-12-31", "2021-12-30") not in got


def test_the_companion_recovers_a_venue_ABSENT_from_the_anchor_body(tmp_path, offline):
    """The measured case, in miniature: a venue that shut before the year-end.  The anchor
    body genuinely has no `.KS`, so no amount of widening what we SAVE can produce it -- only
    holding the preceding trading day can."""
    anchor_body = _body({"(none)": 50000, ".DE": 800}, tag="a")
    comp_body = _body({"(none)": 50000, ".DE": 800, ".KS": 300}, tag="c")
    r = _fetch(tmp_path, {"2021-12-31": anchor_body, "2021-12-30": comp_body},
               [date(2021, 12, 31)], offline, companion_days=1)
    anchor_syms = set(r["df"][r["df"]["date_requested"] == "2021-12-31"]["symbol"])
    comp_syms = set(r["df"][r["df"]["date_requested"] == "2021-12-30"]["symbol"])
    assert not any(s.endswith(".KS") for s in anchor_syms)      # premise
    assert sum(1 for s in comp_syms if s.endswith(".KS")) == 300


def test_the_companion_is_reachable_by_the_READER_fill_layer(tmp_path, offline):
    """END-TO-END, and the only assertion that proves the fetch and the reader agree.  A
    companion written correctly but outside `PriceSource`s fill window would be dead bytes.
    Reads the file back through the real `PriceSource` and asks for the venue AT the anchor."""
    import returns_core as rc
    anchor_body = _body({"(none)": 50000}, tag="a")
    comp_body = _body({"(none)": 50000, ".KS": 300}, tag="c")
    r = _fetch(tmp_path, {"2021-12-31": anchor_body, "2021-12-30": comp_body},
               [date(2021, 12, 31)], offline, companion_days=1)
    assert r["written"] > 0
    ps = rc.PriceSource(str(tmp_path / "out.csv"), anchors=["2021-12-31"])
    ks_name = _sym("c", ".KS", 0)
    assert ps.price(ks_name, "2021-12-31") is not None, (
        "the companion is on disk but the reader cannot see it at the anchor")
    #  and the staleness it introduces is REPORTED, not silent
    rep = ps.fill_report()
    assert int(rep.loc[rep["anchor"] == "2021-12-31", "n_filled"].iloc[0]) >= 300


def test_companion_days_ZERO_restores_the_anchor_only_fetch(tmp_path, offline):
    """The off-switch, so a run can be put back to the previous shape without editing code."""
    bodies = {"2021-12-31": _body(FULL, tag="anch"),
              "2021-12-30": _body(FULL, tag="comp")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31)], offline, companion_days=0)
    assert r["calls"] == 1
    assert set(r["df"]["date_requested"]) == {"2021-12-31"}


def test_a_weekend_companion_costs_NO_call(tmp_path, offline):
    """Same rule as the anchor step-back: the endpoint answers a weekend with a small
    non-empty body, so a weekend is never requested.

    AN EARLIER DRAFT OF THIS TEST PASSED WITHOUT EVER CROSSING A WEEKEND -- it anchored on
    Friday 2021-12-31 and walked back to Thu/Wed/Tue, all weekdays, so the rule it claimed to
    check was never exercised.  Anchored on MONDAY 2021-12-27 the single companion day must
    step over Sunday 12-26 and Saturday 12-25 to reach Friday 12-24, and neither weekend day
    may cost a call."""
    bodies = {"2021-12-27": _body(FULL, tag="a"), "2021-12-24": _body(FULL, tag="b")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 27)], offline, companion_days=1)
    assert date(2021, 12, 27).strftime("%A") == "Monday"          # premise
    assert r["calls"] == 2, "a weekend day was requested"
    assert set(r["df"]["date_requested"]) == {"2021-12-27", "2021-12-24"}
    assert any("Sunday -- not requested" in ln for ln in r["rec"].log_lines)
    assert any("Saturday -- not requested" in ln for ln in r["rec"].log_lines)


def test_the_companion_walk_STOPS_at_the_reader_fill_window(tmp_path, offline):
    """A companion older than `returns_core.DEFAULT_FILL_WINDOW_DAYS` before the anchor is a
    date the reader can never consume, so requesting it spends money for nothing.  Ask for
    far more companion days than the window allows and the walk must stop, not keep paying."""
    import returns_core as rc
    bodies = {"2021-12-31": _body(FULL, tag="a")}
    for dd in range(20, 32):
        bodies[f"2021-12-{dd:02d}"] = _body(FULL, tag=f"d{dd}")
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31)], offline, companion_days=99)
    for req in set(r["df"]["date_requested"]):
        lag = (date(2021, 12, 31) - date(*[int(x) for x in req.split("-")])).days
        assert lag <= rc.DEFAULT_FILL_WINDOW_DAYS, f"{req} is outside the fill window"


def test_a_companion_that_is_ANOTHER_anchor_is_never_written_as_a_companion(tmp_path, offline):
    """2022-12-30 is an anchor of the real grid and 2022-12-29 could be a companion of it.
    The converse is the hazard: a companion landing on a date that IS an anchor would inject a
    second body into that anchor layer -- the "two payloads for one anchor" corruption this
    module already carries a scar from.  Adjacent anchors, so the collision is forced."""
    bodies = {"2021-12-31": _body(FULL, tag="a"), "2021-12-30": _body(FULL, tag="b")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31), date(2021, 12, 30)], offline,
               companion_days=1)
    #  THREE calls, not two, and the arithmetic is the point: two anchor bodies, plus ONE
    #  companion call for the 12-30 anchor (its neighbour 12-29).  The 12-31 anchor companion
    #  would have been 12-30, which is itself an anchor of this run -- already in the file, so
    #  it is skipped AND consumes that anchor companion budget rather than walking further
    #  back and spending an unbudgeted call.
    assert r["calls"] == 3
    assert any("it is an ANCHOR of this run" in ln for ln in r["rec"].log_lines)
    #  the property that matters: no anchor ever wears another date body
    pairs = set(zip(r["df"]["date_requested"], r["df"]["date_actual"]))
    assert pairs == {("2021-12-31", "2021-12-31"), ("2021-12-30", "2021-12-30")}


def test_a_short_companion_body_is_REFUSED_by_the_absolute_floor(tmp_path, offline):
    """The companion faces the payload floor like anything else, so a truncated body cannot
    enter the grid through the side door."""
    bodies = {"2021-12-31": _body(FULL, tag="a"),
              "2021-12-30": _body(SATURDAY_2024, tag="junk")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31)], offline, companion_days=1)
    assert r["calls"] == 2                        # the call was spent
    assert set(r["df"]["date_requested"]) == {"2021-12-31"}   # but nothing was written


def test_a_companion_does_NOT_disturb_the_anchor_body_guards(tmp_path, offline):
    """The floors judge ANCHOR bodies against each other.  A companion of a different size
    must not enter the relative-median calculation, or pulling one could refuse an anchor that
    was fine -- a guard corrupted by the fix meant to help it."""
    bodies = {"2021-12-31": _body(FULL, tag="a"), "2021-12-30": _body(FULL, tag="b"),
              "2020-12-31": _body(FULL, tag="c"), "2020-12-30": _body(FULL, tag="d")}
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31), date(2020, 12, 31)], offline,
               companion_days=1)
    assert r["refused"] == []
    assert r["venue"] == []


def test_the_companion_is_NOT_a_substitute_for_an_anchor_with_no_body(tmp_path, offline):
    """A companion is a fill source, never a stand-in for a missing anchor.

    THE FIRST DRAFT OF THIS TEST HAD A FALSE PREMISE and is worth recording.  It supplied a
    body only at 2021-12-30 and expected the 2021-12-31 anchor to be reported missing -- but
    the anchor own STEP-BACK loop (`max_lookback=4`) reaches 12-30 and accepts it AS the
    anchor body, `date_requested=2021-12-31, date_actual=2021-12-30`.  That is pre-existing,
    correct and deliberate behaviour that predates the companion pull, and the draft was
    asserting against it.

    The real claim is narrower: when NOTHING usable exists within the anchor own lookback, the
    companion machinery must not paper over it.  So the only body here sits outside that
    window."""
    bodies = {"2021-12-20": _body(FULL, tag="far")}   # outside max_lookback of the anchor
    r = _fetch(tmp_path, bodies, [date(2021, 12, 31)], offline, companion_days=1)
    written_anchors = set(r["df"]["date_requested"]) if len(r["df"]) else set()
    assert "2021-12-31" not in written_anchors
    assert any("no usable body" in ln for ln in r["rec"].log_lines)
