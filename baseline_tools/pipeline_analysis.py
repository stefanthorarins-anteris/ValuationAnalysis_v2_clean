"""
pipeline_analysis.py  --  POST-PICK analysis suite for the automatic Sbocker run.

WHAT THIS IS
------------
A STRICTLY ADDITIVE, GUARDED, POST-PICK block that Sbocker.main() calls AFTER the
prospective pick-log stage and BEFORE the (optional, long) delisted ingestion.  It
promotes the offline baseline_tools/ diagnostics into pipeline stages so a single
home-machine overnight run also produces the analysis, instead of hand-run side-scripts.

WHAT IT GRADES (be honest -- printed in the header too)
-------------------------------------------------------
It grades the HISTORICAL point-in-time (PIT) backtest RE-RUN AGAINST TONIGHT'S MODEL:
the same fundamentals/scoring the live pick used, reproduced as-of the historical buy
anchors (buy2021..buy2024 are the CLEAN 36-month windows), evaluated on REAL adjusted-
close prices.  It does NOT grade tonight's live picks -- there is no forward price yet
(the pick-log accrues those over months/years; grading them is the deferred piece).

SAFETY CONTRACT (non-negotiable, mirrors pick_log.run_pick_log_stage)
---------------------------------------------------------------------
  * POST-PICK: runs after the deliverables + postRank pickle + pick-log are written, so
    it CANNOT affect the shipped picks.  It reads a SHALLOW COPY of resdic and never
    writes back into any resdic key the pick path or ingestion reads.
  * ONE try/except PER STAGE (see _run_stage): a failure in one stage banners LOUDLY on
    BOTH stderr+stdout with a full traceback and is SWALLOWED -- it never re-raises and
    never skips the other stages or the run.
  * Imports live INSIDE each stage's try, so a missing/broken analysis module degrades
    loud-but-safe rather than crashing the deliverable pipeline.
  * NO git ops, NO commit/push.  Analysis text goes to stdout (run log) and files under
    output/results/ (gitignored).  Price fetches are BULK-BY-DATE ONLY and the api_key
    is never printed (masked).  The heavy ESTIMATION sub-block is OFF unless
    configdic['run_estimation'] == 1.
"""

import contextlib
import csv as _csv
import os
import re
import sys
import traceback
from datetime import datetime, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# The CLEAN 36-month windows (buy anchor -> +36mo eval anchor) the header advertises.
# Matches depth_horizon_grid's ANCHORS grid: only these two have an eval anchor in data.
#
# THIS LIST IS NO LONGER THE AUTHORITY ON WHICH ANCHORS ARE GRADED -- `dhg.CLEAN_BUY_IDS` is,
# and the two can now differ (buy2020 was promoted 2026-08-30 on the survivorship evidence in
# `dhg.ANCHOR_EXCLUSION_REASONS`).  Every place that PRINTS the graded window set derives it
# from `dhg` via `_clean_window_text()`; this literal survives for exactly one caller,
# `merge_as_of` below, which is a different question -- see the comment there.
_CLEAN_36MO_WINDOWS = [("2021-12-31", "2024-12-31"), ("2022-12-30", "2025-12-31")]

# Default locations for the survivorship (delisted) inputs the PIT reproduction needs.
# Tonight's HOME run keeps its delisted_out under HomeGDrive; a repo-local copy is also
# honored.  configdic['delisted_out'] overrides both.
_HOME = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
_DEFAULT_DELISTED_DIRS = [os.path.join(_HOME, "delisted_out"),
                          os.path.join(_REPO, "delisted_out")]

_PRICES_CSV = os.path.join(_HERE, "price_data", "real_prices.csv")
_PRICES_2025_CSV = os.path.join(_HERE, "price_data", "real_prices_2025.csv")

# Year-end anchors the PriceSource grid needs (2018..2024 in the main file; 2025 in supp).
_MAIN_PRICE_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]  # +12mo -> ...2024-12-31


def _mask_key(k):
    """Mask an api_key for logging: keep first 4 + last 2, star the middle.  NEVER print
    a raw key anywhere in this module."""
    if not k:
        return "<none>"
    k = str(k)
    if len(k) <= 8:
        return "*" * len(k)
    return k[:4] + "*" * (len(k) - 6) + k[-2:]


# --- api-key SCRUBBING (D1): the in-run fetch must never emit the key, even on an  ---
# --- HTTP/network error.  Two layers: (1) route the HTTP call through the           ---
# --- key-scrubbing delisted_ingest.safe_get_bulk_csv; (2) a boundary scrub on BOTH  ---
# --- stdout+stderr during the fetch, and str(exception) masking, as defense-in-depth.---
_APIKEY_RE = re.compile(r"apikey=[^&\s\"']+", re.IGNORECASE)


def _scrub(text, key=None):
    """Strip any apikey=... query-param AND any literal key substring from `text`."""
    s = _APIKEY_RE.sub("apikey=***", str(text))
    if key:
        s = s.replace(str(key), "***")
    return s


class _ScrubStream:
    """Write-through stream wrapper that scrubs the api_key from everything written to
    the underlying stream (boundary mask around the fetch)."""

    def __init__(self, base, key):
        self._base = base
        self._key = key

    def write(self, s):
        self._base.write(_scrub(s, self._key))

    def flush(self):
        self._base.flush()


#  A reference venue must carry at least this many rows for "absent from the new body" to be
#  a fact about the fetch rather than about a handful of names.  Declared here rather than
#  reusing `price_grid_audit.VENUE_MIN_PANEL_NAMES` because that one counts PANEL names and
#  this counts REFERENCE-GRID rows -- same number today, different quantity, and tying them
#  together would make one move when only the other was meant to.
VENUE_MIN_REFERENCE_ROWS = 10

#  A venue surviving at less than this share of its reference count is treated as absent.
#  The same 0.5 as `fetch_prices.SHORT_BODY_MEDIAN_FRACTION`, for the same reason: less than
#  half of a known population is not that population.  It IS a threshold, unlike the
#  categorical zeros `price_grid_audit` restricts itself to -- said out loud because this
#  module does not otherwise carry one.
VENUE_MIN_SURVIVING_SHARE = 0.5


def _venue_of(symbol):
    """`092730.KQ` -> `.KQ`, `META` -> `(none)`.  The same venue key price_grid_audit uses."""
    t = str(symbol)
    return "." + t.rsplit(".", 1)[1] if "." in t else "(none)"


def _venue_reference(candidate_paths, log):
    """{date_requested: {venue: n_rows}} from the FIRST readable previous grid on disk.

    THE COMPARISON IS ANCHOR-MATCHED, AND THAT IS THE WHOLE DESIGN.  A within-run test --
    "every venue seen at another anchor of this run must appear at this one" -- was the
    obvious instrument and it is wrong twice over: it FALSE-POSITIVES on genuine year-end
    venue holidays (.DE/.ST/.IC are legitimately zero at 2018-12-31), and it is SILENT on
    the seven venues the grid never had at any anchor, which are the 1,421 names the whole
    exercise is about.  Matching a PREVIOUS grid anchor by anchor fixes the first problem for
    free: the reference encodes each venue own trading calendar, so a venue that was
    legitimately shut on that date carries no expectation.

    It does NOT fix the second, and cannot -- see the caller for the full blindness list.
    """
    import pandas as pd
    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=["date_requested", "symbol"])
        except Exception as e:      # a truncated or garbled reference is no reference
            log(f"[price-fetch] venue reference {os.path.basename(str(path))} unreadable "
                f"({type(e).__name__}) -- skipped")
            continue
        ref = {}
        venues = df["symbol"].map(_venue_of)
        for (anchor, venue), n in df.groupby([df["date_requested"], venues]).size().items():
            ref.setdefault(str(anchor), {})[venue] = int(n)
        log(f"[price-fetch] venue reference: {os.path.basename(str(path))} "
            f"({len(ref)} anchor(s), {venues.nunique()} venue(s))")
        return ref, str(path)
    log("[price-fetch] NO previous grid on disk -- the per-venue completeness test is "
        "INERT this run (it has no external reference to compare against)")
    return {}, None


def _venue_shortfall(kept_symbols, ref_for_anchor):
    """Venues the reference had at this anchor and this body effectively does not.

    Returns [(venue, n_now, n_ref)], biggest loss first.  `kept_symbols` is what survived the
    local allow-list, so the comparison is like-for-like with the reference, which was
    written through the same filter.
    """
    if not ref_for_anchor:
        return []
    now = {}
    for sym in kept_symbols:
        v = _venue_of(sym)
        now[v] = now.get(v, 0) + 1
    out = []
    for venue, n_ref in ref_for_anchor.items():
        if n_ref < VENUE_MIN_REFERENCE_ROWS:
            continue
        n_now = now.get(venue, 0)
        if n_now < VENUE_MIN_SURVIVING_SHARE * n_ref:
            out.append((venue, n_now, n_ref))
    return sorted(out, key=lambda t: t[2] - t[1], reverse=True)


def _fetch_bulk_scrubbed(baseurl, api_key, anchors, symbols_filter, out_path, log,
                         max_lookback=4, reference_paths=None, companion_days=1):
    """BULK-BY-DATE fetch that CANNOT leak the api_key.  One call per anchor date (whole
    universe per call; NEVER per-symbol), stepping back up to `max_lookback` days on an
    unusable response.  The HTTP call goes through delisted_ingest.safe_get_bulk_csv, which
    strips apikey from the URL and from any exception or warning text.  Reuses fetch_prices'
    KEY-FREE pure helpers -- row parsing AND the payload-acceptance rule, which lives there
    in ONE definition.  Writes the same schema fetch_prices produced.
    Returns (calls, rows_written, refused, venue_findings).

    THIS IS THE PATH THE PIPELINE TAKES, and until 2026-08-22 it was the one WITHOUT any
    completeness test.  `fetch_prices.run_bulk` grew a payload floor and a weekend guard
    while this function -- the only fetch `run_price_fetch_stage` actually calls -- still
    read `if rows:`.  The guard was protecting code production does not execute.  Three
    acceptance tests now apply here, all sourced from `fetch_prices`:

      1. WEEKEND candidates are never requested (no call spent).
      2. ABSOLUTE payload floor, in-line, so the existing step-back moves past a short body.
      3. PER-VENUE completeness against the PREVIOUS GRID at the SAME anchor, in-line, so a
         venue-clustered truncation is stepped past too.
    plus the DEFERRED relative-median floor at write time, which costs no call.

    WHAT (3) CANNOT SEE -- state this beside any green fetch:
      * NOTHING AT ALL when there is no previous grid, which is the pipeline own common
        case: `run_price_fetch_stage` fetches only when the file is ABSENT, so a first fetch
        on a fresh machine has no reference and this test is inert.
      * A venue the REFERENCE also lacks.  The run machine grid holds only
        ['(none)', '.DE', '.IC', '.L', '.ST', '.TO'], so .PA/.KS/.OL/.KQ/.BR/.AS/.LS --
        1,421 names -- carry no expectation and their absence stays silent.  This test can
        DEFEND a venue set; it cannot BOOTSTRAP one.  Only widening the fetch own symbol
        universe does that.
      * An anchor the reference does not cover (a new year), or a venue newly listed since.
      * A venue legitimately delisted since the reference -- that reads as a shortfall and
        would reject every candidate for the anchor, leaving it with NO body.  That is the
        one way this test can cost data, so it has an explicit off-switch:
        `configdic['price_grid_venue_check'] = 0`.  The finding names the venue first, so
        repointing `configdic['price_grid_reference']` is usually the better move.
      * Wrong PRICES.  Every test here is about presence and count.

    THE COMPANION PULL (`companion_days`, default 1) -- and why presence AT THE ANCHOR is a
    different question from the venue being covered.  MEASURED on the dev grid, which is
    ALREADY UNFILTERED (443,893 rows / 90,506 symbols), so the save-side allow-list cannot be
    the cause: `.KS` reads 1, 1, 1350, 1, 1369, 0, 0 across the seven main anchors; `.KQ`
    reads 0, 0, 307, 0, 350, 0, 0; `.OL` reads 1, 1, 243, 1, 290, 275, 0.  The two anchors
    where they DO appear are exactly the two whose body came from a PRE-HOLIDAY `date_actual`
    (2020-12-28 and 2022-12-27).  Those venues shut before the calendar year-end, so the
    anchor body genuinely does not contain them and no amount of widening what we SAVE can
    conjure them.  That is a HOLIDAY problem, not a filter problem, and the two were
    conflated once already.

    So each anchor also pulls the nearest `companion_days` WEEKDAYS strictly before the
    accepted body, and each companion is written under ITS OWN `date_requested` -- never the
    anchor.  That keeps it OUT of PriceSource anchor layer (which selects on
    `date_requested`) and available to `_fill_from_neighbour_dates`, which unions it in per
    symbol, per anchor, add-only, and REPORTS the lag it introduces.  The 2025 supplementary
    file already on disk stores its 2025-12-30 body in exactly this shape, so this is the
    established convention rather than a new one.

    BOUNDED BY THE READER FILL WINDOW, deliberately: a companion older than
    `returns_core.DEFAULT_FILL_WINDOW_DAYS` before the anchor is a date the reader can never
    consume, so a call there buys nothing.  Weekends cost no call, a date already fetched is
    never re-requested, and a companion that would collide with another anchor of this run is
    skipped so it cannot inject a body into a different anchor layer.

    A COMPANION IS NOT AN ANCHOR BODY.  It faces the absolute payload floor and nothing else.
    The venue-shortfall test would fire on it BY CONSTRUCTION -- a different day has different
    venues open, which is the entire reason it is being pulled -- and it is kept out of the
    relative-median floor so it cannot move the median the anchor bodies are judged against.
    It is never a substitute for an anchor that got no body at all.

    WHAT ONE COMPANION DAY BUYS, stated so nobody over-reads it: it recovers venues whose last
    trading day is the weekday before the anchor, which covers 2021-12-31 and 2024-12-31 --
    BOTH legs of the buy2021 clean 36-month window.  A venue that shut EARLIER (the 2020 and
    2022 cases above, three days back) needs `companion_days` raised toward the fill window,
    at roughly +8 calls per extra day.

    ONE PROPERTY WORTH KNOWING, because it makes a bad reference safe: the test only fires
    when the NEW body has FEWER rows for a venue than the reference.  A reference that is
    itself truncated therefore makes this test WEAKER, never spuriously stricter -- so
    pointing it at the very file about to be overwritten (which may be the corrupted one)
    cannot reject a good body.  It degrades toward silence, which is the correct direction
    for a guard that can drop an anchor.
    """
    import delisted_ingest as di
    import fetch_prices as fp

    import returns_core as _rc

    ref, ref_path = _venue_reference(reference_paths or [], log)
    ref_label = os.path.basename(str(ref_path)) if ref_path else "no-reference"
    calls = written = 0
    accepted = []
    companions = []
    venue_findings = []
    #  Every date this run has already spent a call on, plus every ANCHOR of this run.  A
    #  companion may be neither: re-requesting a date costs money for nothing, and a companion
    #  written under a date that is some other anchor would inject a body into that anchor
    #  layer -- the "two payloads for one anchor" failure this file already carries a scar
    #  from.
    seen_dates = set()
    anchor_dates = {a.strftime("%Y-%m-%d") for a in anchors}
    fill_cap = int(_rc.DEFAULT_FILL_WINDOW_DAYS)
    companion_days = max(0, int(companion_days))

    for anchor in anchors:
        a_str = anchor.strftime("%Y-%m-%d")
        got = False
        for back in range(max_lookback + 1):
            d = anchor - timedelta(days=back)
            ds = d.strftime("%Y-%m-%d")
            if fp.is_weekend(d):
                #  NO CALL SPENT.  The endpoint answers a weekend with a small NON-empty
                #  body (2024-12-28, a Saturday: 3,589 rows, 93.8% crypto pairs) and the old
                #  `if rows:` accepted it as the 2024-12-31 anchor.
                log(f"[price-fetch]   skip {ds}: {d.strftime('%A')} -- not requested")
                continue
            url = (f"{baseurl}v4/batch-request-end-of-day-prices"
                   f"?date={ds}&apikey={api_key}")
            calls += 1
            seen_dates.add(ds)
            log(f"[price-fetch] bulk call {calls}: date={ds} "
                f"(anchor {anchor.isoformat()})")
            rows = di.safe_get_bulk_csv(url)  # key-scrubbed on ANY error/warning
            n_payload = len(rows or [])
            if not fp.body_is_acceptable(n_payload):
                log(f"[price-fetch]   REJECTED {ds}: {n_payload} rows is below the absolute "
                    f"floor {fp.MIN_PAYLOAD_ROWS} -- stepping back")
                continue
            kept = []
            for row in rows:
                sym, adj = fp._extract(row)
                if not sym or adj in (None, "", "null"):
                    continue
                if symbols_filter and sym not in symbols_filter:
                    continue
                kept.append((sym, adj))
            short = _venue_shortfall([sym for sym, _a in kept], ref.get(a_str))
            if short:
                detail = "; ".join(f"{v}: {n} now vs {r} in {ref_label}"
                                   for v, n, r in short)
                log(f"[price-fetch]   REJECTED {ds}: venue shortfall at anchor {a_str} -- "
                    f"{detail} -- stepping back")
                venue_findings.append({"anchor": a_str, "date": ds, "shortfall": short,
                                       "reference": ref_label})
                continue
            accepted.append({"anchor": a_str, "date": ds, "n_payload": n_payload,
                             "rows": kept})
            got = True
            accepted_date = d
            log(f"[price-fetch]   OK: {n_payload} rows for {ds} ({len(kept)} kept)")
            break
        if not got:
            log(f"[price-fetch]   WARNING: no usable body for anchor "
                f"{anchor.isoformat()} within {max_lookback} lookback days")
            continue

        #  COMPANION DAYS -- see the docstring.  Venues that shut before the calendar year-end
        #  are absent from the anchor body itself, so the only way to price them AT that
        #  anchor is to also hold the preceding trading day and let the reader union it in.
        probe = accepted_date
        taken = 0
        while taken < companion_days:
            probe = probe - timedelta(days=1)
            if (anchor - probe).days > fill_cap:
                #  Past the reader fill window: a call here buys a date nothing can consume.
                log(f"[price-fetch]   companion stop for {a_str}: reached "
                    f"{probe.isoformat()}, beyond the {fill_cap}-day fill window")
                break
            cs = probe.strftime("%Y-%m-%d")
            if fp.is_weekend(probe):
                dayname = probe.strftime("%A")
                log(f"[price-fetch]   companion skip {cs}: {dayname} -- not requested")
                continue
            #  ALREADY SERVED -> CONSUME THE BUDGET.  A weekend (above) is not a trading
            #  day at all, so it is skipped for free.  These two are different: the date IS
            #  in the file already -- as another anchor body, or as a date this run has
            #  fetched -- and the fill layer reads every row by `date_actual` regardless of
            #  which anchor it was requested for.  The companion purpose is therefore
            #  already met, so walking further back would spend a call the +1-day budget
            #  never promised.  Consuming `taken` caps the run at companion_days calls per
            #  anchor, which is what the call-count estimate in the log line assumes.
            if cs in seen_dates:
                log(f"[price-fetch]   companion skip {cs}: already fetched this run "
                    "-- the fill layer can already reach it")
                taken += 1
                continue
            if cs in anchor_dates:
                log(f"[price-fetch]   companion skip {cs}: it is an ANCHOR of this run "
                    "-- already in the file under its own date")
                taken += 1
                continue
            url = (f"{baseurl}v4/batch-request-end-of-day-prices"
                   f"?date={cs}&apikey={api_key}")
            calls += 1
            seen_dates.add(cs)
            log(f"[price-fetch] bulk call {calls}: date={cs} "
                f"(COMPANION for anchor {anchor.isoformat()})")
            crows = di.safe_get_bulk_csv(url)
            c_payload = len(crows or [])
            #  ABSOLUTE FLOOR ONLY, for the reasons in the docstring.
            if not fp.body_is_acceptable(c_payload):
                log(f"[price-fetch]   companion {cs} REJECTED: {c_payload} rows is below the "
                    f"absolute floor {fp.MIN_PAYLOAD_ROWS} -- not written")
                taken += 1
                continue
            ckept = []
            for row in crows:
                sym, adj = fp._extract(row)
                if not sym or adj in (None, "", "null"):
                    continue
                if symbols_filter and sym not in symbols_filter:
                    continue
                ckept.append((sym, adj))
            companions.append({"anchor": a_str, "date": cs, "n_payload": c_payload,
                               "rows": ckept})
            log(f"[price-fetch]   companion OK: {c_payload} rows for {cs} "
                f"({len(ckept)} kept) -- fill source for anchor {a_str}")
            taken += 1

    #  DEFERRED relative floor.  It costs no call, so it runs at write time rather than
    #  re-entering the step-back loop and spending calls nobody planned.
    refused_idx = fp.refusals_against_median([a["n_payload"] for a in accepted])
    refused = []

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fout:
        w = _csv.writer(fout)
        w.writerow(["date_requested", "date_actual", "symbol", "adjClose"])
        for i, a in enumerate(accepted):
            if i in refused_idx:
                med = refused_idx[i]
                log(f"[price-fetch]   REFUSED at write time: anchor {a['anchor']} body "
                    f"{a['date']} has {a['n_payload']} rows vs a median of {med:.0f} for "
                    f"the run other bodies -- NOT written")
                refused.append({"anchor": a["anchor"], "date": a["date"],
                                "n_payload": a["n_payload"], "median_of_others": med})
                continue
            for sym, adj in a["rows"]:
                w.writerow([a["anchor"], a["date"], sym, adj])
                written += 1
        #  Companions LAST, and under their OWN date_requested.  File order matters only for
        #  the anchor layer keep-first rule, and these are invisible to it by construction.
        for c in companions:
            for sym, adj in c["rows"]:
                w.writerow([c["date"], c["date"], sym, adj])
                written += 1
    return calls, written, refused, venue_findings


def _banner(title, cause=None):
    lines = ["\n" + "!" * 78,
             f"!!! ANALYSIS STAGE FAILED -- {title} !!!",
             "!!! The pick-path deliverables above are UNAFFECTED (this stage is    !!!",
             "!!! post-pick and isolated); this readout is missing this run.        !!!"]
    if cause is not None:
        lines.append(f"!!! Cause: {type(cause).__name__}: {cause}")
    lines.append("!" * 78 + "\n")
    return "\n".join(lines)


def _run_stage(name, fn, *args, **kwargs):
    """Run ONE analysis stage under its OWN guard (mirrors pick_log.run_pick_log_stage).

    A failure prints an unmistakable !!! banner + traceback on BOTH stderr and stdout
    and is SWALLOWED (returns None) so it can never skip the sibling stages or crash the
    run.  The `import` of the stage's module belongs INSIDE `fn`, so an import failure is
    caught here too."""
    print(f"\n[analysis] >>> stage START: {name}", flush=True)
    t0 = datetime.now()
    try:
        out = fn(*args, **kwargs)
        dt = (datetime.now() - t0).total_seconds()
        print(f"[analysis] <<< stage OK: {name}  ({dt:.1f}s)", flush=True)
        return out
    except Exception as e:  # never re-raise
        dt = (datetime.now() - t0).total_seconds()
        b = _banner(f"{name}  (after {dt:.1f}s)", cause=e)
        print(b, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(b, flush=True)
        traceback.print_exc(file=sys.stdout)
        return None


def _reference_paths(configdic, out_path):
    """Candidate previous grids for the per-venue completeness test, best first.

    THE PIPELINE OFTEN HAS NONE, and that is the honest headline: this stage fetches only
    when `out_path` is ABSENT, so the overwrite case -- the one that actually corrupted
    `real_prices.csv` by hand -- belongs to the standalone script, not here.  The candidates
    are therefore (1) whatever the operator names, (2) `out_path` itself for a caller that
    does overwrite, (3) a sibling `.bak`, which is how a previous grid survives on this
    machine today.  No candidate resolving means the test is inert and says so.

    `configdic['price_grid_venue_check'] = 0` returns NO candidates, which disables the test.
    It exists because this is the one guard here that can leave an anchor with no body at
    all (a venue legitimately delisted since the reference reads as a shortfall), and a
    guard with that consequence should be switchable without editing code.
    """
    if str(configdic.get("price_grid_venue_check", 1)) in ("0", "False", "false"):
        return []
    named = configdic.get("price_grid_reference")
    return [named, out_path, out_path + ".bak"]


def _report_fetch_refusals(label, refused, venue_findings, log):
    """Say what was thrown away, unmissably.  A refusal that only shows up as a smaller file
    is the same silent-partial-grid failure the floor exists to prevent."""
    if not refused and not venue_findings:
        return
    bang = "!" * 78
    lines = ["", bang, f"!!! {label} PRICE FETCH REFUSED CONTENT -- the grid is INCOMPLETE"]
    for r in refused:
        lines.append(f"!!!   anchor {r['anchor']} body {r['date']}: {r['n_payload']} rows "
                     f"vs median-of-others {r['median_of_others']:.0f} -- NOT WRITTEN")
    for v in venue_findings:
        detail = "; ".join(f"{ven}: {n} vs {ref}" for ven, n, ref in v["shortfall"])
        lines.append(f"!!!   anchor {v['anchor']} body {v['date']}: venue shortfall vs "
                     f"{v['reference']} -- {detail} -- stepped back")
    lines += ["!!! Re-fetch those anchors deliberately.  Do NOT paste a short body into the",
              "!!! canonical grid by hand -- that is how the file acquired two payloads for",
              "!!! one anchor and lost its provenance.", bang, ""]
    text = chr(10).join(lines)
    print(text, file=sys.stderr, flush=True)
    print(text, flush=True)
    log(f"[price-fetch] {label}: {len(refused)} refused body(ies), "
        f"{len(venue_findings)} venue shortfall(s)")


# --------------------------------------------------------------------------- #
#  Stage 1: guarded BULK-BY-DATE price fetch (no-op/top-up when present)       #
# --------------------------------------------------------------------------- #
def run_price_fetch_stage(resdic, configdic, log):
    """Ensure the grading price grid (real_prices.csv + real_prices_2025.csv) exists.

    NO-OP/top-up when the files are already present (the common case -- prices are
    checked into neither git nor moved out-of-band, but exist on the run machine).  When
    ABSENT, fetch them BULK-BY-DATE ONLY (one call per year-end anchor, whole universe
    per call) via `_fetch_bulk_scrubbed` in THIS module.  The api_key is read from
    configdic (fallback fmpAPIkey.txt) and NEVER printed (masked).  Returns
    dict(main=..., supp=...) of resolved paths (or None).

    THE WHOLE BODY IS SAVED -- no symbol allow-list.  See the long note at the fetch
    itself for why a filter built from tonight's universe silently froze this grid at a
    survivor- and venue-thinned shape.  `resdic` is consequently UNUSED here and kept only
    so every analysis stage shares one signature; the population tonight's universe defines
    is now used by `_audit_price_grid_stage`, which reports rather than discards.

    THE DOCSTRING USED TO SAY "via fetch_prices.run_bulk" AND THAT WAS FALSE.  `run_bulk`
    is called from nowhere in this module; `_fetch_bulk_scrubbed` is the fetch, and it
    exists because it cannot leak the api_key.  The claim mattered: a payload floor and a
    weekend guard were added to `run_bulk` and, on the strength of that sentence, believed
    to be protecting this stage.  They were not.  Both now run here, from the same
    definitions in `fetch_prices`, and `test_fetch_prices` pins the wiring so the sentence
    cannot go stale again.
    """
    import fetch_prices as fp  # KEY-FREE pure helpers only (build_anchor_dates, ...)

    #  PRESENCE DECIDES, FRESHNESS SPEAKS -- and the two used to be different stages.
    #
    #  THE DEFECT (Q-7).  This decision was `not os.path.exists(...)` alone, so a grid written
    #  months ago satisfied it for ever and the freshness signal -- which existed, two stages
    #  down, in `_audit_price_grid_stage` -- had no way to reach the decision it was about.
    #  The audit could shout STALE all it liked; the refetch decision had already been taken
    #  and had never asked.
    #
    #  WHAT IT IS NOW: a THREE-way decision, and only one branch spends money, never by
    #  default.
    #    absent               -> fetch, exactly as before.
    #    present + fresh      -> no fetch, exactly as before.
    #    present + STALE      -> REFUSE TO REFETCH, loudly and by name, and say which key
    #                            authorises it.  The refusal is a refusal to SPEND, not a
    #                            refusal to run: five working analysis stages still get their
    #                            grid, because withholding output would make the spend
    #                            decision for the CEO by starving him of the thing he would
    #                            decide from.  That was the 2026-08-22 ruling and it stands.
    #  `configdic['price_grid_refetch_when_stale'] = 1` is the required override.  There is
    #  deliberately NO automatic-refetch path and no age threshold that silently trips one:
    #  a fetch is ~8 paid calls on a key the house does not own the budget for.
    need_main = not os.path.exists(_PRICES_CSV)
    need_supp = not os.path.exists(_PRICES_2025_CSV)

    stale_findings = None if need_main else _grid_stale_findings(resdic, log)
    if stale_findings:
        if configdic.get("price_grid_refetch_when_stale"):
            need_main = True
            log("[price-fetch] !! REFETCHING A PRESENT GRID because "
                "configdic['price_grid_refetch_when_stale'] is set.")
            log("[price-fetch] !! THIS SPENDS PAID API CALLS.  Reason(s) the grid is stale: "
                + "; ".join(stale_findings[:3]))
        else:
            log("[price-fetch] " + "!" * 62)
            log("[price-fetch] !! THE GRID IS PRESENT AND STALE, AND I WILL NOT REFETCH IT.")
            for finding in stale_findings[:5]:
                log(f"[price-fetch] !!   - {finding}")
            if len(stale_findings) > 5:
                log(f"[price-fetch] !!   ... and {len(stale_findings) - 5} more "
                    "(the audit stage below prints all of them)")
            log("[price-fetch] !! Every backtest number in this run rides THIS grid.")
            log("[price-fetch] !! To refetch, set configdic['price_grid_refetch_when_stale']"
                " = 1 and re-run.")
            log("[price-fetch] !! That costs paid API calls, which is why it is not the "
                "default and why")
            log("[price-fetch] !! nothing here decides it for you.")
            log("[price-fetch] " + "!" * 62)

    if not need_main and not need_supp:
        log(f"[price-fetch] both price files present -- NO fetch "
            f"({os.path.basename(_PRICES_CSV)}, {os.path.basename(_PRICES_2025_CSV)})"
            + ("  [STALE -- refetch REFUSED, see above]" if stale_findings else ""))
        return {"main": _PRICES_CSV, "supp": _PRICES_2025_CSV}

    # Resolve + MASK the key.  Missing key => cannot fetch => banner via raise (guarded).
    api_key = configdic.get("api_key")
    if not api_key:
        key_path = os.path.join(_REPO, "fmpAPIkey.txt")
        if os.path.exists(key_path):
            api_key = fp.read_api_key(key_path)
    if not api_key:
        raise RuntimeError("price fetch needed but no api_key available "
                           "(configdic['api_key'] and fmpAPIkey.txt both empty)")
    log(f"[price-fetch] api_key resolved (masked): {_mask_key(api_key)}")
    baseurl = configdic.get("baseurl") or "https://financialmodelingprep.com/api/"

    #  WHAT WE SAVE: THE WHOLE BODY.  No local allow-list, deliberately.
    #
    #  THE RULE THAT WAS HERE WAS THE DEFECT, not a tuning choice.  It read "keep the
    #  written file small" and built a symbol allow-list out of tonight's `Tickers_df` +
    #  `cdx_df`.  The bulk endpoint returns every symbol that traded on the date and the
    #  call is already paid for in full; the allow-list then threw most of that away
    #  BEFORE writing.  What it kept is TONIGHT'S SCORED SURVIVORS -- and the file it
    #  writes is read for years afterwards by the grading stages, which need a strictly
    #  LARGER set than tonight's universe can ever contain:
    #
    #    * Names that DIED between an anchor and tonight (Atrion, Kirkland Lake, Triton,
    #      QIWI, PBF Logistics, SciPlay, the preferred lines...).  A survivorship-clean
    #      backtest is precisely the thing that needs the names a live universe cannot
    #      hold, so the filter deleted the evidence whose absence it was meant to expose.
    #    * Venues tonight's universe happens not to reach.  On the run machine's grid its
    #      own `[price-audit]` block (2026-08-27) reports SEVEN venues priced at ZERO of the
    #      8 anchors -- `.PA` (572 panel names), `.KS` (327), `.OL` (224), `.KQ` (160),
    #      `.BR` (105), `.AS` (104), `.LS` (33) -- plus `.DE` and `.ST` empty at 2018-12-31,
    #      with the backtest grading 18 of 40 picks.  That grid was written on 2026-07-17
    #      against the older NA1_EU1 universe and, because `need_main` above is a presence
    #      check by design, has been frozen at that shape ever since.
    #
    #      BE EXACT ABOUT HOW MANY OF THE SEVEN THIS CHANGE ACTUALLY RECOVERS: FOUR.  An
    #      earlier version of this note claimed all seven and that was wrong.  Measured on
    #      the DEV grid -- which is already unfiltered, so the allow-list cannot be the cause
    #      -- `.PA` and `.L` are present at EVERY anchor, so `.PA/.BR/.AS/.LS` are genuinely
    #      recovered here.  But `.KS` reads 1, 1, 1350, 1, 1369, 0, 0 across the seven main
    #      anchors, `.KQ` reads 0, 0, 307, 0, 350, 0, 0 and `.OL` reads 1, 1, 243, 1, 290,
    #      275, 0 -- and the anchors where they appear are exactly the two whose body came
    #      from a pre-holiday `date_actual`.  Those three venues (711 of the 1,421 names) are
    #      NOT recovered by removing the filter: they are absent from the anchor body itself,
    #      including at 2021-12-31 and 2024-12-31, BOTH legs of the buy2021 clean window.
    #      That is the HOLIDAY problem and it is fixed on the FETCH side by the companion-day
    #      pull in `_fetch_bulk_scrubbed`, not here.
    #    * The benchmark ETF, which needed its own hand-written exception here to survive
    #      the filter at all.  A second special case for one rule is the rule being wrong,
    #      so the exception is gone with the filter rather than joined by a third.
    #
    #  The structural reason no allow-list belongs here: THE READ SET IS NOT KNOWABLE AT
    #  WRITE TIME.  Any list computed from one night's state is a guess about every future
    #  reader, and it fails toward silently DROPPING data.  That includes the tempting
    #  "universe + delisted registry + benchmark" union -- the registry is an optional
    #  artifact (`delisted_out/`), so on a machine that lacks it the union quietly
    #  re-narrows to exactly the filter being removed here.
    #
    #  WHAT IT COSTS: 13.6 MB.  Unfiltered the main grid is ~444k rows / ~90.5k symbols /
    #  ~15.5 MB, against the filtered 54k rows / 9.6k symbols / 1.8 MB.  It lands on a
    #  gitignored path (`baseline_tools/price_data/`) that moves between machines by Google
    #  Drive, never git, so no size limit is in play; `PriceSource` construction on the
    #  unfiltered file is 0.77s (measured -- see returns_core._fill_from_neighbour_dates),
    #  and every offline tool in baseline_tools/ has been running against the unfiltered
    #  grid on the dev machine all along.  Disk is not a reason to lose a venue.
    #
    #  WHAT CAN STILL BE DROPPED, so nobody reads this as "the grid is now complete":
    #    * rows with no symbol or a null `adjClose` (`fp._extract`) -- not a price;
    #    * a whole anchor body refused by the absolute floor, the relative-median floor or
    #      the venue-shortfall test -- announced by `_report_fetch_refusals`, unmissably;
    #    * whatever the VENDOR itself omits.  FMP's historical bodies are survivor-thinned
    #      (fmp-specialist, 2026-08-20).  This stops US thinning them further; it cannot
    #      un-thin the vendor.
    #
    #  `_fetch_bulk_scrubbed` KEEPS its `symbols_filter` parameter, and the honest reason is
    #  NOT the one an earlier version of this note gave.  That note said "`fetch_prices.py
    #  --symbols` is a deliberate operator choice on the standalone tool", which is wrong
    #  twice: the flag is `--symbols-file`, and it routes through `fetch_prices.run_bulk`,
    #  which has its OWN filter at its own line -- it never reaches this function.  So with
    #  the stage passing None, `_fetch_bulk_scrubbed`s `symbols_filter` is DEAD IN
    #  PRODUCTION; its only live callers are tests.
    #
    #  It is kept anyway, for one reason: `test_pipeline_fetch_guards` needs it to construct
    #  a filtered body for `test_the_reference_is_compared_AFTER_the_local_symbol_filter`,
    #  which pins a real ordering property of the venue-shortfall test (the comparison runs
    #  on the KEPT rows, not the raw payload).  Dropping the parameter would delete that
    #  guard as a side effect.  A reviewer who would rather see the dead parameter go is not
    #  wrong -- it is a loaded gun pointed at exactly the defect above -- and the removal is
    #  a clean follow-up costing two test edits.  What lives HERE is the POLICY: the pipeline
    #  stage never narrows its own grid, pinned by `test_benchmark_in_price_source`.
    symbols_filter = None
    #  COMPANION DAYS: how many weekdays before each anchor to ALSO pull, so venues that shut
    #  before the calendar year-end can be filled in by the reader.  1 (default) costs +8
    #  calls and covers both legs of the buy2021 clean window; see `_fetch_bulk_scrubbed`.
    #  0 restores the anchor-only fetch.  Capped there by the reader fill window.
    companion_days = int(configdic.get("price_companion_days", 1))

    # D1 boundary mask: run the ENTIRE fetch with BOTH stdout+stderr scrubbed, and mask
    # any exception message, so the key cannot surface even on a network/HTTP error path
    # (belt-and-suspenders on top of safe_get_bulk_csv's own scrubbing).
    with contextlib.redirect_stdout(_ScrubStream(sys.stdout, api_key)), \
         contextlib.redirect_stderr(_ScrubStream(sys.stderr, api_key)):
        try:
            if need_main:
                anchors = fp.build_anchor_dates(_MAIN_PRICE_YEARS, hold_months=12)
                log(f"[price-fetch] MAIN grid absent -> bulk-by-date fetch, "
                    f"{len(anchors)} anchor dates + {companion_days} companion day(s) each "
                    f"(~{len(anchors) * (1 + companion_days)} calls): "
                    f"{[a.isoformat() for a in anchors]}")
                calls, written, refused, vf = _fetch_bulk_scrubbed(
                    baseurl, api_key, anchors, symbols_filter, _PRICES_CSV, log,
                    reference_paths=_reference_paths(configdic, _PRICES_CSV),
                    companion_days=companion_days)
                log(f"[price-fetch] MAIN done: {calls} calls, {written} rows -> "
                    f"{os.path.basename(_PRICES_CSV)}")
                _report_fetch_refusals("MAIN", refused, vf, log)
            if need_supp:
                d2025 = fp.nearest_weekday_on_or_before(datetime(2025, 12, 31).date())
                log(f"[price-fetch] SUPP 2025 anchor absent -> bulk-by-date fetch 1 "
                    f"anchor + {companion_days} companion day(s) "
                    f"(~{1 + companion_days} calls) ({d2025.isoformat()})")
                #  ONE anchor, so the relative-median floor is vacuous by construction and
                #  only the absolute floor and the venue test can fire here.  That is exactly
                #  the hole the absolute backstop was put in for.
                calls, written, refused, vf = _fetch_bulk_scrubbed(
                    baseurl, api_key, [d2025], symbols_filter, _PRICES_2025_CSV, log,
                    reference_paths=_reference_paths(configdic, _PRICES_2025_CSV),
                    companion_days=companion_days)
                log(f"[price-fetch] SUPP done: {calls} calls, {written} rows -> "
                    f"{os.path.basename(_PRICES_2025_CSV)}")
                _report_fetch_refusals("SUPP", refused, vf, log)
        except Exception as e:
            # Re-raise with a SCRUBBED message so the guard banner (printed to the REAL
            # streams outside this context) can never carry the key.
            raise RuntimeError(_scrub(f"price fetch failed: {e}", api_key)) from None
    return {"main": _PRICES_CSV if os.path.exists(_PRICES_CSV) else None,
            "supp": _PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None}


def _clean_window_text(horizon_m=36):
    """The graded buy->eval windows, NAMED FROM `dhg.CLEAN_BUY_IDS` rather than restated.

    Two headers printed "buy2021->2024, buy2022->2025" as a literal.  That sentence is what a
    reader takes the run to have graded, so it has to come from the set that decides it --
    otherwise promoting or excluding an anchor silently makes the header a false claim about
    the numbers underneath it, which is the same defect class as the basis stamp.

    GUARDED, because one of its two callers is the SUITE HEADER, which prints BEFORE the
    first `_run_stage` and is therefore outside the swallow that keeps a stage failure from
    costing the run.  A `dhg` import failure there would have lost the entire analysis suite
    to a cosmetic header -- the same shape as the bare `benchmark_return` in a loop, one
    level up.  A header that cannot name the windows degrades to saying so.
    """
    try:
        import depth_horizon_grid as dhg
    except Exception as e:
        return ("window list unavailable: %s: %s -- see the grid report's own flag table"
                % (type(e).__name__, e))
    out = []
    for wid, buy in dhg.BUY_ANCHORS:
        if wid not in dhg.CLEAN_BUY_IDS:
            continue
        idx = dhg.ANCHOR_IDX[buy] + horizon_m // 12
        if idx >= len(dhg.ANCHORS):
            continue
        out.append(f"{wid}->{dhg.ANCHORS[idx][:4]}")
    return ", ".join(out) if out else "(no clean window has an eval anchor in data)"


def _clean_window_text_guarded(horizon_m=36):
    """`_clean_window_text` with the LAST failure mode closed: the body above can still raise
    after the import succeeds (a missing `ANCHOR_IDX` key, a malformed anchor).  Belt and
    braces, because the caller is unguarded by construction and a header is never worth a
    suite."""
    try:
        return _clean_window_text(horizon_m)
    except Exception as e:
        return "window list unavailable (%s: %s)" % (type(e).__name__, e)


def _panel_symbols(resdic):
    """Tonight's scored names, for the price-grid audit.  Same two frames the fetch allow-list
    reads, so the audit measures the grid against exactly the population that will be graded."""
    syms = set()
    for key, col in (("Tickers_df", "symbol"), ("cdx_df", "source")):
        try:
            syms |= set(resdic[key][col].dropna().astype(str))
        except Exception:
            continue
    return syms


def _grid_stale_findings(resdic, log):
    """The audit's STRUCTURAL findings, read at FETCH time so the refetch decision can see
    them.  Returns a list of strings, or None when the question could not be asked.

    ONE DEFINITION, borrowed not copied: this calls `price_grid_audit.audit_price_grid`, the
    same function `_audit_price_grid_stage` reports from, so the fetch decision and the
    printed audit can never disagree about whether the grid is stale.  A second staleness
    rule living here is precisely how the presence-check and the audit came to be two
    unrelated opinions in the first place.

    FULLY GUARDED, AND THE GUARD LEANS TOWARDS SPENDING NOTHING.  Any failure returns None,
    which means "no findings" and therefore NO refetch -- an audit that cannot run must never
    be the thing that authorises a paid call.
    """
    try:
        import price_grid_audit as pga
        rep = pga.audit_price_grid(
            _PRICES_CSV, _panel_symbols(resdic),
            supp_csv=_PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None)
        return list(rep.get("findings") or []) or None
    except Exception as e:
        log(f"[price-fetch] freshness read did not run ({type(e).__name__}: {e}); "
            "falling back to the presence check alone -- NO fetch will be triggered by it.")
        return None


def _audit_price_grid_stage(resdic, configdic, log):
    """STALENESS DETECTION for the saved grading grid -- report only, NEVER a fetch.

    THE DEFECT THIS CLOSES.  `run_price_fetch_stage` decides whether to fetch with
    `not os.path.exists(...)`, a pure presence check, so a `real_prices.csv` written months ago
    satisfied it for ever.  Measured (price_grid_audit against the run machine's grid and the
    08-22 CUR6K panel): the grid dates from 2026-07-17, carries 10,205 symbols, and prices
    4,221 of 5,819 panel names -- while SEVEN venues are unpriceable at EVERY anchor (`.PA` 569
    names, `.KS` 327, `.OL` 224, `.KQ` 159, `.BR` 105, `.AS` 104, `.LS` 33), which is where
    eight of the 08-22 top-20 live.  Every backtest figure below rode that grid and nothing
    said so.  The audit measures THROUGH `returns_core.PriceSource`, so it cannot disagree with
    the grader about what is priceable.

    THE FETCH DECISION IS DELIBERATELY UNTOUCHED.  Absent -> fetch (as before); present -> no
    fetch (as before).  Auto-refetching on staleness would spend money nobody authorised; a
    silent stale grid and a surprise paid refetch are both unacceptable, so this is the third
    option -- say so, unmissably, every run.  `configdic['price_grid_refuse_when_stale']`
    turns the report into a refusal if the CEO wants that instead.
    """
    import price_grid_audit as pga
    if not os.path.exists(_PRICES_CSV):
        log("[price-audit] no grid on disk to audit (the fetch stage above reports on it).")
        return None
    return pga.run_audit(
        _PRICES_CSV, _panel_symbols(resdic),
        supp_csv=_PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None,
        log=log,
        refuse_when_stale=bool(configdic.get("price_grid_refuse_when_stale", 0)),
        #  The audit's headline claim is about what the STAGES compute on, so it has to be
        #  told which route they will use.  Same configdic key `_build_price_source` reads,
        #  so the two cannot disagree.
        price_route=str(configdic.get("price_route", "real") or "real"))


def _audit_price_scale_stage(resdic, configdic, log):
    """VENDOR PRICE-LEVEL audit -- report only, never a correction, never a score change.

    THE DEFECT.  FMP serves `ATRI`'s OHLC and `adjClose` at 1/1000 of the real tape while
    leaving `vwap` on it (`vwap/close` median exactly 1000.0 across all 756 rows; the final
    pinned `vwap` 459.92 matches the ~$460/share cash consideration).  Returns are unaffected
    -- the scaling cancels -- but every use of the price LEVEL is wrong by 1000x, and no level
    screen can find it because 16.9% of the grid's symbols are legitimately under $1 already.

    WHY THE CHECK IS NOT THE ONE THE REGISTER PROPOSED.  Q-38 proposed synthetic-vs-grid.
    Measured on ATRI's own saved fundamentals: FMP scales `marketCap` by the same 1/1000, so
    the synthetic price is 0.70490 against a grid 0.67507 -- a ratio of 1.04.  The proposed
    detector returns "agrees" on the one name known to be broken.  What separates them is the
    BALANCE SHEET, which the scaling does not touch: price/book comes out at 0.0116, i.e. the
    company priced at a hundredth of its own book equity.  See `price_scale_audit`.

    THIS STAGE DOES NOT GATE.  THE PIPELINE NOW DOES, UPSTREAM (Q-48 actioned, 2026-09-01).
    The paragraph that stood here said "changing a score on a heuristic is the CEO's decision.
    So this prints ... and nothing downstream consumes it."  The CEO reopened Q-48 and ruled the
    other way: reading a corrupted vendor number is a BUG, not a scoring preference.
    `nan_policy.price_scale_hits` now refuses the contradicted cells at ingest
    (`the refusal hook in getData_fmp.getFundamentalsData`), so by the time this stage runs, a row under `PB_ALARM` has already
    had `price`, `marketCap`, `bookValuePerShare` and `earningsYield` set to NaN.
    THE CONSEQUENCE FOR THIS STAGE IS THAT ITS SILENCE MEANS SOMETHING DIFFERENT: check A reads
    the two legs the refusal blanks, so on a post-refusal panel it reports what SURVIVED, not
    the run's exposure.  `run_audit` prints an A0 block naming the refused sources first, off
    the `nan_policy.SANITY_REFUSED_COLUMN` stamp -- read that before the ALARM count.
    """
    import price_scale_audit as psa
    panel = None
    for key in ("cdx_df",):
        try:
            panel = resdic[key]
            break
        except Exception:
            continue
    if panel is None or not len(panel):
        log("[price-scale] no cdx_df panel in resdic -- audit did not run.")
        return None
    #  LOG ONLY, no evidence CSV, matching its neighbour `price_grid_audit`.  The alarm list
    #  is single-digit on the measured panel so it fits in the log the CEO already reads, and
    #  a new dated artifact would need a new .gitignore rule and land in the same place Q-29
    #  is still open about.  `price_scale_audit.run_audit(out_csv=...)` remains available for
    #  a standalone investigation.
    #  THE RUN'S OWN RANKING AND THE RUN'S OWN SHIPPED LIST, so the containment sentence is
    #  COMPUTED rather than recited.  `resdic['BoScore_df']` is the FULL pre-carve, pre-veto
    #  Stage-1 frame (postBo assigns it once and stores it unchanged); `resdic['postRank']`
    #  is the Stage-2 survivor set that actually ships.  Both are read defensively -- absent,
    #  the audit says CONTAINMENT NOT CHECKED, which is the honest answer and not an
    #  all-clear.
    def _get(key, col=None):
        try:
            v = resdic[key]
            return list(v[col]) if col else v
        except Exception:
            return None

    return psa.run_audit(
        panel,
        prices_csv=_PRICES_CSV if os.path.exists(_PRICES_CSV) else None,
        supp_csv=_PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None,
        stage1_scores=_get("BoScore_df"),
        shipped_sources=_get("postRank", "source"),
        #  THE RUN'S OWN REFUSAL REPORT, so A0 can print HOW FAR under the cut each refused
        #  source sat.  The ratio is not recoverable from the panel -- `marketCap` is one of
        #  the cells the refusal blanks -- so without this A0 names sources and no numbers.
        refusal_report=_get("inputSanityRefusals"),
        panel_label="LIVE cdx_df -- survivor-only; NO delisted name is in it",
        log=log)


def _alarm_sources(audit_result):
    """The ALARM set out of a `price_scale_audit.run_audit` result, or an empty set."""
    try:
        internal = (audit_result or {}).get("internal")
        if internal is None or not len(internal):
            return set()
        return set(str(x) for x in internal.loc[internal["severity"] == "ALARM", "source"])
    except Exception:
        return set()


def _audit_price_scale_pit_stage(merged, per_anchor, live_audit, survivorship_clean, log):
    """CHECK A OVER THE PIT DEAD-MERGED PANEL -- the population the backtest actually ranks.

    THE BLIND SPOT THIS CLOSES, AND IT WAS THE MOTIVATING NAME.  The live pass reads
    `resdic['cdx_df']`, which is survivor-only: ATRI has **0 rows in it** on 2026-08-29 and
    on 2026-08-31 alike, because it delisted in 2024.  So check A -- written specifically
    because check B CANCELS on the ATRI shape (FMP scales `marketCap` by the same 1/1000, so
    the ratio is ~1.0) -- could not see ATRI either.  Both checks blind to the one name the
    stage exists for, one of them silently, and the module's own blind-spot note disclosed it
    without closing it.

    WHERE ATRI IS LIVE is the PIT dead-merged pool: it appears in the buy2020 carve ("KEPT ON
    UNKNOWN CURRENCY"), and there `bookToPrice` is a Tier-B, Sign +1, HIGHER-IS-BETTER input
    to every backtest ranking -- 1000x too favourable and, until this stage, unmeasured.  A
    contaminated name reaching a graded PIT top-20 moves the beat-rate the CEO reads.

    CHECK B IS DELIBERATELY NOT RUN HERE.  Its comparison is against the saved price grid, and
    the dead names are largely the ones the grid does not carry -- so "no disagreement" would
    be an artifact of absence.  Saying it was not run beats printing a zero.

    WHAT THIS STILL CANNOT SEE.  It ranks `bookToPrice` over the merged panel and checks the
    flagged names against the graded SELECTIONS, but it has no PIT Stage-1 frame, so it makes
    no claim about the margin to the top-100 cutoff -- the live pass's margin sentence has no
    counterpart here, and the audit prints no substitute for it.

    REPORT ONLY *HERE*, AND THAT IS NOW AN ASYMMETRY WORTH KNOWING ABOUT.  Since 2026-09-01 the
    LIVE ingest refuses the contradicted cells (`nan_policy.price_scale_hits`, Q-48) -- but this
    pass reads the PIT dead-merged panel, whose rows were assembled from saved delisted
    fundamentals rather than from tonight's fetch.  Whether they carry the refusal depends
    entirely on whether the pickle that supplied them was built after that date, and today it
    was not.  So the backtest ranking can still be reading a contaminated `bookToPrice` while
    the live scorer is not, and this stage is the only thing that says so.  Do not read a quiet
    LIVE pass as covering this population.
    """
    import price_scale_audit as psa
    if not survivorship_clean:
        #  `_build_pit_inputs` degrades to `merged is dmdic` when the delisted inputs are
        #  absent.  Auditing THAT panel a second time would print a second identical report
        #  under a heading claiming it covered the dead names -- a stage that reads as
        #  coverage it does not have is worse than an absent stage.
        log("[price-scale/PIT] SKIPPED: the run is not survivorship-clean (no delisted "
            "inputs), so the 'dead-merged' panel IS the live one.  The backtest population "
            "is NOT audited this run.")
        return None
    panel = merged.get("cdx_df") if isinstance(merged, dict) else None
    if panel is None or not len(panel):
        log("[price-scale/PIT] no merged cdx_df -- the dead-merged pass did not run.")
        return None

    #  THE GRADED SELECTIONS, POOLED across anchors.  `top20_deduped` is the list each anchor
    #  actually ships in the backtest; `ranking` is the Stage-2 pool it is chosen from and is
    #  the fallback when an anchor carries no deduped list.  Pooling is the right scope for a
    #  containment question asked once: a flagged name reaching ANY graded anchor's list is
    #  the finding.
    selected, pool = [], []
    for _wid, rec in (per_anchor or {}).items():
        selected.extend(rec.get("top20_deduped") or [])
        pool.extend(rec.get("ranking") or [])
    n_anchors = len(per_anchor or {})
    n_pit = int(panel["source"].nunique())
    log("[price-scale/PIT] check A over the DEAD-MERGED panel: %d sources, %d graded anchors, "
        "%d pooled top-20 names" % (n_pit, n_anchors, len(set(selected))))
    out = psa.run_audit(
        panel,
        stage1_scores=None,          # no PIT Stage-1 frame here; the margin is NOT claimed
        shipped_sources=sorted(set(selected)) or sorted(set(pool)),
        run_grid_check=False,
        panel_label="PIT dead-merged cdx_df -- %d sources, %d graded anchors"
                    % (n_pit, n_anchors),
        log=log)

    #  THE NAMES THE LIVE PASS COULD NOT SEE, NAMED.  This difference IS the finding: an
    #  ALARM present here and absent there is a contaminated name living only in the backtest
    #  population -- the ATRI shape exactly.  Printed even when empty, because "the second
    #  pass found nothing new" and "the second pass did not run" must not look alike.
    dead_only = sorted(_alarm_sources(out) - _alarm_sources(live_audit))
    if dead_only:
        log("[price-scale/PIT] ALARM ONLY IN THE BACKTEST POPULATION (invisible to the live "
            "pass): %s" % ", ".join(dead_only))
    else:
        log("[price-scale/PIT] no ALARM name is unique to the dead-merged panel this run.")
    return out


def _build_price_source(log, configdic=None):
    """The outcome-variable price source for the whole analysis suite.

    'real' IS AND REMAINS THE DEFAULT.  `configdic['price_route']` selects another route
    explicitly -- 'real+derived' is the gap-fill (real wherever real has a price, the
    fundamentals-panel total-return leg only where it is empty), which is the one the CEO
    asked for while the price refetch stays deferred.  See derived_prices.GapFillPriceSource,
    and read its refusal report: on the run machine's grid it fills 1,187 names and REFUSES
    ~120 on the listing-currency guard, and those refusals are the correct output rather than
    a gap to paper over.

    THE PANEL IS AN EXTRA INPUT.  A non-real route needs the deep fundamentals panel; if it
    is absent this FALLS BACK TO 'real' with a loud line rather than killing five analysis
    stages over a configuration choice.
    """
    import returns_core as rc
    configdic = configdic or {}
    supp = _PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None
    if not os.path.exists(_PRICES_CSV):
        raise RuntimeError(f"real price grid absent: {_PRICES_CSV} "
                           "(price-fetch stage did not produce it)")
    route = str(configdic.get("price_route", "real") or "real")
    #  INTERIOR-HOLE FILL (D1).  Default OFF: it MOVES MEASURED NUMBERS, on windows that
    #  currently read -100% for a name a later price proves was alive.  Turning it on needs
    #  the fundamentals panel, so it forces a non-real route internally even when the
    #  route is 'real'.
    #
    #  MEASURED BEFORE YOU TURN IT ON: on the run machine's grid it fills 10 holes and
    #  REFUSES 693, 692 of them because the survivors-only panel cannot bridge the gap
    #  (614 of those on `.L`, which is mostly depositary lines the panel does not carry).
    #  So the mechanism is correct and proven and the DATA barely feeds it -- 1.4% of the
    #  hole population.  It is also INERT on the shipped top-20 grading: zero imputed cells
    #  among the graded picks at either clean 36-month anchor, in either exchange scope.
    fill_holes = str(configdic.get("fill_interior_holes", 0)) in ("1", "True", "true")
    if route == "real" and not fill_holes:
        ps = rc.PriceSource(_PRICES_CSV, supp_csv=supp)
        log(f"[price-source] route=real  PriceSource built from "
            f"{os.path.basename(_PRICES_CSV)}"
            + (f" + {os.path.basename(_PRICES_2025_CSV)}" if supp else " (no 2025 supp)"))
        return ps

    import derived_prices as dpx
    if route not in dpx.PRICE_ROUTES:
        raise RuntimeError(f"configdic['price_route']={route!r} is not one of "
                           f"{dpx.PRICE_ROUTES}")
    panel = configdic.get("price_route_panel") or dpx.DEFAULT_PANEL_GLOB
    try:
        ps = dpx.build_price_source(route, prices_csv=_PRICES_CSV, supp_csv=supp,
                                    panel=panel, fill_interior_holes=fill_holes)
    except (FileNotFoundError, KeyError) as e:
        #  NOTE THE ORDERING LIMITATION, stated rather than hidden: the price-grid audit
        #  stage runs BEFORE this one and has already printed a banner naming the CONFIGURED
        #  route, so on a fallback the banner and reality disagree for one run.  This line is
        #  the correction, and it is why it says "every number below" explicitly.
        log(f"[price-source] route={route} UNAVAILABLE ({type(e).__name__}: {e}) -- "
            f"FALLING BACK to route=real.  Every number below is on the real grid, and the "
            f"price-grid audit banner above named {route!r} -- THIS line is the correct one.")
        return rc.PriceSource(_PRICES_CSV, supp_csv=supp)
    log(f"[price-source] route={route}  fill_interior_holes={fill_holes}  (real grid "
        f"{os.path.basename(_PRICES_CSV)} + panel {os.path.basename(str(panel))})")
    inner = getattr(ps, "real", None)
    if hasattr(inner, "imputation_report"):
        d = inner.diagnostics()
        log(f"[price-source]   holes filled={d['n_holes_filled']} "
            f"refused={d['n_holes_refused']} {d['refusal_reasons']}")
    elif hasattr(ps, "imputation_report"):
        d = ps.diagnostics()
        log(f"[price-source]   holes filled={d['n_holes_filled']} "
            f"refused={d['n_holes_refused']} {d['refusal_reasons']}")
    for k, v in ps.diagnostics().items():
        if k in ("route", "n_tickers_gapfilled", "n_tickers_real_priceable",
                 "n_tickers_derived_priceable", "leg_selection", "bias_measurability"):
            log(f"[price-source]   {k} = {v}")
    return ps


# --------------------------------------------------------------------------- #
#  Shared PIT inputs: dead-merge built ONCE, reused across every stage         #
# --------------------------------------------------------------------------- #
def _resolve_delisted_dir(configdic):
    d = configdic.get("delisted_out")
    cands = ([d] if d else []) + _DEFAULT_DELISTED_DIRS
    for c in cands:
        if c and os.path.isdir(c):
            reg = os.path.join(c, "delisted_registry.csv")
            deads = [f for f in os.listdir(c) if f.startswith("dead_fundamentals_")
                     and f.endswith(".pickle")]
            if os.path.exists(reg) and deads:
                deads.sort()
                return os.path.join(c, deads[-1]), reg
    return None, None


def run_level_break_referee_stage(resdic, configdic, log):
    """LEVEL-BREAK REFEREE (D2) -- report only, and it changes NO number.

    Where the real leg makes an extreme single-period move the derived leg does not
    corroborate, the real price is suspect.  `SIMINN.IC` steps 0.2256 -> 9.23 across one
    anchor and reads +36,787% over 36 months; the derived leg reads +25.4%.

    DEFAULT ON, because it prints and nothing else -- no price is overridden.  Set
    `configdic['level_break_referee'] = 0` to silence it.  It needs the fundamentals panel;
    absent that it SKIPS with a line rather than failing the run.

    ITS COST, STATED RATHER THAN HIDDEN: on the default 'real' route nothing else in this
    suite loads the panel, so this stage ADDS a ~276 MB pickle read (tens of seconds) to
    every run.  That is the price of a second opinion and it buys nothing on a clean run.
    If that becomes the wrong trade, turn it off by config -- but turn it off deliberately,
    because a detector defaulted to silent detects nothing.

    PREVENTIVE, NOT CORRECTIVE, and the reason is measured: of the 17 cells flagged on the
    run machine's grid, THREE are the derived leg's own fault (g_derived == 1.000000 exactly,
    both ends priced off the same filing) -- so an automatic "trust the derived leg" rule
    would ship three known-wrong corrections.  None of the 461 names with a >= 5x step
    reaches a top-100 pick in either exchange scope today, so nothing is on fire; this exists
    so the next one is seen before it lands in a headline.
    """
    if str(configdic.get("level_break_referee", 1)) in ("0", "False", "false"):
        log("[referee] level_break_referee disabled by config -- skipped")
        return None
    import derived_prices as dpx
    import returns_core as rc
    if not os.path.exists(_PRICES_CSV):
        log("[referee] no price grid on disk -- skipped")
        return None
    supp = _PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None
    panel = configdic.get("price_route_panel") or dpx.DEFAULT_PANEL_GLOB
    try:
        real = rc.PriceSource(_PRICES_CSV, supp_csv=supp)
        derived = dpx.DerivedPriceSource(panel, benchmark_source=real)
    except (FileNotFoundError, KeyError) as e:
        log(f"[referee] panel unavailable ({type(e).__name__}) -- skipped.  The referee "
            f"needs TWO legs; with one it has no second opinion to offer.")
        return None
    cand = dpx.level_break_candidates(real, derived)
    rep = dpx.level_break_report(real, derived)
    print("\n" + "#" * 72)
    print("# LEVEL-BREAK REFEREE  --  real leg vs derived leg, single-period disagreement")
    print("#" * 72)
    print(f"  corpus: {rep['n_symbols_both_legs_price']} symbols BOTH legs price "
          f"(cut: real step >= {rep['min_real_step']}x AND |log gap| >= "
          f"{rep['min_log_gap']})")
    print(f"  flagged: {rep['n_cells_flagged']} cell(s) / {rep['n_symbols_flagged']} "
          f"symbol(s) -- {rep['n_legs_disagree']} genuine disagreement, "
          f"{rep['n_derived_uninformative']} where the DERIVED leg has no opinion")
    if len(cand):
        print(cand.to_string(index=False))
    else:
        print("  (none at this cut)")
    print(f"  ACTION: {rep['action']}")
    print(f"  BLIND TO: {rep['blind_to']}", flush=True)
    if rep["n_legs_disagree"]:
        log(f"[referee] {rep['n_legs_disagree']} suspect real-price step(s) -- see the "
            f"block above; NOTHING was overridden")
    return rep


def _build_pit_inputs(dmdic, configdic, log):
    """Build (merged, registry) ONCE for the whole suite.

    SURVIVORSHIP-CLEAN path: dead names merged in (dm.merge_dead_into_dmdic, as-of the
    earliest buy anchor) + registry loaded, when the delisted_out inputs are present.
    DEGRADED path: if the delisted inputs are ABSENT, fall back to a SURVIVOR-ONLY run
    (merged == dmdic, empty registry) with a LOUD caveat -- the analysis still runs but
    carries survivorship bias.  Returns (merged, registry, survivorship_clean:bool)."""
    import pandas as pd
    import dead_merge as dm

    dead_path, reg_path = _resolve_delisted_dir(configdic)
    if not dead_path:
        caveat = ("\n" + "!" * 78 + "\n"
                  "!!! ANALYSIS CAVEAT -- delisted_out inputs NOT FOUND !!!\n"
                  "!!! Running the PIT analysis SURVIVOR-ONLY (dead names absent) -- the  !!!\n"
                  "!!! beat-rate / grid / oracle numbers carry SURVIVORSHIP BIAS this run.!!!\n"
                  "!!! Provide delisted_out (registry + dead_fundamentals pickle) for a   !!!\n"
                  "!!! survivorship-clean read.                                           !!!\n"
                  + "!" * 78 + "\n")
        print(caveat, file=sys.stderr, flush=True)
        print(caveat, flush=True)
        return dmdic, pd.DataFrame(), False

    log(f"[pit-inputs] delisted inputs: dead={os.path.basename(dead_path)} "
        f"registry={os.path.basename(reg_path)}")
    dead = pd.read_pickle(dead_path)
    registry = dm.load_registry(reg_path)
    #  NOT "the earliest clean buy anchor" any more, and the old comment saying so became
    #  false the moment buy2020 was promoted.  What this date actually does is fix the
    #  `pit_universe` SNAPSHOT stored on the merged dict -- and NOTHING in this pipeline
    #  grades off that snapshot: `dhg.rank_all_anchors` recomputes the universe per anchor
    #  with `as_of=buy`, and the dead names' rows are unioned into `cdx_df` regardless of this
    #  date, with `reproduce_pit_top` applying its own `date <= D`.  The only consumer of the
    #  stored key is the standalone `run_target_test.py`.  So promoting an anchor EARLIER than
    #  this date does not create a look-ahead -- checked before the promotion, because it
    #  would have been one if `rank_all_anchors` had reused the snapshot.
    #  Left at 2021-12-31 deliberately: moving it would change the merged universe for EVERY
    #  anchor, which is a numbers-moving change and not part of correcting a false comment.
    merge_as_of = _CLEAN_36MO_WINDOWS[0][0]
    log(f"[pit-inputs] dead-merge as-of {merge_as_of} (ONCE; shared across stages) ...")
    merged, stats = dm.merge_dead_into_dmdic(dmdic, dead, registry, as_of=merge_as_of)
    log(f"[pit-inputs] merge: universe={stats.get('universe_size')} "
        f"built_dead={stats.get('built')} gate_fail={stats.get('gate_fail')}")
    return merged, registry, True


# --------------------------------------------------------------------------- #
#  Stage: beat-rate vs URTH (operational-target proxy) -- reuses per_anchor    #
# --------------------------------------------------------------------------- #
def _bench_or_skip(price_source, buy, ev, wid, rc, where):
    """Thin delegation to `returns_core.benchmark_return_or_none` -- see its docstring for
    the composition defect (a strict raise inside a per-anchor loop under a swallowing stage
    guard, so one missing anchor erased the whole stage).

    IT MOVED because this was not the only place with the shape: `gate_attribution` and
    `skill_baseline` had four more bare call sites between them, both running in this same
    suite and both widened by the buy2020 promotion.  Fixing it here and copying the fix
    there is how the two would drift in strictness; one definition in `returns_core`, which
    all of them already import, is the fix.
    """
    return rc.benchmark_return_or_none(price_source, buy, ev, wid, where)


def beat_rate_vs_urth(per_anchor, price_source, log, depths=(10, 20),
                      horizon_m=36, threshold=0.10, merged=None):
    """The operational-target readout on the DEPLOYED FILTER: share of the shipped
    top-N that beat URTH (MSCI World TR proxy) by >= threshold over a `horizon_m` hold,
    on the CLEAN buy anchors.

    Uses the ISSUER-DEDUPED, CARVE-ON top-20 (`per_anchor[wid]["top20_deduped"]`) --
    exactly the general list the pipeline ships (carve partition + issuer-dedup, both
    default ON) -- NOT the raw undeduped pool.  This is the same deduped-top20 basis
    skill_baseline's filter uses.  Pure returns_core URTH path (rc.beat_rate +
    rc.benchmark_return_or_none, still require_exact underneath, so a missing benchmark
    anchor costs one window and not the stage).  Prints a report and returns the
    per-window + pooled beat-rate rows.

    THE "ONLY DIFFERENCE IS THE CARVE" CLAIM WAS FALSE FOR FOUR DAYS, AND IS NOW CHECKABLE
    RATHER THAN ASSERTED.  It was written when the anchor sets diverged, corrected for the
    anchor sets, and left standing while a SECOND divergence -- the Stage-1 solvency veto --
    was still open: this table runs VETOED (`dhg.rank_all_anchors(stage1_veto=True)`) and
    `skill_baseline` ran UN-VETOED, so 25.0% and 25.9% were printed side by side on
    different bases under a sentence saying they were not.  Both stages now stamp their
    basis through `basis_stamp`, and the header points a reader at the two stamps instead of
    promising something this function cannot see from here.  A claim about ANOTHER stage's
    basis belongs in that stage's stamp, not in this one's prose -- which is the general
    form of the defect, not a detail of it."""
    import numpy as np
    import returns_core as rc
    import depth_horizon_grid as dhg
    import target_clauses as _tc

    print("\n" + "#" * 72)
    _print_basis_banner(per_anchor)
    print("# BEAT-RATE vs URTH  --  DEPLOYED FILTER (issuer-deduped, carve-ON top-20)")
    print(f"#   the shipped general list beats MSCI World by >= {threshold*100:.0f}pp?")
    print(f"#   horizon = {horizon_m}mo   benchmark = {rc.BENCHMARK_VARIANT}")
    print(f"#   CLEAN 36mo windows only ({_clean_window_text()}).")
    print("#   (skill_baseline reports the same deduped-top20 selection, carve-OFF.  BOTH")
    print("#    stages stamp their MEASUREMENT BASIS -- read the two stamps before")
    print("#    comparing the two numbers; do not take it on trust from this line.)")
    print("#" * 72)

    rows = []
    pooled_flags = {N: [] for N in depths}
    #  DOWNSIDE-CLAUSE INPUTS.  The second half of the target needs the per-anchor top-20
    #  RETURNS TABLE, not just its beat-rate, so it is captured in the same loop rather than
    #  recomputed later -- one selection, one price read, both clauses.
    clause_inputs = []
    for wid, buy in dhg.BUY_ANCHORS:
        if wid not in per_anchor or wid not in dhg.CLEAN_BUY_IDS:
            continue
        buy_idx = dhg.ANCHOR_IDX[buy]
        eval_idx = buy_idx + horizon_m // 12
        if eval_idx >= len(dhg.ANCHORS):
            continue
        ev = dhg.ANCHORS[eval_idx]
        # DEPLOYED deduped top-20 (the shipped pick); top-10 is its head slice.
        deployed = per_anchor[wid].get("top20_deduped") or per_anchor[wid]["ranking"][:20]
        bench = _bench_or_skip(price_source, buy, ev, wid, rc, "the two-clause / beat-rate table")
        if bench is None:
            continue
        for N in depths:
            top = deployed[:N]
            rdf = rc.compute_returns(top, buy, ev, price_source)
            br, n = rc.beat_rate(rdf, bench, threshold=threshold, missing="fail")
            rows.append({"window": f"{buy}->{ev}", "depth_N": N,
                         "beat_rate": br, "n": n, "bench_ret": bench})
            if N == _tc.CHARTERED_DEPTH:
                #  `n_selected` is len(top), NOT the nominal depth: if the anchor shipped 17
                #  names, counting 20 would invent three phantom unpriced picks and understate
                #  coverage.  The gap that matters is priced-vs-SHIPPED.
                clause_inputs.append({"window": f"{buy}->{ev}", "n_selected": len(top),
                                      "rdf": rdf, "beat_rate": br, "n": n,
                                      "bench": bench})
            # pooled: recompute flags at the per-name level for an honest pooled rate
            inc = rc.included(rdf)
            for _, r in inc.iterrows():
                if r["terminal_flag"]:
                    pooled_flags[N].append(False)  # missing='fail'
                else:
                    pooled_flags[N].append((r["total_return"] - bench) >= threshold)
    hdr = f"  {'window':22} {'N':>4} {'beat_rate':>10} {'n':>5} {'bench':>9}"
    print(hdr)
    for r in rows:
        brs = f"{r['beat_rate']*100:.1f}%" if r["beat_rate"] == r["beat_rate"] else "n/a"
        bes = f"{r['bench_ret']*100:.1f}%" if r["bench_ret"] == r["bench_ret"] else "n/a"
        print(f"  {r['window']:22} {r['depth_N']:>4} {brs:>10} {r['n']:>5} {bes:>9}")
    print("  --- POOLED across clean windows ---")
    pooled = {}
    for N in depths:
        flags = pooled_flags[N]
        rate = float(np.mean(flags)) if flags else float("nan")
        pooled[N] = {"beat_rate": rate, "n": len(flags)}
        rs = f"{rate*100:.1f}%" if rate == rate else "n/a"
        print(f"  POOLED top-{N}: beat_rate={rs} (n={len(flags)})")
    #  COUNTED, not asserted.  This read "2 heavily-overlapping windows" as a literal and
    #  went wrong the moment a third anchor was graded -- in a CAVEAT, whose whole job is to
    #  stop a reader over-reading the n above it.
    _nw = len({r["window"] for r in rows}) if rows else 0
    print(f"  CAVEAT: {_nw} heavily-overlapping window(s) = ONE regime; count-based "
          "(magnitude-")
    print("          blind); missing-eval counts as NOT beating (missing='fail').")

    #  ============ THE SECOND CLAUSE, printed BESIDE the first ============
    #  The beat-rate above is HALF the target.  It was shipped alone for six days and read as
    #  the target, which is the failure mode this block exists to make impossible: the
    #  DOWNSIDE clause is printed in the same stage, from the same selection and the same
    #  price read, so a run can never again report one clause and call it the result.
    two_clause = _two_clause_report(clause_inputs, horizon_m, threshold,
                                    basis=_basis_of(per_anchor))

    # ================= ADDITIVE: per-market-cap-band beat-rate ==================
    # GROUP the deployed general ranking into USD market-cap bands and grade each band
    # SEPARATELY (CEO 2026-07-17): General -> top-20, Mid/Small/Micro -> top-5. Market cap
    # is POINT-IN-TIME as-of buy (latest marketCap_usd <= buy from merged['cdx_df'];
    # historical, NOT today's). Reuses carveOut.MCAP_BANDS + the shared FX/USD path, so
    # selection (production) and grading (here) key off the SAME cutoffs and USD field.
    # Prints each band's member count n. Small bands (<$150M, esp <$50M) are labelled
    # DIRECTIONAL-ONLY -- too few names for a meaningful 60% pass/fail. The existing
    # combined top-20 pooled rate above is UNCHANGED (this is strictly additive).
    band_rows, band_pending = _per_band_beat_rate(
        per_anchor, price_source, merged, horizon_m, threshold)
    return {"per_window": rows, "pooled": pooled,
            "bands": band_rows, "band_pending": band_pending,
            "two_clause": two_clause}


# --------------------------------------------------------------------------- #
#  The TWO-CLAUSE target readout (UPSIDE + DOWNSIDE + the three diagnostics)   #
# --------------------------------------------------------------------------- #
def _pct(x, width=9):
    return f"{x*100:+.1f}%".rjust(width) if x == x else "n/a".rjust(width)


#  THE BASIS READER MOVED TO `basis_stamp` and these three names are now thin delegations.
#  It moved because `depth_horizon_grid` and `scoring_compare` need the same reader and
#  NEITHER CAN IMPORT THIS MODULE -- `pipeline_analysis` imports `dhg`, so the dependency only
#  runs one way.  The alternative was a second copy of the parse living next to the report
#  that prints it, which is how the ejection-count false alarm would have been fixed in one
#  place and not the other.  The names are kept (with the underscore) because the veto tests
#  reach for `pa._basis_of` directly and the delegation is the point being tested: one reader.
def _basis_stamp():
    #  Imported lazily, like every other sibling in this module: `pipeline_analysis` is
    #  imported by the pipeline before `baseline_tools/` is necessarily on the path, so the
    #  path guard travels with the import rather than sitting at module scope.
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import basis_stamp
    return basis_stamp


def _basis_kind(basis):
    return _basis_stamp().kind(basis)


def _basis_of(per_anchor):
    return _basis_stamp().of(per_anchor)


def _print_basis_banner(per_anchor):
    bs = _basis_stamp()
    basis = bs.of(per_anchor)
    for line in bs.banner_lines(basis):
        print(line)
    return basis


def _two_clause_report(clause_inputs, horizon_m, threshold, basis="unknown"):
    """Print and return the charter two-clause verdict, per anchor and pooled.

    ONLY THE TOP-20 IS GRADED HERE.  The charter states both clauses on the top-20 and nowhere
    else; grading depth 10 as well would invent a clause the CEO did not set (the top-10 52.5%
    rung is an UPSIDE rung, not a second target).  The beat-rate table above still reports both
    depths, which is why every line here names its depth.

    BOTH CLAUSES GET THE SAME COVERAGE DISCIPLINE, which is what stops this block contradicting
    itself.  An earlier version applied it to the downside clause only and handed the upside
    clause a beat-rate computed over whichever picks happened to be priceable; since a FAIL on
    either clause sinks the period, the run printed `PERIOD: FAIL` on the same line as
    `DOWNSIDE: INDETERMINATE`.  A beat-rate is bounded on both sides, so partial coverage gives
    a real interval and a partial-coverage FAIL is genuinely provable -- see
    `target_clauses.upside_clause`.

    ONLY THE PRIMARY POLICY IS PRINTED, and the reason is stated in the output rather than
    left for a reader to infer.  `target_clauses.measured` classes every pick with no eval leg
    as unmeasured, so the FLOOR policy has nothing left to act on: every remaining row carries
    the same number in `total_return` and `total_return_floor`, and a FLOOR row would be a
    bit-for-bit copy of the PRIMARY row above it.  Printing four rows where two carry no
    information is its own kind of misleading.  Both policies are still COMPUTED and both are
    still in the returned dict, so nothing downstream loses access; the floor READING is
    `lower_bound`, which is algebraically the number the old FLOOR lower bound produced.
    """
    import pandas as pd
    import returns_core as rc
    import target_clauses as tc

    bar = tc.bond_bar(horizon_m)
    print("\n" + "#" * 72)
    print("# TWO-CLAUSE TARGET  --  BOTH must pass for a period to count as success")
    print(f"#   UPSIDE   : >= 60% of the top-{tc.CHARTERED_DEPTH} beat URTH by "
          f">= {threshold*100:.0f}pp over {horizon_m}mo   [per pick]")
    print(f"#   DOWNSIDE : equal-weight top-{tc.CHARTERED_DEPTH} total return exceeds a flat "
          f"{tc.BOND_RATE_ANNUAL*100:.0f}%/yr bond   [portfolio]")
    print(f"#   bond bar : {bar*100:+.2f}% compounded over {horizon_m}mo")
    print(f"#   BASIS    : {basis}")
    print("#   The clauses are NOT tradable against each other -- that is the point of the")
    print("#   pair: a filter can always buy a higher hit rate with tail risk.")
    print("#" * 72)
    for label, text in (("BOND BAR", tc.BOND_BAR_CAVEAT),
                        ("SOFTNESS", tc.SOFTNESS_CAVEAT),
                        ("POLICY  ", tc.POLICY_CAVEAT),
                        ("COVERAGE", tc.COVERAGE_CAVEAT)):
        print(f"  {label}: {_wrap(text)}")

    if not clause_inputs:
        print("  NO CLEAN WINDOW produced a top-20 -- the two-clause target is UNMEASURED "
              "this run (not a pass, not a fail).")
        return {"bar": bar, "per_anchor": [], "pooled": None, "n_windows": 0,
                "basis": basis}

    per_anchor_out = []
    for ci in clause_inputs:
        for floor in (False, True):
            pol = "floor" if floor else "primary"
            dn = tc.downside_clause(ci["rdf"], ci["n_selected"], horizon_m, floor=floor)
            up = tc.upside_clause(ci["rdf"], ci["bench"], ci["n_selected"],
                                  threshold=threshold, floor=floor)
            per_anchor_out.append({
                "window": ci["window"], "policy": pol, "upside": up, "downside": dn,
                "period": tc.period_verdict(up, dn),
                "diagnostics": tc.diagnostics(ci["rdf"], bar, floor=floor),
                "legacy_missing_fail_rate": ci["beat_rate"], "legacy_n": ci["n"]})

    #  ---- COVERAGE, first, because it conditions every number under it ----
    #  THE COLUMNS PARTITION THE SHIPPED PICKS: measured + stale + buy_only + no_buy =
    #  shipped.  `terminal` used to be printed as a single column ALONGSIDE a `measured` count
    #  that already contained most of it, which is how a reader -- and the clause itself --
    #  could read "16 of 20 measured" off an anchor with 9 terminals.  Split and summing to
    #  the shipped count, the line cannot be read that way: every pick is in exactly one bucket.
    #  `indet` JOINS THE PARTITION, it does not join `stale`.  The columns are printed
    #  BECAUSE they sum to `shipped`; a fifth bucket that existed in the data and not in this
    #  row would have broken the sum silently and re-created the exact mis-reading the
    #  comment above describes, one bucket later.  `cont` is deliberately OUTSIDE the sum
    #  (those picks are already inside `measured`) and is spaced apart in the header to say so.
    print("\n  --- COVERAGE (what the anchor SHIPPED vs what could be MEASURED) ---")
    print(f"  {'window':24} {'shipped':>8} {'measured':>9} {'stale':>7} {'buy_only':>9} "
          f"{'indet':>7} {'no_buy':>7} {'coverage':>9}  | {'cont':>5}")
    for row in per_anchor_out:
        if row["policy"] != "primary":
            continue
        d = row["downside"]
        print(f"  {row['window']:24} {d['n_selected']:>8} {d['n_measured']:>9} "
              f"{d['n_terminal_stale']:>7} {d['n_buy_only']:>9} "
              f"{d.get('n_indeterminate', 0):>7} {d['n_no_buy']:>7} "
              f"{d['coverage']*100:>8.1f}%  | {d.get('n_continued', 0):>5}")
    print("  measured = BOTH legs priced at the chartered anchors. stale = eval leg substituted")
    print("  from an EARLIER anchor (12-24mo old, so not a reading of this window). buy_only =")
    print("  priced at the buy anchor alone. indet = the listing LINE ended for an identified")
    print("  reason (re-domicile / ticker change / preferred call) and no successor return is")
    print("  measurable -- unknown, and the ONE bucket where -100% is positively refuted rather")
    print("  than merely unproven. no_buy = never opened. Those four are UNKNOWN, not flat and")
    print("  not losses -- they are what lo/hi and lower_bound bracket, and they sum with")
    print("  measured to shipped. cont = picks INSIDE measured whose position was followed onto")
    print("  a SUCCESSOR line (issuer_continuity.py); it is not part of the sum.")
    #  THE ONE-LINE Q-42 READOUT, pooled over the shipped top-20 of every clean window.
    #  Printed unconditionally, zeros and all: "the map found nothing in this run" and "the
    #  map did not run" must not print identically.
    pooled_rdf_all = pd.concat([ci["rdf"] for ci in clause_inputs], ignore_index=True)
    print("  " + rc.continuity_report_line(pooled_rdf_all,
                                           where="(pooled shipped top-%d, %d window(s))"
                                                 % (tc.CHARTERED_DEPTH, len(clause_inputs))))
    print(f"  {_wrap(tc.CONTINUITY_CAVEAT)}")

    #  ---- DOWNSIDE ----
    print("\n  --- DOWNSIDE CLAUSE: equal-weight portfolio vs the bond bar ---")
    print(f"  {'window':24} {'pol':>7} {'portfolio':>10} {'lowerbnd':>10} {'flip':>10} "
          f"  {'VERDICT':<14} reason")
    for row in per_anchor_out:
        if row["policy"] != "primary":
            continue
        d = row["downside"]
        print(f"  {row['window']:24} {row['policy']:>7} {_pct(d['portfolio_return'], 10)} "
              f"{_pct(d['lower_bound'], 10)} {_pct(d['flip_return'], 10)}   "
              f"{d['verdict']:<14} {d['verdict_reason']}")

    #  ---- UPSIDE ----
    print("\n  --- UPSIDE CLAUSE: share of the shipped top-20 beating URTH by the bar ---")
    print(f"  {'window':24} {'pol':>7} {'n_beat':>7} {'measured%':>10} {'lo':>8} {'hi':>8} "
          f"  {'VERDICT':<14} reason")
    for row in per_anchor_out:
        if row["policy"] != "primary":
            continue
        u = row["upside"]
        mr = f"{u['rate_measured']*100:.1f}%" if u["rate_measured"] == u["rate_measured"] else "n/a"
        lo = f"{u['lo']*100:.1f}%" if u["lo"] == u["lo"] else "n/a"
        hi = f"{u['hi']*100:.1f}%" if u["hi"] == u["hi"] else "n/a"
        print(f"  {row['window']:24} {row['policy']:>7} {u['n_beat']:>7} {mr:>10} "
              f"{lo:>8} {hi:>8}   {u['verdict']:<14} {u['verdict_reason']}")
    legacy = {(r["window"], r["legacy_missing_fail_rate"]) for r in per_anchor_out}
    print("  FOR CONTINUITY with the beat-rate table above, which uses missing='fail' and so")
    print("  counts every unpriceable pick as a NON-beater rather than as unmeasured:")
    for w, lr in sorted(legacy, key=lambda t: t[0]):
        ls = f"{lr*100:.1f}%" if lr == lr else "n/a"
        print(f"    {w:24} missing='fail' rate = {ls}  (a POINT estimate, not the clause)")

    #  ---- PERIOD ----
    print("\n  --- PERIOD VERDICT: BOTH clauses must pass (charter) ---")
    print(f"  {'window':24} {'pol':>7} {'UPSIDE':<15} {'DOWNSIDE':<15} {'PERIOD':<15}")
    for row in per_anchor_out:
        if row["policy"] != "primary":
            continue
        print(f"  {row['window']:24} {row['policy']:>7} {row['upside']['verdict']:<15} "
              f"{row['downside']['verdict']:<15} {row['period']:<15}")
    #  THE POLICIES CAN NO LONGER DISAGREE, and the check is kept rather than deleted because
    #  the previous run printed a disagreement here and it was read as a price-grid finding.
    #  It was an artefact: the FLOOR row averaged observed returns with assumed -100%s over a
    #  denominator that excluded the picks nothing priced at all.  If this ever fires again it
    #  means a substituted price got back into `target_clauses.measured`.
    for ci in clause_inputs:
        pol = {r["policy"]: r["downside"]["verdict"]
               for r in per_anchor_out if r["window"] == ci["window"]}
        if pol.get("primary") != pol.get("floor"):
            print(f"  ^ {ci['window']}: PRIMARY and FLOOR disagree, which should now be "
                  "IMPOSSIBLE --")
            print("    every measured pick has both legs priced, so the two policies read the")
            print("    same rows. A substituted price has got back into the measured set.")
            print("    TREAT THE NUMBERS ABOVE AS UNSAFE.")

    #  ---- DIAGNOSTICS ----
    print("\n  DIAGNOSTICS -- reported, NEVER gating (charter). Each carries its own n.")
    print(f"  {'window':24} {'pol':>7} {'n':>4} {'p25':>9} {'rank':>5} {'clears':>7} "
          f"{'worst':>9} {'clears':>7} {'below0':>8}")
    for row in per_anchor_out:
        if row["policy"] != "primary":
            continue
        d = row["diagnostics"]
        cp = "-" if d["p25_clears_bar"] is None else ("yes" if d["p25_clears_bar"] else "NO")
        cw = "-" if d["worst_clears_bar"] is None else ("yes" if d["worst_clears_bar"] else "NO")
        print(f"  {row['window']:24} {row['policy']:>7} {d['n']:>4} {_pct(d['p25'])} "
              f"{d['p25_rank']:>5} {cp:>7} {_pct(d['worst'])} {cw:>7} "
              f"{str(d['n_below_zero']) + '/' + str(d['n']):>8}")
    print("  p25 is the ceil(0.25*n)-th SMALLEST pick (the charter \"5th-worst\" at n=20),")
    print("  not an interpolated percentile. 'worst' is EXPECTED to fail the bond bar --")
    print("  it is tracked for magnitude, not as a test.")
    print("  below0 IS NOT THE LOSS RATE OF THE SHIPPED LIST. Its denominator is the picks that")
    print("  could be priced; the unmeasured picks are UNKNOWN, and that population is enriched")
    print("  in names that stopped pricing -- a state losers reach more often than winners. So")
    print("  this share reads OPTIMISTIC, and it moving because the denominator shrank is not")
    print("  the filter improving.")

    #  ---- POOLED ----
    pooled_rdf = pd.concat([ci["rdf"] for ci in clause_inputs], ignore_index=True)
    pooled_selected = sum(ci["n_selected"] for ci in clause_inputs)
    pooled_out = {}
    print(f"\n  --- POOLED across {len(clause_inputs)} clean window(s) (per-name pooling; "
          "NOT the chartered per-anchor clause) ---")
    for floor in (False, True):
        dn = tc.downside_clause(pooled_rdf, pooled_selected, horizon_m, floor=floor)
        dg = tc.diagnostics(pooled_rdf, bar, floor=floor)
        pooled_out["floor" if floor else "primary"] = {"downside": dn, "diagnostics": dg}
        if floor:
            continue          # identical to primary by construction; see the POLICY caveat
        #  `indet` PRINTED HERE TOO.  Without it this line reads
        #  shipped=40 measured=38 stale=0 buy_only=0 no_buy=0 -- four buckets that no
        #  longer sum to `shipped`, which is the same silent-partition break the per-window
        #  table above exists to prevent, one summary line lower.
        print(f"  {'primary':>7}: shipped={dn['n_selected']} "
              f"measured={dn['n_measured']} stale={dn['n_terminal_stale']} "
              f"buy_only={dn['n_buy_only']} indet={dn['n_indeterminate']} "
              f"no_buy={dn['n_no_buy']} "
              f"portfolio={_pct(dn['portfolio_return']).strip()} "
              f"lower_bound={_pct(dn['lower_bound']).strip()} -> {dn['verdict']}"
              f"   | p25={_pct(dg['p25']).strip()} worst={_pct(dg['worst']).strip()} "
              f"below0={dg['n_below_zero']}/{dg['n']}")
    print("  NOTE: the pooled UPSIDE clause is deliberately NOT restated here -- pooling picks")
    print("  across windows is not the chartered per-anchor clause, and the pooled beat-rate")
    print("  already appears in the table above.")
    #  This one printed "the two clean 36mo windows" IMMEDIATELY UNDER a line correctly
    #  reading "POOLED across 3 clean window(s)" -- two adjacent lines contradicting each
    #  other, in the commit whose entire point was to stop exactly that.
    print(f"  CAVEAT: the {len(clause_inputs)} clean 36mo window(s) overlap heavily = ONE "
          "regime, and the pooled")
    print("          n is picks, not independent observations.")
    return {"bar": bar, "per_anchor": per_anchor_out, "pooled": pooled_out,
            "n_windows": len(clause_inputs), "basis": basis}


def _wrap(text, width=86, indent=" " * 12):
    """Fold a caveat onto continuation lines so it stays readable in a run log."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return ("\n" + indent).join(lines)


# --------------------------------------------------------------------------- #
#  Per-market-cap-band beat-rate (ADDITIVE; PIT USD mcap as-of buy)            #
# --------------------------------------------------------------------------- #
def _per_band_beat_rate(per_anchor, price_source, merged, horizon_m, threshold):
    """Pooled per-band beat-rate over the CLEAN windows. Returns (band_rows, pending).
    band_rows: [{band, depth_N, beat_rate, n, directional_only}]. Never raises on the
    banding-specific work -- degrades to a pending/empty read on missing inputs."""
    import numpy as np
    import returns_core as rc
    import depth_horizon_grid as dhg
    import carveOut as co

    merged_cdx = merged.get("cdx_df") if isinstance(merged, dict) else None
    _tdf = merged.get("Tickers_df") if isinstance(merged, dict) else None
    _cols = getattr(_tdf, "columns", [])
    band_names = (dict(zip(_tdf["symbol"], _tdf["name"]))
                  if _tdf is not None and "symbol" in _cols and "name" in _cols else {})

    # ---- POINT-IN-TIME FX (CEO, 2026-08-08) ---------------------------------------
    # The market cap here is already PIT (latest row <= buy).  The RATE was not: it was
    # whatever FX the process happened to hold, i.e. TODAY's spot applied to a 2021
    # market cap, which is a look-ahead-flavoured error -- a 2021 SEK name was banded by
    # a 2026 SEK/USD rate.  `fx_rates.load_pit_rates` reads the dated daily closes pulled
    # from v3/historical-price-full (1 call per currency per range, built OUT OF BAND by
    # `python fx_rates.py --historical --from ... --to ...`), and the conversion path then
    # converts EACH ROW AT ITS OWN DATE'S RATE.
    #
    # NOT A HARD REQUIREMENT, and the fallback is stated rather than silent: with no
    # historical file on disk this keeps the existing spot basis, because refusing would
    # turn every name unknown-mcap and quietly collapse the whole per-band read into
    # "everything is General" -- a worse failure than a labelled approximation in an
    # OFFLINE diagnostic.  Which basis was used is printed with the header, so a band
    # number can never be read without knowing how its FX was resolved.
    # The FX basis must NAME ITSELF ACCURATELY (F-4, reviewer 2026-08-08).  The fallback
    # string used to read "SPOT unset -- today's rate applied to a historical market cap"
    # on every run, which was actively misleading twice over: `baseline_tools/` never
    # installs a feed, so `unset` is the ONLY branch that executes here -- and `unset`
    # means the UNDATED FX_TO_USD CONSTANTS, not today's rate. A reader was being told the
    # wrong thing about the only path that runs. Each state now says what it actually is.
    # A PARTIAL PULL ALSO MUST NOT COLLAPSE SILENTLY: a table holding 3 of 38 currencies
    # loads without complaint and reads as point-in-time while making 35 unknown.
    fx_pit, fx_basis = None, None
    try:
        import fx_rates as fxr
        fx_pit = fxr.load_pit_rates()
        if fx_pit is not None:
            _cov_n, _cov_tot, _cov_f = fx_pit.coverage()
            _lo, _hi = fx_pit.span()
            fx_basis = ("POINT-IN-TIME (dated closes, %d/%d supported currencies, %.0f%%, "
                        "%s..%s)" % (_cov_n, _cov_tot, 100.0 * _cov_f,
                                     _lo.date() if _lo is not None else '?',
                                     _hi.date() if _hi is not None else '?'))
            if _cov_f < 1.0:
                _missing = sorted(set(fxr.supported_currencies()) - set(fx_pit.currencies))
                print("#   !!! PARTIAL PIT FX TABLE: %d of %d currencies missing -- names"
                      % (len(_missing), _cov_tot))
                print("#   !!! reporting in them have NO USD cap here and route to")
                print("#   !!! General. Re-run `python fx_rates.py --historical`.")
                print("#   !!! MISSING: %s" % ', '.join(_missing[:25]))
            #  ---- SILENT EXPIRY: the span is checked AGAINST THE ANCHORS, not just printed --
            #  The committed table is fixed at 2019-01-01..2026-08-07 and static since
            #  2026-08-08.  It covers every currently graded anchor, so nothing is wrong today
            #  and the flag is about TIME: each run drifts further from its end date.  Printing
            #  the span was never the guard -- nothing compared it to anything.  When an anchor
            #  finally lands past it, every row resolves to None, the cap goes NaN, the name
            #  routes to unknown-currency -> General, and the header keeps saying
            #  POINT-IN-TIME.  The look-ahead returns as an absence, which is quieter and no
            #  better.
            #
            #  LOUD, NOT FATAL, and the basis STRING carries it -- a banner scrolls past and a
            #  basis line does not.  Refusing the stage would take the anchors the table DOES
            #  cover down with the one it does not, which is the composition defect
            #  `benchmark_return_or_none` exists to stop one layer over.
            _graded = [b for w, b in dhg.BUY_ANCHORS if w in dhg.CLEAN_BUY_IDS]
            _eval = []
            for _b in _graded:
                _i = dhg.ANCHOR_IDX[_b] + horizon_m // 12
                if _i < len(dhg.ANCHORS):
                    _eval.append(dhg.ANCHORS[_i])
            _out = fx_pit.uncovered(sorted(set(_graded + _eval)))
            if _out:
                fx_basis += (" -- !!! EXPIRED FOR %d GRADED ANCHOR(S): %s"
                             % (len(_out), ', '.join(str(d) for d in _out)))
                print("#   " + "!" * 68)
                print("#   !!! PIT FX TABLE HAS EXPIRED FOR A GRADED ANCHOR.")
                print("#   !!! table span %s..%s (+%dd carry-forward); NOT covered: %s"
                      % (_lo.date() if _lo is not None else '?',
                         _hi.date() if _hi is not None else '?',
                         fxr.PIT_MAX_FORWARD_DAYS, ', '.join(str(d) for d in _out)))
                print("#   !!! Every non-USD name at those anchors resolves to NO USD market")
                print("#   !!! cap and routes to General -- the bands below are WRONG there,")
                print("#   !!! not merely approximate.  Refresh with")
                print("#   !!! `python fx_rates.py --historical --from 2019-01-01 --to <today>`")
                print("#   !!! (one paid call per major currency), or carry a newer")
                print("#   !!! FxRatesHistorical_*.csv onto this machine -- that is free.")
                print("#   " + "!" * 68)
    except Exception as _fe:
        print("#   (PIT FX unavailable: %s: %s)" % (type(_fe).__name__, _fe))
    if fx_basis is None:
        _state = co.fx_source_state()
        _what = {
            'unset': ("the UNDATED carveOut.FX_TO_USD CONSTANTS (no feed installed; this "
                      "is the normal state for baseline_tools/, which never installs one)"),
            'live': "this process's LIVE SPOT rates, applied to historical market caps",
            'failed': ("NOTHING -- the FX feed failed, so every name is unknown-currency "
                       "and every band read below is empty/General"),
        }.get(_state, _state)
        #  THE OLD SENTENCE ADVISED A 35-CALL PAID FETCH AS THE ONLY REMEDY, AND IT IS
        #  USUALLY THE WRONG ONE.  `fx_rates.py --historical` costs one call per MAJOR
        #  supported currency -- 35 of the 38 (GBX/GBp/USD are derived free) -- and on this
        #  house's machines the table it would rebuild ALREADY EXISTS:
        #  `output/FxRatesHistorical_2019-01-01_2026-08-08.csv`, 38/38 currencies, spanning
        #  2019-01-01..2026-08-07, which covers every currently-graded buy anchor.
        #  It does not reach the run machine because `output/` is gitignored, and the FX table
        #  is a large data artifact -- so it travels out-of-band like the pickles, and nobody
        #  ever carried it.  The look-ahead has therefore been costing a false basis for want
        #  of a 2.4MB file copy, while the message on screen asked for money.
        #  BOTH remedies are named now, cheapest first, because a run log that offers only the
        #  expensive one is how the expensive one gets chosen.
        fx_basis = (
            "NOT point-in-time [%s] -- %s.\n"
            "#            TO REMOVE THE LOOK-AHEAD, IN ORDER OF COST:\n"
            "#            (1) FREE -- copy an existing `FxRatesHistorical_*.csv` into this\n"
            "#                machine's repo root or `output/`.  It is gitignored, so it does\n"
            "#                NOT arrive with the code; it must be carried like the pickles.\n"
            "#            (2) PAID -- `python fx_rates.py --historical` rebuilds it at ONE\n"
            "#                API CALL PER MAJOR CURRENCY (35 of the 38 supported).  Only\n"
            "#                needed if no table exists anywhere, or the span misses an\n"
            "#                anchor you intend to grade."
            % (_state, _what))

    band_pending = (merged_cdx is None) or (
        not co.currency_data_present(merged_cdx, fx=fx_pit))
    DIRECTIONAL_MAX_USD = 150e6   # a band whose TOP is <= $150M is directional-only

    print("\n" + "#" * 72)
    print("# PER-BAND BEAT-RATE  --  general ranking GROUPED by USD market cap")
    print("#   General -> top-20 ; Mid/Small/Micro -> top-5 ; PIT mcap as-of buy")
    print("#   FX basis: %s" % fx_basis)
    if band_pending:
        print("#   !!! CURRENCY DATA PENDING (reportedCurrency not yet in this data):   !!!")
        print("#   !!! bands NOT meaningful -> every name reads as General; sub-bands    !!!")
        print("#   !!! empty. Corrects automatically from the next full fetch.          !!!")
    print("#" * 72)

    band_pooled = {lab: [] for lab, *_ in co.MCAP_BANDS}
    for wid, buy in dhg.BUY_ANCHORS:
        if wid not in per_anchor or wid not in dhg.CLEAN_BUY_IDS:
            continue
        buy_idx = dhg.ANCHOR_IDX[buy]
        eval_idx = buy_idx + horizon_m // 12
        if eval_idx >= len(dhg.ANCHORS):
            continue
        ev = dhg.ANCHORS[eval_idx]
        bench = _bench_or_skip(price_source, buy, ev, wid, rc, "the per-band split")
        if bench is None:
            continue
        ranking = per_anchor[wid].get("ranking") or []
        # FULL deduped general ranking (deep enough to fill the sub-bands), then band it.
        if merged_cdx is not None:
            deduped, _drp = co.dedup_ranked(ranking, merged_cdx, band_names)
        else:
            deduped = list(ranking)
        pit_mcu = (co.marketcap_usd_by_source(merged_cdx, as_of=buy, fx=fx_pit)
                   if merged_cdx is not None else {})
        band_seq = {lab: [] for lab, *_ in co.MCAP_BANDS}
        for s in deduped:
            lab = co.band_for_marketcap_usd(pit_mcu.get(s))
            if lab is None:                            # unknown mcap -> General
                lab = co.MCAP_BANDS[0][0]
            band_seq[lab].append(s)
        for label, lo, hi, N in co.MCAP_BANDS:
            top = band_seq[label][:N]
            if not top:
                continue
            rdf = rc.compute_returns(top, buy, ev, price_source)
            for _, r in rc.included(rdf).iterrows():
                if r["terminal_flag"]:
                    band_pooled[label].append(False)   # missing eval = NOT beating
                else:
                    band_pooled[label].append((r["total_return"] - bench) >= threshold)

    print("  --- POOLED across clean windows (per band) ---")
    band_rows = []
    for label, lo, hi, N in co.MCAP_BANDS:
        flags = band_pooled[label]
        rate = float(np.mean(flags)) if flags else float("nan")
        directional = hi <= DIRECTIONAL_MAX_USD
        tag = "DIRECTIONAL-ONLY" if directional else "pass/fail"
        rs = f"{rate*100:.1f}%" if rate == rate else "n/a"
        note = "  [PENDING CURRENCY]" if band_pending and label != co.MCAP_BANDS[0][0] else ""
        print(f"  {label:16} depth<={N:<3} beat_rate={rs:>7} (n={len(flags):>3})  {tag}{note}")
        band_rows.append({"band": label, "depth_N": N, "beat_rate": rate,
                          "n": len(flags), "directional_only": directional})
    print("  CAVEAT: same 2 overlapping windows (one regime); missing-eval = NOT beating.")
    print("          Small bands (<$150M) are thin -> DIRECTIONAL-ONLY, not a 60% pass/fail.")
    return band_rows, band_pending


# --------------------------------------------------------------------------- #
#  Stage: ESTIMATION sub-block (heavy; OFF unless run_estimation == 1)         #
# --------------------------------------------------------------------------- #
def run_estimation_block(dmdic, merged, registry, price_source, configdic, log):
    """The HEAVY parameter-SEARCH block: the tuner / tune_run / rebalance_engine weight &
    cohort sweeps plus the depth-grid weight/carve tuning sweeps.  Runs ONLY when
    configdic['run_estimation'] == 1 (default 0 -> this returns immediately).  Guarded
    by the caller like every other stage."""
    if int(configdic.get("run_estimation", 0) or 0) != 1:
        log("[estimation] run_estimation != 1 -> SKIPPED (default). "
            "Grading/IC/grid/beat-rate/oracle/random already ran above.")
        return None
    log("[estimation] run_estimation == 1 -> running heavy tuning sweeps ...")
    import depth_horizon_grid as dhg

    # (a) depth-grid tuning sweeps: equal-weight and carve-on variants of the grid.
    inputs = dhg.inputs_from_memory(dmdic, merged, registry, log)
    for weights, carve in (("equal", "off"), ("default", "on")):
        log(f"[estimation] depth-grid sweep weights={weights} carve={carve} ...")
        per = dhg.rank_all_anchors(inputs, log, weights=weights, carve=carve)
        cells, pooled, pooled_clean = dhg.compute_grid(per, price_source)
        print("\n" + "#" * 72)
        print(f"# ESTIMATION depth-grid sweep  (weights={weights}, carve={carve})")
        print("#" * 72)
        print(dhg.build_report(per, cells, pooled, pooled_clean), flush=True)

    # (b) weight/cohort tuner sweep (tune_run) -- the expensive Hooke-Jeeves search.
    # Import inside so an absent/rotted tuner degrades loud-but-safe (this whole block
    # is guarded by the caller).  tune_run.main() drives its own default local paths.
    try:
        import tune_run as tr
        log("[estimation] tune_run: launching weight/cohort search (SLOW) ...")
        tr.main()
    except SystemExit:
        pass  # tune_run.main() may sys.exit on missing local inputs; that's fine here
    return True


# --------------------------------------------------------------------------- #
#  Public entry point (called from Sbocker.main, POST-PICK / PRE-INGESTION)   #
# --------------------------------------------------------------------------- #
def run_analysis_suite(resdic, configdic):
    """POST-PICK analysis suite.  Called from Sbocker AFTER the pick-log stage and
    BEFORE delisted ingestion.  Each analysis is a SEPARATELY-guarded stage; a failure
    in one banners loudly and never skips the others or crashes the run.

    Reads a SHALLOW COPY of resdic and never writes into resdic keys the pick path or
    ingestion reads.  Never prints any api_key."""
    log = lambda *a: print("[analysis]", *a, file=sys.stderr, flush=True)
    # sys.path so the baseline_tools modules import cleanly regardless of CWD.
    for p in (_REPO, _HERE):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Shallow copy so top-level key rebinding can never touch the live resdic; stages
    # that mutate frames do so on their own .copy() (build_panel / merge_dead build new
    # frames), so the post-pick invariant holds.
    dmdic = dict(resdic)

    as_of = configdic.get("as_of") or datetime.today().strftime("%Y-%m-%d")
    print("\n" + "=" * 78)
    print("POST-PICK ANALYSIS SUITE  (strictly additive, guarded, post-pick)")
    print("=" * 78)
    print("WHAT THIS GRADES: the HISTORICAL point-in-time backtest RE-RUN AGAINST")
    print("TONIGHT'S MODEL -- the same fundamentals/scoring the live pick used,")
    print(f"reproduced as-of the historical buy anchors ({_clean_window_text_guarded()} =")
    print("the CLEAN 36-month windows), on REAL adjusted-close prices.")
    print("It does NOT grade tonight's live picks: no forward price exists yet (the")
    print("pick-log accrues those; grading them is the deferred separate piece).")
    print(f"run as_of={as_of}   run_estimation={configdic.get('run_estimation', 0)}")
    print("=" * 78, flush=True)

    # ---- Stage 1: prices (guarded) ----
    _run_stage("price-fetch (bulk-by-date, no-op if present)",
               run_price_fetch_stage, resdic, configdic, log)
    #  Its OWN stage, deliberately: the audit must run whether or not the fetch stage did
    #  anything, and a failure inside the audit must not be mistaken for a fetch failure.
    _run_stage("price-grid staleness audit (report only, NO fetch)",
               _audit_price_grid_stage, resdic, configdic, log)
    #  SAME PLACE, DIFFERENT QUESTION.  The staleness audit asks "is this the right grid";
    #  this asks "is any price on it off by a power of ten", which staleness cannot see and a
    #  level screen structurally cannot see either (16.9% of grid symbols are already under
    #  $1).  Report-only, like its neighbour, and for a sharper reason: its evidence touches
    #  names that are LIVE in scoring, and correcting a price would change a score.
    #  CAPTURED, not discarded: the PIT pass below diffs its ALARM set against this one to
    #  name the contaminated rows that exist ONLY in the backtest population.
    price_scale_live = _run_stage("vendor price-scale audit (report only, NO correction)",
                                  _audit_price_scale_stage, resdic, configdic, log)
    #  Referee BEFORE the price source is built: it is about whether the real grid can be
    #  trusted, so its finding belongs above every number that rides on it.
    _run_stage("level-break referee (report only, NO override)",
               run_level_break_referee_stage, resdic, configdic, log)
    price_source = _run_stage("build-price-source", _build_price_source, log, configdic)

    # ---- Stage 2: model-vs-metric (dmdic only; independent of prices/PIT) ----
    def _mvm():
        import model_vs_metric as mvm
        return mvm.run_in_pipeline(dmdic, price_source=price_source, log=log)
    _run_stage("model-vs-metric (persistence/variance/correlation/IC)", _mvm)

    # ---- Stage 3: real-IC + decomposition (needs real prices) ----
    def _ric():
        import real_ic as ric
        return ric.run_in_pipeline(dmdic, price_source=price_source,
                                   real_prices_csv=_PRICES_CSV, log=log)
    _run_stage("real-IC + profit-timing decomposition", _ric)

    # ---- shared PIT inputs (merged/registry) built ONCE for the PIT stages ----
    pit = _run_stage("build-PIT-inputs (dead-merge, ONCE)",
                     _build_pit_inputs, dmdic, configdic, log)
    merged, registry, clean = (pit if pit else (dmdic, None, False))

    # ---- PIT universe exchange scope, resolved ONCE and shared by the two PIT stages.
    #  DEFAULT IS UNCHANGED AND MUST STAY THAT WAY: absent key -> None -> NA1
    #  (NYSE/NASDAQ/TSX), which is what every grid and skill number printed so far was
    #  computed on.  `configdic['pit_exchange_filter'] = 'all'` opens the backtest to the
    #  universe the deployed filter actually scores (KOSPI/KOSDAQ/LSE/XETRA/PAR/STO/OSL/BRU).
    #
    #  ============================================================================
    #  DO NOT SWITCH THIS ON WITHOUT THE PRICE REFETCH.  Three costs, biggest first;
    #  all measured 2026-08-22 against the run machine's grid + the 08-22 CUR6K panel.
    #  ============================================================================
    #
    #  1. no_buy COLLAPSE -- THE REASON THE KNOB STAYS OFF, and the one an earlier version of
    #     this comment understated by an order of magnitude.  The refetch is deferred, so the
    #     Korean/Oslo/Paris names entering the universe have NO price at any anchor.  They
    #     still COMPETE for top-20 slots and then contribute nothing, so the shipped top-20
    #     loses 12 to 19 of its 20 names to `no_buy` (17 of 20 at buy2024 -- a headline
    #     resting on THREE names), and top-20 overlap with the NA1 pick set falls to 4-12 of
    #     20.  The number that comes out is not a wider measurement of the same thing; it is
    #     a three-name average wearing a top-20 label.
    #
    #  2. INTERIOR PRICE HOLES DO REACH THE -1.0 FLOOR, and the earlier claim that the floor
    #     was unreachable was wrong -- it was true only of the WHOLLY-unpriced venues.  A name
    #     priced at the buy anchor and missing at the eval anchor gets status='terminal':
    #     `total_return_floor` puts it at -100% and the default beat-rate policy
    #     (missing='fail') scores it a miss.  Roughly 200 names on the widened scope are in
    #     that state, 164 of them `.L`, and 18 of the 29 `.L` cases at 2021->2024 are priced
    #     again at 2025-12-31 -- provably alive, scored as total losses.  At PICK level the
    #     bite is small (3 picks at depth 100, 1 at depth 50, ZERO at depth <= 20, i.e. about
    #     -2pp on one cell) and at the NA1 default it is ZERO at every anchor, horizon and
    #     depth.  Small, but not nothing, and not "unreachable".
    #
    #  3. COST.  The PIT reproduction goes from ~1,767 to ~4,954 scored live names per anchor.
    #     Not measured -- it needs a pipeline run.
    #
    #  What the widened scope is GOOD for: seeing which names the filter would rank if the
    #  grid could price them.  That is a diagnostic, not a grading run.
    import dead_merge as _dm_scope
    pit_exch = _dm_scope.resolve_exchange_filter(configdic.get("pit_exchange_filter"))
    log("[universe] PIT exchange scope = "
        + ("NA1 (NYSE/NASDAQ/TSX) -- default" if pit_exch is None else str(pit_exch)))

    # ---- Stage 4: depth x horizon grid (returns per_anchor for reuse) ----
    def _grid():
        if price_source is None or registry is None:
            raise RuntimeError("grid stage skipped: price_source or PIT inputs missing")
        import depth_horizon_grid as dhg
        # carve="on" => the DEPLOYED universe (carve partition ON, as shipped).  The grid
        # ranks that universe; the deduped shipped top-20 rides along in per_anchor for
        # the beat-rate stage.  (A carve-off view is available via skill_baseline /
        # -run_estimation; not re-run here to avoid a second ~6min PIT reproduction.)
        return dhg.run_in_pipeline(dmdic, merged, registry, price_source, log=log,
                                   carve="on", exchange_filter=pit_exch)
    grid_out = _run_stage("depth x horizon avg-TR grid (DEPLOYED, carve-ON)", _grid)
    per_anchor = grid_out[1] if grid_out else None

    #  ---- Stage 4b: the price-scale audit AGAIN, over the population the grid just ranked.
    #  HERE and not beside its live twin, for one reason: `merged` and `per_anchor` do not
    #  exist yet up there.  The live pass reads a survivor-only panel, so it cannot see a
    #  delisted name at all -- which is every name the backtest adds, including ATRI, the case
    #  the whole stage was built for.  Running check A a second time over the dead-merged
    #  panel is what makes the audit cover the backtest inputs rather than only the shipped
    #  list.
    _run_stage("vendor price-scale audit -- PIT dead-merged panel (report only)",
               _audit_price_scale_pit_stage, merged, per_anchor, price_scale_live,
               clean, log)

    # ---- Stage 5: beat-rate vs URTH (operational-target proxy; reuses per_anchor) ----
    def _beat():
        if per_anchor is None or price_source is None:
            raise RuntimeError("beat-rate stage skipped: per_anchor/price_source missing")
        return beat_rate_vs_urth(per_anchor, price_source, log, merged=merged)
    _run_stage("beat-rate vs URTH (DEPLOYED filter: deduped, carve-ON)", _beat)

    # ---- Stage 5b: what the Stage-1 solvency GATE actually did -----------------
    #  THE ONE READOUT THAT MAKES THE VETO JUDGEABLE.  Since 2026-08-27 the backtest applies
    #  the gate, and it ejects roughly half the scored pool -- but nothing said whether a
    #  single ejection helped.  A gate whose only visible output is its own ejection count can
    #  be tightened or loosened with no measurable consequence, which is the state the CEO is
    #  in while his next decision is the weights and the gates.
    #
    #  COST, MEASURED: one extra PIT reproduction of the CLEAN anchors only -- two of the
    #  seven, because the rest have no 36-month eval leg and so nothing to attribute over.  A
    #  full seven-anchor pass is ~205s on this panel, so this is ~60s.  The vetoed rankings are
    #  REUSED from the grid stage rather than recomputed.
    def _gate_attr():
        if per_anchor is None or price_source is None or registry is None:
            raise RuntimeError("gate-attribution skipped: per_anchor/price_source/PIT inputs "
                               "missing")
        import gate_attribution as ga
        return ga.run_in_pipeline(dmdic, merged, registry, price_source, per_anchor,
                                  log=log, exchange_filter=pit_exch, carve="on")
    _run_stage("stage-1 gate attribution (vetoed vs un-vetoed counterfactual)", _gate_attr)

    # ---- Stage 6: oracle-best-N + random baseline + decomposition ladder ----
    def _skill():
        if price_source is None or registry is None:
            raise RuntimeError("skill-baseline skipped: price_source/PIT inputs missing")
        import skill_baseline as sb
        #  `stage1_veto=True` EXPLICITLY, even though it is now the module default.  This
        #  is the call site whose silence produced the defect: the grid stage two stages up
        #  passes the veto and stamps VETOED, this one passed nothing and inherited
        #  `reproduce_pit_top`'s OFF, so 25.0% (VETOED) and 25.9% (UN-VETOED) were printed
        #  as a matched pair.  Stating it here means a future change to either default
        #  cannot re-open the gap without someone editing a line that says what it does.
        res = sb.run_skill_baseline(dmdic, merged, registry, price_source,
                                    cadence_months=36, pick_n=20, oracle_ns=(3, 20),
                                    n_draws=1000, seed=0, stage1_veto=True,
                                    exchange_filter=pit_exch, log=log)
        print("\n" + "#" * 72)
        print("# SKILL BASELINE  (oracle-best-N ceiling + random floor + ladder)")
        print("#" * 72)
        print(sb.format_report(res), flush=True)
        return res
    _run_stage("skill-baseline (oracle/random)", _skill)

    # ---- Stage 7: ESTIMATION sub-block (heavy; OFF unless run_estimation==1) ----
    _run_stage("estimation sub-block (tuner/sweeps; run_estimation-gated)",
               run_estimation_block, dmdic, merged, registry, price_source,
               configdic, log)

    print("\n" + "=" * 78)
    print("POST-PICK ANALYSIS SUITE COMPLETE (each stage guarded; picks unaffected)")
    print("=" * 78, flush=True)
