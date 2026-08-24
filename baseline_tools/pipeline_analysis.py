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
                         max_lookback=4, reference_paths=None):
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

    ONE PROPERTY WORTH KNOWING, because it makes a bad reference safe: the test only fires
    when the NEW body has FEWER rows for a venue than the reference.  A reference that is
    itself truncated therefore makes this test WEAKER, never spuriously stricter -- so
    pointing it at the very file about to be overwritten (which may be the corrupted one)
    cannot reject a good body.  It degrades toward silence, which is the correct direction
    for a guard that can drop an anchor.
    """
    import delisted_ingest as di
    import fetch_prices as fp

    ref, ref_path = _venue_reference(reference_paths or [], log)
    ref_label = os.path.basename(str(ref_path)) if ref_path else "no-reference"
    calls = written = 0
    accepted = []
    venue_findings = []

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
            log(f"[price-fetch]   OK: {n_payload} rows for {ds} ({len(kept)} kept)")
            break
        if not got:
            log(f"[price-fetch]   WARNING: no usable body for anchor "
                f"{anchor.isoformat()} within {max_lookback} lookback days")

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

    THE DOCSTRING USED TO SAY "via fetch_prices.run_bulk" AND THAT WAS FALSE.  `run_bulk`
    is called from nowhere in this module; `_fetch_bulk_scrubbed` is the fetch, and it
    exists because it cannot leak the api_key.  The claim mattered: a payload floor and a
    weekend guard were added to `run_bulk` and, on the strength of that sentence, believed
    to be protecting this stage.  They were not.  Both now run here, from the same
    definitions in `fetch_prices`, and `test_fetch_prices` pins the wiring so the sentence
    cannot go stale again.
    """
    import fetch_prices as fp  # KEY-FREE pure helpers only (build_anchor_dates, ...)

    #  PRESENCE, NOT FRESHNESS -- and that is now a stated decision rather than an oversight.
    #  A present-but-stale grid is NOT re-fetched here: the refresh costs API calls and is a
    #  deliberate human action.  Whether the grid still fits tonight's universe is answered by
    #  the SEPARATE audit stage (`_audit_price_grid_stage`), which reports and never spends.
    need_main = not os.path.exists(_PRICES_CSV)
    need_supp = not os.path.exists(_PRICES_2025_CSV)
    if not need_main and not need_supp:
        log(f"[price-fetch] both price files present -- NO fetch "
            f"({os.path.basename(_PRICES_CSV)}, {os.path.basename(_PRICES_2025_CSV)}).")
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

    # Keep the written file small: local symbol allow-list from tonight's universe
    # (the bulk call still returns the whole universe; this only filters what we save).
    syms = set()
    try:
        syms |= set(resdic["Tickers_df"]["symbol"].dropna().astype(str))
        syms |= set(resdic["cdx_df"]["source"].dropna().astype(str))
    except Exception:
        syms = None  # no filter -> save everything (still one call per date)
    # The benchmark ETF (URTH) is NOT a name in tonight's stock universe, so the
    # allow-list above would DROP it -- yet the bulk EOD dump DOES return it and the
    # downstream benchmark stages (beat-rate vs URTH, depth-grid, skill_baseline) fail
    # hard without it.  Force-keep the benchmark symbol so PriceSource.benchmark_series
    # resolves on every fresh fetch.  (No-op when syms is None: everything is saved.)
    if syms is not None:
        import returns_core as rc
        syms.add(rc.BENCHMARK_SYMBOL)

    # D1 boundary mask: run the ENTIRE fetch with BOTH stdout+stderr scrubbed, and mask
    # any exception message, so the key cannot surface even on a network/HTTP error path
    # (belt-and-suspenders on top of safe_get_bulk_csv's own scrubbing).
    with contextlib.redirect_stdout(_ScrubStream(sys.stdout, api_key)), \
         contextlib.redirect_stderr(_ScrubStream(sys.stderr, api_key)):
        try:
            if need_main:
                anchors = fp.build_anchor_dates(_MAIN_PRICE_YEARS, hold_months=12)
                log(f"[price-fetch] MAIN grid absent -> bulk-by-date fetch, "
                    f"{len(anchors)} anchor dates (~{len(anchors)} calls): "
                    f"{[a.isoformat() for a in anchors]}")
                calls, written, refused, vf = _fetch_bulk_scrubbed(
                    baseurl, api_key, anchors, syms, _PRICES_CSV, log,
                    reference_paths=_reference_paths(configdic, _PRICES_CSV))
                log(f"[price-fetch] MAIN done: {calls} calls, {written} rows -> "
                    f"{os.path.basename(_PRICES_CSV)}")
                _report_fetch_refusals("MAIN", refused, vf, log)
            if need_supp:
                d2025 = fp.nearest_weekday_on_or_before(datetime(2025, 12, 31).date())
                log(f"[price-fetch] SUPP 2025 anchor absent -> bulk-by-date fetch 1 "
                    f"date ({d2025.isoformat()})")
                #  ONE anchor, so the relative-median floor is vacuous by construction and
                #  only the absolute floor and the venue test can fire here.  That is exactly
                #  the hole the absolute backstop was put in for.
                calls, written, refused, vf = _fetch_bulk_scrubbed(
                    baseurl, api_key, [d2025], syms, _PRICES_2025_CSV, log,
                    reference_paths=_reference_paths(configdic, _PRICES_2025_CSV))
                log(f"[price-fetch] SUPP done: {calls} calls, {written} rows -> "
                    f"{os.path.basename(_PRICES_2025_CSV)}")
                _report_fetch_refusals("SUPP", refused, vf, log)
        except Exception as e:
            # Re-raise with a SCRUBBED message so the guard banner (printed to the REAL
            # streams outside this context) can never carry the key.
            raise RuntimeError(_scrub(f"price fetch failed: {e}", api_key)) from None
    return {"main": _PRICES_CSV if os.path.exists(_PRICES_CSV) else None,
            "supp": _PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None}


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
    if route == "real":
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
                                    panel=panel)
    except (FileNotFoundError, KeyError) as e:
        #  NOTE THE ORDERING LIMITATION, stated rather than hidden: the price-grid audit
        #  stage runs BEFORE this one and has already printed a banner naming the CONFIGURED
        #  route, so on a fallback the banner and reality disagree for one run.  This line is
        #  the correction, and it is why it says "every number below" explicitly.
        log(f"[price-source] route={route} UNAVAILABLE ({type(e).__name__}: {e}) -- "
            f"FALLING BACK to route=real.  Every number below is on the real grid, and the "
            f"price-grid audit banner above named {route!r} -- THIS line is the correct one.")
        return rc.PriceSource(_PRICES_CSV, supp_csv=supp)
    log(f"[price-source] route={route}  (real grid "
        f"{os.path.basename(_PRICES_CSV)} + panel {os.path.basename(str(panel))})")
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
    merge_as_of = _CLEAN_36MO_WINDOWS[0][0]  # earliest clean buy anchor
    log(f"[pit-inputs] dead-merge as-of {merge_as_of} (ONCE; shared across stages) ...")
    merged, stats = dm.merge_dead_into_dmdic(dmdic, dead, registry, as_of=merge_as_of)
    log(f"[pit-inputs] merge: universe={stats.get('universe_size')} "
        f"built_dead={stats.get('built')} gate_fail={stats.get('gate_fail')}")
    return merged, registry, True


# --------------------------------------------------------------------------- #
#  Stage: beat-rate vs URTH (operational-target proxy) -- reuses per_anchor    #
# --------------------------------------------------------------------------- #
def beat_rate_vs_urth(per_anchor, price_source, log, depths=(10, 20),
                      horizon_m=36, threshold=0.10, merged=None):
    """The operational-target readout on the DEPLOYED FILTER: share of the shipped
    top-N that beat URTH (MSCI World TR proxy) by >= threshold over a `horizon_m` hold,
    on the CLEAN buy anchors.

    Uses the ISSUER-DEDUPED, CARVE-ON top-20 (`per_anchor[wid]["top20_deduped"]`) --
    exactly the general list the pipeline ships (carve partition + issuer-dedup, both
    default ON) -- NOT the raw undeduped pool.  This is the same deduped-top20 basis
    skill_baseline's filter uses, so the numbers are on a comparable footing (skill's
    is carve-OFF; the only intended difference is the carve).  Pure returns_core URTH
    path (rc.beat_rate + rc.benchmark_return, require_exact=True to match skill_baseline
    and fail loudly on a missing benchmark anchor).  Prints a report and returns the
    per-window + pooled beat-rate rows."""
    import numpy as np
    import returns_core as rc
    import depth_horizon_grid as dhg

    print("\n" + "#" * 72)
    print("# BEAT-RATE vs URTH  --  DEPLOYED FILTER (issuer-deduped, carve-ON top-20)")
    print(f"#   the shipped general list beats MSCI World by >= {threshold*100:.0f}pp?")
    print(f"#   horizon = {horizon_m}mo   benchmark = {rc.BENCHMARK_VARIANT}")
    print("#   CLEAN 36mo windows only (buy2021->2024, buy2022->2025).")
    print("#   (skill_baseline reports the same deduped-top20 basis carve-OFF; the")
    print("#    intended difference between the two is the carve.)")
    print("#" * 72)

    rows = []
    pooled_flags = {N: [] for N in depths}
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
        bench = rc.benchmark_return(price_source, buy, ev, require_exact=True)
        for N in depths:
            top = deployed[:N]
            rdf = rc.compute_returns(top, buy, ev, price_source)
            br, n = rc.beat_rate(rdf, bench, threshold=threshold, missing="fail")
            rows.append({"window": f"{buy}->{ev}", "depth_N": N,
                         "beat_rate": br, "n": n, "bench_ret": bench})
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
    print("  CAVEAT: 2 heavily-overlapping windows = ONE regime; count-based (magnitude-")
    print("          blind); missing-eval counts as NOT beating (missing='fail').")

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
            "bands": band_rows, "band_pending": band_pending}


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
        fx_basis = ("NOT point-in-time [%s] -- %s. Run `python fx_rates.py --historical` "
                    "to remove the look-ahead." % (_state, _what))

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
        bench = rc.benchmark_return(price_source, buy, ev, require_exact=True)
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
    print("reproduced as-of the historical buy anchors (buy2021->2024, buy2022->2025 =")
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

    # ---- Stage 5: beat-rate vs URTH (operational-target proxy; reuses per_anchor) ----
    def _beat():
        if per_anchor is None or price_source is None:
            raise RuntimeError("beat-rate stage skipped: per_anchor/price_source missing")
        return beat_rate_vs_urth(per_anchor, price_source, log, merged=merged)
    _run_stage("beat-rate vs URTH (DEPLOYED filter: deduped, carve-ON)", _beat)

    # ---- Stage 6: oracle-best-N + random baseline + decomposition ladder ----
    def _skill():
        if price_source is None or registry is None:
            raise RuntimeError("skill-baseline skipped: price_source/PIT inputs missing")
        import skill_baseline as sb
        res = sb.run_skill_baseline(dmdic, merged, registry, price_source,
                                    cadence_months=36, pick_n=20, oracle_ns=(3, 20),
                                    n_draws=1000, seed=0,
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
