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


def _fetch_bulk_scrubbed(baseurl, api_key, anchors, symbols_filter, out_path, log,
                         max_lookback=4):
    """BULK-BY-DATE fetch that CANNOT leak the api_key.  One call per anchor date (whole
    universe per call; NEVER per-symbol), stepping back up to `max_lookback` days on an
    empty/holiday response.  The HTTP call goes through delisted_ingest.safe_get_bulk_csv,
    which strips apikey from the URL and from any exception/warning text.  Reuses ONLY
    fetch_prices' KEY-FREE pure helpers (row parsing); fetch_prices.py is NOT modified.
    Writes the same schema fetch_prices produced.  Returns (calls, rows_written)."""
    import delisted_ingest as di
    import fetch_prices as fp

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    calls = written = 0
    with open(out_path, "w", newline="") as fout:
        w = _csv.writer(fout)
        w.writerow(["date_requested", "date_actual", "symbol", "adjClose"])
        for anchor in anchors:
            got = False
            for back in range(max_lookback + 1):
                d = anchor - timedelta(days=back)
                ds = d.strftime("%Y-%m-%d")
                url = (f"{baseurl}v4/batch-request-end-of-day-prices"
                       f"?date={ds}&apikey={api_key}")
                calls += 1
                log(f"[price-fetch] bulk call {calls}: date={ds} "
                    f"(anchor {anchor.isoformat()})")
                rows = di.safe_get_bulk_csv(url)  # key-scrubbed on ANY error/warning
                if rows:
                    for row in rows:
                        sym, adj = fp._extract(row)
                        if not sym or adj in (None, "", "null"):
                            continue
                        if symbols_filter and sym not in symbols_filter:
                            continue
                        w.writerow([anchor.strftime("%Y-%m-%d"), ds, sym, adj])
                        written += 1
                    got = True
                    log(f"[price-fetch]   OK: {len(rows)} rows for {ds}")
                    break
            if not got:
                log(f"[price-fetch]   WARNING: no data for anchor "
                    f"{anchor.isoformat()} within {max_lookback} lookback days")
    return calls, written


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


# --------------------------------------------------------------------------- #
#  Stage 1: guarded BULK-BY-DATE price fetch (no-op/top-up when present)       #
# --------------------------------------------------------------------------- #
def run_price_fetch_stage(resdic, configdic, log):
    """Ensure the grading price grid (real_prices.csv + real_prices_2025.csv) exists.

    NO-OP/top-up when the files are already present (the common case -- prices are
    checked into neither git nor moved out-of-band, but exist on the run machine).  When
    ABSENT, fetch them BULK-BY-DATE ONLY (one call per year-end anchor, whole universe
    per call) via fetch_prices.run_bulk, so the downstream analysis is machine-
    independent.  The api_key is read from configdic (fallback fmpAPIkey.txt) and NEVER
    printed (masked).  Returns dict(main=..., supp=...) of resolved paths (or None)."""
    import fetch_prices as fp  # KEY-FREE pure helpers only (build_anchor_dates, ...)

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
                calls, written = _fetch_bulk_scrubbed(baseurl, api_key, anchors, syms,
                                                      _PRICES_CSV, log)
                log(f"[price-fetch] MAIN done: {calls} calls, {written} rows -> "
                    f"{os.path.basename(_PRICES_CSV)}")
            if need_supp:
                d2025 = fp.nearest_weekday_on_or_before(datetime(2025, 12, 31).date())
                log(f"[price-fetch] SUPP 2025 anchor absent -> bulk-by-date fetch 1 "
                    f"date ({d2025.isoformat()})")
                calls, written = _fetch_bulk_scrubbed(baseurl, api_key, [d2025], syms,
                                                      _PRICES_2025_CSV, log)
                log(f"[price-fetch] SUPP done: {calls} calls, {written} rows -> "
                    f"{os.path.basename(_PRICES_2025_CSV)}")
        except Exception as e:
            # Re-raise with a SCRUBBED message so the guard banner (printed to the REAL
            # streams outside this context) can never carry the key.
            raise RuntimeError(_scrub(f"price fetch failed: {e}", api_key)) from None
    return {"main": _PRICES_CSV if os.path.exists(_PRICES_CSV) else None,
            "supp": _PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None}


def _build_price_source(log):
    import returns_core as rc
    supp = _PRICES_2025_CSV if os.path.exists(_PRICES_2025_CSV) else None
    if not os.path.exists(_PRICES_CSV):
        raise RuntimeError(f"real price grid absent: {_PRICES_CSV} "
                           "(price-fetch stage did not produce it)")
    ps = rc.PriceSource(_PRICES_CSV, supp_csv=supp)
    log(f"[price-source] PriceSource built from {os.path.basename(_PRICES_CSV)}"
        + (f" + {os.path.basename(_PRICES_2025_CSV)}" if supp else " (no 2025 supp)"))
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
                      horizon_m=36, threshold=0.10):
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
    return {"per_window": rows, "pooled": pooled}


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
    price_source = _run_stage("build-price-source", _build_price_source, log)

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
                                   carve="on")
    grid_out = _run_stage("depth x horizon avg-TR grid (DEPLOYED, carve-ON)", _grid)
    per_anchor = grid_out[1] if grid_out else None

    # ---- Stage 5: beat-rate vs URTH (operational-target proxy; reuses per_anchor) ----
    def _beat():
        if per_anchor is None or price_source is None:
            raise RuntimeError("beat-rate stage skipped: per_anchor/price_source missing")
        return beat_rate_vs_urth(per_anchor, price_source, log)
    _run_stage("beat-rate vs URTH (DEPLOYED filter: deduped, carve-ON)", _beat)

    # ---- Stage 6: oracle-best-N + random baseline + decomposition ladder ----
    def _skill():
        if price_source is None or registry is None:
            raise RuntimeError("skill-baseline skipped: price_source/PIT inputs missing")
        import skill_baseline as sb
        res = sb.run_skill_baseline(dmdic, merged, registry, price_source,
                                    cadence_months=36, pick_n=20, oracle_ns=(3, 20),
                                    n_draws=1000, seed=0, log=log)
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
