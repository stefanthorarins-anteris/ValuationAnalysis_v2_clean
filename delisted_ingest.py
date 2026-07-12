"""
Delisted-entity (survivorship) ingestion  (investment-filter restructure, design
s4/s5/s9; fetch mechanics = delisted-ingestion-spec-2026-07-12).

WHAT THIS DOES
--------------
ACQUIRES survivorship data and STORES it for later point-in-time analysis:
  1. full paginated delisted-companies REGISTRY (symbol/companyName/exchange/
     ipoDate/delistedDate), None-trap-safe (s1);
  2. entity_id assignment over registry U live occupants, BOTH merge + split
     modes (F4 death-rate band);
  3. N_dead checkpoint (projected added wall-clock) BEFORE the long loop;
  4. two ride-along verification calls (split-adjustment basis; filing-date
     availability) with explicit PASS/FAIL;
  5. dead-entity BULK prices (bulk-by-date ONLY, never per-symbol) with a BBBY
     presence probe + present/absent two-outcome branch, and death-signature
     drawdown classification;
  6. dead-entity FUNDAMENTALS (recent-era-first, resumable) with F-A/F-B applied.

IT MUST NOT CHANGE THE LIVE (as_of=None) TOP-20.  It is gated behind
`-ingest_delisted` (default OFF) -- when off, this module is never imported on a
live run, so the live scoring path is untouched and bit-for-bit.

OFFLINE-TESTABLE BY DESIGN
--------------------------
Every network touch point takes an injectable getter (`get=`/`http_get=`) and an
injectable `sleep=`, so the whole pipeline runs against synthetic fixtures with no
network (see baseline_tools/test_delisted_ingest.py).  The house builds + tests
this OFFLINE; only the CEO runs it live.

ARTIFACT SPLIT (design s8)
--------------------------
  * DATA (multi-GB, Drive, gitignored): delisted_out/*  -- registry parquet, dead
    fundamentals pickle, bulk prices parquet, death-signature parquet, emptyfail
    csv, resume order.
  * MANIFEST + EVENT LOG (small, GitHub): run_logs/*  -- how the house inspects
    the run without pulling the data.
"""
import os
import time
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import getData_gen as gdg
import getData_fmp as gdf
import entity_id as eid
from run_logging import RunLogger

# ------------------------------------------------------------------------- #
# Output locations
DATA_DIR = "delisted_out"          # gitignored -> Google Drive
LOG_DIR = "run_logs"               # committed -> GitHub (manifest + jsonl)

# Tunables (spec s1/s3/s5)
DEFAULT_MAX_PAGES = 500            # max-page guard against a non-terminating feed
PAGE_PACE_S = 0.3                 # inter-page pace (repo idiom, fetch_prices.py:179)
FUND_PACE_S = 0.3                 # inter-name pace on the dead fundamentals loop
DEAD_FUND_LIMIT = 160             # deep-history quarters per dead name
PER_NAME_SEC = 4.5                # spec s5: ~4.5s * N_dead added wall-clock
BULK_TRAILING_MONTHS = 36         # monthly bulk-price grid depth
DEATH_DRAWDOWN = -0.90            # <= -90% peak-to-terminal = death (design s5)
ACQUISITION_DRAWDOWN = -0.50      # >= -50% = acquisition; middle = not-death
PROBE_SYMBOL = "BBBY"             # recently-dead probe (delisted 2023-05)
PROBE_DATE = "2023-01-31"         # a pre-delisting date BBBY should trade on
AAPL_SPLIT_DATE = "2020-08-31"    # AAPL 4:1 split -- the ride-along split-basis date


class DelistedRegistryError(RuntimeError):
    """Raised when the registry pagination cannot complete safely (e.g. a None
    HTTP response mid-pull).  We ABORT rather than silently truncate the registry
    -- a truncated registry re-introduces the survivorship bias this run exists to
    remove."""


# ========================================================================= #
# 1. REGISTRY PULL (full pagination, None-trap-safe)                          #
# ========================================================================= #
def fetch_delisted_registry(baseurl, api_key, get=None, sleep=None,
                            max_pages=DEFAULT_MAX_PAGES, page_pace=PAGE_PACE_S,
                            logger=None):
    """Paginate v3/delisted-companies until a GENUINE empty/short page.

    THE BIAS TRAP (spec s1): gdg.safe_get returns None on repeated HTTP failure
    (incl. a 429 after 3 backoffs).  None is NOT end-of-pagination -- treating it
    as end-of-list on one transient 429 silently truncates the registry and
    re-introduces survivorship bias.  So None -> HARD ERROR + ABORT (raise), never
    a loop-termination.  Only a parsed [] or a short page (< the observed page
    size) terminates the loop.  A max-page guard bounds a non-terminating feed.

    NOTE on the double slash: this pull uses the correct single-slash
    `{baseurl}v3/...` form (spec s1 cleanup).  The pre-existing LIVE page-0 prune
    (getData_gen.py:130) keeps its historical `//v3` form UNTOUCHED -- it is
    server-normalised and identical, and not touching it preserves the live
    bit-for-bit guarantee with zero risk.

    Returns (registry_df, meta) where registry_df has columns
    symbol, companyName, exchange, ipoDate, delistedDate.
    """
    get = get or gdg.safe_get
    sleep = time.sleep if sleep is None else sleep
    rows = []
    page = 0
    page_size = None
    termination = None
    while page < max_pages:
        # NB: the key is in the URL; we log endpoint PATHS only (never keyed URLs),
        # and the RunLogger scrubs the key as a last-resort defence.
        url = f'{baseurl}v3/delisted-companies?page={page}&apikey={api_key}'
        resp = get(url)
        if resp is None:
            if logger:
                logger.data('registry_pagination_error', level='ERROR', page=page,
                            note='safe_get returned None (HTTP failure incl. 429) -- '
                                 'ABORTING; NOT treated as end-of-list (bias trap)')
            raise DelistedRegistryError(
                f"delisted-companies page {page} returned None (repeated HTTP "
                f"failure). Aborting to avoid silently truncating the registry.")
        if not isinstance(resp, list):
            if logger:
                logger.data('registry_pagination_error', level='ERROR', page=page,
                            note=f'unexpected response type {type(resp).__name__}; aborting')
            raise DelistedRegistryError(
                f"delisted-companies page {page} returned a non-list "
                f"({type(resp).__name__}); aborting.")
        n = len(resp)
        if n == 0:
            termination = 'empty_page'
            break
        rows.extend(resp)
        if page_size is None:
            page_size = n           # establish page size from the first page
        if page_size is not None and n < page_size:
            termination = 'short_page'
            page += 1
            break
        page += 1
        if logger:
            logger.data('registry_page_ok', page=page - 1, rows=n, total=len(rows))
        sleep(page_pace)
    else:
        termination = 'max_page_guard'
        if logger:
            logger.data('registry_max_page_guard', level='WARN', max_pages=max_pages,
                        note='hit the max-page guard; registry may be incomplete')

    df = pd.DataFrame(rows)
    keep = ["symbol", "companyName", "exchange", "ipoDate", "delistedDate"]
    present = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]
    if missing and logger:
        logger.data('registry_missing_fields', level='WARN', missing=missing)
    df = df[present] if present else df
    if "symbol" in df.columns:
        df = df.drop_duplicates(subset=[c for c in ("symbol", "delistedDate")
                                        if c in df.columns]).reset_index(drop=True)

    # per-year delistedDate coverage (spec s1 QA byproduct)
    coverage = {}
    if "delistedDate" in df.columns:
        yrs = pd.to_datetime(df["delistedDate"], errors="coerce").dt.year
        coverage = {int(y): int(c) for y, c in yrs.value_counts(dropna=True).sort_index().items()}

    meta = {"pages": page, "total": len(df), "termination": termination,
            "page_size": page_size, "coverage_by_year": coverage}
    if logger:
        logger.data('registry_complete', pages=page, total=len(df),
                    termination=termination, page_size=page_size)
        logger.set_count("registry_total", len(df))
        logger.set_count("registry_pages", page)
        logger.set_coverage("registry_termination", termination)
        logger.set_coverage("delistedDate_by_year", coverage)
    return df, meta


# ========================================================================= #
# 2. ENTITY-ID ASSIGNMENT (both modes, F4 band) + resume order               #
# ========================================================================= #
def assign_registry_entities(registry_df, live_symbols, logger=None):
    """Assign entity_id (merge = scoring default) + entity_id_split (F4 floor) to
    every registry row, running the detection rule over each symbol's registry
    records PLUS the live occupant (design Component 8).

    Returns a registry DataFrame with added columns entity_id, entity_id_split.
    """
    live_set = set(live_symbols or [])
    out_rows = []
    recycled = 0
    if "symbol" not in registry_df.columns:
        return registry_df.assign(entity_id=registry_df.index.astype(str),
                                  entity_id_split=registry_df.index.astype(str))
    for sym, grp in registry_df.groupby("symbol"):
        recs = grp.to_dict("records")
        for i, r in enumerate(recs):
            r["_rid"] = i           # stable key survives assign_entity_ids' copy+sort
        if sym in live_set:
            # RECYCLED TICKER (review ADDENDUM-3 MEDIUM-1): a symbol that is BOTH in
            # the delisted registry AND currently live is, by definition, a reused
            # ticker -- the dead entity delisted at its last delistedDate and a
            # DISTINCT live occupant trades now.  We only have the live symbol (no
            # real live ipoDate), so we synthesize its first-trade at the latest dead
            # delistedDate (a safe LOWER BOUND: a live occupant necessarily post-dates
            # the predecessor's delisting) AND flag has_price_gap.  This lets
            # `_distinct` establish non-overlap + a corroborant so the split fires in
            # BOTH modes: dead => SYM_2, live successor keeps the bare SYM.  Before
            # this fix the live record had ipoDate=None and no gap, so `_distinct`
            # short-circuited to MERGE and the dead BBBY inherited the bare "BBBY"
            # (the masked flagship bug).
            dlds = pd.to_datetime(grp.get("delistedDate"), errors="coerce").dropna() \
                if "delistedDate" in grp.columns else pd.Series([], dtype="datetime64[ns]")
            live_first = dlds.max() if not dlds.empty else None
            recs.append({"symbol": sym, "ipoDate": None,
                         "firstTrade": (live_first.strftime("%Y-%m-%d")
                                        if live_first is not None and pd.notna(live_first)
                                        else None),
                         "delistedDate": None, "companyName": None,
                         "is_live": True, "has_price_gap": True, "_rid": -1})
        merged = {r["_rid"]: r["entity_id"]
                  for r in eid.assign_entity_ids(recs, mode="merge")}
        split = {r["_rid"]: r["entity_id"]
                 for r in eid.assign_entity_ids(recs, mode="split")}
        n_entities_merge = len(set(v for k, v in merged.items() if k != -1))
        if n_entities_merge > 1:
            recycled += 1
        for i, r in enumerate(recs):
            if r.get("is_live"):
                continue
            row = {k: v for k, v in r.items()
                   if not k.startswith("_") and k != "is_live"}
            row["entity_id"] = merged[r["_rid"]]
            row["entity_id_split"] = split[r["_rid"]]
            out_rows.append(row)
    out = pd.DataFrame(out_rows)
    if logger:
        logger.calc('entity_id_assigned', registry_rows=len(out),
                    recycled_symbols=recycled)
        logger.set_count("recycled_symbols", recycled)
        logger.set_count("entities_merge",
                         int(out["entity_id"].nunique()) if not out.empty else 0)
        logger.set_count("entities_split",
                         int(out["entity_id_split"].nunique()) if not out.empty else 0)
    return out


def dead_fetch_order(registry_ent):
    """Recent-era-first order of DEAD entities to fetch (spec s5, design R-D): an
    interrupted long run still captures the most valuable recent cohort.  Keyed on
    the merge-mode entity_id (scoring default); one representative row per entity
    (the one with the latest delistedDate).  Returns a list of dicts
    {symbol, entity_id, delistedDate} sorted by delistedDate DESC (newest first)."""
    if registry_ent.empty:
        return []
    r = registry_ent.copy()
    r["_dld"] = pd.to_datetime(r.get("delistedDate"), errors="coerce")
    r = r.sort_values("_dld", ascending=False)
    order = []
    seen = set()
    for _, row in r.iterrows():
        e = row["entity_id"]
        if e in seen:
            continue
        seen.add(e)
        order.append({"symbol": row["symbol"], "entity_id": e,
                      "ipoDate": row.get("ipoDate"),
                      "delistedDate": row.get("delistedDate")})
    return order


# resume anchor: persist the FULL enumeration so a resumed run replays the EXACT
# same order (review ADDENDUM-3 MEDIUM-2 -- the old anchor was written but never
# read back, so a drifting registry could re-fetch or SKIP a dead name on resume).
_ORDER_COLS = ("entity_id", "symbol", "ipoDate", "delistedDate")


def _write_order(order, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(_ORDER_COLS) + "\n")
        for o in order:
            f.write("\t".join("" if o.get(c) is None else str(o.get(c))
                              for c in _ORDER_COLS) + "\n")


def _read_order(path):
    """Read back the persisted enumeration anchor as the canonical order dicts.
    Ignores the freshly-recomputed order so `-startfromlastindex` is deterministic
    even if the registry drifted between the interrupted run and the resume."""
    order = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        return order
    header = lines[0].split("\t")
    if header != list(_ORDER_COLS):
        # legacy/short anchor (entity_id\tsymbol only): parse best-effort, no dates
        for ln in lines:
            parts = ln.split("\t")
            if len(parts) >= 2:
                order.append({"entity_id": parts[0], "symbol": parts[1],
                              "ipoDate": None, "delistedDate": None})
        return order
    for ln in lines[1:]:
        parts = ln.split("\t")
        parts += [""] * (len(_ORDER_COLS) - len(parts))
        order.append({c: (parts[i] if parts[i] != "" else None)
                      for i, c in enumerate(_ORDER_COLS)})
    return order


# ========================================================================= #
# 3. N_dead CHECKPOINT                                                        #
# ========================================================================= #
def n_dead_checkpoint(order, per_name_sec=PER_NAME_SEC, logger=None):
    """Emit N_dead + projected added wall-clock BEFORE the long fundamentals loop,
    so the CEO can choose to phase (spec s5)."""
    n = len(order)
    added_s = n * per_name_sec
    added_h = added_s / 3600.0
    msg = (f"N_dead = {n} dead entities to fetch. Projected added wall-clock "
           f"~= {per_name_sec}s * {n} = {added_s:.0f}s (~{added_h:.1f}h), "
           f"recent-era-first so an interrupted run keeps the recent cohort.")
    print("\n" + "=" * 70 + f"\n[N_dead CHECKPOINT] {msg}\n" + "=" * 70, flush=True)
    if logger:
        logger.data('n_dead_checkpoint', n_dead=n, per_name_sec=per_name_sec,
                    projected_added_sec=round(added_s), projected_added_hours=round(added_h, 2))
        logger.set_count("n_dead", n)
        logger.set_coverage("projected_added_hours", round(added_h, 2))
    return n, added_s


# ========================================================================= #
# 4. RIDE-ALONG VERIFICATIONS (logic is offline-testable; JSON injected)      #
# ========================================================================= #
def classify_split_adjustment(hist_json, split_date=AAPL_SPLIT_DATE, ratio_min=2.0,
                              adj_tol=0.15):
    """Decide whether FMP historical-price-full `close` is split-UNADJUSTED
    (settles MEDIUM-2 / DcfToPrice denominator basis, design s5A).

    hist_json : parsed historical-price-full JSON (dict with 'historical' list, or
        a bare list) covering a known split (default AAPL 4:1 2020-08-31).
    Returns (passed, values).  PASS = `close` drops sharply across the split
    (close_before/close_after >= ratio_min) while `adjClose` stays continuous
    (|adj_before/adj_after - 1| <= adj_tol) -> close is RAW -> the DcfToPrice raw
    denominator basis is correct.  If close == adjClose across the split (both
    continuous) -> close is ADJUSTED -> FAIL, flag valuation-specialist.
    """
    hist = hist_json.get("historical") if isinstance(hist_json, dict) else hist_json
    if not hist:
        return False, {"error": "no historical rows"}
    df = pd.DataFrame(hist)
    if "date" not in df.columns or "close" not in df.columns:
        return False, {"error": "missing date/close"}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    sd = pd.Timestamp(split_date)
    before = df[df["date"] < sd]
    after = df[df["date"] >= sd]
    if before.empty or after.empty:
        return False, {"error": "split date not spanned by the series"}
    cb, ca = float(before["close"].iloc[-1]), float(after["close"].iloc[0])
    vals = {"split_date": split_date, "close_before": cb, "close_after": ca,
            "close_ratio": (cb / ca) if ca else None}
    if "adjClose" in df.columns:
        ab = float(before["adjClose"].iloc[-1])
        aa = float(after["adjClose"].iloc[0])
        vals.update({"adj_before": ab, "adj_after": aa,
                     "adj_ratio": (ab / aa) if aa else None})
        close_jumps = ca and (cb / ca) >= ratio_min
        adj_continuous = aa and abs(ab / aa - 1.0) <= adj_tol
        passed = bool(close_jumps and adj_continuous)
        vals["interpretation"] = ("close is RAW/unadjusted (DcfToPrice raw "
                                  "denominator OK)" if passed
                                  else "close appears ADJUSTED or ambiguous -- "
                                       "flag valuation-specialist")
        return passed, vals
    # no adjClose to compare -> cannot certify
    return False, {**vals, "error": "no adjClose column to compare"}


def classify_filing_dates(inc_json, min_year=2023):
    """Decide whether income-statement rows carry populated fillingDate/acceptedDate
    and how far back (settles the PIT-availability fallback, design L1).

    Returns (passed, values).  PASS = fillingDate or acceptedDate populated on
    recent rows AND stays populated back to >= min_year (so the availability-date
    filter works directly).  Otherwise the fixed-lag fallback applies before the
    earliest populated year (which is captured either way).
    """
    rows = inc_json if isinstance(inc_json, list) else inc_json.get("historical", [])
    if not rows:
        return False, {"error": "no income rows"}
    df = pd.DataFrame(rows)
    have_filling = "fillingDate" in df.columns
    have_accepted = "acceptedDate" in df.columns
    if not (have_filling or have_accepted):
        return False, {"error": "neither fillingDate nor acceptedDate present"}
    df["_date"] = pd.to_datetime(df.get("date"), errors="coerce")

    def _earliest_populated(col):
        if col not in df.columns:
            return None
        sub = df[df[col].notna() & (df[col].astype(str).str.len() > 0)]
        if sub.empty:
            return None
        yrs = pd.to_datetime(sub["_date"], errors="coerce").dt.year.dropna()
        return int(yrs.min()) if not yrs.empty else None

    earliest_filling = _earliest_populated("fillingDate")
    earliest_accepted = _earliest_populated("acceptedDate")
    earliest = min([y for y in (earliest_filling, earliest_accepted) if y is not None],
                   default=None)
    n_rows = len(df)
    n_pop = int(((df.get("fillingDate").notna() if have_filling else False)
                 | (df.get("acceptedDate").notna() if have_accepted else False)).sum()) \
        if (have_filling or have_accepted) else 0
    passed = bool(earliest is not None and earliest <= min_year)
    vals = {"n_rows": n_rows, "n_populated": n_pop,
            "earliest_filling_year": earliest_filling,
            "earliest_accepted_year": earliest_accepted,
            "earliest_availability_year": earliest,
            "availability_filter_usable_back_to": earliest,
            "fixed_lag_fallback_applies_before_year": earliest}
    return passed, vals


# ========================================================================= #
# 5. DEAD-ENTITY BULK PRICES (bulk-by-date ONLY) + death signature            #
# ========================================================================= #
def monthly_grid(trailing_months=BULK_TRAILING_MONTHS, end=None, extra_dates=None):
    """Month-end (nearest weekday) grid over the trailing window + any extra dates
    (e.g. the probe date and per-cohort delisting months).  ~36-48 dates."""
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    dates = set()
    cur = end.replace(day=1)
    for _ in range(trailing_months + 1):
        me = (cur + pd.offsets.MonthEnd(0))
        while me.weekday() >= 5:
            me -= pd.Timedelta(days=1)
        dates.add(me.normalize())
        cur = (cur - pd.offsets.MonthBegin(1))
    for d in (extra_dates or []):
        d = pd.Timestamp(d)
        while d.weekday() >= 5:
            d -= pd.Timedelta(days=1)
        dates.add(d.normalize())
    return sorted(dates)


def fetch_bulk_prices(dates, dead_symbols, baseurl, api_key, get=None, sleep=None,
                      probe_symbol=PROBE_SYMBOL, probe_date=PROBE_DATE, logger=None,
                      live_symbols=None):
    """Fetch dead-entity prices via v4/batch-request-end-of-day-prices?date=D ONLY
    (one call per date, whole universe; NEVER per-symbol).  Filter each response to
    dead symbols LOCALLY (zero extra calls).

    PRESENT/ABSENT two-outcome branch (spec s3/s6): if dead symbols are ABSENT from
    the bulk endpoint on historical dates, do NOT silently degrade -- log it, mark
    the run's price mode 'delistedDate_presence_only', emit a LOUD warning + a CEO
    decision, and take the WEAKER path (no drawdown magnitude).  The per-symbol bar
    is NEVER violated.

    OUTAGE GUARD (review ADDENDUM-3 LOW-2): a momentary TOTAL outage/throttle (every
    date returns None/empty) would otherwise present as presence_rate=0 and be
    misread as "dead symbols structurally absent", degrading the whole run to the
    weaker presence-only path on a transient blip.  So we separately confirm the
    endpoint is WORKING -- at least one date returned LIVE symbols (a live-symbol hit
    if `live_symbols` is supplied, else any non-empty batch, which the whole-market
    endpoint only returns when it is up).  If it never worked we return the distinct
    mode 'bulk_endpoint_unavailable' (retry the price pull) rather than concluding
    structural absence.

    Returns (prices_df, meta).  prices_df columns: date_requested, date_actual,
    symbol, close, adjClose (whatever the endpoint provides).
    """
    get = get or gdg.safe_get
    sleep = time.sleep if sleep is None else sleep
    dead_set = set(dead_symbols or [])
    live_set = set(live_symbols or [])
    rows = []
    calls = 0
    probe_present = None
    present_dead = set()
    endpoint_working = False       # LOW-2: did the endpoint ever return LIVE data?
    for d in dates:
        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        url = f'{baseurl}v4/batch-request-end-of-day-prices?date={date_str}&apikey={api_key}'
        resp = get(url)
        calls += 1
        if resp is None:
            if logger:
                logger.data('bulk_price_retry', level='WARN', date=date_str,
                            note='None response (HTTP failure); skipping this date')
            continue
        data = resp if isinstance(resp, list) else resp.get("historical", resp) \
            if isinstance(resp, dict) else []
        if not isinstance(data, list):
            data = []
        # LOW-2 outage guard: mark the endpoint as WORKING if it returned live data.
        if data:
            if live_set:
                if any(isinstance(r, dict) and r.get("symbol") in live_set for r in data):
                    endpoint_working = True
            else:
                endpoint_working = True   # whole-market batch is non-empty only when up
        # BBBY presence probe on the probe date (spec s3)
        if date_str == pd.Timestamp(probe_date).strftime("%Y-%m-%d") or \
                (probe_present is None and calls == 1):
            syms_here = {r.get("symbol") for r in data if isinstance(r, dict)}
            hit = probe_symbol in syms_here
            if probe_present is None or date_str == pd.Timestamp(probe_date).strftime("%Y-%m-%d"):
                probe_present = hit
                if logger:
                    logger.data('bulk_probe', date=date_str, probe_symbol=probe_symbol,
                                present=hit)
        for r in data:
            if not isinstance(r, dict):
                continue
            sym = r.get("symbol")
            if dead_set and sym not in dead_set:
                continue
            if sym in dead_set:
                present_dead.add(sym)
            rows.append({"date_requested": date_str, "date_actual": r.get("date", date_str),
                         "symbol": sym, "close": r.get("close"),
                         "adjClose": r.get("adjClose", r.get("adjclose"))})
        sleep(0.3)

    prices_df = pd.DataFrame(rows)
    presence_rate = (len(present_dead) / len(dead_set)) if dead_set else 0.0
    if presence_rate > 0.0:
        mode = "full"
    elif not endpoint_working:
        # LOW-2: the endpoint NEVER returned live data across the whole grid -> this
        # is a transient TOTAL OUTAGE/throttle, NOT evidence that dead names are
        # structurally absent.  Do NOT degrade to presence-only on a blip.
        mode = "bulk_endpoint_unavailable"
        warn = ("BULK-PRICE ENDPOINT UNAVAILABLE: v4/batch-request-end-of-day-prices "
                "returned NO live data on ANY date in the grid -- a transient outage "
                "or throttle, NOT structural dead-symbol absence. NOT concluding dead "
                "names are absent. *** RETRY the price pull *** before trusting a "
                "presence-only death signature.")
        warnings.warn(warn)
        if logger:
            logger.data('bulk_endpoint_unavailable', level='WARN',
                        dead_symbols=len(dead_set), note=warn)
    else:
        mode = "delistedDate_presence_only"
        # LOUD: endpoint IS working (live symbols present) but dead names are
        # unobtainable within the bulk discipline -> weaker path
        warn = ("BULK-PRICE ABSENT: the endpoint is working (live symbols present) "
                "but no dead symbol appeared in v4/batch-request-end-of-day-prices "
                "across the grid. Death-signature DRAWDOWN magnitude cannot be built "
                "within the per-symbol bar. FALLING BACK to delistedDate-presence-"
                "only death detection (no drawdown magnitude). *** CEO DECISION "
                "NEEDED *** whether to relax the per-symbol bar for dead names only. "
                "Not degrading silently.")
        warnings.warn(warn)
        if logger:
            logger.data('bulk_price_absent', level='WARN',
                        dead_symbols=len(dead_set), present=len(present_dead),
                        endpoint_working=endpoint_working, note=warn)
    meta = {"calls": calls, "dates": len(dates), "mode": mode,
            "probe_symbol": probe_symbol, "probe_present": probe_present,
            "endpoint_working": endpoint_working,
            "dead_present": len(present_dead), "dead_total": len(dead_set),
            "presence_rate": round(presence_rate, 4)}
    if logger:
        logger.data('bulk_price_complete', calls=calls, dates=len(dates), mode=mode,
                    dead_present=len(present_dead), dead_total=len(dead_set),
                    presence_rate=round(presence_rate, 4), probe_present=probe_present)
        logger.set_count("bulk_price_calls", calls)
        logger.set_coverage("bulk_price_mode", mode)
        logger.set_coverage("bulk_price_span",
                            [pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
                             pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")] if dates else [])
        logger.set_coverage("bulk_dead_presence_rate", round(presence_rate, 4))
    return prices_df, meta


def slice_to_entity(price_df, ipo, delisted):
    """Bound a per-symbol price series to the entity life-span [ipo, delisted]
    BEFORE death-signature classification (design s5 / Component 8; the BBBY trap)."""
    df = price_df.copy()
    df["date_actual"] = pd.to_datetime(df["date_actual"], errors="coerce")
    lo = pd.to_datetime(ipo, errors="coerce")
    hi = pd.to_datetime(delisted, errors="coerce")
    if pd.notna(lo):
        df = df[df["date_actual"] >= lo]
    if pd.notna(hi):
        df = df[df["date_actual"] <= hi]
    return df.sort_values("date_actual")


def classify_death_signature(prices_df, registry_ent, mode="full", logger=None):
    """Per dead entity: peak-to-terminal drawdown -> {death, acquisition,
    not_death} (design s5).  On mode='delistedDate_presence_only' the drawdown is
    unavailable -> label 'delisted_presence_only' (weaker, un-graded)."""
    out = []
    death = acq = notd = presence_only = 0
    if registry_ent.empty:
        return pd.DataFrame(out)
    reg = registry_ent.drop_duplicates(subset=["entity_id"])
    for _, ent in reg.iterrows():
        eidv = ent["entity_id"]
        sym = ent["symbol"]
        rec = {"entity_id": eidv, "symbol": sym,
               "delistedDate": ent.get("delistedDate")}
        if mode != "full" or prices_df is None or prices_df.empty:
            rec["classification"] = "delisted_presence_only"
            rec["drawdown"] = np.nan
            presence_only += 1
            out.append(rec)
            continue
        ser = slice_to_entity(prices_df[prices_df["symbol"] == sym],
                              ent.get("ipoDate"), ent.get("delistedDate"))
        close = pd.to_numeric(ser["close"], errors="coerce").dropna()
        if len(close) < 2:
            rec["classification"] = "insufficient_prices"
            rec["drawdown"] = np.nan
            presence_only += 1
            out.append(rec)
            continue
        peak = float(close.max())
        terminal = float(close.iloc[-1])
        dd = (terminal / peak - 1.0) if peak > 0 else np.nan
        rec["drawdown"] = dd
        if dd <= DEATH_DRAWDOWN:
            rec["classification"] = "death"; death += 1
        elif dd >= ACQUISITION_DRAWDOWN:
            rec["classification"] = "acquisition"; acq += 1
        else:
            rec["classification"] = "not_death"; notd += 1
        out.append(rec)
    if logger:
        logger.calc('death_signature', mode=mode, death=death, acquisition=acq,
                    not_death=notd, presence_only=presence_only)
        logger.set_distribution("death_signature",
                                {"death": death, "acquisition": acq,
                                 "not_death": notd, "presence_only": presence_only})
    return pd.DataFrame(out)


# ========================================================================= #
# 6. DEAD-ENTITY FUNDAMENTALS (recent-first, resumable, F-A/F-B)              #
# ========================================================================= #
def _slice_statements(stmts, ipo, delisted):
    """Bound each statement DataFrame to the dead entity's life-span [ipo, delisted]
    on its own 'date' column (review ADDENDUM-3 MEDIUM-1 aggravator).  A RECYCLED
    symbol's live-successor quarters -- which FMP returns under the reused ticker --
    must NOT bleed into the dead entity's record (the BBBY trap on the fundamentals
    axis).  A normal dead name (all statements pre-delisting) is unaffected: nothing
    is outside [ipo, delisted], so this is a no-op for it.  Rows with an unparseable
    date are kept (we do not drop what we cannot place).  Returns (stmts, n_trimmed)."""
    lo = pd.to_datetime(ipo, errors="coerce")
    hi = pd.to_datetime(delisted, errors="coerce")
    trimmed = 0
    if pd.isna(lo) and pd.isna(hi):
        return stmts, 0
    for name, s in list(stmts.items()):
        if not hasattr(s, "columns") or "date" not in s.columns:
            continue
        d = pd.to_datetime(s["date"], errors="coerce")
        mask = pd.Series(True, index=s.index)
        if pd.notna(hi):
            mask &= (d <= hi) | d.isna()
        if pd.notna(lo):
            mask &= (d >= lo) | d.isna()
        if not bool(mask.all()):
            trimmed += int((~mask).sum())
            stmts[name] = s[mask].reset_index(drop=True)
    return stmts, trimmed


def fetch_dead_fundamentals(order, baseurl, api_key, period="quarter",
                            limit=DEAD_FUND_LIMIT, start_index=0, http_get=None,
                            sleep=None, logger=None, lastindex_fn=None):
    """Fetch the 5 statement endpoints for each dead entity (recent-era-first),
    with F-A (datefail bypassed) + F-B (lenfail relaxed) via dead_path=True.

    HIGH-1 (review ADDENDUM-3) -- TRANSIENT-ERROR GUARD + failcode/emptyfail SPLIT.
    This is the longest phase and the most exposed to FMP throttling, so:
      * the fetch is routed through `gdg.safe_http_get` (bounded retry/backoff on
        429/5xx + a timeout) by DEFAULT on the live CEO run, instead of a raw
        no-retry `requests.get`; a transient throttle is RETRIED, not dropped.
      * a name that STILL fails after retries is classified by WHY it failed:
          - `emptyfail`  -> the endpoints genuinely returned [] (no data) -- a
                            first-class completeness artifact ("no fundamentals");
          - `failcode`   -> a definitive HTTP 4xx/5xx (incl. an exhausted 429) --
                            a FETCH-UNKNOWN / RE-AUDIT name, NOT "no fundamentals".
        The two buckets are separated so a throttle drop can never masquerade as a
        genuine dead-name absence (the silent-survivorship-within-death hole).  We
        recover WHY from the gate's own bookkeeping: `getFsData_fmp` appends the
        symbol to the `emptyfail` list ONLY on a real empty response, and to
        `tickersfailed` on ANY fail -- so `km==-37707 & ef non-empty` is emptyfail,
        and `km==-37707 & ef empty` is a failcode (datefail/lenfail are disabled on
        the dead path, so failcode is the only other cause).

    Graceful continue: neither bucket aborts the multi-hour loop.  Each kept dead
    name is tagged filing_date_source='fixed_lag' and sliced to its entity life-span
    (MEDIUM-1); short_history (<16q, post-slice) is tagged.

    Returns (results, meta, emptyfail, failcode).  results: {entity_id: {km,fr,inc,
    bs,cf,symbol,short_history,filing_date_source}}.
    """
    sleep = time.sleep if sleep is None else sleep
    # Default the DEAD-path getter to the retry/backoff/timeout getter.  The LIVE
    # scoring path is untouched: it never calls this function and still defaults to
    # raw requests.get inside testForAPIFaults_fmp (bit-for-bit preserved).
    if http_get is None:
        http_get = gdg.safe_http_get
    results = {}
    emptyfail, failcode, short = [], [], []
    n_endpoint_ok = 0
    for idx in range(start_index, len(order)):
        ent = order[idx]
        sym = ent["symbol"]
        try:
            km, fr, inc, bs, cf, tf, lf, dfl, ef = gdf.getFsData_fmp(
                sym, period, limit, baseurl, api_key, 0, [], [], [], [],
                dead_path=True, http_get=http_get)
        except Exception as e:
            # Any residual error escaping the getter (e.g. a getter that returned a
            # bad object) is a FETCH-UNKNOWN, never an emptyfail.  Record + continue.
            failcode.append({"symbol": sym, "entity_id": ent["entity_id"],
                             "whyfail": "exception", "error": str(e)})
            if logger:
                logger.data('dead_fund_failcode', level='WARN', symbol=sym,
                            entity_id=ent["entity_id"], whyfail='exception',
                            error=str(e))
            _persist_resume(lastindex_fn, idx + 1)
            sleep(FUND_PACE_S)
            continue
        if isinstance(km, int) and km == -37707:
            # split emptyfail (genuine absence) from failcode (retryable/fetch-unknown)
            whyfail = "emptyfail" if ef else "failcode"
            rec = {"symbol": sym, "entity_id": ent["entity_id"], "whyfail": whyfail}
            if whyfail == "emptyfail":
                emptyfail.append(rec)
                if logger:
                    logger.data('dead_fund_emptyfail', symbol=sym,
                                entity_id=ent["entity_id"])
            else:
                failcode.append(rec)
                if logger:
                    logger.data('dead_fund_failcode', level='WARN', symbol=sym,
                                entity_id=ent["entity_id"], whyfail='failcode',
                                note='HTTP 4xx/5xx after retries -- re-audit, NOT '
                                     '"no fundamentals"')
            _persist_resume(lastindex_fn, idx + 1)
            sleep(FUND_PACE_S)
            continue
        stmts = {}
        for name, stmt in (("km", km), ("fr", fr), ("inc", inc),
                           ("bs", bs), ("cf", cf)):
            s = stmt.copy()
            s["filing_date_source"] = "fixed_lag"   # dead names -> fixed-lag PIT fallback
            s["entity_id"] = ent["entity_id"]
            stmts[name] = s
        # MEDIUM-1: slice to the dead entity's life-span so a recycled symbol's
        # successor quarters do not contaminate the dead record.
        stmts, trimmed = _slice_statements(stmts, ent.get("ipoDate"),
                                           ent.get("delistedDate"))
        try:
            short_hist = len(stmts["inc"]) < 16
        except Exception:
            short_hist = False
        results[ent["entity_id"]] = {**stmts, "symbol": sym,
                                     "short_history": bool(short_hist),
                                     "filing_date_source": "fixed_lag"}
        n_endpoint_ok += 5
        if short_hist:
            short.append(sym)
        if logger:
            logger.data('dead_fund_ok', symbol=sym, entity_id=ent["entity_id"],
                        quarters=int(len(stmts["inc"])) if hasattr(stmts["inc"], "__len__") else 0,
                        life_trimmed_rows=trimmed, short_history=bool(short_hist))
        _persist_resume(lastindex_fn, idx + 1)
        sleep(FUND_PACE_S)

    meta = {"fetched": len(results), "emptyfail": len(emptyfail),
            "failcode": len(failcode), "short_history": len(short),
            "endpoint_ok": n_endpoint_ok}
    if logger:
        logger.data('dead_fund_complete', fetched=len(results),
                    emptyfail=len(emptyfail), failcode=len(failcode),
                    short_history=len(short))
        logger.set_count("dead_fund_fetched", len(results))
        logger.set_count("dead_fund_emptyfail", len(emptyfail))
        logger.set_count("dead_fund_failcode", len(failcode))
        logger.set_count("dead_fund_short_history", len(short))
    return results, meta, emptyfail, failcode


def _persist_resume(path, idx):
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write(str(idx))
    except Exception:
        pass


# ========================================================================= #
# Persistence helpers (parquet -> csv fallback so a run never dies on IO)     #
# ========================================================================= #
def _write_table(df, basename, data_dir, logger=None):
    os.makedirs(data_dir, exist_ok=True)
    pq = os.path.join(data_dir, basename + ".parquet")
    try:
        df.to_parquet(pq, index=False)
        return pq
    except Exception as e:
        csv = os.path.join(data_dir, basename + ".csv")
        df.to_csv(csv, index=False)
        if logger:
            logger.data('parquet_fallback_csv', level='WARN', basename=basename,
                        error=str(e))
        return csv


# ========================================================================= #
# ORCHESTRATOR                                                                #
# ========================================================================= #
def run_ingest(configdic, live_symbols=None, get=None, http_get=None, sleep=None,
               ridealong=None, data_dir=DATA_DIR, log_dir=LOG_DIR,
               do_fundamentals=True):
    """Full delisted-entity ingestion.  Cheap/high-value acquisition + the two
    verifications run FIRST; the multi-hour dead-fundamentals loop runs LAST
    (recent-era-first, resumable) so an interruption keeps the most valuable data.

    All network touch points are injectable (get/http_get/sleep/ridealong) so the
    whole flow runs offline against fixtures.
    """
    baseurl = configdic["baseurl"]
    api_key = configdic["api_key"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = RunLogger(run_id, out_dir=log_dir, secrets=[api_key])
    try:
        # live universe (for the entity_id union + universe membership)
        if live_symbols is None:
            import getData_gen as _gdg
            live_df = _gdg.get_tickers(configdic["datasource"], baseurl, api_key, [],
                                       configdic["tickerfilter"], sfilt="all", mcapf=-1,
                                       fn="", as_of=None)
            live_symbols = list(live_df["symbol"])
        logger.data('universe_live', live_count=len(live_symbols))
        logger.set_count("live_count", len(live_symbols))

        # 1. registry
        registry, reg_meta = fetch_delisted_registry(
            baseurl, api_key, get=get, sleep=sleep, logger=logger,
            max_pages=int(configdic.get("delisted_max_pages", DEFAULT_MAX_PAGES)))
        # 2. entity ids (both modes) + resume order
        registry_ent = assign_registry_entities(registry, live_symbols, logger=logger)
        union_size = len(set(live_symbols) | set(registry_ent.get("symbol", [])))
        logger.data('universe_union', live=len(live_symbols),
                    dead_ingested=int(registry_ent["entity_id"].nunique())
                    if not registry_ent.empty else 0, union_size=union_size)
        _write_table(registry_ent, "delisted_registry", data_dir, logger=logger)
        # MEDIUM-2: deterministic resume.  On a resume we REPLAY the persisted
        # enumeration anchor (so an integer start_index always points at the same
        # entity even if the registry drifted); on a fresh run we compute + persist
        # the full anchor.
        os.makedirs(data_dir, exist_ok=True)
        order_path = os.path.join(data_dir, "dead_fetch_order.txt")
        resuming = bool(configdic.get("startfromlastindex")) and os.path.exists(order_path)
        if resuming:
            order = _read_order(order_path)
            logger.data('resume_order_loaded', anchor=order_path, n_dead=len(order),
                        note='replaying persisted enumeration anchor for '
                             'deterministic resume (MEDIUM-2)')
        else:
            order = dead_fetch_order(registry_ent)
            _write_order(order, order_path)

        # 3. N_dead checkpoint (BEFORE the long loop)
        n_dead_checkpoint(order, logger=logger)

        # 4. ride-along verifications (2 calls; capture raw JSON; PASS/FAIL)
        ra = ridealong or _default_ridealong(baseurl, api_key, get=get)
        # LOW-3: raw vendor JSON goes to the gitignored Drive data dir, NOT the
        # committed run_logs/ dir (keep raw payloads out of git; the PASS/FAIL +
        # values that the house inspects are logged via logger.verify below).
        split_json = ra["split"]()
        _dump_json(split_json, os.path.join(data_dir, f"ridealong_split_{run_id}.json"))
        split_pass, split_vals = classify_split_adjustment(split_json)
        logger.verify("ridealong_split_adjustment", split_pass, values=split_vals)
        filing_json = ra["filing"]()
        _dump_json(filing_json, os.path.join(data_dir, f"ridealong_filing_{run_id}.json"))
        filing_pass, filing_vals = classify_filing_dates(filing_json)
        logger.verify("ridealong_filing_dates", filing_pass, values=filing_vals)

        # 5. bulk prices (probe + present/absent branch) + death signature
        dead_syms = [o["symbol"] for o in order]
        extra = []
        if "delistedDate" in registry_ent.columns:
            dlds = pd.to_datetime(registry_ent["delistedDate"], errors="coerce").dropna()
            extra = list({d.normalize() for d in dlds})
        grid = monthly_grid(extra_dates=[PROBE_DATE] + extra)
        prices, price_meta = fetch_bulk_prices(grid, dead_syms, baseurl, api_key,
                                               get=get, sleep=sleep, logger=logger,
                                               live_symbols=live_symbols)
        if not prices.empty:
            _write_table(prices, "bulk_prices", data_dir, logger=logger)
        death_df = classify_death_signature(prices, registry_ent,
                                            mode=price_meta["mode"], logger=logger)
        if not death_df.empty:
            _write_table(death_df, "death_signature", data_dir, logger=logger)

        # 6. dead fundamentals (recent-first, resumable) -- the LONG loop, last
        if do_fundamentals:
            lastindex_fn = os.path.join(data_dir, "lastIndexOfRead_delisted.txt")
            start_index = 0
            if configdic.get("startfromlastindex") and os.path.exists(lastindex_fn):
                try:
                    start_index = int(open(lastindex_fn).read().strip() or 0)
                except Exception:
                    start_index = 0
            results, fund_meta, emptyfail, failcode = fetch_dead_fundamentals(
                order, baseurl, api_key, period=configdic.get("period", "quarter"),
                start_index=start_index, http_get=http_get, sleep=sleep,
                logger=logger, lastindex_fn=lastindex_fn)
            # persist dead fundamentals (pickle -> gitignored, Drive) + BOTH the
            # emptyfail (genuine no-data) and the failcode (fetch-unknown / re-audit)
            # sets as first-class manifest artifacts (HIGH-1: the bias is now
            # measurable, not silently folded into "no fundamentals").
            pd.to_pickle(results, os.path.join(data_dir,
                                               f"dead_fundamentals_{run_id}.pickle"))
            ef_df = pd.DataFrame(emptyfail)
            if not ef_df.empty:
                ef_df.to_csv(os.path.join(log_dir, f"dead_emptyfail_{run_id}.csv"),
                             index=False)
            fc_df = pd.DataFrame(failcode)
            if not fc_df.empty:
                fc_df.to_csv(os.path.join(log_dir, f"dead_failcode_reaudit_{run_id}.csv"),
                             index=False)
                logger.data('dead_fund_failcode_set', level='WARN',
                            n_reaudit=len(failcode),
                            note='transient/HTTP-failed dead names -- treat as '
                                 'FETCH-UNKNOWN and re-audit; NOT "no fundamentals"')
            logger.set_coverage("dead_fund", fund_meta)

        manifest = logger.write_manifest()
        print(f"\n[delisted-ingest] DONE. run_id={run_id}. manifest={manifest}",
              flush=True)
        return {"run_id": run_id, "registry": registry_ent, "order": order,
                "prices": prices, "death": death_df, "manifest": manifest,
                "price_mode": price_meta["mode"]}
    finally:
        logger.close()


def _default_ridealong(baseurl, api_key, get=None):
    get = get or gdg.safe_get

    def split():
        return get(f"{baseurl}v3/historical-price-full/AAPL?apikey={api_key}")

    def filing():
        return get(f"{baseurl}v3/income-statement/AAPL?period=quarter&limit=120&apikey={api_key}")

    return {"split": split, "filing": filing}


def _dump_json(obj, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, default=str)
    except Exception:
        pass


if __name__ == "__main__":
    # Standalone entry point (decoupled from the live Sbocker run).  Reads the same
    # configuration so `python delisted_ingest.py -ingest_delisted` works on the CEO
    # machine independently of the 12h live fetch.
    import sys
    import configuration as cf
    cfg = cf.getDataFetchConfiguration(sys.argv[1:])
    run_ingest(cfg)
