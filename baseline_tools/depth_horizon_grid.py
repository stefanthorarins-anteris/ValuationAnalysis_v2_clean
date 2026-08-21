"""
DEPTH x HORIZON  AVERAGE-TOTAL-RETURN GRID  (offline, network-free).

The CEO's orienting diagnostic: NOT the beat-rate -- the equal-weight AVERAGE TOTAL
RETURN of the screen's top-N picks, by pick-depth N and holding horizon, on the
SURVIVORSHIP-CLEAN universe.  Answers: does the RANKING concentrate return at the top
(top-3/5/10 vs top-100?) and how does each depth compare to MSCI World (URTH)?

ARCHITECTURE (per CEO design correction):
  * RETURNS ARE THE PRIMITIVE.  All return math lives in returns_core.compute_returns
    (ticker-agnostic: tickers + two dates -> per-ticker total return, with the
    delisted-terminal bracket).  This script adds NO return math -- it only (1) ranks
    once per buy anchor, (2) slices the top-N tickers, (3) calls the primitive per
    horizon eval date, and (4) applies returns_core's average/excess DERIVED views.

Builds on the certified pipeline as INTERFACES:
  dead_merge.merge_dead_into_dmdic / pit_universe   -> survivorship-clean universe
  stage2_pit.reproduce_pit_top(..., universe_override=pit_universe)
                                                    -> PIT full ranking (depth ~100)
  returns_core.PriceSource / compute_returns / average_return / benchmark_return
Prices: baseline_tools/price_data/real_prices.csv.  Benchmark: URTH (MSCI World TR proxy).

EFFICIENCY: the ranking for a buy anchor does NOT depend on horizon -> rank ONCE per
buy anchor and vary only the eval date.  The dead-frame build (anchor-independent) is
done ONCE and reused across anchors.

No network I/O; never prints the pickle's api_key.

Run:  python baseline_tools/depth_horizon_grid.py --out <file.out> [--csv <file.csv>]
"""

import argparse
import datetime as _dt
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import dead_merge as dm
import stage2_pit as s2
import returns_core as rc
import derived_prices as dpx

# --------------------------------------------------------------------------- #
#  Default inputs (present locally at the HomeGDrive paths; NONE cross git).   #
# --------------------------------------------------------------------------- #
_HOME = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
DEFAULT_PICKLE = os.path.join(
    _HOME, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-07-13_len7879_manelim3692_fails1966.pickle")
DEFAULT_DEAD = os.path.join(_HOME, "delisted_out", "dead_fundamentals_20260713_104350.pickle")
DEFAULT_REGISTRY = os.path.join(_HOME, "delisted_out", "delisted_registry.csv")
DEFAULT_PRICES = os.path.join(_HERE, "price_data", "real_prices.csv")
# Supplementary 2025 prices, merged in-memory to add the 2025-12-31 anchor WITHOUT
# mutating the canonical real_prices.csv (see returns_core.PriceSource._merge_supplementary).
DEFAULT_PRICES_2025 = os.path.join(_HERE, "price_data", "real_prices_2025.csv")

ANCHORS = rc.DEFAULT_ANCHORS
ANCHOR_IDX = {a: i for i, a in enumerate(ANCHORS)}

BUY_ANCHORS = [
    ("buy2018", "2018-12-31"),
    ("buy2019", "2019-12-31"),
    ("buy2020", "2020-12-31"),
    ("buy2021", "2021-12-31"),
    ("buy2022", "2022-12-30"),
    ("buy2023", "2023-12-29"),
    ("buy2024", "2024-12-31"),
]

DEPTHS = [1, 2, 3, 5, 10, 20, 50, 100]
HORIZONS = [12, 24, 36]  # months

# Degeneracy flag (CEO-stated): pre-2021 buy anchors are UNIVERSE-DEGENERATE -- the
# scoring pickle's history cap leaves only a non-representative long-history subset
# (~1,000-1,250 names) live PIT, so a top-100 there is a large slice of the scored pool
# and the depth cut is not meaningful.  buy2024 is the RICHEST anchor (7,863 names >=8q)
# -> CLEAN.
CLEAN_BUY_IDS = {"buy2021", "buy2022", "buy2023", "buy2024"}


# --------------------------------------------------------------------------- #
#  Scoring-config knobs (weights + carve membership)                          #
# --------------------------------------------------------------------------- #
def build_weight_override(mode):
    """mode='default' -> None (production weight vector, live path unchanged).
    mode='equal'   -> {every getPostDict metric key: 1} (naive equal weighting).

    Sets ALL 21 getPostDict keys to 1 (per the brief's "if unsure, set every key
    to 1"). Consequences worth noting: CycleHeat flips from its -0.5 penalty to +1
    (equal-weight has no sign concept); DcfToPrice=1 is INERT (dropped in stage-2,
    so never weighted/summed); BoScore goes 0.1 -> 1.  This is exactly "naive
    equal-weight over the metrics stage-2 actually uses."
    """
    if mode == "default":
        return None
    if mode == "equal":
        import createDicts as cdic
        postBm, postNew = cdic.getPostDict()
        return {k: 1 for k in list(postBm.keys()) + list(postNew.keys())}
    raise ValueError(f"unknown weights mode: {mode!r}")


def carve_general_universe(pit_universe, merged_cdx, tickers_df, log):
    """Filter a survivorship PIT universe to carveOut's GENERAL pool (REIT/Mining/
    Financial cohorts routed out, issuer de-dup, $25M mcap floor) BEFORE ranking.

    Mirrors production's invocation (postBo.py:128): partition_universe over the
    universe's sources with the merged cdx (dead names included) + Tickers_df, default
    sector/industry pickles (auto-found in CWD), mcap_floor=25e6, cohort_head=25.
    partition_universe's MEMBERSHIP decision is score-independent (dedup by
    sector/mcap/symbol-shape, carve by sector, floor by mcap), so a placeholder score
    column yields the identical general set -- verified against carveOut.py.
    """
    import carveOut as co
    bs = pd.DataFrame({"source": sorted(pit_universe), "score": 0.0})
    part = co.partition_universe(bs, merged_cdx, tickers_df,
                                 mcap_floor=25e6, cohort_head=25)
    general = set(part["general"]["source"])
    d = part["diagnostics"]
    log(f"    carve: {d['n_universe']} -> general={d['n_general']} "
        f"(REIT={d['n_REIT']} Mining={d['n_Mining']} "
        f"FIN1={d['n_InvestmentVehicle']} FIN2={d.get('n_FinManager',0)} "
        f"FIN3={d.get('n_BalanceSheetFin',0)} below_floor={d['n_below_floor']})")
    return general


# --------------------------------------------------------------------------- #
#  Input loading (expensive: pickle load + ONE dead-merge build)              #
# --------------------------------------------------------------------------- #
def load_inputs(pickle_path, dead_path, registry_path, log):
    log("Loading base scoring pickle (offline) ...")
    dmdic = pd.read_pickle(pickle_path)
    log("Loading dead-fundamentals pickle + registry ...")
    dead = pd.read_pickle(dead_path)
    registry = dm.load_registry(registry_path)
    live_sources = set(dmdic["cdx_df"]["source"].dropna().unique())

    # Build the dead-name merge ONCE (frames are anchor-independent; PIT membership is
    # applied per anchor via pit_universe + date<=D inside reproduce_pit_top).
    log("Merging dead names into scoring frames (ONCE; anchor-independent build) ...")
    merged, mstats = dm.merge_dead_into_dmdic(
        dmdic, dead, registry, as_of=BUY_ANCHORS[0][1])
    log(f"merge build: universe(as_of[0])={mstats.get('universe_size')} "
        f"built_dead={mstats.get('built')} gate_fail={mstats.get('gate_fail')} "
        f"skipped_collision={mstats.get('skipped_collision')}")

    bm_all = merged["BoMetric_df"].copy()
    bm_all["date"] = pd.to_datetime(bm_all["date"], errors="coerce")
    return {"dmdic": dmdic, "registry": registry, "merged": merged,
            "live_sources": live_sources, "bm_all": bm_all}


def inputs_from_memory(dmdic, merged, registry, log):
    """Build the rank_all_anchors `inputs` dict from ALREADY-LOADED objects (no pickle
    load, no dead-merge rebuild).  Mirrors load_inputs' output contract but takes the
    dead-merge that the pipeline orchestrator built ONCE and shares across stages, so
    the expensive merge_dead_into_dmdic is not repeated per stage.
    """
    live_sources = set(dmdic["cdx_df"]["source"].dropna().unique())
    bm_all = merged["BoMetric_df"].copy()
    bm_all["date"] = pd.to_datetime(bm_all["date"], errors="coerce")
    log(f"    inputs_from_memory: live_sources={len(live_sources)} "
        f"bm_all_rows={len(bm_all)}")
    return {"dmdic": dmdic, "registry": registry, "merged": merged,
            "live_sources": live_sources, "bm_all": bm_all}


def run_in_pipeline(dmdic, merged, registry, price_source, log=None,
                    weights="default", carve="off"):
    """IN-MEMORY entry point for the automatic pipeline (post-pick analysis suite).

    Reproduces the PIT ranking as-of each historical buy anchor USING TONIGHT's model
    (the fresh `dmdic`/`merged` scoring frames), then builds the depth x horizon
    average-total-return grid -- NOT the hardcoded stale DEFAULT_PICKLE that main()
    loads.  This is the HISTORICAL PIT backtest re-run against tonight's model; it does
    NOT analyze tonight's live pick (no forward prices exist yet).

    Returns (report_text, per_anchor).  per_anchor is returned so the beat-rate-vs-URTH
    stage can REUSE the (expensive) rankings instead of reproducing them again.
    Never prints any api_key.
    """
    log = log or (lambda *a: None)
    inputs = inputs_from_memory(dmdic, merged, registry, log)
    per_anchor = rank_all_anchors(inputs, log, weights=weights, carve=carve)
    cells, pooled, pooled_clean = compute_grid(per_anchor, price_source)
    text = build_report(per_anchor, cells, pooled, pooled_clean)
    print("\n" + "#" * 72)
    print(f"# DEPTH x HORIZON avg-total-return GRID  (tonight's model, real prices, "
          f"weights={weights}, carve={carve})")
    print("#" * 72)
    print(text, flush=True)
    return text, per_anchor


# --------------------------------------------------------------------------- #
#  Ranking: once per buy anchor (full ordering to depth 100)                  #
# --------------------------------------------------------------------------- #
def rank_all_anchors(inputs, log, weights="default", carve="off"):
    """Rank all buy anchors under one scoring config.

    weights : 'default' (production weights) | 'equal' (all metric weights = 1).
    carve   : 'off' (full un-carved survivorship universe -- current behaviour) |
              'on'  (universe filtered to carveOut general pool BEFORE ranking).
    weights='default' + carve='off' reproduces the baseline grid bit-for-bit.
    """
    dmdic, registry, merged = inputs["dmdic"], inputs["registry"], inputs["merged"]
    live_sources, bm_all = inputs["live_sources"], inputs["bm_all"]
    weight_override = build_weight_override(weights)
    tickers_df = dmdic.get("Tickers_df")

    per_anchor = {}
    for wid, buy in BUY_ANCHORS:
        log(f"[{wid}] pit_universe + reproducing PIT ranking as-of {buy} "
            f"(weights={weights}, carve={carve}) ...")
        uni = dm.pit_universe(dmdic, registry, as_of=buy)
        if carve == "on":
            uni = sorted(carve_general_universe(uni, merged["cdx_df"], tickers_df, log))
        res = s2.reproduce_pit_top(merged, buy, universe_override=uni,
                                   weight_override=weight_override)
        if res is None:
            log(f"[{wid}] reproduce_pit_top returned None -- skipping")
            continue
        ranking = res["pool_after_norm"]  # full stage-2 ordering (depth up to ~100)

        uni_set = set(uni)
        bm_pit = bm_all[(bm_all["source"].isin(uni_set)) & (bm_all["date"] <= pd.Timestamp(buy))]
        n_pit = int(bm_pit["source"].nunique())
        n_pit_live = int(bm_pit[bm_pit["source"].isin(live_sources)]["source"].nunique())

        per_anchor[wid] = {
            "buy": buy, "ranking": ranking, "rank_depth": len(ranking),
            # DEPLOYED deployed top-20: issuer-DEDUPED (what actually ships), vs
            # `ranking` (=pool_after_norm) which is UNDEDUPED and used for the depth-cut
            # diagnostic.  The beat-rate operational readout uses this deduped list so
            # its number reflects the shipped filter, not the raw pool.
            "top20_deduped": res.get("top20", ranking[:20]),
            "universe_size": len(uni), "n_pit_scored": n_pit,
            "n_pit_live": n_pit_live, "n_pit_dead": n_pit - n_pit_live,
        }
        log(f"[{wid}] pit_universe={len(uni)}  n_pit_scored={n_pit} "
            f"(live={n_pit_live}, dead={n_pit - n_pit_live})  rank_depth={len(ranking)}")
    return per_anchor


# --------------------------------------------------------------------------- #
#  Grid computation -- THIN layer over returns_core                           #
# --------------------------------------------------------------------------- #
def _cell_from_returns(rdf, bench, scope, clean, buy, ev, h, N, n_available):
    """Build one grid cell by applying returns_core DERIVED views to a returns table."""
    c = rc.counts(rdf)
    avg_p = rc.average_return(rdf, floor=False)
    avg_f = rc.average_return(rdf, floor=True)
    return {
        "scope": scope, "clean": clean, "buy": buy, "eval": ev,
        "horizon_m": h, "depth_N": N,
        "n_requested": N, "n_available_rank": n_available,
        "n_included": c["n_included"], "n_missing_buy": c["n_no_buy"],
        "n_affected_eval": c["n_terminal"],
        "avg_ret_primary": avg_p, "avg_ret_floor": avg_f, "bench_ret": bench,
        "excess_primary": rc.excess_return(rdf, bench, floor=False),
        "excess_floor": rc.excess_return(rdf, bench, floor=True),
    }


def compute_grid(per_anchor, price_source):
    cells = []
    pool_rows = []  # per-pick rows tagged with rank_pos, for pooled reconstruction

    for wid, buy in BUY_ANCHORS:
        if wid not in per_anchor:
            continue
        ranking = per_anchor[wid]["ranking"]
        buy_idx = ANCHOR_IDX[buy]
        clean = wid in CLEAN_BUY_IDS

        for h in HORIZONS:
            eval_idx = buy_idx + h // 12
            if eval_idx >= len(ANCHORS):
                continue  # horizon unsupported (no eval anchor in data)
            ev = ANCHORS[eval_idx]
            bench = rc.benchmark_return(price_source, buy, ev)

            # ONE primitive call over the deepest slice; depth views slice this table.
            top = ranking[:max(DEPTHS)]
            rdf_full = rc.compute_returns(top, buy, ev, price_source)

            for N in DEPTHS:
                rdf = rdf_full.iloc[:N]
                cells.append(_cell_from_returns(
                    rdf, bench, wid, clean, buy, ev, h, N, n_available=len(rdf)))

            # pooling rows (a pick at rank r feeds every pooled N>=r)
            for pos, row in enumerate(rdf_full.itertuples(index=False), start=1):
                pool_rows.append({
                    "wid": wid, "clean": clean, "horizon_m": h, "rank_pos": pos,
                    "status": row.status, "ret_primary": row.total_return,
                    "ret_floor": row.total_return_floor, "bench": bench,
                })

    pooled = _pool(pool_rows, clean_only=False)
    pooled_clean = _pool(pool_rows, clean_only=True)
    return cells, pooled, pooled_clean


def _pool(pool_rows, clean_only):
    df = pd.DataFrame(pool_rows)
    out = []
    if df.empty:
        return out
    if clean_only:
        df = df[df["clean"]]
    for h in HORIZONS:
        dh = df[df["horizon_m"] == h]
        if dh.empty:
            continue
        n_anchors = dh["wid"].nunique()
        for N in DEPTHS:
            sel = dh[dh["rank_pos"] <= N]
            incl = sel[sel["status"] != "no_buy"]
            n_missing_buy = int((sel["status"] == "no_buy").sum())
            n_affected = int((incl["status"] == "terminal").sum())
            avg_p = float(incl["ret_primary"].mean()) if len(incl) else float("nan")
            avg_f = float(incl["ret_floor"].mean()) if len(incl) else float("nan")
            avg_bench = float(incl["bench"].mean()) if len(incl) else float("nan")
            out.append({
                "scope": ("POOLED-CLEAN" if clean_only else "POOLED-ALL"),
                "clean": clean_only, "buy": f"{n_anchors} anchors", "eval": "",
                "horizon_m": h, "depth_N": N,
                "n_requested": N, "n_available_rank": int(len(sel)),
                "n_included": int(len(incl)), "n_missing_buy": n_missing_buy,
                "n_affected_eval": n_affected,
                "avg_ret_primary": avg_p, "avg_ret_floor": avg_f, "bench_ret": avg_bench,
                "excess_primary": (avg_p - avg_bench) if avg_p == avg_p else float("nan"),
                "excess_floor": (avg_f - avg_bench) if avg_f == avg_f else float("nan"),
            })
    return out


# --------------------------------------------------------------------------- #
#  Reporting                                                                  #
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"{x*100:7.1f}%" if x == x else "    n/a"


def _grid_block(title, cells_subset, note=""):
    lines = ["", "-" * 100, title]
    if note:
        lines.append(note)
    lines.append("-" * 100)

    horizons = sorted({c["horizon_m"] for c in cells_subset})
    if not horizons:
        lines.append("  (no supported horizons for this scope)")
        return lines

    bl_line = "  BENCHMARK URTH (MSCI World TR proxy):  "
    for h in horizons:
        c = next((c for c in cells_subset if c["horizon_m"] == h), None)
        bl_line += f"  {h}mo={_pct(c['bench_ret']).strip()}"
    lines.append(bl_line)
    lines.append("")

    hdr = f"  {'depth':>6} |"
    for h in horizons:
        hdr += f"  {'avgTR('+str(h)+'mo)':>13}  {'excess':>9}  {'flrTR':>9}  {'[incl/aff/nobuy]':>17} |"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for N in DEPTHS:
        row = f"  {N:>6} |"
        any_cell = False
        for h in horizons:
            c = next((c for c in cells_subset
                      if c["horizon_m"] == h and c["depth_N"] == N), None)
            if c is None:
                row += f"  {'-':>13}  {'-':>9}  {'-':>9}  {'-':>17} |"
                continue
            any_cell = True
            cap = "*" if c["n_available_rank"] < c["n_requested"] else " "
            counts = f"[{c['n_included']}/{c['n_affected_eval']}/{c['n_missing_buy']}]{cap}"
            row += (f"  {_pct(c['avg_ret_primary'])}  {_pct(c['excess_primary'])[:9]}"
                    f"  {_pct(c['avg_ret_floor'])[:9]}  {counts:>17} |")
        if any_cell:
            lines.append(row)
    lines.append("  legend: avgTR=equal-weight avg TOTAL return (primary terminal policy); "
                 "excess=avgTR-benchmark;")
    lines.append("          flrTR=avg TOTAL return under -100% floor for delisted/missing-eval names;")
    lines.append("          [incl/aff/nobuy]=names included / affected-by-terminal-policy / "
                 "excluded-for-missing-buy;  * = ranking shallower than N (n_available<N).")
    return lines


def build_report(per_anchor, cells, pooled, pooled_clean):
    lines = []
    lines.append("=" * 100)
    lines.append("DEPTH x HORIZON  AVERAGE TOTAL-RETURN GRID  (survivorship-clean)")
    lines.append("  Diagnostic of RETURN CONCENTRATION by pick-depth -- NOT the beat-rate.")
    lines.append("  Built on the returns_core.compute_returns primitive (returns-first architecture).")
    lines.append(f"  Generated {_dt.datetime.utcnow().isoformat()}Z  |  offline, no network")
    lines.append("=" * 100)
    lines.append("RETURN DEFINITION: total_return = adjClose_eval / adjClose_buy - 1.")
    lines.append("  adjClose = FMP split+DIVIDEND-adjusted close (same series the beat-rate uses)")
    lines.append("  => dividend-INCLUSIVE total return.  Benchmark = " + rc.BENCHMARK_VARIANT + ".")
    lines.append("")
    lines.append("TERMINAL-VALUE POLICY (load-bearing) for picks MISSING their eval-leg price:")
    lines.append("  PRIMARY: terminal = last available adjClose in real_prices before eval anchor.")
    lines.append("  FLOOR  : terminal = 0  ->  -100% total loss.   Both averages reported per cell.")
    lines.append("  MISSING BUY price -> pick EXCLUDED from that bucket (counted as nobuy).")
    lines.append("  NOTE: 'affected' conflates TRUE delisting with real_prices exchange-coverage")
    lines.append("        GAPS (.L/.DE/.ST/.T/.TO absent at some anchors) -- see caveats.")
    lines.append("")

    lines.append("UNIVERSE & DEGENERACY FLAG (per buy anchor):")
    lines.append(f"  {'anchor':7} {'buy_date':11} {'pit_universe':>12} {'n_pit_scored':>12} "
                 f"{'live':>6} {'dead':>6} {'rank_depth':>10}  flag")
    for wid, buy in BUY_ANCHORS:
        if wid not in per_anchor:
            lines.append(f"  {wid:7} {buy:11}  (ranking unavailable)")
            continue
        a = per_anchor[wid]
        flag = "CLEAN" if wid in CLEAN_BUY_IDS else "DEGENERATE*"
        lines.append(f"  {wid:7} {buy:11} {a['universe_size']:>12} {a['n_pit_scored']:>12} "
                     f"{a['n_pit_live']:>6} {a['n_pit_dead']:>6} {a['rank_depth']:>10}  {flag}")
    lines.append("  * DEGENERATE (pre-2021): only a non-representative long-history subset")
    lines.append("    (~1,000-1,250 names) is available PIT, so a 'top-100' is a large slice of the")
    lines.append("    scored pool -- stage-1 barely filters and the DEPTH cut (top-3 vs top-100) is")
    lines.append("    NOT meaningful.  The depth-concentration question is only truly answerable on")
    lines.append("    the CLEAN windows (buy2021/2022/2023/2024), where the PIT pool is thousands of")
    lines.append("    names.")
    lines.append("")

    for wid, buy in BUY_ANCHORS:
        subset = [c for c in cells if c["scope"] == wid]
        if not subset:
            continue
        flag = "CLEAN" if wid in CLEAN_BUY_IDS else "DEGENERATE (depth cut not meaningful)"
        lines += _grid_block(f"ANCHOR {wid}  (buy {buy})   [{flag}]", subset)

    lines += _grid_block("POOLED -- ALL anchors (per horizon)  [includes DEGENERATE windows]",
                         pooled,
                         note="  Pools every pick across all anchors supporting the horizon. "
                              "Depth-signal CONTAMINATED by degenerate pre-2021 windows.")
    lines += _grid_block("POOLED -- CLEAN anchors only (buy2021/2022/2023/2024)  [the defensible read]",
                         pooled_clean,
                         note="  The depth-concentration question answered on populated PIT "
                              "universes only.")

    lines.append("")
    lines.append("=" * 100)
    lines.append("CAVEATS (do not launder):")
    lines.append("  * PROVISIONAL universe: built on the survivorship-clean dead-merge pipeline;")
    lines.append("    inherits its status.  Returns use REAL adjClose; SELECTION (which names rank)")
    lines.append("    inherits stage-2's synthetic-price scoring caveat (3 low-weight metrics) +")
    lines.append("    dropped DcfToPrice + CycleHeat beta=1.0.")
    lines.append("  * PRICE-DATA COVERAGE DEFECT (input real_prices.csv, NOT this harness): bulk-by-")
    lines.append("    date pulls hit inconsistent exchange batches across anchors (.L/.DE/.ST/.T/.TO")
    lines.append("    absent at some year-ends).  A missing eval-leg from a coverage gap is counted")
    lines.append("    as 'affected' identically to a true delisting -> the affected COUNT is an")
    lines.append("    UPPER bound on true delistings, and the primary-vs-floor spread over-states")
    lines.append("    delisting risk where the miss is really a coverage gap.  Inspect exchange")
    lines.append("    suffixes of affected names before drawing a survivorship conclusion.")
    lines.append("  * MISSING-BUY names are EXCLUDED (not failed) -> a mild survivorship re-entry on")
    lines.append("    the buy leg; watch the nobuy counts, especially on gap-affected exchanges.")
    lines.append("  * A precise delisting terminal needs the death-price series (separate re-fetch);")
    lines.append("    the primary policy (last pre-eval anchor price) is an APPROXIMATION.")
    lines.append("  * DEGENERATE pre-2021 windows: depth cut not meaningful (see flag table).")
    lines.append("  * URTH TR-proxy carries ~0.7pp/36mo tracking drag vs MSCI World Net TR --")
    lines.append("    immaterial against these return magnitudes.")
    lines.append("  * Windows overlap heavily (one macro regime); directional, not a large sample.")
    lines.append("  * Equal-weight across picks (no market-cap / liquidity weighting, no tx costs).")
    lines.append("=" * 100)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pickle", default=DEFAULT_PICKLE)
    ap.add_argument("--dead", default=DEFAULT_DEAD)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--prices", default=DEFAULT_PRICES)
    ap.add_argument("--prices-2025", dest="prices_2025", default=DEFAULT_PRICES_2025,
                    help="supplementary 2025 price CSV merged in-memory for the "
                         "2025-12-31 anchor (canonical real_prices.csv untouched); "
                         "pass '' to disable")
    ap.add_argument("--out", required=True, help="human-readable grid output file")
    ap.add_argument("--csv", default=None, help="optional flat CSV of all cells")
    ap.add_argument("--weights", choices=["default", "equal"], default="default",
                    help="metric weight vector (default=production, equal=all weights 1)")
    ap.add_argument("--carve", choices=["off", "on"], default="off",
                    help="filter universe to carveOut general pool BEFORE ranking")
    #  OUTCOME-VARIABLE ROUTE.  'real' is the default and is bit-for-bit unchanged, so the
    #  certified grid is untouched unless this is passed explicitly.  'derived' swaps in
    #  the panel-based TOTAL-RETURN leg (deeper coverage, dividend-inclusive) -- see
    #  derived_prices.py.  Scoring is NOT affected either way: this only changes what the
    #  return is measured against, never how a name is ranked.
    ap.add_argument("--price-route", dest="price_route",
                    choices=list(dpx.PRICE_ROUTES), default="real",
                    help="outcome-variable price source: 'real' = real_prices*.csv "
                         "(default, unchanged); 'derived' = panel total-return leg")
    ap.add_argument("--panel", default=dpx.DEFAULT_PANEL_GLOB,
                    help="fundamentals panel (pickle path or glob) for --price-route=derived")
    ap.add_argument("--derived-max-lag-days", dest="derived_max_lag_days", type=int,
                    default=None,
                    help="derived route only: withhold a pick staler than N days before "
                         "the anchor (raises fidelity, costs names; default: keep all)")
    #  GUARD TOGGLES -- present so the guard progression is reproducible from the CLI, NOT
    #  because turning a guard off is a supported configuration.  Each default-ON guard
    #  fixes a measured defect; `--derived-no-*` reinstates that defect on purpose.
    ap.add_argument("--derived-no-currency-match", dest="derived_no_currency_match",
                    action="store_true",
                    help="DEFECT REPRODUCTION: stop requiring reportedCurrency == listing "
                         "currency, i.e. compare a reporting-ccy return against a "
                         "listing-ccy adjClose on ~20%% of names")
    ap.add_argument("--derived-no-backfill-reject", dest="derived_no_backfill_reject",
                    action="store_true",
                    help="DEFECT REPRODUCTION: keep rows whose price exactly repeats the "
                         "previous period's (pre-listing backfill)")
    ap.add_argument("--derived-max-period-yield", dest="derived_max_period_yield",
                    type=float, default=dpx.MAX_PERIOD_YIELD,
                    help="derived route only: per-period yield reject ceiling "
                         f"(default {dpx.MAX_PERIOD_YIELD})")
    ap.add_argument("--derived-yield-denominator", dest="derived_yield_denominator",
                    choices=["start", "end"], default="start",
                    help="DEFECT REPRODUCTION when 'end': take the per-period yield over "
                         "period-END market cap, which overstates the yield of any payer "
                         "whose price fell during the period")
    args = ap.parse_args()

    log = lambda *a: print(*a, file=sys.stderr, flush=True)
    for label, path in (("pickle", args.pickle), ("dead", args.dead),
                        ("registry", args.registry), ("prices", args.prices)):
        if not os.path.exists(path):
            log(f"FATAL: missing {label} input: {path}")
            sys.exit(2)

    supp = args.prices_2025 or None
    if supp is not None and not os.path.exists(supp):
        log(f"FATAL: missing prices-2025 input: {supp}")
        sys.exit(2)
    #  The real leg is built either way: on the derived route it still supplies the URTH
    #  benchmark, which can never be derived (the panel holds no ETF rows).
    uses_panel = args.price_route in ("derived", "derived+real")
    price_source = dpx.build_price_source(
        args.price_route, prices_csv=args.prices, supp_csv=supp,
        panel=(args.panel if uses_panel else None),
        **({"max_lag_days": args.derived_max_lag_days,
            "require_listing_currency_match": not args.derived_no_currency_match,
            "reject_repeated_price": not args.derived_no_backfill_reject,
            "max_period_yield": args.derived_max_period_yield,
            "yield_denominator": args.derived_yield_denominator} if uses_panel else {}))
    log(f"[price-source] route={args.price_route}")
    if uses_panel:
        #  SURFACE the timing noise rather than hiding it: ~9-10% of names sit more than
        #  45 days before the anchor, and that is a property of the leg the reader of a
        #  grid needs in front of them.
        d = price_source.diagnostics()
        for k, v in d.items():
            if k != "ic_caveat":
                log(f"[price-source]   {k} = {v}")
        log(f"[price-source]   !! {d['ic_caveat']}")
        log("[price-source] anchor timing (period-end lag vs the anchor's Dec-31):")
        for line in price_source.timing_report().to_string(index=False).splitlines():
            log(f"[price-source]   {line}")
    inputs = load_inputs(args.pickle, args.dead, args.registry, log)
    per_anchor = rank_all_anchors(inputs, log, weights=args.weights, carve=args.carve)
    cells, pooled, pooled_clean = compute_grid(per_anchor, price_source)

    text = build_report(per_anchor, cells, pooled, pooled_clean)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    log(f"grid written to: {args.out}")

    if args.csv:
        pd.DataFrame(cells + pooled + pooled_clean).to_csv(args.csv, index=False)
        log(f"csv written to: {args.csv}")


if __name__ == "__main__":
    main()
