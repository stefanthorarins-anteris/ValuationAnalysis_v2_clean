"""
END-TO-END SURVIVORSHIP-CLEAN BEAT-RATE HARNESS  (offline, network-free).

Produces the CEO's headline number:
    ">= 60% of the top-20 picks beat MSCI World by >= 10pp over a 36-month hold."

...on the SURVIVORSHIP-CLEAN universe (live survivors + dead-but-alive-at-D names),
for buy anchors 2018 / 2019 / 2020 / 2021 year-ends (eval 2021 / 2022 / 2023 / 2024).

PROVISIONAL: the dead-name merge (dead_merge.py) is under independent review in
parallel; this number is NOT yet correctness-certified.  Labelled as such in output.

PIPELINE (all consumed as interfaces -- NOT re-implemented here):
  dead_merge.merge_dead_into_dmdic(dmdic, dead, registry, as_of=D)
                                                   -> merged dmdic + pit_universe
  stage2_pit.reproduce_pit_top(merged, D,
      universe_override=merged['pit_universe'])     -> PIT top-20 INCLUDING dead names
  beat_rate.compute_beat_rate(bundle_dir, missing='fail')  -> the beat-rate math
Prices come from baseline_tools/price_data/real_prices.csv (bulk-by-date pulls).
Benchmark is URTH (iShares MSCI World ETF) adjClose, TR-proxy for MSCI World Net TR.

This script does NO network I/O and never prints the pickle's api_key.

Run:  python baseline_tools/run_target_test.py
"""

import argparse
import datetime as _dt
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import bundle_spec as bspec
import beat_rate as br
import dead_merge as dm
import stage2_pit as s2

# --------------------------------------------------------------------------- #
#  Default inputs (all present locally; NONE cross git -- pickles are ~125MB). #
#  Overridable via CLI so the script is turnkey on either machine.            #
# --------------------------------------------------------------------------- #
_HOME = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
DEFAULT_PICKLE = os.path.join(
    _HOME, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-07-13_len7879_manelim3692_fails1966.pickle")
DEFAULT_DEAD = os.path.join(_HOME, "delisted_out", "dead_fundamentals_20260713_104350.pickle")
DEFAULT_REGISTRY = os.path.join(_HOME, "delisted_out", "delisted_registry.csv")
DEFAULT_PRICES = os.path.join(_HERE, "price_data", "real_prices.csv")

BENCHMARK_SYMBOL = "URTH"
BENCHMARK_VARIANT = ("URTH (iShares MSCI World ETF) adjClose, "
                     "TR-proxy for MSCI World Net TR USD")

# (buy anchor date_requested, eval anchor date_requested) -- 36-month hold.
# Dates are the real_prices `date_requested` year-end anchors (all carry URTH).
WINDOWS = [
    ("buy2018", "2018-12-31", "2021-12-31"),
    ("buy2019", "2019-12-31", "2022-12-30"),
    ("buy2020", "2020-12-31", "2023-12-29"),
    ("buy2021", "2021-12-31", "2024-12-31"),
]

# Settled lookahead policy: SYMMETRIC (both cohorts sliced by the same quarter-start
# `date <= D`).  Enforced inside dead_merge/stage2_pit; recorded here for the manifest
# only -- there is no per-cohort switch to pass any more.
AVAILABILITY_MODE = "symmetric (quarter-start date<=D, both cohorts)"

# NOTE: the former sanitize_dead() int->float bridge has been REMOVED -- that dtype
# coercion now lives INSIDE dead_merge (_floatify, applied per entity before the calc),
# so the crash on int64-zero priceEarningsToGrowthRatio (PMA/MSW) is handled at source.


# --------------------------------------------------------------------------- #
#  DELIVERABLE 1 -- benchmark.csv materialization from URTH.                   #
# --------------------------------------------------------------------------- #
def build_benchmark_df(prices_csv, symbol=BENCHMARK_SYMBOL):
    """Return a DataFrame with bundle_spec BENCHMARK_COLS (date, level) from the
    real_prices rows where symbol==URTH: date <- date_actual, level <- adjClose."""
    rp = pd.read_csv(prices_csv, usecols=["date_actual", "symbol", "adjClose"])
    b = rp[rp["symbol"] == symbol].copy()
    if b.empty:
        raise RuntimeError(f"benchmark symbol {symbol!r} not found in {prices_csv}")
    out = pd.DataFrame({
        "date": pd.to_datetime(b["date_actual"]).dt.strftime("%Y-%m-%d"),
        "level": pd.to_numeric(b["adjClose"], errors="coerce"),
    }).dropna().sort_values("date").reset_index(drop=True)
    assert list(out.columns) == bspec.BENCHMARK_COLS, "benchmark schema drift"
    return out


# --------------------------------------------------------------------------- #
#  Price lookup: (symbol, anchor) -> (date_actual, adjClose)                   #
# --------------------------------------------------------------------------- #
def build_price_lookup(prices_csv, anchors):
    rp = pd.read_csv(prices_csv, usecols=["date_requested", "date_actual", "symbol", "adjClose"])
    rp = rp[rp["date_requested"].isin(anchors)]
    lut = {}
    for r in rp.itertuples(index=False):
        # keep first occurrence per (symbol, anchor)
        key = (r.symbol, r.date_requested)
        if key not in lut:
            lut[key] = (r.date_actual, float(r.adjClose))
    return lut


def _exchange_suffix(symbol):
    return symbol.split(".", 1)[1] if "." in str(symbol) else ""


# --------------------------------------------------------------------------- #
#  DELIVERABLE 2 -- production BUNDLE PACKER.                                  #
#  Per window: merged PIT top-20 + real_prices -> selections.csv / prices.csv. #
# --------------------------------------------------------------------------- #
def build_bundle(bundle_dir, dmdic, dead, registry, prices_csv, windows=WINDOWS,
                 log=print):
    os.makedirs(bundle_dir, exist_ok=True)
    live_sources = set(dmdic["cdx_df"]["source"].dropna().unique())

    anchors = sorted({a for _, b, e in windows for a in (b, e)})
    plut = build_price_lookup(prices_csv, anchors)

    sel_rows, price_rows = [], []
    per_window_meta = {}

    for wid, buy, evd in windows:
        log(f"[{wid}] merging dead names as-of {buy} ...")
        merged, mstats = dm.merge_dead_into_dmdic(
            dmdic, dead, registry, as_of=buy)
        log(f"[{wid}] merge: universe={mstats.get('universe_size')} "
            f"built_dead={mstats.get('built')} gate_fail={mstats.get('gate_fail')}")
        log(f"[{wid}] reproducing PIT top-20 (survivorship-clean) ...")
        res = s2.reproduce_pit_top(merged, buy, universe_override=merged["pit_universe"])
        if res is None:
            raise RuntimeError(f"[{wid}] reproduce_pit_top returned None (empty PIT frame)")
        top20 = res["top20"]

        # AggScore per name (best-effort, from the reproduced postRank)
        pr = res["postRank"]
        agg = dict(zip(pr["source"], pr["AggScore"])) if "AggScore" in pr.columns else {}
        s1 = {s: i + 1 for i, s in enumerate(res.get("stage1_top100", []))}

        n_dead = 0
        for rank, src in enumerate(top20, start=1):
            is_dead = src not in live_sources
            n_dead += int(is_dead)
            suf = _exchange_suffix(src)
            sel_rows.append({
                "window_id": wid, "buy_date": buy, "eval_date": evd, "source": src,
                "stage1_rank": s1.get(src, np.nan), "stage2_rank": rank,
                "BoScore": np.nan, "AggScore": agg.get(src, np.nan),
                "is_top20": True,
                "currency": "USD" if suf == "" else "",
                "non_usd_flag": bool(suf != ""),
                "exchange_suffix": suf,
                "dead_cohort": bool(is_dead),  # diagnostic (ignored by beat_rate math)
            })
            for leg, anchor in (("buy", buy), ("eval", evd)):
                hit = plut.get((src, anchor))
                if hit is not None:
                    price_rows.append({
                        "window_id": wid, "source": src, "leg": leg,
                        "date_actual": hit[0], "adjClose": hit[1],
                    })
        per_window_meta[wid] = {"n_top20": len(top20), "n_dead_cohort": n_dead,
                                "buy": buy, "eval": evd,
                                "universe_size": mstats.get("universe_size")}

    sel = pd.DataFrame(sel_rows)
    prices = pd.DataFrame(price_rows, columns=bspec.PRICES_COLS)
    bench = build_benchmark_df(prices_csv)

    # column order: SELECTIONS_COLS first (schema contract), diagnostics appended
    sel = sel[[c for c in bspec.SELECTIONS_COLS if c in sel.columns]
              + [c for c in sel.columns if c not in bspec.SELECTIONS_COLS]]

    sel.to_csv(os.path.join(bundle_dir, "selections.csv"), index=False)
    prices.to_csv(os.path.join(bundle_dir, "prices.csv"), index=False)
    bench.to_csv(os.path.join(bundle_dir, "benchmark.csv"), index=False)

    manifest = {
        "generated_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "source_pickle": os.path.basename(DEFAULT_PICKLE),
        "config": {
            "windows": [{"window_id": w, "buy": b, "eval": e} for w, b, e in windows],
            "hold_months": 36, "topn_final": 20,
            "availability_mode": AVAILABILITY_MODE,
            "universe": "survivorship-clean (live survivors + dead-alive-at-D)",
            "missing_policy": "fail",
        },
        "stage2_fidelity": {
            "dcf_to_price": "dropped",
            "cyclheat_beta": "const_1.0",
            "boscore_average_basis": "pit",
            "as_reported": False,
            "synthetic_price": ("Stage-2 SCORING uses synthetic price "
                                "(PE*EPS); RETURN legs use REAL adjClose from real_prices.csv"),
        },
        "dedup_version": "entity_id (dead_merge join key)",
        "benchmark_variant": BENCHMARK_VARIANT,
        "n_api_calls": 0,
    }
    with open(os.path.join(bundle_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    ok, problems = bspec.validate_bundle(bundle_dir)
    if not ok:
        raise RuntimeError(f"bundle failed structural validation: {problems}")
    return per_window_meta


# --------------------------------------------------------------------------- #
#  DELIVERABLE 3/4 -- run + report.                                           #
# --------------------------------------------------------------------------- #
def _fmt_pct(x):
    return f"{x*100:5.1f}%" if x == x else "  n/a"


#  A window is flagged COVERAGE-GAP when a large share of its top-20 lacks a
#  buy- or eval-leg price in real_prices.csv -- symptomatic of the exchange-batch
#  coverage inconsistency across anchors (.L absent at 2020-12-31 & 2022-12-30;
#  .DE/.ST/.T present only at 2020/2022; etc.). Such a window's beat-rate is NOT
#  interpretable: missing-eval names auto-FAIL and missing-buy names drop out.
_COVERAGE_GAP_MAX = 8  # > 40% of a 20-name shortlist missing a leg


def report(bundle_dir, per_window_meta, threshold=0.10, missing="fail"):
    result = br.compute_beat_rate(bundle_dir, threshold=threshold, missing=missing)
    detail = result["detail"]

    lines = []
    lines.append("=" * 84)
    lines.append("SURVIVORSHIP-CLEAN BEAT-RATE  --  *** PROVISIONAL ***")
    lines.append("  PENDING INDEPENDENT REVIEW OF THE DEAD-MERGE (dead_merge.py, in review).")
    lines.append("  This number is NOT yet correctness-certified.")
    lines.append("=" * 84)
    lines.append(f"Bar: beat {BENCHMARK_VARIANT}")
    lines.append(f"     by >= {threshold*100:.0f}pp over a 36-month hold.  "
                 f"Missing-eval policy: {missing}  (delisted-at-eval => FAIL).")
    lines.append("")
    hdr = (f"  {'window':7} {'buy->eval':23} {'picks':>5} {'dead':>5} "
           f"{'miss_buy':>8} {'miss_eval':>9} {'n_eval':>6} {'beat':>6} {'bench':>7}  "
           f"{'verdict':7} coverage")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    ok_windows = []
    for wid, b, e in [(w[0], w[1], w[2]) for w in WINDOWS]:
        w = result["per_window"][wid]
        grp = detail[detail["window_id"] == wid]
        meta = per_window_meta[wid]
        miss_buy = int(grp["missing_buy_price"].sum())
        miss_eval = w["n_missing_eval"]
        gap = (miss_buy > _COVERAGE_GAP_MAX) or (miss_eval > _COVERAGE_GAP_MAX)
        cov = "GAP*" if gap else "ok"
        if not gap:
            ok_windows.append(wid)
        beat = w["beat_rate"]
        verdict = "PASS" if (beat == beat and beat >= 0.60) else "FAIL"
        lines.append(
            f"  {wid:7} {b+' -> '+e:23} {meta['n_top20']:>5} "
            f"{meta['n_dead_cohort']:>5} {miss_buy:>8} {miss_eval:>9} "
            f"{w['n_evaluated']:>6} {_fmt_pct(beat):>6} "
            f"{_fmt_pct(w['bench_return']):>7}  {verdict:7} {cov}")

    lines.append("  " + "-" * (len(hdr) - 2))
    pooled = result["pooled_beat_rate"]
    n_dead_total = sum(m["n_dead_cohort"] for m in per_window_meta.values())
    pooled_verdict = "PASS" if result["passes_target"] else "FAIL"
    lines.append(
        f"  POOLED  {'(all 4 windows)':23} "
        f"{sum(m['n_top20'] for m in per_window_meta.values()):>5} "
        f"{n_dead_total:>5} "
        f"{int(detail['missing_buy_price'].sum()):>8} "
        f"{int(detail['missing_eval_price'].sum()):>9} "
        f"{result['pooled_n']:>6} {_fmt_pct(pooled):>6} {'':>7}  {pooled_verdict:7}")

    # coverage-clean pooled (windows with consistent price coverage only)
    clean = detail[detail["window_id"].isin(ok_windows)]
    clean_flags = clean["beat"].dropna()
    clean_rate = float(clean_flags.mean()) if len(clean_flags) else float("nan")
    clean_verdict = "PASS" if (clean_rate == clean_rate and clean_rate >= 0.60) else "FAIL"
    lines.append("")
    lines.append(f"  POOLED, ALL 4 windows       : {_fmt_pct(pooled)}  "
                 f"(n_eval={result['pooled_n']})  -> {pooled_verdict}  "
                 f"[CONTAMINATED by coverage-GAP windows -- do not use as the headline]")
    lines.append(f"  POOLED, COVERAGE-CLEAN only : {_fmt_pct(clean_rate)}  "
                 f"(n_eval={len(clean_flags)}; windows {ok_windows})  -> {clean_verdict}  "
                 f"[the defensible read]   TARGET >= 60%")
    lines.append(f"  USD-only pooled (all)       : {_fmt_pct(result['usd_only_beat_rate'])} "
                 f"(n={result['usd_only_n']}; {result['n_non_usd']} non-USD names, FX deferred)")
    lines.append("")
    lines.append("CAVEATS (do not launder):")
    lines.append("  * PROVISIONAL: dead-merge under independent review; not correctness-certified.")
    lines.append("  * PRICE-DATA COVERAGE DEFECT (input real_prices.csv, NOT the harness): the")
    lines.append("    bulk-by-date pulls hit inconsistent exchange batches across anchors --")
    lines.append("    .L absent at 2020-12-31 & 2022-12-30; .DE/.ST/.T present only at 2020/2022;")
    lines.append("    .TO absent at 2020/2022. -> buy2019 (eval 2022-12-30) auto-FAILs ~15 .L names")
    lines.append("    on missing eval-legs; buy2020 (buy 2020-12-31) drops 17 .L names for want of")
    lines.append("    a buy-leg. Those two windows are NOT interpretable; only buy2018 & buy2021 sit")
    lines.append("    on coverage-consistent anchor pairs. Re-pull those two anchors to repair.")
    lines.append("  * MISSING-BUY names are EXCLUDED from the denominator (beat_rate rule), not")
    lines.append("    failed -- a mild survivorship re-entry on the buy leg; watch miss_buy counts.")
    lines.append("  * URTH TR-proxy carries ~0.7pp/36mo tracking drag vs MSCI World Net TR --")
    lines.append("    immaterial against the 10pp bar.")
    lines.append("  * Windows valid only 2016+ (delisted registry is vendor-thin before ~2016).")
    lines.append("  * Synthetic-price caveat: Stage-2 SCORING uses synthetic price for 3 low-weight")
    lines.append("    metrics; the RETURN legs use REAL adjClose, so the beat-rate itself is on real")
    lines.append("    prices. Selection (which names) inherits the synthetic-price scoring caveat.")
    lines.append("  * Windows overlap heavily (one macro regime); directional, not a large sample.")
    lines.append("=" * 84)
    return "\n".join(lines), result



# --------------------------------------------------------------------------- #
#  PANEL-BASIS GUARD (domain review B1, added 2026-07-26)                     #
# --------------------------------------------------------------------------- #
# This harness produces the charter's PRIMARY criterion (the >=60% beat-rate), and it now
# has a way to be silently WRONG: `dead_merge` calls the LIVE `fillPreReqdf`, so DEAD names
# get today's price (marketCap/weightedAverageShsOut) and today's in-pipeline Graham, while
# a pre-2026-07-19 saved panel's LIVE rows still carry the OLD divided price and FMP's
# quarterly grahamNumber.  Measured on the 07-13 panel: price new/old median 4.0000,
# grahamNumber new/old median 1.9154, so uGrahamNumberToPrice runs 0.4789x on the dead side.
# Stage-2 is z-scored across the pool, so a subpopulation basis split is NOT invariant:
# tbVpRatio -0.844 sigma and grahamNumberToPrice -0.788 sigma, together about -0.054
# AggScore -- roughly 9 rank positions at the top-20 boundary, all of it against dead names,
# which biases the beat-rate UPWARD.
#
# Detection is direct rather than by filename: on a NEW-basis panel
# marketCap / (price * shares) is ~1.0; on an OLD-basis panel it is ~4.0 (the quarterly-PE
# derivation divided the price by ~4).  Refuse by default; --allow-basis-mismatch downgrades
# to a loud banner for someone who explicitly wants the old number.
PRICE_BASIS_TOL = 0.15


def detect_price_basis(cdx_df):
    """('new'|'old'|'unknown', median_ratio) from marketCap / (price * shares)."""
    try:
        mc = pd.to_numeric(cdx_df["marketCap"], errors="coerce")
        pr = pd.to_numeric(cdx_df["price"], errors="coerce")
        sh = pd.to_numeric(cdx_df["weightedAverageShsOut"], errors="coerce")
        r = (mc / (pr * sh)).replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(r) < 100:
            return "unknown", float("nan")
        med = float(r.median())
    except Exception:
        return "unknown", float("nan")
    if abs(med - 1.0) <= PRICE_BASIS_TOL:
        return "new", med
    if abs(med - 4.0) <= 4 * PRICE_BASIS_TOL:
        return "old", med
    return "unknown", med


def assert_panel_basis(dmdic, pickle_path, allow_mismatch=False, log=print):
    """Refuse (or loudly banner) when the panel's basis does not match the live code."""
    basis, med = detect_price_basis(dmdic.get("cdx_df"))
    has_period = "period" in getattr(dmdic.get("cdx_df"), "columns", [])
    msg = ("PANEL BASIS: %s (marketCap/(price*shares) median = %.4f); `period` column %s"
           % (basis, med, "PRESENT" if has_period else "ABSENT"))
    log(msg)
    if basis == "new":
        return True
    bar = "!" * 78
    banner = chr(10).join([
        "", bar,
        "!!! BEAT-RATE PANEL BASIS MISMATCH -- THIS NUMBER WOULD BE INVALID.",
        "!!! " + msg,
        "!!! The panel at",
        "!!!   " + str(pickle_path),
        "!!! carries the OLD price/Graham basis, but dead_merge scores DEAD names",
        "!!! through the LIVE fillPreReqdf (new price + in-pipeline Graham). Dead and",
        "!!! live names would then be on incompatible bases: about -0.054 AggScore,",
        "!!! ~9 rank positions at the top-20 boundary, applied only to dead names,",
        "!!! which biases the beat-rate UPWARD.",
        "!!! FIX: re-point --pickle at a panel fetched with the current code.",
        "!!! Do NOT quote a pre-fetch beat-rate as a 'before' baseline either -- it",
        "!!! carries the same split.",
        bar, "",
    ])
    print(banner, file=sys.stderr, flush=True)
    log(banner)
    if not allow_mismatch:
        raise SystemExit("REFUSING to compute a beat-rate on a basis-mismatched panel "
                         "(pass --allow-basis-mismatch to override deliberately).")
    log("--allow-basis-mismatch given: PROCEEDING on a known-invalid basis.")
    return False

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pickle", default=DEFAULT_PICKLE)
    ap.add_argument("--allow-basis-mismatch", action="store_true",
                    dest="allow_basis_mismatch",
                    help="proceed even if the panel's price/Graham basis does not "
                         "match the live code (the number will be invalid)")
    ap.add_argument("--dead", default=DEFAULT_DEAD)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--prices", default=DEFAULT_PRICES)
    ap.add_argument("--bundle-dir",
                    default=os.path.join(tempfile.gettempdir(), "target_test_bundle"))
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--missing", choices=["fail", "drop", "zero"], default="fail")
    args = ap.parse_args()

    log = lambda *a: print(*a, file=sys.stderr, flush=True)
    for label, path in (("pickle", args.pickle), ("dead", args.dead),
                        ("registry", args.registry), ("prices", args.prices)):
        if not os.path.exists(path):
            log(f"FATAL: missing {label} input: {path}")
            sys.exit(2)

    log("Loading base scoring pickle (offline) ...")
    dmdic = pd.read_pickle(args.pickle)
    # HARD GATE: the panel's price/Graham basis must match the live code (see above).
    assert_panel_basis(dmdic, args.pickle,
                       allow_mismatch=getattr(args, "allow_basis_mismatch", False),
                       log=log)
    log("Loading dead-fundamentals pickle + registry ...")
    dead = pd.read_pickle(args.dead)
    registry = dm.load_registry(args.registry)

    log(f"Building bundle at {args.bundle_dir} ...")
    per_window_meta = build_bundle(args.bundle_dir, dmdic, dead, registry,
                                   args.prices, log=log)

    text, _ = report(args.bundle_dir, per_window_meta,
                     threshold=args.threshold, missing=args.missing)
    print(text)
    log(f"\nbundle written to: {args.bundle_dir}")


if __name__ == "__main__":
    main()
