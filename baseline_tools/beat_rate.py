"""
Beat-rate computation -- the core proxy-target metric, computed HERE, OFFLINE,
as a PURE FUNCTION of the analysis bundle (see bundle_spec.py).

TARGET (charter):
    >= 60% of the filter's TRUE top-20 beat MSCI World by >= 10pp over a
    36-month hold, across the available historical windows.

Per (window, selected top-20 name):
    stock_tr  = eval_adjClose / buy_adjClose - 1     (adjClose is split+div
                adjusted, so this IS a total return -- no separate dividend term)
    bench_tr  = MSCI World total return over the same [buy, eval] window
    beat      = (stock_tr - bench_tr) >= threshold   (threshold default 0.10)

    beat_rate(window) = mean(beat over that window's top-20)
    beat_rate(pooled) = mean(beat over all windows' top-20 pooled)

JUDGMENT CALLS (flagged for reviewer / devils-advocate):
  * MISSING EVAL PRICE (delisted / no data at eval). A stock can vanish by
    acquisition (often a gain) or bankruptcy (a loss). Dropping such names would
    bias the beat-rate UPWARD (survivorship). Default here: missing='fail'
    (counts as NOT beating) and reported separately so the sensitivity is
    visible. Alternatives ('drop', 'zero') are provided but must be justified.
  * NON-USD names: FX is deferred (2026-07-11). We report the beat-rate BOTH
    overall and USD-only, so the FX-noise exposure is explicit rather than
    hidden. A non-USD 36-mo FX swing can be ~the size of the 10pp bar.

This module does NO network I/O and does not touch the big pickles.
"""

import json
import os

import pandas as pd

import benchmark_loader as bl


def _load_bundle_frames(bundle_dir):
    sel = pd.read_csv(os.path.join(bundle_dir, "selections.csv"))
    prices = pd.read_csv(os.path.join(bundle_dir, "prices.csv"))
    bench = bl.load_benchmark(os.path.join(bundle_dir, "benchmark.csv"),
                              date_col="date", level_col="level")
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return sel, prices, bench, manifest


def _price_lookup(prices):
    """(window_id, source, leg) -> adjClose."""
    out = {}
    for r in prices.itertuples(index=False):
        out[(r.window_id, r.source, r.leg)] = float(r.adjClose)
    return out


def compute_returns_table(sel, prices, bench):
    """Build the per-name returns table for the top-20 of every window."""
    plut = _price_lookup(prices)
    top20 = sel[sel["is_top20"].astype(bool)].copy()

    # benchmark return per window (one value each)
    bench_ret = {}
    for wid, grp in top20.groupby("window_id"):
        buy = grp["buy_date"].iloc[0]
        ev = grp["eval_date"].iloc[0]
        bench_ret[wid] = bl.window_return(bench, buy, ev)

    rows = []
    for r in top20.itertuples(index=False):
        wid = r.window_id
        buy_p = plut.get((wid, r.source, "buy"))
        eval_p = plut.get((wid, r.source, "eval"))
        stock_tr = None
        missing_eval = eval_p is None or pd.isna(eval_p) or (eval_p or 0) <= 0
        missing_buy = buy_p is None or pd.isna(buy_p) or (buy_p or 0) <= 0
        if not missing_buy and not missing_eval:
            stock_tr = eval_p / buy_p - 1.0
        rows.append({
            "window_id": wid,
            "source": r.source,
            "stage2_rank": getattr(r, "stage2_rank", None),
            "non_usd_flag": bool(getattr(r, "non_usd_flag", False)),
            "currency": getattr(r, "currency", ""),
            "buy_price": buy_p,
            "eval_price": eval_p,
            "missing_buy_price": missing_buy,
            "missing_eval_price": missing_eval,
            "stock_return": stock_tr,
            "bench_return": bench_ret.get(wid),
        })
    return pd.DataFrame(rows), bench_ret


def _beat_flag(row, threshold, missing):
    if row["missing_buy_price"]:
        # No entry price at all -> the name was never really buyable in-bundle.
        return None  # excluded from denominator regardless of policy
    if row["missing_eval_price"] or pd.isna(row["stock_return"]):
        if missing == "drop":
            return None
        if missing == "zero":
            excess = 0.0 - (row["bench_return"] or 0.0)
            return excess >= threshold
        # 'fail' (default): counts as not beating
        return False
    if pd.isna(row["bench_return"]):
        return None
    excess = row["stock_return"] - row["bench_return"]
    return bool(excess >= threshold)


def compute_beat_rate(bundle_dir, threshold=0.10, missing="fail"):
    """Full offline computation of the proxy target from a bundle directory.

    Returns a dict: per-window rates, pooled rate, USD-only rate, the detail
    table, and the pass/fail against the 60% target.
    """
    sel, prices, bench, manifest = _load_bundle_frames(bundle_dir)
    detail, bench_ret = compute_returns_table(sel, prices, bench)

    detail["beat"] = detail.apply(lambda r: _beat_flag(r, threshold, missing),
                                  axis=1)

    def _rate(df):
        flags = df["beat"].dropna()
        return (float(flags.mean()) if len(flags) else float("nan"), len(flags))

    per_window = {}
    for wid, grp in detail.groupby("window_id"):
        rate, n = _rate(grp)
        per_window[wid] = {
            "beat_rate": rate,
            "n_evaluated": n,
            "n_missing_eval": int(grp["missing_eval_price"].sum()),
            "bench_return": bench_ret.get(wid),
        }

    pooled_rate, pooled_n = _rate(detail)
    usd_only = detail[~detail["non_usd_flag"]]
    usd_rate, usd_n = _rate(usd_only)

    return {
        "threshold_pp": threshold * 100,
        "missing_policy": missing,
        "per_window": per_window,
        "pooled_beat_rate": pooled_rate,
        "pooled_n": pooled_n,
        "usd_only_beat_rate": usd_rate,
        "usd_only_n": usd_n,
        "n_non_usd": int(detail["non_usd_flag"].sum()),
        "target": 0.60,
        "passes_target": (pooled_rate >= 0.60) if pooled_rate == pooled_rate
                         else False,  # NaN-safe
        "detail": detail,
        "manifest": manifest,
    }


def format_report(result):
    """Human-readable summary string (no side effects)."""
    lines = []
    lines.append("=" * 64)
    lines.append("INVESTMENT-FILTER BASELINE -- PROXY TARGET (offline, bundle)")
    lines.append("=" * 64)
    lines.append(f"Bar: beat MSCI World by >= {result['threshold_pp']:.0f}pp "
                 f"over the hold. Missing-eval policy: {result['missing_policy']}")
    lines.append("")
    for wid, w in result["per_window"].items():
        br = w["beat_rate"]
        lines.append(f"  {wid}: beat-rate={br*100:5.1f}%  "
                     f"(n={w['n_evaluated']}, missing_eval={w['n_missing_eval']}, "
                     f"bench_ret={ (w['bench_return'] or float('nan'))*100:5.1f}%)")
    lines.append("")
    lines.append(f"POOLED beat-rate : {result['pooled_beat_rate']*100:5.1f}% "
                 f"(n={result['pooled_n']})   TARGET >= 60%  -> "
                 f"{'PASS' if result['passes_target'] else 'MISS'}")
    lines.append(f"USD-only         : {result['usd_only_beat_rate']*100:5.1f}% "
                 f"(n={result['usd_only_n']}; {result['n_non_usd']} non-USD names "
                 f"flagged lower-confidence, FX deferred)")
    lines.append("")
    lines.append("CAVEATS (do not launder): directional only -- ~3 heavily-"
                 "overlapping windows, one macro regime; survivorship (current-"
                 "tradable universe) biases UP; Stage-2 fidelity flags in manifest.")
    lines.append("=" * 64)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Offline beat-rate from a bundle.")
    ap.add_argument("bundle_dir")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--missing", choices=["fail", "drop", "zero"], default="fail")
    args = ap.parse_args()
    res = compute_beat_rate(args.bundle_dir, args.threshold, args.missing)
    print(format_report(res))
