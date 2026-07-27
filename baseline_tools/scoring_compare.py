"""
N-WAY SCORING COMPARISON on the survivorship-clean depth x horizon return grid.

Holds EVERYTHING constant (dead-merge universe, pickle/registry/prices, clean windows,
returns_core, terminal handling) and varies ONLY the scoring config, so the grids are
apples-to-apples.  Configs:

  ORIGINAL : pre-house-fix scoring   (default weights, un-carved)  -- run on a tree with
             the ranking-affecting correction hunks reverted (a SEPARATE process; this
             script does not do the revert).
  BASELINE : corrected default-weight, un-carved  (== the certified grid).
  CARVE    : corrected default weights, universe filtered to carveOut general pool.
  EQUAL    : all metric weights = 1, un-carved.

Two modes (kept separate so ORIGINAL can run in its own process on reverted code):
  run    : load inputs ONCE, compute the grid for one-or-more configs, dump each
           config's cells to <workdir>/cells_<CONFIG>.csv .
  format : read every <workdir>/cells_*.csv present and emit the side-by-side .out
           (+ a merged CSV).

No network I/O; never prints api_key.

Examples:
  python baseline_tools/scoring_compare.py run --configs baseline,carve,equal --workdir W
  python baseline_tools/scoring_compare.py run --configs original --workdir W   # on reverted tree
  python baseline_tools/scoring_compare.py format --workdir W --out cmp.out --csv cmp.csv
"""

import argparse
import datetime as _dt
import glob
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import returns_core as rc
import depth_horizon_grid as dhg

# config label -> (weights, carve) knobs into rank_all_anchors.
CONFIG_KNOBS = {
    "ORIGINAL": ("default", "off"),   # meaning comes from the REVERTED code it runs on
    "BASELINE": ("default", "off"),
    "CARVE":    ("default", "on"),
    "EQUAL":    ("equal",   "off"),
}
# Display order (left -> right) in the side-by-side.
CONFIG_ORDER = ["ORIGINAL", "BASELINE", "CARVE", "EQUAL"]

DEPTHS = dhg.DEPTHS
HORIZONS = dhg.HORIZONS


def _cells_path(workdir, config):
    return os.path.join(workdir, f"cells_{config}.csv")


def run_configs(configs, workdir, args, log):
    os.makedirs(workdir, exist_ok=True)
    price_source = rc.PriceSource(args.prices)
    inputs = dhg.load_inputs(args.pickle, args.dead, args.registry, log)
    for config in configs:
        weights, carve = CONFIG_KNOBS[config]
        log(f"==== CONFIG {config}  (weights={weights}, carve={carve}) ====")
        per_anchor = dhg.rank_all_anchors(inputs, log, weights=weights, carve=carve)
        cells, pooled, pooled_clean = dhg.compute_grid(per_anchor, price_source)
        df = pd.DataFrame(cells + pooled + pooled_clean)
        df.insert(0, "config", config)
        # attach the per-anchor universe diagnostics (for the flag table)
        diag_rows = [{"config": config, "anchor": wid, **{k: v for k, v in a.items()
                     if k not in ("ranking",)}} for wid, a in per_anchor.items()]
        pd.DataFrame(diag_rows).to_csv(
            os.path.join(workdir, f"diag_{config}.csv"), index=False)
        df.to_csv(_cells_path(workdir, config), index=False)
        log(f"     wrote {_cells_path(workdir, config)} ({len(df)} cells)")


# --------------------------------------------------------------------------- #
#  Formatting the side-by-side                                                #
# --------------------------------------------------------------------------- #
def _pct(x, w=6):
    return f"{x*100:{w}.1f}%" if (x == x and x is not None) else f"{'n/a':>{w}}"


def _signed(x):
    return f"{x*100:+5.1f}" if (x == x and x is not None) else " n/a"


SCOPE_ORDER = [("POOLED-CLEAN", "POOLED, CLEAN anchors 2021/22/23  [HEADLINE / defensible read]"),
               ("buy2021", "ANCHOR buy2021 (buy 2021-12-31)  [CLEAN]"),
               ("buy2022", "ANCHOR buy2022 (buy 2022-12-30)  [CLEAN]"),
               ("buy2023", "ANCHOR buy2023 (buy 2023-12-29)  [CLEAN]"),
               ("POOLED-ALL", "POOLED, ALL anchors  [CONTAMINATED by degenerate pre-2021 -- context only]"),
               ("buy2018", "ANCHOR buy2018 (buy 2018-12-31)  [DEGENERATE]"),
               ("buy2019", "ANCHOR buy2019 (buy 2019-12-31)  [DEGENERATE]"),
               ("buy2020", "ANCHOR buy2020 (buy 2020-12-31)  [DEGENERATE]")]


def _fmt_scope_block(all_df, present_configs, scope, title):
    lines = ["", "-" * 118, title, "-" * 118]
    sdf = all_df[all_df["scope"] == scope]
    if sdf.empty:
        lines.append("  (not present)")
        return lines
    horizons = sorted(sdf["horizon_m"].unique())
    for h in horizons:
        hdf = sdf[sdf["horizon_m"] == h]
        # benchmark (same across configs; take any)
        bench = hdf["bench_ret"].dropna().iloc[0] if hdf["bench_ret"].notna().any() else float("nan")
        lines.append("")
        lines.append(f"  Horizon {h}mo   (URTH benchmark = {_pct(bench).strip()})   "
                     f"cells = avgTR%% (excess vs URTH, pp)")
        hdr = f"  {'depth':>5} |"
        for cfg in present_configs:
            hdr += f"  {cfg:>18} |"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for N in DEPTHS:
            row = f"  {N:>5} |"
            any_cell = False
            for cfg in present_configs:
                c = hdf[(hdf["depth_N"] == N) & (hdf["config"] == cfg)]
                if c.empty:
                    row += f"  {'-':>18} |"
                    continue
                any_cell = True
                r = c.iloc[0]
                cell = f"{_pct(r['avg_ret_primary'],5)}({_signed(r['excess_primary'])})"
                row += f"  {cell:>18} |"
            if any_cell:
                lines.append(row)
    return lines


def format_comparison(workdir, out_path, csv_path, log):
    paths = sorted(glob.glob(os.path.join(workdir, "cells_*.csv")))
    if not paths:
        raise RuntimeError(f"no cells_*.csv found in {workdir}")
    frames = [pd.read_csv(p) for p in paths]
    all_df = pd.concat(frames, ignore_index=True)
    present_configs = [c for c in CONFIG_ORDER if c in set(all_df["config"])]
    log(f"formatting configs present: {present_configs}")

    lines = []
    lines.append("=" * 118)
    lines.append("SCORING-CONFIG COMPARISON  --  depth x horizon AVERAGE TOTAL RETURN "
                 "(survivorship-clean, apples-to-apples)")
    lines.append(f"  Generated {_dt.datetime.utcnow().isoformat()}Z  |  offline, no network")
    lines.append("=" * 118)
    lines.append("Only the SCORING CONFIG varies; dead-merge universe, pickle/registry/prices,")
    lines.append("clean windows, returns_core primitive, and primary-terminal handling are identical.")
    lines.append("")
    lines.append("CONFIGS:")
    lines.append("  ORIGINAL : pre-house-fix scoring (ranking-affecting fixes reverted), default wts, un-carved")
    lines.append("  BASELINE : corrected default-weight, un-carved  (== certified grid)")
    lines.append("  CARVE    : corrected default weights; universe filtered to carveOut general pool pre-rank")
    lines.append("  EQUAL    : all metric weights = 1, un-carved")
    lines.append("")
    lines.append("READS: ORIGINAL->BASELINE = did the house fixes change returns; BASELINE->CARVE = does the")
    lines.append("  carve help; BASELINE vs EQUAL = does weighting beat naive equal weight.")
    lines.append("  Cell = avgTR%(excess-vs-URTH in pp).  avgTR = equal-weight avg TOTAL return, PRIMARY")
    lines.append("  terminal policy.  See per-config grids for floor-policy + affected/nobuy counts.")

    # universe-size sanity line per config (from diag files)
    for cfg in present_configs:
        dpath = os.path.join(workdir, f"diag_{cfg}.csv")
        if os.path.exists(dpath):
            d = pd.read_csv(dpath)
            sizes = ", ".join(f"{r.anchor}={int(r.n_pit_scored)}"
                              for r in d.itertuples(index=False))
            lines.append(f"  [{cfg}] n_pit_scored per anchor: {sizes}")

    for scope, title in SCOPE_ORDER:
        lines += _fmt_scope_block(all_df, present_configs, scope, title)

    lines.append("")
    lines.append("=" * 118)
    lines.append("CAVEATS (do not launder): same as the per-config grids -- PROVISIONAL survivorship")
    lines.append("  dead-merge; real_prices exchange-coverage gaps inflate affected counts; degenerate")
    lines.append("  pre-2021 depth cut not meaningful; URTH TR-proxy; heavily-overlapping windows;")
    lines.append("  equal-weight across picks, no tx costs.  EQUAL flips CycleHeat's -0.5 penalty to +1")
    lines.append("  and zeroes no metric; ORIGINAL is the pre-fix LOGIC on the SAME survivorship universe")
    lines.append("  + SAME real prices (only the scoring correction hunks reverted).")
    lines.append("=" * 118)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    log(f"comparison written to: {out_path}")
    if csv_path:
        all_df.to_csv(csv_path, index=False)
        log(f"merged cells csv: {csv_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="compute + dump cells for one-or-more configs")
    r.add_argument("--configs", required=True,
                   help="comma list of ORIGINAL,BASELINE,CARVE,EQUAL (case-insensitive)")
    r.add_argument("--workdir", required=True)
    r.add_argument("--pickle", default=dhg.DEFAULT_PICKLE)
    r.add_argument("--dead", default=dhg.DEFAULT_DEAD)
    r.add_argument("--registry", default=dhg.DEFAULT_REGISTRY)
    r.add_argument("--prices", default=dhg.DEFAULT_PRICES)

    f = sub.add_parser("format", help="merge dumped cells into the side-by-side output")
    f.add_argument("--workdir", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--csv", default=None)

    args = ap.parse_args()
    log = lambda *a: print(*a, file=sys.stderr, flush=True)

    if args.mode == "run":
        configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
        for c in configs:
            if c not in CONFIG_KNOBS:
                log(f"FATAL: unknown config {c!r}; valid: {list(CONFIG_KNOBS)}")
                sys.exit(2)
        for label, path in (("pickle", args.pickle), ("dead", args.dead),
                            ("registry", args.registry), ("prices", args.prices)):
            if not os.path.exists(path):
                log(f"FATAL: missing {label} input: {path}")
                sys.exit(2)
        run_configs(configs, args.workdir, args, log)
    elif args.mode == "format":
        format_comparison(args.workdir, args.out, args.csv, log)


if __name__ == "__main__":
    main()
