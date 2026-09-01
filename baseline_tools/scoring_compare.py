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

STANDALONE HAND-RUN TOOL -- NOT A PIPELINE STAGE, AND NOTHING HERE EXECUTES ON A RUN.
------------------------------------------------------------------------------------
`pipeline_analysis.run_analysis_suite` does not import this module and no other production
path calls it: grep finds it only in `basis_stamp`'s prose and in tests.  Recorded here
because it stopped being obvious once this module gained BASIS STAMPS on 2026-08-27 -- half
of that batch's work landed in a module the nightly run never touches, and the 08-31 run
review had to establish by grep that it had never executed against real data.  So:

  * its stamps are CORRECT and TESTED (`test_basis_stamp`), and they matter ONLY when a
    person runs this script by hand.  Do not count them as demonstrated on production data.
  * it is deliberately NOT wired in.  A nightly N-way comparison would cost a second, third
    and fourth full PIT reproduction (the single-config grid alone is ~270s on the current
    panel) to answer a question nobody asks every night; the CEO's decisions about weights
    and carve are episodic, and this is the tool reached for when one is live.
  * IF IT IS EVER WIRED IN, this paragraph is wrong and
    `test_basis_stamp.test_scoring_compare_is_still_a_hand_run_tool` will say so.

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
import basis_stamp as bstamp

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


def _build_diagnostic_rows(per_anchor, config):
    """Build diagnostic rows from per_anchor results, excluding nested structures.

    Filters out nested dict/list values that would bloat the CSV with unreadable cells:
    - "ranking": full stage-2 ordering (~100 symbols)
    - "top20_deduped": deduped top-20 list
    - "stage1_veto": veto report dict with rejection flags

    Args:
        per_anchor: dict of anchor_id -> per-anchor-result dict
        config: config name string

    Returns:
        list of dicts suitable for pd.DataFrame()
    """
    return [{"config": config, "anchor": wid, **{k: v for k, v in a.items()
             if k not in ("ranking", "top20_deduped", "stage1_veto")}}
            for wid, a in per_anchor.items()]


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
        #  STAMPED INTO THE CELLS FILE, not only into `diag_`.  `format` is a SEPARATE
        #  invocation reading whatever `cells_*.csv` happen to be in the workdir -- possibly
        #  written by a different process on a differently-versioned tree, which is the whole
        #  reason ORIGINAL exists.  A basis that lives in a sibling file can be stale relative
        #  to the numbers, or absent; travelling in the same row it cannot be.
        df.insert(1, "basis", bstamp.of(per_anchor))
        # attach the per-anchor universe diagnostics (for the flag table)
        diag_rows = _build_diagnostic_rows(per_anchor, config)
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


def _scope_order():
    """CLEAN-first, then pooled-all, then the excluded anchors -- DERIVED from
    `depth_horizon_grid`, not restated here.

    THE HARDCODED VERSION HAD ALREADY DRIFTED, in both directions at once.  Its headline read
    "POOLED, CLEAN anchors 2021/22/23" while `CLEAN_BUY_IDS` had contained buy2024 for weeks,
    so the label under-named the pool it was describing; and buy2024 had NO row of its own at
    all, so the richest-labelled anchor was invisible in the side-by-side.  It also called
    buy2018/19/20 `DEGENERATE`, a word whose stated meaning (a scoring-pickle history cap)
    `-nrperiods 80` has lifted.  A second copy of a set that moves is a second copy that goes
    stale; this reads the set instead.
    """
    clean = [(wid, buy) for wid, buy in dhg.BUY_ANCHORS if wid in dhg.CLEAN_BUY_IDS]
    excl = [(wid, buy) for wid, buy in dhg.BUY_ANCHORS if wid not in dhg.CLEAN_BUY_IDS]
    names = ", ".join(w.replace("buy", "") for w, _ in clean) or "(none)"
    order = [("POOLED-CLEAN",
              f"POOLED, CLEAN anchors {names}  [HEADLINE / defensible read]")]
    order += [(wid, f"ANCHOR {wid} (buy {buy})  [CLEAN]") for wid, buy in clean]
    order.append(("POOLED-ALL",
                  "POOLED, ALL anchors  [includes EXCLUDED anchors -- context only]"))
    #  The reason travels with the label, so a reader of this table sees the same
    #  justification the grid report prints rather than a bare adjective.
    for wid, buy in excl:
        why = (dhg.exclusion_reason(wid) or "").split(":")[0] or "excluded"
        order.append((wid, f"ANCHOR {wid} (buy {buy})  [EXCLUDED -- {why}]"))
    return order


SCOPE_ORDER = _scope_order()


def _fmt_scope_block(all_df, present_configs, scope, title, basis_by_config=None):
    lines = ["", "-" * 118, title, "-" * 118]
    #  REPEATED PER TABLE, not only in the header, because these blocks get copied out of the
    #  .out file one at a time -- a header stamp 200 lines up does not travel with the table
    #  someone pastes into a report.
    if basis_by_config:
        tags = " | ".join("%s=%s" % (c, bstamp.tag(basis_by_config.get(c)))
                          for c in present_configs)
        mixed = len({bstamp.tag(basis_by_config.get(c)) for c in present_configs}) > 1
        lines.append("  BASIS: " + tags
                     + ("   <-- MIXED: columns not comparable" if mixed else ""))
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


def _basis_for_config(all_df, workdir, config):
    """The measurement basis for ONE column of the side-by-side.

    THE DEFECT THIS ADDRESSES IS A SINGLE TABLE MIXING TWO BASES WITH NO LABEL.
    `CONFIG_KNOBS["ORIGINAL"]` is documented as running on a tree with the ranking-affecting
    fixes REVERTED -- i.e. pre-veto code -- while BASELINE/CARVE/EQUAL now rank through
    `rank_all_anchors` on HEAD, which applies the Stage-1 solvency gate.  Four columns of one
    table were therefore on two different bases and nothing said so.  Two tables that each
    name their basis would have been safer than one that names neither.

    READ IN PRIORITY ORDER, and an unrecognised stamp is never defaulted to a basis:
      1. the `basis` column in `cells_<CONFIG>.csv` -- written beside the numbers themselves;
      2. `diag_<CONFIG>.csv`, which has carried a per-anchor `basis` since the stamp existed
         (for cells files written before the cells column did);
      3. UNSTAMPED, said plainly.  A cells file written by an OLDER tree cannot have the
         column, and that is exactly the ORIGINAL case -- so "no stamp" is a real answer
         here, not a lookup failure to paper over.
    """
    cdf = all_df[all_df["config"] == config]
    if "basis" in cdf.columns:
        vals = sorted({str(v) for v in cdf["basis"].dropna().unique() if str(v).strip()})
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            return "MIXED -- one config's cells disagree: " + " | ".join(vals)
    dpath = os.path.join(workdir, "diag_%s.csv" % config)
    if os.path.exists(dpath):
        d = pd.read_csv(dpath)
        if "basis" in d.columns and d["basis"].notna().any():
            per_anchor = {str(r.anchor): {"basis": r.basis}
                          for r in d.itertuples(index=False) if str(r.basis) != "nan"}
            if per_anchor:
                return bstamp.of(per_anchor)
    return bstamp.UNSTAMPED + " -- no basis in cells_%s.csv or diag_%s.csv" % (config, config)


def _basis_block(basis_by_config):
    """The per-config basis, plus a LOUD line when the columns are not comparable."""
    lines = ["", "MEASUREMENT BASIS PER COLUMN  (the Stage-1 solvency gate: applied, or not)"]
    for cfg, b in basis_by_config.items():
        lines.append("  %-9s [%-9s] %s" % (cfg, bstamp.tag(b), b))
    tags = {bstamp.tag(b) for b in basis_by_config.values()}
    if len(tags) > 1:
        lines.append("")
        lines.append("  !! THE COLUMNS OF EVERY TABLE BELOW ARE NOT ALL ON THE SAME BASIS.")
        lines.append("     A vetoed column minus an un-vetoed one measures the VETO plus the "
                     "config change,")
        lines.append("     not the config change -- the ORIGINAL->BASELINE read in particular "
                     "is confounded.")
        lines.append("     Re-run the odd column on the same basis before reading any "
                     "difference as a config effect.")
    if bstamp.tag(basis_by_config.get("ORIGINAL", "")) == "UNSTAMPED":
        lines.append("")
        lines.append("  NOTE on ORIGINAL: it is BY CONSTRUCTION run on a reverted, pre-veto "
                     "tree, so an")
        lines.append("     unstamped ORIGINAL is expected and means UN-VETOED.  An unstamped "
                     "BASELINE, CARVE")
        lines.append("     or EQUAL means the opposite -- something unknown produced it -- "
                     "and is not safe")
        lines.append("     to read the same way.")
    return lines


def format_comparison(workdir, out_path, csv_path, log):
    paths = sorted(glob.glob(os.path.join(workdir, "cells_*.csv")))
    if not paths:
        raise RuntimeError(f"no cells_*.csv found in {workdir}")
    frames = [pd.read_csv(p) for p in paths]
    all_df = pd.concat(frames, ignore_index=True)
    present_configs = [c for c in CONFIG_ORDER if c in set(all_df["config"])]
    log(f"formatting configs present: {present_configs}")
    basis_by_config = {c: _basis_for_config(all_df, workdir, c) for c in present_configs}

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
    lines += _basis_block(basis_by_config)
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
        lines += _fmt_scope_block(all_df, present_configs, scope, title,
                                  basis_by_config=basis_by_config)

    lines.append("")
    lines.append("=" * 118)
    lines.append("CAVEATS (do not launder): same as the per-config grids -- PROVISIONAL survivorship")
    lines.append("  dead-merge; real_prices exchange-coverage gaps inflate affected counts; the")
    lines.append("  EXCLUDED anchors' depth cut is uninformative (they are held out on")
    lines.append("  SURVIVORSHIP -- see depth_horizon_grid.ANCHOR_EXCLUSION_REASONS, printed in")
    lines.append("  the grid report); URTH TR-proxy; heavily-overlapping windows;")
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
