# run_analysis_on_saved_run.py
# Re-run the POST-PICK analysis suite ad-hoc against a SAVED run (offline; no live pick,
# no full pipeline). Parameterized by the fundamentals pickle path so it can be
# re-pointed at the prior nightly pickle when the CEO provides it. Uses the FIXED code
# paths: URTH force-added to the bulk-fetch allow-list (pipeline_analysis) and the
# real_ic.ic_table None-eval guard (real_ic.py line 57).
#
# BENCHMARK (URTH) HANDLING -- chosen policy: DETECT-AND-INSTRUCT. The suite needs URTH
# on the price grid. This script NEVER writes or deletes a price CSV (read-only). It
# checks up front that PriceSource.benchmark_series() resolves; if the local price CSV
# predates the fetch fix (no URTH) it fails loudly with the remedy: delete
# baseline_tools/price_data/real_prices.csv (+ real_prices_2025.csv) so the FIXED
# bulk-by-date fetch re-pulls the grid WITH the benchmark. We deliberately do NOT add a
# separate per-symbol URTH historical fetch: the bulk-by-date endpoint is the one proven
# key-scrubbed on-plan path; a different historical endpoint behaviour is UNVERIFIED in
# this layer and must be confirmed by the fmp-specialist before wiring. On THIS dev
# machine real_prices.csv already contains URTH, so no action is needed. Never prints key.
import argparse
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_HOME = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"

DEFAULT_PICKLE = os.path.join(
    _REPO, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-01-09_len8106_manelim3692_fails1725.pickle")


def _summ(tbl):
    singles = tbl[tbl["metric"] != "COMPOSITE"].sort_values(
        "IC_real", key=lambda s: s.abs(), ascending=False)
    comp = float(tbl[tbl["metric"] == "COMPOSITE"].iloc[0]["IC_real"])
    best = singles.iloc[0]
    return comp, best["metric"], float(best["IC_real"]), comp < float(best["IC_real"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default=DEFAULT_PICKLE)
    ap.add_argument("--delisted-out", default=os.path.join(_HOME, "delisted_out"))
    ap.add_argument("--prices-csv", default=None)
    ap.add_argument("--prices-2025-csv", default=None)
    ap.add_argument("--as-of", default="2026-01-09")
    args = ap.parse_args()

    for p in (_REPO, _HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(_REPO)

    import pandas as pd
    import pipeline_analysis as pa
    import real_ic as ric
    import depth_horizon_grid as dhg
    import skill_baseline as sb

    log = lambda *a: print("[log]", *a, file=sys.stderr, flush=True)
    if args.prices_csv:
        pa._PRICES_CSV = args.prices_csv
    if args.prices_2025_csv:
        pa._PRICES_2025_CSV = args.prices_2025_csv

    t0 = time.time()
    bar = "=" * 78
    print(bar)
    print("AD-HOC ANALYSIS on SAVED RUN (offline; historical PIT backtest vs this model)")
    print("pickle   : " + os.path.basename(args.pickle))
    print("prices   : " + os.path.basename(pa._PRICES_CSV) + " (+2025 supp if present)")
    print("CAVEATS  : JAN-2026 snapshot universe (NOT the live top-20); the CLEAN 36mo grid")
    print("           is only ~2 heavily-overlapping windows -> suggestive, NOT established")
    print("           (per reviewer). Numbers reported as they come out.")
    print(bar, flush=True)

    resdic = pd.read_pickle(args.pickle)
    dmdic = dict(resdic)
    configdic = {"delisted_out": args.delisted_out, "as_of": args.as_of, "run_estimation": 0}

    price_source = pa._build_price_source(log)
    try:
        bs = price_source.benchmark_series()
    except RuntimeError as e:
        raise SystemExit(
            "\n" + ("X" * 78) +
            "\n BENCHMARK ABSENT: " + str(e) + "\n"
            " The local price CSV predates the fetch fix (no URTH). REMEDY: delete\n"
            " baseline_tools/price_data/real_prices.csv (+ real_prices_2025.csv) and re-run\n"
            " the pipeline price-fetch; the FIXED bulk fetch now force-adds URTH to the\n"
            " allow-list, so the re-pulled grid will contain it. Never writes price CSVs.\n"
            + ("X" * 78))
    print("[benchmark] URTH resolved: n=" + str(len(bs)) + " anchors "
          + str(bs.index.min().date()) + ".." + str(bs.index.max().date()), flush=True)

    results = {}
    hashbar = "#" * 78

    print("\n" + hashbar + "\n# STAGE 1: real-IC (composite vs best single metric)\n" + hashbar,
          flush=True)
    try:
        panel = ric.mvm.build_panel(dmdic["cdx_df"])
        real = ric.load_real(pa._PRICES_CSV)
        ov = len(set(panel["source"].unique()) & set(real.index))
        print("real price matrix: " + str(real.shape[0]) + " symbols; overlap=" + str(ov), flush=True)
        HORIZONS = {
            "12m": [("2018-12-31", "2019-12-31"), ("2019-12-31", "2020-12-31"),
                    ("2020-12-31", "2021-12-31"), ("2021-12-31", "2022-12-30"),
                    ("2022-12-30", "2023-12-29"), ("2023-12-29", "2024-12-31")],
            "24m": [("2018-12-31", "2020-12-31"), ("2020-12-31", "2022-12-30"),
                    ("2022-12-30", "2024-12-31")],
            "36m": [("2018-12-31", "2021-12-31"), ("2021-12-31", "2024-12-31")],
        }
        for hz in ("12m", "24m", "36m"):
            pairs = [(b, e) for b, e in HORIZONS[hz]
                     if b in real.columns and e in real.columns]
            tbl, _ = ric.ic_table(panel, real, pairs, hz)
            tbl = tbl.sort_values("IC_real", key=lambda s: s.abs(), ascending=False)
            comp, bm, bic, holds = _summ(tbl)
            print("\n[" + hz + "]  windows=" + str(len(pairs)))
            print(tbl.to_string(index=False, formatters={
                "IC_real": "{:+.3f}".format, "IC_recon": "{:+.3f}".format}))
            print("  -> COMPOSITE IC_real=" + format(comp, "+.3f") + "  vs best single "
                  + str(bm) + " " + format(bic, "+.3f") + "  => composite-below-best "
                  + ("HOLDS" if holds else "does NOT hold"))
            results["realIC_" + hz] = (comp, bm, bic, holds)
        results["real_ic"] = "OK"
    except Exception as e:
        results["real_ic"] = "FAIL " + type(e).__name__ + ": " + str(e)
        traceback.print_exc()

    print("\n" + hashbar + "\n# PIT inputs (survivorship dead-merge)\n" + hashbar, flush=True)
    merged, registry, clean = pa._build_pit_inputs(dmdic, configdic, log)
    print("[pit] survivorship_clean=" + str(clean), flush=True)

    print("\n" + hashbar + "\n# STAGE 2: depth x horizon avg-TR grid (DEPLOYED, carve=on)\n"
          + hashbar, flush=True)
    per_anchor = None
    try:
        report, per_anchor = dhg.run_in_pipeline(dmdic, merged, registry, price_source,
                                                 log=log, carve="on")
        print(report, flush=True)
        results["depth_grid"] = "OK"
    except Exception as e:
        results["depth_grid"] = "FAIL " + type(e).__name__ + ": " + str(e)
        traceback.print_exc()

    print("\n" + hashbar + "\n# STAGE 3: beat-rate vs URTH (deployed filter)\n" + hashbar,
          flush=True)
    try:
        if per_anchor is None:
            raise RuntimeError("per_anchor missing (grid stage failed)")
        pa.beat_rate_vs_urth(per_anchor, price_source, log)
        results["beat_rate"] = "OK"
    except Exception as e:
        results["beat_rate"] = "FAIL " + type(e).__name__ + ": " + str(e)
        traceback.print_exc()

    print("\n" + hashbar + "\n# STAGE 4: skill-baseline (filter vs oracle ceiling vs random floor)\n"
          + hashbar, flush=True)
    try:
        res = sb.run_skill_baseline(dmdic, merged, registry, price_source,
                                    cadence_months=36, pick_n=20, oracle_ns=(3, 20),
                                    n_draws=1000, seed=0, log=log)
        print(sb.format_report(res), flush=True)
        results["skill_baseline"] = "OK"
    except Exception as e:
        results["skill_baseline"] = "FAIL " + type(e).__name__ + ": " + str(e)
        traceback.print_exc()

    print("\n" + hashbar + "\n# STAGE 5: review-reference DATA artifacts (raw metrics + cohort stats)\n"
          + hashbar, flush=True)
    try:
        if _REPO not in sys.path:
            sys.path.insert(0, _REPO)
        import reviewReference as rr
        rr_res = rr.emit_from_saved(dmdic, out_dir=_REPO)
        results["review_ref"] = "OK (case %d)" % rr_res["case"]
    except Exception as e:
        results["review_ref"] = "FAIL " + type(e).__name__ + ": " + str(e)
        traceback.print_exc()

    print("\n" + bar)
    print("SUITE SUMMARY (wall-clock " + format(time.time() - t0, ".1f") + "s)")
    print(bar)
    for k in ("real_ic", "depth_grid", "beat_rate", "skill_baseline", "review_ref"):
        print("  " + k.ljust(16) + " -> " + str(results.get(k, "NOT RUN")))
    print(bar, flush=True)


if __name__ == "__main__":
    main()
