"""
VALIDATION GATE (offline, no network): does the offline Stage-2 PIT reproduction
recover the filter's REAL saved 2023 top-20?

Compares reproduce_pit_top() (from the 2026 pickle, as-of each 2023 date) to the
saved real AggScoreTop40 CSVs. Reports overlap %, the survivorship ceiling, a
cause diagnosis, and DcfToPrice / CycleHeat sensitivity measured on the 2026
saved rankings.

Run:  python baseline_tools/validate_gate.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_pit as s2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKLE = os.path.join(
    REPO, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-01-09_len8106_manelim3692_fails1725.pickle")

GROUND_TRUTH = {
    "2023-01-27": os.path.join(REPO, "results",
                               "AggScoreTop40-2023-01-27_stock_NA1_9546.csv"),
    "2023-01-31": os.path.join(REPO, "results",
                               "AggScoreTop40-2023-01-31_fmp_stock_NA1.csv"),
}


def load_real_top20(csv_path, n=20):
    df = pd.read_csv(csv_path)
    # CSV is sorted by AggScore descending; the top-20 shortlist = first n rows.
    return df.sort_values("AggScore", ascending=False)["source"].head(n).tolist()


def sensitivity_on_2026(dmdic):
    """Measure DcfToPrice & CycleHeat influence on the ACTUAL saved 2026 top-20.

    postRank holds the weighted metric columns and AggScore. Recompute AggScore
    with each metric removed and measure top-20 churn -> a data-grounded bound on
    how much dropping DcfToPrice / neutralising CycleHeat perturbs the shortlist.
    """
    pr = dmdic["postRank"].copy()
    base_top20 = set(pr.sort_values("AggScore", ascending=False)["source"].head(20))
    out = {}
    for metric in ["DcfToPrice", "CycleHeat"]:
        if metric not in pr.columns:
            out[metric] = "column absent in saved postRank"
            continue
        adj = pr.copy()
        adj["AggScore2"] = adj["AggScore"] - adj[metric].fillna(0)
        new_top20 = set(adj.sort_values("AggScore2", ascending=False)["source"].head(20))
        churn = len(base_top20 - new_top20)
        out[metric] = {
            "churn_out_of_20": churn,
            "left_top20": sorted(base_top20 - new_top20),
            "entered_top20": sorted(new_top20 - base_top20),
        }
    return out


def main():
    print("Loading 2026 pickle (offline) ...", flush=True)
    dmdic = pd.read_pickle(PICKLE)
    universe_present = set(dmdic["cdx_df"]["source"].unique())

    print("\n" + "=" * 72)
    print("STAGE-2 PIT REPRODUCTION -- VALIDATION GATE")
    print("Bar: >=18/20 (90%) faithful; <~85% = backtesting a cousin.")
    print("=" * 72)

    for date, csv_path in GROUND_TRUTH.items():
        real20 = load_real_top20(csv_path, 20)
        res = s2.reproduce_pit_top(dmdic, date, na1_only=True)
        rep = s2.overlap_report(res["top20"], real20, universe_present)

        in_pool = [s for s in real20 if s in set(res["stage1_top100"])]
        print(f"\n--- {date} ---")
        print(f"  universe (NA1, PIT): {res['universe_size']} names")
        print(f"  OVERLAP: {rep['overlap_n']}/20 ({rep['overlap_pct']:.0f}%)")
        print(f"  survivorship CEILING (real names still in 2026 data): "
              f"{rep['ceiling_n']}/20 ({rep['ceiling_pct']:.0f}%)")
        print(f"  overlap vs ceiling: {rep['overlap_vs_ceiling_pct']:.0f}% "
              f"of the achievable")
        print(f"  real top-20 that reached my Stage-1 top-100 pool: "
              f"{len(in_pool)}/20")
        print(f"  matched ({len(rep['matched'])}): {rep['matched']}")
        print(f"  missed but PRESENT (repro drift): {rep['missed_present']}")
        print(f"  missed & ABSENT (survivorship, unrecoverable): "
              f"{rep['missed_absent']}")
        print(f"  my extras not in real: {rep['extra']}")

    print("\n" + "=" * 72)
    print("SENSITIVITY (measured on the saved 2026 top-20, real DcfToPrice/beta):")
    sens = sensitivity_on_2026(dmdic)
    for m, v in sens.items():
        print(f"  drop {m}: {v}")
    print("=" * 72)


if __name__ == "__main__":
    main()
