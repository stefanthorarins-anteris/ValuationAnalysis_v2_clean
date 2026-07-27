"""
CHURN-vs-HORIZON DIAGNOSTIC (offline, no network) -- reuses stage2_pit engine.

Measures top-20 overlap (churn = 1 - overlap) between the filter's shortlist
re-ranked at successive as-of dates, as a function of horizon. Two purposes:
  (a) QUANTIFY the short-horizon stability defect;
  (b) MECHANISM: does churn rise with horizon (mispricing-correction) or stay
      flat/low (persistent risk-premium / value-trap)?

KEY LEVER: the stage2_pit engine is FUNDAMENTALS-ONLY and DETERMINISTIC. It
excludes exactly the daily live inputs identified in the audit (DcfToPrice, live
price/beta). So:
  * Re-ranking it at two dates in the SAME quarter gives 20/20 by construction
    -> a clean control. Any short-horizon churn in the REAL filter (15/20 over 4
    days) is therefore attributable to the EXCLUDED noise channels.
  * Its long-horizon churn is a FUNDAMENTALS-ONLY, SURVIVORSHIP-SUPPRESSED LOWER
    BOUND on true churn (names delisted between D and the 2026 snapshot are
    absent at BOTH endpoints, so real churn -- including delistings -- is larger).

Two controlled short-horizon experiments isolate the amplifier channels, holding
quarterly fundamentals FIXED:
  * boscore_noise -> pool-boundary-flip (quantization) channel
  * price_noise   -> Stage-2 price-metric channel

PERF: Stage-1 (scoring the whole universe) is the expensive step and is CACHED
per (as-of date, universe); Stage-2 (100 names) is cheap and re-run per trial.
"""

import os
import sys
import io
import contextlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_pit as s2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKLE = os.path.join(
    REPO, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-01-09_len8106_manelim3692_fails1725.pickle")

_CACHE = {}  # (date, na1_only) -> (BoScore_df, cdx_pit)


def _prep(dmdic, D, na1_only):
    key = (str(D), na1_only)
    if key not in _CACHE:
        with contextlib.redirect_stdout(io.StringIO()):
            bm_pit, cdx_pit = s2.prepare_pit(dmdic, D, na1_only=na1_only)
            bs = s2.stage1_boscore(bm_pit, cdx_pit=cdx_pit)
        _CACHE[key] = (bs, cdx_pit)
    return _CACHE[key]


def top20(dmdic, D, na1_only=False, **kw):
    bs, cdx_pit = _prep(dmdic, D, na1_only)
    with contextlib.redirect_stdout(io.StringIO()):
        return s2.stage2_top(bs, cdx_pit, **kw)


def overlap(a, b):
    return len(set(a) & set(b))


def noise_churn(dmdic, D, channel, levels, n_seeds=6, na1_only=False):
    base = top20(dmdic, D, na1_only=na1_only)
    rows = []
    for lv in levels:
        churns = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            kw = {"rng": rng}
            kw["boscore_noise" if channel == "boscore" else "price_noise_frac"] = lv
            t = top20(dmdic, D, na1_only=na1_only, **kw)
            churns.append((20 - overlap(base, t)) / 20 * 100)
        rows.append({"level": lv, "mean_churn_pct": float(np.mean(churns)),
                     "max_churn_pct": float(np.max(churns))})
    return pd.DataFrame(rows)


def main():
    print("Loading 2026 pickle (offline) ...", flush=True)
    dmdic = pd.read_pickle(PICKLE)

    print("\n" + "=" * 72)
    print("PART 1 - SHORT HORIZON (the clean, high-value part)")
    print("=" * 72)
    t27 = top20(dmdic, "2023-01-27", na1_only=True)
    t31 = top20(dmdic, "2023-01-31", na1_only=True)
    print(f"\n[control] engine(2023-01-27) vs engine(2023-01-31), fundamentals-only:"
          f"  {overlap(t27, t31)}/20")
    print("          REAL filter over the same 4 days (saved runs):  15/20")
    print("          => real short-horizon churn is NOT fundamental; it comes from")
    print("             the daily inputs the engine excludes (DcfToPrice, live price).")

    print("\n[channel isolation] top-20 churn from daily-input noise, fundamentals")
    print("                    held FIXED (anchor 2022-06-30, 6 seeds):")
    D = "2022-06-30"
    bs = noise_churn(dmdic, D, "boscore", [0.003, 0.01, 0.03, 0.05, 0.1])
    pr = noise_churn(dmdic, D, "price", [0.01, 0.02, 0.05, 0.10])
    print("\n  BoScore-jitter (pool-boundary flips; near-cutoff score-gap ~0.003):")
    for r in bs.itertuples(index=False):
        print(f"    +/-{r.level:<6} pts -> mean churn {r.mean_churn_pct:4.0f}%  (max {r.max_churn_pct:.0f}%)")
    print("  Price-jitter (newest-quarter price/marketCap; Stage-1 pool held fixed):")
    for r in pr.itertuples(index=False):
        print(f"    +/-{r.level*100:<4.0f}%     -> mean churn {r.mean_churn_pct:4.0f}%  (max {r.max_churn_pct:.0f}%)")

    print("\n" + "=" * 72)
    print("PART 2 - LONG HORIZON (fundamentals-only; LOWER BOUND, survivorship-suppressed)")
    print("=" * 72)
    long_specs = {
        "2021-12-31": [("+1q", "2022-03-31"), ("+1y", "2022-12-31"),
                       ("+2y", "2023-12-31"), ("+3y", "2024-12-31")],
        "2022-06-30": [("+1q", "2022-09-30"), ("+1y", "2023-06-30"),
                       ("+2y", "2024-06-30"), ("+3y", "2025-06-30")],
    }
    for anchor, horizons in long_specs.items():
        base = top20(dmdic, anchor, na1_only=False)
        print(f"\n  anchor {anchor}:")
        for label, d2 in horizons:
            t2 = top20(dmdic, d2, na1_only=False)
            ov = overlap(base, t2)
            print(f"    {label:5s} (vs {d2}): overlap {ov:2d}/20 ({100*ov/20:3.0f}%)  "
                  f"churn {100*(20-ov)/20:3.0f}%")
    print("\n  CAVEAT: names delisted between anchor and the 2026 snapshot are absent")
    print("  at BOTH endpoints -> true churn (incl. delistings) is HIGHER. Read the")
    print("  SHAPE (does churn rise with horizon), not the absolute rate.")
    print("=" * 72)


if __name__ == "__main__":
    main()
