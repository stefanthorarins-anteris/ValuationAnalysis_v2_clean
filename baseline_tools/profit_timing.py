"""
Part 6 -- PROFIT-TIMING vs CHURN (the CEO's key discriminator). Offline.

For names in the top-20 as-of D that EXIT the top-20 by D+1y: does the profit
accrue WHILE held [D, D+1y] (healthy -- ranking captured the run) or AFTER exit
[D+1y, D+3y] (premature churn -- the ranking dropped them too early = noise =
MODEL problem)?

CAVEAT: reconstructed (synthetic) price returns -- directional only, ~0.73 corr
to real; a real-price fetch would firm up the magnitudes.
"""
import os
import sys
import io
import contextlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_pit as s2
import model_vs_metric as mvm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _price_at(price_panel, sym, d):
    sub = price_panel[(price_panel["source"] == sym) & (price_panel["date"] <= pd.Timestamp(d))]
    return sub["_price"].iloc[-1] if len(sub) else np.nan


def run(D="2021-12-31", D1="2022-12-31", D3="2024-12-31"):
    dmdic = pd.read_pickle(mvm.PICKLE)
    panel = mvm.build_panel(dmdic["cdx_df"])[["source", "date", "_price"]]

    with contextlib.redirect_stdout(io.StringIO()):
        bm0, cdx0 = s2.prepare_pit(dmdic, D, na1_only=False)
        top0 = s2.stage2_top(s2.stage1_boscore(bm0), cdx0)
        bm1, cdx1 = s2.prepare_pit(dmdic, D1, na1_only=False)
        top1 = s2.stage2_top(s2.stage1_boscore(bm1), cdx1)

    exiters = [s for s in top0 if s not in set(top1)]
    stayers = [s for s in top0 if s in set(top1)]

    def _timing(names):
        rows = []
        for s in names:
            p0, p1, p3 = _price_at(panel, s, D), _price_at(panel, s, D1), _price_at(panel, s, D3)
            if any(pd.isna(x) or x <= 0 for x in (p0, p1, p3)):
                continue
            rows.append({"sym": s, "during_hold": p1 / p0 - 1, "after_exit": p3 / p1 - 1,
                         "full_3y": p3 / p0 - 1})
        return pd.DataFrame(rows)

    ex = _timing(exiters)
    st = _timing(stayers)

    print("=" * 72)
    print("6. PROFIT-TIMING vs CHURN (reconstructed price -- DIRECTIONAL, CAVEAT)")
    print(f"   D={D}  ->  D+1y={D1}  ->  D+3y={D3}")
    print("=" * 72)
    print(f"top-20 at D: 20 | exit by D+1y: {len(exiters)} | stay: {len(stayers)}")
    for label, df in [("EXITERS (churned out by +1y)", ex), ("STAYERS", st)]:
        if df.empty:
            print(f"\n  {label}: no priceable names"); continue
        print(f"\n  {label} (n={len(df)}):")
        print(f"    median return DURING hold [D,+1y]:  {df['during_hold'].median()*100:+6.1f}%")
        print(f"    median return AFTER exit  [+1y,+3y]: {df['after_exit'].median()*100:+6.1f}%")
        print(f"    median full 3y [D,+3y]:              {df['full_3y'].median()*100:+6.1f}%")
    if not ex.empty:
        during = ex["during_hold"].median()
        after = ex["after_exit"].median()
        print("\n  READ: ", end="")
        if after > during + 0.05:
            print("profit accrues AFTER exit -> PREMATURE CHURN (ranking dropped them")
            print("        too early) -> churn is largely noise -> MODEL problem.")
        elif during > after + 0.05:
            print("profit accrues DURING hold -> churn looks HEALTHY (ran, then cooled).")
        else:
            print("during ~ after -> ambiguous; no clean timing signal.")
    print("=" * 72)


if __name__ == "__main__":
    run()
