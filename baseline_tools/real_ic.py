"""
Re-run the model-vs-metric IC on REAL adjusted-close forward returns (offline).

The throttled fetch captured 4 real dates: 2018-12-31, 2020-12-28, 2022-12-27,
2024-12-28 (even years). So we test REAL 24-month forward-return IC across buy
dates 2018/2020/2022, plus a 48-month check (2018->2022, 2020->2024), and
compare to the RECONSTRUCTED-price IC over the identical buy dates + horizon.

Purpose: does "composite < best single metric" (the model-problem smoking gun)
HOLD on real returns, and how much do IC magnitudes move vs the proxy?

Caveats kept: 24/36mo horizons differ (fetch throttling forced even-year dates);
metric 'date' are quarter-starts (the pipeline lookahead), applied identically to
real & reconstructed so the COMPARISON is apples-to-apples.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_vs_metric as mvm

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_data", "real_prices.csv")
REAL_DATES = ["2018-12-31", "2020-12-28", "2022-12-27", "2024-12-28"]


def load_real(path=None):
    df = pd.read_csv(path or REAL)
    df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
    # (date_actual, symbol) -> adjClose
    piv = df.pivot_table(index="symbol", columns="date_actual", values="adjClose", aggfunc="last")
    return piv


def metric_xs(panel, asof):
    return panel[panel["date"] <= pd.Timestamp(asof)].groupby("source").tail(1).set_index("source")


def ic_table(panel, real, buy_eval_pairs, horizon_label):
    """Mean Spearman IC per metric (+COMPOSITE) over the given (buy,eval) pairs,
    computed for REAL and RECONSTRUCTED returns side by side."""
    out = {m: {"real": [], "recon": []} for m in mvm.METRICS + ["COMPOSITE"]}
    for buy, ev in buy_eval_pairs:
        xs = metric_xs(panel, buy)
        # reconstructed fwd return from panel _price (nearest <= dates)
        p_buy = metric_xs(panel, buy)["_price"]
        p_ev = metric_xs(panel, ev)["_price"]
        recon_ret = (p_ev / p_buy - 1).replace([np.inf, -np.inf], np.nan)
        # real fwd return
        rb = real[buy] if buy in real.columns else None
        re = real[ev] if ev in real.columns else None
        real_ret = (re / rb - 1).replace([np.inf, -np.inf], np.nan) if (rb is not None and re is not None) else None

        # composite = cross-sectional weighted z-sum
        z = pd.DataFrame(index=xs.index)
        for mm in mvm.METRICS:
            col = pd.to_numeric(xs[mm], errors="coerce")
            z[mm] = (col - col.mean()) / col.std() * mvm.WEIGHTS[mm]
        comp = z.sum(axis=1, min_count=1)

        for m in mvm.METRICS + ["COMPOSITE"]:
            val = comp if m == "COMPOSITE" else pd.to_numeric(xs[m], errors="coerce")
            for tag, ret in (("real", real_ret), ("recon", recon_ret)):
                if ret is None:
                    continue
                j = pd.concat([val.rename("v"), ret.rename("r")], axis=1).dropna()
                if len(j) >= 50:
                    ic, _ = spearmanr(j["v"], j["r"])
                    out[m][tag].append(ic)
    rows = []
    for m in mvm.METRICS + ["COMPOSITE"]:
        rows.append({
            "metric": m,
            "IC_real": np.nanmean(out[m]["real"]) if out[m]["real"] else np.nan,
            "IC_recon": np.nanmean(out[m]["recon"]) if out[m]["recon"] else np.nan,
            "n": len(out[m]["real"]),
        })
    return pd.DataFrame(rows), horizon_label


def profit_timing_real(dmdic, panel, real):
    import io, contextlib
    import stage2_pit as s2
    D, D1, D3 = "2020-12-28", "2022-12-27", "2024-12-28"
    with contextlib.redirect_stdout(io.StringIO()):
        bm0, cdx0 = s2.prepare_pit(dmdic, D, na1_only=False)
        top0 = s2.stage2_top(s2.stage1_boscore(bm0, cdx_pit=cdx0), cdx0)
        bm1, cdx1 = s2.prepare_pit(dmdic, D1, na1_only=False)
        top1 = s2.stage2_top(s2.stage1_boscore(bm1, cdx_pit=cdx1), cdx1)
    ex = [s for s in top0 if s not in set(top1)]
    st = [s for s in top0 if s in set(top1)]

    def _t(names):
        r = []
        for s in names:
            if s not in real.index:
                continue
            p0, p1, p3 = real.at[s, D], real.at[s, D1], real.at[s, D3]
            if any(pd.isna(x) or x <= 0 for x in (p0, p1, p3)):
                continue
            r.append({"during": p1 / p0 - 1, "after": p3 / p1 - 1, "full": p3 / p0 - 1})
        return pd.DataFrame(r)
    return ex, st, _t(ex), _t(st)


def run_in_pipeline(dmdic, price_source=None, real_prices_csv=None, log=None):
    """IN-MEMORY entry point for the automatic pipeline (post-pick analysis suite).

    Re-runs the IC (real vs reconstructed price) + profit-timing decomposition on
    TONIGHT's fresh scoring dict `dmdic`, NOT the hardcoded stale PICKLE that main()
    loads.  Real adjusted-close prices come from real_prices.csv (the same file the
    guarded price-fetch step ensures is present); dmdic supplies the metric panel.

    `real_prices_csv` overrides the default REAL path (machine-independence); if the
    real-price file is missing the caller's guard banners -- this raises rather than
    silently producing an empty table.  price_source is accepted for parity (unused;
    real_ic reads the year-anchor CSV directly).  Never prints any api_key.
    """
    log = log or (lambda *a: None)
    cdx = dmdic.get("cdx_df")
    if cdx is None:
        raise ValueError("dmdic has no 'cdx_df' -- cannot build the metric panel")
    log("[real_ic] building panel + loading real prices (in-memory) ...")
    panel = mvm.build_panel(cdx)
    real = load_real(real_prices_csv)
    ov = len(set(panel['source'].unique()) & set(real.index))
    print("\n" + "#" * 72)
    print("# REAL-IC diagnostic  (tonight's model, real adjusted-close returns)")
    print("#" * 72)
    print(f"real price matrix: {real.shape[0]} symbols x {list(real.columns)}", flush=True)
    print(f"symbols overlapping pipeline universe: {ov}", flush=True)

    print("\n" + "=" * 72)
    print("IC on REAL vs RECONSTRUCTED returns  (24-month horizon; buy 2018/2020/2022)")
    print("=" * 72)
    pairs24 = [("2018-12-31", "2020-12-28"), ("2020-12-28", "2022-12-27"),
               ("2022-12-27", "2024-12-28")]
    tbl, _ = ic_table(panel, real, pairs24, "24m")
    tbl = tbl.sort_values("IC_real", key=lambda s: s.abs(), ascending=False)
    print(tbl.to_string(index=False, formatters={"IC_real": "{:+.3f}".format,
                                                  "IC_recon": "{:+.3f}".format}))
    singles = tbl[tbl["metric"] != "COMPOSITE"]
    comp = tbl[tbl["metric"] == "COMPOSITE"].iloc[0]
    best_real = singles.iloc[0]
    print(f"\n  COMPOSITE IC_real={comp['IC_real']:+.3f} vs best single "
          f"({best_real['metric']}) IC_real={best_real['IC_real']:+.3f}  "
          f"-> smoking gun {'HOLDS' if comp['IC_real'] < best_real['IC_real'] else 'DOES NOT hold'}")

    print("\n" + "=" * 72)
    print("PROFIT-TIMING vs CHURN on REAL prices (D=2020-12 -> +2y -> +4y)")
    print("=" * 72)
    ex, st, exdf, stdf = profit_timing_real(dmdic, panel, real)
    print(f"top-20 at D=2020: exit by +2y={len(ex)}, stay={len(st)}")
    for lbl, d in [("EXITERS", exdf), ("STAYERS", stdf)]:
        if d.empty:
            print(f"  {lbl}: no priceable"); continue
        print(f"  {lbl} (n={len(d)}): during[D,+2y]={d['during'].median()*100:+.1f}%  "
              f"after[+2y,+4y]={d['after'].median()*100:+.1f}%  full={d['full'].median()*100:+.1f}%")
    print("\n[real_ic] DONE.", flush=True)
    return {"ic_24m": tbl, "exiters": exdf, "stayers": stdf}


def main():
    print("Loading pickle + building panel (offline) ...", flush=True)
    dmdic = pd.read_pickle(mvm.PICKLE)
    panel = mvm.build_panel(dmdic["cdx_df"])
    real = load_real()
    print(f"real price matrix: {real.shape[0]} symbols x {list(real.columns)}", flush=True)
    ov = len(set(panel['source'].unique()) & set(real.index))
    print(f"symbols overlapping pipeline universe: {ov}", flush=True)

    print("\n" + "=" * 72)
    print("IC on REAL vs RECONSTRUCTED returns  (24-month horizon; buy 2018/2020/2022)")
    print("=" * 72)
    pairs24 = [("2018-12-31", "2020-12-28"), ("2020-12-28", "2022-12-27"),
               ("2022-12-27", "2024-12-28")]
    tbl, _ = ic_table(panel, real, pairs24, "24m")
    tbl = tbl.sort_values("IC_real", key=lambda s: s.abs(), ascending=False)
    print(tbl.to_string(index=False, formatters={"IC_real": "{:+.3f}".format,
                                                  "IC_recon": "{:+.3f}".format}))
    singles = tbl[tbl["metric"] != "COMPOSITE"]
    comp = tbl[tbl["metric"] == "COMPOSITE"].iloc[0]
    best_real = singles.iloc[0]
    print(f"\n  COMPOSITE IC_real={comp['IC_real']:+.3f} vs best single "
          f"({best_real['metric']}) IC_real={best_real['IC_real']:+.3f}  "
          f"-> smoking gun {'HOLDS' if comp['IC_real'] < best_real['IC_real'] else 'DOES NOT hold'}")

    print("\n  (48-month check: buy 2018->2022, 2020->2024)")
    pairs48 = [("2018-12-31", "2022-12-27"), ("2020-12-28", "2024-12-28")]
    tbl48, _ = ic_table(panel, real, pairs48, "48m")
    tbl48 = tbl48.sort_values("IC_real", key=lambda s: s.abs(), ascending=False)
    c48 = tbl48[tbl48.metric == "COMPOSITE"].iloc[0]["IC_real"]
    b48 = tbl48[tbl48.metric != "COMPOSITE"].iloc[0]
    print(f"    48m: COMPOSITE IC_real={c48:+.3f} vs best ({b48['metric']}) "
          f"{b48['IC_real']:+.3f} -> {'HOLDS' if c48 < b48['IC_real'] else 'DOES NOT hold'}")

    print("\n" + "=" * 72)
    print("PROFIT-TIMING vs CHURN on REAL prices (D=2020-12 -> +2y -> +4y)")
    print("=" * 72)
    ex, st, exdf, stdf = profit_timing_real(dmdic, panel, real)
    print(f"top-20 at D=2020: exit by +2y={len(ex)}, stay={len(st)}")
    for lbl, d in [("EXITERS", exdf), ("STAYERS", stdf)]:
        if d.empty:
            print(f"  {lbl}: no priceable"); continue
        print(f"  {lbl} (n={len(d)}): during[D,+2y]={d['during'].median()*100:+.1f}%  "
              f"after[+2y,+4y]={d['after'].median()*100:+.1f}%  full={d['full'].median()*100:+.1f}%")


if __name__ == "__main__":
    main()
