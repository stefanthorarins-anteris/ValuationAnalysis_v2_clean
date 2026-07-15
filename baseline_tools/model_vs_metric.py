"""
MODEL-vs-METRIC diagnostic (offline, no network, no commit).

Computes, from the saved 2026 pickle:
  1. per-metric PERSISTENCE (lag-1 quarterly autocorr + half-life)
  2. per-metric TIME-VARIANCE (per-name coeff. of variation over time)
  3. cross-metric CORRELATION (redundancy structure)
  4. per-metric PREDICTIVE POWER (information coefficient = rank-corr of the
     metric as-of D with FORWARD return)
  5. COMPOSITE (weighted z-sum) IC vs BEST-SINGLE-METRIC IC
  6. PROFIT-TIMING vs CHURN for names that exit the top-20

CRITICAL CAVEAT (applies to 4/5/6): forward returns use the RECONSTRUCTED
(synthetic) price `getData_fmp.py:171` -- corr ~0.73 to real 36-mo returns,
+/-15-23pp per-name error. So IC/timing signs & RANKINGS are directional; a
real-price fetch would sharpen magnitudes (esp. for price-bearing metrics:
grahamNumberToPrice, bVpRatio, tbVpRatio, freeCashFlowYield, priceGrowth,
earnYield, and DcfToPrice which is unavailable offline entirely).

Metric definitions mirror the Stage-2 loop (postBoRank.py:172-441) at per-quarter
resolution. marketCapRevQuants (cross-sectional rank) and BoScore (a composite)
are excluded from the per-quarter panel; DcfToPrice is unavailable PIT offline
(needs the live DCF endpoint) -- its variance is proxied by price volatility.
"""

import os
import sys
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_pit as s2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKLE = os.path.join(
    REPO, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-01-09_len8106_manelim3692_fails1725.pickle")

WEIGHTS = {  # from createDicts.getPostDict (the AggScore weights)
    "RoA": 2, "earnYield": 2, "grahamNumberToPrice": 1, "bVpRatio": 0.25,
    "revenueGrowth": 1, "incomeQuality": 1, "returnOnEquity": 1,
    "returnOnCapitalEmployed": 1, "currentRatio": 0.35, "grossProfitMargin": 0.75,
    "freeCashFlowYield": 2, "freeCashFlowPerShareGrowth": 1.5, "tbVpRatio": 0.5,
    "EPStoEPSmean": 0.5, "priceGrowth": 0.5, "Altman-Z": 0.5, "Piotroski": 0.75,
    "CycleHeat": -0.5,
    # excluded from panel: DcfToPrice(0.35, unavailable), marketCapRevQuants(0.25), BoScore(0.1)
}


def build_panel(cdx):
    """Per-name, per-quarter metric panel (ascending by date within name)."""
    df = cdx.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["source", "date"])
    g = df.groupby("source", sort=False)

    eps = df["netIncome"] / df["weightedAverageShsOut"]
    fcfps = df["freeCashFlow"] / df["weightedAverageShsOut"]
    df["_eps"] = eps
    m = pd.DataFrame({"source": df["source"], "date": df["date"]})
    m["RoA"] = df["returnOnAssets"]
    m["earnYield"] = df["earningsYield"]
    m["grahamNumberToPrice"] = df["grahamNumber"] / df["price"]
    m["bVpRatio"] = 1.0 / df["pbRatio"]
    m["revenueGrowth"] = g["revenue"].pct_change(4)
    m["incomeQuality"] = df["incomeQuality"]
    m["returnOnEquity"] = df["returnOnEquity"]
    m["returnOnCapitalEmployed"] = df["returnOnCapitalEmployed"]
    m["currentRatio"] = df["currentRatio"]
    m["grossProfitMargin"] = df["grossProfitMargin"]
    m["freeCashFlowYield"] = df["freeCashFlow"] / df["marketCap"]
    m["freeCashFlowPerShareGrowth"] = fcfps.groupby(df["source"]).pct_change(4)
    m["tbVpRatio"] = df["tangibleBookValuePerShare"] / df["price"]
    m["priceGrowth"] = -g["price"].pct_change(1)
    # Altman-Z (row-wise)
    ta = df["totalAssets"].replace(0, np.nan)
    tl = df["totalLiabilities"].replace(0, np.nan)
    m["Altman-Z"] = (1.2 * (df["totalCurrentAssets"] - df["totalCurrentLiabilities"]) / ta
                     + 1.4 * df["totalStockholdersEquity"] / ta
                     + 3.3 * df["operatingIncome"] / ta
                     + 0.6 * df["marketCap"] / tl
                     + 1.0 * df["revenue"] / ta)
    # EPStoEPSmean: eps minus expanding-mean eps (causal)
    exp_mean = g["_eps"] if False else df.groupby("source")["_eps"]
    m["EPStoEPSmean"] = df["_eps"] - df.groupby("source")["_eps"].transform(
        lambda s: s.expanding().mean())
    # CycleHeat: z-score of eps vs expanding history (beta=1), capped [-3,3]
    em = df.groupby("source")["_eps"].transform(lambda s: s.expanding().mean())
    es = df.groupby("source")["_eps"].transform(lambda s: s.expanding().std())
    ch = (df["_eps"] - em) / es.replace(0, np.nan)
    m["CycleHeat"] = ch.clip(-3, 3)
    # Piotroski (9 binaries vs previous quarter within name)
    def _prev(col):
        return df.groupby("source")[col].shift(1)
    ta_p = _prev("totalAssets")
    roa = df["netIncome"] / ta
    roa_p = _prev("netIncome") / ta_p
    p = (
        (df["netIncome"] / ta > 0).astype(float)
        + (df["netCashProvidedByOperatingActivities"] > 0).astype(float)
        + (roa > roa_p).astype(float)
        + (df["netCashProvidedByOperatingActivities"] > df["netIncome"]).astype(float)
        + (df["longTermDebt"] / ta < _prev("longTermDebt") / ta_p).astype(float)
        + (df["currentRatio"] > _prev("currentRatio")).astype(float)
        + (df["weightedAverageShsOut"] <= _prev("weightedAverageShsOut")).astype(float)
        + (df["grossProfitMargin"] > _prev("grossProfitMargin")).astype(float)
        + (df["revenue"] / ta > _prev("revenue") / ta_p).astype(float)
    )
    m["Piotroski"] = p
    # forward reconstructed-price returns
    m["_price"] = df["price"]
    for h in (4, 12):  # 12mo, 36mo (quarters)
        m[f"fwd{h}"] = df.groupby("source")["price"].shift(-h) / df["price"] - 1
    m = m.replace([np.inf, -np.inf], np.nan)
    return m


METRICS = list(WEIGHTS.keys())


def persistence_and_variance(panel):
    rows = []
    grp = panel.groupby("source")
    for metric in METRICS:
        acs, cvs = [], []
        for _, gdf in grp:
            s = gdf[metric].dropna()
            if len(s) >= 8:
                ac = s.autocorr(lag=1)
                if pd.notna(ac):
                    acs.append(ac)
                mu, sd = s.mean(), s.std()
                if mu != 0 and pd.notna(sd):
                    cvs.append(abs(sd / mu))
        med_ac = np.nanmedian(acs) if acs else np.nan
        # half-life in quarters from median autocorr
        hl = (np.log(0.5) / np.log(med_ac)) if (0 < med_ac < 1) else np.nan
        rows.append({
            "metric": metric,
            "median_autocorr": med_ac,
            "half_life_q": hl,
            "median_CV": np.nanmedian(cvs) if cvs else np.nan,
            "n_names": len(acs),
        })
    return pd.DataFrame(rows)


def correlation_matrix(panel, sample_date="2021-12-31"):
    d = pd.Timestamp(sample_date)
    # latest row per name on or before d
    sub = panel[panel["date"] <= d].groupby("source").tail(1)
    X = sub[METRICS].apply(pd.to_numeric, errors="coerce")
    return X.corr(method="spearman")


def information_coefficient(panel, dates, horizon):
    """Mean Spearman IC (metric vs fwd return) across the given as-of dates."""
    rows = []
    for metric in METRICS + ["COMPOSITE"]:
        ics = []
        for d in dates:
            d = pd.Timestamp(d)
            sub = panel[panel["date"] <= d].groupby("source").tail(1)
            fwd = sub[f"fwd{horizon}"]
            if metric == "COMPOSITE":
                # cross-sectional z-sum with AggScore weights (mirrors the model)
                z = pd.DataFrame(index=sub.index)
                for mm in METRICS:
                    col = pd.to_numeric(sub[mm], errors="coerce")
                    z[mm] = (col - col.mean()) / col.std() * WEIGHTS[mm]
                val = z.sum(axis=1, min_count=1)
            else:
                val = pd.to_numeric(sub[metric], errors="coerce")
            ok = val.notna() & fwd.notna()
            if ok.sum() >= 50:
                ic, _ = spearmanr(val[ok], fwd[ok])
                ics.append(ic)
        rows.append({"metric": metric, "mean_IC": np.nanmean(ics) if ics else np.nan,
                     "n_dates": len(ics)})
    return pd.DataFrame(rows).sort_values("mean_IC", key=lambda s: s.abs(), ascending=False)


def run_in_pipeline(dmdic, price_source=None, log=None):
    """IN-MEMORY entry point for the automatic pipeline (post-pick analysis suite).

    Runs parts 1-5 (persistence / variance / correlation / IC / composite-vs-best)
    on the JUST-COMPUTED scoring dict `dmdic` (tonight's fresh cdx_df) -- NOT the
    hardcoded stale PICKLE that main() loads.  This is the crux: wiring main() as-is
    would analyze a stale Jan/Jul snapshot = silently wrong.

    price_source is accepted for signature parity across the suite; parts 1-5 use the
    reconstructed (synthetic) price embedded in the panel (see module CAVEAT), so it is
    not consumed here -- real_ic.run_in_pipeline is the REAL-price IC readout.

    Prints its report to stdout (the run log captures it); returns the IC tables dict.
    Never loads from disk; never prints any api_key.
    """
    log = log or (lambda *a: None)
    cdx = dmdic.get("cdx_df")
    if cdx is None:
        raise ValueError("dmdic has no 'cdx_df' -- cannot build the metric panel")
    log("[model_vs_metric] building metric panel from tonight's cdx_df (in-memory) ...")
    panel = build_panel(cdx)
    print("\n" + "#" * 72)
    print("# MODEL-vs-METRIC diagnostic  (tonight's model, in-memory)")
    print("#   persistence / variance / correlation / IC(reconstructed price)")
    print("#" * 72)
    print(f"panel rows: {len(panel)}, names: {panel['source'].nunique()}", flush=True)

    print("\n" + "=" * 72)
    print("1+2. PERSISTENCE (half-life) & TIME-VARIANCE (per-name CV)")
    print("=" * 72)
    pv = persistence_and_variance(panel).sort_values("median_autocorr", ascending=False)
    print(pv.to_string(index=False,
          formatters={"median_autocorr": "{:.2f}".format,
                      "half_life_q": "{:.1f}".format, "median_CV": "{:.2f}".format}))

    print("\n" + "=" * 72)
    print("3. CROSS-METRIC CORRELATION (Spearman, as-of 2021-12-31) - |corr|>0.5 pairs")
    print("=" * 72)
    C = correlation_matrix(panel)
    pairs = []
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1:]:
            c = C.loc[a, b]
            if pd.notna(c) and abs(c) > 0.5:
                pairs.append((a, b, c))
    pairs.sort(key=lambda x: -abs(x[2]))
    for a, b, c in pairs:
        print(f"   {c:+.2f}  {a} ~ {b}")
    if not pairs:
        print("   (no pair |corr|>0.5 -> metrics largely independent)")

    print("\n" + "=" * 72)
    print("4+5. INFORMATION COEFFICIENT vs FORWARD RECONSTRUCTED-PRICE RETURN")
    print("     (DIRECTIONAL - synthetic price, ~0.73 corr to real; CAVEAT)")
    print("=" * 72)
    ic_dates = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]
    tables = {}
    for h, lbl in [(4, "12-month"), (12, "36-month")]:
        ic = information_coefficient(panel, ic_dates, h)
        tables[lbl] = ic
        print(f"\n  --- {lbl} forward horizon (mean IC over {ic_dates}) ---")
        print(ic.to_string(index=False, formatters={"mean_IC": "{:+.3f}".format}))
        comp = ic[ic["metric"] == "COMPOSITE"]["mean_IC"].iloc[0]
        best = ic[ic["metric"] != "COMPOSITE"].iloc[0]
        print(f"  COMPOSITE IC={comp:+.3f}  vs  best single "
              f"({best['metric']})={best['mean_IC']:+.3f}")
    print("\n[model_vs_metric] DONE (parts 1-5).", flush=True)
    return tables


def main():
    print("Loading pickle (offline) ...", flush=True)
    dmdic = pd.read_pickle(PICKLE)
    panel = build_panel(dmdic["cdx_df"])
    print(f"panel rows: {len(panel)}, names: {panel['source'].nunique()}", flush=True)

    print("\n" + "=" * 72)
    print("1+2. PERSISTENCE (half-life) & TIME-VARIANCE (per-name CV)")
    print("=" * 72)
    pv = persistence_and_variance(panel)
    pv = pv.sort_values("median_autocorr", ascending=False)
    print(pv.to_string(index=False,
          formatters={"median_autocorr": "{:.2f}".format,
                      "half_life_q": "{:.1f}".format, "median_CV": "{:.2f}".format}))

    print("\n" + "=" * 72)
    print("3. CROSS-METRIC CORRELATION (Spearman, as-of 2021-12-31) - |corr|>0.5 pairs")
    print("=" * 72)
    C = correlation_matrix(panel)
    pairs = []
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1:]:
            c = C.loc[a, b]
            if pd.notna(c) and abs(c) > 0.5:
                pairs.append((a, b, c))
    pairs.sort(key=lambda x: -abs(x[2]))
    for a, b, c in pairs:
        print(f"   {c:+.2f}  {a} ~ {b}")
    if not pairs:
        print("   (no pair |corr|>0.5 -> metrics largely independent)")

    print("\n" + "=" * 72)
    print("4+5. INFORMATION COEFFICIENT vs FORWARD RECONSTRUCTED-PRICE RETURN")
    print("     (DIRECTIONAL - synthetic price, ~0.73 corr to real; CAVEAT)")
    print("=" * 72)
    ic_dates = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]
    for h, lbl in [(4, "12-month"), (12, "36-month")]:
        ic = information_coefficient(panel, ic_dates, h)
        print(f"\n  --- {lbl} forward horizon (mean IC over {ic_dates}) ---")
        print(ic.to_string(index=False, formatters={"mean_IC": "{:+.3f}".format}))
        comp = ic[ic["metric"] == "COMPOSITE"]["mean_IC"].iloc[0]
        best = ic[ic["metric"] != "COMPOSITE"].iloc[0]
        print(f"  COMPOSITE IC={comp:+.3f}  vs  best single "
              f"({best['metric']})={best['mean_IC']:+.3f}")

    print("\nDONE (parts 1-5). Part 6 (profit-timing) in profit_timing().", flush=True)


if __name__ == "__main__":
    main()
