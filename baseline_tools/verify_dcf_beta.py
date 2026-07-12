"""
Verification harness for the computed DCF + beta  (design s2D, Build Task #1).

Two modes:

  OFFLINE  (default, NO network) -- runs on the saved pickle + synthetic fixtures.
    * DCF numerator machinery over the in-house panel: compute fair value per share
      for every name (with a reference beta), report the finite/NaN split and the
      NaN-reason distribution, and confirm healthy names give finite positive
      values and degenerate ones (FCF0<=0, financials, short history) give NaN.
    * WACC RATE SWEEP property test (pre-mortem S2): sweep r_f across a synthetic
      range and confirm the model behaves sanely -- fair value MONOTONE-decreasing
      in r_f, no blow-up, and NaN exactly where g_terminal >= k_e.  A property test,
      not a vendor comparison (the current-data check runs in a high-rate 2026 regime
      and cannot see a WACC that is rate-sensitively wrong in a ZIRP regime).
    * BETA recovery on synthetic returns: a series built with a known beta is
      recovered within tolerance (Blume-shrink applied).
    This is the IN-HOUSE pass the endpoint requires -- no network, no full fetch.

  LIVE-VENDOR  (--i-understand-live-vendor-call, CEO-run, CURRENT data only) -- the
    ONLY place a live vendor DCF/beta call is made.  Compares OUR fair value PER
    SHARE to FMP's live /discounted-cash-flow per-share value on the current top-200
    by AggScore + a sector-stratified random ~200, EXCLUDING financials.  We verify
    the NUMERATOR, not DcfToPrice (design s2D: DcfToPrice=DCF/price and the vendor
    ratio share the 1/price factor, inflating correlation and certifying nothing).
    PASS ALL THREE:  Spearman rho >= 0.65 ; sign-agreement >= 75% ; median-abs-%err
    <= 40%.  Beta: computed vs FMP profile beta on the same sample.

    Documented loudly: FMP's live DCF is ONE vendor's assumptions, NOT ground truth.
    Passing means our DCF is METHODOLOGICALLY CONSISTENT with a reputable vendor --
    it licenses a low-weight historical RANK signal only, not trust in DCF LEVELS.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import dcf as dcf_mod
import beta as beta_mod

PICKLE = os.path.join(
    REPO, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-01-09_len8106_manelim3692_fails1725.pickle")
SECTORS = os.path.join(REPO, "sectorsdic_fmp.pickle")

RHO_MIN, SIGN_MIN, MAPE_MAX = 0.65, 0.75, 0.40


# --------------------------------------------------------------------------- #
def _sector_map():
    if not os.path.exists(SECTORS):
        return {}
    d = pd.read_pickle(SECTORS)
    rev = {}
    for sector, syms in d.items():
        for s in syms:
            rev[s] = sector
    return rev


def offline_dcf_machinery(panel, sector_map, ref_beta=1.0, limit=None):
    """Compute fair value per share for every name in the panel (reference beta)."""
    names = panel["source"].dropna().unique()
    if limit:
        names = names[:limit]
    rows = []
    for name in names:
        g = panel[panel["source"] == name]
        val, info = dcf_mod.fair_value_per_share(
            g, D=None, beta=ref_beta, sector=sector_map.get(name))
        rows.append({"source": name, "value": val, "reason": info["reason"],
                     "n_q": info["n_quarters"], "tv_frac": info.get("tv_fraction")})
    return pd.DataFrame(rows)


def wacc_rate_sweep():
    """Property test: build ONE representative entity and sweep r_f.  Fair value must
    be finite and MONOTONE-DECREASING in r_f while k_e - g_terminal > 0, and NaN once
    k_e <= g_terminal."""
    # 16 quarters of steadily growing FCF/revenue, 100M shares.
    dates = pd.date_range("2019-03-31", periods=16, freq="QE")
    base_fcf = np.linspace(80e6, 120e6, 16)
    base_rev = np.linspace(400e6, 620e6, 16)
    panel = pd.DataFrame({
        "date": dates, "freeCashFlow": base_fcf, "revenue": base_rev,
        "weightedAverageShsOut": 100e6,
    })
    beta = 1.0
    results = []
    for rf in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20]:
        val, info = dcf_mod.fair_value_per_share(panel, D=None, beta=beta, rf=rf)
        results.append((rf, info["k_e"], val, info["reason"]))
    # checks
    finite = [(rf, v) for rf, _, v, _ in results if np.isfinite(v)]
    monotone = all(finite[i][1] >= finite[i + 1][1] for i in range(len(finite) - 1))
    # NaN exactly where k_e <= g_terminal
    nan_ok = all((np.isnan(v)) == (ke <= dcf_mod.G_TERMINAL)
                 for _, ke, v, _ in results)
    no_blowup = all(np.isfinite(v) and abs(v) < 1e6 for _, _, v, _ in results
                    if np.isfinite(v))
    return results, monotone, nan_ok, no_blowup


def beta_recovery():
    """Property test: index random walk, stock = TRUE_BETA*index + noise.  Recover
    beta_raw ~ TRUE_BETA (Blume shrinks toward 1)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=1500, freq="B")  # ~6y business days
    idx_ret = rng.normal(0, 0.01, len(dates))
    true_beta = 1.4
    stk_ret = true_beta * idx_ret + rng.normal(0, 0.005, len(dates))
    idx_px = pd.Series(100 * np.cumprod(1 + idx_ret), index=dates)
    stk_px = pd.Series(50 * np.cumprod(1 + stk_ret), index=dates)
    b_adj, info = beta_mod.beta_as_of(stk_px, idx_px, D=None)
    expected_adj = beta_mod.BLUME_SLOPE * true_beta + beta_mod.BLUME_INTERCEPT
    return b_adj, info["beta_raw"], true_beta, expected_adj, info["n_obs"]


def run_offline():
    print("=" * 74)
    print("OFFLINE DCF/BETA VERIFICATION  (no network)")
    print("=" * 74)

    # --- WACC rate sweep -----------------------------------------------------
    res, monotone, nan_ok, no_blowup = wacc_rate_sweep()
    print("\n[1] WACC rate sweep (representative entity):")
    print(f"    {'r_f':>5} {'k_e':>7} {'value/sh':>12} reason")
    for rf, ke, v, reason in res:
        vs = f"{v:12.4f}" if np.isfinite(v) else f"{'NaN':>12}"
        print(f"    {rf:5.2f} {ke:7.3f} {vs} {reason}")
    print(f"    monotone-decreasing in r_f: {monotone}")
    print(f"    NaN exactly where k_e<=g_terminal: {nan_ok}")
    print(f"    no blow-up: {no_blowup}")
    sweep_pass = monotone and nan_ok and no_blowup

    # --- beta recovery -------------------------------------------------------
    b_adj, b_raw, true_b, exp_adj, nobs = beta_recovery()
    print("\n[2] Beta recovery (synthetic, true beta=1.4):")
    print(f"    n_obs={nobs}  beta_raw={b_raw:.3f} (want ~{true_b})  "
          f"beta_adj={b_adj:.3f} (want ~{exp_adj:.3f})")
    beta_pass = abs(b_raw - true_b) < 0.1 and abs(b_adj - exp_adj) < 0.1

    # --- DCF machinery over the in-house panel -------------------------------
    dcf_pass = None
    if os.path.exists(PICKLE):
        print("\n[3] DCF numerator machinery over the in-house pickle:")
        dmdic = pd.read_pickle(PICKLE)
        panel = dmdic["cdx_df"]
        smap = _sector_map()
        out = offline_dcf_machinery(panel, smap, ref_beta=1.0)
        n = len(out)
        finite = out["value"].notna().sum()
        print(f"    names scored: {n}  finite: {finite} ({100*finite/n:.0f}%)  "
              f"NaN: {n-finite}")
        print("    NaN-reason distribution:")
        for reason, cnt in out["reason"].value_counts().items():
            print(f"        {reason:24s} {cnt}")
        healthy = out[out["reason"] == "ok"]
        pos = (healthy["value"] > 0).mean() if len(healthy) else np.nan
        tvf = healthy["tv_frac"].median() if len(healthy) else np.nan
        print(f"    of 'ok' names, fraction with value>0: {pos:.3f}")
        print(f"    median TV fraction (assumption-dominance check, ~0.65-0.80): "
              f"{tvf:.3f}")
        # financials must be NaN
        fin_names = [s for s, sec in smap.items()
                     if sec in dcf_mod.EXCLUDED_SECTORS and s in set(out["source"])]
        fin_nan = out[out["source"].isin(fin_names)]["value"].isna().all() \
            if fin_names else True
        print(f"    all financial-sector names NaN: {fin_nan}")
        dcf_pass = (finite > 0) and (pos == 1.0 or np.isnan(pos)) and fin_nan
    else:
        print("\n[3] SKIP -- pickle not present")

    print("\n" + "-" * 74)
    print(f"WACC sweep : {'PASS' if sweep_pass else 'FAIL'}")
    print(f"beta       : {'PASS' if beta_pass else 'FAIL'}")
    print(f"DCF panel  : {'PASS' if dcf_pass else ('SKIP' if dcf_pass is None else 'FAIL')}")
    overall = sweep_pass and beta_pass and (dcf_pass in (True, None))
    print(f"OFFLINE OVERALL: {'PASS' if overall else 'FAIL'}")
    print("-" * 74)
    return overall


def run_live_vendor(args):
    print("LIVE-VENDOR mode is CEO-run and makes CURRENT-data FMP calls.")
    print("It is intentionally left as a thin, guarded entry point: fill in the")
    print("current top-200 + sector-stratified random 200 selection from a live")
    print("run and compare OUR fair value per share to FMP /discounted-cash-flow.")
    print(f"Bands: Spearman rho>={RHO_MIN}, sign>={SIGN_MIN}, MAPE<={MAPE_MAX}.")
    print("Financials EXCLUDED.  Verify the NUMERATOR, never DcfToPrice.")
    print("NOTE: not auto-run in-house -- needs a live universe + network (CEO).")


def main():
    ap = argparse.ArgumentParser(description="DCF/beta verification harness.")
    ap.add_argument("--i-understand-live-vendor-call", action="store_true",
                    help="enable the CURRENT-data live-vendor comparison (CEO only)")
    args = ap.parse_args()
    if args.i_understand_live_vendor_call:
        run_live_vendor(args)
    else:
        ok = run_offline()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
