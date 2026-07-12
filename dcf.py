"""
Point-in-time two-stage FCFE DCF  (Build Task #1, restructure design 2026-07-12 s2B).

WRITTEN FRESH -- this is NOT a repair of BoDCF.py.  BoDCF.py's formula is wrong
(it discounts by WACC**i instead of (1+WACC)**i, mixes FCF/revenue/GM growth, and
is un-runnable) and is kept only as a record of the intended endpoints.  The model
here is the valuation-specialist's adjudicated spec:

  * Base FCF0  = MEDIAN of the last 8 quarterly TTM-FCF values (rolling 4-quarter
                 sum, then median of 8), as-of D.  FMP freeCashFlow is treated as a
                 LEVERED FCFE proxy -> discount at cost of EQUITY and divide by
                 shares directly (no EV / net-debt bridge).  *Approximation, flagged.*
  * Growth g0  = trailing REVENUE CAGR (min 3y, max 5y), clamped [-5%, +15%].
  * Horizon    = 10y, LINEAR fade g0 -> g_terminal = 2.5%.
                 PV_t = FCF_t / (1+k)**t  (correct discounting).
                 TV   = FCF10*(1+g_t)/(k - g_t), discounted by (1+k)**10.
  * Discount k = CAPM cost of equity  k_e = r_f + beta*ERP; r_f fixed, ERP fixed 5%,
                 beta = the PIT beta (beta.py, already Blume-shrunk/clamped).  We do
                 NOT derive a per-name WACC from historical D/E.
  * NaN (never 0, never forced) if: base FCF0 <= 0; g_terminal >= k; < 12 quarters;
                 or bank / insurer / REIT.  NaN propagates as MISSING to the scorer
                 (see score_missing.py, MED#7) -- it is never neutralised to 0 here.

The function returns fair value PER SHARE (equity value / as-of-D shares).  The
DcfToPrice ratio (and the split-consistent denominator, design s5A/S3) lives in
dcf_to_price.py, so this module has no price dependency and no network dependency.

as_of=None  -> use the full series' latest row as "D" (live behaviour: newest data).
"""
import numpy as np
import pandas as pd

# --- adjudicated constants (valuation-specialist 2026-07-12) --------------------
RF_DEFAULT = 0.04          # risk-free, single fixed value (spec: ~3.5-4%)
ERP = 0.05                 # equity risk premium, fixed
G_TERMINAL = 0.025         # terminal growth
# Minimum k_e - g_terminal spread.  The Gordon terminal value fcf*(1+g)/(k-g) is
# numerically singular as k -> g: at a 0.1% spread the TV multiple is >1000x, an
# untrustworthy LEVEL that would swamp the (rank-only) DcfToPrice signal (LOW-3).
# Contain it AT THE SOURCE -- below this spread return NaN so the channel propagates
# as MISSING (reliability-shrinkage handles it) rather than a finite blow-up leaking
# downstream on the assumption a winsorize/tanh is wired (it is not yet).  Only names
# with k_e within 100bps of g_terminal (very low / negative beta) are affected.
MIN_KE_SPREAD = 0.01
HORIZON = 10               # explicit forecast years
G_CLAMP = (-0.05, 0.15)    # revenue-CAGR clamp
MIN_QUARTERS = 12          # < 12 quarters -> NaN
TTM_MEDIAN_N = 8           # median over last 8 TTM-FCF values
CAGR_MIN_YEARS = 3
CAGR_MAX_YEARS = 5

# sectors treated as bank / insurer / REIT (NaN by spec).  These are the
# sectorsdic_fmp.pickle keys that carry financial-firm FCF-definition problems.
EXCLUDED_SECTORS = frozenset({
    "Financial Services", "Banking", "Insurance", "Real Estate",
})


def _as_of_slice(g, D):
    """Return the per-entity quarterly frame sorted OLDEST->NEWEST, restricted to
    rows on/before D.  Defensive: sorts internally, never assumes input order."""
    g = g.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"]).sort_values("date")
    if D is not None:
        g = g[g["date"] <= pd.Timestamp(D)]
    return g


def _ttm(series):
    """Rolling trailing 4-quarter sum of a NEWEST-first-agnostic ascending series."""
    return series.rolling(4).sum()


def revenue_cagr(rev_ttm, dates):
    """Trailing revenue CAGR over min 3y / max 5y ending at the newest point.
    rev_ttm/dates are ascending.  Returns clamped CAGR or NaN."""
    rev = pd.Series(np.asarray(rev_ttm, float), index=pd.to_datetime(dates))
    rev = rev.dropna()
    rev = rev[rev > 0]
    if len(rev) < 4:
        return np.nan
    end_val = rev.iloc[-1]
    end_dt = rev.index[-1]
    # earliest point at least CAGR_MIN_YEARS back, but no more than CAGR_MAX_YEARS.
    lo = end_dt - pd.DateOffset(years=CAGR_MAX_YEARS)
    hi = end_dt - pd.DateOffset(years=CAGR_MIN_YEARS)
    window = rev[(rev.index >= lo) & (rev.index <= hi)]
    if window.empty:
        # not enough history for a >=3y CAGR
        return np.nan
    start_val = window.iloc[0]
    start_dt = window.index[0]
    years = (end_dt - start_dt).days / 365.25
    if years <= 0 or start_val <= 0:
        return np.nan
    cagr = (end_val / start_val) ** (1.0 / years) - 1.0
    return float(np.clip(cagr, *G_CLAMP))


def fair_value_per_share(entity_panel, D=None, beta=None, rf=RF_DEFAULT,
                         sector=None):
    """Point-in-time two-stage FCFE fair value PER SHARE for one entity.

    Parameters
    ----------
    entity_panel : DataFrame with columns at least
        ['date','freeCashFlow','revenue','weightedAverageShsOut'] for ONE entity.
    D : as-of date (Timestamp/str) or None (=use newest available row).
    beta : the PIT beta for this entity as-of D (Blume-shrunk, from beta.py).
           If None or NaN, k_e cannot be formed -> NaN.
    rf : risk-free rate (fixed).
    sector : optional sector string; bank/insurer/REIT -> NaN.

    Returns
    -------
    (value_per_share, info) where value_per_share is a float or np.nan and info is
    a dict with the reason for NaN and the intermediate quantities (audit/harness).
    """
    info = {"reason": None, "k_e": np.nan, "g0": np.nan, "fcf0": np.nan,
            "shares": np.nan, "n_quarters": 0}

    if sector is not None and sector in EXCLUDED_SECTORS:
        info["reason"] = "excluded_sector"
        return np.nan, info

    g = _as_of_slice(entity_panel, D)
    info["n_quarters"] = len(g)
    if len(g) < MIN_QUARTERS:
        info["reason"] = "insufficient_history"
        return np.nan, info

    if beta is None or not np.isfinite(beta):
        info["reason"] = "no_beta"
        return np.nan, info

    k_e = rf + beta * ERP
    info["k_e"] = k_e
    if not np.isfinite(k_e) or k_e <= G_TERMINAL:
        # g_terminal >= k -> TV explodes / negative; NaN by spec.
        info["reason"] = "g_terminal_ge_wacc"
        return np.nan, info
    if (k_e - G_TERMINAL) < MIN_KE_SPREAD:
        # k barely exceeds g -> near-singular TV; contain at source (LOW-3).
        info["reason"] = "ke_spread_too_small"
        return np.nan, info

    fcf = pd.to_numeric(g["freeCashFlow"], errors="coerce")
    fcf_ttm = _ttm(fcf).dropna()
    if len(fcf_ttm) < 1:
        info["reason"] = "no_ttm_fcf"
        return np.nan, info
    fcf0 = float(fcf_ttm.tail(TTM_MEDIAN_N).median())
    info["fcf0"] = fcf0
    if not np.isfinite(fcf0) or fcf0 <= 0:
        info["reason"] = "nonpositive_fcf0"
        return np.nan, info

    rev_ttm = _ttm(pd.to_numeric(g["revenue"], errors="coerce"))
    g0 = revenue_cagr(rev_ttm, g["date"])
    if not np.isfinite(g0):
        info["reason"] = "no_revenue_cagr"
        return np.nan, info
    info["g0"] = g0

    shares = pd.to_numeric(g["weightedAverageShsOut"], errors="coerce").dropna()
    if shares.empty or shares.iloc[-1] <= 0:
        info["reason"] = "no_shares"
        return np.nan, info
    shares0 = float(shares.iloc[-1])   # as-of-D share count (newest <= D)
    info["shares"] = shares0

    # linear fade g0 -> g_terminal over t=1..HORIZON (g_1=g0, g_HORIZON=g_terminal)
    if HORIZON > 1:
        gpath = [g0 + (G_TERMINAL - g0) * (t - 1) / (HORIZON - 1)
                 for t in range(1, HORIZON + 1)]
    else:
        gpath = [G_TERMINAL]

    pv_sum = 0.0
    fcf_t = fcf0
    for t in range(1, HORIZON + 1):
        fcf_t = fcf_t * (1.0 + gpath[t - 1])
        pv_sum += fcf_t / (1.0 + k_e) ** t

    tv = fcf_t * (1.0 + G_TERMINAL) / (k_e - G_TERMINAL)
    pv_tv = tv / (1.0 + k_e) ** HORIZON

    equity_value = pv_sum + pv_tv
    value_per_share = equity_value / shares0
    info["reason"] = "ok"
    info["tv_fraction"] = pv_tv / equity_value if equity_value else np.nan
    return float(value_per_share), info
