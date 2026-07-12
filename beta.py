"""
Point-in-time beta by trailing-window regression  (Build Task #1, design s2C).

Replaces the betas.xls Jan-2026 snapshot and the live-profile beta.  Adjudicated
spec (valuation-specialist 2026-07-12):

  * Trailing 60-month MONTHLY OLS ending at D; min 24 observations else NaN
    (never the old 1.0 fallback for the standalone / DCF beta).
  * adjClose returns -- split-neutral (a uniform split factor cancels in a return
    ratio), so adjClose is CORRECT for beta (unlike the DcfToPrice LEVEL; see
    dcf_to_price.py / design s5A).
  * Region-matched index, MSCI World fallback -> caller supplies the index series.
  * Blume shrinkage: beta_adj = 0.67*beta_raw + 0.33.
  * Volume-continuity guard: the caller slices prices to the entity life-span
    FIRST; if any zero/missing-volume gap > ~10 trading days in the final 60 days
    -> beta = NaN AND death-classification drops to low-confidence.

In CycleHeat a beta->1.0 fallback is acceptable (neutral multiplier), but that
fallback belongs to the CycleHeat CONSUMER, not here: beta_as_of returns NaN when
the window is too short so the DCF path never silently uses 1.0.
"""
import numpy as np
import pandas as pd

WINDOW_MONTHS = 60
MIN_OBS = 24
BLUME_SLOPE = 0.67
BLUME_INTERCEPT = 0.33
VOLUME_GAP_MAX_DAYS = 10       # max zero/missing-volume run in the final window
VOLUME_GUARD_DAYS = 60         # "final 60 days" over which the gap is measured


def _monthly_returns(prices):
    """Daily adjClose Series (DatetimeIndex) -> monthly simple returns (last obs
    per calendar month)."""
    s = prices.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    monthly = s.resample("ME").last()
    return monthly.pct_change().dropna()


def volume_continuity_ok(volume, D):
    """True if there is no zero/missing-volume run longer than VOLUME_GAP_MAX_DAYS
    trading days within the final VOLUME_GUARD_DAYS days ending at D.  A failure
    means the terminal price series is unreliable (successor stitch / halt) -> the
    caller sets beta=NaN and drops the death-classification to low-confidence."""
    if volume is None or len(volume) == 0:
        return True   # no volume info -> guard cannot fire; caller decides
    v = pd.Series(volume).copy()
    v.index = pd.to_datetime(v.index)
    v = v.sort_index()
    if D is not None:
        v = v[v.index <= pd.Timestamp(D)]
    tail = v.tail(VOLUME_GUARD_DAYS)
    if tail.empty:
        return True
    bad = (tail.isna()) | (tail <= 0)
    # longest consecutive run of bad days
    run = maxrun = 0
    for flag in bad.values:
        run = run + 1 if flag else 0
        maxrun = max(maxrun, run)
    return maxrun <= VOLUME_GAP_MAX_DAYS


def beta_as_of(stock_prices, index_prices, D=None, volume=None):
    """Blume-shrunk PIT beta of stock vs index as-of D.

    Parameters
    ----------
    stock_prices : Series of daily adjClose indexed by date (entity-life-sliced by
        the caller, design Component 4/8).
    index_prices : Series of daily adjClose for the region-matched index (MSCI
        World fallback), indexed by date.
    D : as-of date or None (=use full history / today).
    volume : optional daily volume Series for the volume-continuity guard.

    Returns
    -------
    (beta_adj, info) : beta_adj is float or np.nan; info carries n_obs, beta_raw
    and the reason for NaN (incl. 'volume_gap' which the caller maps to
    low-confidence death classification).
    """
    info = {"reason": None, "n_obs": 0, "beta_raw": np.nan}

    if not volume_continuity_ok(volume, D):
        info["reason"] = "volume_gap"       # caller -> low-confidence + NaN
        return np.nan, info

    sp = pd.Series(stock_prices).copy()
    ip = pd.Series(index_prices).copy()
    sp.index = pd.to_datetime(sp.index)
    ip.index = pd.to_datetime(ip.index)
    if D is not None:
        Dts = pd.Timestamp(D)
        sp = sp[sp.index <= Dts]
        ip = ip[ip.index <= Dts]
    # trailing 60 months ending at the last available date <= D
    if sp.empty or ip.empty:
        info["reason"] = "no_prices"
        return np.nan, info
    end = min(sp.index.max(), ip.index.max())
    start = end - pd.DateOffset(months=WINDOW_MONTHS)
    sp = sp[(sp.index > start) & (sp.index <= end)]
    ip = ip[(ip.index > start) & (ip.index <= end)]

    rs = _monthly_returns(sp)
    ri = _monthly_returns(ip)
    j = pd.concat([rs.rename("s"), ri.rename("i")], axis=1).dropna()
    info["n_obs"] = len(j)
    if len(j) < MIN_OBS:
        info["reason"] = "insufficient_obs"
        return np.nan, info

    var_i = j["i"].var()
    if not np.isfinite(var_i) or var_i == 0:
        info["reason"] = "zero_index_variance"
        return np.nan, info
    beta_raw = j["s"].cov(j["i"]) / var_i
    info["beta_raw"] = float(beta_raw)
    beta_adj = BLUME_SLOPE * beta_raw + BLUME_INTERCEPT
    info["reason"] = "ok"
    return float(beta_adj), info
