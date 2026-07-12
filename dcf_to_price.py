"""
DcfToPrice with a SPLIT-CONSISTENT denominator  (design s2B / s5A, pre-mortem S3).

DcfToPrice = computed DCF fair value per share  /  as-of-D price.

The subtlety (design s5A, code-fact confirmed): the price series used elsewhere is
FMP `adjClose`, which is divided by the CUMULATIVE split factor INCLUDING splits
AFTER D.  The DCF numerator (dcf.fair_value_per_share) is built on the as-of-D
`weightedAverageShsOut` (a PRE-split count for any name that split after D).  So:

  * for BETA (beta.py) adjClose is CORRECT -- a uniform split factor cancels in a
    return ratio;
  * for the DcfToPrice LEVEL adjClose is WRONG as-is -- for any name that split
    after D, adjClose has been divided by the post-D split factor while the share
    count has not, shifting the ratio by that factor.

Fix: the denominator must be on a split basis CONSISTENT with the as-of-D share
count.  Options, in order of preference:
  1. RAW (unadjusted) close as-of D                              -> preferred;
  2. adjClose re-based to D (undo post-D splits) with matched shares.

This module takes the price basis explicitly so the reviewer can confirm the
denominator basis matches the share basis (design s12 reviewer item 4 / S3).
DcfToPrice is a low-weight RANK signal only -- feed it through the SAME
winsorize/tanh as every other metric; its LEVEL is not trusted.
"""
import numpy as np
import pandas as pd


def as_of_price(price_df, D=None, price_col="close", date_col="date",
                basis="raw"):
    """Return the price on/before D on the requested split `basis`.

    price_df : per-symbol price frame with a date column and BOTH a raw close and,
        ideally, an adjClose + a post-D split-factor column when basis != 'raw'.
    basis : 'raw' (unadjusted close -- split-consistent with as-of-D shares, the
        preferred DcfToPrice denominator) or 'adjclose' (returns adjClose as-is;
        ONLY correct if no post-D split -- caller must know this).
    """
    df = price_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    if D is not None:
        df = df[df[date_col] <= pd.Timestamp(D)]
    if df.empty:
        return np.nan
    # LOAD-BEARING ASSUMPTION (MEDIUM-2, pending fmp-specialist confirmation):
    # basis='raw' selects the column literally named `close` as the split-consistent
    # denominator.  This is correct IFF FMP's historical-price-full `close` is genuinely
    # split-UNADJUSTED.  If FMP `close` already incorporates splits (including post-D
    # splits), then for any name that split after D this denominator carries a post-D
    # split factor the as-of-D-shares numerator does not -- silently re-introducing the
    # exact S3 bug this module exists to fix.  Do NOT change the column logic until the
    # fmp-specialist confirms `close` split behaviour; if it is adjusted, re-base to D.
    col = price_col if basis == "raw" else "adjClose"
    if col not in df.columns:
        return np.nan
    val = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(val.iloc[-1]) if not val.empty else np.nan


def dcf_to_price(fair_value_per_share, price, denom_basis="raw"):
    """DcfToPrice ratio.  NaN (never 0) if the numerator is NaN or the price is
    missing/non-positive -- so a degenerate DCF propagates as MISSING (MED#7),
    it is not neutralised here."""
    if fair_value_per_share is None or not np.isfinite(fair_value_per_share):
        return np.nan
    if price is None or not np.isfinite(price) or price <= 0:
        return np.nan
    return float(fair_value_per_share / price)
