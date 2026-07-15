"""
MSCI World benchmark loader (offline).

INPUT WE DO NOT HAVE YET  (flagged -- this module reads a file the CEO sources):
a CSV of MSCI World index levels. This module NEVER fetches anything; it parses
whatever the CEO drops in and normalizes it to a clean date->level series that
the bundle's benchmark.csv is built from.

CEO DECISION REQUIRED (flagged, load-bearing for correctness)
-------------------------------------------------------------
Which MSCI World variant? The stock side uses adjClose (split+DIVIDEND adjusted),
i.e. an implicit TOTAL-return series. To compare apples-to-apples the benchmark
MUST also be total-return:
    RECOMMENDED: "MSCI World Net Total Return, USD"  (dividends reinvested net of
    withholding tax -- the standard investable benchmark).
    WRONG here : "MSCI World Price" index (excludes dividends) -- would understate
    the benchmark by ~1.5-2.5pp/yr, ~5-8pp over 36mo -> silently inflates our
    beat-rate by nearly half the 10pp bar. Do not use the price index.
The chosen variant is recorded in the bundle manifest ("benchmark_variant").

Currency: per the 2026-07-11 FX-deferred decision we compare each stock's
LOCAL-currency return to MSCI World; a return RATIO is currency-neutral, so USD
MSCI World is acceptable as the single yardstick, with non-USD names flagged
lower-confidence (handled in beat_rate.py, not here).
"""

import pandas as pd


def load_benchmark(csv_path, date_col="date", level_col="level"):
    """Read a benchmark CSV -> pd.Series(level) indexed by sorted DatetimeIndex.

    Tolerant of common column-name variants if the defaults are absent.
    """
    df = pd.read_csv(csv_path)

    if date_col not in df.columns:
        for c in ("Date", "DATE", "Exchange Date", "date"):
            if c in df.columns:
                date_col = c
                break
    if level_col not in df.columns:
        for c in ("level", "Level", "Close", "close", "Index Level", "Adj Close",
                  "Value", "value"):
            if c in df.columns:
                level_col = c
                break

    if date_col not in df.columns or level_col not in df.columns:
        raise ValueError(
            f"benchmark CSV must have a date column and a level column; "
            f"got {list(df.columns)}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "level": pd.to_numeric(df[level_col], errors="coerce"),
    }).dropna()
    s = out.set_index("date")["level"].sort_index()
    if s.empty:
        raise ValueError("benchmark series parsed to empty -- check columns/format")
    return s


def level_on_or_before(series, target_date):
    """Nearest index level on or before target_date (fwd-fill semantics)."""
    target_date = pd.Timestamp(target_date)
    earlier = series.loc[:target_date]
    if not earlier.empty:
        return float(earlier.iloc[-1])
    # fallback: earliest available after target
    later = series.loc[target_date:]
    if not later.empty:
        return float(later.iloc[0])
    return float("nan")


def window_return(series, buy_date, eval_date, require_exact=False):
    """Total return of the benchmark over [buy_date, eval_date].

    require_exact (LOW-2 guard, 2026-07-14): when True, both buy_date and
    eval_date MUST be present as exact index anchors in `series`. If an anchor
    is absent the function RAISES rather than silently forward-filling a stale
    earlier level -- forward-fill on a missing EVAL anchor would understate/over-
    state the benchmark by a whole period and silently corrupt the beat-rate.
    Default False preserves the original forward-fill semantics for existing
    callers; the rebalance/tuning path opts in with require_exact=True. (Verified
    2026-07-14: URTH is present at every year-end anchor in the standard grid, so
    turning the guard on does not change any shipped number -- it only fails loud
    if a future anchor is missing.)
    """
    if require_exact:
        b_ts, e_ts = pd.Timestamp(buy_date), pd.Timestamp(eval_date)
        if b_ts not in series.index:
            raise KeyError(f"benchmark missing exact buy anchor {buy_date!r} "
                           f"(require_exact); refusing to forward-fill a stale level")
        if e_ts not in series.index:
            raise KeyError(f"benchmark missing exact eval anchor {eval_date!r} "
                           f"(require_exact); refusing to forward-fill a stale level")
        b, e = float(series.loc[b_ts]), float(series.loc[e_ts])
    else:
        b = level_on_or_before(series, buy_date)
        e = level_on_or_before(series, eval_date)
    if not b or pd.isna(b) or pd.isna(e):
        return float("nan")
    return e / b - 1.0
