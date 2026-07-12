"""
Point-in-time slicing spine  (design s1, s2 L1/L4, s5).

One idea threads the pipeline: an `as_of` date D that flows through universe ->
data slice -> metrics -> scoring.  The invariant is:

    as_of=None  ==  today  ==  live behaviour, BIT-FOR-BIT unchanged.

Every function here returns its input untouched when as_of is None, so wiring a
call site through these helpers can never change a live run.  Only when a real D
is supplied does any filtering happen.

L1 (reporting-lag lookahead) fix
--------------------------------
utils.setDatesToQuarterly stamps the FMP period-END date to the quarter START, and
no filing date is stored.  A Dec-31 report (filed ~Feb-Mar) is stamped Oct-01, so a
naive `date <= D` filter admits it ~5 months early.  We instead filter on an
AVAILABILITY date:

    availability = acceptedDate else fillingDate      (if the columns exist)
                   else  period_end + fixed lag        (fallback, biased LONG)

period_end is reconstructed from the quarter-start stamp (Period.end_time), and the
fallback lag is 90d (quarterly) / 120d (annual) -- biased long so the fallback can
only ever be conservative (no lookahead), per design s2 PIT semantics / pre-mortem
S5.  The quarter-start stamp is kept for DISPLAY only.

NOTE (fetch-fact, deferred): the in-house pickle carries NEITHER fillingDate nor
acceptedDate, so the fallback lag is what runs today.  Whether FMP serves filing
dates historically is an open fmp-specialist question (design s11 Q2) -- until then
every PIT number rests on the fixed-lag fallback and should carry the
filing-date-availability confidence axis (design s2 PIT semantics).
"""
import numpy as np
import pandas as pd

LAG_QUARTERLY_DAYS = 90
LAG_ANNUAL_DAYS = 120


def _period_end_from_quarter_start(dates):
    """Quarter-START stamp -> period-END (design L1). Vectorised, NaT-safe."""
    dt = pd.to_datetime(pd.Series(dates), errors="coerce")
    # end_time of the quarter that STARTS at the stamp
    per = dt.dt.to_period("Q")
    return per.dt.end_time


def availability_date(df, period="quarter"):
    """Return a Series of availability dates for the rows of df.

    Prefers acceptedDate, then fillingDate (if present as columns); otherwise
    period_end + fixed long lag.  df must have a 'date' column (quarter-start stamp).
    """
    n = len(df)
    out = pd.Series([pd.NaT] * n, index=df.index, dtype="datetime64[ns]")
    for col in ("acceptedDate", "fillingDate"):
        if col in df.columns:
            cand = pd.to_datetime(df[col], errors="coerce")
            out = out.fillna(cand)
    # fixed-lag fallback for whatever remains unresolved
    if out.isna().any():
        lag = LAG_ANNUAL_DAYS if period == "annual" else LAG_QUARTERLY_DAYS
        fallback = _period_end_from_quarter_start(df["date"]) + pd.Timedelta(days=lag)
        fallback.index = df.index
        out = out.fillna(fallback)
    return out


def slice_panel_as_of(panel, D=None, period="quarter", date_col="date"):
    """Restrict a per-entity/-quarter panel to rows AVAILABLE on/before D.

    as_of=None -> returns `panel` unchanged (live invariant).  Otherwise keeps rows
    whose availability date <= D.  Does NOT reorder or otherwise mutate columns.
    """
    if D is None:
        return panel
    if panel.empty:
        return panel
    avail = availability_date(panel, period=period)
    keep = avail <= pd.Timestamp(D)
    return panel[keep.values]


def slice_to_entity(series, ipo_date=None, delisted_date=None, date_col="date"):
    """Truncate a price/fundamentals frame to an entity's life-span
    [ipoDate, delistedDate]  (design Component 4/8 -- the BBBY recycled-ticker trap).

    A live current occupant has delisted_date=None -> sliced at its ipoDate only.
    Accepts a DataFrame (uses date_col) or a Series indexed by date.
    """
    if isinstance(series, pd.Series):
        idx = pd.to_datetime(series.index)
        mask = pd.Series(True, index=series.index)
        if ipo_date is not None:
            mask &= idx >= pd.Timestamp(ipo_date)
        if delisted_date is not None:
            mask &= idx <= pd.Timestamp(delisted_date)
        return series[mask.values]
    df = series
    dts = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if ipo_date is not None:
        mask &= dts >= pd.Timestamp(ipo_date)
    if delisted_date is not None:
        mask &= dts <= pd.Timestamp(delisted_date)
    return df[mask.values]
