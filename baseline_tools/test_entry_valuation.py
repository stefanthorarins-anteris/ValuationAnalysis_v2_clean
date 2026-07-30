"""Tests for `epsTTM` and the pick-log entry-valuation columns.

Both were shipped with ZERO tests despite being fully offline-testable, which is the gap these
close.  The pick log is APPEND-ONLY, so a wrong column here is permanent -- that is exactly the
case where the test has to exist before the first row is written, not after.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pick_log as pl


# --------------------------------------------------------------------------- #
#  epsTTM                                                                     #
# --------------------------------------------------------------------------- #
def _tempfund(n, ni, shares, bvps=9.0, freq_months=3):
    return pd.DataFrame({
        "date": pd.date_range("2024-10-01", periods=n, freq="-%dMS" % freq_months),
        "netIncome": np.full(n, ni, dtype=float),
        "weightedAverageShsOut": np.full(n, shares, dtype=float),
        "bookValuePerShare": np.full(n, bvps, dtype=float),
    })


def test_epsTTM_quarterly_sums_four_rows():
    """A quarterly filer's trailing year is 4 rows: 4 x 100 / 50 = 8.0."""
    import getData_fmp as gdf
    out = gdf.stamp_frequency_and_graham(_tempfund(8, 100.0, 50.0))
    assert "epsTTM" in out.columns
    assert float(out["epsTTM"].iloc[0]) == pytest.approx(8.0)
    # the 3 oldest rows have no full trailing year
    assert out["epsTTM"].isna().sum() == 3


def test_epsTTM_semiannual_sums_two_rows():
    """A semi-annual filer's trailing year is 2 rows, not 4: 2 x 100 / 50 = 4.0.  If this
    regressed to 4 rows the P/E would read half its true value for every H1/H2 filer."""
    import getData_fmp as gdf
    import reporting_period as rp
    tf = _tempfund(8, 100.0, 50.0, freq_months=6)
    out = gdf.stamp_frequency_and_graham(tf)
    assert out[rp.FREQ_COLUMN].iloc[0] == rp.SEMIANNUAL
    assert float(out["epsTTM"].iloc[0]) == pytest.approx(4.0)
    assert out["epsTTM"].isna().sum() == 1


def test_epsTTM_is_negative_when_earnings_are_negative():
    """epsTTM itself must carry the sign -- it is the P/E derivation that refuses to publish
    a negative multiple, not this field."""
    import getData_fmp as gdf
    out = gdf.stamp_frequency_and_graham(_tempfund(8, -100.0, 50.0))
    assert float(out["epsTTM"].iloc[0]) == pytest.approx(-8.0)
    assert out["grahamNumber"].isna().all()


def test_epsTTM_shares_the_price_basis():
    """epsTTM must divide by `weightedAverageShsOut` -- the SAME denominator `price` uses --
    so that price/epsTTM is a coherent ratio."""
    import getData_fmp as gdf
    a = gdf.stamp_frequency_and_graham(_tempfund(8, 100.0, 50.0))
    b = gdf.stamp_frequency_and_graham(_tempfund(8, 100.0, 100.0))
    assert float(a["epsTTM"].iloc[0]) == pytest.approx(2 * float(b["epsTTM"].iloc[0]))


# --------------------------------------------------------------------------- #
#  entry valuations                                                           #
# --------------------------------------------------------------------------- #
def _cdx(n_src=8, rows=6, price=10.0, eps=2.0, pb=1.5, graham=12.0, ccy="USD",
         cols_present=("price", "epsTTM", "pbRatio", "grahamNumber", "reportedCurrency")):
    recs = []
    for i in range(n_src):
        for j in range(rows):
            r = {"source": "S%02d" % i,
                 "date": pd.Timestamp("2024-01-01") + pd.DateOffset(months=3 * j)}
            # newest row (last j) carries the values under test; older rows differ, so a
            # regression that averaged instead of taking the newest would be caught
            mult = 1.0 if j == rows - 1 else 0.5
            if "price" in cols_present:
                r["price"] = price * mult
            if "epsTTM" in cols_present:
                r["epsTTM"] = eps * mult
            if "pbRatio" in cols_present:
                r["pbRatio"] = pb * mult
            if "grahamNumber" in cols_present:
                r["grahamNumber"] = graham * mult
            if "reportedCurrency" in cols_present:
                r["reportedCurrency"] = ccy
            recs.append(r)
    return pd.DataFrame(recs)


def test_entry_valuation_reads_the_NEWEST_row_not_an_average():
    v = pl.entry_valuations({"cdx_df": _cdx()})
    r = v["S00"]
    assert r["entry_periodend_price_reporting_ccy"] == pytest.approx(10.0)
    assert r["entry_periodend_trailing_PE"] == pytest.approx(5.0)     # 10 / 2
    assert r["entry_periodend_PB_fmp_basis"] == pytest.approx(1.5)
    assert r["entry_periodend_grahamNumberToPrice"] == pytest.approx(1.2)
    assert r["reporting_currency"] == "USD"


def test_negative_eps_blanks_the_PE_and_never_yields_a_negative_multiple():
    v = pl.entry_valuations({"cdx_df": _cdx(eps=-2.0)})
    assert v["S00"]["entry_periodend_trailing_PE"] == ""
    assert v["S00"]["entry_periodend_price_reporting_ccy"] == pytest.approx(10.0)


def test_zero_eps_blanks_the_PE():
    v = pl.entry_valuations({"cdx_df": _cdx(eps=0.0)})
    assert v["S00"]["entry_periodend_trailing_PE"] == ""


@pytest.mark.parametrize("missing", ["price", "epsTTM", "pbRatio", "grahamNumber",
                                     "reportedCurrency"])
def test_one_missing_column_does_not_blank_the_others(missing):
    """The S4 defect: a single absent cdx column used to blank ALL five fields for EVERY
    ticker.  Each field must now be independently guarded."""
    cols = tuple(c for c in ("price", "epsTTM", "pbRatio", "grahamNumber",
                             "reportedCurrency") if c != missing)
    v = pl.entry_valuations({"cdx_df": _cdx(cols_present=cols)})
    r = v["S00"]
    populated = [k for k, val in r.items() if val != "" and k != "entry_industry_median_n"]
    assert len(populated) >= 2, \
        "dropping %r blanked nearly everything: %s" % (missing, r)
    if missing == "pbRatio":
        assert r["entry_periodend_PB_fmp_basis"] == ""
        assert r["entry_periodend_price_reporting_ccy"] != ""


def test_no_cdx_frame_returns_empty_and_does_not_raise():
    assert pl.entry_valuations({}) == {}
    assert pl.entry_valuations({"cdx_df": pd.DataFrame()}) == {}


def test_industry_median_requires_a_minimum_peer_count():
    """A one-member industry's 'median' is its own value -- a self-comparison that reads as
    'in line with peers' by construction.  It must be suppressed."""
    assert pl.MIN_PEERS_FOR_INDUSTRY_MEDIAN >= 3
    v = pl.entry_valuations({"cdx_df": _cdx(n_src=2)})
    # with 2 names there cannot be an industry meeting the floor
    for r in v.values():
        assert r["entry_industry_median_periodend_PE"] == ""
        assert r["entry_industry_median_n"] == ""


def test_rows_carry_every_declared_column_and_blanks_where_absent():
    frame = pd.DataFrame({"source": ["S00", "S01"], "AggScore": [1.0, 0.5]})
    rows = pl._rows_from_frame(frame, "GENERAL", 2, {}, pl.entry_valuations(
        {"cdx_df": _cdx(n_src=2)}))
    assert len(rows) == 2
    for row in rows:
        for c in pl._VAL_COLS:
            assert c in row


# --------------------------------------------------------------------------- #
#  header-width guard                                                         #
# --------------------------------------------------------------------------- #
def test_append_refuses_on_header_width_drift(tmp_path):
    """Append-only + no per-block header = a column change silently mis-aligns every future
    row against a header that can never be rewritten.  It must refuse."""
    p = tmp_path / "pick_log.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(["as_of", "ticker"])  # stale 2-col header
        f.write("2026-01-01,AAA\n")
    with pytest.raises(RuntimeError, match="SCHEMA DRIFT"):
        pl.append_pick_log([{"as_of": "2026-07-29", "ticker": "BBB"}], path=str(p))


def test_append_writes_full_header_on_a_fresh_file(tmp_path):
    p = tmp_path / "pick_log.csv"
    n = pl.append_pick_log([{"as_of": "d", "ticker": "T"}], path=str(p))
    assert n == 1
    with open(p, encoding="utf-8", newline="") as f:
        hdr = next(csv.reader(f))
    assert hdr == list(pl.PICK_LOG_COLUMNS)
    assert "entry_periodend_price_reporting_ccy" in hdr
    assert "entry_price" not in hdr, "the misleading name must not reappear"


def test_column_names_declare_the_basis():
    """Naming is the whole remedy for S2 -- assert it rather than trust it."""
    cols = pl.PICK_LOG_COLUMNS
    assert "entry_price" not in cols and "entry_PB" not in cols
    assert "reporting_currency" in cols
    for c in cols:
        if c.startswith("entry_periodend"):
            assert "periodend" in c
