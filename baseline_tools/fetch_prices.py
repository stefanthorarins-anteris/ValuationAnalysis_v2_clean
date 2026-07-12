"""
Standalone REAL-PRICE fetch step for the investment-filter baseline.

=============================================================================
 THIS SCRIPT MAKES NETWORK CALLS.  IT IS NEVER RUN BY THE HOUSE / ON THE
 WORK MACHINE AS PART OF DEVELOPMENT.  The CEO runs it MANUALLY -- on either
 the work machine (LOW VOLUME ONLY) or a personal machine -- with no code
 change; only *where* it is invoked differs.
=============================================================================

WHY IT EXISTS
-------------
The pipeline's `price` column is SYNTHETIC (getData_fmp.py:171 reconstructs it
as priceEarningsRatio * netIncome/shares). Its *returns* track real returns but
carry +/-15-23pp per-name error -- too coarse for a +/-10pp beat bar. The
backtest's buy/sell legs must therefore use REAL split/dividend-adjusted closes.

DESIGN GOAL: MINIMUM TOTAL CALL COUNT (CEO, VPN/IT-monitoring concern)
---------------------------------------------------------------------
We fetch the WHOLE universe's adjusted close for ONE date per call using the
bulk-by-date endpoint. We never loop per symbol over the universe.

  Baseline (default: buy 2019/2020/2021-12-31, 36-month hold):
      distinct anchor dates = 3 buy + 3 eval = 6
      => 6 calls nominal (1 per date, if the date is a trading day)
      => holiday/weekend fallback steps back up to `--max-lookback` (default 4)
         days PER DATE, stopping at the first non-empty response.
      => WORST case 6 * (1 + 4) = 30 calls; TYPICAL 6-10 calls.
         (Most anchors are pre-mapped to the nearest weekday, so most dates
          resolve on the first attempt.)

  Optional fuller run:
      N distinct anchor dates  =>  N calls (same 1-call-per-date property).
      e.g. 5 buy-years x (buy+eval) = 10 dates = 10 calls nominal.

  The script PRINTS THE EXACT CALL COUNT it made at the end so exposure can be
  judged at a glance.

  --endpoint bulk   (DEFAULT) : v4/batch-request-end-of-day-prices?date=D
                                -> one call per date, whole universe.
  --endpoint batch  (OPT-IN)  : v3/historical-price-full/A,B,C,D,E  (<=5 syms).
                                THIS IS THE HIGH-VOLUME FALLBACK:
                                ceil(n_symbols/5) calls PER date -> HUNDREDS to
                                THOUSANDS. It refuses to run unless
                                --i-understand-high-volume is also passed, and
                                is intended ONLY for the personal machine.

OUTPUT
------
A single long CSV (default: baseline_tools/price_data/real_prices_<stamp>.csv):
      date_requested, date_actual, symbol, adjClose
The offline checker (Stage-2 backtest) consumes this file by joining
(symbol, date_actual) to the top-20 at each buy date. No network on that side.

  NOTE: keep price_data/ OUT of git (large + regenerable). Recommend adding
  `baseline_tools/price_data/` to .gitignore before any commit.

USAGE (run manually by the CEO)
-------------------------------
  # Baseline, bulk endpoint (6 calls nominal):
  python baseline_tools/fetch_prices.py \
      --api-key-file fmpAPIkey.txt \
      --buy-years 2019,2020,2021 --hold-months 36 \
      --out baseline_tools/price_data/real_prices.csv

  # Restrict the output file to just the symbols we care about (LOCAL filter,
  # costs NO extra calls -- the bulk call still returns the whole universe):
  python baseline_tools/fetch_prices.py ... --symbols-file top_symbols.txt

The same invocation works on either machine. On the work machine keep to the
bulk endpoint and the baseline date set (single-digit call count).
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta

import requests


BASEURL = "https://financialmodelingprep.com/api/"


def read_api_key(path):
    with open(path, "r") as f:
        return f.read().strip()


def nearest_weekday_on_or_before(d):
    """Map a calendar date back to the nearest weekday (Mon-Fri).

    Cheap pre-resolution so most anchors hit a trading day on the first call
    and we don't waste holiday-fallback attempts on Sat/Sun.
    """
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def build_anchor_dates(buy_years, hold_months):
    """Return the sorted set of distinct (buy + eval) anchor dates.

    Buy = <year>-12-31; eval = buy + hold_months (approximated as
    year + hold_months//12 on the same 12-31 anchor to keep it a clean
    calendar window). All anchors are pre-mapped to the nearest weekday.
    """
    hold_years = hold_months // 12
    anchors = set()
    for y in buy_years:
        buy = datetime(y, 12, 31)
        ev = datetime(y + hold_years, 12, 31)
        anchors.add(nearest_weekday_on_or_before(buy).date())
        anchors.add(nearest_weekday_on_or_before(ev).date())
    return sorted(anchors)


def fetch_bulk_for_date(api_key, date_str, timeout=30):
    """One call to the bulk-by-date endpoint. Returns list-of-dict rows or []."""
    url = f"{BASEURL}v4/batch-request-end-of-day-prices?date={date_str}&apikey={api_key}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    # FMP may return JSON (list of dicts) or CSV text depending on tier/params.
    ctype = resp.headers.get("Content-Type", "")
    if "json" in ctype:
        data = resp.json()
        return data if isinstance(data, list) else []
    # CSV fallback parse
    text = resp.text.strip()
    if not text:
        return []
    rows = list(csv.DictReader(text.splitlines()))
    return rows


def _extract(row):
    """Pull (symbol, adjClose) from one bulk row, tolerating key variants."""
    sym = row.get("symbol") or row.get("Symbol")
    adj = (row.get("adjClose") or row.get("adjclose") or row.get("adj_close")
           or row.get("close") or row.get("Close"))
    return sym, adj


def run_bulk(api_key, anchors, max_lookback, symbols_filter, out_path):
    """Fetch each anchor date via the bulk endpoint; step back on empty days."""
    call_count = 0
    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["date_requested", "date_actual", "symbol", "adjClose"])
        for anchor in anchors:
            got = False
            for back in range(max_lookback + 1):
                d = anchor - timedelta(days=back)
                date_str = d.strftime("%Y-%m-%d")
                call_count += 1
                print(f"[call {call_count}] bulk date={date_str} "
                      f"(anchor {anchor}) ...", flush=True)
                try:
                    rows = fetch_bulk_for_date(api_key, date_str)
                except requests.RequestException as e:
                    print(f"    request error: {e}", flush=True)
                    rows = []
                if rows:
                    for row in rows:
                        sym, adj = _extract(row)
                        if not sym or adj in (None, "", "null"):
                            continue
                        if symbols_filter and sym not in symbols_filter:
                            continue
                        w.writerow([anchor.strftime("%Y-%m-%d"), date_str, sym, adj])
                        written += 1
                    got = True
                    print(f"    OK: {len(rows)} rows returned", flush=True)
                    break
                time.sleep(0.3)  # be gentle
            if not got:
                print(f"    WARNING: no data for anchor {anchor} within "
                      f"{max_lookback} lookback days", flush=True)
    return call_count, written


def main():
    ap = argparse.ArgumentParser(description="Real-price fetch (bulk-by-date).")
    ap.add_argument("--api-key-file", default="fmpAPIkey.txt")
    ap.add_argument("--buy-years", default="2019,2020,2021",
                    help="comma-separated buy years (buy date = <y>-12-31)")
    ap.add_argument("--hold-months", type=int, default=36)
    ap.add_argument("--max-lookback", type=int, default=4,
                    help="holiday/weekend fallback: days to step back per anchor")
    ap.add_argument("--symbols-file", default=None,
                    help="optional local symbol allow-list (LOCAL filter, no "
                         "extra calls) -- one symbol per line")
    ap.add_argument("--out", default="baseline_tools/price_data/real_prices.csv")
    ap.add_argument("--endpoint", choices=["bulk", "batch"], default="bulk")
    ap.add_argument("--i-understand-high-volume", action="store_true",
                    help="required to enable the batch (per-symbol) fallback")
    args = ap.parse_args()

    if args.endpoint == "batch" and not args.i_understand_high_volume:
        sys.exit("REFUSING: --endpoint batch makes ceil(n_symbols/5) calls PER "
                 "date (hundreds-thousands). Pass --i-understand-high-volume "
                 "and run ONLY on a personal machine if you truly need it.")
    if args.endpoint == "batch":
        sys.exit("The per-symbol batch fallback is intentionally not implemented "
                 "in this low-volume baseline tool. Use --endpoint bulk.")

    api_key = read_api_key(args.api_key_file)
    buy_years = [int(y) for y in args.buy_years.split(",") if y.strip()]
    anchors = build_anchor_dates(buy_years, args.hold_months)

    symbols_filter = None
    if args.symbols_file:
        with open(args.symbols_file) as f:
            symbols_filter = {ln.strip() for ln in f if ln.strip()}

    print(f"Anchor dates ({len(anchors)} distinct -> {len(anchors)} nominal "
          f"calls): {[a.isoformat() for a in anchors]}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out
    if "{stamp}" in out_path:
        out_path = out_path.format(stamp=stamp)

    calls, written = run_bulk(api_key, anchors, args.max_lookback,
                              symbols_filter, out_path)
    print("=" * 60, flush=True)
    print(f"DONE. TOTAL API CALLS MADE: {calls}", flush=True)
    print(f"Rows written: {written} -> {out_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
