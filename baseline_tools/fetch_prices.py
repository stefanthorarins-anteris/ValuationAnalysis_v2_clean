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

COMPLETENESS OF A BODY IS NOT "len(rows) > 0"  (added 2026-08-22)
----------------------------------------------------------------
`if rows:` was the ONLY completeness test, and three bodies in the saved pulls pass
it while being unusable.  See MIN_PAYLOAD_ROWS / SHORT_BODY_MEDIAN_FRACTION for the
measured counts and the derivation.  Two acceptance tests are applied now:

  * a PAYLOAD FLOOR -- a body far below the size of the run's other bodies is not a
    trading day's worth of prices.  Enforced in two places for two different reasons
    (absolute backstop in-line, relative-to-median deferred).
  * a WEEKEND REFUSAL -- a Saturday/Sunday candidate date is never requested at all,
    because the endpoint answers with a small non-empty body on a weekend rather than
    an empty one, and `if rows:` accepted it.

NEITHER IS A COMPLETENESS TEST, AND THE DIFFERENCE MATTERS.  Both are MAGNITUDE tests.
A body missing one whole small venue (`.PA` is ~569 names out of ~60,000) is 99% of
full size and sails through untouched -- which is exactly the failure that left seven
venues unpriceable at every anchor in the run machine's grid.  What would catch that is
a PER-VENUE within-run test (every exchange suffix seen at the run's other anchors must
be non-empty at this one), which needs no extra calls and is NOT implemented here
because it is outside the change authorised for this file.  Do not read the floor as
"the body is complete".

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


#  ==========================================================================
#  PAYLOAD ACCEPTANCE -- the two tests that replace `if rows:`
#  ==========================================================================
#
#  MEASURED EVIDENCE (row counts per date_actual across every saved pull on this
#  machine, 2026-08-22).  Two dates were pulled TWICE, and the second pull is the
#  control for the first:
#
#      date         first pull        repull        verdict
#      2020-12-31        9,901        55,438        first pull TRUNCATED
#      2022-12-30       15,441        76,690        first pull TRUNCATED
#      2024-12-28        3,589            --        a SATURDAY (93.8% of its rows
#                                                   are USD-suffixed crypto pairs,
#                                                   zero FX); `if rows:` took it
#
#  and the eleven bodies that ARE a trading day, smallest to largest:
#      45,662  51,646  55,438  58,838  59,965  60,152  62,189  64,490  71,985
#      73,758  76,690           (median 60,152)
#
#  So the two populations are cleanly separated: the largest bad body is 15,441 and
#  the smallest good one is 45,662, a factor of 3.0 apart with nothing in between.
#
#  IT IS *NOT* A BYTE TRUNCATION AND *NOT* A REFRESH THROTTLE, and this is why there
#  is no retry-the-same-date loop below.  The short bodies are VENUE-STRUCTURED: the
#  9,901-row body is 98% foreign-suffixed against 48.7% in its own repull (it dropped
#  the US and kept the rest), and the 15,441-row body is roughly half UK+Canada with
#  Germany and Sweden absent entirely.  A throttle returns an error or an empty body,
#  not a coherent venue-clustered 18%-sized one.  The floor is therefore justified BY
#  THE FLOOR -- "a body this far below the run's other bodies is not a trading day" --
#  and not by any theory of what the endpoint was doing.

#  ABSOLUTE BACKSTOP, applied IN-LINE so the existing step-back can move past a short
#  body on the calls it was already allowed to make.  20,000 sits inside the empty band
#  above (largest bad 15,441, smallest good 45,662) with 1.3x clearance below and 2.3x
#  above.  It is deliberately near the BOTTOM of that band rather than the middle,
#  because the failure direction matters: rejecting a real body costs a lookback step,
#  while accepting a truncated one silently corrupts every backtest number computed off
#  that anchor.
#
#  IT IS AN ABSOLUTE COUNT AND SO IT IS THE ONE NUMBER HERE THAT WILL AGE.  The vendor's
#  universe grew 45,662 -> 76,690 rows/day over 2018-2022; a pull reaching much further
#  back (2005, say) could legitimately return a smaller day and would need this lowered.
#  It exists for the case the relative rule below cannot see: a run of ONE date, where a
#  median over the run's own bodies is the body itself and the relative test is vacuous.
MIN_PAYLOAD_ROWS = 20000

#  RELATIVE RULE -- the one the MD asked for, and the one that carries the argument.  A
#  body below this fraction of the MEDIAN of the run's other accepted bodies is refused.
#  Any fraction in (0.257, 0.759) separates the observed populations (15,441/60,152 =
#  0.257; 45,662/60,152 = 0.759); 0.5 is the linear midpoint of that interval to two
#  decimals (0.508) and is a statement rather than a fit: less than half of a typical
#  body is not a body.  On the observed median it lands at 30,076 rows.
#
#  It is DEFERRED to the end of the run, not applied in-line, for a reason that is a
#  charter constraint rather than a nicety: the median is not known until every anchor
#  has been fetched, and re-entering the step-back loop afterwards would spend API calls
#  the CEO did not plan (see DESIGN GOAL above).  So a body that clears MIN_PAYLOAD_ROWS
#  but fails the median test is REFUSED AT WRITE TIME and reported -- it costs no call,
#  and the anchor is simply absent from the output rather than present-and-wrong.
SHORT_BODY_MEDIAN_FRACTION = 0.5


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


def is_weekend(d):
    """True for Saturday/Sunday.

    THE BULK ENDPOINT DOES NOT ANSWER A WEEKEND WITH AN EMPTY BODY -- that is the whole
    point of this predicate.  `2024-12-28` (a Saturday) returned 3,589 rows, 93.8% of them
    USD-suffixed crypto pairs and not one FX row, and `if rows:` accepted it as the
    2024-12-31 anchor.  So a weekend candidate is never requested at all: it saves a call
    AND removes the only body the size floor would have had to catch on a technicality.

    Guarding the REQUESTED date is sufficient here, and deliberately so.  `run_bulk` writes
    the date it asked for into the `date_actual` column, so the requested date IS the
    recorded one; there is no second date to check.
    """
    return d.weekday() >= 5


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


def median_of(counts):
    """Plain median of a list of payload row counts (None for an empty list).

    Local rather than `statistics.median` only so the tie convention is visible: an even
    count averages the two middles, which is what "the size of the run's other bodies"
    should mean.
    """
    xs = sorted(counts)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    return float(xs[mid]) if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def short_against_median(count, median, fraction=None):
    """Is `count` far enough below `median` to refuse the body?  See the constants.

    `fraction=None` reads SHORT_BODY_MEDIAN_FRACTION at CALL time rather than binding it as
    a default at import time.  That is not cosmetic: bound as a default, the constant cannot
    be overridden from outside the module, so the mutation test that reopens the defect
    (`test_MUTATION_with_the_median_fraction_at_zero_...`) silently tested nothing.  It
    caught exactly that.
    """
    if median is None or median <= 0:
        return False
    frac = SHORT_BODY_MEDIAN_FRACTION if fraction is None else fraction
    return count < frac * median


def body_is_acceptable(n_payload):
    """IN-LINE acceptance: is this payload big enough to be a trading day at all?

    The absolute half of the floor, factored out so the STANDALONE script and the PIPELINE
    stage (`pipeline_analysis._fetch_bulk_scrubbed`) apply one rule from one definition.
    They previously shared only the row PARSER, which is how the pipeline path came to keep
    `if rows:` while this file grew a floor -- a guard on a code path production does not
    take.  Reads the constant at call time so a test can move it.
    """
    return n_payload >= MIN_PAYLOAD_ROWS


def refusals_against_median(counts):
    """DEFERRED acceptance: indices of bodies that are far below the run's OTHER bodies.

    `counts` is the accepted payload sizes in fetch order.  Returns
    {index: median_of_the_others} for each body to refuse -- LEAVE-ONE-OUT, so a truncated
    body cannot drag the very median it is tested against (with two anchors and one of them
    truncated, a pooled median sits between them and neither reads as "far below").

    Factored out for the same reason as `body_is_acceptable`: two copies of a correctness
    guard is the drift this project keeps paying for.  A single-element `counts` yields {}
    by construction -- there is no "other body" -- which is exactly why the absolute floor
    above has to exist as well.
    """
    out = {}
    for i, c in enumerate(counts):
        med = median_of(counts[:i] + counts[i + 1:])
        if short_against_median(c, med):
            out[i] = med
    return out


def run_bulk(api_key, anchors, max_lookback, symbols_filter, out_path):
    """Fetch each anchor date via the bulk endpoint; step back past empty/short days.

    Returns (call_count, written, refused) where `refused` is a list of dicts describing
    every anchor whose body was rejected by the DEFERRED median test and is therefore
    ABSENT from the output file.

    WHY THE ROWS ARE BUFFERED AND WRITTEN AT THE END.  The relative floor needs the median
    of the run's other bodies, which does not exist until the last anchor has been fetched,
    so the decision to write an anchor cannot be made while that anchor is in hand.  Peak
    memory is the same order as before (the previous version already held one whole parsed
    body of ~77,000 dicts); what is added is the FILTERED rows of the earlier anchors.

    A REFUSED ANCHOR IS WRITTEN NOWHERE, ON PURPOSE.  A half-body in the file is the
    failure mode that has already cost this project real numbers: it satisfies every
    presence check downstream (`not os.path.exists(...)`, `if rows:`) while being blind to
    whole venues.  An absent anchor at least makes `price_grid_audit` say "anchor X prices
    NONE of tonight's names" out loud.
    """
    call_count = 0
    written = 0
    accepted = []      # [{anchor, date_str, n_payload, rows: [(sym, adj), ...]}, ...]

    for anchor in anchors:
        got = False
        for back in range(max_lookback + 1):
            d = anchor - timedelta(days=back)
            date_str = d.strftime("%Y-%m-%d")
            if is_weekend(d):
                #  NO CALL IS SPENT.  See is_weekend: the endpoint answers a weekend with a
                #  small non-empty body, so this is a correctness guard first and a
                #  call-count saving second.
                print(f"[skip] {date_str} is a {d.strftime('%A')} -- not requested "
                      f"(anchor {anchor})", flush=True)
                continue
            call_count += 1
            print(f"[call {call_count}] bulk date={date_str} "
                  f"(anchor {anchor}) ...", flush=True)
            try:
                rows = fetch_bulk_for_date(api_key, date_str)
            except requests.RequestException as e:
                print(f"    request error: {e}", flush=True)
                rows = []
            n_payload = len(rows)
            if not body_is_acceptable(n_payload):
                #  THE COUNT IS TAKEN ON THE PAYLOAD, BEFORE `symbols_filter`.  The filter
                #  is a local convenience that can legitimately cut a full body to a few
                #  thousand rows; the completeness question is about what the endpoint
                #  returned, not about what we chose to keep.
                if n_payload == 0:
                    print("    empty body -- stepping back", flush=True)
                else:
                    print(f"    REJECTED: {n_payload} rows is below the absolute floor "
                          f"{MIN_PAYLOAD_ROWS} -- not a trading day's worth of prices, "
                          f"stepping back", flush=True)
                time.sleep(0.3)  # be gentle
                continue
            kept = []
            for row in rows:
                sym, adj = _extract(row)
                if not sym or adj in (None, "", "null"):
                    continue
                if symbols_filter and sym not in symbols_filter:
                    continue
                kept.append((sym, adj))
            accepted.append({"anchor": anchor, "date_str": date_str,
                             "n_payload": n_payload, "rows": kept})
            got = True
            print(f"    OK: {n_payload} rows returned ({len(kept)} kept)", flush=True)
            break
        if not got:
            print(f"    WARNING: no usable body for anchor {anchor} within "
                  f"{max_lookback} lookback days", flush=True)

    #  ---- DEFERRED RELATIVE FLOOR.  Each body is judged against the median of the OTHER
    #  accepted bodies, LEAVE-ONE-OUT, so a truncated body cannot drag the very median it
    #  is being tested against: with 2 anchors and one of them truncated a pooled median
    #  sits between them and neither reads as "far below" it.
    refused = []
    for i, med in refusals_against_median([a["n_payload"] for a in accepted]).items():
        a = accepted[i]
        a["refused_against_median"] = med
        refused.append({"anchor": a["anchor"], "date_str": a["date_str"],
                        "n_payload": a["n_payload"], "median_of_others": med})

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["date_requested", "date_actual", "symbol", "adjClose"])
        for a in accepted:
            if "refused_against_median" in a:
                print(f"    REFUSED at write time: anchor {a['anchor']} body "
                      f"{a['date_str']} has {a['n_payload']} rows against a median of "
                      f"{a['refused_against_median']:.0f} for the run's other bodies "
                      f"(< {SHORT_BODY_MEDIAN_FRACTION:g}x) -- NOT written", flush=True)
                continue
            a_str = a["anchor"].strftime("%Y-%m-%d")
            for sym, adj in a["rows"]:
                w.writerow([a_str, a["date_str"], sym, adj])
                written += 1
    return call_count, written, refused


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

    calls, written, refused = run_bulk(api_key, anchors, args.max_lookback,
                                       symbols_filter, out_path)
    print("=" * 60, flush=True)
    print(f"DONE. TOTAL API CALLS MADE: {calls}", flush=True)
    print(f"Rows written: {written} -> {out_path}", flush=True)
    if refused:
        #  NON-ZERO EXIT so a caller cannot treat a partial grid as a completed fetch.  The
        #  file that IS written is still usable -- the refused anchors are simply not in it,
        #  and `price_grid_audit` will say so on the next analysis run.
        print("=" * 60, flush=True)
        print(f"REFUSED {len(refused)} anchor(s) as short bodies -- ABSENT from the "
              f"output:", flush=True)
        for r in refused:
            print(f"  anchor {r['anchor']}  body {r['date_str']}  "
                  f"{r['n_payload']} rows  vs median-of-others "
                  f"{r['median_of_others']:.0f}", flush=True)
        print("Re-fetch those anchors deliberately; do NOT paste a short body into the "
              "canonical grid by hand.", flush=True)
        print("=" * 60, flush=True)
        sys.exit(3)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
