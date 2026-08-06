from datetime import datetime
import sys
import pandas as pd
import requests
import reporting_period as rp
from tqdm import tqdm


def _bar_print(msg):
    """`print()` replacement for the skip-logs below, which fire INSIDE the fetch loop.

    Every call site of this module's gate runs under the tqdm bar in
    `getData_fmp.get_fundamentals_fmp`, and a bare `print()` emits its newline while that
    bar is mid-render with '\\r' -- stranding a fragment of the bar on screen and restarting
    it one line down.  `tqdm.write` clears the live bar, writes, and redraws; with no bar
    alive (the offline/dead-path callers, the tests) it degrades to a plain write.

    PRESENTATION ONLY -- identical text, stream (stdout) and flush behaviour to the
    `print(..., flush=True)` calls it replaces.  Deliberately a local twin of
    `getData_fmp._bar_print` rather than a shared import: this module is imported BY
    getData_fmp, so pulling the helper across would add a cycle to the import graph for two
    lines of code.
    """
    tqdm.write(msg, file=sys.stdout)
    sys.stdout.flush()

def testForAPIFaults_fmp(failcodes,compyear,ticker,period,limit,baseurl,api_key,
                         dead_path=False, http_get=None):
    """Fetch the 5 statement endpoints and evaluate the accept/reject gates.

    LIVE PATH (dead_path=False, the default): behaviour is UNCHANGED and
    bit-for-bit -- the datefail gate (compyear > newest-income-year) and the
    >=16q lenfail gate both apply exactly as before.  Live callers never pass the
    new kwargs, so their calls are identical to the pre-change code.

    ============================ F-A / F-B (DEAD PATH) ============================
    When dead_path=True (the DELISTED-ENTITY INGESTION path), TWO gates are
    DELIBERATELY DISABLED, because they otherwise SILENTLY DESTROY the exact
    population this ingestion exists to capture (delisted-ingestion-spec s0):

      * F-A -- the DATEFAIL gate is BYPASSED.  compyear defaults to (this year - 1)
        = 2025 (the `compyear` default in
        configuration.getDataFetchConfiguration).  A company delisted in 2020 has its newest
        income statement ~2019-2020, so `compyear(2025) > 2020` -> datefail ->
        dropped.  On the live gate ~100% of dead names are eliminated even when
        their fundamentals fetch perfectly.  THE DEAD PATH MUST NOT APPLY THIS
        GATE.  Without this bypass the whole delisted run is worthless.

      * F-B -- the LENFAIL (>=16 quarters) gate is RELAXED.  Rejecting <16q of
        history is survivorship-WITHIN-death (biases the dead set toward long-lived
        names).  On the dead path short history is ACCEPTED here; the caller tags
        `short_history=True` and lets downstream PIT/scoring decide.

    Only `failcode` (HTTP 4xx/5xx) and `emptyfail` (all endpoints return []) still
    fail on the dead path -- an emptyfail dead name is skipped + logged as a
    first-class completeness artifact by the caller (it has no retrievable
    fundamentals), NOT silently discarded by a date/length heuristic.
    ==============================================================================

    http_get : optional callable(url) -> requests.Response, injected for OFFLINE
        testing (fake responses).  Defaults to requests.get so the live path is
        unchanged.
    """
    if http_get is None:
        http_get = requests.get
    failbool = False
    whyfail = 'None'
    calldic = {'km': 'key-metrics', 'fr': 'ratios','inc': 'income-statement', 'bs': 'balance-sheet-statement',
               'cf': 'cash-flow-statement'}
    resplist = []
    respfail = False
    fsdfdic = {}
    for key in calldic.keys():
        resp = http_get(f'{baseurl}v3/{calldic[key]}/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        if resp.status_code in failcodes:
            respfail = True
            failbool = True
            whyfail = 'failcode'
            break
        else:
            # THE LAST UNGUARDED JSON PARSE IN THE FETCH PATH (review B2, fixed 2026-07-31).
            #
            # A THROTTLED 200 CARRYING AN HTML BODY makes `resp.json()` raise
            # JSONDecodeError.  `safe_http_get` correctly hands back the raw Response for a
            # 200, and 200 is not in `failcodes` (400-599), so nothing above catches it --
            # and `getFsData_fmp` has no `try`, while the call site
            # (getData_fmp.get_fundamentals_fmp) sits OUTSIDE the per-ticker guard.  It
            # therefore propagated to Sbocker, which re-raises.
            #
            # WHY THIS ONE MATTERS MORE THAN THE ~700 POST-FETCH CALLS ALREADY HARDENED:
            # this path is ~7,700 tickers x 5 statements = ~38,500 calls (~55x larger), and
            # THERE IS NO RESUME -- utils.write_lastIndexRead runs only AFTER
            # get_fundamentals_fmp RETURNS, so a crash at hour 11 of a 12-hour fetch leaves
            # NOTHING on disk.  Hardening the cheap path and leaving the expensive one bare
            # was the indefensible asymmetry.
            #
            # The handler mirrors the audit H-4 fix two branches down VERBATIM in intent: a
            # malformed 200 body is `emptyfail` -- skipped and logged as a first-class
            # completeness artifact -- because "one unlucky throttled ticker must cost that
            # ticker, not the run".  Not `failcode`: that bucket means a definitive HTTP
            # status, and conflating an unparseable body with one would misreport the cause.
            try:
                _payload = resp.json()
            except Exception as _je:
                respfail = True
                failbool = True
                whyfail = 'emptyfail'
                _body = str(getattr(resp, 'text', ''))[:200].replace('\n', ' ')
                _bar_print('EMPTYFAIL %s (%s): HTTP %s but the body is not JSON (%s) -- ticker '
                           'SKIPPED, run continues. Body head: %r'
                           % (ticker, calldic[key], getattr(resp, 'status_code', '?'),
                              type(_je).__name__, _body))
                break
            resplist.append(_payload)
            fsdfdic[key] = pd.DataFrame(_payload)

    if respfail == False:
        if any([not lst for lst in resplist]):
            failbool = True
            whyfail = 'emptyfail'
        elif dead_path:
            # ===== F-A + F-B: DEAD PATH accepts everything past emptyfail. =====
            # The datefail gate (F-A) and the >=16q lenfail gate (F-B) are NOT
            # applied here -- applying them would silently drop essentially every
            # delisted name (F-A) and bias the survivors long (F-B).  A malformed
            # income statement with no 'date' column is treated as emptyfail (skip
            # + log) rather than raising, so it cannot abort a multi-hour dead loop.
            tempdf = pd.DataFrame(resplist[2])
            if 'date' not in tempdf.columns:
                failbool = True
                whyfail = 'emptyfail'
        else:
            tempdf = pd.DataFrame(resplist[2])
            if 'date' in tempdf.columns:
                # ROW ORDER IS ASSUMED HERE, NOT ESTABLISHED (flagged 2026-08-02, deliberately
                # NOT changed).  `.iloc[0]` is "the newest income statement" ONLY because FMP
                # returns statements newest-first.  On an oldest-first body this gate would
                # compare compyear against the OLDEST year and reject essentially every
                # ticker.  Sorting instead of assuming would change which tickers pass, i.e.
                # the universe -- a behaviour change, so it stays a stated precondition.
                #
                # THE PARSE IS GUARDED (RUN-KILLER FIXED 2026-08-02, CEO-authorised).
                # `datetime.strptime(strdate, '%Y-%m-%d')` raises ValueError on any other date
                # shape and TypeError on None/NaN, and this function sits OUTSIDE the
                # per-ticker guard: getFsData_fmp has no `try`, and its caller invokes it
                # BEFORE entering the per-ticker try/except (getData_fmp.get_fundamentals_fmp).
                # So one malformed date string killed the whole ~38,500-call fetch, WITH NO
                # RESUME -- utils.write_lastIndexRead runs only after get_fundamentals_fmp
                # RETURNS, so a crash at hour 11 of 12 left nothing on disk.
                #
                # HANDLED EXACTLY AS AUDIT H-4 HANDLES ITS SIBLING CASE (the `else` branch
                # below) rather than in a second style: an unusable income-statement date is
                # `emptyfail` -- the ticker is skipped, logged, and lands in the run's
                # completeness counters -- because "one unlucky ticker must cost that ticker,
                # not the run".  NOT `datefail`: that bucket means "we read the date and it was
                # too old", a verdict we cannot reach when the date will not parse.
                # It has not fired yet because FMP's `date` is consistently ISO; that is a
                # property of the vendor, not a guarantee, and the cost of being wrong is 12
                # hours.
                strdate = tempdf['date'].iloc[0]
                try:
                    _newest_year = datetime.strptime(strdate, '%Y-%m-%d').year
                except (TypeError, ValueError) as _de:
                    _newest_year = None
                    failbool = True
                    whyfail = 'emptyfail'
                    _bar_print('EMPTYFAIL %s: income-statement date %r is not YYYY-MM-DD (%s) -- '
                               'ticker SKIPPED, run continues.'
                               % (ticker, strdate, type(_de).__name__))
                if _newest_year is None:
                    pass                        # already recorded as emptyfail above
                elif compyear > _newest_year:
                    failbool = True
                    whyfail = 'datefail'
                else:
                    lentest = [len(resp) for resp in resplist]
                    if period == 'quarter':
                        # HISTORY GATE, in CALENDAR terms (2026-07-25).  16 'quarters'
                        # is 4 years of history -- but a SEMI-ANNUAL filer only issues
                        # 2 rows a year, so demanding 16 rows demanded EIGHT years and
                        # rejected perfectly well-covered LSE names for a reporting
                        # convention.  The bar is rows-per-year x FETCH_HISTORY_YEARS =
                        # 16 quarterly or 8 semi-annual, i.e. the same 4 years either way.
                        # The span is a NAMED constant in reporting_period rather than a
                        # bare `4` here, because the frequency classifier's own recency
                        # window (CLASSIFY_RECENT_DAYS) is deliberately the SAME span and
                        # the two must not drift apart -- the classifier should look at
                        # exactly the history this gate demands.
                        _freq = rp.classify_source(
                            dates=(tempdf['date'] if 'date' in tempdf.columns else None),
                            period_values=(list(tempdf['period'])
                                           if 'period' in tempdf.columns else None))
                        _minrows = rp.rows_per_year(_freq) * rp.FETCH_HISTORY_YEARS
                        if any(j < _minrows for j in lentest):
                            failbool = True
                            whyfail = 'lenfail'
                    elif period == 'annual':
                        if any(j < 4 for j in lentest):
                            failbool = True
                            whyfail = 'lenfail'
                    else:
                        raise Exception('Why is period not either quarter or annual?')
            else:
                # A 200 response whose income statement carries no `date` column is a
                # MALFORMED/ERROR BODY, not a programming error (audit H-4 fix,
                # 2026-07-19).  FMP returns error payloads -- e.g.
                # [{"Error Message": "Limit Reach ..."}] -- with HTTP 200 in throttle
                # states, so `resp.status_code in failcodes` misses them and the old
                # `raise` propagated out of the per-ticker loop and KILLED THE WHOLE ~12h
                # fundamentals run partway through.  One unlucky throttled ticker must
                # cost that ticker, not the run.  Treated as emptyfail -- the SAME
                # graceful skip the dead path already used for this exact case (see F-A /
                # F-B above) -- so the ticker lands in tickersfailed/emptyfail and is
                # visible in the run's completeness counters instead of vanishing.
                failbool = True
                whyfail = 'emptyfail'
                _bar_print('failTests: %s returned HTTP 200 with no `date` column in the '
                           'income statement (malformed/error body) -- skipped as emptyfail, '
                           'run continues. Body head: %.200r'
                           % (ticker, resplist[2]))

    outdic = {}
    if failbool == False:
        outdic = fsdfdic

    return failbool, whyfail, outdic
