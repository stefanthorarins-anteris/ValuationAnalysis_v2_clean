from datetime import datetime
import pandas as pd
import requests
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
        = 2025 (configuration.py:103).  A company delisted in 2020 has its newest
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
            resplist.append(resp.json())
            fsdfdic[key] = pd.DataFrame(resp.json())

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
                strdate = tempdf['date'].iloc[0]
                if compyear > datetime.strptime(strdate, '%Y-%m-%d').year:
                    failbool = True
                    whyfail = 'datefail'
                else:
                    lentest = [len(resp) for resp in resplist]
                    if period == 'quarter':
                        if any(j < 16 for j in lentest):
                            failbool = True
                            whyfail = 'lenfail'
                    elif period == 'annual':
                        if any(j < 4 for j in lentest):
                            failbool = True
                            whyfail = 'lenfail'
                    else:
                        raise Exception('Why is period not either quarter or annual?')
            else:
                raise Exception(f'No column, in dataframe from API, with the name: date')

    outdic = {}
    if failbool == False:
        outdic = fsdfdic

    return failbool, whyfail, outdic
