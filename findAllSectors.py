import os
import glob
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime


def _mask_key(api_key):
    """Return a log-safe fingerprint of the API key -- NEVER the key itself. Used
    everywhere this module logs, so a masked key is the only form that can leak."""
    if not api_key:
        return '<none>'
    k = str(api_key).strip()
    return f'***{k[-4:]}' if len(k) >= 4 else '***'


# Sector-string normalization: single source of truth shared by the full-universe
# producer and the generate-if-missing self-heal builder.
_SECTOR_DICMAPS = {'Financial': 'Financial Services', 'Industrial Goods': 'Industrials',
                   'Biotechnology': 'Healthcare', 'Pharmaceuticals': 'Healthcare',
                   'Retail': 'Consumer Cyclical', 'Banking': 'Financial Services'}


def _normalize_sector_dic(sectordic):
    """Collapse raw FMP sector keys to the canonical set the carve-out expects
    (dicmaps rename + empty/'N/A'/None + <10-member cohorts -> 'Unspecified').
    Identical logic to the original findAllSectorsViaProfile.

    NOTE: the <10-member floor is calibrated for the FULL filtered universe
    (~10-12k names, where every real sector has hundreds). On a tiny ad-hoc batch
    a legitimate sector can fall under the floor and collapse to 'Unspecified' --
    expected, not a bug (the floor exists to drop junk micro-sectors)."""
    newsectordic = {}
    for key in sectordic.keys():
        if key in _SECTOR_DICMAPS:
            newkey = _SECTOR_DICMAPS[key]
        elif key in ['', 'N/A', None]:
            newkey = 'Unspecified'
        elif len(sectordic[key]) < 10:
            newkey = 'Unspecified'
        else:
            newkey = key

        if newkey not in newsectordic:
            newsectordic[newkey] = list(sectordic[key])
        else:
            newsectordic[newkey] = newsectordic[newkey] + sectordic[key]
    return newsectordic


def _fetch_profiles_batched(symbols, baseurl, api_key, batch_size=100, pace=None,
                            max_missing_frac=0.10):
    """Fetch v3/profile for `symbols` in comma-batched calls via the hardened
    getData_gen.safe_get path (bounded timeout + retry/backoff -- NOT bare
    requests.get).  Returns (profiles, n_calls, complete, missing).

    KEY-IN-LOGS (accurate note): the key is passed as a query PARAM so it is NOT in
    the `url` string safe_get logs on failure. HOWEVER, on a connection error or an
    HTTP-status error the underlying requests exception string embeds the RESOLVED
    URL (with `apikey=<KEY>`), and safe_get's warning interpolates that exception --
    so the key CAN appear in the live-console exception text. This is pre-existing
    behavior for ALL FMP calls in the repo; the on-disk log is masked by run_logger,
    so this is not a NEW persisted leak. This module never itself logs the raw key
    (see _mask_key). Do not claim the key "cannot leak" -- it can, to the console.

    `complete` is False when EITHER:
      (i)  any batch failed to return a JSON list (hard error -- None/exception), OR
      (ii) the AGGREGATE shortfall (requested symbols with no returned profile)
           exceeds `max_missing_frac`. This (ii) guard catches a 200 OK that silently
           returns FEWER profiles than requested (e.g. an empty-but-200 batch, or a
           silent drop): those symbols would otherwise get no sector and leak into the
           general pool past carveOut's empty-map abort -- the same failure the
           refuse-to-persist policy exists to stop, reached via a different mechanism.
    A small tail of symbols that legitimately have NO profile (delisted/obscure) stays
    under `max_missing_frac` and is tolerated (over-triggering on those would just
    force an avoidable degrade). Batch-CAP truncation is disproven for batches <=114
    (endpoint probe), so keep batch_size <=100.

    `missing` is the set of requested symbols with no returned profile (for logging)."""
    from getData_gen import safe_get  # lazy import: avoid any import cycle at load
    profiles, n_calls, complete = [], 0, True
    uniq = list(dict.fromkeys(s for s in symbols if isinstance(s, str) and s))
    requested = set(uniq)
    pbar = tqdm(total=len(uniq)) if uniq else None
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i:i + batch_size]
        csv = ','.join(batch)
        resp = safe_get(f'{baseurl}v3/profile/{csv}', params={'apikey': api_key})
        n_calls += 1
        if isinstance(resp, list):
            profiles.extend(resp)
        else:
            # safe_get already retried/backed off; a None here is a hard failure.
            complete = False
        if pbar is not None:
            pbar.update(n=len(batch))
        if pace:
            import time as _t
            _t.sleep(pace)
    if pbar is not None:
        pbar.close()
    # Shortfall guard (ii): a 200 OK can return a list yet silently omit symbols,
    # which the None-check above never catches. Compare returned vs requested.
    returned = {p.get('symbol') for p in profiles if isinstance(p, dict) and p.get('symbol')}
    missing = requested - returned
    if requested and (len(missing) / len(requested)) > max_missing_frac:
        complete = False
    return profiles, n_calls, complete, missing


def buildSectorIndustryMaps(symbols, baseurl, api_key, batch_size=100, pace=None):
    """Build BOTH carve-out maps for an EXPLICIT symbol list (the run's filtered
    universe -- NOT the full available-traded/list), fetching v3/profile in
    comma-batches via the hardened safe_get path. Writes into CWD:

      * sectorsdic_fmp.pickle          UNDATED -- the exact name the sector consumers
                                        read (getData_gen.filterBySector:216 and the
                                        carveOut._load_sector_map / partition_universe
                                        default). Writing the undated name fixes the
                                        producer/consumer filename mismatch.
      * industrydic_fmp_<date>.pickle  DATED   -- the glob carveOut._load_industry_map
                                        matches (newest-dated wins).

    Returns (sectordic, industrydic). Raises RuntimeError on an incomplete fetch so
    the caller degrades rather than persisting a PARTIAL map -- a partial SECTOR map
    is dangerous: it is non-empty, so it slips past carveOut's empty-map abort while
    still leaking miners/REITs into the general pool (the exact failure that abort
    guards). "Incomplete" includes both a hard fetch error AND a large silent shortfall
    of returned profiles (see _fetch_profiles_batched)."""
    profiles, n_calls, complete, missing = _fetch_profiles_batched(
        symbols, baseurl, api_key, batch_size=batch_size, pace=pace)
    if not complete or not profiles:
        raise RuntimeError(
            f'profile fetch incomplete/empty ({len(profiles)} profiles over '
            f'{n_calls} call(s); {len(missing)} requested symbol(s) had no returned '
            f'profile, key {_mask_key(api_key)}) -- refusing to persist a partial map')

    sectordic, industrydic = {}, {}
    for prof in profiles:
        sym = prof.get('symbol') if isinstance(prof, dict) else None
        if not sym:
            continue
        industrydic[sym] = prof.get('industry')
        sec = prof.get('sector')
        sectordic.setdefault(sec, []).append(sym)

    newsectordic = _normalize_sector_dic(sectordic)
    fidag = datetime.today().strftime('%Y-%m-%d')
    pd.to_pickle(newsectordic, 'sectorsdic_fmp.pickle')
    pd.to_pickle(industrydic, f'industrydic_fmp_{fidag}.pickle')
    print(f'[sector/industry build] {len(industrydic)} symbols -> '
          f"sectorsdic_fmp.pickle + industrydic_fmp_{fidag}.pickle "
          f'({n_calls} batched profile call(s); {len(missing)} requested symbol(s) '
          f'had no profile -- tolerated, under threshold; key {_mask_key(api_key)})')
    return newsectordic, industrydic


def ensure_sector_industry_maps(symbols, baseurl, api_key, batch_size=100, pace=None):
    """GENERATE-IF-MISSING hook for the ingestion layer.

    If BOTH the sector map (undated sectorsdic_fmp.pickle) and an industry map
    (industrydic_fmp_*.pickle) are already present in CWD -> DO NOTHING: no rebuild,
    no API calls, the newest cached pickles are reused. Otherwise build BOTH once
    from `symbols` (a single profile fetch yields both, so they are always built as
    a consistent pair).

    Best-effort by design: carveOut is credential-less and cannot self-heal, so this
    lives in ingestion. On ANY fetch failure it logs a KEY-MASKED warning and returns
    WITHOUT writing -- the run then falls through to carveOut's EXISTING behavior
    (industry-missing -> loud degrade to the keyword rule; sector-missing -> the
    existing loud abort). It never introduces a NEW hard dependency or abort.

    Returns True iff a build ran and wrote the maps; False otherwise."""
    sector_present = os.path.exists('sectorsdic_fmp.pickle')
    industry_present = bool(glob.glob('industrydic_fmp_*.pickle'))
    if sector_present and industry_present:
        return False  # idempotent skip -- reuse cached pickles, no rebuild

    try:
        buildSectorIndustryMaps(symbols, baseurl, api_key,
                                batch_size=batch_size, pace=pace)
        return True
    except Exception as e:
        print(f'WARNING: sector/industry map self-heal FAILED -- falling through to '
              f"carveOut's existing degrade path (key {_mask_key(api_key)}): {e}")
        return False


def findAllSectorsViaProfile(baseurl, api_key):
    """Full-universe producer: enumerate available-traded/list, then delegate the
    batched profile fetch + assemble + persist to buildSectorIndustryMaps so the
    maps are built ONCE, consistently, via the same self-heal code path. Kept for
    the full-universe use case; the wired self-heal uses ensure_sector_industry_maps
    on the FILTERED universe instead."""
    resp_assets = requests.get(f'{baseurl}v3/available-traded/list?apikey={api_key}')
    ass_df = pd.DataFrame(resp_assets.json())
    symbols = list(ass_df['symbol'].unique())
    newsectordic, _ = buildSectorIndustryMaps(symbols, baseurl, api_key)
    return newsectordic

## USELESS I THINK
def findAllSectorsViaScreener(baseurl,api_key):

    stocks_seen = []
    sectordic = {}
    exchanges = ['nyse', 'nasdaq', 'amex', 'euronext', 'tsx']
    betaLowerThan = [0.5    ,2.1,1000]
    betaMoreThan = [-1000   ,1.1,2.1]
    priceLowerThan = [1,  7.5,  12,  70, 150,   1000000]
    priceMoreThan = [0,   1,    7.5 , 12,   70,   150]
    isEtf = False
    isActivelyTrading = True
    baseMegaCap = 100000000000
    marketCapMoreThan = [0                  ,baseMegaCap/1000,baseMegaCap/100,baseMegaCap/10,baseMegaCap,
                         baseMegaCap*10]
    marketCapLowerThan = [baseMegaCap/1000 ,baseMegaCap/100 ,baseMegaCap/10 ,baseMegaCap  ,baseMegaCap*10,
                         100*baseMegaCap]

    T = len(exchanges)*len(betaLowerThan)**2*len(priceMoreThan)**2*len(marketCapLowerThan)**2
    tqdm(total=len(ass_df['symbol'].unique()))
    for ex in exchanges:
        for i in range(0,len(betaLowerThan)-1):
            blt = betaLowerThan[i]
            bmt = betaMoreThan[i]
            for j in range(0,len(marketCapLowerThan)-1):
                mcmt = marketCapMoreThan[j]
                mclt = marketCapLowerThan[j]
                for k in range(0,len(priceMoreThan)-1):
                    plt = priceLowerThan[k]
                    pmt = priceMoreThan[k]

                    reqstr = f"{baseurl}v3/stock-screener?marketCapLowerThan={mclt}&marketCapMoreThan={mcmt}&" \
                              f"betaLowerThan={blt}&betaMoreThan={bmt}&" \
                              f"priceLowerThan={plt}&priceMoreThan={pmt}&exchange={ex}&" \
                              f"isEtf=true&isActivelyTrading=true&apikey={api_key}"
                    resp_screen = requests.get(reqstr).json()
                    if len(resp_screen) > 0:
                        sdf = pd.DataFrame(resp_screen)
                        for symbol in sdf['symbol']:
                            stocks_seen.append(symbol)
                            symb_sector = sdf[sdf['symbol'] == symbol].sector
                            if symb_sector not in sectordic:
                                sectordic[symb_sector] = []
                                sectordic[symb_sector].append(symbol)

