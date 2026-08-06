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
    # Total is exact: the loop walks `uniq` in batches and advances by len(batch), which sums
    # to len(uniq).  Nothing inside prints, so no bar-safe writer is needed here; `desc`/`unit`
    # only, so the bar says which stage it belongs to.
    pbar = tqdm(total=len(uniq), desc='Sector/industry profiles', unit='symbol',
                dynamic_ncols=True) if uniq else None
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

    sectordic, industrydic, isindic, volavgdic = {}, {}, {}, {}
    for prof in profiles:
        sym = prof.get('symbol') if isinstance(prof, dict) else None
        if not sym:
            continue
        industrydic[sym] = prof.get('industry')
        #  ISIN: CAPTURE ONLY, NOT WIRED (register K-1, CEO 2026-08-05).
        #  It was ALREADY IN this response and discarded here, so capturing it costs ZERO extra
        #  API calls -- `getData_gen`'s K-1 note says exactly that ("the pipeline ALREADY FETCHES
        #  IT: v3/profile returns `isin`") and it was still being dropped on this line.
        #  WHY IT IS WANTED.  ISIN is the ONLY discriminator for the three dedup groups that
        #  merge correctly but pick the WRONG LINE (Robertet's certificat, Value8's preference
        #  line, Samsung's preferred GDR): same exchange, names identical verbatim, derived price
        #  identical -- no other marker can reach them.  381 of 1,282 groups (29.7%) currently
        #  fall through to an ALPHABETICAL tiebreak.
        #  DO NOT WIRE IT INTO THE DEDUP PICK RULE YET.  It is absent from every existing
        #  artifact, so nothing depending on it can be tested until after a fetch -- and
        #  `getData_gen.isin_same_issuer_groups`'s own note records the reason the naive rule is
        #  unsafe (Heineken N.V. vs Heineken Holding N.V. is a REAL false positive).
        isindic[sym] = prof.get('isin')
        #  volAvg: CAPTURE ONLY, NOT WIRED (MD, 2026-08-05).  Same deal as the ISIN above -- it is
        #  on the SAME already-fetched v3/profile payload, so capturing it costs ZERO extra API
        #  calls, and a fetch is the only chance we ever get to gain a column.
        #  WHY IT IS WANTED, TWICE OVER:
        #    * REGISTER K-1.  An ISIN carries no security-type field, so `_isin_plurality_term`
        #      ABSTAINS on 2-member groups and the three known-wrong picks (Robertet's certificat,
        #      Value8's preference line, Samsung's preferred GDR) stay unresolved. Average volume
        #      is what would actually reach them: the common line is the LIQUID one, and unlike an
        #      ISIN this field is DIRECTIONAL BY CONSTRUCTION (more is the common) rather than an
        #      opaque identifier we would have to invent a convention for.
        #    * REGISTER J-1.  Nothing anywhere in the pipeline screens for liquidity, free float,
        #      volume or spread, while the market-cap bands now go below $50M. This is the first
        #      instrument that could.
        #  WIRED 2026-08-06 into carveOut._investability_key (term 6, below ISIN and above the
        #  alphabetical last resort) -- and the PER-ENTRY AS-OF DATE below is what made wiring it
        #  admissible.  Each entry is {'volAvg': value, 'asof': 'YYYY-MM-DD'} rather than a bare
        #  value, because this map merges NEVER-OVERWRITE and average volume is TIME-VARYING: a
        #  symbol this run did not fetch keeps its previous reading, and without a date a consumer
        #  comparing two lines of one issuer cannot tell a liquidity difference from a
        #  six-month-old reading versus a fresh one.  `carveOut._volavg_liquidity_term` ABSTAINS on
        #  a group whose members' dates disagree, which is only possible because of this field.
        #  The FILENAME still dates the run; it cannot date the ENTRIES, which is the distinction
        #  that mattered.  `carveOut._load_volavg_map` reads the old bare-value shape too.
        volavgdic[sym] = prof.get('volAvg')
        sec = prof.get('sector')
        sectordic.setdefault(sec, []).append(sym)

    newsectordic = _normalize_sector_dic(sectordic)

    # MERGE, NEVER OVERWRITE (2026-08-02).  These two pickles are SHARED,
    # UNIVERSE-INDEPENDENT artifacts: `sectorsdic_fmp.pickle` is a symbol->sector map
    # for the whole world, and the next run to read it may be scoring a much larger
    # pool than the one that built it.  A bare `pd.to_pickle` made the artifact's
    # CONTENT a function of whichever universe happened to run last, so a small run
    # could SHRINK the map and every later run would carve against the remnant --
    # non-empty, so it sails past carveOut's empty-map abort, "while the output still
    # LOOKS carved" (carveOut's own words about exactly this failure).  Unioning makes
    # the map monotonically non-shrinking, which is the property the consumers assume.
    # Newly-fetched entries WIN on conflict: they are fresher, and a company's sector
    # really does get reclassified.
    prev_sector = _read_pickle_or_none('sectorsdic_fmp.pickle')
    prev_industry = _newest_industry_pickle()
    merged_sector, n_kept_s = _merge_sector_dics(prev_sector, newsectordic)
    merged_industry, n_kept_i = _merge_industry_dics(
        _read_pickle_or_none(prev_industry) if prev_industry else None, industrydic)

    fidag = datetime.today().strftime('%Y-%m-%d')
    pd.to_pickle(merged_sector, 'sectorsdic_fmp.pickle')
    pd.to_pickle(merged_industry, f'industrydic_fmp_{fidag}.pickle')
    #  ISIN map: written as its OWN dated artifact, merged with the same
    #  MERGE-NEVER-OVERWRITE discipline as the industry map (it is the identical shape,
    #  symbol -> value, so `_merge_industry_dics` is the right helper and not a coincidence).
    #  A SEPARATE FILE rather than a column on an existing one, deliberately: nothing reads it
    #  yet, and a new file cannot change the schema of an artifact something already consumes.
    _prev_isin = _newest_dated_pickle('isindic_fmp_*.pickle')
    merged_isin, n_kept_x = _merge_industry_dics(
        _read_pickle_or_none(_prev_isin) if _prev_isin else None, isindic)
    pd.to_pickle(merged_isin, f'isindic_fmp_{fidag}.pickle')
    #  volAvg map: its OWN dated artifact, same merge discipline, but PER-ENTRY DATED because
    #  unlike a sector, an industry or an ISIN, average volume is TIME-VARYING.  Merging
    #  never-overwrite still carries FORWARD a reading for any symbol this run did not fetch --
    #  that is right (non-shrinking, same reason as the maps above) -- but now each entry says
    #  WHEN it was read, so `carveOut._volavg_liquidity_term` can refuse to compare a fresh
    #  reading against a stale one instead of doing it silently.  The FILENAME dates the RUN and
    #  never could date the ENTRIES; that was the gap.
    _prev_volavg = _newest_dated_pickle('volavgdic_fmp_*.pickle')
    volavg_dated = {s: {'volAvg': v, 'asof': fidag} for s, v in volavgdic.items()}
    merged_volavg, n_kept_v = _merge_industry_dics(
        _read_pickle_or_none(_prev_volavg) if _prev_volavg else None, volavg_dated)
    pd.to_pickle(merged_volavg, f'volavgdic_fmp_{fidag}.pickle')
    print(f'[sector/industry build] volAvg captured for '
          f'{sum(1 for v in volavgdic.values() if v is not None)} of {len(volavgdic)} symbols -> '
          f'volavgdic_fmp_{fidag}.pickle, each entry stamped asof={fidag} (kept {n_kept_v} '
          f'pre-existing entr(ies), which carry THEIR OWN older asof -- a group with mixed '
          f'dates is skipped by the dedup liquidity term rather than compared). WIRED into '
          f'carveOut._investability_key (register K-1); register J-1 (a liquidity SCREEN) is '
          f'still NOT wired.')
    print(f'[sector/industry build] ISIN captured for {sum(1 for v in isindic.values() if v)} '
          f'of {len(isindic)} symbols -> isindic_fmp_{fidag}.pickle (kept {n_kept_x} '
          f'pre-existing entr(ies)). CAPTURE ONLY -- register K-1 is NOT wired.')
    print(f'[sector/industry build] {len(industrydic)} symbols fetched -> '
          f"sectorsdic_fmp.pickle + industrydic_fmp_{fidag}.pickle "
          f'({n_calls} batched profile call(s); {len(missing)} requested symbol(s) '
          f'had no profile -- tolerated, under threshold; key {_mask_key(api_key)})')
    print(f'[sector/industry build] MERGED with the existing maps: kept {n_kept_s} '
          f'pre-existing sector entr(ies) and {n_kept_i} industry entr(ies) that this '
          f'batch did not cover -> {_sector_symbol_count(merged_sector)} sector / '
          f'{len(merged_industry)} industry symbols on disk (never shrinks).')
    return merged_sector, merged_industry


def _read_pickle_or_none(path):
    try:
        return pd.read_pickle(path) if path and os.path.exists(path) else None
    except Exception:
        return None


def _newest_industry_pickle():
    return _newest_dated_pickle('industrydic_fmp_*.pickle')


def _newest_dated_pickle(pattern):
    """Newest (lexicographically last, i.e. newest ISO date) match, or None.

    Extracted from `_newest_industry_pickle` when the ISIN map was added -- the date is ISO so
    a lexicographic sort IS a chronological one, which is the property both callers rely on."""
    cands = sorted(glob.glob(pattern))
    return cands[-1] if cands else None


def _sector_symbol_count(sectordic):
    return len({s for syms in (sectordic or {}).values() for s in syms})


def _merge_sector_dics(previous, new):
    """Union two sector->[symbols] maps; `new` wins where a symbol appears in both.

    Returns (merged, n_symbols_kept_from_previous).  Kept separate from
    `_normalize_sector_dic` because the on-disk map is ALREADY normalized -- re-running
    the <10-member floor over the union could collapse a real sector that only looks
    thin in this batch."""
    if not previous:
        return new, 0
    new_syms = {s for syms in new.values() for s in syms}
    merged = {}
    kept = set()
    for sector, syms in previous.items():
        keep = [s for s in syms if s not in new_syms]
        kept.update(keep)
        if keep:
            merged[sector] = list(keep)
    for sector, syms in new.items():
        merged.setdefault(sector, [])
        merged[sector] = list(dict.fromkeys(merged[sector] + list(syms)))
    return merged, len(kept)


def _merge_industry_dics(previous, new):
    """Union two symbol->industry maps; `new` wins.  Returns (merged, n_kept)."""
    if not previous:
        return new, 0
    merged = dict(previous)
    kept = len([s for s in previous if s not in new])
    merged.update(new)
    return merged, kept


def ensure_sector_industry_maps(symbols, baseurl, api_key, batch_size=100, pace=None,
                                universe_is_subset=False, universe_name=None):
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

    `universe_is_subset` -- SET IT when `symbols` is a deliberately small, curated
    subset rather than a whole exchange-defined universe (i.e. the `stock_TEST1`
    explicit-membership universe).  Such a run MUST NOT author these shared maps:

      * the maps are SHARED and UNIVERSE-INDEPENDENT, and a 142-symbol map applied to a
        later 10,693-name pool covers 1.3% of it -- non-empty, so it slips straight past
        carveOut's empty-map abort while REIT and Mining leak wholesale;
      * `_normalize_sector_dic`'s <10-member floor is calibrated for the FULL universe.
        MEASURED on the 142-name list: 8 of 12 sector keys collapse and 34.4% of the
        batch lands in 'Unspecified'. NOTE THE LIMIT OF THAT CLAIM -- `Real Estate` (13)
        and `Basic Materials` (16), the only two sectors the carve actually CONSUMES,
        SURVIVE the floor. So the result would be a THIN carve, not NO carve; an earlier
        draft of this comment said 'no carve at all', which overstated it.

    So on a subset universe this SKIPS the build and says so loudly.  It does not
    fabricate a map, and (because `buildSectorIndustryMaps` now merges rather than
    overwrites) it cannot shrink one either.  If no map exists at all, carveOut's
    pre-existing empty-map abort fires with its own banner -- the correct outcome: the
    operator builds the maps once from a full universe (any exchange-defined
    `-tickerfilter`, or `findAllSectorsViaProfile`) and every later run reuses them.

    Returns True iff a build ran and wrote the maps; False otherwise."""
    sector_present = os.path.exists('sectorsdic_fmp.pickle')
    industry_present = bool(glob.glob('industrydic_fmp_*.pickle'))
    if sector_present and industry_present:
        return False  # idempotent skip -- reuse cached pickles, no rebuild

    if universe_is_subset:
        bar = '!' * 78
        print('\n' + bar)
        print('!!! SECTOR/INDUSTRY MAP BUILD SKIPPED -- the active universe is a SUBSET')
        print('!!!   universe : %s  (%d symbols)' % (universe_name or '<unnamed>',
                                                     len(list(symbols))))
        print('!!!   missing  : %s%s'
              % ('sectorsdic_fmp.pickle ' if not sector_present else '',
                 'industrydic_fmp_*.pickle' if not industry_present else ''))
        print('!!! These maps are SHARED and universe-independent. Building them from a')
        print('!!! curated subset would leave a map covering ~1% of a later full run --')
        print('!!! non-empty, so it would slip past carveOut\'s empty-map abort while')
        print('!!! REIT/Mining leaked into the general pool. Refusing to author them.')
        print('!!! FIX: run once with a full exchange-defined universe (e.g.')
        print('!!!      -tickerfilter stock_NA1_EU1), or call')
        print('!!!      findAllSectors.findAllSectorsViaProfile(baseurl, api_key).')
        print('!!! This run will hit carveOut\'s existing loud abort/degrade instead.')
        print(bar + '\n', flush=True)
        return False

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

