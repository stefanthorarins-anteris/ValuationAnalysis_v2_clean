import hashlib
import json
import os
import glob
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime


# =========================================================================== #
#  WHAT THE PROFILE MAPS CAPTURE -- DECLARED ONCE, AND THE GATE READS IT       #
#  (2026-08-14)                                                               #
# =========================================================================== #
#  THE DEFECT THIS CLOSES, and it has already cost one run.  The rebuild gate
#  (`ensure_sector_industry_maps`) decides whether to re-fetch the profile payload from
#  PRESENCE, AGE and COVERAGE.  None of the three can see a CODE change to WHICH FIELDS the
#  maps capture -- so on 2026-08-10 all four maps existed, none was 60 days stale and coverage
#  was above the floor, the gate skipped (correctly, by its own rules), and TWO ALREADY-SHIPPED
#  CAPTURE CHANGES NEVER LANDED.  Every 2026-08-10 pick carried `volAvg_asof = 2026-08-07`, and
#  nothing would have forced a rebuild until 2026-10-06.  `-force_rebuild_maps` was added as
#  the manual answer, and it works -- but it is an OPERATOR REMEMBERING, which is exactly the
#  mechanism that failed.
#
#  THE FIX IS THE SAME LESSON THE PRESENCE CHECK ALREADY LEARNED, ONE LEVEL UP.  That check was
#  wrong because it was frozen at a past revision instead of derived from what the writer
#  WRITES; the gate's own note says "A presence gate must be derived from what the writer
#  WRITES, never from a subset of it".  A FRESHNESS gate must be too.  So the captured FIELD SET
#  is declared here, the writer builds its entries FROM this tuple (it cannot capture a field
#  that is not in it, and cannot omit one that is), the fingerprint is written beside the maps,
#  and the gate rebuilds when the fingerprint on disk differs from the code's.
#
#  A CAPTURE CHANGE NOW TRIGGERS ITS OWN REBUILD, with no flag and nobody remembering.
#  `-force_rebuild_maps` is KEPT and is unchanged -- the two are independent, and it remains
#  the answer for "rebuild for some reason the code cannot see".
#
#  THE FIELDS BELOW ARE THE ONES WITH NAMED CONSUMERS OR PENDING QUESTIONS.  Every one is
#  CAPTURE-ONLY: nothing in the pipeline reads them.  See the long note at the capture site.
PROFILE_EXTRA_CAPTURE_FIELDS = (
    'mktCap',            # the TRADED LINE's own market cap -- the depositary handle
    'ipoDate',           # activates universe_pit.py:97-100, which has never fired live
    'companyName',       # today fetched from Yahoo, one name at a time, at 0.6s spacing
    'isAdr', 'isEtf', 'isFund',
    'cik', 'cusip',      # issuer / instrument identifiers, for the K-1 dedup tiebreak
    'fullTimeEmployees',
)

#  EVERY profile key the writer pulls off the payload, in ONE place.  The earlier waves are
#  listed explicitly because they are read into individually-named dicts (kept that way: they
#  have consumers and comments attached, and renaming them would be a refactor riding a capture
#  change).  The fingerprint covers ALL of them, so removing an old field triggers a rebuild
#  just as adding a new one does.
PROFILE_CAPTURE_FIELDS = tuple(sorted({
    'symbol', 'sector', 'industry', 'isin', 'volAvg',
    'price', 'currency',
    'isActivelyTrading', 'exchange', 'exchangeShortName', 'country', 'beta',
} | set(PROFILE_EXTRA_CAPTURE_FIELDS)))

#  The stamp file.  A SEPARATE artifact rather than a key inside `volavgdic_fmp_*.pickle`,
#  deliberately and for the reason the ISIN map already gives: that pickle is a
#  symbol -> entry map and a non-symbol key in it would be read as a symbol by any consumer
#  that iterates it.  A new file cannot change the schema of an artifact something already
#  consumes.
PROFILE_CAPTURE_SCHEMA_FILE = 'profile_capture_schema.json'


def profile_capture_fingerprint(fields=None):
    """A short, stable fingerprint of the captured profile FIELD SET.

    Order-independent (the tuple is sorted) so re-ordering the declaration cannot trigger a
    spurious rebuild, and content-sensitive so adding OR removing a field does.
    """
    fields = PROFILE_CAPTURE_FIELDS if fields is None else fields
    payload = '\n'.join(sorted(str(f) for f in fields))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def read_capture_schema(directory='.'):
    """The field set + fingerprint recorded beside the maps on disk, or None.

    None means UNKNOWN -- never "unchanged".  Every machine is in that state the first time
    this ships, and the gate must treat it as CHANGED: a missing stamp means we cannot prove
    the maps on disk carry today's fields, and the safe direction for a gate whose only cost is
    ~30 batched profile calls is to rebuild.
    """
    try:
        path = os.path.join(directory, PROFILE_CAPTURE_SCHEMA_FILE)
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) and d.get('fingerprint') else None
    except Exception:
        return None


def write_capture_schema(directory='.', asof=None, verbose=True):
    """Record the field set the maps were just built with.  Best-effort and fully swallowed,
    like every other artifact writer on this path: a stamp must never be able to cost a fetch
    that has already spent its API calls."""
    try:
        payload = {'fingerprint': profile_capture_fingerprint(),
                   'fields': list(PROFILE_CAPTURE_FIELDS),
                   'asof': asof or datetime.today().strftime('%Y-%m-%d')}
        path = os.path.join(directory, PROFILE_CAPTURE_SCHEMA_FILE)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        if verbose:
            print('[maps] capture-schema stamp written to %s (fingerprint %s, %d field(s))'
                  % (path, payload['fingerprint'], len(PROFILE_CAPTURE_FIELDS)), flush=True)
        return payload
    except Exception as _e:
        if verbose:
            print('[maps] WARNING: capture-schema stamp not written (%s: %s) -- the next run '
                  'will treat the schema as UNKNOWN and rebuild, which is the safe direction.'
                  % (type(_e).__name__, _e), flush=True)
        return None


def capture_schema_changed(directory='.'):
    """(changed, why) -- whether the maps on disk were built with TODAY's captured field set.

    Returns `changed=True` when the stamp is absent (unknown != unchanged) or when the
    fingerprint differs, with a `why` string naming the added/removed fields so the run log
    says WHAT changed rather than only THAT something did.
    """
    stamp = read_capture_schema(directory)
    want = profile_capture_fingerprint()
    if stamp is None:
        return True, ('no %s on disk -- the captured field set of the existing maps is UNKNOWN '
                      '(this is the expected state the first time this ships)'
                      % PROFILE_CAPTURE_SCHEMA_FILE)
    if str(stamp.get('fingerprint')) == want:
        return False, ''
    had = set(stamp.get('fields') or [])
    now = set(PROFILE_CAPTURE_FIELDS)
    added, removed = sorted(now - had), sorted(had - now)
    return True, ('captured field set CHANGED since the maps were built (%s -> %s)%s%s'
                  % (stamp.get('fingerprint'), want,
                     '; added: %s' % ', '.join(added) if added else '',
                     '; removed: %s' % ', '.join(removed) if removed else ''))


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
    #  disable=None -- auto-disable off a TTY; see the note at calcScore's Stage-1 bar.
    pbar = tqdm(total=len(uniq), desc='Sector/industry profiles', unit='symbol', disable=None,
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
    #  Profile price + trading currency: capture-only, folded into the volAvg entry below
    #  so a traded VALUE always carries ONE as-of date. See the long note at the capture.
    pricedic, currencydic = {}, {}
    #  Four more capture-only profile fields (2026-08-09), same entry, same as-of date.
    activedic, exchangedic, exchangeshortdic, countrydic, betadic = {}, {}, {}, {}, {}
    #  NINE more (2026-08-14).  Accumulated generically from PROFILE_EXTRA_CAPTURE_FIELDS
    #  rather than as nine named dicts, because the naming pattern above is what made the
    #  capture SET impossible to compare against what is on disk -- see PROFILE_CAPTURE_FIELDS
    #  and `profile_capture_fingerprint`.
    extradic = {f: {} for f in PROFILE_EXTRA_CAPTURE_FIELDS}
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
        #  ---- profile PRICE and CURRENCY: CAPTURE ONLY, NOT WIRED (2026-08-08) ----------
        #  Same argument, third time: both are ALREADY IN this response and were being
        #  discarded on the line above, so capturing them costs ZERO extra API calls, and a
        #  fetch is the only chance we ever get to gain a column.
        #
        #  WHY THEY ARE NEEDED, AND WHY volAvg ALONE IS NOT ENOUGH.  A liquidity floor has
        #  to be denominated in TRADED VALUE, not share count -- ALNTG.PA trades 20,289
        #  shares/day at 0.84, so no share-count floor below 20,000 reaches it, while
        #  DPAM.PA trades 13 shares/day at 855 and is NOT the thinnest name by value.  Value
        #  needs a price, and *** THE PANEL'S OWN `price` CANNOT SUPPLY IT. ***  cdx_df's
        #  price is DERIVED as marketCap / weightedAverageShsOut (getData_fmp), i.e. it is
        #  per ORDINARY SHARE and denominated in the STATEMENT currency (reportedCurrency),
        #  whereas `volAvg` counts the units traded ON THIS LINE in the TRADING currency.
        #  The two do not multiply:
        #    * SHEL.L is QUOTED IN PENCE but REPORTS IN USD -- the derived price and the
        #      traded price are two different numbers in two different currencies;
        #    * SMSN.L and SKHY are out by their GDR ratio, BZ by its ADR ratio, because one
        #      traded unit is not one ordinary share;
        #    * 27 of 100 names on the 2026-08-08 top-100 differ by MORE THAN 2x on that
        #      basis.
        #  And the error CONCENTRATES ON THE EXACT POPULATION A FLOOR TARGETS: depositary
        #  and cross-listed lines are both the thinnest names and the ones whose traded unit
        #  differs from the ordinary share.  A floor built on the derived price would
        #  therefore exclude and retain names on numbers off by up to two orders of
        #  magnitude, silently, and a wrong exclusion leaves NO TRACE -- the name is never
        #  fetched, never scored, and no output says it was dropped.
        #
        #  INFERRING THE TRADING CURRENCY FROM THE TICKER SUFFIX IS NOT A WORKAROUND, IT IS
        #  THE DEFECT.  carveOut.SUFFIX_TO_CURRENCY maps .L -> GBP, which for SHEL.L (a USD
        #  reporter quoted in pence) is wrong by ~100x.  `currency` below is the profile's
        #  own statement of the trading currency and is the only honest source for it.
        #
        #  NOT WIRED.  No consumer reads these yet, deliberately: they are absent from every
        #  existing artifact, so nothing built on them can be tested until a fetch has run
        #  with this capture in place -- the same capture-now-wire-later discipline `isin`
        #  and `volAvg` were given, and for the same reason.  `carveOut._load_volavg_map`
        #  normalises each entry to (volAvg, asof) and IGNORES any other key, so these two
        #  ride along in the same pickle without changing a single consumer today.  They
        #  share the volAvg entry rather than getting pickles of their own because a traded
        #  VALUE is only meaningful when its three inputs carry ONE as-of date; splitting
        #  them across artifacts would reintroduce exactly the stale-vs-fresh comparison the
        #  per-entry date was added to prevent.
        pricedic[sym] = prof.get('price')
        currencydic[sym] = prof.get('currency')
        #  ---- isActivelyTrading / exchange / exchangeShortName / country / beta -------
        #  CAPTURE ONLY, NOT WIRED (CEO-approved 2026-08-09).  Fourth time, same argument:
        #  all five are ALREADY IN this response and were being discarded on the lines
        #  above, so capturing them costs ZERO extra API calls, and a fetch is the only
        #  chance we ever get to gain a column.
        #
        #  PRESENCE IS MEASURED, NOT INFERRED FROM THE VENDOR DOCS.  On a real 100-row
        #  v3/profile payload taken 2026-08-09, every one of the five was non-null on
        #  100/100 rows: `country` was a 2-letter ISO code with 17 distinct values and no
        #  'N/A'; `beta` a float spanning -0.965..3.003 with NO 0.0 placeholder rows (the
        #  usual tell that a vendor is defaulting rather than reporting).
        #
        #  *** WARNING 1 -- `isActivelyTrading` CARRIES NO DISCRIMINATING SIGNAL ON THIS
        #  POPULATION, MEASURED.  It was True on 100/100 of that sample INCLUDING all 39
        #  sampled names that FAILED the previous fetch.  So it separates nothing here.
        #  DO NOT WIRE IT AS A DELISTING, LIVENESS OR TRADEABILITY FILTER EXPECTING IT TO
        #  REJECT ANYTHING -- on the evidence we have it rejects zero names, and a filter
        #  that never fires is worse than absent because it LOOKS like coverage.  It is
        #  captured so that a future fetch can show whether it ever goes False, which is a
        #  question no saved artifact can answer today.  The gate that actually removes
        #  dead names is the delisted prune in getData_gen.get_tickers.
        #
        #  *** WARNING 2 -- `exchange` AND `exchangeShortName` ARE TWO DIFFERENT FIELDS and
        #  both are captured deliberately rather than picking one: `exchange` is the long
        #  venue string ('London Stock Exchange'), `exchangeShortName` the code ('LSE').
        #  They are both free in this payload; guessing which one a future consumer wants
        #  and being wrong would cost a re-fetch, which is the expensive thing here.
        #  NOTE the collision hazard for whoever wires them: `Tickers_df` ALREADY carries
        #  its own `exchangeShortName` from v3/available-traded/list (getData_gen), which
        #  is a DIFFERENT source for the same concept.  Do not assume they agree; if they
        #  are ever joined, the disagreement is a finding, not a merge conflict to resolve
        #  by preference.
        #
        #  *** AND THE ONE ALREADY-CAPTURED FIELD THIS GROUP MAKES EASIER TO MISUSE:
        #  profile `currency` above is the TRADING currency of the listing line, NOT
        #  `reportedCurrency`.  11 of the 100 sampled LSE lines quote in `GBp`, while
        #  fx_rates.py's panel note records that ZERO sources REPORTED in `GBp` on the
        #  2026-08-07 panel -- that was the STATEMENT currency.  So routing profile
        #  `currency` into anything the FX table consumes would take the `GBp` minor-unit
        #  path LIVE FOR THE FIRST TIME.  It does resolve correctly (verified 0.013490),
        #  but "correct" and "unchanged" are not the same claim: that is a real change of
        #  live path and must be made deliberately, with the FX artifact re-read, not
        #  discovered as a side effect of using a field because it was there.
        #
        #  NOT WIRED.  No consumer reads any of the five, and none should until a fetch has
        #  actually run with this capture in place -- the same capture-now-wire-later
        #  discipline `isin`, `volAvg`, `price` and `currency` were given, for the same
        #  reason: they are absent from every existing artifact, so nothing built on them
        #  can be tested first.  They ride in the SAME volavgdic entry under the SAME asof
        #  (see the note at the write below), and `carveOut._load_volavg_map` normalises
        #  each entry to (volAvg, asof) and IGNORES every other key, so no consumer today
        #  changes by a byte.
        activedic[sym] = prof.get('isActivelyTrading')
        exchangedic[sym] = prof.get('exchange')
        exchangeshortdic[sym] = prof.get('exchangeShortName')
        countrydic[sym] = prof.get('country')
        betadic[sym] = prof.get('beta')
        #  ---- THE 2026-08-14 WAVE: mktCap + identity/PIT + fullTimeEmployees ------------
        #  CAPTURE ONLY, NOT WIRED (CEO-approved).  Fifth wave on this payload, same
        #  argument: every field is ALREADY IN the v3/profile response this loop is reading
        #  and was being dropped on the lines above, so capturing costs ZERO extra API calls.
        #  Presence CONFIRMED on a live probe 2026-08-13 and recorded in
        #  APIcallsDocs/endpoint_fields.json -- not inferred from vendor docs.
        #
        #  *** `mktCap` IS THE ONE THAT MATTERS, and it is not a duplicate of anything.
        #  `cdx_df['marketCap']` comes from v3/key-metrics and is a STATEMENT-side figure in
        #  the STATEMENT currency; profile `mktCap` is THIS TRADED LINE's market cap, in the
        #  same terms as profile `price` (captured 2026-08-08) and profile `currency` beside
        #  it.  Having all three from ONE payload at ONE as-of gives an IMPLIED UNIT COUNT
        #  for the line -- `mktCap / price` -- which is INDEPENDENT of the statement's
        #  `weightedAverageShsOut`.  That independence is the whole point: it is the only
        #  handle we have on the DEPOSITARY problem, where a GDR/ADR line's traded units are
        #  a fixed ratio of the ordinary shares the statements count.  MEASURED CASES this
        #  is aimed at: SKHY computes $7.06bn/day of traded value -- 3rd in the top-100 --
        #  on what is essentially SK hynix's HOME line's volume; SMSN.L's profile currency
        #  reads USD at 4,640 and is probably GBp.  Neither is decidable from the panel.
        #  IT IS NOT A FIX FOR EITHER, and must not be described as one -- it is the
        #  MEASUREMENT that makes them decidable, and the decision is a later, separate one.
        #
        #  *** `ipoDate` ACTIVATES A BRANCH THAT HAS NEVER FIRED.  `universe_pit.py:97-100`
        #  already has the code to hold a name out of a point-in-time universe before it was
        #  listed, and it is unreachable for a live name because no artifact carries a
        #  listing date.  Capturing it is the precondition; WIRING IT IS NOT DONE HERE, and
        #  the reason is the standing one -- it is absent from every existing artifact, so
        #  anything built on it is untestable until a fetch has run with this in place.
        #
        #  *** `companyName` REPLACES A YAHOO CALL, EVENTUALLY.  We currently fetch company
        #  names from Yahoo ONE NAME AT A TIME at 0.6s spacing; this field is free here, in
        #  a 100-symbol batch.  Capture now, retire the Yahoo path in a separate change (it
        #  has its own callers and its own failure modes, and folding the two together would
        #  make a capture wave into a refactor).
        #
        #  *** `isAdr` / `isEtf` / `isFund` ARE CAPTURED WITH `isActivelyTrading`'S LESSON
        #  ATTACHED.  That field was captured 2026-08-09 and measured True on 100/100 of a
        #  deliberately ADVERSE sample -- including all 39 names that had FAILED the previous
        #  fetch -- so it discriminates NOTHING on this population and must never be wired as
        #  a liveness filter.  These three are captured WITHOUT a discrimination measurement,
        #  because none exists: no saved artifact carries them, so the only way to measure
        #  their base rates is to capture them and look.  THAT IS THE STATED STATUS -- do not
        #  wire any of the three until its rate has been read off a real panel, and if it
        #  comes back constant, it is an `isActivelyTrading` and gets retired, not used.
        #  `isAdr` is the one with a live question already waiting for it (the depositary
        #  problem above), which is why it is in this wave rather than a later one.
        #
        #  *** `cik` / `cusip` are IDENTIFIERS, and they matter for the SAME reason `isin`
        #  did (register K-1, captured 2026-08-05): issuer-level dedup currently resolves 19
        #  groups on an alphabetical last resort.  `cik` is the SEC issuer key -- exact for
        #  US filers, absent elsewhere -- and `cusip` is instrument-level for North America.
        #  NOTE `cik` IS ALSO ON THE THREE STATEMENT ENDPOINTS; if the two are ever joined,
        #  a disagreement is a FINDING, not a merge conflict to resolve by preference (the
        #  same warning `exchangeShortName` already carries).
        #
        #  *** `fullTimeEmployees` has no consumer and no pending question.  It is here
        #  because it is free, it is the only headcount the pipeline could ever see, and
        #  revenue- or profit-per-employee is a real quality axis we cannot currently ask
        #  about at all.
        #
        #  NOT WIRED.  No consumer reads any of the nine.  They ride the SAME volavgdic
        #  entry under the SAME asof as every earlier wave (`carveOut._load_volavg_map`
        #  normalises to (volAvg, asof) and IGNORES other keys), so no consumer today
        #  changes by a byte.
        for _f in PROFILE_EXTRA_CAPTURE_FIELDS:
            extradic[_f][sym] = prof.get(_f)
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
    #  `price` and `currency` ride in the SAME entry (2026-08-08, capture-only).  They are
    #  the two fields a TRADED-VALUE liquidity floor needs and that the panel cannot supply
    #  -- cdx_df's price is derived (marketCap / weightedAverageShsOut) and denominated in
    #  the STATEMENT currency, which is not the currency or the unit `volAvg` is counted in.
    #  Sharing the entry is what guarantees all three inputs carry ONE as-of date; separate
    #  artifacts would reintroduce the stale-vs-fresh comparison the date was added to stop.
    #  `carveOut._load_volavg_map` normalises to (volAvg, asof) and ignores the extra keys,
    #  so every consumer today is byte-unaffected.
    #  FOUR MORE FIELDS RIDE THE SAME ENTRY (2026-08-09, capture-only): isActivelyTrading,
    #  exchange, exchangeShortName, country, beta.  Same reasoning as `price`/`currency` --
    #  they are only jointly meaningful AT ONE POINT IN TIME (a venue changes, a country of
    #  domicile changes, a beta is a rolling estimate that moves every day), so they share
    #  the one `asof` rather than getting artifacts of their own, which would reintroduce the
    #  stale-vs-fresh comparison the per-entry date exists to prevent.
    #  MERGE-NEVER-OVERWRITE OPERATES ON WHOLE ENTRIES, and that is worth stating: a symbol
    #  this run did not fetch keeps its PREVIOUS entry intact, so it will carry the OLD key
    #  set (no beta, no country) alongside its old asof.  A consumer must therefore treat a
    #  MISSING KEY and a null value as the same thing -- "not captured at that asof" -- and
    #  must never read an absent key as a meaningful False/0.
    #  NINE MORE FIELDS RIDE THE SAME ENTRY (2026-08-14, capture-only) -- see
    #  PROFILE_EXTRA_CAPTURE_FIELDS and the long note at the capture site.  They are spread
    #  from `extradic` rather than named one by one BECAUSE the naming pattern is what let the
    #  capture set drift out of sight of the rebuild gate: the writer can now only ever write
    #  exactly the declared tuple, which is what makes the fingerprint honest.
    volavg_dated = {s: dict({'volAvg': v, 'asof': fidag,
                             'price': pricedic.get(s), 'currency': currencydic.get(s),
                             'isActivelyTrading': activedic.get(s),
                             'exchange': exchangedic.get(s),
                             'exchangeShortName': exchangeshortdic.get(s),
                             'country': countrydic.get(s), 'beta': betadic.get(s)},
                            **{f: extradic[f].get(s) for f in PROFILE_EXTRA_CAPTURE_FIELDS})
                    for s, v in volavgdic.items()}
    merged_volavg, n_kept_v = _merge_industry_dics(
        _read_pickle_or_none(_prev_volavg) if _prev_volavg else None, volavg_dated)
    pd.to_pickle(merged_volavg, f'volavgdic_fmp_{fidag}.pickle')
    #  THE CAPTURE-SCHEMA STAMP, written LAST -- only after every artifact this payload
    #  produces is on disk.  Writing it earlier would let a crash mid-write leave a stamp
    #  asserting a field set the maps do not actually carry, which is worse than no stamp:
    #  the gate would then SKIP the rebuild that would have fixed them.
    write_capture_schema(asof=fidag)
    print(f'[sector/industry build] volAvg captured for '
          f'{sum(1 for v in volavgdic.values() if v is not None)} of {len(volavgdic)} symbols -> '
          f'volavgdic_fmp_{fidag}.pickle, each entry stamped asof={fidag} (kept {n_kept_v} '
          f'pre-existing entr(ies), which carry THEIR OWN older asof -- a group with mixed '
          f'dates is skipped by the dedup liquidity term rather than compared). WIRED into '
          f'carveOut._investability_key (register K-1); register J-1 (a liquidity SCREEN) is '
          f'still NOT wired.')
    print(f'[sector/industry build] profile PRICE captured for '
          f'{sum(1 for v in pricedic.values() if v is not None)} and trading CURRENCY for '
          f'{sum(1 for v in currencydic.values() if v)} of {len(volavgdic)} symbols, folded '
          f'into volavgdic_fmp_{fidag}.pickle under the SAME asof. CAPTURE ONLY -- NOTHING '
          f'reads them yet. They exist because a traded-VALUE liquidity floor (register J-1) '
          f'cannot be built from the panel: cdx_df price is DERIVED (marketCap/shares) in the '
          f'STATEMENT currency, while volAvg counts traded units in the TRADING currency -- '
          f'27 of 100 top-100 names differ by >2x (SHEL.L quotes in PENCE and reports USD; '
          f'SMSN.L/SKHY/BZ differ by their GDR/ADR ratio). The floor waits for these fields.')
    _n_sym = len(volavgdic)
    print(f'[sector/industry build] profile EXTRAS captured into the SAME '
          f'volavgdic_fmp_{fidag}.pickle entry under the SAME asof, of {_n_sym} symbols: '
          f'isActivelyTrading {sum(1 for v in activedic.values() if v is not None)}, '
          f'exchange {sum(1 for v in exchangedic.values() if v)}, '
          f'exchangeShortName {sum(1 for v in exchangeshortdic.values() if v)}, '
          f'country {sum(1 for v in countrydic.values() if v)}, '
          f'beta {sum(1 for v in betadic.values() if v is not None)}. CAPTURE ONLY -- '
          f'NOTHING reads them yet, and two of them carry a warning that must survive this '
          f'run: (1) isActivelyTrading was True on 100/100 of a measured sample INCLUDING '
          f'39 names that FAILED the last fetch, so it is NOT a delisting/liveness filter '
          f'and must not be wired as one; (2) `currency` here is the TRADING currency -- '
          f'LSE lines quote in GBp while ZERO sources REPORT in GBp, so routing it into an '
          f'FX-consuming path takes the GBp minor-unit path live for the first time.')
    _extra_counts = ', '.join(
        '%s %d' % (f, sum(1 for v in extradic[f].values() if v is not None and v != ''))
        for f in PROFILE_EXTRA_CAPTURE_FIELDS)
    print(f'[sector/industry build] 2026-08-14 WAVE captured into the SAME '
          f'volavgdic_fmp_{fidag}.pickle entry under the SAME asof, of {_n_sym} symbols: '
          f'{_extra_counts}. CAPTURE ONLY -- NOTHING reads any of them, and three carry a '
          f'status that must survive this run: (1) `mktCap` is THIS LINE\'s market cap in '
          f'the line\'s own terms, NOT cdx_df[marketCap] (a statement-side figure in the '
          f'STATEMENT currency) -- with profile `price` beside it, mktCap/price is an '
          f'implied unit count independent of weightedAverageShsOut, which is the only '
          f'handle on the depositary problem (SKHY, SMSN.L) and is a MEASUREMENT, not a '
          f'fix; (2) `isAdr`/`isEtf`/`isFund` are captured WITHOUT a discrimination '
          f'measurement because none exists -- read their base rates off this panel before '
          f'wiring any of them, and if one comes back constant it is an `isActivelyTrading` '
          f'and gets retired, not used; (3) `ipoDate` is the precondition for '
          f'universe_pit.py:97-100, a branch that has NEVER fired for a live name -- wiring '
          f'it is a separate change on a panel that actually carries the field.')
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


# Age (days) past which a profile-derived map is called STALE in the run banner.
# Not a hard gate: exceeding it warns LOUDLY, it never blocks or auto-spends calls.
# 60d is a judgement call, sized so a map that missed a whole quarter's worth of
# listings/reclassifications cannot pass unremarked; tune from operator experience.
MAP_STALE_DAYS = 60

# Sector-map coverage (% of the ACTIVE universe carrying a sector) below which a
# rebuild is warranted.  The 2026-08-07 run carved at 84.2% overall / 41.1% on
# KOSDAQ and shipped 32 of its top-100 as sector 'Unknown', so the floor sits
# above that by design.  A rebuild is ~1 batched profile call per 100 symbols --
# ~26 calls on a 2.6k universe, ~100 on a full one -- i.e. sub-1% of a fetch that
# already costs ~16,300 calls.  Cheap enough that refreshing beats carving blind.
MIN_SECTOR_COVERAGE_PCT = 95.0


def _map_age_days(path):
    """Age of a map pickle in days.  Prefers the DATE IN THE FILENAME (the build
    date, which survives a git checkout) and falls back to mtime for the undated
    sectorsdic.  Returns None if the age cannot be determined."""
    try:
        stem = os.path.basename(path)
        for tok in stem.replace('.pickle', '').split('_'):
            try:
                return (datetime.now() - datetime.strptime(tok, '%Y-%m-%d')).days
            except ValueError:
                continue
        return int((datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days)
    except Exception:
        return None


def sector_map_coverage(symbols, sector_pickle='sectorsdic_fmp.pickle'):
    """What FRACTION of the active universe does the cached sector map actually
    cover?  Returns (n_mapped, n_total, pct) -- pct is None when it cannot be
    computed.

    EXISTENCE IS NOT COVERAGE (2026-08-07).  The gate below used to ask only
    'does the pickle exist'.  A map built 2025-12-10 exists perfectly well and
    covered 84.2% of the 2026-08-07 universe -- 41.1% on KOSDAQ, 78.8% on LSE,
    413 of 2,613 names unmapped, 32 of the shipped top-100 carrying
    sectorPickle='Unknown' while the sectorAPI field beside it was 100%
    populated.  That number is one line to print and it is the single number
    that would have caught this eight months earlier, so it is printed on EVERY
    run whether or not it triggers anything."""
    try:
        sectordic = pd.read_pickle(sector_pickle)
        mapped = set()
        for syms in sectordic.values():
            mapped.update(syms)
        total = list(dict.fromkeys(list(symbols)))
        if not total:
            return 0, 0, None
        n_mapped = sum(1 for s in total if s in mapped)
        return n_mapped, len(total), 100.0 * n_mapped / len(total)
    except Exception:
        return None, None, None


def warn_if_maps_stale(stale_days=MAP_STALE_DAYS, verbose=True):
    """RUN-START PRECONDITION CHECK on the four profile-derived maps.

    WHY THIS EXISTS (2026-08-07): the pipeline consumed a sector map built
    2025-12-10 -- eight months old -- and said NOTHING.  The consequences were
    real and invisible: 43.6% sector coverage on KOSDAQ, 77% on LSE, and 32 of
    the shipped top-100 carrying sectorPickle='Unknown'.  A cached map is a
    legitimate optimisation; a cached map of unknown age silently deciding the
    carve is not.  Presence was already checked -- AGE was not checked at all.

    Reports, never blocks: this makes no API call, changes no behaviour, and
    cannot fail a run.  It only makes the state of the maps impossible to miss
    from the top of a run.  Returns {name: {'present','path','age_days','stale'}}.
    """
    specs = [('sectorsdic', 'sectorsdic_fmp.pickle'),
             ('industrydic', 'industrydic_fmp_*.pickle'),
             ('isindic', 'isindic_fmp_*.pickle'),
             ('volavgdic', 'volavgdic_fmp_*.pickle')]
    status = {}
    try:
        for name, pat in specs:
            path = None
            if '*' in pat:
                cands = sorted(glob.glob(pat))
                path = cands[-1] if cands else None
            elif os.path.exists(pat):
                path = pat
            age = _map_age_days(path) if path else None
            status[name] = {'present': path is not None, 'path': path,
                            'age_days': age,
                            'stale': bool(age is not None and age > stale_days)}

        absent = [n for n, s in status.items() if not s['present']]
        stale = [n for n, s in status.items() if s['stale']]
        if verbose and (absent or stale):
            bar = '!' * 78
            print('\n' + bar)
            print('!!! PROFILE-DERIVED MAPS: NOT FRESH -- read this before trusting the carve')
            for name, s in status.items():
                if not s['present']:
                    print('!!!   %-12s MISSING' % name)
                else:
                    age_txt = ('age unknown' if s['age_days'] is None
                               else '%d days old' % s['age_days'])
                    print('!!!   %-12s %s  (%s)%s'
                          % (name, s['path'], age_txt, '  <-- STALE' if s['stale'] else ''))
            if absent:
                print('!!! CONSEQUENCE of a MISSING map: the dedup tiebreak it feeds is')
                print('!!!   STRUCTURALLY SILENT -- issuer groups fall through to the')
                print('!!!   ALPHABETICAL last resort, which prefers a thin dual-class "A"')
                print('!!!   line over its liquid "B" and a preferred over its common.')
            if stale:
                print('!!! CONSEQUENCE of a STALE sector map: names listed since the build')
                print('!!!   carry sector "Unknown", so the carve-out cannot route them and')
                print('!!!   REIT/Mining/financial cohorts leak into the general pool.')
                print('!!!   volAvg is TIME-VARYING -- a stale reading is the worst case.')
            print('!!! FIX: rebuild once from a FULL exchange-defined universe (e.g.')
            print('!!!      -tickerfilter stock_NA1_EU1), or call')
            print('!!!      findAllSectors.findAllSectorsViaProfile(baseurl, api_key).')
            print(bar + '\n', flush=True)
        elif verbose:
            ages = ', '.join('%s=%sd' % (n, s['age_days']) for n, s in status.items())
            print('[maps] all four profile-derived maps present and fresh (%s)' % ages,
                  flush=True)
    except Exception as e:
        if verbose:
            print('[maps] WARNING: staleness check failed safely: %s' % e, flush=True)
    return status


def ensure_sector_industry_maps(symbols, baseurl, api_key, batch_size=100, pace=None,
                                universe_is_subset=False, universe_name=None,
                                force_rebuild=False):
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

    `force_rebuild` -- `-force_rebuild_maps` (CEO, 2026-08-10).  Bypasses the three skip
    conditions (all-present / not-stale / coverage-above-floor) and rebuilds anyway.  It does
    NOT bypass the `universe_is_subset` refusal below: a curated subset must never author
    these shared maps, and "the operator asked for it" is not a reason to write a map that
    covers ~1% of the next full run.  The banner states FORCED vs TRIGGERED explicitly.

    Returns True iff a build ran and wrote the maps; False otherwise.

    THE GATE MUST COVER EVERY ARTIFACT THE WRITER PRODUCES (fixed 2026-08-07).
    `buildSectorIndustryMaps` writes FOUR artifacts from ONE profile payload --
    sectorsdic, industrydic, isindic and volavgdic -- but this gate used to
    short-circuit on the presence of only the first TWO.  isindic/volavgdic were
    added to the writer later and the gate was never widened, so on any machine
    that already held the two older pickles the two newer ones could NEVER be
    born: the build was skipped as 'cached', silently, forever.  That is exactly
    the 2026-08-07 CUR3K run's `isin_map_n: 0` / `volavg_map_n: 0` -- with both
    dedup tiebreaks structurally absent, 19 issuer groups fell through to the
    alphabetical last resort, which systematically prefers a thin dual-class 'A'
    line over its liquid 'B' (TCL-A.TO reached the shipped top-100) and a
    preferred over its common (BMNP).  A presence gate must be derived from what
    the writer WRITES, never from a subset of it frozen at some past revision."""
    sector_present = os.path.exists('sectorsdic_fmp.pickle')
    industry_present = bool(glob.glob('industrydic_fmp_*.pickle'))
    isin_present = bool(glob.glob('isindic_fmp_*.pickle'))
    volavg_present = bool(glob.glob('volavgdic_fmp_*.pickle'))
    missing_desc = ('%s%s%s%s' % (
        '' if sector_present else 'sectorsdic_fmp.pickle ',
        '' if industry_present else 'industrydic_fmp_*.pickle ',
        '' if isin_present else 'isindic_fmp_*.pickle ',
        '' if volavg_present else 'volavgdic_fmp_*.pickle ')).strip() or '<none>'

    # COVERAGE IS PRINTED ON EVERY RUN, unconditionally, before any branch decides
    # anything.  A branch that fires on 100% of runs and says nothing is how this
    # defect survived eight months; this one line is the number that catches it.
    n_mapped, n_total, cov_pct = sector_map_coverage(symbols)
    if cov_pct is None:
        print('[maps] sector-map coverage: UNKNOWN (map unreadable or universe empty)',
              flush=True)
    else:
        print('[maps] sector-map coverage: %.1f%% of the active universe '
              '(%d of %d symbols mapped; %d would carry sector "Unknown")'
              % (cov_pct, n_mapped, n_total, n_total - n_mapped), flush=True)

    map_status = warn_if_maps_stale()
    any_stale = any(s['stale'] for s in map_status.values())
    low_coverage = cov_pct is not None and cov_pct < MIN_SECTOR_COVERAGE_PCT
    all_present = sector_present and industry_present and isin_present and volavg_present
    #  A CODE CHANGE TO *WHICH FIELDS* THE MAPS CAPTURE IS NOW A SKIP CONDITION IN ITS OWN
    #  RIGHT (2026-08-14).  Presence, age and coverage are all properties of the artifacts;
    #  none of them can see that the WRITER now pulls a field it did not pull before, which is
    #  how two shipped capture changes silently failed to land on 2026-08-10.  See
    #  PROFILE_CAPTURE_FIELDS.  An ABSENT stamp counts as CHANGED -- unknown is not unchanged.
    schema_changed, schema_why = capture_schema_changed()

    #  --- THE EXPLICIT OVERRIDE: `-force_rebuild_maps` (CEO, 2026-08-10) -----------------
    #  BYPASSES the three skip conditions above; it does NOT change any of them.  The 60-day
    #  staleness rule is deliberately untouched -- the CEO chose an explicit one-off over
    #  self-maintaining logic, on the grounds that this gate has already had one serious bug
    #  (its presence check covered two of the four artifacts the writer produces, so
    #  isindic/volavgdic could never be born on a machine holding the older two).
    #
    #  WHY IT WAS NEEDED, stated so the flag's purpose is not lost: on the 2026-08-10 run all
    #  four maps existed, none was 60 days stale and coverage was above the floor, so the gate
    #  skipped -- CORRECTLY, by its own rules -- and two capture changes that had already
    #  shipped therefore never landed (`price`/`currency` from 90b0d5f, and
    #  `isActivelyTrading`/`exchange`/`exchangeShortName`/`country`/`beta` from 1e9d353).
    #  Every 2026-08-10 pick carries `volAvg_asof = 2026-08-07`, and nothing would have forced
    #  a rebuild until 2026-10-06.  A CODE change to what the maps CAPTURE has no
    #  representation in a freshness rule that only looks at their AGE.
    if force_rebuild and all_present and not any_stale and not low_coverage and not schema_changed:
        print('[maps] REBUILD **FORCED** by -force_rebuild_maps -- the skip conditions were '
              'ALL SATISFIED (all four maps present, none over the %d-day staleness bar, '
              'sector coverage %s above the %.0f%% floor, captured field set unchanged), so '
              'this run would have reused the cached pickles and spent no API calls. The '
              'operator overrode that explicitly.'
              % (MAP_STALE_DAYS,
                 ('%.1f%%' % cov_pct) if cov_pct is not None else 'UNKNOWN',
                 MIN_SECTOR_COVERAGE_PCT), flush=True)
    elif all_present and not any_stale and not low_coverage and not schema_changed:
        # Idempotent skip -- reuse cached pickles, no rebuild, no API calls.
        # SAY SO.  This branch used to `return False` in total silence.
        #  THE ADVICE THIS USED TO GIVE IS NOW OBSOLETE AND HAS BEEN REMOVED (2026-08-14).  It
        #  said "pass -force_rebuild_maps ... after a change to WHICH FIELDS the maps capture,
        #  which no freshness rule can see".  THE GATE NOW SEES THAT -- the capture-schema
        #  fingerprint is a skip condition, so reaching this branch means the field set on disk
        #  MATCHES the code.  Leaving the old sentence would build a standing
        #  `-force_rebuild_maps` habit around a gate that no longer needs one, and a reflex flag
        #  is how an operator stops reading the gate's verdict at all.
        print('[maps] all four profile-derived maps present, fresh, above the %.0f%% coverage '
              'floor AND built with the current captured field set (fingerprint %s) -- '
              'reusing cached pickles (no rebuild, no API calls). A change to WHICH FIELDS '
              'are captured rebuilds on its own; -force_rebuild_maps is only for a reason the '
              'code cannot see.'
              % (MIN_SECTOR_COVERAGE_PCT, profile_capture_fingerprint()), flush=True)
        return False

    # Otherwise a build is WARRANTED.  State WHY -- FORCED vs TRIGGERED must be
    # distinguishable in the log, so a forced run is never mistaken for the gate having
    # fired on its own (CEO, 2026-08-10).  The API spend that follows is never unexplained.
    triggers = []
    if not all_present:
        triggers.append('missing artifact(s): %s' % missing_desc)
    if any_stale:
        triggers.append('stale map(s): %s (> %d days)'
                        % (', '.join(n for n, s in map_status.items() if s['stale']),
                           MAP_STALE_DAYS))
    if low_coverage:
        triggers.append('sector coverage %.1f%% < %.0f%% floor'
                        % (cov_pct, MIN_SECTOR_COVERAGE_PCT))
    if schema_changed:
        triggers.append('CAPTURE SCHEMA: %s -- the maps on disk cannot carry fields the '
                        'writer only started pulling after they were built, and no freshness '
                        'rule can see that' % schema_why)
    if triggers:
        print('[maps] REBUILD WARRANTED (TRIGGERED by the gate\'s own conditions) -- %s%s'
              % ('; '.join(triggers),
                 ' [-force_rebuild_maps was ALSO passed, but the gate would have rebuilt '
                 'anyway]' if force_rebuild else ''), flush=True)
    else:
        print('[maps] REBUILD WARRANTED (FORCED by -force_rebuild_maps) -- no gate condition '
              'fired; this rebuild is the operator\'s explicit instruction, not the gate\'s '
              'judgement.', flush=True)

    if universe_is_subset:
        bar = '!' * 78
        print('\n' + bar)
        print('!!! SECTOR/INDUSTRY MAP BUILD SKIPPED -- the active universe is a SUBSET')
        print('!!!   universe : %s  (%d symbols)' % (universe_name or '<unnamed>',
                                                     len(list(symbols))))
        print('!!!   missing  : %s' % missing_desc)
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
    #  disable=None -- auto-disable off a TTY; see the note at calcScore's Stage-1 bar.
    #  (This bar is constructed and never driven; left in place rather than deleted because
    #  removing it is a behaviour change to a function nothing in the suite calls, and the
    #  sweep's rule applies to it the same as to any other.)
    tqdm(total=len(ass_df['symbol'].unique()), disable=None)
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

