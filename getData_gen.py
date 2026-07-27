import pandas as pd
import requests
import warnings
import json
import numpy as np
import os
import re
import time


def safe_get(url, params=None, headers=None, timeout=10, retries=3, backoff=1):
    """Perform a GET request with basic retry/backoff, timeout and JSON parsing.

    Returns parsed JSON on success, or None on repeated failures.
    """
    attempt = 0
    while attempt < retries:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError:
                warnings.warn(f"Response from {url} was not valid JSON")
                return None
        except requests.RequestException as e:
            attempt += 1
            if attempt >= retries:
                warnings.warn(f"Failed to GET {url} after {retries} attempts: {e}")
                return None
            # simple exponential backoff
            sleep_for = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_for)
    return None


class _FailedResponse:
    """Stand-in Response returned by ``safe_http_get`` when every attempt raised
    (connection error / timeout).  It carries a FAILING status_code so a downstream
    status-code gate records a definitive, *retryable* ``failcode`` (fetch-unknown)
    instead of either crashing on a ``None`` response or being mislabelled as a
    genuine empty ("no data") response.  See ``safe_http_get`` and the dead-
    fundamentals loop (review ADDENDUM-3 HIGH-1)."""

    def __init__(self, status_code=599, url=None, error=None):
        self.status_code = status_code
        self.url = url
        self.error = error

    def json(self):
        return []


def safe_http_get(url, params=None, headers=None, timeout=10, retries=3, backoff=1,
                  retry_statuses=(429, 500, 502, 503, 504), sleep=None, _get=None):
    """Like ``safe_get`` but returns the raw ``requests.Response`` (NOT parsed JSON),
    with the same retry/backoff/timeout discipline.

    Used by the DELISTED dead-fundamentals loop, whose statement gate needs the
    Response object (``.status_code`` / ``.json()``).  A transient failure -- a 429
    throttle or a 5xx -- is retried with exponential backoff rather than being
    swallowed; only after ``retries`` exhausts do we hand back the last FAILING
    Response, so the caller can record a definitive ``failcode`` (re-audit) and NOT
    silently reclassify a throttled dead name as "no fundamentals" (the HIGH-1 bias
    hole).  If every attempt RAISED (connection error / timeout) we return a
    ``_FailedResponse`` with a failing status_code so the caller never crashes on a
    multi-hour loop.

    ``sleep`` / ``_get`` are injectable for offline testing (no real network)."""
    sleep = time.sleep if sleep is None else sleep
    _get = requests.get if _get is None else _get
    attempt = 0
    last_resp = None
    while attempt < retries:
        try:
            resp = _get(url, params=params, headers=headers, timeout=timeout)
            last_resp = resp
            if getattr(resp, "status_code", None) in retry_statuses:
                attempt += 1
                if attempt >= retries:
                    return resp          # exhausted -> hand back the failing Response
                sleep(backoff * (2 ** (attempt - 1)))
                continue
            return resp                  # non-retryable status (2xx/4xx-non-429) -> done
        except requests.RequestException as e:
            attempt += 1
            if attempt >= retries:
                warnings.warn(f"Failed to GET {url} after {retries} attempts: {e}")
                return last_resp if last_resp is not None else _FailedResponse(url=url, error=str(e))
            sleep(backoff * (2 ** (attempt - 1)))
    return last_resp if last_resp is not None else _FailedResponse(url=url)

def get_tickers(ds, baseurl, api_key, manual_elim=None, tfilt='stock_NA1',sfilt='all', mcapf=-1,fn='',
                as_of=None, registry=None):
    """Build the ticker universe.

    as_of : point-in-time date D (default None).  as_of=None reproduces the live
    universe BIT-FOR-BIT (available-traded/list intersect statement-symbol-lists,
    page-0 delisted prune -- all as today).  Only when a real D is supplied is the
    survivorship-safe PIT membership applied (universe_pit.build_universe): entities
    alive-at-D are retained and not-yet-IPO'd names dropped.

    NOTE (deferred, design Phase 2/3): a FULL PIT universe -- one that RETAINS names
    that delisted after D but are dead today -- needs the paginated delisted
    `registry` (not built yet) AND the page-0-prune inversion.  Until that registry
    is supplied here, build_universe(as_of=D, registry=None) can only filter the
    live occupants by ipoDate<=D; it cannot resurrect already-dead-today names.  The
    as_of=None (live) path -- the one that runs tonight -- is unaffected.
    """
    df = -1
    if ds == 'fromFile':
        # read tickers from CSV file; ensure returned dataframe is assigned to `df`
        df = pd.read_csv(fn)

    elif ds == 'fmp':
        # use safe_get to fetch API endpoints with retries and timeouts
        resp_stockAT_cmp_json = safe_get(f'{baseurl}v3/available-traded/list?apikey={api_key}')
        resp_tckr_json = safe_get(f'{baseurl}v3/financial-statement-symbol-lists?apikey={api_key}')
        resp_stockAT_cmp_df = pd.DataFrame(resp_stockAT_cmp_json) if resp_stockAT_cmp_json else pd.DataFrame()

        # INGESTION CAPTURE (future runs, zero extra calls): available-traded/list
        # carries the raw `type` in {stock,etf,fund,trust} and `name`, which the
        # type-filter below discards for non-stocks. Persist the raw pre-filter
        # table so future runs get a free positive fund/trust tag (feeds the
        # carve-out investment-vehicle detection). Best-effort; never breaks the
        # universe build; does NOT alter the filtered universe returned below.
        try:
            if not resp_stockAT_cmp_df.empty:
                import datetime as _dt
                _keep = [c for c in ['symbol', 'name', 'type', 'exchangeShortName']
                         if c in resp_stockAT_cmp_df.columns]
                if 'symbol' in _keep:
                    _raw = resp_stockAT_cmp_df[_keep].drop_duplicates(subset='symbol')
                    _raw.to_pickle(f"available_traded_raw_{_dt.date.today().isoformat()}.pickle")
        except Exception as _e:
            print(f"WARNING: available-traded raw type/name capture skipped ({_e})")

        resp_tckr_df = pd.DataFrame(resp_tckr_json) if resp_tckr_json else pd.DataFrame()
        resp_tckr_df.columns = ['symbol']

        maskAT = resp_stockAT_cmp_df['symbol'].isin(resp_tckr_df['symbol'])
        tickersAT_df = resp_stockAT_cmp_df[maskAT].drop_duplicates(subset='symbol').reset_index(drop=True)

        df = tickerfilterWrapper(tickersAT_df, tfilt, sfilt, mcapf, baseurl, api_key)

        # GENERATE-IF-MISSING (self-heal the carve-out's sector + industry maps).
        # A fresh git checkout ships neither pickle (both gitignored) and the producer
        # is an orphan, so the carve degrades (industry -> weak keyword fallback) or,
        # on a truly fresh machine, hits carveOut's catastrophic sector-abort. carveOut
        # is credential-less by design and cannot self-heal, so the ingestion layer --
        # which already holds baseurl/api_key and the filtered universe -- is the right
        # home. If BOTH maps are already in CWD this is a no-op (cached pickles reused,
        # no rebuild, no API calls); otherwise both are built ONCE from the filtered
        # universe via batched profile calls. Best-effort: a fetch failure logs a masked
        # warning and falls through to carveOut's existing degrade path -- it never
        # aborts the universe build.
        try:
            from findAllSectors import ensure_sector_industry_maps
            ensure_sector_industry_maps(list(df['symbol']), baseurl, api_key, pace=15)
        except Exception as _e:
            print(f"WARNING: sector/industry map self-heal hook error (non-fatal): {_e}")

    else:
        raise Exception('Not a valid tickers source')

    if manual_elim is None:
        manual_elim = []

    df = df[~df['symbol'].isin(manual_elim)].reset_index(drop=True)

    if as_of is not None:
        # PIT membership over the union of live occupants + delisted registry.
        # Never entered on a live run (as_of=None) -> live universe unchanged.
        import universe_pit as up
        keep = set(up.build_universe(df, registry=registry, as_of=as_of))
        df = df[df['symbol'].isin(keep)].reset_index(drop=True)

    return df

# =========================================================================== #
#  NON-COMMON-INSTRUMENT FILTER (audit M-5)                                    #
# =========================================================================== #
# FMP labels debt, preferred series, warrants, rights and SPAC units as
# type=='stock', so they enter the universe and are scored ON THE COMMON'S
# FUNDAMENTALS (the statements belong to the issuer, not the instrument).  They also
# sit in the Stage-2 z-pool as extra rows, distorting mu/sigma and every percentile
# BEFORE any dedup runs.  Confirmed on the 2026-07-17 universe: 5 Triton preferred
# series (TRTN-PA..-PE), HNNAZ (Hennessy Advisors 4.875% NOTES), IMPPP, GSL-PB, SYF-PA.
#
# WHAT MUST SURVIVE: dual-class COMMONS (TCL-A.TO/TCL-B.TO, ACRI-A.ST, NIVI-B.ST,
# GOOGL/GOOG, UAA/UA, FOXA/FOX, NWSA/NWS, WLYB/WLY, LILAK/LILA, UONEK/UONE, METCB),
# LP/trust UNITS (a partnership's only equity IS units -- DLNG, DMLP, EPD, ET, BIP),
# and foreign ordinaries.
#
# THREE RULES, each validated against this universe rather than assumed:
#
#  A. NAME vocabulary -- the instrument is named as one ("... 4.875% Notes due 2026",
#     "... Warrants", "... Rights", "PERP PFD SER A").  Note bare "Units" is NOT in the
#     vocabulary: it would delete every LP.  Catches 189 names.
#  B. `-P<letters>` suffix on the base symbol = preferred series (US `-PA`, TSX `-PFJ`,
#     Nordic `-PREF`).  Catches 325.  This is the rule the name string CANNOT replace:
#     TRTN-PA..-PE / GSL-PB / SYF-PA all carry the COMMON's name verbatim ("Triton
#     International Limited"), and Triton's common is not even in the universe, so
#     there is no sibling to compare against either.  Safe because the dual-class
#     convention is `-A`/`-B`/`-C`, never `-P`: of 26 `-<single letter>` symbols only 2
#     are `-P*`, and no must-survive name matches.
#  C. SAME-ISSUER SYMBOL EXTENSION -- candidate == a shorter same-name, same-exchange
#     sibling's symbol plus an instrument code, with NO separator (IMPPP = IMPP + P).
#     The tail is restricted to an EXPLICIT WHITELIST because share classes live in the
#     same shape and a permissive rule provably eats real commons: GOOGL = GOOG + "L",
#     UAA = UA + "A", WLYB = WLY + "B", LILAK = LILA + "K", UONEK = UONE + "K",
#     METCB = METC + "B", FOXA, NWSA, CENTA, ASBA, RDIB, PPLC all have this shape.
#     Whitelisting P/R/U/W/Z (+ their two-letter combinations) keeps every one of those
#     -- none of their tails is whitelisted -- while still catching IMPPP and HNNAZ.
#     KNOWN, ACCEPTED MISSES from this conservatism: tails S and V (APOS, VTAS.L,
#     TFGS.L, PEYS.L, SKHYV, ECCV, CECV.DE) are left IN.  Leaving a preferred in is the
#     cheap error; deleting a common is the expensive one.
# Rule A vocabulary.  Every entry requires INSTRUMENT CONTEXT, never a bare word
# (review H1, 2026-07-25).  The first version matched `senior`, `preferred`, `perpetual`,
# `rights`, `notes` and `cumulative` as bare words anywhere in the name and therefore
# DELETED REAL COMMONS: BKD (Brookdale SENIOR Living), SNDA (Sonida SENIOR Living),
# SIA.TO (Sienna SENIOR Living), SNR.L (SENIOR plc), PFBC (PREFERRED Bank), NOTE.ST
# (NOTE AB), LBOW.L (ICG-Longbow SENIOR SECURED UK Property Debt Investments -- a listed
# debt FUND, which the carve should cohort, not delete).  Deleting a common is the
# expensive error and this rule was committing it silently.
#
# Each pattern below was derived by listing EVERY name in the 2026-07-17 universe
# containing the risky word and reading them, not by guessing:
#   senior      23 names -- 20 instruments, ALL of the form "Senior Notes"; the other 3
#               are trade names.  So `senior` is required to be followed by a debt noun.
#   preferred    9 names -- 8 instruments, ALL "Preferred Stock"; PFBC is a bank.
#   perpetual/cumulative -- every instrument use co-occurs with "Preferred Stock", so
#               both words are DROPPED as redundant (and "Perpetuals.com Ltd" exists).
#   rights      15 names -- all SPAC rights, all with Right(s) as the FINAL word.
#   notes       46 names -- NOTE.ST ("NOTE AB (publ)") is a real Swedish common, so
#               `notes` needs a coupon / maturity / seniority / "Notes -<date>" context.
#   %           19 names beyond the above -- every one a coupon (Saratoga 8.00%, Duke
#               5.625%, Bristol Water Cum.Irred.Pref.Shs); zero trade names.  Kept bare.
#   warrants    67 names -- all instruments.  Kept bare (a future "Warrant Technologies"
#               would false-positive; accepted, and it would show in the removal CSV).
_NON_COMMON_NAME_PATTERNS = (
    ('coupon-rate',      r'\d+(?:\.\d+)?\s*%|\b\d+\s+\d/\d\s*%'),
    ('maturity',         r'\bdue\s+(?:19|20)\d\d\b|\bexp(?:iring)?\.?\s+\d'),
    ('senior-debt',      r'\bsenior\s+(?:notes?|debentures?|bonds?|unsecured|sub)'),
    # `subordinated` needs a DEBT noun after it: "Class B Subordinate Voting Shares"
    # (XNDU) is COMMON equity, and "The Law Debenture Corporation p.l.c." (LWDB.L) is a
    # listed investment trust whose trade name contains "Debenture".  Both were removed
    # by the first draft of this rule.  Real debenture instruments all carry a coupon, so
    # bare `debenture` is dropped -- coupon-rate covers them.
    ('subordinated',     r'\bsubordinated\s+(?:notes?|debentures?|bonds?|securit)'
                         r'|\b(?:jr|junior)\.?\s+subordinat'),
    ('preferred-class',  r'\bpreferred\s+(?:stock|shares?|securit|units?)|\bpfd\b'
                         r'|\bpref\.?\s*sh(?:s|ares)?\b|\bcum\.?\s*(?:irred\.?\s*)?pref'),
    # NO bare `depositary` pattern.  "American Depositary Shares" is how a FOREIGN
    # COMMON trades in the US -- ARM (Arm Holdings), PONY (Pony AI), LOT (Lotus Tech),
    # CHA, HDL, and 10 more were all deleted by that pattern, and foreign ordinaries are
    # explicitly on the must-keep list.  The genuine PREFERRED-depositary lines are
    # covered without it: RILYL/RILYP/NEWTP/FCNCN carry "Preferred Stock"
    # (preferred-class) and USB-PS is caught by rule B's `-PS` suffix.
    ('depositary-pfd',   r'\bdepositary\b[^.]{0,120}?\b(?:preferred|pfd)\b'
                         r'|\bdep\s*1/'),
    ('warrant',          r'\bwarrants?\b'),
    ('rights',           r'\brights?\s*\.?\s*$|\bcontingent\s+value\s+rights?\b'),
    ('notes-in-context', r'\bnotes?\s*[-–]\s*\d|\bnotes?\s+d(?:ue|ated)\b'
                         r'|\b(?:fixed|floating)[- ]rate\s+notes?\b'
                         r'|\b(?:jr|sr)\s*(?:sub\s*)?nt\b|\bnt\s*\d'),
)
_NON_COMMON_NAME_RES = tuple((tag, re.compile(pat, re.I))
                             for tag, pat in _NON_COMMON_NAME_PATTERNS)


def _non_common_name_tag(name):
    """The first instrument-context pattern a company name matches, or '' if none."""
    if not isinstance(name, str) or not name:
        return ''
    for tag, rx in _NON_COMMON_NAME_RES:
        if rx.search(name):
            return tag
    return ''

# base symbol ends in a preferred-series marker: -P, -PA, -PFJ, -PREF
_PREFERRED_SUFFIX_RE = re.compile(r'-P[A-Z]{0,3}$')

# Whitelisted instrument tails for rule C (see the note above on why this is a whitelist
# and not "anything short").  EXACTLY what is admitted (the comment used to say
# "P/R/U/W/Z + their two-letter combinations", which understated the third alternative --
# review H1 sub-finding):
#   P            preferred, bare
#   P<letter>    preferred series (PA, PN, PO, PP...)
#   R U W Z      rights / units / warrants / misc-debt, bare
#   <letter>RUWZ any letter FOLLOWED by an instrument code (CW, OW, TW, BU, CZ...) --
#                the SPAC-instrument shape, e.g. ALFUW, TVACW
#   [PRUWZ]<letter>  an instrument code followed by any letter (WR, UU, WW...)
# The share-class letters (A B C D E K L M N O S V) are admitted ONLY in the
# <letter>+instrument-code position, never alone -- which is what keeps GOOGL (GOOG+L),
# UAA (UA+A), WLYB, LILAK, UONEK, METCB, FOXA, NWSA, CENTA, ASBA, RDIB and PPLC.
_INSTRUMENT_TAIL_RE = re.compile(r'^(P[A-Z]?|[A-Z]?[RUWZ]|[PRUWZ][A-Z])$')


def _sym_base(s):
    """Ticker without its exchange suffix ('ACRI-A.ST' -> 'ACRI-A')."""
    return s.rsplit('.', 1)[0] if '.' in s else s


def _sym_exchange_suffix(s):
    return s.rsplit('.', 1)[1] if '.' in s else ''


def filter_non_common_instruments(df, verbose=True, log_csv=True):
    """Drop debt / preferred / warrant / rights lines that FMP types as 'stock'.

    Returns the filtered frame.  Every removal is logged with the rule that caught it
    (loud stdout summary + a dated CSV) so the exclusion list is auditable and never
    silent.  Requires 'symbol'; uses 'name' when present.
    """
    if df is None or 'symbol' not in getattr(df, 'columns', []) or df.empty:
        return df
    sym = df['symbol'].astype(str)
    name = df['name'].astype(str) if 'name' in df.columns else pd.Series('', index=df.index)
    bases = sym.map(_sym_base)

    reason = pd.Series('', index=df.index)

    # rule A -- named as a non-common instrument, with INSTRUMENT CONTEXT required.
    # The specific pattern is recorded (not just 'name-vocabulary') so the removal CSV
    # can be audited pattern by pattern -- that is how the false positives in the first
    # version were found.
    name_tag = name.map(_non_common_name_tag)
    hit_a = name_tag != ''
    reason[hit_a & (reason == '')] = 'name:' + name_tag[hit_a & (reason == '')]

    # rule B -- preferred-series ticker suffix
    hit_b = bases.str.contains(_PREFERRED_SUFFIX_RE)
    reason[hit_b & (reason == '')] = 'preferred-suffix'

    # rule C -- same-issuer symbol extension with a whitelisted instrument tail
    try:
        import carveOut as _co
        norm = name.map(_co._norm_issuer_name)
    except Exception:
        norm = pd.Series('', index=df.index)
    groups = {}
    for s, n, x in zip(sym, norm, sym.map(_sym_exchange_suffix)):
        if n:
            groups.setdefault((n, x), []).append(s)
    ext_hits = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        for cand in members:
            cb = _sym_base(cand)
            for other in members:
                ob = _sym_base(other)
                if len(ob) >= len(cb) or not cb.startswith(ob):
                    continue
                if _INSTRUMENT_TAIL_RE.match(cb[len(ob):]):
                    ext_hits.add(cand)
                    break
    hit_c = sym.isin(ext_hits)
    reason[hit_c & (reason == '')] = 'symbol-extension'

    drop = reason != ''
    if not drop.any():
        return df

    removed = pd.DataFrame({'symbol': sym[drop], 'name': name[drop],
                            'rule': reason[drop]}).reset_index(drop=True)
    if verbose:
        counts = removed['rule'].value_counts().to_dict()
        print("SHARE-CLASS FILTER: removed %d non-common instrument line(s) of %d "
              "(%s)" % (len(removed), len(df), counts), flush=True)
        print("  removed symbols: %s" % ', '.join(sorted(removed['symbol'])), flush=True)
    if log_csv:
        try:
            fidag = pd.Timestamp.today().strftime('%Y-%m-%d')
            fn = f'ExcludedShareClasses_{fidag}.csv'
            removed.to_csv(fn, index=False)
            print(f'  share-class exclusion list written to: {fn}', flush=True)
        except Exception as _e:
            print(f'  WARNING: could not write share-class exclusion list ({_e})', flush=True)

    return df[~drop].reset_index(drop=True)


def tickerfilterWrapper(tickdf,tfilt,sfilt,mcapf,baseurl,api_key):
    df = tickdf
    if tfilt == 'stock_US1':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_US1 = filter_tickers(tickers_df_stock, 'exchangeShortName', ['NYSE', 'NASDAQ'], mcapf, api_key)
        df = tickers_df_stock_US1
    if tfilt == 'stock_NA1':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_NA1 = filter_tickers(tickers_df_stock, 'exchangeShortName', ['NYSE', 'NASDAQ', 'TSX'], mcapf,
                                              api_key)
        df = tickers_df_stock_NA1
    if tfilt == 'stock_WW1_TV':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_WW1_TV = filter_tickers(tickers_df_stock, 'exchangeShortName',
                                                 ['NYSE', 'NASDAQ', 'EURONEXT', 'LSE', 'XETRA'], mcapf, api_key)
        df = tickers_df_stock_WW1_TV
    elif tfilt == 'stock_NA1_EU1':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_NA1_EU1 = filter_tickers(tickers_df_stock, 'exchangeShortName',
                                                   ['NYSE', 'NASDAQ', 'EURONEXT', 'LSE', 'TSX', 'XETRA', 'STO', 'OSE',
                                                    'ICE'], mcapf, api_key)
        df = tickers_df_stock_NA1_EU1
    elif tfilt == 'stock_US1_EU1':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_US1_EU1 = filter_tickers(tickers_df_stock, 'exchangeShortName',
                                                   ['NYSE', 'NASDAQ', 'EURONEXT', 'LSE', 'XETRA', 'STO', 'OSE',
                                                    'ICE'], mcapf, api_key)
        df = tickers_df_stock_US1_EU1
    elif tfilt == 'stock_US1_EU2':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_US1_EU2 = filter_tickers(tickers_df_stock, 'exchangeShortName',
                                                   ['NYSE', 'NASDAQ', 'EURONEXT'], mcapf, api_key)
        df = tickers_df_stock_US1_EU2

    # Drop debt/preferred/warrant/rights lines that FMP types as 'stock' (audit M-5).
    # Placed here, AFTER the type+exchange filters and BEFORE the sector filter, so it
    # applies to every tfilt branch on one code path and so nothing downstream -- the
    # Stage-2 z-pool, the mean-relative Stage-1 baselines, the carve dedup -- ever sees
    # an instrument line.
    df = filter_non_common_instruments(df)

    if sfilt != 'all':
        df = filterBySector(df, sfilt)
    delist_json = safe_get(f'{baseurl}/v3/delisted-companies?page=0&apikey={api_key}')
    delist_df = pd.DataFrame(delist_json) if delist_json else pd.DataFrame()
    delist = list(delist_df['symbol']) if 'symbol' in delist_df.columns else []
    # record delisted tickers to a file for auditing
    try:
        import csv
        fidag = pd.Timestamp.today().strftime('%Y-%m-%d')
        with open(f'delisted_tickers_{fidag}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(delist)
    except Exception:
        # non-fatal: if we can't write, continue
        pass
    df = df[~df['symbol'].isin(delist)].reset_index(drop=True)

    return df

def filterBySector(df,sfilt):
    sectordic = pd.read_pickle('sectorsdic_fmp.pickle')

    dfsfilt = df[df['symbol'].isin(sectordic[sfilt])]

    return dfsfilt

def filter_tickers(ticker_df, colname, cond, mcap, api_key):
    # maybe check if colname is a string and is found in ticker_df
    # maybe check if cond is of type which is valid as a condition
    # start with the full dataframe; apply condition filters if provided
    ntdf = ticker_df.copy()

    if cond and isinstance(cond, str):
        ntdf = ticker_df[ticker_df[colname] == cond]
    elif cond and isinstance(cond, list) and all(isinstance(elem, str) for elem in cond):
        mask = np.zeros(len(ticker_df), dtype=bool)
        for c in cond:
            mask = mask | (ticker_df[colname].values == c)

        ntdf = ticker_df[mask]

    if mcap > 0:
        # use the already-filtered dataframe size when deciding whether to do per-symbol checks
        if len(ntdf) < 3000:
            # iterate over the filtered set and drop those below mcap
            to_drop = []
            for row in ntdf.itertuples():
                symb = row.symbol
                tempjson = safe_get(f'https://financialmodelingprep.com/api/v3/profile/{symb}?apikey={api_key}')
                if tempjson and len(tempjson) > 0:
                    mcap_inst = tempjson[0].get('mktCap')
                    if mcap_inst is None or mcap > mcap_inst:
                        to_drop.append(row.symbol)
                else:
                    # missing profile — treat as fail and drop
                    to_drop.append(row.symbol)

            if to_drop:
                ntdf = ntdf[~ntdf['symbol'].isin(to_drop)]
        else:
            warnings.warn("To many tickers left after other filters to do a market cap screening. It would take forever")
    ntdf.reset_index(drop=True, inplace=True)

    return ntdf

def checkIfValidFS(fs):
    retbool = True
    if any(fs['price'][0:10].isna()):
        retbool = False

    return retbool

def fixAfterGetData(BoMetric_df, cdx_df):
    BoMetric_df = BoMetric_df.dropna(subset=['source'])
    tempfix = BoMetric_df.reset_index(drop=False)
    tempfix2 = tempfix.drop(['index'], axis=1)
    BoMetric_df = tempfix2

    cdx_df = cdx_df.dropna(subset=['source'])
    tempfix = cdx_df.reset_index(drop=False)
    tempfix2 = tempfix.drop(['index'], axis=1)
    cdx_df = tempfix2

    BoMetric_df = forceNumOnDf(BoMetric_df)
    cdx_df = forceNumOnDf(cdx_df)

    return BoMetric_df, cdx_df

def forceNumOnDf(df):
    # Safely coerce each non-identifier column to numeric where possible.
    dftemp = df.copy()
    # Columns to preserve (identifiers / non-numeric)
    preserve = set()
    if 'date' in dftemp.columns:
        preserve.add('date')
    if 'source' in dftemp.columns:
        preserve.add('source')
    # STRING / DATE passthrough columns that MUST survive ingest -- without being listed
    # here pd.to_numeric coerces them to all-NaN, silently destroying the field (this is
    # exactly what happened to reportedCurrency before it was added). None of them reach
    # BoMetric_df, and downstream numeric ops select numeric dtypes or explicit metric
    # lists, so carrying the raw strings through is safe.
    #   reportedCurrency  reporting currency code -> USD market-cap banding
    #                     (carveOut.marketcap_usd_series)
    #   period            'Q1'..'Q4'/'FY' -> tells a 3-month flow from a semi-annual
    #                     filer's 6-month flow (audit C-1)
    #   fillingDate /     filing + acceptance timestamps -> point-in-time availability
    #   acceptedDate      (audit H-2); NB fillingDate is a placeholder = period end on
    #                     ~50% of rows, so acceptedDate is the discriminator
    #   periodEndDate     the RAW fiscal period end, kept beside the quarter-stamped
    #                     `date` (utils.setDatesToQuarterly overwrites `date` in place)
    # calendarYear is deliberately NOT preserved: it is a genuine year and is better off
    # numeric.
    #   grahamUndefinedReason  why a Graham row is undefined (ruling Q1.3) -- a reason
    #                     CODE, so it must survive as a string like the others
    for _passthrough in ('reportedCurrency', 'period', 'fillingDate', 'acceptedDate',
                         'periodEndDate', 'grahamUndefinedReason',
                         'reportingFrequency'):
        if _passthrough in dftemp.columns:
            preserve.add(_passthrough)

    for col in dftemp.columns:
        if col in preserve:
            # skip coercion for identifier columns
            continue
        try:
            # try vectorized coercion for the column; invalid entries become NaN
            coerced = pd.to_numeric(dftemp[col], errors='coerce')
            dftemp[col] = coerced
        except Exception:
            # fallback: try coercing element-wise to be extra defensive
            try:
                dftemp[col] = dftemp[col].apply(lambda x: pd.to_numeric(x, errors='coerce'))
            except Exception:
                # leave the column as-is if it cannot be coerced
                dftemp[col] = dftemp[col]

    dftemp.replace([np.inf, -np.inf], np.nan, inplace=True)
    return dftemp