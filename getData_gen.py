import pandas as pd
import requests
import warnings
import json
import numpy as np
import os
import re
import time
import sys
from tqdm import tqdm


def bar_print(msg):
    """`print()` replacement for anything emitted while a tqdm progress bar is live.

    A bare `print()` puts its newline into the console while tqdm is mid-render with '\\r':
    the bar's line is never cleared, so a fragment of it is stranded on screen and the bar
    restarts one line down.  The warnings in `safe_json_list` below fire from INSIDE the
    per-name loops of the AggScore-CSV and presentation stages, both of which drive a bar,
    and they fire most when the API is throttled -- i.e. exactly when the operator most
    needs to read them.

    `tqdm.write` clears every live bar, writes the line, then redraws.  With no bar alive it
    degrades to a plain write, so offline callers and tests are unaffected.

    PRESENTATION ONLY: identical text, stream (stdout) and flush behaviour to the
    `print(..., flush=True)` calls it replaces.  Lives here because this is the leaf module
    the fetch and post-fetch stages both already import.  (`getData_fmp` and `failTests`
    carry local twins: importing this module from `failTests` would make the import graph
    circular.)
    """
    tqdm.write(msg, file=sys.stdout)
    sys.stdout.flush()


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
    fundamentals loop (review ADDENDUM-3 HIGH-1).

    IT MUST BE RESPONSE-SHAPED, NOT JUST STATUS-SHAPED (review B1, fixed 2026-07-31).
    This shim used to define only `status_code` / `url` / `error` / `json()`.  When the
    2026-07-31 hardening routed `postBoRank._fetch_ticker_dcf` through `safe_http_get`, the
    diagnostic that consumes its 4th return value did `resp_dcf.text[:100]` behind an
    `is not None` guard -- which a `_FailedResponse` SATISFIES -- and raised
    `AttributeError: '_FailedResponse' object has no attribute 'text'`, aborting Stage-2 in
    exactly the dead/hung-endpoint scenario the hardening exists for.  `.text` is the
    second-most-used Response attribute after `.status_code`, so the shim owning it is the
    fix that generalises: any OTHER consumer reaching for `.text` is now also safe, whereas a
    `getattr` at one call site would only have fixed that site.  Kept as a read-only property
    so it cannot be accidentally overwritten and so it stays consistent with `json()`."""

    def __init__(self, status_code=599, url=None, error=None):
        self.status_code = status_code
        self.url = url
        self.error = error

    @property
    def text(self):
        """A body-shaped string describing WHY there is no body."""
        return '<no response body: request failed%s>' % (
            ' (%s)' % self.error if self.error else '')

    @property
    def content(self):
        return self.text.encode('utf-8')

    @property
    def ok(self):
        return False

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

def safe_json_list(url, params=None, headers=None, timeout=10, retries=3, backoff=1,
                   sleep=None, _get=None, label=None, verbose=True):
    """GET a JSON endpoint and return a LIST of records -- never raise, never hang.

    WHY THIS EXISTS (fix, 2026-07-31).  The FETCH loop is hardened (safe_http_get: timeout,
    retries, exponential backoff) but the DELIVERABLE stages were not.  `writeBoAggToCSV`
    (~4-5 calls x 100 names) and `createPresentation` (~7 calls x 20 names) fired ~500-700
    calls as bare `requests.get(url).json()`:
      * NO TIMEOUT -- a hung socket stalls an UNATTENDED run INDEFINITELY, which on a
        12-hour overnight job is the worst failure mode available;
      * NO RETRY -- a single 429 is terminal;
      * `.json()` CHAINED to the call -- a throttled 200 carrying an HTML body raises
        JSONDecodeError, and the loop bodies had no try/except, so the exception propagated
        out of the stage and cost the CSV, the XLSX, the forensic CSV, the postRank pickle
        and the pick-log.
    And it runs IMMEDIATELY AFTER 12+ hours of sustained API load, which is exactly when a
    throttle is most likely.  Losing 12 hours of fetched data to a post-processing network
    error is not an acceptable failure mode, so every one of these calls now degrades to an
    EMPTY LIST instead.

    Returning `[]` is what makes the degradation land at COLUMN granularity rather than
    stage granularity: every consumer in those loops already guards on `len(resp) == 0` and
    writes 'NaN' for that field, so a failed call costs one cell, not the deliverable.

    A dict body (FMP returns `{'Error Message': ...}` on some failures) becomes `[]` unless
    it looks like a single record, in which case it is wrapped -- the same normalisation the
    call sites were doing ad hoc.

    `sleep` / `_get` are injectable for offline testing (no real network).
    """
    resp = safe_http_get(url, params=params, headers=headers, timeout=timeout,
                         retries=retries, backoff=backoff, sleep=sleep, _get=_get)
    status = getattr(resp, 'status_code', None)
    if status != 200:
        if verbose:
            bar_print('  WARNING: %s returned status %s -- that field degrades to NaN for this '
                      'name; the stage continues.' % (label or url, status))
        return []
    try:
        data = resp.json()
    except Exception as _e:
        # THE throttled-200-with-an-HTML-body case.  Bare `.json()` raised here.
        if verbose:
            bar_print('  WARNING: %s returned a 200 with an unparseable body (%s) -- that field '
                      'degrades to NaN for this name; the stage continues.'
                      % (label or url, type(_e).__name__))
        return []
    if isinstance(data, dict):
        if 'Error Message' in data or 'error' in str(data).lower():
            return []
        return [data]
    return data if isinstance(data, list) else []


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

        # LOUD UNIVERSE BANNER, printed HERE -- the point where the universe definition
        # is actually applied -- so every caller (Sbocker, delisted_ingest) gets it, and
        # so it lands BEFORE the multi-hour fundamentals fetch rather than after.  It
        # states the definition fingerprint and, for the four names whose meaning changed
        # on 2026-08-02, that artifacts carrying the same name are NOT comparable on
        # membership.  See universes.run_banner.
        import universes as un
        print(un.run_banner(tfilt), flush=True)

        df = tickerfilterWrapper(tickersAT_df, tfilt, sfilt, mcapf, baseurl, api_key)

        # RESOLVED count against the live-verified expectation.  A run legitimately lands
        # UNDER the expectation (the instrument filter, the sector filter and the delisted
        # prune all remove members after the exchange filter), so this is a sanity read,
        # not a gate -- but a resolved count of ZERO, or one wildly off the expectation,
        # is the exact signature of the dead-exchange-code defect and must be visible.
        _exp = un.expected_count(tfilt)
        print('UNIVERSE %s RESOLVED: %d members (fingerprint %s%s)'
              % (tfilt, len(df), un.definition_fingerprint(tfilt),
                 '' if _exp is None else ', pre-filter expectation ~%d' % _exp),
              flush=True)
        if len(df) == 0:
            print('!!! UNIVERSE %s RESOLVED TO ZERO MEMBERS -- this is the signature of '
                  'an exchange code that matches nothing (the EURONEXT/OSE defect). '
                  'Check universes.UNIVERSES against the live exchange list before '
                  'spending a fetch.' % tfilt, flush=True)

        # PER-CODE FLOOR.  A universe-level total cannot see a dead exchange code -- that
        # is exactly how EURONEXT/OSE hid for the life of the project (the universe still
        # resolved to thousands of names). Per code, a dead code loses 100% of ITSELF and
        # is unmissable. See universes.check_resolved_counts for the measured thresholds.
        if 'exchangeShortName' in getattr(df, 'columns', []):
            _by_code = df['exchangeShortName'].value_counts().to_dict()
            _codes = un.exchanges(tfilt)
            if _codes:
                print('  per-exchange resolved: %s'
                      % ', '.join('%s %d/%d' % (c, int(_by_code.get(c, 0)),
                                                un._VERIFIED_COUNTS.get(c, 0))
                                  for c in _codes), flush=True)
            for _c, _v, _r, _sf in un.check_resolved_counts(tfilt, _by_code):
                print('!!! UNIVERSE %s: exchange code %r returned %d names against a '
                      'verified %d -- %.0f%% SHORT (natural attrition from the instrument '
                      'filter and delisted prune is at most ~%.0f%%). Either the code was '
                      'renamed by FMP or its venue shrank; a code that matches NOTHING is '
                      'the EURONEXT/OSE defect. Re-verify against the live exchange list '
                      'before spending a fetch.'
                      % (tfilt, _c, _r, _v, 100 * _sf,
                         100 * un.RESOLVED_WORST_NATURAL_SHORTFALL), flush=True)

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
        # SUBSET UNIVERSES MUST NOT AUTHOR THE SHARED MAPS (2026-08-02).  This hook is
        # handed the FILTERED universe, and `buildSectorIndustryMaps` used to overwrite
        # `sectorsdic_fmp.pickle` outright -- so once an explicit-membership universe
        # (stock_TEST1, 142 names) could shrink `df`, a test run on a machine lacking the
        # maps would author a 142-symbol map and the NEXT FULL RUN would carve 10,693
        # names against it: non-empty, so past carveOut's empty-map abort, with REIT and
        # Mining leaking wholesale. That hazard is NEW with the selectable-universe work
        # -- `-nrTaT` never shrank `df` here (the cap applies downstream in
        # getData_fmp) -- so the guard belongs with it. `universes.symbols()` is exactly
        # the "membership is an explicit curated list" test.
        try:
            from findAllSectors import ensure_sector_industry_maps
            ensure_sector_industry_maps(list(df['symbol']), baseurl, api_key, pace=15,
                                        universe_is_subset=(un.symbols(tfilt) is not None),
                                        universe_name=tfilt)
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

    THIS FILTER STAYS, AND IT LOSES ITS BURDEN OF COMPLETENESS (2026-08-05).
    carveOut's canonical-choice dedup now collapses each issuer to one line and PICKS the
    canonical member, so a non-common line that has a sibling in the pool no longer needs
    catching HERE -- if it slips through, dedup simply prefers the sibling. That removes
    the property that has actually been failing: this filter had to be COMPLETE, and it
    never could be (it is three ANGLO rules -- English debt vocabulary, the US/Nordic
    `-P<letters>` suffix, and a shared symbol prefix -- so it caught 1 of 196 Korean
    preferreds and 0 of 1,046 restored Continental names).

    BUT IT IS STILL NEEDED, because canonical-choice cannot pick a better sibling that is
    not in the pool. MEASURED on the live 2026-08-04 list, all 51,703 type=='stock' lines
    (i.e. the RAW available-traded capture, NOT intersected with the statement-symbol
    list -- state the population, because the design spec's figures are 902/760/142 on a
    slightly narrower one and the two must not be conflated):
      * it removes 1,097 lines;
      *   912 (83%) have a surviving same-normalised-name sibling in the live list, so
          canonical-choice would have handled them anyway;
      *   185 (17%) DO NOT, and those are catchable only by a detector: name:rights 48,
          name:coupon-rate 42, preferred-suffix 40, name:preferred-class 27,
          name:warrant 22, name:notes-in-context 3, name:maturity 3.
      * RULE C (`symbol-extension`, 468 removals) is the one rule that is now purely a
        COST OPTIMISATION: 0 of its 468 removals lack a sibling, so dedup covers 100% of
        them. Keeping it saves ~5 statement calls per line pre-fetch and keeps them out of
        the Stage-2 z-pool (which dedup does not police); it is no longer load-bearing for
        correctness.
    So: DO NOT WIDEN this filter, and DO NOT "fix" SHARE_CLASS_FILTER_KNOWN_GAPS below --
    WHLRD, WHLRL and BWNB all have siblings in the pool and canonical-choice handles all
    three (BWNB in particular is grouped by the K1 statement key despite the truncated
    FMP name that defeats every name-based rule). Widening a removal rule is where a
    deleted common comes from; that is the expensive error and it is now avoidable.
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


# =========================================================================== #
#  ISIN-BASED SAME-ISSUER DETECTOR -- BUILT, DOCUMENTED, DELIBERATELY NOT WIRED  #
#  (2026-08-02).  Per the CEO's standing practice: write the logic we want, and   #
#  do not apply it until it is decided.  Wiring a NEW removal rule changes the    #
#  production universe, which is a product call, not a bug fix.                   #
#                                                                               #
#  WHY IT EXISTS.  `filter_non_common_instruments`' three rules are all ANGLO:    #
#  rule A is English debt/preferred vocabulary, rule B is the US/Nordic           #
#  `-P<letters>` symbol suffix, rule C needs the instrument symbol to be a        #
#  shorter sibling's symbol PLUS a whitelisted tail.  Restoring 1,046 Continental #
#  European names, the filter removed EXACTLY ZERO of them -- which is equally     #
#  consistent with "clean venues" and "blind filter".  It is the blind filter.    #
#  Measured on the live 2026-08-02 list (1 batched profile call):                 #
#                                                                               #
#    ADMITTED NON-COMMON LINES (different ISIN, IDENTICAL company name, trading   #
#    at a discount to the common -- i.e. the Korean-preferred failure mode):      #
#      CBE.PA    Robertet S.A.  FR0000045601 vs RBT.PA   FR0000039091  -17.9%     #
#                (a French `certificat d'investissement` -- non-voting)           #
#      PREVA.AS  Value8 N.V.    NL0015118803 vs VALUE.AS NL0010661864  -29.9%     #
#                (`PREferente Aandelen` -- cumulative preference shares)          #
#                                                                               #
#    ADMITTED LITERAL DUPLICATES (SAME ISIN, two symbols, ~0.1-0.3% apart --      #
#    two ranking slots for one economic bet):                                     #
#      HAFNIO.OL / HAFNI.OL    Hafnia Limited  SGXZ53070850                       #
#      CATG.PA   / ALCAT.PA    S.A. Catana Group FR0010193052                     #
#                                                                               #
#  NO SYMBOL-SHAPE RULE CAN EVER CATCH THE FIRST TWO: `CBE` shares no prefix with  #
#  `RBT`, and `PREVA` shares none with `VALUE`.  ISIN is the only available         #
#  discriminator -- and the pipeline ALREADY FETCHES IT: `v3/profile` returns       #
#  `isin`, and `findAllSectors._fetch_profiles_batched` pulls profiles for the      #
#  whole universe to build the sector/industry maps.  So the blocker here is        #
#  NEITHER data NOR cost -- it is the product decision.                            #
#                                                                               #
#  CORRECTLY LEFT ALONE by these rules, and the reason each must survive:          #
#    HEIA.AS / HEIO.AS   Heineken N.V. vs Heineken HOLDING N.V. -- the normaliser   #
#                        strips "Holding" so the NAMES collide, but these are two   #
#                        separate issuers with separate financials. Different ISIN  #
#                        AND different fundamentals, so `same_name_different_isin`  #
#                        would flag them: THIS IS WHY THE RULE IS NOT WIRED.       #
#    WWI.OL / WWIB.OL    Wilh. Wilhelmsen A/B -- dual-class COMMONS, must survive.  #
#    CAT31.PA, CRBP2.PA  Credit Agricole regional mutuals: a `certificat            #
#                        cooperatif d'investissement` is the ONLY listed equity of  #
#                        a cooperative, exactly as units are for an LP (which the   #
#                        repo explicitly keeps). Not duplicates -- keep.            #
#                                                                               #
#  The Heineken case is the whole argument for not auto-applying this: the         #
#  same-name-different-ISIN signal has a REAL false positive on the very first     #
#  universe it was tested against, and deleting a common is the expensive error.   #
#  A safe version needs a fundamentals check (do the two lines report the same     #
#  revenue/assets?) on top of the ISIN signal -- which is `carveOut`'s existing     #
#  issuer-fingerprint idea, and belongs there rather than here.                    #
# =========================================================================== #
def isin_same_issuer_groups(symbols, names, isins):
    """(duplicate_isin_groups, same_name_different_isin_groups).

    NOT CALLED BY THE PIPELINE.  Present so that acting on the finding above is a
    wiring change with the logic already written and tested, not a re-derivation.

    duplicate_isin        -- {isin: [symbols]} where one ISIN carries >1 symbol. These
                             are literal duplicate listings; collapsing them is safe.
    same_name_different_isin -- {normalised_name: [(symbol, isin)]} where one issuer
                             name spans >1 ISIN. These are CANDIDATES ONLY: a genuine
                             parent/subsidiary pair (Heineken N.V. vs Heineken Holding
                             N.V.) looks identical to a common/preferred pair here, so
                             this MUST be confirmed against fundamentals before any
                             removal. Deleting a common is the expensive error.
    """
    try:
        import carveOut as _co
        norm = _co._norm_issuer_name
    except Exception:
        norm = lambda x: (x or '').strip().lower()

    by_isin, by_name = {}, {}
    for sym, nm, isin in zip(symbols, names, isins):
        if not isinstance(sym, str) or not sym:
            continue
        if isinstance(isin, str) and isin.strip():
            by_isin.setdefault(isin.strip(), []).append(sym)
        n = norm(nm)
        if n:
            by_name.setdefault(n, []).append((sym, isin))

    dup_isin = {k: sorted(v) for k, v in by_isin.items() if len(v) > 1}
    multi = {}
    for n, pairs in by_name.items():
        distinct = {i for _s, i in pairs if isinstance(i, str) and i.strip()}
        if len(distinct) > 1:
            multi[n] = sorted(pairs)
    return dup_isin, multi


#  The four symbols the detector was verified to flag on the live 2026-08-02 list.
#  Recorded as data so the finding is testable and cannot decay into a comment.
ISIN_DETECTOR_VERIFIED_FINDINGS = {
    'duplicate_isin': (('HAFNI.OL', 'HAFNIO.OL'), ('ALCAT.PA', 'CATG.PA')),
    'non_common_admitted': (('CBE.PA', 'RBT.PA', -17.9), ('PREVA.AS', 'VALUE.AS', -29.9)),
    'known_false_positive': (('HEIA.AS', 'HEIO.AS'),),
}

#  STATUS UPDATE (2026-08-05). The note above ends "a safe version needs a fundamentals
#  check on top of the ISIN signal -- which is carveOut's existing issuer-fingerprint
#  idea, and belongs there rather than here". That is now BUILT: carveOut's K1/K2/K3
#  keys group all four `duplicate_isin` / `non_common_admitted` symbols by FUNDAMENTALS,
#  with no ISIN and no name-similarity rule, and carveOut's canonicity ordering picks the
#  common. So this detector is superseded for the cases it was written to find.
#
#  TWO THINGS DO NOT CHANGE. (1) `same_name_different_isin` STAYS UNWIRED -- it has a
#  real false positive on Heineken and the fundamentals keys do its job with better
#  precision; carveOut separates HEIA.AS from HEIO.AS on netIncome and marketCap, which
#  no name/ISIN rule can. (2) EXACT-ISIN equality remains the natural fourth grouping
#  key (K4) and would be cheap belt-and-braces for HAFNI/HAFNIO and CATG/ALCAT -- it is
#  not wired only because profile ISIN is fetched but never plumbed into `cdx_df`.

# =========================================================================== #
#  KNOWN GAPS IN `filter_non_common_instruments` -- PRE-EXISTING, NOT INTRODUCED  #
#  BY THE UNIVERSE WORK.  Found 2026-08-03 while auditing the curated test        #
#  universe, which is precisely what a reference list is for.  Recorded as data    #
#  so they are testable and cannot decay into prose; NOT fixed here, because       #
#  widening a removal rule changes the production universe (a product decision)    #
#  and because deleting a common is the expensive error.                          #
#                                                                               #
#  Each of these SURVIVES in production today and is scored on its sibling         #
#  common's fundamentals -- the same defect class as the Korean preferreds.        #
# =========================================================================== #
SHARE_CLASS_FILTER_KNOWN_GAPS = {
    # rule C's tail whitelist is `^(P[A-Z]?|[A-Z]?[RUWZ]|[PRUWZ][A-Z])$`. A bare
    # single letter outside RUWZ is not admitted, so a `<common>+D` / `+L` preferred or
    # notes line is invisible even though the sibling common IS in the pool.
    'unwhitelisted-single-letter-tail': (
        ('WHLRD', 'WHLR', 'Wheeler REIT Series D preferred; tail "D" not whitelisted'),
        ('WHLRL', 'WHLR', 'Wheeler REIT notes line; tail "L" not whitelisted'),
    ),
    # FMP TRUNCATES some company names ("Babcock & Wilcox Enterprises, I"), so the
    # normalised name does NOT match the sibling's and the line falls out of rule C's
    # (name, exchange) group ENTIRELY -- no tail rule can help. This one defeats
    # name-based grouping itself, which is why an audit keyed on names missed it too.
    'truncated-company-name': (
        ('BWNB', 'BW', 'Babcock & Wilcox senior notes; FMP name truncated to '
                       '"Babcock & Wilcox Enterprises, I" so it never joins BW\'s group'),
    ),
    # Continental conventions -- see the ISIN detector note above for the measured
    # discounts. Neither rule A (English vocabulary) nor rule B (`-P<letters>`) nor
    # rule C (shared symbol prefix) can reach these.
    'continental-convention': (
        ('CBE.PA', 'RBT.PA', "Robertet certificat d'investissement, -17.9% vs the common"),
        ('PREVA.AS', 'VALUE.AS', 'Value8 cumulative preference shares, -29.9%'),
    ),
}


# =========================================================================== #
#  KOREA -- THE DEDUP DEPENDENCY, EXPRESSED IN CODE                             #
#                                                                               #
#  Korea is admissible in a universe ONLY because canonical-choice issuer dedup   #
#  exists (carveOut, 2026-08-05). 196 preferred symbols across 91 families share   #
#  their common's numeric root AND its company name verbatim and trade at 30-60%   #
#  discounts, so on a cheapness screen they rank STRAIGHT TO THE TOP -- they look   #
#  like the same company at half price, because that is exactly what the data says. #
#  The old share-class filter caught 1 of 196 (rule B keys on a `-P<letters>`       #
#  suffix; the Korean convention is a suffix ON THE ROOT, which no rule saw).       #
#                                                                               #
#  So the dependency is not a comment, it is a GATE: a universe containing KSC or   #
#  KOE does not resolve unless the Korean canonicity marker is present and correct.  #
#  Nobody can enable Korea by editing `universes.py` alone.                          #
#                                                                               #
#  WHAT THE GATE CAN AND CANNOT PROVE -- stated plainly, because overclaiming here   #
#  is exactly the failure the register keeps booking. It runs PRE-FETCH, where no     #
#  statements exist yet, so it proves the PICKING half only:                          #
#    PROVEN pre-fetch  -- the marker demotes every non-`...0` line and demotes NO      #
#                         `...0` common (a pure function of symbol + name), and the     #
#                         live list still has the family shape the marker assumes.      #
#    NOT PROVEN         -- that FMP actually serves a Korean preferred its issuer's      #
#                         statements, i.e. that K1/K3 GROUP the family at all. If they   #
#                         do not, each preferred survives as its own singleton issuer     #
#                         and the marker never gets a sibling to prefer.                  #
#  That second half is a POST-FETCH regression (test_dedup_issuer.py, the Korea gate)     #
#  and it is UNVERIFIED until a real Korea fetch exists. The gate says so out loud.       #
# =========================================================================== #
KOREA_EXCHANGE_CODES = ('KSC', 'KOE')

#  Live-verified 2026-08-04 on `available_traded_raw_2026-08-04.pickle` (51,703
#  type=='stock' lines): (common, *preferred_lines) per family. Recorded as DATA so the
#  finding is testable and cannot decay into prose, exactly like
#  ISIN_DETECTOR_VERIFIED_FINDINGS. Chosen to cover every SHAPE, not just the common one:
#    005930/005935  the textbook numeric preferred (Samsung Electronics, ~29% discount)
#    005380/5/7/9   a family with THREE preferred lines (Hyundai Motor)
#    028260/02826K  the "new-type" preferred whose 6th character is a LETTER -- written
#                   as `\d` the marker misses this shape (15 K + 2 L lines live)
#    336370/K/L     both letter codes in one family (Solus Advanced Materials)
#    009830/009835  the ONE family of 91 whose members do NOT share a name; it is caught
#                   by the instrument-name vocabulary instead ("Pfd Registered Shs")
KOREA_PREFERRED_VERIFIED_FAMILIES = (
    ('005930.KS', '005935.KS'),
    ('005380.KS', '005385.KS', '005387.KS', '005389.KS'),
    ('028260.KS', '02826K.KS'),
    ('336370.KS', '33637K.KS', '33637L.KS'),
    ('009830.KS', '009835.KS'),
)

#  Measured on the same list: Korean families, and how many members each shape has.
KOREA_FAMILY_FACTS = {
    'korean_stock_lines': 1276,
    'multi_line_families': 91,
    'symbols_in_those_families': 196,
    #  How many families have ALL members under ONE normalised name -- i.e. how many the
    #  K3 (name + shares) key can group on names alone. TWO numbers, because the answer
    #  depends on which normaliser you use, and K3 uses keep_holding=True:
    #    89 under keep_holding=True  (what K3 actually sees)
    #    90 under the default        (`Holding` stripped)
    #  The extra split is AMOREPACIFIC: FMP names 002790/002795 "AMOREPACIFIC Holdings
    #  Corp." and the new-type preferred 00279K "AMOREPACIFIC Group", so preserving
    #  `Holding` (the Heineken guard) costs that family its K3 name edge. It is the same
    #  issuer, so 00279K must then be grouped by K1 (statements) or it survives as a
    #  sibling-less preferred -- one of the 91 families whose collapse rests on K1 alone.
    'families_with_one_shared_name_K3': 89,
    'families_with_one_shared_name_default_norm': 90,   # exception: 009830/009835
    'families_containing_a_common': 91,        # every family has its `...0` line
    'sixth_char_counts': {'0': 91, '5': 78, '7': 9, '9': 1, 'K': 15, 'L': 2},
}


def _korean_families(symbols):
    """{(root5, suffix): [symbols]} for 6-character Korean bases -- families only."""
    fam = {}
    for s in symbols:
        if not isinstance(s, str) or '.' not in s:
            continue
        base, suf = s.rsplit('.', 1)
        if suf in ('KS', 'KQ') and len(base) == 6:
            fam.setdefault((base[:5], suf), []).append(s)
    return {k: v for k, v in fam.items() if len(v) > 1}


def assert_korea_dedup_ready(tickdf, tfilt, verbose=True):
    """HARD GATE: raise unless the Korean canonicity rule is present AND correct.

    Called from `tickerfilterWrapper` for any universe wiring KSC/KOE, BEFORE the fetch
    is spent.  Costs zero API calls -- it reads the marker (a pure function) and the
    pre-filter table that is already in memory.
    """
    import carveOut as _co
    tag = getattr(_co, '_non_canonical_tag', None)
    if tag is None:
        raise Exception(
            'UNIVERSE %s includes Korea (%s) but carveOut._non_canonical_tag is MISSING. '
            'Korean preferred lines share their common\'s numeric root and name and trade '
            '30-60%% below it, so without the canonicity marker they rank to the top of '
            'the cheapness screen. Do not fetch Korea. See getData_gen.'
            'KOREA_EXCHANGE_CODES.' % (tfilt, '/'.join(KOREA_EXCHANGE_CODES)))

    # 1. THE MARKER ITSELF, against recorded live-verified families. Pure symbol+name
    #    logic, so this is a real assertion with no data dependency at all.
    names = {}
    if tickdf is not None and 'symbol' in getattr(tickdf, 'columns', []) \
            and 'name' in tickdf.columns:
        names = dict(zip(tickdf['symbol'].astype(str), tickdf['name'].astype(str)))
    problems = []
    for fam in KOREA_PREFERRED_VERIFIED_FAMILIES:
        common, prefs = fam[0], fam[1:]
        grp = list(fam)
        if tag(common, names.get(common, ''), grp):
            problems.append('%s (the COMMON) is marked non-canonical' % common)
        for p in prefs:
            if not tag(p, names.get(p, ''), grp):
                problems.append('%s (a PREFERRED) is NOT marked non-canonical' % p)
    if problems:
        raise Exception(
            'UNIVERSE %s includes Korea but the canonicity marker FAILED its recorded '
            'families: %s. This is the exact defect that makes Korean preferreds rank '
            'first on a cheapness screen. Fix carveOut._non_canonical_tag before '
            'fetching Korea.' % (tfilt, '; '.join(problems)))

    # 2. THE LIVE LIST still has the shape the marker assumes: in every Korean family,
    #    EXACTLY ONE member survives as canonical and it is the `...0` common. This is
    #    what catches FMP renaming or renumbering a venue under us -- the failure mode
    #    that hid the dead EURONEXT/OSE codes for the life of the project.
    fams = _korean_families(list(names) if names else [])
    bad = []
    for key, members in fams.items():
        canon = [s for s in members if not tag(s, names.get(s, ''), members)]
        zeros = [s for s in members if s.rsplit('.', 1)[0][5] == '0']
        if canon != zeros or len(canon) != 1:
            bad.append('%s%s: canonical=%s, `...0`=%s'
                       % (key[0], key[1], canon or 'NONE', zeros or 'NONE'))
    if bad:
        raise Exception(
            'UNIVERSE %s includes Korea but %d of %d Korean families on the LIVE list do '
            'not resolve to exactly one canonical `...0` common: %s. The convention the '
            'marker encodes has moved; re-verify before fetching Korea.'
            % (tfilt, len(bad), len(fams), '; '.join(bad[:5])))

    if verbose:
        print('KOREA DEDUP GATE: PASSED the PICKING half -- carveOut._non_canonical_tag '
              'demotes every preferred and no common across %d recorded families and all '
              '%d Korean families on this list.' % (
                  len(KOREA_PREFERRED_VERIFIED_FAMILIES), len(fams)), flush=True)
        print('  !!! THE GROUPING HALF IS NOT PROVEN HERE AND CANNOT BE: whether FMP '
              'serves a Korean preferred its ISSUER\'S statements (so K1/K3 group the '
              'family at all) needs STATEMENTS, which do not exist pre-fetch. If they do '
              'not, each preferred survives as its own singleton and the marker never '
              'gets a sibling to prefer. Run the Korea MUST-MERGE regression '
              '(test_dedup_issuer.py) against this fetch\'s panel before trusting any '
              'Korean name in the ranking.', flush=True)
    return True


def _apply_issuer_sample(df, tfilt):
    """Keep only rows whose ISSUER NAME falls in the universe's per-exchange sample.

    A no-op (returns `df` unchanged) for every universe with no `sample` key, which is
    every universe except stock_CUR3K -- so this cannot alter an existing scope.

    SAMPLED ON THE NORMALISED ISSUER NAME, NOT ON THE SYMBOL, and that is the whole
    reason the function exists rather than a one-line `.sample(frac=)`.  carveOut's dedup
    edges are PAIRWISE OVER THE POOL, so a per-symbol sample lands UHS without 0LJL.L and
    the pool then holds one line of a two-line issuer -- which is not a smaller test of
    dedup, it is no test of dedup.  It uses `carveOut._norm_issuer_name` -- the pipeline's
    OWN normaliser -- so the sample and the dedup cannot drift onto two different notions
    of "same issuer".

    *** THE NAME HASH ALONE DOES NOT CLOSE A GROUP, AND THE FIRST CUT OF THIS FUNCTION
    ASSUMED IT DID.  Fixed 2026-08-06 after review. ***  The bucket is a function of the
    name, but the THRESHOLD was read PER ROW off that row's exchange, so an issuer with
    lines on an unrated venue (TSX/PAR/AMS, take-all) and a rated one (NYSE/NASDAQ/LSE at
    170) SPLIT whenever its bucket was >= 170 -- 83% of the time.  Measured on the live
    2026-08-04 table: 218 groups split, 209 whole, and five PINNED groups arrived
    incomplete (the pins are by symbol, not by group).  So the rate is now resolved ONCE
    PER ISSUER, against the MOST PERMISSIVE rate among the venues that issuer's lines
    occupy -- `universes.most_permissive_rate`, where the full measurement lives.

    THE RESOLUTION IS POOL-RELATIVE, and the caveat is real: the venues considered are the
    ones this issuer occupies IN THE ALREADY-EXCHANGE-FILTERED FRAME.  A line on a venue
    the universe does not wire at all cannot pull the group in, because it is not there to
    be seen -- so closure is closure WITHIN the universe's exchange set, which is the only
    closure that matters to a pairwise dedup over this pool.

    THE OTHER LIMIT, STATED: the pipeline dedups on a FUNDAMENTALS FINGERPRINT, not on the
    name, so a sibling FMP names differently (BWNB, truncated to "Babcock & Wilcox
    Enterprises, I") is NOT held together by this and can still arrive split.  Closure is
    by construction for SAME-NAMED lines only.

    Rows with no usable name are KEPT, never dropped: an unnamed row cannot be shown to be
    outside the sample, and dropping on missing metadata is how a filter silently shrinks
    a universe (the EURONEXT lesson).
    """
    import universes as un
    rates = un.sample_rates(tfilt)
    if not rates or not len(df):
        return df
    bad = un.check_sample_rates(tfilt)
    if bad:
        # Dead configuration that reads as though it were doing something -- the
        # EURONEXT/OSE defect class. Raise rather than warn: it is a registry typo, and
        # the run costs hours.
        raise Exception(
            'universe %r samples exchange code(s) %s that it does not wire (%s). A rate '
            'keyed on an unwired code is dead configuration -- fix universes.UNIVERSES.'
            % (tfilt, bad, ' '.join(un.exchanges(tfilt) or ())))
    import carveOut as _co
    norm = df['name'].map(_co._norm_issuer_name) if 'name' in df.columns else None
    if norm is None:
        raise Exception(
            'universe %r samples on the issuer NAME but the pre-filter table has no '
            '`name` column -- refusing to fall back to a per-symbol sample, which would '
            'split every cross-listing group and silently stop testing dedup.' % tfilt)
    # PASS 1 -- collect the venues each issuer NAME occupies, so its rate can be resolved
    # once for the whole group instead of once per row.
    venues = {}
    for code, nm in zip(df['exchangeShortName'], norm):
        if nm:
            venues.setdefault(nm, set()).add(code)
    # PASS 2 -- one decision per issuer, applied to every one of its rows.
    decision = {nm: un.issuer_in_sample(nm, un.most_permissive_rate(cs, rates))
                for nm, cs in venues.items()}
    keep = [True if not nm else decision[nm] for nm in norm]   # unnamed -> keep
    out = df[pd.Series(keep, index=df.index)].reset_index(drop=True)
    by_code = out['exchangeShortName'].value_counts().to_dict()
    # The cross-rate count is REPORTED because it is the number the first cut got wrong:
    # an operator can see from the banner that upward closure actually fired.
    cross = sum(1 for cs in venues.values()
                if len({rates.get(c) for c in cs}) > 1)
    print('UNIVERSE %s: issuer-name sample kept %d of %d exchange-filtered row(s) -- %s '
          '(%d issuer name(s), %d of them straddling two sample rates and therefore '
          'resolved at the most permissive one)'
          % (tfilt, len(out), len(df),
             ' '.join('%s=%d' % (c, by_code.get(c, 0))
                      for c in sorted(set(df['exchangeShortName']))),
             len(venues), cross), flush=True)
    return out


def tickerfilterWrapper(tickdf,tfilt,sfilt,mcapf,baseurl,api_key):
    """Apply a named universe scope from `universes.UNIVERSES` to the pre-filter table.

    WAS A SIX-BRANCH if/elif CHAIN WITH THE CODES INLINE (rewritten 2026-08-02).  Three
    things were wrong with that shape and all three are structural, not cosmetic:

      1. THE CODES WERE WRONG AND NOTHING COULD NOTICE.  Four of the six branches
         filtered on `'EURONEXT'` and two on `'OSE'`.  FMP serves neither string, so
         those entries matched ZERO rows and 1,046 statement-bearing common stocks
         (Paris 577, Oslo 224, Brussels 107, Amsterdam 103, Lisbon 35) were absent from
         every run ever made -- silently, because a filter that matches nothing looks
         exactly like a filter whose exchange happens to be small.  `stock_US1_EU2` was
         reduced to US-only.  The codes now live in `universes.py` beside the LIVE
         counts they were verified against, so "does this code match anything" is a
         checked fact rather than an assumption.

      2. AN UNKNOWN NAME RETURNED THE WHOLE WORLD.  `df = tickdf` was the initial value
         and the chain had no `else`, so any tfilt the chain did not recognise fell
         through and returned the ENTIRE ~50,000-name pre-filter table -- unfiltered,
         un-warned, and 5x the intended fetch.  `configuration` validated the name
         against its OWN separate list, so the two lists drifting was the only thing
         standing between the operator and that outcome.  Both now read one registry,
         and an unknown name RAISES.

      3. `if` / `if` / `if`+`elif` MIXED.  The first three branches were bare `if`s and
         only the last four formed a chain, so the code read as though the branches were
         exclusive when two of them were not chained at all.  Harmless only because the
         values happened to be distinct.

    Stage order is type -> INSTRUMENT FILTER (on the full table) -> membership ->
    sector filter -> delisted prune.  The instrument filter MOVED AHEAD of membership
    selection on 2026-08-02; see the note at its call site for why that makes
    "a subset is filtered as well as the whole" true by construction.
    """
    import universes as un

    # RESOLVE THE NAME FIRST, before any filtering or API work: an unknown universe must
    # fail immediately, not after `filter_tickers` has already run (and, with a positive
    # mcap floor, already spent per-symbol profile calls).
    explicit = un.symbols(tfilt)          # raises on an unknown name
    every = un.is_every_exchange(tfilt)
    codes = un.exchanges(tfilt)

    # KOREA GATE, BEFORE ANY WORK IS SPENT. A universe that wires KSC/KOE does not
    # resolve unless the issuer-dedup canonicity marker is present and correct -- see
    # assert_korea_dedup_ready for exactly what that does and does not prove. Placed
    # here, not in universes.py, on purpose: this is where membership is APPLIED, so the
    # dependency cannot be edited around by adding an entry to the registry.
    if codes and any(c in KOREA_EXCHANGE_CODES for c in codes):
        assert_korea_dedup_ready(tickdf, tfilt)
    elif every:
        # stock_FULL1 applies no exchange filter, so it contains Korea by construction.
        # It is the CEO's deliberate "everything" option and is documented as a hazard
        # rather than fenced (universes.stock_FULL1), so the gate WARNS here instead of
        # raising -- but it still has to run, because the Korean preferreds are in it.
        try:
            assert_korea_dedup_ready(tickdf, tfilt)
        except Exception as _ke:
            print('!!! UNIVERSE %s contains Korea (no exchange filter) and the Korea '
                  'dedup gate FAILED: %s\n!!! Korean preferred lines will rank to the top '
                  'of the cheapness screen. Proceeding because FULL is an explicit CEO '
                  'option, NOT because this is safe.' % (tfilt, _ke), flush=True)

    tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)

    # ------------------------------------------------------------------------------ #
    #  INSTRUMENT FILTER FIRST, ON THE FULL TABLE, THEN INTERSECT (moved 2026-08-02).  #
    #                                                                                #
    #  It used to run AFTER membership selection, on the already-narrowed frame. That  #
    #  made its completeness depend on WHICH UNIVERSE WAS ACTIVE, because rule C is    #
    #  PAIRWISE: it recognises an instrument line only by comparing it to a shorter     #
    #  same-name, same-EXCHANGE-SUFFIX sibling. Suffix maps 1:1 to exchange code with   #
    #  one big exception -- every US ticker has NO suffix, so NYSE, NASDAQ, AMEX, OTC    #
    #  and PNK all share the empty one. An enabling sibling can therefore sit on an      #
    #  exchange the active universe excludes, and the instrument line then survives.     #
    #                                                                                #
    #  MEASURED on the live 2026-08-02 table: filtering the full table removes ZCARW    #
    #  (a Zoomcar warrant on NASDAQ) from every US-containing universe, because its     #
    #  enabling common ZCAR trades on OTC, which no universe wires. Exactly one name,   #
    #  and it is a genuine warrant -- so this is a strict improvement, not a rebalance:  #
    #  nothing that used to be kept is newly dropped except that warrant, and nothing   #
    #  that used to be dropped is newly kept (verified for all 8 universes).            #
    #                                                                                #
    #  Filtering first also retires the risk BY CONSTRUCTION rather than by luck: the   #
    #  removal set no longer depends on the active universe at all, so no future subset  #
    #  can be less well filtered than the whole. That property is what matters here --   #
    #  the curated test universe exists to behave like production, and a filter whose    #
    #  strength varied with pool size would have quietly broken that.                    #
    # ------------------------------------------------------------------------------ #
    tickers_df_stock = filter_non_common_instruments(tickers_df_stock)

    if explicit is not None:
        # EXPLICIT-MEMBERSHIP universe (the curated test universe).  Selected by
        # symbol, NOT by exchange, and deliberately NOT by `-nrTaT N`: nrTaT keeps the
        # first N rows of available-traded/list, i.e. an arbitrary positional prefix
        # that under-represents semi-annual filers, non-USD reporters and whole
        # cohorts.  A frozen list is the only way two iterations are comparable.
        want = list(explicit)
        df = tickers_df_stock[tickers_df_stock['symbol'].isin(want)]
        absent = sorted(set(want) - set(df['symbol']))
        if absent:
            # NOT an error: a member can legitimately leave the live list (delisted,
            # moved to OTC, renamed).  But it must be VISIBLE, because a curated
            # universe that quietly shrinks stops covering the categories it was built
            # to cover -- and the coverage claim is the entire value of the list.
            shown = absent[:25]
            print('UNIVERSE %s: %d of %d listed member(s) absent from the live '
                  'type==stock list -- %s%s'
                  % (tfilt, len(absent), len(want), ', '.join(shown),
                     ' (+%d more)' % (len(absent) - len(shown)) if len(absent) > len(shown)
                     else ''), flush=True)
            # A handful is attrition; a large fraction means the list no longer matches
            # what FMP serves, and the run would silently be a different experiment.
            if len(absent) > 0.1 * len(want):
                print('!!! UNIVERSE %s: %.0f%% of the curated list is missing from the '
                      'live universe -- re-curate before trusting this run\'s coverage '
                      '(run `python verify_test_universe.py`).'
                      % (tfilt, 100.0 * len(absent) / len(want)), flush=True)
    elif every:
        # FULL universe: no exchange filter at all, by definition.
        df = tickers_df_stock
    else:
        df = filter_tickers(tickers_df_stock, 'exchangeShortName', list(codes),
                            mcapf, api_key)
        df = _apply_issuer_sample(df, tfilt)

    # ------------------------------------------------------------------------------ #
    #  MUST-INCLUDE UNION, AFTER the base rule and OUTSIDE its exchanges.             #
    #                                                                                #
    #  Placed here rather than inside the branch above so it applies to whatever      #
    #  membership rule ran, and drawn from `tickers_df_stock` -- the FULL              #
    #  post-instrument-filter table -- not from `df`.  Two consequences, both wanted:   #
    #    * a pinned case on an exchange the universe does not wire (EMBELL.ST on STO,   #
    #      EIN.DE / DRW3.DE on XETRA) still arrives, so observing it costs 1 name        #
    #      instead of the 1,375 that wiring STO+XETRA would cost;                       #
    #    * a pinned symbol the INSTRUMENT FILTER removes stays removed.  That is not a   #
    #      leak in the pin, it is the filter doing its job on the one code path (see the #
    #      note above), and it is REPORTED below rather than silently absorbed.          #
    #                                                                                #
    #  AND THE PINS ARE GROUP-CLOSED, ADDED 2026-08-06 (review).  A pin names a SYMBOL,  #
    #  but every case it exists to observe is about a GROUP -- so a pin that arrives      #
    #  without its siblings is a pin that observes nothing.  Closing the base rule        #
    #  upward fixes the pinned groups that STRADDLE two rates (robertet, peyto), but NOT   #
    #  the ones whose lines share one rate and fall out together: `t rowe price`           #
    #  (0KNY.L pinned, TROW not), `sk hynix` (SKHY, no 000660.KS), `f g annuities life`     #
    #  (FG, no FGN).  Those arrived as SINGLETONS -- and a singleton IOB control is not a   #
    #  control.  So each pinned symbol also pulls its SAME-NAME siblings.                  #
    #                                                                                #
    #  RESTRICTED TO THE UNIVERSE'S OWN EXCHANGES, deliberately: an unwired venue is       #
    #  reachable for an EXPLICIT pin (that is the point above) but must not be reachable   #
    #  by implication, or one pinned Samsung line would drag in every Samsung listing FMP   #
    #  serves on a dozen venues. Only applied where a `sample` exists, so no other          #
    #  universe's membership moves by a single row.                                        #
    # ------------------------------------------------------------------------------ #
    pinned = un.must_include(tfilt)
    if pinned:
        want = set(pinned)
        siblings = set()
        if un.sample_rates(tfilt) and 'name' in tickers_df_stock.columns:
            import carveOut as _co
            _norm = tickers_df_stock['name'].map(_co._norm_issuer_name)
            pin_names = {n for n in _norm[tickers_df_stock['symbol'].isin(list(pinned))]
                         if n}
            siblings = set(tickers_df_stock['symbol'][
                _norm.isin(pin_names)
                & tickers_df_stock['exchangeShortName'].isin(list(codes))]) - want
            want |= siblings
        extra = tickers_df_stock[
            tickers_df_stock['symbol'].isin(list(want))
            & ~tickers_df_stock['symbol'].isin(set(df['symbol']))]
        if siblings:
            print('UNIVERSE %s: must-include GROUP CLOSURE -- %d same-issuer sibling(s) of '
                  'a pinned symbol pulled in so no pinned group arrives as a singleton: %s'
                  % (tfilt, len(siblings), ', '.join(sorted(siblings))), flush=True)
        absent = sorted(set(pinned) - set(df['symbol']) - set(extra['symbol']))
        added_pins = set(extra['symbol']) & set(pinned)
        if len(extra):
            df = pd.concat([df, extra], ignore_index=True)
        print('UNIVERSE %s: must-include -- %d of %d pinned symbol(s) added by the union '
              '(%d already selected by the base rule), plus %d sibling(s); %d row(s) total'
              % (tfilt, len(added_pins), len(pinned),
                 len(pinned) - len(added_pins) - len(absent),
                 len(extra) - len(added_pins), len(extra)), flush=True)
        if absent:
            # LOUD, because the entire value of a pinned case is that it is OBSERVED.
            # A pin can legitimately vanish (delisted, renamed, moved venue) or be
            # removed by the instrument filter -- both are findings, neither is silent.
            print('!!! UNIVERSE %s: %d pinned symbol(s) did NOT resolve -- %s. Each one is '
                  'a case this run was built to OBSERVE, so the observation is MISSING, '
                  'not merely absent: check whether it left FMP\'s list or was removed by '
                  'filter_non_common_instruments before reading the dedup results.'
                  % (tfilt, len(absent), ', '.join(absent)), flush=True)

    # The debt/preferred/warrant/rights filter (audit M-5) has ALREADY run, above, on the
    # FULL type=='stock' table -- see the note there. It used to run HERE, after the
    # membership filters, which is what made its completeness depend on the active
    # universe. It still applies to every tfilt branch on one code path, and nothing
    # downstream -- the Stage-2 z-pool, the mean-relative Stage-1 baselines, the carve
    # dedup -- ever sees an instrument line.

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

#  How many of the MOST RECENT periods must carry a price for a ticker to be scoreable.
PRICE_GATE_ROWS = 10


def checkIfValidFS(fs):
    """The ingest PRICE GATE: reject a ticker with any missing price in its most recent
    PRICE_GATE_ROWS periods.  A ticker failing here lands in `pricefail` and never reaches
    scoring (getData_fmp.get_fundamentals_fmp; ~523 names on the 07-17 universe, ~72% non-US).

    ROW ORDER IS LOAD-BEARING AND IS *ASSUMED*, NOT ESTABLISHED HERE (documented 2026-08-02).
    `fs[...][0:PRICE_GATE_ROWS]` is a POSITIONAL slice, so it means "the ten most recent
    periods" ONLY because `fs` is the per-ticker frame straight off the FMP statements, which
    arrive NEWEST-FIRST.  If that frame ever arrived oldest-first -- or were re-sorted by the
    caller -- this gate would silently police the ten OLDEST periods instead, i.e. it would
    admit a ticker whose recent prices are all missing and reject one whose ancient prices
    are.  That is the same defect shape as moatIdentifier's head(n) on oldest-first rows.
    The order is NOT normalised here deliberately: re-sorting would change WHICH rows the gate
    reads on any frame that is not already newest-first, and therefore which tickers enter the
    universe -- a behaviour change, not a refactor.  Stated as an explicit precondition so the
    assumption is visible at the boundary instead of being rediscovered; use
    reporting_period.assert_newest_first at the call site if it ever needs observing.
    """
    retbool = True
    if any(fs['price'][0:PRICE_GATE_ROWS].isna()):
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
    #   reportingFrequencyConflict  'by_period|by_cadence' when the two frequency signals
    #                     disagree for this source, else '' (reporting_period
    #                     .FREQ_CONFLICT_COLUMN).  It MUST reach the saved panel: postBo's
    #                     universe-wide conflict banner decodes it, because
    #                     frequency_by_source short-circuits on the stored verdict and so
    #                     cannot re-detect the conflict itself.  Coerced to NaN here, the
    #                     watchdog goes dark again -- exactly the failure it was just fixed
    #                     for -- which is why it is listed rather than left to chance.
    for _passthrough in ('reportedCurrency', 'period', 'fillingDate', 'acceptedDate',
                         'periodEndDate', 'grahamUndefinedReason',
                         'reportingFrequency', 'reportingFrequencyConflict'):
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