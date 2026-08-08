"""fx_rates.py -- the LIVE, DATED, FAIL-LOUD FX feed for the USD conversions.

WHY THIS MODULE EXISTS
----------------------
`carveOut.FX_TO_USD` was a HARDCODED, UNDATED snapshot carrying its own TODO admitting
it.  Measured against live rates on 2026-08-08 the median absolute drift was ~7% and
THIRTEEN currencies were past 10% (TRY -30.1%, ILS +23.5%, ZAR +14.1%, HUF +13.8%,
NOK +13.0%, INR -12.5%, ISK +12.5%, MYR +11.2%, IDR -11.1%, SEK +11.1%, RUB +11.0%,
CZK +11.0%, CHF +10.5%); the majors were EUR 1.08 -> 1.1562 (+7.1%), GBP 1.27 -> 1.34839
(+6.2%), CAD 0.73 -> 0.717 (-1.8%), KRW 0.00073 -> 0.000706 (-3.3%).

Recomputed on the 2026-08-07 CUR3K panel, the stale table gets **11 universe-membership
decisions wrong at the $25M floor** -- 7 EUR names WRONGLY DELETED (0GJS.L, 0O0E.L,
ALBIZ.PA, ALU10.PA, ALXIL.PA, AMUND.AS, BIG.PA) and 4 wrongly kept (014910.KS, 014915.KS,
075130.KQ, TGO.TO) -- plus 32 names put in the wrong market-cap band.

SCOPE OF THAT NUMBER, which must not be over-read: the floor gates on
`carveOut.currency_data_present` and is a complete NO-OP wherever `reportedCurrency` does
not resolve.  So the 11 wrong decisions are real for POST-FETCH panels (the ones that
carry `reportedCurrency`), NOT for every historical run -- on a pre-2026-07 pickle the
floor excludes nobody at all and the FX table decides nothing.

THE DESIGN, AND THE TRAP THAT DICTATED IT
-----------------------------------------
`v3/quotes/forex` returns all 1,550 pairs in ONE response (~863 KB) -- and it MIXES LIVE
AND DEAD PAIRS.  Probed 2026-08-08: 125 of the 1,550 carried a `timestamp` older than 30
days, the oldest ~1,316 days, EVERY ONE OF THEM SERVED UNDER AN HTTP 200.  A response-level
"did it work" check is therefore worthless.  Freshness is checked **PER RATE**, never per
response.  (Ours all read 0.1-0.2 days old at probe time, so the 7-day threshold is not a
tight fit around observed noise -- it is two orders of magnitude of headroom over what a
healthy pair looks like, and still four times tighter than the nearest dead pair.)

FIVE RULES, in the order they bite:

  1. ONE `v3/quotes/forex` call per run.  A run is ~16,300 calls, so this is ~0.006% of it.
     A per-pair spot fetch would cost 36x and buy nothing.
  2. A rate whose quote timestamp is older than FX_STALE_MAX_DAYS is treated as **ABSENT**.
     Never substituted, never patched from the constants.
  3. A rate that is not within +-FX_SANITY_BAND of its FX_TO_USD constant is REFUSED (also
     -> absent).  This is the new failure mode a live feed has that a constant does not: a
     vendor-side unit flip or a decimal slip.  It is a UNITS check, not an accuracy check --
     TRY is already 30% from its constant, so a tighter band would reject good data.
  4. A rate whose RECIPROCAL pair disagrees (`{CUR}USD * USD{CUR}` not ~= 1) is REFUSED --
     and so is one whose reciprocal is MISSING or STALE, i.e. one the check cannot run on
     at all.  Rule 3 structurally CANNOT catch an inverted quote on a near-parity currency
     (an inverted EUR reads as 0.80x its constant, comfortably inside the band), so an
     unverifiable rate is not accepted on trust: refusing costs a KEPT name, accepting a
     bad one costs a DELETED one.  See FX_RECIPROCAL_TOL.
  5. Missing / stale / refused all route into the code path that ALREADY EXISTS for an
     unresolvable `reportedCurrency`: `carveOut._fx_to_usd` returns None -> the USD cap is
     NaN -> the name is **KEPT** by the $25M floor and routed to General -> the
     `!!! $25M UNIVERSE FLOOR NOT ENFORCED !!!` banner prints and the affected names land in
     `CurrencyFloorFlips_*.csv` as `kept_currency_unknown`.  **ON FX FAILURE THE FLOOR DOES
     NOT RUN ON THE OLD CONSTANTS.**  No parallel machinery is built here, because that
     path already embodies the CEO ruling recorded in `carveOut.partition_universe`'s floor
     block -- *a wrong exclusion is invisible and unrecoverable* -- and a stale rate is the
     same kind of wrong number as a missing currency.

`FX_TO_USD` therefore stops being a RATE SOURCE and becomes the SANITY BAND.  It is still
the source in ONE state only: `fx_source_state() == 'unset'`, i.e. no feed was ever
attempted -- offline tooling, the test suite, `baseline_tools/`.  The production entry point
(`Sbocker.main`) always installs, so production is only ever 'live' or 'failed'.

EVIDENCE (the 2026-08-07 post-mortem rule, in `Sbocker.transfer_outputs_to_drive`)
----------------------------------------------------------------------------------
A stale-FX run and a clean-FX run must NEVER produce identical artifacts.  Two things make
them distinguishable on the receiving machine:
  * `output/FxRates_<date>.csv` -- every supported currency with its rate, its quote
    timestamp, its age and its status.  Written to `output/` DELIBERATELY rather than being
    given its own top-level allowlist pattern: `output/` already ships whole via
    `allowlist_dirs` (see `Sbocker.transfer_outputs_to_drive`), and the last dev correctly
    declined to add a top-level glob for `DedupSurvivorReport_*.csv` for exactly this
    reason -- a directory that ships whole cannot lose evidence to a pattern that stops
    matching after a rename.
  * the `fx_rates` block in `RunProvenance-*.json`, incl. `fx_rates_as_of`.

POINT-IN-TIME FX (the backtest)
-------------------------------
Applying today's spot to a 2021 market cap is a look-ahead-flavoured error.
`v3/historical-price-full/{PAIR}` gives dated daily closes at 1 call per currency per
range; `fetch_historical_rates` pulls them into `output/FxRatesHistorical_*.csv` and
`load_pit_rates` turns that file into a table the conversion path accepts.  See
`carveOut.marketcap_usd_series(fx=...)`.

ENDPOINTS -- verified against our own key, do not re-probe for availability:
  * `GET v3/quotes/forex`                      all 1,550 pairs, one response.
  * `GET v3/historical-price-full/{PAIR}`      dated daily closes (verified to 2022).
  * `GET stable/batch-forex-quotes`            **402 -- NOT ON OUR PLAN.**  Forex is on the
    legacy v3 surface only; anything written against `stable/` will fail.
"""

import datetime as _dt
import os

import pandas as pd

import carveOut as co
import getData_gen as gdg


# --------------------------------------------------------------------------- #
#  Policy constants                                                           #
# --------------------------------------------------------------------------- #

#  A quote older than this is ABSENT, not a rate.  Every pair we use read 0.1-0.2 days
#  old when probed; 125 of the 1,550 pairs the endpoint serves are older than 30 days and
#  the oldest is ~1,316 days -- all under HTTP 200.  7 days sits between "a long weekend
#  plus a holiday" and anything that could be called live.
FX_STALE_MAX_DAYS = 7.0

#  A live rate must land within +-50% of its FX_TO_USD constant or it is REFUSED.
#  This is a UNITS check (a 100x minor-unit flip, an inverted quote, a decimal slip), not
#  an accuracy check.  TRY already sits ~30% from its constant on real data, so a tighter
#  band would start rejecting good rates; a 100x or 1/x error is out by orders of magnitude.
FX_SANITY_BAND = 0.5

#  The band is measured against a FIXED constant, so a currency in a sustained trend walks
#  toward the edge over time and is eventually refused while being perfectly correct.
#  Measured 2026-08-08, TRY is the closest at 0.6975x its constant -- about another 28% of
#  depreciation from being dropped.  A rate whose ratio is within this fraction of the
#  edge is REPORTED (still used) so the drift is seen coming rather than discovered as a
#  name silently leaving the universe.  Remedy: re-seed that constant, not widen the band.
#
#  Read as "fraction of the half-band consumed": 0.5 warns once a rate has drifted at
#  least HALFWAY to refusal, i.e. |ratio - 1| >= 0.25.  On the 2026-08-08 quote that
#  fires on exactly one currency -- TRY at 0.6975x -- and on nothing else (ILS, the
#  runner-up, sits at 1.2343x).  Tuned to be the early warning, not a second alarm.
FX_BAND_EDGE_WARN = 0.5

#  THE SECOND, INDEPENDENT UNITS CHECK -- and the one that catches what the band cannot.
#
#  The +-50% band cannot see an INVERTED quote on a near-parity currency: serving
#  USD-per-EUR as EUR-per-USD turns 1.15593 into 0.86511, which is 0.80x the 1.08 constant
#  -- comfortably INSIDE the band.  Same for GBP and CHF.  Tightening the band is not an
#  option (TRY legitimately sits at 0.70x), so the band alone leaves a real hole.
#
#  The same one response also carries the RECIPROCAL pair `USD{CUR}` for every currency we
#  use (probed 2026-08-08: present and fresh for all 35 major units), and for a correct
#  feed `{CUR}USD * USD{CUR} == 1`.  Measured across all 35: the worst deviation from 1.0
#  was 3.1e-5.  An inverted `{CUR}USD` makes the product `USD{CUR}^2` -- 0.748 for EUR --
#  which is unmissable at ANY rate, near parity included.
#
#  0.02 is ~600x the worst observed deviation: loose enough that a bid/ask or mid
#  convention difference between the two quotes can never reject a good rate, tight enough
#  that an inversion (25%+ off) cannot pass.  The uncaught band is exactly r in
#  [0.990, 1.010], where an inversion would be a <=2% error and cannot move a band edge.
#
#  A MISSING OR STALE RECIPROCAL IS A REFUSAL, NOT A SKIP (F-1, reviewer, 2026-08-08).
#  This was originally a SKIP, on the reasoning that "a missing cross-check is not evidence
#  of a bad rate".  That reasoning is fine in the abstract and WRONG here, because it
#  inverts this module's own asymmetry:
#     * losing a rate is the SAFE direction -- the name is KEPT, the floor does not apply
#       to it, the banner prints and the CSV names it;
#     * accepting a bad rate is the UNRECOVERABLE direction -- the name is DELETED from the
#       universe and no artifact says so.
#  With the skip, an inverted primary whose reciprocal happened to be absent came back
#  `status='ok', rate=0.8651` and, measured on the CUR3K panel, WRONGLY DELETED 16 NAMES
#  (0GJS.L 0O0E.L ALBIZ.PA ALBPK.PA ALDVI.PA ALGTR.PA ALHIT.PA ALHUN.PA ALPOU.PA ALPRG.PA
#  ALU10.PA ALXIL.PA AMUND.AS BIG.PA GRVO.PA POXEL.PA).  All 35 reciprocals were present
#  and fresh when probed, so refusing costs approximately nothing and buys the inversion
#  guarantee outright.  Cost of a false refusal is a kept name; cost of a false accept is a
#  deleted one.  Refuse.
#
#  RESIDUAL, stated honestly and SHARPENED by the reviewer: the check is defeated by ANY
#  PRODUCT-PRESERVING ERROR PAIR, not merely by both directions being inverted.
#  `EURUSD x1.25` together with `USDEUR /1.25` gives a product of exactly 1 and passes both
#  this check and the +-50% band, at a rate 25% high.  Nothing in a single quotes/forex
#  response can distinguish that from a real move; it would take a second, independent
#  source.  Smaller hole than the band alone, not a closed one.
FX_RECIPROCAL_TOL = 0.02

#  COVERAGE (F-2, reviewer, 2026-08-08) -- the half-degradation class, one level up.
#  The USD-alone guard protects the INSTALL boundary; it says nothing about how much of the
#  universe the surviving rates actually cover.  With only {USD, KRW} usable the run
#  installs 'live', currency_data_present is True, no banner fires and `floor_enforced:
#  True` is stamped -- while only 1,365 of 2,611 names get a USD cap.  EUR alone is 23.8%
#  of the universe.  Nothing is wrongly deleted, but a `floor_enforced: true` label on a
#  52%-floored universe is exactly the label-means-something-else defect this project keeps
#  producing.  Below these fractions the degradation is announced LOUDLY and the covered
#  fraction ships in the artifact; the run still proceeds, because per-rate absence is
#  designed behaviour and a partial floor is not a wrong floor.
FX_MIN_SUPPORTED_COVERAGE = 0.90     # of the supported currency set, checked at install
FX_MIN_PANEL_COVERAGE = 0.90         # of the panel's names, checked where the floor runs

QUOTES_ENDPOINT = 'v3/quotes/forex'
HISTORY_ENDPOINT = 'v3/historical-price-full/%s'

#  MINOR-UNIT DENOMINATIONS -- and the single nastiest trap in this file.
#  `GBp` / `GBX` are PENCE, i.e. GBP/100.  There is no `GBpUSD` pair, and an uppercased
#  lookup of 'GBp' hits **GBPUSD**, which would hand back a rate 100x too large for every
#  pence-denominated line.  So minor units are resolved EXPLICITLY, from their major unit,
#  BEFORE any pair lookup -- never by symbol matching.  (The +-50% sanity band would also
#  catch a 100x error, but a correctness rule that relies on a tripwire is not a rule.)
#
#  CORRECT BUT UNEXERCISED, and worth knowing before relying on it: the reviewer measured
#  ZERO sources reporting in `GBp` on the 2026-08-07 CUR3K panel.  So this handling is
#  right by construction and by unit test, NOT proven by any run -- do not cite the run as
#  evidence for it.  It is kept because the field is a vendor string we do not control and
#  `GBp` is what LSE-quoted lines carry when it does appear.
MINOR_UNITS = {'GBp': ('GBP', 100.0), 'GBX': ('GBP', 100.0)}

#  Currencies ADDED to the supported set with this feed (CEO 2026-08-08).  They are
#  quotable and clean on `v3/quotes/forex`, and cost nothing extra once one call fetches
#  all 1,550 pairs.  Their FX_TO_USD entries exist ONLY as sanity anchors (see module
#  docstring) and were seeded from the 2026-08-08 live quote.
ADDED_CURRENCIES = ('PEN', 'MAD')

#  ARS IS DELIBERATELY NOT SUPPORTED -- and the reason is NOT the rate.
#  `ARSUSD` quotes cleanly (0.000667 on 2026-08-08).  The problem is that our ARS
#  *STATEMENT* data is visibly broken.  Measured on BMA's panel: `totalAssets` of 2.32e16
#  sitting between 2.06e13 and 2.42e13 (a 960x break) and `revenue` of 1.2e9 sitting
#  between 1.44e12 and 1.91e12 (1,588x), while `marketCap` runs continuous through both.
#  That is IAS 29 hyperinflation-restatement noise or a straight vendor defect; either way
#  the fundamentals are wrong by three orders of magnitude in places.
#
#  Supplying a CORRECT ARS rate would make this ACTIVELY WORSE, not better: BMA would then
#  score as a confidently-measured $4.9B bank sitting on top of numbers that are wrong by
#  1000x.  An unknown currency keeps the name in the universe, scores it NEUTRAL on the size
#  metric and never lets an absolute USD edge act on it -- which is the correct treatment of
#  data we do not trust.  DO NOT "fix" this by adding 'ARS' to FX_TO_USD.
ABSTAIN_CURRENCIES = {'ARS': ('statement data broken (IAS 29 restatement noise or vendor '
                              'defect): BMA totalAssets 2.32e16 between 2.06e13 and '
                              '2.42e13 = 960x break; revenue 1.2e9 between 1.44e12 and '
                              '1.91e12 = 1,588x; marketCap continuous. A correct rate '
                              'would dress three-orders-of-magnitude-wrong fundamentals '
                              'as a confidently measured $4.9B bank.')}


# --------------------------------------------------------------------------- #
#  One resolved rate                                                          #
# --------------------------------------------------------------------------- #
class FxRate(object):
    """One currency's resolved state.  `usable` is the only thing the pipeline reads;
    everything else exists so the CSV can say WHY."""

    __slots__ = ('currency', 'pair', 'rate', 'quote_ts', 'age_days', 'status',
                 'constant', 'ratio', 'note')

    def __init__(self, currency, pair=None, rate=None, quote_ts=None, age_days=None,
                 status='missing', constant=None, ratio=None, note=''):
        self.currency = currency
        self.pair = pair
        self.rate = rate
        self.quote_ts = quote_ts
        self.age_days = age_days
        self.status = status
        self.constant = constant
        self.ratio = ratio
        self.note = note

    @property
    def usable(self):
        return self.status == 'ok' and self.rate is not None

    @property
    def band_consumed(self):
        """Fraction of the half-band this rate has drifted through: 0.0 = exactly on its
        constant, 1.0 = at the refusal edge.  None when there is no anchor to measure
        against."""
        if self.ratio is None:
            return None
        return abs(self.ratio - 1.0) / FX_SANITY_BAND

    @property
    def near_edge(self):
        c = self.band_consumed
        return bool(self.usable and c is not None and c >= FX_BAND_EDGE_WARN)

    def edge_warning(self):
        """The drift warning as TEXT, so it can travel in the shipped CSV instead of only
        existing as a console line nobody re-reads.  Empty string when not tripped."""
        if not self.near_edge:
            return ''
        return ('DRIFT: %.3fx its anchor (%.10g) = %.0f%% of the way to the +-%.0f%% '
                'refusal edge. Still USED. If it reaches the edge this rate is REFUSED '
                'while being CORRECT, and every %s reporter silently becomes '
                'unknown-currency (kept, unbanded, floor not applied). REMEDY: re-seed '
                'carveOut.FX_TO_USD[%r] with a dated value -- do NOT widen the band.'
                % (self.ratio, self.constant, 100.0 * self.band_consumed,
                   FX_SANITY_BAND * 100, self.currency, self.currency))

    def as_row(self):
        return {
            'currency': self.currency,
            'rate': self.rate,
            'quote_timestamp': (self.quote_ts.isoformat() if self.quote_ts is not None
                                else ''),
            'quote_age_days': (None if self.age_days is None else round(self.age_days, 4)),
            'status': self.status,
            'usable': bool(self.usable),
            'pair': self.pair or '',
            'source_endpoint': QUOTES_ENDPOINT,
            'sanity_constant': self.constant,
            'sanity_ratio': (None if self.ratio is None else round(self.ratio, 6)),
            #  THE ANCHOR-DRIFT WARNING SHIPS (CEO/MD, 2026-08-08).  It used to exist only
            #  as a console print, i.e. in the class of warning this project has spent the
            #  week digging out of: fires on an unattended overnight run, read by nobody,
            #  and the first visible symptom is a currency's names quietly leaving the
            #  floor. These two columns put it in the dated artifact that ships whole, and
            #  the same text goes into RunProvenance-*.json.
            'band_consumed_pct': (None if self.band_consumed is None
                                  else round(100.0 * self.band_consumed, 1)),
            'band_edge_warning': self.edge_warning(),
            'note': self.note,
        }

    def __repr__(self):                                     # pragma: no cover - debug aid
        return 'FxRate(%s, %s, %s, %s)' % (self.currency, self.rate, self.status,
                                           self.age_days)


# --------------------------------------------------------------------------- #
#  Supported set                                                              #
# --------------------------------------------------------------------------- #
def supported_currencies():
    """Every currency this feed will try to resolve, in a stable order.

    It is the FX_TO_USD key set -- FX_TO_USD is the SANITY BAND, so a currency with no
    constant has no band and cannot be admitted.  ADDED_CURRENCIES are in FX_TO_USD too
    (that is what "adding" them means); listing them here is documentation, not a second
    source of truth."""
    return sorted(co.FX_TO_USD)


def _pair_for(currency):
    """The `{CUR}USD` pair symbol for a currency, or None for USD / a minor unit.

    Minor units NEVER get a pair -- see MINOR_UNITS."""
    if currency == 'USD' or currency in MINOR_UNITS:
        return None
    return '%sUSD' % currency


# --------------------------------------------------------------------------- #
#  Parsing / validation of one quotes/forex response                          #
# --------------------------------------------------------------------------- #
def _index_quotes(payload):
    """{PAIRSYMBOL: (price, unix_ts)} from a v3/quotes/forex body.

    Tolerant by construction: a row with no usable symbol or no numeric price is skipped
    rather than raising, because this parses a 1,550-element vendor response on the
    critical path of a 12-hour run."""
    out = {}
    for rec in (payload or []):
        if not isinstance(rec, dict):
            continue
        sym = rec.get('symbol')
        if not isinstance(sym, str) or not sym.strip():
            continue
        try:
            price = float(rec.get('price'))
        except (TypeError, ValueError):
            continue
        ts = rec.get('timestamp')
        try:
            ts = int(ts)
        except (TypeError, ValueError):
            ts = None
        out[sym.strip().upper()] = (price, ts)
    return out


def _utc(ts_unix):
    """Unix seconds -> a NAIVE UTC datetime (naive, to compare against a naive `now`)."""
    if ts_unix in (None, 0):
        return None
    try:
        return (_dt.datetime.fromtimestamp(int(ts_unix), _dt.timezone.utc)
                .replace(tzinfo=None))
    except (OverflowError, OSError, ValueError):
        return None


def resolve_rates(quotes, currencies=None, now=None):
    """Turn an indexed quote table into one FxRate per supported currency.

    PURE -- no network, no clock beyond `now`, no globals mutated.  This is where the
    per-rate staleness check and the sanity band live, so it is the piece worth testing
    directly.

    `quotes` : {PAIR: (price, unix_ts)}  (see _index_quotes)
    `now`    : naive UTC datetime; defaults to utcnow.
    """
    now = _dt.datetime.utcnow() if now is None else now
    currencies = supported_currencies() if currencies is None else list(currencies)
    out = []
    for cur in currencies:
        const = co.FX_TO_USD.get(cur)

        #  USD is the numeraire.  It has no pair and needs no check.
        if cur == 'USD':
            out.append(FxRate(cur, pair=None, rate=1.0, status='ok', constant=const,
                              ratio=1.0, note='numeraire'))
            continue

        #  Minor units resolve from their major unit -- BEFORE any symbol lookup.
        divisor = 1.0
        lookup_cur = cur
        if cur in MINOR_UNITS:
            lookup_cur, divisor = MINOR_UNITS[cur]

        pair = _pair_for(lookup_cur)
        got = quotes.get(pair) if pair else None
        if got is None:
            out.append(FxRate(cur, pair=pair, status='missing', constant=const,
                              note='no %s in the quotes/forex response' % pair))
            continue

        price, ts_unix = got
        rate = price / divisor if divisor != 1.0 else price
        ts = _utc(ts_unix)

        #  PER-RATE FRESHNESS.  125 of 1,550 pairs are dead behind an HTTP 200, so a
        #  response-level check would pass a rate that has not moved in three years.
        if ts is None:
            out.append(FxRate(cur, pair=pair, rate=None, quote_ts=None, status='no_timestamp',
                              constant=const,
                              note='%s carried no usable timestamp -> treated as ABSENT'
                                   % pair))
            continue
        age = (now - ts).total_seconds() / 86400.0
        if age > FX_STALE_MAX_DAYS or age < -1.0:
            out.append(FxRate(cur, pair=pair, rate=None, quote_ts=ts, age_days=age,
                              status='stale', constant=const,
                              note='quote is %.1f days old (max %.1f) -> ABSENT, never '
                                   'substituted' % (age, FX_STALE_MAX_DAYS)))
            continue

        #  SANITY BAND -- the new failure mode a live feed has and a constant does not.
        ratio = (rate / const) if (const not in (None, 0)) else None
        if ratio is None:
            out.append(FxRate(cur, pair=pair, rate=None, quote_ts=ts, age_days=age,
                              status='no_sanity_anchor', constant=const,
                              note='no FX_TO_USD constant to band against -> ABSENT'))
            continue
        if not ((1.0 - FX_SANITY_BAND) <= ratio <= (1.0 + FX_SANITY_BAND)):
            out.append(FxRate(cur, pair=pair, rate=None, quote_ts=ts, age_days=age,
                              status='sanity_reject', constant=const, ratio=ratio,
                              note='live %.10g is %.2fx the %.10g constant -- outside '
                                   '+-%.0f%%; reads as a unit flip / inverted quote, so '
                                   'the rate is REFUSED (-> ABSENT), not used'
                                   % (rate, ratio, const, FX_SANITY_BAND * 100)))
            continue

        #  RECIPROCAL CROSS-CHECK -- catches the inversion the band structurally cannot.
        #  MANDATORY: an absent or stale reciprocal REFUSES the rate (F-1). See the
        #  FX_RECIPROCAL_TOL block for why a skip here deleted 16 real names.
        recip = quotes.get('USD%s' % lookup_cur)
        recip_pair = 'USD%s' % lookup_cur
        if recip is None:
            out.append(FxRate(
                cur, pair=pair, rate=None, quote_ts=ts, age_days=age,
                status='reciprocal_missing', constant=const, ratio=ratio,
                note='%s is absent, so the inversion cross-check cannot run. The rate is '
                     'REFUSED (-> ABSENT) rather than trusted: an unverifiable rate that '
                     'is wrong DELETES names invisibly, while refusing one only KEEPS '
                     'them.' % recip_pair))
            continue
        r_ts = _utc(recip[1])
        r_age = (None if r_ts is None
                 else (now - r_ts).total_seconds() / 86400.0)
        if r_ts is None or not (-1.0 <= r_age <= FX_STALE_MAX_DAYS) or not recip[0]:
            out.append(FxRate(
                cur, pair=pair, rate=None, quote_ts=ts, age_days=age,
                status='reciprocal_stale', constant=const, ratio=ratio,
                note='%s is unusable (%s), so the inversion cross-check cannot run. Rate '
                     'REFUSED (-> ABSENT).'
                     % (recip_pair,
                        'no timestamp' if r_ts is None
                        else ('no price' if not recip[0] else '%.1f days old' % r_age))))
            continue
        product = price * recip[0]                # the MAJOR unit's own product
        if abs(product - 1.0) > FX_RECIPROCAL_TOL:
            out.append(FxRate(
                cur, pair=pair, rate=None, quote_ts=ts, age_days=age,
                status='reciprocal_reject', constant=const, ratio=ratio,
                note='%s (%.10g) x %s (%.10g) = %.6f, not 1 -- the two directions '
                     'disagree, which is what an INVERTED quote looks like. Rate REFUSED '
                     '(-> ABSENT).'
                     % (pair, price, recip_pair, recip[0], product)))
            continue

        out.append(FxRate(cur, pair=pair, rate=rate, quote_ts=ts, age_days=age,
                          status='ok', constant=const, ratio=ratio,
                          note='minor unit of %s (/%g)' % (lookup_cur, divisor)
                               if divisor != 1.0 else ''))
    return out


# --------------------------------------------------------------------------- #
#  The live spot fetch                                                        #
# --------------------------------------------------------------------------- #
def fetch_spot_rates(baseurl, api_key, currencies=None, now=None, _get=None,
                     verbose=True, timeout=60, sleep=None):
    """ONE `v3/quotes/forex` call -> [FxRate].  Never raises, never hangs.

    Routed through `gdg.safe_json_list`, so a throttled 200 with an HTML body, a 429 or a
    dead socket degrades to an empty list -- which resolves to EVERY currency `missing`,
    i.e. the fail-loud path, not a silent fallback to the constants.

    `timeout` is 60s rather than the helper's 10s default: the body is ~863 KB, roughly
    two orders of magnitude larger than the per-ticker statement calls the default was
    sized for."""
    url = '%s%s' % (baseurl, QUOTES_ENDPOINT)
    payload = gdg.safe_json_list(url, params={'apikey': api_key}, timeout=timeout,
                                 label='quotes/forex', verbose=verbose, _get=_get,
                                 sleep=sleep)
    quotes = _index_quotes(payload)
    if verbose:
        print('[fx] %s returned %d pair(s)' % (QUOTES_ENDPOINT, len(quotes)), flush=True)
    return resolve_rates(quotes, currencies=currencies, now=now)


# --------------------------------------------------------------------------- #
#  Evidence                                                                   #
# --------------------------------------------------------------------------- #
def write_fx_rates_csv(rates, run_date=None, outdir='output'):
    """`output/FxRates_<date>.csv` -- the run's rates as EVIDENCE.

    In `output/` on purpose: that directory already ships whole through the Drive
    transfer's `allowlist_dirs`, which is the same precedent (and the same reasoning)
    that kept `DedupSurvivorReport_*.csv` out of the top-level pattern list.  Never
    raises; returns the path, or None."""
    try:
        run_date = run_date or _dt.date.today().strftime('%Y-%m-%d')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        path = os.path.join(outdir, 'FxRates_%s.csv' % run_date)
        pd.DataFrame([r.as_row() for r in rates]).to_csv(path, index=False)
        return path
    except Exception as e:
        print('[fx] WARNING: FxRates CSV not written (%s: %s)' % (type(e).__name__, e),
              flush=True)
        return None


def near_band_edge(rates):
    """Rates that PASSED the sanity band but sit within FX_BAND_EDGE_WARN of its edge.

    An early-warning list, not a rejection list -- the whole point is that a currency
    drifting toward refusal is announced while it is still being used."""
    return [r for r in rates if r.near_edge]


def _provenance(rates, csv_path, as_of):
    ok = [r for r in rates if r.usable]
    bad = [r for r in rates if not r.usable]
    return {
        'fx_rates_as_of': as_of,
        'source_endpoint': QUOTES_ENDPOINT,
        'stale_max_days': FX_STALE_MAX_DAYS,
        'sanity_band': FX_SANITY_BAND,
        'reciprocal_tol': FX_RECIPROCAL_TOL,
        'n_supported': len(rates),
        'n_usable': len(ok),
        #  FETCHED usable rates, i.e. excluding the USD numeraire -- the number that
        #  actually decides whether there is a feed at all (see install_for_run).
        'n_fetched_usable': sum(1 for r in ok if r.pair),
        'n_unusable': len(bad),
        'unusable': {r.currency: r.status for r in bad},
        'newest_quote': max([r.quote_ts.isoformat() for r in ok if r.quote_ts], default=None),
        'oldest_quote': min([r.quote_ts.isoformat() for r in ok if r.quote_ts], default=None),
        'max_age_days': (round(max([r.age_days for r in ok if r.age_days is not None],
                                   default=0.0), 4) if ok else None),
        'evidence_csv': csv_path,
        'abstained': sorted(ABSTAIN_CURRENCIES),
        #  COVERAGE OF THE SUPPORTED SET (F-2).  `n_usable` alone reads as reassurance;
        #  the FRACTION is what says whether a 'live' feed actually covers the universe.
        'supported_coverage': (round(len(ok) / float(len(rates)), 4) if rates else 0.0),
        'supported_coverage_ok': (bool(rates)
                                  and (len(ok) / float(len(rates)))
                                  >= FX_MIN_SUPPORTED_COVERAGE),
        'min_supported_coverage': FX_MIN_SUPPORTED_COVERAGE,
        #  ANCHOR DRIFT, IN THE ARTIFACT (CEO/MD, 2026-08-08).  A console-only warning on an
        #  unattended run is the failure class this project has spent the week removing, so
        #  the drift list travels in the provenance sidecar as well as the FxRates CSV.
        #  {currency: percent of the half-band consumed}; empty is the healthy state.
        'band_edge_warnings': {r.currency: round(100.0 * r.band_consumed, 1)
                               for r in near_band_edge(rates)},
        'band_edge_remedy': ('a currency listed in band_edge_warnings is DRIFTING toward '
                             'refusal by the +-%.0f%% sanity band while being CORRECT; at '
                             'the edge its rate is refused and every name reporting in it '
                             'silently becomes unknown-currency (kept, unbanded, floor not '
                             'applied). REMEDY: re-seed that currency in '
                             'carveOut.FX_TO_USD with a dated value -- do NOT widen the '
                             'band, which is a units check and must stay one.'
                             % (FX_SANITY_BAND * 100)),
    }


# --------------------------------------------------------------------------- #
#  The run-level install                                                      #
# --------------------------------------------------------------------------- #
def install_for_run(baseurl, api_key, run_date=None, now=None, _get=None, verbose=True,
                    outdir='output', sleep=None):
    """Fetch once, validate per rate, INSTALL into carveOut, write the evidence CSV.

    Called ONCE at the top of `Sbocker.main`, BEFORE the ~12-hour fetch -- deliberately, so
    a broken FX feed is visible at launch instead of twelve hours later.  A rate resolved
    at launch is at most ~12h old when the conversions run, three days inside the 7-day
    staleness bar.

    ALWAYS leaves `carveOut.fx_source_state()` at 'live' or 'failed', never 'unset': the
    production path must never silently fall back to the undated constants.  Returns the
    provenance dict that `RunProvenance-*.json` carries."""
    run_date = run_date or _dt.date.today().strftime('%Y-%m-%d')
    try:
        rates = fetch_spot_rates(baseurl, api_key, now=now, _get=_get, verbose=verbose,
                                 sleep=sleep)
    except Exception as e:                       # defence in depth; the fetch is guarded
        print('[fx] WARNING: FX fetch raised (%s: %s)' % (type(e).__name__, e), flush=True)
        rates = []

    table = {r.currency: r.rate for r in rates if r.usable}
    csv_path = write_fx_rates_csv(rates, run_date=run_date, outdir=outdir) if rates else None
    as_of = max([r.quote_ts.isoformat() for r in rates if r.usable and r.quote_ts],
                default=None)
    prov = _provenance(rates, csv_path, as_of)

    #  USD ALONE IS NOT A FEED (defect found in test, 2026-08-08).
    #  USD is the numeraire: it is resolved to 1.0 unconditionally and never fetched.  So a
    #  COMPLETELY DEAD endpoint still produces a one-entry table {'USD': 1.0} -- which is
    #  non-empty, would have installed as 'live', and would have left the run silently
    #  half-degraded: USD reporters floored normally, every non-USD name unknown, and the
    #  FX-UNAVAILABLE banner never printed.  That is precisely the quiet partial
    #  degradation this whole design exists to prevent.  A feed counts as live only if at
    #  least one FETCHED rate (i.e. one with a pair) survived validation.
    n_fetched_ok = sum(1 for r in rates if r.usable and r.pair)
    if n_fetched_ok:
        co.set_live_fx_rates(table, meta=prov)
    else:
        co.mark_fx_unavailable(
            'v3/quotes/forex returned no usable FETCHED rate (USD is the numeraire and '
            'is never fetched, so it does not count as a live feed)', meta=prov)

    if verbose:
        _announce(rates, prov, verbose=True)
    return prov


def _announce(rates, prov, verbose=True):
    """Say what happened, and say it LOUDLY when a currency dropped out.

    A currency that drops out does not mis-price anything -- it makes every name reporting
    in it unknown-mcap, which KEEPS the name and turns the $25M floor off for it.  That is
    the designed degradation, but it is a real change to the universe and this project's
    standing rule is that a universe change is never inferred from a count."""
    if not verbose:
        return
    bad = [r for r in rates if not r.usable]
    #  Report the state that was actually INSTALLED, not the count.  "1/38 usable" beside
    #  the word "installed" would read as a live feed on a run that has none (the USD
    #  numeraire is that 1) -- the same misreading the USD-alone guard exists to prevent.
    print('[fx] %s: %d/%d supported currencies usable (%d fetched), as-of %s '
          '(endpoint %s, stale bar %.0fd, sanity band +-%.0f%%, reciprocal tol %.0f%%)'
          % ('live rates INSTALLED' if prov['n_fetched_usable'] else 'FEED FAILED',
             prov['n_usable'], prov['n_supported'], prov['n_fetched_usable'],
             prov.get('fx_rates_as_of'), QUOTES_ENDPOINT, FX_STALE_MAX_DAYS,
             FX_SANITY_BAND * 100, FX_RECIPROCAL_TOL * 100), flush=True)
    if prov.get('evidence_csv'):
        print('[fx] rates written to %s' % prov['evidence_csv'], flush=True)
    if not prov['n_fetched_usable']:
        bang = '!' * 78
        banner = '\n'.join([
            '', bang,
            '!!! FX FEED UNAVAILABLE -- NO LIVE RATE RESOLVED !!!',
            '!!!   %s returned nothing usable, so NO reportedCurrency' % QUOTES_ENDPOINT,
            '!!!   resolves to a USD rate this run.  The $25M universe floor and the',
            '!!!   market-cap bands DEGRADE (floor keeps every name, bands are skipped)',
            '!!!   and you will see the "$25M UNIVERSE FLOOR NOT ENFORCED" banner below.',
            '!!!   THE FLOOR DOES **NOT** FALL BACK TO THE OLD HARDCODED CONSTANTS --',
            '!!!   that is deliberate: a wrong exclusion is invisible and unrecoverable,',
            '!!!   and a stale rate is the same kind of wrong number as a missing one.',
            '!!!   Run PROCEEDS.  Do NOT read this run\'s universe as floor-filtered.',
            bang, ''])
        import sys as _sys
        print(banner, file=_sys.stderr, flush=True)
        print(banner, flush=True)
        return
    if bad:
        print('[fx] %d currency(ies) did NOT resolve and are treated as UNKNOWN (names '
              'reporting in them are KEPT, unbanded, and the floor does not apply to '
              'them): %s'
              % (len(bad), ', '.join('%s=%s' % (r.currency, r.status) for r in bad)),
              flush=True)
    #  PARTIAL COVERAGE IS ITS OWN BANNER (F-2).  A feed that installs 'live' while
    #  covering a minority of the supported set is the half-degradation the USD-alone
    #  guard does not reach -- it must not be inferred from a count in a log line.
    if prov['n_fetched_usable'] and not prov['supported_coverage_ok']:
        bang = '!' * 78
        banner = '\n'.join([
            '', bang,
            '!!! FX COVERAGE DEGRADED -- the feed is LIVE but only covers %.1f%% of the'
            % (100.0 * prov['supported_coverage']),
            '!!!   supported currency set (%d of %d); the bar is %.0f%%.'
            % (prov['n_usable'], prov['n_supported'],
               100.0 * FX_MIN_SUPPORTED_COVERAGE),
            '!!!   Names reporting in an unresolved currency are KEPT and UNBANDED, and',
            '!!!   the $25M floor does NOT apply to them. The run therefore ships a',
            '!!!   PARTIALLY floored universe. Nothing is wrongly deleted -- but do NOT',
            '!!!   read this run as floor-filtered across the whole universe. The covered',
            '!!!   fraction is in the FxRates CSV and the RunProvenance fx_rates block.',
            '!!!   UNRESOLVED: %s' % ', '.join(r.currency for r in bad),
            bang, ''])
        import sys as _sys
        print(banner, file=_sys.stderr, flush=True)
        print(banner, flush=True)
    #  The console line is the CONVENIENCE copy; the CSV column and the provenance block
    #  are the ones that survive an unattended run and reach the other machine.
    for r in near_band_edge(rates):
        print('[fx] ANCHOR %s -- %s' % (r.currency, r.edge_warning()), flush=True)
    if near_band_edge(rates) and prov.get('evidence_csv'):
        print('[fx]   (also recorded in %s -> band_edge_warning, and in the fx_rates block '
              'of RunProvenance-*.json)' % prov['evidence_csv'], flush=True)


# --------------------------------------------------------------------------- #
#  POINT-IN-TIME rates (the backtest)                                         #
# --------------------------------------------------------------------------- #
#  Applying TODAY's spot to a 2021 market cap is a look-ahead-flavoured error: the band a
#  2021 name is graded in should be decided by the 2021 rate.  `v3/historical-price-full`
#  gives dated daily closes at 1 call per currency per range -- so a whole backtest range
#  costs ~1 call per supported currency, once, cached to disk.
#
#  IT IS A SEPARATE, OPT-IN FETCH and NOT part of the automatic run: the live run needs
#  SPOT, and the PIT table is only meaningful for the offline grading in
#  `baseline_tools/pipeline_analysis`.  Build it with:
#
#      python fx_rates.py --historical --from 2019-01-01 --to 2026-08-08
#
#  which writes `output/FxRatesHistorical_<from>_<to>.csv`.
# --------------------------------------------------------------------------- #
def fetch_historical_rates(baseurl, api_key, currencies=None, start=None, end=None,
                           _get=None, verbose=True, timeout=60, sleep=None):
    """Dated daily closes per currency -> a long DataFrame [date, currency, rate, pair].

    One call per currency per range.  A currency that fails simply contributes no rows --
    which makes it ABSENT in the PIT table, which is the same fail-loud treatment spot
    gets.  Minor units are derived from their major unit's series, never fetched."""
    import time as _time
    sleep = _time.sleep if sleep is None else sleep
    currencies = supported_currencies() if currencies is None else list(currencies)

    #  Fetch only MAJOR units; minor units are derived below.
    majors, derived = [], []
    for cur in currencies:
        if cur == 'USD':
            continue
        if cur in MINOR_UNITS:
            derived.append(cur)
            base = MINOR_UNITS[cur][0]
            if base not in majors and base in co.FX_TO_USD:
                majors.append(base)
            continue
        if cur not in majors:
            majors.append(cur)

    frames = []
    for i, cur in enumerate(majors):
        pair = _pair_for(cur)
        url = '%s%s' % (baseurl, HISTORY_ENDPOINT % pair)
        params = {'apikey': api_key}
        if start:
            params['from'] = str(start)
        if end:
            params['to'] = str(end)
        body = gdg.safe_json_list(url, params=params, timeout=timeout,
                                  label='history/%s' % pair, verbose=False, _get=_get)
        #  historical-price-full answers {'symbol': ..., 'historical': [{date, close,...}]}
        hist = None
        for rec in body:
            if isinstance(rec, dict) and isinstance(rec.get('historical'), list):
                hist = rec['historical']
                break
        if hist is None:
            hist = [r for r in body if isinstance(r, dict) and 'close' in r]
        rows = []
        for r in hist or []:
            try:
                rows.append({'date': pd.Timestamp(r['date']), 'currency': cur,
                             'rate': float(r['close']), 'pair': pair})
            except (KeyError, TypeError, ValueError):
                continue
        if verbose:
            print('[fx-pit] %s: %d dated close(s)' % (pair, len(rows)), flush=True)
        if rows:
            frames.append(pd.DataFrame(rows))
        if i + 1 < len(majors):
            sleep(0.2)

    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=['date', 'currency', 'rate', 'pair']))

    #  USD is the numeraire; give it an explicit series so a PIT lookup never has to
    #  special-case it downstream.
    if len(out):
        dates = sorted(out['date'].unique())
        out = pd.concat([out, pd.DataFrame({'date': dates, 'currency': 'USD',
                                            'rate': 1.0, 'pair': ''})],
                        ignore_index=True)
    #  Minor units: same series, divided.  Never fetched, never symbol-matched.
    for cur in derived:
        base, divisor = MINOR_UNITS[cur]
        src = out[out['currency'] == base]
        if src.empty:
            continue
        cp = src.copy()
        cp['currency'] = cur
        cp['rate'] = cp['rate'] / divisor
        out = pd.concat([out, cp], ignore_index=True)
    return out.sort_values(['currency', 'date']).reset_index(drop=True)


def write_historical_csv(df, start, end, outdir='output'):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    path = os.path.join(outdir, 'FxRatesHistorical_%s_%s.csv' % (start, end))
    df.to_csv(path, index=False)
    return path


#  How far PAST the end of the dated series a lookup may still be answered (F-4, reviewer
#  2026-08-08).  `rate_for` used to carry the last close forward UNBOUNDED -- verified by
#  the reviewer to answer a 2035 date with a 2022 rate, which is a stale number wearing a
#  point-in-time label, i.e. the exact thing this table exists to prevent.  A daily series
#  legitimately has weekend/holiday gaps and a pull is often a day or two behind, so a
#  small carry-forward is correct; a month is not.  Beyond this the answer is None, which
#  routes into the same unknown-currency path everything else does, and the caller says so.
PIT_MAX_FORWARD_DAYS = 31


class PitFxTable(object):
    """Dated FX lookup: the last close ON OR BEFORE a date.

    NEVER interpolates and NEVER reaches FORWARD past a date -- reaching forward is exactly
    the look-ahead this table exists to remove.  A date before the series starts resolves
    to None; so does a date more than PIT_MAX_FORWARD_DAYS past its end, so the table
    cannot silently answer for a period it has no data for.  Both route into the same
    unknown-currency path a missing rate does."""

    def __init__(self, df):
        self._by_cur = {}
        if df is None or not len(df):
            return
        d = df.copy()
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d = d.dropna(subset=['date', 'rate'])
        for cur, grp in d.groupby('currency'):
            g = grp.sort_values('date')
            self._by_cur[cur] = (list(g['date']), list(g['rate'].astype(float)))

    @property
    def currencies(self):
        return sorted(self._by_cur)

    def coverage(self, currencies=None):
        """(covered, total, fraction) of a currency set this table can actually answer.

        A PARTIAL PULL MUST NOT COLLAPSE SILENTLY (F-4): a table holding 3 of 38
        currencies loads without complaint and reads as point-in-time while making 35
        currencies unknown.  The caller checks this and says which it got."""
        want = supported_currencies() if currencies is None else list(currencies)
        have = sum(1 for c in want if c in self._by_cur)
        return have, len(want), (have / float(len(want)) if want else 0.0)

    def span(self):
        """(first, last) date across the whole table, or (None, None)."""
        if not self._by_cur:
            return None, None
        firsts = [d[0][0] for d in self._by_cur.values() if d[0]]
        lasts = [d[0][-1] for d in self._by_cur.values() if d[0]]
        return (min(firsts) if firsts else None), (max(lasts) if lasts else None)

    def rate_for(self, currency, when):
        """USD-per-unit for `currency` as of `when`, or None.

        None when the date precedes the series, and None when it is more than
        PIT_MAX_FORWARD_DAYS past its end -- an unbounded carry-forward is a stale rate
        wearing a point-in-time label."""
        import bisect
        if not isinstance(currency, str):
            return None
        entry = self._by_cur.get(currency.strip())
        if entry is None:
            return None
        dates, rates = entry
        try:
            ts = pd.Timestamp(when)
        except (TypeError, ValueError):
            return None
        if ts is None or pd.isna(ts):
            return None
        i = bisect.bisect_right(dates, ts) - 1
        if i < 0:
            return None
        if (ts - dates[i]).days > PIT_MAX_FORWARD_DAYS:
            return None
        return rates[i]


def load_pit_rates(path=None, outdir='output'):
    """Load the newest `FxRatesHistorical_*.csv` into a PitFxTable, or None if absent.

    Returning None (rather than raising or silently substituting spot) is what lets the
    caller SAY which basis it used."""
    import glob as _glob
    if path is None:
        hits = sorted(_glob.glob(os.path.join(outdir, 'FxRatesHistorical_*.csv')))
        if not hits:
            return None
        path = hits[-1]
    try:
        return PitFxTable(pd.read_csv(path))
    except Exception as e:
        print('[fx-pit] WARNING: could not load %s (%s: %s)' % (path, type(e).__name__, e),
              flush=True)
        return None


# --------------------------------------------------------------------------- #
#  CLI -- the historical pull only.  The spot pull belongs to the pipeline.    #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':                                  # pragma: no cover
    import argparse
    import configuration as cf

    ap = argparse.ArgumentParser(description='FX rates (historical pull / spot probe)')
    ap.add_argument('--historical', action='store_true',
                    help='pull dated daily closes (1 call per currency)')
    ap.add_argument('--from', dest='start', default='2019-01-01')
    ap.add_argument('--to', dest='end', default=_dt.date.today().strftime('%Y-%m-%d'))
    a = ap.parse_args()

    _cfg = cf.getDataFetchConfiguration([])
    _base, _key = _cfg['baseurl'], _cfg['api_key']
    if a.historical:
        _df = fetch_historical_rates(_base, _key, start=a.start, end=a.end)
        print('wrote', write_historical_csv(_df, a.start, a.end))
    else:
        for _r in fetch_spot_rates(_base, _key):
            print('%-4s %-8s %-16s %s' % (_r.currency, _r.status, _r.rate,
                                          _r.quote_ts))
