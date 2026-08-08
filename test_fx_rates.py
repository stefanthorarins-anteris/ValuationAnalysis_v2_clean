"""The live FX feed: per-rate validation, and the fail-loud path it must degrade into.

EVERY TEST HERE IS OFFLINE.  The HTTP layer is injected (`_get=`); nothing in this file
may make a real network call.

WHAT THIS SUITE IS ACTUALLY DEFENDING.  Replacing a constant with a live feed trades one
failure mode (a known-wrong number) for three new ones:
  1. a rate that is quietly DEAD -- `v3/quotes/forex` serves 125 of its 1,550 pairs with a
     timestamp older than 30 days (oldest ~1,316 days) under an HTTP 200, so a
     response-level check passes a three-year-old price;
  2. a rate that is quietly WRONG BY A FACTOR -- a vendor unit flip or an inverted quote;
  3. NO rate at all, silently falling back to the constants the change exists to remove.
The first two are why validation is PER RATE and why FX_TO_USD survives as a sanity band;
the third is why carveOut has a three-state FX source and why 'failed' is not 'unset'.

The measured facts these tests encode were reproduced through this code on the real
2026-08-07 CUR3K panel: the stale table gets 11 universe-membership decisions wrong at the
$25M floor (7 EUR names wrongly deleted, 4 wrongly kept) and puts 32 names in the wrong
market-cap band.
"""

import datetime as dt
import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import carveOut as co
import fx_rates as fx


NOW = dt.datetime(2026, 8, 8, 12, 0, 0)


def _ts(days_ago):
    return int((NOW - dt.timedelta(days=days_ago) - dt.datetime(1970, 1, 1))
               .total_seconds())


#  A body shaped exactly like the real one, with the rates observed on 2026-08-08.
_LIVE_2026_08_08 = {
    'EURUSD': 1.15593, 'GBPUSD': 1.34899, 'KRWUSD': 0.0007103672, 'SEKUSD': 0.10548,
    'TRYUSD': 0.02092441, 'PENUSD': 0.29656, 'MADUSD': 0.10738, 'ARSUSD': 0.0006678322,
}


def _body(pairs=None, age_days=0.2, extra=(), reciprocals=True):
    """A quotes/forex payload: [{symbol, price, timestamp}, ...].

    Emits the RECIPROCAL `USD{CUR}` pair too, because the real response does (probed
    2026-08-08: present and fresh for all 35 major units) and the inversion cross-check
    keys off it.  `reciprocals=False` reproduces a body without them, which must SKIP the
    check rather than fail it."""
    pairs = _LIVE_2026_08_08 if pairs is None else pairs
    out = [{'symbol': s, 'price': p, 'timestamp': _ts(age_days)}
           for s, p in pairs.items()]
    if reciprocals:
        for s, p in pairs.items():
            if s.endswith('USD') and p:
                out.append({'symbol': 'USD' + s[:-3], 'price': 1.0 / p,
                            'timestamp': _ts(age_days)})
    out.extend(extra)
    return out


def _full_body(age_days=0.2):
    """A body covering the WHOLE supported set, each pair quoted exactly at its sanity
    anchor.  Needed for the coverage tests: `_body()` carries 8 pairs, which is a
    legitimately DEGRADED feed and now (correctly) banners as one."""
    pairs = {}
    for cur, const in co.FX_TO_USD.items():
        if cur == 'USD' or cur in fx.MINOR_UNITS:
            continue
        pairs['%sUSD' % cur] = const
    return _body(pairs, age_days=age_days)


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _get_returning(payload, status_code=200):
    return lambda *a, **k: _Resp(payload, status_code)


#  THE FX-STATE ISOLATION FIXTURE LIVES IN THE ROOT `conftest.py` (reviewer F-6).
#  carveOut's FX source is module-global, so a guard that sits in one test FILE protects
#  only that file while the hazard is repo-wide. It was moved rather than duplicated.


# --------------------------------------------------------------------------- #
#  Parsing + the supported set                                                #
# --------------------------------------------------------------------------- #
def test_the_supported_set_is_the_sanity_band_key_set():
    """FX_TO_USD is the BAND now, so a currency with no constant has no band and cannot be
    admitted.  Two lists would drift; there is one."""
    assert fx.supported_currencies() == sorted(co.FX_TO_USD)


def test_PEN_and_MAD_are_supported_and_ARS_is_deliberately_NOT():
    """CEO 2026-08-08.  ARS quotes cleanly (0.000667) -- the abstention is about our ARS
    STATEMENT data being broken by three orders of magnitude, not about the rate, so a
    later reader must not 'fix' it by adding the currency."""
    assert 'PEN' in co.FX_TO_USD and 'MAD' in co.FX_TO_USD
    assert 'ARS' not in co.FX_TO_USD
    assert 'ARS' in fx.ABSTAIN_CURRENCIES
    why = fx.ABSTAIN_CURRENCIES['ARS']
    assert 'BMA' in why and 'marketCap continuous' in why


def test_an_ARS_quote_in_the_body_is_IGNORED_not_adopted():
    """The abstention has to hold against a body that CONTAINS the pair -- which the real
    one does."""
    rows = fx.resolve_rates(fx._index_quotes(_body()), now=NOW)
    assert 'ARS' not in {r.currency for r in rows}


# --------------------------------------------------------------------------- #
#  THE MINOR-UNIT TRAP                                                        #
# --------------------------------------------------------------------------- #
def test_GBp_and_GBX_resolve_to_PENCE_not_to_POUNDS():
    """THE nastiest trap in the module.  There is no `GBpUSD` pair, and an uppercased
    lookup of 'GBp' hits GBPUSD -- which would make every pence-denominated market cap 100x
    too large.  Minor units are resolved from their major unit, explicitly, before any
    symbol lookup."""
    got = {r.currency: r for r in fx.resolve_rates(fx._index_quotes(_body()), now=NOW)}
    assert got['GBP'].rate == pytest.approx(1.34899)
    assert got['GBp'].rate == pytest.approx(1.34899 / 100)
    assert got['GBX'].rate == pytest.approx(1.34899 / 100)
    #  and the evidence records the pair ACTUALLY consulted -- the MAJOR unit's, never a
    #  fabricated 'GBpUSD'
    assert got['GBp'].pair == 'GBPUSD'


def test_a_hypothetical_GBpUSD_pair_in_the_body_cannot_hijack_the_minor_unit():
    """Defence in depth: even if the vendor started serving such a symbol, pence must come
    from GBP/100 and not from whatever that pair says."""
    body = _body(extra=[{'symbol': 'GBPUSD', 'price': 1.34899, 'timestamp': _ts(0.2)},
                        {'symbol': 'GBPUSD'.replace('GBP', 'GBP'), 'price': 1.34899,
                         'timestamp': _ts(0.2)}])
    got = {r.currency: r for r in fx.resolve_rates(fx._index_quotes(body), now=NOW)}
    assert got['GBp'].rate == pytest.approx(1.34899 / 100)


# --------------------------------------------------------------------------- #
#  PER-RATE staleness -- the trap the whole design is built around            #
# --------------------------------------------------------------------------- #
def test_a_stale_rate_is_ABSENT_and_is_NEVER_substituted():
    """125 of 1,550 pairs are dead behind an HTTP 200.  A stale rate must not fall back to
    its constant -- that would be the undated snapshot again, wearing a live label."""
    body = _body({'EURUSD': 1.15593}, age_days=0.2)
    body += [{'symbol': 'SEKUSD', 'price': 0.10548, 'timestamp': _ts(400)}]
    got = {r.currency: r for r in fx.resolve_rates(fx._index_quotes(body), now=NOW)}
    assert got['SEK'].status == 'stale'
    assert got['SEK'].rate is None
    assert got['SEK'].usable is False
    assert got['EUR'].usable is True


def test_freshness_is_checked_PER_RATE_not_per_RESPONSE():
    """The load-bearing property.  One response carries live and dead pairs together, so a
    healthy sibling must not vouch for a dead one."""
    body = (_body({'EURUSD': 1.15593}, age_days=0.1)
            + [{'symbol': 'KRWUSD', 'price': 0.0007103672, 'timestamp': _ts(1316)},
               {'symbol': 'USDKRW', 'price': 1407.72259, 'timestamp': _ts(1316)}])
    got = {r.currency: r for r in fx.resolve_rates(fx._index_quotes(body), now=NOW)}
    assert got['EUR'].usable and not got['KRW'].usable
    assert got['KRW'].status == 'stale'


@pytest.mark.parametrize('age,ok', [(0.0, True), (6.9, True), (7.1, False), (30.0, False)])
def test_the_staleness_bar_is_where_it_says_it_is(age, ok):
    body = _body({'EURUSD': 1.15593}, age_days=age)
    got = fx.resolve_rates(fx._index_quotes(body), currencies=['EUR'], now=NOW)[0]
    assert got.usable is ok


def test_a_missing_timestamp_is_ABSENT_not_assumed_fresh():
    """A quote we cannot date is a quote we cannot trust; assuming fresh would be the
    silent-substitution failure with extra steps."""
    body = [{'symbol': 'EURUSD', 'price': 1.15593}]
    got = fx.resolve_rates(fx._index_quotes(body), currencies=['EUR'], now=NOW)[0]
    assert got.status == 'no_timestamp' and not got.usable


def test_a_missing_pair_is_ABSENT():
    got = fx.resolve_rates({}, currencies=['EUR'], now=NOW)[0]
    assert got.status == 'missing' and not got.usable


def test_USD_is_the_numeraire_and_needs_no_pair():
    got = fx.resolve_rates({}, currencies=['USD'], now=NOW)[0]
    assert got.usable and got.rate == 1.0


# --------------------------------------------------------------------------- #
#  THE SANITY BAND -- the failure mode a live feed has and a constant does not #
# --------------------------------------------------------------------------- #
def test_a_100x_unit_flip_is_REFUSED():
    """The reason FX_TO_USD survives at all.  A vendor serving pence where it served
    pounds is a silent 100x error that no freshness check can see."""
    body = [{'symbol': 'EURUSD', 'price': 115.593, 'timestamp': _ts(0.1)}]
    got = fx.resolve_rates(fx._index_quotes(body), currencies=['EUR'], now=NOW)[0]
    assert got.status == 'sanity_reject' and got.rate is None


def test_the_BAND_ALONE_CANNOT_catch_an_inverted_NEAR_PARITY_quote():
    """THE HOLE, pinned as a fact rather than assumed away.  Serving USD-per-EUR as
    EUR-per-USD turns 1.15593 into 0.86511 -- 0.80x the 1.08 constant, comfortably INSIDE
    +-50%.  Tightening the band is not available: TRY legitimately sits at 0.70x.  This is
    exactly why the reciprocal cross-check exists.

    Driven with a CONSISTENTLY inverted pair (both directions swapped, product still 1) so
    the reciprocal check passes and the BAND is the only thing under test."""
    body = [{'symbol': 'EURUSD', 'price': 1 / 1.15593, 'timestamp': _ts(0.1)},
            {'symbol': 'USDEUR', 'price': 1.15593, 'timestamp': _ts(0.1)}]
    got = fx.resolve_rates(fx._index_quotes(body), currencies=['EUR'], now=NOW)[0]
    assert got.status == 'ok', 'the band alone was expected to pass this'
    assert 0.5 <= got.ratio <= 1.5


def test_the_RECIPROCAL_CROSS_CHECK_catches_the_inversion_the_band_cannot():
    """`{CUR}USD * USD{CUR} == 1` for a correct feed (worst deviation measured across all
    35 pairs on 2026-08-08: 3.1e-5).  An inverted EURUSD makes the product 0.748."""
    body = [{'symbol': 'EURUSD', 'price': 1 / 1.15593, 'timestamp': _ts(0.1)},
            {'symbol': 'USDEUR', 'price': 0.86511, 'timestamp': _ts(0.1)}]
    got = fx.resolve_rates(fx._index_quotes(body), currencies=['EUR'], now=NOW)[0]
    assert got.status == 'reciprocal_reject'
    assert got.rate is None
    assert '0.74' in got.note


def test_the_reciprocal_check_passes_a_CORRECT_feed():
    """It must not become a new way to lose rates: the real body's products are 1.000000."""
    rows = fx.resolve_rates(fx._index_quotes(_body()), now=NOW)
    assert all(r.status != 'reciprocal_reject' for r in rows)
    assert {r.currency for r in rows if r.usable} >= {'EUR', 'GBP', 'KRW', 'TRY'}


def test_an_ABSENT_reciprocal_REFUSES_the_rate():
    """INVERTED 2026-08-08 (reviewer F-1).  This used to assert a SKIP, on the reasoning
    that "a missing cross-check is not evidence of a bad rate" -- sound in the abstract,
    wrong here, because it inverts this module's own asymmetry. An unverifiable rate that
    happens to be inverted is ACCEPTED and DELETES names invisibly; refusing it only KEEPS
    them. Measured cost of the old behaviour on the CUR3K panel: 16 names wrongly deleted."""
    absent = {r.currency: r for r in
              fx.resolve_rates(fx._index_quotes(_body(reciprocals=False)), now=NOW)}
    for cur in ('EUR', 'GBP', 'KRW', 'TRY'):
        assert absent[cur].status == 'reciprocal_missing', (cur, absent[cur].status)
        assert absent[cur].rate is None and not absent[cur].usable
    #  USD is the numeraire and has no reciprocal to check
    assert absent['USD'].usable


def test_a_STALE_or_PRICELESS_reciprocal_REFUSES_the_rate():
    for recip in ({'symbol': 'USDEUR', 'price': 0.86511, 'timestamp': _ts(400)},
                  {'symbol': 'USDEUR', 'price': 0.0, 'timestamp': _ts(0.1)},
                  {'symbol': 'USDEUR', 'price': 0.86511}):
        body = [{'symbol': 'EURUSD', 'price': 1.15593, 'timestamp': _ts(0.1)}, recip]
        got = fx.resolve_rates(fx._index_quotes(body), currencies=['EUR'], now=NOW)[0]
        assert got.status == 'reciprocal_stale', (recip, got.status)
        assert not got.usable


def test_the_INVERTED_rate_that_the_old_SKIP_let_through_is_now_REFUSED():
    """The exact defect, as a behaviour test: an inverted primary with NO reciprocal.
    It used to come back status='ok', rate=0.8651."""
    body = [{'symbol': 'EURUSD', 'price': 1 / 1.15593, 'timestamp': _ts(0.1)}]
    got = fx.resolve_rates(fx._index_quotes(body), currencies=['EUR'], now=NOW)[0]
    assert got.status == 'reciprocal_missing'
    assert got.rate is None, 'an unverifiable rate must never reach the floor'


def test_refusing_is_the_SAFE_direction_and_the_code_says_why():
    """The asymmetry that decided F-1, pinned so a later reader does not re-argue it from
    first principles and re-introduce the skip."""
    src = open(os.path.join(_HERE, 'fx_rates.py'), encoding='utf-8').read()
    assert 'DELETES names invisibly' in src or 'WRONGLY DELETED 16 NAMES' in src
    assert 'refusing' in src.lower()


def test_the_RESIDUAL_is_any_PRODUCT_PRESERVING_error_pair_not_just_inversion():
    """Sharpened by the reviewer.  `EURUSD x1.25` with `USDEUR /1.25` has product exactly 1
    and sits inside +-50%, so it passes BOTH checks at a 25%-high rate.  Nothing in one
    quotes/forex response can distinguish that from a real move."""
    body = [{'symbol': 'EURUSD', 'price': 1.15593 * 1.25, 'timestamp': _ts(0.1)},
            {'symbol': 'USDEUR', 'price': (1 / 1.15593) / 1.25, 'timestamp': _ts(0.1)}]
    got = fx.resolve_rates(fx._index_quotes(body), currencies=['EUR'], now=NOW)[0]
    assert got.usable, 'the residual is real; this test documents it, it is not a bug fix'
    assert got.rate == pytest.approx(1.15593 * 1.25)
    src = open(os.path.join(_HERE, 'fx_rates.py'), encoding='utf-8').read()
    assert 'PRODUCT-PRESERVING ERROR PAIR' in src


def test_the_minor_unit_reciprocal_uses_the_MAJOR_units_product():
    """GBp has no reciprocal of its own; the check must run on GBP's pair, not fabricate
    a 'USDGBp' or compare pence against pounds."""
    got = {r.currency: r for r in fx.resolve_rates(fx._index_quotes(_body()), now=NOW)}
    assert got['GBp'].usable and got['GBp'].rate == pytest.approx(1.34899 / 100)


def test_TRY_at_MINUS_30_PERCENT_is_ACCEPTED_because_the_band_is_a_UNITS_check():
    """The calibration argument, pinned.  TRY really is ~30% from its constant on live
    data, so a band tight enough to 'catch drift' would reject good rates.  The band exists
    to catch ORDERS OF MAGNITUDE, not to police accuracy."""
    got = {r.currency: r for r in fx.resolve_rates(fx._index_quotes(_body()), now=NOW)}
    assert got['TRY'].usable
    assert got['TRY'].ratio == pytest.approx(0.02092441 / 0.030, rel=1e-6)
    assert got['TRY'].ratio < 0.71


def test_a_refused_rate_is_indistinguishable_from_a_missing_one_DOWNSTREAM():
    """Three different causes, ONE behaviour: unknown currency.  That is what lets the
    already-built floor/band degradation absorb all of them without new machinery."""
    stale = fx.resolve_rates(fx._index_quotes(
        [{'symbol': 'EURUSD', 'price': 1.15593, 'timestamp': _ts(99)}]),
        currencies=['EUR'], now=NOW)[0]
    flipped = fx.resolve_rates(fx._index_quotes(
        [{'symbol': 'EURUSD', 'price': 115.593, 'timestamp': _ts(0.1)}]),
        currencies=['EUR'], now=NOW)[0]
    absent = fx.resolve_rates({}, currencies=['EUR'], now=NOW)[0]
    for r in (stale, flipped, absent):
        assert r.rate is None and not r.usable
        co.set_live_fx_rates({'USD': 1.0})
        assert co._fx_to_usd('EUR') is None


def test_the_band_edge_warning_fires_on_TRY_and_on_nothing_else():
    """The anchors AGE.  A currency in a sustained trend eventually gets refused while
    being correct, and that must be seen coming rather than discovered as a name silently
    leaving the universe."""
    rows = fx.resolve_rates(fx._index_quotes(_body()), now=NOW)
    near = {r.currency for r in fx.near_band_edge(rows)}
    assert near == {'TRY'}, near
    try_row = next(r for r in rows if r.currency == 'TRY')
    assert try_row.usable, 'a drifting rate is still USED, not refused'
    assert try_row.band_consumed == pytest.approx(0.605, abs=0.01)


def test_the_anchor_drift_warning_SHIPS_in_the_csv_not_only_on_the_console(tmp_path):
    """CEO/MD 2026-08-08.  A warning that fires on an unattended overnight run and lives
    only in the console is the exact failure class this project spent the week removing.
    It has to be in the dated artifact that travels to the other machine."""
    fx.install_for_run('http://b/', 'KEY', now=NOW, run_date='2026-08-08',
                       outdir=str(tmp_path), verbose=False,
                       _get=_get_returning(_body()))
    got = pd.read_csv(tmp_path / 'FxRates_2026-08-08.csv').set_index('currency')
    assert {'band_consumed_pct', 'band_edge_warning'} <= set(got.columns)
    warn = got.loc['TRY', 'band_edge_warning']
    assert isinstance(warn, str) and warn.startswith('DRIFT:')
    #  the remedy has to be IN the artifact, not only in someone's head
    assert 're-seed' in warn and 'do NOT widen the band' in warn
    assert 'unknown-currency' in warn
    assert got.loc['TRY', 'band_consumed_pct'] == pytest.approx(60.5, abs=1.0)
    #  and a healthy currency carries no warning
    assert not isinstance(got.loc['EUR', 'band_edge_warning'], str) or \
        got.loc['EUR', 'band_edge_warning'] == ''


def test_the_anchor_drift_warning_SHIPS_in_the_provenance_json(tmp_path):
    import json
    import postBo
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       _get=_get_returning(_body()))
    blob = postBo._fx_provenance()
    assert blob['band_edge_warnings'] == {'TRY': pytest.approx(60.5, abs=1.0)}
    assert 're-seed' in blob['band_edge_remedy']
    assert 'do NOT widen the band' in blob['band_edge_remedy']
    json.dumps(blob, default=str)


def test_a_healthy_panel_ships_an_EMPTY_drift_list_not_a_missing_one(tmp_path):
    """Empty is a fact ("checked, nothing drifting"); absent is not."""
    import postBo
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       _get=_get_returning(_body({'EURUSD': 1.15593})))
    assert postBo._fx_provenance()['band_edge_warnings'] == {}


# --------------------------------------------------------------------------- #
#  carveOut's THREE-state FX source                                           #
# --------------------------------------------------------------------------- #
def test_unset_means_the_CONSTANTS_so_offline_tooling_is_unchanged():
    assert co.fx_source_state() == 'unset'
    assert co._fx_to_usd('KRW') == co.FX_TO_USD['KRW']


def test_live_means_ONLY_the_live_table_never_the_constants():
    """The whole point: a currency the feed could not resolve must NOT quietly revert to
    its constant, or the change buys nothing on exactly the currencies it failed on."""
    co.set_live_fx_rates({'KRW': 0.0007103672, 'USD': 1.0})
    assert co.fx_source_state() == 'live'
    assert co._fx_to_usd('KRW') == pytest.approx(0.0007103672)
    assert co._fx_to_usd('SEK') is None, "fell back to the constant"


def test_failed_means_EVERY_currency_is_unknown():
    """THE load-bearing requirement: on FX failure the floor does not run on the old
    constants.  It runs on nothing."""
    co.mark_fx_unavailable('endpoint dead')
    assert co.fx_source_state() == 'failed'
    for cur in ('USD', 'EUR', 'KRW', 'GBp'):
        assert co._fx_to_usd(cur) is None


def test_an_EMPTY_install_is_a_FAILURE_not_an_install():
    """A caller passing {} must not produce a 'live' state with nothing in it."""
    co.set_live_fx_rates({})
    assert co.fx_source_state() == 'failed'


def test_a_failed_feed_ALSO_refuses_a_materialized_marketCap_usd_column():
    """The back door.  `marketCap_usd` is materialized at ingest from whatever FX was live
    THEN; honouring it on a run whose own FX is dead would re-admit a stale number and let
    the floor fire on it -- the exact outcome this design forbids."""
    cdx = pd.DataFrame({'source': ['A'], 'marketCap': [float('nan')],
                        'marketCap_usd': [2e9]})
    assert co.currency_data_present(cdx) is True            # unset: honoured
    assert co.marketcap_usd_series(cdx).notna().any()
    co.mark_fx_unavailable('dead')
    assert co.currency_data_present(cdx) is False           # failed: refused
    assert not co.marketcap_usd_series(cdx).notna().any()


def test_currency_data_present_MIRRORS_the_conversion_it_gates():
    """The divergence that became reachable the moment a rate could fail: when the
    `reportedCurrency` COLUMN exists, the column decides -- it must not fall through to a
    stale materialized column and report 'currency present' while every name converts to
    NaN, which would swap the NOT-ENFORCED banner for a false 'floor applied, 0 excluded'."""
    cdx = pd.DataFrame({'source': ['A'], 'marketCap': [1e9],
                        'reportedCurrency': ['KRW'], 'marketCap_usd': [2e9]})
    co.set_live_fx_rates({'USD': 1.0})                      # KRW did NOT resolve
    assert co.currency_data_present(cdx) is False
    assert not co.marketcap_usd_series(cdx).notna().any()


def test_a_live_rate_really_changes_the_USD_market_cap():
    """The no-op check in reverse: prove the wiring is load-bearing, not decorative."""
    cdx = pd.DataFrame({'source': ['A'], 'marketCap': [1e12],
                        'reportedCurrency': ['KRW']})
    stale = float(co.marketcap_usd_series(cdx).iloc[0])
    co.set_live_fx_rates({'KRW': 0.0007103672})
    live = float(co.marketcap_usd_series(cdx).iloc[0])
    assert stale == pytest.approx(1e12 * 0.00073)
    assert live == pytest.approx(1e12 * 0.0007103672)
    assert stale != live


# --------------------------------------------------------------------------- #
#  install_for_run: the run-level contract                                    #
# --------------------------------------------------------------------------- #
def test_install_leaves_state_LIVE_and_writes_the_evidence_csv(tmp_path):
    prov = fx.install_for_run('http://b/', 'KEY', now=NOW, run_date='2026-08-08',
                              outdir=str(tmp_path), verbose=False,
                              _get=_get_returning(_body()))
    assert co.fx_source_state() == 'live'
    path = tmp_path / 'FxRates_2026-08-08.csv'
    assert path.exists()
    got = pd.read_csv(path)
    #  EVERY supported currency is in the evidence, usable or not -- an absent row and an
    #  unusable row are different facts.
    assert set(got['currency']) == set(fx.supported_currencies())
    assert set(got.columns) >= {'currency', 'rate', 'quote_timestamp', 'status',
                                'source_endpoint'}
    assert (got['source_endpoint'] == 'v3/quotes/forex').all()
    assert prov['fx_rates_as_of']
    assert prov['n_usable'] == int(got['usable'].sum())


def test_install_NEVER_leaves_the_state_UNSET(tmp_path):
    """Production must never silently sit on the undated constants.  A dead endpoint has
    to land in 'failed', not in 'no feed was attempted'."""
    for payload in ([], {'Error Message': 'limit'}, None):
        co.clear_live_fx_rates()
        fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path),
                           verbose=False, sleep=lambda _s: None,
                           _get=_get_returning(payload))
        assert co.fx_source_state() == 'failed', payload


def test_USD_ALONE_IS_NOT_A_FEED(tmp_path):
    """A DEFECT THIS SUITE CAUGHT.  USD is the numeraire -- resolved to 1.0 and never
    fetched -- so a completely dead endpoint still yields a one-entry table {'USD': 1.0}.
    That is non-empty, and it would have installed as 'live': USD reporters floored
    normally, every non-USD name unknown, and the FX-UNAVAILABLE banner never printed.
    A silent HALF-degradation is worse than a loud total one."""
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       sleep=lambda _s: None, _get=_get_returning([]))
    assert co.fx_source_state() == 'failed'
    assert co._fx_to_usd('USD') is None, "USD must not survive as a lone 'live' rate"


def test_a_dead_endpoint_does_NOT_raise_and_does_NOT_fall_back(tmp_path):
    """It runs immediately after (or before) a 12-hour fetch; it may degrade, never abort."""
    import requests

    def _boom(*_a, **_k):
        raise requests.ConnectionError('socket died')

    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       sleep=lambda _s: None, _get=_boom)
    assert co.fx_source_state() == 'failed'
    assert co._fx_to_usd('EUR') is None


def test_a_throttled_200_carrying_HTML_degrades_to_failed(tmp_path):
    """The house's signature vendor failure: a 200 whose body is not JSON."""
    class _Html:
        status_code = 200

        def json(self):
            raise ValueError('Expecting value: line 1 column 1 (char 0)')

    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       sleep=lambda _s: None, _get=lambda *a, **k: _Html())
    assert co.fx_source_state() == 'failed'


def test_the_install_announces_the_degradation_LOUDLY(tmp_path, capsys):
    """A universe change is never inferred from a count in this project."""
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=True,
                       sleep=lambda _s: None, _get=_get_returning([]))
    out = capsys.readouterr().out
    assert 'FX FEED UNAVAILABLE' in out
    assert 'FLOOR' in out.upper()
    assert 'CONSTANTS' in out.upper()
    #  and the summary line must not read as a live install on a run that has none
    assert 'FEED FAILED' in out
    assert 'live rates INSTALLED' not in out


def test_the_provenance_block_is_json_safe_and_states_the_basis(tmp_path):
    import json
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       _get=_get_returning(_body()))
    import postBo
    blob = postBo._fx_provenance()
    assert blob['state'] == 'live'
    assert blob['source_endpoint'] == 'v3/quotes/forex'
    assert blob['stale_max_days'] == fx.FX_STALE_MAX_DAYS
    json.dumps(blob, default=str)          # it goes into RunProvenance-*.json


def test_the_provenance_WARNS_when_no_feed_ran_or_the_feed_died(tmp_path):
    """A run whose FX died and a run whose FX was healthy must not ship indistinguishable
    artifacts -- the 2026-08-07 post-mortem rule."""
    import postBo
    assert 'UNDATED' in postBo._fx_provenance()['warning']
    co.mark_fx_unavailable('dead')
    assert 'did NOT run' in postBo._fx_provenance()['warning']


def test_the_FX_STAMP_is_read_from_MODULE_STATE_not_from_a_loaded_pickle():
    """`-loadbometric` rebuilds resdic from a SAVED panel, which can carry a PREVIOUS run's
    FX stamp.  The STAMP therefore comes from module state, which always describes THIS
    process.  `resdic` is used for ONE thing only -- panel coverage -- because that is a
    property of the data being scored, not of the feed."""
    import inspect
    import postBo
    src = inspect.getsource(postBo._fx_provenance)
    assert 'live_fx_meta' in src and 'fx_source_state' in src
    #  Scan CODE only -- the comments discuss `resdic` by name, and a substring sweep over
    #  the whole source trips on the explanation rather than on a real read (the same trap
    #  test_post_fetch_hardening documents for `requests.get`).
    code = [ln for ln in src.splitlines() if not ln.strip().startswith('#')]
    reads = [ln.strip() for ln in code if 'resdic' in ln]
    assert len(reads) == 2, reads
    assert reads[0].startswith('def _fx_provenance('), reads[0]
    assert "(resdic or {}).get('cdx_df')" in reads[1], reads[1]


def test_the_panel_COVERAGE_ships_and_flags_a_partly_floored_universe(tmp_path):
    """F-2.  `floor_enforced: true` over a half-floored universe is the
    label-means-something-else defect; the covered FRACTION is what qualifies it."""
    import postBo
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       _get=_get_returning(_full_body()))
    #  half the names report in a currency the feed did not resolve
    cdx = pd.DataFrame({'source': ['A', 'B'], 'marketCap': [1e9, 1e9],
                        'reportedCurrency': ['EUR', 'XYZ']})
    blob = postBo._fx_provenance({'cdx_df': cdx})
    assert blob['panel_coverage'] == pytest.approx(0.5)
    assert blob['panel_sources_with_usd_mcap'] == 1 and blob['panel_sources'] == 2
    assert blob['panel_coverage_ok'] is False
    assert 'NOT\nfloor-filtered' in blob['coverage_warning'].replace(' ', '\n') or \
        'floor-filtered end to end' in blob['coverage_warning']


def test_full_panel_coverage_reports_OK(tmp_path):
    import postBo
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       _get=_get_returning(_full_body()))
    cdx = pd.DataFrame({'source': ['A', 'B'], 'marketCap': [1e9, 1e9],
                        'reportedCurrency': ['EUR', 'GBP']})
    blob = postBo._fx_provenance({'cdx_df': cdx})
    assert blob['panel_coverage'] == pytest.approx(1.0)
    assert blob['panel_coverage_ok'] is True
    assert 'coverage_warning' not in blob


def test_a_LIVE_feed_with_thin_COVERAGE_banners_LOUDLY(tmp_path, capsys):
    """The half-degradation one level up: installing 'live' says nothing about how much of
    the supported set survived."""
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=True,
                       _get=_get_returning(_body({'EURUSD': 1.15593})))
    out = capsys.readouterr().out
    assert co.fx_source_state() == 'live'
    assert 'FX COVERAGE DEGRADED' in out
    assert 'PARTIALLY floored' in out
    got = pd.read_csv(tmp_path / ('FxRates_%s.csv' % dt.date.today().strftime('%Y-%m-%d')))
    assert int(got['usable'].sum()) == 2                    # EUR + USD numeraire


def test_the_supported_coverage_fraction_is_in_the_provenance(tmp_path):
    import postBo
    fx.install_for_run('http://b/', 'KEY', now=NOW, outdir=str(tmp_path), verbose=False,
                       _get=_get_returning(_full_body()))
    blob = postBo._fx_provenance()
    assert blob['supported_coverage'] == pytest.approx(1.0)
    assert blob['supported_coverage_ok'] is True


def test_currency_data_present_requires_marketCap_too():
    """F-3: the mirror was still partial.  `marketcap_usd_series` is all-NaN without
    `marketCap`, so reporting 'currency present' against it was a false positive."""
    cdx = pd.DataFrame({'source': ['A'], 'reportedCurrency': ['EUR']})
    assert co.currency_data_present(cdx) is False
    assert not co.marketcap_usd_series(cdx).notna().any()


def test_a_dead_feed_does_not_let_the_SUFFIX_fallback_guess_raw_as_USD():
    """F-5: with every rate unresolvable, a KNOWN suffix used to fall to 1.0 = raw-as-USD.
    DORO.ST would read ~$100M instead of ~$9.5M -- a wrong number wearing a right label,
    in the one state where we explicitly decided not to guess."""
    cdx = pd.DataFrame({'source': ['DORO.ST'], 'marketCap': [962e6],
                        'reportedCurrency': [None]})
    co.mark_fx_unavailable('dead')
    got = co.marketcap_usd_series(cdx, allow_suffix_fallback=True)
    assert not got.notna().any(), 'the dead-feed decision was reversed by the fallback'
    assert co._suffix_fx_to_usd('DORO.ST') is None


# --------------------------------------------------------------------------- #
#  POINT-IN-TIME FX (the backtest)                                            #
# --------------------------------------------------------------------------- #
def _pit():
    """A DAILY-shaped fixture, like the real artifact (~2,050 closes / 7.6 yrs): the
    forward bound (PIT_MAX_FORWARD_DAYS) is only meaningful against realistic spacing."""
    return fx.PitFxTable(pd.DataFrame({
        'date': ['2021-03-30', '2021-03-31', '2026-06-29', '2026-06-30'],
        'currency': ['KRW'] * 4,
        'rate': [0.00093, 0.00092, 0.00071, 0.0007103672]}))


def test_the_pit_table_takes_the_last_close_ON_OR_BEFORE_the_date():
    t = _pit()
    assert t.rate_for('KRW', '2021-03-31') == pytest.approx(0.00092)
    assert t.rate_for('KRW', '2021-04-01') == pytest.approx(0.00092)
    assert t.rate_for('KRW', '2021-03-30') == pytest.approx(0.00093)


def test_the_pit_table_NEVER_reaches_BACKWARD_before_its_series():
    """Reaching outside the series is precisely the look-ahead this table exists to
    remove."""
    assert _pit().rate_for('KRW', '2019-01-01') is None


def test_the_pit_table_does_NOT_carry_the_last_close_FORWARD_FOREVER():
    """F-4 (reviewer): it used to answer a 2035 date with the last close it had, which is
    a stale rate wearing a point-in-time label.  A few days of carry-forward is correct
    (weekends, holidays, a pull a day behind); a year is not."""
    t = _pit()
    assert t.rate_for('KRW', '2026-07-05') == pytest.approx(0.0007103672)   # 5d, fine
    assert t.rate_for('KRW', '2035-01-01') is None
    assert t.rate_for('KRW', '2026-12-31') is None
    #  the boundary itself
    assert t.rate_for('KRW', '2026-07-31') == pytest.approx(0.0007103672)   # 31d
    assert t.rate_for('KRW', '2026-08-01') is None                          # 32d


def test_a_PARTIAL_pit_pull_is_visible_rather_than_silent():
    """F-4 (reviewer): a table holding 3 of 38 currencies loads without complaint and reads
    as point-in-time while making 35 currencies unknown."""
    t = _pit()
    have, total, frac = t.coverage()
    assert have == 1 and total == len(fx.supported_currencies())
    assert frac < 0.05
    lo, hi = t.span()
    assert str(lo.date()) == '2021-03-30' and str(hi.date()) == '2026-06-30'


def test_an_unknown_currency_or_date_resolves_to_None_not_to_spot():
    t = _pit()
    assert t.rate_for('SEK', '2021-03-01') is None
    assert t.rate_for('KRW', None) is None
    assert t.rate_for(None, '2021-03-01') is None


def test_a_pit_conversion_uses_EACH_ROWS_OWN_DATE_not_todays_rate():
    """THE point of item 7.  A 2021 market cap must be banded by the 2021 rate."""
    cdx = pd.DataFrame({'source': ['A', 'A'],
                        'date': ['2021-03-31', '2026-06-30'],
                        'marketCap': [1e12, 1e12],
                        'reportedCurrency': ['KRW', 'KRW']})
    got = co.marketcap_usd_series(cdx, fx=_pit())
    assert float(got.iloc[0]) == pytest.approx(1e12 * 0.00092)
    assert float(got.iloc[1]) == pytest.approx(1e12 * 0.0007103672)
    assert float(got.iloc[0]) != pytest.approx(float(got.iloc[1]))
    #  and the spot path really would have used one rate for both
    flat = co.marketcap_usd_series(cdx)
    assert float(flat.iloc[0]) == pytest.approx(float(flat.iloc[1]))


def test_marketcap_usd_by_source_forwards_the_pit_table_with_as_of():
    """`as_of` picks WHICH market cap; the PIT table picks WHICH RATE.  Both, or the fix is
    half-done."""
    cdx = pd.DataFrame({'source': ['A', 'A'],
                        'date': ['2021-03-31', '2026-06-30'],
                        'marketCap': [1e12, 5e12],
                        'reportedCurrency': ['KRW', 'KRW']})
    got = co.marketcap_usd_by_source(cdx, as_of='2021-12-31', fx=_pit())
    assert got['A'] == pytest.approx(1e12 * 0.00092)


def test_a_row_with_no_usable_date_is_unknown_under_a_pit_table():
    cdx = pd.DataFrame({'source': ['A'], 'date': [None], 'marketCap': [1e12],
                        'reportedCurrency': ['KRW']})
    assert not co.marketcap_usd_series(cdx, fx=_pit()).notna().any()


def test_the_historical_pull_derives_minor_units_and_never_fetches_them():
    """Same trap as spot, on the dated path: pence must come from GBP/100."""
    def _get(url, params=None, headers=None, timeout=None):
        assert 'GBpUSD' not in url, 'a minor unit was FETCHED'
        return _Resp([{'symbol': 'GBPUSD',
                       'historical': [{'date': '2021-01-04', 'close': 1.3672},
                                      {'date': '2021-01-05', 'close': 1.3600}]}])

    got = fx.fetch_historical_rates('http://b/', 'KEY', currencies=['GBP', 'GBp', 'USD'],
                                    _get=_get, verbose=False, sleep=lambda _s: None)
    gbp = got[got['currency'] == 'GBP']['rate'].tolist()
    gbx = got[got['currency'] == 'GBp']['rate'].tolist()
    assert gbp == [pytest.approx(1.3672), pytest.approx(1.3600)]
    assert gbx == [pytest.approx(0.013672), pytest.approx(0.0136)]
    assert set(got[got['currency'] == 'USD']['rate']) == {1.0}


def test_load_pit_rates_returns_None_rather_than_guessing(tmp_path):
    """None is what lets the caller SAY which basis it used, instead of silently
    substituting spot for a point-in-time number."""
    assert fx.load_pit_rates(outdir=str(tmp_path)) is None


def test_the_pit_round_trip_survives_the_csv(tmp_path):
    df = fx.fetch_historical_rates(
        'http://b/', 'KEY', currencies=['KRW'], verbose=False, sleep=lambda _s: None,
        _get=lambda *a, **k: _Resp([{'symbol': 'KRWUSD', 'historical': [
            {'date': '2022-01-03', 'close': 0.00083}]}]))
    fx.write_historical_csv(df, '2022-01-01', '2022-01-31', outdir=str(tmp_path))
    table = fx.load_pit_rates(outdir=str(tmp_path))
    assert table.rate_for('KRW', '2022-01-10') == pytest.approx(0.00083)
    assert table.rate_for('KRW', '2022-06-30') is None, 'forward bound'


# --------------------------------------------------------------------------- #
#  Wiring -- the fix must be IN the pipeline, not merely available            #
# --------------------------------------------------------------------------- #
def test_Sbocker_installs_the_feed_BEFORE_the_fetch():
    """At launch, not at use: a dead FX feed must be visible before 12 hours are spent."""
    import inspect
    import Sbocker
    src = inspect.getsource(Sbocker.main)
    assert 'fxr.install_for_run(' in src
    assert src.index('install_for_run') < src.index('get_fundamentals_fmp')


def test_Sbocker_marks_FX_unavailable_if_the_install_itself_raises():
    """The one path that could leave production on the undated constants."""
    import inspect
    import Sbocker
    src = inspect.getsource(Sbocker.main)
    i = src.index('fxr.install_for_run(')
    assert 'mark_fx_unavailable' in src[i:i + 1500]


def test_the_band_grading_asks_for_PIT_FX_and_states_which_basis_it_used():
    import inspect
    import sys as _sys
    _bt = os.path.join(_HERE, 'baseline_tools')
    if _bt not in _sys.path:
        _sys.path.insert(0, _bt)
    import pipeline_analysis as pa
    src = inspect.getsource(pa._per_band_beat_rate)
    assert 'load_pit_rates' in src
    assert 'fx=fx_pit' in src
    assert 'FX basis' in src


def test_the_stale_snapshot_TODO_is_gone_and_the_constants_say_what_they_are_now():
    """The table is the SANITY BAND now; a reader must not find a TODO telling them to go
    wire the thing that is already wired."""
    src = open(os.path.join(_HERE, 'carveOut.py'), encoding='utf-8').read()
    head = src[:src.index('FX_TO_USD = {')]
    assert 'TODO: wire a live/dated FX source' not in src
    assert 'SANITY BAND' in head


def test_the_TWO_CURRENCIES_warning_sits_next_to_the_table():
    """`reportedCurrency` (statement) vs the profile `currency` (trading) differ for every
    ADR and cross-listing -- SHEL.L quotes in pence and reports USD.  Wiring the trading
    currency here would reintroduce the unit mismatch that halted the liquidity floor, and
    the existing note lives in findAllSectors.py, which is not where an FX author looks."""
    src = open(os.path.join(_HERE, 'carveOut.py'), encoding='utf-8').read()
    head = src[:src.index('FX_TO_USD = {')]
    assert 'TWO' in head and 'reportedCurrency' in head
    assert 'SHEL.L' in head
    assert 'volavgdic' in head or 'findAllSectors' in head
    assert 'pbRatio' in head, 'the panel-wide proof that marketCap is in reportedCurrency'
