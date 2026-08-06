"""Two CEO-decided changes of 2026-08-06, pinned.

REGISTER D-5 -- `marketCapRevQuants` is ABSOLUTE market-cap bands, not pool quartiles.
The properties tested here are the ones that were LOAD-BEARING in the decision, so each is
a guard against a specific way of getting it wrong later:
  * SIX bands, an EVEN count -- an odd count puts a middle band at exactly 0.0, colliding
    with MCAP_QUANT_MISSING, and a mid-cap would become indistinguishable from a name with
    no market cap at all;
  * the range is exactly [-0.5, +0.5] -- widening it would be a weight increase in disguise
    (w = 0.049681 is pinned in test_e2_weight_vector.py and is NOT changed);
  * POOL-INVARIANT -- the same company scores the same in any pool, which is the whole point
    of the change (its score no longer moves because OTHER companies moved);
  * absent / non-positive market cap is NEUTRAL, not the maximum small-cap reward;
  * GATED ON THE REAL `reportedCurrency` (CEO 2026-08-06) -- absolute edges do not cancel a
    systematic currency error the way a pool quartile partly did, so where the reporting
    currency is unknown the name is UNSCORED rather than banded on a coarse exchange-suffix
    guess or on the raw mixed-currency field.  The band-table tests therefore run on a
    currency-PRESENT fixture and the gate has its own section; that split is deliberate --
    re-expressing the table tests as tests of the neutral path would lose the proof that
    every band is reachable.

REGISTER J-1 -- average volume is REPORTED, never screened on.  The tests assert the
report-only property directly (membership and order are byte-identical with and without a
volume map) and that absence reads as absence rather than as zero.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carveOut as co
import stage2_metrics as sm


def _cdx(caps, currency='USD'):
    """A minimal cdx-shaped frame WITH a real `reportedCurrency`, which is what the band is
    gated on (CEO 2026-08-06): the band table can only be exercised on currency-present
    data, because absent currency is now NEUTRAL by design.  Every pickle on disk today
    LACKS the field -- that state is tested separately, by the gate tests below, and the
    band-table tests must not be quietly re-expressed as tests of the neutral path."""
    return pd.DataFrame({'source': [f'S{i}' for i in range(len(caps))],
                         'marketCap': caps,
                         'reportedCurrency': [currency] * len(caps)})


def _cdx_no_currency(caps):
    """The pre-fetch shape: no `reportedCurrency` column at all."""
    return pd.DataFrame({'source': [f'S{i}' for i in range(len(caps))],
                         'marketCap': caps})


# --------------------------------------------------------------------------- #
#  D-5: the band table                                                        #
# --------------------------------------------------------------------------- #
def test_band_count_is_even_so_no_band_collides_with_the_missing_sentinel():
    assert len(sm.MCAP_BAND_SCORES) == 6
    assert len(sm.MCAP_BAND_SCORES) % 2 == 0, 'an odd count puts a band at exactly 0.0'
    assert len(sm.MCAP_BAND_EDGES_USD) == len(sm.MCAP_BAND_SCORES) - 1
    #  THE COLLISION ITSELF: no band may equal the missing sentinel.
    assert sm.MCAP_QUANT_MISSING not in sm.MCAP_BAND_SCORES
    assert len(set(sm.MCAP_BAND_SCORES)) == 6, 'two bands scoring alike is a lost level'


def test_band_range_is_exactly_the_pinned_half_unit_so_the_weight_still_means_what_it_meant():
    assert max(sm.MCAP_BAND_SCORES) == pytest.approx(0.5, abs=1e-12)
    assert min(sm.MCAP_BAND_SCORES) == pytest.approx(-0.5, abs=1e-12)


def test_scores_decrease_monotonically_with_size():
    s = list(sm.MCAP_BAND_SCORES)
    assert s == sorted(s, reverse=True), 'smaller caps must score higher'


def test_edges_are_half_decades():
    """Equal LOG widths, because size / neglect effects are log-linear.  Half-decade steps
    alternate x3 and x10/3, so every consecutive log-ratio is within 5% of half a decade."""
    e = np.asarray(sm.MCAP_BAND_EDGES_USD, dtype=float)
    steps = np.diff(np.log10(e))
    assert np.allclose(steps, 0.5, atol=0.03), steps


@pytest.mark.parametrize('cap, want', [
    (5e6, 0.5),           # deep micro
    (99.9e6, 0.5),        # just under the first edge
    (100e6, 0.3),         # ON the edge -> the HIGHER band (bands are left-closed)
    (299e6, 0.3),
    (300e6, 0.1),
    (999e6, 0.1),
    (1e9, -0.1),
    (2.99e9, -0.1),
    (3e9, -0.3),
    (9.99e9, -0.3),
    (10e9, -0.5),         # ">= $10B"
    (2e12, -0.5),         # a mega-cap is not worse than -0.5
])
def test_each_band_maps_to_its_spec_score(cap, want):
    got = sm.add_mcap_quants(_cdx([cap]))
    assert float(got.iloc[0]) == pytest.approx(want, abs=1e-12), (cap, float(got.iloc[0]))


def test_every_band_is_reachable():
    """All six scores must be ATTAINABLE -- a score level no market cap can produce is a
    reward that can never be given (the reason an 8-band variant was rejected)."""
    caps = [50e6, 200e6, 500e6, 2e9, 5e9, 50e9]
    got = sorted(float(v) for v in sm.add_mcap_quants(_cdx(caps)))
    assert got == sorted(sm.MCAP_BAND_SCORES), got


# --------------------------------------------------------------------------- #
#  D-5: pool-invariance -- the defect the change exists to remove             #
# --------------------------------------------------------------------------- #
def test_a_company_score_does_not_move_when_other_companies_move():
    """Under `pd.qcut` this was FALSE by construction: the same company scored differently
    in a different pool.  It is the half of D-5's complaint that absolute bands close."""
    alone = float(sm.add_mcap_quants(_cdx([200e6])).iloc[0])
    with_giants = sm.add_mcap_quants(_cdx([200e6] + [50e9] * 40))
    with_minnows = sm.add_mcap_quants(_cdx([200e6] + [1e6] * 40))
    assert alone == pytest.approx(0.3, abs=1e-12)
    assert float(with_giants.iloc[0]) == pytest.approx(alone, abs=1e-12)
    assert float(with_minnows.iloc[0]) == pytest.approx(alone, abs=1e-12)


def test_a_duplicated_row_cannot_shift_another_company_band():
    """The row-vs-company cut (which gave the 100-name pool a lopsided 30/31/19/20) is now
    irrelevant: repeating one issuer's rows changes nobody else's score."""
    caps = [50e6, 200e6, 500e6, 2e9, 5e9, 50e9]
    base = sm.add_mcap_quants(_cdx(caps))
    padded = sm.add_mcap_quants(_cdx(caps + [50e9] * 60))
    assert list(padded.iloc[:len(caps)].astype(float)) == list(base.astype(float))


def test_range_is_intact_on_a_degenerate_pool():
    """`duplicates='drop'` used to narrow the quartile range on a clustered pool.  Absolute
    edges cannot collapse, so one row still scores its own band's full value."""
    assert float(sm.add_mcap_quants(_cdx([1e6])).iloc[0]) == pytest.approx(0.5, abs=1e-12)
    same = sm.add_mcap_quants(_cdx([2e9] * 25))
    assert set(float(v) for v in same) == {-0.1}


# --------------------------------------------------------------------------- #
#  D-5: absence stays neutral                                                 #
# --------------------------------------------------------------------------- #
def test_missing_and_non_positive_market_cap_are_neutral_not_best_in_pool():
    got = sm.add_mcap_quants(_cdx([np.nan, 0.0, -5.0, 'junk', 200e6]))
    assert [float(v) for v in got.iloc[:4]] == [sm.MCAP_QUANT_MISSING] * 4
    assert float(got.iloc[4]) == pytest.approx(0.3, abs=1e-12)
    #  and neutral must be STRICTLY WORSE than the most-rewarded real band
    assert sm.MCAP_QUANT_MISSING < max(sm.MCAP_BAND_SCORES)


def test_the_column_carries_seven_levels_at_most():
    """Six bands + the missing sentinel = 7, the figure the `_rank_normal` notes state."""
    got = sm.add_mcap_quants(_cdx([50e6, 200e6, 500e6, 2e9, 5e9, 50e9, np.nan]))
    assert len(set(float(v) for v in got)) == 7


def test_index_is_preserved_so_the_series_aligns_with_the_pool():
    cdx = _cdx([50e6, 5e9])
    cdx.index = [11, 22]
    got = sm.add_mcap_quants(cdx)
    assert list(got.index) == [11, 22]


# --------------------------------------------------------------------------- #
#  D-5: the CURRENCY GATE -- no reportedCurrency, no band (CEO 2026-08-06)     #
# --------------------------------------------------------------------------- #
#  Absolute edges do not cancel a systematic currency error the way a pool quartile partly
#  did, so the band is gated on the REAL reporting currency and guesses NOTHING.  These
#  tests assert the gate itself; the band-table tests above run on currency-present data.
def test_no_reported_currency_means_every_name_is_neutral_not_banded():
    """The state of every pickle on disk: the whole column goes NEUTRAL rather than banding
    a mixed-currency field against absolute USD edges."""
    caps = [50e6, 200e6, 500e6, 2e9, 5e9, 50e9]
    got = sm.add_mcap_quants(_cdx_no_currency(caps))
    assert set(float(v) for v in got) == {sm.MCAP_QUANT_MISSING}
    #  and nothing is MISBANDED -- no real band score is emitted at all
    assert not set(float(v) for v in got) & set(sm.MCAP_BAND_SCORES)


def test_the_raw_mixed_currency_field_is_never_banded():
    """THE defect the gate closes.  13e12 KRW is ~$9.5B (a -0.3 name); read raw it is
    '13 trillion' and lands in the >= $10B band at -0.5.  With no currency it must be
    NEUTRAL -- not -0.5, and not the raw fallback the function used to take."""
    krw_amount = 13e12
    got = float(sm.add_mcap_quants(_cdx_no_currency([krw_amount])).iloc[0])
    assert got == pytest.approx(sm.MCAP_QUANT_MISSING, abs=1e-12), got
    #  with the currency KNOWN it bands on its true USD size instead
    banded = float(sm.add_mcap_quants(_cdx([krw_amount], currency='KRW')).iloc[0])
    assert krw_amount * co.FX_TO_USD['KRW'] == pytest.approx(9.49e9, rel=0.02)
    assert banded == pytest.approx(-0.3, abs=1e-12), banded


def test_the_exchange_suffix_alone_cannot_band_a_name():
    """The suffix is a PRIOR on reporting currency, not a fact (FRES.L reports USD; the .L
    IOB lines are foreign issuers), so the size tilt no longer opts into it."""
    cdx = pd.DataFrame({'source': ['005930.KS', 'DORO.ST'], 'marketCap': [13e12, 962e6]})
    got = sm.add_mcap_quants(cdx)
    assert set(float(v) for v in got) == {sm.MCAP_QUANT_MISSING}


def test_an_unknown_currency_code_is_unknown_not_raw():
    got = float(sm.add_mcap_quants(_cdx([13e12], currency='XYZ')).iloc[0])
    assert got == pytest.approx(sm.MCAP_QUANT_MISSING, abs=1e-12)


def test_a_non_usd_reporter_bands_on_its_usd_size_not_its_local_number():
    """SEK: 962M SEK is ~$91M = the top band (+0.5); read raw it would be +0.3."""
    got = float(sm.add_mcap_quants(_cdx([962e6], currency='SEK')).iloc[0])
    assert got == pytest.approx(0.5, abs=1e-12), got


def test_the_gate_agrees_with_the_band_selection_gate():
    """CONSISTENCY, the reason this was worth changing: `partition_by_marketcap` already
    skipped on `currency_data_present`, and this metric was the one place still guessing.
    The two must now key off the same condition."""
    caps = [50e6, 2e9]
    pending = _cdx_no_currency(caps)
    ready = _cdx(caps)
    assert co.currency_data_present(pending) is False
    assert co.currency_data_present(ready) is True
    assert set(float(v) for v in sm.add_mcap_quants(pending)) == {sm.MCAP_QUANT_MISSING}
    assert set(float(v) for v in sm.add_mcap_quants(ready)) != {sm.MCAP_QUANT_MISSING}


def test_a_materialized_marketCap_usd_column_still_bands():
    """The post-fetch pickle carries `marketCap_usd` beside the raw field (getData_fmp), so
    a frame with no reportedCurrency but a real USD column is NOT pending -- it bands."""
    cdx = _cdx_no_currency([np.nan])
    cdx['marketCap_usd'] = [2e9]
    assert co.currency_data_present(cdx) is True
    assert float(sm.add_mcap_quants(cdx).iloc[0]) == pytest.approx(-0.1, abs=1e-12)


def test_the_neutral_column_is_constant_which_the_normalizer_must_absorb():
    """The pre-fetch consequence, pinned: an all-neutral column has ZERO variance, and the
    normalizer's `sigma == 0 -> 1.0` guard is what stops it dividing into +-inf.  If that
    guard is ever removed this test is the one that says why it mattered."""
    import postBoRank as pbr
    got = sm.add_mcap_quants(_cdx_no_currency([50e6, 200e6, 2e9, 50e9]))
    est = pbr.robust_location_scale(got)
    assert est.status == 'constant'
    z = pbr._rank_normal(got)
    assert np.isfinite(z.to_numpy()).all(), 'a constant column must not produce inf/NaN z'
    assert set(np.round(z.to_numpy(), 12)) == {0.0}, 'no name may gain from absence'


# --------------------------------------------------------------------------- #
#  J-1: volume is REPORTED, never screened on                                 #
# --------------------------------------------------------------------------- #
def test_absent_symbol_reads_as_not_captured_not_as_zero():
    out = co.volavg_report_frame(['AAA', 'BBB'], volavg_map={})
    assert out['volAvg_report'].isna().all(), 'an absent reading must NOT be 0'
    assert list(out['volAvg_asof']) == [co.VOLAVG_STATUS_NOT_CAPTURED] * 2


def test_null_or_zero_reading_is_no_reading_not_a_liquidity_of_zero():
    vmap = {'A': (None, '2026-08-06'), 'B': (0.0, '2026-08-06'),
            'C': (float('nan'), '2026-08-06'), 'D': (-1.0, '2026-08-06')}
    out = co.volavg_report_frame(list('ABCD'), volavg_map=vmap)
    assert out['volAvg_report'].isna().all()
    assert set(out['volAvg_asof']) == {co.VOLAVG_STATUS_NO_READING}


def test_a_real_reading_carries_its_as_of_date():
    out = co.volavg_report_frame(['A'], volavg_map={'A': (123456.0, '2026-08-06')})
    assert float(out['volAvg_report'].iloc[0]) == 123456.0
    assert out['volAvg_asof'].iloc[0] == '2026-08-06'


def test_the_undated_pickle_shape_is_labelled_rather_than_left_blank():
    """A number with no date invites the stale-vs-fresh comparison the dedup term refuses to
    make; the marker is what makes the missing date visible."""
    out = co.volavg_report_frame(['A'], volavg_map={'A': (123456.0, None)})
    assert float(out['volAvg_report'].iloc[0]) == 123456.0
    assert out['volAvg_asof'].iloc[0] == co.VOLAVG_STATUS_UNDATED


def test_every_row_gets_a_status_so_no_cell_is_ambiguous():
    vmap = {'A': (10.0, '2026-08-06')}
    out = co.volavg_report_frame(['A', 'B'], volavg_map=vmap)
    assert len(out) == 2
    assert out['volAvg_asof'].notna().all() and (out['volAvg_asof'] != '').all()


def test_report_frame_aligns_one_row_per_requested_symbol_including_repeats():
    out = co.volavg_report_frame(['A', 'A', 'B'], volavg_map={'A': (5.0, 'd')})
    assert len(out) == 3
    assert list(out['volAvg_report'].isna()) == [False, False, True]


def test_reporting_volume_cannot_reorder_or_drop_anything():
    """THE report-only assertion.  The function returns values only -- it never sees the
    ranking -- so the check is that the caller's own list is untouched and that no
    filtering is possible through this API."""
    syms = ['A', 'B', 'C', 'D']
    before = list(syms)
    rich = co.volavg_report_frame(syms, volavg_map={'A': (1.0, 'd'), 'B': (1e9, 'd')})
    poor = co.volavg_report_frame(syms, volavg_map={})
    assert syms == before
    assert len(rich) == len(poor) == len(syms)          # no name can be excluded
    assert list(rich.columns) == list(poor.columns) == ['volAvg_report', 'volAvg_asof']


def test_the_dedup_tiebreak_is_not_fed_by_the_report_path():
    """The tiebreak reads the map directly and is unchanged: a group inside one order of
    magnitude still ABSTAINS, whatever the report columns say."""
    vmap = {'X': (1.0e6, 'd'), 'X-P': (9.0e5, 'd')}
    assert co._volavg_liquidity_term('X', ['X', 'X-P'], vmap) == 0
    assert co._volavg_liquidity_term('X-P', ['X', 'X-P'], vmap) == 0
    #  a genuine decade gap still speaks -- i.e. the tiebreak was not disturbed
    vmap2 = {'X': (1.0e6, 'd'), 'X-P': (1.0e4, 'd')}
    assert co._volavg_liquidity_term('X', ['X', 'X-P'], vmap2) == 0
    assert co._volavg_liquidity_term('X-P', ['X', 'X-P'], vmap2) == 1
