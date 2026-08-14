"""EVERY COMPUTED QUANTITY MUST BE INVARIANT TO `-nrperiods`, OR SAY WHY NOT.

WHAT THESE PIN (fetch-depth audit, 2026-08-14).  The next fetch moves from `-nrperiods 24` to
`-nrperiods 80` -- deep history for BACKTEST ANCHORING, at ZERO extra API cost (probed live
2026-08-13: `limit=80` returns 80 quarters, `limit=120` returns 120, on the SAME statement
calls).  The CEO's requirement is that fetch DEPTH and metric WINDOW be DECOUPLED: a name's
score must not move because we fetched more rows.

`baseline_tools/depth_sensitivity.py` measured the whole pipeline on the real 2026-08-13 CUR3K
panel (2,629 sources, 1,884 quarterly) re-scored at depth 24 vs a depth-80 extension of the SAME
panel.  EVERY Stage-2 metric, the Montier-C means, the Sloan accruals and the seven mean-bar pass
rates came out BIT-IDENTICAL.  Exactly four things moved:

  1. `CycleHeat` / `EPStoEPSmean` -- DECLARED: their window is 28 quarters, longer than today's
     depth, so it cannot saturate at 24.  Already pinned by test_cycleheat_window.py and
     test_eps_mean_window.py.  Nothing here.
  2. The Stage-1 HISTORY BONUS -- CENSORED at 24, not censored at 80.  Pinned below.
  3. `calcMetrics.peg_pool_median_growth` -- a POOLED CROSS-SECTIONAL bar taken over the WHOLE
     panel, so its value WAS the fetch depth.  This is the one that moved a SCORED criterion
     (PEG, Tier C, w = 0.30): 7.0% of sources changed BoScore through it once the history bonus
     was held fixed.  FIXED by PEG_POOL_WINDOW_NQ; pinned below.
  4. `postBo.moatIdentifier` -- four comparators filtered their inadmissible rows out and THEN
     took head(n), so the window slid backwards without bound.  FIXED to mask-in-place; pinned
     below.

AND ONE NON-FINDING WORTH PINNING SO IT STAYS ONE: `calcScore.getAves2`'s `BoMetric_ave` pooled
median moves with depth on 32 of 41 columns -- and reaches NO score, because `meanBars.mean_bar`
serves a STORED CONSTANT for every weighted mean criterion.  The single declared exception
(`mSalesToMarketCap`) is Tier 'N', w = 0.  The residual measured after pinning the bonus and the
PEG median was EXACTLY ZERO on 2,629 sources, which is the evidence.  A test below fails the
moment that exception is given a weight.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calcMetrics as cm
import calcScore as cs
import createDicts as cdic
import meanBars as mb
import reporting_period as rp
import stage2_metrics as sm


# --------------------------------------------------------------------------- #
#  Panel builders                                                             #
# --------------------------------------------------------------------------- #
def _cdx(n, months=3, eps_path=None, price=100.0, source='AAA', start='2026-01-01'):
    """`n` rows of one source, NEWEST-FIRST -- `peg_local`'s contract (its `.shift(-k)`
    reaches OLDER rows).  `eps_path` is newest-first too."""
    dates = pd.date_range(end=pd.Timestamp(start), periods=n, freq='-%dMS' % months)
    eps = (np.asarray(eps_path, dtype=float) if eps_path is not None
           else np.full(n, 1.0))
    return pd.DataFrame({'date': dates, 'source': source,
                         'netIncomePerShare': eps, 'price': np.full(n, float(price))})


def _deepen(df, extra, months=3, filler=None):
    """`df` with `extra` OLDER rows appended (it is newest-first, so older rows go at the
    END).  The filler EPS is deliberately DIFFERENT from the recent history: if a window is
    honestly capped, no filler value can reach it."""
    oldest = pd.to_datetime(df['date']).min()
    dates = [oldest - pd.DateOffset(months=months * (k + 1)) for k in range(extra)]
    tail = pd.DataFrame({'date': dates, 'source': df['source'].iloc[0],
                         'netIncomePerShare': np.full(extra, float(
                             filler if filler is not None else -50.0)),
                         'price': np.full(extra, float(df['price'].iloc[0]))})
    return pd.concat([df, tail], ignore_index=True)


# =========================================================================== #
#  1.  THE PEG POOLED BAR                                                      #
# =========================================================================== #
def test_peg_pool_median_is_INVARIANT_to_fetch_depth():
    """The headline. Same newest rows, 56 more OLD rows -> the SAME bar."""
    shallow = _cdx(24, eps_path=np.linspace(3.0, 1.0, 24))
    deep = _deepen(shallow, 56, filler=0.05)     # a two-decade tail of near-zero earnings
    med_s, n_s = cm.peg_pool_median_growth(shallow, freq_map={'AAA': rp.QUARTERLY})
    med_d, n_d = cm.peg_pool_median_growth(deep, freq_map={'AAA': rp.QUARTERLY})
    assert n_s == n_d, (n_s, n_d)
    assert med_s == pytest.approx(med_d, abs=0.0, rel=0.0), (med_s, med_d)


def test_the_UNCAPPED_pool_WOULD_have_moved__so_the_test_above_is_not_vacuous():
    """`window_nq=0` restores the pre-fix whole-panel pool.  If THAT does not move on this
    fixture, the test above proves nothing about the cap -- it would only prove the fixture is
    flat.  This is the counterexample half."""
    shallow = _cdx(24, eps_path=np.linspace(3.0, 1.0, 24))
    deep = _deepen(shallow, 56, filler=0.05)
    med_s, n_s = cm.peg_pool_median_growth(shallow, freq_map={'AAA': rp.QUARTERLY},
                                           window_nq=0)
    med_d, n_d = cm.peg_pool_median_growth(deep, freq_map={'AAA': rp.QUARTERLY},
                                           window_nq=0)
    assert n_d > n_s, 'the deep panel must contribute more rows to an uncapped pool'
    assert med_s != pytest.approx(med_d), (med_s, med_d)


def test_the_pool_window_is_STAGE_1s_OWN_SCORING_WINDOW():
    """The bar must describe the population the criterion is actually confronted with.
    Stage-1 scores `calcByTier`'s head(n), n = 8 (`nrScorePeriods` default), so the pool
    window is 8 -- the same argument `meanBars._newest_window` makes for the mean bars."""
    from configuration import getDataFetchConfiguration  # noqa  (import cost only)
    assert cm.PEG_POOL_WINDOW_NQ == 8
    assert mb.calibrate.__defaults__[0] == 8, \
        'meanBars.calibrate window_rows and PEG_POOL_WINDOW_NQ state the same window'


def test_the_pool_window_IS_frequency_scaled_where_stage1s_head_n_is_not():
    """A CROSS-SECTIONAL pool must span the same CALENDAR time for both frequencies, or ~14%
    of the universe puts twice the history into a bar the other 86% also faces.  That is the
    opposite of ruling Q2's decision for Stage-1's own head(n), and deliberately so -- the
    divergence is argued at PEG_POOL_WINDOW_NQ."""
    assert rp.scale_window(cm.PEG_POOL_WINDOW_NQ, 4) == 8
    assert rp.scale_window(cm.PEG_POOL_WINDOW_NQ, 2) == 4      # 4 halves == 8 quarters == 2y


def test_a_semiannual_source_contributes_the_SAME_CALENDAR_SPAN_as_a_quarterly_one():
    q = _cdx(24, months=3, eps_path=np.linspace(3.0, 1.0, 24), source='Q')
    s = _cdx(24, months=6, eps_path=np.linspace(3.0, 1.0, 24), source='S')
    _m, n_q = cm.peg_pool_median_growth(q, freq_map={'Q': rp.QUARTERLY})
    _m, n_s = cm.peg_pool_median_growth(s, freq_map={'S': rp.SEMIANNUAL})
    #  8 quarterly rows vs 4 semi-annual rows, minus each one's own YoY structural lag.
    assert n_q > n_s, (n_q, n_s)
    assert n_s == pytest.approx(n_q / 2.0, abs=1.0), (n_q, n_s)


def test_the_pool_window_is_taken_BEFORE_the_dropna__so_it_cannot_reach_past_itself():
    """A `dropna().head(w)` window slides backwards until it finds w computable rows, which is
    exactly the unbounded shape this whole audit is about.  Here the newest 8 rows are all
    out-of-domain (negative EPS), so the honest answer is "this source contributes NOTHING",
    not "reach back to 2006 for eight usable ones"."""
    bad_recent = np.concatenate([np.full(12, -1.0), np.linspace(3.0, 1.0, 12)])
    df = _cdx(24, eps_path=bad_recent)
    _med, n = cm.peg_pool_median_growth(df, freq_map={'AAA': rp.QUARTERLY})
    assert n == 0, 'the window must not reach past its own bound to find in-domain rows'


# =========================================================================== #
#  2.  THE STAGE-1 HISTORY BONUS                                               #
# =========================================================================== #
def test_history_bonus_is_CENSORED_at_nrperiods_24_and_NOT_at_80():
    """The exact arithmetic behind the one-off score shift, so the number in the report is
    reproducible.  `build_bometric_rows` trims the oldest `rpy` rows, so `-nrperiods 24`
    leaves a fetch-capped quarterly filer with 20 BoMetric rows."""
    at_24 = cs.history_bonus_censored_by(24 - 4)
    at_80 = cs.history_bonus_censored_by(80 - 4)
    assert at_80 is None, 'depth 80 must be able to saturate the bonus'
    assert at_24 == pytest.approx(0.035355, abs=1e-6), at_24
    assert cs.HISTORY_BONUS_MAX - at_24 == pytest.approx(0.014645, abs=1e-6)


def test_the_censored_ceiling_is_FREQUENCY_DEPENDENT_because_the_trim_is_rpy():
    """H-1(a), pinned.  The trim is `rpy`, NOT 4, so a fetch-capped SEMI-ANNUAL filer keeps 22
    rows where a quarterly one keeps 20 -- a HIGHER ceiling and therefore a SMALLER step when
    the fetch deepens.  583 of the 2,629 sources on the 2026-08-13 CUR3K panel (22.2%) are in
    this state, so "every capped name moves by 0.014645" was wrong on a fifth of the panel."""
    q = cs.history_bonus_censored_by(24 - 4)          # quarterly: 20 rows
    s = cs.history_bonus_censored_by(24 - 2)          # semi-annual: 22 rows
    assert q == pytest.approx(0.035355, abs=1e-6)
    assert s == pytest.approx(0.037081, abs=1e-6)
    assert cs.HISTORY_BONUS_MAX - s == pytest.approx(0.012919, abs=1e-6)
    assert s > q, 'the semi-annual ceiling is HIGHER, so its step to saturation is SMALLER'


@pytest.mark.parametrize('own_quarters,expected_delta', [
    (24, 0.000000),      # nothing more to fetch -- a deeper fetch gives this name NOTHING
    (28, 0.003374),
    (32, 0.006478),
    (36, 0.009366),
    (40, 0.012079),
    (44, 0.014645),      # saturated
    (80, 0.014645),      # still saturated -- deeper buys nothing beyond 44
])
def test_the_deep_fetch_step_is_a_CONTINUOUS_function_of_LISTING_AGE_not_a_common_offset(
        own_quarters, expected_delta):
    """H-1(b), and it is the half that matters.

    THE ERROR THIS PINS AGAINST.  `depth_sensitivity.extend_history` pads every eligible source
    to EXACTLY `target_rows`, i.e. it models a universe where every name capped at 24 has >= 80
    quarters of real history.  A real fetch returns `min(depth, own history)`.  So the measured
    "uniform +0.014645" was a property of the HARNESS, not of the fetch -- and every name listed
    between ~6 and ~11 years gets a DIFFERENT step, spanning the full bonus range, inside a
    cohort that is perfectly flat today.  The bonus is SUPPOSED to discriminate on history; what
    was wrong was the claim that the deep run's shift cancels out of a comparison.
    """
    rpy = 4
    rows_24 = min(24, own_quarters) - rpy
    rows_80 = min(80, own_quarters) - rpy
    b = lambda r: cs.HISTORY_BONUS_MAX * math.sqrt(
        min(r, cs.HISTORY_BONUS_SATURATION_ROWS) / cs.HISTORY_BONUS_SATURATION_ROWS)
    assert b(rows_80) - b(rows_24) == pytest.approx(expected_delta, abs=5e-7)


def test_the_step_SPREAD_is_wide_enough_to_break_ties__which_is_why_it_is_not_a_common_offset():
    """The consequence, stated as a number: the spread across the re-ranked band is the FULL
    bonus, and this module's own scale note records that 90.9% of names tie today and break
    ALPHABETICALLY.  A spread of this size is decisive on those ties."""
    b = lambda r: cs.HISTORY_BONUS_MAX * math.sqrt(
        min(r, cs.HISTORY_BONUS_SATURATION_ROWS) / cs.HISTORY_BONUS_SATURATION_ROWS)
    steps = [b(min(80, q) - 4) - b(min(24, q) - 4) for q in range(24, 45, 4)]
    assert max(steps) - min(steps) == pytest.approx(0.014645, abs=1e-5)


def test_the_docstring_does_NOT_claim_a_uniform_shift():
    """The disclosure IS the deliverable here (H-1): this is the text the CEO will compare the
    deep run against, so the retracted claim is pinned out of the file rather than trusted to
    stay out."""
    doc = cs.history_bonus_censored_by.__doc__
    assert 'NOT A UNIFORM SHIFT' in doc
    assert '0.012919' in doc and '0.003374' in doc, 'the corrected numbers must be present'
    assert 'A COMMON shift over that whole cohort cannot re-rank it internally' not in doc


def test_the_bonus_saturates_at_or_above_the_declared_row_count_so_deeper_is_a_NO_OP():
    """Beyond saturation the bonus stops moving, which is what makes depth >= 44 rows
    genuinely fetch-depth INVARIANT rather than merely 'less censored'."""
    for rows in (40, 44, 76, 120, 500):
        assert cs.history_bonus_censored_by(rows) is None, rows


def test_the_bonus_is_still_a_TIEBREAK_and_not_a_criterion_at_the_deeper_depth():
    """HISTORY_BONUS_MAX is unchanged by the deep fetch, so the scale argument recorded at its
    use site still holds: the full bonus is HALF the smallest criterion difference (a Tier-D
    criterion over a full window = 0.1)."""
    assert cs.HISTORY_BONUS_MAX * 2 == pytest.approx(0.1)


# =========================================================================== #
#  3.  THE MOAT WINDOW -- MASK IN PLACE, NEVER FILTER-THEN-head(n)             #
# =========================================================================== #
def _moat_cdx(n, admissible_from, source='AAA'):
    """`n` newest-first rows where the newest `admissible_from` rows are INADMISSIBLE for the
    four masked comparators (grossProfit == 0, netIncomePerShare <= 0, pfcfRatio == 0) and
    everything older is admissible with a distinctive value."""
    dates = pd.date_range(end=pd.Timestamp('2026-01-01'), periods=n, freq='-3MS')
    bad = np.arange(n) < admissible_from
    return pd.DataFrame({
        'date': dates, 'source': source,
        'pfcfRatio': np.where(bad, 0.0, 10.0),
        'grossProfit': np.where(bad, 0.0, 100.0),
        'sellingGeneralAndAdministrativeExpenses': np.full(n, 5.0),
        'depreciationAndAmortization': np.full(n, 2.0),
        'netIncomePerShare': np.where(bad, -1.0, 4.0),
        'capexPerShare': np.full(n, 1.0),
        'grossProfitMargin': np.full(n, 0.5), 'revenue': np.full(n, 1000.0),
        'totalAssets': np.full(n, 2000.0), 'returnOnEquity': np.full(n, 0.05),
        'returnOnAssets': np.full(n, 0.03), 'returnOnCapitalEmployed': np.full(n, 0.04),
        'netProfitMargin': np.full(n, 0.1), 'totalLiabilities': np.full(n, 500.0),
        'totalStockholdersEquity': np.full(n, 1500.0)})


def test_moat_masked_comparators_do_NOT_reach_past_the_window():
    """THE DEFECT: `series[mask].head(20)` reached back until it found 20 SURVIVING rows.  Here
    the newest 20 rows are ALL inadmissible, so a correctly-bounded window has NOTHING to
    average and must return NaN -- while the old filter-then-head would have reached into the
    older rows and returned a real number."""
    import postBo as pb
    df = _moat_cdx(80, admissible_from=20)
    out = pb.moatIdentifier(pd.Series(['AAA']), df, n=20,
                            freq_map={'AAA': rp.QUARTERLY})
    row = out.iloc[0]
    for col in ('FCFyield', 'SGAtoGP', 'DeptoGP', 'CapExtoEarnings'):
        assert pd.isna(row[col]), (col, row[col])


def test_moat_masked_comparators_are_FETCH_DEPTH_INVARIANT():
    """The same name at two depths, same newest rows -> the same 0-11 count and the same four
    comparator values."""
    import postBo as pb
    shallow = _moat_cdx(24, admissible_from=4)
    deep = _moat_cdx(80, admissible_from=4)
    a = pb.moatIdentifier(pd.Series(['AAA']), shallow, n=20,
                          freq_map={'AAA': rp.QUARTERLY}).iloc[0]
    b = pb.moatIdentifier(pd.Series(['AAA']), deep, n=20,
                          freq_map={'AAA': rp.QUARTERLY}).iloc[0]
    assert a['moatScore'] == b['moatScore'], (a['moatScore'], b['moatScore'])
    for col in ('FCFyield', 'SGAtoGP', 'DeptoGP', 'CapExtoEarnings'):
        assert a[col] == pytest.approx(b[col], rel=0, abs=0), col


def test_moat_still_SKIPS_inadmissible_rows_inside_the_window():
    """The fix bounds the window; it must NOT change the admissibility rule.  With 4 bad rows
    inside a 20-row window the value is the mean over the 16 good ones -- not NaN, and not a
    value polluted by the bad rows."""
    import postBo as pb
    df = _moat_cdx(24, admissible_from=4)
    row = pb.moatIdentifier(pd.Series(['AAA']), df, n=20,
                            freq_map={'AAA': rp.QUARTERLY}).iloc[0]
    #  SGAtoGP = 0.15 - mean(5/100 over the 16 admissible rows) = 0.15 - 0.05
    assert row['SGAtoGP'] == pytest.approx(0.10)


# =========================================================================== #
#  4.  THE POOLED BoMetric_ave HAZARD -- SCORE-INERT, AND PINNED THAT WAY      #
# =========================================================================== #
def test_every_WEIGHTED_mean_criterion_is_served_a_STORED_CONSTANT():
    """`getAves2`'s pooled median is depth-sensitive on 32 of 41 columns, so the ONLY thing
    keeping Stage-1 depth-invariant here is that no WEIGHTED mean criterion reads it.
    `meanBars.mean_bar` returns the stored constant, or RAISES -- except for the declared
    exceptions, which must all be weightless."""
    mean_dict = cdic.getDicts()[3]
    for key, spec in mean_dict.items():
        mcol = 'm' + key[0].upper() + key[1:]
        if mcol in mb.NO_BAR:
            assert spec['Tier'] == 'N', (
                '%s keeps the POOLED MEDIAN (meanBars.NO_BAR) but carries Tier %r. A pooled '
                'median moves with `-nrperiods` (measured: 32 of 41 columns), so giving this '
                'criterion a weight would make Stage-1 fetch-depth dependent. Give it a bar '
                'in meanBars.BARS in the same edit.' % (mcol, spec['Tier']))
        else:
            assert mcol in mb.BARS, mcol


def test_the_pooled_median_bar_is_only_reachable_through_declared_exceptions():
    """A sentinel: `mean_bar` must RAISE for an unregistered mean criterion rather than fall
    back to the depth-sensitive pooled median."""
    with pytest.raises(KeyError):
        mb.mean_bar('mSomethingNobodyDeclared', 0.5)


# =========================================================================== #
#  5.  STRUCTURAL -- THE DEEPEST DECLARED WINDOW, AND WHAT IT IMPLIES          #
# =========================================================================== #
#  Rows a metric may read, per declared window basis.  A fetch must be at least this deep for
#  EVERY window to saturate; below it, some metric is measuring the fetch.
DEEPEST_DECLARED_WINDOW_ROWS = max(sm.CYCLEHEAT_BASE_NQ, sm.EPS_MEAN_BASE_NQ,
                                   cs.HISTORY_BONUS_SATURATION_ROWS)


def test_the_planned_fetch_depth_saturates_every_declared_window():
    """`-nrperiods 80` must clear the deepest declared window WITH the rpy history trim taken
    off, or a metric is still reading the fetch.  This is the one number that has to move if
    anybody ever shortens the fetch."""
    planned_depth, trim = 80, rp.DEFAULT_ROWS_PER_YEAR
    assert planned_depth - trim >= DEEPEST_DECLARED_WINDOW_ROWS, (
        'a %d-row fetch leaves %d BoMetric rows against a deepest declared window of %d'
        % (planned_depth, planned_depth - trim, DEEPEST_DECLARED_WINDOW_ROWS))


def test_todays_depth_does_NOT_saturate_every_window__which_is_the_whole_finding():
    """The counterexample half: `-nrperiods 24` does NOT clear it, which is why the deep fetch
    CHANGES CycleHeat, EPStoEPSmean and the history bonus rather than leaving them alone.  If
    this ever starts passing at 24, the constants moved and the report's numbers are stale."""
    assert 24 - rp.DEFAULT_ROWS_PER_YEAR < DEEPEST_DECLARED_WINDOW_ROWS


def test_every_stage2_metric_declares_a_window_basis():
    """The registry is the authority; an unregistered metric would be scored on a silently
    defaulted -- i.e. fetch-depth-dependent -- window.  Re-asserted here because THIS is the
    property the whole depth audit rests on."""
    postBm, postNew = cdic.getPostDict()
    keys = list(postBm) + list(postNew)
    assert sm.unregistered_metrics(keys) == []
    for k in keys:
        w = sm.window_quarters(k, 16)
        assert w is None or w <= DEEPEST_DECLARED_WINDOW_ROWS, (k, w)
