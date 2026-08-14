"""FETCH-DEPTH SENSITIVITY -- does a computed quantity move when we fetch MORE HISTORY?

WHY THIS EXISTS
---------------
The next fetch moves from `-nrperiods 24` to `-nrperiods 80`.  Probed live 2026-08-13:
`limit=80` returns 80 quarters (oldest 2006-09-30) and `limit=120` returns 120 (oldest
1996-09-27) on the SAME statement calls, so DEPTH COSTS ZERO EXTRA API CALLS.  Every panel
ever saved was fetched at 24, reaching back only to 2020-07.

The deep fetch is wanted for BACKTEST ANCHORING.  It must NOT change what a metric MEANS:
a name's score must not move because we happened to fetch more rows.  That is the CEO's
requirement -- "if we want to do a backtest we need the 80 periods, but I think then we
should have the metric at each point only use the last 4 years or last 3 years" -- and it
is the decoupling of FETCH DEPTH from METRIC WINDOW.

Two metrics were already built this way in anticipation (`stage2_metrics.CYCLEHEAT_BASE_NQ`
and `EPS_MEAN_BASE_NQ`, both 28 quarters).  This module is the test that says whether
EVERYTHING ELSE holds, on a REAL panel.  It is deliberately EMPIRICAL: an audit table is an
argument, this is the measurement.

THE TWO ARMS, AND WHY BOTH
--------------------------
  A. DEEP-EXTEND (the direct test of the actual move).  Take a real 24-deep panel and
     PREPEND synthetic OLDER rows until every source is `target_rows` deep, then score both
     arms through the SAME production code.  A quantity whose window is bounded at or below
     the panel's existing depth MUST come out BIT-IDENTICAL.  The CONTENT of the synthetic
     rows is deliberately irrelevant: the property under test is INVARIANCE, so any
     plausible old history that changes an answer is a finding, and a wrong-but-plausible
     history cannot produce a false PASS.  (This is the only direction available -- we
     cannot fetch, and no 80-deep panel exists.)

  B. TRUNCATE (the real-data control).  Compare the real panel truncated to its newest K
     rows against the full real panel.  Uses NO synthetic data, so an arm-A finding that
     also appears here cannot be an artifact of the synthesis.

WHAT "IDENTICAL" MEANS HERE, AND THE DECLARED EXCEPTIONS
--------------------------------------------------------
Three quantities legitimately move between a 24-deep and an 80-deep panel because their
declared window is LONGER THAN 24 and therefore CANNOT SATURATE on today's panel:

  * `CycleHeat`     window 28 quarters (CYCLEHEAT_BASE_NQ) -- 24 rows today, 28 at depth 80
  * `EPStoEPSmean`  window 28 quarters (EPS_MEAN_BASE_NQ)  -- same
  * the Stage-1 HISTORY BONUS, which saturates at `calcScore.HISTORY_BONUS_SATURATION_ROWS`
    ROWS and is therefore CENSORED at today's depth

Everything else must be invariant.  Anything ELSE that moves is a DEFECT and is named.

NO NETWORK I/O.  Pure function of a saved panel.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calcMetrics as cm
import calcScore as cs
import createDicts as cdic
import getData_fmp as gdf
import getData_gen as gdg
import postBoRank as pbr
import reporting_period as rp
import stage2_metrics as sm
import utils as utils


#  Stage-1 scoring window, in rows -- `dmdic['nrScorePeriods']`, default 8
#  (configuration.getDataFetchConfiguration).  Read from the panel where present.
STAGE1_N_DEFAULT = 8
#  Stage-2 ambient scoring window, in quarters -- postBoScoreRanking's `nq` default.
STAGE2_NQ = 16


# --------------------------------------------------------------------------- #
#  1.  THE TWO PANEL TRANSFORMS                                               #
# --------------------------------------------------------------------------- #
def extend_history(cdx_df, target_rows=80, min_rows=None, verbose=True):
    """A COPY of `cdx_df` with synthetic OLDER rows prepended until every ELIGIBLE source is
    `target_rows` deep.  Sources already at or beyond `target_rows` are untouched.

    `min_rows` -- ONLY SOURCES WITH AT LEAST THIS MANY ROWS ARE EXTENDED, and this argument
    is a correctness requirement, not a performance one.  Pass the panel's `nrperiods`.
    At `-nrperiods 24` FMP returns `min(24, what the company has)`, so a source sitting on
    FEWER than 24 rows is limited BY ITS OWN LISTING HISTORY and a deeper fetch gives it
    NOTHING.  Extending such a source models a company that does not exist and produces
    differences that a real depth change could never produce -- measured on the first run of
    this experiment: one 12-row source moved SIXTEEN Stage-2 metrics, purely because padding
    took its head(16) window from 12 real rows to 16.  A prefix-truncated source (data_quality
    removes every row AT OR BEFORE a corruption) gains nothing either: the extra rows are all
    older than the corruption, so they are removed too.

    THE SYNTHETIC CONTENT IS DELIBERATELY UNINTERESTING and cannot manufacture a false
    PASS.  Each added row is a VERBATIM COPY of one of the source's own real rows (cycled
    from its oldest), re-dated backwards at the source's own cadence.  So the values are
    real, finite, self-consistent and on the right scale -- and if a metric is properly
    windowed it can never see them.  A metric that MOVES has, by construction, read rows
    older than its window.

    *** THE ONE THING THIS METHOD MODELS WRONG, AND IT HAS ALREADY PRODUCED A WRONG CLAIM ***
    Every eligible source is padded to EXACTLY `target_rows`, i.e. this models a universe in
    which every fetch-capped name has AT LEAST `target_rows` quarters of real history.  A real
    fetch returns `min(target_rows, the company's own history)`.  For a metric that is a
    function of the WINDOW that is harmless -- the window either saturates or it does not, and
    padding cannot change which.  For a metric that is a function of the ROW COUNT ITSELF it is
    NOT harmless, and there is exactly one such metric: the Stage-1 HISTORY BONUS.

    MEASURED CONSEQUENCE (2026-08-14 review finding H-1).  This harness reported the deep fetch
    as a UNIFORM +0.014645 on every capped name -- max and median identical -- and that was read
    as "a common offset, so it cannot re-rank the cohort".  It is an ARTIFACT: pad every source
    to 80 and every source saturates the bonus, so of course they all move by the same amount.
    In a real fetch the step is a continuous function of listing age (+0.000000 at 24 quarters of
    own history, +0.003374 at 28, +0.006478 at 32, ... saturating only at 44), so the deep run
    RE-RANKS across the whole band.  See `calcScore.history_bonus_censored_by`.

    SO: A ZERO HERE IS STRONG EVIDENCE OF WINDOW INVARIANCE, AND A NON-ZERO HERE IS NOT
    NECESSARILY THE MAGNITUDE A REAL FETCH PRODUCES.  Any quantity that reads a source's ROW
    COUNT rather than a bounded window must be reasoned about analytically, not read off this
    table.  (The `--decompose` arm exists precisely so the bonus can be switched OFF and the
    windowed residual read cleanly; that residual is the number this harness is trustworthy for.)

    DATES step back by 12/rpy MONTHS from the source's own oldest row and are snapped by
    `utils.setDatesToQuarterly`, exactly as the ingest snaps them, so no synthetic row can
    collide with a real quarter.  `periodEndDate` (where present) is moved in lockstep --
    leaving the RAW period end at its old value would give the row two different dates and
    is precisely the kind of skew the frequency classifier reads.

    Row ORDER of the returned frame matches the saved convention: ASCENDING per source,
    sources in their original order.
    """
    if 'source' not in cdx_df.columns or 'date' not in cdx_df.columns:
        raise ValueError('extend_history: cdx_df needs `source` and `date` columns.')
    out = []
    n_ext = n_rows_added = 0
    for src, g in cdx_df.groupby('source', sort=False):
        g = g.copy()
        d = pd.to_datetime(g['date'], errors='coerce')
        #  Ascending is the SAVED convention (data_quality sorts it); assert rather than
        #  assume, because prepending to a newest-first frame would append the FUTURE.
        g = g.assign(_ds_date=d).sort_values('_ds_date', kind='mergesort')
        need = int(target_rows) - len(g)
        if (need <= 0 or g['_ds_date'].isna().all()
                or (min_rows is not None and len(g) < int(min_rows))):
            out.append(g.drop(columns=['_ds_date']))
            continue
        rpy = rp.rows_per_year(str(g[rp.FREQ_COLUMN].iloc[0])
                               if rp.FREQ_COLUMN in g.columns else rp.UNKNOWN)
        step_months = int(round(12.0 / float(rpy)))
        oldest = g['_ds_date'].dropna().iloc[0]
        src_rows = g.drop(columns=['_ds_date'])
        blocks = []
        for k in range(1, need + 1):
            row = src_rows.iloc[[(k - 1) % len(src_rows)]].copy()
            new_date = oldest - pd.DateOffset(months=step_months * k)
            row['date'] = new_date
            if 'periodEndDate' in row.columns:
                row['periodEndDate'] = new_date
            blocks.append(row)
        ext = pd.concat(blocks, ignore_index=True)
        ext = utils.setDatesToQuarterly(ext)
        #  Oldest first, then the real rows -- ascending overall.
        ext = ext.iloc[::-1].reset_index(drop=True)
        out.append(pd.concat([ext, src_rows], ignore_index=True))
        n_ext += 1
        n_rows_added += need
    res = pd.concat(out, ignore_index=True)
    if verbose:
        print('extend_history: %d of %d source(s) extended to %d rows (+%d synthetic rows); '
              'panel %d -> %d rows'
              % (n_ext, cdx_df['source'].nunique(), target_rows, n_rows_added,
                 len(cdx_df), len(res)), flush=True)
    return res


def truncate_history(cdx_df, keep_rows, verbose=True):
    """A COPY of `cdx_df` keeping only each source's NEWEST `keep_rows` rows.  Real data
    only -- the control arm."""
    d = pd.to_datetime(cdx_df['date'], errors='coerce')
    tmp = cdx_df.assign(_ds_date=d)
    keep = (tmp.sort_values(['source', '_ds_date'], ascending=[True, False],
                            kind='mergesort')
               .groupby('source', sort=False).head(int(keep_rows)).index)
    res = cdx_df.loc[cdx_df.index.isin(keep)].copy()
    if verbose:
        print('truncate_history: kept newest %d row(s) per source; panel %d -> %d rows'
              % (keep_rows, len(cdx_df), len(res)), flush=True)
    return res


# --------------------------------------------------------------------------- #
#  2.  THE SCORING ARM -- production functions only                           #
# --------------------------------------------------------------------------- #
def _rebuild_bometric(cdx, n=1, verbose=False):
    """Stage-1 panel rebuilt from `cdx` with the LIVE construction
    (`getData_fmp.build_bometric_rows` + `getData_gen.fixAfterGetData`).

    A near-copy of `baseline_tools/panel_upgrade.rebuild_bometric`, differing in ONE way and
    deliberately: it reads the panel's OWN stored `reportingFrequency` stamp and never
    re-derives it.  Re-deriving would make the classifier a second depth-sensitive input to
    this experiment (`rp.CLASSIFY_RECENT_DAYS` is anchored to the source's newest row, so it
    is depth-safe -- but the point of a control is not to have to argue that).  Both arms
    therefore run on the SAME per-source frequency, and every difference the experiment
    reports is attributable to DEPTH alone.
    """
    if rp.FREQ_COLUMN not in cdx.columns:
        raise ValueError('depth_sensitivity: the panel carries no %r stamp; this experiment '
                         'requires it so both arms share one frequency classification.'
                         % rp.FREQ_COLUMN)
    bm_cols = list(utils.initBoMetric_fromDict()['BoMetric_df'].columns)
    dicts = cdic.getDicts()
    packed = (dicts[2], dicts[3], dicts[5], dicts[4], dicts[6])   # base, mean, unity, diff, special
    frames = []
    for src, g in cdx.iloc[::-1].groupby('source', sort=False):     # -> newest-first
        tf = g.reset_index(drop=True)
        tmp = pd.DataFrame(columns=bm_cols)
        tmp['date'] = tf['date'].values
        tmp['source'] = src
        tmp = utils.setDatesToQuarterly(tmp)
        _rpy = rp.rows_per_year(str(tf[rp.FREQ_COLUMN].iloc[0]))
        frames.append(gdf.build_bometric_rows(tf, tmp, _rpy, n=n, dicts=packed))
    bm = pd.concat(frames, ignore_index=True)
    bm, _ = gdg.fixAfterGetData(bm, cdx.copy())
    if verbose:
        print('  BoMetric rebuilt: %d rows over %d sources'
              % (len(bm), bm['source'].nunique()), flush=True)
    return bm


def _stage1(cdx, bm, n, freq_map, ave_override=None, peg_median_override=None):
    """(scores, pooled_medians, peg_stats) for one arm, through the production seam:
    `getAves2` -> `substitute_peg_crossing` -> `simpleScore_fromDict`, which is exactly the
    order `postBo.postBoWrapper` runs them in.

    The two overrides exist for the CHANNEL DECOMPOSITION and for nothing else: holding one
    cross-sectional baseline at the shallow arm's value isolates how much of the Stage-1
    movement travelled through THAT channel rather than through the windows.  Neither is a
    production option.
    """
    ave = cs.getAves2(bm.copy())
    if peg_median_override is None:
        bm_scored, peg_stats = cm.substitute_peg_crossing(bm, cdx, freq_map=freq_map,
                                                          verbose=False)
    else:
        _real = cm.peg_pool_median_growth
        try:
            cm.peg_pool_median_growth = (
                lambda *_a, **_k: (float(peg_median_override[0]), int(peg_median_override[1])))
            bm_scored, peg_stats = cm.substitute_peg_crossing(bm, cdx, freq_map=freq_map,
                                                              verbose=False)
        finally:
            cm.peg_pool_median_growth = _real
    bar = ave['BoMetric_ave'] if ave_override is None else ave_override
    scores = cs.simpleScore_fromDict(bm_scored, bar, ave['BoMetric_dateAve'],
                                     n, freq_map=freq_map)
    return scores, ave['BoMetric_ave'], peg_stats


#  ---- CHANNEL DECOMPOSITION -------------------------------------------------------------
#  Stage-1's score is a sum of windowed criterion means PLUS two things that are NOT windowed:
#  a HISTORY BONUS read off the source's ROW COUNT, and two CROSS-SECTIONAL baselines pooled
#  over the whole panel (the PEG crossing median and `BoMetric_ave`).  A single "the score
#  moved" number cannot tell those apart, and they need different fixes -- so the experiment
#  peels them off one at a time and reports what each is worth.
CHANNELS = ('history_bonus', 'peg_pool_median', 'pooled_bometric_ave')


def decompose_stage1(cdx_base, cdx_deep, n, verbose=True):
    """How much of the Stage-1 movement each non-windowed channel is worth.

    Four arms, each disabling one more channel.  The RESIDUAL after all three is what
    travelled through the windowed criteria themselves -- which is the number that must be
    ZERO for the windows to be honestly capped.
    """
    import contextlib
    import io
    buf = io.StringIO()
    rows = []
    with contextlib.redirect_stdout(buf):
        bm_b, bm_d = _rebuild_bometric(cdx_base), _rebuild_bometric(cdx_deep)
        fm_b = rp.frequency_by_source(cdx_base, verbose=False)
        fm_d = rp.frequency_by_source(cdx_deep, verbose=False)
        s_b, ave_b, peg_b = _stage1(cdx_base, bm_b, n, fm_b)
        peg_base = (peg_b['median_growth'], peg_b['n_pool_rows'])

        def _arm(bonus_on, peg_ov, ave_ov):
            _keep = cs.HISTORY_BONUS_MAX
            try:
                if not bonus_on:
                    cs.HISTORY_BONUS_MAX = 0.0
                b = _stage1(cdx_base, bm_b, n, fm_b,
                            ave_override=None, peg_median_override=None)[0]
                d = _stage1(cdx_deep, bm_d, n, fm_d,
                            ave_override=(ave_b if ave_ov else None),
                            peg_median_override=(peg_base if peg_ov else None))[0]
            finally:
                cs.HISTORY_BONUS_MAX = _keep
            return b, d

        specs = [('as shipped (all channels live)', True, False, False),
                 ('+ history bonus disabled', False, False, False),
                 ('+ PEG pool median pinned to shallow', False, True, False),
                 ('+ BoMetric_ave pinned to shallow  [RESIDUAL]', False, True, True)]
        for label, bonus, peg_ov, ave_ov in specs:
            b, d = _arm(bonus, peg_ov, ave_ov)
            rows.append(_compare_series(b['score'], d['score'], b['source'], d['source'],
                                        label))
    tab = pd.DataFrame(rows)
    if verbose:
        print('\nSTAGE-1 CHANNEL DECOMPOSITION  (each row disables one MORE channel)\n',
              flush=True)
        print(tab.to_string(index=False), flush=True)
    return tab


def _stage2(cdx, scores, nq=STAGE2_NQ):
    """Every Stage-2 metric for every source, via the PRODUCTION per-ticker function
    `postBoRank._compute_ticker_metrics` -- not a re-implementation, so this experiment
    cannot drift from the scorer it is testing.

    The two things `postBoScoreRanking` does around that function and this must reproduce
    are done here explicitly: the newest-first re-sort and the pool-level `mcapQuants`
    column.  DCF is passed EMPTY -- `DcfToPrice` carries w = 0.000 in the deployed vector
    (`scoringWeights.DEPLOYED`) and a live DCF fetch is network I/O, which this module
    forbids -- so `DcfToPrice` is excluded from the comparison rather than compared on a
    blank.
    """
    postBm, postNew = cdic.getPostDict()
    cdxs = pbr._sort_cdx_newest_first(cdx)
    cdxs['mcapQuants'] = sm.add_mcap_quants(cdxs)
    freq_map = rp.frequency_by_source(cdxs, verbose=False)
    out = pd.DataFrame()
    out['source'] = scores['source'].values
    out = pd.concat([out, pd.DataFrame(columns=list(postBm.keys()))], axis=1)
    out = pd.concat([out, pd.DataFrame(columns=list(postNew.keys()))], axis=1)
    empty_dcf = pd.DataFrame()
    for i, ticker in enumerate(scores['source']):
        tempcdx = cdxs.loc[cdxs['source'] == ticker]
        pbr._compute_ticker_metrics(ticker, tempcdx, empty_dcf, scores, nq, i,
                                    postBm, postNew, out,
                                    rpy=rp.rows_per_year(freq_map, ticker))
    return out


def _side_quantities(cdx, sources):
    """The blocks that sit BESIDE the two scorers and are not covered by `_stage2`:
    the Montier-C / Beneish-M means, Sloan accruals, the moat count and the mean-bar
    calibration.  Each is a real deliverable (the forensic tags reach the CEO's shortlist;
    the calibration can PROPOSE a bar change), so each has to be shown depth-invariant too --
    an audit that stops at the score would have left four live surfaces unmeasured.

    Returns {label: (values, keys)} for `_compare_series`.
    """
    import detectManipulation as dm
    import forensicFlags as ff
    import postBo as pb
    out = {}
    resdic = {'cdx_df': cdx}
    fm = rp.frequency_by_source(cdx, verbose=False)
    _cdf, c_mean, _p = dm.calcMontierC(resdic, sources, freq_map=fm)
    out['FORENSIC MontierC C_Score_mean'] = (c_mean['C_Score_mean'], c_mean['source'])
    _mdf, m_mean, _p = dm.calcBeneishM(resdic, sources, freq_map=fm)
    out['FORENSIC BeneishM M_Score_mean'] = (m_mean['M_Score_mean'], m_mean['source'])
    sl = ff.computeSloanAccruals(cdx, sources, freq_map=fm)
    out['FORENSIC Sloan accruals'] = (sl['sloanAccruals'], sl['source'])
    moat = pb.moatIdentifier(pd.Series(sources), cdx, freq_map=fm)
    out['DISPLAY moatScore'] = (moat['moatScore'], moat['source'])
    return out


def _calibration_arm(bm):
    """`meanBars.calibrate` on one arm -- the seven bars' realised pass rates, which drive
    the breach band and (after two consecutive full-universe breaches) a PROPOSED re-set."""
    md = cdic.getDicts()[3]
    signs = {'m' + k[0].upper() + k[1:]: md[k]['Sign'] for k in md}
    import meanBars as mb
    return mb.calibrate(bm, signs, window_rows=8, n_sources=len(bm['source'].unique()))


def score_arm(cdx, n, label, verbose=True):
    """(stage1_scores, stage2_metrics, pooled_medians, peg_stats) for one panel."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bm = _rebuild_bometric(cdx)
        freq_map = rp.frequency_by_source(cdx, verbose=False)
        s1, med, peg = _stage1(cdx, bm, n, freq_map)
        s2 = _stage2(cdx, s1)
        side = _side_quantities(cdx, list(s1['source']))
        cal = _calibration_arm(bm)
    if verbose:
        print('  arm %-14s : %d source(s), %d panel rows, PEG pool median growth %.6f%% '
              'over %d in-domain row(s)'
              % (label, len(s1), len(cdx), peg['median_growth'], peg['n_pool_rows']),
              flush=True)
    return s1, s2, med, peg, side, cal


# --------------------------------------------------------------------------- #
#  3.  THE COMPARISON                                                         #
# --------------------------------------------------------------------------- #
#  Quantities whose value LEGITIMATELY moves between a 24-deep and a deeper panel, because
#  their declared window is LONGER than today's depth and so cannot saturate on it.  Listed
#  as DECLARED EXCEPTIONS so the report distinguishes "expected, and here is the declaration"
#  from "nobody decided this".
DECLARED_DEEP_WINDOW = {
    'CycleHeat': 'stage2_metrics.CYCLEHEAT_BASE_NQ = %d quarters > today\'s depth'
                 % sm.CYCLEHEAT_BASE_NQ,
    'EPStoEPSmean': 'stage2_metrics.EPS_MEAN_BASE_NQ = %d quarters > today\'s depth'
                    % sm.EPS_MEAN_BASE_NQ,
}
#  Computed on an empty DCF frame in both arms (see `_stage2`), so it carries no information.
EXCLUDED_FROM_COMPARISON = ('DcfToPrice',)


def _compare_series(a, b, key_a, key_b, label, atol=0.0):
    """Rows where two aligned numeric series differ.  NaN == NaN counts as EQUAL."""
    j = pd.merge(pd.DataFrame({'source': key_a, 'v_base': pd.to_numeric(a, errors='coerce')}),
                 pd.DataFrame({'source': key_b, 'v_deep': pd.to_numeric(b, errors='coerce')}),
                 on='source', how='inner')
    both_nan = j['v_base'].isna() & j['v_deep'].isna()
    diff = (j['v_base'] - j['v_deep']).abs()
    changed = (~both_nan) & (~(diff <= atol)).fillna(True)
    return {
        'quantity': label,
        'n_compared': int(len(j)),
        'n_changed': int(changed.sum()),
        'pct_changed': (100.0 * changed.sum() / len(j)) if len(j) else np.nan,
        'max_abs_delta': float(diff[changed].max()) if changed.any() else 0.0,
        'median_abs_delta': float(diff[changed].median()) if changed.any() else 0.0,
    }


def compare_arms(base, deep, quarterly_only_sources=None):
    """One row per computed quantity: did it move, on how many names, by how much."""
    (s1b, s2b, medb, pegb, sideb, calb) = base
    (s1d, s2d, medd, pegd, sided, cald) = deep
    if quarterly_only_sources is not None:
        keep = set(quarterly_only_sources)
        s1b, s1d = s1b[s1b['source'].isin(keep)], s1d[s1d['source'].isin(keep)]
        s2b, s2d = s2b[s2b['source'].isin(keep)], s2d[s2d['source'].isin(keep)]
    rows = [_compare_series(s1b['score'], s1d['score'], s1b['source'], s1d['source'],
                            'STAGE-1 BoScore')]
    cols = [c for c in s2b.columns
            if c != 'source' and c not in EXCLUDED_FROM_COMPARISON]
    for c in cols:
        rows.append(_compare_series(s2b[c], s2d[c], s2b['source'], s2d['source'],
                                    'STAGE-2 ' + c))
    for lbl in sideb:
        vb, kb = sideb[lbl]
        vd, kd = sided[lbl]
        if quarterly_only_sources is not None:
            mb_ = kb.isin(keep)
            md_ = kd.isin(keep)
            vb, kb, vd, kd = vb[mb_], kb[mb_], vd[md_], kd[md_]
        rows.append(_compare_series(vb, vd, kb, kd, lbl))
    rows.append(_compare_series(calb['pass_rate'], cald['pass_rate'],
                                calb['criterion'], cald['criterion'],
                                'MEAN-BAR calibration pass rates'))
    res = pd.DataFrame(rows)
    res['declared'] = [DECLARED_DEEP_WINDOW.get(q.replace('STAGE-2 ', ''), '')
                       for q in res['quantity']]
    pooled = pd.DataFrame([{
        'quantity': 'POOLED PEG median annual growth (calcMetrics.peg_pool_median_growth)',
        'n_compared': 1,
        'n_changed': int(not np.isclose(pegb['median_growth'], pegd['median_growth'],
                                        equal_nan=True)),
        'pct_changed': np.nan,
        'max_abs_delta': abs(float(pegb['median_growth']) - float(pegd['median_growth'])),
        'median_abs_delta': np.nan,
        'declared': '',
    }])
    med_j = pd.concat([medb.rename('base'), medd.rename('deep')], axis=1).dropna(how='all')
    n_med_moved = int((~np.isclose(med_j['base'], med_j['deep'], equal_nan=True)).sum())
    pooled = pd.concat([pooled, pd.DataFrame([{
        'quantity': 'POOLED BoMetric_ave column medians (calcScore.getAves2)',
        'n_compared': int(len(med_j)),
        'n_changed': n_med_moved,
        'pct_changed': 100.0 * n_med_moved / max(1, len(med_j)),
        'max_abs_delta': float((med_j['base'] - med_j['deep']).abs().max()),
        'median_abs_delta': np.nan,
        'declared': '',
    }])], ignore_index=True)
    return pd.concat([res, pooled], ignore_index=True)


# --------------------------------------------------------------------------- #
#  4.  CLI                                                                    #
# --------------------------------------------------------------------------- #
def load_panel(path):
    with open(path, 'rb') as fh:
        d = pickle.load(fh)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--panel', required=True, help='a saved Bometric_dic / Boresults_dic pickle')
    ap.add_argument('--mode', choices=('extend', 'truncate'), default='extend')
    ap.add_argument('--target-rows', type=int, default=80)
    ap.add_argument('--keep-rows', type=int, default=16)
    ap.add_argument('--sources', type=int, default=0,
                    help='limit to the first N sources (0 = all)')
    ap.add_argument('--decompose', action='store_true',
                    help='also attribute the Stage-1 movement to its non-windowed channels')
    ap.add_argument('--out', default='', help='write the comparison table to this CSV')
    args = ap.parse_args(argv)

    d = load_panel(args.panel)
    cdx = d['cdx_df']
    n = int(d.get('nrScorePeriods') or STAGE1_N_DEFAULT)
    if args.sources:
        keep = list(pd.unique(cdx['source']))[:args.sources]
        cdx = cdx[cdx['source'].isin(keep)].copy()
    print('panel: %d row(s), %d source(s), nrperiods=%s, nrScorePeriods=%d'
          % (len(cdx), cdx['source'].nunique(), d.get('nrperiods'), n), flush=True)

    freq = rp.frequency_by_source(cdx, verbose=False)
    quarterly = [s for s, f in freq.items() if f != rp.SEMIANNUAL]
    print('quarterly-or-unknown sources: %d of %d' % (len(quarterly), len(freq)), flush=True)

    if args.mode == 'extend':
        #  ONLY FETCH-CAPPED SOURCES ARE DEEPENED -- see extend_history's `min_rows`.
        arm_a_cdx = cdx
        arm_b_cdx = extend_history(cdx, args.target_rows,
                                   min_rows=int(d.get('nrperiods') or 24))
        labels = ('base(as-fetched)', 'deep(%d rows)' % args.target_rows)
    else:
        arm_a_cdx, arm_b_cdx = truncate_history(cdx, args.keep_rows), cdx
        labels = ('shallow(%d rows)' % args.keep_rows, 'full(as-fetched)')

    base = score_arm(arm_a_cdx, n, labels[0])
    deep = score_arm(arm_b_cdx, n, labels[1])
    table = compare_arms(base, deep, quarterly_only_sources=quarterly)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 60)
    print('\nDEPTH-SENSITIVITY TABLE  (%s vs %s, QUARTERLY filers only)\n' % labels, flush=True)
    print(table.to_string(index=False), flush=True)
    moved = table[(table['n_changed'] > 0) & (table['declared'] == '')]
    if len(moved):
        print('\n!!! %d quantity/quantities moved with NO declared deep window:\n%s'
              % (len(moved), ', '.join(moved['quantity'])), flush=True)
    else:
        print('\nEvery undeclared quantity is IDENTICAL across the two depths.', flush=True)
    if args.decompose:
        decompose_stage1(arm_a_cdx, arm_b_cdx, n)
    if args.out:
        table.to_csv(args.out, index=False)
        print('written to %s' % args.out, flush=True)
    return table


if __name__ == '__main__':
    main()
