import collections
import math
import os
import sys

import createDicts as cdic
import getData_gen as gdg
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

import nan_policy as npol
import stage2_metrics as sm
import reporting_period as rp


# --------------------------------------------------------------------------- #
#  Stage-2 scorer -- postBoScoreRanking, split into single-responsibility
#  helpers.  The per-metric formulas live ONCE in stage2_metrics.py and are
#  shared with the offline reproduction (baseline_tools/stage2_pit.py); this
#  file owns the LIVE orchestration (input checks, the newest-first re-sort, the
#  live DCF fetch, normalisation/weighting/aggregation, and issuer-dedup).
#
#  IT IS ALSO THE ONLY PUBLISHER OF METRIC VALUES.  Three frames leave this file
#  carrying metric-named columns on THREE DIFFERENT BASES, and telling them apart
#  used to depend on the reader knowing which resdic key they came from -- which
#  is how `AggScoreTop100.CycleHeat` shipped as `z x (-0.080)` (an EXACT sign
#  inversion) and how two regression consumers fitted and applied coefficients on
#  disagreeing bases.  Each frame now DECLARES its basis on itself (see the
#  METRIC BASIS section below) and `metric_frame()` is the one way to read metric
#  values off any of them, with the basis stated in the call.
# --------------------------------------------------------------------------- #
def postBoScoreRanking(bmtop, bstop, cdxtop, baseurl, api_key, period='quarter',
                       nq=16, as_of=None, weight_override=None, names=None,
                       dedup_issuers=True, pool_label=None):
    # pool_label : names THIS pool in the missing-data fill report ('general' or a carve-out
    # cohort label).  Diagnostic only -- it reaches no scoring path.
    # as_of : point-in-time date D (default None).  as_of=None reproduces the live
    # Stage-2 ranking BIT-FOR-BIT.  The parameter is threaded here so the PIT DCF/beta
    # engagement (computed point-in-time DcfToPrice + CycleHeat beta, design s2B/s2C)
    # has a live seam; the point-in-time DCF/beta substitution itself is a later
    # (registry-backed, Phase 3+) step and is NOT wired on this path yet.  With
    # as_of=None nothing below branches on it -> live behaviour unchanged.
    # `weight_override` is passed so the score-neutrality guard sees THIS POOL's weights, not
    # just the main vector: a carve-out cohort could give DcfToPrice a weight without the main
    # dict changing, and skipping the fetch would then silently blank a metric that scores.
    _assert_offline_dcf_is_score_neutral(weight_override)
    print('Ranking the top 100 stocks, according to BoScore.')
    sys.stdout.flush()  # Ensure output is printed before progress bar

    # Guarded like the other three (review B1).  This one was missed on the first pass and the
    # sweep test caught it -- which is the partial-sweep defect class this project keeps hitting,
    # so the sweep is asserted rather than trusted.
    _safe_diagnose(_diagnose_inputs, bmtop, bstop, cdxtop)

    postBmRankingDict, postNewRankingDict = cdic.getPostDict()
    postScoreMetric_df = pd.DataFrame()
    postScoreMetric_df['source'] = bstop['source']
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postBmRankingDict.keys())], axis=1)
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postNewRankingDict.keys())], axis=1)

    # Build a stable weight mapping from the post dictionaries so we always have a
    # weight for each metric.  Per-cohort weight vector (carveOut.COHORT_WEIGHTS):
    # weight_override overrides the default weight for any metric it lists.
    # weight_override=None (the general/main pool) keeps the default vector.  A 0
    # weight zeroes that metric's AggScore contribution AND makes the weighted column
    # constant (0) -> neutral in rankOfRanks; it does NOT change cohort membership (a
    # row is dropped only if ALL metrics are NaN, upstream).
    weight_series = {**{k: postBmRankingDict[k]['w'] for k in postBmRankingDict},
                     **{k: postNewRankingDict[k]['w'] for k in postNewRankingDict}}
    if weight_override:
        weight_series = {**weight_series, **weight_override}

    # EVERY METRIC MUST DECLARE ITS WINDOW AND FREQUENCY TREATMENT (stage2_metrics
    # .STAGE2_METRIC_SPEC).  Checked ONCE PER POOL, before any metric is computed, because a
    # metric with no registry entry is exactly how CycleHeat shipped taking neither `nq` nor
    # `rpy` and EPStoEPSmean shipped with its baseline equal to the fetch depth.  A missing
    # entry now REFUSES the run rather than silently defaulting the window -- the same
    # posture as the DcfToPrice weight guard above, and for the same reason: a silently
    # different ranking is worse than no ranking.
    _unregistered = sm.unregistered_metrics(weight_series.keys())
    if _unregistered:
        raise SystemExit(
            "Stage-2 metric(s) %r have NO entry in stage2_metrics.STAGE2_METRIC_SPEC. That "
            "table is the AUTHORITY for each metric's window basis and frequency treatment "
            "and the scorer reads it, so an unregistered metric would be scored on a "
            "silently-defaulted window. Add an entry declaring the window basis and the "
            "frequency treatment (with the reason), then re-run." % (_unregistered,))

    cdxtop = _sort_cdx_newest_first(cdxtop)
    cdxtop['mcapQuants'] = sm.add_mcap_quants(cdxtop)

    # PER-SOURCE REPORTING FREQUENCY, computed ONCE for the pool.  Every windowed /
    # YoY Stage-2 metric below is parameterised by this source's rows-per-year so a
    # semi-annual filer is not scored on 2-year "YoY" growth and 2-year "TTM" windows
    # (audit C-1).  unknown -> quarterly, i.e. unchanged.
    freq_map = rp.frequency_by_source(cdxtop, verbose=True)

    # Note: Bulk endpoints require higher subscription tier, using individual API calls only
    dcf_bulk_dict = {}

    # LABEL THE NaN-POLICY COUNTS WITH THIS POOL (nan_policy.POLICY_COUNTS).  The counter is a
    # run-level accumulator in the same style as NORM_DIAGNOSTICS, and this scorer runs ONCE PER
    # POOL -- general plus the five carve-out cohorts -- so without the label a cohort's
    # coverage/gappiness conversions would be indistinguishable from the general pool's.  It is
    # the observability half of the two-tier policy, and it is a COUNT rather than a per-cell
    # reason channel because a refused-vs-missing channel was explicitly ruled out.
    npol.set_pool(pool_label or 'general')

    pbar = tqdm(total=len(bstop['source'].unique()))
    for tempcntr, ticker in enumerate(bstop['source']):
        tempcdx = cdxtop.loc[cdxtop['source'] == ticker]

        # DCF data (used for the DcfToPrice metric); live per-ticker fetch.
        dcf, dcf_from_bulk, resp_dcf_status, resp_dcf = _fetch_ticker_dcf(
            ticker, baseurl, api_key, dcf_bulk_dict)

        # A DIAGNOSTIC MUST NEVER ABORT THE RUN (review B1, 2026-07-31).  These three calls
        # were unguarded, inside a postBoScoreRanking that is itself unguarded in
        # postBoWrapper, so ANY defect in a print-only helper cost Stage-2 -> no postRank -> no
        # AggScore CSV, no XLSX, no side-lists, no band CSVs, no pick-log.  That is what
        # happened with `resp_dcf.text` on a _FailedResponse.  Fixing only that one attribute
        # leaves the structure that turned a print bug into a total loss, so the structure is
        # fixed too -- this wave already holds `log_conflicts` and `_write_conflict_csv` to the
        # same standard, and the run's own diagnostics should not be the one exception.
        if tempcntr == 0:
            _safe_diagnose(_diagnose_first_ticker_data, ticker, dcf, dcf_from_bulk,
                           resp_dcf_status, resp_dcf, tempcdx)

        _compute_ticker_metrics(ticker, tempcdx, dcf, bstop, nq, tempcntr,
                                postBmRankingDict, postNewRankingDict, postScoreMetric_df,
                                rpy=rp.rows_per_year(freq_map, ticker))

        if tempcntr == 0:
            _safe_diagnose(_diagnose_first_ticker_metrics, ticker, postScoreMetric_df)

        pbar.update(n=1)

    _safe_diagnose(_diagnose_pre_normalize, postScoreMetric_df)

    # --- REVIEW-REFERENCE capture (READ-ONLY; must NOT perturb scoring) ----------
    # Snapshot the RAW per-ticker metrics BEFORE normalizeAndDropNA z-scores them IN
    # PLACE (and before its >4-std outlier drop below), so the human-review reference
    # artifacts (reviewReference.py) reflect true RAW values and the full pre-drop pool
    # membership.  This is a COPY only: it is returned in rankdic['postScoreMetric_raw']
    # and is NEVER read back into scoring / normalization / ranking.  Feeding cohort
    # means or percentiles derived from this back into the score would be cross-sectional
    # sector-neutralization, which is CEO-ratified OFF -- so the capture must stay a pure
    # side-channel.  Because postBoScoreRanking runs once per pool (general + each cohort
    # via postBo), this single line yields raw metrics for every pool automatically.
    postScoreMetric_raw = stamp_metric_basis(postScoreMetric_df.copy(), BASIS_RAW)

    # weight_series is passed so the outlier guard can EXEMPT zero-weight metrics: a
    # w=0 diagnostic column must not clamp (and, pre-fix, must not eject) a name that
    # contributes to no part of the AggScore.  This is the SAME weight_series -- incl.
    # any cohort weight_override -- that weights the columns below, so a metric the
    # cohort zeroed is exempt in that cohort's pool too.
    postScoreMetric_df, outlierlist = normalizeAndDropNA(
        postScoreMetric_df, weight_series=weight_series,
        pool_label=(pool_label or 'general'))
    #  BASIS_Z from here to the weighting step below -- declared on the frame, so a consumer
    #  of resdic['postScoreMetric'] can assert it instead of inferring it from the key name.
    stamp_metric_basis(postScoreMetric_df, BASIS_Z)

    # MISSING-DATA FILL CALIBRATION (2026-08-01) -- EMITS ONLY, changes no score.
    # Placed HERE because this is the one point where both sides exist: postScoreMetric_raw
    # still carries the NaNs, and postScoreMetric_df carries the z's that the fillna(0) has
    # already imputed.  It runs for EVERY pool (general + the five carve-out cohorts), which is
    # the part nobody has measured -- a cohort concentrating most of its weight on two columns
    # makes a single fill far more consequential per cell than the same fill in the general
    # pool.  Read-only on both frames; nothing is assigned back.
    _safe_diagnose(missing_data_fill_report, postScoreMetric_raw, postScoreMetric_df,
                   weight_series, pool=(pool_label or 'general'))

    # WHY each cell was NaN in the first place, as a per-rule COUNT for this pool.  The fill
    # report above says WHERE the imputation lands; this says WHICH RULE created the cell it
    # imputed -- coverage below 0.50, interior gaps, calendar gaps, a boundary limit, or a
    # refusal.  Together they are what section 2 of nan-policy.md asks the run to emit.
    _safe_diagnose(npol.report_counts)

    # Apply weights using the stable weight_series mapping; if a weight is missing,
    # default to 1.
    temp_normpsmdf_weighted = postScoreMetric_df.drop('source', axis=1)
    for col in temp_normpsmdf_weighted.columns:
        w = weight_series.get(col, 1)
        temp_normpsmdf_weighted[col] = postScoreMetric_df[col].values * w
    psmdf_normalized = pd.concat(
        [postScoreMetric_df[postScoreMetric_df.columns.difference(temp_normpsmdf_weighted.columns)],
         temp_normpsmdf_weighted], axis=1)
    #  z x w from here on.  Stamped AFTER the concat, because concat drops attrs whenever its
    #  inputs' attrs differ -- and here one side carries BASIS_Z and the other carries none.
    #  The name `psmdf_normalized` is the trap this stamp defuses: it says "normalized", it is
    #  stored in resdic under that name, and its metric columns are WEIGHTED.
    stamp_metric_basis(psmdf_normalized, BASIS_Z_TIMES_W)

    postRank = getAggScore(psmdf_normalized)
    #  DECLARED INVARIANT, not an accident: getAggScore mutates its argument in place and
    #  returns it, so these two names are ONE object and resdic ends up storing the same
    #  frame twice, under 'psmdf_normalized' and 'postRank'.  Consumers depend on the shape
    #  that follows from it -- resdic['psmdf_normalized'] carries AggScore and
    #  rankOfRanks_diag and is AggScore-descending -- so the aliasing is deliberately KEPT
    #  and asserted here rather than quietly removed: if a future edit makes getAggScore
    #  return a copy, this fails loudly at the source instead of silently changing the
    #  contents and row order of a stored artifact.
    assert postRank is psmdf_normalized, \
        ("getAggScore no longer returns its argument in place: resdic['psmdf_normalized'] "
         "and resdic['postRank'] have silently become DIFFERENT frames (different columns "
         "and row order). Update every consumer of psmdf_normalized before allowing this.")

    tmpcorr = np.corrcoef(list(postRank['BoScore'].values), list(postRank['AggScore'].values))
    BoAggCorr = tmpcorr[0, 1]

    postRank = getRankOfRanks(postRank)

    pbar.close()

    postRank_predupe = postRank.copy()
    postRank, issuer_dupes_dropped = _dedup_issuers_in_ranking(
        postRank, cdxtop, names, dedup_issuers)

    #  THE k PROPERTY, against THIS pool's own panel (E-1 / SQUASH_K).  It runs HERE, and the
    #  position is chosen twice over: it cannot run inside normalizeAndDropNA because the
    #  right-hand side of the inequality is the median-to-rank-20 distance of an AggScore that
    #  does not exist until getAggScore; and it runs AFTER the issuer-dedup because the
    #  rank-20 boundary of the DELIVERABLE is the post-dedup one, and a dual-listing occupying
    #  two slots moves it.  Runs for the general pool AND each cohort -- the cohorts are where
    #  the property is least tested, since every published figure for it is the n=100 general
    #  pool.  EMITS ONLY, and guarded.
    _safe_diagnose(single_column_reach_check, postRank, weight_series,
                   (pool_label or 'general'))

    #  METRIC BASIS PER KEY -- each frame also declares it on itself (metric_basis_of):
    #    postScoreMetric_raw            BASIS_RAW       the metric in its own units
    #    postScoreMetric                BASIS_Z         winsorized cross-sectional z
    #    psmdf_normalized / postRank    BASIS_Z_TIMES_W  z x weight (ONE object -- see above)
    #    postRank_predupe               BASIS_Z_TIMES_W  a copy of postRank before dedup
    #  Read metric values with metric_frame(frame, basis); do NOT infer the basis from the key.
    rankdic = {'postRank': postRank, 'postScoreMetric': postScoreMetric_df,
               'postScoreMetric_raw': postScoreMetric_raw,
               'psmdf_normalized': psmdf_normalized, 'BoAggCorr': BoAggCorr, 'outlierlist': outlierlist,
               'postRank_predupe': postRank_predupe, 'issuer_dupes_dropped': issuer_dupes_dropped}

    return rankdic


def _sort_cdx_newest_first(cdxtop):
    """THE ROW-ORDER BOUNDARY for Stage-2: everything downstream may assume newest-first.

    Every metric in stage2_metrics indexes with .head(w) / .iloc[0] / .iloc[lag] /
    pct_change(-1) assuming the most-recent period is row 0.  data_quality.py sorts cdx
    OLDEST-first and nothing re-sorts it on the way here, so those reads would silently use
    the wrong end (stale windows, sign-flipped growth, time-reversed Piotroski).

    THIS IS THE ONE PLACE the live scorer establishes that order, which is the whole point:
    the individual metrics do NOT re-sort, so there is exactly one site to get right instead
    of nineteen to remember (see reporting_period's "ROW ORDER" section for the shared
    vocabulary, and `moatIdentifier` for what the per-consumer version costs).  It stays a
    WHOLE-FRAME groupwise sort rather than rp.to_newest_first per source, deliberately: a
    per-source sort would change tie order on duplicate-dated (restated) rows, and this
    frame has them.

    Re-sorts a COPY, robustly: dates are coerced to datetime (a naive string sort
    mis-orders mixed/malformed date strings).  The COPY matters because cdx_dftop100 is also
    stored in resdic and must not be mutated.  Per-ticker row count and NaT count are
    asserted unchanged, and the result DECLARES its order so a downstream reader can check
    rather than assume.
    """
    cdxtop = cdxtop.copy()
    _n_before = cdxtop.groupby('source').size()
    cdxtop['date'] = pd.to_datetime(cdxtop['date'], errors='coerce')
    _nat_before = int(cdxtop['date'].isna().sum())
    cdxtop = cdxtop.sort_values(['source', 'date'], ascending=[True, False]).reset_index(drop=True)
    assert cdxtop.groupby('source').size().equals(_n_before), \
        "Stage-2 newest-first re-sort changed per-ticker row counts"
    assert int(cdxtop['date'].isna().sum()) == _nat_before, \
        "Stage-2 newest-first re-sort changed NaT count"
    # Observed, not assumed: the sort above is the guarantee, and this line is what makes the
    # guarantee reportable if a future change (a re-sort, a re-index, a merge) breaks it.
    _safe_diagnose(rp.assert_newest_first, cdxtop, 'Stage-2 cdxtop', by='source')
    return rp.stamp_row_order(cdxtop, rp.NEWEST_FIRST)


# OFFLINE SEAM for the DCF leg (added 2026-07-27; DEFAULT FLIPPED TO SKIP 2026-07-31).
#
# Stage-2's ONLY network dependency is a per-ticker discounted-cash-flow call, and it buys
# NOTHING: it exists to compute `DcfToPrice`, which carries w = 0.000 in the decisional vector
# and is therefore multiplied by zero.  The seam was built so a SAVED panel could be re-scored
# through the deployed path without live calls -- but it was OFF by default, so every
# production run still paid for the calls it could not use.
#
# THE COUNT IS 225 PER RUN, NOT ~100.  `postBoScoreRanking` runs ONCE PER POOL, not once per
# run: the general pool (head(100)) plus five carve-out cohorts (head(25) each).  Verified
# against the shipped 2026-07-17 resdic -- all five cohorts were fully populated at 25 names,
# so 100 + 5x25 = 225 GETs, every one of them feeding a zero-weight metric.  With a standing
# work-machine call-count caution and a ~12h fetch the same night, that is 225 calls of pure
# cost.  Hence: SKIP BY DEFAULT.
#
# PROVABLY SCORE-NEUTRAL TODAY, AND ENFORCED -- the enforcement is the whole point.  Skipping
# the fetch leaves the DCF frame empty, exactly as an HTTP failure already does on any run.
# That is score-neutral only while every weight on `DcfToPrice` is zero, which is a fact about
# today's vectors and NOT a law.  So if the weight ever becomes non-zero,
# `_assert_offline_dcf_is_score_neutral` REFUSES the run rather than emitting a silently
# different ranking with a blank metric.  Flipping the default makes that guard load-bearing on
# EVERY production run instead of only inside the offline tools, which is why it is also
# widened below to cover the per-cohort weight overrides.
#
# TO RE-ENABLE THE LIVE FETCH: VA_OFFLINE_NO_DCF=0 (or 'false' / 'no' / 'off').
# `=1` still means skip, so the six baseline_tools scripts that do
# `os.environ.setdefault('VA_OFFLINE_NO_DCF', '1')` are unaffected -- and they stay
# unaffected now that the flag is read per call rather than at import, because they set it
# BEFORE the first call as well as before the import.
_DCF_ENV = 'VA_OFFLINE_NO_DCF'
_ENV_FALSEY = {'0', 'false', 'no', 'off', 'none', 'null'}


def _env_flag(name, default):
    """Env flag as a real TRUTH test, not a presence test.

    A presence test is the audit-C2 footgun in reverse: with the default now ON, an operator
    writing `VA_OFFLINE_NO_DCF=0` to re-enable the live fetch must actually get the live fetch,
    not be ignored.  An UNSET or empty value takes `default`; anything explicitly falsey is
    False; anything else is True.
    """
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    return str(v).strip().lower() not in _ENV_FALSEY


#  DEFAULT: skip the 225 useless GETs.  The VALUE is read PER CALL, not at import -- see
#  offline_no_dcf().
OFFLINE_NO_DCF_DEFAULT = True

_DCF_BANNER_SHOWN = False


def offline_no_dcf():
    """Whether Stage-2 should SKIP the per-ticker DCF fetch, read AT CALL TIME.

    IT USED TO BE A MODULE CONSTANT EVALUATED AT IMPORT (`OFFLINE_NO_DCF = _env_flag(...)`),
    and that shape caused a real, if latent, fault (fixed 2026-08-02):

      * testing the env semantics required a genuine `importlib.import_module` after
        monkeypatching the variable, because nothing re-read the environment;
      * that mutates `sys.modules`, which monkeypatch does NOT restore, so a test that
        re-imported with `=0` left the WHOLE SESSION holding a postBoRank whose flag said
        FETCH -- verified: the stale module is the same object and keeps the wrong value.
        Any later test, or test FILE, importing postBoRank silently got that module.  The
        suite passed anyway, which is what makes this class of fault dangerous: it is an
        ORDER-DEPENDENT latent fault, not a visible failure.
      * and operationally the import-time read is a footgun in its own right: an operator
        (or a wrapper script) exporting VA_OFFLINE_NO_DCF=0 AFTER postBoRank was first
        imported was silently ignored.

    Reading in the function makes the env var mean what it says whenever it is asked, and
    removes the need for any re-import machinery. Both consumers -- the score-neutrality
    guard and the fetch itself -- call this, so they cannot disagree within a run unless the
    environment genuinely changes mid-run.
    """
    return _env_flag(_DCF_ENV, default=OFFLINE_NO_DCF_DEFAULT)


def _dcf_weights_in_force(weight_override=None):
    """Every weight `DcfToPrice` carries anywhere that can reach a score, as {where: w}.

    WHY THE COHORT OVERRIDES ARE INCLUDED (2026-07-31).  The original guard read only
    `cdic.getPostDict()`.  But `postBoScoreRanking` is also called per carve-out cohort with a
    `weight_override` from `carveOut.COHORT_WEIGHTS`, and a cohort override could give
    `DcfToPrice` a non-zero weight without touching the main vector -- which would slip past a
    guard that only inspects the main one.  All five cohorts are 0.0 today (verified), so this
    widening is precautionary; it matters because flipping the default makes this guard the only
    thing standing between a future weight change and a silently blank metric.
    """
    out = {}
    try:
        postBm, postNew = cdic.getPostDict()
        out['getPostDict'] = float({**postBm, **postNew}
                                   .get('DcfToPrice', {}).get('w', 0) or 0)
    except Exception as _e:                     # a broken dict must not mask the check
        out['getPostDict'] = 'UNREADABLE (%s)' % type(_e).__name__
    if isinstance(weight_override, dict) and 'DcfToPrice' in weight_override:
        try:
            out['weight_override'] = float(weight_override['DcfToPrice'] or 0)
        except (TypeError, ValueError):
            out['weight_override'] = 'UNREADABLE'
    return out


def _assert_offline_dcf_is_score_neutral(weight_override=None):
    """Refuse to skip the DCF fetch if DcfToPrice has acquired a weight ANYWHERE in force."""
    global _DCF_BANNER_SHOWN
    if not offline_no_dcf():
        return
    weights = _dcf_weights_in_force(weight_override)
    bad = {k: v for k, v in weights.items() if not isinstance(v, float) or v != 0.0}
    if bad:
        raise SystemExit(
            "Stage-2 DCF fetch is SKIPPED (%s default, or =1) but DcfToPrice now carries a "
            "non-zero / unreadable weight: %r.  Skipping the fetch is only score-neutral while "
            "that weight is 0 -- REFUSING rather than emitting a silently different ranking "
            "with a blank metric.  Either set %s=0 to fetch it live, or re-zero the weight."
            % (_DCF_ENV, bad, _DCF_ENV))
    # Print ONCE per process, not once per pool: postBoScoreRanking runs 6 times, and a banner
    # repeated 6 times in the run log trains the reader to skip it.
    if not _DCF_BANNER_SHOWN:
        _DCF_BANNER_SHOWN = True
        _how = ('DEFAULT' if os.environ.get(_DCF_ENV) in (None, '')
                else '%s=%s' % (_DCF_ENV, os.environ.get(_DCF_ENV)))
        print("!" * 78, flush=True)
        print("!!! Stage-2 per-ticker DCF fetch SKIPPED (%s) -- saves ~225 live FMP GETs per\n"
              "!!! run (100 general + 5 cohorts x 25).  DcfToPrice w=0.000 in every vector in\n"
              "!!! force, so AggScore is UNAFFECTED; the DCF column reads empty/NaN in any\n"
              "!!! DISPLAY that shows it.  Set %s=0 to fetch it live." % (_how, _DCF_ENV),
              flush=True)
        print("!!! NOTE: this gates STAGE-2 only.  writeBoAggToCSV and createPresentation make\n"
              "!!! their own DCF calls on separate code paths that this flag does NOT touch.",
              flush=True)
        print("!" * 78, flush=True)


def _fetch_ticker_dcf(ticker, baseurl, api_key, dcf_bulk_dict):
    """Fetch (or reuse bulk) DCF data for one ticker and return it as a normalised
    DataFrame.  Returns (dcf_df, dcf_from_bulk, resp_dcf_status, resp_dcf).
    """
    if offline_no_dcf():
        return pd.DataFrame(), False, "offline-skipped", None
    dcf_from_bulk = ticker in dcf_bulk_dict
    resp_dcf = None
    if dcf_from_bulk:
        dcf_data = [dcf_bulk_dict[ticker]]
        resp_dcf_status = "bulk"
    else:
        # Fallback to individual API call.  HARDENED (fix, 2026-07-31): this ran as a bare
        # requests.get with NO TIMEOUT and NO RETRY, ~100 times immediately after 12+ hours of
        # sustained API load.  With no timeout a hung socket stalls the unattended run
        # indefinitely; the existing bare `except` only covered `.json()`, not the GET itself,
        # so a connection error propagated out of the Stage-2 scorer. safe_http_get gives it
        # the same 10s timeout / 3 retries / backoff discipline as the fetch loop, and returns
        # a _FailedResponse rather than raising, so a dead endpoint costs DcfToPrice for this
        # ticker (weight 0.000 in the live vector) instead of the whole ranking.
        resp_dcf = gdg.safe_http_get(
            f'{baseurl}v3/discounted-cash-flow/{ticker}?apikey={api_key}')
        resp_dcf_status = getattr(resp_dcf, 'status_code', None)
        try:
            dcf_data = resp_dcf.json() if resp_dcf_status == 200 else []
        except Exception:
            dcf_data = []

    # Convert bulk data (already in dict format) or API response to DataFrame
    dcf = pd.DataFrame.from_dict(dcf_data) if dcf_data and isinstance(dcf_data, list) else pd.DataFrame()

    # Normalize DCF column names - bulk CSV might have different names than JSON API
    if not dcf.empty:
        column_mapping = {}
        for col in dcf.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if col_lower == 'stockprice' or col_lower == 'stock_price':
                column_mapping[col] = 'Stock Price'
            elif col_lower == 'dcf':
                column_mapping[col] = 'dcf'
        if column_mapping:
            dcf = dcf.rename(columns=column_mapping)

    return dcf, dcf_from_bulk, resp_dcf_status, resp_dcf


def _compute_ticker_metrics(ticker, tempcdx, dcf, bstop, nq, tempcntr,
                            postBmRankingDict, postNewRankingDict, postScoreMetric_df,
                            rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Compute every Stage-2 metric for one ticker and write it into
    postScoreMetric_df.  All formulas come from the shared stage2_metrics module
    (kept in lockstep with the offline reproduction); this function only decides
    which column each result lands in.

    `rpy` is this TICKER's rows-per-year (4 quarterly / 2 semi-annual,
    reporting_period): it is threaded into every windowed or YoY metric so a
    semi-annual filer's windows span the same CALENDAR time as a quarterly peer's.
    rpy=4 is bit-identical to the previous behaviour.
    """
    def setv(col, val):
        postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, col] = val

    # ---- postBmRankingDict metrics ----
    for key1 in postBmRankingDict.keys():
        setv(key1, sm.postbm_metric(key1, postBmRankingDict[key1]['eqMet'], tempcdx, nq,
                                    rpy=rpy))

    # ---- postNewRankingDict metrics ----
    tempfcf = tempcdx.freeCashFlow
    tempshares = tempcdx.weightedAverageShsOut
    tempmcap = tempcdx.marketCap

    # `tempcdx=` is what lets these two see the NAME-level calendar-gap test (nan_policy);
    # every other windowed metric already receives the frame.  They are the SCORING path, so
    # they must pass it -- test_nan_policy pins that this call site does.
    setv('freeCashFlowYield',
         sm.free_cash_flow_yield(tempfcf, tempmcap, nq, rpy=rpy, tempcdx=tempcdx))
    setv('freeCashFlowPerShareGrowth',
         sm.free_cash_flow_per_share_growth(tempfcf, tempshares, nq, rpy=rpy,
                                            tempcdx=tempcdx))
    # EPStoEPSmean's BASELINE window is stage2_metrics.EPS_MEAN_BASE_NQ (28 quarters), NOT
    # the ambient `nq` scoring window of this function -- do NOT "thread nq through" here.
    # `nq` is the head(n) averaging window the postBm/postNew ratio metrics use (16); the
    # mean-reversion baseline needs a full business cycle and is a different quantity. rpy
    # MUST be passed so that baseline spans the same calendar time for a semi-annual filer.
    setv('EPStoEPSmean', sm.eps_to_eps_mean(tempcdx, rpy=rpy))
    setv('marketCapRevQuants', tempcdx.mcapQuants.iloc[0])
    setv('tbVpRatio', sm.tbv_p_ratio(tempcdx, nq, rpy=rpy))
    setv('Altman-Z', sm.altman_z(tempcdx, rpy=rpy))
    setv('Piotroski', sm.piotroski(tempcdx, rpy=rpy))
    # The two Piotroski components extracted as standalone metrics (E-2).  They carry
    # weight ONLY in the FIN-1 cohort vector and 0.000 in the general one, but they are
    # computed for EVERY pool: a metric that exists in one pool and not another would make
    # `postScoreMetric_df`'s schema pool-dependent, and every consumer reads it as fixed.
    setv('shareCountChange', sm.share_count_change(tempcdx, rpy=rpy))
    setv('longTermDebtChange', sm.long_term_debt_change(tempcdx, rpy=rpy))
    setv('priceGrowth', sm.price_growth(tempcdx, nq, rpy=rpy))
    # rpy MUST be passed: CycleHeat is a self-reference z-score, so its window length
    # decides the baseline. It was the only metric in this block taking neither nq nor
    # rpy (fix 2026-07-30); see stage2_metrics.CYCLEHEAT_BASE_NQ.
    setv('CycleHeat', sm.cycleheat(tempcdx, rpy=rpy))

    # DcfToPrice needs the live DCF frame; diagnostic-log missing columns for the
    # first ticker (matches the historical behaviour).
    if not dcf.empty and tempcntr == 0 and not (
            'dcf' in dcf.columns and any(c in dcf.columns for c in
                                         ('Stock Price', 'StockPrice', 'stock_price'))):
        print(f"  WARNING: DCF missing required columns. Available: {list(dcf.columns)}, "
              f"need: ['dcf', price_col]", flush=True)
    setv('DcfToPrice', sm.dcf_to_price(dcf, nq))

    # BoScore is a straight pass-through of the Stage-1 score (weight 0 in the live
    # vector).  Assigned as a Series to preserve the historical index-alignment.
    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, 'BoScore'] = \
        bstop.loc[bstop['source'] == ticker, 'score']


# --------------------------------------------------------------------------- #
#  Diagnostics (stdout only -- no effect on the emitted ranking)              #
# --------------------------------------------------------------------------- #
def _diagnose_inputs(bmtop, bstop, cdxtop):
    print("\n" + "=" * 60, flush=True)
    print("DIAGNOSTIC: Input dataframes check (BEFORE calculations)", flush=True)
    print("=" * 60, flush=True)

    if bmtop.empty:
        print("ERROR: bmtop (BoMetric top 100) is EMPTY!", flush=True)
    else:
        print(f"bmtop shape: {bmtop.shape} (rows, columns)", flush=True)
        print(f"bmtop unique sources: {bmtop['source'].nunique() if 'source' in bmtop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in bmtop.columns:
            print(f"bmtop sample sources: {list(bmtop['source'].head(3).values)}", flush=True)
        print(f"bmtop columns (first 10): {list(bmtop.columns[:10])}", flush=True)
        numeric_cols = bmtop.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            nan_pct = (bmtop[numeric_cols].isna().sum() / len(bmtop) * 100).round(1)
            print(f"bmtop NaN percentage in numeric columns (first 5): {dict(nan_pct.head(5))}", flush=True)

    if bstop.empty:
        print("ERROR: bstop (BoScore top 100) is EMPTY!", flush=True)
    else:
        print(f"\nbstop shape: {bstop.shape} (rows, columns)", flush=True)
        print(f"bstop unique sources: {bstop['source'].nunique() if 'source' in bstop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in bstop.columns:
            print(f"bstop sample sources: {list(bstop['source'].head(3).values)}", flush=True)
        if 'score' in bstop.columns:
            print(f"bstop score stats: min={bstop['score'].min():.4f}, max={bstop['score'].max():.4f}, mean={bstop['score'].mean():.4f}", flush=True)

    if cdxtop.empty:
        print("\nERROR: cdxtop (cdx top 100) is EMPTY!", flush=True)
    else:
        print(f"\ncdxtop shape: {cdxtop.shape} (rows, columns)", flush=True)
        print(f"cdxtop unique sources: {cdxtop['source'].nunique() if 'source' in cdxtop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in cdxtop.columns:
            print(f"cdxtop sample sources: {list(cdxtop['source'].head(3).values)}", flush=True)
        print(f"cdxtop columns (first 10): {list(cdxtop.columns[:10])}", flush=True)
        required_cols = ['freeCashFlow', 'weightedAverageShsOut', 'marketCap', 'grahamNumber', 'price',
                         'tangibleBookValuePerShare', 'totalAssets', 'totalLiabilities', 'totalCurrentAssets',
                         'totalCurrentLiabilities', 'totalStockholdersEquity', 'operatingIncome', 'revenue',
                         'netIncome', 'netCashProvidedByOperatingActivities', 'longTermDebt', 'currentRatio',
                         'grossProfitMargin']
        missing_cols = [col for col in required_cols if col not in cdxtop.columns]
        if missing_cols:
            print(f"WARNING: cdxtop missing required columns: {missing_cols}", flush=True)
        else:
            print(f"cdxtop has all required columns: {required_cols}", flush=True)
        key_cols = [col for col in required_cols if col in cdxtop.columns]
        if key_cols:
            nan_pct = (cdxtop[key_cols].isna().sum() / len(cdxtop) * 100).round(1)
            print(f"cdxtop NaN percentage in key columns: {dict(nan_pct)}", flush=True)

    print("=" * 60 + "\n", flush=True)
    sys.stdout.flush()


def _safe_diagnose(fn, *args, **kwargs):
    """Run a PRINT-ONLY diagnostic; never let it abort the caller.

    Stage-2 has no per-ticker guard and postBoWrapper does not wrap postBoScoreRanking, so an
    exception anywhere in this file costs every deliverable of the night.  A helper whose only
    job is to print must never be able to do that (review B1, 2026-07-31 -- `resp_dcf.text` on
    a _FailedResponse did exactly this).  The failure is reported LOUDLY rather than swallowed:
    a silently-missing diagnostic is how the frequency watchdog went dark for a whole release.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as _e:
        print("WARNING: diagnostic %s FAILED (%s: %s) -- scoring continues, but this "
              "diagnostic's output is MISSING from the run log."
              % (getattr(fn, '__name__', fn), type(_e).__name__, _e), flush=True)
        return None


def _diagnose_first_ticker_data(ticker, dcf, dcf_from_bulk, resp_dcf_status, resp_dcf, tempcdx):
    print(f"\nDIAGNOSTIC: First ticker ({ticker}) data:", flush=True)
    print(f"  DCF source: {'bulk' if dcf_from_bulk else 'individual'}, status: {resp_dcf_status}, empty: {dcf.empty}, shape: {dcf.shape if not dcf.empty else 'N/A'}", flush=True)
    if not dcf.empty:
        print(f"  DCF columns: {list(dcf.columns)}", flush=True)
    print(f"  tempcdx (fundamentals) empty: {tempcdx.empty}, shape: {tempcdx.shape if not tempcdx.empty else 'N/A'}", flush=True)
    if not tempcdx.empty:
        print(f"  tempcdx columns: {list(tempcdx.columns[:5])}...", flush=True)
        print(f"  tempcdx sample values (first row):", flush=True)
        print(f"    freeCashFlow: {tempcdx['freeCashFlow'].iloc[0] if 'freeCashFlow' in tempcdx.columns else 'N/A'}", flush=True)
        print(f"    marketCap: {tempcdx['marketCap'].iloc[0] if 'marketCap' in tempcdx.columns else 'N/A'}", flush=True)
        print(f"    operatingIncome: {tempcdx['operatingIncome'].iloc[0] if 'operatingIncome' in tempcdx.columns else 'N/A'}", flush=True)
    if not dcf_from_bulk and resp_dcf_status != 200:
        # getattr, NOT `.text` (review B1, fixed 2026-07-31).  `resp_dcf is not None` is
        # satisfied by getData_gen._FailedResponse, which had no `.text`, so this line raised
        # AttributeError and killed Stage-2 on the FIRST ticker of each of the 6 pools --
        # precisely in the dead/hung-endpoint case the safe_http_get hardening was added for.
        # _FailedResponse now provides `.text`, so this is belt-and-braces: a diagnostic must
        # not depend on the exact response type it happens to be handed.
        _body = getattr(resp_dcf, 'text', None) if resp_dcf is not None else None
        print(f"  DCF error: {str(_body)[:100] if _body is not None else 'N/A'}", flush=True)
    print(f"  Note: Altman-Z and Piotroski calculated from tempcdx fundamentals", flush=True)


def _diagnose_first_ticker_metrics(ticker, postScoreMetric_df):
    sample_metrics = ['RoA', 'earnYield', 'grahamNumberToPrice', 'freeCashFlowYield', 'BoScore', 'Altman-Z', 'Piotroski', 'CycleHeat']
    print(f"\nDIAGNOSTIC: Sample metric values after calculation for {ticker}:", flush=True)
    for metric in sample_metrics:
        if metric in postScoreMetric_df.columns:
            val = postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, metric].values
            if len(val) > 0 and not pd.isna(val[0]):
                print(f"  {metric}: {val[0]}", flush=True)
            else:
                print(f"  {metric}: NOT CALCULATED (NaN)", flush=True)
        else:
            print(f"  {metric}: COLUMN NOT FOUND", flush=True)
    print(f"  Note: Altman-Z and Piotroski are now calculated from fundamentals (no API needed)", flush=True)


def _diagnose_pre_normalize(postScoreMetric_df):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: postScoreMetric_df statistics before normalizeAndDropNA")
    print("=" * 60)
    print(f"DataFrame shape: {postScoreMetric_df.shape} (rows, columns)")
    print(f"Total rows: {len(postScoreMetric_df)}")
    print(f"Columns: {list(postScoreMetric_df.columns)}")

    if not postScoreMetric_df.empty:
        metric_cols = [col for col in postScoreMetric_df.columns if col != 'source']
        print(f"\nMetric columns (excluding 'source'): {len(metric_cols)}")

        if len(metric_cols) > 0:
            numeric_df = postScoreMetric_df[metric_cols].apply(pd.to_numeric, errors='coerce')

            print("\nNaN statistics per column:")
            nan_counts = numeric_df.isna().sum()
            nan_pct = (nan_counts / len(numeric_df) * 100).round(2)
            for col in metric_cols:
                print(f"  {col}: {nan_counts[col]} NaN ({nan_pct[col]}%)")

            total_cells = len(numeric_df) * len(metric_cols)
            total_nan = numeric_df.isna().sum().sum()
            print(f"\nOverall: {total_nan}/{total_cells} NaN values ({total_nan/total_cells*100:.2f}%)")

            rows_all_nan = (numeric_df.isna().sum(axis=1) == len(metric_cols)).sum()
            print(f"Rows with ALL metrics NaN: {rows_all_nan}/{len(numeric_df)} ({rows_all_nan/len(numeric_df)*100:.2f}%)")

            rows_some_valid = (numeric_df.isna().sum(axis=1) < len(metric_cols)).sum()
            print(f"Rows with at least one valid metric: {rows_some_valid}/{len(numeric_df)} ({rows_some_valid/len(numeric_df)*100:.2f}%)")

            print("\nColumn statistics (for non-NaN values):")
            for col in metric_cols:
                col_data = numeric_df[col].dropna()
                if len(col_data) > 0:
                    print(f"  {col}: mean={col_data.mean():.4f}, min={col_data.min():.4f}, max={col_data.max():.4f}, count={len(col_data)}")
                else:
                    print(f"  {col}: ALL NaN (no valid values)")
        else:
            print("WARNING: No metric columns found!")
    else:
        print("WARNING: DataFrame is empty!")

    print("=" * 60 + "\n")


def _dedup_issuers_in_ranking(postRank, cdxtop, names, dedup_issuers):
    """Issuer-level de-dup of the EMITTED ranking (CEO standing principle: NO
    duplicate issuers in the deployed top-N).  postRank is AggScore-descending, and
    downstream emission (writeBoAggToCSV / createPresentation) takes head(N) off it,
    so collapsing same-issuer lines HERE makes the CEO-reviewed top-20 contain
    DISTINCT issuers.  We keep the HIGHEST-RANKED line per issuer and drop later
    same-issuer lines (share-classes / cross-listings, e.g. TFPM / TFPM.TO) -- the
    SAME rank-based rule and SAME fingerprint (carveOut.dedup_ranked /
    _issuer_components) the backtest harness (stage2_pit.reproduce_pit_top) uses.  So
    live and backtest agree on issuer IDENTITY and economic exposure (one slot per
    issuer) -- NOT necessarily on the specific surviving TICKER: on the carve-ON live
    path the upstream carve already collapsed each issuer to its mcap/sector-preferred
    line, whereas the backtest / carve-OFF path keeps the highest-RANKED line.  Both
    satisfy "distinct issuers".

    This changes ONLY which lines survive into the emitted ranking; no score, no sort
    order, and no other pick logic is touched.  For a LIVE run as_of is NOW, so the
    fingerprint reads CURRENT fundamentals -> merging same-issuer listings is correct
    with NO lookahead (the PIT-purity caveat applies only to backtest dedup at past D).
    If the carve-out already collapsed the universe to one line per issuer upstream,
    this is a safe no-op; it is load-bearing on the carve-off / carve-fallback path
    (and any carve-missed listing).

    Returns (postRank, issuer_dupes_dropped).
    """
    issuer_dupes_dropped = []
    if not dedup_issuers:
        return postRank, issuer_dupes_dropped
    try:
        import carveOut as _co
        ranked_srcs = postRank['source'].tolist()
        # AggScore in: an issuer's clone lines score EXACTLY equal, so which one the
        # sort happened to emit first was arbitrary -- resolve those ties by
        # investability rather than by sort stability (carveOut.dedup_ranked TIE-BREAK).
        _sc = postRank['AggScore'] if 'AggScore' in postRank.columns else None
        kept, issuer_dupes_dropped = _co.dedup_ranked(ranked_srcs, cdxtop,
                                                      names or {}, scores=_sc)
        if issuer_dupes_dropped:
            # HONOUR `kept`'s ORDER, do not merely filter by its MEMBERSHIP (review B1, fixed
            # 2026-08-05).  This line was `postRank[postRank['source'].isin(set(kept))]`, which
            # keeps postRank's own row order and throws the returned SEQUENCE away.
            #
            # THAT IS LIVE, NOT LATENT, AND THE SHARP STATEMENT IS THE DISAGREEMENT: two consumers
            # of the SAME `dedup_ranked` call, in the SAME run, read the result differently --
            # `carveOut.partition_by_marketcap` does `reindex(kept)` and so HONOURS the order,
            # while this site discarded it.  So the two could place one issuer at two different
            # positions from one dedup verdict.  `dedup_ranked` resolves exact-tie clone lines by
            # INVESTABILITY rather than sort stability, and that tie-break is expressed ONLY in
            # the order of `kept`; a membership filter cannot see it.
            # MEASURED SEVERITY on the local panel: the top-20 SETS agree and only the order at
            # #13 diverges -- small, but a divergence between two readings of one verdict is not
            # something to leave in place, and the ejection reading of the mechanism is
            # unquantified rather than excluded.
            # `reindex` on a source-indexed frame is the same idiom carveOut uses, deliberately.
            if postRank['source'].duplicated().any():
                # Cannot happen -- postRank is one row per source -- but a position-preserving
                # reindex on a duplicated index is AMBIGUOUS, and guessing would be exactly the
                # silent mis-ordering this fix removes.  Raise, so the LOUD FALLBACK below names
                # the real cause instead of the dedup quietly reverting to a membership filter.
                raise ValueError(
                    'postBoRank issuer-dedup: postRank carries duplicate `source` rows, so '
                    'honouring dedup_ranked\'s ORDER is ambiguous. postRank is built one row per '
                    'source; a duplicate means an upstream defect, not a dedup problem.')
            _present = set(postRank['source'])
            _kept_present = [s for s in kept if s in _present]
            postRank = (postRank.set_index('source').reindex(_kept_present)
                        .reset_index())
            print("postBoRank issuer-dedup: collapsed %d same-issuer line(s) in the "
                  "ranking -> %s"
                  % (len(issuer_dupes_dropped),
                     ['%s->%s' % (d, k) for d, k in issuer_dupes_dropped]), flush=True)
    except Exception as _e:
        # LOUD FALLBACK (matches the carve-out banner in postBo.postBoWrapper). The
        # emission-time issuer-dedup IS the "no duplicate issuers in the top-20"
        # guarantee; if it fails we still ship (never crash the deliverable) but the
        # emitted top-20 may carry dual-listings / share-classes, so a single quiet
        # stdout line is a defect -- make the degradation IMPOSSIBLE to miss on BOTH
        # streams, exactly like the carve fallback.
        import traceback
        _banner = (
            "\n" + "!" * 78 + "\n"
            "!!! ISSUER-DEDUP DID NOT RUN -- EMITTED TOP-20 MAY CONTAIN DUAL-LISTINGS !!!\n"
            "!!! The ranking was NOT de-duplicated by issuer this run: expect possible !!!\n"
            "!!!   share-class / cross-listing DUPLICATES in the top-N (e.g. TFPM +     !!!\n"
            "!!!   TFPM.TO occupying two slots for one economic bet).                   !!!\n"
            "!!! Cause: %s: %s\n"
            "!!! DO NOT treat this top-20 as issuer-deduplicated.                       !!!\n"
            % (type(_e).__name__, _e)
            + "!" * 78 + "\n")
        print(_banner, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(_banner, flush=True)
        traceback.print_exc(file=sys.stdout)
    return postRank, issuer_dupes_dropped


# --------------------------------------------------------------------------- #
#  ROBUST NORMALISATION  (issue E-1, implemented 2026-08-03)                   #
# --------------------------------------------------------------------------- #
# WHAT REPLACED WHAT, and why it is a RE-PARAMETERISATION rather than a new subsystem.
#
# Until 2026-08-03 the z-path WINSORIZED each raw column at +-3 sigma (`_winsorize_raw`,
# removed in the same change) and then took mu/sigma of the clipped column.  That loop was
# read as an ad-hoc two-pass clip.  IT IS NOT: it is a HUBER PROPOSAL-2 SCALE M-ESTIMATOR,
# and the equivalence is exact, not approximate.  VERIFIED, not asserted:
#
#   The loop's update was  s_{p+1} = sd(clip(x; m_p +- c*s_p)),  m_{p+1} = mean(...).
#   At its fixed point, (y - m)/s == clip((x - m)/s, -c, +c) == psi(u), so
#       s^2 == var_{ddof=1}(psi(u) * s)   <=>   mean_{ddof=0}(psi(u)^2) == (n-1)/n,
#   which IS Huber Proposal 2's estimating equation  mean(psi(u)^2) = b  with b := (n-1)/n
#   instead of the normal-consistency constant b := beta(c).
#   MEASURED on the 18 weighted columns of resdic_2026-07-17_CORRECTED (2026-08-03):
#     * mean(psi^2) at the shipped fixed point = 0.990000 on every n=100 column and
#       0.988889 on the n=90 column -- i.e. EXACTLY (n-1)/n, to 6 dp, on all 18;
#     * an independently written Huber P2 iteration (c=3, b=1, mean-centred) reproduces the
#       shipped sigma to a worst relative difference of 4.5e-10 on all 18 real columns and
#       on 6 synthetic DGPs x 400 replications (clean normal, 5% at +20 at n=100 and n=25,
#       a bunched pair at n=100 and n=25, lognormal).
#
# SO THE OLD DEFECT WAS ONE CONSTANT, NOT THE STRUCTURE.  At c = 3 the Huber weight
# min(1, c/|z|) downweights essentially nothing, which is why the estimator still carried
# 35% sigma inflation under 5% point contamination at n=100 and 119% on a bunched pair at
# cohort size n=25 [sim, normalisation-spec.md N3].  Three things change here and the loop
# itself is KEPT:
#   (1) c: 3.0 -> HUBER_C = 1.5, so the weight actually bites;
#   (2) b: (n-1)/n -> beta(c), the normal-consistency factor, so sigma-hat is unbiased on a
#       clean column (worth ~0.5% at c=3 -- beta(3) = 0.995007 -- and 12% at c=1.5, where it
#       is NOT optional: beta(1.5) = 0.778465);
#   (3) the centre: the iterated MEAN of the clipped column -> the MEDIAN of the observed
#       values, fixed, which is the CEO's "same subset for the mean and the deviation" and is
#       what makes an unavailable metric's fillna(0) land AT the observed median.
# NOTHING IS CLIPPED and NOTHING IS EXCLUDED any more: every value keeps its own number and
# is placed on the resulting ruler (then squashed, see SQUASH_K).  That is what retires both
# of the old exemption lists -- see BOUNDED_DISCRETE_COLUMNS.
HUBER_C = 1.5
# Bound on the Huber fixed-point iteration.  MEASURED, and the COHORTS are what set it -- the
# general pool alone would have justified a much tighter bound and a tighter bound would have
# been wrong:
#     general pool, n=100     worst 71 passes (incomeQuality)
#     the five cohorts, n=25  worst 133 passes (REIT currentRatio); 105-133 on the six
#                             columns that exceed 100, three of them WEIGHTED
#                             (InvestmentVehicle bVpRatio w=0.597 at 127, FinManager
#                             freeCashFlowPerShareGrowth w=0.106 at 128, BalanceSheetFin
#                             incomeQuality w=0.024 at 130)
# The approach is monotone and linear, not oscillatory (verified: s_p increases every pass and
# |s_{p+1} - s_p| falls ~1e-2 -> 1e-6 -> 1e-11 by pass 100), so a bound of 100 would not have
# produced a WRONG scale -- sigma at pass 100 is already right to ~10 significant figures --
# but it would have parked six real columns at status 'maxiter' every run, which trains the
# reader to ignore the status.  500 gives 3.8x headroom over the worst observed case, and each
# pass is one vectorised op over <= 100 rows, so the headroom is free.
HUBER_MAX_PASSES = 500
HUBER_REL_TOL = 1e-13
MAD_TO_SIGMA = 1.4826     # 1/Phi^-1(0.75): makes MAD consistent for sigma on a normal

# CONCAVE BOUNDED SQUASH -- the ALGEBRAIC sigmoid  zeta = k*z / sqrt(k^2 + z^2),  ONE GLOBAL k.
#
# Applied to EVERY value, not only past a boundary: it is C-infinity and zeta'(0) = 1, so
# applying it everywhere is near-identity through the bulk and introduces no kink, whereas a
# boundary-only application would put one exactly at the boundary.
#
# NOT `k*tanh(z/k)`, and this is a concrete float64 defect rather than a preference:
# k*tanh(z/k) equals k EXACTLY in float64 for |z| >= 19.0615*k, i.e. 57.185 at k = 3 (bisected
# 2026-08-03; the spec's 18.99*k / 56.971 is 0.4% low -- tanh(18.99) is one ULP short of 1.0 --
# which changes nothing about the defect).  So tanh RE-CREATES the tie defect the squash exists
# to remove, just further out, and it already saturates one cell on the real panel
# (incomeQuality, honest |z| = 186), on the very column that holds 96.6% of its own raw variance
# in one name.  The algebraic form saturates at |z| ~ 1.8e8, eight orders of margin.
#
# WHY k = 3, and it is arithmetic rather than taste.  The binding property is that NO SINGLE
# COLUMN MAY, ON ITS OWN, CARRY A MEDIAN-SCORING NAME INTO THE TOP 20: the realised worst
# single-column reach max_c |w_c| * max_i |zeta_ic| must stay under the pool's own
# median-to-rank-20 AggScore distance.  On [pool = the 2026-07-17 CORRECTED general top-100]
# that distance is 0.2560 and k = 3 gives 0.2160 (0.84x, 16% margin) against 0.2879 at k = 4
# (1.12x -- VIOLATES) and 0.2666 for the pre-2026-08-03 hard |z| <= 3 bound (1.04x -- the
# shipped design ALREADY violated it).  k = 3 is the largest round value that holds, at a cost
# of 3.9% of the p50->p80 decision-band span.
# THE PROPERTY IS PANEL-DEPENDENT AND IS THEREFORE CHECKED ON EVERY RUN, per pool, by
# single_column_reach_check() -- do not let k = 3 become a number with no panel attached.
SQUASH_K = 3.0

# BOUNDED / DISCRETE columns.  THIS IS NO LONGER A NORMALISATION EXEMPTION -- do not use it
# as one (it was `WINSOR_EXEMPT_BOUNDED` until 2026-08-03).
#
# The old exemption's premise was correct and is now DISSOLVED: these columns were exempt
# from winsorization because clipping a bounded discrete column destroys real structure
# ("winsorizing Piotroski 20x7 + 1x2 moves the worst F-score onto the best").  The robust
# ruler clips nothing, so there is nothing to be exempt from, and MEASURED
# [normalisation-spec.md Test 5] all three take a robust scale cleanly and their sigma-hat
# RISES (Piotroski 1.5594 -> 1.6139, marketCapRevQuants 0.3679 -> 0.4071), so their influence
# FALLS -- the correct direction for the "power transferred to the mechanical metrics" defect.
# What protects the genuinely pathological shapes is now the MEASURED CONDITION (MAD == 0) in
# robust_location_scale, which is the general form this four-name tuple was hand-approximating
# and which does not silently mis-handle the next discrete metric anyone adds.
#
# The tuple survives for ONE purpose only: `normalizeAndDropNA(method=NORM_RANK,
# rank_bounded=False)`, a measurement switch that leaves these columns un-rank-mapped so the
# alternative can be measured.  Production never takes that branch.
# NOTE `mcapQuants` is the cdx-side column name; the postScoreMetric column is
# `marketCapRevQuants`, so that fourth entry has never matched anything in a frame this
# function sees.  Kept rather than quietly dropped (it is inert either way) as the standing
# evidence for why a name list is the wrong instrument.
BOUNDED_DISCRETE_COLUMNS = ('Piotroski', 'CycleHeat', 'marketCapRevQuants', 'mcapQuants')


# --------------------------------------------------------------------------- #
#  NORMALISATION METHOD  (A/B switch, 2026-07-27; RESOLVED 2026-08-03)         #
# --------------------------------------------------------------------------- #
# 'zscore' : THE PRODUCTION PATH.  Since 2026-08-03 this is the ROBUST ruler -- median centre,
#            Huber Proposal-2 scale on the same subset, algebraic squash (see HUBER_C /
#            SQUASH_K and normalizeAndDropNA).  It is NOT the +-3 winsorizer this switch was
#            built to be compared against; that path no longer exists.
# 'rank'   : map the column to ranks, then through the inverse normal CDF.
#
# THE CHOICE IS MADE AND IS NOT OPEN.  CEO, 2026-08-03: STAGE-1 RANKS AND STAGE-2 SCORES
# MAGNITUDES, so rank-normalising Stage-2 would collapse it into a second Stage-1.  The rank
# path is retained as a MEASUREMENT INSTRUMENT only -- do not offer it as a candidate, and do
# not switch the default "later": both rulers change what every weight MEANS, so choosing after
# the weights are set fits weights to a ruler that then moves.
#
# WHAT EACH OF THE TWO ORIGINAL ARGUMENTS FOR 'rank' IS WORTH NOW:
#   (1) CROSS-COLUMN INCOMPARABILITY -- REAL, PARTLY FIXED, AND THE RESIDUAL IS ACCEPTED.
#       A weight w means "w units of AggScore per sigma of THIS column", and a sigma of a
#       skewed column is not a sigma of a symmetric one.  A rank map makes every column the
#       same distribution by construction (1.00x on every band); a location-and-scale ruler
#       cannot, because two columns with identical sigma and different SKEW still deliver
#       different movement per unit of weight.  Measured, max/min per-column span over the 15
#       continuous weighted columns [resdic_2026-07-17_CORRECTED, general top-100]:
#            band        pre-E-1   E-1 robust   rank
#            p25-p75      2.553x     1.397x    1.00x
#            p50-p80      3.057x     2.613x    1.00x   <- the band the rank-20 boundary is in
#       So the middle half is essentially solved (-45%) and the DECISION band improves only 15%.
#       The residual ~2.6x at the shortlist boundary is the ACCEPTED price of magnitude-based
#       scoring (CEO, 2026-08-03), not an open defect.  E-2 should be told: a weight means the
#       same thing to within about +-20% across the middle of every column and to within a
#       factor of ~2.6 at the boundary -- fine for hand-reasoned block budgets, NOT a base on
#       which a 0.293-vs-0.160 split can be read as a precise statement of intent.
#   (2) THE MISSING-DATA REWARD (reviewer finding N1) -- GONE, and not via ranks.
#       Both paths fill an unavailable metric with 0; the question is only where 0 sits.  Under
#       the pre-E-1 mean-centred ruler it was the WINSORIZED MEAN, above the typical name on 14
#       of 18 weighted columns.  Measured on resdic_2026-07-17_CORRECTED (general top-100),
#       advantage := 0 - sum_c w_c * median(z_c over OBSERVED cells):
#            pre-E-1 ruler   advantage +0.0739   median->rank-20 distance 0.2560   = 29%
#            E-1 ruler       advantage  ~2e-18   median->rank-20 distance 0.2396   =  0%
#            columns whose fill beats their own median   14/18  ->  0/18  (tol 1e-12)
#       E-1's median centring makes 0 the observed median EXACTLY -- z = (x - median)/sigma has
#       median 0 by construction, and the squash is monotone with zeta(0) = 0, so it survives
#       the squash.  This is the one place the z-path is now STRICTLY BETTER than the rank map,
#       which cannot make the same claim: `_rank_normal` centres on the median exactly only when
#       a column's values are DISTINCT, and ties displace the centre (`_rank_normal([1, 1, 2])`
#       has mean +0.0260), so on the discrete columns -- `Piotroski` 0..9,
#       `marketCapRevQuants` 5 levels -- the fill is measurably off-median.  Do not restate the
#       rank map's centring as "by construction"; it is not.
#       HISTORICAL NOTE, kept because three mutually inconsistent sets of numbers for this one
#       quantity were in the tree at once and that is the lesson: the figures "+0.1394 against
#       a 0.134 median-to-top-20 distance" (>100%, i.e. missingness alone reaching the
#       shortlist) and "+0.1616 -> +0.0228, 14/18 -> 2/18" were both taken on the SHIPPED pre-fix
#       07-17 panel, whose raw frame no longer exists on this machine, and neither reproduces.
#       EVERY figure above carries its panel for exactly that reason.
#
# DEFAULT: NORM_ZSCORE, now meaning the robust ruler.
NORM_ZSCORE = 'zscore'
NORM_RANK = 'rank'
NORM_METHOD_DEFAULT = NORM_ZSCORE

# Plotting position for the rank -> normal map.  Blom: p_i = (r_i - 3/8) / (n + 1/4).
# Chosen over Van der Waerden r/(n+1) because it is the near-unbiased normal-scores
# constant, i.e. E[z_i] is closest to the expected order statistic, so the resulting
# column's sd is closest to 1 at the pool sizes here (~100 names Stage-2, ~6.5k universe-
# wide).  It is a MONOTONE relabelling either way, so the choice cannot change the ORDER
# within a column -- only the spacing, and therefore how much a given rank gap is worth
# against another column's rank gap.
RANK_PLOT_A = 0.375


def _rank_normal(x):
    """Rank -> inverse-normal ('probit' / normal-scores) map of one metric column.

    * Ties get the AVERAGE rank, so tied names get the SAME score -- required: a
      discrete column (Piotroski, the market-cap quantile codes) is mostly ties, and
      breaking them by row order would inject pure noise into the ranking.
    * NaN keeps NaN and does NOT consume a rank slot, so the percentiles describe the
      OBSERVED population.  The caller's fillna(0) then lands a missing metric AT OR NEAR
      that population's median (finding N1) -- see the tie caveat below for when "near"
      is as good as it gets.
    * CENTRING, stated exactly.  With all values DISTINCT the ranks are symmetric about
      (n+1)/2, so the plotting positions are symmetric about 0.5 and the output has mean
      EXACTLY 0 and median exactly 0.  WITH TIES NEITHER HOLDS: a tied group collapses
      onto one averaged plotting position instead of spanning several, which displaces the
      centre.  `_rank_normal([1, 1, 2])` has mean +0.0260; a 60/40 binary column puts 0
      above ~60% of the pool, not 50%.  So "0 is the median" is a property of
      distinct-valued columns, NOT of this function -- and the discrete columns
      (`Piotroski` 0..9, `marketCapRevQuants` 5 levels) are exactly the ones where it
      fails.  On the 07-17 pool `marketCapRevQuants` alone accounts for +0.0244 of the
      +0.0228 residual missing-data advantage.  Do not restate this as "by construction".
    * sd is CLOSE to but not exactly 1 (finite-sample plotting position, and ties compress
      a column's spread).  That is deliberate and is NOT re-scaled: rescaling each column
      back to unit variance would hand a heavily-tied column the same spread as a
      fully-resolved one, i.e. it would re-weight the vector by tie structure.
    """
    from scipy.special import ndtri            # scipy is a declared dependency
    s = pd.to_numeric(x, errors='coerce')
    valid = s.notna()
    n = int(valid.sum())
    out = pd.Series(np.nan, index=s.index, dtype='float64')
    if n == 0:
        return out
    if n == 1:
        out[valid] = 0.0                        # a single observation IS the median
        return out
    r = s[valid].rank(method='average')
    p = (r - RANK_PLOT_A) / (n + 1.0 - 2.0 * RANK_PLOT_A)
    out[valid] = ndtri(p.to_numpy(dtype='float64'))
    return out


_RobustScale = collections.namedtuple(
    'RobustScale', 'mu sigma status n_passes weight_retained n_obs')


def _huber_beta(c):
    """beta(c) = E[psi(Z)^2] for Z ~ N(0,1), psi = clip(., -c, +c).

    The NORMAL-CONSISTENCY constant of Huber's Proposal 2: dividing mean(psi^2) by it is
    what makes sigma-hat unbiased for sigma on a clean Gaussian column.  Closed form:
        beta(c) = 2*[ c^2*(1 - Phi(c)) + Phi(c) - 0.5 - c*phi(c) ]
    derived from E[min(Z^2, c^2)] = (2*Phi(c) - 1 - 2*c*phi(c)) + 2*c^2*(1 - Phi(c)).
    beta(1.5) = 0.778465, beta(3.0) = 0.995007.

    THE ~1% AT c = 3 IS WHY IT WAS ABSENT AND SURVIVED: the pre-2026-08-03 winsorizer
    effectively used b = (n-1)/n ~ 1, which at c = 3 is a 0.25% bias in sigma-hat and
    therefore invisible.  At c = 1.5 the same omission would be a 12% bias, so it is
    load-bearing here in a way it never was before.

    Uses math.erfc rather than scipy so the constant is a pure deterministic function of c
    (this value multiplies every score; scipy's normal CDF is fine but this has no import
    cost and no version surface).
    """
    c = float(c)
    if not (c > 0.0) or not math.isfinite(c):
        raise ValueError('_huber_beta: c must be finite and positive, got %r' % (c,))
    phi_c = math.exp(-0.5 * c * c) / math.sqrt(2.0 * math.pi)
    Phi_c = 0.5 * math.erfc(-c / math.sqrt(2.0))
    return 2.0 * (c * c * (1.0 - Phi_c) + Phi_c - 0.5 - c * phi_c)


#  beta(c) is called once per column per pool (6 pools x ~21 columns) with one of one or two
#  distinct c values, so memoise it rather than recompute an erfc 126 times.
_HUBER_BETA_CACHE = {}


def huber_beta(c):
    """Memoised _huber_beta."""
    key = float(c)
    if key not in _HUBER_BETA_CACHE:
        _HUBER_BETA_CACHE[key] = _huber_beta(key)
    return _HUBER_BETA_CACHE[key]


def robust_location_scale(x, c=HUBER_C, max_passes=HUBER_MAX_PASSES,
                          rel_tol=HUBER_REL_TOL):
    """LOCATION AND SCALE OF ONE METRIC COLUMN, both estimated on the SAME subset.

    Returns a RobustScale(mu, sigma, status, n_passes, weight_retained, n_obs).

    THE RULE, ONE SENTENCE (normalisation-spec.md N1): a value's contribution to a column's
    location and scale is its estimation weight -- 1 if ordinary, c/|z| if extreme, 0 if not
    observed -- and EVERY value, ordinary or extreme or imputed, is then placed on the
    resulting ruler.  Nothing is clipped, nothing is excluded, no name is ejected.

    LOCATION = THE MEDIAN of the observed values (N2).  Two reasons, and the first is the
    CEO's requirement that the centre and the deviation come from the same subset:
      (a) it centres the squash's near-linear region on where the companies actually are,
          rather than on a mean that skew has dragged into the tail;
      (b) it makes z = 0 -- the value `normalizeAndDropNA` imputes an UNAVAILABLE metric with
          -- sit at the observed median by construction.  MEASURED [pool 2026-07-17
          CORRECTED]: median-centring alone takes the full-missingness AggScore advantage from
          +0.0739 to -0.0000 and the count of columns where being MISSING beats the median
          name from 14 of 18 to 1 of 18.  That is most of issue I-3's arithmetic, obtained as
          a side effect of a decision made for a different reason.
    Location has NO effect on the ranking through the linear part of the path (adding a
    constant to every name's z on a column shifts every AggScore equally), so it acts only
    through those two channels -- and through the squash, which is not linear.

    SCALE = A HUBER PROPOSAL-2 M-ESTIMATE about that median, at c = HUBER_C.  The loop below
    IS the loop the winsorizer used to run (see the HUBER_C block for the verified
    equivalence), with the threshold that bites, the consistency factor beta(c), and the
    centre held at the median instead of drifting with the clip.

    STATUS, and the branch that no longer exists:
      'ok'          the fixed point was reached inside max_passes.
      'degenerate'  MAD == 0, i.e. at least half the observed values are IDENTICAL, so there
                    is no robust scale to estimate -- see the guard below.
      'maxiter'     the bound was exhausted.  sigma is the last iterate, which is a ROBUST
                    scale part-way to its fixed point; it is NOT the raw contaminated sd.
      'constant'    fewer than 2 observed values, or every observed value equal (caught by
                    the degeneracy guard returning sd = 0 / NaN).
    The pre-2026-08-03 winsorizer had a fourth outcome -- `converged=False` -> RETURN THE RAW
    VALUES -- which was the only path by which a fully contaminated sigma could reach the
    score.  IT WAS LATENT, NOT ACTIVE, and the distinction is worth keeping straight because
    the code around it implied otherwise: on the real panel all 18 weighted columns converged,
    nothing reverted, and the worst pass count was 40 (currentRatio) against a bound of 200 --
    the "57 / 50 / 64 passes" figures that sat here were from the earlier pre-correction panel.
    It DID fire on the three pathological shapes documented in that code.  The branch is
    deliberately gone rather than carried over: a partly-converged ROBUST scale is strictly
    better than a converged CONTAMINATED one, so there is nothing to fall back to, and
    'maxiter' is reported rather than silently absorbed.

    THE DEGENERACY GUARD (N4) is the MEASURED CONDITION that replaces the four-name
    `WINSOR_EXEMPT_BOUNDED` list.  MAD == 0 means >= half the column is one value, and there
    the robust scale implodes: MAD is exactly 0, so the fixed point is 0, an unguarded estimator
    returns sigma-hat ~ 1e-14 and every minority name ties at the bound (max|z| ~ 2e20).
    Falling back to the CLASSICAL mean/sd returns a sane ruler on exactly the three shapes that
    also defeated the winsorizer -- re-derived 2026-08-03, because normalisation-spec.md N4
    quotes the right sigmas with the WRONG max|z| for all three:
        99x0 + 1x1e6                sigma = 1.0e5    max|z| = 9.900  (N4 says 10.0)
        Piotroski 20x7 + 1x2        sigma = 1.09109  max|z| = 4.364  (N4 says 4.58; the
                                                     pre-E-1 code comment had 4.36 right)
        CycleHeat 25x(-3) + 2x(+1)  sigma = 1.06752  max|z| = 3.469  (N4 says 3.75)
    It fires on NO column of the real panel: `Piotroski`, `marketCapRevQuants` and `CycleHeat`
    all have MAD > 0 there and take a proper robust scale.

    NaN AND +-inf NEVER ENTER (N1's weight-0 case).  `normalizeAndDropNA` already maps inf to
    NaN before calling, but this function is public and is called directly by the offline
    tools, so it filters non-finite values itself rather than trusting its caller.
    """
    obs = pd.to_numeric(pd.Series(x), errors='coerce')
    obs = obs[np.isfinite(obs.to_numpy(dtype='float64'))]
    n = int(len(obs))
    if n == 0:
        return _RobustScale(np.nan, np.nan, 'constant', 0, np.nan, 0)
    vals = obs.to_numpy(dtype='float64')
    mu = float(np.median(vals))
    if n == 1:
        #  A single observation IS the median, and it has no spread.  sigma = NaN is the
        #  honest answer; the caller maps a non-finite sigma to 1.0, which puts the one name
        #  at z = 0 -- identical to what the classical path did with sd of one value.
        return _RobustScale(mu, np.nan, 'constant', 0, 1.0, 1)
    mad = float(np.median(np.abs(vals - mu))) * MAD_TO_SIGMA
    if not (mad > 0.0) or not math.isfinite(mad):
        # --- N4 DEGENERACY GUARD: >= half the values are identical --------------------
        cmu = float(np.mean(vals))
        csd = float(np.std(vals, ddof=1)) if n > 1 else float('nan')
        status = 'degenerate' if (csd > 0.0 and math.isfinite(csd)) else 'constant'
        return _RobustScale(cmu, csd, status, 0, 1.0, n)

    s, b, status, n_passes = mad, huber_beta(c), 'maxiter', 0
    for p in range(int(max_passes)):
        n_passes = p + 1
        psi = np.clip((vals - mu) / s, -c, c)
        s_new = s * math.sqrt(float(np.mean(psi * psi)) / b)
        if not (s_new > 0.0) or not math.isfinite(s_new):
            #  Unreachable given mad > 0 (mean(psi^2) == 0 requires every value == mu, which
            #  mad == 0 already caught, and psi^2 <= c^2 bounds the growth) -- kept because
            #  this value divides every name's score, so "unreachable" is not a licence to
            #  return a NaN or a zero scale silently.
            return _RobustScale(mu, mad, 'maxiter', n_passes, 1.0, n)
        if abs(s_new - s) <= rel_tol * max(1.0, abs(s)):
            s, status = s_new, 'ok'
            break
        s = s_new

    #  sum w / n: the fraction of estimation weight the column RETAINED.  This is the number
    #  that answers "how much of this column was downweighted" -- and the answer on the real
    #  panel is 87-97%, i.e. 11-24 names per column are PARTIALLY downweighted and NOBODY is
    #  excluded.  That is precisely why the estimator has no step (see the stability test).
    w = np.minimum(1.0, c / np.maximum(np.abs((vals - mu) / s), np.finfo('float64').tiny))
    return _RobustScale(mu, s, status, n_passes, float(np.mean(w)), n)


def squash(z, k=SQUASH_K):
    """CONCAVE BOUNDED SQUASH -- the algebraic sigmoid  zeta = k*z / sqrt(k^2 + z^2).

    Applied to EVERY value including the imputed fill, per SQUASH_K.  Properties, provable
    rather than measured: strictly increasing everywhere (so it collapses NO distinction --
    the whole point, against the 37 the +-3 clip collapsed); zeta(0) = 0 exactly, so a
    median-centred fill stays exactly neutral; zeta'(0) = 1, identity to first order; odd, so
    a cheap outlier and an expensive one are treated symmetrically; |zeta| < k; C-infinity, so
    there is no kink anywhere -- which is WHY it is applied to all values and not only past a
    boundary.

    Computed as k*z/hypot(k, z) rather than k*z/sqrt(k*k + z*z): `np.hypot` is the
    overflow-safe form, and while the real |z| here is O(100), a degenerate pool could in
    principle produce a large one and a silent inf is not an acceptable failure for something
    on the scoring path.
    """
    zz = pd.to_numeric(pd.Series(z), errors='coerce') if not isinstance(z, pd.Series) \
        else pd.to_numeric(z, errors='coerce')
    k = float(k)
    arr = zz.to_numpy(dtype='float64')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(arr == 0.0, 0.0, k * arr / np.hypot(k, arr))
    out = np.where(np.isnan(arr), np.nan, out)
    return pd.Series(out, index=zz.index, dtype='float64')


def normalizeAndDropNA(df, weight_series=None, huber_c=HUBER_C, squash_k=SQUASH_K,
                       method=None, rank_bounded=True, pool_label=None):
    """ROBUSTLY normalise every metric column of one pool; drop a row only when EVERY
    metric is NaN.

    THE Z-PATH IS SIX STEPS, per column, per pool (issue E-1, 2026-08-03 --
    projects/investment-filter/design/normalisation-spec.md N1-N7):

      1. observed = the column's finite values.  NaN and +-inf never enter the ruler.
      2. mu   = median(observed)                                   -- N2
      3. sigma = Huber Proposal-2 M-estimate about that median, c = huber_c, with the
                 MAD == 0 degeneracy guard                         -- N3, N4
      4. z    = (x - mu) / sigma for EVERY name, extremes included -- nothing is clipped
      5. zeta = squash(z, squash_k) = k*z/sqrt(k^2 + z^2)          -- N5, every value
      6. fillna(0), which under median-centring is the observed MEDIAN of the column and
         under the squash is exactly zeta(0) = 0                   -- N6

    WHAT CHANGED FROM THE PRE-2026-08-03 PATH, and it CHANGES SCORES BY DESIGN.  That path
    winsorized each weighted raw column at +-3 sigma and z-scored the clipped column.  Three
    defects, all measured on [pool = baseline_tools/resdic_2026-07-17_CORRECTED.pickle,
    general top-100]:

      (a) THE THRESHOLD DOWNWEIGHTED NOTHING.  The winsorizer IS a Huber P2 scale estimator
          (the equivalence is verified exactly -- see the HUBER_C block), parameterised at
          c = 3 where the weight min(1, c/|z|) essentially never bites.  It left 35% sigma
          inflation under 5% point contamination at n=100 and 119% on a bunched pair at
          cohort size n=25.  Consequence on the real panel: the middle-half dispersion ratio
          across the 16 continuous weighted columns was 2.55x, and 9 of 18 columns had ONE
          company owning more than half the column's raw variance -- i.e. discriminating
          power sat with the mechanical size/quality codes rather than the economic metrics.
      (b) THE HARD +-3 BOUND TIED DISTINCT COMPANIES.  55 cells clipped, all landing at
          exactly |z| = 3.000, which collapsed 37 DISTINCT raw inputs onto identical outputs
          (1,589 distinct inputs -> 1,552 distinct outputs).  A strictly monotone squash
          collapses none, and that is asserted per run below.
      (c) THE CENTRE WAS A CONTAMINATED MEAN, so `fillna(0)` -- how an unavailable metric is
          scored -- landed ABOVE the typical name on 14 of 18 columns, a +0.0739 AggScore
          advantage for full missingness (29% of the median-to-rank-20 distance).  Centring
          on the median takes that to -0.0000 and 1 of 18 (issue I-3, closed as a side
          effect of a decision made for a different reason).

    AND WHAT DID NOT CHANGE: no weight, and no name is ejected or dropped for being extreme.
    Outliers stay in the ranking and out of the ruler -- which is the standing premise, now
    implemented continuously (weight c/|z|) instead of by a discrete in/out cut.  A discrete
    cut was BUILT AND MEASURED AND REJECTED, on stability rather than taste: sweeping one
    name's raw value inward re-scored the OTHER 99 names by up to 3.69 z-units under a
    gap-detection cut, against 0.031 here, because a discrete inclusion decision puts a STEP
    in sigma-hat and sigma-hat divides everyone (spec section 4).

    BOTH OLD EXEMPTION LISTS ARE GONE, and their shared premise is why.  Zero-weight columns
    and bounded/discrete columns were exempt from WINSORIZATION because clipping destroys raw
    values (and, before 2026-07-19, could eject a name).  Nothing is clipped now, so there is
    nothing to be exempt from, and every metric column takes the same ruler:
      * BOUNDED/DISCRETE (`BOUNDED_DISCRETE_COLUMNS`) -- measured to take a robust scale
        cleanly, with sigma-hat RISING (Piotroski 1.5594 -> 1.6139, marketCapRevQuants
        0.3679 -> 0.4071, CycleHeat 1.0260 -> 0.9696), so their influence FALLS.  What
        protects the genuinely pathological shapes is the MAD == 0 condition, which is
        measurable and fires on all three of them, rather than a four-name list that
        silently mis-handles the next discrete metric anyone adds.
      * ZERO-WEIGHT (DcfToPrice / BoScore / priceGrowth) -- SCORE-NEUTRAL by construction,
        because AggScore is sum(z * w) and their w is 0.000, and `rankOfRanks_diag` ranks the
        already-weighted (all-zero) columns.  Normalising them the same way is what keeps
        `resdic['postScoreMetric']` a SINGLE-BASIS frame: before this change three of its 21
        columns were on a different ruler from the other 18, inside a frame every consumer
        reads as normalised.  This is the one part of the change that is a judgement rather
        than a spec requirement; it is recorded here because it moves those three DISPLAY
        columns' values.

    method : NORM_ZSCORE (default, the production path -- everything above applies) or
             NORM_RANK (rank -> inverse-normal; see NORM_RANK's notes and _rank_normal).
             THE RANK PATH IS NOT A CANDIDATE and must not be offered as one: Stage-1 ranks
             and Stage-2 scores MAGNITUDES, so rank-normalising Stage-2 would collapse it
             into a second Stage-1 (CEO, 2026-08-03).  It is retained only as a measurement
             instrument -- the residual comparability gap it would close (~2.6x at the
             rank-20 boundary, against 1.40x on the middle half) is an ACCEPTED price of
             magnitude-based scoring, not an open question.
    rank_bounded : NORM_RANK only.  True (default) rank-maps `BOUNDED_DISCRETE_COLUMNS` along
             with the rest; False leaves them un-mapped, to measure that alternative.  Under
             NORM_RANK the robust ruler is not run at all: the map depends only on the ORDER
             within a column.
    pool_label : label used in the diagnostic banner only ('general' / a cohort name).

    DIAGNOSTICS (N7, score-neutral).  Every call appends one row per column to
    NORM_DIAGNOSTICS and prints the table: mu, sigma, iterations, status, sum w / n, max|z|
    pre-squash, max|zeta|, the realised single-column reach |w| * max|zeta|, and the p50->p80
    span -- that last one being the per-column discriminating power this whole issue is
    about, which nothing printed before.  `single_column_reach_check` then tests the reach
    against the pool's OWN median-to-rank-20 AggScore distance after the score exists,
    because that distance is panel-dependent and the k = 3 property is only meaningful
    against the panel it is measured on.

    Returns (frame, outlierlist).  outlierlist holds ONLY the all-NaN rows that were
    genuinely dropped -- an extreme name is in the ranking, not an outlier.
    """
    df.reset_index(inplace=True, drop=True)

    # Check if dataframe is empty or has no metric columns
    if df.empty:
        print("Warning: Input dataframe is empty.")
        return df, []

    # Replace inf values with nan (modern approach without inplace)
    metric_cols = [col for col in df.columns if col != 'source']

    if len(metric_cols) == 0:
        print("Warning: No metric columns found (only 'source' column present).")
        return df, []

    df_clean = df.copy()
    # Suppress the FutureWarning about downcasting in replace()
    with pd.option_context('future.no_silent_downcasting', True):
        for col in metric_cols:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

    # Drop rows only if ALL metric columns are NaN (completely invalid rows)
    # This is less aggressive than dropping rows with ANY NaN, preserving more data
    nan_counts = df_clean[metric_cols].isna().sum(axis=1)
    dropmask = nan_counts < len(metric_cols)  # Keep rows with at least one valid metric
    outlierlist = list(df_clean['source'][~dropmask].copy())
    dfnona = df_clean[dropmask].copy()

    # Guard: if all rows have NaN, return empty df with warning
    if dfnona.empty:
        print(f"Warning: All {len(df)} rows dropped due to NaN values (all metric columns were NaN). Returning empty dataframe.")
        return dfnona, list(df['source'])

    tempnum = dfnona.drop('source',axis=1).apply(pd.to_numeric, errors='coerce')

    if method is None:
        method = NORM_METHOD_DEFAULT
    if method not in (NORM_ZSCORE, NORM_RANK):
        raise ValueError("normalizeAndDropNA: method must be %r or %r, got %r"
                         % (NORM_ZSCORE, NORM_RANK, method))

    if method == NORM_RANK:
        # --- RANK -> INVERSE-NORMAL ---------------------------------------------------
        # NOT A PRODUCTION CANDIDATE (CEO, 2026-08-03): Stage-1 ranks and Stage-2 scores
        # MAGNITUDES, so rank-normalising Stage-2 collapses it into a second Stage-1.  This
        # branch exists to MEASURE that alternative, nothing else.
        # The robust ruler is not run here, and it is inapplicable rather than skipped: a
        # rank map depends only on the ORDER within a column, and every step of the z-path
        # (median shift, positive scale divide, strictly increasing squash) is
        # order-preserving, so it could not move a single rank.  Verified in
        # test_rank_normalization.test_rank_normal_is_invariant_to_any_strictly_monotone_
        # transform.
        if rank_bounded:
            to_map = list(tempnum.columns)
        else:
            to_map = [c for c in tempnum.columns if c not in BOUNDED_DISCRETE_COLUMNS]
        for col in to_map:
            tempnum[col] = _rank_normal(tempnum[col])
        # fillna(0) is the SAME line the z-path uses, and here 0 is the observed MEDIAN of
        # the column, so an unavailable metric is exactly neutral (finding N1).
        temp_normpsmdf = tempnum.fillna(0)
        skipped = [c for c in tempnum.columns if c not in to_map]
        print("normalizeAndDropNA[rank]: inverse-normal mapped %d column(s) over %d row(s)"
              "%s" % (len(to_map), len(tempnum),
                      ("; LEFT RAW (rank_bounded=False): %s" % skipped) if skipped else ""),
              flush=True)
        dfnona[temp_normpsmdf.columns] = temp_normpsmdf
        return dfnona.copy(), outlierlist

    # --- THE ROBUST RULER: median centre + Huber scale on the SAME subset, then squash ---
    # ONE loop over EVERY metric column -- no exemption list, no per-column threshold, no
    # data-dependent exclusion set.  See the docstring for why both old exemptions died.
    raw_num = tempnum.copy()                     # kept for the ties audit below, read-only
    diag = []
    for col in tempnum.columns:
        raw_col = tempnum[col]
        est = robust_location_scale(raw_col, c=huber_c)
        sigma = est.sigma
        if not np.isfinite(sigma) or sigma == 0.0:
            #  Constant or single-observation column.  sigma = 1 puts every name at z = 0,
            #  which is EXACTLY what the classical path did via
            #  `colstds.replace(0, np.nan).fillna(1)` -- a column with no spread carries no
            #  information and must not be allowed to divide by zero into +-inf.
            sigma = 1.0
        z = (raw_col - est.mu) / sigma
        zeta = squash(z, squash_k)
        observed = raw_col.notna()
        w = float(weight_series.get(col, 1.0) or 0.0) if weight_series is not None else 1.0
        zo, so = zeta[observed].dropna(), z[observed].dropna()
        diag.append({
            'pool': pool_label or 'general', 'column': col, 'weight': w,
            'n_obs': est.n_obs, 'mu': est.mu, 'sigma': est.sigma, 'sigma_used': sigma,
            'status': est.status, 'n_passes': est.n_passes,
            'weight_retained': est.weight_retained,
            'max_abs_z_presquash': float(so.abs().max()) if len(so) else np.nan,
            'max_abs_zeta': float(zo.abs().max()) if len(zo) else np.nan,
            'reach_w_x_max_zeta': (abs(w) * float(zo.abs().max())) if len(zo) else np.nan,
            #  p50 -> p80 IS THE DECISION BAND: the rank-20 boundary of a 100-name pool sits
            #  in it, so this is the per-column discriminating power where it actually
            #  matters.  Measured on OBSERVED cells only -- including the fills would let a
            #  column's missingness rate masquerade as spread.
            'span_p50_p80': (float(zo.quantile(0.80) - zo.quantile(0.50))
                             if len(zo) > 1 else np.nan),
            'span_p25_p75': (float(zo.quantile(0.75) - zo.quantile(0.25))
                             if len(zo) > 1 else np.nan),
        })
        tempnum[col] = zeta

    #  N6: the imputation, AFTER mu-hat and sigma-hat -- an imputation must never set the
    #  scale it is then measured against.  Under median-centring 0 is the column's observed
    #  median and zeta(0) = 0 exactly, so this same one-line fill is now (near-)neutral
    #  instead of the +0.0739 reward it was.  The Type-aware fill percentile of
    #  missing-data-regime.md A6 is issue I-3's remaining half and is NOT implemented here.
    temp_normpsmdf = tempnum.fillna(0.0)

    _emit_norm_diagnostics(diag, huber_c, squash_k)
    NORM_DIAGNOSTICS.extend(diag)

    #  TIES AUDIT, per run and per pool.  Guarded: a defect in an audit must not cost a
    #  12-hour run, but the assertion inside it is a real assertion -- see the function.
    _safe_diagnose(assert_no_collapsed_distinctions, raw_num, tempnum, pool_label, squash_k)

    dfnona[temp_normpsmdf.columns] = temp_normpsmdf
    dfnonanorm = dfnona.copy()

    return dfnonanorm, outlierlist


#  Accumulates one row per column per pool across the whole run (6 pools x ~21 columns = ~126
#  rows), so the run's own normalisation record is inspectable after the fact instead of only
#  scrolling past on stdout.  A LIST, appended to, deliberately: postBoScoreRanking runs once
#  per pool and a per-call dict would leave only the last cohort -- the same single-writer
#  clobber the frequency-conflict CSV hit.  Tests reset it; nothing reads it back into a score.
NORM_DIAGNOSTICS = []


def _emit_norm_diagnostics(diag, huber_c, squash_k):
    """Print the N7 per-column normalisation record.  EMITS ONLY."""
    try:
        if not diag:
            return
        pool = diag[0]['pool']
        #  An ALL-NaN column is not a normalisation event -- it is `DcfToPrice` on every run,
        #  because the per-ticker DCF fetch is skipped by default (OFFLINE_NO_DCF) and the
        #  metric carries w = 0.000.  Bannering an expected phenomenon on every pool of every
        #  run is how a reader learns to skip the banner, so it gets one quiet line.
        empty = [d for d in diag if d['n_obs'] == 0]
        bad = [d for d in diag if d['status'] != 'ok' and d['n_obs'] > 0]
        print("normalizeAndDropNA[robust z]: pool=%s  c=%.3f (beta=%.6f)  k=%.3f  "
              "%d column(s)" % (pool, huber_c, huber_beta(huber_c), squash_k, len(diag)),
              flush=True)
        print("    %-28s %6s %12s %12s %5s %11s %7s %7s %7s %8s %8s"
              % ('column', 'w', 'mu', 'sigma', 'it', 'status', 'sumw/n',
                 'max|z|', 'max|Z|', 'reach', 'p50-p80'), flush=True)
        for d in sorted(diag, key=lambda r: -abs(r['weight'])):
            print("    %-28s %6.3f %12.6g %12.6g %5d %11s %7.3f %7.2f %7.3f %8.4f %8.4f"
                  % (d['column'], d['weight'], d['mu'], d['sigma'], d['n_passes'],
                     d['status'], d['weight_retained'], d['max_abs_z_presquash'],
                     d['max_abs_zeta'], d['reach_w_x_max_zeta'], d['span_p50_p80']),
                  flush=True)
        if empty:
            print("    (no observations, every name imputed at z = 0: %s)"
                  % ', '.join('%s w=%.3f' % (d['column'], d['weight']) for d in empty),
                  flush=True)
        if bad:
            #  LOUD, per column.  'degenerate' means the MAD == 0 guard fired -- at least half
            #  that column is one value and it is on a CLASSICAL mean/sd ruler, which the
            #  reader needs to know.  'maxiter' means the fixed point was not reached inside
            #  the bound (the scale is still robust, just not settled).  Neither fires on the
            #  real panel; if one starts to, it must not be a silent internal outcome.
            print("!" * 78, flush=True)
            for d in bad:
                print("normalizeAndDropNA[robust z]: %-28s status=%-11s n_obs=%-4d "
                      "passes=%-4d  sigma=%.6g" % (d['column'], d['status'], d['n_obs'],
                                                   d['n_passes'], d['sigma']), flush=True)
            print("!" * 78, flush=True)
    except Exception as _e:                              # a print must never cost a run
        print("normalizeAndDropNA: diagnostic emit failed (%s: %s) -- scores UNAFFECTED"
              % (type(_e).__name__, _e), flush=True)


def assert_no_collapsed_distinctions(raw_num, out_num, pool_label=None, k=SQUASH_K):
    """ZERO TIES: no two DISTINCT observed inputs may map to a single output.

    Counted the way the defect was counted -- distinct raw inputs vs distinct outputs, per
    column -- and NOT by appeal to monotonicity, because the failure mode is arithmetic.  Two
    concrete precedents:
      * the +-3 winsorizer this replaced mapped 55 cells onto exactly |z| = 3.000 and
        collapsed 37 distinct values across 18 columns (1,589 -> 1,552);
      * `k*tanh(z/k)` equals k EXACTLY in float64 for |z| >= 18.99*k, so it would re-create
        the same defect further out, and it already saturated one cell on the real panel.
    The algebraic sigmoid saturates only at |z| ~ 1.8e8, so a violation here means either that
    a squash form was swapped in or that sigma-hat has collapsed (check the MAD == 0 guard).

    Returns {column: n_collapsed} (all zeros on a healthy pool).  RAISES on a violation:
    silently shipping a ruler that ties distinct companies is the defect E-1 exists to fix, so
    it is not a warning.  The CALL SITE is wrapped in _safe_diagnose, so an unexpected failure
    of the audit itself degrades to a loud stdout traceback rather than killing the run.
    """
    collapsed = {}
    for col in out_num.columns:
        a = pd.to_numeric(raw_num[col], errors='coerce')
        b = pd.to_numeric(out_num[col], errors='coerce')
        both = a.notna() & b.notna()
        n_in, n_out = int(a[both].nunique()), int(b[both].nunique())
        collapsed[col] = n_in - n_out
    offenders = {c: n for c, n in collapsed.items() if n != 0}
    assert not offenders, (
        "NORMALISATION COLLAPSED DISTINCTIONS (pool=%s, k=%s): %r.  Distinct raw inputs "
        "mapped to a single output -- two different companies now score identically on a "
        "metric where they differ.  A strictly monotone squash cannot do this; the causes are "
        "float64 saturation of the squash or a collapsed sigma-hat."
        % (pool_label or 'general', k, offenders))
    return collapsed


#  The SHORTLIST DEPTH each pool's reach is judged against.  It is NOT one number, and the
#  cohort case is the reason (measured 2026-08-03):
#    * GENERAL pool, n=100 -- the deliverable IS the top-20 (`ntopxlsx`), so rank 20 is the
#      real boundary and it sits comfortably above the pool median.
#    * A CARVE-OUT COHORT, n<=25 -- its side-list is `postRank.head(ntopagg)` = head(100), so
#      the WHOLE cohort is published and the cohort has no shortlist boundary of its own.
#      Rank 20 of 25 is BELOW that pool's median, which makes the inequality vacuous (a
#      negative "distance"): the first version of this check printed VIOLATES on all five
#      cohorts for exactly that reason.  5 is used instead -- the smallest published shortlist
#      depth anywhere in the house (the market-cap bands' top-5) -- and it is a JUDGEMENT, not
#      a spec figure.  It is stated here rather than buried so it can be argued with.
REACH_TOP_N_GENERAL = 20
REACH_TOP_N_COHORT = 5


def single_column_reach_check(postRank, weight_series=None, pool_label=None, top_n=None,
                              diag=None):
    """THE k PROPERTY, tested against THIS run's OWN panel: no single column may, on its own,
    carry a MEDIAN-scoring name into the top `top_n`.  EMITS ONLY -- changes no score.

        max_c |w_c| * max_i |zeta_ic|   <   AggScore(rank top_n) - AggScore(median)

    WHY IT IS CHECKED IN THE RUN RATHER THAN ASSERTED ONCE IN A COMMENT.  The right-hand side
    is PANEL-DEPENDENT -- 0.2560 on the 2026-07-17 CORRECTED panel through the pre-E-1 ruler,
    0.2396 on the same inputs through this one -- and k = 3 was chosen against a figure of
    that size.  A number with no panel attached cannot be checked, and that is exactly how the
    stale "0.134 median-to-top-20 distance" survived for weeks.  So the panel checks itself.
    Measured on that panel: this ruler gives reach 0.2160 (incomeQuality) against distance
    0.2396 = 0.90x, OK with 10% margin; the pre-E-1 hard |z| <= 3 bound gave 0.2666 against
    0.2560 = 1.04x, i.e. the shipped design ALREADY violated it; k = 4 would give 1.12x.

    IT IS A PROPERTY OF THE WEIGHT VECTOR AS MUCH AS OF k, AND THE COHORTS SHOW IT.  On the
    five carve-out cohorts, judged against rank 5 (see REACH_TOP_N_COHORT), two exceed:
    InvestmentVehicle 1.40x and REIT 1.21x, with BalanceSheetFin at exactly 1.00x.  The
    driver is weight CONCENTRATION, not the squash -- InvestmentVehicle's cohort vector puts
    w = 0.597 on `bVpRatio` alone, and a column carrying 60% of the weight is SUPPOSED to be
    able to dominate; that is what the weight says.  No value of k fixes it (rank-5 compliance
    there would need k < 2.15, rank-10 compliance k < 0.5), so it is a weighting finding for
    E-2 and NOT a reason to move k.  Hence the emission below: a BANNER only for the pool whose
    shortlist actually ships, a plain labelled line for a cohort, with |w| printed so the
    attribution is visible rather than inferred.
    """
    try:
        if postRank is None or 'AggScore' not in getattr(postRank, 'columns', []):
            return None
        pool = pool_label or 'general'
        is_general = (pool == 'general')
        if top_n is None:
            top_n = REACH_TOP_N_GENERAL if is_general else REACH_TOP_N_COHORT
        agg = pd.to_numeric(postRank['AggScore'], errors='coerce').dropna()
        n = int(len(agg))
        if n < 3:
            return None
        n_used = min(int(top_n), n)
        ordered = agg.sort_values(ascending=False).to_numpy(dtype='float64')
        boundary = float(ordered[n_used - 1])
        median = float(np.median(ordered))
        distance = boundary - median
        rows = [d for d in (NORM_DIAGNOSTICS if diag is None else diag)
                if d.get('pool') == pool]
        rows = [r for r in rows if np.isfinite(r.get('reach_w_x_max_zeta', np.nan))]
        if not rows:
            return None
        worst = max(rows, key=lambda r: r['reach_w_x_max_zeta'])
        reach = float(worst['reach_w_x_max_zeta'])
        head = ("normalizeAndDropNA[k-property] pool=%-18s worst single-column reach %.4f "
                "(%s, |w|=%.3f x max|zeta|=%.3f)"
                % (pool, reach, worst['column'], abs(worst['weight']),
                   worst['max_abs_zeta']))
        if not (distance > 0):
            #  NOT a violation -- an inapplicable test.  Rank n_used sits at or below this
            #  pool's median, so "carry a median name into the top n_used" is not a statement
            #  about anything.  Say so instead of reporting a negative distance as a failure.
            print(head + "  vs median->rank-%d distance %+.4f  -- NOT APPLICABLE: rank %d is "
                         "at or below the median of a %d-name pool, so there is no boundary "
                         "to reach." % (n_used, distance, n_used, n), flush=True)
            return {'pool': pool, 'reach': reach, 'column': worst['column'],
                    'distance': distance, 'ratio': float('nan'), 'n_used': n_used,
                    'median': median, 'boundary': boundary, 'applicable': False}
        ratio = reach / distance
        line = (head + "  vs median->rank-%d distance %.4f (median %.4f, rank-%d %.4f)  =  "
                       "%.2fx" % (n_used, distance, median, n_used, boundary, ratio))
        if ratio < 1.0:
            print(line + "  OK (margin %.0f%%)" % (100.0 * (1.0 - ratio)), flush=True)
        elif is_general:
            #  The general pool's top-20 IS the deliverable, so here it is a banner.
            print("!" * 78, flush=True)
            print(line + "  <<< VIOLATES on the SHIPPED shortlist: one column can reach the "
                         "top-%d unaided.  k must be RE-DERIVED against this panel, not "
                         "inherited." % n_used, flush=True)
            print("!" * 78, flush=True)
        else:
            #  A cohort.  Real, expected on the concentrated cohort vectors, and a WEIGHTING
            #  fact -- so it is reported plainly and attributed, not bannered.  Bannering a
            #  known consequence of a 0.597 weight every run would train the reader to skip
            #  the line that matters.
            print(line + "  EXCEEDS -- expected where a cohort vector concentrates weight "
                         "(|w|=%.3f on one column); a WEIGHTING finding for E-2, not a k "
                         "defect.  No k satisfies this while one column carries that weight."
                  % abs(worst['weight']), flush=True)
        return {'pool': pool, 'reach': reach, 'column': worst['column'],
                'distance': distance, 'ratio': ratio, 'n_used': n_used,
                'median': median, 'boundary': boundary, 'applicable': True}
    except Exception as _e:
        print("single_column_reach_check failed (%s: %s) -- scores UNAFFECTED"
              % (type(_e).__name__, _e), flush=True)
        return None

MISSING_REPORT_CSV = 'MissingDataFillReport_%s.csv'
_MISSING_CSV_STARTED = set()


def missing_data_fill_report(raw_df, norm_df, weight_series, pool='general',
                             csv=True, verbose=True):
    """WHERE THE fillna(0) IMPUTATION LANDS, per weighted column and per name.  EMITS ONLY.

    WHY IT HAS TO BE PRODUCED BY THE RUN (2026-08-01).  A missing metric is imputed by
    `normalizeAndDropNA`'s post-normalisation `fillna(0)`, i.e. it is scored AT THE POOL
    **MEDIAN** of that column, not as "missing".  (This line said "POOL MEAN" until 2026-08-05
    and was stale from E-1: the ruler has centred each column on its observed MEDIAN since
    2026-08-03, which is precisely why the fill is now (near-)neutral instead of the +0.0739
    advantage it used to be -- so the wrong word here described the defect that had been fixed.)
    Whether the fill is generous or punitive depends on where 0 sits in the column's OBSERVED z
    distribution -- which is a property of THIS pool on THIS panel.  Every fill percentile we hold was measured on a pre-change panel, and the 2026-08-01
    scoring changes (the EPStoEPSmean window cap and the incomeQuality basis) move the very
    distributions those percentiles were measured against.  Re-deriving them later from a stale
    panel would calibrate the new design against the old data, so tonight's run PRODUCES the
    calibration instead.

    THE SIGN OF THE WEIGHT IS PART OF THE READING, AND THE NAIVE PERCENTILE GETS IT BACKWARDS.
    A fill sitting ABOVE a column's median is an ADVANTAGE only if that column's weight is
    POSITIVE.  `CycleHeat` carries w = -0.080, so a fill above its median is a PENALTY -- the
    imputed name is scored as if it were hot.  `fill_effect` below therefore reports
    (percentile - 0.5) * sign(w), not the raw percentile.

    SCORE-NEUTRAL BY CONSTRUCTION: both frames arrive as inputs and are only read; nothing is
    assigned back, and the function returns its own frames.  Fully guarded -- a diagnostic must
    never be able to cost a 12-hour run.
    """
    try:
        wser = dict(weight_series) if weight_series is not None else {}
        wcols = [c for c in raw_df.columns
                 if c != 'source' and float(wser.get(c, 0) or 0) != 0.0]
        if not wcols:
            return None, None
        rows = []
        for c in wcols:
            w = float(wser[c])
            rawc = pd.to_numeric(raw_df[c], errors='coerce')
            zc = pd.to_numeric(norm_df[c], errors='coerce')
            imputed = rawc.isna()
            obs = zc[~imputed].dropna()           # the OBSERVED (non-imputed) z distribution
            if len(obs) == 0:
                pct = np.nan
            else:
                # percentile of the fill value (z = 0) among observed z, 'mean' convention
                pct = 0.5 * ((obs < 0.0).sum() + (obs <= 0.0).sum()) / len(obs)
            eff = np.nan if not np.isfinite(pct) else (pct - 0.5) * (1.0 if w > 0 else -1.0)
            rows.append(dict(
                pool=pool, column=c, weight=round(w, 6),
                n_names=int(len(rawc)), n_imputed=int(imputed.sum()),
                pct_imputed=round(100.0 * imputed.mean(), 2),
                fill_percentile_in_observed_z=(None if not np.isfinite(pct) else round(pct, 4)),
                observed_z_median=(None if len(obs) == 0 else round(float(obs.median()), 4)),
                fill_effect=(None if not np.isfinite(eff) else round(eff, 4)),
                fill_reading=('n/a' if not np.isfinite(eff) else
                              ('ADVANTAGE (fill scores better than the median name)' if eff > 0
                               else 'PENALTY (fill scores worse than the median name)'
                               if eff < 0 else 'neutral')),
                weight_share_of_pool=round(abs(w) / sum(abs(float(wser[k])) for k in wcols), 4),
            ))
        col_df = pd.DataFrame(rows)

        imp = raw_df[wcols].apply(lambda s: pd.to_numeric(s, errors='coerce')).isna()
        name_df = pd.DataFrame({
            'pool': pool,
            'source': raw_df['source'].values,
            'n_imputed_cols': imp.sum(axis=1).values,
            'n_weighted_cols': len(wcols),
            'imputed_weight_share': [
                round(sum(abs(float(wser[c])) for c in wcols if imp.iloc[i][c])
                      / sum(abs(float(wser[k])) for k in wcols), 4)
                for i in range(len(raw_df))],
            'imputed_cols': [', '.join(c for c in wcols if imp.iloc[i][c])
                             for i in range(len(raw_df))],
        })

        if verbose:
            worst = col_df.dropna(subset=['fill_effect']).reindex(
                col_df['fill_effect'].abs().sort_values(ascending=False).index).head(4)
            n_any = int((name_df['n_imputed_cols'] > 0).sum())
            print("MISSING-DATA FILL REPORT [pool=%s]: %d weighted column(s); %d of %d name(s) "
                  "(%.1f%%) carry >=1 imputed column; %d imputed cell(s) total."
                  % (pool, len(wcols), n_any, len(name_df),
                     100.0 * n_any / max(1, len(name_df)), int(col_df['n_imputed'].sum())),
                  flush=True)
            for _, r in worst.iterrows():
                print("    %-28s w=%+.4f  imputed=%3d (%.1f%%)  fill at pct %.3f of observed z"
                      "  -> %s" % (r['column'], r['weight'], r['n_imputed'], r['pct_imputed'],
                                   r['fill_percentile_in_observed_z'] or float('nan'),
                                   r['fill_reading']), flush=True)
            heavy = name_df[name_df['imputed_weight_share'] >= 0.20]
            if len(heavy):
                print("    NAMES SCORED >=20%% ON FILLS: %s"
                      % ', '.join('%s (%.0f%%)' % (r['source'], 100 * r['imputed_weight_share'])
                                  for _, r in heavy.head(10).iterrows()), flush=True)
        if csv:
            _write_missing_csv(col_df, name_df)
        return col_df, name_df
    except Exception as _e:
        print('WARNING: missing-data fill report skipped for pool=%s (%s: %s)'
              % (pool, type(_e).__name__, _e), flush=True)
        return None, None


def _write_missing_csv(col_df, name_df):
    """Append both tables to one dated CSV (a `section` column separates them).

    APPEND, because postBoScoreRanking runs ONCE PER POOL -- general plus five carve-out
    cohorts -- and a per-call overwrite would leave only the last cohort, which is the same
    single-writer clobber the frequency-conflict CSV hit.  Header written once per process.
    """
    try:
        fn = MISSING_REPORT_CSV % pd.Timestamp.today().strftime('%Y-%m-%d')
        a = col_df.assign(section='per_column')
        b = name_df.assign(section='per_name')
        out = pd.concat([a, b], ignore_index=True, sort=False)
        first = fn not in _MISSING_CSV_STARTED
        out.to_csv(fn, index=False, mode='w' if first else 'a', header=first)
        _MISSING_CSV_STARTED.add(fn)
    except Exception as _e:
        print('WARNING: could not write the missing-data fill CSV (%s)' % _e, flush=True)


# =========================================================================== #
#  METRIC BASIS: ONE WAY TO OBTAIN A METRIC VALUE, AND IT NAMES THE BASIS       #
# =========================================================================== #
# THE DEFECT CLASS THIS CLOSES.  Stage-2 emits metric-named columns on THREE bases:
#
#   BASIS_RAW      postScoreMetric_raw   the metric in its own units (pre-normalisation)
#   BASIS_Z        postScoreMetric       winsorized cross-sectional z
#   BASIS_Z_TIMES_W  psmdf_normalized / postRank   z x weight = the AggScore contribution
#
# Nothing on the frames said which, so a reader had to know it from the resdic key. Three
# measured consequences:
#   * `AggScoreTop100-*.csv` published `CycleHeat` straight off postRank under a comment
#     asserting it was the metric.  CycleHeat is winsor-EXEMPT, so its z is an exact affine
#     function of the raw value and w = -0.080 inverted it EXACTLY: corr(published, true) =
#     -1.0000 on the 07-17 panel, i.e. the column's MINIMUM was the pool's HOTTEST name.
#   * the OLS path fitted coefficients in one function and applied them in another; when
#     only the FIT side was un-weighted the two bases diverged for negative weights and the
#     re-ranker inverted -- and BEFORE that, both used `z x w` and the double negation made
#     the result ACCIDENTALLY CORRECT.  A one-sided fix was therefore WORSE than no fix.
#   * `resdic['psmdf_normalized']` IS LITERALLY THE SAME OBJECT as `resdic['postRank']`,
#     because getAggScore mutates its argument in place and returns it -- so a consumer
#     reading a metric column off the innocuously-named "normalized" frame is reading
#     `z x w`.  Two analysis modules needed guard-test exemptions for exactly this.
#
# THE REMEDY IS TWO-PART, and neither part alone is enough:
#   (1) every frame this file emits DECLARES its basis on itself (`stamp_metric_basis`,
#       carried in df.attrs, which survives copy/slice/pickle);
#   (2) `metric_frame(df, basis)` is the ONE accessor, and the basis is a REQUIRED argument
#       -- so the request is explicit AND checkable against the frame's own declaration.
#       Ask for BASIS_Z off a `z x w` frame and you get the un-weighting; ask for something
#       unrecoverable (raw <-> z needs the pool's mu/sigma, which the frame does not carry)
#       and it REFUSES instead of handing back the wrong quantity.
#
# LIMIT, STATED EXACTLY RATHER THAN PAPERED OVER -- measured on pandas 2.3.1, the version in
# requirements, because "attrs propagate" is vaguely true and precisely false:
#     PRESERVED  copy, pickle, head, boolean mask, reset_index, set_index, assign, drop,
#                apply, and concat WHEN EVERY INPUT CARRIES IDENTICAL attrs
#     DROPPED    pd.merge, and pd.concat when the inputs' attrs DIFFER or any is empty
# The consequence that matters: `Sbocker` merges moatScore onto resdic['postRank'], so the
# frame the DELIVERABLE stages receive has LOST its stamp.  An unstamped frame therefore means
# "basis not declared", never "basis is X" -- metric_frame cannot VERIFY such a caller, only
# take it at its word (still strictly better than a silent guess, because the word has to be
# said).  So the stamp hardens the frames INSIDE Stage-2 and the ones that reach a consumer
# unmerged (postScoreMetric_raw, postScoreMetric, psmdf_normalized); the frozen inventory in
# baseline_tools/test_published_columns.py remains the backstop for everything past the merge.
# DO NOT "fix" this by stamping after the merge in Sbocker without also deciding what a stamp
# means on a frame that has since been re-columned -- a stamp that is sometimes stale is worse
# than one that is sometimes absent.
METRIC_BASIS_ATTR = 'va_metric_basis'
BASIS_RAW = 'raw'
BASIS_Z = 'z'
BASIS_Z_TIMES_W = 'z*w'
_METRIC_BASES = (BASIS_RAW, BASIS_Z, BASIS_Z_TIMES_W)


def stamp_metric_basis(df, basis):
    """Declare which basis `df`'s metric columns are on.  Returns the same frame."""
    if basis not in _METRIC_BASES:
        raise ValueError('stamp_metric_basis: unknown basis %r (expected one of %r)'
                         % (basis, _METRIC_BASES))
    try:
        df.attrs[METRIC_BASIS_ATTR] = basis
    except Exception:               # a stamp must never be able to break the scorer
        pass
    return df


def metric_basis_of(df):
    """The declared basis, or None when the frame does not declare one (see the LIMIT note).

    None means UNDECLARED. It does NOT mean raw, and a consumer must not read a default
    into it -- that assumption is the original defect.
    """
    try:
        v = df.attrs.get(METRIC_BASIS_ATTR)
    except Exception:
        return None
    return v if v in _METRIC_BASES else None


def metric_frame(df, basis, cols=None, verbose=False, label=''):
    """THE accessor: metric columns off a Stage-2 frame, on the basis you ASK for.

    `basis` is required and is one of BASIS_RAW / BASIS_Z / BASIS_Z_TIMES_W.  Returns
    (frame, kept_cols, dropped_zero_weight_cols) -- the same shape as
    unweight_postrank_metrics, which is now this function's BASIS_Z implementation.

    Behaviour by (declared basis -> requested basis):
      same                      -> returned unchanged (no arithmetic, so no rounding)
      z*w -> z                  -> divide by each metric's weight; w = 0 columns DROPPED
      z   -> z*w                -> multiply by each metric's weight
      anything involving raw    -> REFUSED (ValueError).  raw <-> z needs the POOL's
                                   mu/sigma, which no single frame carries; guessing here
                                   is how a wrong quantity gets published under a right name.
      undeclared                -> taken at the caller's word, unchanged (cannot verify)
    """
    if basis not in _METRIC_BASES:
        raise ValueError('metric_frame: basis is required and must be one of %r, got %r'
                         % (_METRIC_BASES, basis))
    actual = metric_basis_of(df)
    if actual is None or actual == basis:
        # NOT stamped when the frame arrived UNDECLARED: the caller's basis is a claim this
        # function could not verify, and recording an unverified claim as a declaration would
        # make the stamp sometimes-wrong instead of sometimes-absent.  Sometimes-wrong is the
        # worse failure -- a later reader would trust it.
        out = df.copy()
        if actual is not None:
            stamp_metric_basis(out, actual)
        return out, _weighted_metric_cols(out, cols), []
    if BASIS_RAW in (actual, basis):
        raise ValueError(
            'metric_frame: cannot convert metric basis %r -> %r. The raw <-> z step needs '
            "the POOL's winsorized mu/sigma, which this frame does not carry. Read the raw "
            "metric from resdic['postScoreMetric_raw'] (basis %r) instead of deriving it."
            % (actual, basis, BASIS_RAW))
    if actual == BASIS_Z_TIMES_W and basis == BASIS_Z:
        out, kept, dropped = unweight_postrank_metrics(df, cols=cols, verbose=verbose,
                                                       label=label)
        return stamp_metric_basis(out, BASIS_Z), kept, dropped
    # BASIS_Z -> BASIS_Z_TIMES_W: the production weighting step, exposed so a consumer that
    # needs contributions never re-derives it (and never re-derives it DIFFERENTLY).
    W = _weight_vector()
    out = df.copy()
    kept, dropped = [], []
    for c in _weighted_metric_cols(out, cols):
        w = W.get(c)
        if w is None:
            kept.append(c)
            continue
        if w == 0:
            dropped.append(c)
            continue
        out[c] = pd.to_numeric(out[c], errors='coerce') * w
        kept.append(c)
    return stamp_metric_basis(out, BASIS_Z_TIMES_W), kept, dropped


def assert_metric_basis(df, expected, label=''):
    """Refuse a frame whose DECLARED basis is not `expected`.  Undeclared passes (see LIMIT).

    For a consumer that wants to read the columns directly rather than transform them: it
    turns "I believe this frame holds z" into an assertion the frame itself can contradict.
    """
    actual = metric_basis_of(df)
    if actual is not None and actual != expected:
        raise ValueError(
            'metric basis mismatch%s: this frame declares %r, the caller expects %r. '
            'Use postBoRank.metric_frame(df, %r) to convert, or read the frame that carries '
            'the basis you want.'
            % (' [%s]' % label if label else '', actual, expected, expected))
    return True


def _weight_vector():
    """{metric: weight} from the production dictionaries -- the one weight lookup."""
    postBm, postNew = cdic.getPostDict()
    return {**{k: float(postBm[k]['w']) for k in postBm},
            **{k: float(postNew[k]['w']) for k in postNew}}


def _weighted_metric_cols(df, cols=None):
    """Which columns of `df` are weighted metric columns (the ones a basis applies to)."""
    if cols is not None:
        return [c for c in cols if c in df.columns]
    W = _weight_vector()
    return [c for c in df.columns if c in W]


def unweight_postrank_metrics(df, cols=None, verbose=False, label=''):
    """Recover the METRIC z-scale from a postRank-style frame: divide each metric column by
    its production weight.  Returns (new_df, kept_cols, dropped_zero_cols).

    WHY THIS IS A SHARED HELPER AND NOT INLINE ARITHMETIC (2026-07-30).  postRank's metric
    columns are `z x w`.  Any consumer that wants the metric must un-weight, and CONSUMERS THAT
    MUST AGREE WITH EACH OTHER ARE THE WHOLE PROBLEM: the OLS path fits coefficients in one
    function (`backtest_unified.run_top100_postrank_ols`) and applies them in another
    (`backtest_outputs.compute_ols_weighted_ranking`).  When only the FIT side was un-weighted,
    the two bases diverged for negative weights and the re-ranker inverted -- because
    `standardize(z x w) == sign(w) * standardize(z)`.  Before that, BOTH sides used `z x w` and
    the double negation made the result accidentally correct.  So a one-sided fix was worse
    than no fix, and the durable remedy is that both sides call the SAME function.

    `w = 0` columns are DROPPED, not divided: they are identically +-0.0 in postRank (the
    multiply annihilated them), so there is no information to recover and 0/0 is not a metric.
    Columns with no weight entry (e.g. `moatScore`, which is merged post-weighting and is
    already raw) are left untouched.

    THIS IS NOW `metric_frame`'s BASIS_Z_TIMES_W -> BASIS_Z implementation, and prefer
    `metric_frame(df, BASIS_Z)` in new code: it additionally CHECKS the frame's declared
    basis, so it cannot be pointed at a frame that was never weighted.  The name and the
    (out, kept, dropped) contract are kept because three out-of-file consumers call it
    (backtest_unified, backtest_outputs, backtest_ols_analysis).
    """
    W = _weight_vector()
    out = df.copy()
    if cols is None:
        cols = _weighted_metric_cols(out)
    kept, dropped = [], []
    for c in cols:
        if c not in out.columns:
            continue
        w = W.get(c)
        if w is None:
            kept.append(c)          # not a weighted metric -- leave as-is
            continue
        if w == 0:
            dropped.append(c)
            continue
        out[c] = pd.to_numeric(out[c], errors='coerce') / w
        kept.append(c)
    if verbose:
        print('  %sun-weighted %d postRank metric column(s) to the metric z-scale (signs now '
              'match the metrics); dropped %d zero-weight column(s) %s'
              % (label, len(kept), len(dropped), dropped), flush=True)
    return out, kept, dropped


# =========================================================================== #
#  SUMMATION ORDER MUST BE DETERMINISTIC                                       #
# =========================================================================== #
# THE DEFECT (found independently on two workstreams, fixed 2026-08-02).  Both reducers
# below selected their columns with `list(set(df.columns) - {'source'})`.  A `set` OF
# STRINGS iterates in hash order, and CPython RANDOMISES the string hash seed PER PROCESS
# (PYTHONHASHSEED), so the column order -- and therefore the ORDER OF THE FLOATING-POINT
# ADDITIONS -- differed between runs of the same code on the same data.  Float addition is
# not associative, so `AggScore` was not bit-reproducible across processes.
#
# MEASURED, not inferred.  Scoring the SAME saved frame under five hash seeds produced five
# different AggScore byte-hashes; re-summing one frame's own columns in eight shuffled
# orders moved the row sum by up to 2.220e-16, while the weighted metric columns feeding it
# were bit-identical (0.000e+00).  A parallel dev hit the same thing from the other side:
# 1-3 ULP on 65 of 100 names between two runs of a provably innocent change.
#
# WHY A 1e-16 WOBBLE IS WORTH FIXING -- it is not the magnitude:
#   * ANY exact-equality gate on AggScore flakes intermittently, for a reason nobody
#     diagnoses quickly.  This whole refactor programme is judged on bit-identity harnesses,
#     and those harnesses were being asked to compare a number that moves by itself.
#   * "the ranking changed" could not be distinguished from "the process restarted" at a tie.
#     Rank 20 and rank 21 are separated by ~0.0006 AggScore on the shipped panel, so ties at
#     the shortlist boundary are not hypothetical -- and getRankOfRanks feeds its sum
#     straight into `.rank()`, where a 1-ULP difference IS a rank flip.
#
# THE ORDER CHOSEN is the canonical emission order the weights now define
# (scoringWeights.METRIC_KEYS), NOT a second ordering invented here -- so the sum runs in
# the same order the weight vector is written in.  Any column NOT in the canon (a caller's
# extra column, `moatScore`, a diagnostic) follows in the FRAME'S OWN column order, which is
# itself deterministic.  Nothing is added to or removed from the summed SET: this changes
# only the order, which is why the values may move by an ULP and cannot move by more.
def deterministic_column_order(columns, exclude=('source',)):
    """`columns` in a REPRODUCIBLE order: canonical metric keys first, then the rest as given.

    Replaces `list(set(columns) - {'source'})` at every site that then does something
    ORDER-SENSITIVE with the result (a float sum, a rank, a design matrix).  Never use a set
    for that: the set is the bug.
    """
    try:
        import scoringWeights as _sw
        canon = [k for k in _sw.METRIC_KEYS]
    except Exception:
        # A missing/broken canon must not make the reducer non-deterministic: fall back to
        # the frame's own order, which is still reproducible -- just not canonical.
        canon = []
    cols = [c for c in columns if c not in exclude]
    known = [c for c in canon if c in cols]
    rest = [c for c in cols if c not in known]
    return known + rest


def getAggScore(df):
    #df['AggScore'] = np.nan
    # DETERMINISTIC column order -- see the note above.  `set(...)` here made AggScore
    # differ in its last bits between processes.
    cts = deterministic_column_order(df.columns)
    df['AggScore'] = df[cts].sum(axis=1)
    postRank = df
    postRank.sort_values(by='AggScore',ascending=False,inplace=True)
    postRank.reset_index(drop=True,inplace=True)

    return postRank

#  Columns that must never be ranked INSIDE the rank-of-ranks: AggScore is the weighted
#  SUM of every other column here, so ranking it alongside its own components counts the
#  whole score a second time (audit M1).
ROR_EXCLUDE = ['source', 'AggScore', 'rankOfRanks', 'rankOfRanks_diag']

#  DIAGNOSTIC name.  rankOfRanks orders NOTHING that ships (AggScore does), and it is
#  invariant to weight MAGNITUDE -- only weight SIGNS survive a per-column rank -- so it
#  is an EQUAL-WEIGHT alternative view, not a competing ranking.  It used to ship
#  unlabelled beside AggScore in three CSV families with a visibly different top-5
#  (2026-07-17: AggScore IMPP/RAVE/SYS1.L/INMD/AUDC vs rankOfRanks
#  IMPP/AJ91.DE/RFX.L/AEP.L/CAPD.L), which invites reading it as a second opinion it is
#  not entitled to be.  The `_diag` suffix is the label.
ROR_COLUMN = 'rankOfRanks_diag'


def getRankOfRanks(df):
    """Equal-weight rank-sum DIAGNOSTIC, emitted as ROR_COLUMN.

    Sums each name's per-metric rank and re-ranks the sum.  Because `rank()` discards
    magnitude, this weights every metric EQUALLY and keeps only the sign of the
    production weight -- deliberately a different lens from AggScore, never a substitute
    for it.  AggScore itself is EXCLUDED from the sum (audit M1 fix, 2026-07-19): it was
    previously included as a 22nd ranked column, i.e. the weighted sum of the other 21
    counted twice.  Verified against the shipped 2026-07-17 run: including it reproduced
    the shipped column bit-for-bit, and excluding it moves 80 of 90 names (max 11 rank
    positions).
    """
    postRankOfRanks = pd.DataFrame()
    for col in df.columns:
        if col not in ROR_EXCLUDE:
            postRankOfRanks[col + 'rank'] = df[col].rank(ascending=False,method='dense')

    # DETERMINISTIC column order -- and this one matters MORE than AggScore's, because the
    # sum feeds `.rank()` directly: a 1-ULP difference from a different summation order is a
    # RANK FLIP, not a rounding wobble.  The columns here are the `<metric>rank` names, so
    # the canonical list does not match them and they fall through to the frame's own order
    # -- which is deterministic because the loop above builds them by iterating df.columns.
    cts = deterministic_column_order(postRankOfRanks.columns)
    df[ROR_COLUMN] = postRankOfRanks[cts].sum(1).rank(ascending=True,method='dense')

    return df



def postBoRankingPassFilter(df,mlist,lco,hco):
    pf = df[~df[df.columns.intersection(mlist)].lt(lco).any(axis=1)]
    pf = pf[~pf[pf.columns.intersection(mlist)].gt(hco).any(axis=1)]
    pf.reset_index(inplace=True, drop=True)

    return pf
