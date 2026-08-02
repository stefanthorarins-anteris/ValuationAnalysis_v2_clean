import os
import sys

import createDicts as cdic
import getData_gen as gdg
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    postScoreMetric_df, outlierlist = normalizeAndDropNA(postScoreMetric_df,
                                                         weight_series=weight_series)
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

    setv('freeCashFlowYield', sm.free_cash_flow_yield(tempfcf, tempmcap, nq, rpy=rpy))
    setv('freeCashFlowPerShareGrowth',
         sm.free_cash_flow_per_share_growth(tempfcf, tempshares, nq, rpy=rpy))
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
            keptset = set(kept)
            postRank = postRank[postRank['source'].isin(keptset)].reset_index(drop=True)
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


WINSOR_SIGMA = 3.0        # winsorization threshold, in sigmas of the RAW column
# Safety bound on the mu/sigma <-> clip fixed-point loop.  Set well ABOVE what the real
# pools need: measured on the shipped 2026-07-17 columns, convergence takes up to 64
# passes (currentRatio 57, EPStoEPSmean 50, incomeQuality 64) -- a bound of 20 stopped
# just short of the fixed point and left max|z| = 3.000298 instead of <= 3.  The bound is
# NOT load-bearing for the result: the set of clipped cells stabilises by ~pass 20 and the
# remaining passes only settle sigma in the 6th decimal (verified: identical clipped-cell
# counts and identical AggScore ordering at 200 vs 500 passes).  Each pass is one
# vectorised op over ~100 rows, so the headroom is free.
WINSOR_MAX_PASSES = 200


WINSOR_Z_EPS = 1e-9       # slack on the max|z| <= WINSOR_SIGMA target test

# Columns that CANNOT have a fat tail because they are BOUNDED or DISCRETE by
# construction, and are therefore EXEMPT from winsorization entirely (review H3(c);
# this is the choice made, not just considered):
#   Piotroski            integer 0..9 (9 binary criteria)
#   CycleHeat            hard-capped to [-3, +3] in stage2_metrics.cycleheat_zscore
#   marketCapRevQuants   5 discrete values after fix 16 (-0.5, -1/6, 0, +1/6, +0.5)
# For a bounded discrete column a large |z| is a property of the SPLIT -- how many names
# sit on each level -- not of a heavy tail.  Piotroski 20 names at 7 and one at 2 puts
# the single WORST-F name at |z| = 4.36, and "winsorizing" that moves the worst F-score
# onto the best, destroying real structure to satisfy a threshold that was never meant
# for it.  Leaving them alone is strictly more honest than clipping them.
WINSOR_EXEMPT_BOUNDED = ('Piotroski', 'CycleHeat', 'marketCapRevQuants', 'mcapQuants')


# --------------------------------------------------------------------------- #
#  NORMALISATION METHOD  (A/B switch, 2026-07-27)                             #
# --------------------------------------------------------------------------- #
# 'zscore' : winsorize the RAW column, then (x - mu) / sigma.  THE SHIPPED PATH.
# 'rank'   : map the column to ranks, then through the inverse normal CDF.
#
# WHY 'rank' EXISTS.  Sigma-winsorization assumes an approximately symmetric, roughly
# Gaussian column; several of these metrics are strongly skewed, which the winsorizer
# reduces but cannot remove.  Two consequences the z-path cannot escape:
#   (1) CROSS-COLUMN INCOMPARABILITY.  A weight w only means "w units of AggScore per
#       sigma of THIS column", and a sigma of a skewed column is not a sigma of a
#       symmetric one, so the weight vector's components silently mean different things.
#       After a rank map every column is the SAME distribution by construction, so a
#       weight means one thing everywhere -- which is the precondition for the weight
#       vector to be re-fittable at all.
#   (2) THE MISSING-DATA REWARD (reviewer finding N1).  Both paths fill an unavailable
#       metric with 0, but 0 means different things: under z-scoring 0 is the winsorized
#       MEAN, which on the shipped 07-17 pool sat at the 52nd-65th percentile on 15 of 17
#       weighted columns -- so a name MISSING a metric scored ABOVE the typical name on it.
#       THE TWO NUMBERS, CORRECTED 2026-08-02 (both re-derived from
#       baseline_tools/resdic_2026-07-17_CORRECTED.pickle; the previous pair, "+0.1394
#       against a 0.134 median-to-top-20 distance", was wrong on BOTH and made the reward
#       read as ~104% of the distance to the shortlist, i.e. ~3x its real size):
#           median -> rank-20 AggScore distance        0.2560   (was stated 0.134)
#           full-missingness advantage, 0 - sum w*med(z)  +0.0783   (was stated +0.1394)
#           => ~31% of the distance, NOT ~104%.
#       PANEL-DEPENDENT, so do not re-quote either figure without its panel: on today's
#       code over the same inputs they are 0.2714 and +0.0792 (~29%).  That drift is exactly
#       why the stale pair survived -- a number with no panel attached cannot be checked.
#       Under the rank map 0 is AT OR NEAR the median, so the same fillna(0) is
#       (near-)neutral and most of the reward vanishes without a special case.
#       PRECISELY (do NOT round this up to "0 is the median by construction" -- it is not):
#       the map centres on the median EXACTLY only when the column's observed values are
#       DISTINCT.  Ties displace the centre, because a tied group takes one averaged
#       plotting position instead of spanning several: `_rank_normal([1, 1, 2])` has mean
#       +0.0260, and on a 60/40 binary column the fill sits above ~60% of the pool.  This
#       is not a technicality -- it IS the entire measured residual.  Of the +0.0228 that
#       survives on the 07-17 pool, `marketCapRevQuants` alone contributes +0.0244: five
#       discrete levels, hence massive ties, hence a fill that is materially off-median.
#       Measured effect of the switch: full-missingness advantage +0.1616 -> +0.0228 (-86%),
#       and columns whose fill sits above their own median 14/18 -> 2/18.  A real and large
#       reduction; NOT an elimination.
#       *** UNRECONCILED, FLAGGED 2026-08-02 -- DO NOT QUOTE THIS LINE WITHOUT READING THIS.
#       The pair above is a THIRD set of values for the same two quantities corrected higher
#       up, and it does not reconcile with them.  Re-measured on
#       resdic_2026-07-17_CORRECTED.pickle under the definition that reproduces the corrected
#       figures EXACTLY (advantage = 0 - sum w*median(z), which returns 0.0783 / 0.2560 to
#       four decimals), the switch reads:
#              advantage  +0.0783 (z) -> -0.0186 (rank)
#              fill above its own median  15/18 -> 1/18
#       i.e. the rank path lands the fill slightly BELOW the median here, not +0.0228 above,
#       and the +0.0244 marketCapRevQuants attribution above cannot be checked against a
#       +0.0228 total that does not reproduce.  WHY IT COULD NOT BE SETTLED: the numbers were
#       almost certainly taken on the SHIPPED pre-fix 07-17 panel, and the shipped resdic
#       (HomeGDrive/postRank_2026-07-17_*.pickle) predates the `postScoreMetric_raw` key, so
#       there is no raw frame left on this machine to re-derive them from.  Left in place
#       rather than overwritten with numbers whose panel I cannot state: the DIRECTION (a
#       large reduction, not an elimination) is what this note is load-bearing for, and that
#       survives on both measurements. ***
#
# DEFAULT IS UNCHANGED.  'zscore' remains the production default: the deployed mu weights
# were tuned on the z-path, and a rank map changes what each weight MEANS, so switching
# the live default is a CEO decision that belongs WITH the weight re-fit, not before it.
# The switch exists so the two can be measured side by side on identical inputs.
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


def _winsorize_raw(x, n_sigma=WINSOR_SIGMA, max_passes=WINSOR_MAX_PASSES):
    """Symmetric sigma-winsorization of a RAW metric column, iterated toward max|z| <=
    n_sigma.  Returns (series, n_cells_changed, n_passes, converged).

    Body of the loop is the specified two-pass move: compute mu/sigma, clip the raw
    values to mu +- n_sigma*sigma, recompute mu/sigma on the clipped column.  It is
    REPEATED because ONE pass is not enough, and that is measurable rather than
    theoretical: sigma of the first pass is itself inflated by the outlier being
    clipped, so mu1 +- 3*sigma1 lands far out in the tail, and after re-standardising on
    the (much tighter) clipped column the clipped value can sit at a LARGER |z| than
    before.  On the shipped 2026-07-17 pool a single two-pass left worst-column
    max|z| = 9.90 (EPStoEPSmean) -- WORSE than the +-4 z-clamp it replaced -- with
    sigma1/sigma2 up to 3.23.

    WHAT IS AND IS NOT GUARANTEED (review H3(d) -- the earlier "max|z| <= n_sigma BY
    CONSTRUCTION" claim was FALSE).  Iterating does NOT always reach the target: for a
    near-two-point column the clip ratio is constant, so the minority value decays
    GEOMETRICALLY toward the mode while its z-score -- which is scale-invariant -- never
    moves.  There is no interior fixed point.  Measured cases: 99x0 + 1x1e6 ends with the
    outlier at 4.9e-13 (1e-19 of input) and max|z| still 9.900; Piotroski 20x7/1x2 (n=21)
    burns 74 passes and stays at 4.364; CycleHeat 25x(-3)/2x(+1) (n=27) burns 194 of 200.
    The old exit test compounded this: `tol = 1e-12 * max(1.0, |mu|+sigma)` is an
    ABSOLUTE floor, so once a column's own scale fell below ~1e-12 the test passed
    vacuously and the loop exited OUTSIDE the threshold with no signal at all.

    So the contract is now explicit and two-branched:
      * converged=True  -> max|z| <= n_sigma + WINSOR_Z_EPS is VERIFIED on the returned
                           series (the target test is evaluated directly, not inferred
                           from a proxy).
      * converged=False -> the target is UNREACHABLE for this column's shape.  The
                           ORIGINAL raw values are returned UNCHANGED (never the
                           annihilated ones) and the caller logs it loudly.  A column
                           whose spread is structural is left at its natural z rather
                           than mangled.
    """
    orig = pd.to_numeric(x, errors='coerce')
    y = orig
    for p in range(max_passes):
        m, s = y.mean(), y.std()
        if not np.isfinite(m) or not np.isfinite(s) or s <= 0:
            # constant / all-NaN column: no outlier can exist, target trivially holds
            return orig, 0, p, True
        if float(((y - m) / s).abs().max()) <= n_sigma + WINSOR_Z_EPS:
            changed = int((~np.isclose(orig.to_numpy(dtype='float64'),
                                       y.to_numpy(dtype='float64'),
                                       rtol=0, atol=0, equal_nan=True)).sum())
            return y, changed, p, True
        y2 = y.clip(m - n_sigma * s, m + n_sigma * s)
        # NO-PROGRESS test: BIT-EXACT equality.  It must be exact, not a relative
        # tolerance -- the approach to the target is asymptotic, so a 1e-9 relative test
        # RACES the target test and loses: a lognormal(0,1.2) n=100 column reaches
        # max relative change 6.2e-10 at pass 18, one pass BEFORE max|z| first tests
        # <= 3, so a relative test declared that genuinely-winsorizable column
        # un-winsorizable and left its real fat tail alone.  Exact equality only fires
        # when the clip truly cannot move anything, which is what "no progress" means.
        if np.array_equal(y2.to_numpy(dtype='float64'), y.to_numpy(dtype='float64'),
                          equal_nan=True):
            return orig, 0, p + 1, False      # cannot make progress -> un-winsorizable
        y = y2
    return orig, 0, max_passes, False         # pass bound exhausted -> un-winsorizable


def normalizeAndDropNA(df, weight_series=None, winsor_sigma=WINSOR_SIGMA,
                       method=None, rank_bounded=True):
    """WINSORIZE each weighted RAW metric column at +-winsor_sigma, then
    cross-sectionally z-score; drop a row only when EVERY metric is NaN.

    method : NORM_ZSCORE (default, the shipped path -- everything below applies) or
             NORM_RANK (rank -> inverse-normal; see NORM_RANK's notes and _rank_normal).
             Under NORM_RANK the winsorizer is NOT run: a monotone relabelling of a column
             has the same ranks as the column, so clipping its tail cannot change the
             result.  The winsorizer is retained, not removed -- the z-path still uses it.
    rank_bounded : NORM_RANK only.  True (default) rank-maps the bounded/discrete columns
             (WINSOR_EXEMPT_BOUNDED) along with the rest.  Their WINSORIZATION exemption
             does not carry over, because it rests on a premise the rank map removes: they
             are exempt from clipping because a large |z| there is real structure that
             clipping would destroy, whereas ranking destroys nothing (it is
             order-preserving) and their spacing is ORDINAL anyway -- Piotroski 9 vs 8 is
             "one criterion better", not a measured distance, which is exactly what a
             normal-scores map assumes.  Leaving them un-mapped would also put a raw-scaled
             column beside 13 N(0,1) columns inside a frame every consumer downstream
             reads as normalised, i.e. it would re-introduce the cross-column
             incomparability the method exists to remove.  Set False to measure that
             alternative.

    OUTLIER HANDLING (audit H1/H2 fix 2026-07-19, upgraded 2026-07-25).  Originally
    this EJECTED any row with |z| > 4 in ANY metric column.  Three things were wrong:

      (a) ZERO-WEIGHT metrics could eject a name.  The mask ran over every column
          including the w=0 diagnostics (priceGrowth / DcfToPrice / BoScore), so a
          metric that contributes NOTHING to AggScore could delete a company from
          the ranking.  Proven on the shipped 2026-07-17 run: of the 10 names this
          function ejected, CART was ejected SOLELY on priceGrowth (w = 0.000).
      (b) EJECTION IS ADVERSE SELECTION.  |z| > 4 on a value/quality metric is
          usually the name being EXCEPTIONAL on that axis (highest FCF yield,
          fastest growth, safest balance sheet), i.e. exactly what the score is
          hunting.  Dropping it also silently shortened every top-100 deliverable
          to 90 rows (verified: postScoreMetric_raw = 100 rows, shipped
          postScoreMetric = 90).
      (c) CLAMPING THE Z IS ONLY HALF A FIX (the 2026-07-19 interim, now replaced).
          Clamping AFTER mu/sigma are computed leaves mu/sigma CONTAMINATED by the
          outlier, so every OTHER name's z stays deflated by the sigma the clipped
          value inflated -- currentRatio delivered sigma(z) ~ 0.5, i.e. about half its
          intended weight (audit H2).  And +-4 sigma is a don't-crash bound, not a
          winsorization threshold: a clean n~90 column's true max is |z| ~ 2.6-3.0, so
          a value clamped to 4 still sat above every legitimate name.

    NOW: the RAW column is winsorized at +-winsor_sigma (see _winsorize_raw), and
    mu/sigma for the z-score are computed on the WINSORIZED column -- so the outlier
    neither dominates the score nor distorts anyone else's z.  The NAME IS ALWAYS KEPT.

    WHAT IS GUARANTEED (corrected -- see _winsorize_raw; the earlier "BY CONSTRUCTION"
    wording was false): for every column the winsorizer REPORTS as converged, max|z| <=
    winsor_sigma is verified directly on the result.  A column whose shape makes that
    target unreachable is returned UNTOUCHED at its natural z and named loudly on stdout;
    it is NOT silently left part-clipped, and its raw values are NOT annihilated.

    THREE exemptions, all deliberate:
      * ZERO-WEIGHT columns -- a w=0 column can neither dominate the score nor eject
        anyone, so it stays the honest display-only diagnostic it is;
      * BOUNDED/DISCRETE columns (WINSOR_EXEMPT_BOUNDED) -- they cannot have a fat tail,
        so a large |z| there is real structure;
      * columns the target cannot be reached for (above).
    With no weight_series supplied (the offline baseline_tools callers) every non-bounded
    metric column is winsorized.

    INTERIM, by design.  Sigma-winsorization still assumes an approximately symmetric,
    roughly-Gaussian column.  Several of these metrics are strongly skewed and the
    weights were mu-tuned on the pipeline as it stood, so the principled end state is
    RANK-BASED (inverse-normal) normalization plus a weight RE-FIT -- both of which
    change the weight vector's meaning and therefore need a CEO decision.  This fix
    removes the demonstrable defects (contaminated mu/sigma, a bound that bounds
    nothing) without pre-empting that decision.

    NOT renormalized after clamping, deliberately: the production weights were
    tuned on this pipeline as-is, so rescaling z back to unit variance would
    silently re-weight the whole vector.

    Returns (frame, outlierlist).  outlierlist now holds ONLY the all-NaN rows
    that were genuinely dropped -- winsorized names are NOT outliers, they are in
    the ranking.
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
        # No winsorization pass, because the map depends only on the ORDER within a column
        # and clipping cannot REORDER anything: every name the winsorizer would move keeps
        # its exact rank position relative to every other name.  (Precisely: clipping is
        # weakly monotone, so its only possible effect here is to MERGE the clipped tail
        # into one tied score -- it can never swap two names, and it cannot change any
        # unclipped name's score at all.  Verified in
        # test_rank_normalization.test_rank_normal_is_invariant_to_any_strictly_monotone_
        # transform + ..._clipping_only_merges_the_clipped_tail.)  The fat tail the
        # winsorizer exists to defuse cannot dominate a rank map in the first place, so
        # running it would only destroy raw values for no change in outcome.  Nothing is
        # silently skipped -- it is inapplicable.
        if rank_bounded:
            to_map = list(tempnum.columns)
        else:
            to_map = [c for c in tempnum.columns if c not in WINSOR_EXEMPT_BOUNDED]
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

    # --- PASS 1: winsorize the WEIGHTED RAW columns at +-winsor_sigma -------------
    # Done BEFORE mu/sigma so the z-score below is computed on an UNCONTAMINATED
    # column. w=0 columns are left completely alone (display-only diagnostics).
    if weight_series is None:
        weighted = list(tempnum.columns)                # offline callers: guard all
    else:
        weighted = [c for c in tempnum.columns
                    if float(weight_series.get(c, 1) or 0) != 0]
    guarded = [c for c in weighted if c not in WINSOR_EXEMPT_BOUNDED]
    per_col_clipped, affected_rows, total_clipped = {}, pd.Series(False, index=tempnum.index), 0
    not_converged = {}
    for col in guarded:
        clipped, n_changed, n_passes, converged = _winsorize_raw(tempnum[col], winsor_sigma)
        if not converged:
            # LEFT AT ITS NATURAL Z, deliberately (see _winsorize_raw): the target is
            # unreachable for this column's shape, so clipping would annihilate the raw
            # value without moving a single z-score.
            s = tempnum[col]
            mz = float(((s - s.mean()) / s.std()).abs().max()) if s.std() else float('nan')
            not_converged[col] = (n_passes, mz)
            continue
        if n_changed:
            moved = ~np.isclose(tempnum[col].to_numpy(dtype='float64'),
                                clipped.to_numpy(dtype='float64'),
                                rtol=0, atol=0, equal_nan=True)
            affected_rows |= pd.Series(moved, index=tempnum.index)
            per_col_clipped[col] = (n_changed, n_passes)
            total_clipped += n_changed
        tempnum[col] = clipped
    if total_clipped:
        print(f"normalizeAndDropNA: winsorized {total_clipped} RAW metric cell(s) at "
              f"+-{winsor_sigma} sigma (names KEPT, mu/sigma recomputed after): "
              + ", ".join(f"{c}={n}(x{p} passes)" for c, (n, p) in per_col_clipped.items())
              + f" | affected names: {sorted(dfnona['source'][affected_rows.to_numpy()])}",
              flush=True)
    if not_converged:
        # LOUD, per column: a metric whose spread the winsorizer could not tame is a fact
        # the reader needs, not a silent internal outcome.
        print("!" * 78, flush=True)
        print("normalizeAndDropNA: WINSORIZATION NOT APPLIED to %d column(s) -- the "
              "max|z| <= %s target is UNREACHABLE for their shape (near-two-point /"
              " discrete), so they are LEFT AT THEIR NATURAL z:"
              % (len(not_converged), winsor_sigma), flush=True)
        for c, (p, mz) in not_converged.items():
            print("    %-28s passes=%-4d natural max|z|=%.4f  (raw values UNCHANGED)"
                  % (c, p, mz), flush=True)
        print("!" * 78, flush=True)
    exempt_bounded = [c for c in weighted if c in WINSOR_EXEMPT_BOUNDED]
    if exempt_bounded:
        print("normalizeAndDropNA: bounded/discrete metric(s) EXEMPT from winsorization "
              f"(cannot have a fat tail): {exempt_bounded}", flush=True)
    unguarded = [c for c in tempnum.columns if c not in weighted]
    if unguarded:
        print("normalizeAndDropNA: zero-weight metric(s) EXEMPT from winsorization "
              f"(display-only, cannot dominate or eject): {unguarded}", flush=True)

    # --- PASS 2: mu/sigma on the WINSORIZED columns, then z-score -----------------
    colmeans = tempnum.mean()
    colstds = tempnum.std()
    # Handle division by zero: if std is 0 or NaN, set normalized values to 0
    colstds = colstds.replace(0, np.nan).fillna(1)  # Avoid division by zero
    # subtract the mean and divide by the standard deviation
    temp_normpsmdf = (tempnum - colmeans) / colstds
    # Fill remaining NaN values with 0 (for columns that were all NaN)
    temp_normpsmdf = temp_normpsmdf.fillna(0)

    dfnona[temp_normpsmdf.columns] = temp_normpsmdf
    dfnonanorm = dfnona.copy()

    return dfnonanorm, outlierlist

MISSING_REPORT_CSV = 'MissingDataFillReport_%s.csv'
_MISSING_CSV_STARTED = set()


def missing_data_fill_report(raw_df, norm_df, weight_series, pool='general',
                             csv=True, verbose=True):
    """WHERE THE fillna(0) IMPUTATION LANDS, per weighted column and per name.  EMITS ONLY.

    WHY IT HAS TO BE PRODUCED BY THE RUN (2026-08-01).  A missing metric is imputed by
    `normalizeAndDropNA`'s post-normalisation `fillna(0)`, i.e. it is scored AT THE POOL MEAN
    of that column, not as "missing".  Whether that is generous or punitive depends on where 0
    sits in the column's OBSERVED z distribution -- which is a property of THIS pool on THIS
    panel.  Every fill percentile we hold was measured on a pre-change panel, and the 2026-08-01
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
