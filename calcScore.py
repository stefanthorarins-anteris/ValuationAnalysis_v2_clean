import math

import pandas as pd
import numpy as np
from tqdm import tqdm
import createDicts as cdic
import reporting_period as rp
import meanBars as mb

# --- HISTORY BONUS (register C-7, CEO ruling 2026-08-05) ----------------------
# Named here rather than inline so the two numbers the ruling actually fixes are in one
# place and a test can assert them.  The shape, the scale argument and the honest side
# effects are documented at the single use site in `simpleScore_fromDict`.
HISTORY_BONUS_MAX = 0.05             # bonus at saturation; HALF the smallest criterion
                                     # difference (a Tier-D criterion over a full window)
HISTORY_BONUS_SATURATION_ROWS = 40   # ROWS available to the source, not the head(n) window

def simpleScore_fromDict(bm_df,bm_ave,bm_da,n=8,as_of=None,freq_map=None):
    """Stage-1 per-symbol scoring.

    as_of : point-in-time date D (default None).  as_of=None reproduces the live
    pipeline BIT-FOR-BIT: the PIT slice below is never entered, so every symbol is
    scored over its full panel exactly as today.  Only when a real D is supplied is
    each symbol's panel restricted to rows AVAILABLE on/before D (pit_slice, design
    L1/L4) BEFORE the head(n) scoring window -- so head(n) picks the correct
    as-of-D quarters instead of assuming "newest row == today".
    """
    print(f'Calculating scores for each stock symbol in BoMetric_df')

    # --- ORDERING INVARIANT (Stage-1): calcByTier's .head(n) scoring window
    # assumes each ticker's rows are NEWEST-first. On tonight's data BoMetric_df
    # arrives newest-first (verified 600/600 descending), but NOTHING on the live
    # path enforces it -- data_quality re-sorts only cdx_df, so this is an
    # incidental FMP ingestion order, not an invariant. Defensively re-sort a COPY
    # to newest-first: a no-op when already correct, a fix if the order ever drifts.
    # Dates coerced robustly (a naive string sort mis-orders mixed/malformed dates).
    # Mirrors stage2_pit._sort_newest_first and the Stage-2 re-sort in
    # postBoRank.postBoScoreRanking.
    if 'date' in bm_df.columns:
        bm_df = bm_df.copy()
        _n_before = bm_df.groupby('source').size()
        bm_df['date'] = pd.to_datetime(bm_df['date'], errors='coerce')
        bm_df = bm_df.sort_values(['source', 'date'], ascending=[True, False]).reset_index(drop=True)
        assert bm_df.groupby('source').size().equals(_n_before), \
            "Stage-1 newest-first re-sort changed per-ticker row counts"
    # test
    #    bm_df = BoMetric_df
    #    bm_da = BoMetric_dateAve
    #    bm_ave = datandmetricdic['BoMetric_ave']
    #   BoScore_df['date'] = BoMetric_dateAve.index
    #test
    # PER-SOURCE REPORTING FREQUENCY -- the head(n) Stage-1 scoring window is scaled by
    # it so a semi-annual filer is scored over the same CALENDAR span as a quarterly
    # one.  n=8 rows is 2 years quarterly but FOUR years semi-annual, so without this
    # the two frequencies were graded on different amounts of history.  BoMetric_df
    # carries no `period` column (preReq fields never reach it), so the classifier
    # falls back to date cadence here -- which is the only signal available and is
    # what the 07-17 validation used.  unknown -> quarterly, i.e. unchanged.
    if freq_map is None:
        freq_map = rp.frequency_by_source(bm_df, verbose=True)
    dict_base, dict_mean, dict_diff, dict_unity, dict_special = cdic.getBaseMeanDiffUnitySpecialDicts()
    # SCHEMA / BASIS GATE (review S10, 2026-07-26).  Stage-1's criterion set is data,
    # not code: a metric added or renamed changes the required BoMetric_df columns, so a
    # pickle saved before that change simply cannot be scored.  Without this the failure
    # was a bare KeyError deep in the per-ticker loop with no hint that the PANEL was the
    # problem.  Name the missing columns and say what to do.
    _need = (list(dict_base) + ['m' + k[0].upper() + k[1:] for k in dict_mean]
             + ['d' + k[0].upper() + k[1:] for k in dict_diff]
             + ['u' + k[0].upper() + k[1:] for k in dict_unity] + list(dict_special))
    _missing = [c for c in _need if c not in bm_df.columns]
    if _missing:
        raise KeyError(
            'BoMetric_df is missing %d Stage-1 criterion column(s): %s. This panel was '
            'built by an OLDER version of the metric set and cannot be scored by the '
            'current code -- re-fetch, or check out the code that produced it. Scoring '
            'it anyway would mix metric bases silently.'
            % (len(_missing), _missing))
    _nan_acct = {}
    tbs_df = pd.DataFrame(columns=['score', 'source'])
    tbs_df['source'] = bm_df['source'].unique()
    pbar = tqdm(total=len(bm_df['source'].unique()))

    for ticker in bm_df['source'].unique():
        bmdf_tick = bm_df[bm_df['source'] == ticker]
        if as_of is not None:
            # PIT: keep only rows available on/before D, then score head(n) over them.
            # Never entered on a live run (as_of=None) -> live behaviour unchanged.
            import pit_slice as ps
            bmdf_tick = ps.slice_panel_as_of(bmdf_tick, D=as_of)
            if bmdf_tick.empty:
                continue
        tempscore = 0
        _tick_nan = []
        # STAGE-1 WINDOW IS *NOT* FREQUENCY-SCALED (CEO/specialist ruling Q2, 2026-07-26).
        # Every source is scored over the SAME NUMBER OF ROWS, head(n).
        #
        # WHY, and why this differs from every other window in the pipeline: calcByTier
        # returns `resvec.head(n).mean()` -- the mean of a PASS/FAIL indicator, i.e. an
        # ESTIMATED PROBABILITY that this company passes the criterion.  That estimand is a
        # property of the company; halving n does not change it, it only degrades the
        # estimator by ~sqrt(2).  Calendar-span reasoning is right where a window feeds a
        # LEVEL or a RATIO (TTM sums, YoY lags, Altman year-sums, moat means, the
        # Beneish/Montier windows -- all still scaled, deliberately); here it feeds a COUNT
        # OF BERNOULLI TRIALS, which is a different object.
        #
        # Measured on the 2026-07-17 panel: scaling left the semi-annual MEAN essentially
        # unchanged (10.0024 -> 9.9681, bias ~ 0) but raised its score SD 1.7322 -> 1.9393
        # (+12%), and top-tail selection converts estimator noise straight into gate share:
        # SA share of the RAW Stage-1 top-100 went 31% -> 46% against a ~15% universe base
        # rate.  (An earlier version of this comment also quoted "carved 45% -> 54%" -- that
        # pair was measured on an intermediate working tree and does NOT describe the
        # shipping code; the shipping CARVED+FLOORED general top-100 figure is 44%, and the
        # raw/carved/floored bases must never be quoted interchangeably.)  Bias ~ 0 with a large variance cost => for a GATE, take
        # the low-variance option.  The main counter-argument also failed empirically: the
        # d* diff family (44.5% of Stage-1 summed weight, the block calendar-span would most
        # protect) is frequency-neutral to -0.0095 on Sigma-w 8.30.
        #
        # HONEST COST: head(8) is 4 calendar years for a semi-annual filer, so Stage-1 is
        # SLOWER TO DROP a deteriorating semi-annual name.  Accepted: aggregate bias ~ 0,
        # and recency judgment lives downstream (Stage-2, forensics, R7, the display) on a
        # calendar-correct basis.
        #
        # `freq_map` is still computed and still drives the flow factors and the run banner;
        # it just no longer scales THIS window.  Do not "restore" the scaling here.
        _nw = n
        for key in dict_base:
            temp = calcByTier('base', dict_base[key]['Tier'], dict_base[key]['Sign'], bmdf_tick[key], bm_ave[key],key,_nw,nan_sink=_tick_nan)
            tempscore = tempscore + temp
        for key in dict_mean:
            mkey = "m" + key[0].upper() + key[1:]
            #  THE BAR IS A STORED CONSTANT, NOT THE POOLED MEDIAN (register C-12, 2026-08-06).
            #  `bm_ave[mkey]` is `getAves2`'s median of this column over EVERY company and
            #  EVERY date, so it made the criterion a rank rather than a bar, moved with the
            #  sample, and carried future data into a past row in any backtest.  `mean_bar`
            #  returns the stated constant, or RAISES for a mean criterion nobody has given
            #  one -- see meanBars.  The pooled median is still passed in because one
            #  DECLARED exception (`mSalesToMarketCap`, Tier N, w = 0) still uses it.
            temp = calcByTier('mean', dict_mean[key]['Tier'], dict_mean[key]['Sign'], bmdf_tick[mkey], mb.mean_bar(mkey, bm_ave[mkey]),key,_nw,nan_sink=_tick_nan)
            tempscore = tempscore + temp
        for key in dict_diff:
            dkey = "d" + key[0].upper() + key[1:]
            temp = calcByTier('diff', dict_diff[key]['Tier'], dict_diff[key]['Sign'], bmdf_tick[dkey], bm_ave[dkey],key,_nw,nan_sink=_tick_nan)
            tempscore = tempscore + temp
        for key in dict_unity:
            ukey = "u" + key[0].upper() + key[1:]
            temp = calcByTier('unity', dict_unity[key]['Tier'], dict_unity[key]['Sign'], bmdf_tick[ukey], bm_ave[ukey],key,_nw,nan_sink=_tick_nan)
            tempscore = tempscore + temp
        for key in dict_special:
            temp = calcByTier('special', dict_special[key]['Tier'], dict_special[key]['Sign'], bmdf_tick[key], bm_ave[key],key,_nw,nan_sink=_tick_nan)
            tempscore = tempscore + temp

        # HISTORY BONUS (register C-7, CEO ruling 2026-08-05).  A concave, saturating
        # bonus for a LONGER AVAILABLE PANEL, added to the Stage-1 score.
        #
        #     bonus = HISTORY_BONUS_MAX * sqrt(min(rows, HISTORY_BONUS_SATURATION_ROWS)
        #                                      / HISTORY_BONUS_SATURATION_ROWS)
        #
        # WHAT IT IS NOT.  It does not correct a punishment, because SHORT HISTORY IS NOT
        # PUNISHED TODAY -- calcByTier returns `resvec.head(n).mean()`, and the mean of a
        # 4-row source is the mean over those 4 rows, so a short source has the SAME
        # EXPECTED score as a long one, just a NOISIER estimate of it.  (If any comment
        # anywhere still says short history is penalised, it is wrong: the estimator is
        # unbiased and only its variance moves.)  This bonus therefore pays for ESTIMATOR
        # CONFIDENCE, not for a missing pass.
        #
        # ROWS, NOT CALENDAR SPAN, and that follows from the same fact: the quantity a
        # longer panel buys is more BERNOULLI TRIALS behind each criterion's pass rate, and
        # trials are counted in rows.  So it is deliberately NOT frequency-scaled -- the
        # same ruling (Q2, 2026-07-26) that leaves the head(n) window unscaled.  HONEST
        # CONSEQUENCE, stated rather than implied: for a quarterly filer 40 rows is 10
        # calendar years, for a SEMI-ANNUAL filer the same 40 rows is 20.  The CEO's ruling
        # says "40 quarters"; on a semi-annual source that is 40 ROWS, i.e. a longer
        # calendar span for the same bonus.
        #
        # SCALE -- checked against the score geometry rather than assumed, because the
        # whole point is that it must be decisive for ties and never more:
        #   * max BoScore                       17.85
        #   * rank-100 -> rank-20 span           0.7875   (bonus max = 6.3% of it)
        #   * smallest criterion difference      0.1      (a Tier-D criterion over a full
        #                                                  window), so 0.05 is EXACTLY HALF
        # 90.9% of names share their score with another name and currently break
        # ALPHABETICALLY (the C-13 tiebreak), so a 0.05 bonus decisively resolves ties and
        # flips genuinely close calls, and can NEVER outweigh one full Tier-D criterion
        # differing.  It is a tiebreak with a reason, not a new criterion.
        #
        # WHY sqrt AND NOT log.  Both are concave and both saturate; sqrt was chosen because
        # it spreads MORE of the bonus across the range the panel actually occupies (median
        # panel length is 20 rows): sqrt gives 0.0224 / 0.0354 / 0.0500 at 8 / 20 / 40 rows,
        # a 0.0276 spread, against log's 0.0204.  Required shape properties hold: 40 rows
        # (0.0500) > 30 rows (0.0433); the 30->40 increment (0.0067) is smaller than the
        # 20->30 increment (0.0079); and 80 rows gives EXACTLY the same 0.0500 as 40.
        #
        # SIDE EFFECT, accepted eyes-open: the bonus is monotone in listing age, so it is a
        # small systematic tilt toward long-listed companies.  That is the intended
        # direction (more trials = more trustworthy pass rates), but it IS a tilt.
        _hist_rows = len(bmdf_tick)
        tempscore = tempscore + HISTORY_BONUS_MAX * math.sqrt(
            min(_hist_rows, HISTORY_BONUS_SATURATION_ROWS) / HISTORY_BONUS_SATURATION_ROWS)

        tbs_df.loc[tbs_df['source'] == ticker, 'score'] = tempscore
        if _nan_acct is not None:
            _nan_acct[ticker] = (len(_tick_nan),
                                 float(sum(w for _m, w in _tick_nan)),
                                 sorted({m for m, _w in _tick_nan}))
        pbar.update(n=1)

    pbar.close()
    # NaN-WEIGHT READOUT (ruling Q1.5).  For every source, how many Stage-1 criteria were
    # scored on an entirely non-computable window, and what tier weight that adds up to.
    # This is what makes the Stage-2 missingness coupling visible, and it is the precondition
    # for any future gate-width experiment -- widening head(100) would activate that latent
    # reward, so the incidence has to be measured before anyone widens it.
    #
    # THE MISSINGNESS REWARD IS NOW ~ZERO, AND THE NUMBERS THAT SAID OTHERWISE WERE STALE ON
    # BOTH SIDES (corrected 2026-08-03; the old pair read "+0.1394 AggScore vs a 0.134
    # median->top-20 distance", i.e. it claimed missingness ALONE reached the shortlist).
    # Re-measured on baseline_tools/resdic_2026-07-17_CORRECTED.pickle, general top-100, with
    # advantage := 0 - sum_c w_c * median(z_c over OBSERVED cells):
    #     pre-E-1 ruler (winsorized mean-centred z)  advantage +0.0739  distance 0.2560  29%
    #     E-1 ruler     (median-centred robust z)    advantage  ~2e-18  distance 0.2396   0%
    # So the old figure overstated the reward by ~3.6x, and the E-1 median-centring retires it
    # outright: 0 IS the observed median of every column now, so an unavailable metric is
    # scored exactly at the typical name (columns whose fill beats their own median: 14 of 18
    # before, 0 of 18 after, at a 1e-12 tolerance).  BOTH FIGURES ARE PANEL-DEPENDENT -- do not
    # re-quote either without its panel; a number with no panel attached is how the stale pair
    # survived for weeks.  What remains latent is the OTHER half: this Stage-1 gate is what
    # keeps a name with nothing computable out of the pool in the first place.
    if _nan_acct:
        _tot_w = sum(v[1] for v in _nan_acct.values())
        _worst = sorted(_nan_acct.items(), key=lambda kv: -kv[1][1])[:15]
        print('STAGE-1 NaN ACCOUNTING: %d source(s) have >=1 all-NaN criterion; summed '
              'NaN tier-weight across the universe = %.2f' % (len(_nan_acct), _tot_w),
              flush=True)
        print('  worst by NaN tier-weight: '
              + '; '.join('%s n=%d w=%.2f %s' % (k, v[0], v[1], v[2][:4])
                          for k, v in _worst), flush=True)
    # DETERMINISTIC TIEBREAK (issue C-13, 2026-08-05).  This used to be
    # `sort_values('score', ascending=False)` -- pandas' DEFAULT kind is 'quicksort', which is
    # NOT STABLE, so the order among tied scores depended on an implementation detail of the
    # sort and could differ between runs, pandas versions, or even input orderings of the same
    # universe.  The Stage-1 score is heavily QUANTIZED (a sum of tier weights times k/8
    # window fractions), and 90.9% of names share their exact score with at least one other
    # name -- so the head(100) cut lands INSIDE a tie block essentially always, and which names
    # made the pool was not reproducible.
    #
    # THIS IS NOW LOAD-BEARING RATHER THAN HYGIENE: the veto layer (stage1_veto) sits on top of
    # this boundary and ejects names BEFORE the cut, so a non-reproducible boundary would make
    # the veto's own A/B measurements non-reproducible too.
    #
    # `kind='mergesort'` is pandas' stable sort, and `source` (the ticker) is an explicit
    # secondary key so the tiebreak is a STATED RULE rather than "whatever order the panel
    # arrived in".  Ticker is arbitrary but it is deterministic, total, and carries no
    # information that could bias the pick -- which is exactly what a tiebreak should be.
    tbs_df.sort_values(['score', 'source'], ascending=[False, True],
                       kind='mergesort', inplace=True)
    return tbs_df

def calcByTier(dict,Tier,Sign,metvec,avec,met,n,nan_sink=None):
    """nan_sink: optional list.  When given, a criterion whose SCORING WINDOW is entirely
    non-computable appends (met, tier_weight) -- the accounting behind the per-name
    NaN-weight readout below (ruling Q1.5).  Purely observational; the score is untouched."""
    resvec = pd.DataFrame(columns=[met])
    w = 0
    if Tier == 'S':
        w = 1
    elif Tier == 'A':
        w = 0.75
    elif Tier == 'B':
        w = 0.5
    elif Tier == 'C':
        w = 0.3
    elif Tier == 'D':
        w = 0.1
    else:
        w = 0

    if dict == 'mean':
        testvec = metvec - avec
    elif dict == 'unity':
        testvec = metvec - 1
    else:
        testvec = metvec

    # NaN SCORES AS A FAIL, DELIBERATELY (ruling Q1, 2026-07-26).  `Sign * NaN > 0` is
    # False, so a non-computable criterion contributes 0 to the window mean.  Three
    # reasons it stays that way, recorded because the alternative looks appealing:
    #
    #  1. SEMANTICS.  For the criteria where this bites most, the value is OUT OF DOMAIN
    #     rather than unmeasurable -- Graham on negative EPS_ttm or negative book value has
    #     no published definition at all.  "Undefined" is a real answer about the company,
    #     not a gap in the data.
    #  2. THE COMPLETENESS-FILTER ROLE.  Stage-1 failing NaN is what keeps incomplete names
    #     out of the top-100, and that mattered doubly while Stage-2 REWARDED missingness:
    #     a NaN metric is imputed at z = 0, and under the pre-E-1 mean-centred ruler 0 sat
    #     ABOVE the typical name on 14 of 18 weighted columns, worth +0.0739 AggScore for full
    #     missingness against a median->rank-20 distance of 0.2560 -- 29% of the way to the
    #     shortlist [resdic_2026-07-17_CORRECTED, general top-100].  (An earlier note here said
    #     "+0.1394 against 0.134", i.e. >100% and reaching the top-20 on missingness alone;
    #     both figures were stale, and the pair overstated the reward ~3.6x.)
    #     SINCE 2026-08-03 THAT HALF IS GONE: E-1 centres each column on its observed MEDIAN,
    #     so z = 0 IS the median and the measured advantage is ~2e-18, i.e. zero to float dust,
    #     with 0 of 18 columns rewarding missingness.  THE COMPLETENESS-FILTER ARGUMENT DOES
    #     NOT DEPEND ON THAT REWARD and still stands on its own: a name scored on nothing is
    #     scored on the pool, not on itself, whether or not the pool's centre flatters it.
    #  3. EXCLUDING NaN FROM THE MEAN IS ADVERSELY SELECTED.  head(n) with NaN dropped
    #     scores a name on its SELECTED rows: a loss-maker's computable Graham rows are
    #     exactly its profitable quarters, so it would earn the full Tier-S w=1.0 on 2 of 8
    #     quarters instead of w x 2/8.  That is the same mechanism as the >4-sigma ejection
    #     this project removed, pointed the other way.  An all-NaN column would also poison
    #     tempscore and silently drop the name.
    #
    # The genuine complaint behind "but loss-making cash-generative firms get punished"
    # lives in uIncomeQuality (a ratio whose denominator changes sign), which is fixed
    # separately -- not here.  And this is metric-dependent by design but NEVER
    # TIER-dependent: a tier is a weight, not a semantic, and tier-conditional NaN
    # behaviour would make a name's missingness penalty move at the next weight re-fit.
    resvec[met] = [w if Sign * val > 0 else 0 for val in testvec]
    res = resvec[met].head(n).mean()

    if nan_sink is not None:
        try:
            _win = pd.to_numeric(pd.Series(list(testvec)).head(n), errors='coerce')
            if len(_win) == 0 or _win.isna().all():
                nan_sink.append((met, w))
        except Exception:
            pass

    return res


def getAves2(df):
    print('Getting average values')
    # Ensure 'date' exists and is datetime
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            pass

    # Work on numeric columns only for median calculations
    without_source = df.drop(columns=['source'], errors='ignore')
    # For full median across the dataset (numeric columns only)
    res_fullMean = without_source.select_dtypes(include=[float, int]).median(numeric_only=True)

    # Per-date medians (group by date) — use numeric columns only
    if 'date' in without_source.columns:
        res_withDates = without_source.groupby('date').median(numeric_only=True)
        res_withDates = res_withDates.iloc[::-1].reset_index()
    else:
        res_withDates = pd.DataFrame()

    colslost = set(df.columns) - set(res_fullMean.index)

    meandic = {'BoMetric_ave': res_fullMean, 'BoMetric_dateAve': res_withDates, 'colslost': colslost}
    return meandic

