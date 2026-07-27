import pandas as pd
import requests
import calcMetrics as cm
import json
from tqdm import tqdm
import numpy as np
import warnings
import sys
import traceback
import createDicts as cdic
import getData_gen as gdg
import failTests as ft
import utils as utils
import reporting_period as rp
from datetime import datetime


def get_fundamentals_fmp(Tickers_df, cdx_df, BoMetric_df, baseurl,
                         api_key,compyear, n=1, nrTaT=-1, startindex=0,period='quarter',limit=44):
    print('Fetching financial data from FMP and calculating relevant metrics.')
    if not isinstance(Tickers_df, pd.DataFrame):
        raise Exception('provide a DataFrame')
    if period == 'quarter' and limit < 16:
        raise Exception('Number of periods, if periods are quarters, must be larger than 16')
    tickersfailed = []
    lenfail = []
    datefail = []
    pricefail = []
    pricefailESN = []
    # tickers whose per-ticker PARSE/COMPUTE step raised (see the guard in the loop).
    # First-class completeness artifact, same as the other fail buckets.
    parsefail = []
    emptyfail = []
    hasCurrentYear = []
    if nrTaT < 0 and startindex == 0:
        pbar = tqdm(total=len(Tickers_df))
    elif nrTaT < 0 and startindex > 0:
        pbar = tqdm(total=len(Tickers_df)-startindex)
    else:
        total = min(nrTaT,len(Tickers_df)-startindex)
        pbar = tqdm(total=total)
    cntr = 0
    Tickers_df = Tickers_df.iloc[startindex: ,:]
    preReq_dict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, \
        BoMetric_unity_dict, BoMetric_special_dict = cdic.getDicts()
    for row in Tickers_df.itertuples():
        cntr = cntr + 1
        ticker = row.symbol

        km, fr, inc, bs, cf, tickersfailed, lenfail, datefail, emptyfail = getFsData_fmp(ticker, period, limit,baseurl,
                                                                                         api_key, compyear, tickersfailed,
                                                                                         lenfail, datefail,emptyfail)
        if not (isinstance(km, int) and km == -37707):
            # PER-TICKER GUARD (review B1, 2026-07-26).  One malformed ticker must cost
            # THAT TICKER, not the whole ~12-hour fetch.  This loop body had no
            # try/except while Sbocker re-raises, so any exception in it -- the price
            # length ValueError that prompted this, a ragged statement, an unexpected
            # dtype -- killed the run outright.  failTests was hardened against exactly
            # this class of loss ("one unlucky throttled ticker must cost that ticker,
            # not the run") and the caller that invokes it was left unguarded.
            # The ticker is recorded in tickersfailed/parsefail so it shows up in the
            # run's completeness counters instead of vanishing.
            try:
                tempfund, tempMetric_df = initTempMets(BoMetric_df.columns, cdx_df.columns,
                                                                                       bs['date'], ticker)

                tempfund, hcy = fillPreReqdf(tempfund, preReq_dict, bs, inc, cf, km, fr)
                # READ the one classification stamped by fillPreReqdf (review item 9).
                # It must NOT re-classify: by this point tempfund['date'] has been SNAPPED
                # to quarter-starts, which can turn a semi-annual cadence into a
                # quarterly-looking one, and this site and the Graham site would then
                # disagree for the SAME ticker.
                _rpy_t = rp.rows_per_year(
                    tempfund[rp.FREQ_COLUMN].iloc[0]
                    if rp.FREQ_COLUMN in tempfund.columns else rp.UNKNOWN)
                tempMetric_df = utils.setDatesToQuarterly(tempMetric_df)
                if hcy == 1:
                    hasCurrentYear.append(ticker)

                if not gdg.checkIfValidFS(tempfund):
                    tickersfailed.append(ticker)
                    pricefail.append(ticker)
                    pricefailESN.append(row.exchangeShortName)
                else:
                    tempMetric_df_trimmed = build_bometric_rows(
                        tempfund, tempMetric_df, _rpy_t, n=n,
                        dicts=(BoMetric_base_dict, BoMetric_mean_dict,
                               BoMetric_unity_dict, BoMetric_diff_dict,
                               BoMetric_special_dict))

                    # align schemas (preserve all columns) before concatenation to avoid losing columns
                    cols_union = BoMetric_df.columns.union(tempMetric_df_trimmed.columns)
                    BoMetric_df = BoMetric_df.reindex(columns=cols_union)
                    tempMetric_df_trimmed = tempMetric_df_trimmed.reindex(columns=cols_union)
                    # perform concat while suppressing the specific FutureWarning about
                    # concatenation with empty / all-NA entries (make this local only)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=FutureWarning,
                            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
                        )
                        BoMetric_df = pd.concat([BoMetric_df, tempMetric_df_trimmed], ignore_index=True)
                    # align schemas for cdx as well
                    cols_union_cdx = cdx_df.columns.union(tempfund.columns)
                    cdx_df = cdx_df.reindex(columns=cols_union_cdx)
                    tempfund = tempfund.reindex(columns=cols_union_cdx)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=FutureWarning,
                            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
                        )
                        cdx_df = pd.concat([cdx_df, tempfund], ignore_index=True)
            except Exception as _tick_err:
                tickersfailed.append(ticker)
                parsefail.append(ticker)
                print("PARSE-FAIL %s: %s: %s -- ticker SKIPPED, run continues."
                      % (ticker, type(_tick_err).__name__, _tick_err), flush=True)
                traceback.print_exc(file=sys.stdout)

        if nrTaT > 0 and cntr == nrTaT:
            break
        elif len(tickersfailed) > (cntr + 1)*20:
            break
        pbar.update(n=1)
    pbar.close()

    #BoMetric_df = utils.setDatesToQuarterly(BoMetric_df)
    BoMetric_df, cdx_df = gdg.fixAfterGetData(BoMetric_df, cdx_df)

    # Materialize a USD market-cap column from the just-captured reportedCurrency, so the
    # saved cdx_df is self-describing for market-cap banding (carveOut.MCAP_BANDS). This
    # is the SAME shared FX path the band selection / grading use, so nothing can drift.
    # Guarded + best-effort: never blocks the fetch. No-op (all-NaN) on data lacking
    # reportedCurrency -- consumers then read every name as unknown-mcap (never misbanded).
    try:
        import carveOut as _co
        cdx_df['marketCap_usd'] = _co.marketcap_usd_series(cdx_df).values
    except Exception as _e:
        print(f'WARNING: marketCap_usd materialization skipped ({type(_e).__name__}: {_e})')

    if parsefail:
        print('PARSE-FAIL SUMMARY: %d ticker(s) skipped by the per-ticker guard: %s'
              % (len(parsefail), ', '.join(map(str, parsefail))), flush=True)
    # GRAHAM-UNDEFINED INCIDENCE (ruling Q1.3).  Printed every run so the Tier-S w=1.0
    # gate's NaN population is a measured number rather than a surprise in review.
    try:
        if 'grahamUndefinedReason' in cdx_df.columns:
            _gr = cdx_df['grahamUndefinedReason'].fillna('')
            _tot = len(_gr)
            _und = int((_gr != '').sum())
            print('GRAHAM UNDEFINED: %d of %d rows (%.1f%%) score the Tier-S '
                  'grahamNumberToPrice criterion as a FAIL because the value is undefined; '
                  'by reason: %s'
                  % (_und, _tot, 100.0 * _und / max(1, _tot),
                     _gr[_gr != ''].value_counts().to_dict()), flush=True)
    except Exception as _e:
        print('WARNING: graham-undefined summary skipped (%s)' % _e, flush=True)
    resfunddic = {'BoMetric_df':BoMetric_df,
                  'cdx_df': cdx_df, 'tickersfailed': tickersfailed, 'lenfail': lenfail, 'pricefail': pricefail,
                  'datefail': datefail, 'emptyfail': emptyfail, 'parsefail': parsefail,
                  'cind': cntr, 'hasCurrentYear': hasCurrentYear}
    return resfunddic

def _align_statements_by_date(bs, inc, cf, km, fr):
    """R-E cross-statement date-join (design R-E, getData_fmp.py:48).

    The old fillPreReqdf assigned bs/inc/cf/km/fr columns by RangeIndex POSITION.
    Deep history makes ragged statement lengths (a missing/extra quarter in one
    statement) more likely, and positional assignment then MISPAIRS periods across
    statements, corrupting every cross-statement ratio.  This re-aligns each
    statement to the balance-sheet reference dates by ACTUAL date.

    IDENTITY GUARANTEE (bit-for-bit, behaviour-preserving on the common case): when
    every statement carries the SAME, duplicate-free, identically-ordered date
    vector as bs -- which is the well-formed case for essentially all tickers at the
    current 24-quarter depth -- reindexing to bs['date'] returns each row in the same
    position as the old positional code, so tempfund is byte-identical.  The join
    only DIFFERS when statements are ragged (the bug it fixes).

    SAFETY FALLBACK: if any statement (incl. bs) has duplicate or unusable dates,
    date alignment is ambiguous -> fall back to the original POSITIONAL behaviour for
    that ticker (never worse than today, and the row count is never changed).

    Returns (aligned_dict, used_date_join: bool).  aligned_dict maps
    'bs'/'inc'/'cf'/'km'/'fr' -> a frame reindexed to bs['date'] (or the raw frame on
    fallback).
    """
    stmts = {'bs': bs, 'inc': inc, 'cf': cf, 'km': km, 'fr': fr}
    ref_dates = pd.to_datetime(bs['date'], errors='coerce')
    # unusable if bs dates are non-unique or all-NaT -> positional fallback
    if ref_dates.isna().all() or ref_dates.duplicated().any():
        return stmts, False
    aligned = {}
    for name, sdf in stmts.items():
        if 'date' not in sdf.columns:
            return stmts, False
        s = sdf.copy()
        s_dates = pd.to_datetime(s['date'], errors='coerce')
        if s_dates.duplicated().any():
            # ambiguous mapping for this statement -> positional fallback (whole ticker)
            return stmts, False
        s.index = s_dates
        aligned[name] = s.reindex(ref_dates)
    return aligned, True


def fillPreReqdf(tempfund,preReq_dict,bs,inc,cf,km,fr):
    hcybool = False
    aligned, used_join = _align_statements_by_date(bs, inc, cf, km, fr)
    for key1 in preReq_dict:
        for i in preReq_dict[key1]:
            if key1 == 'bs':
                tempfund[i] = aligned['bs'][i].values if used_join else bs[i]
            elif key1 == 'inc':
                tempfund[i] = aligned['inc'][i].values if used_join else inc[i]
            elif key1 == 'cf':
                tempfund[i] = aligned['cf'][i].values if used_join else cf[i]
            elif key1 == 'km':
                tempfund[i] = aligned['km'][i].values if used_join else km[i]
            elif key1 == 'fr':
                tempfund[i] = aligned['fr'][i].values if used_join else fr[i]
            else:
                #tempfund['shares'] = inc['revenue'] / km['revenuePerShare']
                # PRICE = marketCap / weightedAverageShsOut  (audit C-2 fix, 2026-07-19).
                #
                # It used to be derived as quarterly-PE * quarterly-EPS:
                #   fr['priceEarningsRatio'] * (inc['netIncome'] / inc['weightedAverageShsOut'])
                # but FMP's QUARTERLY priceEarningsRatio is ANNUALISED (price / TTM-ish EPS)
                # while the EPS factor here is a SINGLE quarter, so the product came out at
                # ~1/4 of the real share price.  Proven on the 2026-07-17 panel:
                # marketCap / (price * shares) had median 3.99992 and 69% of all 176,193
                # usable rows inside +-1% of exactly 4.0.
                #
                # The damage was NOT the uniform scale (z-scored metrics are invariant to
                # it) but the Stage-1 Tier-S UNITY test grahamNumber/price > 1, which is
                # scale-SENSITIVE: on the same panel it passed 70.6% of rows on the divided
                # price vs 13.4% on marketCap/shares -- a weight-1.0 criterion that had
                # degenerated into "almost everyone passes".
                #
                # marketCap and weightedAverageShsOut are both already fetched, are in the
                # company's own reporting currency (so the ratio is currency-consistent) and
                # are both as-of the statement period end (verified: 24 distinct marketCap
                # values per source, no lookahead), which is exactly the as-of convention
                # every price-based metric here assumes.
                #
                # +-inf (shares == 0) is normalised to NaN: an undefined price must read as
                # MISSING to checkIfValidFS, which runs on tempfund before forceNumOnDf's
                # inf->NaN sweep.
                #
                # LENGTH SAFETY (review B1, 2026-07-26 -- this crashed the fetch).  On the
                # POSITIONAL-FALLBACK path km/inc are the raw frames, so `km[..] / inc[..]`
                # is an INDEX-ALIGNED Series over the UNION of their indices.  Assigning
                # `.values` from that into a column of len(bs) raises
                #   ValueError: Length of values (7) does not match length of index (6)
                # whenever a statement is longer than the balance sheet -- i.e. exactly the
                # ragged-length case `_align_statements_by_date` exists for, on exactly the
                # path it falls back to.  Both triggers are documented as real in this file
                # (282 sources carry duplicate raw quarters).  The pre-fix line was a plain
                # Series assignment and could not raise; the item-1 price fix turned it into
                # a length assignment.  Reindex onto tempfund's own index instead: aligned
                # by position on the fallback (RangeIndex), correct on the join path, and it
                # yields NaN rather than an exception where a statement is short.
                if used_join:
                    _km, _inc = aligned['km'], aligned['inc']
                    _price = pd.Series(_km['marketCap'].values
                                       / _inc['weightedAverageShsOut'].values,
                                       index=tempfund.index)
                else:
                    _price = (pd.to_numeric(km['marketCap'], errors='coerce')
                              / pd.to_numeric(inc['weightedAverageShsOut'],
                                              errors='coerce')).reindex(tempfund.index)
                tempfund['price'] = _price.replace([np.inf, -np.inf], np.nan)

    tempfund = stamp_frequency_and_graham(tempfund)

    # Keep the RAW fiscal period-end date BEFORE setDatesToQuarterly overwrites `date`
    # with a quarter-start stamp (audit H-2 fix, 2026-07-19).  `date` is deliberately left
    # exactly as it is -- every downstream consumer (the cross-statement date join, the
    # forensic YoY shifts, CycleHeat's restatement tie-break, data_quality's row matching)
    # keys off the quarterly stamp -- so this is ADDITIVE.  The quarter stamp is lossy in
    # the two ways that matter: 52/53-week fiscal drift collapses two different period ends
    # onto ONE quarter (282 sources carry duplicate quarters), and a fiscal year that does
    # not align to calendar quarters cannot be recovered from it afterwards.
    tempfund['periodEndDate'] = tempfund['date'].values
    tempfund = utils.setDatesToQuarterly(tempfund)
    if tempfund['date'].iloc[0].year == datetime.today().year:
        hcybool = True

    return tempfund, hcybool


def build_bometric_rows(tempfund, tempMetric_df, rpy, n=1, dicts=None):
    """Build ONE source's Stage-1 BoMetric rows from its cdx-schema frame.

    EXTRACTED from get_fundamentals_fmp's per-ticker body (2026-07-27) with NO
    behavioural change.  It was previously replicated in
    baseline_tools/dead_merge._build_entity_frames -- a duplication that module's own
    docstring names as its standing drift risk and explicitly proposes extracting.
    The offline panel-upgrade path needed the same loop a THIRD time, which made the
    extraction the cheaper option.  dead_merge and panel_upgrade now both call this.

    CONTRACT:
      * `tempfund` is NEWEST-FIRST (raw FMP statement order) with a positional index:
        `calc_diff`'s shift(-1) is "one period OLDER" and its rolling mean runs on the
        reversed series, and the tail() trim below drops the OLDEST rows.  Feeding an
        oldest-first frame silently inverts every diff metric.
      * `rpy` is THIS source's rows-per-year, read from the ONE classification stamped
        by fillPreReqdf (never re-derived from snapped dates -- review item 9).
      * `tempMetric_df` is the pre-initialised destination frame (initTempMets output,
        already date-snapped), carrying `date` + `source`.

    Returns the TRIMMED frame (oldest `rpy` rows dropped).
    """
    if dicts is None:
        (_preReq, _calc, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict,
         BoMetric_unity_dict, BoMetric_special_dict) = cdic.getDicts()
    else:
        (BoMetric_base_dict, BoMetric_mean_dict, BoMetric_unity_dict,
         BoMetric_diff_dict, BoMetric_special_dict) = dicts

    tempdf = pd.DataFrame()
    tempdf['date'] = tempfund['date']
    # need to lag denominator for Assets, Investment and such [determined before t]

    ratioOpCalcDicts = {**BoMetric_base_dict, **BoMetric_mean_dict, **BoMetric_unity_dict, **BoMetric_diff_dict}
    for key in ratioOpCalcDicts:
        restr = key
        strUp = ratioOpCalcDicts[key]['Upper']
        strDn = ratioOpCalcDicts[key]['Lower']
        tf = cm.calc_simpleRatio(tempfund, strUp, strDn)
        # FLOW-SCALE CORRECTION (specialist annualization ruling, 2026-07-25).
        # A semi-annual row's flow covers 6 months, so any flow/stock or
        # stock/flow Stage-1 ratio reads ~2x (or ~0.5x) purely from the
        # reporting convention.  reporting_period decides which keys are
        # affected, which LEG the flow is on, and whether the test's threshold
        # is absolute (true annualisation) or scale-free (per-quarter basis).
        # Factor is exactly 1.0 for every unaffected key and for quarterly
        # names on the scale-free keys -> no-op.
        _ff = rp.stage1_flow_factor(key, rpy)
        if _ff != 1.0:
            tf = [(v * _ff) if v is not None else v for v in tf]
        if key in BoMetric_base_dict:
            tempMetric_df[restr] = tf
        if key in BoMetric_mean_dict:
            mrestr = "m" + restr[0].upper() + restr[1:]
            tempMetric_df[mrestr] = tf
        if key in BoMetric_unity_dict:
            urestr = "u" + restr[0].upper() + restr[1:]
            tempMetric_df[urestr] = tf
        if key in BoMetric_diff_dict:
            tempdf['forDiff'] = tf
            tf = cm.calc_diff(tempdf, 'forDiff', n, rpy=rpy)
            drestr = "d" + restr[0].upper() + restr[1:]
            tempMetric_df[drestr] = tf

    for key1 in BoMetric_special_dict.keys():
        tf = cm.calc_special(tempfund, key1, n, rpy=rpy)
        tempMetric_df[key1] = tf

    # Drop the OLDEST rows whose YoY/diff windows have no counterpart:
    # `rpy` of them, not a fixed 4 (a semi-annual filer would otherwise
    # lose two full years of history).  Row-based site NOT on the audit's
    # list -- found in the 2026-07-25 sweep.
    return tempMetric_df.drop(tempMetric_df.tail(rpy).index)


def stamp_frequency_and_graham(tempfund):
    """Stamp `reportingFrequency` and the in-pipeline `grahamNumber` (+ undefined reason).

    EXTRACTED from fillPreReqdf (2026-07-27) with NO behavioural change, so the OFFLINE
    panel-upgrade path (baseline_tools/panel_upgrade.py, which rebuilds Stage-1 from a
    SAVED cdx_df rather than from raw statements) executes THIS function rather than a
    replica of it.  The dead-merge module already had to replicate the Stage-1 metric loop
    and that replication is the module's own stated drift risk; this is the same class of
    risk and it is avoidable here, so it is avoided.

    CONTRACT (unchanged from the in-line version):
      * `tempfund` is NEWEST-FIRST -- the raw FMP statement order.  Both the TTM rolling
        sum and `date`-cadence classification depend on it.
      * `tempfund['date']` still holds RAW period-end dates at the live call site
        (setDatesToQuarterly runs AFTER).  The offline caller can only offer SNAPPED
        dates, which is a documented approximation, not a different code path.
      * index must be positional/aligned with the frame's own rows (the live frame is
        freshly built, so RangeIndex).
    Returns the same frame, mutated in place and returned for call-site clarity.
    """
    # GRAHAM NUMBER, computed in-pipeline (review H2 fix, 2026-07-25).
    #
    # FMP's quarterly `grahamNumber` is sqrt(22.5 * EPS_QUARTERLY * BVPS), i.e. HALF the
    # published sqrt(22.5 * EPS_ANNUAL * BVPS) -- proven on the 2026-07-17 panel:
    #   median( FMP graham / sqrt(22.5 * netIncomePerShare_q * bookValuePerShare) )
    #     = 1.0000 with 79.5% of 110,264 rows inside 1%, versus 0.5000 against the
    #     4x-EPS (annualised) form.
    # With `price` fixed to the real share price, the weight-1.0 Tier-S UNITY test
    # `grahamNumber/price > 1` therefore went from 2x too LOOSE (70.6% pass on the old
    # divided price) to 2x too STRICT (13.4%); calibrated is ~42-43%.  Rescaling FMP's
    # field would work numerically, but computing the number outright is the honest fix
    # and removes the dependency on an undocumented FMP convention that could change.
    #
    # EPS_ttm = netIncome_ttm / weightedAverageShsOut(current row), NOT the sum of four
    # quarterly netIncomePerShare values: each quarter's per-share figure uses its OWN
    # share count, so summing them mixes share bases, whereas one TTM earnings total over
    # the current share count is a single consistent basis -- and it is the SAME basis as
    # `price` (marketCap/weightedAverageShsOut) that this ratio is compared against.
    #
    # TTM sums are taken over the SAME set of rows for both inputs (the ttm_aligned_sums
    # convention): a row's TTM is NaN unless all 4 of its trailing quarters are present,
    # so a gap yields "not computable" rather than a 3-quarter sum masquerading as a year.
    #
    # Graham is UNDEFINED for negative earnings or negative book value (the sqrt has no
    # real root and the screen is a value floor for profitable, asset-backed firms), so
    # EPS_ttm <= 0 or BVPS <= 0 -> NaN.  NaN, not 0: Stage-1 scores NaN as a FAIL of this
    # criterion, which is the correct reading of "no Graham floor exists here"; a 0 would
    # be a real computed value that happens to fail.
    #
    # SEMI-ANNUAL -- RESOLVED (2026-07-25): the trailing window is `rpy` rows, so an
    # H1/H2 filer sums its 2 rows for a true 12-month EPS instead of 4 rows / 24 months.
    # THIS TICKER's reporting frequency, decided at ingest.  `period` is authoritative
    # and is present from this fetch onward (fix 14); the date-cadence fallback covers
    # a frame without it.  A semi-annual filer's 2 rows already span 12 months, so its
    # EPS_ttm must sum 2 rows, not 4 -- summing 4 built a 24-month 'annual' EPS and
    # inflated its Graham number by ~sqrt(2), which is exactly the N1 over-pass.
    # ============ THE ONE CLASSIFICATION (review item 9, 2026-07-26) ============
    # Decided HERE and nowhere else, because this is the earliest point where both best
    # signals exist: tempfund['date'] still holds the RAW period-end dates
    # (utils.setDatesToQuarterly has not run yet -- it is further down this function) and
    # `period` is the authoritative fiscal label straight off the income statement.
    #
    # It is STAMPED INTO THE FRAME (rp.FREQ_COLUMN) so every downstream consumer READS this
    # verdict instead of deriving its own: Stage-1, Stage-2, the forensics, the moat,
    # reviewReference and the PIT reproduction all go through rp.frequency_by_source, which
    # prefers this column.  Three sites used to decide independently and two of them could
    # DISAGREE -- snapping moves a period end by up to ~92 days, so a semi-annual filer can
    # look quarterly once snapped (282 sources carry colliding snapped quarters).
    _freq_t = rp.classify_source(
        dates=tempfund['date'] if 'date' in tempfund.columns else None,
        period_values=(list(tempfund['period']) if 'period' in tempfund.columns
                       else None))
    tempfund[rp.FREQ_COLUMN] = _freq_t
    _rpy = rp.rows_per_year(_freq_t)
    _ni = pd.to_numeric(tempfund.get('netIncome'), errors='coerce')
    _sh = pd.to_numeric(tempfund.get('weightedAverageShsOut'), errors='coerce')
    _bvps = pd.to_numeric(tempfund.get('bookValuePerShare'), errors='coerce')
    # tempfund is NEWEST-FIRST here (raw FMP order), so a forward-looking rolling sum on
    # the reversed series gives each row the sum of ITSELF plus the (rpy-1) older rows.
    _pair = pd.concat([_ni, _sh], axis=1)
    _pair = _pair.where(_pair.notna().all(axis=1))       # aligned rows only
    _ni_ttm = _pair.iloc[::-1, 0].rolling(int(_rpy)).sum().iloc[::-1]
    _eps_ttm = _ni_ttm / _sh
    _graham = np.sqrt(22.5 * _eps_ttm.where(_eps_ttm > 0)
                      * _bvps.where(_bvps > 0))
    tempfund['grahamNumber'] = _graham.replace([np.inf, -np.inf], np.nan).values
    # WHY each undefined Graham row is undefined (ruling Q1.3, 2026-07-26).  Graham stays a
    # NaN -> Stage-1 FAIL, but the fail now carries a REASON, per row, so the deferred CEO
    # question -- should that Tier-S w=1.0 slot be cheapness, or the
    # profitability AND asset-backing AND cheapness conjunction it actually is? -- is
    # answerable from run incidence instead of by argument.  Order matters: missing inputs
    # are reported as missing, not mislabelled as negative.
    _reason = pd.Series('', index=tempfund.index, dtype=object)
    _defined = _graham.notna()
    _missing = (~_defined) & (_eps_ttm.isna() | _bvps.isna())
    _neg_eps = (~_defined) & (~_missing) & (_eps_ttm <= 0)
    _neg_bv = (~_defined) & (~_missing) & (~_neg_eps) & (_bvps <= 0)
    _reason[_missing.fillna(False).values] = 'graham_missing_inputs'
    _reason[_neg_eps.fillna(False).values] = 'graham_undefined_negative_eps'
    _reason[_neg_bv.fillna(False).values] = 'graham_undefined_negative_bv'
    tempfund['grahamUndefinedReason'] = _reason.values

    return tempfund

def getFsData_fmp(ticker, period, limit, baseurl, api_key,compyear, tickersfailed, lenfail,datefail,emptyfail,
                  dead_path=False, http_get=None):
    """Fetch the 5 statement endpoints for one ticker and apply the gates.

    dead_path : forwarded to testForAPIFaults_fmp -- on the DELISTED-ENTITY
        ingestion path it BYPASSES the datefail gate (F-A) and RELAXES the >=16q
        lenfail gate (F-B) so dead names are not silently dropped.  Default False
        keeps the live path bit-for-bit.
    http_get : optional injected HTTP getter for offline testing.  When None (the
        LIVE overnight fetch path), it defaults to gdg.safe_http_get -- a bounded
        (timeout + retry/backoff) getter -- so a single stalled/hung FMP endpoint
        cannot hang the ~12h run.  This is BEHAVIOUR-PRESERVING on the happy path:
        safe_http_get returns the SAME requests.Response for a healthy 200 (only a
        10s timeout is added, which a healthy endpoint never trips), so parsed
        fundamentals are byte-identical to the old bare requests.get.  On a
        persistent timeout/connection-error or a retryable 5xx/429 that survives
        retries, safe_http_get hands back a FAILING Response (status_code in the
        400-599 failcodes, or a _FailedResponse(599)); the existing failcode gate
        then records the ticker as a normal fetch failure (tickersfailed) and the
        loop CONTINUES to the next ticker -- it never raises/aborts the run.
    """
    if http_get is None:
        http_get = gdg.safe_http_get
    failcodes = list(range(400, 600))
    failbool, whyfail, outdic = ft.testForAPIFaults_fmp(failcodes,compyear,ticker,period,limit,baseurl,api_key,
                                                        dead_path=dead_path, http_get=http_get)
    if failbool:
        tickersfailed.append(ticker)
        if whyfail == 'datefail':
            datefail.append(ticker)
        elif whyfail == 'lenfail':
            lenfail.append(ticker)
        elif whyfail == 'emptyfail':
            emptyfail.append(ticker)
        km, fr, inc, bs, cf = -37707, -1, -1, -1, -1
    else:
        km = outdic['km'] #pd.DataFrame.from_dict(resp_km.json())
        fr = outdic['fr'] #pd.DataFrame.from_dict(resp_fr.json())
        inc = outdic['inc'] #pd.DataFrame.from_dict(resp_inc.json())
        bs = outdic['bs'] #pd.DataFrame.from_dict(resp_bs.json())
        cf = outdic['cf'] #pd.DataFrame.from_dict(resp_cf.json())

    return km, fr, inc, bs, cf, tickersfailed, lenfail, datefail, emptyfail

def symbchRestock(tckrs_df,baseurl,period,limit,api_key,compyear,timdir='old2new'):
    symbch_df = pd.DataFrame(requests.get(f'https://financialmodelingprep.com/api/v4/symbol_change?apikey={api_key}').json())
    if timdir == 'new2old':
        int = list(set(symbch_df.newSymbol) & set(tckrs_df.symbol))
    else:
        int = list(set(symbch_df.oldSymbol) & set(tckrs_df.symbol))

    succbool_lvl2 = []
    failcodes = list(range(400, 600))
    failers = []
    failers_lvl2 = []
    succnotadded = []
    succ_lvl2 = []
    failstosucc = []

    print(f'Starting symbol restock: {timdir}')
    pbar = tqdm(total=len(int))
    for ticker in int:
        failbool_lvl2_agg = False
        resp_km = requests.get(f'{baseurl}/key-metrics/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_fr = requests.get(f'{baseurl}/ratios/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_inc = requests.get(f'{baseurl}/income-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_bs = requests.get(f'{baseurl}/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_cf = requests.get(f'{baseurl}/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        respstatcodes = [resp_km.status_code, resp_fr.status_code, resp_inc.status_code, resp_bs.status_code,
                         resp_cf.status_code]
        failbool, whyfail = ft.testForAPIFaults(respstatcodes, failcodes,compyear, resp_km, resp_fr, resp_inc, resp_bs, resp_cf)
        if failbool:
            failers.append(ticker)
            if timdir == 'new2old':
                nt_df = symbch_df[symbch_df['newSymbol'] == ticker]['oldSymbol']
            else:
                nt_df = symbch_df[symbch_df['oldSymbol'] == ticker]['newSymbol']
            for i in range(0, len(nt_df)):
                ticker_next = nt_df.iloc[i]
                resp_km = requests.get(f'{baseurl}/key-metrics/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_fr = requests.get(f'{baseurl}/ratios/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_inc = requests.get(f'{baseurl}/income-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_bs = requests.get(f'{baseurl}/balance-sheet-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_cf = requests.get(f'{baseurl}/cash-flow-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                respstatcodes = [resp_km.status_code, resp_fr.status_code, resp_inc.status_code, resp_bs.status_code,resp_cf.status_code]
                failbool_lvl2, whyfail = ft.testForAPIFaults(respstatcodes, failcodes,compyear, resp_km, resp_fr, resp_inc, resp_bs,resp_cf)
                if failbool_lvl2:
                    failbool_lvl2_agg = True
                    failers_lvl2.append(ticker_next)
                elif ticker_next not in list(tckrs_df['symbol']):
                    tckrs_df.loc[tckrs_df['symbol'] == ticker, 'symbol'] = ticker_next
                    failstosucc.append(ticker_next)
                else:
                    succnotadded.append(ticker_next)
            if failbool_lvl2_agg == False:
                succ_lvl2.append(ticker)
        pbar.update(n=1)
    pbar.close()
    fullfail = list(set(failers + failers_lvl2))
    tckrs_df_new = tckrs_df[~tckrs_df['symbol'].isin(fullfail)].reset_index(drop=True)

    if len(failers) > 0:
        pcfixed = len(succ_lvl2)/len(failers)
        pcnotadded = len(succnotadded)/len(failers)

    return tckrs_df_new, failers, failers_lvl2, succ_lvl2, succnotadded, failstosucc, pcfixed, pcnotadded

def initTempMets(dfcols,cdxcols,datevec,ticker):
    tempMetric_df = pd.DataFrame(columns=dfcols)
    tempfund = pd.DataFrame(columns=cdxcols)
    tempfund['date'] = datevec
    tempfund['source'] = ticker
    tempMetric_df['date'] = datevec
    tempMetric_df['source'] = ticker

    return tempfund, tempMetric_df