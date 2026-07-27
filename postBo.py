import calcScore as cs
import postBoRank as pbr
import reporting_period as rp
from detectManipulation import _toNewestFirst
import forensicFlags as ff
import pandas as pd
import requests
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings

# Suppress FutureWarning about DataFrame concatenation with empty/all-NA entries
warnings.filterwarnings('ignore', message='.*concatenation with empty or all-NA entries.*')

def postBoWrapper(dmdic, as_of=None):
    """Scoring orchestration.  as_of (default None) threads the point-in-time date D
    from Sbocker through Stage-1 (simpleScore_fromDict) and Stage-2
    (postBoScoreRanking).  as_of=None -> live behaviour, BIT-FOR-BIT unchanged."""
    import sys
    import numpy as np
    
    # Diagnostic: Check input data BEFORE any calculations
    print("\n" + "="*60, flush=True)
    print("DIAGNOSTIC: postBoWrapper input data check (BEFORE score calculation)", flush=True)
    print("="*60, flush=True)
    
    bmdf = dmdic['BoMetric_df']
    bmav = dmdic.get('BoMetric_ave', {})
    bmda = dmdic.get('BoMetric_dateAve', pd.DataFrame())
    cdx_df = dmdic.get('cdx_df', pd.DataFrame())
    n = dmdic.get('nrScorePeriods', 8)
    
    # Check BoMetric_df
    if bmdf.empty:
        print("ERROR: BoMetric_df is EMPTY!", flush=True)
    else:
        print(f"BoMetric_df shape: {bmdf.shape} (rows, columns)", flush=True)
        print(f"BoMetric_df unique sources: {bmdf['source'].nunique() if 'source' in bmdf.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in bmdf.columns:
            first_source = bmdf['source'].iloc[0]
            print(f"BoMetric_df first source: {first_source}", flush=True)
            # Show sample rows for first source
            first_source_data = bmdf[bmdf['source'] == first_source].head(3)
            print(f"Sample rows for {first_source} (first 3):", flush=True)
            numeric_cols_sample = first_source_data.select_dtypes(include=[np.number]).columns[:5]
            if len(numeric_cols_sample) > 0:
                print(first_source_data[['date', 'source'] + list(numeric_cols_sample)].to_string(), flush=True)
        # Check for NaN in numeric columns
        numeric_cols = bmdf.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            nan_pct = (bmdf[numeric_cols].isna().sum() / len(bmdf) * 100).round(1)
            print(f"\nBoMetric_df NaN percentage in numeric columns (first 5):", flush=True)
            for col, pct in list(nan_pct.head(5).items()):
                print(f"  {col}: {pct}%", flush=True)
    
    # Check cdx_df
    if cdx_df.empty:
        print("\nERROR: cdx_df is EMPTY!", flush=True)
    else:
        print(f"\ncdx_df shape: {cdx_df.shape} (rows, columns)", flush=True)
        print(f"cdx_df unique sources: {cdx_df['source'].nunique() if 'source' in cdx_df.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in cdx_df.columns:
            first_source = cdx_df['source'].iloc[0]
            print(f"cdx_df first source: {first_source}", flush=True)
            # Show sample rows for first source
            first_source_data = cdx_df[cdx_df['source'] == first_source].head(3)
            print(f"Sample rows for {first_source} (first 3):", flush=True)
            key_cols = ['date', 'source', 'marketCap', 'freeCashFlow', 'price', 'totalAssets']
            available_cols = [col for col in key_cols if col in first_source_data.columns]
            if len(available_cols) > 0:
                print(first_source_data[available_cols].to_string(), flush=True)
    
    # Check BoMetric_ave (could be dict, Series, or DataFrame)
    if bmav is None:
        print("\nWARNING: BoMetric_ave is None!", flush=True)
    elif isinstance(bmav, pd.DataFrame):
        if bmav.empty:
            print("\nWARNING: BoMetric_ave DataFrame is empty!", flush=True)
        else:
            print(f"\nBoMetric_ave DataFrame shape: {bmav.shape}", flush=True)
            print(f"BoMetric_ave columns: {list(bmav.columns[:5])}", flush=True)
    elif isinstance(bmav, pd.Series):
        if bmav.empty:
            print("\nWARNING: BoMetric_ave Series is empty!", flush=True)
        else:
            print(f"\nBoMetric_ave Series length: {len(bmav)}", flush=True)
            print(f"BoMetric_ave sample values (first 10):", flush=True)
            for idx, val in bmav.head(10).items():
                print(f"  {idx}: {val}", flush=True)
    elif isinstance(bmav, dict):
        if not bmav:
            print("\nWARNING: BoMetric_ave dict is empty!", flush=True)
        else:
            print(f"\nBoMetric_ave dict has {len(bmav)} keys", flush=True)
            print(f"BoMetric_ave sample values (first 10):", flush=True)
            for key, val in list(bmav.items())[:10]:
                print(f"  {key}: {val}", flush=True)
    else:
        print(f"\nBoMetric_ave type: {type(bmav)}", flush=True)
    
    print("="*60 + "\n", flush=True)
    sys.stdout.flush()
    
    # FREQUENCY MAP FROM cdx_df, not from BoMetric_df (review S3, 2026-07-26).
    # `period` is the authoritative frequency signal and it lands in cdx_df ONLY --
    # BoMetric_df never carries it by design (createDicts preReq fields do not reach
    # it).  Stage-1 was therefore the ONE consumer still deriving frequency from date
    # cadence while Stage-2, the forensics, the moat and reviewReference all used
    # `period`: two sources of truth for the same per-ticker decision, already
    # disagreeing on 2 names (KXIN, OBI.L) and set to diverge properly from the first
    # fetch that carries the field.  Passing the cdx-derived map makes Stage-1 read the
    # same answer as everything downstream.
    # THE UNIVERSE-WIDE FREQUENCY / CONFLICT BANNER (restored, ship-gate item 3, 2026-07-27).
    # verbose=True is LOAD-BEARING and independent of whether Stage-1 consumes the map:
    # this is the run's ONLY universe-wide frequency readout and the only place the
    # period-vs-cadence conflict list + CSV are emitted.  Both pre-ship reads call it "the
    # first thing to read after the fetch".  It got silenced when the map was threaded into
    # Stage-1 with verbose defaulted off, and the Q2 ruling then made Stage-1's use of the
    # map a no-op on the score -- so the wiring's only surviving effect had been to hide
    # the banner.  Keep the call and keep it loud.
    _freq_map = rp.frequency_by_source(dmdic.get('cdx_df'), verbose=True)
    BoScore_df = cs.simpleScore_fromDict(bmdf, bmav, bmda, n, as_of=as_of,
                                        freq_map=_freq_map)

    # --- Phase-1 cohort carve-out (BEFORE the head(100) selection) -------------
    # Partition the full BoScore-ranked universe into a GENERAL pool + three
    # disjoint side-cohorts (REITs / Mining / investment vehicles) and apply a
    # gentle $25M market-cap floor uniformly.  The MAIN shortlist is drawn ONLY
    # from the general pool; each cohort is ranked with the SAME machinery and
    # presented as a labeled side-list.  This only changes WHICH names feed each
    # ranking -- the ranking (and its as_of/live behaviour) is unchanged.
    #
    # ROBUSTNESS (critical path): the carve-out must NEVER be able to destroy the
    # main deliverable.  The partition is wrapped so a failure falls back to the
    # legacy BoScore_df.head(100) (original, no-carve behaviour); the MAIN ranking
    # + resdic are built FIRST, fully independent of the cohorts; and the
    # side-lists run afterwards in a per-cohort guarded best-effort block, so a
    # small/degenerate cohort (e.g. qcut on ~10 names) can crash at most its own
    # side-list -- never the general ranking that already succeeded.
    carve = None
    try:
        import carveOut as co
        carve = co.partition_universe(BoScore_df, cdx_df, dmdic.get('Tickers_df'),
                                      mcap_floor=25e6, cohort_head=25)
        general_scores = carve['general']
        gp_count = len(general_scores)
        diag = carve['diagnostics']
        print(f"CARVE-OUT: general pool = {gp_count} names after cohorts + $25M floor "
              f"(REIT={diag['n_REIT']}, Mining={diag['n_Mining']}, "
              f"FIN1_Vehicle={diag['n_InvestmentVehicle']}, "
              f"FIN2_Manager={diag.get('n_FinManager', 0)}, "
              f"FIN3_BalanceSheet={diag.get('n_BalanceSheetFin', 0)}, "
              f"below_floor={diag['n_below_floor']}, unknown_mcap_kept={diag['n_unknown_mcap']})",
              flush=True)
        if gp_count < 100:
            print(f"CARVE-OUT WARNING: general pool has only {gp_count} names (<100); "
                  f"top-100 selection is short -- top-20 may not fully fill.", flush=True)
    except Exception as e:
        # LOUD FALLBACK. The carve-out/dedup IS the deliverable's integrity guarantee
        # (no issuer duplicates; 0 Basic-Materials / financials in the general pool). A
        # SILENT fallback is itself a defect: on 2026-07-13 it shipped legacy un-carved
        # output that looked like a carved deliverable and nobody noticed. Keep the
        # safety net (never crash the pipeline) but make a fallback IMPOSSIBLE to mistake
        # for success: full traceback + an unmistakable banner on BOTH stdout and stderr.
        import traceback
        banner = (
            "\n" + "!" * 78 + "\n"
            "!!! CARVE-OUT/DEDUP DID NOT RUN -- SHIPPING LEGACY UN-CARVED TOP-100 !!!\n"
            "!!! The general pool is NEITHER de-duped NOR sector-carved this run:  !!!\n"
            "!!!   expect issuer/share-class duplicates AND Basic-Materials /       !!!\n"
            "!!!   financials leaking into the general top-100.                     !!!\n"
            f"!!! Cause: {type(e).__name__}: {e}\n"
            "!!! DO NOT treat this output as a carved deliverable.                  !!!\n"
            + "!" * 78 + "\n")
        # stderr first (survives stdout redirection and tqdm progress-bar noise).
        print(banner, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(banner, flush=True)
        traceback.print_exc(file=sys.stdout)
        carve = None
        general_scores = BoScore_df
        gp_count = len(general_scores)

    # --- MAIN ranking + resdic: built first, independent of the cohorts --------
    BoS_dftop100 = general_scores.head(100)
    BoM_dftop100 = bmdf[bmdf['source'].isin(list(BoS_dftop100.source))].reset_index(drop=True)
    cdx_dftop100 = cdx_df[cdx_df['source'].isin(list(BoS_dftop100.source))].reset_index(drop=True)

    n= 16
    # issuer names for the emission-time issuer-dedup (edge B: name+shares). cdx-based
    # edges (A/C) work without names; passing them widens dual-listing coverage.
    _tdf = dmdic.get('Tickers_df')
    _names = (dict(zip(_tdf['symbol'], _tdf['name']))
              if _tdf is not None and 'symbol' in getattr(_tdf, 'columns', [])
                 and 'name' in getattr(_tdf, 'columns', []) else {})
    rankdic = pbr.postBoScoreRanking(BoM_dftop100, BoS_dftop100, cdx_dftop100, dmdic['baseurl'], dmdic['api_key'],
                                     dmdic['period'],n,as_of=as_of,names=_names)

    # UNWINNED FILTER: -1.5 z-score pass-filter on six metrics (earnYield, grahamNumberToPrice,
    # RoA, EPStoEPSmean, freeCashFlowYield, revenueGrowth). Computed and stored in resdic but
    # NOT wired into any shipped deliverable — the shortlist is built from resdic['postRank']
    # only, so psbrfilter currently filters zero names. Left in place per CEO decision
    # (2026-07-14) pending a future decision to either wire it in (would require a soundness
    # review of the -1.5 cutoff on these 6 metrics) or remove it.
    metricList = ['earnYield', 'grahamNumberToPrice', 'RoA', 'EPStoEPSmean', 'freeCashFlowYield', 'revenueGrowth']
    cutoff = 1.5
    psbrfilter = pbr.postBoRankingPassFilter(rankdic['postRank'],metricList,-cutoff,np.inf)

    regressMetricsOnROR(rankdic)

    resdic = {**rankdic, **{'BoS_dftop100': BoS_dftop100, 'BoM_dftop100': BoM_dftop100, 'cdx_dftop100': cdx_dftop100,
                          'BoScore_df': BoScore_df, 'psbrfilter': psbrfilter,  # NOT WIRED — see above comment
                          'general_pool_count': gp_count}}

    # --- Side-lists: guarded best-effort, AFTER resdic is complete -------------
    # Only runs if the partition succeeded.  Per-cohort try/except so one
    # degenerate cohort degrades to "no side-list for that cohort" (visible
    # warning) without touching the others or the already-built main output.
    carveout_sidelists = {}
    if carve is not None:
        for label, cohort_scores in carve['cohorts'].items():
            try:
                head = cohort_scores.head(25)
                if head.empty:
                    carveout_sidelists[label] = None
                    continue
                bm = bmdf[bmdf['source'].isin(list(head.source))].reset_index(drop=True)
                cd = cdx_df[cdx_df['source'].isin(list(head.source))].reset_index(drop=True)
                # per-cohort weight vector (general/main pool keeps the default)
                wov = co.COHORT_WEIGHTS.get(label)
                print(f"CARVE-OUT side-list '{label}': ranking {len(head)} names"
                      f"{' with per-cohort weights' if wov else ''}", flush=True)
                carveout_sidelists[label] = pbr.postBoScoreRanking(bm, head, cd, dmdic['baseurl'], dmdic['api_key'],
                                                                   dmdic['period'], n, as_of=as_of,
                                                                   weight_override=wov)
            except Exception as e:
                print(f"CARVE-OUT side-list '{label}' FAILED ({type(e).__name__}: {e}); "
                      f"skipping this side-list (main output unaffected).", flush=True)
                carveout_sidelists[label] = None
        resdic['carveout_sidelists'] = carveout_sidelists
        resdic['carveout_labels'] = carve['labels']
        resdic['carveout_diagnostics'] = carve['diagnostics']
        # FULL carved + size-floored membership per cohort (source lists, BEFORE the
        # head(25)/head(100) selection).  Data-only: consumed by the READ-ONLY
        # review-reference artifacts (reviewReference) to build cohort distribution
        # stats / percentiles over EVERY peer, not just the Stage-2-scored subset.
        # Never read back into scoring.
        resdic['carve_full_membership'] = {
            'general': list(general_scores['source']),
            **{lab: list(cs['source']) for lab, cs in carve['cohorts'].items()}}
    else:
        resdic['carveout_sidelists'] = {}
        resdic['carve_full_membership'] = None

    return resdic

def regressMetricsOnROR(rankdic):
    # 'rankOfRanks' -> 'rankOfRanks_diag' (postBoRank.ROR_COLUMN): the emitted column was
    # renamed to mark it a DIAGNOSTIC, not a competing ranking (audit M1).
    ror = pbr.ROR_COLUMN
    regressors = list(set(rankdic['postRank'].columns) - set([ror, 'rankOfRanks', 'AggScore', 'source']))
    regressant = [ror]
    df = rankdic['postRank']
    X = df[regressors]
    y = df[regressant]
    
    # Guard: skip regression if insufficient samples (need at least 1 sample to fit)
    if X.shape[0] < 1:
        print("Warning: Insufficient samples for regression (found 0 samples). Skipping regression.")
        return None
    
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    print("R-Squared:", r_squared)
    coef = model.coef_
    intercept = model.intercept_
    print("Coefficients:", coef)
    print("regressors:", regressors)
    print("Intercept:", intercept)
    print("coefreg:", tuple(zip(regressors, coef.tolist()[0])))
    return None

def writeResWrapper(resdic):
    ntopagg = resdic['ntopagg']
    ntopxlsx = resdic['ntopxlsx']
    fidag = datetime.today().strftime('%Y-%m-%d')
    fb_df = resdic['postRank']

    # --- market-cap band segmentation (ADDITIVE size axis over the general pool) ---
    # GROUP the existing general ranking (postRank) by USD market cap: General (>$300M)
    # -> top-20; Mid ($150-300M)/Small ($50-150M)/Micro (<$50M) -> top-5 each. No re-score,
    # no re-rank -- the ordering is only PARTITIONED. Best-effort + fully guarded so it can
    # NEVER touch the main deliverables. Degrades gracefully when reportedCurrency has not
    # yet flowed (currency_pending): sub-band CSVs are then SKIPPED rather than emitting
    # misbanded output (CEO 2026-07-17: nothing wrong ships now).
    marketcap_bands = None
    try:
        import carveOut as co
        _tdf_b = resdic.get('Tickers_df')
        _cols_b = getattr(_tdf_b, 'columns', [])
        _names_b = (dict(zip(_tdf_b['symbol'], _tdf_b['name']))
                    if _tdf_b is not None and 'symbol' in _cols_b and 'name' in _cols_b else {})
        marketcap_bands = co.partition_by_marketcap(fb_df, resdic.get('cdx_df'), _names_b)
        resdic['marketcap_bands'] = marketcap_bands['bands']
        resdic['marketcap_band_counts'] = marketcap_bands['band_counts']
        resdic['marketcap_currency_pending'] = marketcap_bands['currency_pending']
        resdic['marketcap_band_selective'] = marketcap_bands.get('band_selective', {})
        resdic['marketcap_band_note'] = marketcap_bands.get('band_note', {})
        _pend = marketcap_bands['currency_pending']
        print("MARKET-CAP BANDS: " + ("CURRENCY PENDING (sub-bands suppressed this run) -- " if _pend else "")
              + ", ".join(f"{lab}={marketcap_bands['band_counts'].get(lab, 0)}"
                          for lab, *_ in co.MCAP_BANDS)
              + f" (unknown_mcap->General={marketcap_bands['unknown_mcap']})", flush=True)
        # MIN-N: name the bands whose "top-N" selects nothing, so a zero-selectivity list
        # is never read as a shortlist (audit M5).
        _nonsel = [lab for lab, sel in (marketcap_bands.get('band_selective') or {}).items()
                   if not sel and marketcap_bands['band_counts'].get(lab, 0) > 0]
        if _nonsel:
            print("MARKET-CAP BANDS -- NOT A SELECTION (band size <= requested top-N, "
                  "every member is listed): "
                  + "; ".join(marketcap_bands['band_note'][lab] for lab in _nonsel),
                  flush=True)
    except Exception as _be:
        print(f"WARNING: market-cap banding skipped ({type(_be).__name__}: {_be}); "
              f"main deliverables unaffected.", flush=True)
        marketcap_bands = None

    mscore = resdic['SLmeanMscore']
    cscore = resdic['SLmeanCscore']
    baseurl = resdic['baseurl']
    api_key = resdic['api_key']
    tickerfilter = resdic['tickerfilter']
    datasource = resdic['datasource']
    years = 6

    # Build the per-name forensic-flag table ONCE (offline, no API calls) and route
    # it into every top-N output. Promotes the already-computed M/C forensic signals
    # + Sloan accruals + financial indicator from decoration to decision-support.
    # Covers ntopagg so both the CSV (ntopagg) and the presentation (ntopxlsx) can
    # index into it. FLAGS, NOT VERDICTS: nothing here drops a name.
    flag_df = ff.buildForensicFlagTable(resdic, ntopagg)
    fname_forensic = f'ForensicFlagsTop{ntopagg}-{fidag}_{datasource}_{tickerfilter}.csv'

    # create csv listing the ntopagg stocks. writeBoAggToCSV fetches the API
    # `sector` per name and cross-checks it against the pickle-derived financial
    # classification (ff.applySectorFallback), returning the reconciled flag_df.
    fname_AggScoretop = f'AggScoreTop{ntopagg}-{fidag}_{datasource}_{tickerfilter}.csv'
    flag_df = writeBoAggToCSV(fb_df, mscore, cscore, baseurl, api_key, ntopagg, fname_AggScoretop, flag_df)

    # Write the standalone forensic decision-support CSV AFTER the API-sector
    # cross-check, so it carries the reconciled (conservative) financial classification.
    ff.writeForensicFlagsCSV(flag_df, fname_forensic)
    print(f'Forensic-flag table written to: {fname_forensic}')

    # create presentation xlsx of ntopxlsx stocks. When currency data is present the
    # general top-N is drawn from the General band (>$300M); pending currency -> unchanged.
    fname_presentationtop= f'PresentationTop{ntopxlsx}-{fidag}_{datasource}_{tickerfilter}.xlsx'
    createPresentation(fb_df, mscore, cscore, baseurl, api_key, ntopxlsx, fname_presentationtop, years, flag_df,
                       bands=marketcap_bands)

    # Phase-1 carve-out: write each labeled side-list (REIT / Mining / investment
    # vehicles) as its own compact CSV alongside the main deliverables. Best-effort
    # and self-contained: never raises, and is a no-op when no side-lists are
    # present (e.g. an older resdic), so the main path is unchanged.
    sidelist_fnames = []
    try:
        sidelists = resdic.get('carveout_sidelists') or {}
        for label, sdic in sidelists.items():
            if not sdic or 'postRank' not in sdic:
                continue
            sl_df = sdic['postRank'].head(ntopagg).copy()
            keep = [c for c in ['source', 'AggScore', pbr.ROR_COLUMN] if c in sl_df.columns]
            fname_sidelist = f'SideList_{label}_Top{ntopagg}-{fidag}_{datasource}_{tickerfilter}.csv'
            sl_df[keep].to_csv(fname_sidelist, index=False)
            sidelist_fnames.append(fname_sidelist)
            print(f'Carve-out side-list written to: {fname_sidelist}')
    except Exception as _e:
        print(f'WARNING: carve-out side-list writing skipped ({_e})')

    # Market-cap band CSVs -- one compact CSV per band, MIRRORING the SideList block
    # above. General (>$300M) top-20 + Mid/Small/Micro top-5. Best-effort + self-
    # contained (never raises; no-op when banding was skipped or absent). SKIPPED
    # entirely when currency data is pending, so no misbanded file ever ships (the
    # General-band CSV would just duplicate the top-20 and the sub-bands would be
    # misbanded) -- the feature becomes correct automatically once reportedCurrency flows.
    band_fnames = []
    try:
        if marketcap_bands and not marketcap_bands.get('currency_pending', True):
            _sel_map = marketcap_bands.get('band_selective') or {}
            _note_map = marketcap_bands.get('band_note') or {}
            for label, band_df in (marketcap_bands.get('bands') or {}).items():
                if band_df is None or band_df.empty:
                    continue
                keep = [c for c in ['source', 'AggScore', pbr.ROR_COLUMN] if c in band_df.columns]
                out_band = band_df[keep].copy()
                # MIN-N LABEL (audit M5): carry the selectivity statement INTO the file --
                # a "top-5" over a 4-member band selected nothing, and a CSV read out of
                # context gives the reader no way to know that. Also marked in the filename
                # so it is visible before the file is even opened.
                _note = _note_map.get(label, '')
                out_band['band_selection'] = _note
                _selective = _sel_map.get(label, True)
                _marker = '' if _selective else '_ALLMEMBERS_NOT_A_SELECTION'
                fname_band = (f'MarketCapBand_{label}{_marker}-{fidag}_'
                              f'{datasource}_{tickerfilter}.csv')
                out_band.to_csv(fname_band, index=False)
                band_fnames.append(fname_band)
                print(f'Market-cap band CSV written to: {fname_band}'
                      + (f'   [{_note}]' if _note else ''))
        elif marketcap_bands and marketcap_bands.get('currency_pending', True):
            print('Market-cap band CSVs SKIPPED this run: reportedCurrency not yet in the '
                  'data (correct automatically from the next full fetch).', flush=True)
    except Exception as _e:
        print(f'WARNING: market-cap band CSV writing skipped ({_e})')

    # READ-ONLY review-reference DATA artifacts (RawMetricsTop100 + CohortMetricStats).
    # Computed AFTER scoring from the RAW metrics captured before normalizeAndDropNA
    # (postBoRank -> rankdic['postScoreMetric_raw']); NEVER read back into scoring or the
    # ranking (see reviewReference module docstring -- feeding cohort stats into the score
    # would be cross-sectional sector-neutralization, CEO-ratified OFF). Gated internally
    # on carve labels present (pre-carve run -> skipped with a note) and fully guarded so
    # it can never crash the deliverables. Filenames travel with the other deliverables.
    reviewref_fnames = []
    try:
        import reviewReference as rr
        reviewref_fnames = rr.emit_live(resdic, fidag, datasource, tickerfilter)
    except Exception as _e:
        print(f'WARNING: review-reference artifacts skipped ({type(_e).__name__}: {_e})')

    # Return the human-readable top-N deliverables just written (same pattern as
    # utils.saveWrapper returning its pickle name) so Sbocker.main can copy them to
    # the Drive-synced transfer dir at the pre-ingestion phase boundary. Data-only:
    # nothing here changes scoring/ranking/forensic output.
    return ([fname_AggScoretop, fname_presentationtop, fname_forensic]
            + sidelist_fnames + band_fnames + reviewref_fnames)

def writeBoAggToCSV(fb_df, mscore, cscore, baseurl, api_key, ntopagg, fname_AggScoretop, flag_df=None):
    fbdf_tocsv = fb_df.head(ntopagg)
    symblist = list(fbdf_tocsv['source'])
    #BoComp_tocsv = pd.DataFrame(columns=['source','currentRatio','dividendYield','grahamNumberToPrice','price','beta',
    #                                    'sector','fmpRating','PEratio','M_score','C_score'])
    BoComp_tocsv = pd.DataFrame()
    BoComp_tocsv['source'] = symblist
    #quote_full = pd.DataFrame(requests.get(f'{baseurl}v3/quote/{symblist}?&apikey={api_key}').json())
    crVec = []
    dyVec = []
    GNtPVec = []
    priceVec = []
    margin = []
    dcf2p = []
    betaVec = []
    sectorVec = []
    ratingVec_fmp = []
    pEratioVec = []
    mscoreVec = []
    cscoreVec = []
    # Note: Bulk endpoints require higher subscription tier, using individual API calls only
    profile_bulk_dict = {}
    rating_bulk_dict = {}
    dcf_bulk_dict = {}
    
    print(f'Writing top {ntopagg} stocks to .csv')
    pbar = tqdm(total=ntopagg)
    for row in BoComp_tocsv.itertuples():
        symb = row.source
        temp_resp_km = requests.get(f'{baseurl}v3/key-metrics/{symb}?period=quarter&limit=4&apikey={api_key}').json()
        temp_resp_fr = requests.get(f'{baseurl}v3/ratios/{symb}?period=quarter&limit=4&apikey={api_key}').json()
        
        # Use bulk data for profile, rating, and DCF if available, otherwise fallback to individual calls
        if symb in profile_bulk_dict:
            temp_resp_pr = [profile_bulk_dict[symb]]  # Convert dict to list format
        else:
            temp_resp_pr = requests.get(f'{baseurl}v3/profile/{symb}?apikey={api_key}').json()
        
        if symb in dcf_bulk_dict:
            temp_resp_dcf = [dcf_bulk_dict[symb]]  # Convert dict to list format
        else:
            temp_resp_dcf_raw = requests.get(f'{baseurl}v3/discounted-cash-flow/{symb}?apikey={api_key}')
            temp_resp_dcf = temp_resp_dcf_raw.json()
        
        # Diagnostic: Check what the DCF API actually returns (only for first ticker)
        if len(crVec) == 0:  # Only print for first ticker to avoid spam
            print(f"\nDEBUG: DCF API response for {symb}:")
            print(f"  Status code: {temp_resp_dcf_raw.status_code}")
            print(f"  Response type: {type(temp_resp_dcf)}")
            if isinstance(temp_resp_dcf, list):
                print(f"  Response length: {len(temp_resp_dcf)}")
                if len(temp_resp_dcf) > 0:
                    print(f"  First element type: {type(temp_resp_dcf[0])}")
                    if isinstance(temp_resp_dcf[0], dict):
                        print(f"  First element keys: {list(temp_resp_dcf[0].keys())}")
            elif isinstance(temp_resp_dcf, dict):
                print(f"  Dict keys: {list(temp_resp_dcf.keys())}")
                print(f"  Dict content: {temp_resp_dcf}")
            else:
                print(f"  Response content: {temp_resp_dcf}")
        
        # Handle case where API returns a dict instead of a list (API might have changed)
        if isinstance(temp_resp_dcf, dict):
            # If it's a dict, try to convert to list format or extract error
            if 'Error Message' in temp_resp_dcf or 'error' in str(temp_resp_dcf).lower():
                temp_resp_dcf = []  # Treat as empty
            else:
                # Try to wrap in list if it's a single DCF object
                temp_resp_dcf = [temp_resp_dcf] if temp_resp_dcf else []
        
        # Check if API responses are empty before accessing
        # Check currentRatio
        if len(temp_resp_fr) == 0 or 'currentRatio' not in temp_resp_fr[0]:
            crVec.append('NaN')
        elif type(temp_resp_fr[0]['currentRatio']) == int or type(temp_resp_fr[0]['currentRatio']) == float:
            crVec.append("{:.4f}".format(temp_resp_fr[0]['currentRatio']))
        else:
            crVec.append('NaN')
            
        # Check dividendYield
        if len(temp_resp_km) == 0 or 'dividendYield' not in temp_resp_km[0]:
            dyVec.append('NaN')
        elif type(temp_resp_km[0]['dividendYield']) == int or type(temp_resp_km[0]['dividendYield']) == float:
            dyVec.append("{:.4f}".format(temp_resp_km[0]['dividendYield']*100))
        else:
            dyVec.append('NaN')
            
        # Check grahamNumberToPrice
        if len(temp_resp_km) == 0 or len(temp_resp_pr) == 0:
            GNtPVec.append('NaN')
        elif 'grahamNumber' not in temp_resp_km[0] or 'price' not in temp_resp_pr[0]:
            GNtPVec.append('NaN')
        elif temp_resp_km[0]['grahamNumber'] is None or temp_resp_pr[0]['price'] is None:
            GNtPVec.append('NaN')
        else:
            gtp = (temp_resp_km[0]['grahamNumber']/temp_resp_pr[0]['price'])
            if type(gtp) == int or type(gtp) == float:
                GNtPVec.append("{:.4f}".format(gtp))
            else:
                GNtPVec.append('NaN')
                
        # Check price
        if len(temp_resp_pr) == 0 or 'price' not in temp_resp_pr[0]:
            priceVec.append('NaN')
        else:
            priceVec.append("{:.4f}".format(temp_resp_pr[0]['price']))
            
        # Check beta
        if len(temp_resp_pr) == 0 or 'beta' not in temp_resp_pr[0]:
            betaVec.append('NaN')
        else:
            betaVec.append("{:.4f}".format(temp_resp_pr[0]['beta']))
            
        # Check sector
        if len(temp_resp_pr) == 0 or 'sector' not in temp_resp_pr[0]:
            sectorVec.append('NaN')
        else:
            sectorVec.append(temp_resp_pr[0]['sector'])
            
        # Check priceEarningsRatio
        if len(temp_resp_fr) == 0 or 'priceEarningsRatio' not in temp_resp_fr[0]:
            pEratioVec.append('NaN')
        else:
            perat = temp_resp_fr[0]['priceEarningsRatio']
            if type(perat) == int or type(perat) == float:
                pEratioVec.append("{:.4f}".format(perat))
            else:
                pEratioVec.append('NaN')
                
        # Check rating
        # Use bulk rating data if available, otherwise fallback to individual call
        if symb in rating_bulk_dict:
            temp_resp_rating = [rating_bulk_dict[symb]]
        else:
            temp_resp_rating = requests.get(f'{baseurl}v3/rating/{symb}?apikey={api_key}').json()
        if len(temp_resp_rating) == 0 or 'ratingRecommendation' not in temp_resp_rating[0]:
            ratingVec_fmp.append('NaN')
        else:
            ratingVec_fmp.append(temp_resp_rating[0]['ratingRecommendation'])
            
        # Check M_Score
        if not (mscore[mscore['source'] == symb]['M_Score_mean']).isna().item():
            mcurscore = mscore[mscore['source'] == symb]['M_Score_mean'].item().item()
            if type(mcurscore) == int or type(mcurscore) == float:
                mscoreVec.append("{:.4f}".format(mscore[mscore['source'] == symb]['M_Score_mean'].item()))
            else:
                mscoreVec.append('NaN')
        else:
            mscoreVec.append('NaN')
            
        # Check C_Score
        if not (cscore[cscore['source'] == symb]['C_Score_mean']).isna().item():
            curcscore = cscore[cscore['source'] == symb]['C_Score_mean'].item().item()
            if type(curcscore) == int or type(curcscore) == float:
                cscoreVec.append("{:.4f}".format(cscore[cscore['source'] == symb]['C_Score_mean'].item()))
            else:
                cscoreVec.append('NaN')
        else:
            cscoreVec.append('NaN')

        # Check grossProfitMargin (needs 4 periods)
        if len(temp_resp_fr) == 0 or 'grossProfitMargin' not in temp_resp_fr[0]:
            margin.append('NaN')
        elif len(temp_resp_fr) < 4:
            margin.append('NaN')
        elif type(temp_resp_fr[0]['grossProfitMargin']) == int or type(temp_resp_fr[0]['grossProfitMargin']) == float:
            gpmsum= temp_resp_fr[0]['grossProfitMargin'] + temp_resp_fr[1]['grossProfitMargin'] + temp_resp_fr[2]['grossProfitMargin'] + temp_resp_fr[3]['grossProfitMargin']
            margin.append("{:.4f}".format(gpmsum*25))
        else:
            margin.append('NaN')

        # Check DCF to Price
        if len(temp_resp_dcf) == 0:
            dcf2p.append('NaN')
        elif 'dcf' not in temp_resp_dcf[0]:
            dcf2p.append('NaN')
        elif temp_resp_dcf[0]['dcf'] is None:
            dcf2p.append('NaN')
        elif type(temp_resp_dcf[0]['dcf']) == int or type(temp_resp_dcf[0]['dcf']) == float:
            if 'Stock Price' not in temp_resp_dcf[0]:
                dcf2p.append('NaN')
            elif temp_resp_dcf[0]['Stock Price'] is None:
                dcf2p.append('NaN')
            elif type(temp_resp_dcf[0]['Stock Price']) == int or type(temp_resp_dcf[0]['Stock Price']) == float:
                dcf2p.append("{:.4f}".format(temp_resp_dcf[0]['dcf']/(temp_resp_dcf[0]['Stock Price'])))
            else:
                dcf2p.append('NaN')
        else:
            dcf2p.append('NaN')
        pbar.update(n=1)
    BoComp_tocsv['price'] = priceVec
    BoComp_tocsv['PE-ratio'] = pEratioVec
    BoComp_tocsv['beta'] = betaVec
    BoComp_tocsv['sector'] = sectorVec
    BoComp_tocsv['rating_fmp'] = ratingVec_fmp
    BoComp_tocsv['currentRatio'] = crVec
    BoComp_tocsv['dividendYield'] = dyVec
    BoComp_tocsv['GrahamNumberToPrice'] = GNtPVec
    BoComp_tocsv['GrossProfitMargin_ttm'] = margin
    BoComp_tocsv['DCF-to-Price'] = dcf2p
    BoComp_tocsv['M-Score'] = mscoreVec
    BoComp_tocsv['C-Score'] = cscoreVec
    # Add CycleHeat from postRank data (already calculated in postBoRank)
    if 'CycleHeat' in fbdf_tocsv.columns:
        BoComp_tocsv['CycleHeat'] = fbdf_tocsv['CycleHeat'].values
    # Add moatScore from postRank data (merged from moatIdentifier)
    if 'moatScore' in fbdf_tocsv.columns:
        BoComp_tocsv['moatScore'] = fbdf_tocsv['moatScore'].values

    # Merge in the forensic-flag decision-support columns (offline-computed; no API).
    # These make the M/C flags, their drivers, Sloan accruals, the financial-invalid
    # indicator and the summary tag visible alongside the aggregate scores.
    if flag_df is not None and not flag_df.empty:
        # Cross-check the offline (pickle-based) financial classification against the
        # API `sector` just fetched (sectorVec is aligned to symblist): if EITHER
        # source says bank/insurer/REIT, the name is forensic-invalid (conservative).
        flag_df = ff.applySectorFallback(flag_df, dict(zip(symblist, sectorVec)))
        forensic_cols = ['source', 'isFinancial', 'financialKind', 'forensicValid',
                         'M_flag_gt_-1.78', 'M_drivers', 'C_flag_ge_4', 'C_flags_fired',
                         'sloanAccruals', 'sloan_worstQuintile_inShortlist', 'forensicTag']
        keep = [c for c in forensic_cols if c in flag_df.columns]
        BoComp_tocsv = BoComp_tocsv.merge(flag_df[keep], on='source', how='left')
    BoComp_tocsv.to_csv(fname_AggScoretop)
    pbar.close()
    return flag_df

def createPresentation(finalBoRank_df, mscore, cscore, baseurl, api_key, topn, fname, years, flag_df=None,
                       bands=None):
    #test
    #fname = fname_spreadSheet
    #topn = 20
    #years = 10
    # Market-cap banding (ADDITIVE): when currency data is present, the general top-N is
    # the General band (>$300M) head(topn) -- i.e. postRank[marketCap_usd>300e6].head(topn) --
    # so the xlsx general list matches the banded partition. When banding is absent OR
    # currency is still pending, behaviour is UNCHANGED (byte-identical to before): the
    # general top-N stays postRank.head(topn), so nothing wrong ships before the field flows.
    # TODO: emit each sub-band's top-5 as its own labelled sheet block once the field flows
    # (deferred: the HTML presentation already carries the full banded view; adding sheets
    # here multiplies the per-symbol live API calls).
    _general_df = finalBoRank_df
    if bands and not bands.get('currency_pending', True):
        _gb = (bands.get('bands') or {}).get('General')
        if _gb is not None and not _gb.empty:
            # The General band is pre-capped at the MCAP_BANDS General head_N (=20). If a
            # caller ever requests MORE than that (ntopxlsx > 20), keying the xlsx off the
            # band would SILENTLY shrink the general list -- so fall back to the unbanded
            # head(topn) and warn LOUDLY instead. No effect today (ntopxlsx == 20 == cap).
            if topn > len(_gb):
                print(f"WARNING: createPresentation topn={topn} exceeds General-band size "
                      f"{len(_gb)} (MCAP_BANDS General cap); using unbanded postRank.head({topn}) "
                      f"to avoid silently shrinking the general list.", flush=True)
            else:
                _general_df = _gb
    symblist = list(_general_df['source'].head(topn))
    #eyVec = []
    #quote_full = pd.DataFrame(requests.get(f'{baseurl}v3/quote/{symblist}?&apikey={api_key}').json())

    almago = datetime.today() - timedelta(weeks=5)
    if almago.day >= 5:
        lastlast_5th = almago.replace(day=5)
    else:
        lastlast_5th = (almago - timedelta(days=almago.day)).replace(day=5)
    ll5 = lastlast_5th.strftime('%Y-%d-%m')
    wb = openpyxl.Workbook()
    print(f'Writing top {topn} stocks to an .xlsx file for presentation')
    pbar = tqdm(total=topn)
    for symb in symblist[::-1]:
        km = pd.DataFrame(
            requests.get(f'{baseurl}v3/key-metrics/{symb}?period=annual&limit={years}&apikey={api_key}').json())
        fr = pd.DataFrame(
            requests.get(f'{baseurl}v3/ratios/{symb}?period=annual&limit={years}&apikey={api_key}').json())
        pr = requests.get(
            f'{baseurl}v3/profile/{symb}?apikey={api_key}').json()
        rating = requests.get(
            f'{baseurl}v3/rating/{symb}?apikey={api_key}').json()
        #target = requests.get(
        # f'https://financialmodelingprep.com/api/v4/price-target-consensus?symbol{symb}&apikey={api_key}').json()
        sp = requests.get(
            f'{baseurl}v4/stock_peers?symbol={symb}&apikey={api_key}').json()
        cf = pd.DataFrame(
            requests.get(f'{baseurl}v3/cash-flow-statement/{symb}?period=annual&limit={years}&apikey={api_key}').json())
        dcf_resp = requests.get(f'{baseurl}v3/discounted-cash-flow/{symb}?apikey={api_key}').json()
        dcf = pd.DataFrame.from_dict(dcf_resp) if dcf_resp else pd.DataFrame()
        
        # Check if DCF has required columns, use fallback if empty
        dcf_has_data = not dcf.empty and 'Stock Price' in dcf.columns and 'dcf' in dcf.columns

        NYSEspe = requests.get(f'https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?date={ll5}&exchange=NYSE&apikey={api_key}').json()
        nspe_df = pd.DataFrame(NYSEspe) if isinstance(NYSEspe, list) and len(NYSEspe) > 0 else pd.DataFrame()
        nspe_has_data = not nspe_df.empty and 'sector' in nspe_df.columns and 'pe' in nspe_df.columns
        symb_df = pd.DataFrame(
            columns=['Symbol', 'Date', 'Earnings yield', 'PE-ratio', 'Price-to-book', 'Current ratio',
                     'Dividend yield', 'Price-to-fair value', 'Price'])
        symb_df['Symbol'] = fr['symbol']
        symb_df['Date'] = fr['date']
        symb_df['Earnings yield'] = (km.earningsYield * 100).apply(format_num)
        symb_df['PE-ratio'] = km.peRatio.apply(format_num)
        symb_df['Price-to-book'] = km.ptbRatio.apply(format_num)
        symb_df['Current ratio'] = km.currentRatio.apply(format_num)
        symb_df['Dividend yield'] = (km.dividendYield.fillna(0)*100).apply(format_num)
        symb_df['Price-to-fair value'] = fr.priceFairValue.apply(format_num)
        # Handle empty DCF data (common for non-US stocks)
        if dcf_has_data:
            price = dcf['Stock Price'].apply(format_num)
            symb_df['Price'] = price
            symb_df['Graham number to price'] = (km.grahamNumber/dcf['Stock Price']).apply(format_num)
        else:
            symb_df['Price'] = 'N/A'
            symb_df['Graham number to price'] = 'N/A'

        fcf = cf.freeCashFlow
        shares = fcf/km.freeCashFlowPerShare
        #qdDCFperPrice = quickDCF(fcf,0.12,0,km.interestDebtPerShare*shares,shares,price)

        if symb not in wb.sheetnames:
            ws = wb.create_sheet(symb, 0)
            ws.title = symb
        else:
            ws = wb[symb]
        wb.active = ws

        for r in dataframe_to_rows(symb_df, index=False, header=True):
            ws.append(r)

        for cell in ws['A'] + ws[1]:
            cell.style = 'Pandas'

        bold_font = Font(bold=True)
        psdf_col = len(symb_df.columns)+2
        psdf_row = 1
        
        # Check if profile data is available
        pr_has_data = pr and len(pr) > 0 and isinstance(pr[0], dict)
        
        ws.cell(row=psdf_row, column=psdf_col).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col).value = 'Company'
        ws.cell(row=psdf_row+1, column=psdf_col).value = pr[0].get('companyName', 'N/A') if pr_has_data else 'N/A'

        ws.cell(row=psdf_row, column=psdf_col + 1).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+1).value = 'beta'
        if pr_has_data and pr[0].get('beta') is not None:
            ws.cell(row=psdf_row+1, column=psdf_col+1).value = "{:.4f}".format(pr[0]['beta'])
        else:
            ws.cell(row=psdf_row+1, column=psdf_col+1).value = 'N/A'

        ws.cell(row=psdf_row, column=psdf_col + 2).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+2).value = 'Market Cap'
        if pr_has_data and pr[0].get('mktCap') is not None:
            ws.cell(row=psdf_row+1, column=psdf_col+2).value = "{:,.2f}".format(pr[0]['mktCap']/1000000) + " million"
        else:
            ws.cell(row=psdf_row+1, column=psdf_col+2).value = 'N/A'

        ws.cell(row=psdf_row, column=psdf_col + 3).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+3).value = 'Industry & Sector'
        if pr_has_data and 'industry' in pr[0]:
            ws.cell(row=psdf_row+1, column=psdf_col+3).value = pr[0]['industry']
            ws.cell(row=psdf_row+2, column=psdf_col+3).value = pr[0].get('sector', 'N/A')
        else:
            ws.cell(row=psdf_row+1, column=psdf_col+3).value = 'N/A'
            ws.cell(row=psdf_row+2, column=psdf_col+3).value = 'N/A'

        ws.cell(row=psdf_row, column=psdf_col + 4).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+4).value = 'Sector Average PE-ratio'
        # Handle sector PE ratio - may not be available on all subscriptions
        if nspe_has_data and pr_has_data and 'sector' in pr[0]:
            sector_match = nspe_df[nspe_df['sector'] == pr[0]['sector']]
            if not sector_match.empty:
                secpe = sector_match.pe.iloc[0]
                ws.cell(row=psdf_row+1, column=psdf_col + 4).value = str(round(float(secpe), 4))
            else:
                ws.cell(row=psdf_row+1, column=psdf_col + 4).value = 'N/A'
        else:
            ws.cell(row=psdf_row+1, column=psdf_col + 4).value = 'N/A'

        ws.cell(row=psdf_row + 5, column=psdf_col).font = bold_font
        ws.cell(row=psdf_row + 5, column=psdf_col).value = 'Rating Recommendation'
        # Handle empty rating response
        if rating and len(rating) > 0 and 'ratingRecommendation' in rating[0]:
            ws.cell(row=psdf_row + 6, column=psdf_col).value = rating[0]['ratingRecommendation']
        else:
            ws.cell(row=psdf_row + 6, column=psdf_col).value = 'N/A'

        ws.cell(row=psdf_row + 5, column=psdf_col+2).font = bold_font
        ws.cell(row=psdf_row + 5, column=psdf_col+2).value = '(QD?) DCF per price'
        if dcf_has_data:
            ws.cell(row=psdf_row + 6, column=psdf_col+2).value = str(round(float(dcf['dcf']/dcf['Stock Price']), 4))
        else:
            ws.cell(row=psdf_row + 6, column=psdf_col+2).value = 'N/A'
        #ws.cell(row=psdf_row + 6, column=psdf_col+2).value = qdDCFperPrice

        ws.cell(row=psdf_row+5, column=psdf_col + 4).font = bold_font
        ws.cell(row=psdf_row + 5, column=psdf_col + 4).value = 'List of peers'
        # Handle empty stock peers response
        if sp and len(sp) > 0 and isinstance(sp[0], dict) and 'peersList' in sp[0]:
            peerslist = sp[0]['peersList']
            for i, peer in enumerate(peerslist):
                ws.cell(row=psdf_row + 6 + i, column=psdf_col + 4).value = peer
        else:
            ws.cell(row=psdf_row + 6, column=psdf_col + 4).value = 'N/A'

        # --- Forensic decision-support block (offline; from precomputed flag_df) ---
        # FLAGS, NOT VERDICTS. Surfaces the M/C flags + drivers, Sloan accruals, the
        # financial-invalid indicator and the summary tag so the CEO's manual review
        # sees the risk and its driver. No API call here.
        if flag_df is not None and not flag_df.empty and (flag_df['source'] == symb).any():
            frow = flag_df[flag_df['source'] == symb].iloc[0]
            fcol = psdf_col
            frow0 = psdf_row + 13
            ws.cell(row=frow0, column=fcol).font = bold_font
            ws.cell(row=frow0, column=fcol).value = 'FORENSIC FLAGS (guidance, not a drop)'
            if not bool(frow.get('forensicValid', True)):
                ws.cell(row=frow0, column=fcol + 1).value = (
                    f"INVALID for {frow.get('financialKind', 'financial')} — use financial lens")
            forensic_items = [
                ('Summary tag', frow.get('forensicTag', '')),
                ('Beneish M > -1.78?', 'FLAG' if frow.get('M_flag_gt_-1.78') else 'no'),
                ('  M drivers', frow.get('M_drivers', '') or '-'),
                ('Montier C >= 4?', 'FLAG' if frow.get('C_flag_ge_4') else 'no'),
                ('  C flags fired', frow.get('C_flags_fired', '') or '-'),
                ('Sloan accruals', frow.get('sloanAccruals', '')),
                ('  Sloan worst-quintile (within shortlist)?',
                 'FLAG' if frow.get('sloan_worstQuintile_inShortlist') else 'no'),
                ('Financial (bank/insurer/REIT)?', 'YES' if frow.get('isFinancial') else 'no'),
            ]
            for i, (label, val) in enumerate(forensic_items, start=1):
                ws.cell(row=frow0 + i, column=fcol).font = bold_font
                ws.cell(row=frow0 + i, column=fcol).value = label
                ws.cell(row=frow0 + i, column=fcol + 1).value = ('' if val is None else str(val))

        resize_columns(ws)

        pbar.update(n=1)


    wb.save(fname)
    wb.close()
    pbar.close()

    return None

# Function to resize columns in an Excel sheet
def resize_columns(ws):
    for column_cells in ws.iter_cols():
        length = max(len(str(cell.value)) for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length

def format_num(x):
    return "{:.4f}".format(x)

def moatIdentifier(symblist, cdx_df, n=20, freq_map=None):
    #moatdf = pd.DataFrame(columns=['source','moatScore'])
    #for symb in symbollist:
    # calculate FCFyield (>5-10%), Gross Margin (GrossProfit/Revenue>30%)
    # Calculate sales/Assets (>0.75)
    # Calculate RoE (>15%) and RoA(>10%) and ROIC (>15%)
    # High earnings even when market earnings are down (beta < 1?)
    # Calculate SG&A/GrossProfit (<15%) [sale, general and administrative expenses]
    # Calculate Depreciation/GrossProfit (<10%)
    # Calculate InterestExpenses/OperatingIncome (<15%)
    # Calculate NetMargin [NetIncome/Revenue] (>20%)
    # Calculate Capex/NetIncome (<25%)
    # Calculate TotalLiabilities/ShareholderEquity (<0.8)
    #moatdf = pd.DataFrame(columns=['source','moatScore','FCFyield','GrossMargin','RevtoASS','RoE','RoA','ROIC','SGAtoGP','DeptoGP',
    #                               'InteresttoOI', 'NetMargin','CapExtoEarnings','TLtoEquity'])
    moatdf = pd.DataFrame(columns=['source','moatScore','FCFyield','GrossMargin','RevtoASS','RoE','RoA','ROIC','SGAtoGP','DeptoGP',
                                   'NetMargin','CapExtoEarnings','TLtoEquity'])
    nan_dict = {'source': np.nan,  'moatScore': np.nan, 'FCFyield': np.nan, 'GrossMargin': np.nan, 'RevtoASS': np.nan,
                'RoE': np.nan, 'RoA': np.nan, 'ROIC': np.nan, 'SGAtoGP': np.nan, 'DeptoGP': np.nan,
                'NetMargin': np.nan, 'CapExtoEarnings': np.nan, 'TLtoEquity': np.nan}
    # Every criterion below is a head(n) mean over the name's own history, so `n` is
    # scaled per source to its rows-per-year: n=20 is 5 years of quarters but TEN years
    # of halves, which graded semi-annual filers on twice the history (row-based site
    # NOT on the audit's list -- found in the 2026-07-25 sweep). Display-only metric,
    # but it reaches the CEO as an absolute 0-11 count.
    if freq_map is None:
        freq_map = rp.frequency_by_source(cdx_df)
    tempdf_orig = pd.concat([moatdf, pd.DataFrame([nan_dict])], ignore_index=True)
    # The 11 moat criteria that ARE the score -- each column is already expressed as
    # (value - threshold), so "> 0" means "passes".  Counting these EXPLICITLY (rather
    # than every numeric column) is half of the audit M2 fix below.
    moat_criteria = ['FCFyield', 'GrossMargin', 'RevtoASS', 'RoE', 'RoA', 'ROIC',
                     'SGAtoGP', 'DeptoGP', 'NetMargin', 'CapExtoEarnings', 'TLtoEquity']
    _n_quarterly = n            # the caller's window, expressed in QUARTERS
    for symb in symblist:
        # .copy() -- audit M2 fix (2026-07-19).  `tempdf = tempdf_orig` bound the SAME
        # one-row frame every iteration, so the PREVIOUS ticker's moatScore was still
        # sitting in it when `select_dtypes(include='number') > 0` counted the criteria,
        # and any prior score > 0 was counted as a 12th criterion.  Effect on the shipped
        # 2026-07-17 universe: moatScore too high by exactly +1 on 7,751 of 7,752 names
        # (only the first-processed ticker, whose carried value was still NaN, was right),
        # and the observed minimum was 1 on an 0-11 scale where 0 must be attainable --
        # 496 names shown as "1" are really 0.  The number is display-only (moatScore is
        # merged AFTER Stage-2, so AggScore never saw it), but it reaches the CEO as an
        # absolute count.
        tempdf = tempdf_orig.copy()
        # NEWEST-FIRST BEFORE head(n) (ship-gate BLOCKER, fixed 2026-07-27).
        # Every comparator below is a `.head(n).mean()`, i.e. a RECENCY window -- but
        # cdx_df arrives OLDEST-first (data_quality sorts ascending; verified 7,752 of
        # 7,752 sources), so head(n) was reading the name's OLDEST rows.  Median lag of
        # the window from the newest filing was 1.00 year for a quarterly filer and
        # 7.00 years for a semi-annual one: the 2026-07-25 window scaling (head(20) ->
        # head(10) for rpy=2) AMPLIFIED the pre-existing defect by anchoring a shorter
        # window at the wrong end.  moatScore differed on 50.2% of 400 real names, by up
        # to +-7 points on an 0-11 scale.
        # Uses detectManipulation._toNewestFirst -- the same helper every sibling cdx
        # consumer uses -- rather than a fresh sort, so there is ONE definition of
        # "newest-first" in the pipeline.  It sorts by PARSED date, so it is robust to
        # however the rows happen to arrive.
        cdx_temp = _toNewestFirst(cdx_df[cdx_df['source'] == symb])
        _rpy = rp.rows_per_year(freq_map, symb)
        n = rp.scale_window(_n_quarterly, _rpy)
        # ANNUALIZE the FLOW-over-STOCK comparators (specialist ruling, 2026-07-25).
        # Every bar below is an ANNUAL rule of thumb -- FCF yield > 10%, sales/assets >
        # 0.75, RoE > 15%, RoA > 10%, ROIC > 15% -- but the ratios were per-PERIOD, so a
        # quarterly filer was being asked to earn a full YEAR's return in three months.
        # Five of the eleven comparators were therefore near-unpassable by construction.
        # `af` is a TRUE annualisation (x4 quarterly, x2 semi-annual) because these
        # thresholds are ABSOLUTE; it deliberately changes quarterly names too, and
        # moatScores rise materially as a result.
        # NOT annualized (flow/flow or stock/stock -- scale cancels): GrossMargin,
        # SGAtoGP, DeptoGP, NetMargin, CapExtoEarnings, TLtoEquity.
        af = rp.annualize_factor(_rpy)
        tempdf['source'] = symb
        fcfmask = cdx_temp['pfcfRatio'] != 0
        fcfyield_filter = cdx_temp['pfcfRatio'][fcfmask]
        tempdf['FCFyield'] = (1/fcfyield_filter).head(n).mean()*af-0.1
        tempdf['GrossMargin'] = cdx_temp['grossProfitMargin'].head(n).mean()-0.3
        tempdf['RevtoASS'] = (cdx_temp['revenue']/cdx_temp['totalAssets']).head(n).mean()*af-0.75
        tempdf['RoE'] = cdx_temp['returnOnEquity'].head(n).mean()*af-0.15
        tempdf['RoA'] = cdx_temp['returnOnAssets'].head(n).mean()*af-0.1
        tempdf['ROIC'] = cdx_temp['returnOnCapitalEmployed'].head(n).mean()*af - 0.15
        gpmask = cdx_temp['grossProfit'] != 0
        gp_filter = cdx_temp['grossProfit'][gpmask]
        tempdf['SGAtoGP'] = 0.15-(cdx_temp['sellingGeneralAndAdministrativeExpenses'][gpmask]/gp_filter).head(n).mean()
        tempdf['DeptoGP'] = 0.1 - (cdx_temp['depreciationAndAmortization'][gpmask]/gp_filter).head(n).mean()
        #tempdf['InteresttoOI'] = 0.15 - (cdx_df['interestExpense']/cdx_df['operatingIncome']).head(n).mean()
        tempdf['NetMargin'] = cdx_temp['netProfitMargin'].head(n).mean() - 0.2
        # CapEx/Earnings: |capex| / |NI| < 0.20, GATED ON NI > 0 (domain review S8, fixed
        # 2026-07-26).  It was `0.2 - mean(capexPerShare/netIncomePerShare)`, which
        # free-passed 100.0% of loss-makers vs 44.5% of profitable names.
        # NOTE the audit's stated MECHANISM was wrong and would have produced the wrong fix:
        # it blamed "FMP capexPerShare negative", but on this panel capexPerShare is POSITIVE
        # on 86.5% of 176,604 rows and negative on 0.0000% (median +0.084).  The real cause is
        # the DENOMINATOR -- for NI < 0 the ratio is negative on 83.4% of rows, so
        # `0.2 - negative > 0` is an automatic tick.
        # "Capex is a small share of earnings" is UNDEFINED when there are no earnings, so a
        # loss-making period is NOT-COMPUTABLE (NaN), not a pass: NaN > 0 is False, so it
        # neither ticks the box nor counts against the other ten comparators.
        _ni_ps = pd.to_numeric(cdx_temp['netIncomePerShare'], errors='coerce')
        _cx_ps = pd.to_numeric(cdx_temp['capexPerShare'], errors='coerce')
        _prof = _ni_ps > 0
        tempdf['CapExtoEarnings'] = (0.2 - (_cx_ps[_prof].abs()
                                            / _ni_ps[_prof].abs()).head(n).mean())
        tempdf['TLtoEquity'] = 0.8 - (cdx_temp['totalLiabilities']/cdx_temp['totalStockholdersEquity']).head(n).mean()
        # Count ONLY the 11 criteria -- never moatScore itself, and never any column that
        # happens to be numeric.  NaN > 0 is False, so a non-computable criterion does not
        # pass (an unchanged property of the old code).
        mask = tempdf[moat_criteria].apply(pd.to_numeric, errors='coerce') > 0
        tempdf['moatScore'] = mask.sum(axis=1)

        moatdf = pd.concat([moatdf, tempdf]).reset_index(drop=True)

    moatdf.sort_values(by='moatScore', ascending=False, inplace=True)

    return moatdf


#CLEAN THIS UP
def findHighestOfEachSector(resdic):
    #sectorlist = resdic['sectorlist']
    #sectorlist =  ['all', 'Unspecified', 'Basic Materials', 'Healthcare', 'Financial Services',
    #              'Energy', 'Consumer Cyclical', 'Consumer Defensive', 'Industrials',
    #              'Communication Services', 'Technology', 'Real Estate', 'Utilities','Biotechnology']
    #sectorlist.remove('all')
    #sectorlist.remove('Unspecified')
    sectordic = pd.read_pickle('sectorsdic_fmp.pickle')
    sectorlist = list(sectordic.keys())
    baseurl = resdic['baseurl']
    api_key = resdic['api_key']
    sectorsfound = []
    sectorsnotfound = sectorlist
    bsdf = resdic['BoScore_df'].reset_index(drop=True)
    bsdf['score'] = bsdf['score'].astype(float)
    bshs = pd.DataFrame(columns=['source', 'score', 'sector'])
    symblist = bsdf['source']

    for sector in sectorlist:
        sslist = sectordic[sector]
        if len(bsdf.loc[bsdf['source'].isin(sslist), 'score']) > 0:
            highest_rowid = bsdf.loc[bsdf['source'].isin(sslist), 'score'].idxmax()
            newrow = {'source': bsdf.loc[highest_rowid]['source'], 'score': bsdf.loc[highest_rowid]['score'], 'sector': sector}
            tempdf = pd.DataFrame([newrow])
            bshs = pd.concat([bshs, tempdf], ignore_index=True)
    bshs = bshs.reset_index(drop=True)

    #bshs['sector'] = sectorlist

    # Removed second loop that was searching for sectors not in sectordic
    # The first loop (lines 388-394) already finds the highest-scoring stock for each sector
    # that exists in the current dataset. The second loop was inefficiently searching through
    # all symbols for sectors that don't exist due to sector mapping/consolidation.

    resdic = {**resdic, **{'BoScore_highsectors': bshs}}

    return resdic

#see sector performance
#https://financialmodelingprep.com/api/v3/stock/sector-performance?apikey=YOUR_API_KEY
#https://financialmodelingprep.com/api/v3/historical-sectors-performance?limit=50&apikey={api_key}

## Inspect Owner's Earnings

## Get historical dividends via
#https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/AAPL?apikey=YOUR_API_KEY

## see analyst estimates: https://financialmodelingprep.com/api/v3/analyst-estimates/AAPL?limit=30&apikey=YOUR_API_KEY
