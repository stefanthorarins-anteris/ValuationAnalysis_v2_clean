"""
Data Quality Module

Identifies and removes clearly invalid/corrupted data before any analysis.
Logs removed data to CSV for transparency.

Conservative filters that only catch obvious corruption:
1. Negative prices (impossible)
2. Zero prices (invalid for return calculations)
3. Price-to-MarketCap sanity check (catches API garbage)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

import nan_policy as npol


# --- cross-field plausibility thresholds (see checks 5-7 in check_price_sanity) -----
# Each is set where the combination becomes ARITHMETICALLY IMPOSSIBLE rather than merely
# extreme, because the whole point of the score is to find genuine extremes.
# earnings yield >= 300% (P/E <= 0.33).  NOT set at P/E<=1: a one-off gain (asset sale,
# debt-for-equity swap) genuinely produces a sub-1 P/E for a quarter, and measurement on
# the 2026-07-17 panel found real names sitting at earningsYield 1.08-1.20 with an
# ordinary dividend -- odd but not impossible, so they must NOT be removed.  At >=3x the
# entire market capitalisation earned in one quarter, no combination of real one-offs and
# a normal dividend policy holds together.
_EARNYIELD_IMPOSSIBLE = 3.0
# >=100x market-cap step in ONE quarter on a flat share count.  100x, not 20x, because a
# 20x bar removes REAL market events: measured on the 2026-07-17 panel it flagged GME's
# 2021-01 squeeze (21.6x), SEZL's 2023 rally (28.1x), QUBT (25.6x) and DRUG (48.5x) --
# all genuine, if violent, re-ratings, and flagging them deletes the name's whole history.
# 100x matches the bar Check 4 already uses in this module, and with a flat share count it
# means a >99% collapse or a >9,900% rally in one quarter, which no listed equity does.
_MCAP_BREAK_RATIO = 100.0
_SHARES_FLAT_BAND = 1.10       # "share count essentially unchanged" = within +-10%
# NO reporting-gap bound on the step check -- REMOVED 2026-07-25 (review N2).
#
# A `1.5 x the source's own median cadence` gate was briefly added to make the check
# idempotent across data_quality's two passes.  It was BOTH unnecessary and harmful:
#
#   * UNNECESSARY -- idempotency is STRUCTURAL and stronger than any gap bound.  PASS 3
#     removes every row AT OR BEFORE a source's most-recent corruption date, so what
#     survives is a contiguous DATE SUFFIX of that source's series.  Every removed row is
#     therefore OLDER than every kept row, and no two kept rows that were non-adjacent can
#     become adjacent -- a later pass sees the same interior adjacencies it saw before, so
#     it cannot invent a break.
#   * HARMFUL -- it suppressed 1 of 17 real detections, and specifically the one this
#     module's own comment cites as the motivating example: TAM.L steps 6.99e5 -> 1.06e8
#     (151.2x) with the share count flat to 3 decimals, but across a 365-day gap against a
#     184-day median cadence, so the gate vetoed it and all 21 rows were kept.
#
# The Sbocker self-heal guard (recompute bm_ave if pass 2 ever removes a row) is retained
# as belt-and-braces, since "structurally idempotent" still rests on PASS 3 keeping its
# prefix-removal semantics.


def check_price_sanity(row, price_col='price', mcap_col='marketCap',
                       prev_price=None, prev_mcap=None,
                       prev_row_mcap=None, prev_row_shares=None):
    """
    Check if a price data point is sane.

    prev_price / prev_mcap : the last VALID row's values (existing semantics, used by
        check 4).
    prev_row_mcap / prev_row_shares : the IMMEDIATELY PRECEDING row's values, valid or
        not.  The market-cap step check (6) must use these, not the last-valid ones: a
        "step" is by definition an adjacent-period comparison, and because a flagged row
        does NOT advance the last-valid baseline, keying off prev_mcap makes ONE genuine
        break flag every row after it.  Measured on the 2026-07-17 panel, that cascade
        turned a single real break in TAM.L's 2015 quarter into 20 flagged rows and
        produced 12 spurious flags on SEZL, whose adjacent market-cap steps are all
        between 0.27x and 2.8x.  Optional, so existing callers keep working.

    Returns:
    --------
    tuple: (is_valid, reason) - reason is None if valid
    """
    price = row.get(price_col, np.nan)
    mcap = row.get(mcap_col, np.nan)
    
    # Check 1: Negative price (impossible)
    if pd.notna(price) and price < 0:
        return False, f"negative_price ({price:.4f})"
    
    # Check 2: Zero price (invalid for returns)
    if pd.notna(price) and price == 0:
        return False, "zero_price"
    
    # Check 3: Price-MarketCap sanity
    # If mcap is reasonable but price is absurd, it's likely API corruption
    # A $30B company shouldn't have a $4M stock price
    if pd.notna(price) and pd.notna(mcap) and mcap > 0 and price > 0:
        # Implied shares outstanding
        implied_shares = mcap / price
        
        # If implied shares < 1000, price is likely way too high
        # (Almost no public company has < 1000 shares)
        if implied_shares < 1000:
            return False, f"impossible_price_vs_mcap (price=${price:.2f}, mcap=${mcap/1e9:.2f}B, implied_shares={implied_shares:.0f})"
        
        # If implied shares > 1 trillion, price is likely way too low (or mcap wrong)
        # We're conservative here - some companies have trillions of shares
        if implied_shares > 1e12:
            return False, f"suspicious_price_vs_mcap (price=${price:.6f}, mcap=${mcap/1e9:.2f}B, implied_shares={implied_shares:.0e})"
    
    # Check 4: Extreme quarter-over-quarter jump (if we have previous data)
    # A 1000x jump in one quarter while mcap stays similar is API corruption
    if prev_price is not None and prev_mcap is not None:
        if prev_price > 0 and price > 0:
            price_change_ratio = price / prev_price
            
            # Check if price jumped >100x but mcap stayed within 5x
            if pd.notna(mcap) and pd.notna(prev_mcap) and prev_mcap > 0:
                mcap_change_ratio = mcap / prev_mcap
                
                # Price up 100x but mcap within 5x = corruption
                if price_change_ratio > 100 and 0.2 < mcap_change_ratio < 5:
                    return False, f"price_mcap_mismatch (price {price_change_ratio:.0f}x but mcap {mcap_change_ratio:.2f}x)"
                
                # Price down 100x but mcap within 5x = also suspicious
                if price_change_ratio < 0.01 and 0.2 < mcap_change_ratio < 5:
                    return False, f"price_mcap_mismatch (price {price_change_ratio:.4f}x but mcap {mcap_change_ratio:.2f}x)"

    # =====================================================================
    # CROSS-FIELD PLAUSIBILITY (2026-07-25).  Checks 1-4 above each look at ONE
    # field at a time, so a single corrupt marketCap passes them all -- and one bad
    # market cap makes a name extreme in FOUR COLLINEAR cheapness columns at once
    # (earnYield <-> grahamNumberToPrice r=0.83, bVpRatio <-> tbVpRatio r=0.79), which
    # no per-column winsorization can contain: the name is not an outlier in any single
    # column by enough to be clipped, it is simply "cheap" four times over.
    #
    # EXEMPLAR (real, 2026-07-17): BXP.L's newest quarter reports marketCap $307.6M
    # against $31.2B the quarter before -- a 101x collapse -- while netIncome
    # ($2.15B), weightedAverageShsOut (446.1M), totalStockholdersEquity ($59.1B) and
    # revenue ($13.2B) all continue normally.  Because FMP derives earningsYield and
    # dividendYield FROM that market cap, they come out at 699% and 9.1%, and pbRatio
    # at 0.0052.  check_price_sanity returned (True, None) on that row: the price/mcap
    # RATIO is fine (both moved together), no field is individually out of range, and
    # Check 4's "price moved but mcap didn't" pattern is the exact inverse of this one.
    #
    # DELIBERATELY CONSERVATIVE: these fire only on combinations that cannot be true of
    # any real company, never on merely-extreme values.  Finding extremes is the
    # SCORE's job; this only removes arithmetic impossibilities.
    # -----------------------------------------------------------------------------
    # Check 5: earnings yield >= _EARNYIELD_IMPOSSIBLE (3.0, i.e. P/E <= 0.33) alongside
    # a positive dividend yield.  A firm cannot earn THREE TIMES its entire market
    # capitalisation in a quarter; and if the earnings were real, a token dividend
    # alongside them is incoherent (BXP.L's implied payout ratio is 1.3%).  Requiring
    # BOTH legs keeps this off genuinely cheap deep-value names, which sit at P/E 3-8.
    # The threshold is 3.0 and not 1.0 because measurement found real names at
    # earningsYield 1.08-1.20 with ordinary dividends -- odd, not impossible.
    ey = row.get('earningsYield', np.nan)
    dy = row.get('dividendYield', np.nan)
    if (pd.notna(ey) and pd.notna(dy) and ey >= _EARNYIELD_IMPOSSIBLE
            and dy > 0):
        return False, (f"implausible_yield_pair (earningsYield={ey:.3f} => P/E="
                       f"{1.0/ey:.4f}, dividendYield={dy:.4f})")

    # Check 6: market cap steps by >= _MCAP_BREAK_RATIO between consecutive quarters
    # while the SHARE COUNT is essentially unchanged.  Market cap = price x shares, so
    # with shares flat a 100x cap move requires a 100x price move in one quarter --
    # that is a data break, not a market event.  Guarded on BOTH share counts being
    # present and close, so a genuine capital raise / reverse split (which moves the
    # share count too) is NOT caught.
    #
    # NO reporting-gap bound -- see the note on _MCAP_BREAK_RATIO above.  The comparison is
    # simply against the ADJACENT preceding row, whatever the gap: a >=100x move with the
    # share count flat is a data break at any spacing, and bounding it by cadence vetoed a
    # real 151x break (TAM.L) while buying no idempotency that the prefix-removal structure
    # does not already guarantee.
    shares = row.get('weightedAverageShsOut', np.nan)
    if (prev_row_mcap is not None and pd.notna(mcap) and pd.notna(prev_row_mcap)
            and prev_row_mcap > 0 and mcap > 0
            and pd.notna(shares) and prev_row_shares is not None
            and pd.notna(prev_row_shares) and prev_row_shares > 0):
        mcap_ratio = mcap / prev_row_mcap
        share_ratio = shares / prev_row_shares
        if ((mcap_ratio >= _MCAP_BREAK_RATIO or mcap_ratio <= 1.0 / _MCAP_BREAK_RATIO)
                and (1.0 / _SHARES_FLAT_BAND) <= share_ratio <= _SHARES_FLAT_BAND):
            return False, (f"mcap_step_break (mcap {mcap_ratio:.4g}x with shares "
                           f"{share_ratio:.3f}x)")

    # NOT IMPLEMENTED -- "implied share count (marketCap/price) vs reported share count".
    # It was written, measured, and REMOVED: it cannot detect anything real here.  Once
    # price = marketCap/weightedAverageShsOut (the 2026-07-19 ingest fix) the ratio is
    # identically 1.0 BY CONSTRUCTION, so the check is vacuous going forward; and on the
    # pre-fix data it fired on 1,335 rows across 567 sources clustered at 10-18x, which is
    # the old quarterly-PE price derivation drifting, NOT corrupt data.  A screen that is
    # either vacuous or a false-positive generator earns no place in the pipeline.

    return True, None


def filter_invalid_data(cdx_df, price_col='price', mcap_col='marketCap', 
                        min_periods_required=8, verbose=True):
    """
    Filter out rows with clearly invalid/corrupted price data.
    
    Logic:
    1. Identify all corrupt data points
    2. For each ticker with corruption, find the MOST RECENT corruption date
    3. Remove ALL data at or before that corruption date (keep only newer data)
    4. If remaining data < min_periods_required, remove ticker entirely
    
    This approach: we keep the NEWEST reliable data, since older data before
    corruption may have hidden issues.
    
    Parameters:
    -----------
    cdx_df : DataFrame
        The cdx (fundamentals + price) dataframe
    price_col : str
        Name of price column
    mcap_col : str
        Name of market cap column
    min_periods_required : int
        Minimum data points needed after filtering. If less, remove ticker entirely.
    verbose : bool
        Print summary statistics
    
    Returns:
    --------
    tuple: (clean_df, removed_df)
        - clean_df: DataFrame with invalid rows removed
        - removed_df: DataFrame of removed rows with 'removal_reason' column
    """
    if cdx_df is None or cdx_df.empty:
        return cdx_df, pd.DataFrame()
    
    df = cdx_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by source and date for sequential checking
    df = df.sort_values(['source', 'date']).reset_index(drop=True)
    
    # =========================================================================
    # PASS 1: Identify all corrupt data points
    # =========================================================================
    corrupt_records = []  # (source, date, reason)
    prev_data = {}  # source -> (prev_price, prev_mcap, prev_date)         [last VALID row]
    prev_row = {}   # source -> (prev_mcap, prev_shares)                   [ADJACENT row]
    
    for idx, row in df.iterrows():
        source = row.get('source', 'unknown')
        price = row.get(price_col, np.nan)
        mcap = row.get(mcap_col, np.nan)
        date = row.get('date', None)
        
        prev_price, prev_mcap, prev_date = prev_data.get(source, (None, None, None))
        # The ADJACENT preceding row (valid or not) -- what the market-cap step check
        # needs.  Tracked separately from prev_data (last VALID row) because a flagged
        # row must still advance the adjacent-row baseline, or one break cascades into
        # every later row.  A real capital raise / reverse split moves the share count
        # too, which is why the step check requires shares to be flat.
        prev_row_mcap, prev_row_shares = prev_row.get(source, (None, None))
        shares = row.get('weightedAverageShsOut', np.nan)

        is_valid, reason = check_price_sanity(
            row, price_col, mcap_col, prev_price, prev_mcap,
            prev_row_mcap, prev_row_shares
        )

        prev_row[source] = (mcap, shares)

        if not is_valid:
            corrupt_records.append({
                'source': source,
                'date': date,
                'price': price,
                'marketCap': mcap,
                'reason': reason,
            })
        else:
            if pd.notna(price) and price > 0:
                prev_data[source] = (price, mcap, date)
    
    # NO EARLY RETURN ON "no corruption found" (2026-08-05).  It used to return here, which
    # would have SKIPPED PASS 5 (the primary-presence eject) entirely on any panel whose
    # arithmetic checks all pass -- i.e. the eject would have been conditional on unrelated
    # corruption existing.  The passes below are each empty-safe instead.
    corrupt_df = pd.DataFrame(corrupt_records,
                              columns=['source', 'date', 'price', 'marketCap', 'reason'])
    if verbose and corrupt_df.empty:
        print("No corrupt data found (arithmetic checks).")

    # =========================================================================
    # PASS 2: For each ticker, find most recent corruption date
    # =========================================================================
    # Get the most recent (max) corruption date per ticker
    most_recent_corruption = ({} if corrupt_df.empty
                              else corrupt_df.groupby('source')['date'].max().to_dict())

    if verbose:
        print(f"\nTickers with corruption: {len(most_recent_corruption):,}")

    # =========================================================================
    # PASS 3: Remove all data at or before the most recent corruption date
    # =========================================================================
    removal_records = []
    rows_to_remove = set()
    
    for idx, row in df.iterrows():
        source = row.get('source', 'unknown')
        date = row.get('date', None)
        
        if source in most_recent_corruption:
            corruption_date = most_recent_corruption[source]
            
            # Remove all data at or before the corruption date
            if date <= corruption_date:
                rows_to_remove.add(idx)
                removal_records.append({
                    'source': source,
                    'date': date,
                    'price': row.get(price_col, np.nan),
                    'marketCap': row.get(mcap_col, np.nan),
                    'removal_reason': f"data_before_corruption (corruption at {corruption_date.date()})",
                })
    
    # Create intermediate filtered df
    valid_mask = ~df.index.isin(rows_to_remove)
    filtered_df = df[valid_mask].copy()
    
    # =========================================================================
    # PASS 4: Remove tickers with insufficient remaining data
    # =========================================================================
    ticker_counts = filtered_df.groupby('source').size()
    insufficient_tickers = ticker_counts[ticker_counts < min_periods_required].index.tolist()
    
    if insufficient_tickers:
        # Remove these tickers entirely
        for source in insufficient_tickers:
            source_rows = df[df['source'] == source]
            for idx, row in source_rows.iterrows():
                if idx not in rows_to_remove:  # Don't double-count
                    removal_records.append({
                        'source': source,
                        'date': row.get('date', None),
                        'price': row.get(price_col, np.nan),
                        'marketCap': row.get(mcap_col, np.nan),
                        'removal_reason': f"insufficient_data_after_corruption (<{min_periods_required} periods)",
                    })
        
        # Update filtered_df
        filtered_df = filtered_df[~filtered_df['source'].isin(insufficient_tickers)].copy()

    # =========================================================================
    # PASS 5: PRIMARY-PRESENCE EJECT  (nan-policy.md section 1a / ADDENDUM C)
    # =========================================================================
    # THE CEO'S FIRST TIER, AND IT IS DELIBERATELY *NOT* A NEW GATE.  "we should have some
    # columns such that if there are NaNs, we should just disqualify them."  The source-level
    # exclusion that can express that already exists -- it is this function -- and a second
    # gate is worse than either.  Stage-1 cannot express an eject at all: calcByTier returns a
    # PASS-RATE, so a NaN there is soft degradation (a name failing eight of eight rows on a
    # Tier-S criterion still scores, and four names reach the pool passing ZERO of eight on
    # net-debt-to-EBITDA).  That ruling is ring-fenced and is not reopened.
    #
    # FIVE RAW INPUTS, plus the two ARITHMETIC IMPOSSIBILITIES that `revenue` and `totalAssets`
    # were reclassified into (ADDENDUM C1).  The list, the conditions and the AS-OF-ROW reading
    # all live in `nan_policy` beside the reasoning; this site only applies the verdict and
    # logs it.
    #
    # MEASURED [panel = baseline_tools/resdic_2026-07-17_CORRECTED.pickle]: 117 sources
    # (1.51%) on the CFO limb, 13 on totalAssets <= 0, 36 on revenue < 0, every other limb 0 --
    # union 166 of 7,729 (2.15%) universe and **0 of the 100 deployed pool names**.  It is a
    # TRIPWIRE more than a filter: the eject already happens upstream for the other four limbs
    # (0.00% NaN on the newest row of all 7,729 surviving sources), and 2,738 of the 10,467
    # tickers in Tickers_df never reach Stage-1 at all.  What it buys is that the day a
    # provider gap DOES land on a primary input, the name leaves rather than being scored on
    # the pool.
    #
    # IDEMPOTENT: the ejected sources are gone from `filtered_df`, so a second invocation (this
    # function runs TWICE on the live path, Sbocker.py:490 and :547) finds nothing to remove.
    # LOUD FALLBACK, NOT A RAISE, and not a silent skip either.  `primary_eject` REFUSES a frame
    # that does not carry a primary input, because reporting "0 ejected" for a missing column is
    # a false negative.  But this function sits on the critical path of a ~12-hour run, and the
    # same trade-off already has a precedent in this repo (postBo's carve-out fallback): finish
    # the run, and make it impossible to miss that a tier of the filter did not apply.  The
    # banner deliberately says the output must not be treated as filtered.
    if not filtered_df.empty:
        try:
            _ej = npol.primary_eject(filtered_df, verbose=verbose)
        except Exception as _e:
            print("!" * 78, flush=True)
            print("!!! PRIMARY-PRESENCE EJECT DID NOT RUN (%s: %s)."
                  % (type(_e).__name__, _e), flush=True)
            print("!!! The CEO's first tier -- 'columns such that if there are NaNs, we should\n"
                  "!!! just disqualify them' -- was NOT applied to this frame. Names with an\n"
                  "!!! absent primary input are still in the universe. DO NOT treat this output\n"
                  "!!! as fully filtered.", flush=True)
            print("!" * 78, flush=True)
            _ej = pd.DataFrame()
        if len(_ej):
            # One reason string per source, naming EVERY limb that fired on it (a source can
            # fail more than one), then one removal record per REMOVED ROW -- the same
            # convention PASS 4 uses, so the summary's row count below stays honest and the
            # transparency CSV carries the attribution on every row it deleted.
            _reason = {}
            for _src, _grp in _ej.groupby('source'):
                _limbs = '; '.join(
                    '%s %s (value=%s)' % (r['field'], r['limb'],
                                          'NaN' if pd.isna(r['value']) else '%.6g' % r['value'])
                    for _, r in _grp.iterrows())
                _reason[_src] = 'primary_input_absent [%s]' % _limbs
            _mask = filtered_df['source'].isin(_reason)
            for _idx, _row in filtered_df[_mask].iterrows():
                removal_records.append({
                    'source': _row['source'],
                    'date': _row.get('date', None),
                    'price': _row.get(price_col, np.nan),
                    'marketCap': _row.get(mcap_col, np.nan),
                    'removal_reason': _reason[_row['source']],
                })
            filtered_df = filtered_df[~_mask].copy()
            if verbose:
                print("  primary-presence eject removed %d source(s) entirely (%d row(s))."
                      % (len(_reason), int(_mask.sum())))

    removed_df = pd.DataFrame(removal_records)

    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        n_total = len(df)
        n_removed = len(removal_records)
        n_kept = len(filtered_df)
        
        print(f"\n{'='*60}")
        print("DATA QUALITY FILTER APPLIED")
        print('='*60)
        print(f"Total rows: {n_total:,}")
        print(f"Rows removed: {n_removed:,} ({n_removed/n_total*100:.2f}%)")
        print(f"Rows kept: {n_kept:,}")
        
        # Count original corrupt points
        n_corrupt_points = len(corrupt_df)
        print(f"\nCorrupt data points detected: {n_corrupt_points:,}")
        
        # Breakdown by reason
        if not corrupt_df.empty:
            reason_counts = corrupt_df['reason'].str.split('(').str[0].value_counts()
            print(f"\nCorruption types:")
            for reason, count in reason_counts.items():
                print(f"  {reason.strip()}: {count:,}")
        
        # Tickers removed entirely
        original_tickers = df['source'].nunique()
        remaining_tickers = filtered_df['source'].nunique()
        removed_tickers = original_tickers - remaining_tickers
        
        print(f"\nTickers:")
        print(f"  Original: {original_tickers:,}")
        print(f"  Removed entirely: {removed_tickers:,}")
        print(f"  Remaining: {remaining_tickers:,}")
        
        if insufficient_tickers:
            print(f"\nTickers removed (insufficient data after corruption): {len(insufficient_tickers):,}")
            if len(insufficient_tickers) <= 20:
                print(f"  {insufficient_tickers}")
        
        print('='*60 + '\n')
    
    return filtered_df, removed_df


def save_removed_data(removed_df, filename=None):
    """
    Save removed data to CSV for transparency.
    
    Parameters:
    -----------
    removed_df : DataFrame
        DataFrame of removed rows from filter_invalid_data()
    filename : str, optional
        Custom filename. Default: removed_data_quality_YYYYMMDD.csv
    """
    if removed_df is None or removed_df.empty:
        return None
    
    if filename is None:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"removed_data_quality_{date_str}.csv"
    
    # Ensure output directory exists
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    removed_df.to_csv(filepath, index=False)
    
    print(f"Removed data logged to: {filepath}")
    
    return filepath


def apply_data_quality_filter(dmdic, verbose=True, save_log=True):
    """
    Apply data quality filter to the data dictionary.
    
    This should be called early in the pipeline, before any scoring.
    
    Parameters:
    -----------
    dmdic : dict
        Data dictionary with 'cdx_df' and 'BoMetric_df'
    verbose : bool
        Print summary
    save_log : bool
        Save removed data to CSV
    
    Returns:
    --------
    dict : Updated dmdic with filtered data
    """
    if 'cdx_df' not in dmdic:
        if verbose:
            print("Warning: No cdx_df in data dictionary, skipping quality filter")
        return dmdic
    
    # Filter cdx_df
    clean_cdx, removed_cdx = filter_invalid_data(
        dmdic['cdx_df'], 
        price_col='price', 
        mcap_col='marketCap',
        verbose=verbose
    )
    
    # Get list of affected sources (for filtering BoMetric_df consistently)
    if not removed_cdx.empty:
        # Get sources that had ALL their data removed (completely invalid)
        sources_in_clean = set(clean_cdx['source'].unique())
        sources_original = set(dmdic['cdx_df']['source'].unique())
        completely_removed = sources_original - sources_in_clean
        
        if verbose and len(completely_removed) > 0:
            print(f"Tickers completely removed (all data invalid): {len(completely_removed)}")
            if len(completely_removed) <= 20:
                print(f"  {list(completely_removed)}")
        
        # Save log
        if save_log:
            save_removed_data(removed_cdx)
    
    # Update dictionary
    dmdic['cdx_df'] = clean_cdx
    #
    # ACCUMULATE, NEVER ASSIGN (fixed 2026-08-07).
    #
    # This filter runs TWICE in a single pipeline run (Sbocker.py:490 and :554).
    # Pass 1 does the real work and records what it removed.  Pass 2 is correctly
    # IDEMPOTENT -- there is nothing left to remove -- so it returned an EMPTY frame,
    # and this line then OVERWROTE pass 1's record with it.  The shipped pickle
    # therefore asserted "data quality removed nothing" on the exact run where it had
    # removed 82 sources, which is why 3140 - 445 != 2613 could not be reconciled from
    # the artifact at all.  The idempotency was right; destroying the evidence of the
    # first pass was the bug.  Concatenating makes the record survive any number of
    # passes, and a second pass that genuinely removes nothing simply adds nothing.
    _prior_removed = dmdic.get('removed_data_quality')
    if _prior_removed is not None and len(_prior_removed) > 0:
        if removed_cdx is not None and len(removed_cdx) > 0:
            dmdic['removed_data_quality'] = pd.concat(
                [_prior_removed, removed_cdx], ignore_index=True)
        else:
            dmdic['removed_data_quality'] = _prior_removed
    else:
        dmdic['removed_data_quality'] = removed_cdx

    # Scalar counters stamped at the SAME site as the frame, so the reconciliation
    # 3140 - 445 - n_dq_removed_sources == 2613 can be checked straight off the
    # pickle without re-deriving anything.  `_dq_source_col` is resolved defensively:
    # the removed frame carries the source identifier under whichever column this
    # pipeline version uses, and a counter that raises would be worse than one that
    # reports None.
    try:
        _rem = dmdic.get('removed_data_quality')
        if _rem is not None and len(_rem) > 0:
            _src_col = next((c for c in ('source', 'symbol', 'ticker', 'source_id')
                             if c in _rem.columns), None)
            dmdic['n_dq_removed_rows'] = int(len(_rem))
            if _src_col:
                _srcs = sorted(set(_rem[_src_col].dropna().tolist()))
                dmdic['n_dq_removed_sources'] = len(_srcs)
                dmdic['dq_removed_source_list'] = _srcs
            else:
                dmdic['n_dq_removed_sources'] = None
                dmdic['dq_removed_source_list'] = []
        else:
            dmdic['n_dq_removed_rows'] = 0
            dmdic['n_dq_removed_sources'] = 0
            dmdic['dq_removed_source_list'] = []
        if verbose:
            print(f"[data_quality] cumulative removals this run: "
                  f"{dmdic.get('n_dq_removed_sources')} source(s), "
                  f"{dmdic.get('n_dq_removed_rows')} row(s) "
                  f"(accumulated across all filter passes)")
    except Exception as _e:
        if verbose:
            print(f"[data_quality] WARNING: removal counters not stamped: {_e}")


    # Also filter BoMetric_df -- by SOURCE *and* by ROW (audit H-1 fix, 2026-07-19).
    #
    # Only the source-level filter existed, so a ticker whose corrupt-era rows were
    # surgically removed from cdx_df kept those SAME quarters in BoMetric_df -- the
    # Stage-1 scoring frame.  Measured on the 2026-07-17 run: 3,522 rows across 551
    # sources that this filter had removed from cdx_df were still being scored (and still
    # feeding the cross-sectional median every mean-relative Stage-1 test compares
    # against).  "Partially cleaned" meant cleaned for Stage-2 and untouched for Stage-1.
    #
    # Row matching is by (source, date), the pair both frames are keyed on -- they are
    # built per ticker from the same date vector, and both go through
    # utils.setDatesToQuarterly.  SUBTRACTIVE by design: only pairs this pass explicitly
    # recorded in removed_cdx are dropped, so a BoMetric_df row with no cdx counterpart is
    # never silently deleted, and a second (idempotent) invocation removes nothing.
    if 'BoMetric_df' in dmdic and not clean_cdx.empty:
        valid_sources = set(clean_cdx['source'].unique())
        bm = dmdic['BoMetric_df']
        original_bm_len = len(bm)
        bm = bm[bm['source'].isin(valid_sources)].copy()
        n_after_source = len(bm)

        n_rowdrop = 0
        if (not removed_cdx.empty and 'date' in removed_cdx.columns
                and 'date' in bm.columns):
            rem_pairs = set(zip(removed_cdx['source'],
                                pd.to_datetime(removed_cdx['date'], errors='coerce')))
            bm_pairs = zip(bm['source'], pd.to_datetime(bm['date'], errors='coerce'))
            keep_row = np.fromiter((p not in rem_pairs for p in bm_pairs),
                                   dtype=bool, count=len(bm))
            n_rowdrop = int((~keep_row).sum())
            bm = bm[keep_row].copy()

        dmdic['BoMetric_df'] = bm

        if verbose:
            new_bm_len = len(bm)
            if original_bm_len != new_bm_len:
                print(f"BoMetric_df filtered: {original_bm_len:,} -> {new_bm_len:,} rows "
                      f"({original_bm_len - n_after_source:,} by removed SOURCE, "
                      f"{n_rowdrop:,} by removed ROW (source,date))")

    return dmdic


# Convenience function for standalone use
def filter_pickle_data(pickle_path, output_pickle_path=None, verbose=True):
    """
    Load a pickle, apply quality filter, optionally save cleaned version.
    
    Parameters:
    -----------
    pickle_path : str
        Path to input pickle
    output_pickle_path : str, optional
        Path for cleaned pickle. If None, doesn't save.
    verbose : bool
        Print details
    
    Returns:
    --------
    dict : Cleaned data dictionary
    """
    import pickle
    
    with open(pickle_path, 'rb') as f:
        dmdic = pickle.load(f)
    
    print(f"Loaded: {pickle_path}")
    
    dmdic = apply_data_quality_filter(dmdic, verbose=verbose, save_log=True)
    
    if output_pickle_path:
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(dmdic, f)
        print(f"Saved cleaned data to: {output_pickle_path}")
    
    return dmdic


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply data quality filter to pickle file")
    parser.add_argument('--input', '-i', required=True, help='Input pickle path')
    parser.add_argument('--output', '-o', help='Output pickle path (optional)')
    
    args = parser.parse_args()
    
    filter_pickle_data(args.input, args.output)
