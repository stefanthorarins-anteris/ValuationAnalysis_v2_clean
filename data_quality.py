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
import transfer_utils as _tu   # EVIDENCE_DIR: where the run's evidence CSVs are written


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


def guarded_row_pass(df, fn, banner_lines, verbose=True):
    """Apply ONE row-removing pass so that the rows and their RECORD move together.

    Returns `(df, records, removed_sources)` -- `df` unchanged and both lists empty if the
    pass failed.

    WHY THIS EXISTS (reviewer F-6, 2026-08-08).  PASS 0 and PASS 0b were each written as

        try:
            mask, records = <compute>
            if mask.any():
                df = df[~mask]          # <-- frame mutated
                <verbose printing>      # <-- can raise
        except Exception:
            records = []                # <-- record destroyed, rows already gone

    so anything raising AFTER the drop deleted rows while asserting that nothing had been
    removed -- names leaving the universe with no artifact naming them, which is the exact
    defect class this filter exists to make impossible.  (The faithful reproduction is a
    PRINT that raises -- a Windows console-codepage failure, say -- because the verbose
    printing was the only code between the mutation and the `except`.  Under the old shape
    that gave `clean=8, removed=0`: ten rows gone, no record.)

    TWO HALVES, AND BOTH ARE NOW ENFORCED RATHER THAN TRUSTED:

      1. ORDERING.  `fn` runs inside the guard and only computes; the mutation happens
         OUTSIDE the guard and only after the record is in hand; no printing happens
         anywhere near either.  There is no ordering left for a future editor to get wrong.
      2. COMPLETENESS (reviewer F-11, 2026-08-08).  The seam used to drop `mask.sum()` rows
         and return whatever `fn` handed back, UNCHECKED -- so an `fn` returning
         `(one_true_mask, [])` delivered F-6's exact harm THROUGH THE SEAM BUILT TO CLOSE
         IT.  Neither shipped `fn` does that, but "atomic by construction" was only half
         true while the record half was still by care.  A pass whose record does not cover
         every row it masks is now REFUSED.

    THE REFUSAL DIRECTION IS DELIBERATE.  On any violation the frame is returned UNCHANGED
    and both lists empty: not removing rows is visible and recoverable (the names are still
    there, and the banner says the pass did not apply), whereas removing them unrecorded is
    invisible and permanent.  Availability loses to integrity here, which is the whole
    point of F-6.

    `fn(df)` must return `(row_mask, [record])`, must emit ONE record per masked row, and
    must not mutate `df`.
    """
    try:
        mask, records = fn(df)
        records = list(records or [])
    except Exception as e:
        print("!" * 78, flush=True)
        for line in banner_lines:
            print(line % {'err': '%s: %s' % (type(e).__name__, e)}
                  if '%(err)s' in line else line, flush=True)
        print("!" * 78, flush=True)
        return df, [], []

    n_masked = 0 if (mask is None or not len(mask)) else int(mask.sum())

    #  THE COMPLETENESS CHECK, BEFORE ANY ROW MOVES.  Checked in BOTH directions: records
    #  without a mask is a record of a removal that never happened, which corrupts the
    #  reconciliation just as surely as the reverse.
    if len(records) != n_masked:
        print("!" * 78, flush=True)
        print("!!! A ROW-REMOVING PASS DID NOT RECORD WHAT IT REMOVED -- PASS REFUSED.",
              flush=True)
        print("!!!   rows masked for removal : %d" % n_masked, flush=True)
        print("!!!   removal records returned: %d" % len(records), flush=True)
        print("!!! Every removed row must carry a record, or names leave the universe with\n"
              "!!! no artifact naming them and the reconciliation cannot balance. NOTHING\n"
              "!!! was removed by this pass; the rows are STILL IN THE PANEL. Fix the pass\n"
              "!!! -- do NOT relax this check.", flush=True)
        for line in banner_lines:
            if '%(err)s' not in line:
                print(line, flush=True)
        print("!" * 78, flush=True)
        return df, [], []

    if not n_masked:
        return df, [], []

    removed_sources = sorted(set(df.loc[mask, 'source'])) if 'source' in df.columns else []
    df = df[~mask].reset_index(drop=True)
    return df, records, removed_sources


def filter_invalid_data(cdx_df, price_col='price', mcap_col='marketCap',
                        min_periods_required=8, verbose=True, sanity_refusals=None):
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
    sanity_refusals : DataFrame, optional
        The run's `inputSanityRefusals` report (`nan_policy.refuse_impossible_cells`).  The
        first pass below RESTORES the pre-refusal values from it for the duration of its own
        row checks, and nothing else reads it.  REQUIRED FOR CORRECTNESS, not a nicety -- the
        reasoning and the measured before/after live beside the restoration itself.  Absent,
        that pass announces it is running blind rather than silently skipping the checks whose
        inputs were blanked.

        THIS DOCSTRING DELIBERATELY NAMES NEITHER THE PASS NUMBERS NOR THE ROW-CHECK FUNCTION.
        `test_vendor_contamination.test_the_quarantine_runs_BEFORE_the_arithmetic_checks` reads
        this function's SOURCE and asserts a textual ORDERING over those exact tokens; a
        parameter doc sits above every pass, so mentioning one here inverts the ordering and
        breaks a real guard to make room for a comment.  Two drafts of this note did exactly
        that.  Add no such token above the body.

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

    # The source set AS IT ARRIVED, captured before PASS 0/0b drop rows from `df`.  The
    # summary's "Tickers: Original / Removed entirely" lines are derived from this, not
    # from the post-pass frame -- otherwise a source removed ENTIRELY by PASS 0b (the
    # reporting-currency exclusion) would silently shrink the "Original" count and report
    # itself as zero tickers removed.
    sources_in = set(df['source']) if 'source' in df.columns else set()

    # =========================================================================
    # PASS 0: VENDOR-CONTAMINATION QUARANTINE  (vendor_contamination.py)
    # =========================================================================
    # NAMED, DATED, EVIDENCED windows where FMP serves ANOTHER ISSUER'S statements under
    # this ticker.  The founding case is `058820.KQ` (CMG Pharmaceutical, KOSDAQ) carrying
    # CHIPOTLE's income statement and balance sheet for 2020-03-31 -> 2022-09-30, labelled
    # KRW and matching CMG to the dollar, then SNAPPING to genuine KRW at 2022-12-31.
    #
    # WHY IT IS *HERE* AND NOT IN A FILTER OF ITS OWN.  It is a DATA-SIDE rule and it must
    # survive the next full fetch -- FMP still serves the bad rows today, so a re-fetch
    # re-ingests them verbatim.  Putting it in this function means it inherits, for free,
    # every property the removal machinery already has: the rows land in the transparency
    # CSV with a reason string, they propagate to BoMetric_df by (source, date) through
    # the row-level filter below, and the whole thing is idempotent across this function's
    # two invocations per run because the rows are simply gone the second time.
    #
    # IT RUNS FIRST, BEFORE THE ARITHMETIC CHECKS, ON PURPOSE.  The contaminated rows would
    # otherwise be the ADJACENT PRECEDING ROW for the market-cap step check, i.e. corrupt
    # data acting as the baseline that decides whether real data looks like a break.
    # Removing a leading window also leaves a contiguous DATE SUFFIX, which is exactly the
    # shape PASS 3 already guarantees, so nothing downstream sees a new adjacency.
    #
    # NOT CAUGHT BY ANY EXISTING CHECK, and could not be: `marketCap` runs CONTINUOUS in
    # real KRW straight through the window, so every market-cap-based sanity rule in this
    # module passes, and the 13x scale break at the boundary is an order of magnitude below
    # _MCAP_BREAK_RATIO (and is in the wrong field anyway).
    #
    # ATOMIC BY CONSTRUCTION (reviewer F-6, fixed 2026-08-08): `guarded_row_pass` computes
    # the record inside the guard and mutates the frame outside it, so rows can never leave
    # while their record is discarded.
    #
    # WHAT IS AND IS NOT NON-FATAL, STATED EXACTLY (reviewer F-12, 2026-08-08).  This comment
    # used to say "LOUD, never silent, never fatal", which was true of the old shape and is
    # now true of only HALF the code.  The line matters because it is read as a promise about
    # a ~12-hour run, so it must not overstate:
    #
    #   * THE `fn` LIMB IS NON-FATAL.  A failure computing the mask or the records (a bad
    #     rule, a missing module, an unreadable column) is caught, announced with the banner
    #     below, and the run continues with the rows STILL PRESENT.  Same trade-off as the
    #     primary-presence eject.
    #   * EVERYTHING AFTER THE GUARD IS DELIBERATELY FATAL.  `sorted(set(...))` over the
    #     `source` column, the drop itself, and the verbose print blocks below all sit
    #     OUTSIDE any handler, and are genuinely fatal -- verified for a raising print (a
    #     Windows console-codepage failure) and for `sorted()` on a mixed-type `source`
    #     column.  THIS IS THE FIX, NOT AN OVERSIGHT: the print is exactly what used to
    #     raise between the mutation and the `except` and silently discard the record, so
    #     re-wrapping this region would walk F-6 straight back.  A crash here loses the run
    #     and loses nothing else; the old behaviour lost ten rows and told nobody.
    #     Integrity over availability, chosen on purpose.
    def _quarantine(frame):
        import vendor_contamination as vc
        return vc.quarantine_records(frame, price_col=price_col, mcap_col=mcap_col)

    df, quarantine_records, _q_src = guarded_row_pass(
        df, _quarantine,
        ["!!! VENDOR-CONTAMINATION QUARANTINE DID NOT RUN (%(err)s).",
         "!!! Known-contaminated vendor rows (another issuer's statements served\n"
         "!!! under this ticker) are STILL IN THIS PANEL. The backtest reads the\n"
         "!!! affected window. DO NOT treat this output as quarantined."],
        verbose=verbose)

    if verbose and _q_src:
        import vendor_contamination as _vc_p
        print("VENDOR-CONTAMINATION QUARANTINE: removed %d row(s) across %d "
              "source(s): %s" % (len(quarantine_records), len(_q_src),
                                 ', '.join(_q_src)))
        for _r in _vc_p.QUARANTINE_RULES:
            if _r.source in _q_src:
                print("  %s" % _r.label())

    # =========================================================================
    # PASS 0b: REPORTING-CURRENCY EXCLUSION  (currency_exclusions.py)
    # =========================================================================
    # NAMED, DATED, EVIDENCED reporting currencies whose STATEMENTS are refused outright.
    # Today that is `ARS`, on three grounds stated in full in `currency_exclusions`: this
    # vendor is wrong on ARS by three orders of magnitude on BMA in this very panel; no ARS
    # name can be valued in USD at all (no admitted rate, so no band and a NEUTRAL size
    # tilt); and separating a correctly IAS 29-restated series from a mishandled one needs
    # a per-name audit this codebase does not do.
    #
    # WHY IT IS AN EXCLUSION AND NOT THE FX ABSTENTION IT REPLACES (CEO, 2026-08-08).
    # `fx_rates.ABSTAIN_CURRENCIES` refuses the ARS->USD rate and KEEPS the name, scored
    # neutral on the size tilt.  That fixes the CURRENCY.  On BMA the statements themselves
    # are wrong by three orders of magnitude, so every metric downstream of them is
    # contaminated -- and the name goes on feeding the cross-sectional medians that every
    # mean-relative Stage-1 test scores every OTHER company against.  The abstention is kept
    # as the backstop for paths that never reach this filter (see the concrete list in
    # `currency_exclusions`: `getData_fmp` converts on the RAW panel before this filter runs,
    # and four `baseline_tools` entry points never import `data_quality` at all).
    #
    # SOURCE-SCOPED, so the whole name leaves -- which is also what keeps the universe
    # reconciliation balanced: an entirely-removed source drops out of the panel and lands
    # in `removed_df`, so `resolved == panel + failed + removed + residual` still holds.
    # (See apply_data_quality_filter, where the removed-source counter is derived from the
    # sources that actually LEFT rather than from every source with a removed row.)
    #
    # WHY IT RUNS HERE, CORRECTED (reviewer F-3, 2026-08-08).  This block used to claim
    # PASS 0's rationale -- "an ARS row would otherwise be the adjacent preceding row for
    # the market-cap step check".  THAT MECHANISM CANNOT OPERATE HERE: PASS 1 keys both of
    # its baselines (`prev_data`, `prev_row`) strictly PER SOURCE, and 0b removes the source
    # ENTIRELY, so no ARS row is ever adjacent to another name's row.  The rationale was
    # copied and was wrong.
    # THE ORDER IS STILL RIGHT, for a different and checkable reason: if an excluded source
    # also trips an arithmetic check, PASS 2/3 would record `data_before_corruption` rows
    # for rows PASS 0b has already recorded as `currency_excluded` -- the same rows logged
    # twice under two different reasons, inflating `n_dq_removed_rows` and making the
    # transparency CSV self-contradictory about why a name left.  Running the exclusion
    # first makes those rows absent from `df` before PASS 1 ever iterates.
    #
    # A PANEL WITHOUT `reportedCurrency` CANNOT BE FILTERED THIS WAY, and that fact now
    # reaches an ARTIFACT, not just the console (reviewer F-5): `verbose=False` callers used
    # to lose it entirely.
    #
    # ATOMICITY: the same `guarded_row_pass` seam as PASS 0 -- see its docstring, and see
    # PASS 0's note on exactly which half is non-fatal.  Same split here: the `fn` limb is
    # caught and announced; the drop and the verbose block below are deliberately fatal.
    _cx_status = []
    try:
        import currency_exclusions as cx
        _cx_status = cx.status_rows(df)
    except Exception as _se:
        _cx_status = [{'status': 'ERROR', 'currency': '', 'source': '',
                       'n_rows_in_currency': '', 'n_rows_total': '', 'minority_label': '',
                       'watched_currencies': '',
                       'note': '%s: %s' % (type(_se).__name__, _se)}]

    def _currency_exclude(frame):
        import currency_exclusions as _cx
        return _cx.exclusion_records(frame, price_col=price_col, mcap_col=mcap_col)

    df, currency_excl_records, _c_src = guarded_row_pass(
        df, _currency_exclude,
        ["!!! REPORTING-CURRENCY EXCLUSION DID NOT RUN (%(err)s).",
         "!!! Names reporting in a currency whose STATEMENTS this pipeline refuses\n"
         "!!! (ARS) are STILL IN THIS PANEL and are being scored. DO NOT treat this\n"
         "!!! output as currency-filtered."],
        verbose=verbose)

    if verbose:
        if _c_src:
            print("REPORTING-CURRENCY EXCLUSION: removed %d source(s) entirely "
                  "(%d row(s)): %s"
                  % (len(_c_src), len(currency_excl_records), ', '.join(_c_src)))
            for _row in _cx_status:
                if _row.get('status') == 'EXCLUDED':
                    print("  %s: %s on %s/%s rows%s"
                          % (_row['source'], _row['currency'],
                             _row['n_rows_in_currency'], _row['n_rows_total'],
                             '  <-- MINORITY LABEL, possible stray vendor value'
                             if _row.get('minority_label') else ''))
        elif _cx_status and _cx_status[0].get('status') == 'NOT_APPLIED':
            print("REPORTING-CURRENCY EXCLUSION: NOT APPLIED -- %s. Watched currencies "
                  "(%s) are still present in this frame if it contains any."
                  % (_cx_status[0].get('note', ''),
                     _cx_status[0].get('watched_currencies', '')))

    # =========================================================================
    # PASS 1: Identify all corrupt data points
    # =========================================================================
    corrupt_records = []  # (source, date, reason)
    prev_data = {}  # source -> (prev_price, prev_mcap, prev_date)         [last VALID row]
    prev_row = {}   # source -> (prev_mcap, prev_shares)                   [ADJACENT row]

    #  ---- SECTION-5 RESTORATION, AND WHY THIS GATE MUST NOT SEE A REFUSAL --------------
    #  A section-5 refusal blanks a cell so the SCORER abstains on it.  This gate is not a
    #  scorer: it DELETES corrupt history, and every check below is `pd.notna`-guarded, so a
    #  blanked input does not make a check abstain -- it makes the check PASS.  Since the
    #  price-scale rule (2026-09-01) blanks `price`, `marketCap` and `earningsYield`, checks
    #  3, 5 and 6 would skip on exactly the rows most likely to be corrupt, and PASS 3 would
    #  stop removing the prefix it used to remove.  Measured on an ATRI-shape fixture: two
    #  `mcap_step_break` flags before, none after; rows PASS 3 would delete 7 -> 0.
    #  So PASS 1 -- AND ONLY PASS 1 -- reads the pre-refusal values back out of the run's own
    #  refusal report.  Nothing is written back into the frame; the scorer still sees NaN.
    _restore = npol.refusal_restore_map(sanity_refusals)
    _n_restored = 0
    _occ_seen = {}
    if (sanity_refusals is not None and len(sanity_refusals)
            and npol.SANITY_REFUSED_COLUMN not in df.columns and verbose):
        #  A REPORT WITH NO STAMP COLUMN restores nothing, and the branch below cannot say so
        #  because it is gated on the column being present.  Silent in that combination until
        #  now; it means the frame lost the stamp somewhere between the refusal and here,
        #  which is precisely the failure the ingest-passthrough fix was written for.
        print('!!! DATA-QUALITY: a section-5 refusal report with %d record(s) was supplied but '
              'the frame carries NO `%s` column, so NOTHING can be restored and checks 3/5/6 '
              'will skip on every refused row. The stamp was lost upstream.'
              % (len(sanity_refusals), npol.SANITY_REFUSED_COLUMN), flush=True)
    if not _restore and npol.SANITY_REFUSED_COLUMN in df.columns:
        _stamped = int(df[npol.SANITY_REFUSED_COLUMN].astype(str).str.len().gt(0).sum())
        if _stamped and verbose:
            #  LOUD, NEVER SILENT.  "no refusals happened" and "the report was not passed to
            #  me" are different statements about the data and must never print the same.
            print('!!! DATA-QUALITY PASS 1 IS RUNNING BLIND ON %d REFUSED ROW(S): the frame '
                  'carries section-5 refusal stamps but no refusal report was supplied, so '
                  'checks 3/5/6 will SKIP on those rows and any prefix removal they would '
                  'have triggered will NOT happen. Pass `sanity_refusals=` (the run keeps it '
                  'as `inputSanityRefusals`).' % _stamped, flush=True)

    for idx, row in df.iterrows():
        source = row.get('source', 'unknown')
        date = row.get('date', None)
        #  Restore ONLY for the checks, ONLY on rows that were actually refused, and ONLY
        #  the fields THIS ROW's own stamp names.
        #
        #  THE STAMP IS WHAT MAKES THE KEY SAFE.  `(source, date)` IS NOT UNIQUE on this
        #  panel -- 296 duplicate-key rows across 76 sources on the 2026-08-11 CUR3K panel,
        #  AAPL among them -- so a map keyed on it alone would restore one row's pre-refusal
        #  values onto a DIFFERENT row that shares its date and was never refused, handing
        #  the checks below numbers that do not belong to the row they are judging.  No
        #  refusal intersects a duplicate key today, so this is latent rather than live; it
        #  is closed rather than documented because the fix is free.  `sanityRefusedFields`
        #  is written PER ROW by section 5, so intersecting the map with it restores exactly
        #  the cells that were actually blanked on this row and nothing else.
        #  The occurrence index disambiguates DUPLICATE (source, date) rows -- see
        #  `nan_policy.refusal_restore_map`.  Counted in frame order on both sides.
        _k = (str(source), npol._normalise_refusal_date(date))
        _occ_seen[_k] = _occ_seen.get(_k, -1) + 1
        _rest = _restore.get(_k + (_occ_seen[_k],)) if _restore else None
        if _rest:
            _stamped = {t for t in str(row.get(npol.SANITY_REFUSED_COLUMN, '') or '').split(
                npol._SANITY_REFUSED_SEP) if t}
            _rest = {k: v for k, v in _rest.items() if k in _stamped} if _stamped else {}
        if _rest:
            row = dict(row)
            row.update(_rest)
            _n_restored += 1
        price = row.get(price_col, np.nan)
        mcap = row.get(mcap_col, np.nan)
        
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
            #  `price`/`marketCap` here are the RESTORED values on a restored row -- the
            #  ones that actually tripped the check.  `corrupt_df` feeds the verbose summary
            #  only; the artifact a reader keeps is `removed_df`, and the explanation is
            #  attached there (see `refused_fields` in the removal records below).
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
    if verbose and _restore:
        print('Section-5 restoration: PASS 1 evaluated %d refused row(s) against their '
              'PRE-REFUSAL values (%d row(s) in the refusal report).'
              % (_n_restored, len(_restore)), flush=True)
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
    # SEEDED WITH PASS 0, not re-initialised.  The quarantine's rows must reach the same
    # `removed_df` every other removal reaches -- that frame is what drives the
    # transparency CSV, the (source, date) propagation into BoMetric_df, and the
    # `n_dq_removed_*` counters the universe reconciliation balances against.  Dropping
    # them here would delete the rows while asserting nothing had been removed, which is
    # precisely the accumulate-never-assign defect fixed on 2026-08-07 further down.
    # PASS 0b joins it for the identical reason: an excluded name that left the panel
    # without a row in `removed_df` would be a name that vanished with no artifact naming
    # it, and it would break the universe reconciliation on the way out.
    removal_records = list(quarantine_records) + list(currency_excl_records)
    rows_to_remove = set()

    for idx, row in df.iterrows():
        source = row.get('source', 'unknown')
        date = row.get('date', None)

        if source in most_recent_corruption:
            corruption_date = most_recent_corruption[source]
            
            # Remove all data at or before the corruption date
            if date <= corruption_date:
                rows_to_remove.add(idx)
                #  WHY `refused_fields` IS ON EVERY REMOVAL RECORD.  This frame IS the
                #  transparency CSV.  On a row section 5 refused, the `price` and `marketCap`
                #  columns below read NaN -- while the corruption that caused the removal was
                #  found on the PRE-REFUSAL values restored in the first pass.  Two different
                #  numbers for one (source, date), with nothing saying why, is how a reader
                #  concludes the CSV is broken.  Naming the refused fields ON THE ROW makes
                #  the NaN self-explaining and reconciles the two halves.  Empty string on
                #  every ordinary row, so nothing else in the file changes shape.
                removal_records.append({
                    'source': source,
                    'date': date,
                    'price': row.get(price_col, np.nan),
                    'marketCap': row.get(mcap_col, np.nan),
                    'refused_fields': str(row.get(npol.SANITY_REFUSED_COLUMN, '') or ''),
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
                        'refused_fields': str(row.get(npol.SANITY_REFUSED_COLUMN, '') or ''),
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
                    'refused_fields': str(_row.get(npol.SANITY_REFUSED_COLUMN, '') or ''),
                    'removal_reason': _reason[_row['source']],
                })
            filtered_df = filtered_df[~_mask].copy()
            if verbose:
                print("  primary-presence eject removed %d source(s) entirely (%d row(s))."
                      % (len(_reason), int(_mask.sum())))

    removed_df = pd.DataFrame(removal_records)
    #  UNIFORM, NOT RAGGED, AND ALWAYS PRESENT.  PASS 0 (quarantine), PASS 0b (currency) and
    #  PASS 5 (primary-presence eject) build records without `refused_fields`, so
    #  `pd.DataFrame` would leave NaN there while the arithmetic passes write ''.  Worse, on a
    #  run where ONLY those passes remove anything the column would be ABSENT ENTIRELY and the
    #  transparency CSV's header would vary between runs.  A column that means "nothing was
    #  refused" in three spellings -- '', NaN, and missing -- is a column a reader has to
    #  guess at, so it is created unconditionally and normalised once.
    if len(removed_df):
        if 'refused_fields' not in removed_df.columns:
            removed_df['refused_fields'] = ''
        removed_df['refused_fields'] = removed_df['refused_fields'].fillna('').astype(str)

    # The PASS 0b status rides out on the frame's `attrs` (the same channel the Stage-2
    # frames use to declare their basis) so that `apply_data_quality_filter` -- which is
    # where this module is allowed to touch the filesystem -- can write it to
    # `CurrencyExclusionStatus_<date>.csv` (repo root since 2026-08-10).  A pure filter must not write files;
    # a status that only exists in a console line is the asymmetry F-5 is about.
    try:
        removed_df.attrs['currency_exclusion_status'] = list(_cx_status)
    except Exception:
        pass

    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        # n_total counts the frame AS IT ARRIVED (passes 0 and 0b have already dropped
        # their rows from `df`), so the removed-percentage stays honest rather than quietly
        # excluding them from their own denominator.
        n_total = len(df) + len(quarantine_records) + len(currency_excl_records)
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
        
        # Tickers removed entirely -- measured against the ARRIVING source set, so the
        # PASS 0b exclusions are counted rather than erased from their own denominator.
        original_tickers = len(sources_in)
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


def run_identifier(dmdic):
    """A short string identifying the RUN a removal record came from.

    Reviewer F-9: `removed_data_quality_*.csv` carried a timestamp and nothing else, so a
    file could not be tied to the run that produced it -- and this directory now receives
    removal records from the pipeline AND from a standalone `backtest_unified` invocation,
    which are not the same thing.  Built from the stamps the run already carries
    (`universes.provenance` -> configdic -> dmdic); never re-derived, never guessed, and
    explicitly `unknown-unstamped-run` when the panel predates stamping."""
    if not isinstance(dmdic, dict):
        return 'unknown-unstamped-run'
    universe = dmdic.get('universe') or dmdic.get('tickerfilter')
    fingerprint = dmdic.get('universe_fingerprint')
    if not universe and not fingerprint:
        return 'unknown-unstamped-run'
    return '%s@%s' % (universe or 'unknown', fingerprint or 'unstamped')


def save_removed_data(removed_df, filename=None, run_id=''):
    """
    Save removed data to CSV for transparency.

    Parameters:
    -----------
    removed_df : DataFrame
        DataFrame of removed rows from filter_invalid_data()
    filename : str, optional
        Custom filename. Default: removed_data_quality_YYYYMMDD.csv
    run_id : str, optional
        Run identifier stamped into every row (see run_identifier).
    """
    if removed_df is None or removed_df.empty:
        return None

    if filename is None:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"removed_data_quality_{date_str}.csv"

    # AT THE REPO ROOT since 2026-08-10 (CEO), not in `output/`: this file names the sources
    # the data-quality filter REMOVED and the reason for each, so it is the sole on-disk
    # record of a decision about the universe -- and `output/` demonstrably did not reach the
    # other machine on the 2026-08-10 run.  See `transfer_utils.EVIDENCE_DIR`.
    output_dir = _tu.EVIDENCE_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #  The run stamp goes in as the FIRST column, on a copy -- the caller's frame is not
    #  mutated by having been logged.
    out = removed_df.copy()
    if 'run_id' not in out.columns:
        out.insert(0, 'run_id', run_id or 'unknown-unstamped-run')

    filepath = os.path.join(output_dir, filename)
    out.to_csv(filepath, index=False)

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
        verbose=verbose,
        #  THE RUN'S OWN REFUSAL REPORT.  Without it PASS 1 cannot see the values
        #  section 5 blanked, and it silently stops deleting rows it used to delete --
        #  see `nan_policy.refusal_restore_map`.  `.get`, not `[]`: a panel loaded with
        #  `-loadbometric` predates the key, and PASS 1 announces the blindness itself.
        sanity_refusals=dmdic.get('inputSanityRefusals'),
    )
    
    _run_id = run_identifier(dmdic)

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
            save_removed_data(removed_cdx, run_id=_run_id)

    # THE REPORTING-CURRENCY STATUS SHIPS ON EVERY PASS, INCLUDING THE PASSES WHERE THE
    # RULE COULD NOT RUN (reviewer F-5).  It is written OUTSIDE the `if not removed_cdx.
    # empty` block above on purpose: the case this exists to record -- a panel with no
    # `reportedCurrency` column -- removes nothing, so gating it on a non-empty removal
    # frame would drop exactly the fact it is meant to carry.
    if save_log:
        try:
            import currency_exclusions as _cx_w
            _status = (removed_cdx.attrs.get('currency_exclusion_status')
                       if hasattr(removed_cdx, 'attrs') else None)
            if _status:
                _cx_w.write_status(_status, run_id=_run_id)
        except Exception as _we:
            if verbose:
                print(f"[data_quality] WARNING: currency-exclusion status not written: {_we}")

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
    # pickle without re-deriving anything.  `_src_col` is resolved defensively:
    # the removed frame carries the source identifier under whichever column this
    # pipeline version uses, and a counter that raises would be worse than one that
    # reports None.
    #
    # `n_dq_removed_sources` COUNTS THE SOURCES THAT ACTUALLY LEFT (fixed 2026-08-08).
    #
    # It used to count every source with ANY removed row -- a different quantity the moment
    # a removal is PARTIAL.  `Sbocker.print_universe_reconciliation` balances
    #
    #     resolved == panel + fetch_failures + filter_removals + residual
    #
    # and that identity is about names LEAVING THE UNIVERSE.  A partially-removed source is
    # in the panel AND in the removed frame, so counting it here subtracts it twice and
    # drives the residual NEGATIVE.
    #
    # MEASURED on the 2026-08-07 CUR3K panel under the current tree: the old definition
    # counts **108** sources against a true **84**, i.e. residual **-24**, and fires the
    # "UNIVERSE DOES NOT RECONCILE -- 24 name(s) UNACCOUNTED FOR" banner on a run that
    # reconciles perfectly.  A reconciliation that cries wolf is worse than none, because
    # the next REAL residual is then read as the same known noise.
    #
    # THE MISCOUNT IS NOT A QUARANTINE ARTEFACT AND LONG PREDATES IT (corrected 2026-08-08,
    # reviewer F-2; an earlier version of this comment said "83 against a true 82" and
    # blamed 817df52 -- both wrong, and the wrong attribution was relayed upward).  Of the
    # 24 partially-removed sources, **23 are ordinary `data_before_corruption` PASS-3
    # trims** -- a source loses its pre-corruption rows and survives with the rest -- and
    # only **1** is the `058820.KQ` quarantine.  PASS 3 has behaved this way for as long as
    # it has existed, so the counter has been wrong on every panel with any partial trim;
    # the quarantine merely added a 24th case.  What actually kept it invisible until now is
    # the accumulate-never-assign bug fixed on 2026-08-07, which destroyed the removal frame
    # before anything could count it.
    #
    # Partial removals are NOT swept under the rug -- they get their own counter and list
    # below.  They are real removals with real evidence in the transparency CSV; they are
    # simply not universe EXITS, which is the only thing the identity is about.
    try:
        _rem = dmdic.get('removed_data_quality')
        _panel_srcs = set()
        _clean = dmdic.get('cdx_df')
        if _clean is not None and len(_clean) and 'source' in _clean.columns:
            _panel_srcs = set(_clean['source'].dropna().tolist())
        if _rem is not None and len(_rem) > 0:
            _src_col = next((c for c in ('source', 'symbol', 'ticker', 'source_id')
                             if c in _rem.columns), None)
            dmdic['n_dq_removed_rows'] = int(len(_rem))
            if _src_col:
                _touched = set(_rem[_src_col].dropna().tolist())
                _gone = sorted(_touched - _panel_srcs)
                _partial = sorted(_touched & _panel_srcs)
                dmdic['n_dq_removed_sources'] = len(_gone)
                dmdic['dq_removed_source_list'] = _gone
                dmdic['n_dq_partially_removed_sources'] = len(_partial)
                dmdic['dq_partially_removed_source_list'] = _partial
            else:
                dmdic['n_dq_removed_sources'] = None
                dmdic['dq_removed_source_list'] = []
                dmdic['n_dq_partially_removed_sources'] = None
                dmdic['dq_partially_removed_source_list'] = []
        else:
            dmdic['n_dq_removed_rows'] = 0
            dmdic['n_dq_removed_sources'] = 0
            dmdic['dq_removed_source_list'] = []
            dmdic['n_dq_partially_removed_sources'] = 0
            dmdic['dq_partially_removed_source_list'] = []
        if verbose:
            print(f"[data_quality] cumulative removals this run: "
                  f"{dmdic.get('n_dq_removed_sources')} source(s) removed ENTIRELY "
                  f"(the universe-reconciliation figure), "
                  f"{dmdic.get('n_dq_partially_removed_sources')} source(s) PARTIALLY "
                  f"trimmed and still in the panel, "
                  f"{dmdic.get('n_dq_removed_rows')} row(s) total "
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
