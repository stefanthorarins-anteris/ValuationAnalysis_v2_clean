"""
forensicFlags.py

Promotes the pipeline's already-computed forensic signals from *decoration* to
*decision-support* on the top-N shortlist. It is deliberately OFFLINE and makes
NO API calls: it consumes artifacts already present in resdic
(postRank, cdx_df, and the detectManipulation output: mscore_df / cscore_df /
SLmeanMscore / SLmeanCscore / problemlist_*), plus the local sector pickle.

Guiding principle (CEO): FLAGS, NOT VERDICTS. Nothing here drops a name from the
top-20. Beneish/Altman/accruals are false-positive-prone (esp. high-growth) and
INVALID for financials (banks/insurers/REITs); the CEO's manual review is the gate.
This module only surfaces the risk clearly, with the DRIVER (which components), so
the CEO sees why a name flagged, not just a number.

Signals surfaced per name:
  - Beneish M-score flag  (M > -1.78, i.e. the stored M_Score_mean > 0) + the
    components DRIVING M upward (adverse-direction decomposition around neutral,
    annotated with the underlying ratio direction so negative-coefficient
    components like SGAI/LVGI don't read as inverted).
  - Montier C-score flag  (C >= 4) + which of the 6 red flags fired (the fired set
    is computed consistently with the C_Score it sits beside).
  - Sloan accruals ratio  (NI - CFO)/avg(TotalAssets); flag = worst quintile WITHIN
    the shortlist (a within-run RANK artifact -- always flags ~20% -- not an
    absolute accruals red line; the raw value is shown for magnitude).
  - Financial (bank/insurer/REIT) indicator -> forensic readings INVALID for these.
    Classification is conservative and cross-checks TWO sector sources (the local
    pickle and, via applySectorFallback, the authoritative API/CSV sector): if
    EITHER says financial, the name is forensic-invalid.
  - Summary tag: clean / single-flag: dig-deeper / multi-flag: concern
    (financials are tagged forensic-invalid and NOT counted).
"""

import os
import warnings
import numpy as np
import pandas as pd

from detectManipulation import (invrollsumTTM, _toNewestFirst,
                                C_FLAG_CUTOFF, C_FLAG_COLS)
import reporting_period as rp

# --- Beneish component decomposition constants -------------------------------
# Coefficients as used in detectManipulation.calcBeneishM (the +1.78 shift there
# folds the -1.78 manipulator cutoff into the stored M_Score, so stored M > 0 ==
# standard M > -1.78). Neutral (no-manipulation) benchmark is 1.0 for the ratio
# indices and 0.0 for TATA. contribution_vs_neutral = coef * (value - neutral);
# a POSITIVE contribution pushes M up toward the manipulator cutoff, so those are
# the components "driving" a flag. This is an exact linear decomposition of M
# around the neutral baseline -- a directional driver read, not a per-component
# threshold (we do NOT claim any single component independently signals fraud).
M_COMPONENTS = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']
M_COEF = {'DSRI': 0.92, 'GMI': 0.528, 'AQI': 0.404, 'SGI': 0.892,
          'DEPI': 0.115, 'SGAI': -0.172, 'LVGI': -0.327, 'TATA': 4.679}
M_NEUTRAL = {'DSRI': 1.0, 'GMI': 1.0, 'AQI': 1.0, 'SGI': 1.0,
             'DEPI': 1.0, 'SGAI': 1.0, 'LVGI': 1.0, 'TATA': 0.0}
# Quarterly defaults; both are scaled per source by rp.scale_window(.., rpy) so they
# cover the same CALENDAR span for a semi-annual filer (2 rows / 1 row) and match the
# window detectManipulation actually averaged over.
M_WINDOW = 4  # matches SLmeanMscore = M_Score.head(scale_window(4, rpy)).mean()

# Montier C-score red flags (each column > 0 counts one flag per period in
# calcMontierC, and C_Score_mean averages those per-period counts over head(2)).
C_FLAGS = C_FLAG_COLS          # imported: ONE definition of the six flag columns
C_WINDOW = 2  # matches SLmeanCscore = C_Score.head(scale_window(2, rpy, min=1)).mean()
# THE surface-for-review flag is C >= C_FLAG_CUTOFF (a review flag, not an auto-drop,
# so higher sensitivity is intended).  The cutoff is now IMPORTED from
# detectManipulation, which is also what its own problemlist_Cscore uses -- previously
# this module used `>= 4` while detectManipulation used a stricter `> 4`, and the
# resulting `C_flag_ge_4` / `legacyProblemC_strict_gt4` columns CONTRADICTED each other
# in the same CSV row for every name scoring exactly 4.0 (12 of 90 on 2026-07-17).

# Sectors for which Beneish / Altman / Sloan-accruals are INVALID (business model
# differs: banks/insurers/REITs). Keys reflect the raw sectorsdic_fmp.pickle AND
# the FMP profile/CSV `sector` labels (both use 'Financial Services'/'Real Estate';
# 'Banking'/'Insurance' are FMP *industry* labels, kept as harmless extra keys).
FINANCIAL_SECTORS = {'Financial Services', 'Banking', 'Insurance', 'Real Estate'}
REIT_SECTORS = {'Real Estate'}

SLOAN_TOP_QUINTILE = 0.80  # flag names at/above the 80th pct of Sloan WITHIN shortlist


def _load_sector_map(sector_pickle='sectorsdic_fmp.pickle'):
    """symbol -> sector, from the local sector pickle (dict sector -> [symbols]).
    Returns {} if the pickle is absent (the caller then emits a visible warning and
    the financial exclusion falls back to the API/CSV sector only)."""
    if not os.path.exists(sector_pickle):
        return {}
    sectordic = pd.read_pickle(sector_pickle)
    symb2sector = {}
    for sector, symbols in sectordic.items():
        for s in symbols:
            # first assignment wins; sector lists are largely disjoint
            symb2sector.setdefault(s, sector)
    return symb2sector


def _sector_is_financial(sector):
    """(is_financial, financial_kind) for a single sector label ('' kind if not)."""
    if sector in REIT_SECTORS:
        return True, 'REIT / real-estate'
    if sector in FINANCIAL_SECTORS:
        return True, 'bank / insurer (financial)'
    return False, ''


def _classify_financial(sector, fallback_sector=None):
    """(is_financial, forensic_valid, financial_kind) from up to two sector sources:
    the local pickle `sector` and an optional authoritative API/CSV `fallback_sector`.

    CONSERVATIVE: if EITHER source labels the name a bank/insurer/REIT it is
    financial -> forensic_valid=False. This never lets a real financial slip through
    as forensic-valid just because one source is Unknown/mismapped. A real-estate
    label from either source wins the REIT kind."""
    is_fin_p, kind_p = _sector_is_financial(sector)
    is_fin_f, kind_f = _sector_is_financial(fallback_sector) if fallback_sector else (False, '')
    if not (is_fin_p or is_fin_f):
        return False, True, ''
    if 'REIT' in kind_p or 'REIT' in kind_f:
        kind = 'REIT / real-estate'
    else:
        kind = kind_p or kind_f
    return True, False, kind


def _summary_tag(is_fin, m_finite, c_finite, m_flag, c_flag, sloan_flag):
    """Summary guidance tag (never an auto-drop). Financials are forensic-invalid
    and their M/C/Sloan flags are NOT counted."""
    if is_fin:
        return 'financial: forensic-invalid (use financial lens)'
    if not (m_finite and c_finite):
        return 'data-incomplete: dig-deeper'
    nflags = int(m_flag) + int(c_flag) + int(sloan_flag)
    if nflags == 0:
        return 'clean'
    if nflags == 1:
        return 'single-flag: dig-deeper'
    return 'multi-flag: concern'


def computeSloanAccruals(cdx_df, symblist, freq_map=None):
    """Standalone Sloan accruals = (NI_ttm - CFO_ttm) / avg(TotalAssets).

    NI/CFO are TTM flows (trailing-4-quarter sums); TotalAssets is a stock, so the
    denominator averages beginning and ending TA of the TTM window. The per-symbol
    frame is normalized to NEWEST-FIRST here (via _toNewestFirst, same explicit
    orientation the M/C forensics use) so index 0 IS the most recent quarter and
    index 4 is 4 quarters earlier -- REGARDLESS of how cdx_df happens to be ordered.
    The upstream cdx_df is oldest-first and is left untouched. Higher (more positive)
    = more accruals = lower earnings quality. Returns DataFrame[source, sloanAccruals]."""
    out = pd.DataFrame({'source': symblist})
    if freq_map is None:
        freq_map = rp.frequency_by_source(cdx_df)
    vals = []
    for symb in symblist:
        sub = _toNewestFirst(cdx_df[cdx_df['source'] == symb])
        # `rpy` rows span one YEAR: the TTM flow sums, the beginning-of-window asset
        # level, and the minimum history all follow the source's reporting frequency
        # (4 quarters OR 2 halves).  A semi-annual filer previously had a 24-month
        # 'TTM' numerator over a 24-month asset average -- both are now 12 months.
        _rpy = rp.rows_per_year(freq_map, symb)
        if len(sub) < _rpy + 1:
            vals.append(np.nan)
            continue
        ni = pd.to_numeric(sub['netIncome'], errors='coerce')
        cfo = pd.to_numeric(sub['netCashProvidedByOperatingActivities'], errors='coerce')
        ta = pd.to_numeric(sub['totalAssets'], errors='coerce')
        ni_ttm = invrollsumTTM(ni, _rpy)   # newest-first: iloc[0] = TTM to most recent period
        cfo_ttm = invrollsumTTM(cfo, _rpy)
        ni_recent = ni_ttm.iloc[0]
        cfo_recent = cfo_ttm.iloc[0]
        ta_end = ta.iloc[0]        # most recent
        ta_begin = ta.iloc[_rpy]   # one YEAR earlier (4 quarters or 2 halves)
        avg_ta = (ta_end + ta_begin) / 2.0
        if pd.isna(ni_recent) or pd.isna(cfo_recent) or pd.isna(avg_ta) or avg_ta == 0:
            vals.append(np.nan)
        else:
            vals.append((ni_recent - cfo_recent) / avg_ta)
    out['sloanAccruals'] = vals
    return out


def _mscore_drivers(mscore_df, symb, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """String of Beneish components driving M upward for one symbol, sorted by
    contribution magnitude, e.g. 'TATA(+2.34, ratio>neut); LVGI(+0.05, ratio<neut)'.
    '' if none/insufficient.

    Each driver is annotated with the direction of the underlying ratio vs its
    neutral (1.0, or 0.0 for TATA). This matters for the NEGATIVE-coefficient
    components (SGAI coef -0.172, LVGI coef -0.327): a positive contribution there
    means the ratio FELL BELOW neutral (ratio<neut) -- e.g. leverage DECLINED --
    which without the annotation reads backwards ('LVGI drives M up' looks like
    'leverage is a problem' when leverage actually dropped)."""
    sub = mscore_df[mscore_df['symbol'] == symb]
    if sub.empty:
        return ''
    window = sub.head(rp.scale_window(M_WINDOW, rpy))
    drivers = []
    for comp in M_COMPONENTS:
        if comp not in window.columns:
            continue
        val = pd.to_numeric(window[comp], errors='coerce').mean()
        if pd.isna(val) or np.isinf(val):
            continue
        contrib = M_COEF[comp] * (val - M_NEUTRAL[comp])
        if contrib > 0:
            ratio_dir = 'ratio>neut' if val > M_NEUTRAL[comp] else 'ratio<neut'
            drivers.append((comp, contrib, ratio_dir))
    drivers.sort(key=lambda x: x[1], reverse=True)
    return '; '.join(f'{c}(+{v:.2f}, {d})' for c, v, d in drivers)


def _cscore_fired(cscore_df, symb, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """String of Montier red flags that fired for one symbol, e.g. 'NICFOdiv; TAgr'.

    Consistent with the C_Score it sits beside: calcMontierC counts, PER PERIOD, each
    flag column > 0 (detectManipulation:55) and C_Score_mean averages that per-period
    count over the same head(C_WINDOW) window. So a flag is listed here iff it is > 0
    in AT LEAST ONE period of that window -- which guarantees every flag contributing
    to the shown score appears in the list. (The prior rule used the windowed MEAN of
    each flag > 0, which could HIDE a flag that fired strongly in one period and
    reversed in the other, producing fewer listed flags than the C_Score implied.)"""
    sub = cscore_df[cscore_df['symbol'] == symb]
    if sub.empty:
        return ''
    window = sub.head(rp.scale_window(C_WINDOW, rpy, minimum=1))
    fired = []
    for flag in C_FLAGS:
        if flag not in window.columns:
            continue
        vals = pd.to_numeric(window[flag], errors='coerce')
        if (vals > 0).any():   # per-period > 0, mirrors calcMontierC's count (NaN -> False)
            fired.append(flag)
    return '; '.join(fired)


def buildForensicFlagTable(resdic, topn, sector_fallback=None):
    """Build the per-name forensic-flag table for the top-`topn` shortlist.

    Pure/offline: consumes resdic['postRank'] (rank order), resdic['cdx_df'], and
    the detectManipulation artifacts (mscore_df, cscore_df, SLmeanMscore,
    SLmeanCscore, problemlist_Mscore, problemlist_Cscore). Makes no API calls.

    `sector_fallback` (optional dict symbol -> authoritative API/CSV sector) is
    cross-checked against the local pickle sector for the financial-invalid
    determination. In the production pipeline it is usually left None here and the
    API sectors are folded in later by applySectorFallback (the profile fetch that
    yields them happens after this table is built); pass it directly when the map is
    already known (e.g. offline tests).

    Returns a DataFrame, one row per shortlisted name, in rank order.
    """
    postRank = resdic['postRank']
    cdx_df = resdic['cdx_df']
    symblist = list(postRank['source'].head(topn))

    mscore_df = resdic.get('mscore_df', pd.DataFrame())
    cscore_df = resdic.get('cscore_df', pd.DataFrame())
    SLmeanMscore = resdic.get('SLmeanMscore', pd.DataFrame(columns=['source', 'M_Score_mean']))
    SLmeanCscore = resdic.get('SLmeanCscore', pd.DataFrame(columns=['source', 'C_Score_mean']))
    problem_M = set(resdic.get('problemlist_Mscore', []))
    problem_C = set(resdic.get('problemlist_Cscore', []))

    symb2sector = _load_sector_map()
    if not symb2sector:
        msg = ('forensicFlags: sector pickle (sectorsdic_fmp.pickle) is empty/absent '
               '-- the Stage-0 financial exclusion is DISABLED for pickle-sourced '
               'sectors; financials may be mislabeled forensic-valid unless an API '
               'sector_fallback is supplied. Restore the pickle or pass sector_fallback.')
        warnings.warn(msg, RuntimeWarning)
        print('WARNING: ' + msg)
    fallback = sector_fallback or {}

    # ONE frequency classification for the whole table (same source of truth as the
    # Stage-2 scorer and detectManipulation, all derived from this cdx_df).
    freq_map = rp.frequency_by_source(cdx_df)
    sloan = computeSloanAccruals(cdx_df, symblist, freq_map)
    sloan_map = dict(zip(sloan['source'], sloan['sloanAccruals']))

    # Sloan worst-quintile threshold WITHIN the shortlist (financials -- from EITHER
    # sector source -- excluded so their invalid readings don't shift the cutoff).
    valid_sloan = [sloan_map[s] for s in symblist
                   if not _classify_financial(symb2sector.get(s, 'Unknown'),
                                              fallback.get(s))[0]
                   and pd.notna(sloan_map.get(s))]
    sloan_cut = np.nanquantile(valid_sloan, SLOAN_TOP_QUINTILE) if len(valid_sloan) >= 5 else np.nan

    rows = []
    for rank, symb in enumerate(symblist, start=1):
        sector = symb2sector.get(symb, 'Unknown')
        api_sector = fallback.get(symb)
        is_fin, forensic_valid, fin_kind = _classify_financial(sector, api_sector)

        # M-score
        m_row = SLmeanMscore[SLmeanMscore['source'] == symb]['M_Score_mean']
        m_mean = pd.to_numeric(m_row, errors='coerce').iloc[0] if len(m_row) else np.nan
        m_finite = pd.notna(m_mean) and not np.isinf(m_mean)
        m_flag = bool(m_finite and m_mean > 0)  # stored M > 0 == standard M > -1.78
        _rpy = rp.rows_per_year(freq_map, symb)
        m_drivers = (_mscore_drivers(mscore_df, symb, _rpy)
                     if not mscore_df.empty else '')

        # C-score
        c_row = SLmeanCscore[SLmeanCscore['source'] == symb]['C_Score_mean']
        c_mean = pd.to_numeric(c_row, errors='coerce').iloc[0] if len(c_row) else np.nan
        c_finite = pd.notna(c_mean) and not np.isinf(c_mean)
        c_flag = bool(c_finite and c_mean >= C_FLAG_CUTOFF)  # THE flag: C >= 4
        c_fired = (_cscore_fired(cscore_df, symb, _rpy)
                   if not cscore_df.empty else '')

        # Sloan
        sloan_val = sloan_map.get(symb, np.nan)
        sloan_flag = bool(forensic_valid and pd.notna(sloan_val)
                          and pd.notna(sloan_cut) and sloan_val >= sloan_cut)

        tag = _summary_tag(is_fin, m_finite, c_finite, m_flag, c_flag, sloan_flag)

        rows.append({
            'rank': rank,
            'source': symb,
            'sectorPickle': sector,
            'sectorAPI': api_sector if api_sector else '',
            'isFinancial': is_fin,
            'forensicValid': forensic_valid,
            'financialKind': fin_kind,
            'M_score_mean': round(m_mean, 4) if m_finite else np.nan,
            'M_flag_gt_-1.78': m_flag,
            'M_drivers': m_drivers,
            'C_score_mean': round(c_mean, 4) if c_finite else np.nan,
            'C_flag_ge_4': c_flag,
            'C_flags_fired': c_fired,
            'sloanAccruals': round(sloan_val, 4) if pd.notna(sloan_val) else np.nan,
            'sloan_worstQuintile_inShortlist': sloan_flag,
            'inLegacyProblemlist_M': symb in problem_M,
            # detectManipulation's own problemlist_Cscore.  RENAMED from
            # 'legacyProblemC_strict_gt4': that header asserted a `> 4` cutoff that no
            # longer exists (both sides are now C_FLAG_CUTOFF), so the old name would
            # keep claiming a threshold the code does not use.  It now agrees with
            # C_flag_ge_4 except where the score is non-finite (which the problemlist
            # also collects, deliberately -- that is the cross-check it provides).
            'problemlistC': symb in problem_C,
            'forensicTag': tag,
        })

    return pd.DataFrame(rows)


def applySectorFallback(flag_df, api_sector_map):
    """Cross-check the pickle-derived financial classification against an
    authoritative API/CSV sector map and reconcile CONSERVATIVELY.

    In the production pipeline the per-name profile fetch (which yields the API
    `sector`) happens AFTER buildForensicFlagTable, so the API sectors are folded in
    here. For every name: if EITHER the pickle sector (already in the row) or the API
    sector says bank/insurer/REIT, the name becomes financial -> forensic-invalid.
    Recomputes isFinancial / forensicValid / financialKind, the Sloan worst-quintile
    cutoff (with the -- now possibly larger -- financial set excluded) and the summary
    tag. Pure/offline; returns a NEW DataFrame (input unchanged).

    `api_sector_map`: dict symbol -> API sector label ('NaN'/''/None treated as
    unknown, i.e. no fallback signal)."""
    if flag_df is None or flag_df.empty or not api_sector_map:
        return flag_df

    df = flag_df.copy()

    is_fin_list, valid_list, kind_list, api_list = [], [], [], []
    for _, row in df.iterrows():
        pickle_sector = row.get('sectorPickle', 'Unknown')
        api_sector = api_sector_map.get(row['source'])
        if api_sector in (None, '', 'NaN', 'nan'):
            api_sector = None
        is_fin, valid, kind = _classify_financial(pickle_sector, api_sector)
        is_fin_list.append(is_fin)
        valid_list.append(valid)
        kind_list.append(kind)
        api_list.append(api_sector if api_sector else '')

    df['sectorAPI'] = api_list
    df['isFinancial'] = is_fin_list
    df['forensicValid'] = valid_list
    df['financialKind'] = kind_list

    # Recompute the Sloan worst-quintile cutoff with the reconciled financial set
    # excluded (so newly-identified financials no longer contaminate the quantile).
    mask_valid = df['forensicValid'].astype(bool)
    sv = pd.to_numeric(df.loc[mask_valid, 'sloanAccruals'], errors='coerce').dropna()
    sloan_cut = np.nanquantile(sv, SLOAN_TOP_QUINTILE) if len(sv) >= 5 else np.nan

    new_sloan_flag, new_tag = [], []
    for _, row in df.iterrows():
        valid = bool(row['forensicValid'])
        sval = pd.to_numeric(pd.Series([row['sloanAccruals']]), errors='coerce').iloc[0]
        sflag = bool(valid and pd.notna(sval) and pd.notna(sloan_cut) and sval >= sloan_cut)
        m_finite = pd.notna(row.get('M_score_mean'))
        c_finite = pd.notna(row.get('C_score_mean'))
        m_flag = bool(row.get('M_flag_gt_-1.78'))
        c_flag = bool(row.get('C_flag_ge_4'))
        tag = _summary_tag(bool(row['isFinancial']), m_finite, c_finite, m_flag, c_flag, sflag)
        new_sloan_flag.append(sflag)
        new_tag.append(tag)

    df['sloan_worstQuintile_inShortlist'] = new_sloan_flag
    df['forensicTag'] = new_tag
    return df


def writeForensicFlagsCSV(flag_df, fname):
    """Persist the forensic-flag table to CSV (the standalone decision-support
    artifact; also feeds the presentation / AggScore outputs)."""
    flag_df.to_csv(fname, index=False)
    return fname
