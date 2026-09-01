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
                                C_FLAG_CUTOFF, C_FLAG_COLS, M_COMPONENTS,
                                absent_components, abstention_reason)
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
# `M_COMPONENTS` is IMPORTED from detectManipulation, not restated here (2026-08-16): the
# forensic-gap penalty counts absent components against the same list this file decomposes, and
# two copies of "the eight components" is how a ninth one gets added to one of them.
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


#  A BLANKED FLAG IS NOT A FALSE (P-4, 2026-08-29).  `M_flag_gt_-1.78` / `C_flag_ge_4` are
#  blanked on a row with no score (see buildForensicFlagTable), which makes the published
#  column three-valued -- True / False / blank -- and the blank arrives as `''` in memory and
#  as NaN after a CSV round-trip.  `bool(float('nan')) is True`, so a plain truthiness test on
#  the round-tripped column would INVENT a manipulation flag on the very names the pipeline
#  said it could not assess: the exact inversion of the rule that blanked them, and precisely
#  the "red flag next to a high-quality indicator" the CEO has named as the opposite of what
#  he wants.  ONE reader for every consumer, so a new surface cannot re-derive it wrongly.
def _flag_true(v):
    """True only for a flag that SAYS true; a blank/NaN/absent flag is not a False, and is
    certainly not a True.  Accepts the three shapes the column reaches a consumer in: a
    Python/NumPy bool (in-memory), the string 'True'/'False' (object column read back from
    CSV), and the blank ('' in memory, NaN after the round-trip)."""
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes')
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return bool(v)


#  A FORENSIC-VALIDITY CELL IS ALSO THREE-VALUED, AND ITS BLANK IS THE DANGEROUS ONE
#  (Q-44, 2026-08-31).  `forensicValid` says whether the Beneish / Montier / Sloan models
#  APPLY to a name at all -- it is a statement about the business model, not about the data --
#  and every consumer so far has read it with `bool(row.get('forensicValid', True))`, i.e. it
#  DEFAULTS TO VALID.  That default is the same class of error `_flag_true` closes one column
#  to the left, pointing the other way: `M_flag_gt_-1.78` blank defaulting to True invents a
#  finding, `forensicValid` blank defaulting to True SUPPRESSES one -- it asserts "the fraud
#  models apply to this name" about a name nobody classified, and every downstream
#  low-confidence marker keyed off it then silently declines to fire.
#
#  SO THE ABSENCE GETS ITS OWN VALUE.  `None` means UNDETERMINED: not valid, not invalid, not
#  assessed.  A caller that needs a binary must decide which way undetermined falls AND SAY SO
#  AT ITS OWN CALL SITE -- which is the whole point, because the two consumers want opposite
#  defaults (a display marker should fire on undetermined; a Sloan-quantile exclusion should
#  not).  ONE reader, so a new surface cannot re-derive it wrongly.
def published_forensic_validity(v):
    """Tri-state read of a published `forensicValid` cell: True / False / None.

    `None` for a blank -- absent key, `''`, NaN after a CSV round-trip, or the string 'nan'
    that an object column reads back as.  NEVER True: a name nobody classified is not a name
    classified as forensically valid, and defaulting it to True is precisely how the deck's
    low-confidence guard came to be unable to fire (Q-44).  Accepts the shapes the column
    reaches a consumer in: a Python/NumPy bool in memory, 'True'/'False' from a CSV, and the
    blank in either of its two forms."""
    if v is None:
        return None
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ('true', '1', 'yes'):
            return True
        if t in ('false', '0', 'no'):
            return False
        return None                      # '' and 'nan' both land here
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return bool(v)


def validity_display(v, valid='applies', invalid='INVALID (financial lens)',
                     blank='not determined'):
    """How a tri-state forensic validity is RENDERED, in one place -- the `flag_display`
    of this column.  The undetermined state gets its own words: showing it as `applies` is
    the Q-44 defect itself, and showing it as `INVALID` overstates a classification nobody
    made."""
    t = published_forensic_validity(v)
    if t is None:
        return blank
    return valid if t else invalid


#  THE TEXT COLUMNS NEED A READER TOO, AND FOR THE SAME REASON (Q-66, 2026-09-01).  Every
#  optional string this module publishes -- `M_drivers`, `M_abstain_reason`, `forensicReason`,
#  `forensicNote`, `C_flags_fired` -- is `''` in memory and NaN after a CSV round-trip, and
#  every consumer so far wrote `str(x or '')`.  That is WRONG in the same way `bool(nan)` is:
#  `float('nan') or ''` short-circuits to the NaN, because NaN is TRUTHY, so the expression
#  returns the literal string `'nan'` -- which is not empty, so an `if reason:` branch FIRES
#  and the page renders the word "nan" where a sentence belongs.
#  IT WAS NOT HYPOTHETICAL: it put `Why there is no M / C score: nan` on all five Mining pages
#  of the 2026-08-31 deck -- the five pages the Q-66 extension exists to populate.
def cell_text(v):
    """A published optional-text cell as a plain string: '' for every shape of absence.

    Absence is `None`, `''`, NaN, and the strings 'nan'/'none' an object column reads back as.
    One reader, so a new surface cannot re-derive it wrongly -- the `flag_display` of the text
    columns."""
    if v is None:
        return ''
    try:
        if not isinstance(v, str) and pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return '' if s.lower() in ('nan', 'none') else s


def flag_display(v, yes='FLAG', no='no', blank='not assessed'):
    """How a possibly-blank forensic flag is RENDERED, in one place.  The blank gets its own
    word: rendering it as `no` is the P-4 defect itself (a negative finding asserted where the
    pipeline abstained), and rendering it as `FLAG` is the NaN-truthiness trap above."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return blank
    try:
        if not isinstance(v, str) and pd.isna(v):
            return blank
    except (TypeError, ValueError):
        pass
    return yes if _flag_true(v) else no


#  A COHORT REFUSAL AND A DATA GAP ARE DIFFERENT FACTS AND MUST NOT SHARE A TAG (Q-66).
#  `_summary_tag` used to branch on `is_fin` -- the SECTOR classification -- which is only one
#  of the two sources that can rule the models inapplicable.  A REIT whose sector map came back
#  `Unknown` but whose CARVE LABEL says REIT would have fallen through to
#  `data-incomplete: dig-deeper`, i.e. "we tried and the vendor let us down" about a name we
#  deliberately declined to measure.  The first parameter is now the APPLICABILITY VERDICT, and
#  `kind` decides only the WORDING -- the 'financial: forensic-invalid' string is preserved
#  verbatim for the sector-classified case because that is the tag already in every shipped
#  artifact and in the tests that pin it.
def _summary_tag(inapplicable, m_finite, c_finite, m_flag, c_flag, sloan_flag,
                 kind='', label=''):
    """Summary guidance tag (never an auto-drop). A name the forensic models do not apply to
    is tagged as such and its M/C/Sloan flags are NOT counted."""
    if inapplicable:
        #  `if kind:` ON THE RAW CELL WOULD BE A TRUTHINESS TRAP.  `financialKind` is blank in
        #  memory but reads back from a CSV as NaN, and `bool(float('nan'))` is True -- so a
        #  reconciled row with no kind would have taken the 'financial' wording on the strength
        #  of a missing value.  Same family as `_flag_true`, one column over.
        kind = '' if kind is None else str(kind).strip()
        if kind.lower() in ('nan', 'none'):
            kind = ''
        label = '' if label is None else str(label).strip()
        if label.lower() in ('nan', 'none'):
            label = ''
        if kind:
            return 'financial: forensic-invalid (use financial lens)'
        return ('cohort %s: forensic-inapplicable (see forensicReason)' % label
                if label else 'forensic-inapplicable (see forensicReason)')
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


def cohort_applicability(label):
    """(applies, reason, note) for one carve label -- the ONE reader of the Q-66 ruling.

    `applies` is TRI-STATE: True (computed), False (refused, and `reason` says why in words),
    None (undetermined -- 'general', or a label this run does not know).  `note` is a caveat
    that ships WITH a computed score and is '' otherwise; a caveat is not a refusal and the
    two never share a field.

    Delegates to `carveOut`, which owns the labels AND the argument behind each verdict.  It is
    imported lazily and its absence is survivable: a caller with no carve information gets
    `(None, '', '')` and behaves exactly as this module did before Q-66 -- the general pool's
    reading must not depend on a module it never needed."""
    if not label:
        return None, '', ''
    try:
        import carveOut as _co
    except Exception:                                    # pragma: no cover - import guard
        return None, '', ''
    return (_co.cohort_forensic_validity(label),
            _co.cohort_forensic_reason(label),
            _co.cohort_forensic_note(label))


def buildForensicFlagTable(resdic, topn, sector_fallback=None, cohort_members=None,
                           carve_labels=None):
    """Build the per-name forensic-flag table for the top-`topn` shortlist AND, when
    `cohort_members` is supplied, for every carve-cohort side-list beside it (Q-66).

    Pure/offline: consumes resdic['postRank'] (rank order), resdic['cdx_df'], and
    the detectManipulation artifacts (mscore_df, cscore_df, SLmeanMscore,
    SLmeanCscore, problemlist_Mscore, problemlist_Cscore). Makes no API calls.

    `sector_fallback` (optional dict symbol -> authoritative API/CSV sector) is
    cross-checked against the local pickle sector for the financial-invalid
    determination. In the production pipeline it is usually left None here and the
    API sectors are folded in later by applySectorFallback (the profile fetch that
    yields them happens after this table is built); pass it directly when the map is
    already known (e.g. offline tests).

    `cohort_members`: {carve label -> ordered symbols}, normally each side-list's own
    `postRank['source']`.  `carve_labels`: the RUN'S OWN label Series (symbol -> label, i.e.
    `resdic['carveout_labels']`).  Both optional; omitting them reproduces the pre-Q-66
    general-pool-only table exactly.

    WHY THE LABEL AND NOT THE SECTOR PICKLE.  The pickle is undated and rebuilt every run, and
    the copy in a working tree is a DIFFERENT TAXONOMY from the copy a run used -- re-deriving
    a cohort from it flipped six of the 2026-08-29 deck's 45 pages in both directions.  The
    carve label travels in the same pickle as the shortlist, so it is contemporaneous with the
    names by construction.  See the block above `carveOut.COHORT_FORENSIC_VALIDITY`.

    THREE-VALUED, NEVER DEFAULTING TO A NUMBER.  Each name's forensic reading is COMPUTED,
    REFUSED (`forensicReason` says why, and every score/flag column is blank -- '' for the
    flags, NaN for the scores), or UNDETERMINED.  A refused name is NOT scored and then hidden:
    the numbers are never produced, so no later surface can find them and publish them.

    Returns a DataFrame, one row per name, cohort rows after the general rows, `rank` counting
    from 1 WITHIN each pool (a cohort's rank 1 is its own best name, not name 101).
    """
    postRank = resdic['postRank']
    cdx_df = resdic['cdx_df']
    #  POOLS, IN ORDER.  `pool` and a per-pool `rank` are what let one table carry six
    #  shortlists without the reader having to guess whether row 104 is "general, rank 104" or
    #  "REIT, rank 4".  A name that somehow appears in two pools is kept in the FIRST (general
    #  wins), so the table has exactly one row per name and `merge(..., on='source')` at every
    #  downstream surface cannot silently duplicate rows.
    pools = [('general', list(postRank['source'].head(topn)))]
    _seen = set(pools[0][1])
    for _lab, _members in (cohort_members or {}).items():
        _m = [x for x in (list(_members) if _members is not None else []) if x not in _seen]
        _seen.update(_m)
        if _m:
            pools.append((_lab, _m))
    symblist = [s for _lab, _m in pools for s in _m]
    label_of = {}
    if carve_labels is not None:
        try:
            label_of = {k: v for k, v in dict(carve_labels).items() if isinstance(v, str)}
        except Exception:                                # pragma: no cover - defensive
            label_of = {}

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
    #  ONE CUTOFF PER POOL (Q-66).  The flag's own name says "worst quintile WITHIN the
    #  shortlist", and once cohort names share this table there are SIX shortlists, not one.
    #  A single pooled cutoff would have two failure modes at once: it would move the GENERAL
    #  pool's flags (24 Mining names entering its quantile is a change to an artifact the CEO
    #  already reviewed, made as a side effect of adding rows), and it would compare a miner's
    #  accruals against an industrials-heavy pool it was never meant to be ranked in.  The
    #  general pool's cutoff is computed over exactly the names it was computed over before.
    def _pool_cut(members):
        vals = [sloan_map[s] for s in members
                if not _classify_financial(symb2sector.get(s, 'Unknown'),
                                           fallback.get(s))[0]
                and pd.notna(sloan_map.get(s))]
        return np.nanquantile(vals, SLOAN_TOP_QUINTILE) if len(vals) >= 5 else np.nan
    sloan_cut_by_pool = {lab: _pool_cut(members) for lab, members in pools}

    rows = []
    for pool_label, members in pools:
        sloan_cut = sloan_cut_by_pool[pool_label]
        for rank, symb in enumerate(members, start=1):
            sector = symb2sector.get(symb, 'Unknown')
            api_sector = fallback.get(symb)
            is_fin, sector_valid, fin_kind = _classify_financial(sector, api_sector)
            #  THE RUN'S OWN LABEL FIRST, the pool it was listed under second.  They agree for
            #  every cohort row; they differ only if `carve_labels` is absent (older pickle) or
            #  disagrees with the side-list it came from, and in both cases the pool the name
            #  was actually ranked in is the better answer than nothing.
            carve_label = label_of.get(symb, pool_label)
            label_applies, label_reason, label_note = cohort_applicability(carve_label)
            #  CONSERVATIVE JOIN, in the same direction as `_classify_financial`'s: EITHER
            #  source saying "the models do not apply" is decisive, and neither can promote a
            #  name the other refused.  For the general pool `label_applies` is None, so this
            #  is `sector_valid` exactly -- the pre-Q-66 value, unchanged.
            forensic_valid = bool(sector_valid and label_applies is not False)
            if label_applies is False:
                forensic_reason = label_reason
            elif is_fin:
                forensic_reason = ('the Beneish / Montier / Sloan models do not apply to a '
                                   '%s' % (fin_kind or 'financial'))
            else:
                forensic_reason = ''
            forensic_note = label_note if forensic_valid else ''

            #  ---- THE REFUSAL (Q-66) -----------------------------------------------------
            #  A NAME THE MODELS DO NOT APPLY TO IS NOT SCORED, and it is not scored HERE --
            #  before any number is read out of `SLmeanMscore` -- rather than scored and then
            #  hidden at the presentation layer.  Two reasons, and the second is the one that
            #  matters: (a) a number that exists in the frame is a number the next surface
            #  written against this table will find and publish, which is exactly how the deck
            #  came to print a green Sloan tick on REITs; and (b) `forensicReason` beside the
            #  blank is what makes this the THIRD value -- refused-with-a-reason -- rather than
            #  a hole indistinguishable from a vendor gap.
            #  BLANK IS `''`, NOT NaN, for the flags, for the reason set out at
            #  `M_flag_gt_-1.78` below: `bool(float('nan')) is True`, so a NaN blank would
            #  invent a manipulation flag on the very names we just declined to assess.
            if not forensic_valid:
                m_mean = np.nan
                m_finite = False
                m_flag = False
                m_drivers = ''
                c_mean = np.nan
                c_finite = False
                c_flag = False
                c_fired = ''
                #  ONE SENTENCE, NOT TWO ACCOUNTS OF ONE ROW.  `M_abstain_reason` and
                #  `forensicReason` must not disagree about why the cell is empty, so the
                #  refusal writes both from the same string.
                m_abstain_reason = forensic_reason + ' -- not measured, and not charged'
                sloan_val = sloan_map.get(symb, np.nan)
                #  THE SLOAN FLAG BECOMES BLANK TOO, not False.  `False` here says "assessed,
                #  and not in the worst quintile" about a name excluded from the quantile
                #  entirely -- the P-4 defect one column to the left, pointing the same way.
                #  The raw `sloanAccruals` VALUE is still published: it is an arithmetic ratio,
                #  not a verdict, and suppressing it would hide the magnitude the reader may
                #  still want.  Its verdict is what is withheld.
                sloan_flag = ''
                #  THE LABEL OUTRANKS THE SECTOR IN THE *TAG* TOO, so the tag and
                #  `forensicReason` cannot give two accounts of one row.  Passing `fin_kind`
                #  unconditionally produced a cohort whose rows split between
                #  'financial: forensic-invalid' and 'cohort REIT: forensic-inapplicable'
                #  purely on whether the sector pickle happened to know the name -- and the
                #  reason column beside them already said the cohort.  When the LABEL is what
                #  refused, the tag says the cohort; the financial wording survives for a name
                #  the SECTOR refused and no cohort ruling covered (every general-pool row).
                tag = _summary_tag(True, m_finite, c_finite, m_flag, c_flag, False,
                                   kind=('' if label_applies is False else fin_kind),
                                   label=carve_label)
            else:
                # M-score
                m_row = SLmeanMscore[SLmeanMscore['source'] == symb]['M_Score_mean']
                m_mean = pd.to_numeric(m_row, errors='coerce').iloc[0] if len(m_row) else np.nan
                m_finite = pd.notna(m_mean) and not np.isinf(m_mean)
                m_flag = bool(m_finite and m_mean > 0)  # stored M > 0 == standard M > -1.78
                _rpy = rp.rows_per_year(freq_map, symb)
                #  NO VERDICT, NO DRIVER BREAKDOWN (P-2, CEO 2026-08-17).  `_mscore_drivers` averages
                #  each component over the window INDEPENDENTLY, so it happily returns a decomposition
                #  for a name whose M_Score is NaN -- the components it lists are the ones that WERE
                #  computable, and the reason the name has no score is one it silently omits.  On the
                #  2026-08-13 top-100 all 21 `data-incomplete: dig-deeper` rows shipped a populated
                #  `M_drivers`; RMV.L's read `SGAI(+0.01)...` beside an abstention caused by the gross
                #  margin, and PSI.TO's was four components all at `+0.00`.  A breakdown asserts "here
                #  is what drove the verdict" where there is no verdict, which is the same "say nothing
                #  rather than something unfounded" principle the abstention itself rests on.
                #  NOTHING IS LOST: `M_abstain_reason` beside it names the missing vendor input, and
                #  this is the ONE place to blank it -- `ForensicFlagsTop100`, `AggScoreTop100`, the
                #  XLSX forensic block and the HTML deck's R5 rule all read this column.
                m_drivers = (_mscore_drivers(mscore_df, symb, _rpy)
                             if (m_finite and not mscore_df.empty) else '')
                #  The abstention's REASON, from the same two functions the penalty bucket uses.
                _absent = absent_components(mscore_df, symb, _rpy)
                _mwin = (pd.to_numeric(mscore_df[mscore_df['symbol'] == symb]['M_Score'],
                                       errors='coerce').head(rp.scale_window(M_WINDOW, _rpy))
                         if not mscore_df.empty else pd.Series(dtype='float64'))
                m_abstain_reason = abstention_reason(_absent, m_finite,
                                                     int(_mwin.notna().sum()), len(_mwin))
                #  A FORENSICALLY INVALID NAME DID NOT FAIL TO BE MEASURED -- THE MODEL DOES NOT APPLY
                #  (review H-2, 2026-08-17).  The ad-hoc penalty exempts these names for exactly this
                #  reason; saying "no usable vendor data" beside a `financial: forensic-invalid` tag
                #  would describe a measurement we never attempted, and the two artifacts would give
                #  the reader two different accounts of one row.
                #  THAT RULE NOW LIVES IN THE REFUSAL BRANCH ABOVE, and the `if is_fin ...` override
                #  that used to sit here is DELETED rather than left as belt-and-braces (Q-66).  It
                #  became UNREACHABLE the moment the refusal moved earlier -- `forensic_valid` is
                #  `not is_fin` ANDed with the cohort ruling, so reaching this line proves `is_fin`
                #  is False -- and a dead branch whose comment says it fires is a worse artifact
                #  than no branch: the next reader trusts it and looks for the behaviour elsewhere.
                #  The SENSE it carried is preserved -- the refusal still ends "-- not measured,
                #  and not charged" -- but the sentence in front of it is now `forensicReason`,
                #  which names all three models rather than Beneish alone and covers the cohort
                #  refusal as well as the sector one.  One string, so the CSV, the XLSX and the
                #  deck cannot give three accounts of one empty cell.

                # C-score
                c_row = SLmeanCscore[SLmeanCscore['source'] == symb]['C_Score_mean']
                c_mean = pd.to_numeric(c_row, errors='coerce').iloc[0] if len(c_row) else np.nan
                c_finite = pd.notna(c_mean) and not np.isinf(c_mean)
                c_flag = bool(c_finite and c_mean >= C_FLAG_CUTOFF)  # THE flag: C >= 4
                #  THE SAME RULE ON THE C SIDE (P-2), AND ITS MEASURED EFFECT TODAY IS ZERO -- said
                #  here so nobody reads the symmetry as evidence of a live defect.  `C_Score` is a
                #  COUNT (`(cols > 0).sum()`, NaN > 0 = False), so `C_Score_mean` is NaN only when a
                #  name produces no forensic rows at all: 0 of 2,629 on the 2026-08-13 panel.  The
                #  guard is here because the asymmetry -- one column self-limiting, its neighbour not
                #  -- is exactly how the M side acquired this defect, not because a row needs it now.
                c_fired = (_cscore_fired(cscore_df, symb, _rpy)
                           if (c_finite and not cscore_df.empty) else '')

                # Sloan -- the worst quintile of THIS POOL (see `_pool_cut`)
                sloan_val = sloan_map.get(symb, np.nan)
                sloan_flag = bool(pd.notna(sloan_val) and pd.notna(sloan_cut)
                                  and sloan_val >= sloan_cut)

                tag = _summary_tag(False, m_finite, c_finite, m_flag, c_flag, sloan_flag,
                                   kind=fin_kind, label=carve_label)

            rows.append({
                #  `pool` + a per-pool `rank` (Q-66).  Before this table carried cohort rows,
                #  `rank` alone was unambiguous; with six shortlists in one file it is not, and
                #  a reader (or a merge) that assumes global rank would mis-order five of them.
                'pool': pool_label,
                'rank': rank,
                'source': symb,
                #  THE RUN'S OWN CARVE LABEL, carried as PROVENANCE for the applicability
                #  verdict beside it -- so a reader can tell a refusal decided by the label
                #  from one decided by the sector map, and so a re-derivation from a
                #  working-tree sector pickle is never necessary.
                'carveLabel': carve_label,
                'sectorPickle': sector,
                'sectorAPI': api_sector if api_sector else '',
                'isFinancial': is_fin,
                'forensicValid': forensic_valid,
                'financialKind': fin_kind,
                'M_score_mean': round(m_mean, 4) if m_finite else np.nan,
                #  NO VERDICT, NO BOOLEAN VERDICT (P-4, 2026-08-29) -- the same rule as `M_drivers`
                #  above, one column to its left.  `M_flag_gt_-1.78 = False` on a row with no
                #  M-score asserts "this name is NOT a manipulator" about a name the pipeline has
                #  just declared unassessable; on the 2026-08-29 top-100 that was 17 of 97 rows,
                #  FIVE of them inside the top-20 (ranks 2, 4, 7, 9, 10).  A False here is not the
                #  absence of a finding, it is a finding -- and the reader has no way to tell it
                #  from the 78 rows where the model actually ran and returned a real 'no'.
                #  THE COLUMN BECOMES THREE-VALUED (True / False / blank) and therefore OBJECT
                #  dtype.  `''` is the blank, not `np.nan`, deliberately: the two are identical in
                #  the CSV (both write an empty cell) but not in memory, and `bool(np.nan)` is
                #  True -- so a NaN blank would turn any truthiness test still to be written into a
                #  FALSE RED FLAG, which is worse than the defect being fixed.  `''` also matches
                #  the blank `M_drivers` / `C_flags_fired` already use in the neighbouring columns.
                #  Consumers must go through `_flag_true` / `flag_display`, which handle all three.
                'M_flag_gt_-1.78': m_flag if m_finite else '',
                'M_drivers': m_drivers,
                #  WHY THE CELL BESIDE IT IS EMPTY (CEO, 2026-08-16).  After the O-13 domain
                #  guards a fifth of the shortlist abstains, and a blank M with no explanation
                #  reads as a broken tool rather than as a refusal -- which is the CEO's standing
                #  "presentation must be correctly suggestive" constraint applied to a hole.
                #  '' for a name that HAS a verdict, so the column is self-limiting; the same
                #  string is what `adhoc_penalty` charged the name on, from one shared function,
                #  so the CSV cannot explain the abstention differently from the bucket.
                'M_abstain_reason': m_abstain_reason,
                'C_score_mean': round(c_mean, 4) if c_finite else np.nan,
                #  THE SAME RULE ON THE C SIDE, AND ITS MEASURED EFFECT TODAY IS ZERO -- stated
                #  rather than implied, exactly as `C_flags_fired` states it: `C_Score` is a COUNT,
                #  so `C_score_mean` is NaN only for a name that produces no forensic rows at all
                #  (0 of 97 on the 2026-08-29 top-100).  Present because an asymmetry between two
                #  adjacent columns -- one self-limiting, one not -- is how the M side acquired
                #  this defect in the first place.
                'C_flag_ge_4': c_flag if c_finite else '',
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
                #  THE THIRD VALUE, IN WORDS (Q-66).  `forensicValid=False` says the models do
                #  not apply; this says WHY, in the sentence the deck and the XLSX render.  A
                #  refusal with no reason is indistinguishable, on the page, from an omission,
                #  which is the whole complaint that produced this change.  '' when the name
                #  was scored -- self-limiting, like `M_drivers` and `M_abstain_reason`.
                'forensicReason': forensic_reason,
                #  A CAVEAT ON A SCORE THAT EXISTS -- never a reason it does not.  Only Mining
                #  carries one today (calibration + the near-degenerate AQI base); see
                #  `carveOut.COHORT_FORENSIC_NOTE`.  Kept in its own column so a caveat can
                #  never be rendered where a refusal belongs.
                'forensicNote': forensic_note,
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
    unknown, i.e. no fallback signal).

    THREE THINGS CHANGED HERE FOR Q-66, all of them "this function must not undo what
    `buildForensicFlagTable` just decided":

    1. IT CANNOT PROMOTE A COHORT-REFUSED NAME BACK TO VALID.  It re-derived `forensicValid`
       from the two SECTOR sources alone, so a name the carve label had ruled inapplicable
       would come back valid the moment its API sector read `Industrials` -- silently undoing
       the ruling one function later, in a step whose docstring says it only ever ADDS
       financials.  The carve label travels on the row (`carveLabel`) precisely so this
       reconciliation can read it.
    2. A NAME THAT BECOMES INAPPLICABLE HERE IS BLANKED, not just re-tagged.  Previously a
       general-pool name that the API sector reclassified as a bank kept its computed M-Score
       and C-Score and merely gained a `financial: forensic-invalid` tag -- a printed number
       beside a sentence saying the model does not apply to it.  ZERO rows on the 2026-08-31
       run (all 97 were already valid from both sources), so this is closing the route rather
       than fixing a live wrong cell -- said plainly rather than implied.
    3. THE SLOAN CUTOFF IS PER POOL, matching `buildForensicFlagTable`.  Pooling all six
       shortlists into one quantile would move the GENERAL pool's flags as a side effect of
       the table having gained cohort rows."""
    if flag_df is None or flag_df.empty or not api_sector_map:
        return flag_df

    df = flag_df.copy()
    #  A pre-Q-66 frame has neither column; `.get` on a Series is not available, so the
    #  columns are resolved once here and every loop below reads the resolved list.  Absent
    #  `carveLabel` -> `cohort_applicability` returns None -> the sector sources decide, which
    #  is exactly the pre-Q-66 behaviour.
    labels = (list(df['carveLabel']) if 'carveLabel' in df.columns else [''] * len(df))
    pools = (list(df['pool']) if 'pool' in df.columns else ['general'] * len(df))

    is_fin_list, valid_list, kind_list, api_list, reason_list = [], [], [], [], []
    for (_, row), lab in zip(df.iterrows(), labels):
        pickle_sector = row.get('sectorPickle', 'Unknown')
        api_sector = api_sector_map.get(row['source'])
        if api_sector in (None, '', 'NaN', 'nan'):
            api_sector = None
        is_fin, sector_valid, kind = _classify_financial(pickle_sector, api_sector)
        label_applies, label_reason, _ = cohort_applicability(lab)
        valid = bool(sector_valid and label_applies is not False)
        if label_applies is False:
            reason = label_reason
        elif is_fin:
            reason = ('the Beneish / Montier / Sloan models do not apply to a %s'
                      % (kind or 'financial'))
        else:
            reason = ''
        is_fin_list.append(is_fin)
        valid_list.append(valid)
        kind_list.append(kind)
        reason_list.append(reason)
        api_list.append(api_sector if api_sector else '')

    df['sectorAPI'] = api_list
    df['isFinancial'] = is_fin_list
    df['forensicValid'] = valid_list
    df['financialKind'] = kind_list
    df['forensicReason'] = reason_list

    #  BLANK THE SCORES OF ANYTHING NEWLY REFUSED, before the cutoff is computed off them.
    newly_refused = [not v for v in valid_list]
    if any(newly_refused):
        for _col, _blank in (('M_score_mean', np.nan), ('M_flag_gt_-1.78', ''),
                             ('M_drivers', ''), ('C_score_mean', np.nan),
                             ('C_flag_ge_4', ''), ('C_flags_fired', '')):
            if _col in df.columns:
                #  `.astype(object)` first: assigning '' into a bool column would coerce the
                #  whole column, and assigning it into a float column raises on some pandas
                #  builds.  The three-valued columns are object dtype by design (see
                #  `M_flag_gt_-1.78` in the builder).
                if _blank == '':
                    df[_col] = df[_col].astype(object)
                df.loc[newly_refused, _col] = _blank
        if 'M_abstain_reason' in df.columns:
            df['M_abstain_reason'] = df['M_abstain_reason'].astype(object)
            df.loc[newly_refused, 'M_abstain_reason'] = [
                r + ' -- not measured, and not charged'
                for r, bad in zip(reason_list, newly_refused) if bad]
        if 'forensicNote' in df.columns:
            df['forensicNote'] = df['forensicNote'].astype(object)
            df.loc[newly_refused, 'forensicNote'] = ''

    # Recompute the Sloan worst-quintile cutoff with the reconciled financial set
    # excluded (so newly-identified financials no longer contaminate the quantile),
    # ONE CUTOFF PER POOL -- see the docstring's point 3.
    sloan_cut_by_pool = {}
    for pool in dict.fromkeys(pools):
        mask = [p == pool and v for p, v in zip(pools, valid_list)]
        sv = pd.to_numeric(df.loc[mask, 'sloanAccruals'], errors='coerce').dropna()
        sloan_cut_by_pool[pool] = (np.nanquantile(sv, SLOAN_TOP_QUINTILE)
                                   if len(sv) >= 5 else np.nan)

    new_sloan_flag, new_tag = [], []
    for (_, row), lab, pool in zip(df.iterrows(), labels, pools):
        valid = bool(row['forensicValid'])
        sloan_cut = sloan_cut_by_pool.get(pool, np.nan)
        sval = pd.to_numeric(pd.Series([row['sloanAccruals']]), errors='coerce').iloc[0]
        #  BLANK, NOT False, when the models do not apply -- the same three-valued rule the
        #  builder applies (a `False` here asserts "assessed, and not in the worst quintile"
        #  about a name excluded from the quantile).
        sflag = (bool(pd.notna(sval) and pd.notna(sloan_cut) and sval >= sloan_cut)
                 if valid else '')
        m_finite = pd.notna(row.get('M_score_mean'))
        c_finite = pd.notna(row.get('C_score_mean'))
        #  `_flag_true`, NOT `bool(...)` (P-4): the published columns are now three-valued and
        #  a blank arrives here as `''` in memory or NaN from a CSV, and `bool(np.nan)` is
        #  True.  The tag itself is not currently reachable through that path -- `_summary_tag`
        #  returns `data-incomplete` before it reads either flag whenever a score is missing --
        #  so this is closing the route, not a live wrong tag.
        m_flag = _flag_true(row.get('M_flag_gt_-1.78'))
        c_flag = _flag_true(row.get('C_flag_ge_4'))
        #  Same label-outranks-sector rule as the builder's, so the reconciliation cannot
        #  re-word a tag the builder already agreed with `forensicReason`.
        _label_refused = cohort_applicability(lab)[0] is False
        tag = _summary_tag(not valid, m_finite, c_finite, m_flag, c_flag,
                           _flag_true(sflag),
                           kind=('' if _label_refused else row.get('financialKind', '')),
                           label=lab)
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
