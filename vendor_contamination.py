"""vendor_contamination.py -- one company's statements served under another's ticker.

THE CASE THAT BUILT THIS MODULE
-------------------------------
`058820.KQ` (CMG Pharmaceutical, KOSDAQ) carries **Chipotle Mexican Grill's** income
statement and balance sheet for the window **2020-03-31 -> 2022-09-30**, labelled
`reportedCurrency=KRW`, matching `CMG` TO THE DOLLAR.  Verified on the 2026-08-07 CUR3K
panel: quarterly revenue runs 1.36e9 / 1.60e9 / 1.61e9 / 1.74e9 ... 2.22e9 -- Chipotle's
actual quarterly revenue, in dollars, with a KRW label -- and `totalAssets` sits at
5.4-6.8e9 against the same company's genuine 1.69-1.79e11 KRW from the next quarter on.
At the 2022-12-31 statement it SNAPS to real KRW values: a ~13x scale break INSIDE ONE
NAME'S HISTORY, which reads as spectacular growth exactly at the boundary.

Three properties made it hard to see and are worth stating because they constrain any
detector:
  * `marketCap` is UNAFFECTED -- it runs continuous in real KRW across the boundary.  So
    every market-cap sanity check in `data_quality` passes, and any "scale break" screen
    keyed on cap-vs-assets divergence is fighting the wrong signal.
  * `cik` is ZEROED ("0000000000") rather than copied, so CIK is NOT a usable detector.
  * The collision is by **COMPANY NAME** -- "CMG Pharmaceutical" -> leading token "CMG" ->
    Chipotle's ticker -- NOT by ticker root; `058820` collides with nothing.

**IT IS LIVE AT THE API TODAY** (re-probed 2026-08-08; FMP still serves it).  A re-fetch
re-ingests it verbatim, which is why the fix has to be a DATA-SIDE RULE that survives the
next full fetch rather than a one-off scrub of a pickle.

BLAST RADIUS, established before building anything:
  * **Stage-1 scoring is untouched.**  `calcScore` re-sorts each source newest-first and
    scores only the newest n=8 rows; the corrupt rows are positions 15-24 of 24, and the
    name appears in no shipped 2026-08-07 output.
  * **The backtest IS hit.**  `backtest_unified`'s CLI defaults to
    `--buy_years 2020,2021,2022` -- exactly the corrupt window.

WHAT THIS MODULE DOES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------------
1. `QUARANTINE_RULES` -- a NAMED, DATED, EVIDENCED list of (source, window) rows to drop.
   Applied by `data_quality.filter_invalid_data` as its first pass, so the removal flows
   through the SAME transparency CSV and the SAME BoMetric_df row-propagation every other
   removal uses, and so it survives every re-fetch.  Not a magic constant in a filter.
2. `detect_shared_fundamentals` -- the panel-internal detector that FOUND it.  ZERO API
   calls: hash every `(date, revenue, totalAssets)` triple, find sources sharing >= 3 of
   them, and flag the pairs whose NORMALISED COMPANY NAMES do not match.
3. It does **NOT** auto-quarantine what the detector finds.  Legitimate cross-listings
   dominate the raw matches; the name comparison separates them but does not adjudicate
   them.  On the 9,012-source NA1_EU1 panel all 26 zero-shared-token pairs were LEGITIMATE
   (Corteva/EIDP, MicroStrategy/Strategy rename, NOW/DNOW, Adeia/Xperi, preferreds and
   warrants).  A detector with that profile is a REPORT, and a human promotes a finding
   into `QUARANTINE_RULES`.

   MEASURED ON THE 2026-08-07 CUR3K PANEL (re-measured after the F-7 tokeniser fix):
   425 pairs share >= 3 triples; 3 are name-mismatches and 1 is name-unknown.
     * `058820.KQ`/`CMG` and `058820.KQ`/`0HXW.L` -- the real contamination (twice, because
       Chipotle also has an LSE line).
     * `ALTA.PA`/`AREIT.PA` (Altarea SCA / Altareit SCA) -- a NEW flag, and a LEGITIMATE
       parent/subsidiary pair.  It previously cleared only because `sca`, a legal form, was
       counted as a shared distinctive token; a legal form must never be the thing that
       vouches for a match, so removing it is right and this flag is the price.  Recorded
       here so the next reader does not re-investigate it.
     * `0RJ6.L`/`LOUP.PA` (both "L.d.c. S.a.") -- name-unknown: the name normalises to
       nothing, so it is reported as UNCOMPARABLE rather than called a mismatch.
   So: 1 real finding, 1 known-benign flag, 1 uncomparable, out of 425 raw matches.

REJECTED, AND RECORDED SO IT IS NOT REBUILT
-------------------------------------------
The **scale-break detector** (`totalAssets` jumps >= 5x while `marketCap` moves <= 1.6x)
was evaluated and REJECTED.  It fires on 119 of 2,613 sources, is dominated by de-SPACs,
IPOs and first-full-balance-sheet events, and ranks `058820.KQ` only 24th.  Do not build it.
"""

import datetime as _dt
import os
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  1. THE QUARANTINE                                                          #
# --------------------------------------------------------------------------- #
class QuarantineRule(object):
    """One named contamination window.  `start`/`end` are INCLUSIVE statement dates."""

    __slots__ = ('source', 'start', 'end', 'reason', 'evidence', 'added')

    def __init__(self, source, start, end, reason, evidence, added):
        self.source = source
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.reason = reason
        self.evidence = evidence
        self.added = added

    def label(self):
        return 'vendor_contamination [%s %s..%s: %s]' % (
            self.source, self.start.date(), self.end.date(), self.reason)


QUARANTINE_RULES = [
    QuarantineRule(
        source='058820.KQ',
        start='2020-03-31', end='2022-09-30',
        reason="another issuer's statements served under this ticker "
               "(Chipotle Mexican Grill / CMG), labelled KRW",
        evidence=(
            "Income statement and balance sheet match CMG (Chipotle) to the dollar over "
            "the window while carrying reportedCurrency=KRW; at 2022-12-31 the series "
            "SNAPS to genuine KRW values (~13x scale break inside one name's history, "
            "which reads as spectacular growth at the boundary). marketCap is unaffected "
            "and runs continuous in real KRW throughout, so no market-cap check can see "
            "it. cik is zeroed ('0000000000'), not copied, so CIK is not a detector. The "
            "collision is by COMPANY NAME ('CMG Pharmaceutical' -> 'CMG'), not ticker "
            "root. STILL LIVE AT THE API on 2026-08-08, so a re-fetch re-ingests it. "
            "Stage-1 is untouched (the rows are positions 15-24 of 24 under calcScore's "
            "newest-first head(8)); the backtest IS hit, since backtest_unified defaults "
            "to --buy_years 2020,2021,2022."),
        added='2026-08-08'),
]


def _quarter_start(ts):
    """The first day of the calendar quarter containing `ts` -- the one date convention
    both of this panel's date columns agree on.

    WHY THIS IS NEEDED.  A row has TWO dates: `periodEndDate` (the vendor's statement
    date, e.g. 2022-09-30) and `date` (that same row after `utils.setDatesToQuarterly`,
    i.e. 2022-07-01).  The quarantine window is stated in VENDOR dates, so matching it
    against `date` directly would be off by a quarter.  Normalising BOTH sides to the
    quarter start makes the rule identical under either convention and immune to a future
    change in which column is canonical -- checked on the real panel: it selects exactly
    the 10 contaminated rows of 058820.KQ's 24 and excludes the 2022-12-31 statement that
    begins the genuine series."""
    if ts is None or pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    return pd.Timestamp(ts.year, ((ts.month - 1) // 3) * 3 + 1, 1)


def quarantine_mask(df, rules=None):
    """Row mask (True = QUARANTINED) for a cdx/BoMetric-shaped frame.

    Keys off `periodEndDate` when it is present and parseable, else `date`; both are
    normalised to the quarter start (see _quarter_start).  Returns an all-False mask for
    an unusable frame -- this must never be the thing that crashes a 12-hour run."""
    rules = QUARANTINE_RULES if rules is None else rules
    cols = getattr(df, 'columns', [])
    if df is None or not len(df) or 'source' not in cols:
        return pd.Series(False, index=getattr(df, 'index', None), dtype=bool)

    primary = (pd.to_datetime(df['periodEndDate'], errors='coerce')
               if 'periodEndDate' in cols else pd.Series(pd.NaT, index=df.index))
    fallback = (pd.to_datetime(df['date'], errors='coerce')
                if 'date' in cols else pd.Series(pd.NaT, index=df.index))
    eff = primary.where(primary.notna(), fallback)
    qs = eff.map(_quarter_start)

    mask = pd.Series(False, index=df.index, dtype=bool)
    for rule in rules:
        mask |= ((df['source'] == rule.source)
                 & qs.notna()
                 & (qs >= _quarter_start(rule.start))
                 & (qs <= _quarter_start(rule.end)))
    return mask


def quarantine_records(df, rules=None, price_col='price', mcap_col='marketCap'):
    """(mask, [removal_record]) in the shape `data_quality` already logs removals in, so
    the quarantine appears in `output/removed_data_quality_*.csv` beside every other
    removal and propagates to BoMetric_df through the existing (source, date) pairing."""
    rules = QUARANTINE_RULES if rules is None else rules
    mask = quarantine_mask(df, rules)
    if not mask.any():
        return mask, []
    by_source = {r.source: r for r in rules}
    out = []
    for _idx, row in df[mask].iterrows():
        rule = by_source.get(row.get('source'))
        out.append({
            'source': row.get('source'),
            'date': row.get('date', None),
            'price': row.get(price_col, np.nan),
            'marketCap': row.get(mcap_col, np.nan),
            'removal_reason': (rule.label() if rule is not None
                               else 'vendor_contamination [unnamed rule]'),
        })
    return mask, out


# --------------------------------------------------------------------------- #
#  2. THE DETECTOR                                                            #
# --------------------------------------------------------------------------- #
#  Panel-internal and FREE: it reads the frame the fetch just produced and makes zero API
#  calls.  Three shared (date, revenue, totalAssets) triples between two DIFFERENT issuers
#  is not a coincidence -- those are three independent multi-digit numbers agreeing on
#  three independent dates.
MIN_SHARED_TRIPLES = 3

#  A triple shared by more than this many sources is DEGENERATE (a placeholder value, an
#  all-zero filing) rather than evidence, and pairing it up is quadratic for no signal.
MAX_SOURCES_PER_TRIPLE = 20

#  Significant digits the triple is hashed at.  Float noise from the ingest arithmetic must
#  not break a match; 10 digits is far finer than any real coincidence and far coarser than
#  representation noise.
_SIGFIG = 10

#  Legal-form and share-class noise.  Dropped BEFORE comparing, so "Corteva, Inc." vs
#  "EIDP, Inc." does NOT match on 'inc' -- the whole point is to compare the DISTINCTIVE
#  part of the name.  Kept deliberately short: an over-long list starts deleting real
#  words ("Pharmaceutical", "Grill") and turns every pair into a mismatch.
_GENERIC_NAME_TOKENS = {
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'companies',
    'ltd', 'limited', 'plc', 'llc', 'lp', 'llp', 'sa', 'sas', 'sarl', 'ag', 'nv',
    'bv', 'se', 'spa', 'srl', 'oyj', 'oy', 'ab', 'asa', 'as', 'aps', 'kgaa', 'gmbh',
    'pte', 'pt', 'tbk', 'bhd', 'sdn', 'psc', 'pjsc', 'ojsc', 'jsc', 'kk',
    #  'sca' added 2026-08-08 (reviewer F-7): a Luxembourg/French legal form
    #  (societe en commandite par actions) and the SOLE reason ALTA.PA / AREIT.PA
    #  cleared as a name match -- a legal form must never be the thing that clears a pair.
    'sca',
    'holding', 'holdings', 'group', 'groupe', 'grupo', 'the', 'and', 'of', 'de',
    'class', 'cl', 'series', 'ordinary', 'shares', 'share', 'common', 'stock',
    'adr', 'ads', 'gdr', 'reg', 'sponsored', 'unsponsored', 'new', 'sa/nv',
}


def _fold_accents(s):
    """'Societe Generale' from 'Societe Generale' with any accents -- NFKD-decompose and
    drop the combining marks, so an accented letter stays a LETTER."""
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c))


def normalise_name(name):
    """A company name reduced to its DISTINCTIVE tokens.

    Accents FOLDED (not split on), lower-cased, punctuation split out, legal forms and
    share-class noise dropped.  An empty result means "no distinctive tokens" -- which is
    treated as UNCOMPARABLE, not as a mismatch, because a name we cannot read is not
    evidence of contamination.

    ACCENT FOLDING IS LOAD-BEARING (F-7, reviewer 2026-08-08).  The tokeniser split on
    `[^0-9a-z]`, i.e. on every non-ASCII letter, so "Societe Generale" (accented) shattered
    into {'soci', 'rale'} -- and the fragment 'soci' then appeared in 23 different names on
    the panel, any one of which could clear a pair as a "shared distinctive token".  A
    tokeniser that manufactures collisions out of accented characters is worse than useless
    on a European universe: it makes the name check, which is the ONLY thing separating
    real contamination from legitimate cross-listings, silently unreliable exactly where
    the universe is densest."""
    if not isinstance(name, str):
        return set()
    toks = re.split(r'[^0-9a-z]+', _fold_accents(name).lower())
    return {t for t in toks if t and t not in _GENERIC_NAME_TOKENS and len(t) > 1}


def _triples(cdx_df):
    """{(date, revenue, totalAssets): {source, ...}} over the panel.

    Rows with a missing revenue or totalAssets are SKIPPED (a NaN cannot corroborate
    anything), as are all-zero pairs -- a shell filing zero-zero on both fields matches
    every other shell filing and is noise, not a shared identity."""
    need = ('source', 'date', 'revenue', 'totalAssets')
    cols = getattr(cdx_df, 'columns', [])
    if cdx_df is None or any(c not in cols for c in need):
        return {}
    d = cdx_df[list(need)].copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d['revenue'] = pd.to_numeric(d['revenue'], errors='coerce')
    d['totalAssets'] = pd.to_numeric(d['totalAssets'], errors='coerce')
    d = d.dropna(subset=['date', 'revenue', 'totalAssets'])
    d = d[(d['revenue'] != 0) | (d['totalAssets'] != 0)]

    out = defaultdict(set)
    fmt = '%%.%dg' % _SIGFIG
    for src, dt, rev, ta in zip(d['source'], d['date'], d['revenue'], d['totalAssets']):
        out[(dt.value, fmt % rev, fmt % ta)].add(src)
    return out


def detect_shared_fundamentals(cdx_df, names=None, min_shared=MIN_SHARED_TRIPLES):
    """Source pairs sharing >= `min_shared` (date, revenue, totalAssets) triples.

    Returns a DataFrame ordered worst-first (name mismatches, then shared count):
      source_a, source_b, n_shared, name_a, name_b, shared_tokens, name_match, verdict,
      first_shared_date, last_shared_date

    `name_match` is the discriminator, NOT the detection: sharing three triples is what
    makes a pair INTERESTING (cross-listings do it legitimately, all day); the names not
    matching is what makes it SUSPECT.  A pair whose names cannot be read (either side
    normalises to nothing) is reported as `name_unknown` rather than being called a
    mismatch."""
    names = names or {}
    triples = _triples(cdx_df)

    shared = defaultdict(list)
    for key, srcs in triples.items():
        if len(srcs) < 2 or len(srcs) > MAX_SOURCES_PER_TRIPLE:
            continue
        srcs = sorted(srcs)
        for i in range(len(srcs)):
            for j in range(i + 1, len(srcs)):
                shared[(srcs[i], srcs[j])].append(key[0])

    rows = []
    for (a, b), dates in shared.items():
        if len(dates) < min_shared:
            continue
        na, nb = names.get(a), names.get(b)
        ta, tb = normalise_name(na), normalise_name(nb)
        if not ta or not tb:
            match, verdict = None, 'name_unknown'
        else:
            common = ta & tb
            match = bool(common)
            verdict = 'name_match' if match else 'NAME_MISMATCH'
        rows.append({
            'source_a': a, 'source_b': b, 'n_shared': len(dates),
            'name_a': na or '', 'name_b': nb or '',
            'shared_tokens': ' '.join(sorted((ta & tb) if (ta and tb) else set())),
            'name_match': match, 'verdict': verdict,
            'first_shared_date': str(pd.Timestamp(min(dates)).date()),
            'last_shared_date': str(pd.Timestamp(max(dates)).date()),
        })

    out = pd.DataFrame(rows, columns=['source_a', 'source_b', 'n_shared', 'name_a',
                                      'name_b', 'shared_tokens', 'name_match', 'verdict',
                                      'first_shared_date', 'last_shared_date'])
    if len(out):
        out['_rank'] = out['verdict'].map({'NAME_MISMATCH': 0, 'name_unknown': 1}).fillna(2)
        out = (out.sort_values(['_rank', 'n_shared'], ascending=[True, False])
                  .drop(columns=['_rank']).reset_index(drop=True))
    return out


def run_detector_stage(cdx_df, names=None, run_date=None, outdir='output', verbose=True):
    """The standing POST-FETCH check.  Never raises; returns the flag frame (possibly
    empty) and writes `output/VendorContaminationFlags_<date>.csv`.

    The CSV goes in `output/` rather than getting its own top-level transfer pattern for
    the same reason `DedupSurvivorReport_*.csv` does: `output/` already ships whole via
    Sbocker's `allowlist_dirs`, so evidence cannot be lost to a pattern that stops
    matching after a rename.  It ships because it is the ONLY on-disk record of this
    check having run -- the evidence rule stated in
    `Sbocker.transfer_outputs_to_drive`."""
    try:
        flags = detect_shared_fundamentals(cdx_df, names=names)
    except Exception as e:
        print('[contamination] WARNING: detector did not run (%s: %s)'
              % (type(e).__name__, e), flush=True)
        return pd.DataFrame()

    n_mis = int((flags['verdict'] == 'NAME_MISMATCH').sum()) if len(flags) else 0
    if verbose:
        print('[contamination] shared-fundamentals check: %d source pair(s) share >= %d '
              '(date, revenue, totalAssets) triple(s); %d have NON-MATCHING company names.'
              % (len(flags), MIN_SHARED_TRIPLES, n_mis), flush=True)
        if n_mis:
            bang = '!' * 78
            print('\n' + bang, flush=True)
            print('!!! VENDOR CONTAMINATION SUSPECTED -- %d pair(s) share fundamentals '
                  'across' % n_mis, flush=True)
            print('!!! DIFFERENT company names. This is the shape of the 058820.KQ / CMG',
                  flush=True)
            print('!!! case (Chipotle\'s statements served under a KOSDAQ ticker).',
                  flush=True)
            print('!!! These are REPORTED, NOT removed: legitimate cross-listings and',
                  flush=True)
            print('!!! renames look identical to a machine (Corteva/EIDP, NOW/DNOW,',
                  flush=True)
            print('!!! MicroStrategy/Strategy). Judge each, then add a NAMED rule to',
                  flush=True)
            print('!!! vendor_contamination.QUARANTINE_RULES for any that is real.',
                  flush=True)
            for _, r in flags[flags['verdict'] == 'NAME_MISMATCH'].head(25).iterrows():
                print('!!!   %-12s %-12s shared=%-3d  %r  vs  %r'
                      % (r['source_a'], r['source_b'], r['n_shared'],
                         r['name_a'], r['name_b']), flush=True)
            print(bang + '\n', flush=True)

    try:
        run_date = run_date or _dt.date.today().strftime('%Y-%m-%d')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        path = os.path.join(outdir, 'VendorContaminationFlags_%s.csv' % run_date)
        #  Written even when EMPTY.  "The check ran and found nothing" and "the check did
        #  not run" are different facts, and only one of them is fine.
        flags.to_csv(path, index=False)
        if verbose:
            print('[contamination] flags written to %s (%d row(s))' % (path, len(flags)),
                  flush=True)
    except Exception as e:
        print('[contamination] WARNING: flag CSV not written (%s: %s)'
              % (type(e).__name__, e), flush=True)
    return flags
