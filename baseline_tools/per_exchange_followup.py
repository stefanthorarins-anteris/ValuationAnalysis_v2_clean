"""READ-ONLY follow-up to per_exchange_completeness.py. Offline, no network.

Three questions, all decision-relevant to the 1,046-name European restoration:

  Q1  NAME-LEVEL sector/industry map coverage for the five restored markets. A headcount
      ratio (214 Oslo entries vs 224 incoming) is NOT coverage: the 214 could be a
      disjoint set of symbols. Measured two ways -- an exact intersection on the curated
      live-verified names, and a large-n proxy validated against markets where the panel
      gives ground truth.
  Q2  Where the pool-wide sector-coverage guard actually crosses its thresholds, and what
      a per-exchange check would have to look like to fire on a blackout while staying
      quiet on the legitimate share of names with no FMP profile.
  Q3  False-positive exposure of a marketCap-keyed issuer-dedup edge, measured across the
      whole panel rather than asserted.

EMITS ONLY. No pipeline module is imported for mutation; nothing is written back.
"""

import os
import sys
import itertools
import collections

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import carveOut as co
import universes as un
import per_exchange_completeness as pxc

RESTORED = ('PA', 'AS', 'BR', 'LS', 'OL')
#  Verified incoming counts, from universes.provenance (PAR 577, OSL 224, BRU 107,
#  AMS 103, LIS 35). Keyed by SUFFIX so it lines up with everything else here.
INCOMING = {'PA': 577, 'OL': 224, 'BR': 107, 'AS': 103, 'LS': 35}


def suf(s):
    return s.rsplit('.', 1)[1].strip().upper() if isinstance(s, str) and '.' in s else 'US'


def q1_exact_curated(sm, im):
    """Exact name-level intersection on the curated, live-verified restored names."""
    rows = []
    for s, tag, _r in un.TEST_UNIVERSE:
        if suf(s) not in RESTORED:
            continue
        sec = sm.get(s)
        rows.append({'symbol': s, 'market': suf(s), 'tag': tag,
                     'in_sector_map': s in sm,
                     'sector_value': repr(sec),
                     'usable_sector': co._is_known_sector(sec),
                     'in_industry_map': s in im,
                     'industry_value': im.get(s)})
    return pd.DataFrame(rows)


def q1_proxy(sm, im, panel_measured=None):
    """Large-n proxy: of the symbols FMP served a profile for in each market (industry-map
    membership), what share carry a USABLE sector in the on-disk sector map?

    Validated on the six markets already in the panel, where per_exchange_completeness
    measured the true figure directly.
    """
    rows = []
    for g in sorted({suf(s) for s in im}):
        syms = [s for s in im if suf(s) == g]
        in_sm = [s for s in syms if s in sm]
        usable = [s for s in syms if co._is_known_sector(sm.get(s))]
        smsyms = [s for s in sm if suf(s) == g]
        sm_usable = [s for s in smsyms if co._is_known_sector(sm.get(s))]
        rows.append({
            'market': g,
            'industry_map_n': len(syms),
            'sector_map_n': len(smsyms),
            'sector_map_usable_n': len(sm_usable),
            'ind_syms_present_in_sector_map_pct': 100.0 * len(in_sm) / len(syms),
            'ind_syms_with_USABLE_sector_pct': 100.0 * len(usable) / len(syms),
            'incoming_verified_n': INCOMING.get(g),
            'panel_measured_usable_pct': (panel_measured or {}).get(g),
        })
    return pd.DataFrame(rows).set_index('market')


def q1_disjointness(sm, im):
    """Is the headcount coincidence real overlap, or two different symbol sets?"""
    rows = []
    for g in RESTORED:
        smsyms = {s for s in sm if suf(s) == g}
        sm_usable = {s for s in smsyms if co._is_known_sector(sm.get(s))}
        imsyms = {s for s in im if suf(s) == g}
        rows.append({
            'market': g,
            'incoming_verified': INCOMING[g],
            'sector_map_usable': len(sm_usable),
            'industry_map': len(imsyms),
            'usable_AND_in_industry_map': len(sm_usable & imsyms),
            'usable_but_NOT_in_industry_map': len(sm_usable - imsyms),
            'in_industry_map_but_NO_usable_sector': len(imsyms - sm_usable),
            'headcount_ratio_pct': 100.0 * min(1.0, len(sm_usable) / INCOMING[g]),
            'name_level_ceiling_pct': 100.0 * len(sm_usable & imsyms) / INCOMING[g],
        })
    return pd.DataFrame(rows).set_index('market')


def q1_industry_backfill(sm, im, labels):
    """Could an industry-to-sector backfill recover the uncarvable population?

    Measured on the pool AND on the five incoming markets, because the answer decides
    whether the sector-coverage uncertainty matters at all.
    """
    df = pd.DataFrame({'source': list(labels.index)})
    df['grp'] = df['source'].map(pxc.lgroup)
    df['usable_sector'] = df['source'].map(lambda s: co._is_known_sector(sm.get(s)))
    df['usable_industry'] = df['source'].map(
        lambda s: bool(im.get(s)) and im.get(s) not in co._UNKNOWN_SECTORS)
    unc = df[~df['usable_sector']]
    g = unc.groupby('grp')
    pool = pd.DataFrame({'uncarvable_n': g.size(),
                         'with_usable_industry_n': g['usable_industry'].sum(),
                         'recoverable_pct': 100 * g['usable_industry'].mean()})
    rows = []
    for m in RESTORED:
        ims = {s for s in im if suf(s) == m}
        no_sec = {s for s in ims if not co._is_known_sector(sm.get(s))}
        with_ind = {s for s in no_sec
                    if im.get(s) and im.get(s) not in co._UNKNOWN_SECTORS}
        usable = {s for s in ims if co._is_known_sector(sm.get(s))}
        rows.append({'market': m, 'incoming': INCOMING[m], 'profiled_syms': len(ims),
                     'usable_sector': len(usable), 'no_usable_sector': len(no_sec),
                     'of_those_with_usable_INDUSTRY': len(with_ind),
                     'industry_recovery_pct': 100 * len(with_ind) / max(1, len(no_sec)),
                     'sector_OR_industry_pct': 100 * len(usable | with_ind) / len(ims)})
    return pool, pd.DataFrame(rows).set_index('market')


def q2_guard_curve(sm, labels, thresholds=(0.75, 0.50)):
    """Where the POOL-WIDE guard reading lands as incoming names are added, and how many
    wholly-uncovered names it would take to trip each threshold."""
    pool = list(labels.index)
    N = len(pool)
    present = sum(1 for s in pool if s in sm)
    rows = []
    for newN in (1046, 900, 800, 700, 642, 577, 500, 300, 100):
        r = {'new_names': newN}
        for c in (0.0, 0.25, 0.50, 0.75, 0.95):
            r['cov_%d%%' % int(100 * c)] = 100.0 * (present + c * newN) / (N + newN)
        rows.append(r)
    curve = pd.DataFrame(rows).set_index('new_names')
    trip = []
    for thr in thresholds:
        # present / (N + x) = thr  ->  x = present/thr - N   (worst case: 0% coverage)
        trip.append({'threshold': thr,
                     'uncovered_new_names_needed_to_trip': present / thr - N,
                     'restoration_brings': 1046})
    return curve, pd.DataFrame(trip).set_index('threshold'), N, present


def q2_per_exchange_bound(p0=0.84, alpha=0.01):
    """Binomial lower-tail bound: the lowest usable-sector fraction a LEGITIMATE market of
    size n can show at the alpha level, given the pooled healthy rate p0.

    This is the per-exchange threshold. It auto-widens for small n, which is what keeps it
    quiet on Iceland (n=14) while still firing on the IOB family (n=385).
    """
    from math import comb
    rows = []
    for n in (5, 10, 14, 20, 35, 50, 100, 200, 385, 577, 1000):
        cum = 0.0
        frac = 1.0
        for k in range(n + 1):
            cum += comb(n, k) * p0 ** k * (1 - p0) ** (n - k)
            if cum > alpha:
                frac = k / n
                break
        rows.append({'n': n, 'lower_bound_pct': 100 * frac})
    return pd.DataFrame(rows).set_index('n')


#  The three duplicate-issuer pairs the fix has to catch (verified in the 2026-08-03 audit).
DEDUP_TARGETS = (('0KNY.L', 'TROW'), ('0J3H.L', 'HIG'), ('0R28.L', 'NEM'))


def _dedup_context(d):
    cdx = d['cdx_df']
    syms = list(d['BoScore_df']['source'])
    names = dict(zip(d['Tickers_df']['symbol'], d['Tickers_df']['name']))
    latest = co._latest_raw(cdx, ['revenue', 'netIncome', 'totalAssets',
                                  'weightedAverageShsOut', 'marketCap'])

    def val(s, c):
        if s in latest.index and c in latest.columns:
            v = latest.at[s, c]
            if pd.notna(v) and np.isfinite(v):
                return round(float(v), 4)
        return None

    def rel(a, b):
        if a is None or b is None:
            return None
        dn = max(abs(a), abs(b))
        return abs(a - b) / dn if dn else 0.0

    comps, _l, _v = co._issuer_components(syms, cdx, names)
    comp_of = {m: r for r, mem in comps.items() for m in mem}
    mcg = collections.defaultdict(list)
    for s in syms:
        v = val(s, 'marketCap')
        if v is not None and v > 0:
            mcg[v].append(s)
    return syms, names, val, rel, comp_of, mcg


def q3_edge_variants(d):
    """Yield and FALSE-POSITIVE exposure of each candidate marketCap-keyed dedup edge.

    Every variant is keyed on byte-identical POSITIVE marketCap; they differ only in the
    guard applied on top. Reported per variant: how many new pairs it joins, how many
    distinct issuer components it merges, how many joined pairs have DIVERGENT normalized
    names (the FP-candidate population that then needs eyeballing), and whether it catches
    the three known duplicate-issuer pairs.
    """
    syms, names, val, rel, comp_of, mcg = _dedup_context(d)

    def nm(s):
        return co._norm_issuer_name(names.get(s, ''))

    def namematch(a, b):
        return bool(nm(a)) and nm(a) == nm(b)

    def near_k(a, b, k, tol=0.05):
        ok = sum(1 for c in ('revenue', 'netIncome', 'totalAssets')
                 if (rel(val(a, c), val(b, c)) if rel(val(a, c), val(b, c)) is not None
                     else 9) <= tol)
        return ok >= k

    def shares_within(a, b, tol):
        r = rel(val(a, 'weightedAverageShsOut'), val(b, 'weightedAverageShsOut'))
        return r is not None and r <= tol

    variants = {
        'D0 marketCap alone (NAIVE)': lambda a, b: True,
        'D1 + name match': namematch,
        'D2 + shares within 5pct': lambda a, b: shares_within(a, b, 0.05),
        'D3 + >=2 of rev/NI/TA near-equal': lambda a, b: near_k(a, b, 2),
        'D4 + name match + shares 5pct': lambda a, b: namematch(a, b) and shares_within(a, b, 0.05),
        'D5 + (name match OR >=2 near-equal) + shares 5pct':
            lambda a, b: (namematch(a, b) or near_k(a, b, 2)) and shares_within(a, b, 0.05),
        'D6 + all 3 near-equal': lambda a, b: near_k(a, b, 3),
    }
    rows, detail = [], {}
    for label, ok in variants.items():
        joins = []
        for _k, g in mcg.items():
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    a, b = g[i], g[j]
                    if comp_of[a] == comp_of[b]:
                        continue
                    if ok(a, b):
                        joins.append((a, b))
        nd = [(a, b) for a, b in joins if nm(a) != nm(b)]
        caught = [p[0] for p in DEDUP_TARGETS
                  if any({a, b} == set(p) for a, b in joins)]
        rows.append({'variant': label, 'new_pairs': len(joins),
                     'component_merges': len({(comp_of[a], comp_of[b]) for a, b in joins}),
                     'name_divergent_pairs': len(nd),
                     'targets_caught': '%d/3' % len(caught),
                     'which': ','.join(caught)})
        detail[label] = nd
    return pd.DataFrame(rows).set_index('variant'), detail
