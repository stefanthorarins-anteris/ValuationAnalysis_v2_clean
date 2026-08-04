"""verify_test_universe.py -- OFFLINE report on what the curated TEST universe covers.

    python verify_test_universe.py

WHY THIS EXISTS RATHER THAN A COMMENT CLAIMING COVERAGE
======================================================
`universes.TEST_UNIVERSE` asserts that its 142 names span both reporting frequencies,
several exchanges and currencies including the newly-restored European ones, all five
carve-out cohorts, the market-cap bands down to sub-50M IN REPORTING CURRENCY (see the
label note below -- NOT USD), and a set of named edge cases.
A claim like that decays: companies accrete history and graduate out of the
short-history slot, names delist, exchanges reclassify, and a cohort tag moves when the
sector map is rebuilt.  A comment would go stale silently.  This measures it instead,
against the repo's OWN classifiers (`reporting_period` for frequency,
`carveOut.classify` for cohort, the saved run's own fail buckets for the gate cases), so
the coverage claim is OBSERVED on every check rather than trusted.

TWO SOURCES OF TRUTH, KEPT SEPARATE ON PURPOSE
----------------------------------------------
  * PANEL-OBSERVED -- measured from a locally saved `Bometric_dic-*.pickle`.  Covers the
    ~114 members that were in the universe before the fix.
  * MANIFEST-DECLARED -- the newly-restored European members (PAR/AMS/BRU/LIS/OSL) are
    in NO saved panel, because the broken exchange codes meant no run ever fetched them.
    Their frequency and currency were verified by live API call on 2026-08-02 and are
    recorded in the manifest rationale.  They are reported as DECLARED, never merged
    into the observed counts, because presenting a hand-recorded fact as a measurement
    is exactly how an unverifiable claim gets laundered into a verified one.

NO NETWORK.  This script makes zero API calls.
"""

import glob
import os
import re
import sys

import pandas as pd

import universes as un

TEST_NAME = 'stock_TEST1'

#  Exchange codes the test universe must span -- the full stock_NA1_EU1 set, so the
#  test universe itself exercises the restored codes rather than only the old ones.
REQUIRED_EXCHANGES = tuple(un.exchanges('stock_NA1_EU1'))

#  Cohorts that must be populated, and how many members each needs to make the
#  cohort code path do real work.  12 is enough to RUN cohort-relative scoring; it is
#  nowhere near enough for the resulting numbers to mean anything (see universes.py).
REQUIRED_COHORTS = ('REIT', 'Mining', 'InvestmentVehicle', 'FinManager',
                    'BalanceSheetFin')
MIN_PER_COHORT = 10

#  Tags whose presence is the whole point of the curation.
REQUIRED_TAGS = ('eu-restored', 'edge-dupdate', 'edge-dedup', 'edge-zerocap',
                 'edge-lenfail', 'edge-datefail', 'edge-pricefail', 'edge-tiny',
                 'filter-remove', 'filter-keep', 'ccy-extreme', 'ccy-isk', 'ccy-gbp',
                 'ccy-sek', 'ccy-cad', 'cohort-fill')


#  Panel filenames carry the run date as `..._all_YYYY-MM-DD_len...`.
_PANEL_DATE_RE = re.compile(r'_all_(\d{4}-\d{2}-\d{2})_')


def _panel_date(path):
    m = _PANEL_DATE_RE.search(os.path.basename(path))
    return m.group(1) if m else ''


def panel_universe(path):
    """The universe name embedded in a panel filename, or None if unrecognisable.

    Matches the LONGEST registry name, because the names nest as substrings:
    `stock_NA1` occurs inside `stock_NA1_EU1`, so a shortest/first match would
    mis-attribute every default-universe panel.
    """
    base = os.path.basename(path)
    for name in sorted(un.names(), key=len, reverse=True):
        if ('_%s_' % name) in base:
            return name
    return None


def is_explicit_list_panel(path):
    """True when this panel came from a CURATED, EXPLICIT-TICKER-LIST universe.

    The discriminator is the registry's own (`universes.symbols(name) is not None`) -- the
    same field `universes.run_banner` keys its TEST-UNIVERSE warning off -- rather than a
    hard-coded 'TEST1'.  A second curated universe added later is covered automatically.
    """
    u = panel_universe(path)
    return u is not None and un.symbols(u) is not None


def newest_panel(pattern='Bometric_dic-*.pickle', production_only=False):
    """Newest locally saved panel, or None.  A panel is dev-machine-only (gitignored,
    ~140 MB), so absence is normal and must degrade to a partial report, not an error.

    TWO DEFECTS FIXED HERE (2026-08-04), and the second is the one that matters:

    1. "Newest" SORTED LEXICOGRAPHICALLY, so it was not newest.  Panel names embed the
       universe before the date, and `stock_TEST1` > `stock_NA1_EU1` on a string compare, so
       a curated-universe panel outranked a production one REGARDLESS of date.  Now sorted on
       the embedded run date (mtime as the tie-break), which is what the name claims.

    2. But fixing the sort alone does NOT fix the bug it caused, and that is the point:
       2026-08-04 > 2026-01-08, so a date sort ALSO picks the curated panel.  The real defect
       is SEMANTIC -- `derive_divergent` reconstructs "the production pool" from the panel's
       own `Tickers_df`, and a curated panel's Tickers_df contains only the ~140 curated
       members.  Deriving the production pool from the test universe and then reconciling the
       test universe against it is CIRCULAR: siblings that production sees (BW, HRZN, WHLR)
       cannot exist in a 126-name panel, so every declared open group reads as
       over-declared.  `production_only=True` excludes curated panels for exactly that call.

    This was reproducible for anyone the moment a `stock_TEST1` run wrote a panel -- which the
    README recommends as the standard iteration route -- so it was a live trap, not a
    theoretical one.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = glob.glob(os.path.join(here, pattern))
    if production_only:
        cands = [p for p in cands if not is_explicit_list_panel(p)]
    if not cands:
        return None
    return max(cands, key=lambda p: (_panel_date(p), os.path.getmtime(p)))


def _exchange_of(symbol, tickers_df):
    if tickers_df is not None and 'symbol' in tickers_df.columns:
        hit = tickers_df[tickers_df['symbol'] == symbol]
        if len(hit):
            return str(hit.iloc[0]['exchangeShortName'])
    #  Fall back to the ticker SUFFIX, which is deterministic per exchange on FMP
    #  (verified 2026-08-02: every .PA is PAR, every .OL is OSL, and so on).  Used only
    #  for members absent from the saved panel -- i.e. the newly-restored ones.
    suffix_map = {'PA': 'PAR', 'AS': 'AMS', 'BR': 'BRU', 'LS': 'LIS', 'OL': 'OSL',
                  'L': 'LSE', 'DE': 'XETRA', 'ST': 'STO', 'TO': 'TSX', 'IC': 'ICE'}
    if '.' in symbol:
        return suffix_map.get(symbol.rsplit('.', 1)[1], '?')
    return '?'          # a suffix-less ticker is US, but NYSE vs NASDAQ is unknowable


def _declared_frequency(reason):
    """Frequency the MANIFEST claims, for members no saved panel can measure."""
    r = reason.lower()
    if 'semi-annual' in r or 'semiannual' in r:
        return 'semiannual'
    if 'quarterly' in r:
        return 'quarterly'
    return None


def _declared_currency(reason):
    m = re.match(r'^([A-Z]{2,6})/([A-Z]{3})\b', reason)
    return m.group(2) if m else None


def collect(panel_path=None, verbose=True):
    """Measure the test universe's coverage.  Returns a plain dict (assertable)."""
    manifest = un.test_universe_manifest()
    members = [s for s, _t, _r in manifest]
    tags = {s: t for s, t, _r in manifest}
    reasons = {s: r for s, _t, r in manifest}

    out = {
        'name': TEST_NAME,
        'fingerprint': un.definition_fingerprint(TEST_NAME),
        'n_members': len(members),
        'n_unique': len(set(members)),
        'members': tuple(members),
        'tag_counts': {t: sum(1 for x in tags.values() if x == t) for t in set(tags.values())},
        'missing_rationale': tuple(s for s in members if not reasons[s].strip()),
        'panel': None,
        'panel_observed': {},
        'manifest_declared': {},
        'exchange_counts': {},
        'cohort_counts': {},
        'frequency_counts': {},
        'unclassified': (),
    }

    panel_path = panel_path or newest_panel()
    if panel_path is None:
        if verbose:
            print('NO saved Bometric_dic-*.pickle found -- reporting MANIFEST-DECLARED '
                  'coverage only. Panel-observed coverage is unavailable on this machine.')
        tickers_df = None
    else:
        out['panel'] = os.path.basename(panel_path)
        d = pd.read_pickle(panel_path)
        cdx = d['cdx_df']
        tickers_df = d.get('Tickers_df')
        names_map = dict(zip(tickers_df['symbol'], tickers_df['name'])) if tickers_df is not None else {}

        import reporting_period as rp
        import carveOut as co

        freq = rp.frequency_by_source(cdx, verbose=False)
        if isinstance(freq, tuple):
            freq = freq[0]

        in_panel = [s for s in members if s in set(cdx['source'])]
        sub = cdx[cdx['source'].isin(in_panel)]
        g = sub.groupby('source')
        rows = g.size()
        dup = g['date'].apply(lambda s: int(len(s) - s.nunique()))
        ni = g['netIncome'].first()
        mc = g['marketCap'].first()

        sector_map = co._load_sector_map()
        industry_map = co._load_industry_map()
        fund = co._latest_fundamentals(cdx)
        lab, _rsn = co.classify(in_panel, sector_map, fund,
                                {s: names_map.get(s, '') for s in in_panel}, industry_map)

        obs = {}
        for s in in_panel:
            obs[s] = {
                'exchange': _exchange_of(s, tickers_df),
                'frequency': freq.get(s),
                'cohort': lab.get(s),
                'rows': int(rows.get(s, 0)),
                'dup_dates': int(dup.get(s, 0)),
                'loss_maker': bool(ni.get(s, 0) < 0),
                'marketCap_rc': float(mc.get(s, float('nan'))),
            }
        out['panel_observed'] = obs

        #  fail buckets from the saved run -- the ONLY evidence for the gate cases
        out['fail_buckets'] = {k: tuple(s for s in members if s in set(d.get(k, [])))
                               for k in ('lenfail', 'datefail', 'emptyfail', 'pricefail')}
        out['unclassified'] = tuple(s for s in members if s not in obs)

    #  manifest-declared facts for everything the panel cannot see
    for s in out['unclassified'] or members:
        if s in out['panel_observed']:
            continue
        out['manifest_declared'][s] = {
            'tag': tags[s],
            'exchange': _exchange_of(s, tickers_df),
            'frequency_declared': _declared_frequency(reasons[s]),
            'currency_declared': _declared_currency(reasons[s]),
        }

    #  combined spreads: exchange is knowable for every member (suffix fallback);
    #  frequency/cohort only where observed, plus declared shown separately.
    ex = {}
    for s in members:
        code = (out['panel_observed'].get(s) or out['manifest_declared'].get(s, {})).get('exchange', '?')
        ex[code] = ex.get(code, 0) + 1
    out['exchange_counts'] = ex
    for s, v in out['panel_observed'].items():
        out['cohort_counts'][v['cohort']] = out['cohort_counts'].get(v['cohort'], 0) + 1
        out['frequency_counts'][v['frequency']] = out['frequency_counts'].get(v['frequency'], 0) + 1

    return out


def derive_divergent(panel_path=None, production=None):
    """Re-derive the divergent member set with the pipeline's OWN issuer grouping.

    Uses `carveOut._issuer_components` -- the function `dedup_to_issuers` and
    `dedup_ranked` both resolve issuer identity with -- NOT a normalised-name match.  That
    distinction is the whole point: name-matching finds cross-listings but cannot tell a
    cross-listing from a baby bond or a preferred GDR, because those carry the common's
    name AND its statements verbatim.  The first closure audit used names, declared one
    open group, and was wrong in three places.

    Returns (derived, note).  `derived` maps member -> sorted missing siblings.

    LIMITATION, stated rather than hidden: a fingerprint needs statements, so members or
    siblings absent from the saved panel are invisible here.  Those divergences are
    evidenced from live-list membership instead and are listed in
    `universes.OPEN_GROUPS_LIVE_LIST_ONLY`.

    THE PANEL MUST BE A PRODUCTION ONE (`production_only=True`, fixed 2026-08-04).  This
    function's job is to compare the CURATED universe against the pool PRODUCTION sees, and it
    reconstructs that pool from the panel's own `Tickers_df`.  Handed a curated-universe panel
    it therefore derives "production" from the very 140 names under test -- circular, and it
    reports every declared open group as over-declared because the siblings production sees
    are not in a 126-name panel.  Caller-supplied `panel_path` is honoured as-is; only the
    DEFAULT is constrained.
    """
    panel_path = panel_path or newest_panel(production_only=True)
    if panel_path is None:
        return None, 'no saved panel on this machine -- derivation unavailable'
    import carveOut as co
    d = pd.read_pickle(panel_path)
    cdx = d['cdx_df']
    tickers = d.get('Tickers_df')
    names = (dict(zip(tickers['symbol'], tickers['name']))
             if tickers is not None and 'name' in getattr(tickers, 'columns', []) else {})
    members = set(un.symbols(TEST_NAME))
    panel_syms = set(cdx['source'].unique())
    if production is None:
        #  PRODUCTION STAND-IN, built offline from the panel's own Tickers_df.
        #  The raw panel is NOT a stand-in for production: it was fetched before
        #  `filter_non_common_instruments` existed, so it still contains preferred and
        #  warrant lines (MS-PA, WHLRP, SQFTP, NCPLW, BW-PA...). Using it raw reported six
        #  spurious divergences against siblings production never sees. Applying the same
        #  instrument filter and exchange restriction the pipeline applies reconstructs a
        #  faithful pool without needing the live list.
        tdf = tickers
        if tdf is not None and 'symbol' in getattr(tdf, 'columns', []):
            import getData_gen as gdg
            kept = gdg.filter_non_common_instruments(tdf.copy(), verbose=False,
                                                    log_csv=False)
            wired = set(un.exchanges('stock_NA1_EU1'))
            if 'exchangeShortName' in kept.columns:
                kept = kept[kept['exchangeShortName'].isin(wired)]
            production = set(kept['symbol'])
        else:
            production = panel_syms
    pool = sorted(set(production) & panel_syms)
    comps, _latest, _val = co._issuer_components(pool, cdx, names)
    derived = {}
    for _root, group in comps.items():
        if len(group) < 2:
            continue
        mine = [s for s in group if s in members]
        missing = [s for s in group if s not in members]
        if mine and missing:
            for m in mine:
                derived.setdefault(m, set()).update(missing)
    return {k: sorted(v) for k, v in derived.items()}, 'derived via carveOut._issuer_components'


def reconcile_open_groups(panel_path=None, production=None):
    """Reconcile the DERIVED divergent set against the DECLARED one, BOTH directions.

    Returns a list of problem strings; empty means the declaration is exact.
      * derived-but-not-declared  -> a real HIDDEN divergence (the defect direction);
      * declared-but-not-derived  -> only acceptable for the entries whose evidence is
        live-list membership rather than a fundamentals fingerprint.
    """
    derived, note = derive_divergent(panel_path, production)
    if derived is None:
        return ['reconciliation unavailable: %s' % note]
    declared = set(un.TEST_UNIVERSE_OPEN_GROUPS)
    allowed_undermined = set(un.OPEN_GROUPS_LIVE_LIST_ONLY)
    problems = []

    undeclared = sorted(set(derived) - declared)
    if undeclared:
        problems.append(
            'HIDDEN DIVERGENCE -- derived but NOT declared: %s (each has a same-issuer '
            'sibling in the pool that is absent from the list)'
            % {k: derived[k] for k in undeclared})

    extra = sorted(declared - set(derived) - allowed_undermined)
    if extra:
        problems.append(
            'OVER-DECLARED -- declared open but no divergence derivable, and not listed '
            'in OPEN_GROUPS_LIVE_LIST_ONLY: %s' % extra)

    #  where BOTH agree, the sibling sets must match too -- a declaration naming the wrong
    #  sibling is a declaration that does not describe the divergence it claims to.
    for m in sorted(set(derived) & declared):
        want = set(derived[m])
        got = set(un.TEST_UNIVERSE_OPEN_GROUPS[m][:-1])
        if want != got:
            problems.append('%s: declared siblings %s but derivation says %s'
                            % (m, sorted(got), sorted(want)))

    #  every declared sibling must be genuinely absent from the list
    members = set(un.symbols(TEST_NAME))
    for m, spec in un.TEST_UNIVERSE_OPEN_GROUPS.items():
        if m not in members:
            problems.append('%s is declared an open group but is not a member' % m)
        for sib in spec[:-1]:
            if sib in members:
                problems.append('%s is declared an absent sibling of %s but IS a member'
                                % (sib, m))
    return problems


def gaps(cov):
    """Category requirements NOT met.  Empty tuple = full coverage.

    Deliberately returns the gaps rather than raising: the honest state of a curated
    list is that some slots decay (a 2024 IPO graduates out of `edge-lenfail`), and the
    right response is to SEE that and re-curate, not to have a green test that stopped
    looking.
    """
    problems = []
    if cov['n_unique'] != cov['n_members']:
        problems.append('duplicate members in the list')
    if cov['missing_rationale']:
        problems.append('members with no rationale: %s' % (cov['missing_rationale'],))

    missing_ex = [c for c in REQUIRED_EXCHANGES if cov['exchange_counts'].get(c, 0) == 0]
    #  A suffix-less US ticker resolves to '?' without a panel, so NYSE/NASDAQ are only
    #  checkable when a panel is present.
    if cov['panel'] is None:
        missing_ex = [c for c in missing_ex if c not in ('NYSE', 'NASDAQ')]
    if missing_ex:
        problems.append('exchange codes with no member: %s' % missing_ex)

    missing_tags = [t for t in REQUIRED_TAGS if cov['tag_counts'].get(t, 0) == 0]
    if missing_tags:
        problems.append('required tags with no member: %s' % missing_tags)

    if cov['panel_observed']:
        thin = {c: cov['cohort_counts'].get(c, 0) for c in REQUIRED_COHORTS
                if cov['cohort_counts'].get(c, 0) < MIN_PER_COHORT}
        if thin:
            problems.append('cohorts below %d members: %s' % (MIN_PER_COHORT, thin))
        for f in ('quarterly', 'semiannual'):
            if cov['frequency_counts'].get(f, 0) == 0:
                problems.append('no %s filer observed' % f)
        obs = cov['panel_observed']
        if not any(v['dup_dates'] > 0 for v in obs.values()):
            problems.append('no member with duplicate-dated rows')
        if not any(v['marketCap_rc'] <= 0 or v['marketCap_rc'] != v['marketCap_rc']
                   for v in obs.values()):
            problems.append('no member with a zero/missing marketCap')
        if not any(v['loss_maker'] for v in obs.values()):
            problems.append('no loss-making member')
        if not any(v['marketCap_rc'] > 0 and v['marketCap_rc'] < 5e7 for v in obs.values()):
            problems.append('no sub-50M-in-REPORTING-CURRENCY member')
        if not cov.get('fail_buckets', {}).get('lenfail'):
            problems.append('no member currently failing the history gate (lenfail) -- '
                            'the short-history slot has graduated; re-curate')
    return tuple(problems)


def report(panel_path=None):
    cov = collect(panel_path)
    bar = '=' * 78
    print(bar)
    print('  TEST UNIVERSE COVERAGE -- %s' % cov['name'])
    print('  fingerprint : %s' % cov['fingerprint'])
    print('  members     : %d (%d unique)' % (cov['n_members'], cov['n_unique']))
    print('  panel       : %s' % (cov['panel'] or 'NONE (manifest-declared only)'))
    print(bar)

    print('\nEXCHANGE SPREAD (every member; suffix fallback for names absent from the panel)')
    for c, n in sorted(cov['exchange_counts'].items(), key=lambda kv: -kv[1]):
        mark = ' <- restored 2026-08-02' if c in ('PAR', 'AMS', 'BRU', 'LIS', 'OSL') else ''
        print('  %-8s %3d%s' % (c, n, mark))

    print('\nTAG INVENTORY (why each member is in the list)')
    for t, n in sorted(cov['tag_counts'].items(), key=lambda kv: -kv[1]):
        print('  %-16s %3d' % (t, n))

    if cov['panel_observed']:
        obs = cov['panel_observed']
        n = len(obs)
        print('\nPANEL-OBSERVED (%d of %d members)' % (n, cov['n_members']))
        print('  frequency      : %s' % cov['frequency_counts'])
        print('                   semi-annual share %.0f%% of observed (universe-wide '
              '~14%%; over-weighted ON PURPOSE -- the recent defects concentrated there)'
              % (100.0 * cov['frequency_counts'].get('semiannual', 0) / max(n, 1)))
        print('  cohorts        : %s' % cov['cohort_counts'])
        print('  loss-makers    : %d (%.0f%%; universe-wide ~36%%)'
              % (sum(1 for v in obs.values() if v['loss_maker']),
                 100.0 * sum(1 for v in obs.values() if v['loss_maker']) / max(n, 1)))
        print('  zero/NaN cap   : %d' % sum(1 for v in obs.values()
                                            if not (v['marketCap_rc'] > 0)))
        #  `marketCap` is stored in each company's REPORTING CURRENCY, mixed across the
        #  universe (DORO.ST = SEK 962M is about $92M), so this is NOT a USD band and must
        #  not be read as one. A USD band needs the `marketCap_usd` the currency work adds.
        print('  sub-50M cap in REPORTING CCY (not USD): %d' % sum(1 for v in obs.values()
                                            if 0 < v['marketCap_rc'] < 5e7))
        print('  dup-dated rows : %s' % {s: v['dup_dates'] for s, v in obs.items()
                                         if v['dup_dates'] > 0})
        print('  gate buckets   : %s' % cov.get('fail_buckets'))

    if cov['manifest_declared']:
        print('\nMANIFEST-DECLARED (%d members no saved panel can measure -- the newly'
              '-restored\nEuropean names, live-verified 2026-08-02, NOT merged into the '
              'observed counts)' % len(cov['manifest_declared']))
        for s, v in sorted(cov['manifest_declared'].items()):
            print('  %-11s %-5s freq=%-11s ccy=%-4s [%s]'
                  % (s, v['exchange'], v['frequency_declared'] or '-',
                     v['currency_declared'] or '-', v['tag']))

    #  ISSUER-GROUP CLOSURE.  Reported, not silently assumed: carveOut's dedup edges are
    #  pairwise over the pool, so a member whose same-issuer sibling is absent is deduped
    #  differently here than in production. 17 partners were added to close 16 such
    #  groups; the one that cannot be closed without importing known non-common lines is
    #  DECLARED. Full closure verification needs the LIVE list (this script is offline),
    #  so what is checked here is that the declaration still matches the list.
    print('\nISSUER-GROUP CLOSURE')
    n_partners = cov['tag_counts'].get('dedup-partner', 0)
    print('  closure partners in the list : %d' % n_partners)
    mem = set(cov['members'])
    for member, spec in sorted(un.TEST_UNIVERSE_OPEN_GROUPS.items()):
        sibs, why = spec[:-1], spec[-1]
        basis = ('fingerprint-derivable'
                 if member in un.OPEN_GROUPS_FINGERPRINT_DERIVABLE else 'live-list only')
        state = 'OK (declared)' if (member in mem and not any(s in mem for s in sibs)) \
            else 'STALE -- the declaration no longer matches the list'
        print('  DECLARED OPEN GROUP %-8s siblings absent by design: %-16s [%s; %s]'
              % (member, ', '.join(sibs), state, basis))
        print('      %s' % why)

    #  RECONCILED, not asserted: re-derive with carveOut's own issuer grouping and compare
    #  BOTH directions. An under-declared list hides divergence; an over-declared one is
    #  just as misleading. The guard this replaces was a loop with no assertion.
    rec = reconcile_open_groups(cov.get('panel') and
                               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            cov['panel']))
    if rec:
        print('  RECONCILIATION PROBLEMS:')
        for p in rec:
            print('    - %s' % p)
    else:
        print('  RECONCILIATION: derived divergent set matches the declaration exactly '
              '(both directions).')

    g = gaps(cov)
    print('\n' + bar)
    if g:
        print('  COVERAGE GAPS (%d) -- the list needs re-curating:' % len(g))
        for p in g:
            print('    - %s' % p)
    else:
        print('  COVERAGE: every required category is populated.')
    print('  REMINDER: pool-relative output from this universe (z-scores, percentiles,')
    print('  top-100, top-20) is NOT comparable to production. See universes.py.')
    print(bar)
    return cov, g


if __name__ == '__main__':
    _cov, _g = report(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0)
