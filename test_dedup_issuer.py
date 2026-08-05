"""Regression gates for ONE-LISTING-PER-ISSUER dedup (carveOut, 2026-08-05).

WHOLLY OFFLINE.  Not one test here touches the network.  Two evidence bases, kept apart
because conflating them is how an unverified claim gets a green tick:

  PANEL-JAN  `postRank_2026-01-09_fmp_stock_NA1_EU1.pickle` -- 8,106 lines WITH
             fundamentals (NA + Europe-incl-London).  Real statements, so the GROUPING
             keys can actually be exercised.  It contains NO Asian and NO Amsterdam /
             Paris fundamentals, which is exactly why the Continental and Korean
             MUST-MERGE cases below are gated rather than asserted.
  LIVE-AUG   `available_traded_raw_2026-08-04.pickle` -- 51,703 type=='stock' lines,
             SYMBOLS AND NAMES ONLY, no fundamentals.  Enough to exercise the PICKING
             half (a pure function of symbol + name) and the family shapes, never the
             grouping half.

WHAT IS STILL UNVERIFIED, AND SAYS SO OUT LOUD.  The MUST-NOT-MERGE case the register
cares most about (HEIA.AS vs HEIO.AS) and the whole Korea MUST-MERGE gate need
fundamentals that do not exist locally.  Those tests SKIP with an explicit reason rather
than passing vacuously, and `test_the_unverified_gates_are_declared` asserts that the
skip list is DECLARED -- so "we never ran it" cannot decay into "it passed".
"""

import os

import numpy as np
import pandas as pd
import pytest

import carveOut as co
import getData_gen as gdg

_HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(_HERE, 'postRank_2026-01-09_fmp_stock_NA1_EU1.pickle')
LIVE = os.path.join(_HERE, 'available_traded_raw_2026-08-04.pickle')


# --------------------------------------------------------------------------- #
#  FIXTURES                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def raw_capture():
    if not os.path.exists(LIVE):
        pytest.skip('LIVE-AUG name capture absent: %s' % os.path.basename(LIVE))
    return pd.read_pickle(LIVE)


@pytest.fixture(scope='module')
def live(raw_capture):
    return raw_capture[raw_capture['type'] == 'stock'].reset_index(drop=True)


@pytest.fixture(scope='module')
def panel(raw_capture):
    """PANEL-JAN with names resolved from the FULL capture (every `type`, not just
    'stock').

    THE NAME BASIS IS LOAD-BEARING AND IS STATED RATHER THAN ASSUMED.  K3 keys on the
    name, so which name table you hand it changes the grouping slightly: resolving from
    the full capture gives 6,328 components (the figure the design spec measured and the
    figures pinned below), while resolving from the type=='stock' rows ONLY gives 6,330 --
    two lines that PANEL-JAN carries are typed non-stock in the August capture, so they
    lose their K3 edge.  Both are correct answers to different questions; the pinned
    numbers belong to this one.  Production takes names from the filtered universe
    (type=='stock'), so expect the production figure to be the 6,330-flavoured one.  The
    ZERO-REGRESSION property below holds on either basis -- it compares old and new under
    the SAME names.
    """
    if not os.path.exists(PANEL):
        pytest.skip('PANEL-JAN absent: %s' % os.path.basename(PANEL))
    d = pd.read_pickle(PANEL)
    syms = list(d['moatdf']['source'])
    nm = dict(zip(raw_capture['symbol'], raw_capture['name']))
    names = {s: nm.get(s, '') for s in syms}
    comps, latest, val = co._issuer_components(syms, d['cdx_df'], names)
    root = {s: r for r, m in comps.items() for s in m}
    return {'d': d, 'cdx': d['cdx_df'], 'syms': syms, 'names': names,
            'comps': comps, 'root': root, 'val': val,
            'ranked': list(d['postRank']['source'])}


def _same_group(panel, *symbols):
    """(verdict, present) -- verdict in {'merged', 'split', 'absent'}."""
    present = [s for s in symbols if s in panel['root']]
    if len(present) < 2:
        return 'absent', present
    return ('merged' if len({panel['root'][s] for s in present}) == 1
            else 'split'), present


# --------------------------------------------------------------------------- #
#  GROUPING -- THE ZERO-REGRESSION / STRICT-SUPERSET PROPERTY                   #
#                                                                               #
#  This is the change's own success criterion (the design spec is explicit that it #
#  must NOT be judged by whether the top-20 improves).  The numbers are pinned as   #
#  DATA so a future edit that quietly loses merges cannot pass.                     #
# --------------------------------------------------------------------------- #
PANEL_JAN_GROUPING = {
    'lines': 8106,
    'components': 6328,
    'multi_line_groups': 1282,
    'lines_dropped': 1778,
    'pairs': 2842,
    #  The pre-change A/B/C edge set, measured on the same panel before the rewrite.
    'previous_components': 6437,
    'previous_multi_line_groups': 1236,
    'previous_lines_dropped': 1669,
    'previous_pairs': 2595,
}


def _pairs(comps):
    out = set()
    for m in comps.values():
        m = sorted(m)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                out.add((m[i], m[j]))
    return out


def test_grouping_matches_the_measured_specification(panel):
    f = PANEL_JAN_GROUPING
    comps = panel['comps']
    assert len(panel['syms']) == f['lines']
    assert len(comps) == f['components']
    assert sum(1 for m in comps.values() if len(m) > 1) == f['multi_line_groups']
    assert len(panel['syms']) - len(comps) == f['lines_dropped']
    assert len(_pairs(comps)) == f['pairs']


def test_the_new_grouping_is_a_STRICT_SUPERSET_of_the_old_one(panel):
    """ZERO REGRESSIONS is the acceptance bar, and it is not implied by "more merges":
    a rewrite can add 300 pairs while silently dropping 30.  So the old A/B/C edge set is
    RECOMPUTED here from its own definition and every pair it found must survive."""
    syms, cdx, names = panel['syms'], panel['cdx'], panel['names']
    latest = co._latest_raw(cdx, ['revenue', 'netIncome', 'totalAssets',
                                  'weightedAverageShsOut', 'marketCap'])

    def val(s, c):
        if s in latest.index and c in latest.columns:
            v = latest.at[s, c]
            if pd.notna(v) and np.isfinite(v):
                return round(float(v), 4)
        return None

    fp_cols = ['revenue', 'netIncome', 'totalAssets', 'weightedAverageShsOut']
    parent = {s: s for s in syms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def near_equal(a, b, cols, tol=0.05):
        for c in cols:
            va, vb = val(a, c), val(b, c)
            if va is None or vb is None:
                return False
            denom = max(abs(va), abs(vb))
            if denom == 0.0:
                if va != vb:
                    return False
                continue
            if abs(va - vb) / denom > tol:
                return False
        return True

    fpmap, nsmap, shmap = {}, {}, {}
    for s in syms:
        v = [val(s, c) for c in fp_cols]
        if all(x is not None for x in v):
            fpmap.setdefault(tuple(v), []).append(s)
        # edge B used the pre-change normaliser, i.e. `Holding` STRIPPED = today's default
        n = co._norm_issuer_name(names.get(s, ''))
        sh = val(s, 'weightedAverageShsOut')
        if n and sh is not None:
            nsmap.setdefault((n, sh), []).append(s)
        if sh is not None and sh > 0:
            shmap.setdefault(sh, []).append(s)
    for grp in shmap.values():
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                if near_equal(grp[i], grp[j], ['revenue', 'netIncome', 'totalAssets']):
                    union(grp[i], grp[j])
    for grp in list(fpmap.values()) + list(nsmap.values()):
        for s in grp[1:]:
            union(grp[0], s)
    old = {}
    for s in syms:
        old.setdefault(find(s), []).append(s)

    f = PANEL_JAN_GROUPING
    assert len(old) == f['previous_components'], (
        'the recomputed A/B/C baseline no longer reproduces the recorded one (%d vs %d) '
        '-- the panel or the normaliser default changed, so the superset claim below is '
        'about a different baseline' % (len(old), f['previous_components']))
    old_pairs, new_pairs = _pairs(old), _pairs(panel['comps'])
    regressions = sorted(old_pairs - new_pairs)
    assert not regressions, (
        'NOT A SUPERSET: %d pair(s) the previous edge set merged are now split, e.g. %s'
        % (len(regressions), regressions[:10]))
    assert len(new_pairs - old_pairs) == f['pairs'] - f['previous_pairs']


def test_grouping_is_SCOPE_INVARIANT(panel):
    """The property the previous scheme did NOT have, and the reason the pairwise
    completeness worry dissolves for the DROP decision: every key is a hash bucket over
    ONE line's own fields, so grouping the emitted 87 rows alone must give exactly what
    grouping all 8,106 and projecting down gives.  This is a claim about the RULE, so it
    is verified rather than restated."""
    ranked = panel['ranked']
    sub_names = {s: panel['names'].get(s, '') for s in ranked}
    sub, _l, _v = co._issuer_components(ranked, panel['cdx'], sub_names)
    projected = {}
    for s in ranked:
        projected.setdefault(panel['root'][s], []).append(s)
    assert ({frozenset(m) for m in sub.values()}
            == {frozenset(m) for m in projected.values()})
    assert sum(1 for m in sub.values() if len(m) > 1) == 19
    assert len(sub) == 65


def test_no_grouping_key_uses_a_tolerance():
    """The fix is a FIELD REMOVED, not a tolerance ADDED.  A future edit that reintroduces
    a threshold would reintroduce the chance-collision surface the design rejected, so the
    absence of the retired knobs is asserted rather than trusted."""
    assert not hasattr(co, '_XLIST_FUND_TOL')
    assert not hasattr(co, '_fund_near_equal')
    assert 'weightedAverageShsOut' not in co._K1_COLS, (
        'shares are the ONE listing-dependent field (register B-7); putting them back '
        'into the statement key is the original defect')


# --------------------------------------------------------------------------- #
#  MUST-NOT-MERGE                                                              #
# --------------------------------------------------------------------------- #
#  Eight name-collision groups where the ISSUERS GENUINELY DIFFER.  A fundamentals
#  fingerprint separates issuers that name matching merges, and this is where that
#  claim is cashed.
MUST_NOT_MERGE_SAME_NAME = (
    ('FBNC', 'FBP', 'FNLC'),        # three different First Bancorps
    ('IBCP', 'INDB'),
    ('GHC', 'GHM'),
    ('DOM.L', 'DPZ'),
    ('OBDC', 'OWL'),
    ('ATER', 'ATN.L'),
    ('SST', 'SYS1.L'),
    ('TORO', 'TTC'),
)


@pytest.mark.parametrize('group', MUST_NOT_MERGE_SAME_NAME,
                         ids=['/'.join(g) for g in MUST_NOT_MERGE_SAME_NAME])
def test_MUST_NOT_MERGE_distinct_issuers_that_share_a_name(panel, group):
    verdict, present = _same_group(panel, *group)
    if verdict == 'absent':
        pytest.skip('fewer than two members in PANEL-JAN: %s' % (present,))
    assert verdict == 'split', (
        '%s were MERGED -- these are different companies. A false merge here deletes a '
        'real issuer from the universe, which is the expensive error.' % (present,))


def test_MUST_NOT_MERGE_heineken_names_cannot_fire_K3(live):
    """HEIA.AS (Heineken N.V.) vs HEIO.AS (Heineken HOLDING N.V.) -- the register's one
    LIVE must-not-merge: both arrive on the AMS code the Europe fix restored.

    THIS TEST COVERS THE NAME HALF ONLY, AND THAT LIMIT IS THE POINT.  K3 is the one key
    whose input is a NAME, so it is the one key a name collision could fire; with
    `Holding` preserved the two normalise apart and K3 provably cannot fire.  K1
    (netIncome) and K2 (marketCap) are what must keep them apart, and neither can be
    checked here -- there are no Amsterdam fundamentals in any local panel.  See
    test_MUST_NOT_MERGE_heineken_fundamentals for the gate that closes it."""
    nm = dict(zip(live['symbol'], live['name']))
    assert 'HEIA.AS' in nm and 'HEIO.AS' in nm, (
        'Heineken lines absent from the live capture -- the case cannot be checked')
    a = co._norm_issuer_name(nm['HEIA.AS'], keep_holding=True)
    b = co._norm_issuer_name(nm['HEIO.AS'], keep_holding=True)
    assert a != b, ('K3 CAN fire on Heineken: %r == %r. `Holding` must survive '
                    'normalisation for the K3 key.' % (a, b))
    #  And the collision is REAL under the default normaliser, so the flag is doing work
    #  rather than guarding a hypothetical.
    assert co._norm_issuer_name(nm['HEIA.AS']) == co._norm_issuer_name(nm['HEIO.AS'])


def test_MUST_NOT_MERGE_heineken_fundamentals(panel):
    """THE OPEN GATE.  Assert HEIA.AS and HEIO.AS land in DIFFERENT components on real
    fundamentals, and record their revenue / netIncome / totalAssets / marketCap /
    weightedAverageShsOut as data the way ISIN_DETECTOR_VERIFIED_FINDINGS does.

    Mechanism it is testing: Holding CONSOLIDATES Heineken N.V. and reports materially
    different netIncome (roughly half, the minority interest), and its market cap is
    around half at a holding discount -- so K1 cannot fire and K2 cannot fire.  That is
    the reason netIncome must stay a REQUIRED and EXACT member of K1.

    SKIPS until a fetch with Amsterdam fundamentals exists.  It must not be deleted or
    weakened to make the suite green: an un-runnable gate is an OPEN item, not a pass."""
    verdict, present = _same_group(panel, 'HEIA.AS', 'HEIO.AS')
    if verdict == 'absent':
        pytest.skip('UNVERIFIED GATE -- no Amsterdam fundamentals in PANEL-JAN (present: '
                    '%s). Re-run against the first fetch that includes AMS.' % (present,))
    assert verdict == 'split', (
        'HEIA.AS and HEIO.AS MERGED. These are two separate issuers; Heineken Holding '
        'consolidates Heineken N.V. Check that netIncome is still REQUIRED and EXACT in '
        'carveOut._K1_COLS and that the K2 marketCap key has not been loosened.')
    for s in ('HEIA.AS', 'HEIO.AS'):
        print('  %s: %s' % (s, {c: panel['val'](s, c) for c in
                                ('revenue', 'netIncome', 'totalAssets', 'marketCap',
                                 'weightedAverageShsOut')}))


def test_MUST_NOT_MERGE_credit_agricole_CCIs(panel, live):
    """CAT31.PA / CRBP2.PA are DIFFERENT COOPERATIVES (Credit Mutuel Toulouse 31 vs Brie
    Picardie) with different names and different statements, so no key can group them.

    The register's original worry was that a SHARE-CLASS FILTER would delete them -- a
    `certificat cooperatif d'investissement` is the ONLY listed equity of a cooperative,
    exactly as units are for an LP.  Canonical-choice cannot delete a group's only
    member, so that worry dissolves structurally; this test pins BOTH halves."""
    nm = dict(zip(live['symbol'], live['name']))
    for s in ('CAT31.PA', 'CRBP2.PA'):
        if s not in nm:
            pytest.skip('%s absent from the live capture' % s)
    assert (co._norm_issuer_name(nm['CAT31.PA'], keep_holding=True)
            != co._norm_issuer_name(nm['CRBP2.PA'], keep_holding=True)), (
        'the two cooperatives normalise to one name, so K3 could fire on them')
    #  Neither may be removed by the share-class filter either.
    sub = live[live['symbol'].isin(['CAT31.PA', 'CRBP2.PA'])].copy()
    kept = gdg.filter_non_common_instruments(sub, verbose=False, log_csv=False)
    assert set(kept['symbol']) == {'CAT31.PA', 'CRBP2.PA'}, (
        'the share-class filter deleted a cooperative CCI -- its only listed equity')
    verdict, present = _same_group(panel, 'CAT31.PA', 'CRBP2.PA')
    if verdict == 'absent':
        pytest.skip('UNVERIFIED on fundamentals -- no Paris fundamentals in PANEL-JAN')
    assert verdict == 'split'


# --------------------------------------------------------------------------- #
#  MUST-MERGE                                                                  #
# --------------------------------------------------------------------------- #
#  Cases PANEL-JAN can actually see.  Each is a distinct MECHANISM, not a repeat:
MUST_MERGE_ON_PANEL = (
    (('BW', 'BW-PA', 'BWNB'), 'K1 groups a senior-notes line whose FMP name is TRUNCATED '
                              '("Babcock & Wilcox Enterprises, I"), which defeats every '
                              'name-based rule'),
    (('AEM', 'AEM.TO', '0R2J.L'), 'three venues incl. an LSE IOB depositary line'),
    (('BKE', '0HQ7.L'), 'the cited exact-AggScore IOB pair'),
    (('EXEL', '0IJO.L'), 'the second cited exact-AggScore IOB pair'),
    (('HNNA', 'HNNAZ'), 'the notes line that beat its common ON SCORE (0.1159 vs 0.1026)'),
    (('HEN.DE', 'HEN3.DE'), 'dual-class ordinary + preferred, one issuer'),
    (('TCL-A.TO', 'TCL-B.TO'), 'dual-class commons'),
    (('ELUX-A.ST', 'ELUX-B.ST'), 'dual-class commons, Nordic'),
    (('ACRI-A.ST', 'ACRI-B.ST'), 'the pair only the name+shares key K3 catches'),
    (('0R4M.L', 'LUG.TO', 'LUG.ST'), 'FX-shifted statements -- K1 cannot fire, K2 does'),
    (('HRZN', 'HTFB', 'HTFC'), 'baby bonds carrying the common\'s fundamentals'),
    (('GSL', 'GSL-PB'), 'preferred series occupying a second top-20 slot'),
    (('SYF', 'SYF-PA'), 'the second such preferred'),
)


@pytest.mark.parametrize('group,why', MUST_MERGE_ON_PANEL,
                         ids=['/'.join(g) for g, _w in MUST_MERGE_ON_PANEL])
def test_MUST_MERGE_on_panel(panel, group, why):
    verdict, present = _same_group(panel, *group)
    if verdict == 'absent':
        pytest.skip('fewer than two members in PANEL-JAN: %s' % (present,))
    assert verdict == 'merged', '%s did not collapse to one issuer (%s)' % (present, why)


#  Cases NO local panel can see: Amsterdam / Paris / Oslo fundamentals do not exist here.
#  Declared so the gap is a KNOWN OPEN ITEM rather than an absence nobody notices.
MUST_MERGE_NEEDS_A_FETCH = (
    (('CBE.PA', 'RBT.PA'), "Robertet certificat d'investissement, -17.9%. THE SOFTEST "
                           'CASE: both lines are named "Robertet S.A." verbatim so K3 '
                           'fires if shares match, but CBE carries NO canonicity marker '
                           '-- picking falls through to shares/mcap then '
                           'punctuation/length. A `certificat` name marker or the ISIN '
                           'key K4 would firm it up.'),
    (('PREVA.AS', 'VALUE.AS'), 'Value8 cumulative preference shares, -29.9%. Same '
                               'mechanism, same softness unless FMP labels PREVA.'),
    (('HAFNI.OL', 'HAFNIO.OL'), 'same ISIN, same name -- HAFNI survives on length'),
    (('CATG.PA', 'ALCAT.PA'), 'same ISIN, same name -- ALCAT survives on length'),
    (('WWI.OL', 'WWIB.OL'), 'Wilh. Wilhelmsen dual-class: identical statements, so no '
                            'fundamentals key CAN separate them and the ruling is that '
                            'they merge'),
)


@pytest.mark.parametrize('group,why', MUST_MERGE_NEEDS_A_FETCH,
                         ids=['/'.join(g) for g, _w in MUST_MERGE_NEEDS_A_FETCH])
def test_MUST_MERGE_continental(panel, live, group, why):
    """Names are checked NOW (all that local data supports); grouping is gated on a fetch.

    The name check is not a formality: for every one of these the whole merge rests on
    the two lines carrying the SAME company name, so if FMP renames one, the case breaks
    and this test says so without needing fundamentals."""
    nm = dict(zip(live['symbol'], live['name']))
    have = [s for s in group if s in nm]
    if len(have) == len(group):
        norms = {co._norm_issuer_name(nm[s], keep_holding=True) for s in group}
        assert len(norms) == 1, (
            '%s no longer share a normalised name (%s) -- K3 will not fire and this merge '
            'now depends on K1/K2 alone. %s' % (list(group), sorted(norms), why))
    verdict, present = _same_group(panel, *group)
    if verdict == 'absent':
        pytest.skip('UNVERIFIED GATE -- no fundamentals for %s in any local panel; '
                    'names check out. %s' % (list(group), why))
    assert verdict == 'merged', '%s did not collapse (%s)' % (present, why)


def test_every_duplicate_group_in_the_emitted_ranking_collapses(panel):
    """The 19 duplicate groups in the emitted 87-row ranking, and the headline
    consequence: 87 lines are 65 distinct issuers, and reaching 20 distinct issuers
    consumes 27 raw positions because 7 of the raw top 20 are duplicates."""
    ranked = panel['ranked']
    kept, dropped = co.dedup_ranked(ranked, panel['cdx'],
                                    {s: panel['names'].get(s, '') for s in ranked},
                                    scores=panel['d']['postRank']['AggScore'])
    assert len(kept) == 65
    assert len(set(kept)) == 65
    assert len(dropped) == len(ranked) - 65
    #  Each dropped line's survivor must be a DIFFERENT line of the SAME issuer.
    for drop, surv in dropped:
        assert drop != surv
        assert panel['root'][drop] == panel['root'][surv]
    #  The raw top 20 really is 7 duplicates deep -- the number that motivates the change.
    seen, distinct = set(), 0
    for i, s in enumerate(ranked, 1):
        r = panel['root'][s]
        if r not in seen:
            seen.add(r)
            distinct += 1
        if distinct == 20:
            assert i == 27, ('reaching 20 distinct issuers consumed %d raw positions, '
                             'expected 27' % i)
            break


# --------------------------------------------------------------------------- #
#  PICKING -- CANONICITY FIRST                                                 #
# --------------------------------------------------------------------------- #
def test_canonicity_beats_share_count_and_market_cap(panel):
    """The measured claim that reordered the key: share-count-first and marketCap-first
    both pick a structurally non-canonical line in ~54% of groups, because FMP serves the
    ISSUER's share count and market cap to every one of its lines.  Canonicity-first is
    under 1%.  Pinned as an INEQUALITY plus a ceiling, not as exact counts, so ordinary
    data drift does not make it fail for the wrong reason."""
    groups = [m for m in panel['comps'].values() if len(m) > 1]
    val, names = panel['val'], panel['names']

    def failures(keyfn):
        n = 0
        for m in groups:
            surv = sorted(m, key=lambda s: keyfn(s, m))[0]
            if co._non_canonical_tag(surv, names.get(surv, ''), m):
                n += 1
        return n

    by_shares = failures(lambda s, m: (-(val(s, 'weightedAverageShsOut') or -1.0), s))
    by_mcap = failures(lambda s, m: (-(val(s, 'marketCap') or -1.0), s))
    by_spec = failures(lambda s, m: co._investability_key(s, val, None, names, m))
    print('  non-canonical survivors: shares-first %d, mcap-first %d, canonicity-first %d'
          ' of %d groups' % (by_shares, by_mcap, by_spec, len(groups)))
    assert by_shares > 0.4 * len(groups)
    assert by_mcap > 0.4 * len(groups)
    assert by_spec <= 0.01 * len(groups), (
        'canonicity-first picked a non-common line in %d of %d groups (>1%%)'
        % (by_spec, len(groups)))
    assert by_spec < by_shares / 10.0


def test_shares_still_break_ties_INSIDE_a_canonicity_tier(panel):
    """The CEO's share-count rule is not discarded, it is DEMOTED to where it works.
    Two lines in the same canonicity class must order by share count."""
    val = panel['val']
    fake = {'A': {'weightedAverageShsOut': 100.0, 'marketCap': 5.0},
            'B': {'weightedAverageShsOut': 200.0, 'marketCap': 5.0}}
    vf = lambda s, c: fake[s][c]
    assert sorted(['A', 'B'], key=lambda s: co._investability_key(
        s, vf, None, {'A': '', 'B': ''}, ['A', 'B']))[0] == 'B'
    del val


def test_canonicity_overrides_RANK_not_just_ties(panel):
    """THE SURVIVOR-RULE CHANGE.  HNNAZ (notes) outscored HNNA 0.1159 vs 0.1026 -- not a
    tie -- and therefore used to survive.  The score of a notes line is the ISSUER's
    fundamentals attached to an instrument the CEO is not buying, so canonicity now wins.
    """
    ranked = panel['ranked']
    if 'HNNA' not in ranked or 'HNNAZ' not in ranked:
        pytest.skip('the Hennessy pair is not in this ranking')
    assert ranked.index('HNNAZ') < ranked.index('HNNA'), (
        'HNNAZ no longer outranks HNNA on this panel, so it cannot demonstrate a '
        'rank OVERRIDE')
    kept, _dropped = co.dedup_ranked(ranked, panel['cdx'],
                                     {s: panel['names'].get(s, '') for s in ranked})
    assert 'HNNA' in kept and 'HNNAZ' not in kept, (
        'the higher-RANKED notes line survived; canonicity did not override rank')


def test_sector_tagging_no_longer_decides_the_surviving_ticker(panel):
    """`sector_map` used to be criterion 1, which let FMP's sector TAGGING choose the
    ticker the CEO sees.  The key must now be insensitive to it."""
    val, names = panel['val'], panel['names']
    for m in list(panel['comps'].values())[:400]:
        if len(m) < 2:
            continue
        a = sorted(m, key=lambda s: co._investability_key(s, val, None, names, m))
        b = sorted(m, key=lambda s: co._investability_key(
            s, val, {s: 'Technology' for s in m[1:]}, names, m))
        assert a == b


NON_CANONICAL_MARKER_CASES = (
    ('0HQ7.L', '', 'lse-iob'),
    ('0R4M.L', '', 'lse-iob'),
    ('GSL-PB', 'Global Ship Lease, Inc.', 'preferred-suffix'),
    ('TD-PFJ.TO', 'The Toronto-Dominion Bank', 'preferred-suffix'),
    ('HNNAZ', 'Hennessy Advisors, Inc. 4.875% Notes due 2026', 'name-vocabulary'),
    ('009835.KS', 'HANWHA SOLUTIONS Corp. Pfd Registered Shs Non-Voting',
     'name-vocabulary'),
    ('005935.KS', 'Samsung Electronics Co., Ltd.', 'korea-preferred'),
    ('02826K.KS', 'Samsung C&T Corporation', 'korea-preferred'),
    ('33637L.KS', 'Solus Advanced Materials Co. Ltd.', 'korea-preferred'),
    #  MUST be canonical:
    ('AEM', 'Agnico Eagle Mines Limited', ''),
    ('GOOGL', 'Alphabet Inc.', ''),
    ('TCL-A.TO', 'Transcontinental Inc.', ''),
    ('005930.KS', 'Samsung Electronics Co., Ltd.', ''),
    ('BRIM.IC', 'Brim hf.', ''),
    ('EPD', 'Enterprise Products Partners L.P.', ''),
)


@pytest.mark.parametrize('sym,name,expect', NON_CANONICAL_MARKER_CASES,
                         ids=[c[0] for c in NON_CANONICAL_MARKER_CASES])
def test_non_canonical_marker(sym, name, expect):
    tag = co._non_canonical_tag(sym, name, [sym])
    if expect == '':
        assert tag == '', '%s was demoted as %r but is a common' % (sym, tag)
    else:
        assert tag.startswith(expect), '%s -> %r, expected %r' % (sym, tag, expect)


def test_symbol_extension_marker_needs_the_GROUP():
    """Marker (d) is the one RELATIVE marker, and it is relative to the ISSUER GROUP
    rather than the whole pool -- which is what keeps it scope-invariant where
    getData_gen rule C is not."""
    assert co._non_canonical_tag('IMPPP', 'Imperial Petroleum Inc.', ['IMPPP']) == ''
    assert co._non_canonical_tag('IMPPP', 'Imperial Petroleum Inc.',
                                 ['IMPP', 'IMPPP']) == 'symbol-extension'
    #  Different exchange suffix -> a genuine foreign listing, not a tail.
    assert co._non_canonical_tag('IMPPP.L', 'Imperial Petroleum Inc.',
                                 ['IMPP', 'IMPPP.L']) == ''


def test_the_ORDERING_tail_is_permissive_where_the_REMOVAL_tail_must_not_be():
    """THE INVERSION, CASHED IN, AND THE ONE PLACE THE TWO RULES MUST DISAGREE.

    getData_gen's tail is a hand-audited WHITELIST because there a false positive DELETES
    a common. Here a false positive only picks the sibling, so the tail is permissive --
    which is what lets it see the un-whitelisted single-letter tails (CIMN = CIM + "N", a
    NOTES line that otherwise sat in the canonical tier and won its group on marketCap).

    The price is that dual-class commons in the same shape ARE demoted; that is harmless
    only because the dual-class ruling merges them with their sibling, so the sibling
    common survives. Both halves are asserted, because the second is what makes the first
    safe."""
    import getData_gen as _gg
    #  The un-whitelisted tails the removal rule cannot see, which this one can.
    for sym, sib in (('CIMN', 'CIM'), ('WHLRD', 'WHLR'), ('WHLRL', 'WHLR')):
        tail = sym[len(sib):]
        assert not _gg._INSTRUMENT_TAIL_RE.match(tail), (
            '%r is now whitelisted for REMOVAL -- if that is intended it is a universe '
            'change, not a picking change' % tail)
        assert co._non_canonical_tag(sym, '', [sib, sym]) == 'symbol-extension'
    #  Dual-class commons ARE demoted here (and must NOT be in getData_gen).
    for sym, sib in (('GOOGL', 'GOOG'), ('UAA', 'UA'), ('METCB', 'METC')):
        assert co._non_canonical_tag(sym, '', [sib, sym]) == 'symbol-extension'
        assert co._non_canonical_tag(sib, '', [sib, sym]) == '', (
            'the SIBLING must stay canonical -- otherwise demoting the dual-class line '
            'is not harmless, it just moves the arbitrariness')
        assert not _gg._INSTRUMENT_TAIL_RE.match(sym[len(sib):]), (
            'getData_gen would now DELETE %s, a real common' % sym)


def test_a_missing_marker_degrades_the_TICKER_and_never_the_MEMBERSHIP(panel):
    """THE ARCHITECTURAL PROPERTY, ASSERTED.  Disable every canonicity marker and the set
    of surviving ISSUERS must be identical -- only WHICH LINE survives may change.  That
    is the whole reason this pile of heuristics is safe HERE and was not safe as a removal
    rule, and it is the property that makes a marker false positive cost nothing.

    Run over the WHOLE panel, not the emitted 87 rows: on those 87 the deterministic tail
    (shares, mcap, digit-prefix, punctuation, length, alphabetical) happens to reach the
    same tickers the markers do, so the 87-row slice cannot exercise this at all.  Over
    all 8,106 lines the markers change 19 picks."""
    syms, names = panel['syms'], panel['names']
    kept, _d = co.dedup_ranked(syms, panel['cdx'], names)
    real = co._non_canonical_tag
    try:
        co._non_canonical_tag = lambda *_a, **_k: ''
        kept_blind, _d2 = co.dedup_ranked(syms, panel['cdx'], names)
    finally:
        co._non_canonical_tag = real
    assert len(kept_blind) == len(kept), (
        'disabling the markers changed the NUMBER OF ISSUERS -- membership must depend '
        'only on grouping, never on the ordering markers')
    root = panel['root']
    assert {root[s] for s in kept_blind} == {root[s] for s in kept}, (
        'disabling the markers changed WHICH ISSUERS survive, not just which line')
    changed = [(a, b) for a, b in zip(kept, kept_blind) if a != b]
    assert changed, ('the markers changed no pick anywhere on 8,106 lines, so this test '
                     'is not exercising them')
    print('  markers change the surviving ticker in %d group(s), e.g. %s'
          % (len(changed), changed[:5]))


def test_the_markers_are_load_bearing_and_by_how_much(panel):
    """HOW MUCH of the improvement is the MARKERS and how much is the reordering?  Stated
    as a measurement because "0.47% thanks to the detectors" would be an overclaim: the
    deployed key's 3.1% failure rate falls to 1.1% from DROPPING the sector-map criterion
    alone, and the markers take it from 25 groups to 6."""
    groups = [m for m in panel['comps'].values() if len(m) > 1]
    val, names = panel['val'], panel['names']

    def failures(keyfn):
        n = 0
        for m in groups:
            surv = sorted(m, key=lambda s: keyfn(s, m))[0]
            if co._non_canonical_tag(surv, names.get(surv, ''), m):
                n += 1
        return n

    spec = failures(lambda s, m: co._investability_key(s, val, None, names, m))
    tail_only = failures(lambda s, m: (-(val(s, 'weightedAverageShsOut') or -1.0),
                                       -(val(s, 'marketCap') or -1.0),
                                       1 if s[:1].isdigit() else 0,
                                       sum(c in '-.' for c in s), len(s), s))
    print('  non-canonical survivors: with markers %d, tail only %d, of %d groups'
          % (spec, tail_only, len(groups)))
    assert spec < tail_only, (
        'the canonicity markers change nothing measurable -- either they are broken or '
        'the deterministic tail already dominates, and the docstring claim is wrong')


# --------------------------------------------------------------------------- #
#  KOREA -- THE GATE THAT UNBLOCKS ASIA                                        #
# --------------------------------------------------------------------------- #
def test_korea_families_have_the_recorded_shape(live):
    """91 multi-line families over 196 symbols, every family containing its `...0`
    common, and 90 of 91 sharing one normalised name.  Recorded as data in
    getData_gen.KOREA_FAMILY_FACTS; re-derived here so the fact cannot decay."""
    f = gdg.KOREA_FAMILY_FACTS
    kr = [s for s in live['symbol'].astype(str) if s.rsplit('.', 1)[-1] in ('KS', 'KQ')]
    assert len(kr) == f['korean_stock_lines']
    fams = gdg._korean_families(kr)
    assert len(fams) == f['multi_line_families']
    assert sum(len(v) for v in fams.values()) == f['symbols_in_those_families']
    nm = dict(zip(live['symbol'], live['name']))
    assert all(any(s.rsplit('.', 1)[0][5] == '0' for s in v) for v in fams.values()), (
        'a Korean family has no `...0` common -- the marker would demote every member')
    #  Both name bases, because K3 uses keep_holding=True and that costs one family
    #  (AMOREPACIFIC) its name edge -- see KOREA_FAMILY_FACTS.
    one_name_k3 = sum(1 for v in fams.values()
                      if len({co._norm_issuer_name(nm.get(s, ''), keep_holding=True)
                              for s in v}) == 1)
    one_name_def = sum(1 for v in fams.values()
                       if len({co._norm_issuer_name(nm.get(s, '')) for s in v}) == 1)
    assert one_name_k3 == f['families_with_one_shared_name_K3']
    assert one_name_def == f['families_with_one_shared_name_default_norm']


def test_korea_marker_demotes_every_preferred_and_no_common(live):
    """THE PICKING HALF, over all 91 live families.  This is the half that is genuinely
    verified without a fetch, because the marker is a pure function of symbol and name."""
    nm = dict(zip(live['symbol'].astype(str), live['name'].astype(str)))
    fams = gdg._korean_families(list(nm))
    bad = []
    for key, members in fams.items():
        canon = [s for s in members if not co._non_canonical_tag(s, nm[s], members)]
        zeros = [s for s in members if s.rsplit('.', 1)[0][5] == '0']
        if canon != zeros or len(canon) != 1:
            bad.append((key, canon, zeros))
    assert not bad, ('%d of %d Korean families do not resolve to exactly one canonical '
                     '`...0` common: %s' % (len(bad), len(fams), bad[:5]))


def test_the_korean_line_code_is_a_CHARACTER_not_a_digit(live):
    """A LETTER in the 6th position is Korea's "new-type" preferred, and 17 live lines
    have one (15 K + 2 L: Samsung C&T 02826K.KS, Hanjin Kal 18064K.KS, SK Inc 03473K.KS,
    Solus 33637K/33637L.KS...).  Every one carries its common's name VERBATIM, so the
    name vocabulary does not catch them either.  Written as `\\d` the marker would call
    all 17 canonical -- which is why this is asserted separately from the family test."""
    nm = dict(zip(live['symbol'].astype(str), live['name'].astype(str)))
    fams = gdg._korean_families(list(nm))
    letters = [s for v in fams.values() for s in v
               if not s.rsplit('.', 1)[0][5].isdigit()]
    assert len(letters) == (gdg.KOREA_FAMILY_FACTS['sixth_char_counts']['K']
                            + gdg.KOREA_FAMILY_FACTS['sixth_char_counts']['L'])
    for s in letters:
        assert co._non_canonical_tag(s, nm[s], fams[(s.rsplit('.', 1)[0][:5],
                                                     s.rsplit('.', 1)[1])]) != '', (
            '%s (6th char %r) was not demoted' % (s, s.rsplit('.', 1)[0][5]))


def test_the_korea_gate_RAISES_without_the_canonicity_marker(live):
    """The dependency is enforced, not documented: remove the marker and a Korean
    universe must refuse to resolve."""
    real = co._non_canonical_tag
    try:
        del co._non_canonical_tag
        with pytest.raises(Exception, match='(?i)korea'):
            gdg.assert_korea_dedup_ready(live, 'stock_ASIA1', verbose=False)
    finally:
        co._non_canonical_tag = real
    assert gdg.assert_korea_dedup_ready(live, 'stock_ASIA1', verbose=False) is True


def test_the_korea_gate_RAISES_when_the_marker_is_WRONG(live):
    """A marker that is present but broken must fail the gate too -- presence is not
    correctness, and a `\\d`-based Korean rule is exactly the "present but wrong" case."""
    real = co._non_canonical_tag
    try:
        co._non_canonical_tag = lambda *_a, **_k: ''
        with pytest.raises(Exception, match='(?i)canonicity marker FAILED'):
            gdg.assert_korea_dedup_ready(live, 'stock_ASIA1', verbose=False)
    finally:
        co._non_canonical_tag = real


def test_KOREA_MUST_MERGE_the_196_symbols_collapse_to_91_issuers(panel, live):
    """THE GATE THAT UNBLOCKS ASIA, AND IT IS OPEN.

    On real Korean fundamentals, the 196 symbols in the 91 families must collapse to 91
    issuers with the `...0` common surviving each.  Until this passes, the claim that FMP
    serves Korean preferreds their ISSUER'S statements -- i.e. that K1/K3 group the family
    at all -- is UNVERIFIED, and the picking rule has nothing to pick between.

    SKIPS on every local panel by construction: none contains Korean fundamentals.  Do
    not delete it and do not relax it into a name-only check; it is the one test that
    turns Korea from "should work" into "does"."""
    kr_in_panel = [s for s in panel['syms'] if s.rsplit('.', 1)[-1] in ('KS', 'KQ')]
    if not kr_in_panel:
        pytest.skip('UNVERIFIED GATE -- no Korean fundamentals in PANEL-JAN (0 of 8,106 '
                    'lines are .KS/.KQ). Run against the first stock_ASIA1 fetch.')
    nm = dict(zip(live['symbol'].astype(str), live['name'].astype(str)))
    fams = gdg._korean_families(kr_in_panel)
    unmerged, wrong_survivor = [], []
    for key, members in fams.items():
        roots = {panel['root'][s] for s in members if s in panel['root']}
        if len(roots) != 1:
            unmerged.append((key, sorted(members)))
            continue
        group = panel['comps'][roots.pop()]
        surv = sorted(group, key=lambda s: co._investability_key(
            s, panel['val'], None, {s: nm.get(s, '') for s in group}, group))[0]
        if surv.rsplit('.', 1)[0][5] != '0':
            wrong_survivor.append((key, surv))
    assert not unmerged, ('%d Korean families did NOT collapse: %s -- FMP is not serving '
                          'the issuer\'s statements to the preferred lines, so grouping '
                          'cannot see them' % (len(unmerged), unmerged[:5]))
    assert not wrong_survivor, ('a non-`...0` line survived in %d family(ies): %s'
                                % (len(wrong_survivor), wrong_survivor[:5]))


# --------------------------------------------------------------------------- #
#  THE OPEN ITEMS, DECLARED                                                    #
# --------------------------------------------------------------------------- #
#  Tests that CANNOT run on local data, and why.  Declared as data so a reader (and the
#  test below) can tell "never ran" from "passed" -- the distinction the whole file rests
#  on.  Remove an entry only when the data that unblocks it exists.
UNVERIFIED_GATES = {
    'test_MUST_NOT_MERGE_heineken_fundamentals':
        'needs Amsterdam fundamentals (HEIA.AS / HEIO.AS). THE ONE LIVE MUST-NOT-MERGE.',
    'test_MUST_NOT_MERGE_credit_agricole_CCIs':
        'names + share-class filter verified; grouping needs Paris fundamentals.',
    'test_MUST_MERGE_continental':
        'names verified for all five pairs; grouping needs AMS/PAR/OSL fundamentals.',
    'test_KOREA_MUST_MERGE_the_196_symbols_collapse_to_91_issuers':
        'needs a Korea fetch. Blocks trusting any Korean name in the ranking.',
}


def test_the_unverified_gates_are_declared():
    """A gate that skips is an OPEN item.  This asserts every declared gate still exists
    as a test function, so the declaration cannot rot -- and so deleting an awkward gate
    to get a green suite breaks the suite instead."""
    here = globals()
    for fn in UNVERIFIED_GATES:
        assert fn in here and callable(here[fn]), (
            '%s is declared as an unverified gate but no longer exists as a test' % fn)
