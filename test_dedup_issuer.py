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
import zlib

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


# =========================================================================== #
#  ISIN WIRED INTO THE SURVIVOR PICK -- register K-1 (CEO 2026-08-05)           #
#                                                                               #
#  The load-bearing test is the FIRST one: with no ISIN data -- the state of      #
#  every pickle in the repo today -- the pick must be BIT-IDENTICAL to the        #
#  pre-ISIN rule.  The others pin what the term does and, just as importantly,     #
#  what it does NOT do (it does not resolve the three known-wrong groups).         #
# =========================================================================== #

def _pre_isin_key(sym, val_fn, names=None, group=()):
    """The survivor key EXACTLY as it stood before ISIN was wired (terms 1-4 + symbol).
    Duplicated here on purpose: a test that re-derives the old behaviour from the new
    code could not detect the new code changing it."""
    nm = (names or {}).get(sym, '') if names else ''
    noncanon = 1 if co._non_canonical_tag(sym, nm, group) else 0
    sh = val_fn(sym, 'weightedAverageShsOut')
    sh = sh if sh is not None else -1.0
    mc = val_fn(sym, 'marketCap')
    mc = mc if mc is not None else -1.0
    digitpfx = 1 if sym[:1].isdigit() else 0
    punct = sum(ch in '-.' for ch in sym)
    return (noncanon, -sh, -mc, digitpfx, punct, len(sym), sym)


def test_isin_absent_is_bit_identical(panel):
    """*** THE PROPERTY THAT MATTERS MOST.  No isindic_fmp_*.pickle exists yet, so the
    CEO may run the pipeline before the next profile build; on that path the survivor of
    EVERY group must be the one the pre-ISIN rule picked. ***
    Checked with an EMPTY map and with the map the process would actually load, and
    checked as an ORDERING (the full sorted member list), not just the winner."""
    val, names = panel['val'], panel['names']
    live_map = co._isin_map_cached()
    n_groups = 0
    for m in panel['comps'].values():
        if len(m) < 2:
            continue
        n_groups += 1
        old = sorted(m, key=lambda s: _pre_isin_key(s, val, names, m))
        for imap in ({}, live_map):
            new = sorted(m, key=lambda s: co._investability_key(
                s, val, None, names, m, imap))
            assert new == old, (
                'ISIN changed the ordering of %s with isin_map=%r; the no-ISIN path is '
                'NOT bit-identical' % (m, 'live' if imap is live_map else 'empty'))
    assert n_groups > 100, 'panel produced too few multi-line groups to be evidence'


def test_isin_absent_means_absent_not_silently_populated():
    """The bit-identity above is only reassuring if the map really is empty today.  If a
    profile build has landed and this fails, that is INFORMATION, not a defect -- but it
    must not pass unnoticed, because the no-op claim stops holding at that moment."""
    assert co._load_isin_map() == {}, (
        'an isindic_fmp_*.pickle now exists -- ISIN is LIVE in the survivor pick. '
        'Re-measure the alphabetical-tail groups and update register K-1.')


def test_isin_plurality_decides_the_multi_venue_common():
    """WHAT THE TERM ACTUALLY BUYS.  Three lines, one issuer: the common cross-listed on
    two venues (same ISIN) plus a single odd line (different ISIN).  Nothing above the
    alphabetical tail separates them, and the odd line sorts FIRST alphabetically, so
    without ISIN it would win.  Plurality must hand it to the common."""
    fake = {s: {'weightedAverageShsOut': 100.0, 'marketCap': 5.0}
            for s in ('AAA', 'ZZZ', 'ZZY')}
    vf = lambda s, c: fake[s][c]
    names = {s: 'Same Issuer NV' for s in fake}
    group = ['AAA', 'ZZZ', 'ZZY']
    imap = {'AAA': 'NL0000000001', 'ZZZ': 'NL0000000002', 'ZZY': 'NL0000000002'}
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}))[0] == 'AAA', 'precondition: alphabet picks AAA'
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, imap))[0] == 'ZZY', (
        'ISIN plurality did not lift the two-venue common over the singleton line')


def test_isin_ABSTAINS_on_a_two_member_group_with_two_isins():
    """*** THE HONEST NEGATIVE.  This is the shape of ALL THREE known-wrong groups
    (CBE.PA/RBT.PA certificat, PREVA.AS/VALUE.AS preference, SMSD.L/SMSN.L preferred
    GDR): two members, two distinct ISINs, plurality 1-1.  An ISIN carries no
    security-type field, so it CANNOT say which is the common, and this test pins that
    the code does not pretend otherwise -- it abstains and the alphabetical tail still
    picks the NON-COMMON.  Register K-1 stays OPEN. ***"""
    fake = {s: {'weightedAverageShsOut': 100.0, 'marketCap': 5.0}
            for s in ('CBE.PA', 'RBT.PA')}
    vf = lambda s, c: fake[s][c]
    names = {s: 'Robertet SA' for s in fake}
    group = ['CBE.PA', 'RBT.PA']
    imap = {'CBE.PA': 'FR0000045551', 'RBT.PA': 'FR0000039091'}
    for m in group:
        assert co._isin_plurality_term(m, group, imap) == 0, (
            'a 1-1 ISIN split must abstain (return 0), not invent a direction')
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, imap))[0] == 'CBE.PA', (
        'the certificat no longer wins -- if this fails, some rule started reading '
        'canonicity out of an ISIN, which is exactly what must not happen silently')


def test_isin_never_outranks_a_canonicity_marker():
    """NEVER WORSE THAN TODAY.  Even a plurality of ISINs on the non-canonical side must
    not lift a marker-detected line over a clean one -- ISIN sits BELOW canonicity."""
    fake = {s: {'weightedAverageShsOut': 100.0, 'marketCap': 5.0}
            for s in ('AEM', '0R2J.L', '0R2K.L')}
    vf = lambda s, c: fake[s][c]
    names = {s: 'Agnico Eagle Mines Limited' for s in fake}
    group = ['AEM', '0R2J.L', '0R2K.L']
    imap = {'AEM': 'CA0084741085', '0R2J.L': 'GB00B000001', '0R2K.L': 'GB00B000001'}
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, imap))[0] == 'AEM', (
        'an IOB line won on ISIN plurality; the ISIN term is placed too high')


#  ---- THE TWO PROPERTIES THAT MAKE "NO ISIN DATA" NEUTRAL  (reviewer, 2026-08-05) ---- #
#  The first cut returned the abstain value 0 for an unmapped member while the             #
#  discriminating branch emitted only values <= -1, so 0 was the WORST value in the         #
#  term's own range and a member the profile map merely LACKED sorted BELOW a member        #
#  holding a SINGLETON ISIN.  ISIN DATA AVAILABILITY was deciding survivors.  These two     #
#  tests pin the properties; each is also a case-exhaustive consequence of the code shape   #
#  (see the note in `_isin_plurality_term`), so they are regression guards on a proof, not  #
#  the evidence for it.                                                                     #
def _tied_group(members):
    """Members that are IDENTICAL on every term above ISIN (canonicity, shares, marketCap,
    digit-prefix, punctuation, length), so the ISIN term and then the alphabet decide."""
    fake = {s: {'weightedAverageShsOut': 100.0, 'marketCap': 5.0} for s in members}
    return (lambda s, c: fake[s][c]), {s: 'Same Issuer NV' for s in members}


def test_isin_absence_ties_with_a_singleton_and_is_never_the_worst_value():
    """PROPERTY (a), at the value level -- the reviewer's exact call.  'No ISIN for this
    line' and 'an ISIN only this line holds' both mean ISIN TELLS US NOTHING HERE, so they
    must be the SAME value; and no member may score above (i.e. sort after) every active
    value just for being unmapped."""
    group = ['AAA', 'BBB', 'CCC', 'DDD']
    imap = {'AAA': 'US1111111111', 'BBB': 'US1111111111',
            'CCC': 'GB2222222222', 'DDD': None}
    t = {m: co._isin_plurality_term(m, group, imap) for m in group}
    assert t == {'AAA': -2, 'BBB': -2, 'CCC': -1, 'DDD': -1}, (
        'absence must tie with the SINGLETON at -1, not fall out at 0 (the abstain value, '
        'which is the WORST value in this term\'s range); got %r' % t)
    assert t['DDD'] <= max(v for v in t.values()), (
        'the unmapped member is the unique worst value -- availability is deciding the pick')
    #  ...and at the ordering level: DDD must not be pushed behind the singleton CCC.
    vf, names = _tied_group(group)
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, imap)) == ['AAA', 'BBB', 'CCC', 'DDD']


def test_isin_mixed_availability_cannot_reorder_otherwise_tied_members():
    """PROPERTY (a), at the ordering level and the way it actually bites.  Two members
    otherwise tied on every term, one UNMAPPED and one holding a SINGLETON, must order
    exactly as if both were unmapped -- populating the map for one of them and not the
    other may not move the survivor."""
    group = ['AAA', 'BBB', 'YYY', 'ZZZ']          # AAA/BBB are the two-venue common
    vf, names = _tied_group(group)
    common = {'AAA': 'US1111111111', 'BBB': 'US1111111111'}
    variants = {
        'both unmapped':          {**common, 'YYY': None, 'ZZZ': None},
        'YYY singleton only':     {**common, 'YYY': 'GB2222222222', 'ZZZ': None},
        'ZZZ singleton only':     {**common, 'YYY': None, 'ZZZ': 'GB3333333333'},
        'both singletons':        {**common, 'YYY': 'GB2222222222', 'ZZZ': 'GB3333333333'},
        'unusable not missing':   {**common, 'YYY': 'nan', 'ZZZ': float('nan')},
    }
    orders = {k: sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, m)) for k, m in variants.items()}
    assert len(set(map(tuple, orders.values()))) == 1, (
        'ISIN availability alone reordered an otherwise-tied group: %r' % orders)
    assert list(orders['both unmapped'])[:2] == ['AAA', 'BBB'], (
        'precondition: the plurality still lifts the two-venue common -- if this fails the '
        'invariance above is vacuous because the term stopped discriminating at all')


def test_isin_group_wide_abstention_survives_mixed_availability():
    """PROPERTY (b).  When the term discriminates NOTHING it must change NO ordering -- and
    that has to hold when only SOME members are mapped, which is exactly where the first cut
    broke it (the abstain test used to be taken after this line's own ISIN was read, so an
    unmapped member got 0 while a mapped one got a real value).  The abstain decision is now
    a function of (isin_map, group) alone, so it cannot differ between members."""
    group = ['AAA', 'BBB', 'CCC']
    vf, names = _tied_group(group)
    #  Every abstain shape, each with mixed availability inside it.
    for label, imap in (
            ('one distinct usable ISIN, others unmapped',
             {'AAA': 'US1111111111', 'BBB': None, 'CCC': None}),
            ('one distinct usable ISIN shared, one unmapped',
             {'AAA': 'US1111111111', 'BBB': 'US1111111111', 'CCC': None}),
            ('no ISIN held by more than one member, one unmapped',
             {'AAA': 'US1111111111', 'BBB': 'GB2222222222', 'CCC': None}),
            ('nothing usable at all',
             {'AAA': 'nan', 'BBB': None, 'CCC': ''})):
        vals = [co._isin_plurality_term(m, group, imap) for m in group]
        assert vals == [0, 0, 0], (
            '%s: abstention must be the literal 0 for EVERY member (auditable as a value, '
            'not as a cancellation); got %r' % (label, vals))
        assert sorted(group, key=lambda s: co._investability_key(
            s, vf, None, names, group, imap)) == sorted(
                group, key=lambda s: co._investability_key(s, vf, None, names, group, {})), (
            '%s: the term abstained but the ordering still moved' % label)


def test_unusable_isin_values_cannot_group(panel):
    """None / NaN / '' / short junk must not count as a shared ISIN -- otherwise every
    unmapped line in a group would look like one big plurality."""
    group = ['AAA', 'BBB', 'CCC']
    for bad in (None, float('nan'), '', '  ', 'N/A', 12345):
        imap = {'AAA': bad, 'BBB': bad, 'CCC': 'US0000000001'}
        assert co._isin_plurality_term('AAA', group, imap) == 0
        assert co._isin_plurality_term('CCC', group, imap) == 0, (
            'a single usable ISIN among junk must abstain, not win on count 1'
        )
    del panel


# =========================================================================== #
#  volAvg AS THE SURVIVOR DISCRIMINATOR (register K-1, wired 2026-08-06)        #
#                                                                               #
#  Same three properties the ISIN block pins, in the same order, because the      #
#  wiring is deliberately the same shape: (1) NO DATA -> BIT-IDENTICAL, (2) what    #
#  it actually buys, (3) every condition under which it must ABSTAIN.  The reason   #
#  volAvg is wanted at all is that ISIN could NOT reach the three known-wrong      #
#  picks -- it abstains on a 2-member group with 2 distinct ISINs -- while volume   #
#  is DIRECTIONAL BY CONSTRUCTION: the common is the liquid line.                  #
# =========================================================================== #

def _pre_volavg_key(sym, val_fn, names=None, group=(), isin_map=None):
    """The survivor key EXACTLY as it stood after ISIN and before volAvg (terms 1-5 +
    symbol).  Duplicated rather than derived, for the same reason `_pre_isin_key` is: a
    test that re-derives the old behaviour from the new code cannot detect the new code
    changing it."""
    nm = (names or {}).get(sym, '') if names else ''
    noncanon = 1 if co._non_canonical_tag(sym, nm, group) else 0
    sh = val_fn(sym, 'weightedAverageShsOut')
    sh = sh if sh is not None else -1.0
    mc = val_fn(sym, 'marketCap')
    mc = mc if mc is not None else -1.0
    digitpfx = 1 if sym[:1].isdigit() else 0
    punct = sum(ch in '-.' for ch in sym)
    imap = {} if isin_map is None else isin_map
    return (noncanon, -sh, -mc, digitpfx, punct, len(sym),
            co._isin_plurality_term(sym, group, imap), sym)


def test_volavg_absent_is_bit_identical(panel):
    """*** THE PROPERTY THAT MATTERS MOST, AND THE ONE THE BRIEF ASKED TO BE ASSERTED.
    No volavgdic_fmp_*.pickle exists yet, so every saved pickle and every run before the
    next profile build must produce the survivor -- and the whole ORDERING -- that the
    pre-volAvg rule produced. ***
    Checked with an EMPTY map and with the map the process would actually load, and as an
    ORDERING rather than just a winner, because a term that reorders the LOSERS is still a
    behaviour change (dedup_ranked's audit trail reads the loser list)."""
    val, names = panel['val'], panel['names']
    live_map = co._volavg_map_cached()
    n_groups = 0
    for m in panel['comps'].values():
        if len(m) < 2:
            continue
        n_groups += 1
        old = sorted(m, key=lambda s: _pre_volavg_key(s, val, names, m, {}))
        for vmap in ({}, live_map):
            new = sorted(m, key=lambda s: co._investability_key(
                s, val, None, names, m, {}, vmap))
            assert new == old, (
                'volAvg changed the ordering of %s with volavg_map=%r; the no-volAvg path '
                'is NOT bit-identical'
                % (m, 'live' if vmap is live_map else 'empty'))
    assert n_groups > 100, 'panel produced too few multi-line groups to be evidence'


def test_volavg_absent_means_absent_not_silently_populated():
    """The bit-identity above is only reassuring if the map really is empty today.  If a
    profile build has landed and this fails, that is INFORMATION, not a defect -- but it
    must not pass unnoticed, because the no-op claim stops holding at that moment."""
    assert co._load_volavg_map() == {}, (
        'a volavgdic_fmp_*.pickle now exists -- volAvg is LIVE in the survivor pick. '
        'Re-measure the alphabetical-tail groups and update register K-1.')


def _k1_shaped_group():
    """The K-1 shape: two SAME-EXCHANGE lines, names identical VERBATIM, identical
    issuer-level share count and market cap (so the derived price is identical too), no
    -P suffix, no shared prefix, neither a .KS line.  Every canonicity marker is ruled out
    BY CONSTRUCTION, so the key falls to the alphabetical tail and the NON-COMMON wins."""
    group = ['CBE.PA', 'RBT.PA']
    fake = {s: {'weightedAverageShsOut': 100.0, 'marketCap': 5.0} for s in group}
    vf = lambda s, c: fake[s][c]
    names = {s: 'Robertet SA' for s in group}
    return group, vf, names


def test_the_alphabetical_tail_really_does_pick_the_wrong_line_today():
    """The premise of the whole change, asserted rather than left in prose: with no volAvg
    data the Robertet-shaped group hands the win to the certificat."""
    group, vf, names = _k1_shaped_group()
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}, {}))[0] == 'CBE.PA'


def test_volavg_hands_the_group_to_the_LIQUID_line():
    """WHAT THE TERM ACTUALLY BUYS.  Same group, same everything, plus volume readings a
    decade apart: the common must now win despite sorting LAST alphabetically."""
    group, vf, names = _k1_shaped_group()
    vmap = {'CBE.PA': (1.2e3, '2026-08-06'), 'RBT.PA': (4.5e4, '2026-08-06')}
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}, vmap))[0] == 'RBT.PA', (
        'volAvg did not demote the thinly-traded certificat')


def test_volavg_cannot_outrank_a_canonicity_marker():
    """POSITION IN THE KEY IS THE SAFETY ARGUMENT.  Give the NON-COMMON a million times the
    volume of the common and put a canonicity marker on it: canonicity must still win, or
    the measured 0.47% canonicity-first failure rate has regressed."""
    group = ['SMSD.L', 'SMSN.L']
    fake = {s: {'weightedAverageShsOut': 100.0, 'marketCap': 5.0} for s in group}
    vf = lambda s, c: fake[s][c]
    #  A name-vocabulary marker on the preferred line is the realistic case.
    names = {'SMSD.L': 'Samsung Electronics Co., Ltd. Pfd Registered Shs Non-Voting',
             'SMSN.L': 'Samsung Electronics Co., Ltd.'}
    assert co._non_canonical_tag('SMSD.L', names['SMSD.L'], group), \
        'the fixture no longer carries a canonicity marker, so it proves nothing'
    vmap = {'SMSD.L': (1e9, '2026-08-06'), 'SMSN.L': (1e3, '2026-08-06')}
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}, vmap))[0] == 'SMSN.L'


def _isin_vs_volavg_fixture():
    """The group where the two terms DISAGREE -- the only fixture that can observe their
    relative order.  ISIN plurality favours the ZZ* pair (two lines, one ISIN); volume
    favours AAA by three orders of magnitude.  This is also the shape the CEO's reasoning
    names: several lines sharing one ISIN against a single line with its own."""
    group = ['AAA', 'ZZY', 'ZZZ']
    fake = {s: {'weightedAverageShsOut': 100.0, 'marketCap': 5.0} for s in group}
    vf = lambda s, c: fake[s][c]
    names = {s: 'Same Issuer NV' for s in group}
    imap = {'ZZY': 'NL0000000001', 'ZZZ': 'NL0000000001', 'AAA': 'US0000000002'}
    vmap = {'AAA': (1e7, '2026-08-06'), 'ZZY': (1e4, '2026-08-06'),
            'ZZZ': (1e3, '2026-08-06')}
    return group, vf, names, imap, vmap


def test_volavg_OUTRANKS_ISIN():
    """*** CEO RULING 2026-08-06: volume above ISIN. ***  This test asserted the OPPOSITE
    until that ruling -- the first cut put volAvg below ISIN as the conservative choice and
    flagged it as not obviously correct.  The reasoning for the swap: volume is DIRECTIONAL
    BY CONSTRUCTION (the common IS the liquid line), whereas ISIN plurality is an IDENTITY
    INFERENCE that can point the wrong way -- three depositary lines sharing one ISIN
    against a common carrying its own hand plurality to a depositary receipt, which is
    exactly this fixture."""
    group, vf, names, imap, vmap = _isin_vs_volavg_fixture()
    #  Both terms must ACTUALLY SPEAK here, or the test proves nothing about their order.
    assert len({co._volavg_liquidity_term(s, group, vmap) for s in group}) > 1, \
        'the fixture must have volAvg actively speaking, or the test is vacuous'
    assert len({co._isin_plurality_term(s, group, imap) for s in group}) > 1, \
        'the fixture must have ISIN actively speaking, or the test is vacuous'
    winner = sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, imap, vmap))[0]
    assert winner == 'AAA', (
        'ISIN plurality outranked volume (winner %r) -- term order 5/6 has been swapped '
        'back, against the CEO ruling' % winner)


def test_ISIN_still_decides_every_group_volAvg_ABSTAINS_on():
    """The other half of the swap, and the one that stops it being a regression: promoting
    volume must not have DISPLACED ISIN, only preceded it.  Same fixture, volumes moved
    inside one order of magnitude so volAvg abstains -- ISIN must then decide exactly as it
    did before 2026-08-06."""
    group, vf, names, imap, _vmap = _isin_vs_volavg_fixture()
    vmap = {'AAA': (1.0e4, '2026-08-06'), 'ZZY': (1.1e4, '2026-08-06'),
            'ZZZ': (1.2e4, '2026-08-06')}
    assert {co._volavg_liquidity_term(s, group, vmap) for s in group} == {0}
    winner = sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, imap, vmap))[0]
    assert winner in ('ZZY', 'ZZZ'), (
        'volAvg abstained but ISIN no longer decides (winner %r) -- the reordering broke '
        'the 2026-08-05 ISIN behaviour' % winner)


def test_volavg_does_NOT_speak_across_a_POWER_OF_TEN_boundary():
    """*** THE REGRESSION TEST FOR THE DEFECT THE REVIEWER CAUGHT.  The first cut bucketed
    with `-int(floor(log10(v)))`, so 9,900 -> 3 and 10,100 -> 4 and a 2% DIFFERENCE SPOKE
    WITH FULL FORCE -- flatly contradicting the stated rationale that near-ties tie. ***

    Worse than useless: with the non-common just above the boundary and its common just
    below, the term ACTIVELY SELECTED THE NON-COMMON in the exact groups it was added for,
    and since volAvg is re-read every fetch a line drifting across a power of ten flipped
    the survivor between runs.  The fix compares VALUES (max/min >= 10), which has no edge
    anywhere.  Both orientations are checked, because the bucketed version was wrong in
    both and only one of them loses a K-1 group."""
    group, vf, names = _k1_shaped_group()
    for lo_sym, hi_sym in (('CBE.PA', 'RBT.PA'), ('RBT.PA', 'CBE.PA')):
        vmap = {lo_sym: (9900.0, '2026-08-06'), hi_sym: (10100.0, '2026-08-06')}
        terms = [co._volavg_liquidity_term(s, group, vmap) for s in group]
        assert terms == [0, 0], (
            'volAvg spoke across the 10,000 boundary (%r): a 2%% volume difference decided '
            'a survivor. terms=%r' % (vmap, terms))
        #  *** THE ORDERING ASSERTION CHANGED 2026-08-08, and the change is deliberate. ***
        #  It used to require the whole ordering to equal the PRE-volAvg one.  That is no
        #  longer the contract: a 2% difference now DOES decide this group, via the weak raw
        #  term the CEO added below ISIN, because the alternative fallback was the raw
        #  alphabet.  What must NOT happen -- and what this test exists for -- is the
        #  DECADE term claiming it: a 2% margin must never be reported with the confident
        #  term's authority.  So the surviving invariant is about ATTRIBUTION.
        keys = {s: co._investability_key(s, vf, None, names, group, {}, vmap)
                for s in group}
        order = sorted(group, key=keys.__getitem__)
        assert co._deciding_term(keys[order[0]], keys[order[1]]) == 'volavg_raw', (
            'a 2%% volume difference was attributed to a term other than volavg_raw '
            '(%r) -- the power-of-ten edge is back in the DECADE term'
            % co._deciding_term(keys[order[0]], keys[order[1]]))


def test_volavg_decides_at_exactly_TEN_TIMES_and_not_below_it():
    """The threshold itself, pinned from both sides so a later edit to
    `_VOLAVG_DECIDING_RATIO` cannot pass silently.  9.99x abstains, 10x decides -- and note
    that the boundary that remains flips between "volume decides" and "alphabet decides",
    never between "picks A" and "picks B", which is why it is tolerable where the
    power-of-ten edge was not."""
    group, vf, names = _k1_shaped_group()
    for ratio, should_speak in ((9.99, False), (10.0, True), (10.01, True)):
        vmap = {'CBE.PA': (1000.0, '2026-08-06'),
                'RBT.PA': (1000.0 * ratio, '2026-08-06')}
        spoke = len({co._volavg_liquidity_term(s, group, vmap) for s in group}) > 1
        assert spoke is should_speak, 'ratio %s: spoke=%s' % (ratio, spoke)
        if should_speak:
            assert sorted(group, key=lambda s: co._investability_key(
                s, vf, None, names, group, {}, vmap))[0] == 'RBT.PA'


def test_volavg_term_is_SCALE_INVARIANT_which_is_what_kills_the_absolute_EDGE():
    """The structural reason the fix has no power-of-ten edge, asserted rather than argued:
    multiplying every reading in a group by any positive factor leaves every term
    unchanged, because the rule reads only RATIOS.  The bucketed version failed this for
    almost every factor -- which is another way of saying it had an absolute edge."""
    group, _vf, _names = _k1_shaped_group()
    base = {'CBE.PA': (1.2e3, '2026-08-06'), 'RBT.PA': (4.5e4, '2026-08-06')}
    want = [co._volavg_liquidity_term(s, group, base) for s in group]
    assert len(set(want)) > 1, 'the base fixture must be speaking'
    for k in (1e-6, 0.37, 3.0, 7.3, 1e5, 9.9, 10.0):
        scaled = {s: (v * k, d) for s, (v, d) in base.items()}
        got = [co._volavg_liquidity_term(s, group, scaled) for s in group]
        assert got == want, 'scaling by %s changed the term: %r != %r' % (k, got, want)


#  Each case is (label, vmap, EXPECTED SURVIVOR, EXPECTED decided_by).
#
#  *** THE LAST TWO FIELDS WERE ADDED 2026-08-08 AND THEY ARE THE POINT (reviewer F2). ***
#  The first rewrite of the test below relaxed its assertion to
#  `decided in ('volavg_raw', 'alphabetical')` because 2 of these 9 cases legitimately
#  changed behaviour when the raw term landed.  That relaxation ACCEPTED THE VERY LEAK THE
#  GUARD EXISTS TO PREVENT: a mutation that SKIPS a NaN / non-numeric reading instead of
#  abstaining the group -- so the survivor is decided by DATA AVAILABILITY -- passed the
#  whole suite green.  The two triggers that expose it (`a non-numeric reading`, `a NaN
#  reading`) live ONLY here, in no other test.  So the expectation is now pinned PER CASE
#  and BY NAME: seven of the nine must still be decided by the raw ALPHABET with the
#  alphabetically-first line surviving, and only the two genuine ratio-abstentions may
#  reach `volavg_raw`.  Hardcoded, never derived from the guard -- a test that recomputes
#  its expectation from the code under test cannot detect that code changing.
VOLAVG_ABSTAIN_CASES = (
    ('empty map', {}, 'CBE.PA', 'alphabetical'),
    ('one member unmapped -- a survivor must never be decided by DATA AVAILABILITY',
     {'CBE.PA': (1.2e3, '2026-08-06')}, 'CBE.PA', 'alphabetical'),
    ('a None reading', {'CBE.PA': (None, '2026-08-06'), 'RBT.PA': (4.5e4, '2026-08-06')},
     'CBE.PA', 'alphabetical'),
    ('a zero reading -- log10(0) is undefined and "no volume" is not "least volume"',
     {'CBE.PA': (0.0, '2026-08-06'), 'RBT.PA': (4.5e4, '2026-08-06')},
     'CBE.PA', 'alphabetical'),
    ('a non-numeric reading',
     {'CBE.PA': ('n/a', '2026-08-06'), 'RBT.PA': (4.5e4, '2026-08-06')},
     'CBE.PA', 'alphabetical'),
    ('a NaN reading',
     {'CBE.PA': (float('nan'), '2026-08-06'), 'RBT.PA': (4.5e4, '2026-08-06')},
     'CBE.PA', 'alphabetical'),
    ('MIXED as-of dates -- merge-never-overwrite carries a STALE reading forward, and '
     'comparing it against a fresh one compares two market regimes',
     {'CBE.PA': (1.2e3, '2026-02-01'), 'RBT.PA': (4.5e4, '2026-08-06')},
     'CBE.PA', 'alphabetical'),
    #  --- the only two of the nine the raw term may reach: the guard PASSES (every member
    #  has a usable reading on one date) and only the DECADE term abstains, on the ratio.
    ('both within ONE ORDER OF MAGNITUDE (2.25x) -- the DECADE term has no opinion, but '
     'from 2026-08-08 the weak raw term takes it for the more liquid line',
     {'CBE.PA': (2.0e4, '2026-08-06'), 'RBT.PA': (4.5e4, '2026-08-06')},
     'RBT.PA', 'volavg_raw'),
    ('a 2% difference STRADDLING A POWER OF TEN -- the DECADE term must stay silent (the '
     'bucketed first cut let this decide with full force); the raw term may take it, but '
     'only under its own name',
     {'CBE.PA': (1.01e4, '2026-08-06'), 'RBT.PA': (9.9e3, '2026-08-06')},
     'CBE.PA', 'volavg_raw'),
)


@pytest.mark.parametrize('label,vmap,exp_surv,exp_decided', VOLAVG_ABSTAIN_CASES,
                         ids=[c[0][:40] for c in VOLAVG_ABSTAIN_CASES])
def test_volavg_abstains_and_abstention_is_the_LITERAL_zero(
        label, vmap, exp_surv, exp_decided):
    """The DECADE term must abstain on all nine, as the LITERAL 0 for every member --
    auditable as a value, not as a cancellation -- and the survivor and the deciding term
    must be EXACTLY the pinned ones.

    The strong per-case expectation is what makes this a guard rather than a smoke test:
    seven of the nine are cases where `_volavg_comparable_values` refuses the whole group,
    and every one of those must still fall to the raw ALPHABET with `CBE.PA` surviving.  If
    a change ever lets an unusable reading be SKIPPED rather than abstain the group, the
    mapped line wins on data availability and `RBT.PA` survives -- which these assertions
    catch and a membership test over ('volavg_raw', 'alphabetical') does not."""
    group, vf, names = _k1_shaped_group()
    vals = [co._volavg_liquidity_term(s, group, vmap) for s in group]
    assert vals == [0, 0], (
        '%s: expected literal-0 abstention for every member, got %r' % (label, vals))
    keys = {s: co._investability_key(s, vf, None, names, group, {}, vmap) for s in group}
    order = sorted(group, key=keys.__getitem__)
    assert order[0] == exp_surv, (
        '%s: survivor is %r, expected %r -- if the expected survivor was the '
        'alphabetically-first line, an unusable reading has stopped abstaining the group '
        'and the pick is now decided by DATA AVAILABILITY' % (label, order[0], exp_surv))
    decided = co._deciding_term(keys[order[0]], keys[order[1]])
    assert decided == exp_decided, (
        '%s: decided_by is %r, expected %r' % (label, decided, exp_decided))
    assert decided != 'volavg', (
        '%s: the decade term abstained but was reported as the deciding term' % label)


def test_volavg_abstain_decision_never_reads_the_symbol_under_test():
    """The reviewer's ISIN lesson, applied pre-emptively: the decision to SPEAK is a
    function of (volavg_map, group) alone, so it cannot differ between two members of one
    group -- which is what made the first ISIN cut decide on data availability."""
    group, _vf, _names = _k1_shaped_group()
    for _label, vmap, _exp_surv, _exp_decided in VOLAVG_ABSTAIN_CASES:
        terms = {s: co._volavg_liquidity_term(s, group, vmap) for s in group}
        assert len(set(terms.values())) == 1, (
            'members of one group disagreed about whether to abstain: %r' % terms)


def test_volavg_loader_reads_BOTH_the_dated_and_the_undated_shape(tmp_path):
    """findAllSectors wrote a BARE value before 2026-08-06 and a {'volAvg','asof'} dict
    after.  A pickle in either shape must load, because the CEO's other machine may hold
    the older one -- and an undated entry must come back with asof None so the mixed-date
    guard treats a wholly-undated map as self-consistent rather than refusing to work."""
    dated = tmp_path / 'volavgdic_fmp_2026-08-06.pickle'
    pd.to_pickle({'A': {'volAvg': 123.0, 'asof': '2026-08-06'}}, dated)
    assert co._load_volavg_map(str(dated)) == {'A': (123.0, '2026-08-06')}
    undated = tmp_path / 'volavgdic_fmp_2026-08-05.pickle'
    pd.to_pickle({'A': 123.0, 'B': None}, undated)
    assert co._load_volavg_map(str(undated)) == {'A': (123.0, None), 'B': (None, None)}


def test_an_entirely_undated_map_still_discriminates():
    """The corollary: an OLD undated pickle has asof None for every entry, which is
    self-consistently "all the same unknown date", so it is allowed to speak.  Refusing it
    would silently disable the fix on the one machine that already has a map."""
    group, vf, names = _k1_shaped_group()
    vmap = {'CBE.PA': (1.2e3, None), 'RBT.PA': (4.5e4, None)}
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}, vmap))[0] == 'RBT.PA'


# =========================================================================== #
#  THE RUN ANSWERS ITS OWN QUESTION ABOUT THE TIEBREAK  (2026-08-06)           #
#                                                                              #
#  The report frame recorded neither term value, so on the three K-1 groups it   #
#  was impossible to tell from the run's own output whether volAvg SPOKE AND LOST #
#  or ABSTAINED -- and group closure enlarged exactly those groups with the IOB   #
#  lines most likely to report volAvg 0/null, which is an abstention trigger.     #
# =========================================================================== #

_TERM_COLS = ('decided_by', 'dropped_vol_t', 'survivor_vol_t',
              'dropped_isin_t', 'survivor_isin_t')


def _dedup_on_panel(panel, isin=None, volavg=None):
    """`dedup_to_issuers` over the panel, optionally with injected maps."""
    d, names = panel['d'], panel['names']
    bs = pd.DataFrame({'source': panel['syms']})
    old_i, old_v = co._ISIN_MAP_CACHE, co._VOLAVG_MAP_CACHE
    try:
        if isin is not None:
            co._ISIN_MAP_CACHE = isin
        if volavg is not None:
            co._VOLAVG_MAP_CACHE = volavg
        return co.dedup_to_issuers(bs, d['cdx_df'], {}, names)
    finally:
        co._ISIN_MAP_CACHE, co._VOLAVG_MAP_CACHE = old_i, old_v


def test_the_new_audit_columns_DO_NOT_CHANGE_THE_PICK(panel):
    """*** THE PROPERTY THE BRIEF ASKED TO BE ASSERTED: the columns are OBSERVATIONAL.
    The survivor of every group must be bit-identical to an INDEPENDENT re-derivation that
    sorts each group with `_investability_key` directly -- so neither the added columns nor
    the memoisation of the key inside `dedup_to_issuers` moved a single pick. ***
    Re-derived rather than compared against a stored list, because a stored list would only
    say "unchanged since the day I stored it"."""
    out = _dedup_on_panel(panel)
    val, names = panel['val'], panel['names']
    for members in panel['comps'].values():
        expect = sorted(members, key=lambda s: co._investability_key(
            s, val, None, names, members))[0]
        for m in members:
            assert out['member_to_survivor'][m] == expect, (
                'survivor of %s moved: %r vs the independently re-derived %r'
                % (sorted(members), out['member_to_survivor'][m], expect))
    assert len(out['survivors']) == len(panel['comps'])


def test_the_report_frame_carries_BOTH_TERMS_for_EVERY_MEMBER_of_a_group(panel):
    """Per-GROUP readability, which is the requirement: one row per DROPPED member plus the
    survivor's value repeated on it covers every member, and the winner's value alone
    cannot distinguish abstain from spoke-and-lost."""
    rep = _dedup_on_panel(panel)['diagnostics']['report']
    for c in _TERM_COLS:
        assert c in rep.columns, 'the report frame is missing %r' % c
    assert len(rep) > 100, 'panel produced too few dropped rows to be evidence'
    sizes = rep['issuer_group'].value_counts()
    for grp, n_rows in sizes.items():
        assert n_rows == len(grp.split('|')) - 1, (
            'group %s has %d dropped rows for %d members -- the frame is not per-member '
            'complete' % (grp, n_rows, len(grp.split('|'))))
    for _i, r in rep.iterrows():
        members = r['issuer_group'].split('|')
        assert r['dropped'] in members and r['survivor'] in members


def test_decided_by_names_the_TERM_THE_SORT_ACTUALLY_USED(panel):
    """`decided_by` must be re-derivable from the two keys, and never blank on a real
    dropped row (the last key term is the unique symbol, so two members cannot tie)."""
    rep = _dedup_on_panel(panel)['diagnostics']['report']
    val, names = panel['val'], panel['names']
    for _i, r in rep.iterrows():
        members = r['issuer_group'].split('|')
        ks = co._investability_key(r['survivor'], val, None, names, members)
        kd = co._investability_key(r['dropped'], val, None, names, members)
        assert r['decided_by'] == co._deciding_term(ks, kd)
        assert r['decided_by'] in co._KEY_TERM_NAMES, (
            '%s -> %s: decided_by=%r is not a key term'
            % (r['dropped'], r['survivor'], r['decided_by']))
        #  and the survivor really did win on it
        i = co._KEY_TERM_NAMES.index(r['decided_by'])
        assert ks[i] < kd[i]


def test_NO_DATA_reads_as_BLANK_and_not_as_a_DECISION(panel):
    """*** THE HONEST-NO-DATA REQUIREMENT.  Every pickle in existence has neither map, so
    both terms are a constant 0 -- and a column of zeros reads exactly like "the term looked
    and found every line comparable", which is a different and much stronger claim than
    "there was nothing to look at". ***"""
    out = _dedup_on_panel(panel, isin={}, volavg={})
    rep, diag = out['diagnostics']['report'], out['diagnostics']
    assert diag['volavg_map_n'] == 0 and diag['isin_map_n'] == 0
    for c in ('dropped_vol_t', 'survivor_vol_t', 'dropped_isin_t', 'survivor_isin_t'):
        assert rep[c].isna().all(), (
            '%s is not blank with no map on disk -- a constant 0 there would read as a '
            'liquidity/identity finding that was never made' % c)
    assert not (rep['decided_by'].isin(('volavg', 'isin_plurality'))).any(), \
        'a term with no map cannot have decided anything'


def _k1_cdx(syms):
    """A minimal cdx frame for a K-1-shaped group: identical issuer-level share count and
    market cap (so the derived price is identical too), one date, same exchange."""
    n = len(syms)
    return pd.DataFrame({'source': syms, 'date': ['2026-01-01'] * n,
                         'weightedAverageShsOut': [100.0] * n,
                         'marketCap': [5.0] * n, 'price': [0.05] * n,
                         'netIncome': [7.0] * n, 'revenue': [9.0] * n})


def test_ABSTAIN_and_SPOKE_AND_LOST_are_DISTINGUISHABLE_in_the_frame():
    """*** THE QUESTION THE COLUMNS EXIST TO ANSWER, on a K-1-shaped group run through the
    real function. ***  Same group three times: volumes a decade apart (the term SPEAKS and
    the illiquid line loses on it), volumes inside one decade (ABSTAINS, the alphabet
    decides), and mixed as-of dates (REFUSES the comparison -- also an abstention, and the
    state the IOB lines in the closed K-1 groups can actually produce).  The three must not
    look alike in the frame."""
    syms = ['CBE.PA', 'RBT.PA']
    cdx = _k1_cdx(syms)
    names = {s: 'Robertet S.A.' for s in syms}
    bs = pd.DataFrame({'source': syms})

    def _run(vmap):
        old_v, old_i = co._VOLAVG_MAP_CACHE, co._ISIN_MAP_CACHE
        try:
            co._VOLAVG_MAP_CACHE, co._ISIN_MAP_CACHE = vmap, {}
            return co.dedup_to_issuers(bs, cdx, {}, names)
        finally:
            co._VOLAVG_MAP_CACHE, co._ISIN_MAP_CACHE = old_v, old_i

    r = _run({'CBE.PA': (1.2e3, '2026-08-06'), 'RBT.PA': (4.5e4, '2026-08-06')})
    rep = r['diagnostics']['report']
    if rep.empty:
        pytest.skip('the fixture did not group -- grouping is exercised elsewhere')
    row = rep.iloc[0]
    assert (row['survivor'], row['dropped']) == ('RBT.PA', 'CBE.PA')
    assert (row['survivor_vol_t'], row['dropped_vol_t']) == (0, 1), \
        'SPOKE-AND-LOST must read as survivor 0 / dropped 1'
    assert row['decided_by'] == 'volavg'

    row = _run({'CBE.PA': (4.0e4, '2026-08-06'),
                'RBT.PA': (4.5e4, '2026-08-06')})['diagnostics']['report'].iloc[0]
    #  *** CHANGED 2026-08-08. ***  Inside one decade the DECADE term still abstains --
    #  which is what the vol_t columns must show -- but the group no longer falls to the
    #  alphabet: the weak `volavg_raw` term now takes it, and takes it for the COMMON
    #  (4.5e4 > 4.0e4).  This fixture used to be the standing demonstration that the
    #  certificat wins a known-wrong group; it is now the demonstration that it does not.
    assert row['survivor'] == 'RBT.PA', (
        'the raw volume tiebreak did not reach an in-decade K-1 group -- the certificat '
        'still wins, which is the defect it was added to fix')
    assert (row['survivor_vol_t'], row['dropped_vol_t']) == (0, 0), \
        'ABSTAIN must read as 0 for EVERY member -- that is what separates it from a loss'
    assert row['decided_by'] == 'volavg_raw'
    #  ... and the RAW readings are on the row, which is what makes the margin judgeable.
    assert row['survivor_volAvg'] == 4.5e4 and row['dropped_volAvg'] == 4.0e4
    assert row['survivor_volAvg_asof'] == row['dropped_volAvg_asof'] == '2026-08-06'

    row = _run({'CBE.PA': (1.2e3, '2026-02-01'),
                'RBT.PA': (4.5e4, '2026-08-06')})['diagnostics']['report'].iloc[0]
    assert (row['survivor_vol_t'], row['dropped_vol_t']) == (0, 0), \
        'a REFUSED comparison is an abstention and must read as one'
    #  The date-disagreement abstention SURVIVES into the raw term (CEO requirement), so
    #  this group still falls to the alphabet -- unlike the in-decade case above.
    assert row['decided_by'] == 'alphabetical'
    #  And the report now SHOWS why: two different as-of dates on one group is exactly the
    #  state that looks like a liquidity difference and is not one.  Before these columns
    #  this row was indistinguishable from a genuine near-tie.
    #  The SURVIVOR is CBE.PA (it won on the alphabet), and CBE.PA is the STALE line -- so
    #  the group was decided alphabetically while carrying a Feb reading against an Aug one.
    #  That is precisely the case the old report could not express.
    assert row['survivor_volAvg_asof'] == '2026-02-01'
    assert row['dropped_volAvg_asof'] == '2026-08-06'


def test_a_MISSING_reading_on_ONE_member_reads_as_ABSTAIN_not_as_a_LOSS():
    """The state group closure MADE MORE LIKELY: an added IOB sibling reporting volAvg 0 or
    null.  `_volavg_liquidity_term` abstains for the WHOLE group (condition 1), so a reader
    must see three zeros -- not "the null line lost"."""
    syms = ['CBE.PA', 'RBT.PA', '0NZN.L']
    names = {s: 'Robertet S.A.' for s in syms}
    bs = pd.DataFrame({'source': syms})
    old_v, old_i = co._VOLAVG_MAP_CACHE, co._ISIN_MAP_CACHE
    try:
        co._ISIN_MAP_CACHE = {}
        co._VOLAVG_MAP_CACHE = {'CBE.PA': (1.2e3, '2026-08-06'),
                                'RBT.PA': (4.5e4, '2026-08-06'),
                                '0NZN.L': (0, '2026-08-06')}     # the IOB line, volAvg 0
        rep = co.dedup_to_issuers(bs, _k1_cdx(syms), {}, names)['diagnostics']['report']
    finally:
        co._VOLAVG_MAP_CACHE, co._ISIN_MAP_CACHE = old_v, old_i
    if rep.empty:
        pytest.skip('the fixture did not group -- grouping is exercised elsewhere')
    assert set(rep['survivor_vol_t']) == {0} and set(rep['dropped_vol_t']) == {0}, \
        'one unusable reading must abstain the whole group, not demote the null line'
    assert set(rep['decided_by']) <= {'canonicity', 'alphabetical'}


def test_the_deciding_term_does_not_invent_a_decision_out_of_NaN():
    """`val_fn` can serve NaN for an unmeasured share count, and NaN != NaN -- so a naive
    comparison would name `shares` as the deciding term for two lines that are both merely
    unmeasured.  A fabricated decision in the one column added so nobody has to guess."""
    #  TEN elements as of 2026-08-08 (the raw-volume tiebreak sits between isin_plurality
    #  and alphabetical).  `_deciding_term` asserts the arity, so a stale literal here fails
    #  loudly rather than silently mislabelling the column.
    k = (0, float('nan'), float('nan'), 0, 0, 4, 0, 0, 0, 'AAAA')
    other = k[:-1] + ('ZZZZ',)
    assert co._deciding_term(k, other) == 'alphabetical'
    assert co._same_key_term(float('nan'), float('nan'))
    assert not co._same_key_term(float('nan'), 1.0)


# =========================================================================== #
#  RAW volAvg AS THE LAST TIEBREAK BEFORE THE ALPHABET (CEO ruling 2026-08-08)  #
#                                                                               #
#  The decade term above only speaks on a >=10x gap; the groups it ties fell to   #
#  the RAW ALPHABET, which correlates with nothing.  This block pins the same     #
#  three properties: (1) NO DATA -> BIT-IDENTICAL, (2) what it buys, (3) every    #
#  condition under which it must ABSTAIN -- plus (4) that it cannot outrank ANY   #
#  term above it, which is the entire safety argument for a WEAK signal.          #
# =========================================================================== #

def _pre_volraw_key(sym, val_fn, names=None, group=(), isin_map=None, volavg_map=None):
    """The survivor key EXACTLY as it stood after the decade term + ISIN and BEFORE the raw
    volume term.  Duplicated rather than derived, same reason as `_pre_volavg_key`."""
    nm = (names or {}).get(sym, '') if names else ''
    noncanon = 1 if co._non_canonical_tag(sym, nm, group) else 0
    sh = val_fn(sym, 'weightedAverageShsOut')
    sh = sh if sh is not None else -1.0
    mc = val_fn(sym, 'marketCap')
    mc = mc if mc is not None else -1.0
    digitpfx = 1 if sym[:1].isdigit() else 0
    punct = sum(ch in '-.' for ch in sym)
    imap = {} if isin_map is None else isin_map
    vmap = {} if volavg_map is None else volavg_map
    return (noncanon, -sh, -mc, digitpfx, punct, len(sym),
            co._volavg_liquidity_term(sym, group, vmap),
            co._isin_plurality_term(sym, group, imap), sym)


def test_volraw_absent_is_bit_identical(panel):
    """*** THE PROPERTY THE BRIEF ASKED TO BE ASSERTED. ***  Every pickle written before
    2026-08-08 has no volavgdic map, so with NO MAP the whole ORDERING -- not just the
    winner, because a term that reorders the LOSERS still changes the audit trail -- must be
    what the pre-2026-08-08 key produced.

    *** SCOPE CORRECTED 2026-08-08 (reviewer F3).  THIS COVERS THE ABSENT-MAP PATH ONLY. ***
    The first cut also iterated over "the map the process would actually load", which read
    like coverage of the POPULATED case and was not: the baseline was computed with an empty
    map in both iterations, so on a machine with no pickle the second pass merely re-asserted
    that an empty map changes nothing, and on the run machine (where the 2026-08-08 build
    left 3,127 entries) it would FAIL -- 111 of 1,282 groups reorder, which is the feature
    working, not a defect.  Bit-identity is a claim about the ABSENT map and nothing else.
    The populated map is covered by `test_volraw_under_a_POPULATED_map_...` below."""
    val, names = panel['val'], panel['names']
    n_groups = 0
    for m in panel['comps'].values():
        if len(m) < 2:
            continue
        n_groups += 1
        old = sorted(m, key=lambda s: _pre_volraw_key(s, val, names, m, {}, {}))
        new = sorted(m, key=lambda s: co._investability_key(
            s, val, None, names, m, {}, {}))
        assert new == old, (
            'the raw volume term changed the ordering of %s with an ABSENT map; the '
            'no-map path is NOT bit-identical' % (m,))
    assert n_groups > 100, 'panel produced too few multi-line groups to be evidence'


def _synthetic_volavg_map(symbols, asof='2026-08-08'):
    """A POPULATED volAvg map over real panel symbols, deterministic and dense.

    Deliberately synthetic rather than the on-disk map: no volavgdic pickle exists on a dev
    machine, so a test that depends on one is vacuous exactly where it matters.  Values are
    spread over ~3 decades so that some groups clear the 10x decade gap and others sit inside
    it -- i.e. both the confident term and the weak raw term get exercised -- and EVERY entry
    carries ONE as-of date so the guard passes and the terms actually speak.
    *** USES crc32, NOT `hash()`. ***  Python randomises `hash()` on str per process
    (PYTHONHASHSEED), so a map built from it would give a DIFFERENT assignment on every run
    -- which for a test whose assertions depend on how many groups the term reaches means
    intermittent failures that reproduce on nobody's machine.  crc32 is stable across
    processes and versions.
    """
    out = {}
    for sym in sorted(symbols):
        h = zlib.crc32(sym.encode('utf-8'))
        out[sym] = (float(1000 * (1 + (h % 997))), asof)
    return out


def test_volraw_under_a_POPULATED_map_cannot_outrank_ANY_term_above_it(panel):
    """*** THE COVERAGE THAT WAS MISSING (reviewer F3), AND THE STATE THE PIPELINE IS
    ACTUALLY IN. ***  Since the 2026-08-08 build the maps are populated, so every assertion
    that only ever ran against an empty map was testing a configuration the run no longer
    has.  This exercises the raw term against a DENSE map over real panel groups.

    THE INVARIANT, and it is the whole safety argument for a weak signal: the raw term sits
    at position 8, so it may only break ties that terms 1-7 left.  Concretely -- for every
    PAIR of members whose keys already differ on terms 1-7, their relative order must be
    IDENTICAL with and without the map.  Only pairs tied on all of 1-7 may move.  That is
    strictly stronger than checking the winner, because it also pins the loser ordering the
    audit trail reads."""
    val, names = panel['val'], panel['names']
    all_syms = {s for m in panel['comps'].values() for s in m}
    vmap = _synthetic_volavg_map(all_syms)
    n_groups = n_reordered = n_pairs_pinned = 0
    for m in panel['comps'].values():
        if len(m) < 2:
            continue
        n_groups += 1
        pre = {s: _pre_volraw_key(s, val, names, m, {}, vmap) for s in m}
        post = {s: co._investability_key(s, val, None, names, m, {}, vmap) for s in m}
        if sorted(m, key=pre.__getitem__) != sorted(m, key=post.__getitem__):
            n_reordered += 1
        for a in m:
            for b in m:
                if a >= b:
                    continue
                #  Terms 1-7 = everything in the pre-raw key except its trailing symbol.
                if pre[a][:-1] == pre[b][:-1]:
                    continue                      # tied above -- the raw term MAY decide
                n_pairs_pinned += 1
                assert (pre[a] < pre[b]) == (post[a] < post[b]), (
                    'the raw volume term reversed %r vs %r, which terms 1-7 had already '
                    'separated -- it is outranking a term above it' % (a, b))
    assert n_groups > 100, 'panel produced too few multi-line groups to be evidence'
    assert n_pairs_pinned > 100, (
        'no pair was actually separated by terms 1-7, so this test asserted nothing')
    assert n_reordered > 0, (
        'a DENSE volAvg map reordered ZERO groups -- the raw term is not reachable on this '
        'panel, so this test is vacuous and the populated case is still uncovered')


def test_volraw_under_a_POPULATED_map_only_ever_takes_groups_from_the_ALPHABET(panel):
    """The other half of F3's missing coverage, stated as the promise the CEO was given:
    the volume terms reach EXACTLY the groups that would otherwise fall to raw alphabet.  So
    for every group, the deciding term with the map populated must either be UNCHANGED from
    the no-map run, or be one of the two VOLUME terms where the no-map run said
    `alphabetical` -- never a term above, and never a group that some real marker had
    already decided.

    BOTH volume terms are legitimate takers here and the test counts them separately: a
    dense map over three decades leaves some groups more than 10x apart (the confident
    `volavg` term takes those) and others inside one decade (`volavg_raw` takes those).
    Requiring `volavg_raw` alone was wrong and this test caught it -- FPAR-D.ST/FPAR-A.ST
    clears the decade gap under the synthetic map."""
    val, names = panel['val'], panel['names']
    all_syms = {s for m in panel['comps'].values() for s in m}
    vmap = _synthetic_volavg_map(all_syms)
    n_taken = 0
    taken_by = {}
    for m in panel['comps'].values():
        if len(m) < 2:
            continue
        bare = {s: co._investability_key(s, val, None, names, m, {}, {}) for s in m}
        full = {s: co._investability_key(s, val, None, names, m, {}, vmap) for s in m}
        b_ord = sorted(m, key=bare.__getitem__)
        f_ord = sorted(m, key=full.__getitem__)
        b_dec = co._deciding_term(bare[b_ord[0]], bare[b_ord[1]])
        f_dec = co._deciding_term(full[f_ord[0]], full[f_ord[1]])
        if f_dec == b_dec:
            continue
        assert b_dec == 'alphabetical', (
            'group %r was decided by %r without the map and by %r with it -- a volume term '
            'took a group that a real marker had already decided' % (m, b_dec, f_dec))
        assert f_dec in ('volavg', 'volavg_raw'), (
            'group %r changed deciding term to %r, which is neither volume tiebreak'
            % (m, f_dec))
        taken_by[f_dec] = taken_by.get(f_dec, 0) + 1
        n_taken += 1
    assert n_taken > 0, 'no volume term took an alphabetical group -- test is vacuous'
    assert taken_by.get('volavg_raw', 0) > 0, (
        'the WEAK RAW term took no group at all (%r) -- this test would then be covering '
        'only the decade term, which is exactly the gap F3 reported' % taken_by)


def test_volraw_decides_a_group_the_DECADE_term_ties():
    """WHAT THE TERM BUYS.  A K-1-shaped group inside one order of magnitude: the decade
    term abstains (so this is exactly a group that fell to the alphabet), and the raw term
    must now hand it to the MORE LIQUID line even though it sorts LAST alphabetically."""
    group, vf, names = _k1_shaped_group()
    vmap = {'CBE.PA': (3.0e4, '2026-08-08'), 'RBT.PA': (4.5e4, '2026-08-08')}
    assert {co._volavg_liquidity_term(s, group, vmap) for s in group} == {0}, \
        'the fixture must have the DECADE term abstaining, or it proves nothing'
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}, vmap))[0] == 'RBT.PA'


def test_volraw_cannot_outrank_ISIN_or_ANY_term_above_it():
    """POSITION IS THE SAFETY ARGUMENT FOR A WEAK SIGNAL.  Same fixture the ISIN/volAvg
    order test uses, with volumes moved INSIDE one order of magnitude so the decade term
    abstains: ISIN must still decide, exactly as it did before 2026-08-08, and the raw term
    must not steal the group."""
    group, vf, names, imap, _ = _isin_vs_volavg_fixture()
    #  AAA is the most liquid, so if the raw term outranked ISIN the winner would be AAA.
    vmap = {'AAA': (1.2e4, '2026-08-08'), 'ZZY': (1.1e4, '2026-08-08'),
            'ZZZ': (1.0e4, '2026-08-08')}
    assert {co._volavg_liquidity_term(s, group, vmap) for s in group} == {0}
    winner = sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, imap, vmap))[0]
    assert winner in ('ZZY', 'ZZZ'), (
        'the raw volume term outranked ISIN plurality (winner %r) -- it was placed ABOVE '
        'term 6 instead of below it' % winner)


def test_volraw_cannot_outrank_a_canonicity_marker():
    """A million times the volume on the NON-COMMON must still lose to a canonicity marker,
    or the measured 0.47% canonicity-first failure rate has regressed."""
    group = ['SMSD.L', 'SMSN.L']
    fake = {s: {'weightedAverageShsOut': 100.0, 'marketCap': 5.0} for s in group}
    vf = lambda s, c: fake[s][c]
    names = {'SMSD.L': 'Samsung Electronics Co., Ltd. Pfd Registered Shs Non-Voting',
             'SMSN.L': 'Samsung Electronics Co., Ltd.'}
    #  Inside one order of magnitude, so ONLY the raw term can be speaking.
    vmap = {'SMSD.L': (9.0e4, '2026-08-08'), 'SMSN.L': (1.0e4, '2026-08-08')}
    assert {co._volavg_liquidity_term(s, group, vmap) for s in group} == {0}
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}, vmap))[0] == 'SMSN.L'


@pytest.mark.parametrize('label,vmap', [
    ('a member missing from the map',
     {'CBE.PA': (3.0e4, '2026-08-08')}),
    ('a member with a NULL reading',
     {'CBE.PA': (3.0e4, '2026-08-08'), 'RBT.PA': (None, '2026-08-08')}),
    ('a member with a ZERO reading',
     {'CBE.PA': (3.0e4, '2026-08-08'), 'RBT.PA': (0.0, '2026-08-08')}),
    ('DISAGREEING as-of dates',
     {'CBE.PA': (3.0e4, '2026-08-08'), 'RBT.PA': (4.5e4, '2026-02-01')}),
])
def test_volraw_abstains_and_abstention_is_the_LITERAL_zero(label, vmap):
    """*** THE CEO'S TWO EXPLICIT REQUIREMENTS, ASSERTED. ***
    (a) an absent / zero / null reading must neither WIN nor be DEMOTED -- so the whole
        group abstains and the survivor is whatever the alphabet gave it, unchanged;
    (b) the date-disagreement abstention SURVIVES into the raw term, where it matters more:
        a raw comparison acts on differences far smaller than the drift a stale reading
        accumulates.
    Abstention must be the LITERAL 0 for every member, not merely a tie that cancels, so the
    report can read it as a value."""
    group, vf, names = _k1_shaped_group()
    assert {co._volavg_raw_liquidity_term(s, group, vmap) for s in group} == {0}, \
        'the raw term did not abstain on: %s' % label
    #  ... and therefore the group falls to the alphabet, exactly as it does today.
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}, vmap))[0] == 'CBE.PA'


def test_volraw_absence_is_never_read_as_a_volume_of_ZERO():
    """The sharpest form of (a): if absence were treated as 0 the unmapped member would be
    DEMOTED and the mapped one would win on DATA AVAILABILITY alone.  Map ONLY the
    alphabetically-LAST line and assert it does NOT win."""
    group, vf, names = _k1_shaped_group()
    vmap = {'RBT.PA': (4.5e4, '2026-08-08')}
    assert co._volavg_raw_liquidity_term('RBT.PA', group, vmap) == 0
    assert sorted(group, key=lambda s: co._investability_key(
        s, vf, None, names, group, {}, vmap))[0] == 'CBE.PA', (
        'a mapped member beat an unmapped one -- the survivor was decided by DATA '
        'AVAILABILITY, which is the defect the shared abstention guard exists to prevent')


def test_volraw_has_its_OWN_name_in_decided_by():
    """The answer to the "wrong-but-confident tiebreak LOOKS principled" objection: a pick
    made on a 1.03x margin must NOT be reported under the confident decade term's name."""
    assert 'volavg_raw' in co._KEY_TERM_NAMES
    assert (co._KEY_TERM_NAMES.index('volavg_raw')
            > co._KEY_TERM_NAMES.index('isin_plurality'))
    assert (co._KEY_TERM_NAMES.index('volavg_raw')
            < co._KEY_TERM_NAMES.index('alphabetical'))
    group, vf, names = _k1_shaped_group()
    vmap = {'CBE.PA': (4.37e4, '2026-08-08'), 'RBT.PA': (4.5e4, '2026-08-08')}   # 1.03x
    keys = {s: co._investability_key(s, vf, None, names, group, {}, vmap) for s in group}
    assert co._deciding_term(keys['RBT.PA'], keys['CBE.PA']) == 'volavg_raw'


def test_the_shared_abstention_guard_is_ONE_function_for_BOTH_terms():
    """The extraction is the reason the two terms cannot drift apart.  Every case the guard
    rejects must abstain BOTH terms, together."""
    group = ['AAA', 'BBB']
    for vmap in ({}, {'AAA': (1e5, 'd')}, {'AAA': (1e5, 'd'), 'BBB': (0, 'd')},
                 {'AAA': (1e5, 'd1'), 'BBB': (1e2, 'd2')}):
        assert co._volavg_comparable_values(group, vmap) is None
        assert {co._volavg_liquidity_term(s, group, vmap) for s in group} == {0}
        assert {co._volavg_raw_liquidity_term(s, group, vmap) for s in group} == {0}


# --------------------------------------------------------------------------- #
#  THE RAW VOLUMES ON THE DEDUP REPORT (CEO 2026-08-08)                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('vmap,exp_val,exp_asof', [
    ({}, None, co.VOLAVG_STATUS_NOT_CAPTURED),
    ({'X': (None, '2026-08-08')}, None, co.VOLAVG_STATUS_NO_READING),
    ({'X': (0.0, '2026-08-08')}, None, co.VOLAVG_STATUS_NO_READING),
    ({'X': (1234.0, None)}, 1234.0, co.VOLAVG_STATUS_UNDATED),
    ({'X': (1234.0, '2026-08-08')}, 1234.0, '2026-08-08'),
])
def test_the_three_kinds_of_absence_stay_DISTINGUISHABLE_from_a_real_zero(
        vmap, exp_val, exp_asof):
    """`not-captured` / `no-reading` / `undated-capture` must not collapse into each other
    or into a genuine reading -- the report columns and the per-name CSV columns share ONE
    implementation (`_volavg_reading`) so they cannot drift."""
    import math as _m
    v, a = co._volavg_reading('X', vmap)
    assert a == exp_asof
    if exp_val is None:
        assert _m.isnan(v)
    else:
        assert v == exp_val
    #  ... and the per-name CSV frame agrees, because it is the same function.
    fr = co.volavg_report_frame(['X'], volavg_map=vmap)
    assert fr['volAvg_asof'].iloc[0] == exp_asof


# =========================================================================== #
#  AN AMBIGUOUS SECTOR ROUTES TO GENERAL  (CEO, 2026-08-10)                    #
# =========================================================================== #
def _dedup_sectors(groups, sector_map, names=None):
    """Run `dedup_to_issuers` over synthetic issuer groups and return
    (propagated sector per survivor, the conflicts it reported).

    `cdx_df` carries one row per symbol with matching `weightedAverageShsOut` so the
    grouping key collapses each group -- this test is about the SECTOR VOTE, not about the
    grouping, which has its own tests above.
    """
    rows, _names, ranked = [], {}, []
    for gi, (members, nm) in enumerate(groups):
        for s in members:
            rows.append({'source': s, 'date': '2026-03-31', 'price': 10.0,
                         'weightedAverageShsOut': 1000.0 + gi, 'marketCap': 10000.0 + gi,
                         'totalAssets': 1.0, 'revenue': 1.0})
            _names[s] = nm
            ranked.append(s)
    cdx = pd.DataFrame(rows)
    #  `names` matters now: `_non_canonical_tag`'s name-vocabulary rule is what identifies a
    #  baby bond, so a fixture that passes only symbols would silently give every line a vote.
    _names = {**_names, **(names or {})}
    ded = co.dedup_to_issuers(pd.DataFrame({'source': ranked, 'score': range(len(ranked))}),
                              cdx, sector_map, _names)
    return ded['sector_override'], ded['diagnostics']['sector_conflicts']


def test_a_TIED_sector_conflict_routes_the_issuer_to_GENERAL_not_into_a_cohort():
    """*** THE RULE THE `MAS` CASE MOTIVATED. ***  A DEAD TIE between two equity lines used to
    be resolved IN FAVOUR OF THE COHORT SECTOR, i.e. decided on no evidence at all -- and a
    cohort carries SPECIALIST SOLVENCY FLAGS, so guessing INTO one is strictly more dangerous
    than guessing out of one.  The general pool's flag set is the one designed for a company
    we cannot classify.

    WHAT ACTUALLY HAPPENS TO `MAS` ITSELF, corrected 2026-08-10 once the run's real sector map
    was recovered (the repo's copy is a DIFFERENT TAXONOMY -- see the warning in `carveOut`):
    **it stays in Mining.**  Its own primary line is tagged `Basic Materials`; only its LSE IOB
    sibling said `Consumer Cyclical`, and the equity-voter rule strips that vote, so no tie
    survives to resolve.  The vendor genuinely files a building-products company as a materials
    name, and no tiebreak can improve on a correct-but-unhelpful tag on the only line that has
    one -- that is the parked business-model cohort work.
    So this fixture pins the RULE on a synthetic tie, not that name's fate.
    """
    override, conflicts = _dedup_sectors(
        [(['MAS', 'MAS.SW'], 'Masco Corporation')],
        {'MAS': 'Consumer Cyclical', 'MAS.SW': co.MINING_SECTOR})
    assert override['MAS'] == co.AMBIGUOUS_SECTOR
    assert len(conflicts) == 1 and conflicts[0][2] == co.AMBIGUOUS_SECTOR
    #  ...and the sentinel must actually ROUTE to general through the real classifier, not
    #  merely be a different string.
    labels, _reasons = co.classify(['MAS'], override, pd.DataFrame(), {'MAS': 'Masco'})
    assert labels['MAS'] == 'general'


def test_the_ambiguous_sentinel_is_KNOWN_and_matches_NO_cohort_sector():
    """It must survive `_is_known_sector` -- an ambiguous issuer is not an UNMAPPED one; we
    know its tags and cannot choose between them -- while matching none of the three cohort
    sectors, so `classify` routes it by its existing default rather than by a new branch."""
    assert co._is_known_sector(co.AMBIGUOUS_SECTOR)
    assert co.AMBIGUOUS_SECTOR not in (co.REIT_SECTOR, co.MINING_SECTOR,
                                       co.FINANCIAL_SECTOR)
    assert co.AMBIGUOUS_SECTOR not in co._UNKNOWN_SECTORS


def test_only_the_issuers_EQUITY_lines_get_a_sector_vote():
    """*** THE `AFG` CASE (reviewer S2, 2026-08-10). ***  American Financial Group is tagged
    Financial Services x1 -- its equity -- against Industrials x2, which are `AFGB` and `AFGE`,
    its SUBORDINATED DEBENTURES.  Plurality alone therefore moved a P&C insurer out of
    BalanceSheetFin and into the general pool, and NOT as an ambiguity: it was a confident
    wrong answer, unflagged.

    A bond line carries the vendor's classification of the INSTRUMENT, not of the ISSUER, and
    there can be several of them against one equity line -- so they can carry the plurality
    outright.  The filter is `_non_canonical_tag`, the same function the survivor pick already
    trusts to tell an instrument from a common line, so the vote and the pick cannot disagree
    about what a non-equity line is.
    """
    override, conflicts = _dedup_sectors(
        [(['AFG', 'AFGB', 'AFGE'], 'American Financial Group, Inc.')],
        {'AFG': co.FINANCIAL_SECTOR, 'AFGB': 'Industrials', 'AFGE': 'Industrials'},
        names={'AFG': 'American Financial Group, Inc.',
               'AFGB': 'American Financial Group, Inc. 5.875% Subordinated Debentures',
               'AFGE': 'American Financial Group, Inc. 5.125% Subordinated Debentures'})
    surv = conflicts[0][0]
    assert override[surv] == co.FINANCIAL_SECTOR, (
        'two debenture lines outvoted the equity and demoted an insurer out of its cohort')
    assert override[surv] != co.AMBIGUOUS_SECTOR, (
        'this is not an ambiguity -- the issuer HAS a clear equity tag; routing it to general '
        'as "ambiguous" would be a second wrong answer')
    #  THE DIAGNOSTIC MUST SHOW BOTH SIDES.  A report restricted to the voters would hide that
    #  a bond line disagreed and was not counted, which is the thing a reader needs to see.
    entry = conflicts[0][1]
    assert entry['all_lines'] == {co.FINANCIAL_SECTOR: 1, 'Industrials': 2}
    assert entry['equity_voters'] == {co.FINANCIAL_SECTOR: 1}


def test_a_group_of_ONLY_non_equity_lines_keeps_its_sector_rather_than_losing_it():
    """The degenerate case, ruled rather than left to fall out: with no equity line there is
    nothing to protect, so the old behaviour stands.  Dropping the sector entirely would push
    a real issuer to general for having only preferred lines in the panel."""
    override, _c = _dedup_sectors(
        [(['RE-PA', 'RE-PB'], 'Some REIT Inc.')],
        {'RE-PA': co.REIT_SECTOR, 'RE-PB': co.REIT_SECTOR},
        names={'RE-PA': 'Some REIT Inc. 6% Series A Preferred',
               'RE-PB': 'Some REIT Inc. 5% Series B Preferred'})
    assert set(override.values()) == {co.REIT_SECTOR}


def test_a_PLURALITY_still_wins_so_a_mistagged_sibling_cannot_demote_a_real_REIT():
    """THE PROTECTIVE HALF OF THE OLD RULE, PRESERVED.  The cohort preference existed so a
    baby-bond line mistagged 'Industrials' could not demote a REIT issuer out of its cohort.
    That intent is correct and survives -- through the PLURALITY, which is the evidence the
    intent was really appealing to.  Only the DEAD TIE changed hands."""
    override, conflicts = _dedup_sectors(
        [(['RE', 'RE-PA', 'RE-PB'], 'Some REIT Inc.')],
        {'RE': co.REIT_SECTOR, 'RE-PA': co.REIT_SECTOR, 'RE-PB': 'Industrials'})
    surv = conflicts[0][0]
    assert override[surv] == co.REIT_SECTOR, (
        '2 REIT tags against 1 mistagged Industrials is a plurality, not a tie -- the issuer '
        'must stay in its cohort')
    labels, _r = co.classify([surv], override, pd.DataFrame(), {surv: 'Some REIT Inc.'})
    assert labels[surv] == 'REIT'


def test_an_UNCONFLICTED_issuer_is_untouched():
    """The change must bite ONLY on a conflict.  A group whose members agree keeps its sector
    exactly as before, so the carve is unchanged for the overwhelming majority."""
    override, conflicts = _dedup_sectors(
        [(['M1', 'M1-B'], 'Miner Corp')],
        {'M1': co.MINING_SECTOR, 'M1-B': co.MINING_SECTOR})
    assert not conflicts
    assert set(override.values()) == {co.MINING_SECTOR}
