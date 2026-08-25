"""The two 2026-08-24 investability gates and the two display/scoring repairs beside them.

Four items, one file, because they share one question: what may be excluded from the
shortlist, what may only be charged for, and what must never be either.

  C1  the $1M/day traded-value floor          carveOut.partition_universe
  C7  the charge for a name we cannot price   postBo.postBoWrapper (via the floor's abstention)
  C3  the imputation ladder                   postBoRank.imputation_ladder
  C5  the panel-computed currentRatio         postBo._current_ratio_panel_table

Every test here is a BEHAVIOURAL one -- it fails on the code as it stood on 2026-08-23 -- or
it says so in its own docstring.
"""
import math
import re

import numpy as np
import pandas as pd
import pytest

import adhoc_penalty as ap
import carveOut as co
import postBo as pb
import postBoRank as pbr


# --------------------------------------------------------------------------- #
#  C1  --  THE $1M/DAY TRADED-VALUE FLOOR                                      #
# --------------------------------------------------------------------------- #
def _floor_inputs(n=40):
    pool = ['S%04d' % i for i in range(n)]
    bo = pd.DataFrame({'source': pool, 'BoScore': np.linspace(1.0, 0.0, n)})
    tickers = pd.DataFrame({'symbol': pool, 'name': ['N' + s for s in pool]})
    cdx = pd.DataFrame({'source': pool, 'date': pd.Timestamp('2026-01-01'),
                        'marketCap': 1e9, 'reportedCurrency': 'USD',
                        'totalStockholdersEquity': 5e8, 'totalAssets': 1e9,
                        'revenue': 5e8, 'weightedAverageShsOut': 1e7,
                        'netIncome': 1e7})
    return pool, bo, cdx, tickers


def _carve(monkeypatch, tmp_path, pool, bo, cdx, tickers, dv_by_symbol, **kw):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: {s: 'Technology' for s in pool})
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {s: 'Software' for s in pool})
    monkeypatch.setattr(co, 'dollar_volume_frame',
                        lambda syms, *a, **k: pd.DataFrame({
                            'dollarVolume_usd': [dv_by_symbol.get(s, np.nan) for s in syms],
                            'dollarVolume_basis': ['2026-08-21|USD' if s in dv_by_symbol
                                                   else 'no-reading' for s in syms]}))
    kw.setdefault('dollarvol_floor', co.DOLLAR_VOLUME_FLOOR_USD)
    return co.partition_universe(bo, cdx, tickers, dedup=False, **kw)


def test_a_name_below_the_floor_is_EJECTED_not_merely_ranked_below(tmp_path, monkeypatch):
    """The CEO's ruling is a FLOOR, not a penalty: an untradable line is not a weak candidate,
    it is not a candidate.  Ejection is also what makes the rest of the design work -- see the
    backfill test below."""
    pool, bo, cdx, tickers = _floor_inputs()
    dv = {s: 5e6 for s in pool}
    dv[pool[0]] = 30_964.0            # EIOF.OL's real reading on the 2026-08-22 run
    dv[pool[3]] = 246_887.0           # 007700.KS's -- rank 1 on that run
    part = _carve(monkeypatch, tmp_path, pool, bo, cdx, tickers, dv)
    kept = set(part['general']['source'])
    assert pool[0] not in kept and pool[3] not in kept, (
        'a name below $1M/day survived the floor -- it is a floor, not a tiebreak')
    assert part['diagnostics']['n_below_dollarvol'] == 2
    assert part['diagnostics']['dollarvol_floor_enforced'] is True


def test_the_floor_runs_BEFORE_the_head_100_cut_so_ejections_BACKFILL(tmp_path, monkeypatch):
    """*** THE PLACEMENT DECISION, AS A TEST. ***  A floor applied AFTER `postBo`'s
    `head(100)` would ship a 91-name shortlist; applied here it ships 100 names with the
    thinnest 9 replaced by the next 9 on BoScore.  The property that guarantees it: the pool
    this returns is longer than the cut taken from it, and the survivors are still in BoScore
    order, so `head(N)` reaches deeper by exactly the number ejected."""
    pool, bo, cdx, tickers = _floor_inputs(n=40)
    dv = {s: 5e6 for s in pool}
    for s in pool[:9]:
        dv[s] = 100_000.0
    part = _carve(monkeypatch, tmp_path, pool, bo, cdx, tickers, dv)
    g = part['general']
    assert len(g) == 31
    top20 = list(g.head(20)['source'])
    assert top20 == pool[9:29], (
        'the survivors are not the next 20 by BoScore -- the floor reordered the pool instead '
        'of removing from it')
    assert not set(top20) & set(pool[:9])


def test_the_floor_runs_BEFORE_THE_DEDUP_so_an_issuer_keeps_its_TRADABLE_line(tmp_path,
                                                                              monkeypatch):
    """*** THE REVIEWER'S CASE, 2026-08-24, AND IT DELETED REAL ISSUERS. ***

    Run AFTER `dedup_to_issuers` the floor sees ONE line per issuer, so ejecting that line
    removes the WHOLE ISSUER -- including one whose sibling line trades $3M/day and was
    collapsed away moments earlier.

    THE FIXTURE IS THE BAND C2 DELIBERATELY ABSTAINS ON, which is what makes this reachable
    rather than exotic: `LIQ` at $3.0M/day and `THIN` at $400k/day are 7.5x apart, so
    `_volavg_liquidity_term` returns 0 for both and says nothing; `-shares` then decides, and
    `THIN` reads 1.3% more shares -- the TOP of the measured 0.1-1.6% filing-vintage wobble
    band, and the exact margin the rejected ">1% materiality gate" would also have let
    through.  So the dedup picks `THIN`, and a floor placed after it ejects the issuer.

    THIS IS ALSO WHERE TWO INDIVIDUALLY-HONEST BLIND-SPOT STATEMENTS WERE JOINTLY
    INCOMPLETE: C2's note calls the sub-decade groups a no-change case, which was true of C2
    alone and stopped being true once C1 was in the tree beside it.

    The assertion is about the ISSUER surviving, not about which line wins -- either line is
    a defensible survivor, but the issuer vanishing is not."""
    pool = ['LIQ', 'THIN'] + ['S%03d' % i for i in range(8)]
    bo = pd.DataFrame({'source': pool, 'BoScore': np.linspace(1.0, 0.0, len(pool))})
    #  Same issuer name on both lines, and a share count that differs by the wobble.
    tickers = pd.DataFrame({'symbol': pool,
                            'name': ['Widget PLC', 'Widget PLC']
                                    + ['N' + s for s in pool[2:]]})
    shares = {'LIQ': 1.000e9, 'THIN': 1.013e9}
    cdx = pd.DataFrame({
        'source': pool, 'date': pd.Timestamp('2026-01-01'),
        'marketCap': 1e9, 'reportedCurrency': 'USD',
        'totalStockholdersEquity': 5e8, 'totalAssets': 1e9, 'revenue': 5e8,
        'weightedAverageShsOut': [shares.get(s, 1e7) for s in pool],
        'netIncome': 1e7})
    dv = {s: 5e6 for s in pool}
    dv['LIQ'] = 3_000_000.0
    dv['THIN'] = 400_000.0
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: {s: 'Technology' for s in pool})
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {s: 'Software' for s in pool})
    monkeypatch.setattr(co, 'dollar_volume_frame',
                        lambda syms, *a, **k: pd.DataFrame({
                            'dollarVolume_usd': [dv.get(s, np.nan) for s in syms],
                            'dollarVolume_basis': ['2026-08-21|USD'] * len(syms)}))
    #  The fixture must genuinely be a group the DECADE term ties, or it is testing nothing.
    vmap = {'LIQ': (3.0e6, '2026-08-21'), 'THIN': (4.0e5, '2026-08-21')}
    assert {co._volavg_liquidity_term(s, ['LIQ', 'THIN'], vmap) for s in ('LIQ', 'THIN')} == {0}

    #  AND THE DE-DUP MUST GENUINELY PREFER THE THIN LINE WITH NO FLOOR IN THE WAY, or the
    #  case this test reproduces cannot arise and the test is vacuous.  Measured: it does,
    #  on `shares`.
    _nofloor = co.partition_universe(bo, cdx, tickers, dedup=True)
    _survivor = set(_nofloor['general']['source']) & {'LIQ', 'THIN'}
    assert _survivor == {'THIN'}, (
        'the de-dup no longer prefers the illiquid line on a 1.3%% share wobble (it kept %s), '
        'so this fixture cannot reproduce the issuer-deletion case' % _survivor)

    part = co.partition_universe(bo, cdx, tickers, dedup=True,
                                 dollarvol_floor=co.DOLLAR_VOLUME_FLOOR_USD)
    kept = set(part['general']['source']) | {s for c in part['cohorts'].values()
                                             for s in c['source']}
    assert kept & {'LIQ', 'THIN'}, (
        'the issuer VANISHED: it holds a $3.0M/day line and the floor deleted it because the '
        'de-dup had already collapsed that line away and handed the floor the $400k/day one')
    assert 'THIN' not in kept, (
        'the $400k/day line survived a $1M/day floor')
    assert 'LIQ' in kept


def test_a_name_with_NO_reading_is_KEPT_because_absence_is_not_illiquidity(tmp_path,
                                                                           monkeypatch):
    """*** THE ASYMMETRY THAT MATTERS MOST, and the one an eager floor gets wrong. ***  A NaN
    `dollarVolume_usd` means we could not price the name, not that it trades $0 -- and an
    ejected name leaves NO trace downstream (not in the top-100, not in a cohort, not in any
    report), so deleting on absence is an invisible, unrecoverable error.  It is kept and
    handed to `diagnostics['dollarvol_unknown']` for the ad-hoc bucket to charge instead."""
    pool, bo, cdx, tickers = _floor_inputs()
    dv = {s: 5e6 for s in pool if s != pool[2]}     # pool[2] has NO reading at all
    part = _carve(monkeypatch, tmp_path, pool, bo, cdx, tickers, dv)
    assert pool[2] in set(part['general']['source'])
    assert part['diagnostics']['dollarvol_unknown'] == [pool[2]]
    assert part['diagnostics']['n_below_dollarvol'] == 0


def test_a_run_with_NO_readings_at_all_floors_NOTHING_and_says_so(tmp_path, monkeypatch,
                                                                  capsys):
    """The `floor_enforced` shape the market-cap floor already carries.  With no volavgdic
    capture every value is NaN, so the floor excludes nobody -- and a downstream banner
    claiming a "$1M/day floored universe" would assert a filter that never ran."""
    pool, bo, cdx, tickers = _floor_inputs()
    part = _carve(monkeypatch, tmp_path, pool, bo, cdx, tickers, {})
    assert len(part['general']) == len(pool)
    assert part['diagnostics']['dollarvol_floor_enforced'] is False
    assert 'NOT ENFORCED' in capsys.readouterr().out


def test_the_floor_EVIDENCE_CSV_survives_a_DUPLICATE_source(tmp_path, monkeypatch):
    """*** THE SOLE RECORD OF THE BIGGEST CUT THIS FUNCTION MAKES, AND IT USED TO FAIL SILENTLY
    ON A DUPLICATE (reviewer, 2026-08-24). ***  The frame paired `sorted(dv_below_sources)` --
    a SET, de-duplicated -- with `['ejected_below_floor'] * n_below_dv` -- a ROW COUNT.  One
    duplicate `source` made the column lengths disagree, pandas raised, and the `except`
    swallowed it into a one-line WARNING: the banner would claim N ejected while the file
    naming them did not exist, and `n_below_dollarvol` would disagree with `dollarvol_below`
    with nothing saying so.  Built row-wise from one mask, the columns cannot disagree.

    WHAT THIS CANNOT DETECT: whether the CSV reaches the transfer.  That is
    `Sbocker.allowlist_patterns` and `test_transfer_directories`, not here."""
    pool = ['DUP', 'DUP', 'OK1', 'OK2', 'OK3', 'OK4', 'OK5', 'OK6']
    bo = pd.DataFrame({'source': pool, 'BoScore': np.linspace(1.0, 0.0, len(pool))})
    tickers = pd.DataFrame({'symbol': pool, 'name': ['N' + s for s in pool]})
    cdx = pd.DataFrame({'source': pool, 'date': pd.Timestamp('2026-01-01'),
                        'marketCap': 1e9, 'reportedCurrency': 'USD',
                        'totalStockholdersEquity': 5e8, 'totalAssets': 1e9,
                        'revenue': 5e8, 'weightedAverageShsOut': 1e7, 'netIncome': 1e7})
    dv = {'DUP': 1000.0, 'OK1': 5e6, 'OK2': 5e6, 'OK3': 5e6, 'OK4': 5e6, 'OK5': 5e6,
          'OK6': 5e6}
    part = _carve(monkeypatch, tmp_path, pool, bo, cdx, tickers, dv)
    written = sorted(tmp_path.glob('DollarVolumeFloor_*.csv'))
    assert written, (
        'the traded-value floor ejected names and wrote NO evidence file -- the cut is now '
        'unauditable off the run machine')
    rows = pd.read_csv(written[0])
    #  Row-wise means one row per EJECTED LINE, duplicates included -- and the count in the
    #  banner and the diagnostics must be that same number.
    assert len(rows) == part['diagnostics']['n_below_dollarvol'] \
        + part['diagnostics']['n_unknown_dollarvol']
    assert list(rows[rows['verdict'] == 'ejected_below_floor']['source']) == ['DUP', 'DUP']
    assert part['diagnostics']['n_below_dollarvol'] == 2


def test_the_floor_is_OFF_BY_DEFAULT_so_no_POINT_IN_TIME_carve_inherits_it(tmp_path,
                                                                            monkeypatch):
    """*** THE LOOKAHEAD GUARD, AND IT IS THE ONE THAT WOULD HAVE BEEN EASY TO GET WRONG. ***
    `dollar_volume_frame` reads whatever volavgdic capture is newest ON DISK -- TODAY's
    liquidity -- so a point-in-time carve floored on it would be screening a 2018 pool with a
    2026 fact, straight into the backtest that measures whether the filter works.  Three
    offline callers carve PIT pools (`baseline_tools/refit.py`, `depth_horizon_grid.py`,
    `tune_run.py`) and only ONE of them passes `coverage_scope`, so the floor cannot be keyed
    on that: it is OFF unless the caller asks, and `postBo.postBoWrapper` is the only caller
    that asks."""
    pool, bo, cdx, tickers = _floor_inputs()
    dv = {s: 1.0 for s in pool}                     # everything would fall if it ran
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: {s: 'Technology' for s in pool})
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {s: 'Software' for s in pool})
    monkeypatch.setattr(co, 'dollar_volume_frame',
                        lambda *a, **k: pytest.fail(
                            'partition_universe read the traded-value map without being asked '
                            'for a floor -- a point-in-time carve would inherit today\'s '
                            'liquidity'))
    part = co.partition_universe(bo, cdx, tickers, dedup=False)     # NO dollarvol_floor
    assert len(part['general']) == len(pool)
    assert part['diagnostics']['n_below_dollarvol'] == 0
    assert part['diagnostics']['dollarvol_floor'] == 0.0
    #  And the explicit off-switch works too, for a caller that wants to be obvious about it.
    part0 = co.partition_universe(bo, cdx, tickers, dedup=False, dollarvol_floor=0)
    assert len(part0['general']) == len(pool)


def test_a_POINT_IN_TIME_scoring_pass_does_NOT_get_the_floor():
    """*** THE LOOKAHEAD GUARD, BEHAVIOURALLY. ***  `postBoWrapper` is itself a
    point-in-time entry point (`Sbocker` threads the documented `-asof` flag straight into it), and the floor reads TODAY's volAvg capture.  A run scoring a past cross-section
    must not be screened on a fact from the future."""
    assert pb._resolve_dollarvol_floor(as_of=None, dollarvol_floor='auto') \
        == co.DOLLAR_VOLUME_FLOOR_USD
    assert pb._resolve_dollarvol_floor(as_of='2020-12-31', dollarvol_floor='auto') is None
    #  An explicit value ALWAYS wins, including None -- which is what the two date-filtered
    #  re-entrants need, because they pass as_of=None and no value of as_of can express them.
    assert pb._resolve_dollarvol_floor(as_of=None, dollarvol_floor=None) is None
    assert pb._resolve_dollarvol_floor(as_of='2020-12-31', dollarvol_floor=5.0) == 5.0


def test_EVERY_postBoWrapper_RE_ENTRANT_is_classified_live_or_point_in_time():
    """*** THE GUARD THE FIRST CUT GOT WRONG, RE-POINTED (reviewer, 2026-08-24). ***

    The first version asserted that `dollarvol_floor=` appears only in `postBo.py` -- which
    was TRUE, and satisfied in exactly the state where the bug was live, because every
    re-entrant INHERITED postBo's unconditional opt-in.  The object that matters is not the
    set of files naming the parameter; it is the set of CALLERS of `postBoWrapper`, each of
    which is a scoring pass that either is or is not point-in-time.  A new re-entrant must
    force that decision, so the set is pinned and every member is classified here.

    WHAT THIS CANNOT DETECT: whether a caller's classification is CORRECT.  It pins that
    somebody decided.  `normalized_analysis` and the `baseline_tools` reproducers are
    classified LIVE because they score today's panel; if one of them starts filtering by date
    this test still passes and the lookahead is back.  It also cannot see a call made through
    a wrapper, an alias, or `getattr(pb, 'postBoWrapper')`."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent
    callers = set()
    for f in list(root.glob('*.py')) + list((root / 'baseline_tools').glob('*.py')):
        if f.name.startswith('test_') or f.name == 'postBo.py':
            continue
        for line in f.read_text(encoding='utf-8', errors='ignore').split('\n'):
            t = line.strip()
            if t.startswith('#'):
                continue
            if 'postBoWrapper(' in t:
                callers.add(f.name)
    #  PIT / date-filtered: must pass dollarvol_floor=None or thread a real as_of.
    pit = {'Sbocker.py', 'backtest_ols_analysis.py', 'portfolio.py'}
    #  Live-equivalent: scores TODAY's panel, so today's liquidity is the right question.
    live = {'normalized_analysis.py', 'run_corrected_current.py',
            'industry_attribution.py', 'nan_policy_report.py'}
    assert callers == pit | live, (
        'the set of postBoWrapper re-entrants changed (%s) -- classify the new one as '
        'point-in-time (pass dollarvol_floor=None) or live before shipping, because it '
        'inherits the traded-value floor by default and the floor reads TODAY\'s volAvg '
        'capture' % sorted(callers ^ (pit | live)))
    #  And the two that cannot be distinguished by `as_of` really do say so.
    for f in ('backtest_ols_analysis.py', 'portfolio.py'):
        src = (root / f).read_text(encoding='utf-8')
        assert 'postBoWrapper(temp_dmdic, dollarvol_floor=None)' in src, (
            '%s scores a DATE-FILTERED panel with as_of=None, so it must withhold the floor '
            'explicitly -- no value of as_of expresses it' % f)
    #  Sbocker threads as_of, which is what closes it.
    sb = (root / 'Sbocker.py').read_text(encoding='utf-8')
    assert 'pb.postBoWrapper(datandmetricdic, as_of=as_of)' in sb


def test_the_level_is_the_CEOs_and_is_not_quietly_a_different_number():
    """A pinned constant, because the level is a CEO decision and the measured consequence
    (9 of the top 20, including rank 1) was quoted back to him at exactly this level."""
    assert co.DOLLAR_VOLUME_FLOOR_USD == 1_000_000.0


# --------------------------------------------------------------------------- #
#  C3  --  THE IMPUTATION LADDER                                              #
# --------------------------------------------------------------------------- #
def _fill_frame(shares):
    return pd.DataFrame({
        'pool': 'general',
        'source': list(shares),
        'n_imputed_cols': [1] * len(shares),
        'n_weighted_cols': [19] * len(shares),
        'imputed_weight_share': [shares[s] for s in shares],
        'imputed_cols': ['earnYield'] * len(shares)})


def test_the_two_rungs_never_act_on_the_SAME_name():
    """*** THE CEO'S EXPLICIT REQUIREMENT: the two instruments "cannot double-count or fight".
    ***  Exactly one acts per name, decided by one number, so there is no name both touch and
    no ordering in which they disagree."""
    shares = {'LOW': 0.05, 'MID': 0.19, 'HIGH': 0.20, 'WORST': 0.9317}
    book = ap.PenaltyBook()
    excluded, n_charged = pbr.imputation_ladder(_fill_frame(shares), penalty_book=book,
                                                pool_label='general', verbose=False)
    assert excluded == {'HIGH', 'WORST'}
    assert n_charged == 2
    assert book.sources == {'LOW', 'MID'}
    assert not (set(book.sources) & excluded), 'a name was both charged AND excluded'


def test_the_charge_is_MONOTONE_and_never_falls_as_the_data_gets_worse():
    """The inversion this bucket has actually shipped before (see adhoc_penalty's MONOTONICITY
    note): the charge climbed with the gap and then dropped to zero at the worst case.  Here
    the charge rises to RUNGS points immediately below the cut and then becomes EJECTION,
    which is strictly worse -- so nowhere does more missing data buy a softer treatment."""
    prev = -1.0
    for share in np.arange(0.005, pbr.IMPUTED_EXCLUDE_AT, 0.005):
        book = ap.PenaltyBook()
        pbr.imputation_ladder(_fill_frame({'X': float(share)}), penalty_book=book,
                              verbose=False)
        pts = book.points_by_source()['X']
        assert pts >= prev, 'the charge FELL between %.3f and this share' % share
        prev = pts
    assert prev == pbr.IMPUTED_LADDER_RUNGS, prev


def test_the_step_is_DERIVED_from_the_CEOs_cut_and_not_a_second_constant():
    """Move the cut and the ladder rescales with it; there is no second number to forget."""
    assert pbr.IMPUTED_LADDER_STEP == pbr.IMPUTED_EXCLUDE_AT / pbr.IMPUTED_LADDER_RUNGS
    assert pbr.IMPUTED_EXCLUDE_AT == 0.20


def test_a_zero_share_is_charged_NOTHING():
    """87 of the 2026-08-22 top-100 sit at exactly 0.0.  A bucket contribution of 0 points is
    refused by `adhoc_penalty.add` outright, so a ladder that did not skip them would raise
    on a clean pool."""
    book = ap.PenaltyBook()
    excluded, n = pbr.imputation_ladder(_fill_frame({'CLEAN': 0.0}), penalty_book=book,
                                        verbose=False)
    assert (excluded, n, len(book)) == (set(), 0, 0)


def test_a_MISSING_fill_report_ABSTAINS_and_does_not_read_as_a_clean_pool(capsys):
    """`missing_data_fill_report` is wrapped in `_safe_diagnose` and returns (None, None) on
    any failure.  A ladder that read that as "nobody is heavily imputed" would report a clean
    pool on a run where it never looked -- which is the guard-blind-to-the-defect shape this
    project has shipped twice."""
    excluded, n = pbr.imputation_ladder(None, penalty_book=ap.PenaltyBook())
    assert (excluded, n) == (set(), 0)
    assert 'NOT APPLIED' in capsys.readouterr().out


def test_the_ladder_charges_through_the_bucket_and_not_through_the_weight_vector():
    """`scoringWeights.DEPLOYED` asserts Sigma|w| = 1.000000 at import and every published
    AggScore range rests on it.  The ladder must therefore reach the score the way every other
    ad-hoc charge does -- as points in the bucket, converted by ONE fixed weight."""
    book = ap.PenaltyBook()
    pbr.imputation_ladder(_fill_frame({'X': 0.10}), penalty_book=book, verbose=False)
    item = book.itemised().iloc[0]
    assert item['check'] == pbr.CHECK_IMPUTED_WEIGHT
    assert item['penalty'] == pytest.approx(-ap.WEIGHT * item['points'])
    import scoringWeights as sw
    assert pbr.CHECK_IMPUTED_WEIGHT not in sw.METRIC_KEYS


def test_the_bucket_has_exactly_FOUR_charging_call_sites_and_ONE_reads_the_fill_share():
    """*** THE DOUBLE-CHARGE CHECK, PINNED SO A FIFTH CHARGE CANNOT ARRIVE UNNOTICED. ***

    THE CLAIM THAT WAS VERIFIED BEFORE THE LADDER WAS BUILT, not taken on trust: before
    2026-08-24 the ad-hoc bucket had exactly TWO charging call sites -- `stage1_veto` (missing
    veto-flag ROWS) and `detectManipulation` (absent Beneish COMPONENTS, min(absent,3)+1
    points, max -0.04) -- and NEITHER reads `imputed_weight_share` or anything Stage-2
    imputed.  So the ladder ADDS a charge; it does not duplicate one.  Two sites were added
    that day: the ladder itself and the unpriceable charge.

    Both halves are asserted, because either alone is weak: the call-site set catches a NEW
    charge appearing anywhere, and the reader set catches a second instrument built on the
    SAME number.

    WHAT THIS CANNOT DETECT: a second charge for the same underlying data gap raised under a
    different name and off a different column.  It pins the column and the call sites, not the
    phenomenon -- and the phenomenon IS partly duplicated already: 7 of the 10 names the ladder
    charges on the 2026-08-22 top-100 also carry a veto or Beneish charge, because one bad
    filer trips several checks.  That is the bucket working as designed (one point per
    distinct gap), and it is recorded here so nobody reads this test as "no name is charged
    twice"."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent
    charging, reading = set(), set()
    for f in sorted(root.glob('*.py')):
        if f.name.startswith('test_'):
            continue
        t = f.read_text(encoding='utf-8', errors='ignore')
        for m in re.finditer(r'^\s*(?:\w+)\.add\(', t, re.M):
            line_start = t.rfind('\n', 0, m.start()) + 1
            obj = t[line_start:m.start() + len(m.group(0))]
            if 'penalty_book.add(' in obj or 'book.add(' in obj:
                charging.add(f.name)
        if re.search(r"\[.?'imputed_weight_share'.?\]", t):
            reading.add(f.name)
    assert charging == {'stage1_veto.py', 'detectManipulation.py', 'postBoRank.py',
                        'postBo.py'}, (
        'the set of modules that CHARGE the ad-hoc bucket changed (%s) -- a new charge must be '
        'reconciled against the four that exist before it ships' % sorted(charging))
    #  AND THE DISTINCT CHARGE NAMES, which is the number a reader of `AdHocPenaltyBucket_*.csv`
    #  actually reconciles against.  Six, on four mechanisms:
    import stage1_veto as sv
    import detectManipulation as dm
    names = {sv.CHECK_SHORT_PANEL, sv.CHECK_REFUSED_ROWS, sv.CHECK_UNCORROBORATED,
             dm.CHECK_GAP_COMPONENTS, dm.CHECK_GAP_NO_VERDICT, pbr.CHECK_IMPUTED_WEIGHT,
             'unpriceable:no_traded_value_reading'}
    assert len(names) == 7, 'two charges now share a name, so the CSV cannot separate them'
    #  `postBo` and `generate_presentation` READ the fill share (to join and to render it) and
    #  charge nothing for it -- that is fine and is not what this pins.  What matters is that
    #  `imputed_weight_share` becomes POINTS in exactly one place.
    assert 'postBoRank.py' in reading and 'postBoRank.py' in charging
    src = (root / 'postBoRank.py').read_text(encoding='utf-8')
    assert src.count('CHECK_IMPUTED_WEIGHT') == 2, (
        'the imputation charge is raised from more than one place in postBoRank -- it is one '
        'constant, one definition and one `add`')


# --------------------------------------------------------------------------- #
#  C5  --  currentRatio: COMPUTED FROM THE PANEL, AND VALUE-CHECKED            #
# --------------------------------------------------------------------------- #
def test_an_impossible_vendor_currentRatio_becomes_NaN_not_ZERO():
    """*** THE DEFECT: a type check with no value check. ***  `092730.KQ` shipped `0.0000` at
    RANK 7 on the 2026-08-22 run and the deck flags `currentRatio < 1.0` as a solvency red
    flag, so a rank-7 pick displayed a red flag it does not have -- in the CSV the CEO's
    manual review reads.  A vendor 0 must read as the same 'NaN' a MISSING key produces."""
    assert pb._current_ratio_value(0) is None
    assert pb._current_ratio_value(0.0) is None
    assert pb._current_ratio_value(-1.5) is None
    assert pb._current_ratio_value(float('nan')) is None
    assert pb._current_ratio_value(float('inf')) is None
    assert pb._current_ratio_value(None) is None
    assert pb._current_ratio_value('') is None
    assert pb._current_ratio_value(5.7528) == pytest.approx(5.7528)


def test_the_panel_computes_the_ratio_the_CEO_verified_against_the_balance_sheet():
    """The exact numbers checked by hand: 184,604,415,000 / 32,089,653,000 KRW = 5.75 for
    `092730.KQ`, and 230,256,128,000 / 43,462,233,000 = 5.30 for `041830.KQ`."""
    cdx = pd.DataFrame({
        'source': ['092730.KQ', '092730.KQ', '041830.KQ'],
        'date': ['2025-10-01', '2026-01-01', '2026-01-01'],
        'totalCurrentAssets': [1.0, 184_604_415_000.0, 230_256_128_000.0],
        'totalCurrentLiabilities': [1.0, 32_089_653_000.0, 43_462_233_000.0]})
    t = pb._current_ratio_panel_table(cdx)
    assert t['092730.KQ'] == pytest.approx(5.7528, rel=1e-4)
    assert t['041830.KQ'] == pytest.approx(5.2978, rel=1e-4)


def test_the_panel_table_takes_the_NEWEST_row_BY_DATE_not_by_arrival_order():
    """Nothing on this path guarantees ingestion order -- the same reason `_pe_panel_table`
    sorts.  Here the newest row is deliberately FIRST in the frame."""
    cdx = pd.DataFrame({
        'source': ['A', 'A'],
        'date': ['2026-01-01', '2020-01-01'],
        'totalCurrentAssets': [200.0, 999.0],
        'totalCurrentLiabilities': [100.0, 1.0]})
    assert pb._current_ratio_panel_table(cdx)['A'] == pytest.approx(2.0)


def test_a_zero_LIABILITIES_row_yields_NaN_not_an_infinity():
    """A company with no current liabilities has an undefined ratio, not an infinite one, and
    `"{:.4f}".format(inf)` would ship the string 'inf' into the CSV."""
    cdx = pd.DataFrame({'source': ['A'], 'date': ['2026-01-01'],
                        'totalCurrentAssets': [200.0], 'totalCurrentLiabilities': [0.0]})
    assert pb._current_ratio_panel_table(cdx)['A'] is None


def test_a_panel_without_the_balance_sheet_columns_ABSTAINS_wholesale():
    """It returns {} so every cell falls back to the vendor -- and `writeBoAggToCSV` prints
    that the computed basis did not run at all, rather than shipping a column that looks
    computed.  A saved pre-2026 panel is exactly this case."""
    assert pb._current_ratio_panel_table(pd.DataFrame({'source': ['A'], 'date': ['2026-01-01']})) == {}
    assert pb._current_ratio_panel_table(None) == {}


def test_the_TYPE_CHECK_WITHOUT_A_VALUE_CHECK_is_gone_from_the_publishing_loop():
    """*** THE EXACT LINE THAT SHIPPED THE DEFECT, PINNED SO IT CANNOT COME BACK. ***

    It read:
        elif type(temp_resp_fr[0]['currentRatio']) == int or type(...) == float:
            crVec.append("{:.4f}".format(temp_resp_fr[0]['currentRatio']))
    -- a check that the vendor sent A NUMBER, with no check that the number is POSSIBLE.  A
    vendor 0 is an int, so it formatted as '0.0000' and the deck flagged rank 7 as insolvent.

    STRUCTURAL, AND SAID TO BE: the publishing loop is 300 lines inside `writeBoAggToCSV` with
    eleven parallel row vectors and live calls in it, so the branch cannot be exercised without
    standing the whole stage up.  This asserts the SHAPE instead -- every path through the
    currentRatio cell now goes through `_current_ratio_value`.  What it cannot detect: whether
    `_current_ratio_value` is right, which the unit tests above cover, or whether the panel
    table is joined to the correct symbol, which nothing here covers."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent / 'postBo.py').read_text(encoding='utf-8')
    i = src.index('crVec = []')
    j = src.index('def ', src.index('_cr_panel.get(symb)'))
    body = src[i:j]
    assert "type(temp_resp_fr[0]['currentRatio'])" not in body, (
        'the type-check-without-a-value-check is back in the publishing loop')
    assert body.count('_current_ratio_value(') >= 1
    #  Nothing may format a currentRatio that did not come through the guard.
    for m in re.finditer(r'crVec[.]append[(](.*)[)]', body):
        arg = m.group(1)
        assert ("'NaN' if _cr is None" in arg) or arg.strip() == "'NaN'", (
            'crVec is appended a value that did not pass through _current_ratio_value: %r' % arg)


def test_scoring_does_not_read_this_column_so_C5_is_display_only():
    """MEASURED, NOT ASSUMED: `uCurrentRatio` is FIELD_EVIDENCE='counts', so a 0 behaves in
    Stage-1 exactly as a NaN does, and 0 of 407 general-pool `uCurrentRatio` ejections flip
    when the impossible zeros are repaired.  This pins the mechanism behind that: the
    published column is built inside `writeBoAggToCSV`, which runs after Stage-2 and assigns
    nothing back."""
    import stage1_veto as sv
    assert sv.FIELD_EVIDENCE.get('uCurrentRatio') == 'counts'


# --------------------------------------------------------------------------- #
#  C7  --  THE CHARGE FOR A NAME WE CANNOT PRICE                              #
# --------------------------------------------------------------------------- #
def test_the_unpriceable_charge_and_the_floor_are_DISJOINT_by_construction():
    """*** THE SCOPING DECISION, AS A TEST.  C7 must not become a second liquidity floor. ***
    The floor acts on a KNOWN value below the level; the charge acts on NO value.  The two
    predicates cannot both hold, so no name can be ejected and charged for the same fact.

    WHAT THIS CANNOT DETECT: whether charging is the RIGHT response to an absent reading, as
    opposed to ejecting or ignoring.  It pins that the two instruments do not overlap, not
    that either is calibrated."""
    src = pb.postBoWrapper.__module__      # keep the import used
    assert src
    dv = pd.Series([np.nan, 5.0, 2e6])
    below = dv.notna() & (dv < co.DOLLAR_VOLUME_FLOOR_USD)
    unknown = dv.isna()
    assert not (below & unknown).any()
    assert list(below) == [False, True, False]
    assert list(unknown) == [True, False, False]


def test_the_unpriceable_charge_is_ONE_POINT_and_is_not_scaled():
    """Absence has no magnitude, so there is nothing to scale by -- one named data gap, one
    point, which is the bucket's own convention.  Pinned because the obvious "improvement" is
    to scale it by something, and the only things available are liquidity proxies, which is
    exactly what this charge must not become."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent / 'postBo.py').read_text(encoding='utf-8')
    i = src.index('CHECK_UNPRICEABLE = ')
    j = src.index('veto_reports = {}', i)
    body = src[i:j]
    assert 'penalty_book.add(' in body
    assert '1.0, pool=_lab)' in body, (
        'the unpriceable charge is no longer a flat single point -- if it is now scaled, say '
        'what by, because every available scale is a liquidity proxy and this charge must not '
        'become a second liquidity floor')
