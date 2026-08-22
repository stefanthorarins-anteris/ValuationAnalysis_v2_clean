"""
price_grid_audit.py -- is the saved price grid still the right grid for TONIGHT's universe?

WHY THIS EXISTS
---------------
`pipeline_analysis.run_price_fetch_stage` decided whether the grading price grid needed
fetching with

    need_main = not os.path.exists(_PRICES_CSV)

a PURE PRESENCE CHECK.  A file called `real_prices.csv` therefore satisfied it for ever.

MEASURED, this module against the run machine's actual grid + its 2025 supplement and the 08-22
CUR6K panel: the grid was written 2026-07-17 (36 days before that run) and carries 10,205
symbols; of 5,819 panel names, 4,221 are priced at at least one anchor (72.5%).  The count is
the less serious half.  SEVEN WHOLE VENUES are unpriceable at EVERY anchor -- `.PA` (569 panel
names), `.KS` (327), `.OL` (224), `.KQ` (159), `.BR` (105), `.AS` (104), `.LS` (33), i.e. 1,421
names -- and `.DE` (680), `.ST` (679) and `.IC` (21) are additionally empty at 2018-12-31.
Eight of the 08-22 top-20 sit on venues the grid is blind to.  Every backtest number the
analysis suite printed rode that frozen grid and nothing said so.

(On the repo-local dev grid the shape is different and worth knowing: coverage is 98.7% but
`.DE`/`.ST`/`.IC` are empty at 2018-12-31, 2019-12-31, 2021-12-31 and 2024-12-31 and `.KS`/`.KQ`
additionally at 2023-12-29 -- venue holidays on those year-ends.  `_merge_supplementary` unions
a holiday-adjacent day into the anchor for 2025 ONLY, so the same calendar problem is unfixed at
every earlier anchor.  The audit surfaces it; generalising that union is a separate, already-
authorised change and is NOT done here.)

WHAT THIS MODULE DOES, AND DELIBERATELY DOES NOT DO
---------------------------------------------------
It DETECTS and REPORTS.  It never fetches, never touches the network, and never changes the
fetch decision: an absent file is still fetched, a present file is still not.  Making staleness
trigger a re-fetch would spend the CEO's money on a schedule nobody approved, and a surprise
paid refetch is as unacceptable as a silent stale grid.  So the third option: say so, loudly,
every run.

WARN OR REFUSE IS THE CEO'S CALL, AND IT IS A SWITCH, NOT A GUESS
-----------------------------------------------------------------
Default is WARN-AND-PROCEED.  Two reasons, both about honesty rather than convenience:
  * The defect is PARTIAL -- 4,221 of 5,819 names ARE priced -- so refusing kills five working
    analysis stages over numbers that are biased, not garbage.  A refusing stage produces
    nothing, which is the same trade the carveOut coverage scope had to make.
  * The remedy costs money the house cannot authorise.  A refusal would therefore convert a
    warning into a standing outage until the CEO paid, i.e. it would make the spend decision
    for him by withholding output.
Set `configdic['price_grid_refuse_when_stale'] = 1` to make a structural finding raise instead.

IT MEASURES THROUGH THE GRADER, NOT THROUGH THE FILE (2026-08-22, reviewer C2)
-----------------------------------------------------------------------------
The first version harvested its own anchor axis out of the CSV's `date_requested` column and
counted rows itself.  That made it capable of DISAGREEING WITH THE THING IT IS AUDITING, and it
did: the local grid carries 9 `date_requested` values including `2025-12-30`, while the grader
works over 8 anchors, because `returns_core.PriceSource._merge_supplementary` deliberately
UNIONS `2025-12-30` into the `2025-12-31` anchor -- precisely for the venues that do not trade
on the 31st.  So the audit reported `venue .T has 3787 panel name(s) but ZERO priced at
anchor(s) 2025-12-31` (and `.KS` 1,228, `.BK` 943, `.DE` 2,100-vs-8) about names the pipeline
prices perfectly well.  A false STALE is not cosmetic here: with `price_grid_refuse_when_stale`
enabled it is a PERMANENT hard refusal on a non-defect.

Every coverage question is therefore answered by `returns_core.PriceSource.price()` over
`returns_core.DEFAULT_ANCHORS` -- the same object, the same anchor list and the same
"unpriced" rule (`None`, `0`, or NaN; see `compute_returns`, returns_core.py:159-163) that the
depth-grid uses.  The audit can no longer be right about a grid the grader reads differently,
or wrong about one it reads fine.

A CONSEQUENCE WORTH STATING: this audits the REAL-PRICE ROUTE, which is what
`pipeline_analysis._build_price_source` feeds every analysis stage today.  If the derived-price
route (`derived_prices.build_price_source`) is ever wired into the pipeline, this module's claim
about what the stages below it compute on goes stale and must be re-pointed.

THRESHOLDS: THERE ARE NONE, ON PURPOSE
--------------------------------------
Every trigger below is a CATEGORICAL ZERO, not a chosen cut.  "Overlap fell below X%" would
need a measured healthy reference and the house does not have one for this grid, so overlap and
file vintage are REPORTED at every run and trigger NOTHING.  What triggers is "this anchor
priced none of tonight's names" and "this venue has no rows anywhere" -- facts that need no
threshold to interpret, and which a scalar overlap percentage would have hidden completely
(the run machine's grid overlaps 72.5% while being structurally blind to SEVEN whole venues).
"""
import os
from datetime import datetime

#  A venue is judged WHOLLY ABSENT only if the panel carries enough names there for
#  "none of them priced" to be a fact about the fetch rather than about a coincidence.  With
#  ten or more names, zero-at-every-anchor cannot be sampling.  Below it the finding is still
#  REPORTED (see `venues`), it just does not by itself declare the grid stale.
VENUE_MIN_PANEL_NAMES = 10

#  Symbols with no exchange suffix (US listings, as FMP returns them).
NO_SUFFIX = '(none)'


def suffix_of(symbol):
    """`'092730.KQ' -> '.KQ'`, `'META' -> '(none)'`.  The venue key used throughout."""
    s = str(symbol)
    if '.' not in s:
        return NO_SUFFIX
    return '.' + s.rsplit('.', 1)[1]


def _is_priced(px):
    """The GRADER's rule for "this name has a usable leg here", not a rule of our own.

    `returns_core.compute_returns` treats a leg as absent when it is `None`, `0`, or NaN
    (returns_core.py:159-163) and otherwise uses it.  Reproduced exactly -- including NOT
    excluding negatives, which are nonsense in a price but which the grader would consume, so
    an audit that filtered them would under-report coverage relative to the numbers actually
    printed.
    """
    if px is None:
        return False
    try:
        f = float(px)
    except (TypeError, ValueError):
        return False
    if f == 0.0 or f != f:      # zero, or NaN
        return False
    return True


def audit_price_grid(prices_csv, panel_symbols, supp_csv=None, anchors=None):
    """Measure the saved grid against tonight's panel, THROUGH the grader's own price source.

    Pure apart from reading the CSVs; returns a dict.  `anchors` defaults to
    returns_core.DEFAULT_ANCHORS -- imported, never restated here, so the audit's anchor space
    cannot drift from the grid the depth-grid and beat-rate stages actually traverse.
    """
    import returns_core as rc

    if not os.path.exists(prices_csv):
        raise FileNotFoundError('no price grid to audit: %r' % (prices_csv,))
    anchors = list(rc.DEFAULT_ANCHORS if anchors is None else anchors)
    #  THE grader's object, built the way pipeline_analysis._build_price_source builds it, so
    #  the supplementary-anchor union (2025-12-30 -> 2025-12-31) is applied here identically.
    ps = rc.PriceSource(prices_csv, anchors=anchors,
                        supp_csv=supp_csv if (supp_csv and os.path.exists(supp_csv)) else None)

    panel = {str(x) for x in panel_symbols
             if x is not None and str(x) and str(x).lower() != 'nan'}

    priced_at = {a: {x for x in panel if _is_priced(ps.price(x, a))} for a in anchors}
    overlap = set().union(*priced_at.values()) if priced_at else set()
    per_anchor = {a: len(priced_at[a]) for a in anchors}

    #  Per venue, per anchor.  This is the check a scalar overlap cannot make.
    panel_by_venue = {}
    for sym in panel:
        panel_by_venue.setdefault(suffix_of(sym), set()).add(sym)

    venues = {}
    for venue, members in sorted(panel_by_venue.items()):
        pa = {a: len(members & priced_at[a]) for a in anchors}
        zero_anchors = [a for a in anchors if pa[a] == 0]
        venues[venue] = {
            'n_panel': len(members),
            'n_priced_any_anchor': len(members & overlap),
            'per_anchor': pa,
            'zero_anchors': zero_anchors,
            'absent_everywhere': len(zero_anchors) == len(anchors) and bool(anchors),
        }

    #  THE BENCHMARK IS ITS OWN COVERAGE QUESTION, and the only one that can still kill a stage
    #  after the carve unblocks.  `PriceSource.benchmark_series` RAISES when URTH is absent from
    #  the grid entirely, and `returns_core.benchmark_return` needs it at the specific anchors a
    #  window spans -- so the beat-rate stage can fail for a reason that has nothing to do with
    #  per_anchor.  URTH is never in the panel (it is an ETF, not a scored name), so the venue
    #  table above is structurally blind to it.  Reported per anchor, as a fact.
    bench_sym = rc.BENCHMARK_SYMBOL
    bench = {a: _is_priced(ps.price(bench_sym, a)) for a in anchors}
    bench_missing = [a for a in anchors if not bench[a]]

    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(prices_csv))
        age_days = (datetime.now() - mtime).days
    except OSError:
        mtime, age_days = None, None

    rep = {
        'path': prices_csv,
        'supp_path': supp_csv if (supp_csv and os.path.exists(supp_csv)) else None,
        'mtime': mtime.isoformat(timespec='seconds') if mtime else None,
        'age_days': age_days,
        'anchors': anchors,
        'n_grid_symbols': len({sym for (sym, _a) in ps._lut}),
        'n_panel': len(panel),
        'n_overlap': len(overlap),
        'overlap_frac': (len(overlap) / len(panel)) if panel else 0.0,
        'per_anchor': per_anchor,
        'venues': venues,
        'benchmark_symbol': bench_sym,
        'benchmark_per_anchor': bench,
        'benchmark_missing_anchors': bench_missing,
        'findings': [],
    }

    #  ---- STRUCTURAL FINDINGS.  Categorical zeros only; see the module docstring. --------
    f = rep['findings']
    if not anchors:
        f.append('the audit was given NO anchors to measure over')
    if panel and not overlap:
        f.append('NOT ONE of the %d panel names is priced at any anchor' % len(panel))
    for a in anchors:
        if panel and per_anchor[a] == 0:
            f.append("anchor %s prices NONE of tonight's %d panel names" % (a, len(panel)))
    for venue, v in venues.items():
        if v['n_panel'] < VENUE_MIN_PANEL_NAMES:
            continue
        if v['absent_everywhere']:
            f.append('venue %s is WHOLLY ABSENT: %d panel name(s) there, 0 priced at any of '
                     'the %d anchors' % (venue, v['n_panel'], len(anchors)))
        elif v['zero_anchors']:
            f.append('venue %s has %d panel name(s) but ZERO priced at anchor(s) %s'
                     % (venue, v['n_panel'], ', '.join(v['zero_anchors'])))
    if anchors and bench_missing == anchors:
        f.append('the benchmark %s is absent from EVERY anchor -- PriceSource.benchmark_series '
                 'will RAISE and the beat-rate/excess stages cannot run at all' % bench_sym)
    elif bench_missing:
        f.append('the benchmark %s is missing at anchor(s) %s -- any window spanning one of '
                 'them has no benchmark leg' % (bench_sym, ', '.join(bench_missing)))

    rep['verdict'] = 'STALE' if f else 'OK'
    return rep


def format_audit(rep):
    """Human-readable block.  Reports the un-thresholdable numbers too, clearly labelled."""
    L = []
    L.append('PRICE-GRID AUDIT  %s' % rep['path'])
    L.append('  written %s (%s day(s) ago)   %d symbol(s) on the grid, %d anchor(s)'
             % (rep['mtime'], rep['age_days'], rep['n_grid_symbols'], len(rep['anchors'])))
    L.append('  measured THROUGH returns_core.PriceSource over DEFAULT_ANCHORS -- the same '
             'object, anchors and unpriced-rule the grading stages use')
    L.append("  tonight's panel: %d name(s); priced somewhere in the grid: %d (%.1f%%)"
             % (rep['n_panel'], rep['n_overlap'], 100.0 * rep['overlap_frac']))
    L.append('  (vintage and overlap % are REPORTED ONLY -- no measured healthy reference '
             'exists for them, so neither triggers anything)')
    L.append('  panel names priced, per anchor:')
    for a in rep['anchors']:
        L.append('    %s  %5d' % (a, rep['per_anchor'][a]))
    L.append('  per venue (panel names / priced at any anchor):')
    for venue, v in sorted(rep['venues'].items(),
                           key=lambda kv: -kv[1]['n_panel']):
        flag = ''
        if v['absent_everywhere']:
            flag = '   <-- WHOLLY ABSENT from the grid'
        elif v['zero_anchors']:
            flag = '   <-- zero at %s' % ', '.join(v['zero_anchors'])
        L.append('    %-8s %5d / %5d%s'
                 % (venue, v['n_panel'], v['n_priced_any_anchor'], flag))
    L.append('  benchmark %s, per anchor (never in the panel, so it needs its own line):'
             % rep['benchmark_symbol'])
    L.append('    ' + '  '.join('%s=%s' % (a, 'y' if rep['benchmark_per_anchor'][a] else 'NO')
                                for a in rep['anchors']))
    L.append('  VERDICT: %s' % rep['verdict'])
    for x in rep['findings']:
        L.append('    - %s' % x)
    return chr(10).join(L)


def banner(rep):
    """The unmissable form, for a STALE verdict."""
    bang = '!' * 78
    lines = ['', bang,
             '!!! PRICE GRID IS STALE FOR THIS UNIVERSE -- BACKTEST NUMBERS ARE AFFECTED !!!',
             '!!!   grid written %s (%s day(s) ago)' % (rep['mtime'], rep['age_days']),
             '!!!   %d of %d panel name(s) priced (%.1f%%)'
             % (rep['n_overlap'], rep['n_panel'], 100.0 * rep['overlap_frac'])]
    for x in rep['findings']:
        lines.append('!!!   * %s' % x)
    lines += [
        '!!! EVERY number the analysis stages below print is computed on this grid, so any',
        '!!! name it cannot price is silently ABSENT from those averages -- and a whole',
        '!!! missing venue biases them by market, not just by count.',
        '!!! NOTHING WAS FETCHED. This is a report, not a remedy: refreshing the grid costs',
        '!!! API calls and is a deliberate human action (baseline_tools/fetch_prices.py).',
        bang, '']
    return chr(10).join(lines)


def run_audit(prices_csv, panel_symbols, supp_csv=None, log=None, refuse_when_stale=False,
              out_streams=None):
    """Audit + emit.  Returns the report dict.

    `refuse_when_stale` raises RuntimeError on a structural finding, for the CEO to switch on
    (`configdic['price_grid_refuse_when_stale']`).  Default False -- see the module docstring
    for why warn-and-proceed is the default and why that is a decision rather than a habit.
    """
    import sys
    rep = audit_price_grid(prices_csv, panel_symbols, supp_csv=supp_csv)
    text = format_audit(rep)
    streams = out_streams if out_streams is not None else (sys.stdout,)
    if rep['verdict'] == 'STALE':
        b = banner(rep) + chr(10) + text
        print(b, file=sys.stderr, flush=True)
        for st in streams:
            print(b, file=st, flush=True)
        if log:
            log('[price-audit] STALE: %s' % '; '.join(rep['findings']))
        if refuse_when_stale:
            raise RuntimeError(
                'price grid STALE and price_grid_refuse_when_stale is set: %s'
                % '; '.join(rep['findings']))
    else:
        for st in streams:
            print(text, file=st, flush=True)
        if log:
            log('[price-audit] OK: %d of %d panel names priced'
                % (rep['n_overlap'], rep['n_panel']))
    return rep
