"""
pick_log.py  --  prospective, append-only pick log (forward, un-gameable track record)

Every automatic Sbocker run appends the day's GENERAL top-N shortlist plus each of the
five cohort side-lists (REIT / Mining / FIN1 / FIN2 / FIN3) as NEW, immutable rows to a
long/tidy `pick_log.csv` in the repo root -- ONE ROW PER (run, list, stock).

WHY THIS EXISTS: every value number the filter reports today is BACKTEST (reconstructed
history, tunable until it looks good). This log stamps each run's picks with the run date
BEFORE any outcome exists, so it cannot be gamed after the fact. It is the only
survivorship-free way to accrue a FORWARD record of what the filter actually picked, to be
graded LATER (months/years out) against a benchmark. The grader is a SEPARATE, deferred
piece; this module is only the WRITER.

APPEND-ONLY IS THE CRUX. The writer NEVER reads-modifies-rewrites existing rows. If the log
exists it appends (no header); if not it creates it with a header. Re-running the same
`as_of` does NOT overwrite -- it appends a new block with a fresh `logged_at`, so history is
complete and duplicate re-runs stay VISIBLE, never lost. (An outcome grader later reads this
log; the writer's sole contract is that no prior byte is ever mutated.)

SCHEMA DRIFT SELF-HEALS BY MOVING, NEVER BY EDITING (2026-08-22). If PICK_LOG_COLUMNS no
longer matches the header on disk, the writer MOVES the old file into `_quarantine/` under a
dated name -- bytes untouched, no row rewritten, no value backfilled -- announces the split
loudly on both streams, and starts a fresh log with the current header. It used to REFUSE,
which on the CEO's run machine meant zero forward picks recorded on every run (08-20, 08-22)
because the remedy was a manual file move only a human at that machine could perform. A
grader must therefore read `_quarantine/pick_log_*.csv` alongside `pick_log.csv`; the
history is complete across the pair, never inside one file. An UNREADABLE header still
refuses -- see append_pick_log.

ROBUSTNESS: run_pick_log_stage() -- the entry point Sbocker calls -- wraps the whole stage so
a pick-log failure LOGS LOUDLY (a !!! banner + traceback on BOTH stdout and stderr, patterned
on the carve-out fallback in postBo.py) but NEVER crashes the deliverable/run. A missing
pick-log is recoverable; a crashed pipeline is not.
"""

import csv
import os
import subprocess
import sys
from datetime import datetime

import carveOut as _co

# Repo-root log file (this module lives in the repo root alongside its callers).
PICK_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pick_log.csv')

# Fixed column order -- the header written once on file creation. Long/tidy: one row per
# (run, list, stock).
#  ENTRY VALUATION (added 2026-07-29).  The log recorded WHAT was picked and its AggScore but
#  NOTHING about the valuation paid, which made the CEO's stated sell trigger -- "P/E going up"
#  -- uninstrumentable: with no entry multiple on record there is nothing for a later multiple
#  to be compared against.  These columns are captured AT ENTRY, on the same append-only row.
#
#  =====================================================================================
#  THE NAMES DECLARE THE BASIS, AND THE BASIS IS NOT A TRADED PRICE.  READ THIS.
#  =====================================================================================
#  Every figure here is derived from the STATEMENT PERIOD-END, because that is what `cdx_df`
#  carries -- it is `marketCap / weightedAverageShsOut` at the period end, NOT the quote on the
#  day the name was picked.  Measured on the 2026-07-17 panel: median divergence from the live
#  quote 21.2%, with 76% of names more than 10% apart, and top-20 rows 3.5-6.5 months stale.
#  It is also in the company's OWN REPORTING CURRENCY (WATR.L reads 3.83 against a 302.50 quote),
#  which is why `reporting_currency` is logged beside it -- a number with no currency is not a
#  price.  The columns were briefly named `entry_price` / `entry_PB`; they were renamed before
#  any row was ever written, because `pick_log.csv` is APPEND-ONLY and a mis-named column would
#  have been permanent and unrepairable.
#
#  WHY NOT FETCH THE REAL QUOTE (the honest alternative, deliberately deferred): it is the
#  better number and it is cheap -- a handful of calls for a 20-name list, on the machine where
#  the pipeline already spends ~12 hours of API budget.  It is NOT wired here because the CEO's
#  standing rule is to confirm an external endpoint with a verifying call BEFORE wiring it, and
#  that call cannot be made from this machine.  Wiring an unverified network dependency into an
#  APPEND-ONLY log is the one place a silent failure is permanent.  So: declare the basis now,
#  add `entry_quote_*` columns once the quote endpoint is confirmed on the home machine.
#
#  `entry_industry_median_PE` is the peer yardstick: a P/E of 14 means opposite things in Marine
#  Shipping and in Software, so an absolute multiple alone cannot support a "re-rated" judgement.
#
#  UNIVERSE PROVENANCE (added 2026-08-04, audit H-5).  Every OTHER artifact the pipeline emits
#  carries the universe stamp (the saved panel, `resdic`, the postRank pickle, RunProvenance.json)
#  -- this file, the only one that cannot be regenerated, carried NOTHING.  A pick made against
#  the 140-name curated TEST universe was therefore indistinguishable from a production pick in
#  the exact instrument the forward beat-rate target rests on, and since the log is append-only
#  the omission would have been permanent for every row already written.
#
#  BOTH FIELDS, NOT ONE.  The NAME alone is insufficient and that is the whole reason the
#  fingerprint exists: `stock_NA1_EU1` denotes a DIFFERENT universe before and after the
#  2026-08-02 European restoration (1,046 names apart), so two rows can share a name and describe
#  different pools.  The name is what the deliverable FILENAMES are built from (so it is what a
#  reader will try to match on); the FINGERPRINT is what actually settles whether two rows are
#  comparable.  Both are read from `resdic` -- the same object Sbocker stamps for the postRank
#  pickle -- never re-derived from configdic, so all artifacts of a run agree by construction
#  rather than by two call sites happening to match (see Sbocker's stamp comment).
PICK_LOG_COLUMNS = ['as_of', 'logged_at', 'filter_commit',
                    'universe', 'universe_fingerprint', 'list', 'rank',
                    'ticker', 'company', 'aggscore',
                    'reporting_currency',
                    'entry_periodend_price_reporting_ccy',
                    'entry_periodend_trailing_PE',
                    'entry_periodend_PB_fmp_basis',
                    'entry_periodend_grahamNumberToPrice',
                    'entry_industry_median_periodend_PE',
                    'entry_industry_median_n']

# Cohort carve-label (from carveOut) -> pick-log `list` tag. Referencing the carveOut
# constants keeps this in lock-step if a label is ever renamed there. Iteration order here
# is the log's cohort order: REIT, MINING, FIN1, FIN2, FIN3.
_COHORT_LABEL_MAP = [
    ('REIT', 'REIT'),
    ('Mining', 'MINING'),
    (_co.FIN1_VEHICLE, 'FIN1'),
    (_co.FIN2_MANAGER, 'FIN2'),
    (_co.FIN3_BALSHEET, 'FIN3'),
]

_GENERAL_LIST = 'GENERAL'


_REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _git_short_hash():
    """Best-effort short git hash of VALUATION_REPO HEAD, with a '-dirty' suffix when the
    working tree has uncommitted changes (so the commit stamp is HONEST about what actually
    ran, not just the last commit). Returns 'unknown' on ANY failure (no git, detached,
    subprocess error) -- MUST NOT raise, so the run never dies over it."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=_REPO_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=10,
        )
        if out.returncode == 0:
            h = out.stdout.decode('utf-8', 'replace').strip()
            if h:
                # Best-effort dirty check; on ANY git failure fall back to the clean hash.
                try:
                    st = subprocess.run(
                        ['git', 'status', '--porcelain'],
                        cwd=_REPO_DIR,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=10,
                    )
                    if st.returncode == 0 and st.stdout.decode('utf-8', 'replace').strip():
                        return h + '-dirty'
                except Exception:
                    pass
                return h
    except Exception:
        pass
    return 'unknown'


def _resolve_as_of(as_of):
    """The run's as-of date as a string. as_of=None means live/today -> stamp today's date
    (matches the deliverable filename date), so every block carries a meaningful as_of."""
    if as_of:
        return str(as_of)
    return datetime.today().strftime('%Y-%m-%d')


def _names_map(resdic):
    """symbol -> company name from resdic['Tickers_df'] (merged in from datandmetricdic).
    Returns {} if unavailable -- company is then blank, never a crash."""
    tdf = resdic.get('Tickers_df')
    cols = getattr(tdf, 'columns', [])
    if tdf is not None and 'symbol' in cols and 'name' in cols:
        return dict(zip(tdf['symbol'], tdf['name']))
    return {}


def _warn_empty_general(frame):
    """LOUD warning (BOTH streams) when the GENERAL shortlist logs zero rows -- a silently
    empty forward record is exactly the failure this log exists to prevent, so it can NEVER
    pass quietly under the stage's success line. Mirrors the side-list warning, escalated."""
    if frame is None:
        reason = "frame is None (resdic['postRank'] missing)"
    elif len(frame) == 0:
        reason = "frame is empty (0 rows)"
    elif 'source' not in getattr(frame, 'columns', []):
        reason = "frame lacks the expected 'source' column"
    else:
        reason = "no rows survived head(depth)"
    banner = (
        "\n" + "!" * 78 + "\n"
        "!!! PICK-LOG WARNING -- GENERAL SHORTLIST LOGGED ZERO ROWS !!!\n"
        "!!! The general top-N frame (resdic['postRank']) was unusable this run:  !!!\n"
        f"!!!   {reason}.\n"
        "!!! The forward record has NO general picks for this run -- INVESTIGATE;  !!!\n"
        "!!! do NOT treat this pick-log block as complete.                         !!!\n"
        + "!" * 78 + "\n")
    print(banner, file=sys.stderr, flush=True)
    print(banner, flush=True)


#  Sentinels for a run whose universe could NOT be established from `resdic`.  Sbocker stamps
#  `resdic` on EVERY path (fetch, -loadbometric, -loadboresults), so a missing stamp means
#  something upstream of here broke -- and a BLANK universe column would silently reintroduce
#  exactly the defect these columns were added to close, permanently, on an append-only row.
#  An explicit "unknown" is worth more than a blank: a blank reads as "not applicable", an
#  unknown reads as "unestablished", which is the truth. Distinct from Sbocker's
#  `unknown-unstamped-panel` (a LOADED panel that predates stamping -- a legitimate, differently
#  -caused unknown) so the two cases stay tellable apart in the log.
UNKNOWN_UNIVERSE = 'unknown'
UNKNOWN_UNIVERSE_FINGERPRINT = 'unknown-unstamped-resdic'


def _warn_unstamped_universe(name):
    """LOUD warning (BOTH streams) when this run's picks are logged with no establishable
    universe fingerprint. Not fatal -- a pick recorded with an honest `unknown` provenance is
    still worth more than no forward record at all -- but it can never pass quietly, because
    the rows are permanent and a later grader has no way to ask what pool they came from."""
    banner = (
        "\n" + "!" * 78 + "\n"
        "!!! PICK-LOG WARNING -- NO UNIVERSE FINGERPRINT ON THIS RUN'S PICKS !!!\n"
        "!!! resdic carries no 'universe_fingerprint' (universe name seen: %r).\n"
        "!!! Sbocker stamps resdic on every path, so this indicates an upstream    !!!\n"
        "!!! provenance failure. The rows are still logged -- APPEND-ONLY, so they !!!\n"
        "!!! are permanent -- but stamped %r, meaning a later\n"
        "!!! grader CANNOT establish which pool these picks came from. Do NOT treat !!!\n"
        "!!! them as comparable to a stamped run.                                   !!!\n"
        % (name or '', UNKNOWN_UNIVERSE_FINGERPRINT)
        + "!" * 78 + "\n")
    print(banner, file=sys.stderr, flush=True)
    print(banner, flush=True)


def _warn_fingerprint_drift(name, prev_fp, now_fp, prev_as_of, n_prev_rows):
    """LOUD warning (BOTH streams) when a universe NAME is running under a DIFFERENT
    definition than the last time it was logged. WARN ONLY -- see check_fingerprint_drift."""
    banner = (
        "\n" + "!" * 78 + "\n"
        "!!! UNIVERSE DEFINITION DRIFT -- %r MEANS SOMETHING DIFFERENT NOW !!!\n"
        "!!!   last logged fingerprint : %s  (as_of %s, %d row(s))\n"
        "!!!   this run's fingerprint  : %s\n"
        "!!! The NAME is unchanged, so nothing else in the pipeline notices. The\n"
        "!!! fingerprint covers the EXCHANGE SET, the SAMPLE RATES and the MUST-INCLUDE\n"
        "!!! list, so at least one of those three moved -- and a sample-rate change can\n"
        "!!! move membership while leaving the member COUNT flat, which is exactly the\n"
        "!!! gap the wallclock guard and the cohort-sum pin do NOT close.\n"
        "!!! WARN ONLY, BY DESIGN (CEO 2026-08-06): the run PROCEEDS UNCHANGED, nothing\n"
        "!!! is recomputed and nothing is refused. This is your cue to decide whether\n"
        "!!! rows under the two fingerprints may be POOLED when grading -- they are the\n"
        "!!! same name but not the same universe. Both fingerprints are on the rows.\n"
        % (name, prev_fp, prev_as_of, n_prev_rows, now_fp)
        + "!" * 78 + "\n")
    print(banner, file=sys.stderr, flush=True)
    print(banner, flush=True)


def check_fingerprint_drift(name, now_fp, path=PICK_LOG_PATH):
    """Has `name` been logged before under a DIFFERENT definition fingerprint?  WARN ONLY.

    Closes the residual the tightened wallclock guard and the cohort-sum pin leave open: both
    trip on a MEMBER-COUNT move, so a definition change that leaves the count roughly unmoved
    (a sample rate retuned, one exchange swapped for a similar one, a must-include edited)
    passes them silently.  `universes.definition_fingerprint` already hashes exactly those
    three things and is already stamped on every pick-log row -- what was missing was anybody
    COMPARING this run's stamp against the recorded one.  That is all this does.

    WARN ONLY, deliberately (CEO 2026-08-06), mirroring the bar-calibration failsafe: on a
    mismatch it warns and RECORDS, and changes NOTHING -- no refusal, no auto-recompute, no
    write to the append-only log beyond this run's own rows.  The drift is a judgement call
    about whether historical rows may be POOLED with new ones, and that judgement is the
    CEO's; a hard failure here would block a legitimate deliberate redefinition.

    Returns a dict describing the drift, or None when there is nothing to say (first run for
    this name, unchanged fingerprint, unstamped/unknown either side, or no readable log).

    Reads with the `csv` module, NOT pandas, to match the rest of this file -- pick_log has no
    module-level pandas import, and an earlier draft of this function used `pd` anyway.  The
    resulting NameError was swallowed by a broad `except Exception: return None` and the check
    silently never fired: a drift detector that reports "no drift" because it crashed is the
    same silent-null defect this whole change set exists to remove.  Hence also the NARROW
    except below -- an unreadable/legacy log is expected and returns None quietly, but a
    programming error must surface, not masquerade as a clean result.
    """
    if not name or name == UNKNOWN_UNIVERSE:
        return None
    if not now_fp or now_fp == UNKNOWN_UNIVERSE_FINGERPRINT:
        return None            # already warned by _warn_unstamped_universe
    now_fp = str(now_fp).strip()
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        with open(path, 'r', encoding='utf-8', newline='') as rf:
            rows = list(csv.DictReader(rf))
    except (OSError, UnicodeDecodeError, csv.Error):
        #  Unreadable, or a log written before universe provenance existed. Not this
        #  function's job to diagnose -- append_pick_log's schema check owns that.
        return None
    prior = [r for r in rows
             if str(r.get('universe', '')).strip() == str(name)
             and str(r.get('universe_fingerprint') or '').strip()
             not in ('', UNKNOWN_UNIVERSE_FINGERPRINT)]
    if not prior:
        return None
    #  Compare against the MOST RECENT prior stamp for this name; earlier ones may include a
    #  drift already warned about on a previous run. `as_of` is ISO yyyy-mm-dd, so a plain
    #  string sort is chronological; ties keep file order, which is append order.
    prior.sort(key=lambda r: str(r.get('as_of') or ''))
    prev_fp = str(prior[-1]['universe_fingerprint']).strip()
    if prev_fp == now_fp:
        return None
    rec = {'universe': name, 'previous_fingerprint': prev_fp,
           'current_fingerprint': now_fp,
           'previous_as_of': str(prior[-1].get('as_of') or ''),
           'previous_rows_logged': len(prior),
           'distinct_prior_fingerprints':
               len({str(r['universe_fingerprint']).strip() for r in prior}),
           'detected_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'action_taken': 'WARN ONLY -- run proceeded unchanged (CEO 2026-08-06)'}
    _warn_fingerprint_drift(name, prev_fp, now_fp, rec['previous_as_of'],
                            rec['previous_rows_logged'])
    #  RECORD it as a dated CSV beside the other dated drift reports (CurrencyFloorFlips,
    #  ExcludedShareClasses). Best-effort: failing to write the note must not affect the run,
    #  and the banner has already fired either way.
    try:
        _fn = 'UniverseDefinitionDrift_%s.csv' % datetime.now().strftime('%Y-%m-%d')
        _p = os.path.join(os.path.dirname(os.path.abspath(path)), _fn)
        _new = not os.path.exists(_p) or os.path.getsize(_p) == 0
        with open(_p, 'a', encoding='utf-8', newline='') as wf:
            w = csv.DictWriter(wf, fieldnames=list(rec.keys()))
            if _new:
                w.writeheader()
            w.writerow(rec)
        print('[PICK-LOG] universe-definition drift recorded to: %s' % _p, flush=True)
    except (OSError, UnicodeEncodeError, csv.Error) as _e:
        print('[PICK-LOG] WARNING: could not record the drift note (%s) -- the banner '
              'above is the only record for this run.' % _e, flush=True)
    return rec


def _universe_stamp(resdic):
    """(universe_name, universe_fingerprint) for this run, READ FROM `resdic`.

    Deliberately not re-derived from configdic/universes: `resdic`'s stamp already accounts for
    the load paths, where the data's universe is the LOADED panel's and NOT the active
    `-tickerfilter` (stamping the current filter onto loaded data would manufacture provenance).
    Reading the same key Sbocker wrote is what makes the pick-log row agree with the postRank
    pickle and the deliverables by construction."""
    name = resdic.get('universe')
    fp = resdic.get('universe_fingerprint')
    name = '' if name is None else str(name).strip()
    fp = '' if fp is None else str(fp).strip()
    if not fp:
        _warn_unstamped_universe(name)
        return (name or UNKNOWN_UNIVERSE), UNKNOWN_UNIVERSE_FINGERPRINT
    return (name or UNKNOWN_UNIVERSE), fp


_VAL_COLS = ['reporting_currency',
             'entry_periodend_price_reporting_ccy', 'entry_periodend_trailing_PE',
             'entry_periodend_PB_fmp_basis', 'entry_periodend_grahamNumberToPrice',
             'entry_industry_median_periodend_PE', 'entry_industry_median_n']
_EMPTY_VAL = {c: '' for c in _VAL_COLS}

#  Minimum peer count for an industry median to be published.  One FMP industry in this
#  universe has a SINGLE member, whose "industry median P/E" is just its own P/E -- a
#  self-comparison that would read as "in line with peers" by construction.
MIN_PEERS_FOR_INDUSTRY_MEDIAN = 5


def entry_valuations(resdic):
    """ticker -> entry valuation dict, from the NEWEST cdx row per source.

    PERIOD-END values, not traded prices, and not window averages -- see the long note at
    PICK_LOG_COLUMNS for what they are and are not.  The Stage-2 metric frame holds head(nq)
    MEANS of these quantities (which is what the score wants); this reads the newest row so the
    figure is at least a single point in time rather than a trailing average of several.

    `entry_periodend_trailing_PE` uses `epsTTM`, the canonical rpy-aware trailing EPS stamped by
    getData_fmp.stamp_frequency_and_graham on the SAME share basis as `price` -- deliberately
    not FMP's `earningsYield`, which is computed against FMP's own price.  A non-positive EPS
    yields a BLANK, never a negative number: a negative P/E is not a cheaper valuation and must
    never sort or compare as one.

    EACH FIELD IS GUARDED INDEPENDENTLY.  A first version wrapped all of them in one try, so a
    single missing cdx column (`pbRatio`, say) blanked ALL five for EVERY ticker -- one absent
    field silently destroying four present ones.  Now a missing column costs only its own
    column.  Everything remains best-effort: a missing entry valuation is recoverable, a crashed
    pipeline is not (the module's standing rule).
    """
    out = {}
    try:
        import numpy as _np
        import pandas as _pd
        cdx = resdic.get('cdx_df')
        if cdx is None or 'source' not in getattr(cdx, 'columns', []):
            print('[PICK-LOG] no cdx_df: entry valuations blank.', flush=True)
            return out
        d = cdx.copy()
        d['date'] = _pd.to_datetime(d['date'], errors='coerce')
        # cdx is stored ascending per source, so the LAST row is the newest; sort defensively
        d = d.sort_values(['source', 'date'])
        newest = d.groupby('source', sort=False).tail(1)
        if 'source' not in newest.columns:
            return out
        newest = newest.set_index('source')

        def _num(col):
            """Independent per-column read: absent -> an all-NaN series, never an exception."""
            if col not in newest.columns:
                print('[PICK-LOG] cdx column %r absent; that column logs blank (others are '
                      'unaffected).' % col, flush=True)
                return _pd.Series(_np.nan, index=newest.index, dtype='float64')
            return _pd.to_numeric(newest[col], errors='coerce')

        px, eps = _num('price'), _num('epsTTM')
        pb, gr = _num('pbRatio'), _num('grahamNumber')
        ccy = (newest['reportedCurrency'].astype(object)
               if 'reportedCurrency' in newest.columns
               else _pd.Series('', index=newest.index, dtype=object))

        pe = (px / eps.where(eps > 0)).replace([_np.inf, -_np.inf], _np.nan)
        gtp = (gr / px.where(px != 0)).replace([_np.inf, -_np.inf], _np.nan)

        # industry median P/E over the whole universe (peers, not the shortlist), with a
        # MINIMUM peer count so a one-member industry cannot self-compare.
        med_by_ind, n_by_ind, ind = {}, {}, {}
        try:
            import carveOut as _co2
            ind = _co2._load_industry_map()
            tmp = _pd.DataFrame({'pe': pe})
            tmp['ind'] = [ind.get(s) for s in tmp.index]
            g = tmp.dropna(subset=['pe', 'ind']).groupby('ind')['pe']
            n_by_ind = g.size().to_dict()
            med_by_ind = {k: v for k, v in g.median().to_dict().items()
                          if n_by_ind.get(k, 0) >= MIN_PEERS_FOR_INDUSTRY_MEDIAN}
        except Exception as _ie:
            print('[PICK-LOG] industry median unavailable (%s: %s); that column logs blank.'
                  % (type(_ie).__name__, _ie), flush=True)

        def _f(v):
            return '' if v is None or _pd.isna(v) else float(v)

        for s in newest.index:
            _i = ind.get(s)
            out[s] = {
                'reporting_currency': ('' if _pd.isna(ccy.get(s)) else str(ccy.get(s) or '')),
                'entry_periodend_price_reporting_ccy': _f(px.get(s)),
                'entry_periodend_trailing_PE': _f(pe.get(s)),
                'entry_periodend_PB_fmp_basis': _f(pb.get(s)),
                'entry_periodend_grahamNumberToPrice': _f(gtp.get(s)),
                'entry_industry_median_periodend_PE': _f(med_by_ind.get(_i)),
                'entry_industry_median_n': (n_by_ind.get(_i, '')
                                            if _i in med_by_ind else ''),
            }
    except Exception as _e:
        print('[PICK-LOG] entry valuations unavailable (%s: %s); logging blanks.'
              % (type(_e).__name__, _e), flush=True)
    return out


def _rows_from_frame(frame, list_tag, depth, names, vals=None):
    """Emit (rank, ticker, company, aggscore, entry_*) partial rows from a postRank-style
    frame, rank 1-based within the frame's existing (AggScore-descending) order, head(depth)."""
    rows = []
    if frame is None or 'source' not in getattr(frame, 'columns', []):
        return rows
    vals = vals or {}
    head = frame.head(depth)
    has_agg = 'AggScore' in head.columns
    for rank, (_, r) in enumerate(head.iterrows(), start=1):
        ticker = r['source']
        company = names.get(ticker, '')
        aggscore = r['AggScore'] if has_agg else ''
        row = {
            'list': list_tag,
            'rank': rank,
            'ticker': '' if ticker is None else str(ticker),
            'company': '' if company is None else str(company),
            'aggscore': '' if aggscore is None else aggscore,
        }
        row.update(vals.get(ticker, _EMPTY_VAL))
        rows.append(row)
    return rows


def build_pick_log_rows(resdic, as_of=None, logged_at=None, filter_commit=None):
    """Build the full list of pick-log row dicts for this run: GENERAL top-N + the five
    cohort side-lists, each tagged and rank-stamped. Depths MATCH the emitted deliverables:
      GENERAL     -> ntopxlsx (the presentation top-N; =20 by default) from resdic['postRank']
      side-lists  -> ntopagg  (the side-list CSV depth; =100 by default) from
                     resdic['carveout_sidelists'][label]['postRank']
    Every row carries the same run-level as_of / logged_at / filter_commit / universe /
    universe_fingerprint stamps."""
    as_of_str = _resolve_as_of(as_of)
    if logged_at is None:
        logged_at = datetime.now().isoformat()
    if filter_commit is None:
        filter_commit = _git_short_hash()
    universe, universe_fp = _universe_stamp(resdic)
    #  WARN-ONLY definition-drift check (CEO 2026-08-06). Runs here because this is the one
    #  place per run that already holds BOTH the current stamp and a persistent record of the
    #  previous one; it cannot alter `universe`/`universe_fp` or the rows that follow.
    check_fingerprint_drift(universe, universe_fp)
    names = _names_map(resdic)
    vals = entry_valuations(resdic)

    # Depth of the emitted GENERAL deliverable = the presentation top-N (ntopxlsx). Falls
    # back to 20 (the config default) if absent. Side-lists match the side-list CSV = ntopagg.
    general_depth = resdic.get('ntopxlsx', 20)
    sidelist_depth = resdic.get('ntopagg', 100)

    partial = []
    # GENERAL: the carve-deduped general pool -- the exact frame the emitted top-N comes from.
    gen_frame = resdic.get('postRank')
    gen_rows = _rows_from_frame(gen_frame, _GENERAL_LIST, general_depth, names, vals)
    if not gen_rows:
        _warn_empty_general(gen_frame)
    partial += gen_rows

    # Five cohort side-lists, in fixed REIT/MINING/FIN1/FIN2/FIN3 order.
    sidelists = resdic.get('carveout_sidelists') or {}
    for carve_label, list_tag in _COHORT_LABEL_MAP:
        sdic = sidelists.get(carve_label)
        if not sdic or 'postRank' not in sdic:
            # A degenerate/failed cohort side-list (already warned upstream): note and skip;
            # its absence is visible by a missing block, never a crash.
            print(f"[PICK-LOG] side-list '{list_tag}' ({carve_label}) absent this run; "
                  f"skipping (no rows logged for it).", flush=True)
            continue
        partial += _rows_from_frame(sdic['postRank'], list_tag, sidelist_depth, names,
                                   vals)

    # Stamp the run-level columns onto every row.
    rows = []
    for p in partial:
        row = {'as_of': as_of_str, 'logged_at': logged_at, 'filter_commit': filter_commit,
               'universe': universe, 'universe_fingerprint': universe_fp}
        row.update(p)
        rows.append(row)
    return rows


#  Directory a SUPERSEDED pick log is moved into when the writer's schema no longer matches
#  the header on disk.  Sits beside the log, and is ALREADY gitignored (`.gitignore`
#  `_quarantine/`), so a quarantined forensic record never enters git history.  The naming
#  convention `pick_log_<reason>_<YYYY-MM-DD>.csv` matches what the pre-2026-08-22 refusal
#  message told a human to do by hand, and what is already in there from the 2026-08-04
#  TESTUNIVERSE quarantine -- so the auto-move is indistinguishable from the manual one.
QUARANTINE_DIRNAME = '_quarantine'


def _quarantine_target(path, reason, today=None, quarantine_dir=None):
    """Destination for a superseded pick log: `<dir>/_quarantine/<stem>_<reason>_<date>.csv`.

    NEVER RETURNS AN EXISTING PATH.  Two quarantines on the SAME DAY (two runs, or a
    re-run after a partial recovery) would otherwise collide on the date, and the second
    would destroy the first -- which is the one outcome this whole path exists to prevent.
    On a collision the candidate is suffixed `-2`, `-3`, ... until it is free, so the
    sequence is readable and no earlier record is ever the loser.
    """
    d = quarantine_dir or os.path.join(os.path.dirname(os.path.abspath(path)),
                                       QUARANTINE_DIRNAME)
    stem = os.path.splitext(os.path.basename(path))[0]
    day = today or datetime.today().strftime('%Y-%m-%d')
    base = '%s_%s_%s' % (stem, reason, day)
    cand = os.path.join(d, base + '.csv')
    n = 2
    while os.path.exists(cand):
        cand = os.path.join(d, '%s-%d.csv' % (base, n))
        n += 1
    return cand


def quarantine_pick_log(path, reason, today=None, quarantine_dir=None):
    """MOVE the existing log aside, BYTE-IDENTICAL, and return the destination path.

    `os.rename` -- deliberately NOT `os.replace` and NOT copy-then-delete.  `os.replace`
    silently clobbers an existing destination; a copy-then-delete has a window in which the
    only copy of an append-only forensic record is a partial one.  `os.rename` moves the
    inode within the filesystem (the quarantine dir is a sibling of the log, so this is
    always same-filesystem) and, on Windows, RAISES rather than overwrites if the
    destination appeared after `_quarantine_target` checked -- so the collision guard is
    belt AND braces.

    Raises OSError (with a message naming the manual fallback) if the move is refused --
    the realistic case being the log open in a spreadsheet, which locks it on Windows.
    Refusing here is correct: if the old file cannot be moved, the new header cannot be
    written, and nothing may be appended under the old one.
    """
    dest = _quarantine_target(path, reason, today=today, quarantine_dir=quarantine_dir)
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.rename(path, dest)
    except OSError as e:
        raise OSError(
            'PICK-LOG QUARANTINE FAILED: could not move %s aside to %s (%s: %s). The '
            'existing log is UNTOUCHED and NOTHING was written. The usual cause on Windows '
            'is the file being open in a spreadsheet -- close it and re-run. Otherwise move '
            'it by hand to that path; do NOT hand-edit its header or pad its rows.'
            % (path, dest, type(e).__name__, e)) from None
    return dest


def append_pick_log(rows, path=PICK_LOG_PATH):
    """APPEND-ONLY write. Creates the file with a header if it does not exist (or is empty);
    otherwise appends rows with NO header. NEVER reads or rewrites existing content -- the
    file is only ever opened in append mode, so prior rows are physically untouchable here.
    Returns the number of rows appended."""
    file_exists = os.path.exists(path)
    size = os.path.getsize(path) if file_exists else 0
    write_header = (not file_exists) or (size == 0)

    # HEADER-WIDTH GUARD (added 2026-07-29; became a SELF-HEALING QUARANTINE 2026-08-22).
    # This file is APPEND-ONLY and has NO header on an append, so if PICK_LOG_COLUMNS gains or
    # loses a column, every subsequent block silently mis-aligns against the header already on
    # disk -- and because nothing may be rewritten, the damage is permanent.  Checked ONLY on
    # the append path (a fresh file writes its own header).
    #
    # WHAT CHANGED, AND WHY THE REFUSAL HAD TO GO.  From 2026-07-29 to 2026-08-22 the drift
    # branch RAISED, on the reasoning that a stopped forensic record is recoverable and a
    # half-migrated one is not.  The first half of that is still true; the second half turned
    # out to be an argument for the wrong remedy.  The universe columns widened the schema from
    # 8 to 17, so the CEO's run machine carried a pre-2026-08-04 pick_log.csv and the stage
    # raised on EVERY run: the 08-20 and 08-22 runs both recorded ZERO forward picks.  The log
    # is the ONE artifact the pipeline cannot regenerate -- a pick not stamped the night it was
    # made can never be stamped honestly later -- and the fix (move the file aside) was only
    # ever executable by a human standing at that machine.  So the refusal was not a pause; it
    # was a permanent, silent-to-the-target outage that compounded one lost run per night.
    #
    # THE QUARANTINE IS NOT A MIGRATION, and that distinction is the whole point.  The invariant
    # the refusal defended -- NO BYTE OF AN EXISTING ROW IS EVER REWRITTEN -- is preserved
    # exactly: the old file is MOVED (os.rename, same filesystem, bytes untouched) into
    # `_quarantine/` under a dated, collision-proof name, and stays a complete, readable record
    # of its own era under its own header.  Then a fresh pick_log.csv is created with the
    # current header.  Backfilling the old rows remains REJECTED for the original reason: the
    # only honest value for a pre-2026-08-04 row's universe is "unknown", so a backfill buys a
    # uniform schema by rewriting an append-only file for no information gain.
    #
    # WHAT IT COSTS, stated rather than hidden: the forward record is now SPLIT ACROSS FILES,
    # so a grader must read `_quarantine/pick_log_*.csv` alongside `pick_log.csv` to see the
    # whole history.  That is strictly better than the two alternatives on offer (a permanent
    # hole in the record, or one file whose columns mean different things in different blocks),
    # and the split is announced loudly on both streams every time it happens.
    if file_exists and size > 0 and not write_header:
        _hdr_err = None
        try:
            with open(path, 'r', encoding='utf-8', newline='') as hf:
                existing = next(csv.reader(hf), None)
        except Exception as _he:
            existing = None
            _hdr_err = '%s: %s' % (type(_he).__name__, _he)
        # An UNREADABLE / absent header on a non-empty file STILL RAISES (tightened 2026-08-04;
        # deliberately NOT converted to a quarantine 2026-08-22).  The drift case is a KNOWN,
        # diagnosed schema change -- we know exactly what the old file is and that moving it
        # loses nothing.  Here we know nothing: the file may be mid-write by another process, or
        # the disk may be damaged, and moving a file we cannot read is a guess dressed as a
        # remedy.  Cannot establish the on-disk schema => must not append to it, and must not
        # quietly relocate it either.
        if not existing:
            raise RuntimeError(
                'PICK-LOG HEADER UNREADABLE: %s is %d byte(s) long but no header row could be '
                'read from it (%s). The on-disk schema therefore cannot be established, and '
                'appending would risk permanently mis-aligning an append-only forensic record. '
                'This case is NOT auto-quarantined (unlike a schema drift): the file cannot be '
                'read, so nothing here knows what it is. FIX: inspect it by hand; move it aside '
                '(e.g. to _quarantine/pick_log_<reason>_<YYYY-MM-DD>.csv) to let the next run '
                'create a fresh log. Do NOT hand-edit it into shape.'
                % (path, size, _hdr_err or 'file has no parseable first row'))
        if list(existing) != list(PICK_LOG_COLUMNS):
            _missing = [c for c in existing if c not in PICK_LOG_COLUMNS]
            _added = [c for c in PICK_LOG_COLUMNS if c not in existing]
            # The reason slug names the ERA the quarantined file belongs to when we can
            # recognise it, and says only "schemadrift" when we cannot -- a guessed-precise
            # name on a file we did not diagnose would be worse than a vague true one.
            _reason = ('preuniverse'
                       if {'universe', 'universe_fingerprint'}.issubset(set(_added))
                       else 'schemadrift')
            _dest = quarantine_pick_log(path, _reason)
            bang = '!' * 78
            banner = '\n'.join([
                '', bang,
                '!!! PICK-LOG SCHEMA DRIFT -- EXISTING LOG QUARANTINED, FRESH LOG STARTED !!!',
                '!!!   was : %s  (%d columns)' % (path, len(existing)),
                '!!!   now : %s  (%d columns)' % ('writer schema', len(PICK_LOG_COLUMNS)),
                '!!!   moved to: %s' % _dest,
                '!!!   removed: %s' % _missing,
                '!!!   added  : %s' % _added,
                '!!! The moved file is BYTE-IDENTICAL and remains a complete record under its',
                '!!! own header -- no row was rewritten, nothing was backfilled. A fresh',
                '!!! pick_log.csv is being created with the current header.',
                '!!! CONSEQUENCE FOR ANY GRADER: the forward record is now SPLIT. Read the',
                '!!! quarantined file ALONGSIDE pick_log.csv or the history is incomplete.',
                bang, ''])
            print(banner, file=sys.stderr, flush=True)
            print(banner, flush=True)
            # The log no longer exists at `path`; this run creates it and writes the header.
            file_exists = False
            size = 0
            write_header = True

    # LOW-2 -- trailing-partial-row seam: if a PRIOR run was crash-truncated mid-row (its
    # last byte is not a newline), pad ONE newline at EOF first, so this run's block cannot
    # merge onto the dangling partial row. Still append-only: we add a byte at the end and
    # NEVER rewrite an existing row (a normally-written file already ends in '\n', so this
    # is a no-op except after a truncation). Reading the last byte does not mutate the file.
    if file_exists and size > 0:
        with open(path, 'rb') as rf:
            rf.seek(-1, os.SEEK_END)
            last_byte = rf.read(1)
        if last_byte != b'\n':
            with open(path, 'a', encoding='utf-8', newline='') as pf:
                pf.write('\n')

    # encoding='utf-8' is MANDATORY: the universe holds non-US issuers whose names carry
    # non-cp1252 characters; on Windows the default text encoding (cp1252) would raise
    # UnicodeEncodeError on the first such name and abort the whole stage -> zero forward
    # picks that run. utf-8 makes the writer platform-independent.
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        if write_header:
            w.writerow(PICK_LOG_COLUMNS)
        for r in rows:
            w.writerow([r.get(c, '') for c in PICK_LOG_COLUMNS])
    return len(rows)


def write_pick_log(resdic, as_of=None, path=PICK_LOG_PATH,
                   logged_at=None, filter_commit=None):
    """Build this run's rows and append them. Returns the number of rows written. Raises on a
    genuine error (so tests see it); the guarded stage below swallows it loudly for the run."""
    rows = build_pick_log_rows(resdic, as_of=as_of, logged_at=logged_at,
                               filter_commit=filter_commit)
    n = append_pick_log(rows, path=path)
    return n


def run_pick_log_stage(resdic, as_of=None, path=PICK_LOG_PATH):
    """Pipeline entry point (called from Sbocker.main AFTER the deliverables are emitted).

    Wraps the whole stage so a pick-log failure LOGS LOUDLY but NEVER crashes the run --
    a missing pick-log entry is recoverable; a crashed deliverable pipeline is not. On
    success prints a one-line confirmation; on failure prints an unmistakable !!! banner
    plus the traceback on BOTH stderr and stdout (patterned on the carve-out fallback in
    postBo.py). Returns the rows-written count, or None on failure."""
    try:
        n = write_pick_log(resdic, as_of=as_of, path=path)
        print(f"[PICK-LOG] appended {n} rows to {path} "
              f"(as_of={_resolve_as_of(as_of)}); append-only, no prior row touched.",
              flush=True)
        return n
    except Exception as e:
        import traceback
        banner = (
            "\n" + "!" * 78 + "\n"
            "!!! PICK-LOG STAGE FAILED -- NO FORWARD PICKS RECORDED THIS RUN !!!\n"
            "!!! The append-only prospective track record was NOT updated for this   !!!\n"
            "!!! run. The deliverables above are UNAFFECTED (this stage is isolated), !!!\n"
            "!!! but today's picks are missing from pick_log.csv -- re-run or append  !!!\n"
            "!!! manually to keep the forward record complete.                        !!!\n"
            f"!!! Cause: {type(e).__name__}: {e}\n"
            + "!" * 78 + "\n")
        # stderr first (survives stdout redirection / tqdm noise), then stdout.
        print(banner, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(banner, flush=True)
        traceback.print_exc(file=sys.stdout)
        return None
