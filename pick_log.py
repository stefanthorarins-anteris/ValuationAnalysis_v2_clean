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
PICK_LOG_COLUMNS = ['as_of', 'logged_at', 'filter_commit', 'list', 'rank',
                    'ticker', 'company', 'aggscore']

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


def _rows_from_frame(frame, list_tag, depth, names):
    """Emit (rank, ticker, company, aggscore) partial rows from a postRank-style frame,
    rank 1-based within the frame's existing (AggScore-descending) order, head(depth)."""
    rows = []
    if frame is None or 'source' not in getattr(frame, 'columns', []):
        return rows
    head = frame.head(depth)
    has_agg = 'AggScore' in head.columns
    for rank, (_, r) in enumerate(head.iterrows(), start=1):
        ticker = r['source']
        company = names.get(ticker, '')
        aggscore = r['AggScore'] if has_agg else ''
        rows.append({
            'list': list_tag,
            'rank': rank,
            'ticker': '' if ticker is None else str(ticker),
            'company': '' if company is None else str(company),
            'aggscore': '' if aggscore is None else aggscore,
        })
    return rows


def build_pick_log_rows(resdic, as_of=None, logged_at=None, filter_commit=None):
    """Build the full list of pick-log row dicts for this run: GENERAL top-N + the five
    cohort side-lists, each tagged and rank-stamped. Depths MATCH the emitted deliverables:
      GENERAL     -> ntopxlsx (the presentation top-N; =20 by default) from resdic['postRank']
      side-lists  -> ntopagg  (the side-list CSV depth; =100 by default) from
                     resdic['carveout_sidelists'][label]['postRank']
    Every row carries the same run-level as_of / logged_at / filter_commit stamps."""
    as_of_str = _resolve_as_of(as_of)
    if logged_at is None:
        logged_at = datetime.now().isoformat()
    if filter_commit is None:
        filter_commit = _git_short_hash()
    names = _names_map(resdic)

    # Depth of the emitted GENERAL deliverable = the presentation top-N (ntopxlsx). Falls
    # back to 20 (the config default) if absent. Side-lists match the side-list CSV = ntopagg.
    general_depth = resdic.get('ntopxlsx', 20)
    sidelist_depth = resdic.get('ntopagg', 100)

    partial = []
    # GENERAL: the carve-deduped general pool -- the exact frame the emitted top-N comes from.
    gen_frame = resdic.get('postRank')
    gen_rows = _rows_from_frame(gen_frame, _GENERAL_LIST, general_depth, names)
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
        partial += _rows_from_frame(sdic['postRank'], list_tag, sidelist_depth, names)

    # Stamp the run-level columns onto every row.
    rows = []
    for p in partial:
        row = {'as_of': as_of_str, 'logged_at': logged_at, 'filter_commit': filter_commit}
        row.update(p)
        rows.append(row)
    return rows


def append_pick_log(rows, path=PICK_LOG_PATH):
    """APPEND-ONLY write. Creates the file with a header if it does not exist (or is empty);
    otherwise appends rows with NO header. NEVER reads or rewrites existing content -- the
    file is only ever opened in append mode, so prior rows are physically untouchable here.
    Returns the number of rows appended."""
    file_exists = os.path.exists(path)
    size = os.path.getsize(path) if file_exists else 0
    write_header = (not file_exists) or (size == 0)

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
