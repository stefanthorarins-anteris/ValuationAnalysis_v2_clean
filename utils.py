import pandas as pd
import sys
import createDicts as cdic
import os
import csv
from datetime import datetime

PRICE_BASIS_TOL = 0.15


def check_panel_basis(dmdic, fname='<unknown>', verbose=True):
    """Report whether a LOADED panel's price/Graham basis matches the current code.

    Detects by measurement, not by filename: marketCap / (price * shares) is ~1.0 on a
    panel built with the current ingest (price = marketCap/weightedAverageShsOut) and ~4.0
    on any panel built before 2026-07-19 (the quarterly-PE derivation divided the price by
    about four).  Returns 'new' | 'old' | 'unknown'.  Never raises and never blocks -- the
    reload path is a legitimate developer workflow -- but it must not be SILENT.
    """
    basis, med, has_period = 'unknown', float('nan'), False
    try:
        cdx = dmdic.get('cdx_df')
        has_period = 'period' in getattr(cdx, 'columns', [])
        mc = pd.to_numeric(cdx['marketCap'], errors='coerce')
        pr = pd.to_numeric(cdx['price'], errors='coerce')
        sh = pd.to_numeric(cdx['weightedAverageShsOut'], errors='coerce')
        r = (mc / (pr * sh)).replace([float('inf'), float('-inf')], pd.NA).dropna()
        if len(r) >= 100:
            med = float(r.median())
            if abs(med - 1.0) <= PRICE_BASIS_TOL:
                basis = 'new'
            elif abs(med - 4.0) <= 4 * PRICE_BASIS_TOL:
                basis = 'old'
    except Exception:
        pass
    if not verbose:
        return basis
    if basis == 'new':
        print('PANEL BASIS OK: %s carries the CURRENT price basis '
              '(marketCap/(price*shares) median %.4f); `period` column %s.'
              % (fname, med, 'present' if has_period else 'absent'), flush=True)
        return basis
    bar = '!' * 78
    msg = [
        '', bar,
        '!!! LOADED PANEL BASIS MISMATCH -- OLD/NEW METRIC BASES WILL BE MIXED.',
        '!!!   panel : ' + str(fname),
        '!!!   marketCap/(price*shares) median = %s  (expect ~1.0 for the current basis,'
        % ('%.4f' % med if med == med else 'n/a'),
        '!!!           ~4.0 for a pre-2026-07-19 panel)   basis = %s' % basis,
        '!!!   `period` column: %s' % ('present' if has_period else 'ABSENT -> frequency '
                                       'falls back to date cadence'),
        '!!! Consequence: Stage-1 scores this panel on the OLD price/Graham basis while',
        '!!! Stage-2 applies the CURRENT per-quarter corrections to the same cdx_df.',
        '!!! uGrahamNumberToPrice runs about 2x loose on an old panel (71.5% vs 38.9% pass).',
        '!!! Any ranking or beat-rate from this run is a hybrid and is NOT comparable to a',
        '!!! freshly-fetched one. Re-fetch for anything decisional.',
        bar, '',
    ]
    out = chr(10).join(msg)
    print(out, file=sys.stderr, flush=True)
    print(out, flush=True)
    return basis


#  A LOADED PANEL'S *BoMetric* BASIS, which `check_panel_basis` above cannot see (2026-08-05).
#  THE HAZARD IS THE SAME SHAPE AS THE ONE THAT FUNCTION EXISTS FOR, one frame further in.  The
#  2026-08-05 NaN-policy change altered two Stage-1 CRITERION COLUMNS rather than any cdx field:
#  `uGrahamNumberToPrice` now carries the boundary value 0.0 on adverse rows instead of NaN, and
#  `PEG` is computed locally instead of read off the vendor field.  Neither renames a column, so
#  `calcScore`'s schema gate -- which is column-EXACT on NAMES -- passes a stale panel silently,
#  and a `-loadbometric` run would score old criterion columns with new code.  cdx-level ratios
#  cannot detect it: cdx is unchanged.
#
#  THE DETECTOR IS AN EXACT ZERO, and that is why it is reliable rather than a heuristic.
#  `uGrahamNumberToPrice` is grahamNumber/price -- a ratio of two continuous positive quantities,
#  which lands on exactly 0.0 with probability zero.  So ANY exact zeros mean the boundary
#  imputation ran; NONE, alongside a large NaN rate, means it did not.  Measured on the
#  2026-07-17 CORRECTED panel rebuilt with the current code: 53,267 of 148,081 rows (36.0%) are
#  exactly 0.0 and 687 remain NaN; on the pre-change rebuild, 0 are exactly 0.0 and 53,954 are
#  NaN.  Emits only, never raises, never blocks -- the reload path is a legitimate workflow, but
#  it must not be silent.
BOMETRIC_BOUNDARY_COLUMN = 'uGrahamNumberToPrice'


def check_bometric_basis(dmdic, fname='<unknown>', verbose=True):
    """Report whether a LOADED BoMetric_df carries the post-2026-08-05 criterion basis.

    Returns 'new' | 'old' | 'unknown'.  'unknown' when the column is absent or the panel is too
    small for the incidence to mean anything -- a small TEST universe can legitimately contain no
    adverse Graham row at all, and calling that 'old' would be a false alarm.
    """
    basis, n_zero, n_nan, n = 'unknown', 0, 0, 0
    try:
        bm = dmdic.get('BoMetric_df')
        v = pd.to_numeric(bm[BOMETRIC_BOUNDARY_COLUMN], errors='coerce')
        n = int(len(v))
        n_nan = int(v.isna().sum())
        n_zero = int((v == 0.0).sum())
        if n >= 1000:
            if n_zero > 0:
                basis = 'new'
            elif n_nan >= 0.05 * n:
                #  no boundary values AND a substantial non-computable share = the pre-change
                #  build.  The second clause matters: a panel with no adverse rows at all would
                #  otherwise be mislabelled.
                basis = 'old'
    except Exception:
        pass
    if not verbose:
        return basis
    if basis == 'new':
        print('BoMETRIC BASIS OK: %s carries the CURRENT criterion basis (%s exactly 0.0 on '
              '%d of %d rows = the adverse-Graham boundary; %d still NaN = genuine missing '
              'inputs).' % (fname, BOMETRIC_BOUNDARY_COLUMN, n_zero, n, n_nan), flush=True)
        return basis
    if basis == 'unknown':
        print('BoMETRIC BASIS UNKNOWN for %s (%s absent, or only %d row(s) -- too few for the '
              'incidence to mean anything). Not a warning; just not determinable.'
              % (fname, BOMETRIC_BOUNDARY_COLUMN, n), flush=True)
        return basis
    bar = '!' * 78
    msg = [
        '', bar,
        '!!! LOADED BoMetric PANEL PREDATES THE 2026-08-05 NaN POLICY.',
        '!!!   panel : ' + str(fname),
        '!!!   %s: 0 rows exactly 0.0, %d of %d NaN (%.1f%%).'
        % (BOMETRIC_BOUNDARY_COLUMN, n_nan, n, 100.0 * n_nan / max(1, n)),
        '!!! Consequence: this panel carries the OLD criterion columns -- `PEG` from the VENDOR',
        '!!! field on a quarter-over-quarter growth leg, and `uGrahamNumberToPrice` NaN on',
        '!!! adverse rows instead of the derived boundary -- while the rest of the run applies',
        '!!! the current code. No column was RENAMED, so calcScore\'s schema gate cannot see it.',
        '!!! Stage-1 scores are a hybrid: the PEG criterion alone moves from a 0.2050 to a',
        '!!! 0.2670 pass rate between the two bases. Re-fetch (or rebuild the panel) for',
        '!!! anything decisional.',
        bar, '',
    ]
    out = chr(10).join(msg)
    print(out, file=sys.stderr, flush=True)
    print(out, flush=True)
    return basis


def loadWrapper(type,loaddic):
    if type == 'metric':
        lbmfn = loaddic['loadBoMetricfname']
        if loaddic['loadBoMetric']:
            xdic = pd.read_pickle(lbmfn)
        else:
            xdic = initBoMetric_fromDict()
    elif type == 'results':
        lbrfn = loaddic['loadBoResultsfname']
        if loaddic['loadBoResults']:
            xdic = pd.read_pickle(lbrfn)
    else:
        raise Exception('Illegal type in loading. Only metric and results allowed')

    xdic #metricdic or resdic
    return xdic

def saveWrapper(type,savedata):
    fidag = datetime.today().strftime('%Y-%m-%d')
    tf = savedata['tickerfilter']
    sf = savedata['sectorfilter']
    lentdf = savedata['BoMetric_df']['source'].nunique()
    # `manelim<N>` in the filename is a PROVENANCE claim about this run, so it must count
    # the list that was actually APPLIED to the universe -- not the list that happened to
    # be loaded from disk (audit C-3).  'manualelim_applied' is set by Sbocker AFTER the
    # configdic spread; the fallback keeps older saved dicts (and any other caller)
    # working unchanged.
    if 'manualelim_applied' in savedata:
        nrmanelim = len(savedata['manualelim_applied'])
    else:
        nrmanelim = len(savedata['manualelimtickers'])
    lentfail = len(savedata['tickersfailed'])
    ds = savedata['datasource']
    fname_bmdf = f'Bo{type}_dic-{ds}_{tf}_{sf}_{fidag}_len{lentdf}_manelim{nrmanelim}_fails{lentfail}.pickle'
    pd.to_pickle(savedata, fname_bmdf)
    # Return the written path so callers can incrementally transfer this exact
    # artifact at the phase boundary (crash-resilience).  Backward compatible:
    # existing callers that ignore the return value are unaffected.
    return fname_bmdf

def initBoMetric_fromDict():
    preReq_dict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict,BoMetric_special_dict = cdic.getDicts()
    BMdfcollist = ['date']
    complist = ['date']
    cdxcollist = ['date']

    for key in BoMetric_Calc_dict:
        ops = BoMetric_Calc_dict[key]['Operation']
        for o in ops:
            if o == 'n':
                coln = key
            elif o == 'm':
                coln = "m" + key[0].upper() + key[1:]
            elif o == 'u':
                coln = "u" + key[0].upper() + key[1:]
            elif o == 'd':
                coln = "d" + key[0].upper() + key[1:]
            BMdfcollist.append(coln)

    for key1 in BoMetric_special_dict:
        BMdfcollist.append(key1)

    #  THE VETO CHANNEL (CEO, 2026-08-07).  Columns computed and CARRIED on the panel for
    #  `stage1_veto.POOL_FLAGS`, and NEVER scored: they come from `cdic.getVetoDict()`, which is
    #  NOT part of `getDicts`'s tuple, so `calcScore.simpleScore_fromDict` -- which iterates the
    #  five scoring dicts above -- cannot reach them.  They are appended LAST and before
    #  `source`, so no existing column's position moves.
    #  Declared here rather than left to the fetch: the schema is what tells
    #  `stage1_veto.missing_columns` whether a panel can carry a cohort's gate at all.
    for key2 in cdic.getVetoDict():
        BMdfcollist.append(key2)

    for key in preReq_dict:
        cdxcollist.extend(preReq_dict[key])

    BoTckr_count = pd.DataFrame({string: [0] for string in BMdfcollist[1:]})
    #BMdfcollist = ['date'] + BMdfcollist
    #cdxcollist = ['date'] + BMdfcollist
    BoMetric_sum = pd.DataFrame(columns=BMdfcollist)
    BMdfcollist.append('source')
    cdxcollist.append('source')
    BoMetric_df = pd.DataFrame(columns=BMdfcollist)
    cdx_df = pd.DataFrame(columns=cdxcollist)

    #period = 'quarter'
    #limit = 8
    #temp_resp = requests.get(f'{baseurl}/key-metrics/AAPL?period={period}&limit={limit + 4}&apikey={api_key}')
    #if temp_resp.status_code in range(400, 600):
    #    raise Exception('Something wrong with API, I suppose')
    #else:
    #    temp_resp_df = pd.DataFrame(temp_resp.json())

    #BoMetric_df['date'] = temp_resp_df['date']
    #BoMetric_sum['date'] = temp_resp_df['date']
    #cdx_df['date'] = temp_resp_df['date']

    metricdic = {'BoMetric_df': BoMetric_df, 'BoMetric_sum': BoMetric_sum,
                 'BoTckr_count': BoTckr_count, 'cdx_df': cdx_df}
    return metricdic

def writeManElimToFile(dmdic, manualelimtickers):
    """Write the run's exclusion list IN THE SCHEMA THE LOADER READS.

    IT USED TO WRITE ONE CSV ROW OF BARE TICKERS -- `writer.writerow(manualelimtickers)`.
    That is why writer and loader did not even share a schema: the writer emitted a headerless
    row of names and the loader (`configuration`) took `templist[0]` as the whole list, so
    nothing carried a date, a reason or an expiry, and a file written in 2023 was still
    authoritative in 2026.  This now writes `exclusions.EXCLUSION_HEADER` and the loader
    refuses anything else, so the two cannot drift apart again.

    `dmdic['exclusion_entries_next']` is the merged, dated entry list built in Sbocker.  The
    `manualelimtickers` argument is retained for the legacy call shape and is used ONLY as a
    fallback when the caller has no entries -- in which case the names are written as
    `transient_fetch`, the category that says the least, rather than silently promoted into a
    permanent ban.
    """
    import exclusions as _x
    ds = dmdic['datasource']
    fidag = datetime.today().strftime('%Y-%m-%d')
    mefn = f'ExclusionList_{ds}_{fidag}.csv'
    entries = dmdic.get('exclusion_entries_next')
    if not entries:
        entries = _x.propose_from_run(manualelimtickers, [])
    #  CARRY THE ROTATION COUNTER FORWARD (CEO, 2026-08-24).  `write_exclusions` ADVANCES it,
    #  so the next run holds out the next 10% slice and the cycle walks every slot.  The
    #  verdict is None when the run loaded no list; the counter is then omitted and the next
    #  armed run starts at slot 0 -- correct, merely not continuous.
    _xv = dmdic.get('exclusion_verdict')
    _x.write_exclusions(mefn, entries,
                        cycle=(getattr(_xv, 'cycle', None) if _xv is not None else None))
    live = [e for e in entries if e.status == 'live']
    #  THE LOOP IS SHIPPED **INERT**, AND THAT IS STATED HERE RATHER THAN LEFT TO BE DISCOVERED
    #  (review S10).  The run WRITES `ExclusionList_<ds>_<date>.csv` and the loader READS
    #  `exclusions.DEFAULT_EXCLUSION_FILE` = `ExclusionList_fmp.csv`, so the file this writes is
    #  never the file the next run reads.  That is deliberate -- it is safety lock (b), and it
    #  is why "this change alone cannot alter a run" holds even once a schema-conforming file
    #  exists on disk -- but an earlier draft described the accumulation as a working loop,
    #  which it is not until the CEO arms it.  The dated filename is also what preserves the
    #  audit trail: an undated file that round-tripped would overwrite its own history every
    #  run, which is the property that let a 2023 list masquerade as current.
    print('EXCLUSIONS WRITTEN: %r -- %d entr(ies), %d live, %d retained as expired '
          'evidence. Categories: %s\n'
          '    NOT YET ARMED (by design): the next run reads %r, which is NOT this file. '
          'To apply this list, review it, copy it to %r and pass -manelimtickers 1.'
          % (mefn, len(entries), len(live), len(entries) - len(live),
             ', '.join(sorted({e.category for e in entries})) or '(none)',
             _x.DEFAULT_EXCLUSION_FILE, _x.DEFAULT_EXCLUSION_FILE), flush=True)
    return mefn

def get_lastIndexRead(lastindex_fn):
    """Resume index for a universe's `lastIndexOfRead_<ds>_<filter>.txt`.

    THE WHITELIST IS NOW DERIVED, NOT HARDCODED (2026-08-02).  It used to be a literal
    list of four filenames, and `configuration` accepted SIX filter names -- so
    `-startfromlastindex` with `stock_US1_EU1` or `stock_US1_EU2` raised
    'Not Implemented' for a filter the CLI had just validated.  A hardcoded whitelist
    beside a growing set of universes is a defect generator: every universe added would
    be born unresumable, and the failure surfaces only on the RESUME, i.e. after a
    partial multi-hour fetch has already been paid for.
    `universes.resume_filenames()` contains every historical name verbatim, so no
    resume file already on disk is orphaned; the two previously-missing ones now work.
    """
    import universes as un
    allowedfn = un.resume_filenames('fmp')
    if lastindex_fn in allowedfn:
        if not os.path.exists(lastindex_fn):
            with open(lastindex_fn, "w") as file:
                file.write('%d' % 0)
                startindex = 0
                print('File didnt exist, but filename is allowed. I created the file and set the starting index to 0')
        else:
            with open(lastindex_fn) as f:
                lines_list = f.readlines()
                startindex = int(lines_list[0])
    else:
        raise Exception('Not Implemented: %r is not a resume file for any known universe. '
                        'Expected one of: %s' % (lastindex_fn, ', '.join(allowedfn)))

    return startindex

def write_lastIndexRead(lastindex_fn, currentIndex = 0):
    with open(f'{lastindex_fn}', 'w') as f:
        f.write('%d' % currentIndex)

    return None


def setDatesToQuarterly(df):
    df['date'] = pd.PeriodIndex(df.date, freq='Q').to_timestamp()

    return df