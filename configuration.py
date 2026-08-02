from datetime import datetime
import argparse
import csv
import os
import utils

# ===========================================================================
#  ARGUMENT PARSING (argparse refactor, 2026-08-02)
#  --------------------------------------------------------------------------
#  This file used to parse `args` by hand with the idiom
#
#       if '-flag' in args:
#           i = args.index('-flag')
#           value = args[i + 1]
#
#  which produced THREE real defects, because the index variable and the flag
#  it belongs to are only related by a name the reader has to check:
#    * -compyear      read args[id + 1]  (`id`  = the -datasource index)
#    * -sectorfilter  read args[imb + 1] (`imb` = the -mcapBelow  index)
#    * -manelimfilename read args[ibmfn + 1] (audit C-3, 2026-07-19)
#  plus a LATENT instance: `ima` was bound by BOTH -mcapAbove and -fsMAnumber
#  and was correct only because each use happened to follow its own binding,
#  so reordering the blocks would have silently broken it.
#
#  argparse removes the class outright: a flag's value can no longer be
#  fetched through another flag's index because there are no indexes.
#
#  FOUR BEHAVIOURS OF THE OLD PARSER ARE DELIBERATELY PRESERVED -- argparse
#  does NOT give any of them by default, and each would be a regression:
#
#   1. IT NEVER EXITS THE PROCESS.  `_RaisingArgumentParser` turns argparse's
#      `error()`/`exit()` into a raised Exception.  Stock argparse prints a
#      usage message and calls sys.exit(2); this module is IMPORTED by tests
#      and by Sbocker/backtest/portfolio/normalized_analysis/delisted_ingest,
#      so a hard exit on a bad argument would kill a caller that today gets a
#      catchable Exception.
#   2. UNKNOWN TOKENS ARE IGNORED.  `parse_known_args` keeps the old
#      leniency: `'-flag' in args` simply never looked at anything else, so
#      unknown flags and stray positionals passed silently.  Callers rely on
#      this -- the test-suite convention is a leading dummy token
#      (`['x', '-compyear', 'thisYear']`), which stock `parse_args` would
#      reject as "unrecognized arguments".
#   3. NO -h/--help INTERCEPT (`add_help=False`).  Today `-h` is just an
#      unknown token and is ignored; argparse's default help action would
#      print usage and exit the process.
#   4. FIRST OCCURRENCE WINS.  `args.index()` returns the FIRST match, so
#      `-period quarter -period annual` yielded 'quarter'.  argparse keeps the
#      LAST value by default, so every valued flag uses action='append' and
#      reads element [0] via `_first_value` to hold the old semantics.
#
#  `allow_abbrev=False` because the old parser matched flags EXACTLY: it must
#  not start accepting `-nrp` for `-nrperiods`.
# ===========================================================================


class _RaisingArgumentParser(argparse.ArgumentParser):
    """An ArgumentParser that RAISES instead of terminating the interpreter.

    Preserved behaviour #1 above.  `configuration` is an imported module, and
    every current caller passes an explicit list (`[]`, `sys.argv[1:]`, or a
    test's literal) and expects a catchable Exception on a bad argument -- not
    SystemExit.  SystemExit is a BaseException, so it would also slip straight
    through the `except Exception` handlers used elsewhere in the pipeline and
    through `pytest.raises(Exception)` in the suite.
    """

    def error(self, message):
        raise Exception('configuration argument error: %s' % message)

    def exit(self, status=0, message=None):
        raise Exception('configuration argument error: %s'
                        % (message or 'argument parsing aborted'))


#  Valued flags: action='append' + nargs='?' so that
#    * a repeated flag keeps its FIRST value (preserved behaviour #4), and
#    * a flag given with NO value yields [None] rather than an argparse exit,
#      which lets each flag raise its OWN historical error message below.
_VALUED = dict(action='append', nargs='?', default=None)
#  Presence-only flags: truth comes from PRESENCE, never from a value.
_PRESENT = dict(action='store_true', default=False)


def _build_parser():
    p = _RaisingArgumentParser(prog='configuration', add_help=False,
                               allow_abbrev=False)
    for flag in ('-tickerfilter', '-datasource', '-mcapAbove', '-mcapBelow',
                 '-sectorfilter', '-period', '-nrperiods', '-nrTaT', '-compyear',
                 '-fsMAnumber', '-nrScorePeriods', '-ntopagg', '-ntopxlsx',
                 '-savebometric', '-saveboresults', '-loadbometric',
                 '-loadboresults', '-symbolChangeRestock', '-bometricfilename',
                 '-boresultsfilename', '-manelimtickers', '-manelimfilename',
                 '-asof', '-delisted_max_pages', '-portfolioTest',
                 '-backtest_buy_years', '-backtest_eval_years', '-backtest_topn',
                 '-run_estimation', '-transfer_dir'):
        p.add_argument(flag, **_VALUED)
    for flag in ('-newOnly', '-ingest_delisted', '-startfromlastindex',
                 '-runbacktest', '-no_transfer'):
        p.add_argument(flag, **_PRESENT)
    return p


def _given(ns, dest):
    """True iff the flag APPEARED at all (the old `'-flag' in args` test)."""
    return getattr(ns, dest) is not None


def _first_value(ns, dest):
    """The FIRST value given for a valued flag, or None if it carried no value.

    Preserved behaviour #4: `args.index()` found the first occurrence, so a
    repeated flag must keep its first value, not argparse's default last one.
    """
    seq = getattr(ns, dest)
    return None if seq is None else seq[0]


def _require(ns, dest, missing_msg):
    """Value of a valued flag, raising `missing_msg` when the value is absent.

    The old code bounds-checked FIVE flags with a bespoke message (-asof,
    -manelimtickers, -manelimfilename, -run_estimation, -transfer_dir) and left
    the rest to fail with an opaque `IndexError: list index out of range` when
    the flag was the final token.  Every flag now reports in that same
    bespoke style; the five original messages are preserved VERBATIM because
    they are the ones already documented and pinned by tests.
    """
    value = _first_value(ns, dest)
    if value is None:
        raise Exception(missing_msg)
    return value


def getDataFetchConfiguration(args):
    # Tokenise ONCE, up front.  `parse_known_args` discards unrecognised
    # tokens exactly as the old `'-flag' in args` scan did (preserved
    # behaviour #2); `_unknown` is intentionally unused.
    ns, _unknown = _build_parser().parse_known_args(list(args))

    # Assign tickerfilter
    tickerfilterlist = ['stock_NA1', 'stock_US1', 'stock_WW1_TV','stock_NA1_EU1', 'stock_US1_EU1', 'stock_US1_EU2']
    if _given(ns, 'tickerfilter'):
        tickerfilter = _require(ns, 'tickerfilter', '-tickerfilter requires an argument')
        if tickerfilter not in tickerfilterlist:
            raise Exception('-tickerfilter argument not valid')
    else:
        tickerfilter = tickerfilterlist[3]

    # Assign datasource
    datasourcelist = ['fmp']
    if _given(ns, 'datasource'):
        datasource = _require(ns, 'datasource', '-datasource requires an argument')
        if datasource not in datasourcelist:
            raise Exception('-datasource argument not valid')
    else:
        datasource = datasourcelist[0]
    # Getting the associated API baseurl and setting the api_key, for the datasource 'fmp'
    if datasource == 'fmp':
        api_key_fname = 'fmpAPIkey.txt'
        api_key = open('fmpAPIkey.txt', 'r').read()
        baseurl = "https://financialmodelingprep.com/api/"

    # Assign filtering on market cap band to filter
    # NOTE: both values are overwritten with -1 immediately below (the flags are
    # not implemented).  The int() conversion is kept because it is REACHABLE
    # behaviour: `-mcapAbove abc` raises ValueError today and must keep doing so.
    if _given(ns, 'mcapAbove'):
        print('-mcapAbove not yet implemented. Will be ignored')
        mcapUL = int(_require(ns, 'mcapAbove', '-mcapAbove requires an argument'))
    else:
        mcapUL = -1
    if _given(ns, 'mcapBelow'):
        print('-mcapBelow not yet implemented. Will be ignored')
        mcapLL = int(_require(ns, 'mcapBelow', '-mcapBelow requires an argument'))
    else:
        mcapLL = -1

    mcapUL, mcapLL = [-1,-1]

    #sectorlist = ['all', 'Basic Materials', 'Healthcare', 'Financial Services', 'Energy', 'Consumer Cyclical',
    # 'Biotechnology', 'Consumer Defensive', 'Pharmaceuticals', 'Industrials', 'Communication Services', 'Technology',
    # 'Real Estate', 'Utilities', 'Media', 'Hotels, Restaurants & Leisure', 'Food Products', 'Machinery',
    # 'Electrical Equipment', 'Commercial Services & Supplies', 'Semiconductors', 'Construction',
    # 'Textiles, Apparel & Luxury Goods', 'Metals & Mining', 'Retail', 'Logistics & Transportation', 'Road & Rail',
    # 'Chemicals', 'Professional Services', 'Insurance', 'Airlines', 'Aerospace & Defense', 'Telecommunication',
    # 'Services', 'Consumer Goods', 'Trading Companies & Distributors', 'Banking', 'Consumer products', 'Packaging',
    # 'Conglomerates']
    sectorlist = ['all', 'Unspecified', 'Basic Materials', 'Healthcare', 'Financial Services',
                  'Energy', 'Consumer Cyclical', 'Consumer Defensive', 'Industrials',
                  'Communication Services', 'Technology', 'Real Estate', 'Utilities']
    if _given(ns, 'sectorfilter'):
        print('Limited implementation of sector filter')
        # WRONG-INDEX BUG (fixed 2026-07-31): this read args[imb+1] -- `imb` is the
        # -mcapBelow index, so `isf` was computed and DISCARDED.  Reproduced: `-sectorfilter
        # Technology` alone raised UnboundLocalError (imb unbound); with -mcapBelow also
        # present it read the -mcapBelow VALUE and raised '-sectorfilterr argument not valid'.
        # Unusable in every form.  Same family as the -compyear defect below and as the
        # audit's C-3/L-5 fixes.  The 2026-08-02 argparse refactor makes the whole family
        # unrepresentable: there is no index to get wrong.
        sectorfilter = _require(ns, 'sectorfilter', '-sectorfilter requires an argument')
        if sectorfilter not in sectorlist:
            raise Exception('-sectorfilter argument not valid')
    else:
        sectorfilter = 'all'


    #Assign period of the data
    periodlist = ['quarter', 'annual']
    if _given(ns, 'period'):
        period = _require(ns, 'period', '-period requires an argument')
        if period not in periodlist:
            raise Exception('-period argument is not valid')
    else:
        period = 'quarter'

    #Assign number of periods to fetch
    if _given(ns, 'nrperiods'):
        nrperiods = int(_require(ns, 'nrperiods', '-nrperiods requires an argument'))
    else:
        nrperiods = 6 * 4

    #nr of Tickers at a Time
    if _given(ns, 'nrTaT'):
        nrTaT = int(_require(ns, 'nrTaT', '-nrTaT requires an argument'))
    else:
        nrTaT = -1

    # Get comparison year (default last year)
    if _given(ns, 'compyear'):
        # WRONG-INDEX BUG (fixed 2026-07-31): this read args[id+1] -- `id` is the -datasource
        # index, so `ic` was computed and DISCARDED.  Reproduced: `-compyear thisYear` alone
        # raised UnboundLocalError; with `-datasource fmp` it read args[1] ('fmp') and raised
        # 'compyear argument is not valid'.  Unusable in every form -- and `compyear` is
        # LOAD-BEARING: it drives the datefail gate (failTests.py), which is what rejects a
        # ticker whose newest statement predates the comparison year.
        compyearstr = _require(ns, 'compyear', '-compyear requires an argument')
        if compyearstr == 'lastYear':
            compyear = datetime.now().year - 1
        elif compyearstr == 'thisYear':
            compyear = datetime.now().year
        else:
            raise Exception('compyear argument is not valid')
    else:
        compyear = datetime.now().year - 1

    # Set how many datapoints used in moving average of the entries of the financial statments fetched
    if _given(ns, 'fsMAnumber'):
        fsMAnumber = int(_require(ns, 'fsMAnumber', '-fsMAnumber requires an argument'))
    else:
        fsMAnumber = 1

    # Set number of periods used in averaging when calculating score for each metric
    if _given(ns, 'nrScorePeriods'):
        nrScorePeriods = int(_require(ns, 'nrScorePeriods', '-nrScorePeriods requires an argument'))
    else:
        nrScorePeriods = 8

    # number of stocks in top list and the presentation, respectively
    if _given(ns, 'ntopagg'):
        ntopagg = int(_require(ns, 'ntopagg', '-ntopagg requires an argument'))
    else:
        ntopagg =  100

    if _given(ns, 'ntopxlsx'):
        ntopxlsx = int(_require(ns, 'ntopxlsx', '-ntopxlsx requires an argument'))
    else:
        ntopxlsx = 20

    # PRESENCE-ONLY flag: 1 iff -newOnly appears.  It takes NO value, so no
    # value can ever be misread as a truth (see the -manelimtickers note below).
    newOnly = 1 if ns.newOnly else 0

    # Assign values to saving and loading bools
    # VALUED flags (not presence): `-savebometric 0` must DISABLE.  The
    # `int(...) > 0` test is a TRUTH test on the value, deliberately NOT a
    # presence test -- the ALLOW_MERGE_CONTENT_MISMATCH footgun (fixed
    # 2026-07-31) was exactly a presence test standing in for a truth test,
    # where `=0` ACTIVATED the override.  Kept verbatim.
    if _given(ns, 'savebometric'):
        saveBoMetric = 1 if int(_require(ns, 'savebometric', '-savebometric requires an argument')) > 0 else 0
    else:
        saveBoMetric = 1

    # Assign booleans on saving and loading
    if _given(ns, 'saveboresults'):
        saveBoResults = 1 if int(_require(ns, 'saveboresults', '-saveboresults requires an argument')) > 0 else 0
    else:
        saveBoResults = 1

    if _given(ns, 'loadbometric'):
        loadBoMetric = 1 if int(_require(ns, 'loadbometric', '-loadbometric requires an argument')) > 0 else 0
    else:
        loadBoMetric = 0

    if _given(ns, 'loadboresults'):
        loadBoResults = 1 if int(_require(ns, 'loadboresults', '-loadboresults requires an argument')) > 0 else 0
    else:
        loadBoResults = 0

    if loadBoMetric:
        if saveBoMetric:
            print('Since loadBoMetric is set to unity, saveBoMetric is disabled')
            saveBoMetric = 0
    if loadBoResults:
        if saveBoResults:
            print('Since loadBoResults is set to unity, saveBoResults is disabled')
            saveBoResults =  0

    # Set boolean that determines whether symbol changes are affecting fetched data
    if _given(ns, 'symbolChangeRestock'):
        symbchRestock = 1 if int(_require(ns, 'symbolChangeRestock', '-symbolChangeRestock requires an argument')) > 0 else 0
    else:
        symbchRestock = 0

    # Assign loading filenames of Metrics, Results and elimination list of Tickers
    if _given(ns, 'bometricfilename'):
        loadBoMetricfname = _require(ns, 'bometricfilename', '-bometricfilename requires a filename argument')
    else:
        loadBoMetricfname = 'Bometric_dic-fmp_stock_NA1_EU1_all_2023-03-16_len6728_manelim3692_fails6729.pickle'

    if _given(ns, 'boresultsfilename'):
        loadBoResultsfname = _require(ns, 'boresultsfilename', '-boresultsfilename requires a filename argument')
    else:
        loadBoResultsfname = 'Boresults_dic-fmp_stock_NA1_EU1_all_2023-03-16_len6728_manelim3692_fails6729.pickle'

    # Assign boolean and filename to manual elimination of ticker symbols before fetching data
    if _given(ns, 'manelimtickers'):
        # int(): argv values are STRINGS and the string '0' is TRUTHY, so
        # `-manelimtickers 0` used to switch the filter ON.  This flag only became
        # reachable once the -manelimfilename else-branch stopped force-setting it to 1.
        # This is a VALUED flag for that reason -- do NOT turn it into a presence flag.
        manelimtickersbool = int(_require(ns, 'manelimtickers',
                                          '-manelimtickers requires a 0/1 argument'))
    else:
        manelimtickersbool = 0

    # THREE bugs fixed here (audit C-3 / L-5, 2026-07-19):
    #   1. the filename was read from args[ibmfn + 1] -- ibmfn is the
    #      -bometricfilename index, so -manelimfilename either picked up the WRONG
    #      argument or raised (imefn was computed and then never used);
    #   2. no bounds check, so -manelimfilename as the FINAL arg raised an opaque
    #      IndexError (same pattern as the -asof guard below);
    #   3. the else branch FORCED manelimtickersbool = 1, so NOT passing the flag
    #      turned manual elimination ON with a hardcoded 2023 file and silently
    #      overrode whatever -manelimtickers had said.  That is how the stale 3,692-name
    #      list came to be loaded on every run.  Omitting the flag now leaves
    #      -manelimtickers in charge and only supplies the DEFAULT filename.
    if _given(ns, 'manelimfilename'):
        manelimtick_fname_toget = _require(ns, 'manelimfilename',
                                           '-manelimfilename requires a filename argument')
    else:
        manelimtick_fname_toget = 'ManualEliminationTickersList_fmp_2023-02-14.csv'

    # Point-in-time as-of date D (design 2026-07-12 restructure).  Default None =
    # today / live run (reproduces current behaviour bit-for-bit).  Pass an ISO date
    # (YYYY-MM-DD) to run the pipeline as-of that past date (survivorship-safe PIT
    # universe + availability-date metric slice).  Tonight's full deep-fetch is a
    # LIVE run -> omit this flag (as_of stays None).
    if _given(ns, 'asof'):
        # LOW-B fix: bounds-check so `-asof` as the FINAL arg (no date) raises a
        # clear error instead of an opaque IndexError.
        as_of = _require(ns, 'asof', '-asof requires a date argument (YYYY-MM-DD)')
        # MEDIUM-B guard (review addendum 2): the -asof path is only PARTIALLY
        # point-in-time.  simpleScore_fromDict applies the row-level availability
        # slice (L1/L4), but the cross-sectional baseline bm_ave / getAves2 (L2) and
        # the per-ticker means (L3) are still computed over the FULL panel, and
        # DCF/beta (L5/L6) are not substituted.  So a -asof run STILL embeds L2/L3
        # lookahead and must NOT be treated as clean PIT.  Warn loudly so a
        # partial-PIT run is never mistaken for a clean one.  (Tonight is as_of=None
        # -> this never fires on the live run; the guard is wired for when -asof is
        # used later.)
        import warnings as _w
        _w.warn(
            "PARTIAL-PIT: -asof applies ONLY the row-level availability slice "
            "(L1/L4). The cross-sectional baseline (L2, getAves2/bm_ave), the "
            "per-ticker means (L3), and DCF/beta substitution (L5/L6) are NOT yet "
            "point-in-time -- this run STILL embeds L2/L3 lookahead. Do NOT treat "
            "its output as clean point-in-time.")
    else:
        as_of = None

    # -ingest_delisted (default OFF): gate for the survivorship / delisted-entity
    # ingestion (delisted_ingest.run_ingest).  When OFF the ingestion module is
    # never imported and the live path is untouched / bit-for-bit.  Turn ON for the
    # full survivorship deep-fetch.  Optional -delisted_max_pages bounds the
    # registry pagination guard.
    # PRESENCE-ONLY (both -ingest_delisted and -startfromlastindex).
    ingest_delisted = 1 if ns.ingest_delisted else 0
    if _given(ns, 'delisted_max_pages'):
        delisted_max_pages = int(_require(ns, 'delisted_max_pages',
                                          '-delisted_max_pages requires an argument'))
    else:
        delisted_max_pages = 500
    startfromlastindex = 1 if ns.startfromlastindex else 0

    if _given(ns, 'portfolioTest'):
        # NOTE: stays a STRING when given, while the default is the INT -1.
        # That mixed type is pre-existing and is preserved deliberately.
        portfoliotestyear = _require(ns, 'portfolioTest', '-portfolioTest requires an argument')
    else:
        portfoliotestyear = -1

    # Unified backtesting parameters
    runbacktest = 1 if ns.runbacktest else 0

    if _given(ns, 'backtest_buy_years'):
        backtest_buy_years = [int(y) for y in _require(
            ns, 'backtest_buy_years', '-backtest_buy_years requires an argument').split(',')]
    else:
        backtest_buy_years = None  # Will use defaults in backtest_unified

    if _given(ns, 'backtest_eval_years'):
        backtest_eval_years = [int(y) for y in _require(
            ns, 'backtest_eval_years', '-backtest_eval_years requires an argument').split(',')]
    else:
        backtest_eval_years = None

    if _given(ns, 'backtest_topn'):
        backtest_topn = int(_require(ns, 'backtest_topn', '-backtest_topn requires an argument'))
    else:
        backtest_topn = 100

    # -run_estimation (VALUED, default 0): gate for the HEAVY estimation sub-block
    # inside the post-pick analysis suite (tuner / tune_run / rebalance_engine + the
    # depth-grid weight/carve tuning sweeps).  DEFAULT 0 -> the estimation block is
    # SKIPPED entirely.  The grading / IC / depth-grid / beat-rate / oracle / random
    # READOUTS still run by default (each self-guarded), because they are cheap and
    # measure tonight's model; only the expensive parameter-SEARCH is behind this gate.
    # Same valued-arg shape as -backtest_topn (read via configdic.get('run_estimation')).
    if _given(ns, 'run_estimation'):
        run_estimation = int(_require(ns, 'run_estimation',
                                      '-run_estimation requires an integer argument (0 or 1)'))
    else:
        run_estimation = 0

    # ---- Drive transfer target resolution (OPT-OUT, default ON) -------------
    # Transfer now runs BY DEFAULT (opt-out).  The pipeline copies the output
    # allowlist to the Google-Drive-synced folder at end-of-run so run outputs
    # reach the house automatically.  The previous behaviour was OPT-IN via
    # -transfer_dir, and omitting the flag silently did NO transfer -- last
    # night's run lost its outputs that way (2026-07-16 fix: never silently
    # not-transfer).
    #
    # Enable / disable -- ONLY an EXPLICIT opt-out turns transfer off.  The point
    # of opt-out is that you must DELIBERATELY opt out; an accidentally-empty env
    # var is a mistake, not an intent to disable, so it falls through to ON:
    #   (default, no flag)          -> transfer ON  (default target)
    #   -no_transfer                -> transfer OFF (explicit toggle)
    #   value == 'none' (flag/env)  -> transfer OFF (explicit token; case/space-insensitive)
    #   -transfer_dir <path>        -> transfer ON, target overridden to <path>
    #   empty / whitespace / unset  -> transfer ON, falls through to default (NEVER off)
    #
    # Target resolution order when transfer is ON and no explicit -transfer_dir
    # <path> is given:
    #   1. env var VALUATION_TRANSFER_DIR, if set to a real (non-empty) value
    #   2. documented operator-runbook default: E:\drive\valuationTransfer
    #
    # configdic carries:
    #   'transfer_dir'             -> resolved target path (str) when ON, else None
    #   'transfer_disabled_reason' -> None when ON; the disabling flag string when OFF
    DEFAULT_TRANSFER_DIR = r'E:\drive\valuationTransfer'
    transfer_dir = None
    transfer_disabled_reason = None

    explicit_transfer_dir = None
    if _given(ns, 'transfer_dir'):
        explicit_transfer_dir = _require(ns, 'transfer_dir',
                                         '-transfer_dir requires a directory path argument')

    # 'none' (after strip+lowercase) is the ONLY explicit disable token -- applied
    # IDENTICALLY to the flag AND the env var.  Empty/whitespace is NOT a disable:
    # it falls through so an accidentally-blank value can never silently turn
    # transfer off (never-miss intent).
    # -no_transfer is PRESENCE-ONLY: it takes no value, so `-no_transfer 0`
    # cannot mean "do transfer" -- the value would be an ignored stray token,
    # exactly as before.
    if ns.no_transfer:
        transfer_disabled_reason = '-no_transfer'
    elif explicit_transfer_dir is not None and explicit_transfer_dir.strip().lower() == 'none':
        transfer_disabled_reason = '-transfer_dir none'
    elif explicit_transfer_dir is not None and explicit_transfer_dir.strip():
        transfer_dir = explicit_transfer_dir.strip()
    else:
        # No usable -transfer_dir (unset, or empty/whitespace) -> env, then default.
        env_dir = os.environ.get('VALUATION_TRANSFER_DIR')
        if env_dir is not None and env_dir.strip().lower() == 'none':
            transfer_disabled_reason = 'VALUATION_TRANSFER_DIR=none'
        elif env_dir is not None and env_dir.strip():
            transfer_dir = env_dir.strip()
        else:
            transfer_dir = DEFAULT_TRANSFER_DIR

    # Sanity guard: a resolved target that is NOT an absolute path would create a
    # stray junk dir under the run's cwd and could falsely look like success.
    # Warn loudly here (the launch probe additionally refuses to create it).
    if transfer_dir is not None and not os.path.isabs(transfer_dir):
        print("[TRANSFER] WARNING: resolved transfer target %r is NOT an absolute "
              "path -- this could create a stray directory under the run's working "
              "dir. Check VALUATION_TRANSFER_DIR / -transfer_dir." % transfer_dir)

    # Skip loading manual elimination CSV when loading metrics (it's already in the pickle file)
    if loadBoMetric:
        manualelimtickers = []
    elif manelimtickersbool:
        with open(manelimtick_fname_toget, 'r') as file:
            reader = csv.reader(file)
            templist = list(reader)
            manualelimtickers = templist[0]
    else:
        manualelimtickers = []

    # Inform of consistency
    if loadBoMetric or loadBoResults:
        print('Note that loading might overwrite other arguments.')

    lastindex_fn = 'lastIndexOfRead_' + datasource + '_' + tickerfilter + '.txt'
    if ns.startfromlastindex:
        startindex = utils.get_lastIndexRead(lastindex_fn)
    else:
        startindex = 0

    # get the starting index for getting data for fundamentals


    configdic = {'tickerfilter': tickerfilter, 'datasource': datasource, 'baseurl': baseurl, 'api_key': api_key,
                 'period': period, 'nrperiods': nrperiods, 'nrTaT': nrTaT, 'compyear': compyear, 'newOnly': newOnly,
                 'fsMAnumber': fsMAnumber, 'startindex': startindex, 'mcapUL': mcapUL, 'mcapLL': mcapLL,
                 'saveBoMetric': saveBoMetric, 'saveBoResults': saveBoResults, 'loadBoMetric': loadBoMetric,
                 'loadBoResults': loadBoResults, 'symbchRestock': symbchRestock, 'loadBoMetricfname': loadBoMetricfname,
                 'loadBoResultsfname': loadBoResultsfname, 'manualelimtickers': manualelimtickers,
                 # recorded so the run can STATE which list it loaded (manual-elim provenance)
                 'manualelimtick_fname_toget': manelimtick_fname_toget,
                 'manelimtickersbool': manelimtickersbool,
                 'lastindex_fn': lastindex_fn, 'nrScorePeriods': nrScorePeriods, 'ntopagg': ntopagg,
                 'ntopxlsx': ntopxlsx, 'sectorfilter': sectorfilter, 'portfoliotestyear': portfoliotestyear,
                 'sectorlist': sectorlist,
                 'runbacktest': runbacktest, 'backtest_buy_years': backtest_buy_years,
                 'backtest_eval_years': backtest_eval_years, 'backtest_topn': backtest_topn,
                 'as_of': as_of, 'ingest_delisted': ingest_delisted,
                 'delisted_max_pages': delisted_max_pages,
                 'startfromlastindex': startfromlastindex, 'transfer_dir': transfer_dir,
                 'transfer_disabled_reason': transfer_disabled_reason,
                 'run_estimation': run_estimation}

    return configdic
