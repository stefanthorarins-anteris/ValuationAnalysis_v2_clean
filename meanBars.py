"""meanBars.py  --  THE ABSOLUTE BARS FOR THE STAGE-1 `mean` FAMILY  (register C-12, CEO 2026-08-06)

WHAT THIS REPLACES, AND WHY IT IS NOT A TUNING CHANGE
-----------------------------------------------------
`calcScore.calcByTier('mean', ...)` scores `Sign * (value - avec) > 0`, and `avec` was
`getAves2`'s `res_fullMean` -- the MEDIAN OF THE COLUMN POOLED ACROSS EVERY COMPANY AND
EVERY DATE in the panel being scored.  Three separate defects, and only the third is fatal:

  1. IT IS NOT A BAR, IT IS A RANK.  A criterion whose threshold is the pool's own median
     passes ~half the pool BY CONSTRUCTION, whatever the pool is worth.  In a market where
     every name is expensive, "cheaper than the median name" is still expensive; the
     criterion cannot say so.
  2. IT MOVES WITH THE SAMPLE.  Add an exchange, change the manual-elimination list, or
     re-run on a different universe, and every name's pass/fail on seven criteria moves
     without any company's fundamentals changing.  Two runs are then not comparable.
  3. IT IS A LOOKAHEAD CHANNEL.  The median is pooled over EVERY DATE IN THE PANEL, so a
     2015 row is scored against a bar computed partly from 2025 data.  In any backtest that
     is future information reaching a past decision -- the one defect class that can
     manufacture a beat-rate out of nothing.

So the bars below are STORED CONSTANTS.  They are not fitted and they are not tuned: each
one is a STATED ECONOMIC CLAIM about what "good enough" means for that criterion, and the
measured pass rate is recorded ALONGSIDE it as evidence that the claim lands somewhere
sane on today's panel -- NOT as the thing that set it.  That distinction is the whole
point; a bar set by aiming at a pass rate is the pooled median again with extra steps.

THE ARITHMETIC IS UNCHANGED -- DO NOT "FIX" THE FLOW LEG
--------------------------------------------------------
`reporting_period.STAGE1_FLOW_CORRECTION` classifies `earningsYield`,
`freeCashFlowToMarketCap` and `salesToMarketCap` as SCALE-FREE -> `per_quarter`, because a
cross-sectional test against a pooled median is invariant to a global constant.  That
classification is STILL CORRECT after this change, and the reason is worth stating because
the obvious reading is the wrong one: the mode `per_quarter` vs `annualize` describes what
basis the COLUMN is put on, and the bars here are STATED ON THE PER-QUARTER BASIS THAT
FACTOR PRODUCES.  A bar being absolute does NOT mean the column must be annualised -- it
means the bar and the column must agree about the basis, and they do.

Switching either yield leg to `annualize` would multiply the column by 4 against an
unchanged bar and SILENTLY QUADRUPLE the effective threshold (the 4%/yr earnings-yield bar
would become 16%/yr).  If you ever change that table, change these two entries in the same
edit -- and `test_mean_bars.py` will fail if you change one without the other.

THE TWO YIELD BARS ARE DELIBERATELY DIFFERENT (CEO, 2026-08-06).  `mEarningsYield` is
4.0%/yr and `mFreeCashFlowToMarketCap` is 3.0%/yr.  They look like a matched pair and they
are not: the earnings bar is set to a CLASSICAL ANCHOR (P/E <= 25, Graham's 2x-AAA) and
carries the gate's value stance; the FCF bar is a floor on cash reality with no such anchor.
Do not harmonise them -- see both `rationale` fields, and the test that pins the asymmetry.

The two yield bars are therefore written as `<annual rate> / rp.DEFAULT_ROWS_PER_YEAR`
rather than as a bare decimal, so the basis is legible at the point of definition and the
divisor is the SAME constant `per_quarter_factor` divides by.

WHAT IS NOT HERE
----------------
`salesToMarketCap` is Tier 'N' (w = 0) and is deliberately EXCLUDED -- see `NO_BAR`.  It
keeps the pooled median, which costs nothing because a weightless criterion contributes 0
to the score either way.  Giving an inert criterion a grounded bar would be inventing an
economic claim nobody needs.

REPORTING A BREACH IS NOT PROPOSING A RE-SET  (2026-08-07)
----------------------------------------------------------
The failsafe band emits two different kinds of statement and they carry DIFFERENT EVIDENCE
BARS.  `breach` says "on the cells in front of me, this bar's pass rate is outside
[25%, 75%]" -- true or false at any sample size, so it is always computed and always
reported.  `proposed_constant` says "the bar is wrong and here is a better number" -- a
claim about the world, which needs a representative universe, a persistent breach and the
production seam.  `advisory` gates the SECOND ONLY.  See MIN_FULL_UNIVERSE_SOURCES for the
defect that made this explicit.

TO CHANGE A BAR: edit `value` in ONE place here.  Nothing else mirrors these numbers.
"""

import numpy as np
import pandas as pd

import reporting_period as rp

#  Quarters per year -- the SAME constant `rp.per_quarter_factor` divides by, imported
#  rather than written as 4 so the two can never disagree.
_Q = float(rp.DEFAULT_ROWS_PER_YEAR)

#  --- THE BARS ---------------------------------------------------------------------------
#  Keyed by the BoMetric panel's `m`-prefixed COLUMN NAME (what `calcScore` has in hand),
#  not by the `createDicts.BoMetric_mean_dict` key.
#
#  Each entry:
#    value            the bar, on the panel column's OWN basis (per-quarter for the flows)
#    units            what the number is, in words -- the basis is load-bearing, see above
#    rationale        the ECONOMIC claim.  This is what set the number.
#    pass_rate_at_set measured share of observed cells passing, on the panel named below.
#                     EVIDENCE THAT THE CLAIM IS SANE, NOT THE THING THAT SET IT.
#    panel_at_set     which panel that measurement was taken on
#    date_set         when
#    round_to         quantum the CALIBRATION report rounds a proposed re-set to (its own
#                     units -- see `annual_basis`)
#    annual_basis     True when the bar is naturally quoted as an ANNUAL rate, so the
#                     calibration report rounds the ANNUAL number and divides by _Q
BARS = {
    'mBookToPrice': {
        'value': 0.50,
        'units': 'book equity / market cap (stock/stock, no flow correction)',
        'rationale': 'P/B <= 2.0.  Pay at most twice book for the equity.',
        'pass_rate_at_set': 0.511,
        'panel_at_set': '2026-08 full universe, newest-8 window',
        'date_set': '2026-08-06',
        'round_to': 0.05,
        'annual_basis': False,
    },
    'mEarningsYield': {
        #  4.0%/yr on a PER-QUARTER column -- see the flow-leg section of the docstring.
        'value': 0.04 / _Q,
        'units': 'earnings / market cap, PER QUARTER (annual rate / %g)' % _Q,
        'rationale': '4.0%/yr earnings yield => P/E <= 25, roughly Graham\'s 2x-AAA anchor. '
                     'CHOSEN OVER THE ~50%-PASS ALTERNATIVE (3.0%/yr, P/E <= 33.3) BY THE CEO, '
                     'and the reason is the one thing a pass rate cannot express: the GATE '
                     'itself should carry a value stance rather than delegating all cheapness '
                     'to Stage-2.  So 43.5% is the RIGHT pass rate here, not a calibration '
                     'miss -- and it sits comfortably inside the 25-75% failsafe band. '
                     'DELIBERATELY NOT THE SAME LEVEL AS `mFreeCashFlowToMarketCap` (3.0%/yr): '
                     'the earnings bar carries the value stance and has a classical anchor to '
                     'stand on; the FCF bar is a floor on cash reality with no such anchor. '
                     'DO NOT HARMONISE THEM -- the asymmetry is the decision.',
        'pass_rate_at_set': 0.435,
        'panel_at_set': '2026-08 full universe, newest-8 window',
        'date_set': '2026-08-06',
        'round_to': 0.005,          # 0.5pp, on the ANNUAL rate
        'annual_basis': True,
    },
    'mDebtEquityRatio': {
        'value': 0.50,
        'units': 'total debt / book equity (stock/stock)',
        'rationale': 'Debt no more than half of book equity.  Sign is -1, so the test '
                     'is `ratio < 0.50`.',
        'pass_rate_at_set': 0.504,
        'panel_at_set': '2026-08 full universe, newest-8 window',
        'date_set': '2026-08-06',
        'round_to': 0.05,
        'annual_basis': False,
    },
    'mFreeCashFlowToMarketCap': {
        #  3.0%/yr on a PER-QUARTER column -- see the flow-leg section of the docstring.
        'value': 0.03 / _Q,
        'units': 'free cash flow / market cap, PER QUARTER (annual rate / %g)' % _Q,
        'rationale': '3.0%/yr FCF yield -- a floor on cash reality, not a value stance. '
                     'DELIBERATELY LOWER THAN `mEarningsYield` (4.0%/yr): that bar is set to '
                     'a classical P/E anchor and is the one carrying the gate\'s value '
                     'stance; this one has no such anchor and is not trying to. The two '
                     'yield bars are NOT a matched pair -- do not tidy them into one number.',
        'pass_rate_at_set': 0.487,
        'panel_at_set': '2026-08 full universe, newest-8 window',
        'date_set': '2026-08-06',
        'round_to': 0.005,          # 0.5pp, on the ANNUAL rate
        'annual_basis': True,
    },
    'mEquityToAssets': {
        'value': 0.40,
        'units': 'book equity / total assets (stock/stock)',
        'rationale': 'Liabilities <= 1.5x equity.  E/A >= 0.40 <=> L/E <= 1.5.',
        'pass_rate_at_set': 0.531,
        'panel_at_set': '2026-08 full universe, newest-8 window',
        'date_set': '2026-08-06',
        'round_to': 0.05,
        'annual_basis': False,
    },
    'mGrossProfitMargin': {
        'value': 0.40,
        'units': 'gross profit / revenue (ratio, already unitless)',
        'rationale': 'The non-commodity line: 40% gross margin is roughly where a business '
                     'is selling something other than a price.',
        'pass_rate_at_set': 0.510,
        'panel_at_set': '2026-08 full universe, newest-8 window',
        'date_set': '2026-08-06',
        'round_to': 0.05,
        'annual_basis': False,
    },
    'mNetProfitMargin': {
        'value': 0.05,
        'units': 'net income / revenue (ratio, already unitless)',
        'rationale': 'Modestly profitable after everything.  Low deliberately -- this is a '
                     'floor, and the margin LEVEL that matters is industry-specific.',
        'pass_rate_at_set': 0.487,
        'panel_at_set': '2026-08 full universe, newest-8 window',
        'date_set': '2026-08-06',
        'round_to': 0.01,
        'annual_basis': False,
    },
}

#  Mean-family columns that DELIBERATELY keep the pooled median, with the reason.  A key
#  here is a DECISION; a key in neither table is an ERROR (see `mean_bar`).
NO_BAR = {
    'mSalesToMarketCap': "Tier 'N' (w = 0) -- inert, so an absolute bar would be an "
                         "economic claim invented for a criterion that scores nothing. "
                         "Give it a bar in BARS the moment it is given a weight.",
}


def mean_bar(mcol, pooled_median):
    """The bar `calcScore` must test `mcol` against.

    Returns the stored constant when there is one, the pooled median for a DECLARED
    exception, and RAISES otherwise.  The raise is the point: a mean criterion added
    later would otherwise silently inherit the sample-dependent, lookahead-bearing
    median this module exists to remove, and it would do so with no visible symptom.
    """
    spec = BARS.get(mcol)
    if spec is not None:
        return float(spec['value'])
    if mcol in NO_BAR:
        return pooled_median
    raise KeyError(
        "meanBars: no absolute bar for Stage-1 mean criterion %r, and it is not a declared "
        "exception. A mean criterion is scored as `value - bar`, so without an entry it "
        "would fall back to the POOLED MEDIAN -- sample-dependent and a lookahead channel "
        "in any backtest (register C-12). Add it to meanBars.BARS with a stated rationale, "
        "or to meanBars.NO_BAR with the reason it needs none." % (mcol,))


#  =========================================================================== #
#  THE FAILSAFE BAND -- REPORT AND RECORD, NEVER AUTO-ADJUST (CEO, 2026-08-06)  #
#  =========================================================================== #
#  A stored constant cannot drift with the sample, which is the point -- but it also
#  cannot notice that the WORLD moved, or that a bar was wrong when it was set.  So the
#  run measures each bar's realised pass rate and WRITES IT DOWN.  It never changes a bar.
#
#  WHY IT NEVER AUTO-ADJUSTS.  A bar that re-fits itself to keep its pass rate in a band
#  IS the pooled median again, just with a slower time constant -- and it would re-open
#  the lookahead channel through the back door.  The band is a WATCHDOG on a human
#  decision, not a controller.
#
#  DENOMINATOR = OBSERVED CELLS ONLY.  Guard-refused and non-computable cells are excluded
#  from BOTH numerator and denominator, so the reported rate cannot move when DATA
#  COVERAGE moves.  This is material on exactly one criterion today: `mDebtEquityRatio` is
#  guard-refused (`equity_positive`) on ~6.0% of window cells, and counting those as
#  failures would report a rate ~6pp low and could trip the band on a coverage change
#  rather than on anything about the companies.
BAND_LOW = 0.25
BAND_HIGH = 0.75

#  --------------------------------------------------------------------------------------
#  `advisory` GATES THE ACTION, NEVER THE VERDICT  (CEO framing, fixed 2026-08-07)
#  --------------------------------------------------------------------------------------
#  REPORTING A BREACH AND PROPOSING A RE-SET ARE DIFFERENT ACTS WITH DIFFERENT EVIDENCE BARS.
#  Conflating them is what this fixes:
#
#    * `breach` is a statement ABOUT THE PANEL IN HAND -- "this bar's realised pass rate on
#      these cells is outside [25%, 75%]".  That is true or false at ANY n, and a small
#      universe does not make it less true.  So `breach` is now COMPUTED AND REPORTED ALWAYS.
#    * A RE-SET PROPOSAL is a claim about THE WORLD -- "the bar is wrong and here is a better
#      number" -- and that DOES need a representative universe.  `advisory` gates it, together
#      with `streak_participant` and `BREACH_RUNS_TO_PROPOSE`.
#
#  THE DEFECT THIS REPLACES, recorded because it is the class this repo keeps finding: `breach`
#  used to be FORCED TO 0 whenever `advisory` was 1.  On the 2026-08-07 run (2,613 kept sources)
#  the CSV therefore reported `breach=0` on all seven bars and that was read upward as "all seven
#  bars held" -- a test no run under 5,000 sources could fail.  A column whose name says
#  "breached" and whose value says "not measured" is a number meaning something other than its
#  label.
#
#  WHY NOT THE TWO OBVIOUS FIXES.  Disabling the floor for the test run destroys the guard rail
#  (a thin sample could then move a bar).  Growing the test universe past 5,000 KEPT sources
#  costs roughly 6,000 resolved and ~8 hours against the ~4 today, which defeats the purpose of
#  a fast-iteration universe.  Separating verdict from action costs nothing and gives the honest
#  answer at any n.
#
#  AND IT IS NOT HYPOTHETICAL.  Per-venue pass rates on `mGrossProfitMargin` today: US 49.7%,
#  LSE 48.5%, TSX 37.8%, Paris 34.6%, KOSPI 23.3% -- already under BAND_LOW on one venue.  A
#  full global run will breach that bar ON COMPOSITION, and the verdict has to be trustworthy
#  before then.
#
#  A run below this floor is still not judged: it emits `advisory=1`, `breach_streak=0` and
#  never a `proposed_constant`.  Its pass rates remain a property of the sample it was given --
#  which is precisely the thing constants exist to stop mattering.  THE INVARIANT IS UNCHANGED:
#  A SMALL OR UNREPRESENTATIVE UNIVERSE CANNOT MOVE A BAR.
MIN_FULL_UNIVERSE_SOURCES = 5000

#  A breach must PERSIST before it proposes anything: one bad fetch, one truncated pull,
#  one exchange arriving late must not be able to move a bar.  `proposed_constant` is
#  therefore populated only from the SECOND consecutive full-universe breach.
BREACH_RUNS_TO_PROPOSE = 2

CALIBRATION_COLUMNS = ('criterion', 'constant', 'n_observed', 'n_pass', 'pass_rate',
                       'breach', 'proposed_constant', 'breach_streak', 'advisory',
                       'streak_participant')


def _newest_window(bm_df, window_rows):
    """The newest `window_rows` rows per source, newest-first -- the SAME population
    Stage-1 scores over (`calcScore.calcByTier`'s head(n)).  Re-sorted here rather than
    assumed, exactly as `stage1_veto._evaluate` does, because nothing on the live path
    guarantees ingestion order."""
    df = bm_df
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(['source', 'date'], ascending=[True, False])
    return df.groupby('source', sort=False).head(window_rows)


def _round_proposal(spec, median_value):
    """Round an empirical median to the criterion's OWN units.  For a bar quoted as an
    annual rate the rounding happens on the ANNUAL number and is divided back -- rounding
    the per-quarter number to the same quantum would be 4x coarser than intended."""
    if median_value is None or not np.isfinite(median_value):
        return None
    q = float(spec['round_to'])
    if spec.get('annual_basis'):
        annual = float(median_value) * _Q
        return (round(annual / q) * q) / _Q
    return round(float(median_value) / q) * q


def calibrate(bm_df, mean_dict_signs, window_rows=8, prior_streaks=None,
              n_sources=None, streak_participant=True):
    """One row per bar: observed cells, passes, rate, breach, proposal.  Pure -- no I/O.

    `mean_dict_signs` maps the m-prefixed column to the criterion's Sign, read from the
    live registry rather than mirrored here (a mirrored sign table is the defect class
    this repo keeps finding).
    `prior_streaks` maps criterion -> the previous full-universe run's breach streak.

    `breach` IS COMPUTED AT ANY `n_sources`, INCLUDING AN ADVISORY RUN.  It is a true statement
    about the panel in hand and the report exists to carry it.  `advisory` gates the CONSEQUENCE
    only -- streak, and therefore proposal.  See MIN_FULL_UNIVERSE_SOURCES.

    `streak_participant` -- WHETHER THIS REPORT MAY ADVANCE OR SEED THE HYSTERESIS LEDGER.
    Only the PRODUCTION scoring seam (`postBo.postBoWrapper`) is a participant; every
    research/offline caller is not.  A non-participant still computes and records `breach`
    -- that is a true statement about the panel and the report exists to carry it -- but its
    `breach_streak` is pinned at 0 and it can never emit a `proposed_constant`.  See
    `_prior_streaks` for why the FILENAME could not carry this.
    """
    win = _newest_window(bm_df, window_rows)
    if n_sources is None:
        n_sources = win['source'].nunique() if 'source' in win.columns else 0
    advisory = int(n_sources < MIN_FULL_UNIVERSE_SOURCES)
    participant = int(bool(streak_participant))
    #  MAY THIS REPORT TOUCH THE HYSTERESIS LEDGER AT ALL -- the ONE gate on the ACTION side,
    #  named once so the streak and the proposal cannot drift apart.  An advisory run is
    #  excluded from reading `prior_streaks` too: chaining a streak off a full run and then
    #  proposing on a thin one would move a bar on a sample.
    ledger = bool(participant) and not advisory
    prior_streaks = (prior_streaks or {}) if ledger else {}

    rows = []
    for mcol, spec in BARS.items():
        bar = float(spec['value'])
        sign = mean_dict_signs.get(mcol)
        if sign is None or mcol not in win.columns:
            #  Reported as a zero-evidence row rather than skipped: a criterion missing
            #  from the panel is a fact the report should carry, not one it should hide.
            rows.append({'criterion': mcol, 'constant': bar, 'n_observed': 0, 'n_pass': 0,
                         'pass_rate': None, 'breach': 0, 'proposed_constant': None,
                         'breach_streak': 0, 'advisory': advisory,
                         'streak_participant': participant})
            continue

        vals = pd.to_numeric(win[mcol], errors='coerce')
        obs = vals[vals.notna()]
        n_obs = int(len(obs))
        #  Same comparison calcByTier makes -- `Sign * (value - bar) > 0`.
        n_pass = int(((float(sign) * (obs - bar)) > 0).sum())
        rate = (n_pass / n_obs) if n_obs else None

        #  THE VERDICT -- true at any n, never suppressed.  `advisory` is NOT a term here.
        breached = int(rate is not None and (rate < BAND_LOW or rate > BAND_HIGH))
        #  THE CONSEQUENCE -- gated.  A non-ledger run is pinned at 0 rather than advancing OR
        #  resetting; `_prior_streaks` skips such reports on read, so a standing streak from a
        #  full run survives an advisory run in between.
        streak = ((prior_streaks.get(mcol, 0) + 1) if breached else 0) if ledger else 0
        proposal = None
        if ledger and streak >= BREACH_RUNS_TO_PROPOSE and n_obs:
            proposal = _round_proposal(spec, float(obs.median()))

        rows.append({'criterion': mcol, 'constant': bar, 'n_observed': n_obs,
                     'n_pass': n_pass, 'pass_rate': rate, 'breach': breached,
                     'proposed_constant': proposal, 'breach_streak': streak,
                     'advisory': advisory, 'streak_participant': participant})
    return pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))


def _prior_streaks(directory='.', exclude_basename=None):
    """Breach streaks from the most recent NON-ADVISORY calibration report on disk.

    Advisory reports are skipped rather than treated as a clean run, and THAT IS THE MECHANISM
    that lets an advisory run report a breach honestly without touching the ledger: such a
    report carries `breach_streak = 0`, but because it is skipped here that 0 can neither
    advance a streak nor RESET one a full run recorded.  A TEST universe is invisible to the
    hysteresis, in both directions.

    `exclude_basename` SKIPS THE FILE THIS RUN IS ABOUT TO WRITE, and it is not a nicety.
    `postBoWrapper` is the production seam but it is ALSO re-entered by the offline research
    tools (`baseline_tools/nan_policy_report` calls it TWICE in one process, and
    `industry_attribution`, `run_corrected_current`, `normalized_analysis`, `portfolio` and
    `backtest_ols_analysis` each call it on a full panel).  Without this, the second arm would
    read the first arm's report and manufacture a streak of 2 -- a bar-change proposal out of
    ONE panel, which is exactly what the hysteresis exists to prevent.  The report is named by
    date + universe, so "same basename" == "same run's repeated arms, or a same-day re-run".

    THE BASENAME MATCH WAS NOT ENOUGH (fix, 2026-08-06).  The name is `<date>_<universe>`, so
    it only ever excluded a same-day run under the SAME universe label.  `backtest_ols_analysis`
    builds a `temp_dmdic` with NO `universe` key, so its report lands as
    `MeanBarCalibration-<today>_unknown.csv` -- a DIFFERENT basename, non-advisory whenever its
    PIT-sliced panel clears MIN_FULL_UNIVERSE_SOURCES -- and the day's production run would then
    read it and chain a streak of 2 off a research panel.  One door along from the case the
    basename closed, and unreachable by any filename rule: the identity that matters is not WHICH
    FILE but WHICH RUN, and a research tool can produce any filename it likes.

    So participation is now DECLARED, in the `streak_participant` column, and it is an OPT-IN
    held by the production seam alone (`postBo.postBoWrapper`).  The polarity is the point: a
    research tool written tomorrow is a non-participant without its author knowing this exists.
    `exclude_basename` is KEPT -- it is the cheaper guard for the genuine same-run re-entry
    (`nan_policy_report` calls the seam twice in one process) and the two are independent.

    A report with NO `streak_participant` column is treated as a NON-participant: it predates
    this fix and cannot be attributed.  That discards any streak standing on disk at the
    changeover, which fails toward FEWER proposals -- the safe direction for a failsafe whose
    only output is a proposal.

    RESIDUAL, STATED NOT HIDDEN: nothing here can stop a caller that explicitly passes
    `streak_participant=True`.  That is deliberate -- it is now a written claim in a diff a
    reviewer reads, not an accident of a filename.  And the streak still only ever reaches a
    PROPOSAL a human must accept: no path in this module writes a constant."""
    import glob
    import os
    try:
        files = sorted(glob.glob(os.path.join(directory, 'MeanBarCalibration-*.csv')),
                       key=os.path.getmtime, reverse=True)
        for fn in files:
            if exclude_basename and os.path.basename(fn) == exclude_basename:
                continue
            prev = pd.read_csv(fn)
            if 'streak_participant' not in prev.columns or int(pd.to_numeric(
                    prev['streak_participant'], errors='coerce').fillna(0).max()) != 1:
                continue
            if 'advisory' in prev.columns and int(pd.to_numeric(
                    prev['advisory'], errors='coerce').fillna(1).max()) == 1:
                continue
            if 'breach_streak' not in prev.columns:
                continue
            return {str(r.criterion): int(r.breach_streak)
                    for r in prev.itertuples() if pd.notna(r.breach_streak)}
    except Exception:
        pass
    return {}


def emit_calibration(bm_df, mean_dict_signs, universe='unknown', window_rows=8,
                     directory='.', verbose=True, streak_participant=False):
    """Write `MeanBarCalibration-<date>_<universe>.csv` and return the frame.

    BEST-EFFORT AND FULLY SWALLOWED, like `reporting_period._write_conflict_csv`: this is a
    watchdog on a 12-hour run and must never be able to abort it.  WRITTEN ALWAYS, even
    with no breach -- the file's PRESENCE is the evidence the check ran, so its absence is
    a signal rather than an ambiguity.

    `streak_participant` DEFAULTS TO FALSE AT THIS SEAM ON PURPOSE.  This is the I/O boundary
    every offline research tool reaches through, so the safe value has to be the default one;
    the single production caller opts in by name.  See `_prior_streaks` for the mechanism and
    for the same-day-different-universe hole this closes.
    """
    try:
        import os
        base = ('MeanBarCalibration-%s_%s.csv'
                % (pd.Timestamp.today().strftime('%Y-%m-%d'), universe))
        cal = calibrate(bm_df, mean_dict_signs, window_rows=window_rows,
                        streak_participant=streak_participant,
                        prior_streaks=_prior_streaks(directory, exclude_basename=base))
        fn = os.path.join(directory, base)
        cal.to_csv(fn, index=False)
        if verbose:
            print('  mean-bar calibration written to: %s' % fn, flush=True)
            if int(cal['advisory'].max() or 0):
                #  ADVISORY IS ABOUT THE PROPOSAL, NOT THE VERDICT (2026-08-07).  Breaches
                #  below are REAL and are reported; what this run cannot do is advance a
                #  streak or propose a new constant.
                print('  MEAN-BAR CALIBRATION IS ADVISORY -- fewer than %d sources, so this is '
                      'not a full-universe run. Any BREACH below is still a TRUE statement '
                      'about this panel and is reported as such; what an advisory run cannot '
                      'do is advance a breach streak or propose a re-set.'
                      % MIN_FULL_UNIVERSE_SOURCES, flush=True)
            for r in cal[cal['breach'] == 1].itertuples():
                #  WARNING AND NOTHING ELSE.  The bar is not touched.
                print('  MEAN-BAR BAND WARNING: %s pass rate %.3f is outside [%.2f, %.2f] '
                      '(constant %.6g, %d observed cells, breach run %d of %d). NO BAR WAS '
                      'CHANGED -- %s'
                      % (r.criterion, r.pass_rate, BAND_LOW, BAND_HIGH, r.constant,
                         r.n_observed, r.breach_streak, BREACH_RUNS_TO_PROPOSE,
                         ('ADVISORY RUN: the breach is reported but cannot advance the streak '
                          'or propose a re-set') if int(r.advisory)
                         else ('proposed re-set %.6g, for a HUMAN to accept or reject'
                               % r.proposed_constant) if r.proposed_constant is not None
                         else 'no proposal until the breach persists'), flush=True)
        return cal
    except Exception as _e:
        if verbose:
            print('  WARNING: mean-bar calibration did not run (%s: %s)'
                  % (type(_e).__name__, _e), flush=True)
        return None
