## Description: we read preReq_dict and put into a new dataframe, relVars. We read calcFromfs_dict and for each key we
#               lookup upper and lower and calculate the ratio at each time, adding the result into a new dataframe we
#               call BoPrep_df. If it is in the diff dict we calculate the 4 quarter
#               difference as well. If it is in the mean, we add the sum of the ratio to a dataframe which
#               continuously holds the total sum of that ratio, as well as updating a counter of tickers processed. We
#               also update the mean. We then loop over the "provided" dictionaries and add them to BoPref_df

from collections import defaultdict
## Dict for fundamental calculation
# Sales to inventory should probably be S' - I' > 0, not (S/I)' > 0

import macroConditions as mcond
import scoringWeights as sw          # SINGLE SOURCE OF TRUTH for every scoring weight


# --- KNOWN DUPLICATE STAGE-1 CRITERION (made explicit, 2026-08-02) ------------
# `EPS` and `netIncomePerShare` in BoMetric_diff_dict are BYTE-IDENTICAL criterion
# specs -- same Upper ('netIncomePerShare'), same Lower ('Identity'), same Sign (+1) --
# differing ONLY in Tier: 'B' (w = 0.5) vs 'N' (w = 0).  So the panel carries two
# byte-identical columns, dEPS and dNetIncomePerShare (verified `.equals()` True on the
# 2026-07-17 panel, 148,172 rows, zero NaN mismatches), and the Tier-N one is scored at
# weight 0.
#
# MEASURED, not assumed (2026-08-02): calcScore.calcByTier maps any tier outside
# S/A/B/C/D to w = 0, so the criterion returns exactly 0.0 for every name (max = min =
# 0.0 over 400 names, no NaN), and deleting the registry entry outright leaves Stage-1
# BITWISE identical on the saved panel (identical source order, max |score diff| = 0.0).
# So Tier N genuinely contributes nothing TODAY.
#
# IT IS STILL NOT DELETED, deliberately.  A registry entry is cheap; the risk is one
# tier edit away.  If a future re-weighting promotes `netIncomePerShare` off Tier N, the
# SAME criterion would silently score TWICE -- 0.5 + its new weight on one quantity --
# and nothing in the pipeline would say so.  Keeping the entry and declaring the pair
# here means `test_scoring_weights_single_source.py` fails the moment both carry weight,
# which is a decision prompt rather than a silent double-count.
#
# Removing one entry is ALSO not free of side effects, which is the other reason to
# leave it: it drops a column from BoMetric_df (so an older saved panel and a newer one
# no longer have the same schema, and calcScore's schema gate is column-EXACT on what it
# needs), and it changes the Stage-1 NaN-accounting readout's per-name criterion COUNT
# and name list (the `_nan_acct` readout in calcScore.simpleScore_fromDict, fed by
# calcByTier's `nan_sink`) even though the summed tier weight it reports is
# unaffected.  Neither touches a score, but both are visible output.
#
# Each entry is (BoMetric_diff_dict key that CARRIES the weight, the inert TWIN).
DUPLICATE_DIFF_CRITERIA = (('EPS', 'netIncomePerShare'),)


# --- THE SIGN-INVERTING CRITERIA (fixed 2026-08-04) ---------------------------
# THE DEFECT, in one line: a ratio whose ADVERSE quantity sits in the DENOMINATOR does not
# fail -- it INVERTS SIGN and reads as the best possible value.
#
#   `earnings/price`, `book/price`, `FCF/price`  -- the YIELD forms -- go NEGATIVE when the
#   numerator is adverse, and a negative yield fails on its own.  Self-correcting.
#   `price/earnings`, `price/book`, `price/FCF`, `debt/equity`, `netDebt/EBITDA`, `NI/equity`
#   go negative and read as CHEAP / UNLEVERED / PROFITABLE.  Perverse.
#
# That asymmetry is why Stage-2 was nearly clean (it is written in yield form throughout) and
# Stage-1 was not (it carried five price-in-numerator forms, three at the top three tiers).
# MEASURED over the 61,481 head(8) rows of the 7,729-source 2026-07-17 CORRECTED panel:
#   mPfcfRatio            Tier S, w 1.0  23,034 of 30,386 passes (75.8%) had NEGATIVE FCF
#   uNetDebtToEBITDA      Tier A, w 0.75 11,824 of 31,352 passes (37.7%) had EBITDA <= 0 -- and
#                                        THIS IS THE LARGEST SINGLE CHANGE IN THE SET in weight
#                                        terms, larger than the mPfcfRatio headline.  It is TWO
#                                        cells, and an earlier version of this comment counted
#                                        only the first: 6,319 (20.2%) are the perverse
#                                        netDebt>0/EBITDA<0 case, and a further 5,276 (16.8%) are
#                                        the net-cash-with-negative-EBITDA case.  Both are
#                                        removed; see the unity-dict entry for the full 4-cell
#                                        decomposition and why refusing the second is a judgment
#                                        call rather than a bug fix.
#   PEG                   Tier C         11,665 of 24,292 passes (48.0%) on an UNDEFINED growth
#                                        leg -- the second-largest behavioural change in the set
#   dEffectiveTaxRate     Tier C         7,734 of 26,311 passes (29.4%) had a NEGATIVE rate
#   mPbRatio / dPbRatio   Tier B         3,657 / 1,816 passes on NEGATIVE equity
#   mDebtEquityRatio      Tier C         3,658 of 30,598 passes (12.0%) on NEGATIVE equity
#   returnOnEquity        Tier C         2,255 of 22,402 passes (10.1%) on equity<0 AND NI<0
#   dFreeCashFlowToEquity Tier B         1,746 passes on NEGATIVE equity
# `mPfcfRatio` was the headline: Stage-1's HIGHEST-weighted cheapness criterion awarded three
# quarters of its passes to cash-BURNING rows, unbounded below, and 59 of the 100 deployed
# pool names collected at least one.  Nobody ever meant "burning cash makes you cheap".
#
# TWO FIXES, AND WHICH ONE IS USED IS NOT A STYLE CHOICE:
#
#  1. INVERT to the yield form -- used where the numerator CANNOT go negative (price, debt),
#     so only the denominator flips.  Inverting moves the sign-crossing quantity into the
#     NUMERATOR over a strictly-positive denominator, and the criterion then self-corrects
#     with no guard, no NaN and no new state.  THE SIGN NECESSARILY FLIPS (-1 -> +1): the
#     metric's good direction reverses.  Applied to `pfcfRatio` -> `freeCashFlowToMarketCap`
#     and `pbRatio` -> `bookToPrice`.  This also removes a Stage-1/Stage-2 convention
#     divergence: Stage-2 already scores `freeCashFlowYield` and `bVpRatio` (= 1/pbRatio).
#
#  2. GUARD the denominator (the `Guard` key below) -- used where inverting is UNAVAILABLE or
#     would introduce a fresh defect.  The out-of-domain rows become NaN, which
#     calcScore.calcByTier already scores as a fail.  THE SIGN DOES NOT CHANGE, because the
#     metric's direction does not change -- only its domain shrinks.  Getting that distinction
#     wrong per criterion is the failure this comment exists to prevent.
#
# WHY THE FOUR GUARDED ONES ARE NOT INVERTED -- measured, not asserted:
#   debtEquityRatio        equity/debt would be +inf on the 5.02% of head(8) rows that are
#                          DEBT-FREE (debtEquityRatio == 0 exactly; 6.20% of the panel), and
#                          +inf -> NaN -> FAIL means a debt-free balance sheet fails a
#                          leverage-SAFETY test.  That is precisely the defect
#                          `assetsToLongTermLiabilities` was inverted to FIX (see its entry),
#                          so inverting here would re-create it in a new place.
#   freeCashFlowToEquity   the inverse is equity/FCF, which puts a SIGN-CROSSING FLOW in the
#                          denominator -- strictly worse than the ratio it replaces.  There is
#                          no inversion that fixes a ratio whose BOTH operands cross zero.
#   netDebtToEBITDA        same: net cash and negative EBITDA both occur, so neither operand
#                          is safe as a denominator.
#   returnOnEquity         equity/NI is not a return on anything, and NI crosses zero.
#
# WHAT WAS DELIBERATELY *NOT* CHANGED, so a future reader does not read the omission as an
# oversight:
#   `dSalesToInventory`  -- inventory -> 0 gives +inf, which would PASS.  Different defect
#       (the limit lands on the GOOD side, not an inversion) and no inventory is a BUSINESS
#       MODEL, not an adverse condition.  Tier N, w = 0, inert.  Pinned by
#       test_sign_conventions.py::test_salesToInventory_stays_weight_zero.
#   `dReturnOnCapitalEmployed`, `dReturnOnTangibleAssets` -- measured members of the same
#       family, REPORTED and left alone.  Both are small: capitalEmployed (= totalAssets -
#       totalCurrentLiabilities) goes negative on 813 head(8) rows and contributes 395 passes
#       (1.28%); tangible assets go negative on 90 rows and contribute 30 passes (0.10%).
#       Neither is guarded, and neither is claimed fixed.
#   `returnOnAssets` -- NOT a member of this family, but a SPEC-vs-CODE defect found while
#       fixing it, recorded here because it sits in the same registry.  Its base-dict entry
#       declares `netIncome / totalAssets`, but build_bometric_rows builds the ratio from the
#       MERGED dict, where the diff entry (FMP's `returnOnAssets` field) wins -- so the Tier-S
#       w=1.0 base column has always been the FMP field, not the declared expression.  Measured
#       near-equivalent (median ratio 1.0000, 97.3% of 175,699 rows within 1%, 99.7% sign
#       agreement), so it is a documentation defect today and NOT a scoring one.  Left alone:
#       changing it would change a w=1.0 criterion's basis.  Pinned by
#       test_sign_conventions.test_shared_keys_agree_on_basis_or_are_declared_exceptions.
#
# `PEG` WAS in this list and IS NOW FIXED (follow-up, same day) -- its vendor formula was
# established arithmetically, so the guard condition became determinable.  See its entry in
# BoMetric_special_dict.  It is the second-largest behavioural change in the set at -48.0% of
# its passes.  (This paragraph exists because the earlier text said "DO NOT tidy it in" and
# survived the follow-up that did exactly that, on purpose.)
#
# The `Guard` values name predicates in `calcMetrics.STAGE1_DOMAIN_GUARDS`; the declaration
# lives HERE, beside the metric, and the arithmetic lives THERE, beside the ratio code.


def getDicts():
    # sub dicts
    # Possible entries later:
    #   'netOperatingAssets': {'Upper': '?', 'Lower': '?', 'Tier': 'B'}
    #   'dSalesToEmployees': {'Upper': 'revenue', 'Lower': 'Employees', 'Tier': 'B'},
    #   Defensive Internal Ratio

    preReq_dict = {'bs': ['totalAssets', 'longTermDebt', 'inventory', 'totalStockholdersEquity', 'totalLiabilities',
                          'totalCurrentAssets', 'totalCurrentLiabilities','propertyPlantEquipmentNet', 'otherCurrentAssets',
                          # --- CAPTURE ONLY, NOT WIRED (2026-08-05) --------------------------
                          # `totalDebt` and `cashAndCashEquivalents` are ALREADY IN the paid
                          # v3/balance-sheet-statement response and were discarded at ingest, so
                          # capturing them costs ZERO extra API calls -- the same free capture as
                          # `eps` / `period` / `reportedCurrency` below.
                          #
                          # WHY THEY ARE WANTED.  Together they give the REAL OPERAND
                          # `netDebt = totalDebt - cashAndCashEquivalents`, which is what the
                          # three-branch leverage rule (calcMetrics.net_debt_three_branch) needs
                          # and does not have: today it recovers sign(netDebt) as
                          # sign(ratio) x sign(EBITDA proxy), which is unrecoverable wherever the
                          # proxy is zero and is a proxy near zero everywhere else.
                          #
                          # A SAVED PICKLE CAN NEVER GAIN A COLUMN, which is the whole reason to
                          # capture now: the upcoming fetch is the only cheap chance.  They are
                          # ABSENT FROM EVERY EXISTING PICKLE, so NOTHING may read them yet --
                          # DO NOT rewire the leverage rule to them until a panel that carries
                          # them exists, or the rule becomes untestable on saved data.
                          'totalDebt', 'cashAndCashEquivalents',
                          # `shortTermDebt`: CAPTURE ONLY, NOT WIRED -- and it is here for
                          # REGISTER B-8, not for the leverage rule (2026-08-05).
                          #
                          # B-8 IS THAT FMP CONFLATES "no long-term debt" WITH "not disclosed":
                          # `longTermDebt` is 0.00% NaN and 25.33% EXACTLY ZERO on the local
                          # panel, so the provider sends 0 rather than omitting the key -- which
                          # makes `longTermDebtChange`'s NaN branch unreachable and, because that
                          # metric carries a NEGATIVE weight, awards a PASS in FIN-1, the one
                          # cohort where leverage IS the solvency signal.
                          #
                          # A KEY-PRESENCE MARKER WOULD NOT FIX IT, AND WAS DELIBERATELY NOT
                          # BUILT.  Presence is ALREADY OBSERVABLE -- an absent key becomes NaN
                          # when the response list is framed, so the measured 0.00% NaN rate IS
                          # the presence measurement, and it says the key is present on every
                          # row.  A boolean presence column would therefore be CONSTANT TRUE: it
                          # records nothing and the next reader would trust it.  The
                          # discriminator B-8 needs does not exist in this response.
                          # WHAT DOES DISCRIMINATE, and it is free on this same call: a row with
                          # `longTermDebt == 0` while `totalDebt > 0` (or `shortTermDebt > 0`) is
                          # evidence of NON-DISCLOSURE OR MISALLOCATION, not of debt-freedom,
                          # whereas all three at zero is a genuinely unlevered balance sheet.
                          # That is a real test and it is offline-repairable from the pickles
                          # once these columns exist -- which is why only the CAPTURE half rides
                          # this fetch.  THE SCORING HALF (what `longTermDebtChange` should do
                          # once the cross-check is available) IS DELIBERATELY NOT IN THIS BUILD.
                          'shortTermDebt'],
                   'inc': ['netIncome', 'grossProfit', 'revenue', 'weightedAverageShsOut', 'weightedAverageShsOutDil', 'depreciationAndAmortization',
                           'sellingGeneralAndAdministrativeExpenses', 'operatingIncome','interestExpense',
                           # eps / epsdiluted: ALREADY IN the paid v3/income-statement response and
                           # discarded at ingest until now, so capturing them costs ZERO extra API
                           # calls -- the same free capture as `period` / `reportedCurrency` below.
                           #
                           # WHY THEY ARE WANTED.  `eps` is income available to COMMON, and it is the
                           # field FMP's own `priceEarningsToGrowthRatio` is built from (established
                           # arithmetically -- see calcMetrics._peg_growth_defined).  Exact parity with
                           # that field therefore needs `eps`; `netIncomePerShare` is a near-perfect
                           # proxy but NOT identical (sign agreement 92.8%, median absolute error
                           # 2.5%), and the PEG domain guard currently runs on the proxy.
                           #
                           # THEY POPULATE ON THE NEXT FULL FETCH AND ARE ABSENT FROM EVERY SAVED
                           # PICKLE, like the four provenance fields below.  Nothing reads them yet, so
                           # that absence costs nothing today -- do NOT add a consumer that requires
                           # them without checking the panel actually carries them.
                           #
                           # AND ONE REASON TO EVENTUALLY COMPUTE PEG OURSELVES RATHER THAN CONSUME
                           # THE FIELD: the vendor's growth denominator is a DIFFERENCE OF TWO
                           # 2-DECIMAL-ROUNDED EPS FIGURES, so for small sequential changes its PEG is
                           # dominated by quantisation noise, and 179 of 2,969 rows carry an exact-zero
                           # growth artifact.  Rebuilding it from an unrounded per-share series removes
                           # that noise.
                           'eps', 'epsdiluted',
                           # `ebitda`: CAPTURE ONLY, NOT WIRED (2026-08-05).  Already in the paid
                           # v3/income-statement response and discarded at ingest.  It is the
                           # vendor's OWN EBITDA -- the quantity FMP will not tell us today, and
                           # the reason `net_debt_three_branch` has to reconstruct the sign of
                           # EBITDA from `operatingIncome + depreciationAndAmortization`.  With
                           # it, that proxy (and its ~229 unrecoverable zero-proxy rows) can be
                           # retired.  ABSENT FROM EVERY SAVED PICKLE -- capture now, wire later.
                           'ebitda',
                           # reportedCurrency: the statement's reporting currency (USD/SEK/EUR/...).
                           # Captured (was discarded at ingest) so marketCap -- stored in this same
                           # reporting currency -- can be converted to USD for market-cap banding
                           # (carveOut.marketcap_usd_series). A string column; rides through unused by
                           # every ratio calc. Populates on the next full fetch; absent on saved pickles.
                           'reportedCurrency',
                           # --- REPORTING-PERIOD PROVENANCE (audit C-1 / H-2, 2026-07-19) ---
                           # All four are ALREADY in the paid v3/income-statement response and were
                           # simply discarded at ingest.  Like reportedCurrency they ride through
                           # every ratio calculation unused (they never reach BoMetric_df, whose
                           # columns come from the metric dicts, not from preReq_dict) and populate
                           # from the NEXT full fetch; they are absent on saved pickles.
                           #
                           # period       'Q1'..'Q4' / 'FY'.  THE missing field behind the biggest
                           #              open data defect: FMP labels a SEMI-ANNUAL filer's H1/H2 as
                           #              Q2/Q4 carrying true 6-MONTH flows, and with no period field
                           #              nothing downstream can tell a 3-month flow from a 6-month
                           #              one -- so those names are scored on ~2x flows against
                           #              quarterly peers, and iloc[4] is 2 years back for them (see
                           #              stage2_metrics.piotroski).  Capturing it is the
                           #              precondition for annualising-or-excluding them.
                           # calendarYear FMP's FISCAL-year label -- NOT the calendar year of the
                           #              period-end date.  Verified: TAM.L's 2026-03-31 row carries
                           #              calendarYear=2025.  Never use it as a calendar anchor or to
                           #              derive a date; it disambiguates 52/53-week drift only.
                           # fillingDate  filing date.  Real for SEC filers (30-51d after period end)
                           #              but a PLACEHOLDER equal to the period end for ~50% of rows
                           #              (mostly non-US), so a point-in-time slice must use it only
                           #              where acceptedDate > period end and fall back to a fixed lag
                           #              otherwise -- do not treat it as availability truth blindly.
                           # acceptedDate timestamp the filing was accepted; the discriminator for the
                           #              fillingDate-placeholder test above.
                           'period', 'calendarYear', 'fillingDate', 'acceptedDate'],
                   'cf': ['freeCashFlow', 'netCashProvidedByOperatingActivities','netCashUsedProvidedByFinancingActivities',
                          'dividendsPaid'],
                   'km': ['netIncomePerShare', 'pbRatio', 'earningsYield', 'pfcfRatio', 'grahamNumber', 'grahamNetNet',
                          'marketCap', 'returnOnTangibleAssets', 'incomeQuality', 'bookValuePerShare', 'netDebtToEBITDA',
                          'daysSalesOutstanding', 'capexPerShare', 'tangibleBookValuePerShare',
                          'dividendYield', 'payoutRatio'],
                   'fr': ['returnOnEquity', 'debtEquityRatio', 'currentRatio', 'grossProfitMargin','netProfitMargin',
                          'effectiveTaxRate', 'returnOnCapitalEmployed', 'returnOnAssets', 'priceEarningsToGrowthRatio',
                          'daysOfInventoryOutstanding','capitalExpenditureCoverageRatio'],
                   'other': ['price']
                   }


    #n is > 0; d is difference > 0; m is larger than the mean; u is larger than unity
    #
    # THIS DICT IS THE COLUMN SCHEMA AND IT CARRIES A SECOND COPY OF `Sign`.
    # `utils.initBoMetric_fromDict` builds BoMetric_df's columns from the keys + 'Operation'
    # here, while the ARITHMETIC and the SCORED sign come from the four operational dicts
    # below.  So every key's Sign exists in TWO places, and a sign fix applied to only one of
    # them is silent -- the score changes with no error anywhere.  That is this project's worst
    # historical bug class, so it is now PINNED: test_sign_conventions.py asserts the two
    # registries agree on Sign, on Operation, and on key set, for every criterion.  Change a
    # Sign here and you must change it below (or the test fails, which is the point).
    BoMetric_Calc_dict =   {'currentRatio':                 {'Operation': ['u', 'd'],   'Sign': 1},
                            'returnOnAssets':               {'Operation': ['n','d'],    'Sign': 1},
                            'debtEquityRatio':              {'Operation': ['m'],        'Sign': -1},
                            'effectiveTaxRate':             {'Operation': ['d'],        'Sign': -1},
                            'returnOnCapitalEmployed':      {'Operation': ['d'],        'Sign': 1},
                            'grossProfitMargin':            {'Operation': ['d','m'],    'Sign': 1},
                            # WAS 'pfcfRatio' (price/FCF), Sign -1 -- INVERTED to the yield
                            # form 2026-08-04, so Sign is now +1.  See the mean-dict entry.
                            'freeCashFlowToMarketCap':      {'Operation': ['m'],        'Sign': 1},
                            'earningsYield':                {'Operation': ['m'],        'Sign': 1},
                            'returnOnTangibleAssets':       {'Operation': ['d'],        'Sign': 1},
                            # WAS 'pbRatio' (price/book), Sign -1 -- INVERTED to the yield
                            # form 2026-08-04, so Sign is now +1.  See the mean/diff entries.
                            'bookToPrice':                  {'Operation': ['m', 'd'],   'Sign': 1},
                            'grahamNetNet':                 {'Operation': ['n'],        'Sign': 1},
                            'netIncomePerShare':            {'Operation': ['d'],        'Sign': 1},
                            'grossProfitToAssets':          {'Operation': ['d'],        'Sign': 1},
                            'grahamNumberToPrice':          {'Operation': ['u'],        'Sign': 1},
                            'salesToAssets':                {'Operation': ['d'],        'Sign': 1},
                            # INVERTED to longTermDebt/totalAssets (ruling Q1.2, 2026-07-26) --
                            # see the BoMetric_diff_dict entry for the full reasoning.
                            'assetsToLongTermLiabilities':  {'Operation': ['d'],        'Sign': -1},
                            'salesToMarketCap':             {'Operation': ['m'],        'Sign': 1},
                            'CFO':                          {'Operation': ['n'],        'Sign': 1},
                            # DELIBERATELY UNCHANGED -- see the diff-dict entry.  Tier N (w=0)
                            # and the defect there is inventory -> 0 giving +inf on the GOOD
                            # side, which is not the sign-inversion defect and is not adverse.
                            'salesToInventory':             {'Operation': ['d'],        'Sign': 1},
                            'grossProfit':                  {'Operation': ['d'],        'Sign': 1},
                            'freeCashFlowToEquity':         {'Operation': ['d'],        'Sign': 1},
                            'CFOtoMarketCap':               {'Operation': ['d'],        'Sign': 1},
                            # incomeQuality's UNITY test is REPLACED by the sign-safe
                            # difference CFOlessEarnings (see BoMetric_special_dict).

                            'revenue':                      {'Operation': ['d'],        'Sign': 1},
                            'sharesOutstanding':            {'Operation': ['d'],        'Sign': -1},
                            'EPS':                          {'Operation': ['d'],        'Sign': 1},
                            'EquityToAssets':               {'Operation': ['m'],        'Sign': 1},
                            # `netDebtToEBITDA` USED TO BE HERE with Operation ['u'], Sign -1.
                            # It became a `special` (the three-branch leverage rule, CEO
                            # 2026-08-05), and a `special`'s column is created from
                            # BoMetric_special_dict in utils.initBoMetric_fromDict -- NOT from
                            # this dict.  Leaving the entry here would create an EMPTY
                            # `uNetDebtToEBITDA` column that nothing ever fills.
                            #
                            # NEW (2026-08-05): interestCoverage = operatingIncome /
                            # interestExpense, tested against unity (i.e. `> 1x covered`).
                            # Sign +1 -- more coverage is better.  BOTH LEGS ARE THE SAME
                            # PERIOD'S FLOW, so the ratio is frequency-invariant and it
                            # deliberately has NO reporting_period.STAGE1_FLOW_CORRECTION entry.
                            'interestCoverage':             {'Operation': ['u'],        'Sign': 1},
                            'netProfitMargin':              {'Operation': ['m'],        'Sign': 1},
                          }

    BoMetric_base_dict =    {
        'returnOnAssets':   {'Upper': 'netIncome',                              'Lower': 'totalAssets', 'Tier': 'S', 'Sign': 1},
        'grahamNetNet':     {'Upper': 'grahamNetNet',                           'Lower': 'Identity', 'Tier': 'N', 'Sign': 1},
        'CFO':              {'Upper': 'netCashProvidedByOperatingActivities',   'Lower': 'Identity', 'Tier': 'S', 'Sign': 1}
        }

    BoMetric_diff_dict =    {
        'returnOnTangibleAssets':       {'Upper': 'returnOnTangibleAssets',                 'Lower': 'Identity',                'Tier': 'B', 'Sign': 1},
        'returnOnAssets':               {'Upper': 'returnOnAssets',                         'Lower': 'Identity',                'Tier': 'S', 'Sign': 1},
        'grossProfitToAssets':          {'Upper': 'grossProfit',                            'Lower': 'totalAssets',             'Tier': 'A', 'Sign': 1},
        # NOT inverted (ruling Q1.2, 2026-07-26).  inventory == 0 on 41.44% of rows, and the
        # limit of revenue/inventory as inventory -> 0 IS the good side -- so this is the same
        # defect class as assetsToLongTermLiabilities.  Left alone for two reasons: Tier 'N'
        # means w = 0, so it cannot move a score; and for a services firm inventory == 0 means
        # the ratio is NOT APPLICABLE rather than infinitely good, so NaN is the more honest
        # reading than a synthetic best-case. Revisit if this metric is ever given weight.
        #
        # RE-EXAMINED AND DELIBERATELY LEFT UNCHANGED by the 2026-08-04 sign-inversion fix.
        # It is NOT a member of that family, on two counts, and both matter:
        #  1. THE FAILURE IS THE OPPOSITE SHAPE.  The sign-inversion defect is a ratio going
        #     NEGATIVE and reading as best-in-class.  Here nothing goes negative -- revenue and
        #     inventory are both >= 0.  The limit as inventory -> 0 is +INFINITY, which lands on
        #     the metric's BEST side and would score a PASS: an inventory-free business acing an
        #     inventory-TURN test.  The project's boundary rule refuses exactly this case
        #     ("where the limit lands at the metric's BEST value, the rule does not apply and
        #     the metric must be REFUSED"), and w = 0 already achieves the refusal.
        #  2. NO INVENTORY IS A BUSINESS MODEL, NOT AN ADVERSE CONDITION.  A consultancy has no
        #     inventory because of what it is, not because something went wrong.  Guarding it
        #     would be asserting a deterioration that did not happen.
        # SCALE, so nobody has to re-measure it to decide: 3,641 sources (47.1%) and 26 of the
        # 100 deployed pool names have at least one such row.  INERT TODAY at Tier N (w = 0) --
        # and that inertness is the ONLY thing keeping it harmless.
        # PINNED: test_sign_conventions.py::test_salesToInventory_stays_weight_zero fails the
        # moment this leaves Tier N, so activating it is a VISIBLE EVENT with this comment
        # attached rather than a silent re-weighting that quietly turns +inf into a pass.
        'salesToInventory':             {'Upper': 'revenue',                                'Lower': 'inventory',               'Tier': 'N', 'Sign': 1},
        'salesToAssets':                {'Upper': 'revenue',                                'Lower': 'totalAssets',             'Tier': 'N', 'Sign': 1},
        'grossProfitMargin':            {'Upper': 'grossProfitMargin',                      'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        # GUARDED, not inverted (sign-inversion fix, 2026-08-04).  effectiveTaxRate is
        # incomeTaxExpense / incomeBeforeTax, so a PRE-TAX LOSS (or a tax credit) makes the
        # rate NEGATIVE -- and this is a DIFF test with Sign -1, i.e. a FALLING tax rate
        # scores.  A name going from +25% to -50% therefore reads as a large improvement in
        # tax efficiency when what actually happened is that it started losing money.
        # MEASURED: 7,736 of 26,313 passes (29.4%) sit on a rate < 0.
        # The admissible domain of a tax RATE is >= 0, so the guard refuses the negative rows
        # and the diff then has no defined change to score.  SIGN STAYS -1 -- the direction
        # (lower tax is better) is unchanged; only the domain shrinks.
        # LIMIT, stated because it bounds what this fix can claim: NEITHER operand
        # (incomeTaxExpense, incomeBeforeTax) is in preReq_dict, so they are not on cdx_df and
        # the OPERAND signs cannot be recovered -- only the ratio's own sign.  So the
        # both-operands-negative case (a tax CREDIT on a PRE-TAX LOSS, which yields a
        # NORMAL-LOOKING POSITIVE rate) is NOT detectable here and is NOT fixed.  It is a
        # separate, unmeasured channel that needs those two fields at ingest.
        # A rate > 1 is also out of the natural domain but is NOT the inversion defect and is
        # NOT guarded here: 1,267 head(8) rows, contributing only 63 passes (0.24%).
        # TIER 'N' (w = 0) SINCE 2026-08-05 -- WAS TIER 'C' (w = 0.30).  CEO DECISION: the
        # criterion is REMOVED FROM THE GATE, not repaired.  The guard above stays (it is still
        # the right domain if the criterion is ever restored), and so does the measurement that
        # motivated it -- what changed is the ECONOMICS.  Its apparent correlation with quality
        # was traced to a DATA-AVAILABILITY channel, not to tax efficiency: `effectiveTaxRate`
        # is NaN on 25.7% of rows, so the criterion was substantially measuring "does this
        # company report a usable tax line", which is a filer/coverage property and not a
        # statement about the business.  A criterion that scores completeness under the name of
        # a business quantity is worse than an absent one, because its weight is spent on the
        # wrong axis while its NAME says otherwise.
        # NOT DELETED, for the reason DUPLICATE_DIFF_CRITERIA gives at the top of this module: a
        # registry entry is cheap, deleting it drops a panel column (a schema change for every
        # saved pickle) and changes the NaN-accounting readout, and keeping it means restoring
        # the criterion is a one-character edit with this note attached.
        'effectiveTaxRate':             {'Upper': 'effectiveTaxRate',                       'Lower': 'Identity',                'Tier': 'N', 'Sign': -1, 'Guard': 'tax_rate_nonnegative'},
        'currentRatio':                 {'Upper': 'currentRatio',                           'Lower': 'Identity',                'Tier': 'B', 'Sign': 1},
        # INVERTED: longTermDebt/totalAssets, Sign -1 (ruling Q1.2, 2026-07-26).
        # It was totalAssets/longTermDebt with Sign +1.  longTermDebt == 0 on 17,824 of
        # 177,350 panel rows (10.05%) -- DEBT-FREE firms -- and TA/0 is +-inf, which
        # forceNumOnDf turns into NaN, which calcByTier scores as a FAIL.  So a debt-free
        # balance sheet FAILED a leverage-SAFETY test, and it failed it for having no debt.
        # The general rule: if the limit of the ratio as the denominator -> 0 is the GOOD
        # side, then inf -> NaN -> fail is backwards.  Inverting puts the zero-able quantity
        # in the NUMERATOR, so debt-free = 0.0 = best and is DEFINED; NaN rows drop from
        # 11.05% to 0.35% (only genuinely-corrupt totalAssets <= 0).  Sign flips to -1
        # because the metric direction reverses: falling leverage is now the good side.
        # WHAT THIS DID AND DID NOT FIX -- measured, because the difference matters:
        #   FIXED     the ratio is now DEFINED for a debt-free firm (0.0 instead of NaN):
        #             0.0% of debt-free rows had a value before, 96.6% do now.  So the
        #             criterion no longer fails them on a MISSING-DATA technicality, and the
        #             direction is now economically right (falling leverage scores).
        #   NOT FIXED they still mostly FAIL, for a different and legitimate reason: this is
        #             a DIFF test, and a permanently debt-free firm has d(0) = 0, which is
        #             not > 0.  85.7% of debt-free rows with a computable diff still fail
        #             (85.6% of them have a zero diff).  Only the 14.3% that actually
        #             DE-levered in the period pass.
        # That is true of EVERY diff test here -- "no change" is not "improvement" -- and is
        # not specific to this metric.  Do not describe the inversion as "debt-free firms no
        # longer fail a leverage test": they largely still do, on a defensible basis rather
        # than on missingness.
        'assetsToLongTermLiabilities':  {'Upper': 'longTermDebt',                           'Lower': 'totalAssets',             'Tier': 'B', 'Sign': -1},
        'grossProfit':                  {'Upper': 'grossProfit',                            'Lower': 'Identity',                'Tier': 'N', 'Sign': 1},
        'returnOnCapitalEmployed':      {'Upper': 'returnOnCapitalEmployed',                'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        # NOT inverted (ruling Q1.2, 2026-07-26).  totalStockholdersEquity == 0 on only 39 of
        # 177,350 rows (0.02%), and -- decisively -- the zero-denominator rule does NOT apply:
        # the limit of FCF/equity as equity -> 0 is +inf when FCF > 0 and -inf when FCF < 0,
        # so there is no single 'good side' to point at.  Zero book equity is also an
        # economically degenerate state (technically insolvent), which NaN describes better
        # than either infinity.
        #
        # GUARDED equity > 0 (sign-inversion fix, 2026-08-04).  Q1.2 above ruled on equity
        # == 0; the defect fixed here is equity < 0, which Q1.2 did not consider.  BOTH
        # operands cross zero, so this is a genuine double-negative ratio: FCF < 0 with
        # equity < 0 gives a POSITIVE FCF-to-equity, and the diff then scores a deteriorating
        # cash position as an improving one.  MEASURED on the head(8) window, by sign cell:
        #     FCF>0 equity>0   35,067 rows   pass 0.6224     (the normal case)
        #     FCF<0 equity>0   21,026 rows   pass 0.2929     (correctly penalised)
        #     FCF>0 equity<0    1,469 rows   pass 0.3376
        #     FCF<0 equity<0    2,106 rows   pass 0.5921  <- BOTH NEGATIVE, passing MORE often
        #                                                   than the FCF<0 equity>0 case
        # 1,743 passes total on equity < 0.  The last two cells are what the guard removes.
        # SIGN STAYS +1: the direction (rising cash-to-equity is good) is unchanged; the
        # denominator is not moved, so nothing reverses.  This is the criterion where getting
        # that right matters most -- it sits next to two entries whose Sign DID flip.
        'freeCashFlowToEquity':         {'Upper': 'freeCashFlow',                           'Lower': 'totalStockholdersEquity', 'Tier': 'B', 'Sign': 1, 'Guard': 'equity_positive'},
        'CFOtoMarketCap':               {'Upper': 'netCashProvidedByOperatingActivities',   'Lower': 'marketCap',               'Tier': 'B', 'Sign': 1},
        # DUPLICATE OF 'EPS' BELOW -- byte-identical spec, differing only in Tier ('N',
        # w = 0, so inert today).  See DUPLICATE_DIFF_CRITERIA at the top of this module
        # before giving this a tier: promoting it double-counts the same quantity.
        'netIncomePerShare':            {'Upper': 'netIncomePerShare',                      'Lower': 'Identity',                'Tier': 'N', 'Sign': 1},
        # INVERTED from `pbRatio` (price/book, Sign -1) to book/price, Sign +1, AND GUARDED
        # equity > 0 (sign-inversion fix + review blocker 1, 2026-08-04).
        #
        # THE INVERSION ALONE DOES NOT FIX THE DIFF FORM, and this was caught in review after
        # being wrongly reported as cleared.  When equity < 0 BOTH LEGS of equity/marketCap
        # invert, so a RISING MARKET CAP drives a negative ratio toward zero, the diff comes out
        # POSITIVE, and the row passes -- i.e. a company getting MORE EXPENSIVE while its equity
        # deficit is flat or worsening passes a CHEAPNESS criterion.  MEASURED on the 1,313
        # post-inversion passes that still sat on equity < 0: only 869 (66.2%) were
        # equity-driven; 444 (33.8%) passed on the market-cap leg alone, 434 of them with market
        # cap RISING against flat-or-worse equity.
        # METHOD NOTE, because the first check was wrong: an earlier pass validated these rows
        # by testing whether "book-to-price rose", which IS THE PASS CONDITION -- a tautology
        # that returned 99.7% and cleared nothing.  A residual has to be decomposed by WHICH LEG
        # moved, never by re-asking the criterion.
        # So the DIFF form is REFUSED when equity <= 0: the change in a book YIELD is undefined
        # when the book value has no meaningful sign.  Because the guard is applied to the LEVEL
        # and NaN propagates through calc_diff's subtraction, this is automatically TWO-SIDED --
        # refused whenever EITHER period is inadmissible, which is what "no defined change"
        # means.
        # THE MEAN FORM IS DELIBERATELY *NOT* GUARDED -- see the BoMetric_mean_dict entry.
        # 1,816 of 31,017 passes (5.9%) sat on equity < 0 before the inversion.
        # Basis: totalStockholdersEquity / marketCap, the RAW fields, rather than 1/pbRatio --
        # same reasoning as the in-pipeline grahamNumber (see stamp_frequency_and_graham): do
        # not depend on an undocumented FMP per-share convention that could change.  VERIFIED
        # equivalent on the panel: median(equity/marketCap over 1/pbRatio) = 1.0000, 98.7% of
        # 175,723 rows within 1%, 99.98% same sign.
        # Stock/stock, so NO flow-scale correction -- as `pbRatio` had none.
        'bookToPrice':                  {'Upper': 'totalStockholdersEquity',                'Lower': 'marketCap',               'Tier': 'B', 'Sign': 1, 'Guard': 'equity_positive'},
        'revenue':                      {'Upper': 'revenue',                                'Lower': 'Identity',                'Tier': 'A', 'Sign': 1},
        'sharesOutstanding':            {'Upper': 'weightedAverageShsOut',                  'Lower': 'Identity',                'Tier': 'B', 'Sign': -1},
        # THE WEIGHT-BEARING half of the duplicate pair (the other is
        # 'netIncomePerShare' above, Tier 'N').  See DUPLICATE_DIFF_CRITERIA.
        'EPS':                          {'Upper': 'netIncomePerShare',                      'Lower': 'Identity',                'Tier': 'B', 'Sign': 1}
                             }

    BoMetric_mean_dict =    {
        # INVERTED from `pbRatio`, Sign -1 -> Sign +1.  See the diff-dict `bookToPrice` entry
        # for the basis choice and the measured equivalence; 3,657 of this criterion's 32,349
        # passes (11.3%) sat on NEGATIVE equity reading as maximally cheap.
        #
        # NO GUARD HERE, AND THAT ASYMMETRY WITH THE DIFF FORM IS THE POINT (review blocker 1).
        # The inversion ALREADY fixes the LEVEL test outright: negative equity gives a negative
        # book yield, which loses the mean test against a positive panel median on its own.
        # MEASURED post-inversion: 0 passes on equity < 0, from 3,657 before.  So there is
        # nothing left to guard, and guarding anyway would COST something real -- a guarded row
        # leaves the pool median, and this criterion is scored as `value - median(column)`, so
        # dropping 3,661 legitimate negative observations would MOVE THE BAR for every name that
        # was always in domain.  A negative book yield is a true measurement of a real company,
        # not a domain error; it belongs in the ruler.  Stage-2's `bVpRatio` keeps them for the
        # same reason.
        # The DIFF form is different because there the level's sign corrupts the CHANGE, which
        # the inversion cannot repair -- so it is guarded and this is not.
        'bookToPrice':          {'Upper': 'totalStockholdersEquity',    'Lower': 'marketCap',   'Tier': 'B', 'Sign': 1},
        'salesToMarketCap':     {'Upper': 'revenue',                    'Lower': 'marketCap',   'Tier': 'N', 'Sign': 1},
        'earningsYield':        {'Upper': 'earningsYield',              'Lower': 'Identity',    'Tier': 'S', 'Sign': 1},
        # GUARDED equity > 0, NOT inverted (sign-inversion fix, 2026-08-04).  debtEquityRatio
        # is totalDebt/totalStockholdersEquity, so NEGATIVE equity makes it negative, which on
        # a Sign -1 test reads as LESS levered than any real company -- 3,658 of 30,598 passes
        # (12.0%).  A book-insolvent balance sheet scoring as unlevered is the inversion.
        # NOT inverted, and this is measured rather than preferred: equity/debt is +inf on the
        # 5.02% of head(8) rows with debtEquityRatio == 0 exactly (DEBT-FREE names; 6.20% of
        # the panel), and +inf -> NaN -> FAIL would make a debt-free company fail a leverage
        # test -- the same defect `assetsToLongTermLiabilities` was inverted to fix.  Trading
        # a 12.0% inversion for a 5.02% new one is not a fix.
        # SIGN STAYS -1: less leverage is still better.  Only the domain shrinks.
        'debtEquityRatio':      {'Upper': 'debtEquityRatio',            'Lower': 'Identity',    'Tier': 'C', 'Sign': -1, 'Guard': 'equity_positive'},
        # INVERTED from `pfcfRatio` (price/FCF, Sign -1) to FCF/marketCap, Sign +1
        # (sign-inversion fix, 2026-08-04).  THIS WAS THE LARGEST DEFECT MEASURED IN THE
        # PIPELINE: Stage-1's highest-weighted cheapness criterion (Tier S, w = 1.0) awarded
        # 23,041 of its 30,378 passes (75.9%) to rows with NEGATIVE free cash flow, because
        # price/FCF goes negative -- and unbounded below, so a small cash burn reads as
        # arbitrarily cheap.  59 of the 100 deployed pool names collected at least one.
        # The yield form is self-correcting: negative FCF over a positive market cap is a
        # negative yield, which loses the mean test against a positive panel median.
        # Basis: the RAW freeCashFlow / marketCap, not 1/pfcfRatio (same reasoning as
        # bookToPrice).  VERIFIED equivalent: median ratio 1.0000, 95.6% of 171,273 rows
        # within 1%, 99.56% same sign.
        # THE FLOW LEG MOVES WITH THE INVERSION: reporting_period.STAGE1_FLOW_CORRECTION
        # carried 'pfcfRatio' as ('flow_den', 'per_quarter') because the flow was the
        # DENOMINATOR; it is now the NUMERATOR, so the entry is ('flow_num', 'per_quarter').
        # Leaving the old leg would have applied the semi-annual correction BACKWARDS -- x2
        # where x0.5 was needed, a 4x error on every semi-annual name.
        'freeCashFlowToMarketCap': {'Upper': 'freeCashFlow',            'Lower': 'marketCap',   'Tier': 'S', 'Sign': 1},
        'EquityToAssets':       {'Upper': 'totalStockholdersEquity',    'Lower': 'totalAssets', 'Tier': 'D', 'Sign': 1},
        'grossProfitMargin':    {'Upper': 'grossProfitMargin',          'Lower': 'Identity',    'Tier': 'B', 'Sign': 1},
        'netProfitMargin':      {'Upper': 'netProfitMargin',            'Lower': 'Identity',    'Tier': 'C', 'Sign': 1},
                             }

    BoMetric_unity_dict =    {
        'currentRatio':         {'Upper': 'currentRatio',       'Lower': 'Identity',    'Tier': 'S', 'Sign': 1},
        # BOUNDARY-IMPUTED on its ADVERSE rows (nan-policy.md ADDENDUM A, 2026-08-05).
        # grahamNumber = sqrt(22.5 * EPS_ttm * BVPS) is undefined for EPS_ttm <= 0 or BVPS <= 0,
        # and 99.1% of that non-computability is ADVERSE (negative EPS 22,103 rows, negative book
        # 1,109) against 0.9% a genuine gap (missing inputs 208) -- the strongest evidence in
        # this project that "undefined" and "missing" are different objects.  The CEO's rule for
        # the adverse half: "just put it like earnings were close to 0."  Taken as a LIMIT, that
        # is grahamNumber/price -> 0.0, and the unity test then yields -1.0 = a FAIL.
        # SO THIS IS BEHAVIOUR-IDENTICAL TO TODAY on all 23,212 adverse rows, deliberately, and
        # the identity is the confirmation rather than the disappointment: the fail becomes
        # DERIVED ("there is no earnings-based valuation floor to compare this price against")
        # instead of INCIDENTAL ("the number was missing"), so the criterion no longer relies on
        # NaN-scores-as-a-fail to reach the right answer.  The 208 missing-input rows stay NaN.
        # Predicate in calcMetrics.STAGE1_BOUNDARY_IMPUTATIONS, limit + derivation in
        # nan_policy.BOUNDARY_LIMIT.  Sign STAYS +1 and the Tier is untouched.
        # TIER 'N' (w = 0) SINCE 2026-08-05 -- WAS TIER 'S' (w = 1.0).  CEO DECISION, and it is
        # a DEMOTION of the criterion, not a repair of it: the boundary imputation above stays
        # exactly as it is.  Reason: the metric is REMOVED FROM THE STAGE-1 GATE while it is
        # retained in Stage-2 (`grahamNumberToPrice`, Tier 3 there pending audit D3) -- so the
        # two stages now DELIBERATELY disagree about it, and that asymmetry is the decision.
        # Stage-1 is a completeness/quality gate over the whole universe, where a criterion that
        # is undefined-and-adverse on 37.5% of rows (23,212 of 61,832 newest-8) spends a w = 1.0
        # Tier-S budget mostly on stating that a company lost money -- which returnOnAssets,
        # CFO and CFOlessEarnings already state at Tier S, three times over.  Stage-2 RANKS a
        # pool of 100 survivors, where the same column is a cheapness reading among many.
        # DO NOT "fix the inconsistency" by demoting the Stage-2 metric to match -- it is
        # retained there by explicit CEO decision (2026-08-05).  Sigma-w for Stage-1 goes
        # 18.65 -> 17.85 together with the dEffectiveTaxRate demotion and the interestCoverage
        # addition; see the note on `interestCoverage` below.
        'grahamNumberToPrice':  {'Upper': 'grahamNumber',       'Lower': 'price',       'Tier': 'N', 'Sign': 1, 'Boundary': 'graham_adverse'},
        # NEW CRITERION (CEO, 2026-08-05).  interestCoverage = operatingIncome /
        # interestExpense, tested against UNITY, i.e. "does one period's operating profit cover
        # one period's interest bill at least once".  TIER 'B' (w = 0.50) -- and the tier is a
        # DECISION, not a measurement:
        #   * It is a meaningful leverage instrument in its own right and the pipeline had none
        #     on the FLOW side -- every existing leverage criterion reads a STOCK
        #     (netDebtToEBITDA, debtEquityRatio, assetsToLongTermLiabilities, EquityToAssets).
        #     Interest coverage is the question "can it service the debt out of current
        #     earnings", which is the one a stock ratio cannot answer.
        #   * It must NOT outrank the existing stock-leverage criterion at Tier A, hence B and
        #     not A: netDebtToEBITDA remains the primary leverage carrier.
        #   * It partially offsets the 1.30 of Stage-1 weight removed the same day by demoting
        #     `uGrahamNumberToPrice` (Tier S, 1.00) and `dEffectiveTaxRate` (Tier C, 0.30).
        #     Sigma-w over the Stage-1 registry goes 18.65 -> 17.85.  Any tool that hard-codes
        #     18.65 (baseline_tools/verify_part5_defects.py prints it) is now stale.
        # BOTH LEGS ARE THE SAME PERIOD'S FLOW, so the ratio is frequency-invariant: it gets NO
        # reporting_period.STAGE1_FLOW_CORRECTION entry, deliberately, and the unity bar is a
        # per-period bar that means the same thing for a semi-annual filer as for a quarterly
        # one.  (Contrast netDebtToEBITDA, a STOCK over a flow, which needs the factor.)
        # GUARDED `interestExpense > 0`, and the guard is the substantive half of the design.
        # FMP reports 0 for a debt-free name, so without it every debt-free company would come
        # out +/-inf -> NaN -> FAIL a debt-safety test for having no debt -- exactly the defect
        # `assetsToLongTermLiabilities` was inverted to fix.  Refusing the row instead hands the
        # leverage question to `netDebtToEBITDA`, whose net-cash branch PASSES a debt-free name
        # on an explicit operand condition.  So the two criteria are complementary by
        # construction rather than by coincidence.
        # SIGN +1: more coverage is better.
        # NOT MEASURED ON A PANEL, and this is the one caveat on the whole change: both inputs
        # are already fetched, but no saved pickle carries a `uInterestCoverage` column, so its
        # pass rate and its correlation with the rest of the gate are UNKNOWN until the next
        # fetch.  It is added on economic grounds, at a mid tier, for that reason.
        'interestCoverage':     {'Upper': 'operatingIncome',    'Lower': 'interestExpense', 'Tier': 'B', 'Sign': 1, 'Guard': 'interest_expense_positive'},
        }

    BoMetric_special_dict ={
        # THE THREE-BRANCH LEVERAGE RULE (CEO, 2026-08-05).  `netDebtToEBITDA` WAS A `unity`
        # CRITERION HERE-ABOVE with `Guard: ebitda_positive`; it is now a `special`, because the
        # rule the CEO ruled for has THREE branches and neither `Guard` (a refusal mask) nor
        # `Boundary` (a finite-limit fill) can express one.  THE FULL REASONING, the four
        # measured sign cells, why the net-cash branch must never compute the ratio, how
        # sign(netDebt) is recovered and what that recovery cannot do lives in
        # `calcMetrics.net_debt_three_branch` -- beside the arithmetic, once.
        #
        # SIGN IS NOW +1, AND THE FLIP IS NOT A BUG.  The old unity column held the LEVERAGE
        # RATIO and was scored `-(ratio - 1) > 0`; the new column holds a VERDICT-BEARING
        # margin (positive = passes), so higher is better.  Read the sign-convention rule at
        # the top of this module: a form change that moves the tested quantity flips the sign,
        # a domain shrink does not.  Tier 'A' and w = 0.75 are UNCHANGED.
        #
        # THE PANEL COLUMN IS RENAMED `uNetDebtToEBITDA` -> `netDebtToEBITDA` (a `special`'s
        # column is its key, unprefixed, like `returnOnEquity` and
        # `capitalExpenditureCoverageRatio`).  That is a SCHEMA CHANGE: `calcScore`'s schema
        # gate will refuse any panel built before today, which is correct -- an older panel
        # cannot be scored by this rule.
        'netDebtToEBITDA':                  {'Tier': 'A', 'Sign': 1},
        # ---- the pre-existing specials -------------------------------------------------
        # (the old netDebtToEBITDA unity commentary follows, kept because it records the
        #  measurement that justified the 2026-08-04 guard this rule supersedes)
        # THE GUARD KEYED ON THE DENOMINATOR ALONE, AND THAT WAS THE WHOLE POINT.  This
        # criterion has FOUR sign cells and only ONE of them is the defect.  Measured over the
        # 61,481 head(8) rows (sign(netDebt) recovered as sign(ratio) x sign(EBITDA proxy)):
        #   netDebt>0 EBITDA>0  33,615 rows  pass 0.2222  23.8% of passes  normal, correct
        #   netDebt<0 EBITDA>0  11,844 rows  pass 0.9998  37.8% of passes  GENUINE NET CASH,
        #                                                                 CORRECT, MUST SURVIVE
        #   netDebt>0 EBITDA<0   6,324 rows  pass 0.9992  20.2% of passes  THE DEFECT: has debt
        #                                                                 AND no earnings, and
        #                                                                 scores as safest
        #   netDebt<0 EBITDA<0   9,200 rows  pass 0.5739  16.8% of passes  net cash + loss
        #
        # Both the GENUINE and the PERVERSE cell have exactly ONE negative operand -- they are
        # told apart by WHICH operand, never by HOW MANY.  A "both operands negative" rule
        # would miss the entire 20.2% defect and instead transform the 16.8% cell, which was
        # never the measured defect.  Guarding the DENOMINATOR leaves the 11,844 genuine
        # net-cash rows untouched BY CONSTRUCTION, because their denominator is admissible --
        # not by a carve-out that could rot.  VERIFIED end to end: 11,842 of 11,842 genuine
        # passes survive, 0 of 2,990 sources lose one.
        #
        # THE TOTAL REMOVED IS 11,824 OF 31,352 PASSES (37.7%), NOT 20.2%.  Guarding the
        # denominator refuses EVERY row with EBITDA <= 0, so it takes BOTH negative-EBITDA
        # cells: the 6,319 perverse ones AND the 5,276 net-cash-with-negative-EBITDA ones (plus
        # ~229 zero-EBITDA rows).  Stated explicitly because quoting only the 20.2% understates
        # this change by ~1.9x and it is, in weight terms (w = 0.75), THE LARGEST SINGLE CHANGE
        # in the sign-inversion batch -- larger than the mPfcfRatio headline.
        # THE SECOND CELL IS A JUDGMENT CALL, NOT A BUG FIX, and is flagged as such: a company
        # with net CASH and negative EBITDA has nothing to service, so one could argue it should
        # pass.  It is refused here because |netCash|/|EBITDA loss| < 1 is not a debt-service
        # measurement of anything -- with no EBITDA there is no service capacity to measure --
        # and refusing never rewards.  Reversible: admit `EBITDA <= 0 AND netDebt < 0` if ruled
        # otherwise.
        # EBITDA is the `operatingIncome + depreciationAndAmortization` PROXY (FMP does not
        # give the EBITDA behind its own netDebtToEBITDA), so the cell counts are indicative
        # near zero; the guard's DIRECTION does not depend on the proxy.
        # (end of the retained 2026-08-04 commentary.)
        # CFO - netIncome > 0 -- the sign-SAFE replacement for the old
        # uIncomeQuality unity test (domain review S1, fixed 2026-07-26).
        #
        # It used to test FMP's incomeQuality (= CFO/NI) against 1.0.  That is a RATIO whose
        # DENOMINATOR CHANGES SIGN, so the test inverts for loss-makers.  Measured on the
        # 2026-07-17 panel (175,503 usable rows, Tier S, w=1.0):
        #     NI>0 CFO>0 healthy                        pass 0.742
        #     NI>0 CFO<0 profit but no cash             pass 0.016  (correctly fails)
        #     NI<0 CFO>0 loss but CASH-GENERATIVE, the GOOD case   pass 0.034
        #     NI<0 CFO<0 loss AND burning cash          pass 0.316  <- ~10x the good case
        # 12,241 rows -- 14.6% of ALL passes of this w=1.0 criterion -- were companies
        # losing money AND burning cash.  The difference form fixes the inversion outright:
        # the good case goes to 1.000 and profit-without-cash to 0.000.
        #
        # This is exactly the form Piotroski p4 already uses (stage2_metrics.piotroski:
        # `CFO > netIncome`), so the pipeline now states the accrual test one way.
        # SEMANTICS, stated so the remaining 0.672 on loss-and-burning is not read as a
        # miss: this is an EARNINGS-QUALITY (accruals) test, not a profitability test -- a
        # loss-maker whose cash outflow is smaller than its accounting loss genuinely has
        # better earnings quality.  Profitability is tested separately and at Tier S by
        # returnOnAssets and the CFO>0 base test.
        'CFOlessEarnings':                  {'Tier': 'S', 'Sign': 1},
        # COMPUTED LOCALLY, AND THE DOMAIN MOVED WITH IT (2026-08-05).  The criterion is still
        # `1/PEG - 1 > 0`, i.e. exactly 0 < PEG < 1, still Tier C, still Sign +1.  What changed
        # is that `PEG` is no longer FMP's `priceEarningsToGrowthRatio`: it is computed from
        # full-precision inputs on a TRAILING-YEAR basis with a ONE-YEAR growth horizon.  The
        # vendor formula that was reverse-engineered, the three defects in it (quarter-over-
        # quarter horizon, an annualised P/E divided by a quarterly percentage, and a growth leg
        # differenced from two 2-decimal-rounded figures), and the horizon reasoning all live in
        # `calcMetrics.peg_local` -- one place, beside the arithmetic.  CEO ruling behind it: "In
        # general, we should compute things we can rather than using the FMP."
        #
        # THE `Guard` KEY IS GONE, AND THAT IS NOT A REVERT OF THE 2026-08-04 FIX -- READ THIS
        # BEFORE RESTORING IT.  PEG is still REFUSED on exactly the states that used to be
        # guarded plus one more, and the refusal is stricter, not looser.  What changed is WHERE
        # the domain is stated.  A `Guard` is a predicate on the RAW frame whose signature
        # carries no `rpy`, so a PEG guard would have to re-derive the filer's frequency from the
        # stamp while `peg_local` receives it from the caller -- two statements of one domain,
        # resolved from two different places, which is the silently-divergent pair this repo
        # keeps getting bitten by.  PEG's domain is also INTRINSIC to its formula (without a
        # positive trailing EPS there is no P/E for a growth rate to be compared against, so the
        # value does not exist rather than being inadmissible), which a `Guard` cannot express.
        # So it lives in `calcMetrics.peg_local`, once.
        #
        # THE DOMAIN ITSELF MOVED, and that is the substantive half.  The shipped guard required
        # BOTH eps legs positive, because the vendor's ratio-form growth leg is meaningless
        # across a sign change.  The local growth leg divides by |base|, so a NEGATIVE base is
        # admissible -- which is exactly what makes the TURNAROUND expressible.  What is still
        # refused is a non-positive CURRENT trailing EPS: no earnings, no P/E, nothing for a
        # growth rate to be compared against; and that single condition removes BOTH of the old
        # false-pass cells (where PE < 0 cancelled against growth < 0 into a positive PEG).
        #
        # THE FOUR SIGN CELLS, BEFORE -> AFTER, on the 61,832 newest-8 rows of the panel:
        #   eps_now>0 prev>0      34,500 / 12,673 passes  ->  34,569 / 12,513   (0.367 -> 0.362)
        #   eps_now<=0 prev>0      5,177 /      0 refused ->   4,225 /      0   refused
        #   eps_now<=0 prev<=0    17,035 /      0 refused ->  17,321 /      0   refused
        #   eps_now>0 prev<=0      5,089 /      0 FAILED  ->   4,489 /    773   (0.172)
        # Overall criterion pass rate 0.2050 -> 0.2149.  NOTE WHERE THAT COMES FROM: the normal
        # cell is essentially UNCHANGED (-0.53pp), so the fixed 0<PEG<1 bar has NOT been loosened
        # by the horizon change; the whole movement is the crossing rows going from AUTO-FAILED to
        # SCORED-ON-THEIR-OWN-P/E.
        # THE CROSSING CELL IS NERFED (CEO, 2026-08-05): it takes the POOL's median growth rate,
        # so the DEPTH OF THE PRIOR LOSS no longer enters the answer. Crossing 0.172 against
        # normal 0.362 = 0.48x, where before the nerf it was 0.890 = 2.46x. Lower, and the
        # mechanism is plain: a just-crossed trailing EPS is tiny, so the trailing P/E is
        # genuinely high and fails against a 6.72%/yr median. So those 5,089 rows are now
        # MEASURED rather than auto-failed -- NOT "now passing". See calcMetrics.peg_local.
        'PEG':                              {'Tier': 'C', 'Sign': 1},
        # GUARDED equity > 0 (sign-inversion fix, 2026-08-04) -- applied inside
        # calcMetrics.calc_special, since this criterion has a formula rather than a ratio spec.
        # ROE = netIncome/equity, so NEGATIVE equity with a NEGATIVE net income gives a
        # POSITIVE ROE that CLEARS THE 12% HURDLE while the company is losing money.  This is
        # the family's clean double negative, and the measurement says so: of the 22,402
        # passes, 2,255 (10.1%) are netIncome<0 AND equity<0 (that cell passes at 0.9295),
        # while netIncome>0 with equity<0 passes ZERO times out of 1,235 rows.  So guarding the
        # denominator and detecting the double negative remove the SAME 2,255 rows here -- the
        # two rules coincide on this criterion, and the guard is chosen for consistency with
        # the rest of the family.
        # SIGN STAYS +1: higher return on equity is still better.
        'returnOnEquity':                   {'Tier': 'C', 'Sign': 1, 'Guard': 'equity_positive'},
        'capitalExpenditureCoverageRatio':  {'Tier': 'C', 'Sign': 1},
                            }

    return preReq_dict, BoMetric_Calc_dict , BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict,BoMetric_special_dict

#  =========================================================================== #
#  THE VETO CHANNEL -- COLUMNS THAT ARE COMPUTED AND CARRIED BUT NEVER SCORED   #
#  (CEO, 2026-08-07)                                                           #
#  =========================================================================== #
#  `stage1_veto.POOL_FLAGS` gives each carve-out cohort the red flags that MEAN something on its
#  balance sheet.  Three of the four columns those cohort flag sets need did not exist on any
#  panel; this dict is what builds them.
#
#  IT IS A SEPARATE DICT AND NOT FOUR MORE ENTRIES IN `BoMetric_special_dict`, AND THAT IS THE
#  WHOLE POINT.  Every entry in the five SCORING dicts (base / mean / diff / unity / special)
#  carries a `Tier` and a `Sign`, and `calcScore.simpleScore_fromDict` iterates those five dicts
#  and hands each entry to `calcByTier` -- so a veto column declared there would have silently
#  added FOUR WEIGHTED STAGE-1 CRITERIA TO EVERY POOL, the general pool included.  That is a far
#  larger change than a veto, and it is not the one that was ruled for.  Entries here carry NO
#  `Tier` and NO `Sign` BY CONSTRUCTION: there is nothing for a scorer to read even if one
#  reached them, and `getDicts`'s return tuple -- the thing every scoring caller unpacks -- does
#  not contain this dict at all.  `_assert_veto_never_scored` below turns that into a checked
#  invariant instead of a comment.
#
#  THE ADMISSIBILITY GATE LIVES IN THE COLUMN, NOT IN `stage1_veto`.  Each entry's `Guard` names
#  a `calcMetrics.STAGE1_DOMAIN_GUARDS` predicate, exactly as the scoring dicts do, so an
#  inadmissible row reaches the veto as NaN and NO CONDITION IN `stage1_veto` CAN INVERT.  This
#  is the same discipline the eight sign-inverting criteria of 2026-08-04/05 were fixed with; do
#  not restate a domain inside a flag lambda.
#
#  NOT ON ANY EXISTING PANEL.  `ebitda` and `cashAndCashEquivalents` are CAPTURE-ONLY additions
#  from 2026-08-05 (see preReq_dict), so these columns exist only from a fetch made after that
#  change.  On an older panel `apply_veto` declines the affected POOL with `missing_columns` set
#  rather than raising.  NOTHING HERE IS BACKTESTABLE -- do not measure these on a saved pickle.
BoMetric_veto_dict = {
    #  REIT.  `ebitda / interestExpense`, tested `> 1` -- does the rent cover the interest bill.
    #  GUARDED `interestExpense > 0`: FMP reports 0 for a name with no interest expense, and
    #  without the guard the ratio would be +/-inf -> NaN and read as "cannot cover its
    #  interest" on a name with no interest to cover -- the measured
    #  `uInterestCoverage`-on-a-debt-free-name defect (1,668 sources, 21.5% of the universe).
    #  Its `FIELD_EVIDENCE` ruling is `not_evidence` (BENIGN) for that reason.  18 of 67 abstain.
    'reitEbitdaInterestCoverage': {'Guard': 'interest_expense_positive'},
    #  MINING.  The column holds EBITDA; the veto tests `> 0`.  GUARDED `revenue > 0`, so a
    #  PRE-PRODUCTION explorer is refused rather than read as a producer failing to earn on its
    #  ore.  `not_evidence` (BENIGN); 43 of 218 abstain, and they are precisely the names
    #  `cashRunwayOneYear` then judges.  11 of 218 fail.
    'producerEbitdaPositive':     {'Guard': 'revenue_positive'},
    #  MINING.  `cash + CFO x rpy`, tested `> 0`.  THE HORIZON IS STATUTORY, not chosen: IAS 1.25
    #  / ASC 205-40 require a going-concern assessment over at least TWELVE MONTHS, and `rpy` is
    #  what makes it twelve months for a semi-annual filer too.  NO GUARD: both operands measured
    #  0.00% NaN, so there is no benign refusal channel -- `counts`.  7 of 218 fail, all
    #  pre-revenue explorers.
    'cashRunwayOneYear':          {},
    #  MINING.  `totalStockholdersEquity`, tested `> 0`.  NO GUARD -- always admissible; the
    #  field is never absent and a degenerate one is adverse on any reading (`counts`, same shape
    #  as `returnOnAssets`).  4 of 218 fail.
    'equityPositive':             {},
}


def getVetoDict():
    """The VETO channel's column declarations -- see `BoMetric_veto_dict`.

    A SEPARATE ACCESSOR rather than an eighth element of `getDicts`'s tuple, deliberately.  Every
    caller that unpacks `getDicts` is a SCORING caller, and the invariant this channel exists to
    hold is that a veto column can never be reached as a scoring criterion.  Keeping it out of
    that tuple entirely makes the invariant structural: there is no unpack site at which the
    veto dict could be mistaken for `BoMetric_special_dict`.
    """
    return {k: dict(v) for k, v in BoMetric_veto_dict.items()}


def _assert_veto_never_scored():
    """A VETO COLUMN MUST NOT BE A SCORING CRITERION.  Checked at import, so it cannot reach a
    fetch, and checked against the COLUMN NAMES the scoring dicts actually produce -- not against
    their keys -- because the base/mean/unity/diff forms prefix theirs (`mBookToPrice`,
    `uCurrentRatio`, `dReturnOnAssets`) and a collision would be with the PREFIXED name.

    Two ways this could go wrong and both are covered: a veto key added to a scoring dict (it
    would then carry a Tier and be weighted into every pool's Stage-1 score), and a veto entry
    that carries a `Tier`/`Sign` in the hope of being scored somewhere.
    """
    (_preReq, calc, base, mean, diff, unity, special) = getDicts()
    scored = set(special)
    for key, spec in calc.items():
        for o in spec['Operation']:
            scored.add(key if o == 'n' else o + key[0].upper() + key[1:])
    clash = sorted(scored & set(BoMetric_veto_dict))
    if clash:
        raise AssertionError(
            'createDicts: %d veto column(s) are ALSO Stage-1 scoring criteria: %s. A veto column '
            'is computed and carried but NEVER scored; a name in both channels would be silently '
            'weighted into every pool\'s Stage-1 score by calcScore.calcByTier, which is a much '
            'larger change than a veto and is not what was ruled for.' % (len(clash), clash))
    graded = sorted(k for k, v in BoMetric_veto_dict.items()
                    if 'Tier' in v or 'Sign' in v)
    if graded:
        raise AssertionError(
            'createDicts: veto entrie(s) %s declare a Tier/Sign. Veto columns are not scored, so '
            'a Tier or a Sign on one is either dead weight or an attempt to score it -- neither '
            'is allowed. Move it to a scoring dict deliberately, or drop the key.' % graded)


_assert_veto_never_scored()


def getBaseMeanDiffUnitySpecialDicts():
    preReq_dict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict, BoMetric_special_dict = getDicts()

    return BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict, BoMetric_special_dict
def _buildPostDicts(weights):
    """Assemble the two Stage-2 ranking dicts from a flat {metric: w} vector.

    THE SHAPE IS THE CONTRACT.  postBoRank consumes these as
    `pd.DataFrame(columns=postBmRankingDict.keys())`, so the key ORDER is load-bearing;
    it is taken from scoringWeights.POSTBM_EQMET / POSTNEW_KEYS, which is the same
    order these dicts have always been written in.  Fresh inner dicts are built on
    every call so a caller that mutates the result cannot reach the canon.
    """
    postBmRankingDict = {k: {'eqMet': eq, 'w': weights[k]}
                         for k, eq in sw.POSTBM_EQMET.items()}
    postNewRankingDict = {k: {'w': weights[k]} for k in sw.POSTNEW_KEYS}
    return postBmRankingDict, postNewRankingDict


def getPostDict(macroAdj=1):
    """DECISIONAL Stage-2 weights -- the deployed mu theory prior.

    THE NUMBERS AND THEIR FULL PROVENANCE LIVE IN `scoringWeights.DEPLOYED`, which is
    the SINGLE SOURCE OF TRUTH for every scoring weight in the repo (single-source
    refactor, 2026-08-02).  Read that file to change a weight or to find out where
    0.072 on incomeQuality came from, which three metrics are deliberately zeroed and
    why, and why CycleHeat is negative -- all of it moved there VERBATIM, not
    summarised, so there is exactly one place to look.

    Nothing about this function's OUTPUT changed in the move: same two dicts, same keys
    in the same order, same eqMet mapping, same 'w' values and types.  Only the source
    of the numbers moved.  `getPostDict_legacy()` still holds the pre-2026-07-14
    vector for A/B.  `macroAdj` is unused, as before.
    """
    return _buildPostDicts(sw.DEPLOYED)


def getPostDict_legacy(macroAdj=1):
    """Pre-2026-07-14 double-counted DEFAULT weights (the certified 30.0% target-cell
    baseline). Retained for A/B against the promoted mu theory prior now decisional in
    getPostDict(); NOT decisional. Identical keys/eqMet/ordering to getPostDict -- only
    the 'w' values differ, so swapping this in reproduces the pre-promotion picks.

    Values in `scoringWeights.LEGACY` (same single source; the two vectors are held on
    ONE canonical key set so neither can silently drift off the other).  Note it keeps
    DcfToPrice = 0.35 where the deployed vector has 0.000 -- a real, deliberate
    difference, pinned by test_scoring_weights_single_source.py."""
    return _buildPostDicts(sw.LEGACY)

def getMetricDicts():
    preReqDict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict,BoMetric_unity_dict, BoMetric_special_dict = getDicts()

    return BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict
def dictCheckValid():
    dictCheckbool = True
    preReq_dict = getPreReqDict()
    BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, BoMetric_unity_dict = getMetricDicts()
    testdict = {**BoMetric_base_dict,**BoMetric_mean_dict,**BoMetric_diff_dict,**BoMetric_unity_dict,}
    matchVec = []
    for baseMet in testdict.keys():
        for entry in testdict[baseMet]:
            matchVec = []
            if entry == 'Upper' or entry == 'Lower':
                x = testdict[baseMet][entry]
                for fdl in preReq_dict:
                    if x in preReq_dict[fdl] or x == 'Identity':
                        matchVec.append(True)
                    else:
                        matchVec.append(False)
                if not any(matchVec):
                    print(x)
                    dictCheckbool = False

    # Check for duplicates
    bigPreReqList = []
    for key in preReq_dict:
        bigPreReqList = bigPreReqList + preReq_dict[key]

    if len(bigPreReqList) > len(list(set(bigPreReqList))):
        print('preReq_dict has duplicates')
        dictCheckbool = False
    # Check for unnecessary elements in preReq_dict


    return dictCheckbool

def getPreReqDict():
    #preReqDict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict,BoMetric_unity_dict, BoMetric_special_dict = getDicts()
    dictList = getDicts()

    return dictList[0]