"""universes.py -- THE single source of truth for the ticker-universe scopes.

WHY THIS MODULE EXISTS (2026-08-02)
===================================
The universe definition used to live in THREE places that could -- and did -- drift
apart:

  * `getData_gen.tickerfilterWrapper`  held the EXCHANGE CODES per filter,
  * `configuration.getDataFetchConfiguration` held the list of VALID FILTER NAMES,
  * `utils.get_lastIndexRead`          held a hardcoded whitelist of RESUME FILENAMES.

Each drift is silent and each has a real failure mode:

  1. THE DEFECT THIS MODULE WAS BUILT TO FIX.  `tickerfilterWrapper` filtered
     `exchangeShortName` on the strings ``'EURONEXT'`` and ``'OSE'``.  FMP serves
     NEITHER.  Its codes are ``PAR`` / ``AMS`` / ``BRU`` / ``LIS`` (the four Euronext
     markets) and ``OSL`` (Oslo).  Both strings matched ZERO rows, so every filter that
     named them silently returned a smaller universe than it claimed, and
     **1,046 statement-bearing common stocks were never in any run** -- the entire
     French, Dutch, Belgian, Portuguese and Norwegian small/mid-cap opportunity set,
     which is exactly the neglected-small-cap pool the filter's thesis targets.
     Corroborated three ways: (a) the live `available-traded/list` intersected with
     `financial-statement-symbol-lists` returns 0 rows for both codes; (b) the saved
     2026-01-08 panel's `Tickers_df` contains only NASDAQ/NYSE/LSE/TSX/STO/XETRA/ICE;
     (c) the deployed top-100 carries not one `.PA`/`.AS`/`.BR`/`.LS`/`.OL` suffix.
     `stock_US1_EU2` was the worst case -- NYSE + NASDAQ + EURONEXT, i.e. a
     "US + Europe" universe that was silently US-ONLY.

  2. A NAME THAT VALIDATES BUT DOES NOT RESOLVE.  `tickerfilterWrapper` was a chain of
     `if`/`elif` with `df = tickdf` as the fall-through, so a filter name accepted by
     `configuration` but missing from the wrapper would have returned the WHOLE
     ~50,000-name pre-filter table with no error.  `resolve()` raises instead.

  3. AN UNRESUMABLE UNIVERSE.  `utils.get_lastIndexRead`'s whitelist listed four
     filenames and omitted `stock_US1_EU1` / `stock_US1_EU2`, so `-startfromlastindex`
     raised 'Not Implemented' for two filters that `configuration` accepted. The
     whitelist is now DERIVED from this registry, which fixes those two as a byproduct
     and means a newly-added universe can never be born unresumable.

EVERY EXCHANGE CODE BELOW WAS VERIFIED AGAINST LIVE FMP DATA on 2026-08-02, not read
from `Notes/stockExchangeList.txt` -- that file is stale and actively wrong (it labels
Korea's `KSC` as Kuwait and Iceland's `ICE` as Intercontinental Exchange).  The counts
in `_VERIFIED_COUNTS` are members of `available-traded/list` INTERSECT
`financial-statement-symbol-lists` with `type == 'stock'`, i.e. the exact set
`get_tickers` feeds to the exchange filter.

CONTINUITY -- AND THE THING IT COSTS
====================================
Artifact filenames (`Bometric_dic-fmp_stock_NA1_EU1_*`) and the resume file
(`lastIndexOfRead_fmp_<filter>.txt`) embed the filter NAME, so every existing name
still resolves and nothing orphans.  But the fix means `stock_NA1_EU1` now DENOTES A
DIFFERENT UNIVERSE than the one that produced every existing artifact.  Two numbers,
kept apart because conflating them is how a misleading figure gets made:

  * its DEFINITION now sums to 11,497 statement-bearing stocks across its exchange
    codes, where it used to sum to 10,451;
  * RESOLVED through the full path (type filter, exchange filter, instrument filter,
    delisted prune) it yields 10,693 today against 9,647 on the old definition -- both
    measured against the live lists (2026-08-02 counts, re-measured 2026-08-03 after
    the instrument filter moved ahead of membership selection, which removes one more
    name -- the ZCARW warrant -- from every US-containing universe).

Either way the delta is the same 1,046 names.

THE CHANGE IS PURELY ADDITIVE, measured rather than assumed (the instrument filter's
rule C is pairwise, so adding 1,046 lines COULD in principle have knocked out an
existing name; it did not -- resolving both definitions against the live lists gives
+1,046 / -0).  So the new membership is a strict SUPERSET of the old.  That is worth
knowing and it is NOT the same as comparability: every retained name is still there,
but the POOL it is scored against grew by ~11%, so a panel built before 2026-08-02 and
one built after are NOT comparable on membership, and any pooled statistic (a z-score,
a percentile cut, a top-100 selection, a beat-rate) differs for that reason alone --
even for a name present in both.  This project already
carries two irreconcilable beat-rate figures from an undocumented basis change; that is
why `definition_fingerprint()` exists and why `MEMBERSHIP_CHANGED_2026_08_02` makes the
run BANNER say so out loud rather than leaving it to be rediscovered.
"""

import hashlib
import os

# --------------------------------------------------------------------------- #
#  Live-verified statement-bearing common-stock counts per exchangeShortName.   #
#  Source: available-traded/list INTERSECT financial-statement-symbol-lists,     #
#  type == 'stock', 2026-08-02.  Recorded so a future reader can tell a code     #
#  that BROKE from a code that merely shrank.                                   #
# --------------------------------------------------------------------------- #
_VERIFIED_COUNTS = {
    # North America
    'NYSE': 2270, 'NASDAQ': 3871, 'AMEX': 256, 'TSX': 662, 'TSXV': 698,
    'NEO': 56, 'CNQ': 291, 'OTC': 9937, 'PNK': 60,
    # Europe -- currently wired
    'LSE': 2253, 'XETRA': 684, 'STO': 691, 'ICE': 20,
    # Europe -- RESTORED by this fix
    'PAR': 577, 'AMS': 103, 'BRU': 107, 'LIS': 35, 'OSL': 224,
    # Europe -- served by FMP, deliberately NOT wired (see EUROPE_NOT_WIRED)
    'CPH': 138, 'HEL': 154, 'SIX': 225, 'MIL': 209, 'BME': 153, 'VIE': 84,
    'ATH': 79, 'DUB': 20, 'WSE': 382, 'IST': 144, 'FSX': 239, 'STU': 23,
    'DUS': 12, 'HAM': 10, 'MUN': 6, 'PRA': 8, 'BUD': 6, 'TAL': 1, 'RIS': 1,
    'DXE': 11, 'IOB': 30, 'AQS': 3,
    # Asia -- WIRED 2026-08-05 (see ASIA_LIKELY_INVESTABLE). Same counts and same
    # provenance as ASIA_CANDIDATE_CODES; they belong HERE as well, because
    # `check_resolved_counts` skips any code whose verified count is 0 -- so an Asia code
    # absent from this dict would have its per-code dead-code floor SILENTLY DISABLED,
    # which is precisely how EURONEXT/OSE hid for the life of the project.
    'JPX': 3644, 'HKSE': 2376, 'KSC': 857, 'KOE': 398, 'ASX': 1586, 'SES': 237,
    # Asia -- served, deliberately NOT wired (access-excluded; see ASIA_ACCESS_EXCLUDED)
    'TAI': 1027, 'TWO': 1081,
}

#  The two codes that matched NOTHING.  Kept as data, not prose, so the test suite can
#  assert they never come back into a filter definition.
DEAD_CODES = {
    'EURONEXT': ('PAR', 'AMS', 'BRU', 'LIS'),
    'OSE': ('OSL',),
}

#  Euronext-the-GROUP also operates Dublin (DUB, 20), Oslo (OSL, 224) and Milan
#  (MIL, 209) today.  The original code listed `EURONEXT` and `OSE` as SEPARATE
#  entries, so the faithful reading of intent is "the four classic Euronext markets,
#  plus Oslo as its own line" -- which is what is wired.  DUB and MIL are therefore
#  a DELIBERATE omission, not an oversight; wiring them is a CEO call, not a bug fix.
EURONEXT_CLASSIC = ('PAR', 'AMS', 'BRU', 'LIS')

#  European codes FMP serves that no filter references.  Documented, not wired --
#  adding any of them changes the universe, which is a product decision.
EUROPE_NOT_WIRED = ('CPH', 'HEL', 'SIX', 'MIL', 'BME', 'VIE', 'ATH', 'DUB',
                    'WSE', 'IST')

#  US codes FMP serves that no filter references.  AMEX (NYSE American) is the
#  notable one: 256 statement-bearing common stocks, a genuine US small-cap venue,
#  absent from `stock_US1` and from every other filter.  Flagged for a CEO decision
#  rather than added, because "US only" changing meaning is a universe change.
US_NOT_WIRED = ('AMEX', 'TSXV', 'NEO', 'CNQ', 'OTC', 'PNK')

# --------------------------------------------------------------------------- #
#  ASIA -- NOW WIRED, AND ONLY BECAUSE THE DEDUP BLOCKER IS CLOSED (2026-08-05). #
#                                                                               #
#  The blocker below was never data: every code here returns statement-bearing    #
#  stocks on the current key.  It was DEDUP.  Korean preferred lines share the     #
#  common's numeric root AND its company name verbatim and trade at 30-60%        #
#  discounts, so on a cheapness screen they rank STRAIGHT TO THE TOP -- they look  #
#  like the same company at half price, because that is precisely what the data    #
#  says.  The share-class filter caught 1 of 196: rule B keys on a `-P<letters>`   #
#  suffix and the Korean convention is a suffix ON THE ROOT, which no rule saw.    #
#                                                                               #
#  WHAT CHANGED: carveOut's canonical-choice dedup (2026-08-05) groups an issuer's #
#  lines on FUNDAMENTALS and then picks the canonical member, and its Korean marker #
#  demotes every line whose 6th character is not `0`.  Verified on the live         #
#  2026-08-04 list: in ALL 91 multi-line Korean families (196 symbols) the unique    #
#  canonical member is exactly the `...0` common -- zero commons demoted, zero        #
#  preferreds missed.                                                              #
#                                                                               #
#  THE DEPENDENCY IS ENFORCED IN CODE, NOT BY THIS COMMENT.  Any universe wiring    #
#  KSC/KOE goes through `getData_gen.assert_korea_dedup_ready`, which RAISES if the  #
#  canonicity marker is absent or wrong -- so Korea cannot be enabled by editing     #
#  this registry alone.                                                            #
#                                                                               #
#  AND WHAT IS STILL UNVERIFIED, LOUDLY: that gate proves the PICKING half only.     #
#  Whether FMP actually serves a Korean preferred its ISSUER'S statements -- i.e.     #
#  whether the family GROUPS at all -- needs statements, and no Korean fundamentals   #
#  exist locally.  If it does not, each preferred survives as its own singleton        #
#  issuer and the marker never gets a sibling to prefer.  That is a POST-FETCH         #
#  regression (test_dedup_issuer.py, the Korea MUST-MERGE gate) and it must be run     #
#  against the first Korea fetch before any Korean name in the ranking is trusted.     #
# --------------------------------------------------------------------------- #
ASIA_CANDIDATE_CODES = {
    'JPX': 3644,    # Japan
    'HKSE': 2376,   # Hong Kong
    'TAI': 1027, 'TWO': 1081,   # Taiwan (main + OTC board)
    'ASX': 1586,    # Australia
    'SES': 237,     # Singapore
    'KSC': 857, 'KOE': 398,     # Korea -- admissible ONLY with the dedup gate, see above
}
ASIA_BLOCKER = (
    'CLOSED 2026-08-05 by carveOut canonical-choice dedup + the '
    'getData_gen.assert_korea_dedup_ready gate. It was never a data gap: every Asia code '
    'returns statements on the current key. It was DEDUP -- Korean preferreds (KSC/KOE) '
    'share the common\'s numeric root and company name, trade 30-60% below it, and the '
    'share-class filter caught 1 of 196. RESIDUAL, UNVERIFIED: that FMP serves a Korean '
    'preferred its issuer\'s STATEMENTS (so the family groups at all) is unproven until '
    'the post-fetch Korea MUST-MERGE regression runs on a real Korea fetch.')

#  THE LIKELY-INVESTABLE ASIA SET (CEO, 2026-08-05): "Let's drop what is unlikely for me
#  to be able to invest in. Keep the ones that are likely, have the full universe as an
#  option. I'll check with my broker later."
#  Kept: Japan, Hong Kong, Korea, Australia, Singapore -- all reachable through a normal
#  international broker. Counts are the live-verified statement-bearing figures above.
ASIA_LIKELY_INVESTABLE = ('JPX', 'HKSE', 'KSC', 'KOE', 'ASX', 'SES')

#  DELIBERATELY EXCLUDED, and they are the BULK of what Asia would otherwise add: these
#  generally need local or qualified-foreign-investor access.
#    India     NSE + BSE   5,828 statement-bearing names
#    China A   SHH + SHZ   4,362   (Shanghai + Shenzhen A-shares)
#    Taiwan    TAI + TWO   2,108   (matches ASIA_CANDIDATE_CODES: 1,027 + 1,081)
#  Only the Taiwan figure comes from this module's own live-verified counts; the India and
#  China figures are as supplied with the CEO's decision and are NOT in _VERIFIED_COUNTS,
#  so treat them as the decision's basis rather than as measured here. On the raw live
#  2026-08-04 available-traded table the corresponding type=='stock' line counts are
#  NSE 2,427 + BSE 3,731, SHH 1,989 + SHZ 2,395, TAI 1,052 + TWO 1,084 -- upper bounds,
#  since that table is not intersected with the statement-symbol list.
ASIA_ACCESS_EXCLUDED = {
    'NSE': 'India -- generally needs local/FPI access',
    'BSE': 'India -- generally needs local/FPI access',
    'SHH': 'China A-shares (Shanghai) -- QFII/Stock-Connect access',
    'SHZ': 'China A-shares (Shenzhen) -- QFII/Stock-Connect access',
    'TAI': 'Taiwan main board -- foreign-investor registration',
    'TWO': 'Taiwan OTC board -- foreign-investor registration',
}

#  ASIA/EM CODES FMP SERVES THAT ARE IN NEITHER SET. Not a decision -- an omission made
#  visible, in the same spirit as EUROPE_NOT_WIRED, so nobody reads ASIA_LIKELY_INVESTABLE
#  as "all the investable Asia there is". Raw type=='stock' counts, live 2026-08-04:
#  SET 878 (Thailand), JKT 761 (Indonesia), KLS 299 (Malaysia), SAU 368 (Saudi),
#  TLV 478 (Israel). Wiring any of them is a CEO call.
ASIA_NOT_WIRED = ('SET', 'JKT', 'KLS', 'SAU', 'TLV')


def asia_codes():
    """Every Asia code the module knows, wired or not (see ASIA_LIKELY_INVESTABLE for
    the set an actual universe uses)."""
    return tuple(sorted(ASIA_CANDIDATE_CODES))


def korea_codes():
    """The Korean exchange codes -- the ones whose presence triggers the dedup gate."""
    return ('KSC', 'KOE')


# --------------------------------------------------------------------------- #
#  THE CURATED TEST UNIVERSE                                                    #
#                                                                               #
#  WHY A FROZEN LIST AND NOT `-nrTaT N`.  `-nrTaT` truncates to the FIRST N rows #
#  of `available-traded/list` (getData_fmp: `Tickers_df.iloc[startindex:]` then   #
#  `cntr == nrTaT`).  That is an arbitrary positional prefix, so it systematically#
#  under-represents semi-annual filers, non-USD reporters and whole cohorts, and  #
#  its membership moves whenever FMP reorders the list.  Iterating on it produces #
#  confident wrong conclusions.  This list is instead FIXED, CHECKED IN, and      #
#  REPRESENTATIVE BY CONSTRUCTION.                                               #
#                                                                               #
#  Derived ONCE (2026-08-02) from the saved 2026-01-08 panel using the repo's own #
#  classifiers -- reporting_period for frequency, carveOut.classify for cohort,   #
#  the run's own lenfail/datefail/pricefail buckets for the gate cases -- then    #
#  every member re-verified present in the LIVE list and confirmed to sit on a    #
#  `stock_NA1_EU1` exchange.  It is a SUBSET of the production universe; that is  #
#  enforced by test, because a test universe containing names production cannot   #
#  see would test nothing.                                                       #
#                                                                               #
#  !!! WHAT THIS UNIVERSE CANNOT TEST -- READ THIS BEFORE QUOTING ANY NUMBER !!! #
#  Every pool-relative computation in the pipeline is MEANINGLESS here.          #
#  Cross-sectional z-scores, percentile cuts, cohort-relative scoring, the        #
#  top-100 pool and the top-20/top-5 selections are all functions of POOL         #
#  COMPOSITION.  On 142 names a z-score is computed against 142 peers instead of  #
#  ~11,500, and the "top 20" is the top 15% of the universe rather than the top   #
#  0.17%.  Scores, ranks and any beat-rate from a test-universe run are NOT       #
#  COMPARABLE TO PRODUCTION and must never be reported as a result.  The test     #
#  universe answers "does the code path run correctly on this shape of input",    #
#  never "what does the filter pick".  `run_banner()` says this on every run.     #
#                                                                               #
#  Second-order tag meanings:                                                    #
#    eu-restored   -- a member of the newly-restored European exchanges           #
#    edge-*        -- a named edge case, taken from the pipeline's own artifacts   #
#    filter-remove -- MUST be removed by filter_non_common_instruments             #
#    filter-keep   -- MUST survive it (the delete-a-common failure mode)           #
#    ccy-*         -- a specific reporting-currency / magnitude case               #
#    cohort-fill   -- deterministic stratified fill; the rule is in the reason     #
# --------------------------------------------------------------------------- #
TEST_UNIVERSE = (
    # --- the newly-restored European exchanges (the whole point of the fix) ---
    ('ALLUX.PA', 'eu-restored',
     'PAR/EUR, semi-annual (live-verified 2026-08-02): small French industrial on Euronext Growth Paris'),
    ('AI.PA', 'eu-restored',
     'PAR/EUR: Air Liquide. A NEW cross-listing pair created by the fix -- its partner AIL.DE (XETRA) is a member too, so issuer-dedup actually FIRES on a restored code. carveOut dedup edges are PAIRWISE over the pool, so the partner has to be IN THE LIST or this rationale is a claim the list does not honour (it was, until 2026-08-03)'),
    ('NEDAP.AS', 'eu-restored',
     'AMS/EUR, semi-annual (live-verified): small Dutch industrial'),
    ('SHELL.AS', 'eu-restored',
     'AMS/EUR: Shell plc, large cap. A THREE-venue cross-listing (AMS + SHEL.L + SHEL on NYSE), all three members, so dedup collapses a genuine multi-venue group that spans a restored code'),
    ('CAMB.BR', 'eu-restored',
     'BRU/EUR, semi-annual (live-verified): small Belgian metals/chemicals recycler -- also probes the Mining carve on a restored code'),
    ('COR.LS', 'eu-restored',
     'LIS/EUR, quarterly (live-verified): Corticeira Amorim, Portuguese mid cap'),
    ('GALP.LS', 'eu-restored',
     'LIS/EUR: Galp Energia -- second Lisbon name so the code is not represented by one ticker'),
    ('GYL.OL', 'eu-restored',
     'OSL/NOK, quarterly, 40 rows over 9.8yr (live-verified): small Norwegian publisher'),
    ('RING.OL', 'eu-restored',
     'OSL/NOK, quarterly (live-verified): Norwegian savings bank -- BalanceSheetFin cohort on a restored code'),
    ('SALM.OL', 'eu-restored',
     'OSL/NOK: SalMar ASA, Norwegian salmon farming mid cap'),
    ('ACKB.BR', 'eu-restored',
     'BRU/EUR: Ackermans & van Haaren -- second Brussels name so the code is not represented by a single ticker'),

    # --- ISSUER-GROUP CLOSURE (added 2026-08-03, CORRECTED 2026-08-03) ----------
    #  carveOut's dedup edges are PAIRWISE OVER THE POOL, so a member whose same-issuer
    #  sibling is absent is deduped DIFFERENTLY here than in production -- the exact
    #  production/test divergence this list exists to prevent. A name-based audit found
    #  SIXTEEN such members, not the three that were spotted by eye.
    #
    #  THE FIRST VERSION OF THIS BLOCK ASSERTED "each is a legitimate common or ordinary
    #  cross-listing, never an instrument line". THAT WAS FALSE, and it was false because
    #  the audit that produced this block grouped by NORMALISED NAME while the pipeline
    #  dedups with `carveOut._issuer_components`, which union-finds a FUNDAMENTALS
    #  FINGERPRINT. Name-matching finds cross-listings; it does NOT tell a cross-listing
    #  apart from a baby bond or a preferred GDR, because those carry the common's name
    #  and statements verbatim. Two of the seventeen were instrument lines:
    #
    #    HTFC   -- a Horizon Technology Finance BABY BOND. Byte-identical fundamentals to
    #              HRZN, quoted at 24.85 (near the $25 par of a US retail note) against
    #              HRZN's 4.32, and listed on NYSE while the common is on NASDAQ.
    #    SMSD.L -- the Samsung Electronics PREFERRED GDR. SMSD.L/SMSN.L = 0.708, and the
    #              Seoul pair 005935.KS/005930.KS (preferred/common) = 0.728. It is the
    #              KOREAN PREFERRED TRAP the Asia note above warns about -- a 29% discount
    #              on identical fundamentals, i.e. exactly what ranks to the top of a
    #              cheapness screen -- imported into the reference list by hand.
    #
    #  Both are REMOVED. The lesson is recorded rather than just the fix: a closure audit
    #  must use the pipeline's OWN issuer grouping, and price behaviour is what separates
    #  a cross-listing from an instrument line once names and fundamentals both match.
    #  Groups that CANNOT be closed without re-importing such a line are DECLARED in
    #  TEST_UNIVERSE_OPEN_GROUPS and reconciled against a fresh derivation by test.
    ('AIL.DE', 'dedup-partner',
     "XETRA line of L'Air Liquide -- closes AI.PA's group so dedup FIRES on a restored code (PAR)"),
    ('SHEL.L', 'dedup-partner',
     'LSE line of Shell plc -- with SHELL.AS and SHEL makes a three-venue group spanning a restored code (AMS)'),
    ('SHEL', 'dedup-partner',
     'NYSE line of Shell plc -- the third leg of that group'),
    ('STLA', 'dedup-partner',
     "NYSE line of Stellantis -- closes 8TI.DE's group (which also carries the zero-marketCap edge case)"),
    ('STLAP.PA', 'dedup-partner',
     'PAR line of Stellantis -- a SECOND dedup group spanning a restored code, and an extra Paris member'),
    ('MS', 'dedup-partner',
     "NYSE common of Morgan Stanley -- closes DWD.DE's group. Also the sibling that makes the MS-PE preferred a complete rule-B demonstration"),
    ('0QYU.L', 'dedup-partner',
     'LSE international-order-book line of Morgan Stanley -- the third leg of that group'),
    ('WDC.DE', 'dedup-partner',
     'XETRA line of Western Digital -- closes the WDC / 0QZF.L group, which is also the duplicate-dated-rows case'),
    ('LIN.DE', 'dedup-partner',
     "XETRA line of Linde plc -- closes LIN's group (Mining cohort)"),
    ('IHG', 'dedup-partner',
     "NYSE line of InterContinental Hotels -- closes IHG.L's group (the GBp magnitude case)"),
    ('CGI.L', 'dedup-partner',
     "LSE line of Canadian General Investments -- closes CGI.TO's group (semi-annual CAD FIN-1)"),
    ('0NNU.L', 'dedup-partner',
     "LSE line of Nedap N.V. -- closes NEDAP.AS's group, so dedup fires on AMS too"),
    ('0LUS.L', 'dedup-partner',
     "LSE line of Welltower -- closes WELL's group (REIT cohort)"),
    ('0LHX.L', 'dedup-partner',
     "LSE line of U.S. Global Investors -- closes GROW's group (FIN-1 cohort)"),
    ('SMSN.L', 'dedup-partner',
     'Samsung Electronics COMMON GDR on LSE -- tracks 005930.KS (SMSN.L/BC94.L = 0.989). '
     "It does NOT close BC94.L's group: the third line SMSD.L is the PREFERRED GDR and "
     'is deliberately excluded, so BC94.L and SMSN.L are both DECLARED open groups'),

    # --- edge cases, taken from the pipeline's OWN fail/quality artifacts ---
    ('FLO', 'edge-dupdate',
     'US quarterly with 5 DUPLICATE-DATED rows on the 2026-01-08 panel -> exercises _align_statements_by_date positional fallback'),
    ('WDC', 'edge-dupdate',
     'US quarterly, 4 duplicate-dated rows AND a cross-listed LSE twin (0QZF.L) -> dupdate + dedup in one name'),
    ('0QZF.L', 'edge-dedup',
     'the LSE line of Western Digital -- the pre-existing cross-listing that must collapse to one issuer with WDC'),
    ('SYPR', 'edge-dupdate',
     'US micro cap (~$49M) with 4 duplicate-dated rows -> dupdate at the $50M band boundary'),
    ('MCFT', 'edge-zerocap',
     'marketCap == 0 on the panel -> the missing/zero market-cap path and the pending-currency band degrade'),
    ('8TI.DE', 'edge-zerocap',
     'XETRA/EUR semi-annual Stellantis line with marketCap == 0 -> zero cap on a NON-USD reporter'),
    ('MGX', 'edge-lenfail',
     'LENFAILED the >=16q history gate on the 2026-01-08 run (2024 IPO, short history). TIME-VARYING: accretes rows and will eventually pass -- see the checker note'),
    ('IBTA', 'edge-lenfail',
     'LENFAILED the history gate on 2026-01-08 (2024 IPO) -- second short-history case so the category does not empty when one graduates'),
    ('TMO.L', 'edge-datefail',
     'DATEFAILED on 2026-01-08 (newest statement predates compyear). TIME-VARYING: depends on -compyear, so treat as observed-not-asserted'),
    ('SMWH.L', 'edge-pricefail',
     'PRICEFAILED on 2026-01-08 -> the price-unavailable branch'),
    ('BDRX', 'edge-tiny',
     'NASDAQ semi-annual loss-maker with marketCap ~159 (reporting ccy) -> the extreme small-cap + loss + semi-annual corner'),

    # --- the share-class / instrument filter, BOTH directions ---
    ('MS-PE', 'filter-remove',
     'a Morgan Stanley PREFERRED series -- filter_non_common_instruments MUST remove it (rule B). Present so the filter is demonstrably firing, not assumed to'),
    ('GIPRW', 'filter-remove',
     'a WARRANT line (Generation Income Properties) -- MUST be removed, and ONLY rule C '
     'can do it: FMP gives the warrant the COMMON\'s name verbatim ("Generation Income '
     'Properties, Inc.", no "Warrants" token), so rule A does not fire and there is no '
     '-P suffix for rule B. RULE C IS PAIRWISE: it recognises GIPRW as GIPR+"W" only '
     'when the shorter same-name same-exchange sibling GIPR is ALSO in the universe, so '
     'a universe SUBSET that omits the sibling silently DISABLES the rule. GIPR is in '
     'the REIT fill below and is flagged load-bearing there'),
    ('INVE-A.ST', 'filter-keep',
     'STO dual-class COMMON -- must SURVIVE the filter and then dedup to one issuer with INVE-B.ST (the must-not-delete side)'),
    ('INVE-B.ST', 'filter-keep',
     'the B line of the same issuer -- the dedup partner'),

    # --- currency / magnitude extremes (the marketCap-in-reporting-currency hazard) ---
    ('TLK', 'ccy-extreme',
     'NYSE-listed IDR reporter, marketCap ~3.0e14 in reporting currency -> the worst case for USD banding'),
    ('KB', 'ccy-extreme',
     'NYSE-listed KRW reporter, ~4.4e13 -> BalanceSheetFin + a second exotic reporting currency'),
    ('ISB.IC', 'ccy-isk',
     'Iceland/ISK BalanceSheetFin -- the ICE code plus ISK magnitudes'),
    ('BRIM.IC', 'ccy-isk',
     'Iceland/ISK small cap (Brim hf.) -- second ICE name'),
    ('ICESEA.IC', 'ccy-isk',
     'Iceland/ISK smallest ICE member -> ISK + small cap'),
    ('IHG.L', 'ccy-gbp',
     'LSE semi-annual, GBp-denominated, very large reported marketCap -> pence-vs-pounds magnitude case'),
    ('LXB.ST', 'ccy-sek',
     'STO semi-annual SEK loss-maker, small cap -- one of only 5 semi-annual Stockholm names'),
    ('BONAS.ST', 'ccy-sek',
     'STO semi-annual SEK REIT -> semi-annual REIT outside the LSE'),
    ('CGI.TO', 'ccy-cad',
     'TSX semi-annual CAD InvestmentVehicle -> semi-annual + CAD + FIN-1'),

    # ======================================================================= #
    #  DETERMINISTIC STRATIFIED FILL -- the bulk that gives each cohort enough
    #  members to exercise cohort scoring at all.  Selection rule, run ONCE on
    #  2026-08-02 and then FROZEN here (the list is the artifact; re-running the
    #  rule against live data would defeat the stability requirement):
    #    within each (cohort x stratum), take names with the DEEPEST history
    #    first, then by marketCap -- ASCENDING in the small-cap stratum (we want
    #    the genuinely tiny end) and DESCENDING in mid/large (ascending there
    #    re-picked the bottom of every band and pushed the loss-maker share to
    #    40 percent).  marketCap > 0 is required for the FILL, because sorting
    #    cap ascending put every zero-cap name first and made 27 percent of the
    #    first draft zero-cap -- zero cap is a WANTED case but belongs in the
    #    explicit picks (MCFT / 8TI.DE) where it is deliberate and counted.
    #  Strata per cohort: semiannual-LSE, semiannual-other, quarterly-small
    #  (<50M), quarterly-mid (50-300M), quarterly-large (>300M), loss-maker.
    #  The bracket after each name records what it was OBSERVED to be on the
    #  2026-01-08 panel; (rc) = reporting currency, NOT USD.
    # ======================================================================= #

    # --- deterministic stratified fill: cohort REIT ---
    ('TRAF.L', 'cohort-fill',
     'semiannual-LSE | Trafalgar Property Group Plc [LSE, semiannual, cap 222k(rc), 24 rows, LOSS]'),
    ('FLK.L', 'cohort-fill',
     'semiannual-LSE | Fletcher King Plc [LSE, semiannual, cap 3.74m(rc), 24 rows, profit]'),
    ('DGR.DE', 'cohort-fill',
     'semiannual-other | Deutsche Grundst?cksauktionen AG [XETRA, semiannual, cap 11.9m(rc), 24 rows, LOSS]'),
    ('SMWN.DE', 'cohort-fill',
     'semiannual-other | SM Wirtschaftsberatungs AG [XETRA, semiannual, cap 21.7m(rc), 24 rows, LOSS]'),
    ('GIPR', 'cohort-fill',
     'quarterly-small | Generation Income Properties, Inc. [NASDAQ, quarterly, cap 5.01m(rc), 24 rows, LOSS] LOAD-BEARING: it is also the rule-C anchor that makes the explicit GIPRW warrant removable -- do not drop it when re-curating this fill'),
    ('WHLR', 'cohort-fill',
     'quarterly-small | Wheeler Real Estate Investment Trust, Inc. [NASDAQ, quarterly, cap 6.07m(rc), 24 rows, profit]'),
    ('BRT', 'cohort-fill',
     'quarterly-mid | BRT Apartments Corp. [NYSE, quarterly, cap 282m(rc), 24 rows, LOSS]'),
    ('RMR', 'cohort-fill',
     'quarterly-mid | The RMR Group Inc. [NASDAQ, quarterly, cap 272m(rc), 24 rows, profit]'),
    ('BEKE', 'cohort-fill',
     'quarterly-large | KE Holdings Inc. [NYSE, quarterly, cap 145bn(rc), 24 rows, profit]'),
    ('WELL', 'cohort-fill',
     'quarterly-large | Welltower Inc. [NYSE, quarterly, cap 120bn(rc), 24 rows, profit]'),
    ('SQFT', 'cohort-fill',
     'loss-maker | Presidio Property Trust, Inc. [NASDAQ, quarterly, cap 6.48m(rc), 24 rows, LOSS]'),

    # --- deterministic stratified fill: cohort Mining ---
    ('ARK.L', 'cohort-fill',
     'semiannual-LSE | Arkle Resources PLC [LSE, semiannual, cap 1.65k(rc), 24 rows, LOSS]'),
    ('PREM.L', 'cohort-fill',
     'semiannual-LSE | Premier African Minerals Limited [LSE, semiannual, cap 65.7k(rc), 24 rows, LOSS]'),
    ('IONR', 'cohort-fill',
     'semiannual-other | ioneer Ltd [NASDAQ, semiannual, cap 6.01m(rc), 24 rows, LOSS]'),
    ('PME.TO', 'cohort-fill',
     'semiannual-other | Sentry Select Primary Metals Corp. [TSX, semiannual, cap 18.6m(rc), 24 rows, profit]'),
    ('ZNWD.L', 'cohort-fill',
     'quarterly-small | Zinnwald Lithium Plc [LSE, quarterly, cap 29.5k(rc), 24 rows, LOSS]'),
    ('RKDA', 'cohort-fill',
     'quarterly-small | Arcadia Biosciences, Inc. [NASDAQ, quarterly, cap 4.74m(rc), 24 rows, profit]'),
    ('SAU.TO', 'cohort-fill',
     'quarterly-mid | St. Augustine Gold and Copper Limited [TSX, quarterly, cap 298m(rc), 24 rows, LOSS]'),
    ('ADN.TO', 'cohort-fill',
     'quarterly-mid | Acadian Timber Corp. [TSX, quarterly, cap 297m(rc), 24 rows, profit]'),
    ('PKX', 'cohort-fill',
     'quarterly-large | POSCO Holdings Inc. [NYSE, quarterly, cap 22.3e12(rc), 24 rows, profit]'),
    ('LIN', 'cohort-fill',
     'quarterly-large | Linde plc [NASDAQ, quarterly, cap 223bn(rc), 24 rows, profit]'),
    ('ZIOC.L', 'cohort-fill',
     'loss-maker | Zanaga Iron Ore Company Limited [LSE, semiannual, cap 75.2k(rc), 24 rows, LOSS]'),
    ('KP2.L', 'cohort-fill',
     'any | Kore Potash plc [LSE, semiannual, cap 159k(rc), 24 rows, LOSS]'),

    # --- deterministic stratified fill: cohort InvestmentVehicle ---
    ('MAC.L', 'cohort-fill',
     'semiannual-LSE | Marechale Capital Plc [LSE, semiannual, cap 1.32m(rc), 24 rows, LOSS]'),
    ('TIR.L', 'cohort-fill',
     'semiannual-LSE | Tiger Alpha Plc [LSE, semiannual, cap 1.4m(rc), 24 rows, LOSS]'),
    ('SVE.DE', 'cohort-fill',
     'semiannual-other | Shareholder Value Beteiligungen AG [XETRA, semiannual, cap 59.4m(rc), 24 rows, profit]'),
    ('PZS.DE', 'cohort-fill',
     'semiannual-other | Scherzer & Co. AG [XETRA, semiannual, cap 64.9m(rc), 24 rows, profit]'),
    ('WINV.L', 'cohort-fill',
     'quarterly-small | Worsley Investors Limited [LSE, quarterly, cap 9.11m(rc), 24 rows, profit]'),
    ('GROW', 'cohort-fill',
     'quarterly-small | U.S. Global Investors, Inc. [NASDAQ, quarterly, cap 35.7m(rc), 24 rows, profit]'),
    ('TPVG', 'cohort-fill',
     'quarterly-mid | TriplePoint Venture Growth BDC Corp. [NYSE, quarterly, cap 233m(rc), 24 rows, profit]'),
    ('VEFAB.ST', 'cohort-fill',
     'quarterly-mid | Vef Ab [STO, quarterly, cap 203m(rc), 24 rows, profit]'),
    ('INDU-A.ST', 'cohort-fill',
     'quarterly-large | AB Industriv?rden (publ) [STO, quarterly, cap 161bn(rc), 24 rows, profit]'),
    ('INDU-C.ST', 'cohort-fill',
     'quarterly-large | AB Industriv?rden (publ) [STO, quarterly, cap 161bn(rc), 24 rows, profit]'),
    ('GUN.L', 'cohort-fill',
     'loss-maker | Gunsynd Plc [LSE, semiannual, cap 1.57m(rc), 24 rows, LOSS]'),
    ('WSL.L', 'cohort-fill',
     'any | Worldsec Limited [LSE, semiannual, cap 1.75m(rc), 24 rows, LOSS]'),

    # --- deterministic stratified fill: cohort FinManager ---
    ('QBT.L', 'cohort-fill',
     'semiannual-LSE | Quantum Blockchain Technologies Plc [LSE, semiannual, cap 16k(rc), 24 rows, LOSS]'),
    ('SEED.L', 'cohort-fill',
     'semiannual-LSE | Seed Innovations Limited [LSE, semiannual, cap 3.09m(rc), 24 rows, profit]'),
    ('MKZ-UN.TO', 'cohort-fill',
     'semiannual-other | Mackenzie Master Limited Partnership [TSX, semiannual, cap 2.6m(rc), 24 rows, profit]'),
    ('PDV.TO', 'cohort-fill',
     'semiannual-other | Prime Dividend Corp. [TSX, semiannual, cap 4.88m(rc), 24 rows, profit]'),
    ('AAB.TO', 'cohort-fill',
     'quarterly-small | Aberdeen International Inc. [TSX, quarterly, cap 4.81m(rc), 24 rows, profit]'),
    ('NCPL', 'cohort-fill',
     'quarterly-small | Netcapital Inc. [NASDAQ, quarterly, cap 10.4m(rc), 24 rows, LOSS]'),
    ('O4B.DE', 'cohort-fill',
     'quarterly-mid | OVB Holding AG [XETRA, quarterly, cap 288m(rc), 24 rows, profit]'),
    ('HRZN', 'cohort-fill',
     'quarterly-mid | Horizon Technology Finance Corporation [NASDAQ, quarterly, cap 263m(rc), 24 rows, profit]'),
    ('NMR', 'cohort-fill',
     'quarterly-large | Nomura Holdings, Inc. [NYSE, quarterly, cap 2.83e12(rc), 24 rows, profit]'),
    ('DWD.DE', 'cohort-fill',
     'quarterly-large | Morgan Stanley [XETRA, quarterly, cap 250bn(rc), 24 rows, profit] -- '
     'its partners MS (NYSE) and 0QYU.L are members, so production and the test universe '
     'dedup this issuer IDENTICALLY. Until 2026-08-03 only this XETRA line was in the list, '
     'so production dropped it at dedup and a test run kept it: a live production/test '
     'divergence inside the artifact built to avoid divergence'),
    ('MA10.DE', 'cohort-fill',
     'loss-maker | Binect AG [XETRA, semiannual, cap 6.24m(rc), 24 rows, LOSS]'),
    ('FKE.L', 'cohort-fill',
     'any | Fiske plc [LSE, semiannual, cap 7.68m(rc), 24 rows, profit]'),

    # --- deterministic stratified fill: cohort BalanceSheetFin ---
    ('LEND.L', 'cohort-fill',
     'semiannual-LSE | Sancus Lending Group Limited [LSE, semiannual, cap 3.04m(rc), 24 rows, profit]'),
    ('MFX.L', 'cohort-fill',
     'semiannual-LSE | Manx Financial Group PLC [LSE, semiannual, cap 27.5m(rc), 24 rows, profit]'),
    ('FRS.DE', 'cohort-fill',
     'semiannual-other | Foris AG [XETRA, semiannual, cap 18.4m(rc), 24 rows, LOSS]'),
    ('ALG.DE', 'cohort-fill',
     'semiannual-other | ALBIS Leasing AG [XETRA, semiannual, cap 59.3m(rc), 24 rows, profit]'),
    ('AMGO.L', 'cohort-fill',
     'quarterly-small | Amigo Holdings PLC [LSE, quarterly, cap 2.01m(rc), 24 rows, LOSS]'),
    ('AIHS', 'cohort-fill',
     'quarterly-small | Senmiao Technology Limited [NASDAQ, quarterly, cap 3m(rc), 24 rows, LOSS]'),
    ('PCB', 'cohort-fill',
     'quarterly-mid | PCB Bancorp [NASDAQ, quarterly, cap 299m(rc), 24 rows, profit]'),
    ('CBAN', 'cohort-fill',
     'quarterly-mid | Colony Bankcorp, Inc. [NYSE, quarterly, cap 297m(rc), 24 rows, profit]'),
    ('SHG', 'cohort-fill',
     'quarterly-large | Shinhan Financial Group Co., Ltd. [NYSE, quarterly, cap 35.4e12(rc), 24 rows, profit]'),
    ('MUFG', 'cohort-fill',
     'quarterly-large | Mitsubishi UFJ Financial Group, Inc. [NYSE, quarterly, cap 27.3e12(rc), 24 rows, profit]'),
    ('CRWN.TO', 'cohort-fill',
     'loss-maker | Crown Capital Partners Inc. [TSX, quarterly, cap 3.41m(rc), 24 rows, LOSS]'),
    ('HUIZ', 'cohort-fill',
     'any | Huize Holding Limited [NASDAQ, quarterly, cap 7.48m(rc), 24 rows, profit]'),

    # --- deterministic stratified fill: cohort general ---
    ('CLON.L', 'cohort-fill',
     'semiannual-LSE | Clontarf Energy plc [LSE, semiannual, cap 2.62k(rc), 24 rows, LOSS]'),
    ('EME.L', 'cohort-fill',
     'semiannual-LSE | Empyrean Energy Plc [LSE, semiannual, cap 3.82k(rc), 24 rows, LOSS]'),
    ('CIZ.L', 'cohort-fill',
     'semiannual-LSE | Cizzle Biotechnology Holdings Plc [LSE, semiannual, cap 6.15k(rc), 24 rows, LOSS]'),
    ('AEG.L', 'cohort-fill',
     'semiannual-LSE | Active Energy Group Plc [LSE, semiannual, cap 8.1k(rc), 24 rows, LOSS]'),
    ('TRP.L', 'cohort-fill',
     'semiannual-LSE | Tower Resources plc [LSE, semiannual, cap 11.2k(rc), 24 rows, LOSS]'),
    ('XMF-A.TO', 'cohort-fill',
     'semiannual-other | M Split Corp. [TSX, semiannual, cap 1.3m(rc), 24 rows, LOSS]'),
    ('CHNR', 'cohort-fill',
     'semiannual-other | China Natural Resources, Inc. [NASDAQ, semiannual, cap 4.95m(rc), 24 rows, LOSS]'),
    ('RTC.DE', 'cohort-fill',
     'semiannual-other | RealTech AG [XETRA, semiannual, cap 5.6m(rc), 24 rows, LOSS]'),
    ('A6T.DE', 'cohort-fill',
     'semiannual-other | artec technologies AG [XETRA, semiannual, cap 5.81m(rc), 24 rows, LOSS]'),
    ('NC5A.DE', 'cohort-fill',
     'semiannual-other | NorCom Information Technology GmbH & Co. K [XETRA, semiannual, cap 6.36m(rc), 24 rows, LOSS]'),
    ('GNLN', 'cohort-fill',
     'quarterly-small | Greenlane Holdings, Inc. [NASDAQ, quarterly, cap 5.45k(rc), 24 rows, LOSS]'),
    ('PPBT', 'cohort-fill',
     'quarterly-small | Purple Biotech Ltd. [NASDAQ, quarterly, cap 15.1k(rc), 24 rows, LOSS]'),
    ('AKTX', 'cohort-fill',
     'quarterly-small | Akari Therapeutics, Plc [NASDAQ, quarterly, cap 33k(rc), 24 rows, LOSS]'),
    ('SPRB', 'cohort-fill',
     'quarterly-small | Spruce Biosciences, Inc. [NASDAQ, quarterly, cap 43.2k(rc), 24 rows, LOSS]'),
    ('0HH6.L', 'cohort-fill',
     'quarterly-small | Aqua Metals, Inc. [LSE, quarterly, cap 91.3k(rc), 24 rows, LOSS]'),
    ('AQMS', 'cohort-fill',
     'quarterly-small | Aqua Metals, Inc. [NASDAQ, quarterly, cap 91.3k(rc), 24 rows, LOSS]'),
    ('F3C.DE', 'cohort-fill',
     'quarterly-mid | SFC Energy AG [XETRA, quarterly, cap 300m(rc), 24 rows, LOSS]'),
    ('OLY.TO', 'cohort-fill',
     'quarterly-mid | Olympia Financial Group Inc. [TSX, quarterly, cap 300m(rc), 24 rows, profit]'),
    ('BW', 'cohort-fill',
     'quarterly-mid | Babcock & Wilcox Enterprises, Inc. [NYSE, quarterly, cap 299m(rc), 24 rows, profit]'),
    ('QUAD', 'cohort-fill',
     'quarterly-mid | Quad/Graphics, Inc. [NYSE, quarterly, cap 297m(rc), 24 rows, profit]'),
    ('FTG.TO', 'cohort-fill',
     'quarterly-mid | Firan Technology Group Corporation [TSX, quarterly, cap 297m(rc), 24 rows, profit]'),
    ('BC94.L', 'cohort-fill',
     'quarterly-large | Samsung Electronics Co., Ltd. [LSE, quarterly, cap 565e12(rc), 24 rows, profit]'),
    ('EC', 'cohort-fill',
     'quarterly-large | Ecopetrol S.A. [NYSE, quarterly, cap 74.2e12(rc), 24 rows, profit]'),
    ('TSM', 'cohort-fill',
     'quarterly-large | Taiwan Semiconductor Manufacturing Company [NYSE, quarterly, cap 33.8e12(rc), 24 rows, profit]'),
    ('TM', 'cohort-fill',
     'quarterly-large | Toyota Motor Corporation [NYSE, quarterly, cap 32.5e12(rc), 24 rows, profit]'),
    ('TYT.L', 'cohort-fill',
     'quarterly-large | Toyota Motor Corporation [LSE, quarterly, cap 32.5e12(rc), 24 rows, profit]'),
    ('KEP', 'cohort-fill',
     'quarterly-large | Korea Electric Power Corporation [NYSE, quarterly, cap 25.2e12(rc), 24 rows, profit]'),
    ('AIEA.L', 'cohort-fill',
     'loss-maker | AIREA plc [LSE, semiannual, cap 105k(rc), 24 rows, LOSS]'),
    ('HSDT', 'cohort-fill',
     'any | Solana Company [NASDAQ, quarterly, cap 160k(rc), 24 rows, LOSS]'),
    ('FARN.L', 'cohort-fill',
     'any | Faron Pharmaceuticals Oy [LSE, semiannual, cap 254k(rc), 24 rows, LOSS]'),
    ('CNSP', 'cohort-fill',
     'any | CNS Pharmaceuticals, Inc. [NASDAQ, quarterly, cap 278k(rc), 24 rows, LOSS]'),
    ('XTIA', 'cohort-fill',
     'any | XTI Aerospace, Inc. [NASDAQ, quarterly, cap 324k(rc), 24 rows, LOSS]'),
    ('PIP.L', 'cohort-fill',
     'any | PipeHawk plc [LSE, semiannual, cap 610k(rc), 24 rows, LOSS]'),
)


# =========================================================================== #
#  THE ~3,000-NAME AD-HOC CURATED UNIVERSE  (CEO, 2026-08-06)                   #
#                                                                               #
#  "Something in between, we need to test the dedup properly also so just create #
#   an ad-hoc curated with roughly 3k stocks."                                   #
#                                                                               #
#  WHAT IT IS FOR.  It is the FIRST FETCH after ~20 changes landed on 2026-08-05  #
#  -- deliberately small, because the expectation is that something needs a tweak #
#  and a re-fetch, and nobody wants to discover that 21 hours in.  It is a        #
#  THROWAWAY SHAKEDOWN UNIVERSE, not a production scope.                         #
#                                                                               #
#  WHY NEITHER EXISTING OPTION WORKED.  `stock_TEST1` is 142 names, so a top-100  #
#  cut is 70% of the pool and means nothing.  A North-America-only scope would    #
#  not exercise CROSS-LISTING at all, which is the thing being tested.            #
#                                                                               #
#  THE SHAPE IS A BASE RULE **PLUS** AN EXPLICIT MUST-INCLUDE LIST, and the       #
#  "plus" is the whole design.  A pure sampling rule cannot be trusted to contain #
#  the specific measured dedup cases -- and a later edit to the rate would drop   #
#  them silently, which is exactly the failure `TEST_UNIVERSE` was built to stop.  #
#  So the measured cases are pinned BY SYMBOL, unioned in AFTER the base rule,    #
#  and a test asserts every one of them resolves.                                #
#                                                                               #
#  ---------------------------------------------------------------------------  #
#  THE BASE RULE: SAMPLE ON THE NORMALISED ISSUER **NAME**, NOT ON THE SYMBOL.    #
#  ---------------------------------------------------------------------------  #
#  This is the one non-obvious decision in the module and it is what makes the    #
#  universe able to test dedup AT ALL.                                           #
#                                                                               #
#  A per-SYMBOL sample (hash the ticker, keep 17%) SPLITS EVERY CROSS-LISTING     #
#  GROUP: UHS lands in, 0LJL.L does not, and the pool then contains one line of   #
#  a two-line issuer.  carveOut's dedup edges are PAIRWISE OVER THE POOL, so a    #
#  half-present group is not "a smaller test of dedup", it is NO test of dedup --  #
#  the same production/test divergence documented at TEST_UNIVERSE_OPEN_GROUPS,   #
#  reintroduced wholesale.  Sampling on `carveOut._norm_issuer_name` instead      #
#  puts every line of an issuer in the SAME BUCKET.                               #
#                                                                               #
#  BUT THE BUCKET IS ONLY HALF THE DECISION, AND THE FIRST CUT OF THIS UNIVERSE   #
#  GOT IT WRONG.  The bucket is a function of the NAME; the THRESHOLD it is       #
#  compared against is a function of the EXCHANGE.  Holding one half constant     #
#  closes nothing: with TSX/PAR/AMS unrated and NYSE/NASDAQ/LSE at 170, every     #
#  cross-listed issuer with bucket >= 170 SPLIT -- measured at 218 split groups    #
#  against 209 whole (51%) on the live 2026-08-04 table, including BMO/BMO.TO,    #
#  BCE/BCE.TO, BN/BN.TO and AEM/AEM.TO/0R2J.L, all of which arrive WHOLE in       #
#  stock_NA1_EU1.  The "splits ZERO" measurement that shipped with that cut was   #
#  real but covered only Korean families -- KSC and KOE SHARE rate 250, so they    #
#  cannot split against each other and the CROSS-RATE case was never measured.     #
#  Since the pins are by SYMBOL and not by GROUP, it reached the pinned cases too: #
#  five pinned groups arrived incomplete, `robertet s a` among them.               #
#                                                                               #
#  SO CLOSURE IS TWO RULES, NOT ONE (fixed 2026-08-06): hash the NAME, and then    #
#  resolve each issuer name ONCE against the MOST PERMISSIVE RATE among the        #
#  exchanges its lines occupy (`most_permissive_rate`) -- a line on an unrated,     #
#  take-all venue pulls the whole group in.  Cost of closing upward, measured:      #
#  +243 members (3,015 -> 3,258), ~+1,215 calls, ~+20 min of fetch.  THAT is what   #
#  makes a group entirely IN or entirely OUT.                                      #
#                                                                               #
#  WHAT THE NAME-HASH DOES **NOT** CLOSE, stated because the limit is real: it     #
#  groups by NAME, while the pipeline dedups by FUNDAMENTALS FINGERPRINT           #
#  (`carveOut._issuer_components`).  Name-matching finds cross-listings; it does   #
#  NOT catch a line FMP names differently from its sibling -- BWNB, whose name is   #
#  TRUNCATED to "Babcock & Wilcox Enterprises, I", is the documented case.  Those  #
#  groups can still arrive split.  Closure is BY CONSTRUCTION FOR SAME-NAMED       #
#  LINES ONLY, which covers the cross-listings, the Korean preferred families and  #
#  all three K-1 wrong-pick groups, and does not cover a renamed sibling.          #
#                                                                               #
#  IT IS SHA1, NOT `hash()`.  Python's `hash()` of a str is SALTED PER PROCESS     #
#  (PYTHONHASHSEED), so a `hash()`-based sample would return a DIFFERENT UNIVERSE  #
#  on every invocation -- unreproducible, and silently so.  It is also NOT a       #
#  positional prefix (`-nrTaT N`), for the reasons in the TEST_UNIVERSE note: a    #
#  prefix of `available-traded/list` moves whenever FMP reorders the list.         #
#  The sample is therefore a PURE FUNCTION of the issuer name -- stable across      #
#  runs, machines and list re-orderings, and re-derivable without checking in       #
#  3,000 tickers.                                                                 #
#                                                                               #
#  ---------------------------------------------------------------------------  #
#  WHY THESE EXCHANGES, AND WHY NOT ONE REGION                                    #
#  ---------------------------------------------------------------------------  #
#  The measured dedup cases LIVE on specific venues, so the venue list is derived  #
#  from them rather than chosen for tidiness:                                     #
#    NYSE/NASDAQ  UHS, AEM, MU, the veto ejects, the Chimera NOTES line            #
#    LSE          the 0-prefixed IOB lines (0LJL.L, 0KNY.L, 0QQF.L, 0HQ7.L) and    #
#                 the Samsung GDR pair -- the untradeable-at-size class (J-1)      #
#    TSX          AEM.TO, PEY.TO -- the different-historical-statements merge      #
#    PAR          Robertet + its certificat (CBE.PA)                              #
#    AMS          Heineken/Heineken Holding (MUST NOT merge) and Value8 + PREVA.AS  #
#    KSC/KOE      the preferred families that share the common's numeric root       #
#  TSX / PAR / AMS are TAKEN WHOLE (662 + 577 + 103 statement-bearing = 1,342):     #
#  they are small enough to cost little and each carries a case that a sample could #
#  thin.  NYSE / NASDAQ / LSE / KSC / KOE are SAMPLED, because taking them whole is  #
#  9,649 names -- the 21-hour run this universe exists to avoid.                    #
#                                                                               #
#  KOREA IS GATED, same as stock_ASIA1: `getData_gen.assert_korea_dedup_ready`     #
#  must pass or this universe does not resolve.                                   #
#                                                                               #
#  ---------------------------------------------------------------------------  #
#  !!! POOL-RELATIVE OUTPUT IS STILL NOT PRODUCTION OUTPUT !!!                     #
#  ---------------------------------------------------------------------------  #
#  Weaker than the TEST_UNIVERSE warning but NOT absent, and the difference is the  #
#  point of choosing ~3,000 over 142.  A z-score here is computed against ~3,000    #
#  peers instead of ~11,500, and a top-100 cut is the top ~3.3% of the pool rather  #
#  than the top ~0.9%.  So the top-100 selection and the veto's survivor count are  #
#  REAL NUMBERS on a REAL POOL -- which is what the CEO asked for -- but they are    #
#  still NOT COMPARABLE to a stock_NA1_EU1 run, because the pool differs.  Read     #
#  this run for BEHAVIOUR ("does the veto eject who we expect, does the dedup       #
#  merge what we measured"), never as a pick list.                                  #
# =========================================================================== #

#  Taken WHOLE.  Small, and each carries a measured case a sample could thin.
CURATED3K_TAKE_ALL = ('TSX', 'PAR', 'AMS')

#  SAMPLED, as a count out of CURATED3K_SAMPLE_DENOMINATOR issuer-name buckets.
#  Rates were CHOSEN BY MEASUREMENT against the live 2026-08-04 available-traded table
#  (see CURATED3K_SIZING below), not picked to look round.
CURATED3K_SAMPLED = {
    'NYSE': 170, 'NASDAQ': 170, 'LSE': 170,     # 17.0% of issuer names
    'KSC': 250, 'KOE': 250,                     # 25.0% -- Korea needs enough whole
                                                # preferred FAMILIES to be worth fetching
}
CURATED3K_SAMPLE_DENOMINATOR = 1000

#  Measured on the live 2026-08-04 `available_traded_raw` snapshot, type=='stock',
#  scaled to statement-bearing by each code's own live verified/raw ratio (NYSE 0.973,
#  NASDAQ 0.913, LSE 0.984, TSX 0.962, PAR 0.991, AMS 0.990, KSC 0.995, KOE 0.959).
#  These are ESTIMATES of the resolved size, not assertions -- the instrument filter, the
#  delisted prune and the sector filter all remove members afterwards, and the raw
#  snapshot is NOT intersected with `financial-statement-symbol-lists`.
#
#  RE-MEASURED 2026-08-06 AFTER THE GROUP-CLOSURE FIX, and the numbers moved -- upward
#  closure means a cross-listed issuer that misses the 17% threshold is now KEPT if any of
#  its lines sits on a take-all venue, so the sampled codes gained.  Replayed end-to-end
#  through `filter_non_common_instruments` + the base rule + the pin/sibling union over the
#  live 2026-08-04 table: 3,015 -> 3,258 members (+243, ~+1,200 calls, ~+20 min of fetch).
#  That cost is the price of the closure property and it was accepted deliberately.
CURATED3K_SIZING = {
    'TSX': 641, 'PAR': 582, 'AMS': 104,         # taken whole
    'NYSE': 445, 'NASDAQ': 620, 'LSE': 529,     # sampled at 17.0%, closed UPWARD
    'KSC': 223, 'KOE': 111,                     # sampled at 25.0%, closed UPWARD
    'STO': 1, 'XETRA': 2,                       # pins on unwired venues (EMBELL.ST etc.)
}
CURATED3K_ESTIMATED_MEMBERS = sum(CURATED3K_SIZING.values())      # 3,258

#  PER-COHORT FILL, MEASURED not guessed (2026-08-06).  Derived by replaying the base rule
#  over the live 2026-08-04 table, intersecting with the saved 2026-01-08 panel (1,719
#  sources) so that `carveOut.classify` has the REAL fundamentals its FIN-1 fingerprint
#  needs, then applying the resulting shares to the 3,045 estimate.
#
#  WHY THE PANEL AND NOT THE LIVE TABLE.  Classifying off the sector map ALONE puts
#  InvestmentVehicle at 2, which is wrong and would have been a bad answer to "is every
#  cohort populated": FIN-1 is identified by an equity/assets FINGERPRINT plus the name
#  rule, and with no fundamentals only the name half can fire.  The panel has
#  fundamentals, so this is the honest basis.
#
#  IT IS A LOWER BOUND ON COVERAGE, TWICE OVER: the panel predates PAR/AMS/Korea entirely
#  (so those 1,942 names contribute nothing to the shares), and it predates the instrument
#  filter.  Estimates, not assertions -- deliberately NOT pinned by a test, because the
#  real numbers arrive with the fetch and a hardcoded expectation would then be an
#  obstacle rather than a check.
#
#  BASIS, 2026-08-06: the SHARES are the measured quantity; the COUNTS are those shares
#  applied to `CURATED3K_ESTIMATED_MEMBERS`.  They previously summed to 3,045 -- the
#  PRE-group-closure member count -- so the table and the coded member count disagreed by
#  213 names and a reader adding the column up got the wrong universe size.  Re-scaled to
#  the closed count (x1.070) with the panel shares unchanged; `test_universes` now asserts
#  the sum matches, so the two cannot drift apart again.
CURATED3K_COHORT_ESTIMATE = {
    'general': 2419,            # 74.2% -- comfortably above 100, so the top-100 cut and
                                #          the veto's survivor count are real numbers
    'Mining': 356,              # 10.9%
    'BalanceSheetFin': 252,     #  7.7%  FIN-3
    'FinManager': 96,           #  2.9%  FIN-2
    'REIT': 95,                 #  2.9%
    'InvestmentVehicle': 40,    #  1.2%  FIN-1 -- the thinnest, and the one to watch
}

#  Fetch cost, on the SAME basis as TEST_UNIVERSE_API_CALLS: 5 statement calls per source
#  plus 3 for the universe build.  Two independent readings of the per-call rate, kept
#  apart because they disagree and the disagreement is the honest range:
#    * the curated 142-name run   713 calls in 8-15 min  ->  0.67-1.26 s/call
#    * the 2026-01-08 production run, 12h over 9,012 kept + 1,725 failed = 10,737
#      sources ATTEMPTED (a failed source still costs its calls)  ->  0.80 s/call
#  At 0.80-1.15 s/call, ~16,300 calls is 3.6-5.2 HOURS.  Against the ~12h/7,700 rough
#  arithmetic this lands slightly LOW, and the reason is that the 12h denominator should
#  be sources ATTEMPTED (10,737), not sources kept.
#  BOTH FIGURES WERE STALE UNTIL 2026-08-06 (reviewer): the comment said ~15,228 calls and
#  the band said (3.4, 4.9), i.e. the PRE-group-closure 3,045-member universe.  The
#  wallclock guard's +-0.5 h tolerance was wide enough to absorb the error, which is why
#  nothing failed -- it is tightened to +-0.05 h alongside this, so the guard actually
#  guards.  The band is a DERIVED quantity: re-derive it from CURATED3K_API_CALLS whenever
#  the member count moves.
CURATED3K_API_CALLS = 5 * CURATED3K_ESTIMATED_MEMBERS + 3        # ~16,293
CURATED3K_WALLCLOCK_HOURS = (3.6, 5.2)

# --------------------------------------------------------------------------- #
#  THE MUST-INCLUDE LIST.  Every entry is a case that has been MEASURED and that #
#  this run exists to OBSERVE.  Unioned in AFTER the base rule, so no edit to a   #
#  sample rate can drop one, and `test_universes` asserts each one resolves.      #
#                                                                               #
#  A must-include is NOT restricted to the base rule's exchanges -- it is applied  #
#  to the full post-instrument-filter table by symbol.  That is deliberate: it is  #
#  how EMBELL.ST (Stockholm), EIN.DE and DRW3.DE (XETRA) can be observed without   #
#  wiring STO and XETRA and paying for another 1,375 names.                        #
#                                                                               #
#  Tags:                                                                          #
#    must-merge      two lines that MUST collapse to one issuer                     #
#    must-not-merge  two lines that MUST STAY SEPARATE                              #
#    k1-wrongpick    merges correctly, picks the WRONG line today (register K-1)     #
#    lse-iob         untradeable-at-size IOB line (register J-1)                     #
#    veto-eject      on the superseded 5-flag veto's 17-name ejection list           #
#    veto-clean      clean on every criterion -- the CONTROL group                   #
#    group-closure   a sibling present only to close an issuer group                 #
# --------------------------------------------------------------------------- #
CURATED3K_MUST_INCLUDE = (
    # --- cross-listings that MUST MERGE to one line -------------------------- #
    ('UHS', 'must-merge',
     'Universal Health Services (NYSE). With 0LJL.L its fundamentals are BYTE-IDENTICAL, '
     'and the pair occupied TWO of the top 36 on the 2026-07-17 panel. The new dedup was '
     'verified OFFLINE to catch them; this run is the live confirmation. UHS is ALSO on '
     'the veto-eject list, so one name carries both observations'),
    ('0LJL.L', 'must-merge',
     'the LSE international-order-book line of Universal Health Services -- the other half '
     'of the byte-identical pair, and itself an IOB line and a veto eject'),
    ('AEM', 'must-merge',
     'Agnico Eagle Mines (NYSE). Merges with AEM.TO, but the two lines carry DIFFERENT '
     'HISTORICAL STATEMENTS -- aligned on their 22 common dates, netIncome differs on 20 of '
     '22, operatingIncome and interestExpense on 22 of 22, with no NaN mismatch anywhere. '
     'Register B-9, killed as a no-alternative issue. WHAT TO WATCH is WHICH LINE SURFACES, '
     'because every history-integrating metric is listing-dependent'),
    ('AEM.TO', 'must-merge',
     'the Toronto line of Agnico Eagle -- the other half of B-9, and the reason TSX is in '
     'the base rule at all'),
    ('0R2J.L', 'group-closure',
     'the LSE line of Agnico Eagle. Present for GROUP CLOSURE, not as a case: carveOut '
     'dedup edges are PAIRWISE over the pool, so observing the AEM/AEM.TO pick with the '
     'third line absent would observe a DIFFERENT group than production sees. All three '
     'carry the identical derived price 168.56 and three different AggScores '
     '(-0.3605 / -0.3320 / -0.3635)'),

    # --- MUST NOT MERGE ------------------------------------------------------ #
    ('HEIA.AS', 'must-not-merge',
     'Heineken N.V. (Amsterdam), the OPERATING company. THE FALSE-POSITIVE GUARD: '
     'Heineken Holding CONSOLIDATES it and reports about HALF the netIncome, so if these '
     'two ever merge, `netIncome` has stopped being required-and-exact in the grouping key '
     '-- which is the failure mode `getData_gen.isin_same_issuer_groups` warns about by '
     'name. AMS is taken whole so the base rule already selects it; pinned anyway, because '
     '"the rule happens to include it" is exactly the guarantee this list exists to replace'),
    ('HEIO.AS', 'must-not-merge',
     'Heineken Holding N.V. -- the consolidating parent, ~half the netIncome. The other '
     'half of the must-NOT-merge pair'),

    # --- the three known WRONG-LINE picks (register K-1, still open) ---------- #
    #  All three are SAME-EXCHANGE, with names identical verbatim and an identical
    #  DERIVED price (price = marketCap / weightedAverageShsOut, both issuer-level, so
    #  they cancel).  Every canonicity marker (a)-(e) is ruled out BY CONSTRUCTION, so
    #  the key falls through to the ALPHABETICAL tail and the NON-COMMON wins:
    #  'CBE' < 'RBT', 'PREVA' < 'VALUE', 'SMSD' < 'SMSN'.  THAT is the behaviour to
    #  observe -- and volAvg (wired 2026-08-06) is what should now change it.
    ('RBT.PA', 'k1-wrongpick',
     'Robertet SA (Paris), the COMMON. Should win its group'),
    ('CBE.PA', 'k1-wrongpick',
     "Robertet's CERTIFICAT D'INVESTISSEMENT (Paris). WINS today on the alphabetical tail "
     "('CBE' < 'RBT'). Observe whether volAvg now demotes it"),
    ('VALUE.AS', 'k1-wrongpick',
     'Value8 N.V. (Amsterdam), the COMMON. Should win its group'),
    ('PREVA.AS', 'k1-wrongpick',
     "Value8's PREFERENCE line (Amsterdam). WINS today ('PREVA' < 'VALUE')"),
    ('SMSN.L', 'k1-wrongpick',
     'Samsung Electronics COMMON GDR (LSE). SMSN.L/BC94.L = 0.989'),
    ('SMSD.L', 'k1-wrongpick',
     'Samsung Electronics PREFERRED GDR (LSE). WINS today on the alphabetical tail. '
     'SMSD.L/SMSN.L = 0.708 against the Seoul preferred/common ratio '
     '005935.KS/005930.KS = 0.728 -- a 29% discount on identical fundamentals, i.e. exactly '
     'what ranks to the TOP of a cheapness screen. NOTE: it is DELIBERATELY EXCLUDED from '
     'TEST_UNIVERSE (see TEST_UNIVERSE_OPEN_GROUPS) because a 142-name reference list must '
     'not contain instrument lines. It is INCLUDED HERE for the opposite reason -- this '
     'universe exists to OBSERVE the wrong pick, not to avoid it'),
    ('BC94.L', 'group-closure',
     'the third Samsung line on LSE -- closes the group SMSD.L/SMSN.L would otherwise be '
     'deduped against differently than in production'),
    ('005930.KS', 'k1-wrongpick',
     'Samsung Electronics COMMON in Seoul -- the 6th character is `0`, so the Korea '
     'canonicity marker should keep it and demote its preferred'),
    ('005935.KS', 'k1-wrongpick',
     'Samsung Electronics PREFERRED in Seoul (6th character `5`). The named case behind the '
     'whole Korea gate: it must be DEMOTED by the korea-preferred marker, not merely '
     'out-sorted. THE UNVERIFIED HALF: whether FMP serves it its ISSUER\'S STATEMENTS -- i.e. '
     'whether the family GROUPS at all -- is unproven until this fetch (ASIA_BLOCKER '
     'residual). If it does not group, it survives as its own singleton and the marker '
     'never gets a sibling to prefer'),

    # --- the LSE IOB lines (register J-1: untradeable at size) --------------- #
    ('0KNY.L', 'lse-iob',
     'IOB line, and one of the SIX names clean on every forensic criterion -- so it is a '
     'veto CONTROL and a liquidity case in one'),
    ('0QQF.L', 'lse-iob',
     'IOB line of Mikron Holding AG, also a clean-on-every-criterion control (the issuer '
     'was unnamed here until 2026-08-06 -- not wrong, but unreadable for an operator)'),
    ('0HQ7.L', 'lse-iob',
     'IOB line of Buckle Inc. Derived price 54.80, IDENTICAL to BKE -- the pair that '
     'demonstrates price cannot discriminate an issuer\'s lines'),

    # --- the 17 names the superseded five-flag veto ejected ------------------ #
    #  THE 17 IS AN UPPER BOUND, and that is WHY they are here.  It comes from the
    #  SUPERSEDED five-flag set; the CURRENT set differs, and observing WHICH of these
    #  the live veto now ejects -- and whether it ejects something not on this list -- is
    #  the point.  UHS and 0LJL.L are also above as the must-merge pair.
    ('SBH', 'veto-eject', 'Sally Beauty Holdings (NYSE)'),
    ('NWPX', 'veto-eject',
     'NWPX Infrastructure (NASDAQ) -- FMP\'s name since the issuer renamed; the why-string '
     'carried the PRE-RENAME name until 2026-08-06'),
    ('BKE', 'veto-eject',
     'Buckle Inc (NYSE) -- also the common half of the 0HQ7.L identical-price pair'),
    ('BVFL', 'veto-eject', 'BV Financial (NASDAQ)'),
    ('DDI', 'veto-eject', 'DoubleDown Interactive (NASDAQ)'),
    ('EMBELL.ST', 'veto-eject',
     'Embellence Group (STOCKHOLM). STO is NOT in the base rule -- this is the case that '
     'proves a must-include reaches outside the base exchanges'),
    ('FG', 'veto-eject', 'F&G Annuities & Life (NYSE)'),
    ('GMR.L', 'veto-eject', 'Gaming Realms (LSE)'),
    ('JEL.L', 'veto-eject', 'Jersey Electricity (LSE)'),
    ('MU', 'veto-eject', 'Micron Technology (NASDAQ)'),
    ('PBYI', 'veto-eject', 'Puma Biotechnology (NASDAQ)'),
    ('PEY.TO', 'veto-eject', 'Peyto Exploration & Development (Toronto)'),
    ('RFX.L', 'veto-eject', 'Ramsdens Holdings (LSE)'),
    ('SKHY', 'veto-eject',
     'SK hynix Inc. (NASDAQ) -- the unsponsored US line of the Seoul semiconductor maker. '
     'THE WHY-STRING NAMED TWO UNRELATED ISSUERS UNTIL 2026-08-06 (tickers that merely look '
     'similar; the wrong names are recorded in the review, not repeated here). FMP returns '
     '"SK hynix Inc." for BOTH this symbol and 000660.KS, which is exactly WHY the '
     '000660.KS sibling fires under group closure (CUR3K_UNPINNED_SIBLINGS) -- a manifest '
     'naming the wrong company would have made that closure look like a defect'),
    ('STRT', 'veto-eject', 'Strattec Security (NASDAQ)'),

    # --- the 6 clean on EVERY criterion: the veto CONTROL group -------------- #
    #  A veto with no observed non-ejections is not an observed veto.  (0KNY.L and
    #  0QQF.L are in this cohort too and are listed above as IOB lines.)
    ('EIN.DE', 'veto-clean', 'Einhell Germany (XETRA) -- outside the base rule'),
    ('NEXN', 'veto-clean', 'Nexxen International (NASDAQ)'),
    ('DRW3.DE', 'veto-clean',
     'Drägerwerk preference line (XETRA) -- outside the base rule'),
    ('KFY', 'veto-clean', 'Korn Ferry (NYSE)'),

    # --- the Chimera NOTES line --------------------------------------------- #
    ('CIM', 'group-closure',
     'Chimera Investment Corp COMMON (NYSE) -- the sibling CIMN needs, and a REIT'),
    ('CIMN', 'k1-wrongpick',
     'a Chimera NOTES line. CAUGHT WINNING ITS GROUP during the dedup build, on marketCap, '
     'while scoring as a SUCCESS against the labeller -- because the labeller IS the rule '
     '(`_non_canonical_tag` recognises no marker on it). It is the named reason the 0.47% '
     'canonicity-first failure rate is a LOWER BOUND and must not be quoted as the failure '
     'rate. Watch whether it still wins'),
)

CURATED3K_MUST_INCLUDE_SYMBOLS = tuple(s for s, _t, _w in CURATED3K_MUST_INCLUDE)


def curated3k_manifest():
    """The must-include list as (symbol, tag, why) triples -- the self-documenting record,
    same shape and same purpose as `test_universe_manifest`."""
    return tuple(CURATED3K_MUST_INCLUDE)


# =========================================================================== #
#  THE REGISTRY.  One entry per universe scope.  Both `configuration` (which     #
#  validates the -tickerfilter argument) and `getData_gen` (which applies it)    #
#  read THIS, so a name that validates always resolves and vice versa.          #
#                                                                               #
#  Keys:                                                                        #
#    label       one-line human description, printed in the run banner           #
#    exchanges   tuple of exchangeShortName codes, or None                       #
#    symbols     explicit ticker tuple (test universe), or None                  #
#    every_exchange  True = apply NO exchange filter (the FULL universe)         #
#    was         the pre-2026-08-02 exchange tuple, when the definition CHANGED;  #
#                None when this name means exactly what it always meant           #
#    note        any caveat the operator must see before trusting the output      #
#                                                                               #
#  EXACTLY ONE of exchanges / symbols / every_exchange is set per entry.         #
#                                                                               #
#  TWO OPTIONAL keys, added 2026-08-06 for stock_CUR3K and read via `.get` so     #
#  every other entry is untouched (a required key would have meant editing all     #
#  ten entries to say "not me", which is noise, not information):                 #
#    sample        {exchangeShortName: n} -- keep only issuer NAMES whose bucket    #
#                  is < n out of CURATED3K_SAMPLE_DENOMINATOR. A code in            #
#                  `exchanges` but absent from `sample` is taken WHOLE.             #
#                  THE RATE IS RESOLVED PER ISSUER, NOT PER ROW: an issuer's whole    #
#                  group is decided by the MOST PERMISSIVE rate among the venues its  #
#                  lines occupy (`most_permissive_rate`), which is what keeps          #
#                  cross-rate groups from splitting. See that function.               #
#    must_include  tuple of symbols unioned in AFTER the base rule, from the FULL    #
#                  post-instrument-filter table -- so a must-include is NOT           #
#                  restricted to `exchanges`.                                        #
# =========================================================================== #
_US = ('NYSE', 'NASDAQ')
_EU_CORE = ('LSE', 'XETRA', 'STO', 'ICE')
_EU_RESTORED = EURONEXT_CLASSIC + ('OSL',)

UNIVERSES = {
    # ---- unchanged by the fix: they never named a dead code -------------- #
    'stock_US1': dict(
        label='US ONLY -- NYSE + NASDAQ (6,141 pre-filter)',
        exchanges=_US, symbols=None, every_exchange=False, was=None,
        note='AMEX (NYSE American, 256 statement-bearing US commons) is NOT included; '
             'see US_NOT_WIRED -- widening "US only" is a CEO decision, not a bug fix.'),
    'stock_NA1': dict(
        label='NORTH AMERICA -- NYSE + NASDAQ + TSX (6,803 pre-filter)',
        exchanges=_US + ('TSX',), symbols=None, every_exchange=False, was=None,
        note='TSX only; TSXV (698), CNQ (291) and NEO (56) are not wired.'),

    # ---- CHANGED by the fix: each named EURONEXT and/or OSE -------------- #
    'stock_NA1_EU1': dict(
        label='NORTH AMERICA + EUROPE, CORRECTED -- NYSE NASDAQ TSX LSE XETRA STO ICE '
              '+ PAR AMS BRU LIS OSL (11,497 pre-filter)',
        exchanges=_US + ('TSX',) + _EU_CORE + _EU_RESTORED,
        symbols=None, every_exchange=False,
        was=_US + ('TSX', 'EURONEXT', 'LSE', 'XETRA', 'STO', 'OSE', 'ICE'),
        note='THE DEFAULT. Restored 1,046 names (PAR 577, OSL 224, BRU 107, AMS 103, '
             'LIS 35) that no run had ever seen. Membership differs from every '
             'pre-2026-08-02 artifact carrying this same name.'),
    'stock_US1_EU1': dict(
        label='US + EUROPE, CORRECTED -- NYSE NASDAQ LSE XETRA STO ICE '
              '+ PAR AMS BRU LIS OSL (10,835 pre-filter)',
        exchanges=_US + _EU_CORE + _EU_RESTORED,
        symbols=None, every_exchange=False,
        was=_US + ('EURONEXT', 'LSE', 'XETRA', 'STO', 'OSE', 'ICE'),
        note='Same 1,046-name restoration as stock_NA1_EU1, without Canada.'),
    'stock_US1_EU2': dict(
        label='US + EURONEXT ONLY -- NYSE NASDAQ + PAR AMS BRU LIS (6,963 pre-filter)',
        exchanges=_US + EURONEXT_CLASSIC,
        symbols=None, every_exchange=False,
        was=_US + ('EURONEXT',),
        note='THE WORST PRE-FIX CASE: EURONEXT was its ONLY non-US code, so this '
             '"US + Europe" universe was silently US-ONLY (identical to stock_US1). '
             'It now actually contains Europe.'),
    'stock_WW1_TV': dict(
        label='TRADINGVIEW-ISH WORLD -- NYSE NASDAQ LSE XETRA + PAR AMS BRU LIS '
              '(9,900 pre-filter)',
        exchanges=_US + ('LSE', 'XETRA') + EURONEXT_CLASSIC,
        symbols=None, every_exchange=False,
        was=_US + ('EURONEXT', 'LSE', 'XETRA'),
        note='Named "WW" but contains no Asia and no Nordics; the name overstates it.'),

    # ---- NEW: ASIA (2026-08-05, unblocked by canonical-choice dedup) ------ #
    'stock_ASIA1': dict(
        label='ASIA, LIKELY-INVESTABLE ONLY -- JPX HKSE KSC KOE ASX SES '
              '(9,098 pre-filter)',
        exchanges=ASIA_LIKELY_INVESTABLE, symbols=None, every_exchange=False, was=None,
        note='Asia ALONE (no US/Europe) -- the universe to fetch when validating Asia, '
             'notably the post-fetch Korea MUST-MERGE regression. EXCLUDES India '
             '(NSE/BSE 5,828), China A-shares (SHH/SHZ 4,362) and Taiwan (TAI/TWO 2,108) '
             'as generally needing local/foreign-investor access -- see '
             'ASIA_ACCESS_EXCLUDED; those three are the BULK of what Asia would add. '
             'KOREA IS GATED: getData_gen.assert_korea_dedup_ready must pass or this '
             'universe does not resolve. Its GROUPING half is UNVERIFIED until a real '
             'Korea fetch is regression-tested (ASIA_BLOCKER residual).'),
    'stock_NA1_EU1_ASIA1': dict(
        label='NORTH AMERICA + EUROPE + LIKELY-INVESTABLE ASIA -- stock_NA1_EU1 '
              '+ JPX HKSE KSC KOE ASX SES (20,595 pre-filter)',
        exchanges=_US + ('TSX',) + _EU_CORE + _EU_RESTORED + ASIA_LIKELY_INVESTABLE,
        symbols=None, every_exchange=False, was=None,
        note='The default universe PLUS likely-investable Asia -- roughly +9,100 '
             'statement-bearing names on top of 11,497, i.e. ~+21h of fundamentals fetch. '
             'Same Korea gate and same UNVERIFIED grouping residual as stock_ASIA1. '
             'Pool composition changes, so NO pooled statistic (z-score, percentile, '
             'top-100, beat-rate) from this universe is comparable to a stock_NA1_EU1 '
             'run -- match artifacts by universe_fingerprint, not by name.'),

    # ---- NEW ------------------------------------------------------------- #
    'stock_TEST1': dict(
        label='CURATED TEST UNIVERSE -- 142 fixed names (140 after the instrument '
              'filter), representative by construction (~713 calls, ~12 min fetch)',
        exchanges=None, symbols=tuple(s for s, _t, _w in TEST_UNIVERSE),
        every_exchange=False, was=None,
        note='POOL-RELATIVE OUTPUT IS MEANINGLESS HERE. z-scores, percentile cuts, '
             'cohort scoring, the top-100 pool and the top-20/top-5 selections are all '
             'functions of pool composition; on 142 names they are not comparable to '
             'production. Use it to check that code paths RUN, never to read a pick.'),
    'stock_CUR3K': dict(
        label='AD-HOC CURATED ~3,250 -- TSX PAR AMS whole + NYSE NASDAQ LSE @17% and '
              'KSC KOE @25% of issuer NAMES resolved PER ISSUER at the most permissive '
              'rate, plus 40 pinned dedup/veto cases and their siblings '
              '(3,258 pre-filter) -- ~16,300 calls, ~3.6-5.2 h fetch',
        exchanges=CURATED3K_TAKE_ALL + tuple(sorted(CURATED3K_SAMPLED)),
        symbols=None, every_exchange=False, was=None,
        sample=dict(CURATED3K_SAMPLED),
        must_include=CURATED3K_MUST_INCLUDE_SYMBOLS,
        note='THE FIRST FETCH AFTER THE 2026-08-05 CHANGES, deliberately small so a '
             'needed tweak costs ~4 h to rediscover instead of ~21 h. The base rule '
             'samples on the NORMALISED ISSUER NAME (not the symbol), so every line of '
             'an issuer is IN or OUT together and cross-listing groups arrive CLOSED -- '
             'a per-symbol sample would split them and test nothing. 39 measured '
             'dedup/veto cases are pinned by symbol so no rate edit can drop them '
             '(asserted in test_universes). KOREA IS GATED: '
             'getData_gen.assert_korea_dedup_ready must pass. POOL-RELATIVE OUTPUT IS '
             'REAL BUT NOT PRODUCTION -- a top-100 cut is the top ~3.3% of this pool '
             'against ~0.9% in production, so read BEHAVIOUR, never a pick list.'),
    'stock_FULL1': dict(
        label='FULL -- every exchange FMP serves with statements (~49,000 names)',
        exchanges=None, symbols=None, every_exchange=True, was=None,
        note='NOT THE DEFAULT, by CEO decision -- intended only once the issue register '
             'is worked through. Applies NO exchange filter, so it includes OTC (9,937) '
             'and PNK (60), which dominate it and are a data-quality minefield, plus '
             'ALL of Asia -- including Korea (now handled by canonical-choice dedup, but '
             'with the grouping half still UNVERIFIED -- see ASIA_BLOCKER) and the '
             'access-excluded India / China A / Taiwan venues that stock_ASIA1 leaves '
             'out on purpose. This IS the CEO\'s "full universe as an option". '
             'FETCH-TIME HAZARD: 49,071 names x 5 statement calls = ~245,000 calls. At '
             'the ~1 s/call the 12h/10,693-name production run implies, that is roughly '
             '60 HOURS -- a single flag turns a 12-hour job into a multi-day one. '
             'Recorded, not guarded: the CEO asked for this universe to exist, not to '
             'be fenced.'),
}

#  The names whose DEFINITION changed on 2026-08-02.  Derived, not hand-listed, so it
#  cannot fall out of step with the registry.
MEMBERSHIP_CHANGED_2026_08_02 = frozenset(
    n for n, d in UNIVERSES.items() if d['was'] is not None)

DEFAULT_UNIVERSE = 'stock_NA1_EU1'


def names():
    """Valid -tickerfilter values.  `configuration` validates against this."""
    return sorted(UNIVERSES)


def _entry(name):
    try:
        return UNIVERSES[name]
    except KeyError:
        raise Exception(
            '-tickerfilter argument not valid: %r. Valid universes: %s'
            % (name, ', '.join(names())))


def exchanges(name):
    """Exchange codes for `name`, or None when the universe is not exchange-defined."""
    return _entry(name)['exchanges']


def symbols(name):
    """Explicit ticker tuple for `name`, or None when membership is by exchange."""
    return _entry(name)['symbols']


def is_every_exchange(name):
    return _entry(name)['every_exchange']


def sample_rates(name):
    """{exchangeShortName: keep_n_out_of_CURATED3K_SAMPLE_DENOMINATOR}, or {} when the
    universe takes every member of its exchanges.

    A code present in `exchanges` but ABSENT from this dict is taken WHOLE -- absence
    means "no sampling", never "sample at zero", so a typo'd code cannot silently empty
    an exchange.  `check_sample_rates` asserts the dict names no code the universe does
    not wire, which is the other half of that guard.
    """
    return dict(_entry(name).get('sample') or {})


def must_include(name):
    """Symbols unioned in AFTER the base rule, or () when there are none.

    NOT restricted to the universe's `exchanges`: applied to the full
    post-instrument-filter table by symbol, which is how a Stockholm or XETRA case is
    observable without wiring (and paying for) those venues.
    """
    return tuple(_entry(name).get('must_include') or ())


def check_sample_rates(name):
    """Return the sampled codes that the universe does not actually wire.

    Empty list = the sample dict and the exchange tuple agree.  A non-empty one is the
    EURONEXT/OSE defect in a new costume: a rate keyed on a code no filter selects is
    dead configuration that reads as though it were doing something.
    """
    rates = sample_rates(name)
    if not rates:
        return []
    codes = set(exchanges(name) or ())
    return sorted(c for c in rates if c not in codes)


#  Namespaced so the bucket for a given issuer name can never collide with some future
#  hash over a different quantity in this module.
_SAMPLE_HASH_PREFIX = 'issuer:'


def issuer_sample_bucket(norm_name):
    """Stable bucket in [0, CURATED3K_SAMPLE_DENOMINATOR) for a NORMALISED issuer name.

    SHA1, NOT `hash()`.  Python salts `hash()` of a str per process (PYTHONHASHSEED), so
    a `hash()`-based sample would resolve to a DIFFERENT universe on every invocation --
    and silently, because both runs look equally plausible.  This is a pure function of
    the name: stable across runs, machines and any re-ordering of
    `available-traded/list`.

    The caller supplies the ALREADY-NORMALISED name (`carveOut._norm_issuer_name`) so
    this module does not have to import carveOut, and so the sample and the pipeline's
    own issuer grouping cannot drift onto two different normalisations.
    """
    h = hashlib.sha1((_SAMPLE_HASH_PREFIX + (norm_name or '')).encode('utf-8'))
    return int(h.hexdigest()[:8], 16) % CURATED3K_SAMPLE_DENOMINATOR


def issuer_in_sample(norm_name, rate):
    """Is this issuer name kept at `rate` out of CURATED3K_SAMPLE_DENOMINATOR?

    `rate` None or >= the denominator means "take everything", so an unsampled code needs
    no special case at the call site.

    ONE RATE ONLY.  This answers the per-rate question; it is NOT the membership decision
    for an issuer whose lines straddle two rates -- see `most_permissive_rate`, which is
    what the caller must resolve FIRST.
    """
    if rate is None or rate >= CURATED3K_SAMPLE_DENOMINATOR:
        return True
    if rate <= 0:
        return False
    return issuer_sample_bucket(norm_name) < int(rate)


def most_permissive_rate(codes, rates):
    """The ONE rate an issuer's whole group is decided by: the most permissive rate among
    the exchange codes its lines occupy.  `None` = take the group whole.

    *** THIS IS WHAT MAKES GROUPS ARRIVE CLOSED.  HASHING THE NAME IS NOT ENOUGH. ***
    Fixed 2026-08-06 after review; the original cut got this WRONG and the error was
    measured, not hypothetical.

    THE DEFECT.  The bucket is a function of the NAME, but the THRESHOLD it is compared
    against was a function of the EXCHANGE -- so a two-part decision was being made with
    only its first part held constant.  TSX/PAR/AMS carry no rate (kept unconditionally)
    while NYSE/NASDAQ/LSE keep buckets < 170, so ANY issuer with lines on both sets split
    whenever its bucket was >= 170 -- 83% of the time.  Replayed through the real code path
    over the live 2026-08-04 table: of the multi-line groups that appeared in the pool at
    all, 218 arrived SPLIT against 209 whole -- 51%.  BMO/BMO.TO, BCE/BCE.TO, BN/BN.TO,
    AEM/AEM.TO/0R2J.L all arrive WHOLE in stock_NA1_EU1, so the SAMPLE was introducing the
    divergence it existed to avoid.  The "splits ZERO" measurement that shipped with the
    first cut was real but covered only Korean families, and KSC/KOE SHARE rate 250 -- so
    they cannot split against each other and the cross-rate case was never measured.

    AND IT REACHED THE PINNED CASES, because the pins are by SYMBOL, not by GROUP: five
    pinned groups arrived incomplete, `robertet s a` among them (RBT.PA + CBE.PA present,
    0NZN.L on LSE absent) -- one of the three K-1 wrong-pick groups, observed as a 2-member
    group where production sees 3.

    WHY UPWARD (most permissive) AND NOT DOWNWARD.  Closing downward -- dropping the whole
    group when any of its lines is out -- also closes it, but it thins the universe by
    exactly the cross-listed names this universe exists to test, and it would drop pinned
    groups entirely.  Closing upward costs members (measured: +243, 3,015 -> 3,258, ~+1,215
    calls, ~+20 min of fetch) and buys the property outright.  A line on an unrated
    (take-all) code therefore pulls the WHOLE group in.

    `codes` empty -> None (take it): an issuer with no usable venue cannot be shown to be
    outside the sample, and dropping on missing metadata is how a filter silently shrinks
    a universe (the EURONEXT lesson).
    """
    eff = None
    seen = False
    for c in codes:
        r = rates.get(c)
        if r is None or int(r) >= CURATED3K_SAMPLE_DENOMINATOR:
            return None                        # a take-all venue pulls the group in whole
        r = int(r)
        eff = r if not seen else max(eff, r)
        seen = True
    return eff if seen else None


def label(name):
    return _entry(name)['label']


def note(name):
    return _entry(name)['note']


def expected_count(name):
    """Sum of live-verified per-exchange counts, or the member count for an explicit
    list.  None for the every-exchange universe (no per-code sum applies).

    A SANITY EXPECTATION, not an assertion: a real run legitimately lands under it,
    because the instrument filter, the delisted prune and the sector filter all remove
    members AFTER the exchange filter.

    A SAMPLED universe's expectation is the per-code verified count SCALED BY ITS RATE,
    plus the must-include list.  Returning the unscaled sum would be actively misleading
    -- it would advertise 10,991 for a universe built to be ~3,000, i.e. exactly the
    "expected members" line an operator reads to sanity-check the run length.

    BUT A PER-CODE SCALING CANNOT MODEL PER-ISSUER CLOSURE, and after the 2026-08-06
    group-closure fix that gap is material: a cross-listed issuer that misses the 17%
    threshold is now KEPT because one of its lines sits on a take-all venue, and no
    per-code factor can express "kept for a reason that lives on a different code".  The
    formula understated the closed universe by 135 (3,123 vs a MEASURED 3,258 -- ~25 min of
    fetch), so where a REPLAYED measurement exists it is used instead of the formula.  A
    measurement beats a model here; the formula stays as the fallback for any future
    sampled universe that has not been replayed.

    KNOWN SILENT-STALENESS PATH, NAMED RATHER THAN CLOSED (reviewer, 2026-08-06).  The
    `stock_CUR3K` branch is special-cased BY NAME, so it is exact today but keeps returning
    the STORED measurement if the universe DEFINITION moves -- with no symptom, because the
    number still looks plausible.  A `definition_fingerprint` (hash the exchange set + sample
    rates + must-include list, and refuse the stored figure when it does not match) is the
    real close and is NOT DONE -- deliberately deferred rather than rushed in ahead of the
    fetch.  THE CHEAPER PROTECTION IS IN PLACE INSTEAD: `test_universes` now derives the
    wallclock band from `CURATED3K_API_CALLS` at +-0.05 h (was +-0.5 h, wide enough to absorb
    the 3,045 -> 3,258 drift) and asserts `CURATED3K_COHORT_ESTIMATE` sums to the coded member
    count, so a definition change that moves the member count trips a test.  It does NOT catch
    a definition change that leaves the count unmoved; that residual is the fingerprint's.
    """
    d = _entry(name)
    if d['symbols'] is not None:
        return len(d['symbols'])
    if d['every_exchange']:
        return None
    if name == 'stock_CUR3K':
        return CURATED3K_ESTIMATED_MEMBERS
    rates = sample_rates(name)
    total = 0.0
    for c in d['exchanges']:
        v = _VERIFIED_COUNTS.get(c, 0)
        r = rates.get(c)
        if r is None or r >= CURATED3K_SAMPLE_DENOMINATOR:
            total += v
        else:
            total += v * (float(r) / CURATED3K_SAMPLE_DENOMINATOR)
    #  Must-includes are a UNION, so a member the base rule already selected is not
    #  counted twice -- but which ones those are is only knowable against the live table,
    #  so this over-counts by the overlap.  Over-counting is the safe direction for a
    #  sanity expectation and the count is 39, i.e. ~1%.
    total += len(must_include(name))
    return int(round(total))


# --------------------------------------------------------------------------- #
#  RESOLVED-COUNT FLOOR, PER EXCHANGE CODE (2026-08-02).                        #
#                                                                               #
#  Only ZERO used to be special-cased, which is precisely why `EURONEXT` and      #
#  `OSE` went unnoticed for the life of the project: the UNIVERSE still resolved  #
#  to thousands of names, so nothing looked wrong.  A universe-LEVEL ratio cannot  #
#  fix that -- losing OSL costs 224 of 11,497 names, i.e. 1.9%, while the NORMAL   #
#  shortfall from the instrument filter and the delisted prune is 7-12%.  The     #
#  signal is drowned before it exists.                                            #
#                                                                               #
#  PER-CODE is the granularity that works, because a dead code loses 100% of      #
#  ITSELF.  Measured on the live 2026-08-02 table, the worst natural per-code      #
#  shortfall is 13.5% (NYSE -- preferred lines concentrate there; NASDAQ 11.4%,    #
#  TSX 6.9%, and every European code 0-1%).  A dead code sits at 100%.  A 40% cut  #
#  therefore separates them with ~26 points of margin below and 60 above.          #
# --------------------------------------------------------------------------- #
RESOLVED_SHORTFALL_WARN_ABOVE = 0.40
RESOLVED_WORST_NATURAL_SHORTFALL = 0.135      # NYSE, measured 2026-08-02


def check_resolved_counts(name, resolved_by_code):
    """Compare a run's per-exchange resolved counts against the verified counts.

    `resolved_by_code` : {exchangeShortName: n} from the resolved universe.
    Returns a list of (code, verified, resolved, shortfall_fraction) for codes that
    came back more than RESOLVED_SHORTFALL_WARN_ABOVE below their verified count --
    empty list means every code delivered roughly what it should.

    Returns [] for a universe that is not exchange-defined (an explicit ticker list has
    its own absent-member report; the FULL universe has no per-code expectation).
    """
    codes = exchanges(name)
    if not codes:
        return []
    out = []
    for c in codes:
        v = _VERIFIED_COUNTS.get(c, 0)
        if v <= 0:
            continue
        r = int(resolved_by_code.get(c, 0))
        shortfall = 1.0 - (r / float(v))
        if shortfall > RESOLVED_SHORTFALL_WARN_ABOVE:
            out.append((c, v, r, shortfall))
    return out


def definition_fingerprint(name):
    """Short stable hash of what `name` MEANS, so an artifact can be matched to the
    universe definition that produced it.

    This is the guard against the failure the CONTINUITY note describes: a filter NAME
    is not sufficient provenance once a name's definition can change.  The fingerprint
    is over the SORTED membership basis (exchange codes, or the explicit ticker list),
    so it is insensitive to the order they happen to be written in but changes the
    moment a code or a ticker is added or removed.

    A SAMPLE RATE AND A MUST-INCLUDE LIST ARE PART OF WHAT THE NAME MEANS, so both are in
    the basis: changing a rate from 17% to 20% changes the membership just as surely as
    adding an exchange code, and an artifact must not claim the same provenance across
    that change.  Appended AFTER the exchange basis and only when non-empty, so every
    pre-2026-08-06 universe fingerprints to exactly the value it did before.
    """
    d = _entry(name)
    if d['symbols'] is not None:
        basis = 'symbols:' + ','.join(sorted(d['symbols']))
    elif d['every_exchange']:
        basis = 'every-exchange'
    else:
        basis = 'exchanges:' + ','.join(sorted(d['exchanges']))
    rates = sample_rates(name)
    if rates:
        basis += '|sample/%d:' % CURATED3K_SAMPLE_DENOMINATOR + ','.join(
            '%s=%d' % (c, rates[c]) for c in sorted(rates))
    mi = must_include(name)
    if mi:
        basis += '|must:' + ','.join(sorted(mi))
    return hashlib.sha1(basis.encode('utf-8')).hexdigest()[:12]


def provenance(name):
    """The universe-definition stamp to embed in every saved artifact.

    Plain JSON-able types only -- it is written into the run's pickle and read back by
    tooling that must not need this module to interpret it.
    """
    d = _entry(name)
    return {
        'universe': name,
        'universe_label': d['label'],
        'universe_fingerprint': definition_fingerprint(name),
        'universe_exchanges': list(d['exchanges']) if d['exchanges'] else None,
        'universe_symbols': list(d['symbols']) if d['symbols'] else None,
        'universe_every_exchange': bool(d['every_exchange']),
        'universe_expected_count': expected_count(name),
        'universe_sample': dict(sample_rates(name)) or None,
        'universe_sample_denominator': (
            CURATED3K_SAMPLE_DENOMINATOR if sample_rates(name) else None),
        'universe_must_include': list(must_include(name)) or None,
        'universe_definition_changed': name in MEMBERSHIP_CHANGED_2026_08_02,
        'universe_previous_exchanges': list(d['was']) if d['was'] else None,
        'universe_codes_verified': '2026-08-02',
        'universe_note': d['note'],
    }


def resume_filename(name, datasource='fmp'):
    """The `lastIndexOfRead_<ds>_<name>.txt` path for a universe.

    Single source for the resume filename so `utils.get_lastIndexRead`'s whitelist is
    DERIVED rather than hardcoded -- which is what previously left stock_US1_EU1 and
    stock_US1_EU2 unresumable.
    """
    _entry(name)
    return 'lastIndexOfRead_%s_%s.txt' % (datasource, name)


def resume_filenames(datasource='fmp'):
    """Every legal resume filename.  Because every historical universe NAME is still in
    the registry, this contains every historical filename verbatim -- no
    `lastIndexOfRead_fmp_stock_*.txt` already on disk is orphaned by the refactor."""
    return sorted(resume_filename(n, datasource) for n in UNIVERSES)


def run_banner(name, resolved_count=None):
    """The LOUD launch-time universe banner, printed BEFORE the long fetch.

    Says what universe is running, what it means, its fingerprint, and -- when the
    definition changed on 2026-08-02, or when the universe is the TEST or FULL one --
    why the output must not be compared to older artifacts or read as a production pick.
    Returns the string (the caller prints it) so it is testable without capturing
    stdout.
    """
    d = _entry(name)
    bar = '=' * 78
    bang = '!' * 78
    out = [bar, '  UNIVERSE: %s' % name, '  %s' % d['label'],
           '  definition fingerprint : %s   (exchange codes verified live 2026-08-02)'
           % definition_fingerprint(name)]
    exp = expected_count(name)
    if exp is not None:
        out.append('  expected members      : ~%d before the instrument/delisted/sector '
                   'filters' % exp)
    if resolved_count is not None:
        out.append('  RESOLVED members      : %d' % resolved_count)
    if d['exchanges']:
        out.append('  exchanges             : %s' % ' '.join(d['exchanges']))
    if d['symbols'] is not None:
        out.append('  explicit ticker list  : %d names (frozen in universes.TEST_UNIVERSE)'
                   % len(d['symbols']))
    rates = sample_rates(name)
    if rates:
        whole = [c for c in (d['exchanges'] or ()) if c not in rates]
        out.append('  SAMPLED on issuer NAME: %s   (out of %d buckets)'
                   % (' '.join('%s=%d' % (c, rates[c]) for c in sorted(rates)),
                      CURATED3K_SAMPLE_DENOMINATOR))
        out.append('  taken WHOLE           : %s' % (' '.join(whole) or '(none)'))
    mi = must_include(name)
    if mi:
        out.append('  must-include (pinned) : %d symbol(s), unioned in AFTER the base '
                   'rule' % len(mi))
    out.append('  note                  : %s' % d['note'])

    if name in MEMBERSHIP_CHANGED_2026_08_02:
        gained = sorted(set(d['exchanges']) - set(d['was']))
        lost = sorted(set(d['was']) - set(d['exchanges']))
        out += [
            bang,
            '!!! THIS UNIVERSE NAME CHANGED MEANING ON 2026-08-02.',
            '!!!   was : %s' % ' '.join(d['was']),
            '!!!   now : %s' % ' '.join(d['exchanges']),
            '!!!   dead codes removed : %s   (they matched ZERO rows)' % ' '.join(lost),
            '!!!   codes added        : %s' % ' '.join(gained),
            '!!! Artifacts written BEFORE this date carry the same NAME but a DIFFERENT',
            '!!! membership. Do NOT compare this run to them on membership, and do NOT',
            '!!! compare any POOLED statistic (z-score, percentile, top-100, beat-rate)',
            '!!! across the boundary -- the pool changed, so those numbers moved for that',
            '!!! reason alone. Match artifacts by universe_fingerprint, not by name.',
            bang,
        ]

    if d['symbols'] is not None:
        out += [
            bang,
            '!!! TEST UNIVERSE -- SCORES FROM THIS RUN ARE NOT PRODUCTION SCORES.',
            '!!! Every pool-relative computation (cross-sectional z-scores, percentile',
            '!!! cuts, cohort-relative scoring, the top-100 pool, the top-20 and the',
            '!!! per-band top-5s) is a function of POOL COMPOSITION. With %d names the'
            % len(d['symbols']),
            '!!! pool is ~1/76th of production (140 vs 10,693 resolved), so a',
            '!!! "top 20" here is the top ~14% of the universe rather than the top ~0.19%.',
            '!!! Ranks and scores from this run are',
            '!!! NOT comparable to a production run and must never be quoted as a result.',
            '!!! It answers "does the code path run correctly", never "what is cheap".',
            bang,
        ]

    if rates:
        out += [
            bang,
            '!!! SAMPLED UNIVERSE -- A SHAKEDOWN SCOPE, NOT A PRODUCTION ONE.',
            '!!! Membership is a deterministic sample of ISSUER NAMES plus %d pinned'
            % len(mi),
            '!!! cases, so the POOL is ~%s members against ~10,693 in production.'
            % (('%d' % exp) if exp is not None else '?'),
            '!!! Every pool-relative number (z-scores, percentile cuts, cohort scoring,',
            '!!! the top-100 pool, the top-20 and the per-band top-5s) is therefore REAL',
            '!!! but NOT COMPARABLE to a production run: a top-100 cut here is roughly the',
            '!!! top 3% of the pool against ~0.9% in production. Read this run for',
            '!!! BEHAVIOUR -- what the dedup merges, which line survives, whom the veto',
            '!!! ejects -- never as a pick list. Match artifacts by universe_fingerprint.',
            bang,
        ]

    if d['every_exchange']:
        out += [
            bang,
            '!!! FULL UNIVERSE -- no exchange filter at all. Includes OTC (9,937) and',
            '!!! PNK (60), which dominate it, and ALL of Asia including Korea and the',
            '!!! access-excluded India / China A / Taiwan venues. Korean preferreds are',
            '!!! now demoted by carveOut canonical-choice dedup, but whether the family',
            '!!! GROUPS at all is UNVERIFIED on real Korean statements (see',
            '!!! universes.ASIA_BLOCKER); if it does not, they rank at the top of the',
            '!!! cheapness screen exactly as before. ~5x the current fetch cost.',
            bang,
        ]

    out.append(bar)
    return '\n'.join(out)


def test_universe_manifest():
    """The TEST universe as (symbol, tag, rationale) triples -- the self-documenting
    record.  Nothing in the pipeline needs it; the checker and the report read it."""
    return tuple(TEST_UNIVERSE)


#  MEMBERS LISTED vs MEMBERS FETCHED.  The list is 142 names; `filter_non_common_
#  instruments` then removes MS-PE and GIPRW by design (they are in the list precisely so
#  that filter is demonstrably firing), leaving 140 names actually fetched.  Both numbers
#  are stated because a run reporting "140 of 142" is CORRECT and must not be mistaken
#  for shrinkage.  Verified 2026-08-02 against the live list.
#
#  The first draft of the fill also picked up BW-PA, CDR-PB and CDR-PC -- real preferred
#  lines that the saved 2026-01-08 panel still contains because that panel predates the
#  instrument filter.  They were removed at fetch time and silently thinned the very
#  cohorts they were selected to populate, so the fill's candidate pool now excludes
#  anything the instrument filter would delete.
TEST_UNIVERSE_LISTED = len(TEST_UNIVERSE)
TEST_UNIVERSE_EFFECTIVE = 140
TEST_UNIVERSE_FILTERED_BY_DESIGN = ('MS-PE', 'GIPRW')

# --------------------------------------------------------------------------- #
#  ISSUER-GROUP CLOSURE -- AND EVERY GROUP DELIBERATELY LEFT OPEN.               #
#  (2026-08-03; the declared set was WRONG on the first pass -- see below.)       #
#                                                                               #
#  carveOut's dedup edges are PAIRWISE OVER THE POOL, so a member whose            #
#  same-issuer sibling is missing gets deduped differently here than in            #
#  production.  15 legitimate siblings are in the list to close such groups.       #
#                                                                               #
#  A group is left OPEN when closing it would require re-importing a NON-COMMON    #
#  line -- a preferred, a baby bond, a notes line.  Those are exactly what the      #
#  reference list must not contain (they carry the common's fundamentals and        #
#  trade at a discount, so they rank high on a cheapness screen), and the           #
#  share-class filter does not catch them: see                                     #
#  getData_gen.SHARE_CLASS_FILTER_KNOWN_GAPS.                                      #
#                                                                               #
#  WHY THIS BLOCK IS AUTHORITATIVE RATHER THAN ASPIRATIONAL.  The first version    #
#  declared ONE open group (WHLR) and was wrong: replaying                          #
#  `carveOut._issuer_components` -- the function the pipeline ACTUALLY dedups        #
#  with -- over the post-instrument-filter production pool showed BW open too, and   #
#  removing the two instrument lines that had been mis-added as "partners" (HTFC,    #
#  SMSD.L) opened HRZN, BC94.L and SMSN.L as well.  A declared-open list that is     #
#  SHORTER than reality hides divergence; one that is LONGER is equally misleading.  #
#  So `verify_test_universe` re-derives the divergent set and reconciles it against  #
#  this dict IN BOTH DIRECTIONS, and a test asserts the reconciliation.  The         #
#  previous guard for this was a loop with two `continue`s and NO assertion --        #
#  decorative, which is worse than no guard at all, because the suite stayed green.  #
#                                                                               #
#  Each value is (*sibling_symbols, reason).  The siblings are present in            #
#  PRODUCTION and deliberately absent HERE.                                          #
# --------------------------------------------------------------------------- #
TEST_UNIVERSE_OPEN_GROUPS = {
    'WHLR': ('WHLRD', 'WHLRL',
             'Wheeler REIT. Both siblings are non-common lines (Series D preferred / '
             'notes) with byte-identical fundamentals to WHLR, and the share-class filter '
             'catches neither (tails "D" and "L" are outside rule C\'s whitelist). '
             'Production dedups this issuer over 3 lines, the list over 1. Accepted: '
             'importing them would corrupt the cohort counts, and dropping WHLR would cost '
             'a real small-cap REIT.'),
    'BW': ('BWNB',
           'Babcock & Wilcox. BWNB is a SENIOR NOTES line with byte-identical '
           'fundamentals to BW. It is invisible to rule C for a different reason from '
           'Wheeler: FMP TRUNCATES its name to "Babcock & Wilcox Enterprises, I", so it '
           'never joins BW\'s (name, exchange) group at all -- which is also why the '
           'first, name-based closure audit missed this divergence entirely.'),
    'HRZN': ('HTFB', 'HTFC',
             'Horizon Technology Finance. BOTH siblings are BABY BONDS with '
             'byte-identical fundamentals to HRZN. HTFC is quoted 24.85 -- near the '
             '$25 par of a US retail note -- against HRZN at 4.32, and sits on NYSE '
             'while the common is on NASDAQ; it was briefly in this list as a '
             '"dedup-partner" before being caught. HTFB is declared too because it '
             'WAS in the 2026-01-08 production pool, which is what the offline '
             'reconciliation replays against, but has since LEFT FMP\'s tradable list '
             '-- a matured note -- so live production today sees only HTFC. Declared '
             'as of the panel era so the reconciliation is EXACT; expect HTFB to drop '
             'out when the panel is refreshed.'),
    'BC94.L': ('SMSD.L',
               'Samsung Electronics. SMSD.L is the PREFERRED GDR, not a second common '
               'line: SMSD.L/SMSN.L = 0.708 against the Seoul preferred/common ratio '
               '005935.KS/005930.KS = 0.728. This is the KOREAN PREFERRED TRAP the Asia '
               'note warns about, and it was imported into this list by hand before being '
               'caught. Excluded deliberately; the divergence is the price of keeping it '
               'out.'),
    'SMSN.L': ('SMSD.L',
               'the other Samsung COMMON GDR in the same open group as BC94.L -- both are '
               'divergent for the one reason, that SMSD.L (preferred) is excluded.'),
}

#  TWO EVIDENCE BASES, kept apart so the reconciliation can be exact instead of
#  approximately right.
#
#  `carveOut._issuer_components` groups by FUNDAMENTALS FINGERPRINT, so it can only see a
#  divergence when BOTH the member and its missing sibling have statements in a saved
#  panel.  For these three, both do, and a replay over the production pool reproduces the
#  divergence exactly -- that is the strong form, and `verify_test_universe` requires the
#  derivation and this dict to agree on it in BOTH directions.
OPEN_GROUPS_FINGERPRINT_DERIVABLE = ('BW', 'HRZN', 'WHLR')
#
#  The Samsung entries CANNOT be fingerprint-derived offline: SMSD.L and SMSN.L have no
#  rows in any saved panel (the 2026-01-08 fetch failed or skipped them), so no
#  fundamentals exist to fingerprint.  Their divergence rests on LIVE-LIST MEMBERSHIP,
#  which is just as checkable and needs no fundamentals: SMSD.L is in the production
#  universe (LSE, statement-bearing, survives the instrument filter) and is not in this
#  list.  Recorded explicitly so that "declared but not derived" is a KNOWN, justified
#  state rather than something a reader has to guess about -- an over-declared list is as
#  misleading as an under-declared one.
OPEN_GROUPS_LIVE_LIST_ONLY = ('BC94.L', 'SMSN.L')

# --------------------------------------------------------------------------- #
#  WHAT EACH dedup-partner CLAIMS TO CLOSE -- AS DATA, NOT AS PROSE.            #
#                                                                               #
#  The rationale strings say this in English, and the first guard tried to check   #
#  them by regex-scanning for ticker-shaped tokens.  That cannot work: a rationale  #
#  legitimately names symbols it does NOT close (SMSN.L cites SMSD.L as the         #
#  EXCLUDED preferred, and 005930.KS as price evidence), and a regex cannot tell a   #
#  claim from a counter-example -- so the check either misfired or, as shipped,      #
#  asserted nothing at all.                                                        #
#                                                                               #
#  Stated as data instead, so the claim is machine-checkable: partner -> the member  #
#  whose issuer group it closes, or None for a partner that explicitly does NOT      #
#  close one.  A partner mapped to X asserts X's group is CLOSED, so a test can       #
#  require X not to appear in TEST_UNIVERSE_OPEN_GROUPS -- which is exactly the        #
#  contradiction HTFC shipped (it claimed to close HRZN while the group stayed open).  #
# --------------------------------------------------------------------------- #
DEDUP_PARTNER_CLOSES = {
    'AIL.DE': 'AI.PA',
    'SHEL.L': 'SHELL.AS', 'SHEL': 'SHELL.AS',
    'STLA': '8TI.DE', 'STLAP.PA': '8TI.DE',
    'MS': 'DWD.DE', '0QYU.L': 'DWD.DE',
    'WDC.DE': 'WDC',
    'LIN.DE': 'LIN',
    'IHG': 'IHG.L',
    'CGI.L': 'CGI.TO',
    '0NNU.L': 'NEDAP.AS',
    '0LUS.L': 'WELL',
    '0LHX.L': 'GROW',
    #  SMSN.L closes NOTHING: its group-mate SMSD.L is the preferred GDR and is
    #  deliberately excluded, so BC94.L and SMSN.L are both DECLARED open. Recorded as
    #  None rather than omitted, so a reader cannot mistake it for an oversight.
    'SMSN.L': None,
}

#  Fetch cost, so "minutes not hours" is a number rather than a hope.  `failTests`
#  issues 5 statement calls per ticker (key-metrics, ratios, income-statement,
#  balance-sheet-statement, cash-flow-statement) and the universe build costs 3
#  (available-traded/list, financial-statement-symbol-lists, delisted page 0).
#  The ~12h/9,000-name production fetch runs at ~1 s/call, which is the basis for the
#  wall-clock estimate.
TEST_UNIVERSE_API_CALLS = 5 * TEST_UNIVERSE_LISTED + 3          # 648
TEST_UNIVERSE_WALLCLOCK_MIN = (8, 15)                           # minutes, observed rate
