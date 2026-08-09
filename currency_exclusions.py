"""currency_exclusions.py -- reporting currencies whose STATEMENTS we refuse to score.

THE CASE THAT BUILT THIS MODULE
-------------------------------
`ARS`.  Argentine reporters restate under **IAS 29 (hyperinflationary economies)**, and on
the one name where it can be checked the vendor's handling of that currency is outright
broken.  Measured on the 2026-08-07 CUR3K panel, over the identical 2020-04 -> 2026-01
window:

  =========  ==================  ==================  ===================================
  source     revenue 1st -> last  max ADJACENT ratio  reading
  =========  ==================  ==================  ===================================
  `BMA`      4.18e10 -> 1.91e12   **1,588x**          vendor defect + inflation
  `CRESY`    3.38e10 -> 2.53e11   **5.0x**            no defect visible
  =========  ==================  ==================  ===================================

  * `BMA` (Banco Macro), 2025-10-01 row: `totalAssets` = 2.325e16 sitting between 2.056e13
    and 2.420e13 -- a **961x break** -- and `revenue` = 1.201e9 sitting between 1.443e12
    and 1.907e12 -- a **1,588x break** -- while `marketCap` runs 4.12e12 / 8.68e12 / 7.30e12
    straight through, i.e. CONTINUOUS.  Nothing keyed on market cap can see this.
  * `CRESY` (Cresud) shows **NO defect of any kind**.  Its worst adjacent ratio is 5.0x --
    an ordinary quarter-on-quarter move -- and its nominal rise over the window is only
    **7.5x** against `BMA`'s **45.6x**.  7.5x is *below* Argentine inflation for the
    period, which is what a **correctly IAS 29-restated** series looks like.

**READ THAT SECOND ROW BEFORE EDITING THIS FILE.**  An earlier version of this docstring
claimed the inflation-reads-as-growth effect was "visible in BOTH" and cited `BMA`'s 45x as
carrying `CRESY`.  That was **wrong** -- the 45x is `BMA`'s alone, and `CRESY`'s 7.5x is
evidence *against* the same reading.  `CRESY` is removed by the CURRENCY rule and by
nothing else.  Do not re-attach `BMA`'s evidence to it.

WHAT ACTUALLY JUSTIFIES REMOVING A CLEAN-LOOKING ARS NAME
---------------------------------------------------------
Three things, none of which is "its numbers look wrong":

  1. **This vendor's ARS handling is demonstrably defective in this very panel** (`BMA`, by
     three orders of magnitude, invisible to every market-cap check).  A currency whose
     feed is known-broken on one of its two names is not a currency we can spot-clear the
     other name on.
  2. **No ARS name can be valued in USD at all.**  `ARS` is refused by
     `fx_rates.ABSTAIN_CURRENCIES` and absent from `carveOut.FX_TO_USD`, so an ARS reporter
     has no `marketCap_usd`, sits in no market-cap band, and scores NEUTRAL on the size
     tilt while still competing for the same shortlist slots.
  3. **Clearing one would take a per-name restatement audit this codebase does not do.**
     Telling a properly-restated IAS 29 series from a mishandled one means checking the
     restatement vintage of every row against the filing's own price index.  Nothing here
     does that, and a screen cannot: the rejected scale-break detector fires on 119 of
     2,613 sources and is dominated by de-SPACs and IPOs.

So the honest statement is: **`CRESY` is removed because it reports in ARS, full stop.**
That is the CEO's ruling and it is the right one (see below); it is *not* a finding about
`CRESY`.

WHY EXCLUSION AND NOT ABSTENTION (CEO ruling, 2026-08-08)
---------------------------------------------------------
The previous treatment was `fx_rates.ABSTAIN_CURRENCIES`: refuse to supply an ARS->USD
rate, keep the name, score it NEUTRAL on the size tilt.  That fixes the **currency** only.
It is insufficient, because the currency was never the whole problem: a name whose
`totalAssets` is wrong by three orders of magnitude in places, and whose entire history is
denominated in a unit that moved 45x inside the window, has EVERY metric downstream of
those statements contaminated -- Altman-Z, Piotroski, the yields, the growth terms, the
Stage-1 pass rates and the cross-sectional medians they are compared against.  Abstaining
addresses one band of one metric while the name keeps voting on all the others, including
on the medians every other company is scored against.

So the name leaves the universe.  `ABSTAIN_CURRENCIES` is KEPT as a second line -- see
"THE TWO RULES ARE NOT REDUNDANT" below.

WHY CURRENCY-SCOPED AND NOT NAME-SCOPED (CEO ruling, 2026-08-08; agreed here)
-----------------------------------------------------------------------------
The rule keys on `reportedCurrency == 'ARS'`, not on the tickers `BMA` / `CRESY`.

  * The defect is the **reporting regime and this vendor's handling of it**, not the two
    companies.  The next ARS reporter to enter the universe inherits all three conditions
    above in full, and a name list would admit it silently.  A name list also cannot
    express the real reason, so the next reader deletes it as unexplained.
  * It is honestly BLUNTER than the per-name evidence, and `CRESY` is the proof: it looks
    clean and is removed anyway.  That is a deliberate, stated over-reach.  The cost is two
    names out of 2,613 (0.08%), neither of which reaches the shipped top-100 on the
    2026-08-07 run; the alternative is spot-clearing names on a feed we have caught being
    wrong by 1000x on the same currency.  If an ARS reporter with a defensible, audited
    statement basis ever matters, the right response is a NAMED exception here carrying
    that audit, not deleting the rule.
  * It is keyed on a **vendor field we do not control**.  If FMP mislabels
    `reportedCurrency`, the rule misses; if the column is absent (every panel fetched
    before `reportedCurrency` was folded into ingest), the rule CANNOT fire at all and says
    so out loud rather than reporting a clean zero.  See `applicable`.

THE TWO RULES ARE NOT REDUNDANT -- AND HERE ARE THE EXACT PATHS
----------------------------------------------------------------
`fx_rates.ABSTAIN_CURRENCIES['ARS']` stays.  This exclusion runs inside
`data_quality.filter_invalid_data`; anything that converts a market cap WITHOUT going
through that function is covered by the abstention and by nothing else.  Enumerated rather
than asserted (reviewer, 2026-08-08):

  * **Inside the main pipeline, BEFORE this filter ever runs.** `getData_fmp` materialises
    `cdx_df['marketCap_usd'] = carveOut.marketcap_usd_series(cdx_df)` on the RAW freshly
    fetched panel, and `Sbocker` calls `apply_data_quality_filter` only afterwards.  So on
    every full fetch the ARS names are converted before they are excluded.  With the
    abstention in place that conversion yields NaN, which is correct; delete it and the
    saved panel ships a confident USD market cap for a name whose assets are wrong by 1000x.
  * **Four `baseline_tools` entry points reach `carveOut` and never import `data_quality`
    at all**: `depth_horizon_grid.py`, `tune_run.py`, `pipeline_analysis.py`,
    `stage2_pit.py`.  On those paths this module is not in the call graph.

Exclusion is the primary rule; abstention is the backstop for every path that bypasses it.
Removing either one alone leaves a live hole.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
------------------------------------------
It does NOT detect hyperinflation, scale breaks, or restatement bases.  There is no screen
here and there should not be one: the scale-break detector was already evaluated and
REJECTED for this codebase (see `vendor_contamination`'s closing note -- it fires on 119 of
2,613 sources and is dominated by de-SPACs and IPOs).  This is a NAMED, DATED, EVIDENCED
list a human maintains, in the same shape as `vendor_contamination.QUARANTINE_RULES`, and
for the same reason: the judgement is not mechanisable, so it is written down instead.

WHERE THE EVIDENCE GOES
-----------------------
Two artifacts, both under `output/`, which ships whole:
  * `removed_data_quality_*.csv` -- one row per removed row, carrying the rule's evidence
    AND this source's own ARS-row fraction (`ARS on 24/24 rows`), so a genuine reporter and
    a single stray vendor label can never read the same.
  * `CurrencyExclusionStatus_<date>.csv` -- written on EVERY invocation, APPEND-mode,
    including the invocations where the rule **could not run**.  "Applied" used to be
    evidenced and "could not apply" used to be a console line only; three `baseline_tools`
    callers pass `verbose=False`, so on those paths the miss had no witness at all.

MEASURED THROUGH THIS CODE
--------------------------
  * 2026-08-07 CUR3K panel (61,007 rows, 2,613 sources): **2 sources / 48 rows removed**
    (`BMA` 24/24 ARS, `CRESY` 24/24 ARS).  Neither appears in `AggScoreTop100-2026-08-07`,
    so the effect on that run's shipped output is ZERO.
  * 2026-01-08 NA1_EU1 panel (9,012 sources): the panel has **no `reportedCurrency`
    column** (it predates the ingest change), so the rule cannot fire.  `BMA` and `CRESY`
    ARE in that panel and are NOT removed from it.  This is announced AND recorded.
"""

import datetime as _dt
import os

import numpy as np
import pandas as pd


#  The column the rule keys on.  It is the STATEMENT currency (the one `marketCap` is
#  denominated in), not the listing's trading currency -- see the block comment above
#  `carveOut.FX_TO_USD` for why confusing the two is the standing trap in this codebase.
CURRENCY_COL = 'reportedCurrency'


class CurrencyExclusionRule(object):
    """One reporting currency whose statements are refused, with its evidence attached."""

    __slots__ = ('currency', 'reason', 'evidence', 'added')

    def __init__(self, currency, reason, evidence, added):
        self.currency = currency
        self.reason = reason
        self.evidence = evidence
        self.added = added

    def label(self):
        """The `removal_reason` string that travels into
        `output/removed_data_quality_*.csv`.  It carries the EVIDENCE, not just the verdict:
        that CSV is the shipped record of why a name left the universe, and a reason a
        reader has to go looking for is the defect this project spent the week removing."""
        return 'currency_excluded [%s: %s | %s]' % (
            self.currency, self.reason, self.evidence)


EXCLUDED_CURRENCIES = {
    'ARS': CurrencyExclusionRule(
        currency='ARS',
        #  THE REASON IS ABOUT THE CURRENCY, NOT ABOUT WHICHEVER NAME IS BEING REMOVED.
        #  An earlier draft said "plus a vendor-side scale defect", which is TRUE of BMA
        #  and FALSE of CRESY -- and it was stamped onto CRESY's rows.  A removal record
        #  must not assert of a name something that was measured on a different name.
        reason='reports in ARS -- IAS 29 hyperinflation regime, no admitted ARS->USD rate, '
               'and this vendor is demonstrably wrong on ARS in this panel',
        evidence=(
            "CURRENCY-SCOPED RULE (CEO 2026-08-08): every ARS reporter is removed, whether "
            "or not its own series looks damaged. Grounds, measured on the 2026-08-07 "
            "CUR3K panel: (1) VENDOR DEFECT, on BMA -- totalAssets 2.325e16 between "
            "2.056e13 and 2.420e13 (961x break) and revenue 1.201e9 between 1.443e12 and "
            "1.907e12 (1,588x break), with marketCap continuous through both, so no "
            "market-cap sanity check can see it; BMA's nominal revenue also rises 45.6x "
            "over the window. (2) NO USD VALUATION IS POSSIBLE -- ARS is refused by "
            "fx_rates.ABSTAIN_CURRENCIES and absent from carveOut.FX_TO_USD, so an ARS "
            "name has no marketCap_usd, sits in no band, and scores NEUTRAL on the size "
            "tilt while competing for the same slots. (3) NO PER-NAME CLEARANCE EXISTS -- "
            "separating a correctly IAS 29-restated series from a mishandled one needs a "
            "restatement-vintage audit this codebase does not perform. "
            "EXPLICITLY NOT CLAIMED OF EVERY ARS NAME: CRESY shows NO defect (worst "
            "adjacent ratio 5.0x vs BMA's 1,588x; 7.5x nominal rise vs BMA's 45.6x, i.e. "
            "BELOW Argentine inflation, which is what correct restatement looks like). "
            "CRESY is removed by this rule alone. Panel effect: 2 sources, 48 rows."),
        added='2026-08-08'),
}

#  WHY ARS IS THE ONLY ENTRY, given the rule is stated as a REGIME principle.
#  Because no other currency in the panel meets the three grounds above -- checked, not
#  assumed.  The 2026-08-07 CUR3K panel carries 23 reporting currencies (ARS AUD BRL CAD
#  CHF CNY DKK EUR GBP ILS INR JPY KRW MAD NOK PEN PHP PLN SEK SGD TWD USD ZAR).  The only
#  other currency that would plausibly raise the IAS 29 question is **TRY**, and there is
#  **no TRY reporter in this panel at all** -- so the question is not live, and adding TRY
#  now would be a rule with no referent and no evidence behind it.
#  IF A TRY (or VES, or LBP) REPORTER EVER APPEARS, this is the decision to make
#  deliberately, with the three grounds re-checked against it -- note that ground (2) does
#  NOT hold for TRY (it IS in carveOut.FX_TO_USD, at 0.030), so TRY would not simply
#  inherit ARS's case.  Recorded here so the asymmetry is a decision, not an oversight.


def applicable(df):
    """(ok, note) -- can this rule even run on `df`?

    A panel WITHOUT `reportedCurrency` (anything fetched before the field was folded into
    ingest) cannot be filtered by reporting currency, and reporting "0 excluded" for it
    would be a false negative of exactly the kind this module exists to prevent.  The
    caller states the note; it never guesses."""
    cols = getattr(df, 'columns', [])
    if df is None or not len(df):
        return False, 'frame is empty'
    if 'source' not in cols:
        return False, "frame has no 'source' column"
    if CURRENCY_COL not in cols:
        return False, ("panel carries no %r column (it predates the currency ingest), so "
                       "reporting-currency exclusions CANNOT be applied to it"
                       % CURRENCY_COL)
    return True, ''


def _normalised_currency(df):
    """`reportedCurrency` as an upper-cased, stripped string Series.  Vendor strings arrive
    with stray case and whitespace; a rule that misses because of a trailing space is a
    rule that silently does not exist."""
    return (df[CURRENCY_COL].astype(str).str.strip().str.upper()
            .replace({'NAN': '', 'NONE': ''}))


def excluded_sources(df, rules=None):
    """{source: (rule, n_rows_in_currency, n_rows_total)} for every source with AT LEAST
    ONE row in an excluded reporting currency.

    SOURCE-SCOPED, not row-scoped, and that is the point.  The defect is the reporting
    basis of the company's history, so removing only the ARS-labelled rows would leave a
    truncated series that still gets scored -- and a name that CHANGED reporting currency
    mid-history is the case where the restatement question is worst, not a case for keeping
    the clean half.

    THE COUNTS ARE RETURNED BECAUSE ONE STRAY LABEL DELETES A WHOLE NAME (reviewer F-4).
    Measured on the 2026-08-07 CUR3K panel: 44 sources carry more than one reporting
    currency, and **5 of them have exactly ONE minority-currency row** (`NUAG.TO`, `QRC.TO`,
    `TFII`, `TFII.TO`, `TRX.TO` -- each 23 USD/CAD rows and 1 of the other; three further
    sources have a single UNPARSEABLE row, which is not a currency and does not count).  A
    single stray `ARS` label on any of them would delete a 24-quarter North American history
    under an IAS 29 reason, and the record would read **identically** to `BMA`'s genuine
    24-of-24.  The behaviour is deliberate and unchanged -- a name whose currency label we
    cannot trust is not a name we can score -- but the RECORD must be able to tell the two
    apart, so the fraction travels into the removal reason.

    Returns {} for a frame the rule cannot run on; the caller checks `applicable`."""
    rules = EXCLUDED_CURRENCIES if rules is None else rules
    ok, _note = applicable(df)
    if not ok or not rules:
        return {}
    cur = _normalised_currency(df)
    hit = cur.isin(set(rules))
    if not hit.any():
        return {}
    totals = df['source'].value_counts()
    out = {}
    for src, ccy in zip(df.loc[hit, 'source'], cur[hit]):
        #  First hit wins; a source cannot be excluded twice, and the currency that
        #  triggered it is the one recorded.
        if src not in out:
            n_ccy = int(((df['source'] == src) & hit).sum())
            out[src] = (rules[ccy], n_ccy, int(totals.get(src, 0)))
    return out


def is_minority_label(n_in_currency, n_total):
    """True when the excluded currency is a MINORITY of the source's rows -- i.e. the
    'one stray vendor label' shape rather than a genuine ARS reporter.  Not a different
    action (the name still leaves); it is what makes the case visible to a human."""
    return bool(n_total) and (2 * int(n_in_currency) < int(n_total))


def exclusion_records(df, rules=None, price_col='price', mcap_col='marketCap'):
    """(row_mask, [removal_record]) in the shape `data_quality` already logs removals in.

    `row_mask` covers EVERY row of an excluded source -- including rows that are not
    themselves ARS-labelled -- because the exclusion is source-scoped.  Each record carries
    the rule's evidence AND this source's own currency fraction, so
    `output/removed_data_quality_*.csv` (which ships whole with `output/`) can never read
    the same for a 1-of-24 stray label as for a genuine 24-of-24 reporter."""
    rules = EXCLUDED_CURRENCIES if rules is None else rules
    hits = excluded_sources(df, rules)
    if not hits:
        return pd.Series(False, index=getattr(df, 'index', None), dtype=bool), []
    mask = df['source'].isin(hits)
    reasons = {}
    for src, (rule, n_ccy, n_tot) in hits.items():
        if rule is None:
            reasons[src] = 'currency_excluded [unnamed rule]'
            continue
        tag = '%s on %d/%d rows%s' % (
            rule.currency, n_ccy, n_tot,
            ' -- MINORITY LABEL, possible stray vendor value' if
            is_minority_label(n_ccy, n_tot) else '')
        reasons[src] = '%s [%s]' % (rule.label(), tag)
    records = []
    for _idx, row in df[mask].iterrows():
        records.append({
            'source': row.get('source'),
            'date': row.get('date', None),
            'price': row.get(price_col, np.nan),
            'marketCap': row.get(mcap_col, np.nan),
            'removal_reason': reasons.get(row.get('source'),
                                          'currency_excluded [unnamed rule]'),
        })
    return mask, records


# --------------------------------------------------------------------------- #
#  The status artifact -- "could not apply" has to ship too                   #
# --------------------------------------------------------------------------- #
def status_rows(df, rules=None, applied=None, note=None):
    """The rows describing WHAT THIS INVOCATION DID, including doing nothing.

    "Applied" was evidenced (the removal CSV) and "could not apply" was console-only --
    the wrong asymmetry, and three `baseline_tools` callers pass `verbose=False`, so on
    those paths the miss had no witness at all.  One row per excluded source when the rule
    fired; exactly one row saying so when it could not."""
    rules = EXCLUDED_CURRENCIES if rules is None else rules
    ok, auto_note = applicable(df)
    ok = ok if applied is None else bool(applied)
    note = auto_note if note is None else note
    watched = ', '.join(sorted(rules))
    if not ok:
        return [{'status': 'NOT_APPLIED', 'currency': '', 'source': '',
                 'n_rows_in_currency': '', 'n_rows_total': '', 'minority_label': '',
                 'watched_currencies': watched, 'note': note}]
    hits = excluded_sources(df, rules)
    if not hits:
        return [{'status': 'APPLIED_NO_MATCH', 'currency': '', 'source': '',
                 'n_rows_in_currency': '', 'n_rows_total': '', 'minority_label': '',
                 'watched_currencies': watched,
                 'note': 'rule ran; no source reports in a watched currency'}]
    return [{'status': 'EXCLUDED', 'currency': rule.currency, 'source': src,
             'n_rows_in_currency': n_ccy, 'n_rows_total': n_tot,
             'minority_label': is_minority_label(n_ccy, n_tot),
             'watched_currencies': watched, 'note': rule.reason}
            for src, (rule, n_ccy, n_tot) in sorted(hits.items())]


def write_status(rows, outdir='output', run_date=None, run_id=''):
    """APPEND `rows` to `output/CurrencyExclusionStatus_<date>.csv`.

    APPEND, not overwrite, and that is load-bearing: `filter_invalid_data` runs TWICE per
    pipeline run and the second pass is correctly idempotent, so an overwriting writer
    would replace pass 1's "EXCLUDED BMA, CRESY" with pass 2's "no match" -- the exact
    accumulate-never-assign defect this repo fixed on 2026-08-07, rebuilt in a new file.

    Written even when the rule matched nothing: "the check ran and found nothing" and "the
    check could not run" are different facts, and the `VendorContaminationFlags_*.csv`
    precedent is to record both.  Never raises."""
    try:
        run_date = run_date or _dt.date.today().strftime('%Y-%m-%d')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        path = os.path.join(outdir, 'CurrencyExclusionStatus_%s.csv' % run_date)
        frame = pd.DataFrame(list(rows))
        frame.insert(0, 'run_id', run_id or 'unknown-unstamped-run')
        frame.insert(1, 'written_at', _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        header = not os.path.exists(path)
        frame.to_csv(path, mode='a', header=header, index=False)
        return path
    except Exception as e:                                  # pragma: no cover - guard
        print('[currency_exclusions] WARNING: status CSV not written (%s: %s)'
              % (type(e).__name__, e), flush=True)
        return None
