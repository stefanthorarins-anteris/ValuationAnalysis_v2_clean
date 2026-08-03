ValuationAnalysis — pipeline README

Overview

This repository collects fundamentals from FinancialModelingPrep (FMP), computes many derived metrics, aggregates them into scores, and ranks stocks for further analysis. The main orchestrator script is `Sbocker.py` which:

- reads configuration (CLI flags) via `configuration.py`,
- retrieves and filters tickers via `getData_gen.py`,
- fetches fundamentals and computes metrics via `getData_fmp.py`,
- computes scores via `calcScore.py`,
- ranks and post-processes results via `postBo.py`, and
- runs manipulation checks via `detectManipulation.py`.

New additions in this branch

- `normalized_analysis.py`: builds a normalized version of `BoMetric_df` using per-date winsorized z-scores (cross-sectional), runs the same scoring and post-processing on the normalized data, and saves the results as `results_normalized_YYYY-MM-DD.pickle`. Also collects a `timing_maybe_off` list (tickers whose latest metric date is in the current calendar year) for auditing lookahead/recency concerns.

- `rolling_backtest.py`: a simple rolling-window backtest harness. It:
  - builds scores from the latest available metrics as-of each rebalance date,
  - selects top-N tickers, constructs a simple portfolio (equal or linear-decay weights),
  - computes portfolio returns between rebalance dates using prices in `cdx_df`, and
  - reports monthly return series for the tested strategies.

- `getData_gen.py` now writes a dated CSV `delisted_tickers_YYYY-MM-DD.csv` with the current delisted ticker list for auditing.

Guidance — how to run

1) Install dependencies (per-user recommended):

```powershell
python -m pip install --user -r requirements.txt
```

2) Run full pipeline (default behavior — fetches all tickers, computes metrics, scores, and saves results):

```powershell
python .\Sbocker.py
```

3) Run with saved metrics (no fetch, only post-processing):

```powershell
python .\Sbocker.py -loadbometric 1 -bometricfilename <your_pickle_filename>
```

4) Run a small test pipeline — use the **curated TEST universe**, not `-nrTaT`:

```powershell
python .\Sbocker.py -tickerfilter stock_TEST1
```

142 fixed, checked-in names (~713 API calls, ~12 minutes) chosen to span both reporting
frequencies, every exchange in `stock_NA1_EU1` including the restored European ones, all
five carve-out cohorts, the market-cap bands down to sub-$50M, and the known edge cases
(duplicate-dated rows, zero market cap, history-gate failures, preferred/warrant lines).
Cross-listing issuer groups are closed wherever closing them does not require re-importing a
preferred or notes line; the handful that would are DECLARED in `universes.TEST_UNIVERSE_OPEN_GROUPS`
and reconciled against a fresh derivation by test, so dedup divergence from production is bounded
and named rather than assumed away.
Membership is frozen, so two iterations are comparable. Run
`python .\verify_test_universe.py` to see the coverage measured rather than asserted.

> **`-nrTaT 50` is NOT a substitute.** It keeps the *first* N rows of FMP's ticker list —
> an arbitrary positional prefix that under-represents semi-annual filers, non-USD
> reporters and whole cohorts, and whose membership shifts whenever FMP reorders the list.
> It is fine for "does the process start"; it is not fine for iterating on behaviour.
>
> **Scores from `stock_TEST1` are not production scores.** Z-scores, percentile cuts,
> cohort scoring and the top-20/top-5 selections all depend on pool composition, so a
> 142-name pool produces numbers that are not comparable to a full run. The run banner
> says so on every invocation.

5) Load both metrics and results (pure report generation, no fetch or recompute):

```powershell
python .\Sbocker.py -loadbometric 1 -bometricfilename <metrics_pickle> -loadboresults 1 -boresultsfilename <results_pickle>
```

6) Run normalized analysis (produces `results_normalized_YYYY-MM-DD.pickle`):

```powershell
python .\normalized_analysis.py
```

7) Run the simple rolling backtest (loads saved metric data, computes rolling returns with annual rebalancing by default):

```powershell
python .\rolling_backtest.py
```

**Common CLI flags for `Sbocker.py`:**

- `-loadbometric 1`: load saved metrics pickle instead of fetching from API.
- `-bometricfilename <name>`: specify the metrics pickle to load.
- `-loadboresults 1`: load saved results pickle instead of recomputing.
- `-boresultsfilename <name>`: specify the results pickle to load.
- `-nrTaT <N>`: limit to the first N tickers processed. A *positional prefix*, not a sample — for a small representative run use `-tickerfilter stock_TEST1` instead (see step 4).
- `-startfromlastindex 1`: resume from the last read index (uses `lastIndexOfRead_*` file).
- `-tickerfilter <name>`: **choose the universe.** Defaults to `stock_NA1_EU1`. Definitions live in
  `universes.py` (one source of truth; exchange codes verified against live FMP data 2026-08-02):

  Counts measured 2026-08-02 against the live FMP lists. **`pre-filter`** = the sum over the
  universe's exchange codes; **`resolved`** = what a run actually fetches, after the type filter,
  the share-class/instrument filter and the delisted prune. The two differ by ~7%; don't quote one
  as the other.

  | name | scope | pre-filter | resolved |
  |---|---|---|---|
  | `stock_TEST1` | curated TEST universe, 142 fixed names — **use this for iteration** | 142 | 140 |
  | `stock_US1` | US only: NYSE + NASDAQ | 6,141 | 5,393 |
  | `stock_NA1` | North America: + TSX | 6,803 | 6,009 |
  | `stock_US1_EU2` | US + Euronext (PAR AMS BRU LIS) | 6,963 | 6,215 |
  | `stock_WW1_TV` | US + LSE + XETRA + Euronext | 9,900 | 9,149 |
  | `stock_US1_EU1` | US + Europe | 10,835 | 10,077 |
  | `stock_NA1_EU1` | **default** — North America + Europe | 11,497 | 10,693 |
  | `stock_FULL1` | everything FMP serves with statements (incl. OTC + Asia) — **not** the default | — | 49,071 |

  ⚠️ **`stock_NA1_EU1`, `stock_US1_EU1`, `stock_US1_EU2` and `stock_WW1_TV` changed meaning on
  2026-08-02.** They filtered on `EURONEXT` and `OSE`, which are not FMP exchange codes and matched
  **zero** rows; the real codes are `PAR`/`AMS`/`BRU`/`LIS` and `OSL`. That restored **1,046
  statement-bearing common stocks** no run had ever fetched. Artifacts written before that date carry
  the same *name* but a different *membership*, so do not compare them on membership or on any pooled
  statistic (z-score, percentile, top-100, beat-rate). Every artifact is now stamped with a
  `universe_fingerprint`; match on that, not on the name.
- `-period <quarter|annual>`: set data period. Defaults to `quarter`.
- `-nrperiods <N>`: number of periods to fetch. Defaults to 24 (6 years of quarters).
- `-compyear <lastYear|thisYear>`: comparison year for metrics. Defaults to last year.

Notes & recommendations

- The normalized analysis produces a parallel set of results using cross-sectional, per-date z-scores. This helps to ensure that each metric contributes comparably to the aggregate score.

- The `timing_maybe_off` flag is conservative: it flags tickers that include a metric row dated in the current calendar year (possible recency/lookahead). This is intended as an audit flag, not a filter.

- The rolling-backtest is intentionally simple and uses nearest available prices on or after rebalance dates; it does not model transaction costs, market impact, liquidity constraints, or slippage. Use it to compare relative strategy behavior; add transaction-cost modeling before claiming live implementability.

- If you upgrade pandas in the future and see dtype changes, consider applying stricter dtype casting or dropping columns that are fully empty after collection.

Next improvements to consider

- Add explicit lookback/lag windows for each metric to remove any lookahead bias.
- Add a transaction-cost model and turnover constraints to the backtest.
- Add sector- and size-neutralization options when aggregating metrics.
- Add unit tests that mock API responses for reliable regression testing.
- Lower API Overhead in post-analysis.

