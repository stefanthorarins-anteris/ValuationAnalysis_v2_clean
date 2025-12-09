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

4) Run a small test pipeline (processes only first 50 tickers to verify the pipeline):

```powershell
python .\Sbocker.py -nrTaT 50
```

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
- `-nrTaT <N>`: limit to first N tickers processed (useful for testing; e.g., `-nrTaT 50` for a quick 50-ticker run).
- `-startfromlastindex 1`: resume from the last read index (uses `lastIndexOfRead_*` file).
- `-tickerfilter <name>`: choose exchange/region filter (e.g., `stock_NA1`, `stock_US1_EU1`). Defaults to `stock_NA1_EU1`.
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

