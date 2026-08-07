# ValuationAnalysis codebase review

## Scope and verification standard

This is a static, read-only review of the pipeline initialized by `Sbocker.py`. No application code was changed and no live API or full pipeline run was performed.

Concrete defects below are included only where the relevant definition, caller, and control flow were checked. Broader design comments are labeled as recommendations rather than defects.

## Executive opinion

This is a serious financial-research codebase with considerably more domain awareness than its structure initially suggests. It explicitly addresses point-in-time selection, survivorship bias, delisted companies, duplicate issuers, restatements, corrupt price data, forensic accounting signals, and agreement between live and offline scoring.

The central weakness is not the investment logic. It is that a large amount of correctness depends on mutable dictionaries, implicit DataFrame schemas, stage ordering, and conventions spread across modules. The research logic has outgrown the original script-oriented architecture.

My concise assessment:

- Financial-domain sophistication: high
- Awareness of data and backtest hazards: high
- Live-pipeline maintainability: medium-low
- Operational resilience: improving but inconsistent
- Confidence in the intended live ranking flow: reasonably good
- Confidence in `-asof` as a clean historical simulation: intentionally limited by the code itself

## Pipeline as implemented

The main decisional flow is:

1. `configuration.getDataFetchConfiguration()` builds a configuration dictionary.
2. `getData_gen.get_tickers()` builds the universe.
3. `getData_fmp.get_fundamentals_fmp()` fetches five FMP datasets and constructs fundamentals and derived metrics.
4. `data_quality.apply_data_quality_filter()` prunes invalid observations.
5. `postBo.postBoWrapper()` runs Stage 1 scoring, carve-out/cohort handling, and Stage 2 ranking.
6. Moat, forensic, and manipulation outputs decorate the ranked results.
7. CSV, XLSX, pickle, and append-only pick-log artifacts are emitted.
8. Optional analysis, delisted ingestion, unified backtesting, and Drive transfer run afterward.

This ordering is also accurately summarized in the header of `Sbocker.py`.

## What is particularly good

### Financial-data hazards are treated as first-class concerns

The code does not assume that ticker identity, statement alignment, or historical availability is trivial. Examples include:

- `_align_statements_by_date()` in `getData_fmp.py`, which avoids silently combining different fiscal periods by positional index.
- Issuer-level deduplication for share classes and cross-listings.
- Delisted-entity ingestion and separate survivorship-analysis tooling.
- Data-quality filtering before scoring.
- Deterministic handling of duplicate-dated/restated quarters in `stage2_metrics.prepare_eps_series()`.

These are substantive safeguards, not cosmetic abstractions.

### Stage 2 has a valuable single source of truth

Keeping Stage-2 metric formulas in `stage2_metrics.py`, shared by live and point-in-time reproduction code, is one of the strongest architectural choices in the repository. It directly reduces the risk that a backtest validates a formula different from the one used for live selections.

### Output durability is thoughtfully handled

`Sbocker.py` persists and optionally transfers artifacts at multiple phase boundaries. Long optional stages occur after the primary ranking and deliverables are written. This is sensible for an overnight, API-dependent process.

### Recent code is more explicit about assumptions

The newer modules contain useful explanations of availability limitations, fallback behavior, cohort behavior, and live/offline invariants. The repository is moving in a better direction: important assumptions are increasingly visible near the code that implements them.

## Confirmed defects and inconsistencies

### 1. The configured manual-elimination list is discarded on a fresh run

`configuration.py` loads `manualelimtickers` and places it in `configdic`. In `Sbocker.main()`, the value is read:

```python
manualelimtickers, baseurl = configdic['manualelimtickers'], configdic['baseurl']
```

It is immediately replaced with an empty list:

```python
manualelimtickers = []
```

The empty list is then passed to `get_tickers()`. Therefore the configured manual-elimination CSV has no effect on the fresh-fetch universe through this path. This is a confirmed control-flow issue, not a stylistic concern.

### 2. Fresh data is quality-filtered twice

On the fresh-fetch branch, `apply_data_quality_filter()` is called once before saving metrics. It is then called again unconditionally after the load/fetch branch converges.

The second call is largely idempotent with respect to already-removed rows, but it is not a pure no-op:

- it repeats a full scan;
- it replaces `removed_data_quality` with the result of the second scan, which will commonly be empty;
- it can obscure which observations were removed by the first pass;
- it creates duplicate operational noise and makes stage ownership unclear.

Loaded metrics need the common filter; freshly fetched metrics do not need both calls.

### 3. The legacy `-portfolioTest` path is nonfunctional

This was checked across both `configuration.py`, `Sbocker.py`, and `portfolio.py`.

- `configuration.py` stores the CLI value as a string rather than converting it to an integer.
- `Sbocker.py` compares that value with integer zero using `portfoliotestyear > 0`, which raises `TypeError` when the flag is supplied under Python 3.
- If that conversion were fixed, `Sbocker.py` calls `portfolio.portfolioBacktestWrapper()`, but no function with that name exists in `portfolio.py`.
- The code following the branch assumes `resdic` exists, while the portfolio branch never assigns it.

The newer `-runbacktest`/`backtest_unified.py` path is separate. This finding applies specifically to the legacy `-portfolioTest` branch.

### 4. `BoDCF.py` cannot be imported

The module defines:

```python
def BoDCF(resdic,mos=0.66,baseurl,api_key):
```

Python does not allow non-default parameters after a default parameter, so parsing the module raises `SyntaxError`. A repository-wide reference scan found no imports or calls from other modules, indicating that this is currently disconnected legacy code rather than a defect in the active `Sbocker.py` path.

The module also contains a `getWACC()` call whose positional argument order does not match its definition. That reinforces the conclusion that this file is stale, but it does not affect the active pipeline while the module remains unused.

### 5. HTTP reliability is inconsistent across stages

`getData_gen.py` contains retry/backoff/timeout helpers, and parts of delisted ingestion also use bounded requests. However, the primary statement-fetch path defaults to bare `requests.get`, and `postBo.py` makes many direct FMP calls without explicit timeouts or retry handling.

This is operationally significant because Python Requests has no timeout by default. A stalled endpoint can therefore hold an otherwise successful run indefinitely. The reporting functions also make repeated per-symbol requests after ranking has completed, so optional enrichment can prevent final deliverables from completing cleanly.

This is a confirmed implementation inconsistency. Whether it has caused observed failures would require runtime evidence.

### 6. A DCF debug path relies on a variable that is only conditionally assigned

In `postBo.writeBoAggToCSV()`, `temp_resp_dcf_raw` is assigned only in the fallback branch used when a symbol is absent from `dcf_bulk_dict`. The subsequent first-ticker diagnostic reads `temp_resp_dcf_raw.status_code` unconditionally.

At present all three bulk dictionaries are initialized empty and never populated in that function, so the fallback branch always runs and the active behavior does not fail. If bulk DCF loading is later enabled as the comments suggest, the first bulk hit would make the diagnostic reference an unassigned local variable. This is a latent defect in the advertised bulk/fallback structure, not a current failure under the empty-dictionary implementation.

## Point-in-time assessment

The codebase should receive credit for explicitly acknowledging that `-asof` is not yet a clean point-in-time simulation. `configuration.py` emits a warning stating that the path is partial-PIT and identifies remaining lookahead in cross-sectional baselines, per-ticker means, and DCF/beta substitution.

Additional evidence appears in `stage2_metrics.py`, which states that `cdx_df` lacks filing-date/`acceptedDate` information for disambiguating restatements.

Accordingly, the accurate conclusion is:

- Row-level date slicing exists and is intentionally threaded through Stage 1 and Stage 2.
- The project is actively designed around point-in-time concerns.
- The `-asof` flag must not yet be interpreted as a fully information-available historical replay.
- This is a known, documented limitation rather than a hidden bug.

The most important research improvement would be to retain and use filing/acceptance availability dates throughout the canonical fundamentals panel, then compute every cross-sectional baseline strictly from information available at the evaluation date.

## Architectural recommendations

These are recommendations, not claims that the present output is wrong.

### Break `Sbocker.main()` into explicit stages

`main()` currently owns configuration, ingestion, filtering, scoring, persistence, reporting, logging, optional analysis, backtesting, and transfer. A safer shape would be small stage functions such as:

```text
build_universe
fetch_fundamentals
validate_panels
score_universe
build_rankings
decorate_results
write_deliverables
run_optional_analysis
```

Each stage should accept and return an explicit contract and be independently testable.

### Replace accumulating dictionaries with validated contracts

`datandmetricdic` and `resdic` are flexible, but required keys and DataFrame schemas are implicit. Lightweight dataclasses plus schema checks would expose missing keys, wrong date types, absent columns, and unexpected sort order at the boundary where the problem begins.

### Centralize FMP access

One FMP client should own:

- timeouts;
- retry/backoff behavior;
- rate-limit handling;
- JSON/status validation;
- caching where appropriate;
- injectable transports for tests;
- API-key redaction in errors and logs.

The existing helpers provide a good starting point, but all live request paths need to use the same policy.

### Separate decisional data from presentation enrichment

The ranked output should be considered complete before profile, rating, peers, sector PE, and presentation-only DCF requests occur. Enrichment failures should yield explicit missing fields without jeopardizing the core ranking artifact.

### Write an executable metric specification

For each score component, document and validate:

- economic meaning and expected direction;
- source fields;
- availability lag;
- transformation and winsorization;
- missing-value treatment;
- sector/size treatment;
- weight;
- minimum history requirement.

Much of this knowledge already exists, but it is distributed across dictionaries, functions, and comments.

## Testing observations

There is meaningful test coverage around newer high-risk areas, including Stage-2 behavior, PIT machinery, issuer deduplication, delisted ingestion, returns, pick logging, and manipulation detection.

The test suite is not uniform, however. Several root-level files named `test_*.py` behave more like environment-dependent diagnostic scripts: they read a real API key, expect dated local pickle files, perform live HTTP calls, or execute at import time. Those should be distinguished from deterministic unit tests so a test command has a clear meaning.

The highest-value additional tests would be:

1. A small end-to-end `Sbocker` orchestration test with all I/O injected.
2. A regression test proving manual eliminations reach `get_tickers()`.
3. Tests for fresh-versus-loaded quality-filter stage behavior.
4. A configuration/control-flow test for every CLI branch.
5. Contract tests for required columns and date ordering at each scoring boundary.
6. HTTP failure tests covering timeout, malformed JSON, rate limiting, and partial endpoint availability.

## Suggested priority order

1. Fix or retire the confirmed dead branches: manual eliminations, `-portfolioTest`, and `BoDCF.py`.
2. Remove the duplicate quality-filter pass while preserving the first-pass removal audit.
3. Route all FMP calls through one bounded, testable client.
4. Define and validate pipeline-stage schemas.
5. Complete filing-availability-based PIT handling before relying on historical results as clean simulations.
6. Gradually decompose `Sbocker.main()` without rewriting the financial formulas.

## Final view

I would not rewrite this system from scratch. The difficult part—the domain knowledge and awareness of subtle financial-data failure modes—is already present. A rewrite would risk losing that knowledge.

The better path is to preserve the metric and ranking behavior, surround it with explicit contracts and deterministic tests, and progressively make the orchestrator thinner. The codebase is valuable; its main need is for the software structure to catch up with the sophistication of the research logic.
