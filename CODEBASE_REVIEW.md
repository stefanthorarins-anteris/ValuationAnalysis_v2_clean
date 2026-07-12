# ValuationAnalysis — Codebase Review

**Date:** 2026-07-12  
**Scope:** Opinion-only review; no code changes.

---

## What This Is

This is a **fundamental equity screening and ranking pipeline**, not a web app or library. Roughly ~14k lines across ~56 Python modules. The flow is:

**FMP API → metric computation → tier-weighted scoring → ranking/post-processing → Excel/CSV outputs → forensic checks → optional backtest**

The domain is clear: Graham-style value, quality ratios, cash-flow emphasis, manipulation screens (Beneish M, Montier C), moat heuristics, and DCF/beta valuation overlays. That reads like a real stock-picking workflow, not a tutorial project.

---

## What Impresses Me

### 1. Methodological Honesty

The codebase is unusually candid about its own biases. The `-asof` path warns that PIT is only partial. `forensicFlags.py` says "flags, not verdicts." The README admits the rolling backtest is simplified. `pit_slice.py` documents the filing-date lag problem and why fixed-lag fallback is conservative. That level of epistemic hygiene is rare in quant code.

### 2. The Recent PIT/Survivorship Work Is Serious

`pit_slice.py`, `universe_pit.py`, `entity_id.py`, the rewritten `dcf.py`, `beta.py`, `delisted_ingest.py`, and `baseline_tools/test_pit_engine.py` feel like a deliberate restructure, not a patch. The `as_of=None` → live behavior unchanged invariant is a good design choice: you can add rigor without breaking production runs.

### 3. Domain Depth in the Scoring Model

`createDicts.py` encodes a rich metric taxonomy (base / mean-relative / unity / diff / special) with tier weights (S/A/B/C/D). The scoring in `calcScore.py` is interpretable: each metric contributes positively only when it moves in the expected direction, averaged over N quarters. That's easier to defend to a human reviewer than a black-box model.

### 4. Practical Operational Concerns Are Handled

- Pickle caching for expensive API runs
- Resume via `lastIndexOfRead_*`
- `data_quality.py` catching API garbage (negative prices, price/mcap mismatches)
- Manual elimination lists
- `timing_maybe_off` and normalized analysis for cross-sectional comparability

Someone has run this at scale (~9k tickers) and hit real-world data problems.

### 5. Forensic Layer Is Thoughtful

`forensicFlags.py` is well-designed: offline, no extra API calls, decomposes Beneish drivers, handles financial-sector invalidity, and uses within-shortlist quintile ranking for Sloan accruals rather than pretending there's a universal threshold.

---

## What Concerns Me

### 1. Architecture: Evolved Research Scripts, Not a Product

Everything flows through giant dicts (`datandmetricdic`, `resdic`) passed between modules. There's no package structure, no typed interfaces, no clear layer boundaries. That's fine for a solo researcher, but it makes the system hard to reason about, test end-to-end, or hand off. `Sbocker.py` even re-imports everything inside `main()`.

### 2. Naming and Polish Signal "Long-Lived Research Code"

`Sbocker.py`, `BoMetric`, `getAves_fuckedTTT`, `Boresults_dic-...` filenames — this has clearly grown organically. Some modules (`dcf.py`, `forensicFlags.py`) read like spec-driven engineering; others (`postBo.py`) have large diagnostic print blocks that feel like active debugging.

### 3. Latent Bugs in Configuration

`configuration.py` has several copy-paste issues (e.g. sector filter indexing `imb` instead of `isf`, `compyear` using `id` instead of `ic`). They may not fire on default paths, but they're the kind of thing that bites when you actually use those flags.

### 4. Performance Will Hurt at Full Universe Scale

The fetch loop in `getData_fmp.py` is ticker-by-ticker with `pd.concat` in the loop. Scoring iterates every unique symbol. At ~9k names this works because you cache pickles — but recomputing from scratch will be slow, and there's no parallelism despite FMP being the bottleneck.

### 5. PIT Is Started but Not Finished

The cross-sectional baseline (`getAves2` / `bm_ave`) and per-ticker means still use the full panel when `-asof` is set. So historical backtests can still embed lookahead unless you're careful. The infrastructure is there; the scoring layer hasn't caught up yet.

### 6. Testing Is Narrow

`baseline_tools/` has solid synthetic tests for the PIT engine, DCF, beta, entity splitting. But there's no mocked-API integration test for the full pipeline, and `requirements.txt` has no version pins — reproducibility risk over time.

### 7. Repo Hygiene

Pickles, CSVs, and XLSX outputs sit alongside source. `.gitignore` was fixed recently, but the repo still feels like a working directory rather than a clean artifact. The `old/` folder and commented-out blocks add noise.

---

## The Scoring Philosophy (My Read)

This isn't trying to be a factor model or ML alpha engine. It's a **composite quality/value score** with:

- Hard gates (fail tests, manual eliminations, data quality)
- Soft tier weighting (not all metrics equal)
- Cross-sectional context (mean-relative variants)
- Momentum-of-fundamentals (diff variants over quarters)
- Post-hoc forensic vetting on the shortlist

That's a defensible approach for a fundamental investor who wants a ranked funnel, not a trading signal with statistical significance claims. The `baseline_tools/` suite (`real_ic.py`, `beat_rate.py`, `churn_horizon.py`) suggests you're starting to evaluate whether the score actually predicts — which is the right next question.

---

## Overall Verdict

**This is a credible, practitioner-built equity research system in mid-transition from "works for live screening" to "defensible for historical backtesting."**

| Dimension | Rating (subjective) |
|-----------|---------------------|
| Domain knowledge | Strong |
| Live screening utility | Strong |
| Backtest / PIT rigor | In progress, thoughtfully designed |
| Code architecture | Weak-to-moderate |
| Test coverage | Moderate for new modules, thin for pipeline |
| Documentation | Better than average for this style of project |
| Production readiness | Good for a personal/small-team workflow |

If I were advising without touching code: the highest-leverage improvements would be finishing PIT through L2/L3 (cross-sectional baselines), adding version-pinned dependencies, and extracting the scoring/fetch layers into testable units — not rewriting the scoring model itself, which already encodes real investment judgment.

The recent commit history (`point-in-time / survivorship-safe pipeline`) tells me this is actively being hardened, not abandoned. That's the most important signal: the author knows where the weaknesses are and is working on the right ones.

---

## Key Modules Reference

| Module | Role |
|--------|------|
| `Sbocker.py` | Main orchestrator |
| `configuration.py` | CLI flag parsing |
| `getData_gen.py` / `getData_fmp.py` | Ticker universe + FMP fetch |
| `calcMetrics.py` / `createDicts.py` | Metric definitions and computation |
| `calcScore.py` / `postBo.py` | Scoring and post-processing |
| `detectManipulation.py` / `forensicFlags.py` | Beneish M, Montier C, accruals flags |
| `pit_slice.py` / `universe_pit.py` / `entity_id.py` | Point-in-time / survivorship engine |
| `dcf.py` / `beta.py` / `dcf_to_price.py` | Valuation overlays |
| `backtest_unified.py` / `rolling_backtest.py` | Backtesting harnesses |
| `baseline_tools/` | PIT tests, IC analysis, benchmarking |