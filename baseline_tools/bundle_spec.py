"""
The ANALYSIS BUNDLE -- the HOME->HERE interface for the investment-filter baseline.

WHY A BUNDLE
------------
Git is the cross-machine sync channel and GitHub hard-rejects files >100 MB.
The full fundamentals pickles are ~138 MB each -> they CANNOT cross machines via
git and are NOT the interface. They live locally on each machine (used HERE only
for dev/test of this code). Production data crosses as this compact bundle.

  HOME machine  (has full pickles + does the API fetch)
     -> runs Stage-2 point-in-time selection (needs full BoMetric_df/cdx_df)
     -> runs the real-price fetch (fetch_prices.py, bulk-by-date, ~6 calls)
     -> emits THIS BUNDLE  (target: a few MB; hard cap <100 MB)
     -> `git push`
  HERE machine  (offline, network-free)
     -> `git pull`
     -> validate_bundle() + run the beat-rate analysis as a PURE FUNCTION of
        the bundle (beat_rate.py). No network, no big pickles.

SIZE
----
selections: n_windows * ~100 names * ~30 cols   (3*100*30 ~ 9k cells)
prices:     n_windows * <=100 names * 2 dates    (3*100*2  ~ 600 rows)
benchmark:  a handful of index levels
=> kilobytes-to-low-MB. Comfortably < 100 MB.

LAYOUT (a directory, or a zip of it)
------------------------------------
  <bundle>/
    manifest.json     provenance + config + fidelity flags (see MANIFEST_KEYS)
    selections.csv    the point-in-time Stage-1/Stage-2 rankings per window
    prices.csv        REAL split/div-adjusted closes for the SELECTED names
    benchmark.csv     MSCI World index levels (see benchmark_loader.py)

This module is the single source of truth for the schema. Both the HOME-side
packer and the HERE-side checker import these constants so the contract can't
silently drift. It performs NO network I/O.
"""

# ---- selections.csv -------------------------------------------------------
# One row per (window, selected symbol). "Selected" = in the point-in-time
# top-100 (Stage-1 by BoScore) with Stage-2 AggScore computed; is_top20 marks
# the real shortlist the target is about.
SELECTIONS_COLS = [
    "window_id",        # e.g. "buy2019"
    "buy_date",         # ISO date, point-in-time rebalance
    "eval_date",        # ISO date, buy + hold
    "source",           # symbol (post-dedup primary listing)
    "stage1_rank",      # rank by BoScore (1 = best)
    "stage2_rank",      # rank by AggScore (1 = best); the shortlist order
    "BoScore",
    "AggScore",
    "is_top20",         # bool: stage2_rank <= 20
    "currency",         # best-effort; "USD" or other (for FX confidence flag)
    "non_usd_flag",     # bool: True if currency != USD (lower-confidence)
    "exchange_suffix",  # e.g. "", ".ST", ".L" (evidence for the currency guess)
    # NOTE: the 21 AggScore component metrics MAY be appended as extra columns
    # (prefixed "m_") for diagnostics; they are optional and ignored by the
    # HERE-side beat-rate math.
]

# ---- prices.csv -----------------------------------------------------------
# Real adjusted closes for the selected names, at each window's buy & eval leg.
# adjClose is split+dividend adjusted, so a per-name TOTAL return is simply
# eval_adjClose / buy_adjClose - 1  (no separate dividend term needed).
PRICES_COLS = [
    "window_id",
    "source",           # symbol
    "leg",              # "buy" or "eval"
    "date_actual",      # the trading day actually used (holiday-resolved)
    "adjClose",         # split+dividend-adjusted close
]

# ---- benchmark.csv --------------------------------------------------------
# MSCI World index levels. MUST be a TOTAL-RETURN variant (Net TR recommended)
# to be apples-to-apples with adjClose (which reinvests dividends). See
# benchmark_loader.py for the CEO decision this implies.
BENCHMARK_COLS = [
    "date",             # ISO date
    "level",            # index level
]

MANIFEST_KEYS = [
    "generated_utc",
    "source_pickle",        # which snapshot the selection was computed from
    "config",               # {ntopagg, ntopxlsx, hold_months, buy_years, nq, ...}
    "stage2_fidelity",      # {dcf_to_price: "dropped"|"fmp_live"|"reconstructed",
                            #  cyclheat_beta: "const_1.0"|"fmp_profile"|"rolling",
                            #  boscore_average_basis: "pit"|"full_snapshot",
                            #  as_reported: false  # restatement caveat}
    "dedup_version",        # universe_dedup rule version applied
    "benchmark_variant",    # e.g. "MSCI World Net TR USD"
    "n_api_calls",          # exact count the fetch made (exposure record)
]


def validate_bundle(bundle_dir):
    """Offline structural check. Returns (ok, problems:list[str])."""
    import os
    import json
    import csv

    problems = []

    def _check_csv(name, required):
        path = os.path.join(bundle_dir, name)
        if not os.path.exists(path):
            problems.append(f"missing {name}")
            return
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        missing = [c for c in required if c not in header]
        if missing:
            problems.append(f"{name} missing columns: {missing}")

    _check_csv("selections.csv", SELECTIONS_COLS)
    _check_csv("prices.csv", PRICES_COLS)
    _check_csv("benchmark.csv", BENCHMARK_COLS)

    man = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(man):
        problems.append("missing manifest.json")
    else:
        try:
            with open(man) as f:
                m = json.load(f)
            for k in MANIFEST_KEYS:
                if k not in m:
                    problems.append(f"manifest.json missing key: {k}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"manifest.json unreadable: {e}")

    return (len(problems) == 0), problems
