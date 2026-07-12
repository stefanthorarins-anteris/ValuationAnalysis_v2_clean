"""
Universe de-duplication for the investment-filter baseline (OFFLINE).

PROBLEM (evidence: AggScoreTop100-2026-01-09 rows 1-2 = GSL / GSL-PB, identical
PE=2.4544, beta=0.9130, currentRatio=2.0085; also .ST/.L foreign lines): FMP
returns the SAME issuer fundamentals for secondary listings -- preferred shares
(-P*), units, warrants, and foreign depositary lines. These occupy multiple
top-20 slots for ONE economic bet, polluting the shortlist the target measures.

APPROACH -- two independent signals, applied conservatively:
  (1) SYMBOL-PATTERN: flag obvious secondary classes by suffix.
  (2) FUNDAMENTAL FINGERPRINT: rows whose LARGE-magnitude fundamentals
      (revenue, netIncome, totalAssets, weightedAverageShsOut) are byte-identical
      at the same latest report date are the same issuer. Exact equality across
      several independent quantities is effectively impossible by coincidence, so
      it is a strong same-issuer signal. Within a fingerprint group we keep ONE
      "primary" listing and drop the rest.

JUDGMENT POINTS (flagged -- reviewer must scrutinise, not auto-trust):
  * DUAL-CLASS commons (BRK-A/BRK-B, GOOG/GOOGL) are economically distinct yet
    share fundamentals. Default `collapse_dual_class=True` -> one bet per issuer,
    because the target counts issuer-level winners. Set False to keep both.
  * A preferred/secondary line with NO common sibling in the set is KEPT (better
    to keep a possibly-valid holding than to wrongly drop it).
  * "Primary" pick within a group is a heuristic: prefer a symbol with no
    punctuation, then the shortest, then alphabetical. It does NOT use liquidity
    (not available offline cheaply). Flagged as a heuristic, not ground truth.

This module NEVER hits the network and NEVER mutates its input. It returns the
kept symbols plus a dry-run report so the removals can be inspected before use.
"""

import re

import numpy as np
import pandas as pd

DEDUP_VERSION = "v1-2026-07-11"

# Suffix patterns for secondary classes. NYSE uses '-' before the class letter;
# some feeds use '.'. Preferred = -P, -PA..-PZ. Warrants/units/rights below.
_PREFERRED_RE = re.compile(r"[-.]P[A-Z]?$")
_WARRANT_RE = re.compile(r"([-.]W[TS]?|\.WS)$")
_UNIT_RE = re.compile(r"[-.]U$")
_RIGHT_RE = re.compile(r"[-.]R(T)?$")
# Dual-class trailing single letter, e.g. BRK-A / BRK-B  (or .A/.B)
_DUALCLASS_RE = re.compile(r"[-.][A-Z]$")

FINGERPRINT_COLS = ["revenue", "netIncome", "totalAssets", "weightedAverageShsOut"]


def classify_symbol(sym):
    """Return one of: 'preferred','warrant','unit','right','dualclass','primary'."""
    if _PREFERRED_RE.search(sym):
        return "preferred"
    if _WARRANT_RE.search(sym):
        return "warrant"
    if _UNIT_RE.search(sym):
        return "unit"
    if _RIGHT_RE.search(sym):
        return "right"
    if _DUALCLASS_RE.search(sym):
        return "dualclass"
    return "primary"


def _primary_score(sym):
    """Lower is 'more primary'. No punctuation < has punctuation; then shorter."""
    has_punct = 1 if re.search(r"[-.]", sym) else 0
    return (has_punct, len(sym), sym)


def _latest_rows(df):
    """One row per source: the latest (max-date) record."""
    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.sort_values("date")
    return d.groupby("source", as_index=False).tail(1).set_index("source")


def dedup_universe(df, collapse_dual_class=True,
                   fingerprint_cols=FINGERPRINT_COLS):
    """Collapse a fundamentals DataFrame to one primary listing per issuer.

    Parameters
    ----------
    df : DataFrame with at least 'source' (symbol) and the fingerprint columns
         (and ideally 'date'). Typically cdx_df, or cdx_df restricted to a
         candidate set (e.g. the point-in-time top-100).
    collapse_dual_class : if True, dual-class commons that share a fingerprint
         are collapsed to one; if False they are kept.

    Returns
    -------
    kept : list[str]         symbols to keep (primary listings)
    dropped_map : dict       dropped_symbol -> kept_symbol it collapsed into
    report : DataFrame       one row per input symbol with the decision + reason
    """
    latest = _latest_rows(df)
    symbols = list(latest.index)

    avail_fp = [c for c in fingerprint_cols if c in latest.columns]

    # Build fingerprint per symbol (only when all fp cols present & finite).
    def _fp(sym):
        if not avail_fp:
            return None
        vals = []
        for c in avail_fp:
            v = latest.at[sym, c]
            if pd.isna(v) or not np.isfinite(v):
                return None
            vals.append(round(float(v), 6))
        return tuple(vals)

    fps = {s: _fp(s) for s in symbols}

    # Group symbols by fingerprint (ignore None fingerprints -> singleton groups).
    groups = {}
    singletons = []
    for s in symbols:
        fp = fps[s]
        if fp is None:
            singletons.append(s)
        else:
            groups.setdefault(fp, []).append(s)

    kept = set()
    dropped_map = {}
    report_rows = []

    # Fingerprint groups: choose one primary, collapse the rest.
    for fp, members in groups.items():
        if len(members) == 1:
            s = members[0]
            kept.add(s)
            report_rows.append((s, "keep", "unique-fingerprint", ""))
            continue
        # Multiple members share issuer fundamentals.
        classes = {m: classify_symbol(m) for m in members}
        only_dualclass_diff = all(c in ("primary", "dualclass")
                                  for c in classes.values())
        if only_dualclass_diff and not collapse_dual_class:
            for m in members:
                kept.add(m)
                report_rows.append((m, "keep", "dualclass-kept (toggle)", ""))
            continue
        primary = sorted(members, key=_primary_score)[0]
        kept.add(primary)
        report_rows.append((primary, "keep", "fingerprint-group-primary",
                            "|".join(members)))
        for m in members:
            if m == primary:
                continue
            dropped_map[m] = primary
            report_rows.append((m, "drop",
                               f"same-issuer-as:{primary} ({classes[m]})", ""))

    # Singletons (no reliable fingerprint): keep, but note obvious secondaries.
    for s in singletons:
        cls = classify_symbol(s)
        note = "no-fingerprint"
        if cls != "primary":
            note = f"no-fingerprint;pattern={cls} (kept: no sibling matched)"
        kept.add(s)
        report_rows.append((s, "keep", note, ""))

    report = pd.DataFrame(report_rows,
                          columns=["source", "decision", "reason", "group"])
    report = report.sort_values(["decision", "source"]).reset_index(drop=True)
    return sorted(kept), dropped_map, report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Dry-run universe dedup over a fundamentals CSV/pickle.")
    ap.add_argument("path", help="CSV or pickle with 'source' + fundamentals")
    ap.add_argument("--no-collapse-dual-class", action="store_true")
    args = ap.parse_args()
    if args.path.endswith(".pickle") or args.path.endswith(".pkl"):
        data = pd.read_pickle(args.path)
    else:
        data = pd.read_csv(args.path)
    kept, dropped, rep = dedup_universe(
        data, collapse_dual_class=not args.no_collapse_dual_class)
    print(f"kept {len(kept)} symbols; dropped {len(dropped)}")
    print(rep[rep["decision"] == "drop"].to_string(index=False))
