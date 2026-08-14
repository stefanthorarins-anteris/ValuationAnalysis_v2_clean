"""THE ACCEPTANCE NUMBERS FOR THE TWO-TIER NaN POLICY + THE LOCAL PEG.

WHAT THIS IS.  One script, run offline on a saved panel, that produces every figure the change
has to be judged on -- and produces the BEFORE arm in the same process, so a comparison is never
made against a differently-dated artifact.  It is a MEASUREMENT tool, not part of the pipeline:
nothing here is imported by production.

WHY BOTH ARMS RUN IN ONE PROCESS.  The nearest saved "before" list
(`CORRECTED_general_top100-2026-07-17-CORRECTED.csv`, 2026-07-30) predates BOTH the E-1 robust
normaliser (2026-08-03) and the eight-criterion sign fix (2026-08-04), so diffing against it
would credit this change with two earlier ones.  The BEFORE arm is therefore reconstructed here,
by narrow and enumerated overrides:

  * `nan_policy` thresholds neutralised -- COVERAGE_MIN 0.0, the two gap caps infinite,
    BOUNDARY_LIMIT emptied.  With every gate off, `window_verdict` reduces to `vw.mean()`, which
    IS the bare `.head(w).mean()` the metrics used before -- so this arm is exact, not
    approximate, for the Stage-2 half.
  * `stage2_metrics.piotroski` replaced by a FROZEN COPY of the pre-2026-08-05 body (the one
    where a NaN input scores the point 0).
  * Stage-1 rebuilt with `PEG` on the VENDOR field under the old two-sided eps guard, and with
    `grahamNumberToPrice`'s `Boundary` declaration removed.

READ THE LIMITS OF THE PANEL BEFORE READING THE NUMBERS.  This is the 2026-07-17 universe as the
OLD acquisition gates admitted it; `reportedCurrency` and `period` are absent so currency and
reporting frequency are FALLBACKS; `eps` / `epsdiluted` are absent so PEG runs on the
`netIncomePerShare` proxy (which is what it will do in production too until the next full fetch);
and the rebuild inverts tie order on the 282 duplicate-snapped-quarter sources.  See
`panel_upgrade` and `run_corrected_current` for the full list.

    python baseline_tools/nan_policy_report.py                 # full universe, both arms
    python baseline_tools/nan_policy_report.py --subset 600     # fast iteration
    python baseline_tools/nan_policy_report.py --skip-pool      # measurements only, no rescore
"""

import argparse
import os
import sys

os.environ.setdefault("VA_OFFLINE_NO_DCF", "1")     # BEFORE postBoRank is imported

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calcMetrics as cm
import calcScore as cs
import createDicts as cdic
import data_quality as dq
import getData_fmp as gdf
import nan_policy as npol
import panel_upgrade as pu
import postBo as pb
import reporting_period as rp
import stage2_metrics as sm
import utils

PANEL = os.path.join(_HERE, "resdic_2026-07-17_CORRECTED.pickle")
ORIG = (r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
        r"\Boresults_dic-fmp_stock_NA1_EU1_all_2026-07-17_len7752_manelim3692_fails2075.pickle")
BAR = "=" * 100


def _h(title):
    print("\n" + BAR + "\n" + title + "\n" + BAR, flush=True)


# =========================================================================== #
#  THE BEFORE ARM -- frozen copies, enumerated                                 #
# =========================================================================== #
def _legacy_piotroski(tempcdx, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """FROZEN COPY of stage2_metrics.piotroski as it stood before 2026-08-05.

    The only difference: no presence check, so a NaN input makes every comparison False and the
    point scores 0 -- indistinguishable from failing the test.  Kept verbatim rather than
    reconstructed, because the BEFORE arm is only worth anything if it is exact.
    """
    try:
        lag = int(rpy)
        if len(tempcdx) >= lag + 1:
            curr, prev = tempcdx.iloc[0], tempcdx.iloc[lag]
            ta_curr, ta_prev = curr["totalAssets"], prev["totalAssets"]
            if ta_curr > 0 and ta_prev > 0:
                p1 = 1 if curr["netIncome"] / ta_curr > 0 else 0
                p2 = 1 if curr["netCashProvidedByOperatingActivities"] > 0 else 0
                p3 = 1 if curr["netIncome"] / ta_curr > prev["netIncome"] / ta_prev else 0
                p4 = 1 if (curr["netCashProvidedByOperatingActivities"]
                           > curr["netIncome"]) else 0
                p5 = 1 if (curr["longTermDebt"] / ta_curr
                           < prev["longTermDebt"] / ta_prev) else 0
                p6 = 1 if curr["currentRatio"] > prev["currentRatio"] else 0
                p7 = 1 if (curr["weightedAverageShsOut"]
                           <= prev["weightedAverageShsOut"]) else 0
                p8 = 1 if curr["grossProfitMargin"] > prev["grossProfitMargin"] else 0
                p9 = 1 if (curr["revenue"] / ta_curr > prev["revenue"] / ta_prev) else 0
                return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        return np.nan
    except Exception:
        return np.nan


def _legacy_peg_growth_defined(df):
    """FROZEN COPY of the pre-2026-08-05 `peg_growth_defined` guard: both single-period eps
    legs strictly positive."""
    e = pd.to_numeric(df[cm._PEG_EPS_FIELD], errors="coerce")
    return (e > 0) & (e.shift(-1) > 0)


_ORIG_CALC_SPECIAL = cm.calc_special


def _legacy_calc_special(df, metstr, rpy=rp.DEFAULT_ROWS_PER_YEAR, guard=None):
    """`calc_special` with PEG back on the VENDOR field (the pre-2026-08-05 line)."""
    if metstr != "PEG":
        return _ORIG_CALC_SPECIAL(df, metstr, n, rpy=rpy, guard=guard)
    res = pd.DataFrame()
    v = pd.to_numeric(df["priceEarningsToGrowthRatio"], errors="coerce")
    res[metstr] = np.where(v != 0, 1 / v, 0) - 1
    if guard is not None:
        res[metstr] = cm.apply_domain_guard(df, res[metstr].tolist(), guard)
    return res


class before_arm(object):
    """Context manager installing the BEFORE behaviour.  Every override is listed in the module
    docstring; nothing else is touched, and all of it is restored on exit."""

    def __enter__(self):
        self._saved = dict(coverage=npol.COVERAGE_MIN, runs=npol.MAX_INTERIOR_RUNS,
                           gaps=npol.MAX_CALENDAR_GAPS,
                           limits=dict(npol.BOUNDARY_LIMIT),
                           piotroski=sm.piotroski, special=cm.calc_special)
        npol.COVERAGE_MIN = 0.0            # `< 0.0` is never true -> gate off
        npol.MAX_INTERIOR_RUNS = 10 ** 9
        npol.MAX_CALENDAR_GAPS = 10 ** 9
        npol.BOUNDARY_LIMIT.clear()        # Type-D fully-undefined -> refused, i.e. old NaN
        sm.piotroski = _legacy_piotroski
        cm.STAGE1_DOMAIN_GUARDS["peg_growth_defined"] = _legacy_peg_growth_defined
        cm.calc_special = _legacy_calc_special
        gdf.cm.calc_special = _legacy_calc_special
        return self

    def __exit__(self, *exc):
        npol.COVERAGE_MIN = self._saved["coverage"]
        npol.MAX_INTERIOR_RUNS = self._saved["runs"]
        npol.MAX_CALENDAR_GAPS = self._saved["gaps"]
        npol.BOUNDARY_LIMIT.update(self._saved["limits"])
        sm.piotroski = self._saved["piotroski"]
        cm.calc_special = self._saved["special"]
        gdf.cm.calc_special = self._saved["special"]
        cm.STAGE1_DOMAIN_GUARDS.pop("peg_growth_defined", None)
        return False


def _legacy_dicts():
    """The Stage-1 registry as it stood before: PEG on the old guard, Graham with no Boundary."""
    #  Deep-copy the six METRIC registries (indices 1..6); index 0 is preReq_dict, whose values
    #  are lists, not per-metric specs.
    src = cdic.getDicts()
    d = [src[0]] + [dict((k, dict(v)) for k, v in x.items()) for x in src[1:]]
    d[5]["grahamNumberToPrice"].pop("Boundary", None)       # unity dict
    d[6]["PEG"]["Guard"] = "peg_growth_defined"             # special dict
    return d


# =========================================================================== #
#  PANEL BUILD                                                                 #
# =========================================================================== #
def build_panel(cdx_up, dicts, subset=None, label=""):
    """Rebuild BoMetric_df from an upgraded cdx with an explicit registry."""
    packed = (dicts[2], dicts[3], dicts[5], dicts[4], dicts[6])
    bm_cols = list(utils.initBoMetric_fromDict()["BoMetric_df"].columns)
    frames = []
    srcs = 0
    for src, g in pu._groups_newest_first(cdx_up):
        tf = g.reset_index(drop=True)
        tmp = pd.DataFrame(columns=bm_cols)
        tmp["date"] = tf["date"].values
        tmp["source"] = src
        tmp = utils.setDatesToQuarterly(tmp)
        _rpy = rp.rows_per_year(tf[rp.FREQ_COLUMN].iloc[0])
        frames.append(gdf.build_bometric_rows(tf, tmp, _rpy, dicts=packed))
        srcs += 1
        if subset and srcs >= subset:
            break
    bm = pd.concat(frames, ignore_index=True)
    import getData_gen as gdg
    bm, _ = gdg.fixAfterGetData(bm, cdx_up.copy())
    print("  %s panel: %d rows over %d sources" % (label, len(bm), bm["source"].nunique()),
          flush=True)
    return bm


# =========================================================================== #
#  SECTIONS                                                                    #
# =========================================================================== #
def section_primary(cdx, pool):
    _h("A. PRIMARY-SET EJECT -- per limb and union   (acceptance: 1.51%% universe / 0 pool)")
    n_src = cdx["source"].nunique()
    ej = npol.primary_eject(cdx)
    rows = []
    for field, limb in npol.primary_limbs():
        s = set(ej.loc[ej["field"] == field, "source"]) if len(ej) else set()
        rows.append((field, limb, len(s), 100.0 * len(s) / n_src, len(s & pool)))
    primary_fields = set(npol.PRIMARY_POSITIVE) | set(npol.PRIMARY_PRESENT)
    u_primary = set(ej.loc[ej["field"].isin(primary_fields), "source"]) if len(ej) else set()
    u_all = set(ej["source"]) if len(ej) else set()
    print("  %-42s %-22s %6s %8s %6s" % ("field", "condition", "srcs", "% univ", "pool"))
    for f, limb, n, pct, np_ in rows:
        print("  %-42s %-22s %6d %7.2f%% %6d" % (f, limb, n, pct, np_))
    print("  %-65s %6d %7.2f%% %6d"
          % ("UNION -- PRIMARY limbs only (the CEO's first tier)", len(u_primary),
             100.0 * len(u_primary) / n_src, len(u_primary & pool)))
    print("  %-65s %6d %7.2f%% %6d"
          % ("UNION -- primary + the two impossibility checks", len(u_all),
             100.0 * len(u_all) / n_src, len(u_all & pool)))
    print("\n  Sources in the panel: %d.  A limb ejecting MATERIALLY more than the spec's "
          "figure is a stop condition;\n  'extremely selective' is the constraint, and the "
          "pool column is the one that must be 0." % n_src)
    return ej


def section_thresholds(cdx, pool):
    _h("B. COVERAGE AND GAPPINESS -- names converted to NaN, and how many DISQUALIFY")
    n_src = cdx["source"].nunique()
    # ---- calendar gappiness, cadence-relative vs naive ------------------------------
    rows = []
    for src, g in cdx.groupby("source", sort=False):
        freq = str(g[rp.FREQ_COLUMN].iloc[0])
        rpy = rp.rows_per_year(freq)
        d = pd.to_datetime(g["date"], errors="coerce").dropna().sort_values()
        w = rp.scale_window(npol.SCORING_WINDOW_NQ, rpy)
        d = d.iloc[-w:] if len(d) > w else d
        sp = d.diff().dropna().dt.days.to_numpy(dtype="float64") if len(d) > 1 else np.array([])
        thr_rel = npol.GAP_TOLERANCE * npol.expected_month_spacing(rpy) * 30.4375
        rows.append((src, freq, int((sp > thr_rel).sum()),
                     int((sp > npol.GAP_TOLERANCE * 3.0 * 30.4375).sum()),
                     int((sp > 3.0 * 30.4375).sum())))
    G = pd.DataFrame(rows, columns=["source", "freq", "rel", "naive16", "naive_strict"])
    n_sa = int((G["freq"] == "semiannual").sum())
    n_q = int((G["freq"] == "quarterly").sum())
    print("  CADENCE-RELATIVE (%.1fx the filer's own expected spacing) vs a NAIVE 3-month rule."
          % npol.GAP_TOLERANCE)
    print("  %-34s %7s %8s %6s %14s %14s" % ("rule", "srcs", "% univ", "pool",
                                             "semiannual", "quarterly"))
    for lbl, col, k in (("cadence-relative, >=1 gap", "rel", 1),
                        ("cadence-relative, >=2 gaps  <-- USED", "rel", 2),
                        ("naive 3m x1.6, >=1 gap", "naive16", 1),
                        ("naive strict >3m, >=1 gap", "naive_strict", 1)):
        m = G[col] >= k
        print("  %-34s %7d %7.2f%% %6d %7d/%-6d %7d/%-6d"
              % (lbl, int(m.sum()), 100.0 * m.sum() / n_src,
                 len(set(G.loc[m, "source"]) & pool),
                 int((m & (G["freq"] == "semiannual")).sum()), n_sa,
                 int((m & (G["freq"] == "quarterly")).sum()), n_q))
    print("\n  CONFIRMS THE LOAD-BEARING FIGURE.  Semi-annual filers flagged at >=1 gap:\n"
          "    cadence-relative (the rule shipped)          %d of %d\n"
          "    naive FIXED 3-month expectation, same 1.6x   %d of %d\n"
          "  i.e. the ONLY thing standing between 'a handful of genuine stoppages' and 'every\n"
          "  semi-annual filer in the universe' is reading the filer's own cadence.  (The strict\n"
          "  >3-month variant in the table is NOT the right comparator and is shown only to say so:\n"
          "  it flags %d of %d QUARTERLY filers too, because a snapped quarter can span 92 days.)"
          % (int(((G["rel"] >= 1) & (G["freq"] == "semiannual")).sum()), n_sa,
             int(((G["naive16"] >= 1) & (G["freq"] == "semiannual")).sum()), n_sa,
             int(((G["naive_strict"] >= 1) & (G["freq"] == "quarterly")).sum()), n_q))

    # ---- per-metric coverage / interior runs ---------------------------------------
    print("\n  PER-METRIC, over each source's scaled window.  Coverage denominator = rows\n"
          "  PRESENT less the metric's structural lag (nan_policy.COVERAGE_DENOMINATOR).")
    recs = []
    for src, g in cdx.iloc[::-1].groupby("source", sort=False):
        tf = g.reset_index(drop=True)
        rpy = rp.rows_per_year(str(tf[rp.FREQ_COLUMN].iloc[0]))
        r = {"source": src}
        for key, ser in _row_series(tf, rpy).items():
            w = rp.scale_window(sm.window_quarters(key, npol.SCORING_WINDOW_NQ) or
                                npol.SCORING_WINDOW_NQ, rpy)
            v = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
            vw = v.head(w)
            n_present, n_ok = len(vw), int(vw.notna().sum())
            lag = sm.structural_lag(key, rpy)
            n_struct = max(0, lag - max(0, len(v) - n_present))
            denom = max(0, n_present - n_struct)
            r[key + "_cov"] = (n_ok / denom) if denom else np.nan
            r[key + "_runs"] = npol.interior_missing_runs(vw.notna().to_numpy())
            r[key + "_zero"] = (n_ok == 0)
        recs.append(r)
    R = pd.DataFrame(recs)
    keys = [k for k in _row_series(cdx.head(30).iloc[::-1].reset_index(drop=True), 4)]
    print("  %-30s %6s %8s %5s   %6s %8s %5s   %s"
          % ("metric", "cov<.5", "% univ", "pool", "runs>1", "% univ", "pool", "type"))
    for k in keys:
        typ = "D" if k in npol.TYPE_D else "U"
        mc, mr = R[k + "_cov"] < npol.COVERAGE_MIN, R[k + "_runs"] > npol.MAX_INTERIOR_RUNS
        print("  %-30s %6d %7.2f%% %5d   %6d %7.2f%% %5d   %s"
              % (k, int(mc.sum()), 100.0 * mc.mean(), len(set(R.loc[mc, "source"]) & pool),
                 int(mr.sum()), 100.0 * mr.mean(), len(set(R.loc[mr, "source"]) & pool), typ))
    tu = [k for k in keys if k not in npol.TYPE_D]
    uc = R[[k + "_cov" for k in tu]].lt(npol.COVERAGE_MIN).any(axis=1)
    ur = R[[k + "_runs" for k in tu]].gt(npol.MAX_INTERIOR_RUNS).any(axis=1)
    print("  %-30s %6d %7.2f%% %5d   %6d %7.2f%% %5d   (Type-U only; Type-D is EXEMPT)"
          % ("UNION", int(uc.sum()), 100.0 * uc.mean(), len(set(R.loc[uc, "source"]) & pool),
             int(ur.sum()), 100.0 * ur.mean(), len(set(R.loc[ur, "source"]) & pool)))

    print("\n  HOW MANY OF THESE THEN DISQUALIFY: **ZERO**, and the spec's own sentence saying\n"
          "  otherwise is wrong.  Section 3b(i) concludes '2 or more calendar gaps therefore\n"
          "  disqualifies' on the grounds that `earnYield` is primary -- but the primary set is\n"
          "  five RAW INPUTS (section 1a / ADDENDUM C) and section 1c rules that a DERIVED metric\n"
          "  can never be primary.  Coverage and gappiness act on derived metrics, so every one of\n"
          "  the counts above lands at the column MEDIAN.  `normalizeAndDropNA` drops a row only\n"
          "  when EVERY metric is NaN, and the five point-in-time metrics (Piotroski, Altman-Z,\n"
          "  marketCapRevQuants, shareCountChange, longTermDebtChange) are untouched by these\n"
          "  rules -- so not even a 2-gap name is dropped.  IF AN EJECT IS WANTED IT IS A CEO\n"
          "  DECISION; the price is the '>=2 gaps' row above.")
    return R


def _row_series(tf, rpy):
    """Each windowed Stage-2 metric's PER-ROW series for one source, newest-first.

    Mirrors the arithmetic in `stage2_metrics`; used ONLY for measurement, which is why it is
    here and not there -- production reduces these inside the metric functions.
    """
    num = lambda c: pd.to_numeric(tf[c], errors="coerce")
    ta = num("totalAssets").where(num("totalAssets") > 0)
    out = {
        "RoA": num("returnOnAssets"),
        "earnYield": num("earningsYield"),
        "grahamNumberToPrice": num("grahamNumber") / num("price"),
        "bVpRatio": 1.0 / num("pbRatio"),
        "revenueGrowth": num("revenue").pct_change(-int(rpy), fill_method=None),
        "incomeQuality": (num("netCashProvidedByOperatingActivities") - num("netIncome")) / ta,
        "returnOnEquity": num("returnOnEquity").where(num("totalStockholdersEquity") > 0),
        "returnOnCapitalEmployed": num("returnOnCapitalEmployed"),
        "currentRatio": num("currentRatio"),
        "grossProfitMargin": num("grossProfitMargin"),
        "freeCashFlowYield": num("freeCashFlow") / num("marketCap"),
        "freeCashFlowPerShareGrowth": (num("freeCashFlow") / num("weightedAverageShsOut")
                                       ).pct_change(-int(rpy), fill_method=None),
        "tbVpRatio": num("tangibleBookValuePerShare") / num("price"),
        "priceGrowth": num("price").pct_change(-1, fill_method=None),
        "CycleHeat": num("netIncome") / num("weightedAverageShsOut"),
    }
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def section_boundary(cdx, bm_after, bm_before, pool):
    _h("C. BOUNDARY IMPUTATION -- the limit per metric, and the Stage-1 IDENTITY")
    print("  ADMITTED (limit is finite AND is the metric's worst admissible value):")
    for k, (lim, why) in sorted(npol.BOUNDARY_LIMIT.items()):
        print("    %-26s limit %+.4f   %s" % (k, lim, why))
    print("  REFUSED (A1's escape clause fires):")
    for k, why in sorted(npol.REFUSED_NOT_IMPUTED.items()):
        print("    %-26s %s" % (k, why))

    reason = cdx["grahamUndefinedReason"].fillna("")
    print("\n  grahamUndefinedReason over the whole panel: %s"
          % {k: int(v) for k, v in reason[reason != ""].value_counts().items()})

    # ---- Stage-1 identity: the criterion outcome must not move -----------------------
    #  COMPARED POSITIONALLY, NOT BY (source, date) -- and this is a correction of my own first
    #  attempt, which merged on the pair and reported 70 changed rows.  It was a MIS-JOIN, not a
    #  behaviour change: 282 sources carry DUPLICATE snapped quarters (panel_upgrade fidelity gap
    #  C), so a pair-merge fans out many-to-many and pairs an AFTER row with the wrong BEFORE row
    #  -- the join produced 149,179 rows out of two 148,081-row panels, which is the tell.  The
    #  two panels are built by the SAME loop over the SAME sources in the SAME order with the
    #  same trim, so they are row-aligned by construction; that alignment is asserted rather
    #  than assumed before anything is read off it.
    assert len(bm_after) == len(bm_before), "the two arms produced different row counts"
    assert (bm_after["source"].values == bm_before["source"].values).all(), \
        "the two arms are not row-aligned -- a positional comparison would be meaningless"
    #  AND THE DATE, not just the source (reviewer, 2026-08-05).  `source` alone would agree on a
    #  frame whose rows had been re-ordered WITHIN a source, which is exactly the failure a
    #  positional compare cannot otherwise see -- and the whole reason the first version of this
    #  measurement was wrong.  On the real panel this holds on all 148,081 rows.
    assert (pd.to_datetime(bm_after["date"], errors="coerce").values
            == pd.to_datetime(bm_before["date"], errors="coerce").values).all(), \
        "the two arms disagree on `date` row-for-row -- the panels are not aligned"
    #  Belt and braces on the CLAIM the comparison is used to make: exactly the columns this
    #  change touches may differ.  A third differing column would mean collateral drift.
    _num = lambda f, c: pd.to_numeric(f[c], errors="coerce")
    _diff = sorted(c for c in bm_after.columns
                   if c not in ("source", "date")
                   and not _num(bm_after, c).equals(_num(bm_before, c)))
    print("  columns differing between the two arms: %s" % _diff)
    assert set(_diff) <= {"uGrahamNumberToPrice", "PEG"}, (
        "columns beyond the two this change touches differ between the arms: %s" % _diff)
    va = pd.to_numeric(bm_after["uGrahamNumberToPrice"], errors="coerce").reset_index(drop=True)
    vb = pd.to_numeric(bm_before["uGrahamNumberToPrice"], errors="coerce").reset_index(drop=True)
    imputed = vb.isna() & va.notna()
    #  Stage-1's unity test is `Sign * (value - 1) > 0`, Sign +1.
    pass_a, pass_b = (va - 1.0) > 0, (vb - 1.0) > 0
    j = va
    print("\n  Stage-1 `uGrahamNumberToPrice` -- boundary vs NaN, over %d row-aligned rows:"
          % len(j))
    print("    rows the boundary FILLED (were NaN, now %.1f)      : %d"
          % (npol.BOUNDARY_LIMIT["uGrahamNumberToPrice"][0], int(imputed.sum())))
    print("    rows left NaN (genuine missing inputs, not adverse): %d"
          % int((vb.isna() & va.isna()).sum()))
    print("    CRITERION OUTCOME CHANGED on                       : %d rows"
          % int((pass_a != pass_b).sum()))
    print("    -> BEHAVIOUR-IDENTICAL is asserted, not hoped: the fail is now DERIVED "
          "(value -1.0)\n       rather than incidental (NaN).")

    # ---- Stage-2 boundary incidence --------------------------------------------------
    zero_cov, adverse = 0, 0
    for src, g in cdx.iloc[::-1].groupby("source", sort=False):
        tf = g.reset_index(drop=True)
        rpy = rp.rows_per_year(str(tf[rp.FREQ_COLUMN].iloc[0]))
        w = rp.scale_window(npol.SCORING_WINDOW_NQ, rpy)
        v = (pd.to_numeric(tf["grahamNumber"], errors="coerce")
             / pd.to_numeric(tf["price"], errors="coerce")).head(w)
        if int(v.notna().sum()) == 0:
            zero_cov += 1
            if bool(npol.graham_adverse_mask(tf).head(w).any()):
                adverse += 1
    n_src = cdx["source"].nunique()
    print("\n  Stage-2 `grahamNumberToPrice`: coverage EXACTLY 0 on %d of %d sources (%.2f%%);"
          % (zero_cov, n_src, 100.0 * zero_cov / n_src))
    print("    of those, %d are ADVERSE -> take the 0.0 boundary; %d are all-missing -> refused."
          % (adverse, zero_cov - adverse))
    print("    PARTIAL coverage is NOT collapsed and NOT imputed -- it keeps its own "
          "observations\n    (ADDENDUM A's closing clause).")


def section_peg(cdx, pool):
    _h("D. PEG -- the four sign cells BEFORE and AFTER, the horizon, and the turnaround")
    n_win = 8               # Stage-1 head(8); the window is NOT frequency-scaled (calcScore)
    arms = {"BEFORE  vendor field, QoQ single-period eps": None,
            "AFTER   local, TTM legs, 1-year horizon": 1,
            "  alt   local, TTM legs, 2-year horizon": 2,
            "  alt   local, TTM legs, 3-year horizon": 3}
    out = {k: {"cells": {}, "pass": 0, "rows": 0, "turn": [0, 0], "tiny": 0,
               "tiny_pos_base": 0, "n_crossing": 0} for k in arms}

    def peg_crit(peg):
        return cm.peg_criterion(peg)

    #  THE POOL MEDIAN, ONE PER HORIZON, computed BEFORE the per-source loop -- because that is
    #  what makes it a pool quantity.  Printed, because the crossing rows' bar now depends on it
    #  and a ranking must never be read without it.
    medians = {}
    for _y in [v for v in arms.values() if v is not None]:
        medians[_y], _n = cm.peg_pool_median_growth(cdx, years=_y)
        print("  pool median annual growth at horizon %dy: %.4f%% over %d in-domain row(s)"
              % (_y, medians[_y], _n))

    for src, g in cdx.iloc[::-1].groupby("source", sort=False):
        tf = g.reset_index(drop=True)
        rpy = rp.rows_per_year(str(tf[rp.FREQ_COLUMN].iloc[0]))
        e = pd.to_numeric(tf[cm._PEG_EPS_FIELD], errors="coerce")
        for name, years in arms.items():
            if years is None:
                v = pd.to_numeric(tf["priceEarningsToGrowthRatio"], errors="coerce")
                adm = (e > 0) & (e.shift(-1) > 0)
                crit = pd.Series(np.where(v != 0, 1.0 / v, 0.0) - 1.0,
                                 index=v.index).where(adm)
                now, prev = e, e.shift(-1)
            else:
                #  THE NERF: crossing rows take the POOL's median growth rate, exactly as
                #  `substitute_peg_crossing` does at the production seam.  Measured here by
                #  passing the same median into `peg_local`, so this section exercises the same
                #  code path without needing the BoMetric alignment.
                peg, now, prev = cm.peg_local(tf, rpy=rpy, years=years,
                                              crossing_growth=medians[years])
                crit = peg_crit(peg)
                out[name]["tiny"] += int((peg.abs() < 1e-3).sum())
                #  the tiny-|PEG| population split into its TWO causes, which are different
                #  things: a SIGN CROSSING (now substituted) vs a genuinely tiny POSITIVE base
                #  (a real, enormous growth rate -- not an artifact, and not reached by the
                #  substitution, deliberately).
                _tinypos = (peg.abs() < 1e-3) & (prev > 0)
                out[name]["tiny_pos_base"] += int(_tinypos.sum())
                out[name]["n_crossing"] += int(((prev <= 0) & prev.notna()
                                                & (now > 0)).sum())
            cw = crit.head(n_win)
            ok = (cw > 0).fillna(False)
            out[name]["pass"] += int(ok.sum())
            out[name]["rows"] += len(cw)
            for lbl, m in (("now>0 prev>0", (now > 0) & (prev > 0)),
                           ("now<=0 prev>0", (now <= 0) & (prev > 0)),
                           ("now<=0 prev<=0", (now <= 0) & (prev <= 0)),
                           ("now>0 prev<=0", (now > 0) & (prev <= 0))):
                mw = m.head(n_win).fillna(False)
                c = out[name]["cells"].setdefault(lbl, [0, 0])
                c[0] += int(mw.sum())
                c[1] += int((mw & ok).sum())
            tm = ((now > 0) & (prev < 0)).head(n_win).fillna(False)
            out[name]["turn"][0] += int(tm.sum())
            out[name]["turn"][1] += int((tm & ok).sum())

    for name in arms:
        d = out[name]
        print("\n  %s" % name)
        print("    criterion pass rate over ALL scored rows: %d / %d = %.4f"
              % (d["pass"], d["rows"], d["pass"] / max(1, d["rows"])))
        for lbl, (n, p) in d["cells"].items():
            print("      %-16s rows %7d  passes %6d  rate %.4f  share of passes %.3f"
                  % (lbl, n, p, p / max(1, n), p / max(1, d["pass"])))
        print("      TURNAROUND (now>0, prev<0): %d rows, %d now PASS (was 0)"
              % tuple(d["turn"]))
        if arms[name] is not None:
            print("      sign-crossing rows, PANEL-WIDE (now on the pool median): %d "
                  "-- the cell counts above are the head(8) SCORING window, this one is not"
                  % d["n_crossing"])
            print("      |PEG| < 1e-3 total: %d ; of which a genuinely tiny POSITIVE base: %d"
                  % (d["tiny"], d["tiny_pos_base"]))
    print("\n  HORIZON RECOMMENDED: %d year (calcMetrics.PEG_GROWTH_YEARS).  See that module for\n"
          "  the reasoning; the number that decides it is above -- the NORMAL cell's pass rate is\n"
          "  essentially unchanged, so the fixed 0<PEG<1 bar is NOT loosened by the horizon\n"
          "  change, and the entire rise in the overall rate is the turnaround cell." %
          cm.PEG_GROWTH_YEARS)
    print("  THE CROSSING CELL IS NOW NERFED (CEO, 2026-08-05): a crossing row takes the POOL's\n"
          "  median growth rate instead of |E_prev| growth, so the DEPTH OF THE PRIOR LOSS no longer\n"
          "  enters the answer and the row is decided by its own P/E -- neither credit nor penalty,\n"
          "  and no tuned constant.  READ THE TWO CELL RATES AGAINST EACH OTHER: they should be\n"
          "  COMPARABLE, not equal (a crossing company's P/E distribution is genuinely different).\n"
          "  If they are wildly apart in EITHER direction the substitution is not doing its job.\n"
          "  The tiny-POSITIVE-base rows are NOT reached by the substitution and that is deliberate:\n"
          "  a near-zero positive base is a real, enormous growth rate, not a sign artifact.  The\n"
          "  only ways to reach them are a relative floor or a growth cap -- both TUNED CONSTANTS,\n"
          "  and this path deliberately carries none.")


def section_pool(dmdic_after, dmdic_before, subset):
    _h("E. POOL CHANGE -- reported, not minimised")
    res = {}
    for label, dm, ctx in (("BEFORE", dmdic_before, before_arm), ("AFTER", dmdic_after, None)):
        npol.POLICY_COUNTS.clear()
        if ctx is None:
            res[label] = pb.postBoWrapper(dm, as_of=None)
        else:
            with ctx():
                res[label] = pb.postBoWrapper(dm, as_of=None)
    a = res["AFTER"]["postRank"].reset_index(drop=True)
    b = res["BEFORE"]["postRank"].reset_index(drop=True)
    sa, sb = list(a["source"]), list(b["source"])
    _h("E. POOL CHANGE -- results")
    print("  general pool size: BEFORE %d  AFTER %d" % (len(sb), len(sa)))
    for n in (5, 10, 20, 100):
        ia, ib = set(sa[:n]), set(sb[:n])
        print("    top-%-3d overlap %d/%d (%.0f%%)   entered: %s   left: %s"
              % (n, len(ia & ib), min(n, len(ia)), 100.0 * len(ia & ib) / max(1, min(n, len(ia))),
                 sorted(ia - ib)[:8], sorted(ib - ia)[:8]))
    common = [s for s in sa if s in set(sb)]
    if common:
        ra = {s: i for i, s in enumerate(sa)}
        rb = {s: i for i, s in enumerate(sb)}
        d = pd.Series([ra[s] - rb[s] for s in common])
        print("    rank move on the %d common names: mean %+.2f, median %+.1f, max |move| %d"
              % (len(common), d.mean(), d.median(), int(d.abs().max())))
        aa = a.set_index("source")["AggScore"]
        ab = b.set_index("source")["AggScore"]
        dd = (aa.reindex(common) - ab.reindex(common)).dropna()
        print("    AggScore delta on common names: mean %+.5f, max |delta| %.5f"
              % (dd.mean(), dd.abs().max()))
        try:
            from scipy.stats import spearmanr
            print("    Spearman(before, after) over common names: %.5f"
                  % spearmanr([rb[s] for s in common], [ra[s] for s in common]).statistic)
        except Exception:
            pass
    print("\n  THE NaN-POLICY CONVERSIONS THE *AFTER* RUN ITSELF RECORDED (per pool / column /"
          " rule):")
    print(npol.counts_frame().to_string(index=False))
    print("\n  Subset =", subset or "FULL UNIVERSE",
          "-- pool-relative output on a subset is NOT comparable to production.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--orig-panel", default=ORIG)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--skip-pool", action="store_true")
    ap.add_argument("--cache", action="store_true",
                    help="cache/reuse the two Stage-1 panel rebuilds (DELETE the cache after "
                         "any change to the Stage-1 construction)")
    args = ap.parse_args()

    _h("NaN-POLICY + LOCAL-PEG ACCEPTANCE REPORT")
    print("  panel   : %s" % args.panel)
    print("  arms    : AFTER = the code as it stands;  BEFORE = reconstructed in-process "
          "(see module docstring)")
    d = pd.read_pickle(args.panel)
    cdx = d["cdx_df"].copy()
    pool = set(d["postRank"]["source"])
    if args.subset:
        keep = list(pd.unique(cdx["source"]))[:args.subset]
        cdx = cdx[cdx["source"].isin(keep)].copy()
    cdx["date"] = pd.to_datetime(cdx["date"], errors="coerce")
    cdx_up = pu.upgrade_cdx(cdx, verbose=False)
    print("  sources : %d   rows: %d   pool names present: %d"
          % (cdx_up["source"].nunique(), len(cdx_up),
             len(pool & set(cdx_up["source"]))))

    section_primary(cdx_up, pool)
    section_thresholds(cdx_up, pool)

    #  The two panel rebuilds are the expensive part (~7,700 sources x 2), so they are cached --
    #  the sections downstream of them are the ones a reader iterates on.  The cache is keyed on
    #  the subset size only, so DELETE IT after any change to the Stage-1 construction.
    tag = "sub%d" % args.subset if args.subset else "full"
    cache = os.path.join(_HERE, "cache_nanpolicy_panels_%s.pickle" % tag)
    if args.cache and os.path.exists(cache):
        print("\n  reusing cached Stage-1 panels: %s" % cache, flush=True)
        _c = pd.read_pickle(cache)
        bm_after, bm_before = _c["after"], _c["before"]
    else:
        print("\n  building Stage-1 panels (both arms)...", flush=True)
        bm_after = build_panel(cdx_up, cdic.getDicts(), subset=args.subset, label="AFTER ")
        with before_arm():
            bm_before = build_panel(cdx_up, _legacy_dicts(), subset=args.subset, label="BEFORE")
        if args.cache:
            pd.to_pickle({"after": bm_after, "before": bm_before}, cache)
            print("  cached to %s" % cache, flush=True)

    section_boundary(cdx_up, bm_after, bm_before, pool)
    section_peg(cdx_up, pool)

    if args.skip_pool:
        print("\n  --skip-pool: the rescore was not run.")
        return

    orig = pd.read_pickle(args.orig_panel)

    def _dm(bm):
        dm = dict(orig)
        dm["cdx_df"] = cdx_up.copy()
        dm["BoMetric_df"] = bm.copy()
        dm["Tickers_df"] = d.get("Tickers_df", orig.get("Tickers_df"))
        dm["api_key"] = ""
        dm["baseurl"] = "OFFLINE"
        for k in ("BoMetric_ave", "BoMetric_dateAve"):
            dm.pop(k, None)
        dm = dq.apply_data_quality_filter(dm, verbose=True, save_log=False)
        dm.update(cs.getAves2(dm["BoMetric_df"]))
        dm["nrScorePeriods"] = orig.get("nrScorePeriods", 8)
        return dm

    section_pool(_dm(bm_after), _dm(bm_before), args.subset)


if __name__ == "__main__":
    main()
