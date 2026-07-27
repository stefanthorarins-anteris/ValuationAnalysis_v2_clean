"""
Dead-name MERGE into the scored universe  (survivorship-clean scoring, OFFLINE).

PURPOSE
-------
The scoring pickle (`Boresults_dic-...`) is SURVIVOR-ONLY: its `cdx_df` /
`BoMetric_df` contain only names still traded at the snapshot date, so a name that
delisted before the snapshot can never enter the PIT top-20 -- survivorship bias on
the project's core axis.  The delisted-fundamentals pickle (`dead_fundamentals_*.pickle`,
7,194 entities) carries the raw FMP statement frames (km/fr/inc/bs/cf) for those dead
names.  This module transforms each dead entity's raw frames into the SAME
`cdx_df` / `BoMetric_df` schema the scorer consumes -- reusing the PRODUCTION transform
(getData_fmp.fillPreReqdf + calcMetrics + createDicts dicts, AND the production
fixAfterGetData/forceNumOnDf inf-scrub) so dead names are scored apples-to-apples with
survivors -- and unions them into the scoring frames under survivorship-safe
point-in-time (PIT) membership (universe_pit.build_universe / entity_id.alive_as_of).

NO network I/O.  Pure function of the two pickles + the delisted registry CSV.

LOOKAHEAD POLICY (CEO-SETTLED: symmetric, option A)
---------------------------------------------------
A PIT score as-of D must use only statements available at D.  The LIVE frames carry no
fillingDate/acceptedDate; the dead pickle's fillingDate is a lag-0 placeholder
(`filing_date_source == 'fixed_lag'` for all 7,194 entities).  The SETTLED decision is
the SYMMETRIC one: dead rows are stamped to the quarter START by the SAME production
`utils.setDatesToQuarterly` as live, and the scorer slices BOTH cohorts by the SAME
`date <= D` cut (stage2_pit.reproduce_pit_top).  There is therefore NO per-cohort
availability switch in this module -- that would be inert decorative machinery implying
a guarantee it does not deliver.  If delisted_ingest ever emits REAL dead filing dates,
the asymmetry MUST be re-litigated explicitly (do not silently start using them here).

WHAT IS DECIDED HERE
--------------------
  * JOIN KEY = entity_id (registry `entity_id`; == bare symbol except recycled
    tickers).  Dead rows enter with source = entity_id.
  * PIT MEMBERSHIP = universe_pit.build_universe(as_of=D, registry) -- a dead name is
    in the as-of-D universe only if alive_as_of D (ipoDate<=D<delistedDate).
  * EXCHANGE SCOPE = the merged universe is exchange-matched to the na1_only baseline:
    {NA1 live survivors} UNION {NA1 dead-but-alive-at-D}, nothing else, so a top-20
    delta reflects SURVIVORSHIP alone, not universe expansion (see pit_universe).
  * COLLISION handling: 273 dead entity_ids also appear as LIVE sources (names that
    delisted 2025-26, still in the Jan-2026 survivor snapshot -- SAME entity, not
    recycled tickers).  Default `collision='prefer_live'` skips the dead copy.
  * INF PARITY: dead frames go through gdg.fixAfterGetData (=> forceNumOnDf, inf->NaN,
    getData_gen.py:317) BEFORE the concat, exactly as every live row does, so a
    zero-denominator inf in a distressed dead name cannot corrupt the Stage-1 pool
    median (getAves2 does not scrub inf, calcScore.py:137).

DRIFT NOTE (for the reviewer): the per-entity build loop below is a faithful replica
of getData_fmp.get_fundamentals_fmp:53-146 (the ONLY duplication -- all arithmetic
still lives in calcMetrics/createDicts/getData_gen, called here).  It is duplicated
rather than extracted because getData_fmp.py is imported by a file under concurrent
edit; a follow-up could extract getData_fmp `_build_entity_frames` and have both
call it.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calcMetrics as cm
import reporting_period as rp
import createDicts as cdic
import getData_gen as gdg
import utils as utils
import entity_id as eid
import universe_pit as up

# NA1 exchange set -- the na1_only scoring baseline (stage2_pit.NA1_EXCHANGES).
# Duplicated as a module constant to avoid importing stage2_pit (which pulls in
# postBoRank -> requests/matplotlib) at merge-module import time; kept in lockstep.
NA1_EXCHANGES = ("NYSE", "NASDAQ", "TSX")


# --------------------------------------------------------------------------- #
#  Per-entity transform  (replica of get_fundamentals_fmp:53-146, offline)    #
# --------------------------------------------------------------------------- #
_ID_COLS = ("date", "symbol")


def _floatify(df):
    """Coerce every non-identifier column to numeric float -- matching the LIVE frames'
    representation.  Live statement frames arrive float-typed; the dead pickle stores
    raw FMP frames whose numeric columns can be int64 OR object-with-python-ints (e.g.
    priceEarningsToGrowthRatio carries an exact 0 as int64 for PMA/MSW and as an object
    0 for CDIX).  production calcMetrics.calc_special computes 1/priceEarningsToGrowthRatio
    BEFORE np.where masks it; on an int/object 0 pandas takes a Python scalar path and
    raises ZeroDivisionError, whereas float 0.0 yields inf and IS masked.  Coercing to
    float (int 0 -> 0.0, object 0 -> 0.0) reproduces the live path exactly.  This is a
    DTYPE fix applied BEFORE the calc -- deliberately distinct from (and in addition to)
    the forceNumOnDf inf->NaN scrub applied AFTER concat.  `date`/`symbol` are preserved
    (date is the cross-statement join key in fillPreReqdf); other string columns
    (cik/period/link/...) are not consumed by the transform and coerce harmlessly.
    """
    d = df.copy()
    for c in d.columns:
        if c in _ID_COLS:
            continue
        # to_numeric alone PRESERVES int64 for a pure-integer column; force float64 so
        # int 0 -> 0.0 (the numpy inf path), matching live exactly.
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    return d


def _build_entity_frames(entity, source, cdx_cols, bm_cols, n=1):
    """Transform one dead entity's raw statements into (cdx_row_df, bm_row_df).

    entity : the dead-pickle value dict with 'km','fr','inc','bs','cf' DataFrames.
    source : the entity_id to stamp as `source` (join key).
    Mirrors the production per-ticker body EXACTLY (initTempMets -> fillPreReqdf ->
    calc loop -> tail(rpy) trim), calling the real calcMetrics/createDicts functions.
    Returns (None, None) if the price gate (checkIfValidFS) fails, as production does.
    NOTE: the inf->NaN scrub (fixAfterGetData/forceNumOnDf) is applied ONCE on the
    concatenated frames in dead_to_scoring_frames, exactly as production applies it
    once at the end of the ingest loop (getData_fmp.py:146).
    """
    km, fr, inc, bs, cf = entity["km"], entity["fr"], entity["inc"], entity["bs"], entity["cf"]
    if any(not isinstance(x, pd.DataFrame) or x.empty for x in (km, fr, inc, bs, cf)):
        return None, None
    # DTYPE fix (int 0 -> 0.0) BEFORE the calc, so the whole transform behaves exactly
    # like the live float path -- see _floatify.
    km, fr, inc, bs, cf = (_floatify(x) for x in (km, fr, inc, bs, cf))

    (preReq_dict, _calc, BoMetric_base_dict, BoMetric_mean_dict,
     BoMetric_diff_dict, BoMetric_unity_dict, BoMetric_special_dict) = cdic.getDicts()

    # initTempMets(BoMetric_df.columns, cdx_df.columns, bs['date'], source)
    tempMetric_df = pd.DataFrame(columns=bm_cols)
    tempfund = pd.DataFrame(columns=cdx_cols)
    tempfund["date"] = bs["date"].values
    tempfund["source"] = source
    tempMetric_df["date"] = bs["date"].values
    tempMetric_df["source"] = source

    try:
        import getData_fmp as gdf
        tempfund, _hcy = gdf.fillPreReqdf(tempfund, preReq_dict, bs, inc, cf, km, fr)
    except Exception:
        return None, None
    tempMetric_df = utils.setDatesToQuarterly(tempMetric_df)

    if not gdg.checkIfValidFS(tempfund):
        return None, None

    tempdf = pd.DataFrame()
    tempdf["date"] = tempfund["date"]
    # LOCKSTEP with the live ingest (review H4, 2026-07-25): this loop is a copy of
    # getData_fmp's Stage-1 metric construction, so it must carry the SAME per-source
    # rows-per-year branching and the SAME Stage-1 flow-scale correction.  Without them
    # the dead-merged panel was a HYBRID and the PIT beat-rate it feeds was
    # non-comparable to live for every semi-annual name.
    # READ the one classification stamped by the live fillPreReqdf (review item 9) --
    # never re-derive it from the already-SNAPPED date column.
    _rpy = rp.rows_per_year(
        tempfund[rp.FREQ_COLUMN].iloc[0] if rp.FREQ_COLUMN in tempfund.columns
        else rp.UNKNOWN)
    ratioOpCalcDicts = {**BoMetric_base_dict, **BoMetric_mean_dict,
                        **BoMetric_unity_dict, **BoMetric_diff_dict}
    for key in ratioOpCalcDicts:
        restr = key
        strUp = ratioOpCalcDicts[key]["Upper"]
        strDn = ratioOpCalcDicts[key]["Lower"]
        tf = cm.calc_simpleRatio(tempfund, strUp, strDn)
        _ff = rp.stage1_flow_factor(key, _rpy)
        if _ff != 1.0:
            tf = [(v * _ff) if v is not None else v for v in tf]
        if key in BoMetric_base_dict:
            tempMetric_df[restr] = tf
        if key in BoMetric_mean_dict:
            tempMetric_df["m" + restr[0].upper() + restr[1:]] = tf
        if key in BoMetric_unity_dict:
            tempMetric_df["u" + restr[0].upper() + restr[1:]] = tf
        if key in BoMetric_diff_dict:
            tempdf["forDiff"] = tf
            tf = cm.calc_diff(tempdf, "forDiff", n, rpy=_rpy)
            tempMetric_df["d" + restr[0].upper() + restr[1:]] = tf
    for key1 in BoMetric_special_dict.keys():
        tempMetric_df[key1] = cm.calc_special(tempfund, key1, n, rpy=_rpy)

    tempMetric_df_trimmed = tempMetric_df.drop(tempMetric_df.tail(_rpy).index)
    return tempfund, tempMetric_df_trimmed


# --------------------------------------------------------------------------- #
#  Registry + universe helpers                                                #
# --------------------------------------------------------------------------- #
def load_registry(path):
    """Load the delisted registry CSV (entity_id, symbol, ipoDate, delistedDate,
    exchange, ...).  Dates coerced to Timestamp."""
    reg = pd.read_csv(path)
    for c in ("ipoDate", "delistedDate"):
        if c in reg.columns:
            reg[c] = pd.to_datetime(reg[c], errors="coerce")
    return reg


def _live_na1(dmdic, exch):
    """Live survivors restricted to the `exch` exchange set, exactly as the na1_only
    scoring baseline does (Tickers_df.exchangeShortName in exch), intersected with the
    actual cdx sources so the set is scoring-meaningful."""
    tk = dmdic["Tickers_df"]
    live_syms = set(tk.loc[tk["exchangeShortName"].isin(exch), "symbol"])
    cdx_sources = set(dmdic["cdx_df"]["source"].dropna().unique())
    return live_syms & cdx_sources


def pit_universe(dmdic, registry, as_of, exchange_filter=None):
    """As-of-D universe, EXCHANGE-MATCHED to the na1_only baseline:

        {live survivors on `exch`}  UNION  {dead-registry entities on `exch` alive@D}

    `exchange_filter` defaults to the NA1 set (NYSE/NASDAQ/TSX) so the merged run scores
    the SAME exchange scope as the baseline -- the top-20 delta then isolates
    survivorship, not universe expansion.  The SAME `exch` set is applied to BOTH the
    live survivors (via Tickers_df) and the dead names (via build_universe's
    exchange_filter), closing the two-variables-at-once confound.

    as_of=None returns the live NA1 survivors unchanged (live invariant)."""
    exch = set(NA1_EXCHANGES) if exchange_filter is None else set(exchange_filter)
    live = _live_na1(dmdic, exch)
    if as_of is None:
        return sorted(live)
    # Dead side: registry entities alive@D on `exch`.  Empty live_symbols -> the union
    # inside build_universe adds nothing from live; the exchange_filter is applied to
    # the registry rows (which DO carry `exchange`).
    dead = set(up.build_universe([], registry=registry, as_of=as_of,
                                 exchange_filter=exch))
    return sorted(live | dead)


# --------------------------------------------------------------------------- #
#  The merge                                                                  #
# --------------------------------------------------------------------------- #
def _resolve_registry_row(reg_by_sym, sym):
    """Return (entity_id, in_registry, ambiguous) for a dead symbol.

    Recycled/duplicate symbols can have >1 registry row; pick DETERMINISTICALLY the
    most-recent delistedDate (matching the 'most-recent occupant keeps the bare symbol'
    convention, entity_id.py) and flag the ambiguity so it is never silent."""
    if reg_by_sym is None or sym not in reg_by_sym.index:
        return sym, False, False
    row = reg_by_sym.loc[sym]
    ambiguous = False
    if isinstance(row, pd.DataFrame):
        ambiguous = True
        if "delistedDate" in row.columns:
            row = row.sort_values("delistedDate").iloc[-1]
        else:
            row = row.iloc[0]
    ent = row.get("entity_id", sym)
    return (ent if isinstance(ent, str) and ent else sym), True, ambiguous


def dead_to_scoring_frames(dead, registry, cdx_cols, bm_cols,
                           entities=None, live_sources=None,
                           collision="prefer_live", n=1, verbose=False):
    """Build (cdx_dead, bm_dead) -- dead-name rows in cdx_df / BoMetric_df schema,
    inf-scrubbed to live parity via gdg.fixAfterGetData.

    dead        : the dead-fundamentals pickle dict {symbol -> entity dict}.
    registry    : delisted-registry DataFrame (for entity_id join + collision test).
    cdx_cols/bm_cols : the live-frame columns (schema template).
    entities    : optional iterable of dead symbols to build (default: all).
    live_sources: set of live `source` values (for collision detection).
    collision   : 'prefer_live' (skip dead copy of a source already live) |
                  'keep_both' (append with a '_dead' suffix on the entity_id) |
                  'prefer_dead' (append bare -- NOT recommended; can double-count).
    Returns (cdx_dead, bm_dead); build stats attached to .attrs['build_stats'].
    """
    reg_by_sym = registry.set_index("symbol") if "symbol" in registry.columns else None
    live_sources = set(live_sources or set())
    syms = list(entities) if entities is not None else list(dead.keys())

    cdx_parts, bm_parts = [], []
    built = skipped_collision = gate_fail = not_in_registry = ambiguous_registry = 0
    for sym in syms:
        entity = dead.get(sym)
        if entity is None:
            continue
        entity_id, in_reg, ambiguous = _resolve_registry_row(reg_by_sym, sym)
        if not in_reg:
            not_in_registry += 1
        if ambiguous:
            ambiguous_registry += 1

        source = entity_id
        if entity_id in live_sources:
            if collision == "prefer_live":
                skipped_collision += 1
                continue
            if collision == "keep_both":
                source = f"{entity_id}_dead"
            # prefer_dead -> keep bare source (may double-count; caller warned)

        cdx_row, bm_row = _build_entity_frames(entity, source, cdx_cols, bm_cols, n=n)
        if cdx_row is None:
            gate_fail += 1
            continue
        cdx_parts.append(cdx_row)
        bm_parts.append(bm_row)
        built += 1

    cdx_dead = (pd.concat(cdx_parts, ignore_index=True)
                if cdx_parts else pd.DataFrame(columns=cdx_cols))
    bm_dead = (pd.concat(bm_parts, ignore_index=True)
               if bm_parts else pd.DataFrame(columns=bm_cols))

    # CONFORM TO THE LIVE SCHEMA EXACTLY (ship-gate, 2026-07-27).  `cdx_cols`/`bm_cols` are
    # the schema TEMPLATE this function's contract promises to reproduce, but the frames are
    # built by calling the LIVE fillPreReqdf, which now stamps derived columns the template
    # may not have (`reportingFrequency`, `periodEndDate`, `grahamUndefinedReason`).  Without
    # reindexing, a merge against an OLDER live panel produced a frame with columns present
    # on the dead rows and absent on the live ones -- precisely the generation-mixing the
    # merge-content gate below refuses.  Reindex, and REPORT the difference rather than
    # silently dropping it, so a genuinely new live column shows up as a schema drift signal.
    _extra_cdx = [c for c in cdx_dead.columns if c not in cdx_cols]
    _extra_bm = [c for c in bm_dead.columns if c not in bm_cols]
    if _extra_cdx or _extra_bm:
        print('dead_merge: conforming dead frames to the live schema -- dropping columns the '
              'live template does not have (cdx: %s ; bm: %s). If the LIVE panel is current '
              'these should be empty; a non-empty list means the panel predates the live '
              'ingest.' % (_extra_cdx, _extra_bm), flush=True)
    cdx_dead = cdx_dead.reindex(columns=cdx_cols)
    bm_dead = bm_dead.reindex(columns=bm_cols)

    # INF->NaN parity with the live pipeline: run BOTH frames through the SAME
    # post-ingest fixup production applies once (getData_fmp.py:146).  Guards the
    # Stage-1 pool median against a zero-denominator inf in a distressed dead name.
    if not cdx_dead.empty or not bm_dead.empty:
        bm_dead, cdx_dead = gdg.fixAfterGetData(bm_dead, cdx_dead)

    stats = {"requested": len(syms), "built": built,
             "skipped_collision": skipped_collision, "gate_fail": gate_fail,
             "not_in_registry": not_in_registry,
             "ambiguous_registry": ambiguous_registry}
    cdx_dead.attrs["build_stats"] = stats
    bm_dead.attrs["build_stats"] = stats
    if verbose:
        print("dead_to_scoring_frames:", stats, flush=True)
    return cdx_dead, bm_dead


def merge_dead_into_dmdic(dmdic, dead, registry, as_of=None,
                          collision="prefer_live", exchange_filter=None,
                          entities=None, n=1, verbose=False):
    """Return a NEW dmdic whose cdx_df / BoMetric_df include the dead names, plus the
    exchange-matched as-of-D PIT union universe.

    as_of=None -> returns dmdic UNCHANGED (live invariant): no dead names, exactly the
    survivor-only behaviour of today.  Only a real D triggers the merge.

    exchange_filter -> exchange scope for BOTH cohorts (default NA1 = baseline match).

    Returns (new_dmdic, stats).  new_dmdic adds:
        'pit_universe'     : sorted entity_ids in the as-of-D universe (exchange-matched)
        'dead_merge_stats' : build + universe stats (incl. residual-survivorship drops)
    Live keys are shallow-copied; cdx_df / BoMetric_df are replaced with unioned copies.
    """
    if as_of is None:
        return dmdic, {"merged": False, "reason": "as_of=None (live invariant)"}

    cdx_cols = list(dmdic["cdx_df"].columns)
    bm_cols = list(dmdic["BoMetric_df"].columns)
    live_sources = set(dmdic["cdx_df"]["source"].dropna().unique())

    cdx_dead, bm_dead = dead_to_scoring_frames(
        dead, registry, cdx_cols, bm_cols, entities=entities,
        live_sources=live_sources, collision=collision, n=n, verbose=verbose)

    universe = pit_universe(dmdic, registry, as_of, exchange_filter=exchange_filter)

    # M-3: dead-pickle entities ABSENT from the registry cannot get PIT membership
    # (no delistedDate -> excluded from the universe) -> residual survivorship.  Count
    # and surface LOUDLY; never a silent drop.
    reg_syms = set(registry["symbol"]) if "symbol" in registry.columns else set()
    considered = set(entities) if entities is not None else set(dead.keys())
    dropped_no_registry = sorted(considered - reg_syms)

    new = dict(dmdic)  # shallow copy of the container
    new["cdx_df"] = pd.concat([dmdic["cdx_df"], cdx_dead], ignore_index=True)
    new["BoMetric_df"] = pd.concat([dmdic["BoMetric_df"], bm_dead], ignore_index=True)

    # MERGE-CONTENT GATE (ship-gate item 2, 2026-07-27).
    # The Stage-1 schema gate checks column PRESENCE; a merge can satisfy it and still be
    # scored-but-wrong, because the dead side is built by TODAY's code and the live side by
    # whatever built the panel.  Two measured cases on a real merge:
    #   * `CFOlessEarnings` (Tier S, w=1.0) 100.00% NaN on LIVE rows and 3.60% on dead --
    #     the column exists, so presence passes, and every live name fails a top-weight
    #     criterion on missing data;
    #   * `dAssetsToLongTermLiabilities` live rows hold the OLD quantity (totalAssets/LTD)
    #     scored with the NEW Sign=-1, i.e. the sign is inverted against the data.
    # So the check has to be on CONTENT, per column, per side of the merge -- the same shape
    # as the price-basis refusal, which is why it reuses that posture: refuse by default,
    # explicit override to proceed.
    # SCOPE = the ACTUAL Stage-1 criterion columns, derived from the same criterion dicts
    # calcScore's schema gate uses -- NOT every column in the frame.  Scanning all columns
    # made the gate over-broad and its message untrue: it refused a merge on
    # `uIncomeQuality`, a criterion RETIRED on 2026-07-26 (replaced by `CFOlessEarnings`),
    # which lingers in older panels and is read by nothing.  A retired column cannot cause
    # a scoring error, so refusing on it blocks a valid merge; only a column the scorer
    # actually reads can make the merged panel scored-wrong.  (Found by
    # test_skill_baseline::test_integration_determinism_and_ordering, 2026-07-27.)
    _b, _m, _d, _u, _s = cdic.getBaseMeanDiffUnitySpecialDicts()
    _crit_cols = set(
        list(_b)
        + ['m' + k[0].upper() + k[1:] for k in _m]
        + ['d' + k[0].upper() + k[1:] for k in _d]
        + ['u' + k[0].upper() + k[1:] for k in _u]
        + list(_s))
    _live_src = set(dmdic["BoMetric_df"]["source"].dropna().unique())
    _dead_src = set(bm_dead["source"].dropna().unique()) if len(bm_dead) else set()
    _bad = []
    if _dead_src:
        _bmm = new["BoMetric_df"]
        _is_live = _bmm["source"].isin(_live_src)
        _is_dead = _bmm["source"].isin(_dead_src)
        for _c in _bmm.columns:
            if _c in ("source", "date") or _c not in _crit_cols:
                continue
            _lv = pd.to_numeric(_bmm.loc[_is_live, _c], errors="coerce")
            _dv = pd.to_numeric(_bmm.loc[_is_dead, _c], errors="coerce")
            if len(_lv) == 0 or len(_dv) == 0:
                continue
            _lnan, _dnan = _lv.isna().mean(), _dv.isna().mean()
            # "present but empty on ONE side only" == the two sides were built by
            # different generations of the metric set.
            if _lnan > 0.99 and _dnan < 0.50:
                _bad.append((_c, "live %.2f%% NaN vs dead %.2f%% NaN" % (100 * _lnan,
                                                                        100 * _dnan)))
            elif _dnan > 0.99 and _lnan < 0.50:
                _bad.append((_c, "dead %.2f%% NaN vs live %.2f%% NaN" % (100 * _dnan,
                                                                        100 * _lnan)))
    if _bad:
        _bar = "!" * 78
        _msg = chr(10).join(
            ["", _bar,
             "!!! DEAD/LIVE MERGE CONTENT MISMATCH -- THE MERGED PANEL WOULD BE SCORED WRONG.",
             "!!! %d Stage-1 column(s) are populated on ONE side of the merge only, i.e. the"
             % len(_bad),
             "!!! live panel and the dead frames were built by DIFFERENT generations of the",
             "!!! metric set. Column presence passes; the CONTENT does not:"]
            + ["!!!   %-34s %s" % (c, why) for c, why in _bad[:12]]
            + ["!!! A Tier-S w=1.0 criterion that is all-NaN on the live side fails every live",
               "!!! name on missing data, and a renamed/inverted metric scores the OLD quantity",
               "!!! with the NEW sign. FIX: re-fetch the live panel with the current code.",
               _bar, ""])
        print(_msg, file=sys.stderr, flush=True)
        print(_msg, flush=True)
        if not os.environ.get("ALLOW_MERGE_CONTENT_MISMATCH"):
            raise SystemExit(
                "REFUSING to return a dead/live merged panel whose Stage-1 columns are "
                "populated on one side only (%d column(s): %s). Re-fetch the live panel, or "
                "set ALLOW_MERGE_CONTENT_MISMATCH=1 to proceed on a known-invalid basis."
                % (len(_bad), ", ".join(c for c, _ in _bad[:6])))
        print("ALLOW_MERGE_CONTENT_MISMATCH set: PROCEEDING on a known-invalid basis.",
              flush=True)
    new["pit_universe"] = universe

    stats = dict(cdx_dead.attrs.get("build_stats", {}))
    stats.update({
        "merged": True, "as_of": str(pd.Timestamp(as_of).date()),
        "universe_size": len(universe),
        "exchange_scope": sorted(set(NA1_EXCHANGES) if exchange_filter is None
                                 else set(exchange_filter)),
        "dead_dropped_not_in_registry": len(dropped_no_registry),
    })
    new["dead_merge_stats"] = stats
    if stats["dead_dropped_not_in_registry"]:
        print(f"WARNING [dead_merge M-3]: {stats['dead_dropped_not_in_registry']} dead "
              f"entities absent from the registry -> excluded from the PIT universe "
              f"(residual survivorship). e.g. {dropped_no_registry[:10]}", flush=True)
    if stats.get("ambiguous_registry"):
        print(f"NOTE [dead_merge]: {stats['ambiguous_registry']} dead symbols had >1 "
              f"registry row -> resolved by most-recent delistedDate (not silent).",
              flush=True)
    if verbose:
        print("merge_dead_into_dmdic:", stats, flush=True)
    return new, stats
