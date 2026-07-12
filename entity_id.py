"""
Entity-identity for recycled ticker symbols  (design s2A / Component 8, ruling R3).

A ticker reused by a NEW company after a prior entity delisted must be a DISTINCT
entity (TICKER, TICKER_2, ...).  This prevents two failures:
  (a) a new company wrongly excluded/contaminated because a same-symbol predecessor
      died (the dead predecessor's delistedDate must NOT prune the newcomer);
  (b) a price/fundamental series bleeding across the boundary (the BBBY trap: dead
      BBBY returned a continuous 2021->2026 series with live 2026 volume).

Detection rule (design s2A)
---------------------------
Group all records for a symbol; order by time.  Two records A (older, delistedDate
= d_A) and B (newer, ipoDate/first-trade = f_B) on the same symbol are DISTINCT iff
  1. chronological non-overlap:  f_B > d_A - tau   (tau ~ 0-30d admin tolerance), AND
  2. at least one discontinuity corroborant:
       (i)   price gap:   no trading in (d_A, f_B), then resumes;
       (ii)  fundamentals reset:  B's statements do not extend before f_B;
       (iii) identity change:  companyName differs materially (CIK if available).

If lifespans OVERLAP -> SAME entity (administrative duplicate) -> MERGE, do not
split.  PRECISION-FIRST for scoring: when signals conflict, prefer merging unless
the price gap is unambiguous (a false split fragments one entity's history).

entity_id
---------
The CURRENT/live occupant keeps the bare TICKER (so as_of=None and the live pipeline
are unchanged).  Prior distinct entities are TICKER_2, TICKER_3, ... in reverse
chronological order (most-recent prior = _2).  Each entity_id carries its own
(ipoDate, delistedDate, companyName, exchange).

F4 caveat (death-counting, design s9 part 3)
--------------------------------------------
Merge-on-overlap (precision-first) is right for SCORING but WRONG for
DEATH-COUNTING (a missed split hides a death).  `assign_entity_ids` therefore takes
a `mode` in {'merge','split'} so the caller can run BOTH thresholds and report the
death-rate as a BAND (split-first = floor).  Never use one precision direction for
both uses.
"""
import numpy as np
import pandas as pd

ADMIN_TOLERANCE_DAYS = 30


def _to_ts(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    ts = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(ts) else ts


def _distinct(rec_a, rec_b, mode="merge"):
    """Are older rec_a and newer rec_b distinct entities?  rec_* are dicts with
    ipoDate/delistedDate/companyName (and optional cik, has_price_gap,
    fundamentals_reset).  `mode` sets the precision direction (F4)."""
    d_a = _to_ts(rec_a.get("delistedDate"))
    f_b = _to_ts(rec_b.get("ipoDate")) or _to_ts(rec_b.get("firstTrade"))
    if d_a is None or f_b is None:
        # cannot establish non-overlap; precision-first -> merge (unless split mode
        # is told there IS a corroborated gap)
        if mode == "split" and rec_b.get("has_price_gap"):
            return True
        return False

    tau = pd.Timedelta(days=ADMIN_TOLERANCE_DAYS)
    non_overlap = f_b > (d_a - tau)
    if not non_overlap:
        return False   # lifespans overlap -> same entity, merge

    # corroborants
    price_gap = bool(rec_b.get("has_price_gap", False))
    fundamentals_reset = bool(rec_b.get("fundamentals_reset", False))
    name_a = (rec_a.get("companyName") or "").strip().lower()
    name_b = (rec_b.get("companyName") or "").strip().lower()
    identity_change = bool(name_a) and bool(name_b) and name_a != name_b
    cik_a, cik_b = rec_a.get("cik"), rec_b.get("cik")
    if cik_a and cik_b and cik_a != cik_b:
        identity_change = True

    corroborated = price_gap or fundamentals_reset or identity_change
    if mode == "split":
        # death-counting floor: non-overlap alone splits (maximise deaths found)
        return True
    # scoring / merge-first: require at least one corroborant (precision-first)
    return corroborated


def assign_entity_ids(records, mode="merge"):
    """Assign entity_id to a list of same-symbol records.

    records : list of dicts, each with at least 'symbol', 'ipoDate',
              'delistedDate', 'companyName', and optional 'exchange','cik',
              'has_price_gap','fundamentals_reset','is_live' (True for the current
              traded occupant).
    mode    : 'merge' (precision-first, for scoring) or 'split' (floor, death-count).

    Returns the records (copies) with an 'entity_id' key set.  The live occupant (or,
    absent an explicit live flag, the record with no delistedDate / the newest) keeps
    the bare symbol; prior distinct entities get _2, _3, ... reverse-chronologically.
    """
    if not records:
        return []
    recs = [dict(r) for r in records]
    symbol = recs[0]["symbol"]

    # order newest -> oldest by (delistedDate or ipoDate or far-future for live)
    def _sortkey(r):
        if r.get("is_live") or _to_ts(r.get("delistedDate")) is None:
            return pd.Timestamp.max
        return _to_ts(r.get("delistedDate")) or pd.Timestamp.min
    recs.sort(key=_sortkey, reverse=True)

    # walk newest->oldest, deciding at each adjacent boundary whether to split
    groups = [[recs[0]]]
    for older in recs[1:]:
        newer = groups[-1][-1]
        # newer is the more-recent record; older is candidate predecessor
        if _distinct(older, newer, mode=mode):
            groups.append([older])          # start a new (older) entity
        else:
            groups[-1].append(older)         # merge into current entity

    # groups[0] = most-recent (live) = bare symbol; groups[1] = _2; ...
    for gi, grp in enumerate(groups):
        eid = symbol if gi == 0 else f"{symbol}_{gi + 1}"
        for r in grp:
            r["entity_id"] = eid
    return recs


def alive_as_of(entity, D, unknown_ipo_alive=False):
    """Membership predicate (design Component 2): entity is alive at D iff
    ipoDate <= D and (delistedDate is null or delistedDate > D).  A new company is
    thus included on its OWN merits -- a dead predecessor's delistedDate never prunes
    it (fixes failure (a)).

    unknown_ipo_alive : how to treat a MISSING ipoDate.  Default False = precision-
        first: an entity whose birth date we do not know is NOT assumed alive at an
        arbitrary (possibly pre-existence) D -- otherwise a name that IPO'd after D
        with a blank ipoDate would leak into a historical universe (mild lookahead,
        LOW-1).  Pass True only when the caller has independently established the
        entity existed at D (e.g. a live occupant handled by build_universe's union)."""
    D = pd.Timestamp(D)
    ipo = _to_ts(entity.get("ipoDate"))
    dld = _to_ts(entity.get("delistedDate"))
    if ipo is None:
        if not unknown_ipo_alive:
            return False
    elif ipo > D:
        return False
    if dld is not None and dld <= D:
        return False
    return True
