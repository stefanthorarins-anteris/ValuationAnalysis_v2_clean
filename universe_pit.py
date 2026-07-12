"""
Survivorship-safe universe as a function of D and entity_id  (design s3 / Component
2).

Universe membership becomes a function of the as-of date D over a UNION of currently
-traded names and the delisted registry, keyed on entity_id (entity_id.py).

  * as_of=None  -> today: exact current behaviour.  The caller keeps the existing
    getData_gen live path (available-traded/list ∩ statement-symbol-lists, page-0
    prune) -- build_universe(None) simply returns the live symbols as bare
    entity_ids, so nothing about a live run changes.
  * as_of=D     -> PIT membership over the union of entities via alive_as_of:
    RETAIN entities alive-at-D even if dead now; DROP entities not yet IPO'd or
    already dead at D.  This inverts the live prune (which correctly drops dead names
    -- you cannot buy a dead stock TODAY -- but is wrong for a historical D).

This module is the PREDICATE + union; it deliberately does not fetch or read the
live endpoints (that stays in getData_gen).  It is pure and offline-testable.
"""
import pandas as pd

import entity_id as eid


def _ts(x):
    """Coerce to Timestamp or None (NaT-safe)."""
    if x is None:
        return None
    t = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(t) else t


def _live_records(live_symbols):
    """Normalise the live-occupant input to a list of dicts carrying at least
    'symbol' (and, when the caller passes a DataFrame, any 'entity_id'/'ipoDate'/
    'exchange' it knows).  A bare iterable of tickers becomes {'symbol': s}."""
    if isinstance(live_symbols, pd.DataFrame):
        return live_symbols.to_dict("records")
    return [{"symbol": s} for s in live_symbols]


def build_universe(live_symbols, registry=None, as_of=None, exchange_filter=None):
    """Return the set of entity_ids in the universe as-of `as_of`.

    live_symbols : iterable of currently-traded symbols (bare tickers), OR a
        DataFrame with a 'symbol' column (optionally 'entity_id'/'ipoDate'/
        'exchange').
    registry : optional DataFrame of delisted-registry entities with columns
        entity_id, symbol, ipoDate, delistedDate, exchange (design Component 3).
    as_of : None (=today, live behaviour) or a date D.
    exchange_filter : optional set of allowed exchanges.

    Returns a sorted list of entity_ids.

    as_of=D forms the UNION of (1) the delisted-registry entities alive_as_of D and
    (2) the live occupants (survivors).  Forming this union INSIDE the function is
    load-bearing: iterating the registry alone would silently drop every survivor
    whenever the caller passes a delisted-only registry -- REVERSE survivorship, the
    dominant original-bias class (design s0) inverted.  Survivors are alive today by
    construction, so they are never dropped from a historical universe; a live name
    is excluded at D only when we can POSITIVELY establish it had not yet IPO'd by D.
    """
    if as_of is None:
        # live: bare symbols only, exactly as today.  The page-0 prune etc. remain
        # the caller's job -- this returns the live occupants untouched.
        return sorted({r["symbol"] for r in _live_records(live_symbols)})

    D = pd.Timestamp(as_of)
    members = set()

    # (1) delisted-registry entities alive at D: RETAIN delisted-later survivors,
    #     DROP names dead-before-D or not-yet-IPO'd at D (alive_as_of predicate).
    reg_by_eid = {}
    if registry is not None and "entity_id" in registry.columns:
        reg = registry.copy()
        for col in ("ipoDate", "delistedDate"):
            if col in reg.columns:
                reg[col] = pd.to_datetime(reg[col], errors="coerce")
        for _, row in reg.iterrows():
            ent = row.to_dict()
            reg_by_eid[ent["entity_id"]] = ent
            if exchange_filter is not None and ent.get("exchange") not in exchange_filter:
                continue
            if eid.alive_as_of(ent, D):
                members.add(ent["entity_id"])

    # (2) UNION IN the live occupants (survivors).  The live occupant keeps the bare
    #     symbol as its entity_id (entity_id.py), so map symbol -> bare entity_id and
    #     borrow ipoDate/exchange from the registry row when the live input omits it.
    for rec in _live_records(live_symbols):
        sym = rec["symbol"]
        ent_id = rec.get("entity_id", sym)
        ref = reg_by_eid.get(ent_id, {})
        exch = rec.get("exchange", ref.get("exchange"))
        if exchange_filter is not None and exch is not None and exch not in exchange_filter:
            continue
        ipo = _ts(rec.get("ipoDate")) or _ts(ref.get("ipoDate"))
        if ipo is not None and ipo > D:
            # positively not-yet-public at D -> genuinely absent, not a survivor drop
            continue
        members.add(ent_id)

    return sorted(members)
