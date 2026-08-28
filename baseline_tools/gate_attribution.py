"""WHAT THE STAGE-1 SOLVENCY GATE ACTUALLY DID -- per anchor, per flag, on real returns.

WHY THIS MODULE EXISTS.  On 2026-08-27 the Stage-1 veto ran inside the backtest for the first
time and ejected 955-1,182 names per anchor; on the live pool it took 1,455 names to 902.  Not
one line of output anywhere in this tree said whether a single one of those ejections HELPED.
The run printed a beat-rate and a portfolio return on the VETOED basis and that was all, so the
gate could have been tightened, loosened or inverted and the only visible change would have
been the ejection count -- a measurement of the gate's SIZE, never of its EFFECT.

THE CEO'S NEXT DECISION IS THE WEIGHTS AND THE GATES, and he has been told the coverage problem
is not cheaply fixable and has ruled it last.  So the question this module answers is the one
the EXISTING coverage can still answer: of the names a flag ejected, what did they go on to do,
and would we have bought them anyway?

THE LOAD-BEARING COMPARISON IS THE SWAP, NOT THE EJECTION SET.  Most of what a gate ejects
never came near the top-20 and is irrelevant to a 20-name portfolio: ejecting a name ranked
900th changes nothing anyone would have bought.  What matters is the names the gate removed
that WOULD HAVE ENTERED THE SHIPPED TOP-20, and the names that took their slots.  Those two
sets are the whole of the gate's effect on an equal-weight top-20: every other pick is common
to both lists and cancels exactly.  So the readout is built around

    DROPPED  = un-vetoed top-20  MINUS  vetoed top-20      (the picks the gate cost)
    ADDED    = vetoed top-20     MINUS  un-vetoed top-20   (the picks the gate bought)

and the effect on the equal-weight return is (|DROPPED| / 20) * (mean ADDED - mean DROPPED).

A DROPPED NAME IS NOT AUTOMATICALLY A GATE KILL, and conflating the two would overstate the
gate.  The veto runs BEFORE the `head(topn_stage1)` cut, so an ejection PROMOTES the next name
into the Stage-1 pool -- and the Stage-2 normalisation is computed OVER that pool.  A name can
therefore leave the top-20 because the pool it was normalised against changed, without ever
having been ejected.  DROPPED is split against the veto's own `ejected` set so the two causes
are never added together.

A GATE CANNOT BE CREDITED FOR REMOVING A NAME WE CANNOT PRICE.  This is the rule that decides
whether the whole readout is honest, because the coverage that made the target clause
INDETERMINATE applies here unchanged and applies WORSE: ejected names are enriched in exactly
the companies whose prices stop -- that is most of what a solvency gate is for.  So every
cohort here is counted THREE WAYS and never two:

    measured-and-beat / measured-and-not / UNKNOWN

An unpriceable ejected name is UNKNOWN.  It is not a save, it is not a loss, and it is never
folded into a mean.  `measured` is `target_clauses.measured`, i.e. `status == 'ok'` -- both
legs priced at the chartered anchors -- so a stale substituted price cannot be counted as a
gate outcome any more than it can be counted as a target observation.

THE PER-FLAG ROLLUP IS THE WEAKEST THING HERE AND IT SAYS SO.  `stage1_veto.EJECT_MIN_FLAGS`
is 1, so a name is ejected the moment ANY flag fails and most ejected names fail several.  A
per-flag count of ejections therefore SUMS TO MORE THAN THE EJECTION TOTAL (at buy2021:
46+400+580+300+453 = 1,779 against 1,125 ejected) and is not a partition of anything.  Two
different counts are reported and they answer different questions:

    n_any     -- the flag fired on this name.  Inflated; sums past the total; useful only for
                 "how often does this flag have an opinion".
    n_solely  -- this flag is the ONLY one that fired, so REMOVING THE FLAG UN-EJECTS THE NAME.
                 This is the counterfactual count, and it is the one that answers "what happens
                 if I drop this flag" -- which is the CEO's actual question.

At top-20 depth both counts are small enough that the NAME-LEVEL TABLE is the real output and
the rollup is a convenience.  The module prints the names.

BOTH BASES ARE REPORTED SIDE BY SIDE, from `_basis_of`-style stamps carried on each ranking, so
a vetoed figure is never silently compared against an un-vetoed one -- the failure the
measurement-basis banner exists to prevent.

NO NETWORK, NO FETCH.  Everything is a function of two rankings and a price source.
"""

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import returns_core as rc
import target_clauses as tc

#  Graded at the CHARTERED depth and horizon, from the one definition, so this readout and the
#  target clause can never drift into describing different portfolios.
DEPTH_N = tc.CHARTERED_DEPTH
HORIZON_M = tc.CHARTERED_HORIZON_M
BEAT_THRESHOLD = 0.10

#  The name the CEO raised by hand, carried as a standing spotlight rather than a one-off
#  script: at buy2022 TRTN is ejected on `netDebtToEBITDA` and is the ONLY one of the 14
#  delisted picks the gate would have removed -- and arguably the healthiest of them (a
#  container lessor whose ~20x net-debt/EBITDA IS its industry's capital structure, positive
#  and rising net income and equity at the buy quarter).  Whatever the aggregate says, the
#  specific name is reported.
DEFAULT_SPOTLIGHT = ("TRTN",)


# --------------------------------------------------------------------------- #
#  Cohort arithmetic -- one place that decides what is counted and what is not #
# --------------------------------------------------------------------------- #
def cohort(names, buy, ev, price_source, bench, threshold=BEAT_THRESHOLD):
    """Equal-weight summary of an arbitrary set of picks, counted THREE WAYS.

    `names` is a SET OF PICKS, not a set of measurements: `n` is how many names are in it and
    `n_measured` is how many of them the price grid can value over [buy, ev].  The difference
    is the finding, exactly as it is in `target_clauses`.

    Returned keys:
      n, n_measured, n_unknown   the three-way split's denominator and its unknown bucket
      mean_return                equal-weight mean over the MEASURED names only.  NOT the
                                 cohort's return when n_unknown > 0, and labelled that way
                                 wherever it is printed.
      n_beat / n_below_zero      counts over the MEASURED names, each carrying n_measured
      names / measured / unknown the actual tickers, because at these n's the names ARE the
                                 output and an aggregate over four picks is decoration
      per_name                   {ticker: total_return} for the measured ones
    """
    names = list(dict.fromkeys(names))          # de-dup, order preserved
    out = {"n": len(names), "n_measured": 0, "n_unknown": len(names),
           "mean_return": float("nan"), "n_beat": 0, "n_below_zero": 0,
           "names": names, "measured": [], "unknown": list(names), "per_name": {}}
    if not names:
        out["n_unknown"] = 0
        return out
    rdf = rc.compute_returns(names, buy, ev, price_source)
    m = tc.measured(rdf)
    if not len(m):
        return out
    r = pd.to_numeric(m["total_return"], errors="coerce")
    keep = m[r.notna()]
    vals = pd.to_numeric(keep["total_return"], errors="coerce").astype(float)
    measured_names = list(keep["ticker"])
    out["measured"] = measured_names
    out["unknown"] = [t for t in names if t not in set(measured_names)]
    out["n_measured"] = len(measured_names)
    out["n_unknown"] = len(out["unknown"])
    out["mean_return"] = float(vals.mean())
    out["n_below_zero"] = int((vals < 0.0).sum())
    out["per_name"] = {t: float(v) for t, v in zip(measured_names, vals)}
    if bench == bench:
        out["n_beat"] = int(((vals - bench) >= threshold).sum())
    else:
        out["n_beat"] = 0
    return out


def _flip_to_match(dropped, added):
    """What the UNPRICEABLE dropped picks would have had to average for the swap to have been
    return-neutral.

    The idiom is `target_clauses.flip_return`'s and it is here for the same reason: a swap
    measured over 3 of 5 dropped names is not a verdict, but "the two we cannot price would
    have had to average +180% for the gate to have broken even" IS actionable, and a reader
    can judge it without being handed a false point estimate.  NaN when nothing is unknown
    (there is nothing to flip) or when neither side has a measurement to flip against.
    """
    if dropped["n_unknown"] == 0 or dropped["n"] == 0:
        return float("nan")
    #  The ADDED side must be measurable -- there is nothing to flip TOWARDS otherwise.  The
    #  DROPPED side does NOT: a cohort with nothing measured contributes S = 0, and "the one
    #  name we could not price would have had to return +X" is the most useful form this
    #  number takes, which is exactly the TRTN case.
    if added["n_measured"] == 0:
        return float("nan")
    #  Solve  (S_dropped + n_unknown * x) / n_dropped  ==  mean(added measured)
    s = ((dropped["mean_return"] * dropped["n_measured"])
         if dropped["n_measured"] else 0.0)
    return (added["mean_return"] * dropped["n"] - s) / dropped["n_unknown"]


# --------------------------------------------------------------------------- #
#  Per-anchor attribution                                                     #
# --------------------------------------------------------------------------- #
def attribute_anchor(wid, buy, ev, vetoed, unvetoed, price_source,
                     depth_n=DEPTH_N, threshold=BEAT_THRESHOLD):
    """One anchor: what the gate dropped, what it bought, and what each flag is answerable for.

    `vetoed` / `unvetoed` are the per-anchor dicts `depth_horizon_grid.rank_all_anchors`
    returns for the SAME anchor under `stage1_veto=True` / `False`.
    """
    v_top = list((vetoed.get("top20_deduped") or vetoed.get("ranking") or [])[:depth_n])
    u_top = list((unvetoed.get("top20_deduped") or unvetoed.get("ranking") or [])[:depth_n])
    vr = vetoed.get("stage1_veto") or {}
    ejected = set(vr.get("ejected") or [])
    ejected_flags = dict(vr.get("ejected_flags") or {})

    bench = rc.benchmark_return(price_source, buy, ev, require_exact=True)

    dropped_all = [t for t in u_top if t not in set(v_top)]
    added = [t for t in v_top if t not in set(u_top)]
    #  THE SPLIT THAT KEEPS THE GATE FROM BEING BLAMED (OR CREDITED) FOR THE POOL SHIFT.
    dropped_by_veto = [t for t in dropped_all if t in ejected]
    dropped_other = [t for t in dropped_all if t not in ejected]

    coh = dict(
        vetoed_top=cohort(v_top, buy, ev, price_source, bench, threshold),
        unvetoed_top=cohort(u_top, buy, ev, price_source, bench, threshold),
        dropped_by_veto=cohort(dropped_by_veto, buy, ev, price_source, bench, threshold),
        dropped_other=cohort(dropped_other, buy, ev, price_source, bench, threshold),
        added=cohort(added, buy, ev, price_source, bench, threshold),
    )

    #  ---- per-flag, over the TOP-20 KILLS only (the pool-wide counts are in the veto
    #  report and are a measure of the gate's size, not of its effect) ----
    by_flag = {}
    for t in dropped_by_veto:
        fl = list(ejected_flags.get(t) or [])
        for f in fl:
            e = by_flag.setdefault(f, {"n_any": 0, "n_solely": 0, "any": [], "solely": []})
            e["n_any"] += 1
            e["any"].append(t)
            if len(fl) == 1:
                e["n_solely"] += 1
                e["solely"].append(t)
    for f, e in by_flag.items():
        e["cohort_any"] = cohort(e["any"], buy, ev, price_source, bench, threshold)
        e["cohort_solely"] = cohort(e["solely"], buy, ev, price_source, bench, threshold)

    #  ---- THE FUNNEL: where a ~1,000-name ejection actually lands ----
    #  THE NUMBER THAT MAKES THE EJECTION COUNT LEGIBLE.  "1,014 of 2,148 ejected" is a
    #  measure of the gate's SIZE and it invites the reading that half the shortlist changed.
    #  It did not.  Stage-1 cuts to `topn_stage1` names before Stage-2 scores anything, so an
    #  ejection can only matter if it lands INSIDE that pool -- and a name that fails a
    #  solvency flag is much rarer among high-BoScore names than in the pool at large.  Three
    #  counts, narrowing, so the reader can see the gate's reach collapse:
    #      pool  ->  the names that reached Stage-2  ->  the shipped top-N
    u_pool = list(unvetoed.get("ranking") or [])
    v_pool = list(vetoed.get("ranking") or [])
    funnel = {
        "n_scored": int(vr.get("n_in") or 0),
        "n_ejected": int(vr.get("n_ejected") or 0),
        "n_unvetoed_pool": len(u_pool),
        "n_unvetoed_pool_ejected": len([t for t in u_pool if t in ejected]),
        "pool_overlap": len(set(u_pool) & set(v_pool)),
        "n_unvetoed_top_ejected": len([t for t in u_top if t in ejected]),
        "top_overlap": len(set(u_top) & set(v_top)),
    }
    #  Per flag, the same narrowing, on the COUNTERFACTUAL (`solely`) count at each level --
    #  because "drop this flag and N names come back" is the only per-flag number that
    #  answers a tuning question.
    solely_pool = _solely_counts(ejected_flags)
    flag_funnel = {}
    for f in sorted(set(list(solely_pool) + list((vr.get("by_flag") or {})))):
        in_pool = [t for t in u_pool
                   if t in ejected and list(ejected_flags.get(t) or []) == [f]]
        in_top = [t for t in u_top
                  if t in ejected and list(ejected_flags.get(t) or []) == [f]]
        flag_funnel[f] = {"any_pool_wide": int((vr.get("by_flag") or {}).get(f, 0)),
                          "solely_pool_wide": int(solely_pool.get(f, 0)),
                          "solely_in_scored_pool": len(in_pool),
                          "solely_in_top": len(in_top),
                          "names_in_scored_pool": in_pool}

    #  ---- the swap, which IS the gate's effect on an equal-weight top-20 ----
    k = len(dropped_all)
    swap = {"k": k,
            "effect_measured": (
                (float(k) / depth_n) * (coh["added"]["mean_return"]
                                        - coh["dropped_by_veto"]["mean_return"])
                if (coh["added"]["n_measured"] and coh["dropped_by_veto"]["n_measured"])
                else float("nan")),
            "flip_return": _flip_to_match(coh["dropped_by_veto"], coh["added"])}

    return {"wid": wid, "buy": buy, "eval": ev, "bench": bench,
            "basis_vetoed": vetoed.get("basis"), "basis_unvetoed": unvetoed.get("basis"),
            "n_ejected_pool": int(vr.get("n_ejected") or 0),
            "n_in_pool": int(vr.get("n_in") or 0),
            "pool_by_flag": dict(vr.get("by_flag") or {}),
            "pool_solely_by_flag": _solely_counts(ejected_flags),
            "v_top": v_top, "u_top": u_top,
            "dropped_all": dropped_all, "dropped_by_veto": dropped_by_veto,
            "dropped_other": dropped_other, "added": added,
            "ejected_flags": {t: list(ejected_flags.get(t) or []) for t in dropped_by_veto},
            "cohorts": coh, "by_flag": by_flag, "swap": swap,
            "funnel": funnel, "flag_funnel": flag_funnel,
            "depth_n": depth_n}


def _solely_counts(ejected_flags):
    """Pool-wide: how many ejections each flag is SOLELY responsible for.

    The counterfactual count -- remove the flag and these names come back.  `by_flag` in the
    veto report is the `n_any` version and sums past the ejection total, which makes it easy to
    read as "this flag ejected 580 names" when the honest sentence is "580 ejected names failed
    this flag, most of them failing others too".
    """
    out = {}
    for _s, fl in (ejected_flags or {}).items():
        fl = list(fl or [])
        if len(fl) == 1:
            out[fl[0]] = out.get(fl[0], 0) + 1
    return dict(sorted(out.items()))


def spotlight_rows(anchors, vetoed_by_wid, unvetoed_by_wid, price_source,
                   tickers=DEFAULT_SPOTLIGHT, depth_n=DEPTH_N, threshold=BEAT_THRESHOLD):
    """A named ticker's whole story at every anchor: ranked? ejected? on what? priced? return?

    A standing spotlight rather than an ad-hoc query because the aggregate at these n's cannot
    settle a case like TRTN, and the CEO asked about that name specifically.  A ticker that is
    nowhere near the pool still gets a row saying so -- absence is an answer.
    """
    rows = []
    for wid, buy, ev in anchors:
        v = vetoed_by_wid.get(wid) or {}
        u = unvetoed_by_wid.get(wid) or {}
        vr = v.get("stage1_veto") or {}
        ejected_flags = dict(vr.get("ejected_flags") or {})
        v_top = list((v.get("top20_deduped") or [])[:depth_n])
        u_top = list((u.get("top20_deduped") or [])[:depth_n])
        u_rank_full = list(u.get("ranking") or [])
        bench = rc.benchmark_return(price_source, buy, ev, require_exact=True)
        for t in tickers:
            c = cohort([t], buy, ev, price_source, bench, threshold)
            rows.append({
                "wid": wid, "buy": buy, "eval": ev, "ticker": t,
                "in_unvetoed_top20": t in u_top,
                "in_vetoed_top20": t in v_top,
                "unvetoed_rank": (u_rank_full.index(t) + 1) if t in u_rank_full else None,
                "ejected": t in (set(vr.get("ejected") or [])),
                "ejected_flags": list(ejected_flags.get(t) or []),
                "measured": c["n_measured"] == 1,
                "total_return": c["per_name"].get(t, float("nan")),
                "bench": bench,
            })
    return rows


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
def attribute(vetoed_by_wid, unvetoed_by_wid, price_source, anchors=None,
              depth_n=DEPTH_N, horizon_m=HORIZON_M, threshold=BEAT_THRESHOLD,
              spotlight=DEFAULT_SPOTLIGHT):
    """Attribution over every anchor present in BOTH passes with an eval anchor in the grid."""
    import depth_horizon_grid as dhg
    pairs = []
    for wid, buy in dhg.BUY_ANCHORS:
        if anchors is not None and wid not in set(anchors):
            continue
        if wid not in vetoed_by_wid or wid not in unvetoed_by_wid:
            continue
        if wid not in dhg.CLEAN_BUY_IDS:
            continue
        ei = dhg.ANCHOR_IDX[buy] + horizon_m // 12
        if ei >= len(dhg.ANCHORS):
            continue
        pairs.append((wid, buy, dhg.ANCHORS[ei]))
    per_anchor = [attribute_anchor(wid, buy, ev, vetoed_by_wid[wid], unvetoed_by_wid[wid],
                                   price_source, depth_n=depth_n, threshold=threshold)
                  for wid, buy, ev in pairs]
    return {"per_anchor": per_anchor, "horizon_m": horizon_m, "depth_n": depth_n,
            "threshold": threshold,
            "spotlight": spotlight_rows(pairs, vetoed_by_wid, unvetoed_by_wid, price_source,
                                        tickers=spotlight, depth_n=depth_n,
                                        threshold=threshold)}


# --------------------------------------------------------------------------- #
#  Report                                                                     #
# --------------------------------------------------------------------------- #
def _pct(x, w=9):
    return (f"{x*100:+.1f}%".rjust(w)) if x == x else "n/a".rjust(w)


def _three_way(c):
    """'2 beat / 1 lost / 3 UNKNOWN (of 6)' -- the sentence the whole module is built around."""
    return ("%d beat / %d below zero / %d UNKNOWN  (measured %d of %d)"
            % (c["n_beat"], c["n_below_zero"], c["n_unknown"], c["n_measured"], c["n"]))


def format_report(res):
    L = []
    P = L.append
    P("#" * 72)
    P("# STAGE-1 GATE ATTRIBUTION  --  what the solvency veto did to the SHIPPED top-%d"
      % res["depth_n"])
    P("#   horizon = %dmo   beat bar = +%.0fpp vs URTH" % (res["horizon_m"],
                                                           res["threshold"] * 100))
    P("#" * 72)
    P("  THE GATE'S EFFECT ON A TOP-%d IS THE SWAP AND NOTHING ELSE. Names on both lists" % res["depth_n"])
    P("  cancel; only the picks the gate REMOVED from the un-vetoed top-%d and the picks that"
      % res["depth_n"])
    P("  took their slots can move the portfolio. Ejections further down the pool are a")
    P("  measure of the gate's SIZE, not of its effect, and are reported separately.")
    P("  A GATE IS NOT CREDITED FOR REMOVING A NAME WE CANNOT PRICE. Every cohort is counted")
    P("  three ways -- beat / below zero / UNKNOWN -- and an unpriceable ejection stays UNKNOWN.")
    if not res["per_anchor"]:
        P("")
        P("  NO ANCHOR has both a vetoed and an un-vetoed ranking with a %dmo eval leg --"
          % res["horizon_m"])
        P("  the attribution is UNMEASURED this run (not 'the gate did nothing').")
        return "\n".join(L)

    for a in res["per_anchor"]:
        c = a["cohorts"]
        P("")
        P("=" * 72)
        P("  %s   %s -> %s   benchmark %s" % (a["wid"], a["buy"], a["eval"], _pct(a["bench"]).strip()))
        P("    BASIS vetoed   : %s" % a["basis_vetoed"])
        P("    BASIS un-vetoed: %s" % a["basis_unvetoed"])
        P("")
        P("  --- FUNNEL: where the ejections actually land ---")
        f = a["funnel"]
        P("    %-46s %5d" % ("names Stage-1 scored", f["n_scored"]))
        P("    %-46s %5d   (%.0f%% of the pool)"
          % ("ejected by the veto", f["n_ejected"],
             100.0 * f["n_ejected"] / f["n_scored"] if f["n_scored"] else float("nan")))
        P("    %-46s %5d   of %d"
          % ("...that were in the un-vetoed Stage-2 pool", f["n_unvetoed_pool_ejected"],
             f["n_unvetoed_pool"]))
        P("    %-46s %5d   of %d"
          % ("...that were in the un-vetoed top-%d" % a["depth_n"],
             f["n_unvetoed_top_ejected"], a["depth_n"]))
        P("    %-46s %5d   of %d"
          % ("top-%d slots that changed at all" % a["depth_n"],
             a["depth_n"] - f["top_overlap"], a["depth_n"]))
        P("    THE EJECTION COUNT IS A MEASURE OF THE GATE'S SIZE, NOT ITS EFFECT. Stage-1 cuts")
        P("    to the top pool before Stage-2 scores anything, so an ejection can only matter if")
        P("    it lands inside that pool -- and a name that fails a solvency flag is far rarer")
        P("    among high-BoScore names than in the pool at large.")
        P("")
        P("  --- THE TWO TOP-%d, SIDE BY SIDE ---" % a["depth_n"])
        P("    %-14s %-10s %-10s %s" % ("list", "mean(meas)", "n_meas/n", "three-way"))
        for lbl, key in (("un-vetoed", "unvetoed_top"), ("VETOED (shipped)", "vetoed_top")):
            x = c[key]
            P("    %-16s %s   %2d/%-3d  %s"
              % (lbl, _pct(x["mean_return"]), x["n_measured"], x["n"], _three_way(x)))
        P("    the two means are over DIFFERENT measured subsets, so their difference is not")
        P("    the gate's effect. The swap below is.")
        P("")
        P("  --- THE SWAP: what the gate removed, and what replaced it ---")
        P("    %d of %d picks changed." % (a["swap"]["k"], a["depth_n"]))
        P("    DROPPED by the veto  (%d): %s"
          % (len(a["dropped_by_veto"]), ", ".join(a["dropped_by_veto"]) or "-"))
        if a["dropped_other"]:
            P("    DROPPED for other reasons (%d): %s"
              % (len(a["dropped_other"]), ", ".join(a["dropped_other"])))
            P("      ^ NOT gate kills. The veto runs before the Stage-1 head() cut, so an")
            P("        ejection promotes a name into the pool the Stage-2 normalisation is")
            P("        computed over; these left the top-%d because that pool moved."
              % a["depth_n"])
        P("    ADDED by the veto    (%d): %s"
          % (len(a["added"]), ", ".join(a["added"]) or "-"))
        P("")
        P("    %-22s %s   %s" % ("cohort", "mean(meas)", "three-way"))
        for lbl, key in (("dropped (gate kills)", "dropped_by_veto"),
                         ("dropped (pool shift)", "dropped_other"),
                         ("added (replacements)", "added")):
            x = c[key]
            if x["n"] == 0:
                P("    %-22s %s   -" % (lbl, "n/a".rjust(9)))
                continue
            P("    %-22s %s   %s" % (lbl, _pct(x["mean_return"]), _three_way(x)))
        sw = a["swap"]
        P("")
        P("    EFFECT on the equal-weight top-%d, over the MEASURED members of each side:"
          % a["depth_n"])
        P("      (%d/%d) x (mean added - mean dropped) = %s"
          % (sw["k"], a["depth_n"], _pct(sw["effect_measured"]).strip()))
        if c["dropped_by_veto"]["n_unknown"]:
            P("      NOT A VERDICT: %d of the %d dropped picks are unpriceable. They would have"
              % (c["dropped_by_veto"]["n_unknown"], c["dropped_by_veto"]["n"]))
            P("      had to average %s for the gate to have been return-neutral here."
              % _pct(sw["flip_return"]).strip())
            P("      UNKNOWN dropped: %s" % ", ".join(c["dropped_by_veto"]["unknown"]))
        if c["added"]["n_unknown"]:
            P("      UNKNOWN added  : %s" % ", ".join(c["added"]["unknown"]))
        P("")
        P("    beat-count on the swap: dropped %d beaters of %d measured (%d unknown); added"
          % (c["dropped_by_veto"]["n_beat"], c["dropped_by_veto"]["n_measured"],
             c["dropped_by_veto"]["n_unknown"]))
        P("      %d beaters of %d measured (%d unknown). Net on the observed part: %+d beaters"
          % (c["added"]["n_beat"], c["added"]["n_measured"], c["added"]["n_unknown"],
             c["added"]["n_beat"] - c["dropped_by_veto"]["n_beat"]))
        P("      out of the shipped %d -- and the unknowns on both sides can move it further"
          % a["depth_n"])
        P("      in either direction, so it is a reading of the priced part, not a verdict.")

        P("")
        P("  --- NAME BY NAME (the real output at this n) ---")
        P("    %-14s %-10s %-9s %s" % ("dropped pick", "return", "beat?", "flags that ejected it"))
        for t in a["dropped_by_veto"]:
            r = c["dropped_by_veto"]["per_name"].get(t)
            beat = ("-" if r is None else
                    ("YES" if (r - a["bench"]) >= res["threshold"] else "no"))
            P("    %-14s %-10s %-9s %s"
              % (t, ("UNPRICEABLE" if r is None else "%+.1f%%" % (r * 100)), beat,
                 ", ".join(a["ejected_flags"].get(t) or [])))
        for t in a["added"]:
            r = c["added"]["per_name"].get(t)
            beat = ("-" if r is None else
                    ("YES" if (r - a["bench"]) >= res["threshold"] else "no"))
            P("    %-14s %-10s %-9s %s"
              % ("+" + t, ("UNPRICEABLE" if r is None else "%+.1f%%" % (r * 100)), beat,
                 "(replacement)"))

        P("")
        P("  --- PER FLAG, over the top-%d kills ONLY ---" % a["depth_n"])
        if not a["by_flag"]:
            P("    no ejected name reached the un-vetoed top-%d at this anchor, so NO FLAG has"
              % a["depth_n"])
            P("    an effect to measure here. That is a real answer: whatever the gate did to")
            P("    the pool, it did not change what this anchor would have bought via ejection.")
        else:
            P("    %-24s %6s %8s %s" % ("flag", "n_any", "n_solely", "solely-ejected names"))
            for f in sorted(a["by_flag"]):
                e = a["by_flag"][f]
                P("    %-24s %6d %8d %s"
                  % (f, e["n_any"], e["n_solely"], ", ".join(e["solely"]) or "-"))
            P("    n_any counts every kill the flag fired on and SUMS PAST the kill total")
            P("    (EJECT_MIN_FLAGS=1, so names fail several). n_solely is the counterfactual:")
            P("    drop that flag and exactly those names come back.")
            for f in sorted(a["by_flag"]):
                e = a["by_flag"][f]
                if not e["n_solely"]:
                    continue
                x = e["cohort_solely"]
                P("      %-22s solely-ejected: mean(meas) %s   %s"
                  % (f, _pct(x["mean_return"]), _three_way(x)))

        P("")
        P("  --- PER FLAG, THE SAME FUNNEL: what dropping the flag would give back ---")
        P("    %-24s %9s %9s %9s %7s"
          % ("flag", "any(pool)", "sole(pool)", "sole(top100)", "sole(top%d)" % a["depth_n"]))
        for fl in sorted(a["flag_funnel"]):
            e = a["flag_funnel"][fl]
            P("    %-24s %9d %9d %9d %7d"
              % (fl, e["any_pool_wide"], e["solely_pool_wide"],
                 e["solely_in_scored_pool"], e["solely_in_top"]))
        P("    any(pool) sums to %d against %d actually ejected -- it is NOT a partition"
          % (sum(a["pool_by_flag"].values()) if a["pool_by_flag"] else 0, a["n_ejected_pool"]))
        P("    (EJECT_MIN_FLAGS=1, so most ejected names fail several flags). sole(...) is the")
        P("    counterfactual at each depth: drop the flag and exactly that many names return.")
        P("    THE ONLY COLUMN THAT CAN MOVE THE SHIPPED LIST IS THE LAST ONE.")

    #  ---- spotlight ----
    if res.get("spotlight"):
        P("")
        P("=" * 72)
        P("  SPOTLIGHT -- named tickers, whatever the aggregate says")
        P("  %-9s %-8s %-7s %-7s %-9s %-11s %s"
          % ("anchor", "ticker", "u-rank", "in u20", "ejected", "return", "flags"))
        for r in res["spotlight"]:
            ret = ("UNPRICEABLE" if r["total_return"] != r["total_return"]
                   else "%+.1f%%" % (r["total_return"] * 100))
            P("  %-9s %-8s %-7s %-7s %-9s %-11s %s"
              % (r["wid"], r["ticker"],
                 ("-" if r["unvetoed_rank"] is None else str(r["unvetoed_rank"])),
                 ("YES" if r["in_unvetoed_top20"] else "no"),
                 ("YES" if r["ejected"] else "no"), ret,
                 ", ".join(r["ejected_flags"]) or "-"))
        P("  u-rank is the name's position in the UN-VETOED stage-2 ordering; '-' means it is")
        P("  not in that ordering at all (not scored, or below its depth).")

    P("")
    P("  CAVEATS (do not launder):")
    P("   * n IS TINY. The swap is a handful of names per anchor over two heavily-overlapping")
    P("     36mo windows = ONE regime. Nothing here is a per-flag verdict; it is a named list.")
    P("   * COVERAGE CUTS AGAINST THE GATE'S CRITICS AND ITS DEFENDERS EQUALLY. An unpriceable")
    P("     ejected name is UNKNOWN. It is not evidence the gate saved us from a delisting and")
    P("     it is not evidence the gate cost us a winner.")
    P("   * SELECTION MOVES, NOT JUST MEMBERSHIP: the veto ejects before the Stage-1 head() cut,")
    P("     so the Stage-2 normalisation population changes too. `dropped (pool shift)` is that")
    P("     channel, and it is NOT attributable to any flag.")
    P("   * The un-vetoed pass is a COUNTERFACTUAL RE-RANK, not a historical figure. It is what")
    P("     tonight's model would have picked with the gate off, on tonight's panel.")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Pipeline entry point                                                       #
# --------------------------------------------------------------------------- #
def run_in_pipeline(dmdic, merged, registry, price_source, per_anchor_vetoed, log=None,
                    exchange_filter=None, carve="on", weights="default",
                    spotlight=DEFAULT_SPOTLIGHT):
    """Re-rank the CLEAN anchors with the veto OFF and attribute the difference.

    REUSES the vetoed rankings the grid stage already paid for; the only new cost is a second
    PIT reproduction of the anchors that have a 36-month eval leg -- TWO of the seven, because
    the other five have no eval anchor in the grid and so nothing to compare returns over.
    MEASURED on this panel: a full seven-anchor ranking pass is ~205s (~30s per anchor), so the
    two clean anchors are about a minute inside a multi-hour run.

    Returns the attribution dict; prints the report.
    """
    import depth_horizon_grid as dhg
    log = log or (lambda *a: None)
    wanted = [wid for wid, buy in dhg.BUY_ANCHORS
              if wid in dhg.CLEAN_BUY_IDS
              and wid in (per_anchor_vetoed or {})
              and dhg.ANCHOR_IDX[buy] + HORIZON_M // 12 < len(dhg.ANCHORS)]
    if not wanted:
        print("\nSTAGE-1 GATE ATTRIBUTION: no clean anchor with a %dmo eval leg was ranked "
              "this run -- UNMEASURED, which is not the same as 'the gate did nothing'."
              % HORIZON_M, flush=True)
        return None
    log("[gate-attr] re-ranking %s with the Stage-1 veto OFF (counterfactual) ..."
        % ", ".join(wanted))
    inputs = dhg.inputs_from_memory(dmdic, merged, registry, log)
    unvetoed = dhg.rank_all_anchors(inputs, log, weights=weights, carve=carve,
                                    exchange_filter=exchange_filter, stage1_veto=False,
                                    anchors=wanted)
    res = attribute(per_anchor_vetoed, unvetoed, price_source, anchors=wanted,
                    spotlight=spotlight)
    print("\n" + format_report(res), flush=True)
    return res
