"""
Live-path issuer-dedup self-checks.

The deployed top-20 (postBo -> postBoScoreRanking -> head(N) emission) must contain
DISTINCT issuers: a dual-listing / share-class (e.g. TFPM / TFPM.TO) must collapse to
ONE line, matching the backtest harness's dedup.

UPDATED 2026-08-05 -- THE SURVIVOR RULE CHANGED, DELIBERATELY.  These checks used to
assert "keep the highest-RANKED line", and two of them asserted it directly.  That rule
is RETIRED: canonicity now overrides rank (carveOut.dedup_ranked), because the score of a
notes / preferred / depositary line is the ISSUER's fundamentals attached to an instrument
the CEO is not buying, and because 10 of 19 duplicate groups differ on score with
IDENTICAL price and IDENTICAL fundamentals -- the difference is history depth, so the old
rank rule was systematically selecting on noise.  The issuer still occupies its BEST RANK
POSITION; only which ticker represents it changed.  Nothing here was weakened to make the
suite green: the checks now assert the new rule, plus the property that survives both
(one line per issuer, order preserving).

postBoScoreRanking itself is not cleanly offline-runnable (its DcfToPrice/CycleHeat-beta
paths need the network -- the reason stage2_pit re-implements the metric loop), so we
test (1) the EXACT function the live emission invokes -- carveOut.dedup_ranked -- on a
realistic TFPM/TFPM.TO fixture, and (2) a wiring guard that postBoScoreRanking calls it
and exposes the audit interface.

Run:  python baseline_tools/test_live_dedup.py
"""
import inspect
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import pandas as pd
import carveOut as co


def _cdx(rows):
    """rows: list of (source, revenue, netIncome, totalAssets, shares, marketCap)."""
    recs = []
    for s, rev, ni, ta, sh, mc in rows:
        recs.append({"source": s, "date": "2026-03-31", "revenue": rev,
                     "netIncome": ni, "totalAssets": ta,
                     "weightedAverageShsOut": sh, "marketCap": mc})
    return pd.DataFrame(recs)


# Real-ish TFPM figures (from the pickle): identical shares across listings; the CAD
# line's revenue/NI/TA differ from the USD line by <0.3% (a common reporting currency).
TFPM_US = ("TFPM", 144956892.0, 115309331.0, 2225851396.0, 206573855.0, 7.17e9)
TFPM_TO = ("TFPM.TO", 144583873.0, 115012604.0, 2220123603.0, 206573855.0, 7.17e9)
AAA = ("AAA", 1.0e8, 2.0e7, 9.0e8, 5.0e7, 1.5e9)
CRUS = ("CRUS", 2.0e9, 3.0e8, 3.0e9, 5.5e7, 5.0e9)

NAMES = {"TFPM": "Triple Flag Precious Metals Corp.",
         "TFPM.TO": "Triple Flag Precious Metals Corp.",
         "AAA": "Alpha Alpha Inc.", "CRUS": "Cirrus Logic, Inc."}


def test_K2_marketcap_groups_the_pair_with_no_names():
    """The pair collapses with NO names at all.  It used to be edge C (exact shares +
    near-equal fundamentals, 5% tolerance) that did this; that edge is RETIRED and K2
    (exact marketCap, an issuer-level number identical across an issuer's lines) does it
    with no tolerance."""
    cdx = _cdx([AAA, TFPM_US, TFPM_TO, CRUS])
    ranked = ["AAA", "TFPM", "TFPM.TO", "CRUS"]     # TFPM ranked ABOVE TFPM.TO
    kept, dropped = co.dedup_ranked(ranked, cdx, names={})
    assert kept == ["AAA", "TFPM", "CRUS"], kept
    assert dropped == [("TFPM.TO", "TFPM")], dropped
    print(f"  [ok] K2 (no names): {ranked} -> {kept}  dropped {dropped}")


def test_K3_name_plus_shares_when_every_issuer_level_field_is_FX_shifted():
    """K3 (same normalised name + EXACT shares) collapses the pair even when EVERY
    issuer-level field on the CAD line is FX-shifted, so neither K1 (statements) nor K2
    (marketCap) can fire.  Share count is currency-invariant, which is why K3 is retained
    from the old edge set rather than subsumed.

    NOTE the survivor here is TFPM.TO, and that is the SPECIFIED ordering, not a defect:
    both lines are commons (neither carries a canonicity marker), shares tie, so the key
    falls to -marketCap -- and this FIXTURE deliberately inflates the CAD line's market cap
    by 1.35x, which real data does not do (marketCap is an issuer-level number, identical
    across an issuer's lines in 1,250 of 1,282 real groups).  The assertion is therefore on
    ONE-LINE-PER-ISSUER, not on which of two commons wins a synthetic currency mismatch.

    `isin_map={}` IS LOAD-BEARING HERE (added 2026-08-13, when K4 was wired).  This
    fixture is built from REAL symbols, and K4's default map is whichever
    `isindic_fmp_*.pickle` sits in the repo root -- which knows that TFPM and TFPM.TO are
    one issuer and merges them on ISIN alone.  That merge is CORRECT, and it is precisely
    what would make this test stop testing K3: the no-names branch below would collapse
    for a reason that has nothing to do with names or shares.  So the ISIN key is switched
    off explicitly, and this test means what its name says."""
    tfpm_to_fx = ("TFPM.TO", 195_000_000.0, 155_000_000.0, 3_000_000_000.0,
                  206573855.0, 9.6e9)   # ~1.35x FX on rev/NI/TA AND on marketCap
    cdx = _cdx([AAA, TFPM_US, tfpm_to_fx, CRUS])
    ranked = ["AAA", "TFPM", "TFPM.TO", "CRUS"]
    # without names, nothing can group them: K1 and K2 both differ by the FX shift
    kept_noname, _ = co.dedup_ranked(ranked, cdx, names={}, isin_map={})
    assert "TFPM.TO" in kept_noname and "TFPM" in kept_noname, kept_noname
    # WITH names, K3 (name + shares) collapses it to ONE line at the group's best rank
    kept, dropped = co.dedup_ranked(ranked, cdx, NAMES, isin_map={})
    assert len(kept) == 3 and len([s for s in kept if s.startswith("TFPM")]) == 1, kept
    assert kept.index([s for s in kept if s.startswith("TFPM")][0]) == 1, (
        "the issuer must occupy its BEST rank position (index 1)", kept)
    assert len(dropped) == 1 and set(dropped[0]) == {"TFPM", "TFPM.TO"}, dropped
    print(f"  [ok] K3 (name+shares, everything FX-shifted): collapses -> {kept}")


def test_CANONICITY_overrides_rank_and_order_is_preserved():
    """THE SURVIVOR RULE, restated for the new rule.  A distinct issuer below the dup is
    still promoted as the slot frees, and the issuer still sits at its BEST rank
    position -- but the surviving TICKER is now the canonical line REGARDLESS of which
    line ranked higher.  Demonstrated on a preferred, which is the case that matters:
    under the old rule a preferred that outranked its common SURVIVED."""
    cdx = _cdx([TFPM_US, TFPM_TO, AAA, CRUS])
    ranked = ["TFPM", "TFPM.TO", "AAA", "CRUS"]     # dup at ranks 1,2
    kept, dropped = co.dedup_ranked(ranked, cdx, NAMES)
    assert kept == ["TFPM", "AAA", "CRUS"], kept          # TFPM.TO removed, rest shift up
    assert dropped == [("TFPM.TO", "TFPM")], dropped

    # A PREFERRED ranked ABOVE its common: the common still survives, at rank 1.
    pfd = ("TFPM-PA", TFPM_US[1], TFPM_US[2], TFPM_US[3], TFPM_US[4], TFPM_US[5])
    cdx2 = _cdx([pfd, TFPM_US, AAA, CRUS])
    names2 = dict(NAMES, **{"TFPM-PA": "Triple Flag Precious Metals Corp."})
    kept2, dropped2 = co.dedup_ranked(["TFPM-PA", "TFPM", "AAA", "CRUS"], cdx2, names2)
    assert kept2 == ["TFPM", "AAA", "CRUS"], kept2
    assert dropped2 == [("TFPM-PA", "TFPM")], dropped2
    print("  [ok] canonicity overrides rank + order-preserving promotion")


def test_no_dupes_is_noop():
    cdx = _cdx([AAA, CRUS])
    ranked = ["AAA", "CRUS"]
    kept, dropped = co.dedup_ranked(ranked, cdx, NAMES)
    assert kept == ranked and dropped == [], (kept, dropped)
    print("  [ok] no duplicate issuers -> no-op")


def test_live_wiring_present():
    """postBoScoreRanking must expose the dedup interface AND actually CALL dedup_ranked
    as LIVE code (verified on the AST, so a commented-out / doc-only mention cannot pass)."""
    import ast
    import textwrap
    import postBoRank as pbr
    sig = inspect.signature(pbr.postBoScoreRanking)
    assert "names" in sig.parameters, sig
    assert "dedup_issuers" in sig.parameters, sig
    assert sig.parameters["dedup_issuers"].default is True

    # Parse the function bodies; comments/docstrings are dropped by the parser, so only a
    # real executable call survives.  The dedup was extracted into the private helper
    # _dedup_issuers_in_ranking (readability refactor), so inspect BOTH the orchestrator
    # and that helper: assert (1) postBoScoreRanking WIRES the helper in, (2) the helper
    # actually CALLS dedup_ranked as live code, and (3) the audit keys are emitted as live
    # code (not just named in a comment).
    called = set()
    assigned_str = set()
    for fn in (pbr.postBoScoreRanking, pbr._dedup_issuers_in_ranking):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute):
                    called.add(f.attr)
                elif isinstance(f, ast.Name):
                    called.add(f.id)
            # dict literal keys for the audit interface (rankdic = {... 'issuer_dupes_dropped': ...})
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assigned_str.add(node.value)
    assert "_dedup_issuers_in_ranking" in called, "orchestrator does not WIRE the dedup helper (AST)"
    assert "dedup_ranked" in called, "live ranker does not CALL carveOut.dedup_ranked (AST)"
    assert "issuer_dupes_dropped" in assigned_str, "audit key not emitted as live code"
    assert "postRank_predupe" in assigned_str, "pre-dedup audit frame not emitted as live code"
    print("  [ok] postBoScoreRanking wires + CALLS dedup_ranked (AST) + emits audit keys + params")


if __name__ == "__main__":
    print("Live-path issuer-dedup self-checks")
    test_K2_marketcap_groups_the_pair_with_no_names()
    test_K3_name_plus_shares_when_every_issuer_level_field_is_FX_shifted()
    test_CANONICITY_overrides_rank_and_order_is_preserved()
    test_no_dupes_is_noop()
    test_live_wiring_present()
    print("ALL LIVE-DEDUP SELF-CHECKS PASSED")
