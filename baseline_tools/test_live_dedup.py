"""
Live-path issuer-dedup self-checks.

The deployed top-20 (postBo -> postBoScoreRanking -> head(N) emission) must contain
DISTINCT issuers: a dual-listing / share-class (e.g. TFPM / TFPM.TO) must collapse to
ONE line, keeping the highest-RANKED line, matching the backtest harness's dedup.

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


def test_edge_c_shares_plus_nearequal():
    """Edge C: EXACT shares + near-equal fundamentals -> collapse, no names needed."""
    cdx = _cdx([AAA, TFPM_US, TFPM_TO, CRUS])
    ranked = ["AAA", "TFPM", "TFPM.TO", "CRUS"]     # TFPM ranked ABOVE TFPM.TO
    kept, dropped = co.dedup_ranked(ranked, cdx, names={})
    assert kept == ["AAA", "TFPM", "CRUS"], kept
    assert dropped == [("TFPM.TO", "TFPM")], dropped
    print(f"  [ok] edge C (no names): {ranked} -> {kept}  dropped {dropped}")


def test_edge_b_name_plus_shares_fx():
    """Edge B: same normalized name + EXACT shares collapses even when the CAD line's
    revenue is FX-shifted well beyond edge C's tolerance."""
    tfpm_to_fx = ("TFPM.TO", 195_000_000.0, 155_000_000.0, 3_000_000_000.0,
                  206573855.0, 9.6e9)   # ~1.35x FX on rev/NI/TA -> edge C fails
    cdx = _cdx([AAA, TFPM_US, tfpm_to_fx, CRUS])
    ranked = ["AAA", "TFPM", "TFPM.TO", "CRUS"]
    # without names edge C fails on the FX gap -> NOT collapsed
    kept_noname, _ = co.dedup_ranked(ranked, cdx, names={})
    assert "TFPM.TO" in kept_noname, kept_noname
    # WITH names, edge B (name+shares) collapses it
    kept, dropped = co.dedup_ranked(ranked, cdx, NAMES)
    assert kept == ["AAA", "TFPM", "CRUS"], kept
    assert dropped == [("TFPM.TO", "TFPM")], dropped
    print(f"  [ok] edge B (name+shares, FX rev): collapses -> {kept}")


def test_keeps_highest_ranked_and_order():
    """Survivor = the highest-RANKED line; order preserved; a distinct issuer that
    was below the dup is promoted up as the slot frees."""
    cdx = _cdx([TFPM_US, TFPM_TO, AAA, CRUS])
    ranked = ["TFPM", "TFPM.TO", "AAA", "CRUS"]     # dup at ranks 1,2
    kept, dropped = co.dedup_ranked(ranked, cdx, NAMES)
    assert kept == ["TFPM", "AAA", "CRUS"], kept          # TFPM.TO removed, rest shift up
    assert dropped == [("TFPM.TO", "TFPM")], dropped
    # if TFPM.TO were ranked ABOVE TFPM, IT would be the survivor (rank rule)
    ranked2 = ["TFPM.TO", "TFPM", "AAA", "CRUS"]
    kept2, dropped2 = co.dedup_ranked(ranked2, cdx, NAMES)
    assert kept2 == ["TFPM.TO", "AAA", "CRUS"], kept2
    assert dropped2 == [("TFPM", "TFPM.TO")], dropped2
    print(f"  [ok] rank-based survivor + order-preserving promotion")


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

    # Parse the function body; comments/docstrings are dropped by the parser, so only a
    # real executable call survives.  Assert there is an ast.Call to `dedup_ranked` AND
    # that the audit keys are assigned as live code (not just named in a comment).
    src = textwrap.dedent(inspect.getsource(pbr.postBoScoreRanking))
    tree = ast.parse(src)
    called = set()
    assigned_str = set()
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
    assert "dedup_ranked" in called, "live ranker does not CALL carveOut.dedup_ranked (AST)"
    assert "issuer_dupes_dropped" in assigned_str, "audit key not emitted as live code"
    assert "postRank_predupe" in assigned_str, "pre-dedup audit frame not emitted as live code"
    print("  [ok] postBoScoreRanking CALLS dedup_ranked (AST) + emits audit keys + params")


if __name__ == "__main__":
    print("Live-path issuer-dedup self-checks")
    test_edge_c_shares_plus_nearequal()
    test_edge_b_name_plus_shares_fx()
    test_keeps_highest_ranked_and_order()
    test_no_dupes_is_noop()
    test_live_wiring_present()
    print("ALL LIVE-DEDUP SELF-CHECKS PASSED")
