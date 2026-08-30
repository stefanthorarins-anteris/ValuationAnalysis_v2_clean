"""The MEASUREMENT BASIS stamp -- ONE reader for the string `stage2_pit` writes.

WHY THIS MODULE EXISTS.  Every PIT figure this project published before 2026-08-27 was
UN-VETOED: nothing under `baseline_tools/` applied the Stage-1 solvency gate.  So a number
carrying no basis is ambiguous, and a vetoed number set beside an un-vetoed one is simply
wrong.  `stage2_pit.reproduce_pit_top` stamps `basis` per anchor for exactly that reason and
`depth_horizon_grid.rank_all_anchors` carries it through.

THE READER USED TO LIVE IN `pipeline_analysis` AND ONLY ONE CONSUMER COULD REACH IT.  The
commit that introduced the stamp said "basis is stamped everywhere"; it was stamped in
`rank_all_anchors`'s log lines and in the two-clause banner, and NOWHERE ELSE.  The grid
report and `scoring_compare` -- the two artifacts a reader actually keeps -- printed nothing,
so the first vetoed grid report ever produced is shaped identically to every un-vetoed one in
the archive.  `depth_horizon_grid` cannot import `pipeline_analysis` (that is the cycle:
`pipeline_analysis` imports `dhg`), which is why the reader is here instead of copied.  A
second implementation of this parse is the failure mode to avoid, not a second stamp.

THE REGEX IS A PARSER OF `stage2_pit`'s FORMAT STRING and that coupling is the reason the
kind/magnitude split exists at all -- see `kind`.
"""

import re

#  The per-anchor MAGNITUDE inside a basis stamp.  `stage2_pit` writes
#  "VETOED (stage-1 solvency gate applied, 1125 ejected)", so the ejection COUNT is part of
#  the string -- and the count differs at every anchor by construction, because the pools are
#  different sizes.
EJECTED_RE = re.compile(r",\s*(\d+)\s+ejected\)")

UNSTAMPED = "un-vetoed (basis not stamped)"


def kind(basis):
    """The KIND of measurement basis, with the per-anchor magnitude stripped out.

    WHY THIS EXISTS.  The collapse used to run on the WHOLE stamp, so seven anchors that
    gated identically produced seven distinct strings and the 2026-08-28 run printed
    `BASIS: MIXED -- anchors disagree` followed by seven lines differing only in an ejection
    count (955, 991, 1037, 1125, 1134, 1144, 1182).  Nothing disagreed.  The banner exists to
    stop a vetoed number being compared against an un-vetoed one, and a false MIXED devalues
    it exactly like a false alarm devalues any other alarm.

    THE COUNT IS NOT DISCARDED, it is moved: `of` reports the counts separately, per anchor,
    where they read as what they are -- how much each anchor's pool was cut -- instead of as
    a disagreement about the measurement basis.

    DECLINED STAMPS KEEP THEIR PARENTHETICAL, deliberately.  "un-vetoed (veto DECLINED: panel
    missing netDebtToEBITDA)" and "un-vetoed (veto DECLINED: panel missing uCurrentRatio)" are
    genuinely different bases -- different flags never ran -- so those still read as MIXED.
    Only the ejection count, which is a magnitude and not a basis, is normalised away.
    """
    if not basis:
        return UNSTAMPED
    return EJECTED_RE.sub(")", str(basis))


def of(per_anchor):
    """The measurement BASIS carried by a `rank_all_anchors` result -- vetoed or not.

    Collapses the per-anchor stamps to a KIND for a header, reports the per-anchor ejection
    counts beside it, and REFUSES to guess when the anchors are on genuinely different bases.
    """
    stamps = {wid: ((v or {}).get("basis") or UNSTAMPED)
              for wid, v in (per_anchor or {}).items()}
    if not stamps:
        return "unknown (no anchors)"
    kinds = sorted({kind(b) for b in stamps.values()})
    if len(kinds) > 1:
        return "MIXED -- anchors disagree: " + " | ".join(kinds)
    counts = {}
    for wid, b in stamps.items():
        m = EJECTED_RE.search(str(b))
        if m:
            counts[wid] = int(m.group(1))
    if not counts:
        return kinds[0]
    ns = sorted(counts.values())
    return ("%s -- %d anchor(s), %d-%d ejected each (%s)"
            % (kinds[0], len(stamps), ns[0], ns[-1],
               ", ".join("%s=%d" % (w, counts[w]) for w in sorted(counts))))


def tag(basis):
    """A COLUMN-WIDTH tag for the same stamp, for tables that cannot carry the full string.

    Deliberately COARSE and deliberately NOT reconstructible into the full stamp: it answers
    "may I put these two numbers side by side" and nothing else.  Anything it cannot classify
    comes back `UNKNOWN` rather than being defaulted to a basis -- an unrecognised stamp is
    not evidence of an un-vetoed run, and defaulting it would be the original defect in
    miniature.
    """
    s = str(basis or "").strip()
    #  ORDER MATTERS HERE, TWICE, and both orderings were got wrong once.
    #
    #  MIXED IS TESTED FIRST because `of()` builds a MIXED string by JOINING the disagreeing
    #  kinds -- so a pool containing one unstamped anchor produces
    #  "MIXED -- anchors disagree: VETOED (...) | un-vetoed (basis not stamped)", which
    #  CONTAINS the not-stamped substring.  Testing "not stamped" first collapsed that to
    #  UNSTAMPED, and the consequence was not cosmetic: in `scoring_compare` a mixed column
    #  and a genuinely unstamped column then carried the SAME tag, the
    #  "NOT ALL ON THE SAME BASIS ... confounded" warning was SUPPRESSED because the tag set
    #  had collapsed to one value, and a VETOED column printed "[UNSTAMPED]" beside its own
    #  VETOED string.  A disagreement is the one thing this tag must never swallow.
    if s.startswith("MIXED"):
        return "MIXED"
    #  THEN not-stamped, before the `un-vetoed` prefix below: `UNSTAMPED` is the string
    #  "un-vetoed (basis not stamped)", so a plain `startswith("un-vetoed")` would classify an
    #  UNSTAMPED artifact as UN-VETOED -- answering "I do not know what produced this" with a
    #  definite basis.  Matched on a SUBSTRING because callers append their own provenance
    #  ("... -- no basis in cells_X.csv").
    if not s or "basis not stamped" in s or s.lower().startswith("unknown"):
        return "UNSTAMPED"
    if "DECLINED" in s:
        return "DECLINED"
    if s.startswith("VETOED"):
        return "VETOED"
    if s.startswith("un-vetoed"):
        return "un-vetoed"
    return "UNKNOWN"


def banner_lines(basis, width=72):
    """The standing warning that goes with any stamped figure, as LINES.

    Returned rather than printed so the report builders (which accumulate a list and write a
    file) and the run log (which prints) share one wording.  Two copies of this paragraph
    drifting apart is how a caveat stops being read.
    """
    return ["#" * width,
            f"# MEASUREMENT BASIS: {basis}",
            "#   Every PIT figure published by this project BEFORE 2026-08-27 was UN-VETOED --",
            "#   the backtest path never applied the Stage-1 solvency gate.  Do NOT compare a",
            "#   number on one basis against a number on the other.",
            "#" * width]
