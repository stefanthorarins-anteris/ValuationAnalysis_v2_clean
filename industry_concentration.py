"""INDUSTRY COUNTER + CONCENTRATION reporting for the top-100 / top-20.  INFORMATIONAL ONLY.

WHY THIS MODULE EXISTS
----------------------
The shipped top-100 can sit heavily in one industry -- the 2026-07-17 corrected list holds
**11 Marine Shipping** of 100 and **7 of 20** -- and NOTHING in the deliverables said so.  The
CEO reviews the shortlist by hand, so concentration is something he has to SEE; the filter must
not act on it (CEO standing ruling: no hard gates in the filtering logic).  So this module
produces text, and only text.

NOT A GATE.  Nothing here is read back into scoring, ranking, membership, banding or the
carve-out.  Every function takes a list of symbols and returns strings; none mutates its
inputs.  `test_industry_concentration.py` pins that (bit-identity of postRank + the AggScore
frame across a report call) because "informational" is a property that rots silently.

WHERE THE CODE CAME FROM
------------------------
`concentration_line` and the cycle-cluster table were written for
`baseline_tools/industry_attribution.py` (the "which fix created the shipping concentration"
ablation) and were reachable only from an analysis script.  They are LIFTED here verbatim --
repo root, no `baseline_tools` dependency -- so the pipeline can import them and there is
exactly ONE concentration calculation in the repo.  `industry_attribution` now re-exports
these names, so `ia.concentration_line(...)` still works for its existing caller
(`baseline_tools/run_corrected_current.py`).

THREE DELIBERATE DIFFERENCES IN THE LIFT, all in `concentration_line`, none of which changes
a count (they are the reasons to read the diff carefully):

  1. LATENT-DEFECT FIX.  An unlabelled name resolves as `ind.get(s) or UNKNOWN` rather than
     `ind.get(s, "UNKNOWN")`.  The old form treats a symbol that is PRESENT in the map with a
     null/empty industry as an industry literally named `None`, which could win the "N of M in
     ONE industry" line and trip the warning -- while the universe side of the same function
     already filtered falsy values (`if ind.get(s)`), so the two halves disagreed.  The current
     `industrydic_fmp_2026-07-13.pickle` has 0 null values out of 18,333 symbols, so this is
     bit-identical on today's data.
  2. TIES SORT ALPHABETICALLY instead of by first insertion.  `Counter.most_common` breaks ties
     by the order names were seen, i.e. by RANK, so the same 100 names in a different order
     printed a different line and a diff of two run logs was noisy for no reason.  Which member
     of an exact tie is named as "the ONE industry" was arbitrary before and is arbitrary now;
     it is merely deterministic now.
  3. `Unclassified N` no longer competes for a `top_k` slot -- it is appended after the named
     industries.  It was already excluded from the "one industry" line and the warning; this
     makes the head of the line consistently the top-k INDUSTRIES.

TAXONOMY: `industry`, not `sector`.  Measured coverage is ~100% for industry against ~87% for
sector, and industry is what the CEO asked for.  Labels come from `industrydic_fmp_*.pickle`
(18,333 symbols, 156 FMP industries) via `carveOut._load_industry_map` -- already the
pipeline's FIN-2/FIN-3 classifier and already loaded by the deck, so no new taxonomy and
nothing extra to maintain.

UNCLASSIFIED IS NEVER DROPPED.  A concentration report that silently omits the names it could
not label is worse than none: the reader cannot tell 11-of-100 from 11-of-100-minus-9-unknown.
So the count is always reported, and always as its own line -- an unclassified bucket is a DATA
GAP, not an industry, and must not compete for the "one industry" line.
"""

from collections import Counter

import numpy as np

UNKNOWN = "UNKNOWN"


def industry_map(industry_pickle=None):
    """symbol -> FMP industry label, via the pipeline's own classifier.

    Imported lazily: this module is imported from `postBo.writeResWrapper` and from the
    offline deck, and neither should pay a `carveOut` import (which pulls pandas + globs the
    repo for pickles) merely to have the reporting helpers in scope.
    """
    import carveOut as co
    return co._load_industry_map(industry_pickle)


#  CYCLE CLUSTERS -- the grouping `top20-real-value-verification.md` S6.3 step 3 actually asks
#  for.  Grouping by industry LABEL under-reports the exposure that matters: on the 07-17
#  corrected list it reads "Marine Shipping 40%" when the real commodity-cycle exposure is
#  8 shipping + 4 oil & gas E&P = 60%.  Two names in different FMP industries can share one
#  cycle; the doctrine's warning is about the CYCLE, not the tag.  Both views are reported --
#  the label view stays because it is what the CSVs and the deck show.
CYCLE_CLUSTERS = {
    "Commodity/Freight cycle": [
        "Marine Shipping", "Oil & Gas Exploration & Production", "Oil & Gas Midstream",
        "Oil & Gas Refining & Marketing", "Oil & Gas Equipment & Services",
        "Oil & Gas Integrated", "Oil & Gas Drilling", "Thermal Coal", "Coking Coal",
        "Railroads", "Trucking", "Integrated Freight & Logistics",
        "Steel", "Aluminum", "Copper", "Other Industrial Metals & Mining",
        "Gold", "Silver", "Other Precious Metals & Mining", "Uranium",
        "Agricultural Inputs", "Chemicals", "Specialty Chemicals", "Lumber & Wood Production",
        "Paper & Paper Products", "Building Materials", "Airlines",
    ],
    "Rate/Credit cycle": [
        "Banks - Regional", "Banks - Diversified", "Credit Services", "Mortgage Finance",
        "Insurance - Property & Casualty", "Insurance - Life", "Insurance - Diversified",
        "Insurance - Reinsurance", "Insurance - Specialty", "Insurance Brokers",
        "Capital Markets", "Asset Management", "Financial - Data & Stock Exchanges",
        "REIT - Diversified", "REIT - Office", "REIT - Retail", "REIT - Residential",
        "REIT - Industrial", "REIT - Hotel & Motel", "REIT - Mortgage",
        "REIT - Healthcare Facilities", "REIT - Specialty",
    ],
    "Consumer cycle": [
        "Auto Manufacturers", "Auto Parts", "Auto & Truck Dealerships",
        "Residential Construction", "Homebuilding", "Furnishings, Fixtures & Appliances",
        "Apparel Manufacturing", "Apparel Retail", "Footwear & Accessories",
        "Restaurants", "Lodging", "Resorts & Casinos", "Travel Services",
        "Leisure", "Department Stores", "Specialty Retail",
    ],
}
_CYCLE_OF = {ind: cl for cl, inds in CYCLE_CLUSTERS.items() for ind in inds}


def cycle_of(industry):
    """Cycle cluster for an FMP industry label, or None if it is not cycle-classified.
    None is NOT a cluster -- an unclassified name is counted separately, never pooled."""
    return _CYCLE_OF.get(industry)


def _fmt_enr(x):
    """Enrichment factor.  `%.0fx` collapses every sub-1.5x value to '0x' or '1x', which
    reads as "no signal" for a 1.4x and as "one times" for a 0.6x -- both wrong.  Use a
    decimal below 10x."""
    if x is None or not np.isfinite(x):
        return "n/a"
    return ("%.1fx" % x) if x < 10 else ("%.0fx" % x)


def _label(ind, symbol):
    """The industry label for one symbol.  Falsy (absent, None, '') -> UNKNOWN."""
    return ind.get(symbol) or UNKNOWN


def industry_counts(top_sources, ind=None):
    """(named_counts, n_unclassified, n_total) for one ranked list.

    `named_counts` is a list of (industry, count) sorted by count DESCENDING, ties broken
    alphabetically so two runs over the same list print identically (`Counter.most_common`
    orders ties by first-insertion, i.e. by rank, which makes a diff of two run logs noisy for
    no reason).  The unclassified bucket is returned SEPARATELY and never appears in
    `named_counts` -- see the module docstring.

    THE single count in the repo: `counter_block`, `concentration_line` and every caller read
    this, so a divergence between the run log, the deck and the analysis tooling is not
    expressible.
    """
    if ind is None:
        ind = industry_map()
    top = list(top_sources)
    c = Counter(_label(ind, s) for s in top)
    n_unclassified = c.pop(UNKNOWN, 0)
    named = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return named, n_unclassified, len(top)


def counter_block(top_sources, list_label, ind=None, indent="  "):
    """THE INDUSTRY COUNTER the CEO asked for, as a list of lines.

    Every industry present is listed -- count-descending, so concentration is visible at a
    glance instead of requiring arithmetic.  Singletons are collapsed onto wrapped `1 x N`
    lines: they are still named IN FULL (nothing is hidden), but ~35 one-line entries would
    bury the 11 at the top, which is the number the block exists to surface.
    """
    if ind is None:
        ind = industry_map()
    named, n_unc, n_tot = industry_counts(top_sources, ind)
    if not n_tot:
        return ["%sINDUSTRY COUNTER (%s): empty list -- nothing to count."
                % (indent, list_label)]
    out = ["%sINDUSTRY COUNTER -- %s (%d names, informational only: affects no score, rank "
           "or membership)" % (indent, list_label, n_tot)]
    multi = [(name, k) for name, k in named if k > 1]
    singles = [name for name, k in named if k == 1]
    for name, k in multi:
        out.append("%s  %4d  %s  (%.0f%% of the list)"
                   % (indent, k, name, 100.0 * k / n_tot))
    if singles:
        out.append("%s     1  x %d:" % (indent, len(singles)))
        line = ""
        for name in singles:
            # ` | `, NOT `, ` -- FMP industry labels CONTAIN commas ("Furnishings, Fixtures &
            # Appliances", "Gambling, Resorts & Casinos", "Paper, Lumber & Forest Products"),
            # so a comma-joined list reads as five industries where there are two.
            piece = (" | " if line else "") + name
            if len(line) + len(piece) > 84:
                out.append("%s          %s" % (indent, line))
                line = name
            else:
                line += piece
        if line:
            out.append("%s          %s" % (indent, line))
    out.append("%s  unclassified (no industry label -- a data gap, not an industry): %d"
               % (indent, n_unc))
    return out


def concentration_line(top_sources, universe_sources, ind=None, top_k=6, warn_share=0.25):
    """The deck pre-flight's INDUSTRY CONCENTRATION line, as one string.

    Automates `top20-real-value-verification.md` S6.3 step 3, which currently asks the CEO to
    tally industries by hand.  The COUNT alone is not the signal -- 3 of 20 in an industry that
    is 15% of the universe is nothing, while 3 of 20 in one that is 0.8% is 19x.  So the line
    carries the universe base rate and the enrichment factor, which is what turns a tally into
    something interpretable.

    Uses `industrydic_fmp_*.pickle` -- 18,333 symbols over 156 FMP industries, already loaded
    by the deck (`get_industry`) and already the pipeline's primary FIN-2/FIN-3 classifier -- so
    this introduces no new taxonomy and nothing to maintain.

    The `warn_share` lines are PRINTED WARNINGS ONLY.  They return in the string; no caller
    branches on them, and nothing in the pipeline may.
    """
    if ind is None:
        ind = industry_map()
    top = list(top_sources)
    if not top:
        return "INDUSTRY CONCENTRATION: empty top list -- nothing to report."
    known_uni = [s for s in universe_sources if ind.get(s)]
    uni_c = Counter(ind[s] for s in known_uni)
    n_uni = len(known_uni)

    # ---- view 1: by INDUSTRY LABEL (what the CSVs and the deck show) ------------------
    named, n_unc, _ = industry_counts(top, ind)
    parts = []
    for name, k in named[:top_k]:
        br = uni_c.get(name, 0) / n_uni if n_uni else float("nan")
        enr = ((k / len(top)) / br) if br else float("inf")
        parts.append("%s %d (base %.2f%%, %s)" % (name, k, 100 * br, _fmt_enr(enr)))
    if n_unc:
        parts.append("Unclassified %d" % n_unc)
    out = ["INDUSTRY CONCENTRATION (top-%d): %s" % (len(top), " * ".join(parts))]

    # UNKNOWN is NOT an industry and must never win the "one industry" line or trip the
    # warning: 6 unclassified names are a DATA GAP, not a concentration.
    if named:
        top_name, top_n = named[0]
        br = uni_c.get(top_name, 0) / n_uni if n_uni else float("nan")
        enr = ((top_n / len(top)) / br) if br else float("inf")
        out.append("  -> %d of %d in ONE industry (%s): universe base rate %.2f%%, "
                   "enrichment %s" % (top_n, len(top), top_name, 100 * br, _fmt_enr(enr)))
    if n_unc:
        out.append("  -> %d of %d UNCLASSIFIED (a data gap, not a concentration)"
                   % (n_unc, len(top)))

    # ---- view 2: by CYCLE CLUSTER (what S6.3 step 3 asks for) -------------------------
    cyc = Counter()
    for s in top:
        cl = cycle_of(ind.get(s))
        if cl:
            cyc[cl] += 1
    uni_cyc = Counter()
    for s in known_uni:
        cl = cycle_of(ind[s])
        if cl:
            uni_cyc[cl] += 1
    if cyc:
        cparts = []
        for cl, k in cyc.most_common():
            br = uni_cyc.get(cl, 0) / n_uni if n_uni else float("nan")
            enr = ((k / len(top)) / br) if br else float("inf")
            cparts.append("%s %d/%d = %.0f%% (base %.1f%%, %s)"
                          % (cl, k, len(top), 100 * k / len(top), 100 * br, _fmt_enr(enr)))
        out.append("  CYCLE CLUSTER (the grouping S6.3 step 3 asks for -- a shared cycle can "
                   "span several industry tags):")
        for p in cparts:
            out.append("    " + p)

    # THE TWO WARNINGS ARE INDEPENDENT (fix, 2026-07-30).  The first version gated the
    # label-level warning on the cycle view existing (`if cyc: ... elif ...`), so a list that
    # was 50% Software plus ONE shipping name produced a cycle block and NO warning at all --
    # the label concentration was SUPPRESSED by the presence of an unrelated cycle name, and it
    # would have fired before the cycle view was added.  A second view must never silence the
    # first: a 50%-one-industry list is a concentration whether or not that industry maps to a
    # cycle, so both conditions are now evaluated separately and both can fire.
    if named:
        worst_ind, worst_ind_n = named[0]
        if worst_ind_n / len(top) >= warn_share:
            _mapped = cycle_of(worst_ind)
            out.append("  !! %.0f%% of the list is ONE INDUSTRY (%s)%s -- a concentration on "
                       "its own terms."
                       % (100 * worst_ind_n / len(top), worst_ind,
                          "" if _mapped else ", with no cycle mapping"))
    if cyc:
        worst_cl, worst_n = cyc.most_common(1)[0]
        if worst_n / len(top) >= warn_share:
            out.append("  !! %.0f%% of the list sits in ONE CYCLE (%s). Where both warnings "
                       "fire, the doctrine's cyclicality warning applies to THIS number -- it "
                       "is the larger exposure -- but read both, before position sizing."
                       % (100 * worst_n / len(top), worst_cl))
    return "\n".join(out)


def report_lines(top100_sources, top20_sources, universe_sources=None, ind=None,
                 labels=("top-100", "top-20")):
    """The full block for the run log: the COUNTER for both lists, plus the interpretive
    concentration line (base rate + enrichment + cycle view) when a universe is available.

    `universe_sources` is optional on purpose.  The counter is the deliverable the CEO asked
    for and needs nothing but the ranked list; the base-rate/enrichment view needs the scored
    universe, so it is added when the caller has one and skipped -- with a note, never
    silently -- when it does not.  Pure: returns lines, prints nothing, mutates nothing.

    `labels` is a parameter and not two literals because `ntopagg` / `ntopxlsx` are CLI-settable
    (`-ntopagg 50`): a hardcoded "top-100" heading over 50 names would be a caption that lies.
    The header still states the ACTUAL name count separately, so a ranking that is shallower
    than the requested depth -- the shipped 07-17 postRank is 90 deep -- is visible rather than
    implied.
    """
    if ind is None:
        ind = industry_map()
    bar = "-" * 92
    out = [bar,
           "INDUSTRY COUNTER / CONCENTRATION -- INFORMATIONAL ONLY.  Not a gate: no score, "
           "rank,",
           "membership, band or carve-out decision reads any number below.",
           bar]
    if not ind:
        out.append("  UNAVAILABLE: no industrydic_fmp_*.pickle found, so no name can be "
                   "labelled.")
        out.append(bar)
        return out
    for label, srcs in zip(labels, (top100_sources, top20_sources)):
        out.extend(counter_block(srcs, label, ind=ind))
        if universe_sources is not None and len(list(srcs)):
            out.extend("  " + l for l in
                       concentration_line(srcs, universe_sources, ind=ind).split("\n"))
        elif universe_sources is None:
            out.append("  (universe base rate / enrichment not shown: no scored universe "
                       "passed to the reporter)")
        out.append("")
    out.append(bar)
    return out
