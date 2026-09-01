"""ISSUER CONTINUITY  --  follow the ISSUER, not the ticker, through a line discontinuity.

THE DEFECT THIS EXISTS FOR (register Q-42).  `returns_core.compute_returns` looks up ONE
symbol at two anchors.  When an issuer's LISTING LINE changes -- a re-domicile, a ticker
change, an exchange move, a share-class reorganisation, a preferred-series redemption --
the old line stops pricing and the primitive fires its terminal policy: `total_return_floor`
reads -100% and the FLOOR readings (the depth-grid `flrTR` column, `beat_rate(missing=...)`,
`target_clauses.lower_bound`) book the pick as a TOTAL CAPITAL LOSS.  A LINE ENDING AND A
COMPANY DYING ARE DIFFERENT EVENTS.  The 2026-08-27 post-mortem on the 14 vanished backtest
picks found ten likely acquisitions, two confirmed not-deaths and ZERO wipeouts: in this
panel a delisting is overwhelmingly NOT capital loss, so the floor is not a conservative
default, it is a wrong answer with a conservative shape.

SEVERITY.  S3 -- this corrupts BACKTEST MEASUREMENT.  Nothing here touches which names the
live screen selects tonight; it changes what the harness says those selections were worth.

WHY A STATIC TABLE, AND NOT A DERIVATION FROM DATA ALREADY ON DISK
------------------------------------------------------------------
Deriving successors automatically was the preferred option and it was tried against the
three candidate sources; all three fail, and the third fails in a way that DECIDES the
question:

  * `isindic_fmp_*.pickle` -- an ISIN per symbol would be the ideal join key.  The cache on
    disk holds 3,129 symbols out of the grid's 90,507 and contains NONE of the four lines at
    issue (VMD, VMD.TO, CMRE, CMRE-PE).  Filling it is a paid fetch.
  * `delisted_tickers_*.csv` -- a bare comma-separated symbol list.  No company name, no
    delisting date, no successor field.  There is nothing in it to join ON.
  * `universe_dedup`'s FUNDAMENTAL FINGERPRINT (revenue / netIncome / totalAssets /
    weightedAverageShsOut byte-identical => same issuer) is a real same-ISSUER detector that
    is already in the tree and already reviewed -- AND IT IS THE WRONG DETECTOR HERE, which
    is the argument that settles it.  Same issuer does not mean same SECURITY.  CMRE-PE and
    CMRE share an issuer and therefore share a fingerprint, but a holder of the 8.50% Series
    E preferred does NOT receive the common's return when the series is called: they receive
    cash at par.  A fingerprint-driven successor map would have measured CMRE-PE at CMRE's
    common-stock return -- swapping a fabricated -100% for a fabricated +25%, which is not an
    improvement, it is the same defect wearing better clothes.

So: a hand-written table, one row per line, EACH ROW CARRYING ITS REASON AND ITS EVIDENCE,
verifiable by anyone against the price grid without spending a call.  It costs nothing at
measurement time and it cannot silently grow: a new row is a code change and a review event.

TWO ROW KINDS, AND THE SECOND ONE IS THE POINT
----------------------------------------------
  successor = "<SYM>"  -- SHARE-FOR-SHARE CONTINUITY.  The holder's position carried onto
                          that line, so the successor line's own return over the SAME two
                          anchors IS the holder's return.
  successor = None     -- IDENTIFIED DISCONTINUITY, UNMEASURABLE EXIT.  We know the line's
                          end was not a death (evidence in the row), and we do NOT know what
                          it paid.  The pick is INDETERMINATE: unmeasured, never -100%, and
                          never silently dropped -- it keeps a status of its own and is
                          counted in every coverage partition.

  ABSENCE OUTRANKS A FABRICATED VERDICT.  This is the P-4/P-5 rule applied one layer down.

THE SUCCESSOR'S OWN TWO LEGS, NEVER A SPLICE.  The return is computed as
`succ(eval) / succ(buy) - 1` -- BOTH legs off the successor line.  It is NOT
`succ(eval) / old(buy) - 1`.  That distinction is load-bearing and it is a second, smaller
defect the post-mortem named: VMD.TO's buy leg is CAD 6.60 and VMD's eval leg is USD 8.02,
so splicing them books an unquantified FX return (8.02/6.60 = +21.5%, wrong by ~32pp against
the true +53.6%).  Taking both legs off one line keeps the ratio inside one currency by
construction.

WHAT THAT DOES NOT FIX, STATED PLAINLY.  The resulting number is a return in the SUCCESSOR's
currency (USD for VMD) where an unbroken VMD.TO would have given a CAD return, and over
2021->2024 those differ by the CAD/USD move.  The panel's existing convention is already
"each line's return in its own currency, compared to a USD benchmark", so this stays inside
the convention rather than breaking it -- but the line changed currency mid-hold and that
residual is real.  It is NOT converted here: `fx_rates` could supply anchor spot rates, but
FX-converting ONE name in a panel where nothing else is converted would make this pick
inconsistent with the other ~99%.  Recorded, not papered over.

WHAT THIS TABLE CANNOT DO
-------------------------
  * IT CANNOT FIND THE LINES NOBODY ENTERED.  Coverage is exactly the rows below.  Every
    other discontinuity in the panel still books the old terminal/floor reading, and nothing
    here detects that a row is MISSING.  The 12 other vanished picks from the 2026-08-27
    post-mortem are not in this table because their outcomes were never settled.
  * IT CANNOT VERIFY ITS OWN ROWS.  `validate` checks the table's SHAPE (no duplicates, no
    self-reference, no chains) and `verify_against_grid` checks that the price grid is
    CONSISTENT with each row's claim.  Neither can tell you the corporate action actually
    happened that way -- that came from the 2026-08-27 settlement work, and the citation is
    in each row.
  * IT CANNOT DETECT A ROW THAT HAS GONE STALE.  If a future price refetch gives VMD.TO a
    2024 anchor price, the continuity path stops firing for that window (the eval leg exists,
    so the primitive never asks) -- which is the correct degradation -- but a row whose
    successor symbol was re-used by a different company would go on resolving silently.
    `entity_id.py` exists precisely because FMP recycles tickers; this table does not consult
    it.

NO I/O, NO NETWORK, NO PIPELINE IMPORTS.  Pure data plus four pure functions.
"""

#  ONE ROW PER DISCONTINUOUS LINE.
#
#  line            the symbol as it appears in the price grid and in the pick log
#  successor       the continuing line whose own return IS the holder's return, or None
#  event           the corporate action, in a few words
#  reason          why the successor is (or is not) the right line to measure
#  evidence        what on disk supports it -- checkable without an API call
#  source          who settled it and when
#  currency        (old -> new); "" when the line did not change currency
CONTINUITY_TABLE = [
    {
        "line": "VMD.TO",
        "successor": "VMD",
        "event": "re-domicile + exchange move (TSX -> Nasdaq)",
        "reason": (
            "Viemed Healthcare is ONE issuer with two listing lines. The TSX line stopped "
            "after 2022-12-30; the Nasdaq line continued uninterrupted. A holder was never "
            "dispossessed -- the shares became the continuing line share-for-share -- so "
            "VMD's own two-anchor return IS what the position returned."),
        "evidence": (
            "real_prices.csv: VMD.TO priced 2018-12-31..2022-12-30 (5.23, 8.13, 9.95, 6.60, "
            "10.27) and ABSENT at 2023-12-29 onward; VMD priced at EVERY anchor "
            "2018-12-31..2025-12-31 (incl. 5.22 @2021-12-31, 8.02 @2024-12-31 -> +53.64%). "
            "The two series overlap for five anchors, which is what shows they are the same "
            "issuer cross-listed rather than a ticker recycled after a death."),
        "source": "design/vanished-picks-settled-2026-08-27.md, Deliverable 5 (fmp-specialist)",
        "currency": "CAD -> USD",
    },
    {
        "line": "CMRE-PE",
        "successor": None,
        "event": "preferred-series redemption inside a living issuer",
        "reason": (
            "Costamare's 8.50% Series E preferred stopped pricing after 2023-12-29 at 24.24, "
            "just under its $25 par. The COMMON (CMRE) is priced through 2025-12-31 at 15.57 "
            "and is plainly trading, so the issuer did not die and -100% is refuted by the "
            "grid itself. But the common is NOT the successor: a called preferred pays CASH "
            "AT PAR, its holder does not receive common shares, and CMRE's return is a "
            "different security's return. The actual redemption price and date are in no "
            "artifact on disk. UNMEASURABLE, therefore INDETERMINATE -- not a loss, and not "
            "a guess at par-plus-accrued."),
        "evidence": (
            "real_prices.csv: CMRE-PE priced 2018-12-31..2023-12-29 (12.80, 17.72, 18.80, "
            "21.26, 21.81, 24.24), ABSENT at 2024-12-31 and 2025; CMRE priced at every "
            "anchor through 2025-12-31 (8.91 @2024-12-31, 15.57 @2025-12-31)."),
        "source": "design/vanished-picks-settled-2026-08-27.md, Deliverable 5 (fmp-specialist)",
        "currency": "",
    },
]

REQUIRED_FIELDS = ("line", "successor", "event", "reason", "evidence", "source", "currency")

#  Status written into the returns table for an identified discontinuity whose exit value
#  cannot be measured.  Distinct from 'terminal' ON PURPOSE: 'terminal' means "the eval leg
#  is missing and we do not know why", and its FLOOR reading of -100% is a defensible strict
#  bound for an unknown.  This means "the eval leg is missing and we know the line did not
#  die", where -100% is not a bound, it is a false statement.
STATUS_INDETERMINATE = "indeterminate"


def validate(table=None):
    """SHAPE checks on the table.  Raises ValueError; returns the table on success.

    Checks what code can check: every field present, no duplicate `line`, no row pointing at
    itself, and NO CHAINS (a successor that is itself a mapped line).  Chains are refused
    rather than resolved transitively because nobody has verified one and a two-hop
    resolution would be inventing a corporate history out of two rows written independently.
    """
    table = CONTINUITY_TABLE if table is None else table
    lines = set()
    for i, row in enumerate(table):
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError("continuity row %d (%r) missing field(s): %s"
                             % (i, row.get("line"), missing))
        line = row["line"]
        if not line or not isinstance(line, str):
            raise ValueError("continuity row %d has a non-string/empty `line`: %r" % (i, line))
        if line in lines:
            raise ValueError("continuity table has TWO rows for %r; one line, one successor"
                             % line)
        lines.add(line)
        succ = row["successor"]
        if succ is not None and (not isinstance(succ, str) or not succ):
            raise ValueError("continuity row %r has a non-string/empty successor: %r"
                             % (line, succ))
        if succ == line:
            raise ValueError("continuity row %r points at itself" % line)
        for field in ("reason", "evidence", "source"):
            if not str(row.get(field) or "").strip():
                raise ValueError("continuity row %r has an empty %s; every row must carry "
                                 "its own justification" % (line, field))
    for row in table:
        if row["successor"] in lines:
            raise ValueError(
                "continuity CHAIN refused: %r -> %r and %r is itself a mapped line. "
                "Resolve it to the final line in ONE row, with the evidence for the whole "
                "path." % (row["line"], row["successor"], row["successor"]))
    return table


def load(table=None):
    """{line: row} for the validated table.  Cheap; call it per run, not per pick."""
    return {row["line"]: row for row in validate(table)}


def _is_price(p):
    """The primitive's own unpriced rule, restated so the two cannot drift: None, 0 and NaN
    are all 'no price' (returns_core.compute_returns applies the same test to the buy leg)."""
    if p is None:
        return False
    try:
        p = float(p)
    except (TypeError, ValueError):
        return False
    return p == p and p != 0.0


def resolve(line, buy_date, eval_date, price_source, cmap):
    """What the continuity map says about `line` over [buy_date, eval_date].

    Returns None when the line is NOT mapped (the caller keeps its existing behaviour), else
    a dict:
      kind          'continued' | 'indeterminate'
      successor     the successor symbol, or None
      buy_px        successor's price at buy_date   (None when indeterminate)
      eval_px       successor's price at eval_date  (None when indeterminate)
      total_return  the successor line's own return, BOTH LEGS OFF THAT LINE (None when
                    indeterminate)
      note          a one-line audit string for the returns table's `continuity` column

    A mapped line whose successor cannot price BOTH anchors comes back 'indeterminate', NOT
    the old terminal reading.  That is deliberate and it is the only place this module makes
    a pick's PRIMARY number disappear: for a line we have positive evidence did not die,
    BOTH old readings are refusals of the same kind -- the FLOOR's -100% is contradicted by
    the evidence in the row, and PRIMARY's last-observed price is a substituted number 12 or
    24 months stale that `target_clauses.measured` already refuses to call an observation.
    Neither observes the chartered window, so the honest output is that there isn't one.
    """
    row = cmap.get(line)
    if row is None:
        return None
    succ = row["successor"]
    if succ:
        p_buy = price_source.price(succ, buy_date)
        p_eval = price_source.price(succ, eval_date)
        if _is_price(p_buy) and _is_price(p_eval):
            cur = (" [%s]" % row["currency"]) if row["currency"] else ""
            return {
                "kind": "continued",
                "successor": succ,
                "buy_px": float(p_buy),
                "eval_px": float(p_eval),
                "total_return": float(p_eval) / float(p_buy) - 1.0,
                "note": "%s->%s %s%s" % (line, succ, row["event"], cur),
            }
        unpriced = [d for d, p in ((buy_date, p_buy), (eval_date, p_eval))
                    if not _is_price(p)]
        why = "successor %s unpriced at %s" % (succ, " and ".join(unpriced))
    else:
        why = "no successor line: %s" % row["event"]
    return {"kind": "indeterminate", "successor": succ, "buy_px": None, "eval_px": None,
            "total_return": None, "note": "%s INDETERMINATE (%s)" % (line, why)}


def verify_against_grid(price_source, anchors=None, table=None):
    """Is the PRICE GRID consistent with what each row claims?  Report, never assert.

    For each row: the last anchor at which `line` is priced, and (for a successor row)
    whether the successor is priced at anchors the line is not.  A row claiming continuity
    whose successor prices nothing is a row that can only ever produce INDETERMINATE, and
    that is worth seeing before a run rather than after.

    THIS DOES NOT VERIFY THE CORPORATE ACTION.  A row can be perfectly consistent with the
    grid and still name the wrong successor; only the settlement work behind the row's
    `source` field speaks to that.
    """
    cmap = load(table)
    anchors = (list(anchors) if anchors is not None
               else list(getattr(price_source, "anchors", [])))
    out = []
    for line, row in cmap.items():
        line_anchors = [a for a in anchors if _is_price(price_source.price(line, a))]
        succ = row["successor"]
        succ_anchors = ([a for a in anchors if _is_price(price_source.price(succ, a))]
                        if succ else [])
        out.append({
            "line": line,
            "successor": succ or "",
            "event": row["event"],
            "line_priced_anchors": len(line_anchors),
            "line_last_priced": line_anchors[-1] if line_anchors else "",
            "successor_priced_anchors": len(succ_anchors),
            "successor_last_priced": succ_anchors[-1] if succ_anchors else "",
            "covers_the_gap": bool(succ_anchors) and bool(line_anchors)
                              and succ_anchors[-1] > line_anchors[-1],
        })
    return out
