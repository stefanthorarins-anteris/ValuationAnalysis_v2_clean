"""THE DECK PUBLISHES ONE P/E, ON THE CSV'S BASIS, AND SAYS WHICH  (register Q-91 S2).

THE DEFECT THESE TESTS PIN.  `generate_presentation.section_c_valuation` rendered one
`<td>P/E</td>` label over TWO DIFFERENT RATIOS: a name inside `AggScoreTop100` got that
CSV's `PE-ratio`, and a name outside it got `1 / earnYield` -- Stage-2's SIXTEEN-QUARTER
WINDOWED MEAN earnings yield.  Measured on the 2026-09-01 run, the two readings disagree by
a median of 0.87x and by as much as 1.93x on the same name (CRTO: 16.76 against 8.70), so
this was not a rounding difference -- it was two different facts wearing one word.

WHAT "FAILS ON THE UNFIXED CODE" MEANS HERE, because the house has 32 logged instances of a
test that pins the defect it covers or passes for the wrong reason.  The unfixed file has no
`build_pe_display` at all, so a test that only touched the new helper would fail with an
`AttributeError` -- a real failure, but one that proves nothing about the page.  So the
RENDER tests below go through `section_c_valuation`, which exists in BOTH versions, and
assert on the `<tr>` it emits.  On the unfixed file they fail because the cell says `2.50`
and the note column says `traded or yield inv`; on the fixed file they pass because it says
`3.81` and names the basis.  Verified by restoring the pre-change file byte-identical and
running this module against it (2026-09-03).

THE ANTI-DRIFT TEST IS THE IMPORTANT ONE.  `test_deck_cell_equals_the_csv_cell` does not
hard-code what a P/E is; it asserts the deck's cell EQUALS `postBo._pe_cell` on the same
panel frame.  A future change to either side that re-bases only one of them fails here,
which is the failure mode -- two definitions of one published number drifting apart -- that
this whole register item exists to remove.

OFFLINE.  Nothing here makes a network call: the panel fixtures are synthetic, and the one
artifact-backed test reads saved pickles/CSVs from the run dir and skips when they are not
on this machine.
"""

import os

import numpy as np
import pandas as pd
import pytest

import generate_presentation as gp
import postBo as pb


REPO = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive\pipeline"
RUN_DATE = "2026-09-01"


# =========================================================================== #
#  FIXTURES -- THE SHAPE OF THE REAL PANEL                                     #
# =========================================================================== #
#  The TNK numbers are the REAL ones off the saved run's panel (newest row), the same pair
#  `test_display_basis` uses, so BOTH readings that were in circulation are reproducible
#  from one fixture and neither is a number invented to make a test pass:
#
#      price (marketCap / weightedAverageShsOut) = 64.87      epsTTM = 17.012959
#      earningsYield                             = 0.100081   (per quarter, rpy = 4)
#
#      1 / (4 x 0.100081) = 2.4980   <- what the deck showed for an in-CSV name on 09-01
#      64.87 / 17.012959  = 3.8130   <- the one basis it shows now
_TNK_PRICE = 64.87
_TNK_EPSTTM = 17.012959
_TNK_PE_OLD = 2.4980
_TNK_PE_NEW = 3.8130


def _panel_rows(source, price, eps_ttm, n=8, refused='', ccy='USD'):
    """`n` quarterly rows for one source, NEWEST FIRST, on the real panel's schema.

    Newest-first deliberately: `_pe_ttm_panel_table` re-sorts by DATE rather than trusting
    arrival order, and a fixture that arrives pre-sorted the way the function wants would
    not exercise that.  The OLDER rows carry deliberately different numbers, so a helper
    that read the wrong row would produce a visibly wrong ratio rather than the right one.
    """
    rows = []
    for k in range(n):
        rows.append({
            'source': source,
            'date': pd.Timestamp('2026-04-01') - pd.DateOffset(months=3 * k),
            'reportedCurrency': ccy,
            #  the newest row carries the real pair; older rows are shifted so that reading
            #  one of them gives a DIFFERENT answer, not the same one.
            'price': price if k == 0 else price * 0.5,
            'epsTTM': eps_ttm if k == 0 else eps_ttm * 2.0,
            'marketCap': price * 1.0e6,
            'weightedAverageShsOut': 1.0e6,
            'netIncome': eps_ttm * 1.0e6 / 4.0,
            npol_col(): refused if k == 0 else '',
        })
    return rows


def npol_col():
    """The refusal-stamp column name, taken from `nan_policy` rather than spelled here."""
    import nan_policy as _npol
    return _npol.SANITY_REFUSED_COLUMN


@pytest.fixture
def panel():
    """A panel with one of each outcome the published P/E can have.

    TNK        a normal name -- our panel answers.
    LOSSY      intact panel, NEGATIVE trailing EPS -- a loss-maker, no meaningful P/E.
    REFUSD     `marketCap` refused by nan_policy section 5 on the newest row.  Note that
               `price` is left FINITE, which is the shape that matters: `price` is derived
               and stamped BEFORE the refusal runs, so a refused name really can carry a
               plausible-looking price, and a value-first implementation would publish a
               P/E built on an input the pipeline judged corrupt.
    """
    rows = []
    rows += _panel_rows('TNK', _TNK_PRICE, _TNK_EPSTTM)
    rows += _panel_rows('LOSSY', 41.0, -3.25)
    rows += _panel_rows('REFUSD', 88.0, 4.0, refused='marketCap')
    return pd.DataFrame(rows)


# =========================================================================== #
#  1.  THE DECK'S NUMBER **IS** THE CSV'S NUMBER                               #
# =========================================================================== #

def test_deck_cell_equals_the_csv_cell(panel):
    """The anti-drift pin: deck cell == `postBo._pe_cell` on the same frame, per name.

    Stated as an EQUALITY WITH THE PRODUCER rather than against a literal, so it fails if
    EITHER side is re-based independently.  A literal-only test would keep passing while the
    CSV moved out from under the deck, which is exactly how the two got out of step.
    """
    table, absent = gp.build_pe_display(panel)
    csv_table = pb._pe_ttm_panel_table(panel)
    csv_refused = pb._pe_refused_sources(panel)
    assert set(table) == set(csv_table)
    for src in csv_table:
        csv_cell, csv_basis = pb._pe_cell(pb._pe_ttm_ratio(csv_table, src), src, csv_refused)
        deck_value, deck_basis = table[src]
        assert deck_basis == csv_basis, src
        if csv_cell == 'NaN':
            assert deck_value is None, src
        else:
            assert deck_value == pytest.approx(float(csv_cell)), src
    assert absent == pb.PE_BASIS_UNAVAILABLE


def test_the_basis_is_ttm_and_not_the_earnyield_inverse(panel):
    """TNK reads 3.8130 (epsTTM / period-end price), NOT 2.4980 (1 / (rpy x earnYield)).

    Both numbers come off the SAME panel row, so this cannot pass by the fixture being thin:
    a helper that took the other reading would land on 2.4980 and fail here.
    """
    table, _ = gp.build_pe_display(panel)
    value, basis = table['TNK']
    assert value == pytest.approx(_TNK_PE_NEW, abs=1e-4)
    assert value != pytest.approx(_TNK_PE_OLD, abs=1e-2)
    assert basis == pb.PE_BASIS_TTM
    assert basis == 'epsTTM/periodEndPrice(reportedCurrency)'


# =========================================================================== #
#  2.  ABSENCE IS RENDERED, AND IT SAYS WHICH ABSENCE                          #
# =========================================================================== #

def test_a_loss_maker_and_a_refusal_are_different_facts(panel):
    """Both blank, both for DIFFERENT reasons, and the tokens say which.

    Folding them into one token would be the cheap fix and would lose the only thing the
    CEO can act on: "the pipeline judged this name's inputs to contradict each other" and
    "this company lost money over the trailing year" are not the same news.
    """
    table, _ = gp.build_pe_display(panel)
    assert table['LOSSY'] == (None, pb.PE_BASIS_NO_TTM_EPS)
    assert table['REFUSD'] == (None, pb.PE_BASIS_REFUSED)
    assert pb.PE_BASIS_NO_TTM_EPS != pb.PE_BASIS_REFUSED
    #  and each has a sentence for the page, not just a token for the spreadsheet
    for tok in (pb.PE_BASIS_NO_TTM_EPS, pb.PE_BASIS_REFUSED, pb.PE_BASIS_UNAVAILABLE):
        assert gp.PE_ABSENT_REASONS.get(tok), tok


def test_the_refusal_is_tested_before_the_value(panel):
    """REFUSD carries a FINITE price and a positive epsTTM, so the ratio computes.

    It must still publish nothing: the refusal lands on `marketCap`, which is what `price`
    is derived from, so a value-first order would ship a number built on an input the
    pipeline already judged impossible and label it as ours.  This is the ordering bug that
    was live until 2026-09-03, and it is observable only because `price` survives the
    refusal -- which is why the fixture leaves it finite.
    """
    assert pb._pe_ttm_ratio(pb._pe_ttm_panel_table(panel), 'REFUSD') is not None
    table, _ = gp.build_pe_display(panel)
    assert table['REFUSD'][0] is None


def test_a_name_outside_the_panel_is_unavailable_not_a_number(panel):
    """A name the panel has never heard of gets the CSV's `unavailable` token."""
    table, absent = gp.build_pe_display(panel)
    assert 'NOSUCH' not in table
    assert absent == 'unavailable'


def test_an_unreadable_panel_reports_a_deck_failure_not_a_universe_fact():
    """No `price`/`epsTTM` -> `panel-unreadable`, which is a DIFFERENT token from `unavailable`.

    A wholesale failure that printed as N individually-unavailable names would read as a
    fact about those companies.  It is a fact about this deck.  `postBo` learned to make the
    same distinction in its own log ("THE WHOLE COLUMN IS EMPTY"); the deck says it in the
    cell.
    """
    empty = pd.DataFrame({'source': ['A'], 'date': [pd.Timestamp('2026-04-01')]})
    table, absent = gp.build_pe_display(empty)
    assert table == {}
    assert absent == gp.PE_BASIS_PANEL_UNREADABLE
    assert absent != 'unavailable'
    assert gp.PE_ABSENT_REASONS.get(absent)


def test_pe_published_never_substitutes_another_quantity(panel):
    """The accessor answers `(None, reason)` for an unknown name -- it does NOT reach for
    `earnYield`, or for anything else that happens to be computable.

    `raw_metric` is wired to return a perfectly good earnings yield here.  The OLD code
    would have inverted it and printed a P/E; a different quantity under the same label is
    worse than a blank, so the correct answer is the blank.
    """
    table, absent = gp.build_pe_display(panel)
    builder = object.__new__(gp.PresentationBuilder)
    builder.data = {'pe_display': table, 'pe_absent_basis': absent}
    builder.raw_metric = lambda t, m: 0.08 if m == 'earnYield' else np.nan   # 1/0.08 = 12.5
    assert builder.pe_published('TNK')[0] == pytest.approx(_TNK_PE_NEW, abs=1e-4)
    assert builder.pe_published('NOSUCH') == (None, 'unavailable')


# =========================================================================== #
#  3.  THE RENDERED PAGE  -- the tests that fail on the unfixed file           #
# =========================================================================== #
#  These go through `section_c_valuation`, which exists in both versions, so their failure
#  on the pre-change file is a statement about the PAGE and not about a missing symbol.

_HAVE_RUN = os.path.isdir(RUN_DIR) and bool(
    [f for f in os.listdir(RUN_DIR) if f.startswith('postRank_%s' % RUN_DATE)]
    if os.path.isdir(RUN_DIR) else [])

requires_run = pytest.mark.skipif(
    not _HAVE_RUN, reason="saved %s run not on this machine" % RUN_DATE)


@pytest.fixture(scope='module')
def builder():
    """One real builder off the saved run.  Module-scoped: loading the run's pickles is the
    expensive part and every render test wants the same one.  `augment=False` -> no Yahoo
    fetch, no HTTP client imported, zero network."""
    if not _HAVE_RUN:
        pytest.skip("saved %s run not on this machine" % RUN_DATE)
    data = gp.load_run_data(RUN_DIR, gp.VALUATION_REPO, run_date=RUN_DATE)
    return gp.PresentationBuilder(data, augment=False)


def _pe_row(html):
    """The `<tr>` for the P/E, as (value-cell, note-cell) with tags stripped."""
    import re
    m = re.search(r'<td><strong>P/E</strong></td><td>(.*?)</td><td>(.*?)</td>', html, re.S)
    assert m, "the page has no P/E row at all"
    strip = lambda t: ' '.join(re.sub(r'<[^>]+>', '', t).split())
    return strip(m.group(1)), strip(m.group(2))


@requires_run
def test_rendered_page_shows_the_ttm_pe_not_the_csv_column(builder):
    """TNK's page reads 3.81, and the note column NAMES the basis.

    ON THE UNFIXED FILE THIS FAILS TWICE, and both failures are about what the CEO sees:
    the value cell reads `2.50` (the 2026-09-01 CSV predates the P/E re-basing and still
    carries the superseded column) and the note cell reads `traded or yield inv`, which
    names two possible bases and commits to neither.
    """
    value, note = _pe_row(builder.section_c_valuation('TNK', 'General'))
    assert value.startswith('3.81'), value
    assert '2.50' not in value, value
    assert note.startswith('epsTTM/periodEndPrice(reportedCurrency)'), note
    assert 'traded or yield inv' not in note


@requires_run
def test_every_rendered_pe_on_the_deck_declares_one_basis(builder):
    """Across every page the deck renders, the P/E note is the SAME basis token (or a named
    absence).  This is the register item stated directly: one label, one basis.

    It walks the real page plan rather than a chosen name, because the defect was invisible
    on any single page -- it only existed BETWEEN pages.
    """
    #  BUILT WITHOUT REFERENCE TO THE NEW SYMBOLS, deliberately.  Naming
    #  `gp.PE_ABSENT_REASONS` here would make this test fail on the pre-change file with an
    #  `AttributeError` -- a failure that says nothing about the page.  Taking the tokens
    #  from `postBo` (which both versions import fine) makes it fail on the unfixed file for
    #  the reason that matters: the note column says `traded or yield inv`.
    allowed = {pb.PE_BASIS_TTM, pb.PE_BASIS_REFUSED, pb.PE_BASIS_NO_TTM_EPS,
               pb.PE_BASIS_UNAVAILABLE, 'panel-unreadable'}
    seen, pages = set(), 0
    for ticker, cohort in _page_list(builder):
        html = builder.section_c_valuation(ticker, cohort)
        if '<td><strong>P/E</strong></td>' not in html:
            continue        # this cohort's Section C carries no P/E row at all
        pages += 1
        _value, note = _pe_row(html)
        token = note.split()[0] if note else ''
        assert any(note.startswith(a) or token == a for a in allowed), (ticker, note)
        seen.add(note.split('(')[0] if note.startswith('epsTTM') else note)
    assert pages > 0, "no page rendered a P/E -- the test proved nothing"
    assert len(seen) == 1, "more than one basis is on the deck: %r" % (seen,)


@requires_run
def test_an_absent_pe_renders_a_reason_and_no_number(builder):
    """Drop TNK from the resolved table and the cell must say WHY, not print an em-dash and
    not fall back to anything.  An unexplained blank reads as a broken tool -- the same
    argument that put `M_abstain_reason` on the page."""
    original = builder.data['pe_display']
    try:
        builder.data['pe_display'] = {k: v for k, v in original.items() if k != 'TNK'}
        builder._eval_cache = {}
        value, note = _pe_row(builder.section_c_valuation('TNK', 'General'))
        assert gp.PE_ABSENT_REASONS['unavailable'] in value, value
        assert '3.81' not in value and '2.50' not in value, value
        assert note.startswith('unavailable'), note
    finally:
        builder.data['pe_display'] = original
        builder._eval_cache = {}


def _page_list(builder):
    """(ticker, cohort) for every page the deck plans, from the run's own artifacts."""
    data = builder.data
    out = [(t, 'General') for t in list(data['postrank_df']['source'].head(20))]
    for label, df in (data.get('carveout_sidelists') or {}).items():
        frame = df.get('postRank') if isinstance(df, dict) else df
        if frame is None or getattr(frame, 'empty', True):
            continue
        out += [(t, label) for t in list(frame['source'].head(5))]
    return out
