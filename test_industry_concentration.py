"""THE ENFORCEMENT for the industry counter (CEO, 2026-08-04).

The counter reports how many of the top-100 / top-20 sit in each FMP industry -- the 07-17
corrected list holds 11 Marine Shipping of 100 and 7 of 20, and no deliverable said so.  It is
INFORMATIONAL ONLY: the standing ruling is that there are no hard gates in the filtering logic,
so a concentration is a fact for the CEO to weigh, never an input to a score, a rank, a
membership, a band or a carve-out.

WHAT IS PINNED HERE, and why each is a defect if it fails:

  1. IT WRITES NOTHING AND MUTATES NOTHING.  The frames that get pickled and published are
     bit-identical across a counter call, and the counter opens no file and writes no CSV.
     "Informational" is a property that rots silently, so it is asserted rather than asserted-in-
     a-comment.  (Full-run bit-identity of the AggScore CSV cannot be tested here -- that path
     makes ~500 live FMP calls -- but the counter runs AFTER every deliverable is written and
     never receives the AggScore frame, and item 1 + item 4 are what make that structural claim
     checkable.)
  2. ONE COUNT IN THE REPO.  The pipeline run log, the deck and `baseline_tools` all resolve to
     the same `industry_counts`.  A second implementation that drifted would put a different
     shipping number in the deck than in the log.
  3. UNCLASSIFIED IS ALWAYS REPORTED AND NEVER WINS.  A report that silently drops the names it
     could not label is worse than none; equally, 6 unlabelled names are a DATA GAP and must not
     be reported as "6 of 20 in ONE industry" or trip the concentration warning.
  4. THE COUNTED TOP-20 IS THE PRESENTED TOP-20.  `generalTopN` is the single selection shared
     by the deck, the XLSX and the counter, and it is byte-identical to the inline logic it
     replaced in `createPresentation` on all four branches.
  5. THE REAL ARTIFACT.  On the saved 2026-07-17 corrected list the counter must report exactly
     11 Marine Shipping in the top-100 and 7 in the top-20 -- the numbers the CEO quoted.

No network, no API key.  Run it the repo way: `pytest . --ignore=baseline_tools`.
"""
import os

import pandas as pd
import pytest

import industry_concentration as ic
import postBo as pb


# --------------------------------------------------------------------------- #
#  Fixtures                                                                   #
# --------------------------------------------------------------------------- #
def _ind():
    """A small symbol -> industry map with every shape that breaks something: a heavy
    concentration, a cycle-spanning second industry, a comma-bearing label, a symbol PRESENT
    with a null label, and symbols absent from the map entirely."""
    m = {}
    for i in range(7):
        m['SHIP%d' % i] = 'Marine Shipping'
    for i in range(3):
        m['OIL%d' % i] = 'Oil & Gas Exploration & Production'
    m['FURN0'] = 'Furnishings, Fixtures & Appliances'
    m['SOFT0'] = 'Software - Application'
    m['NULL0'] = None                     # present-but-unlabelled: must read as unclassified
    return m                              # 'GONE0' / 'GONE1' are absent on purpose


def _top20():
    return (['SHIP%d' % i for i in range(7)] + ['OIL%d' % i for i in range(3)]
            + ['FURN0', 'SOFT0', 'NULL0', 'GONE0'])


def _postrank(n=40):
    """A postRank-shaped frame: 'source' + AggScore + the ROR column, ordered."""
    src = _top20() + ['X%03d' % i for i in range(n - len(_top20()))]
    return pd.DataFrame({'source': src,
                         'AggScore': [1.0 - 0.01 * i for i in range(len(src))],
                         'marketCap_usd': [1e9] * len(src)})


# --------------------------------------------------------------------------- #
#  1. IT WRITES NOTHING AND MUTATES NOTHING                                   #
# --------------------------------------------------------------------------- #
def test_counter_leaves_the_published_frames_bit_identical():
    """The ranking frame the pipeline pickles and publishes must be untouched by reporting."""
    ind = _ind()
    fb = _postrank()
    before = fb.copy(deep=True)
    universe = sorted(set(fb['source']) | {'ZZZ0', 'ZZZ1'})

    ic.report_lines(list(fb['source'].head(100)), list(fb['source'].head(20)),
                    universe_sources=universe, ind=ind)
    ic.counter_block(list(fb['source'].head(20)), 'top-20', ind=ind)
    ic.concentration_line(list(fb['source'].head(20)), universe, ind=ind)
    pb.generalTopN(fb, None, 20, warn=False)

    pd.testing.assert_frame_equal(fb, before, check_exact=True)
    # the industry map is an input too -- a reporting pass must not backfill labels into it
    assert ind == _ind()


def test_counter_opens_no_file_and_writes_no_csv(monkeypatch, tmp_path):
    """Informational means informational: no artifact, no side-channel, not even a temp file."""
    ind = _ind()
    fb = _postrank()

    def _boom_open(*a, **k):
        raise AssertionError('the industry counter opened a file: %r' % (a,))

    def _boom_csv(*a, **k):
        raise AssertionError('the industry counter wrote a CSV')

    monkeypatch.setattr('builtins.open', _boom_open)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', _boom_csv)
    monkeypatch.setattr(pd.Series, 'to_csv', _boom_csv)

    lines = ic.report_lines(list(fb['source'].head(100)), list(fb['source'].head(20)),
                           universe_sources=sorted(set(fb['source'])), ind=ind)
    assert lines and all(isinstance(l, str) for l in lines)


def test_the_block_says_it_is_not_a_gate():
    """The reader of a run log must be told the number is inert; a bare tally invites the
    inference that the pipeline acted on it."""
    text = "\n".join(ic.report_lines(_top20(), _top20()[:20],
                                     universe_sources=_top20(), ind=_ind()))
    assert 'INFORMATIONAL ONLY' in text
    assert 'Not a gate' in text


# --------------------------------------------------------------------------- #
#  2. ONE COUNT IN THE REPO                                                   #
# --------------------------------------------------------------------------- #
def test_baseline_tools_reexports_the_same_objects():
    """`baseline_tools/industry_attribution.py` used to OWN this code; it must now be the same
    function object, not a copy that can drift."""
    import importlib
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'baseline_tools'))
    try:
        ia = importlib.import_module('industry_attribution')
    except Exception as e:                       # pragma: no cover - heavy optional deps
        pytest.skip('industry_attribution unavailable: %s: %s' % (type(e).__name__, e))
    assert ia.concentration_line is ic.concentration_line
    assert ia.industry_counts is ic.industry_counts
    assert ia.cycle_of is ic.cycle_of
    assert ia.CYCLE_CLUSTERS is ic.CYCLE_CLUSTERS


def test_deck_and_run_log_count_identically():
    """The deck banner and the run-log block must resolve to one count. Both call
    `ic.industry_counts`, so this pins that the deck has not grown its own tally."""
    import generate_presentation as gp
    ind = _ind()
    fb = _postrank()
    html = gp.industry_counter_banner(fb, list(fb['source'].head(20)), ind, cdx_df=fb)
    named, n_unc, n_tot = ic.industry_counts(list(fb['source'].head(20)), ind)
    assert 'INDUSTRY COUNTER' in html
    # every counted industry, with its count, appears in the rendered block
    for name, k in named:
        assert '<b>%d</b>' % k in html
    assert 'Marine Shipping' in html
    assert 'unclassified <b>%d</b>' % n_unc in html
    assert n_tot == 20


def test_deck_banner_degrades_loudly_without_an_industry_map():
    """No map is a DATA GAP the reader must be told about -- not an empty block."""
    import generate_presentation as gp
    html = gp.industry_counter_banner(_postrank(), _top20(), {}, cdx_df=None)
    assert 'unavailable' in html and 'INDUSTRY COUNTER' in html


# --------------------------------------------------------------------------- #
#  3. UNCLASSIFIED IS ALWAYS REPORTED AND NEVER WINS                          #
# --------------------------------------------------------------------------- #
def test_unclassified_counted_not_dropped():
    """NULL0 is present in the map with a null label; GONE0 is absent from it. Both are
    unclassified, and neither may become an industry named 'None'."""
    named, n_unc, n_tot = ic.industry_counts(_top20(), _ind())
    assert n_tot == 14
    assert n_unc == 2                              # NULL0 + GONE0
    assert sum(k for _, k in named) + n_unc == n_tot
    assert all(isinstance(name, str) and name not in ('None', ic.UNKNOWN)
               for name, _ in named)


def test_unclassified_reported_even_when_zero():
    """A silent 0 is indistinguishable from a forgotten line."""
    lines = ic.counter_block(['SHIP0', 'SOFT0'], 'top-2', ind=_ind())
    assert any('unclassified' in l and l.rstrip().endswith('0') for l in lines)


def test_unclassified_never_wins_the_one_industry_line_or_the_warning():
    """9 unlabelled names out of 10 is a data gap, not a 90% concentration."""
    top = ['GONE%d' % i for i in range(9)] + ['SOFT0']
    line = ic.concentration_line(top, ['SOFT0', 'SHIP0'], ind=_ind())
    assert 'in ONE industry (Software - Application)' in line
    assert '9 of 10 UNCLASSIFIED' in line
    assert 'ONE INDUSTRY' not in line               # the 25% warning must NOT fire on UNKNOWN


def test_counts_are_descending_and_tie_stable():
    named, _, _ = ic.industry_counts(_top20(), _ind())
    assert [k for _, k in named] == sorted([k for _, k in named], reverse=True)
    assert named[0] == ('Marine Shipping', 7)
    # same multiset, different order in -> identical output (ties broken alphabetically, not by
    # rank, so diffing two run logs is not noise)
    assert ic.industry_counts(list(reversed(_top20())), _ind()) == \
        ic.industry_counts(_top20(), _ind())


def test_singletons_are_pipe_separated_because_labels_contain_commas():
    """'Furnishings, Fixtures & Appliances' comma-joined with its neighbours reads as three
    industries. Every singleton is still named in full -- nothing is truncated away."""
    lines = ic.counter_block(_top20(), 'top-14', ind=_ind())
    text = "\n".join(lines)
    assert 'Furnishings, Fixtures & Appliances' in text
    assert ' | ' in text
    assert 'Software - Application' in text


def test_warning_fires_on_concentration_and_is_only_text():
    line = ic.concentration_line(_top20(), sorted(set(_top20()) | {'Z%d' % i
                                                                  for i in range(200)}),
                                 ind=_ind())
    assert 'ONE INDUSTRY (Marine Shipping)' in line
    assert isinstance(line, str)


def test_headings_follow_the_requested_depth_not_a_literal():
    """`-ntopagg 50` must not print a 'top-100' caption over 50 names, and the ACTUAL name count
    must be stated separately so a ranking shallower than the requested depth is visible."""
    fb = _postrank(40)
    text = "\n".join(ic.report_lines(list(fb['source'].head(50)), list(fb['source'].head(10)),
                                     universe_sources=None, ind=_ind(),
                                     labels=('top-50', 'top-10')))
    assert 'top-50 (40 names' in text          # requested 50, ranking is only 40 deep
    assert 'top-10 (10 names' in text
    assert 'top-100' not in text


def test_empty_list_is_stated_not_crashed():
    assert 'empty' in ic.counter_block([], 'top-20', ind=_ind())[0]
    assert 'empty' in ic.concentration_line([], ['SHIP0'], ind=_ind())


# --------------------------------------------------------------------------- #
#  4. THE COUNTED TOP-20 IS THE PRESENTED TOP-20                              #
# --------------------------------------------------------------------------- #
def _generalTopN_preRefactor(finalBoRank_df, bands, topn, printed):
    """The logic that lived inline in createPresentation before 2026-08-04, verbatim."""
    _general_df = finalBoRank_df
    if bands and not bands.get('currency_pending', True):
        _gb = (bands.get('bands') or {}).get('General')
        if _gb is not None and not _gb.empty:
            if topn > len(_gb):
                printed.append('warn')
            else:
                _general_df = _gb
    return _general_df


@pytest.mark.parametrize('bands, topn', [
    (None, 20),                                                    # no banding at all
    ({'currency_pending': True, 'bands': {}}, 20),                 # currency pending
    ({'currency_pending': False, 'bands': {}}, 20),                # banded, no General
    ({'currency_pending': False, 'bands': {'General': 'EMPTY'}}, 20),   # empty General
    ({'currency_pending': False, 'bands': {'General': 'BAND'}}, 20),    # the live case
    ({'currency_pending': False, 'bands': {'General': 'BAND'}}, 25),    # topn > band size
])
def test_generalTopN_matches_the_inline_logic_it_replaced(bands, topn, capsys):
    fb = _postrank()
    band_df = fb.head(20).copy()
    if bands and 'General' in (bands.get('bands') or {}):
        token = bands['bands']['General']
        bands = {**bands, 'bands': {'General': (band_df if token == 'BAND'
                                                else fb.iloc[0:0].copy())}}
    printed = []
    expect = _generalTopN_preRefactor(fb, bands, topn, printed)
    got = pb.generalTopN(fb, bands, topn)
    pd.testing.assert_frame_equal(got, expect, check_exact=True)
    # the LOUD warning survives the extraction, and only on the shrink case
    assert ('WARNING' in capsys.readouterr().out) is bool(printed)


def test_generalTopN_warn_false_is_silent_but_selects_identically(capsys):
    fb = _postrank()
    bands = {'currency_pending': False, 'bands': {'General': fb.head(20).copy()}}
    loud = pb.generalTopN(fb, bands, 25, warn=True)
    assert 'WARNING' in capsys.readouterr().out
    quiet = pb.generalTopN(fb, bands, 25, warn=False)
    assert capsys.readouterr().out == ''
    pd.testing.assert_frame_equal(loud, quiet, check_exact=True)


def test_counter_counts_the_band_general_top20_when_banded():
    """When currency data flows, the presented top-20 is the General band -- and so is the
    counted one. A counter that kept counting postRank.head(20) would describe a list nobody
    sees."""
    fb = _postrank(40)
    band = fb[fb['source'].str.startswith(('OIL', 'SOFT', 'X'))].head(20).copy()
    bands = {'currency_pending': False, 'bands': {'General': band}}
    counted = list(pb.generalTopN(fb, bands, 20, warn=False)['source'].head(20))
    assert counted == list(band['source'])
    named, _, _ = ic.industry_counts(counted, _ind())
    assert dict(named).get('Marine Shipping') is None      # the band excluded them


# --------------------------------------------------------------------------- #
#  5. THE REAL ARTIFACT -- the numbers the CEO quoted                         #
# --------------------------------------------------------------------------- #
_CORRECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_tools',
                          'CORRECTED_general_top100-2026-07-17-CORRECTED.csv')


@pytest.mark.skipif(not os.path.exists(_CORRECTED),
                    reason='saved 2026-07-17 corrected top-100 not present on this machine')
def test_real_corrected_top100_holds_eleven_marine_shipping():
    ind = ic.industry_map()
    if not ind:
        pytest.skip('no industrydic_fmp_*.pickle on this machine')
    src = list(pd.read_csv(_CORRECTED)['source'])
    n100, unc100, tot100 = ic.industry_counts(src[:100], ind)
    n20, unc20, tot20 = ic.industry_counts(src[:20], ind)
    assert (tot100, tot20) == (100, 20)
    assert dict(n100)['Marine Shipping'] == 11
    assert dict(n20)['Marine Shipping'] == 7
    assert n100[0] == ('Marine Shipping', 11)          # it is the TOP industry, not just present
    assert (unc100, unc20) == (0, 0)                   # ~100% industry coverage on this list
