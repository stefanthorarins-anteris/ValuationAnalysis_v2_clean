"""The 2026-08-13 REPORTING EXPANSION: currency, traded value, coverage, ejections.

WHAT THIS FILE PINS, and why each one needs pinning rather than trusting the code to stay
right.  All four defects below were DISPLAY defects -- none of them moved a rank -- which is
exactly why nothing already in the suite caught them: every existing guard watches the
scoring path.  The shortlist display is what the CEO reads to choose what to investigate, so
a wrong number here is acted on even though nothing computed it wrongly.

  N-3  the display columns MIXED TWO CURRENCIES.  `GrahamNumberToPrice` was
       `key-metrics.grahamNumber` (reportedCurrency) over `profile.price` (trading
       currency), with no FX.  Measured on the shipped 2026-08-13 top-100: SKHY published
       6763.0518 and SHEL.L 0.0113, against a true panel-computed range of 0.074-4.75.
       `price` had the same disease with no currency column beside it, and the HTML deck
       printed both `price` and `marketCap` with a hard-coded `$`.
  N-3b `dividendYield` was ONE QUARTER's vendor yield published under a name every reader
       takes as annual (median true/published ratio 3.09 across the shipped payers).
  N-4  a name could be ~93% imputed, sit at rank 55, be tagged `clean`, and NOTHING said so.
  N-5  the Stage-1 veto -- 573 of 1,404 general-pool names on 2026-08-13 -- shipped no list
       of what it removed.

EVERY TEST HERE FAILS AGAINST THE PRE-FIX CODE.  The structural ones assert the absence of
the exact expressions that produced the wrong numbers; the behavioural ones call functions
that did not exist.  The numeric constants are MEASURED off the shipped 2026-08-13 run and
are cited as such, so a future reader can re-derive them rather than trust them.
"""
import io
import inspect
import os

import numpy as np
import pandas as pd
import pytest

import carveOut as co
import generate_presentation as gp
import postBo as pb
import reporting_period as rp
import stage1_veto as sv


# =========================================================================== #
#  FIXTURES -- synthetic panels reproducing the SHAPES of the real named cases #
# =========================================================================== #
#  Deliberately synthetic rather than reading the shipped run off disk: the pipeline data
#  lives on one machine's Drive mount and a test that silently skips when it is absent is a
#  test that protects nothing.  The NUMBERS are the real ones (2026-08-13 panel), so the
#  shapes are not invented -- only the frame around them is.

def _panel(rows):
    """A minimal cdx_df: the columns the display helpers read, nothing else."""
    return pd.DataFrame(rows)


def _code_only(fn):
    """`inspect.getsource(fn)` with COMMENT LINES **and DOCSTRINGS** removed.

    Necessary, not fastidious: this repo documents a removed defect by QUOTING it in the prose
    that replaces it -- which is the right thing for a reader and fatal for a substring
    assertion, because the guard then finds its own explanation and reports the bug as
    present.  It has now happened twice: once in a comment (`f"${price:.2f}"`) and once in a
    docstring (`_page_tickers`), so BOTH kinds of prose are stripped rather than just the one
    that happened to bite first.

    A deliberately simple triple-quote state machine: these are ordinary functions with
    ordinary docstrings, and a real Python tokeniser here would be more machinery than the
    assertion is worth.
    """
    out, in_doc = [], False
    for ln in inspect.getsource(fn).splitlines():
        st = ln.strip()
        if in_doc:
            if '"""' in st:
                in_doc = False
            continue
        if st.startswith('#'):
            continue
        if st.startswith('"""') or st.startswith('r"""'):
            body = st.split('"""', 1)[1]
            if '"""' not in body:            # a one-line docstring closes on its own line
                in_doc = True
            continue
        out.append(ln)
    return '\n'.join(out)


@pytest.fixture
def cross_listed_panel():
    """SKHY / SHEL.L / TNK -- the three currency shapes that broke the Graham column.

    SKHY  reports KRW, quotes USD  -> vendor ratio was 6763x too high
    SHEL.L reports USD, quotes GBp -> vendor ratio was 88x too low
    TNK   reports USD, quotes USD  -> vendor ratio was already right (the control)
    """
    rows = []
    for src, cur, graham, price in (
            #  Panel values off the 2026-08-13 newest row for each name.
            ('SKHY', 'KRW', 236466.0, 80700.0),          # -> 2.9302
            ('SHEL.L', 'USD', 58.5091, 38.8857),         # -> 1.5046
            ('TNK', 'USD', 201.0, 80.66)):               # -> 2.4919 (shape only)
        for i, d in enumerate(pd.date_range('2025-04-01', periods=4, freq='QS')[::-1]):
            rows.append({'source': src, 'date': d, 'reportedCurrency': cur,
                         'grahamNumber': graham, 'price': price,
                         'marketCap': price * 1e9, 'marketCap_usd': np.nan,
                         'dividendsPaid': -1.0e7,
                         rp.FREQ_COLUMN: rp.QUARTERLY})
    return _panel(rows)


@pytest.fixture
def dividend_panel():
    """SHEL.L's real 2026-08-13 shape: four quarters of dividends against one market cap.

    Chosen because it is the case where the two readings differ MOST informatively -- the
    published per-quarter number was 0.9957% and the real trailing yield is 3.87%, i.e. the
    column understated Shell's income by ~4x while looking entirely plausible.
    """
    mcap = 2.173e11
    rows = []
    for i, d in enumerate(pd.date_range('2025-07-01', periods=4, freq='QS')[::-1]):
        rows.append({'source': 'SHEL.L', 'date': d, 'reportedCurrency': 'USD',
                     'marketCap': mcap, 'dividendsPaid': -mcap * 0.009663,
                     'grahamNumber': 58.5, 'price': 38.9,
                     rp.FREQ_COLUMN: rp.QUARTERLY})
    return _panel(rows)


@pytest.fixture
def profile_map():
    """A volavgdic profile map: the real 2026-08-13 entries for the named cases."""
    return {
        'SHEL.L': {'volAvg': 9288640, 'asof': '2026-08-13', 'price': 3322.5,
                   'currency': 'GBp'},
        'TNK': {'volAvg': 367467, 'asof': '2026-08-13', 'price': 80.89,
                'currency': 'USD'},
        '0QQF.L': {'volAvg': 1200, 'asof': '2026-08-13', 'price': 5.23,
                   'currency': 'GBp'},                       # the thin-market case
        'NOPRICE': {'volAvg': 5000, 'asof': '2026-08-13', 'price': None,
                    'currency': 'USD'},
        'NOCCY': {'volAvg': 5000, 'asof': '2026-08-13', 'price': 10.0,
                  'currency': None},                         # a pre-capture entry
        'BADCCY': {'volAvg': 5000, 'asof': '2026-08-13', 'price': 10.0,
                   'currency': 'ZZZ'},
    }


LIVE_FX = {'USD': 1.0, 'GBp': 0.0134987, 'KRW': 0.0007068736, 'CAD': 0.71743}


# =========================================================================== #
#  N-3a  GrahamNumberToPrice: FX-free by construction, never the vendor ratio  #
# =========================================================================== #

def test_the_vendor_FX_MIXED_graham_ratio_is_GONE_from_the_CSV_writer():
    """The exact expression that produced SKHY = 6763.0518 must not be in the source.

    STRUCTURAL, and that is the point: the value defect is invisible on a US-only sample --
    every control name (TNK, STRT) was already correct -- so a numeric test on a plausible
    fixture can pass while the bug is fully present.  What has to be pinned is that the
    numerator and denominator no longer come from two different currencies' worth of vendor
    JSON.
    """
    src = _code_only(pb.writeBoAggToCSV)
    assert "temp_resp_km[0]['grahamNumber']/temp_resp_pr[0]['price']" not in src, (
        "the published GrahamNumberToPrice is dividing a reportedCurrency grahamNumber by a "
        "trading-currency profile price again -- this is the defect that shipped SKHY at "
        "6763x and SHEL.L at 0.0113x on the same run")
    assert "'grahamNumber'" not in src, (
        "writeBoAggToCSV is reading the vendor's grahamNumber field again; the published "
        "ratio must come from the pipeline's own panel/metric, not from key-metrics")


def test_graham_to_price_is_FX_FREE_on_every_cross_listing(cross_listed_panel):
    """The panel ratio is unaffected by the trading currency, which is the whole fix.

    SKHY (KRW statements, USD quote) and SHEL.L (USD statements, GBp quote) both land in the
    same order of magnitude as the USD-only control, because numerator and denominator are
    now both statement-currency quantities off one row.
    """
    got = pb._graham_to_price_panel_latest(cross_listed_panel)
    assert got['SKHY'] == pytest.approx(2.9302, rel=1e-3)
    assert got['SHEL.L'] == pytest.approx(1.5046, rel=1e-3)
    #  The whole top-100 lands in a single order of magnitude once the currencies agree.
    #  Measured on the shipped 2026-08-13 panel: 0.0694 .. 4.8962.
    assert all(0.01 < v < 100 for v in got.values())


def test_the_published_graham_prefers_the_SCORED_metric_and_NAMES_its_fallback(
        cross_listed_panel):
    """Primary = the number that actually ranked the name; fallback = the panel's newest row.

    Publishing a DIFFERENT number under a scored metric's own name is how `CycleHeat` came to
    ship a column whose correlation with the truth was -1.0000, so the primary basis is the
    literal scored value.  The fallback exists because Stage-2 imputes a metric it cannot
    window (STRT and PET.TO on the shipped run), and a blank there is less useful than a real
    newest-row reading -- but the caller has to be able to SAY which names those are.
    """
    raw = pd.DataFrame({'source': ['SKHY', 'SHEL.L', 'TNK'],
                        'grahamNumberToPrice': [3.9909, 1.4749, np.nan]})
    vals, basis = pb._graham_to_price_published(raw, cross_listed_panel)
    assert vals['SKHY'] == pytest.approx(3.9909)
    assert basis['SKHY'] == pb.GRAHAM_BASIS_SCORED
    #  TNK's scored value is NaN here -> the panel answers, and says so.
    assert basis['TNK'] == pb.GRAHAM_BASIS_PANEL
    assert vals['TNK'] == pytest.approx(2.4919, rel=1e-3)


def test_the_published_graham_NEVER_falls_back_to_the_vendor():
    """With neither a scored value nor a panel, the answer is ABSENT -- not the vendor's.

    The vendor's number IS the defect, so a "better than nothing" fallback to it would
    restore the bug for precisely the names we know least about.
    """
    vals, basis = pb._graham_to_price_published(None, None)
    assert vals == {} and basis == {}


# =========================================================================== #
#  N-3b  dividendYield: trailing twelve months, computed, not a vendor quarter #
# =========================================================================== #

def test_the_vendor_QUARTERLY_dividend_yield_is_GONE_from_the_CSV_writer():
    src = _code_only(pb.writeBoAggToCSV)
    assert "['dividendYield']*100" not in src, (
        "the published dividendYield is again `key-metrics?period=quarter.dividendYield * "
        "100`, i.e. ONE QUARTER's yield under a name every reader takes as annual")


def test_dividend_yield_is_a_TRAILING_YEAR_not_one_quarter(dividend_panel):
    """~3.87% for SHEL.L, not the ~0.97% the per-quarter vendor field published."""
    got = pb._dividend_yield_ttm_from_panel(dividend_panel)
    assert got['SHEL.L'] == pytest.approx(3.865, rel=1e-2)


def test_a_semiannual_filer_sums_TWO_rows_for_a_year_not_four():
    """rpy-aware, like every other flow quantity here.

    Summing 4 rows for an H1/H2 filer builds a TWENTY-FOUR month 'annual' dividend and
    doubles the reported yield -- the same defect class `stamp_frequency_and_graham` fixed
    for EPS.
    """
    mcap = 1.0e9
    rows = [{'source': 'SEMI', 'date': d, 'marketCap': mcap,
             'dividendsPaid': -1.0e7, 'reportedCurrency': 'EUR',
             rp.FREQ_COLUMN: rp.SEMIANNUAL}
            for d in pd.date_range('2024-06-30', periods=4, freq='2QS')[::-1]]
    got = pb._dividend_yield_ttm_from_panel(_panel(rows))
    #  2 rows x 1e7 / 1e9 = 2%.  The 4-row bug would report 4%.
    assert got['SEMI'] == pytest.approx(2.0, rel=1e-6)


def test_an_INCOMPLETE_dividend_window_is_REFUSED_not_partially_summed():
    """A 3-of-4 sum masquerading as a year would UNDERSTATE the yield -- the direction that
    reads as safe, which is why it must be refused rather than published."""
    rows = [{'source': 'GAPPY', 'date': d, 'marketCap': 1.0e9,
             'dividendsPaid': (-1.0e7 if i else np.nan), 'reportedCurrency': 'USD',
             rp.FREQ_COLUMN: rp.QUARTERLY}
            for i, d in enumerate(pd.date_range('2025-07-01', periods=4, freq='QS')[::-1])]
    assert 'GAPPY' not in pb._dividend_yield_ttm_from_panel(_panel(rows))


def test_a_POSITIVE_dividendsPaid_is_REFUSED_not_sign_flipped():
    """`dividendsPaid` is a cash OUTFLOW.  A positive one means something we do not
    understand, and guessing is how a wrong number acquires a right-looking label."""
    rows = [{'source': 'ODD', 'date': d, 'marketCap': 1.0e9,
             'dividendsPaid': 1.0e7, 'reportedCurrency': 'USD',
             rp.FREQ_COLUMN: rp.QUARTERLY}
            for d in pd.date_range('2025-07-01', periods=4, freq='QS')[::-1]]
    assert 'ODD' not in pb._dividend_yield_ttm_from_panel(_panel(rows))


# =========================================================================== #
#  N-3c  the price column carries the currency it is quoted in                 #
# =========================================================================== #

def test_the_AggScore_CSV_writes_a_priceCurrency_COLUMN_beside_price():
    src = _code_only(pb.writeBoAggToCSV)
    assert "BoComp_tocsv['priceCurrency']" in src, (
        "the `price` column ships with no currency beside it again -- 000660.KS at 1,616,000 "
        "(KRW) and TNK at 76.21 (USD) then sit in one column indistinguishable")
    #  Adjacency is part of the fix: column order is what a human reads.
    assert src.index("BoComp_tocsv['price']") < src.index("BoComp_tocsv['priceCurrency']")
    assert (src.index("BoComp_tocsv['priceCurrency']")
            < src.index("BoComp_tocsv['PE-ratio']"))


def test_the_currency_label_is_never_GUESSED_from_the_exchange_suffix():
    """SHEL.L quotes GBp and reports USD; the LSE `0*.L` lines are foreign issuers.  An
    unknown currency must produce an explicit 'unknown', never a plausible symbol."""
    out = gp.price_with_currency(3322.50, None)
    assert '$' not in out
    assert 'currency unknown' in out


def test_a_non_USD_price_is_NOT_rendered_with_a_dollar_sign():
    assert '$' not in gp.price_with_currency(1616000.0, 'KRW')
    assert 'KRW' in gp.price_with_currency(1616000.0, 'KRW')
    assert 'GBp' in gp.price_with_currency(3322.5, 'GBp')


def test_the_HTML_no_longer_hard_codes_a_dollar_sign_on_price_or_market_cap():
    """The deck rendered `f"${price:.2f}"` and `f"${mktcap/1e9:.2f}B"` unconditionally.

    `cdx_df['marketCap']` is a reportedCurrency quantity, so the second one printed
    000660.KS as "$1890724.65B" (KRW 1.89e15 wearing a dollar sign) and SKHY as
    "$568935.00B".  Both are the same defect as the Graham ratio, in the deck.
    """
    src = _code_only(gp.PresentationBuilder.section_a_identity)
    assert 'f"${price:.2f}"' not in src
    assert 'f"${mktcap/1e9:.2f}B"' not in src
    assert 'marketCap_usd' in src, (
        "the deck must convert the market cap with the pipeline's own marketCap_usd -- the "
        "SAME column the market-cap bands partition on -- not print the raw reported field")


# =========================================================================== #
#  NEW  traded value per day                                                   #
# =========================================================================== #

def test_dollar_volume_is_volume_x_price_x_the_TRADING_currency_rate(profile_map):
    """TNK: 367,467 sh x $80.89 = $29.72M/day (2026-08-13, measured)."""
    df = co.dollar_volume_frame(['TNK'], profile_map=profile_map, fx=LIVE_FX)
    assert df['dollarVolume_usd'].iloc[0] == pytest.approx(29_724_405.6, rel=1e-4)
    assert df['dollarVolume_basis'].iloc[0] == '2026-08-13|USD'


def test_the_PENCE_minor_unit_path_is_LIVE_and_correct(profile_map):
    """SHEL.L is the first production use of GBp anywhere in this codebase.

    Zero sources REPORT in GBp, so the minor-unit entries in the FX table have never been
    looked up on the live path; feeding the TRADING currency in fires them for the first
    time.  9,288,640 sh x 3322.5 GBp x 0.0134987 = $416.6M/day, which is the right order for
    Shell's London line.  The GBP (not GBp) rate would give $41.6bn -- a factor of 100, which
    is exactly what a minor-unit mistake looks like and why this is pinned.
    """
    df = co.dollar_volume_frame(['SHEL.L'], profile_map=profile_map, fx=LIVE_FX)
    got = df['dollarVolume_usd'].iloc[0]
    assert got == pytest.approx(416_590_216, rel=1e-4)
    assert got < 1e9, "the GBP major unit was used for a GBp quote -- 100x too large"


def test_the_THREE_kinds_of_dollar_volume_absence_stay_DISTINGUISHABLE(profile_map):
    """A name we could not price and a name that trades nothing must not look alike.

    Same rule -- and the same shared `_volavg_reading` helper -- as the `volAvg_asof` column
    beside it, so the two volume columns cannot disagree about whether a reading exists.
    """
    syms = ['NOTHERE', 'NOPRICE', 'NOCCY', 'BADCCY']
    df = co.dollar_volume_frame(syms, profile_map=profile_map, fx=LIVE_FX)
    assert df['dollarVolume_usd'].isna().all(), "an unpriceable name must not read as $0/day"
    assert list(df['dollarVolume_basis']) == [
        co.VOLAVG_STATUS_NOT_CAPTURED,
        co.DOLLARVOL_STATUS_NO_PRICE,
        co.DOLLARVOL_STATUS_NO_CURRENCY,
        co.DOLLARVOL_STATUS_FX_UNRESOLVED % 'ZZZ']


def test_a_REFUSED_live_rate_is_indistinguishable_from_an_unknown_currency(profile_map):
    """`_fx_to_usd`'s contract: a stale/refused rate must NOT silently fall through to the
    sanity constants, because it is the same kind of wrong number as no rate at all."""
    df = co.dollar_volume_frame(['SHEL.L'], profile_map=profile_map, fx={'USD': 1.0})
    assert np.isnan(df['dollarVolume_usd'].iloc[0])
    assert df['dollarVolume_basis'].iloc[0].startswith('fx-unresolved')


def test_the_trading_currency_is_NOT_wired_into_the_market_cap_conversion():
    """The standing rule at the top of carveOut, pinned rather than trusted to a comment.

    `marketCap` is a reportedCurrency quantity; converting it with the profile (trading)
    currency is the exact unit mismatch that halted the liquidity floor.  The new dollar
    volume converts with the trading currency -- correct for a trading quantity -- and this
    test is what stops the two from being confused later.
    """
    src = _code_only(co.marketcap_usd_series)
    assert 'trading_currency' not in src
    assert '_load_volavg_profile_map' not in src
    assert 'reportedCurrency' in src


# =========================================================================== #
#  N-4  imputed_weight_share reaches the shortlist a human reads               #
# =========================================================================== #

def test_the_per_name_imputation_table_is_CARRIED_not_only_printed():
    """`missing_data_fill_report` computed `imputed_weight_share` all along and it reached
    only a dated CSV nobody opens beside a shortlist."""
    import postBoRank as pbr
    src = _code_only(pbr.postBoScoreRanking)
    assert 'missing_fill_by_name' in src


def test_the_AggScore_CSV_joins_imputed_weight_share():
    src = _code_only(pb.writeBoAggToCSV)
    assert "BoComp_tocsv['imputed_weight_share']" in src
    assert 'missing_fill_df' in inspect.signature(pb.writeBoAggToCSV).parameters


def test_an_ABSENT_fill_table_OMITS_the_column_rather_than_writing_zeros():
    """A zero here would assert "nothing was imputed", which is the one thing this column
    exists to stop anyone believing by default."""
    src = _code_only(pb.writeBoAggToCSV)
    i = src.index("BoComp_tocsv['imputed_weight_share']")
    tail = src[i:i + 2000]
    assert 'fillna(0)' not in tail and '= 0' not in tail.split('else')[0]
    assert 'OMITTED' in tail or 'omitted' in tail


# =========================================================================== #
#  N-5  the veto ships its ejection list                                       #
# =========================================================================== #

def test_the_veto_report_records_WHICH_flags_ejected_WHICH_name():
    """`by_flag` counts ejections per flag and `ejected` lists the names; NEITHER can answer
    "why was THIS name removed", which is the one question the shipped CSV exists to answer.

    `bad` is local to `apply_veto`, so if the per-name mapping is not captured while the call
    is running it cannot be recovered from the report afterwards -- which is exactly why
    reconstructing the 2026-08-13 ejections required a full offline re-score.
    """
    #  Reuses the existing veto suite's panel builder rather than a second copy of it: a
    #  divergent fixture is how two tests come to disagree about what the gate does.
    from test_stage1_veto import _panel as veto_panel, _scores as veto_scores
    src = {'BAD': {'uCurrentRatio': 0.5}, 'ALSOBAD': {'returnOnAssets': -0.01},
           'GOOD': {}}
    _, rep = sv.apply_veto(veto_scores(src), veto_panel(src), enabled=True, verbose=False)
    assert rep['ejected'] == ['ALSOBAD', 'BAD']
    assert rep['ejected_flags'] == {'ALSOBAD': ['returnOnAssets'],
                                    'BAD': ['uCurrentRatio']}
    #  A survivor must NOT appear -- the map is the ejection reason, not a flag dump.
    assert 'GOOD' not in rep['ejected_flags']


def test_the_ejection_csv_names_every_ejected_source_and_the_flags_that_did_it(tmp_path):
    reports = {'general': {'enabled': True, 'applies': True, 'n_in': 1404, 'n_ejected': 2,
                           'ejected': ['AAA', 'BBB'],
                           'ejected_flags': {'AAA': ['uInterestCoverage', 'uAltmanZ'],
                                             'BBB': ['uPiotroski']}}}
    p = sv.write_ejection_csv(reports, path=str(tmp_path / 'ej.csv'))
    df = pd.read_csv(p)
    ej = df[df['status'] == 'EJECTED'].set_index('source')
    assert sorted(ej.index) == ['AAA', 'BBB']
    assert ej.loc['AAA', 'n_flags'] == 2
    assert 'uInterestCoverage' in ej.loc['AAA', 'flags']
    #  ONE ROW PER NAME, so the file's row count means "names removed".
    assert len(ej) == 2


def test_the_ejection_csv_keeps_NO_EJECTIONS_distinct_from_DID_NOT_RUN(tmp_path):
    """The distinction this whole module exists to preserve, carried into the artifact."""
    clean = sv.write_ejection_csv(
        {'general': {'enabled': True, 'applies': True, 'n_in': 900, 'ejected': []}},
        path=str(tmp_path / 'clean.csv'))
    off = sv.write_ejection_csv(
        {'general': {'enabled': False, 'applies': True, 'n_in': 900, 'ejected': []}},
        path=str(tmp_path / 'off.csv'))
    na = sv.write_ejection_csv(
        {'REIT': {'enabled': True, 'applies': False, 'ejected': [],
                  'not_applicable_reason': 'two flags undefined on this cohort'}},
        path=str(tmp_path / 'na.csv'))
    notes = [pd.read_csv(p)['note'].iloc[0] for p in (clean, off, na)]
    assert len({str(n) for n in notes}) == 3, (
        'a clean veto, an OFF veto and an out-of-scope pool must not produce the same note')
    assert 'ejected nobody' in notes[0]
    assert 'OFF' in notes[1]


def test_the_ejection_csv_is_written_EVEN_WHEN_NOTHING_WAS_EJECTED(tmp_path):
    """Its PRESENCE is the evidence the gate was reached."""
    p = sv.write_ejection_csv({}, path=str(tmp_path / 'empty.csv'))
    assert p is not None and os.path.exists(p)


def test_the_ejection_csv_TRAVELS_to_the_other_machine():
    """Root-level artifacts demonstrably reach Drive and `output/` demonstrably did not, so
    a new evidence file needs BOTH the root location and the allowlist glob."""
    import Sbocker
    pats = inspect.getsource(Sbocker).split('allowlist_patterns')[1][:4000]
    assert 'Stage1VetoEjections_*.csv' in pats
    assert sv.EJECTION_CSV % '2026-08-13' == 'Stage1VetoEjections_2026-08-13.csv'
    assert '/' not in sv.EJECTION_CSV and '\\' not in sv.EJECTION_CSV


def test_the_deck_reads_the_veto_report_that_was_ALREADY_in_every_pickle():
    """`stage1_veto` has been in every Boresults pickle since 2026-08-05 and the deck simply
    never read it, so the report said nothing about the largest edit made to the pool."""
    src = _code_only(gp.load_run_data)
    assert "boresults_dic.get('stage1_veto')" in src


# =========================================================================== #
#  THE PRESENTATION RULE: a marker's valence must match its subject            #
# =========================================================================== #
#  "putting a red flag next to high quality earning indicator is the exact opposite of what
#  I mean" (CEO, 2026-08-13).  These are the tests that hold each new marker pointing the
#  right way -- including the two that are deliberately ONE-SIDED.

def test_HIGH_traded_volume_earns_NO_positive_marker():
    """Liquidity is not merit, and this filter's thesis is NEGLECTED names -- so rewarding a
    liquid name would suggest the opposite of the strategy.  The high end is a bare number."""
    cell = gp.dollar_volume_cell(7.06e9)
    assert 'flag' not in cell
    assert 'GREEN' not in cell and '🟢' not in cell
    assert '$7.06B' in cell


def test_THIN_traded_volume_is_AMBER_and_worded_as_a_CONSTRAINT_not_a_verdict():
    """Amber, not red: nothing is excluded on volume (register J-1), and the caution is about
    the reader's ability to size and exit -- not a claim about the business."""
    cell = gp.dollar_volume_cell(8_477.0)
    assert 'AMBER' in cell and 'RED' not in cell
    assert 'thin market' in cell
    assert 'NOT a judgement on the business' in cell


def test_the_thin_marker_fires_ONLY_below_the_reference_line():
    assert 'AMBER' not in gp.dollar_volume_cell(gp.THIN_LIQUIDITY_USD_PER_DAY + 1)
    assert 'AMBER' in gp.dollar_volume_cell(gp.THIN_LIQUIDITY_USD_PER_DAY - 1)


def test_a_HEAVILY_IMPUTED_name_reads_as_THIN_EVIDENCE_not_as_a_bad_company():
    """The wording is epistemic on purpose: the risk is that we know little about the name."""
    cell = gp.data_coverage_cell(0.9317)
    assert 'RED' in cell
    assert '93% imputed' in cell and 'thin evidence' in cell
    assert '6.8% measured' in cell


def test_a_FULLY_MEASURED_name_gets_NO_marker_at_all():
    """The complement of the rule above: a marker that fires on every name is noise, and the
    top-100 MEDIAN imputation is 0.0000, so silence is the normal state."""
    cell = gp.data_coverage_cell(0.0)
    assert 'flag' not in cell
    assert '100.0% measured' in cell


def test_the_coverage_marker_says_it_is_INDEPENDENT_of_the_forensic_tag():
    """The exact misread this closes: `forensicTag = clean` on a 93%-imputed name.  Both STRT
    and PET.TO shipped that way at ranks 55 and 71 on 2026-08-13."""
    cell = gp.data_coverage_cell(0.9003)
    assert 'forensic' in cell.lower()
    assert 'independent' in cell.lower()


def test_the_coverage_marker_sits_BESIDE_the_forensic_tag():
    """Adjacency is load-bearing -- the two facts have to be next to each other for a reader
    to see that they are different facts."""
    src = _code_only(gp.PresentationBuilder.section_b_flags)
    assert src.index('Forensic:') < src.index('Score coverage:')


def test_the_veto_banner_is_NEUTRAL_not_a_warning():
    """A hard exclusion applied before ranking is the filter working as designed, and it
    describes the SCOPE of what the reader is looking at.  Styling it like the red
    `.basis-warning` would suggest something went wrong."""
    html = gp.veto_scope_banner({'general': {'enabled': True, 'applies': True,
                                             'n_in': 1404, 'n_ejected': 573}})
    assert 'industry-counter' in html
    assert 'basis-warning' not in html
    assert '573 of 1404' in html and '40.8%' in html
    assert 'not a warning to weigh' in html


def test_the_veto_banner_distinguishes_DID_NOT_RUN_from_OFF_from_APPLIED():
    assert 'DID NOT RUN' in gp.veto_scope_banner({})
    assert 'was OFF' in gp.veto_scope_banner(
        {'general': {'enabled': False, 'applies': True, 'n_in': 900, 'n_ejected': 0}})
    assert gp.veto_scope_banner(None) == ''


def test_the_legend_STATES_the_valence_of_every_field_added():
    """A marker whose direction the reader has to infer can be inferred backwards, which is
    the failure the CEO named.  The two one-sided markers say so, so their SILENCE is not
    read as approval."""
    leg = gp.PresentationBuilder.__dict__['_icon_legend'](object.__new__(
        gp.PresentationBuilder))
    for phrase in ('own currency', 'converted to USD', 'no positive marker',
                   'thin market', 'independent of the forensic tag'):
        assert phrase.lower() in leg.lower(), f'the legend does not state: {phrase}'


# =========================================================================== #
#  ABSENCE, EVERYWHERE, READS AS ABSENCE                                       #
# =========================================================================== #

def test_every_new_deck_field_degrades_to_a_DASH_on_a_pre_expansion_run():
    """`priceCurrency` / `dollarVolume_usd` / `imputed_weight_share` first exist from
    2026-08-13.  An older run must render "—", never a confident 0."""
    assert gp.dollar_volume_cell(None) == '—'
    assert gp.data_coverage_cell(None) == '—'
    assert gp.price_with_currency(None, 'USD') == '—'
    assert gp.money_format(None) == '—'


def test_agg_val_returns_the_DEFAULT_for_a_column_this_run_does_not_have():
    b = object.__new__(gp.PresentationBuilder)
    b.data = {'aggscore_df': pd.DataFrame({'source': ['TNK'], 'price': [80.89]})}
    assert b.agg_val('TNK', 'price') == 80.89
    assert b.agg_val('TNK', 'priceCurrency') is None      # column absent on this run
    assert b.agg_val('NOPE', 'price') is None


# =========================================================================== #
#  REVIEW FIXES (2026-08-13) -- H-1..H-5, L-2                                  #
# =========================================================================== #

def test_H3_a_non_payer_does_not_render_a_NEGATIVE_zero_yield():
    """`-0.0 / mc` is a NEGATIVE zero by IEEE-754 and `_fmt4` printed it as `'-0.0000'`.

    It hit 20 of the shipped 2026-08-13 top-100 -- a fifth of the shortlist showing a
    negative-signed dividend yield -- and it was a REGRESSION against the vendor column it
    replaced, which printed a plain `0.0`.
    """
    rows = [{'source': 'NOPAY', 'date': d, 'marketCap': 1.0e9, 'dividendsPaid': 0.0,
             'reportedCurrency': 'USD', rp.FREQ_COLUMN: rp.QUARTERLY}
            for d in pd.date_range('2025-07-01', periods=4, freq='QS')[::-1]]
    got = pb._dividend_yield_ttm_from_panel(_panel(rows))['NOPAY']
    assert got == 0.0
    assert not np.signbit(got), 'negative zero -- renders as -0.0000'
    assert pb._fmt4(got) == '0.0000'


def test_H2_the_graham_column_ships_its_BASIS_beside_it(cross_listed_panel):
    """Two bases in one column, stated in a column -- not only in the run log.

    The same change added `dollarVolume_basis` and `volAvg_asof` already sat beside
    `volAvg_report`, so leaving Graham as the one two-basis column with no basis field was an
    inconsistency inside one diff rather than a considered difference.
    """
    src = _code_only(pb.writeBoAggToCSV)
    assert "BoComp_tocsv['GrahamNumberToPrice_basis']" in src
    assert (src.index("BoComp_tocsv['GrahamNumberToPrice']")
            < src.index("BoComp_tocsv['GrahamNumberToPrice_basis']"))
    #  And the values are the real per-name bases, not a constant.
    raw = pd.DataFrame({'source': ['SKHY', 'SHEL.L', 'TNK'],
                        'grahamNumberToPrice': [3.9909, 1.4749, np.nan]})
    _, basis = pb._graham_to_price_published(raw, cross_listed_panel)
    assert {basis['SKHY'], basis['TNK']} == {pb.GRAHAM_BASIS_SCORED, pb.GRAHAM_BASIS_PANEL}


def test_L2_a_WHOLESALE_graham_failure_is_reported_not_silent():
    """Both helpers return `{}` on any exception, so a renamed panel column would write an
    all-NaN column.  The fallback line alone would say nothing -- it reports the MIX."""
    src = _code_only(pb.writeBoAggToCSV)
    assert '_gtp_missing' in src
    assert 'THE WHOLE COLUMN IS EMPTY' in inspect.getsource(pb.writeBoAggToCSV)


def test_H4_a_report_without_flag_detail_does_NOT_claim_ZERO_flags(tmp_path):
    """`n_flags = 0` on an ejected name is an AFFIRMATIVE FALSE CLAIM.

    A name reaches the EJECTED rows only by failing at least `EJECT_MIN_FLAGS`, so zero
    contradicts the gate itself.  Verified on the real 2026-08-13 report, which predates
    `ejected_flags`: 617 ejected names, every one written `n_flags = 0`.  Same rule this
    change already applies to `imputed_weight_share` (omitted, not zeroed) and `priceCurrency`
    (NaN, not guessed): absence is not zero.
    """
    legacy = {'general': {'enabled': True, 'applies': True, 'n_in': 1404,
                          'n_ejected': 2, 'ejected': ['AAA', 'BBB']}}      # no ejected_flags
    df = pd.read_csv(sv.write_ejection_csv(legacy, path=str(tmp_path / 'legacy.csv')),
                     keep_default_na=False)
    ej = df[df['status'] == 'EJECTED']
    assert len(ej) == 2, 'the ejections themselves must still be reported'
    assert set(ej['n_flags'].astype(str)) == {''}, 'a blank, never a 0'
    assert (ej['flags'] == '').all()
    assert ej['note'].str.contains('not recorded').all()
    assert ej['note'].str.contains('WAS ejected').all(), (
        'the note must say the ejection is real and only the REASON is unrecorded')


def test_H4_a_report_WITH_flag_detail_still_reports_the_counts(tmp_path):
    """The complement: the blank is for absence only, never for a report that has the data."""
    rich = {'general': {'enabled': True, 'applies': True, 'n_in': 10, 'n_ejected': 1,
                        'ejected': ['AAA'],
                        'ejected_flags': {'AAA': ['uCurrentRatio', 'returnOnAssets']}}}
    df = pd.read_csv(sv.write_ejection_csv(rich, path=str(tmp_path / 'rich.csv')),
                     keep_default_na=False)
    ej = df[df['status'] == 'EJECTED'].iloc[0]
    assert int(ej['n_flags']) == 2 and ej['note'] == ''


def test_H5_a_clone_line_carries_its_COUNTERPART_in_the_basis(profile_map):
    """"Flagged, not patched" has to reach the CSV cell, not just a docstring.

    SKHY computes $7.06bn/day -- third in the shipped top-100, above TSM -- because the vendor
    reports SK Hynix's HOME line volume against the Nasdaq depositary price.  No ratio is
    invented; the run's OWN contamination artifact already pairs the two lines.
    """
    df = co.dollar_volume_frame(['TNK', 'SHEL.L'], profile_map=profile_map, fx=LIVE_FX,
                                clone_map={'TNK': 'clone-suspect:0EAQ.L'})
    b = list(df['dollarVolume_basis'])
    assert b[0] == '2026-08-13|USD|clone-suspect:0EAQ.L'
    assert b[1] == '2026-08-13|GBp', 'an unpaired name must not be marked'


def test_H5_the_counterpart_map_is_SYMMETRIC_and_names_the_verdict(tmp_path):
    """Both members of a pair are marked, and a same-company pairing (`name_match`) is kept
    distinct from a different-company one (`NAME_MISMATCH`) -- they are different defects."""
    import vendor_contamination as vc
    f = tmp_path / 'VendorContaminationFlags_2026-08-13.csv'
    pd.DataFrame([
        {'source_a': '000660.KS', 'source_b': 'SKHY', 'verdict': 'name_match'},
        {'source_a': '058820.KQ', 'source_b': 'CMG', 'verdict': 'NAME_MISMATCH'},
    ]).to_csv(f, index=False)
    m = vc.clone_counterparts(path=str(f))
    assert m['SKHY'] == 'clone-suspect:000660.KS'
    assert m['000660.KS'] == 'clone-suspect:SKHY'          # symmetric
    assert m['CMG'] == 'contamination-suspect:058820.KQ'
    assert m['058820.KQ'] == 'contamination-suspect:CMG'


def test_H5_the_STRONGER_verdict_wins_a_multiply_paired_name(tmp_path):
    """A name mixed up with a DIFFERENT company is the more serious finding and must not be
    overwritten by a benign cross-listing pairing found later in the same file."""
    import vendor_contamination as vc
    f = tmp_path / 'VendorContaminationFlags_2026-08-13.csv'
    pd.DataFrame([
        {'source_a': 'X', 'source_b': 'BAD', 'verdict': 'NAME_MISMATCH'},
        {'source_a': 'X', 'source_b': 'TWIN', 'verdict': 'name_match'},
    ]).to_csv(f, index=False)
    assert vc.clone_counterparts(path=str(f))['X'].startswith('contamination-suspect:')


def test_H5_an_absent_contamination_file_costs_the_MARKER_not_the_NUMBER(tmp_path):
    import vendor_contamination as vc
    assert vc.clone_counterparts(path=str(tmp_path / 'nope.csv')) == {}


# --- H-1: the deck renders more names than the AggScore CSV contains ------------------

def _deck(profile_map, fill=None, clone=None):
    """A real `PresentationBuilder` with only the attributes these tests touch.

    `object.__new__` rather than a stand-in subclass, deliberately: binding
    `gp.PresentationBuilder._field` in a class body would be evaluated at IMPORT time, so
    against pre-fix code the whole module would fail COLLECTION instead of failing its tests
    -- and a guard that errors at collection tells you nothing about which behaviour is
    missing.  It also keeps the tests exercising the real methods rather than copies.
    """
    b = object.__new__(gp.PresentationBuilder)
    b.data = {'aggscore_df': pd.DataFrame({'source': ['TNK'], 'price': [80.89],
                                           'priceCurrency': ['USD']}),
              'profile_map': profile_map, 'fx_table': LIVE_FX,
              'fill_by_name': fill or {}, 'clone_map': clone or {}, 'fx_label': None}
    b._offline_cache = {}
    return b


def test_H1_a_name_OUTSIDE_the_top100_CSV_still_gets_its_fields(profile_map):
    """THE DEFECT: `agg_val` read only the general top-100 AggScore CSV, but the deck renders
    56 pages -- top-20 plus cohort, side-list and market-cap-band names -- of which only 31
    were in that CSV.  The other 25 rendered an em-dash for Price, Traded/day and Score
    coverage.  Every input needed already ships with the run."""
    d = _deck(profile_map, fill={'0QQF.L': 0.42})
    assert d.agg_val('0QQF.L', 'dollarVolume_usd') == pytest.approx(84.72, rel=1e-2)
    assert d.agg_val('0QQF.L', 'priceCurrency') == 'GBp'
    assert d.agg_val('0QQF.L', 'imputed_weight_share') == 0.42
    assert d.agg_val('0QQF.L', 'price') == 5.23


def test_H1_the_thin_market_caution_is_no_longer_SWALLOWED(profile_map):
    """The measured cost of the defect: the deck fired the caution twice and suppressed it
    four times, on cohort pages ~19x thinner than the top-100.  A blank where a caution
    belongs reads as "no concern" -- the silence-as-approval failure the legend exists to
    prevent, and the one place the change actively mis-suggested."""
    d = _deck(profile_map)
    assert 'thin market' in gp.dollar_volume_cell(d.agg_val('0QQF.L', 'dollarVolume_usd'))


def test_H1_the_AggScore_CSV_still_WINS_wherever_it_has_the_name(profile_map):
    """Adding the fallback must change no in-CSV cell."""
    d = _deck(profile_map)
    assert d._field('TNK', 'price') == (80.89, 'aggscore')
    assert d._field('0QQF.L', 'price')[1] == 'offline'


def test_H1_the_offline_lookup_is_LAZY_so_it_cannot_drift_from_the_LAYOUT():
    """Building the table over `_page_tickers()` was WRONG and was caught: that is the YAHOO
    FETCH set (top-20 + top-5 per cohort = 45), while the banded layout renders 56.  Eleven
    pages were blanked for a second, unrelated reason.  Keying off the ticker being rendered
    removes the bug class rather than fixing the instance."""
    assert '_page_tickers' not in _code_only(gp.PresentationBuilder._resolve_offline)
    assert '_page_tickers' not in _code_only(gp.PresentationBuilder.offline_field)


def test_H1_the_deck_uses_THIS_RUNS_profile_map_not_a_repo_root_glob():
    """`carveOut._volavg_profile_map_cached()` globs the REPO ROOT for the newest capture --
    right for the pipeline, WRONG for a deck handed an arbitrary `--run-dir`.

    Caught exactly that way in testing: the deck rendered `price_asof 2026-08-11` on an 08-13
    run, blanked CBSM.PA (absent from the older map) and gave ORIA.PA $7,337/day against this
    run's $5,915.  Same-date or nothing -- `resolve_run_artifacts` refuses the same
    substitution for the same reason.
    """
    res = _code_only(gp.PresentationBuilder._resolve_offline)
    assert '_volavg_profile_map_cached' not in res, 'that is the repo-root glob'
    assert "self.data.get('profile_map')" in res
    assert 'profile_map=pmap' in res
    load = _code_only(gp.load_run_data)
    assert 'profile_map_for_run(run_dir, run_date)' in load, 'must be resolved SAME-DATE'
    #  AND THE DECK MUST NOT NAME THE ARTIFACT ITSELF.  `test_universes` pins the set of
    #  modules that may name `volavgdic_fmp_` to {findAllSectors, carveOut, Sbocker}; the
    #  first version of this fix added a fourth and that guard caught it.  A module that
    #  knows the filename is one step from doing its own `read_pickle` and reading raw
    #  entries, which is exactly how the single-seam guarantee lapses unnoticed.
    assert 'volavgdic_fmp_' not in inspect.getsource(gp), (
        'the presentation generator must ask carveOut for a run-scoped capture, not build '
        'the path -- see '
        'test_universes.test_the_volavg_pickle_still_has_exactly_ONE_reading_seam')
    assert co.profile_map_for_run('no-such-dir', '2026-08-13') == {}


def test_H1_an_absent_same_date_capture_BLANKS_rather_than_borrowing_another_run():
    """A price and a traded value from a different day are worse than an em-dash."""
    d = _deck({})                                   # no capture for this run
    assert d.agg_val('0QQF.L', 'price') is None
    assert d.agg_val('0QQF.L', 'dollarVolume_usd') is None
    assert gp.dollar_volume_cell(d.agg_val('0QQF.L', 'dollarVolume_usd')) == '—'


def test_H1_the_deck_converts_with_THIS_RUNS_FX_and_labels_the_anchor_fallback():
    """The deck never calls `install_for_run`, so carveOut's FX state is 'unset' here and an
    unqualified conversion would silently use the sanity ANCHORS -- which are a UNITS check,
    not a rate source (SHEL.L: $391.9M on anchors vs $416.6M live), and which produce a
    BYTE-IDENTICAL basis string.  So the run's dated table is preferred, and an anchor
    fallback is labelled on every cell."""
    load = _code_only(gp.load_run_data)
    assert 'FxRates_{d}.csv' in load
    assert "fx_label = 'fx=anchors'" in load
    df = co.dollar_volume_frame(['TNK'], profile_map={
        'TNK': {'volAvg': 100, 'asof': '2026-08-13', 'price': 10.0, 'currency': 'USD'}},
        fx=LIVE_FX, fx_label='fx=anchors')
    assert df['dollarVolume_basis'].iloc[0] == '2026-08-13|USD|fx=anchors'


def test_H1_a_fallback_PRICE_is_labelled_with_its_capture_date():
    """Two captures of the same vendor field, hours apart.  Fine for a display price; not
    fine to leave unlabelled -- that is the silent two-basis column this change removes."""
    src = _code_only(gp.PresentationBuilder.section_a_identity)
    assert "_px_origin == 'offline'" in src
    assert 'price_asof' in src and 'capture' in src


# =========================================================================== #
#  THE TTM WINDOW IS POSITIONAL AND ABSTAINS  (fix 2026-08-14)                 #
# =========================================================================== #
#  `ttm_sum` / `ttm_aligned_sums` were `df[col].dropna().head(rpy)` -- a window over the newest
#  rpy PRESENT VALUES, not over the newest rpy PERIODS.  Two limbs, both of which make
#  "trailing twelve months" a claim the number does not own:
#    LIMB 1  fewer than rpy values -> it summed whatever it found and labelled it TTM.
#    LIMB 2  a hole inside the window -> `dropna` walked older, so the sum spanned MORE than
#            twelve months while still being divided into twelve-month quantities.
#  MEASURED on the shipped 2026-08-13 CUR3K panel, over the 15,774 (source, column) pairs
#  these helpers read: LIMB 1 fires 0 times (latent on this data, not active); LIMB 2 fires
#  21 times (0.133%), median span 1.00 years and worst 1.50, and 0 times on the 600 shortlist
#  pairs.  Driving the real 2026-08-13 presentation build, 10 of 22,516 helper invocations
#  change, at four sites, every one from a number to an abstention.
#  BOTH LIMBS ARE PINNED HERE.  A test for the live one only would let the latent one come
#  back the first time a panel arrives with a short column.

def test_ttm_sum_ABSTAINS_when_the_window_cannot_be_filled():
    """LIMB 1: fewer than `rpy` periods is not a shorter year, it is no year."""
    df = pd.DataFrame({'freeCashFlow': [10.0, 20.0, np.nan, np.nan, np.nan, np.nan]})
    assert np.isnan(gp.ttm_sum(df, 'freeCashFlow', rpy=4)), (
        'two populated quarters must not be published as a trailing YEAR')
    #  ...and a semi-annual filer reading the SAME frame has a real, full window of 2.
    assert gp.ttm_sum(df, 'freeCashFlow', rpy=2) == 30.0
    #  a frame SHORTER than the window abstains for the same reason
    assert np.isnan(gp.ttm_sum(pd.DataFrame({'x': [1.0, 2.0]}), 'x', rpy=4))
    #  and the all-absent case is unchanged (it already returned NaN)
    assert np.isnan(gp.ttm_sum(pd.DataFrame({'x': [np.nan] * 8}), 'x', rpy=4))


def test_ttm_sum_does_NOT_reach_PAST_the_window_to_fill_it():
    """LIMB 2 -- the limb that actually fires on the shipped panel.

    A hole at row 1 used to make the sum span FIVE quarters (rows 0,2,3,4) while every
    consumer divides it into, and compares it against, twelve-month quantities.  It is not a
    smaller measurement of the same thing; it is a different quantity in the same units."""
    df = pd.DataFrame({'revenue': [100.0, np.nan, 100.0, 100.0, 100.0, 100.0]})
    assert np.isnan(gp.ttm_sum(df, 'revenue', rpy=4)), (
        'the newest four PERIODS are not all present, so there is no TTM; the old code '
        'returned 400.0 by reaching to row 4 -- a fifteen-month "year"')
    #  no hole -> unchanged, and this is the case that must NOT regress
    clean = pd.DataFrame({'revenue': [100.0, 110.0, 120.0, 130.0, 140.0]})
    assert gp.ttm_sum(clean, 'revenue', rpy=4) == 460.0


def test_ttm_aligned_sums_window_is_positional_for_every_column_at_once():
    """The old form kept the columns aligned WITH EACH OTHER but let the shared window slide
    older, so a paired ratio could be internally consistent and still not be a year."""
    df = pd.DataFrame({'netIncome': [10.0, np.nan, 10.0, 10.0, 10.0],
                       'netCashProvidedByOperatingActivities': [8.0, 8.0, 8.0, 8.0, 8.0]})
    got = gp.ttm_aligned_sums(df, ['netIncome',
                                   'netCashProvidedByOperatingActivities'], rpy=4)
    assert all(np.isnan(v) for v in got), got
    clean = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 99.0], 'b': [5.0, 5.0, 5.0, 5.0, 99.0]})
    assert gp.ttm_aligned_sums(clean, ['a', 'b'], rpy=4) == (10.0, 20.0)
    #  and the two helpers now agree BY CONSTRUCTION on the same frame -- they did not before
    assert gp.ttm_aligned_sums(clean, ['a'], rpy=4)[0] == gp.ttm_sum(clean, 'a', rpy=4)


def test_the_TTM_helpers_no_longer_carry_the_dropna_then_head_expression():
    """Structural, because the two behavioural tests above would both pass on a form that
    re-introduced `dropna().head(rpy)` behind a length check that happened to be equivalent
    on those fixtures.  The expression itself is what was wrong."""
    for fn in (gp.ttm_sum, gp.ttm_aligned_sums):
        src = _code_only(fn)
        assert '.dropna()' not in src, (
            '%s must window POSITIONALLY; `dropna` before `head` is the defect itself' % fn.__name__)
        assert '.head(' in src


# ---------------------------------------------------------------------------
#  THE ABSTENTION SAYS WHY, ON THE DECK (CEO, 2026-08-16).  After the O-13 domain guards a
#  fifth of the shortlist carries `data-incomplete: dig-deeper`; a tag that says only
#  "incomplete" reads as a broken tool at that rate.
# ---------------------------------------------------------------------------

def _deck_with_forensic(agg_row, forensic_df=None, carveout_labels=None):
    b = object.__new__(gp.PresentationBuilder)
    b.data = {'aggscore_df': pd.DataFrame([agg_row]),
              'postrank_df': pd.DataFrame({'source': [agg_row['source']],
                                           'AggScore': [0.5], 'moatScore': [7.0]}),
              'forensic_df': forensic_df,
              #  Q-44: the deck's forensic-applicability guard reads the run's own carve
              #  labels for the pages the AggScore CSV does not contain.
              'carveout_labels': carveout_labels,
              'profile_map': {}, 'fx_table': None, 'fill_by_name': {}, 'clone_map': {},
              'fx_label': None}
    b._offline_cache = {}
    b._validity_cache = {}
    return b


def test_the_deck_says_WHY_a_name_has_no_M_score():
    d = _deck_with_forensic({'source': 'RMV.L',
                             'forensicTag': 'data-incomplete: dig-deeper',
                             'M_abstain_reason': 'no usable vendor data for: gross margin (GMI)'})
    html = d.section_b_flags('RMV.L')
    assert 'data-incomplete' in html
    assert 'gross margin (GMI)' in html, html
    assert 'forensic-why' in html


def test_the_deck_reason_is_STYLED_apart_from_the_forensic_tag_it_follows():
    """A data gap rendered in the tag's alarm-red would read as a forensic red flag -- the
    exact inversion the CEO ruled against for the imputed-share marker ("a red flag beside a
    high-quality earnings indicator inverts the read and is worse than no marker")."""
    css = gp.PresentationBuilder._get_css(object.__new__(gp.PresentationBuilder))
    assert '.forensic-why' in css, 'the reason has no style of its own'
    #  the tag's own colour is the alarm red; the reason must not inherit it
    block = css.split('.forensic-why')[1].split('}')[0]
    assert '#d9534f' not in block, block
    src = _code_only(gp.PresentationBuilder.section_b_flags)
    assert 'forensic-why' in src


def test_a_name_WITH_an_M_score_gets_no_reason_and_no_extra_markup():
    d = _deck_with_forensic({'source': 'TNK', 'forensicTag': 'clean',
                             'M_abstain_reason': ''})
    html = d.section_b_flags('TNK')
    assert 'forensic-why' not in html, html
    assert 'clean' in html


def test_the_deck_falls_back_to_the_FORENSIC_csv_when_the_aggscore_one_lacks_the_column():
    """An older AggScore CSV predates the column; the forensic CSV of the same run carries it.
    Absence of the reason must never blank the TAG, which is the pre-2026-08-16 behaviour and
    still the right one."""
    fdf = pd.DataFrame({'source': ['CFX.L'],
                        'M_abstain_reason': ['no usable vendor data for: asset quality (AQI)']})
    d = _deck_with_forensic({'source': 'CFX.L',
                             'forensicTag': 'data-incomplete: dig-deeper'}, forensic_df=fdf)
    html = d.section_b_flags('CFX.L')
    assert 'asset quality (AQI)' in html, html
    d2 = _deck_with_forensic({'source': 'CFX.L', 'forensicTag': 'clean'})
    assert 'forensic-why' not in d2.section_b_flags('CFX.L')


# ---------------------------------------------------------------------------
#  P-2 SWEEP -- "an explanation of a thing that is not there", deck edition
# ---------------------------------------------------------------------------

def test_an_ABSENT_value_outranks_LOW_CONFIDENCE_in_both_the_icon_and_the_note():
    """*** P-5, 2026-08-29: absence outranks low confidence. ***  `compute_verdict` tested
    `low_conf` BEFORE the value, so a metric that is both low-confidence and MISSING rendered
    as 🟡 "merely uncertain" -- the weaker and less actionable of the two facts, and the one
    that reads as a number we distrust rather than a number we do not have.  The P-2 sweep of
    2026-08-17 fixed only the SENTENCE and pinned the icon, saying in as many words that a
    later ruling should update this test rather than quietly relax it.  This is that update.

    TWO-SIDED, and the second side is the one that could be broken silently: a metric that is
    low-confidence and PRESENT must still be 🟡 with the low-confidence sentence, or the fix has
    traded a wrong icon for a blind one that never reports low confidence at all.
    """
    #  one key from EACH branch that carries the low-confidence short-circuit: VERDICT_RULES
    #  (M-Score, sloan) and VERDICT_FLOORS (cash_conv).  `incomeQuality` is VERDICT_GRAY and
    #  never reaches the branch, so including it would have tested nothing.
    for key in ('M-Score', 'sloan', 'cash_conv'):
        st_missing, note_missing = gp.compute_verdict(key, np.nan, None, low_conf=True)
        assert st_missing == 'gray', (key, st_missing)        # ⚪, not 🟡
        assert note_missing == 'value unavailable', (key, note_missing)
        assert 'value present' not in note_missing, (key, note_missing)
        #  ...and a name that HAS the value still gets the low-confidence amber and sentence
        st_present, note_present = gp.compute_verdict(key, 0.5, None, low_conf=True)
        assert st_present == 'neutral', (key, st_present)
        assert note_present == 'value present but flagged low-confidence (see 🚩/forensic)', note_present
        #  the icon actually rendered differs, not just the internal state name
        assert gp.VERDICT_GLYPH[st_missing] != gp.VERDICT_GLYPH[st_present]
    #  the ordinary (not low-confidence) missing case is untouched
    assert gp.compute_verdict('M-Score', np.nan, None, low_conf=False)[0] == 'gray'
    #  cohort suppression still outranks BOTH -- a metric with no rule for this cohort says so
    #  rather than reporting a missing value it was never going to judge.
    assert gp.compute_verdict('M-Score', np.nan, 'FinManager', low_conf=True) == (
        'gray', 'no universal rule for this cohort')
    #  THE BELT BEHIND THE BRACE.  `_low_conf_note`'s absent-value branch is now unreachable
    #  through `compute_verdict`; it is kept so that a future re-ordering cannot restore the
    #  false SENTENCE as well as the icon, and is exercised directly here so it cannot rot.
    assert 'value present' not in gp._low_conf_note(np.nan)
    assert 'unavailable' in gp._low_conf_note(np.nan)


def test_the_deck_LEGEND_states_the_precedence_it_renders():
    """*** P-5. ***  The icon meanings are a documented spec ("Part C"); the code and the
    documented meaning must not drift apart, and the only copy of that spec the CEO actually
    reads is the legend printed on the page.  So the legend is asserted here alongside the
    behaviour: ⚪ must advertise BOTH of its meanings (no rule, and no value), and the
    precedence must be stated where the reader is.

    WHAT THIS CANNOT CATCH: it checks that the sentences EXIST, not that they are true.  A
    change that re-ordered `compute_verdict` and left the legend intact would pass this test
    and fail the one above -- which is why both are here.
    """
    legend = gp.PresentationBuilder._icon_legend(None)   # `self` is unused by the legend
    assert 'or no value' in legend, legend
    assert 'absence is the stronger fact' in legend, legend
    src = inspect.getsource(gp.compute_verdict)
    #  the ORDER, read off the source: the value test precedes the low-confidence test in both
    #  branches that carry one.  A pure-behaviour test cannot see a branch nobody exercises.
    assert src.index("st = _verdict_state_band(r, value)") < src.index("if low_conf:")
    assert src.index("if np.isnan(v):") < src.rindex("if low_conf:")


def test_the_deck_R5_rule_cannot_fire_off_a_driver_string_that_has_no_M_score():
    """*** P-2. ***  R5 reads `M_drivers` from the published CSV and fires "Beneish drivers
    (DSRI/SGI/AQI) elevated" on a token match.  The column is BLANK for any name without an
    M-score (forensicFlags.buildForensicFlagTable), so the deck inherits the fix and must not
    carry its own recomputation that would defeat it.

    Q-66 MOVED THE READ, NOT THE RULE.  The read that used to sit inline in `evaluate_page`
    now sits inside `forensic_scores`, the one method both `evaluate_page` and Section F
    resolve through -- so the assertion moves to its new site, plus the assignment that carries
    it into the rule.  The two assertions that actually hold the defect shut (no local
    recomputation; the rule keys off the published string) are unchanged.

    THE FORM CHANGED TOO, AND ON PURPOSE.  `str(r0.get('M_drivers') or '')` returns the literal
    `'nan'` for a blank that has been through a CSV, because NaN is truthy.  It was harmless
    for THIS rule only by luck -- R5's token test happens not to match 'NAN' -- and it stopped
    being harmless when R3 began printing the driver string.  It reads through `_text` now, and
    the useless-half guard below is what stops that becoming "assert some function was
    called"."""
    src = inspect.getsource(gp)
    assert "'M_drivers': _text(r0.get('M_drivers'))" in src
    #  and `_text` really neutralises both shapes of blank -- exercised, not assumed
    b = object.__new__(gp.PresentationBuilder)
    b.data = {'aggscore_df': pd.DataFrame([{'source': 'X', 'M-Score': 1.0, 'C-Score': 1.0,
                                            'M_drivers': np.nan}]),
              'forensic_df': None, 'carveout_labels': None}
    b._validity_cache = {}
    b._forensic_score_cache = {}
    assert b.forensic_scores('X')['M_drivers'] == '', b.forensic_scores('X')
    assert "m_drivers = _fs['M_drivers']" in src, \
        'the rule must be fed from the one published-column reader'
    assert '_mscore_drivers' not in src, 'the deck must READ the published column, not rebuild it'
    #  and the rule still keys off that variable, so a blank column disarms it
    assert "mdriver_hit = any(tok in m_drivers.upper()" in src


# =========================================================================== #
#  Q-44 -- A GUARD STRUCTURALLY BLIND TO WHAT SITS BENEATH IT                 #
# =========================================================================== #
#  `evaluate_page` set `low_conf_forensic = not forensic_valid`, and resolved `forensicValid`
#  from `AggScoreTop100` ALONE with a `True` default.  That CSV is the GENERAL pool, which
#  `carveOut` has already stripped of every financial, so `False` does not occur in it (0 rows
#  across all seven saved runs, 08-13 -> 08-29) -- and 25 of the 45 pages on the 2026-08-29
#  deck are not in the file at all and took the default.  The guard could not fire on any deck
#  the pipeline produces.  It shipped nothing wrong; it silently shipped no coverage while the
#  page read as covered.
#
#  MEASURED, OFFLINE, ON THE 2026-08-29 RUN (`--augment off`, no network, no pipeline run):
#  45 pages, 1,665 verdict cells / 1,755 `compute_verdict` calls; under the fix the guard
#  resolves 25 pages valid and 20 invalid (was: 45 valid, 0 invalid) and TEN cells change
#  icon, every one of them `sloan` on a REIT (5) or InvestmentVehicle (5) page.


def _validity_deck(agg_df, labels=None, forensic_df=None):
    b = object.__new__(gp.PresentationBuilder)
    b.data = {'aggscore_df': agg_df, 'forensic_df': forensic_df,
              'carveout_labels': labels}
    b._validity_cache = {}
    return b


def test_Q44_the_guard_resolves_the_pages_that_are_NOT_in_the_AggScore_CSV():
    """*** Q-44, 2026-08-31. ***  The whole defect in one assertion: a carve-cohort page has
    no row in `AggScoreTop100`, and must NOT therefore be treated as forensically valid.

    The run's own `carveout_labels` is the source, and it is the source ON PURPOSE -- see the
    comment on `_resolve_forensic_validity`.  It travels inside the same Boresults pickle as
    the shortlist, so it cannot disagree with the run; the sector pickle can and does (it is
    undated, rebuilt per run, and the two copies on this machine are different taxonomies --
    classifying the 45 pages from the repo copy instead of the run's flips SIX of them).
    """
    agg = pd.DataFrame({'source': ['TNK'], 'forensicValid': [True]})
    labels = pd.Series({'TNK': 'general', 'HST': 'REIT', 'PSEC': 'InvestmentVehicle',
                        'FRP.L': 'FinManager', 'NMIH': 'BalanceSheetFin',
                        'DPM.TO': 'Mining'}, name='carve_label')
    d = _validity_deck(agg, labels)
    #  the CSV still wins where it has the name
    assert d.forensic_validity('TNK') == (True, 'AggScoreTop100')
    #  ...and the four financial cohorts are now REACHED, which they never were
    for t in ('HST', 'PSEC', 'FRP.L', 'NMIH'):
        state, src = d.forensic_validity(t)
        assert state is False, (t, state, src)
        assert 'carve label' in src, (t, src)
    #  TWO-SIDED: a cohort that is NOT financial must keep its real verdicts, or the fix has
    #  replaced a guard that never fires with one that always does.
    assert d.forensic_validity('DPM.TO')[0] is True


def test_Q44_an_unresolvable_page_is_UNDETERMINED_and_the_default_is_not_True():
    """The exact defect class P-4 fixed at source, in the other column: the DEFAULT.

    NOTHING ON THE 2026-08-29 DECK EXERCISES THIS -- all 45 pages resolve (25 valid, 20
    invalid) -- so it is asserted here rather than measured.  It is the state a future deck
    reaches the moment it renders a name outside the top-100 whose carve label is 'general',
    or reads a Boresults pickle that predates `carveout_labels`.
    """
    agg = pd.DataFrame({'source': ['TNK'], 'forensicValid': [True]})
    d = _validity_deck(agg, pd.Series({'TNK': 'general'}, name='carve_label'))
    #  a name nobody classified
    assert d.forensic_validity('NEWNAME') == (None, 'unresolved')
    #  'general' is NOT 'non-financial': an UNMAPPED name also lands in general, so the label
    #  certifies nothing.  This is the assertion that stops the fix reintroducing the default
    #  one level further down.
    assert co.cohort_forensic_validity('general') is None
    assert co.cohort_forensic_validity(None) is None
    assert co.cohort_forensic_validity('NoSuchCohort') is None
    #  no carve labels at all (a pre-label pickle) -> undetermined, never valid
    assert _validity_deck(agg, None).forensic_validity('HST') == (None, 'unresolved')


def test_Q44_a_BLANK_forensicValid_in_the_CSV_falls_through_instead_of_reading_as_valid():
    """The published column is three-valued and arrives as NaN through a CSV.  The old read
    (`str(...).strip().lower() not in ('false','0','no','nan','none','')`) mapped the blank to
    False -- i.e. INVALID -- which is the opposite over-claim: it would assert 'this is a
    financial' about a name nobody classified.  Neither: fall through to the next source."""
    agg = pd.DataFrame({'source': ['HST', 'TNK'], 'forensicValid': [np.nan, True]})
    labels = pd.Series({'HST': 'REIT', 'TNK': 'general'}, name='carve_label')
    d = _validity_deck(agg, labels)
    state, src = d.forensic_validity('HST')
    assert state is False and 'carve label' in src, (state, src)
    #  ...and with nothing behind it, a blank is UNDETERMINED, not INVALID and not VALID
    assert _validity_deck(agg, None).forensic_validity('HST') == (None, 'unresolved')


def test_Q44_the_ForensicFlags_CSV_is_a_real_fallback_not_a_duplicate():
    """The AggScore CSV is a MERGE and the offline reduced schema can drop the column; the
    forensic CSV of the same run carries it independently."""
    agg = pd.DataFrame({'source': ['HST'], 'CycleHeat': [0.1]})       # no forensicValid at all
    fdf = pd.DataFrame({'source': ['HST'], 'forensicValid': [False]})
    d = _validity_deck(agg, None, forensic_df=fdf)
    assert d.forensic_validity('HST') == (False, 'ForensicFlagsTop100')


def test_Q44_evaluate_page_no_longer_carries_the_default_that_made_the_guard_dormant():
    """A BEHAVIOURAL test cannot see a guard that never fires -- that is the whole shape of
    this defect -- so the wiring is asserted structurally as well: the literal default is
    gone, and the guard keys off `is not True` (three states) rather than `not` (two)."""
    src = inspect.getsource(gp.PresentationBuilder.evaluate_page)
    assert 'forensic_valid = True' not in src, src[:400]
    assert 'low_conf_forensic = not forensic_valid' not in src
    assert 'self.forensic_validity(ticker)' in src
    assert 'forensic_state is not True' in src
    #  and the resolver never returns the old default for an unknown name
    rsrc = inspect.getsource(gp.PresentationBuilder._resolve_forensic_validity)
    assert "return None, 'unresolved'" in rsrc, rsrc


def test_Q44_the_marker_says_WHICH_of_the_two_low_confidence_facts_it_is():
    """`INVALID because it is a bank` and `we never classified it` are different facts and an
    unnamed amber collapses them.  Three DISTINCT notes, and the valid state fires nothing."""
    notes = gp.PresentationBuilder._VALIDITY_NOTE
    assert not notes[True]                          # valid -> no marker at all
    assert notes[False] and notes[None]
    assert notes[False] != notes[None]
    assert 'do not apply' in notes[False]
    assert 'could not determine' in notes[None]
    #  the reason reaches the rendered hover note, and the bare-`True` caller is unchanged
    n = gp.compute_verdict('sloan', 0.02, 'REIT', low_conf=notes[False])[1]
    assert 'value present but flagged low-confidence' in n and 'do not apply' in n
    assert gp.compute_verdict('sloan', 0.02, 'REIT', low_conf=True)[1] == (
        u'value present but flagged low-confidence (see \U0001f6a9/forensic)')
    #  P-5 still outranks it: an ABSENT value is still white even with a named reason
    assert gp.compute_verdict('sloan', np.nan, 'REIT', low_conf=notes[None]) == (
        'gray', 'value unavailable')


def test_Q44_every_carve_label_the_pipeline_can_emit_has_a_STATED_validity():
    """The structural guard that would have caught this being written wrong.  A cohort added
    to `classify` without an entry here inherits `None` silently -- fail-safe in direction,
    but it would put a whole cohort permanently in the undetermined state with nobody
    noticing.  Read off `classify`'s own source so the two cannot drift."""
    src = inspect.getsource(co.classify)
    emitted = {'general'}
    for tok, const in (("label = 'REIT'", 'REIT'), ("label = 'Mining'", 'Mining')):
        assert tok in src
        emitted.add(const)
    for const in (co.FIN1_VEHICLE, co.FIN2_MANAGER, co.FIN3_BALSHEET):
        emitted.add(const)
    missing = emitted - set(co.COHORT_FORENSIC_VALIDITY)
    assert not missing, 'carve labels with no stated forensic validity: %s' % sorted(missing)
    #  the financial cohorts are the point of the table -- pin their direction, not just
    #  their presence
    assert co.COHORT_FORENSIC_VALIDITY['REIT'] is False
    assert co.COHORT_FORENSIC_VALIDITY[co.FIN1_VEHICLE] is False
    assert co.COHORT_FORENSIC_VALIDITY[co.FIN2_MANAGER] is False
    assert co.COHORT_FORENSIC_VALIDITY[co.FIN3_BALSHEET] is False
    assert co.COHORT_FORENSIC_VALIDITY['Mining'] is True


def test_Q44_the_page_STATES_the_applicability_it_could_not_read_from_the_CSV():
    """A bare em-dash beside the word `Forensic` on 25 of 45 pages reads as `nothing to
    report`.  The page now says which of the three states it is, and names its source."""
    labels = pd.Series({'HST': 'REIT'}, name='carve_label')
    d = _deck_with_forensic({'source': 'HST'}, carveout_labels=labels)
    html = d.section_b_flags('HST')
    assert 'forensic models INVALID here' in html, html
    assert 'carve label REIT' in html, html
    #  its own class -- the M-abstention reason is a different fact and must stay separable
    assert 'forensic-applic' in html and 'forensic-why' not in html
    #  a name the run classified VALID says nothing extra (silence is correct there)
    d2 = _deck_with_forensic({'source': 'TNK', 'forensicTag': 'clean', 'forensicValid': True})
    assert 'forensic-applic' not in d2.section_b_flags('TNK')
    #  and the undetermined state is worded as an ABSENCE, not as a clean reading
    d3 = _deck_with_forensic({'source': 'ZZZZ'})
    h3 = d3.section_b_flags('ZZZZ')
    assert 'NOT DETERMINED' in h3 and 'unresolved' in h3, h3


def test_Q44_the_LEGEND_states_the_third_state_it_now_renders():
    """Same rule as P-5's legend test: the only copy of the icon spec the CEO reads is the
    legend on the page, so a state the code can render and the legend cannot explain is a
    silent spec drift."""
    legend = gp.PresentationBuilder._icon_legend(None)
    assert 'forensic models apply' in legend, legend
    assert 'no forensic classification for it exists in this run' in legend, legend


# =========================================================================== #
#  Q-66 -- THE FORENSIC SCORES REACH THE COHORT PAGES, OR SAY WHY THEY DO NOT  #
# =========================================================================== #
#  Q-44 closed the APPLICABILITY half of this: it made the deck resolve "do the forensic
#  models apply to this name" for all 45 pages instead of 20.  It explicitly left the SCORES
#  alone -- its note said M/C "stay CSV-only ... there is no offline source for the other 25",
#  which was true then.  It is not now: `forensicFlags.buildForensicFlagTable` writes a row for
#  every cohort side-list name, scored where the models apply and blank-with-a-reason where
#  they do not.  So the 25 cohort pages stop rendering a forensic layer that is blank BY
#  OMISSION and start rendering one that is either populated or explained.
#
#  MEASURED, OFFLINE, ON THE 2026-08-31 RUN (`--no-augment`, no network, no pipeline run):
#  45 pages / 1,665 verdict cells; 39 cells change, ALL of them on cohort pages (Mining 29,
#  REIT 5, InvestmentVehicle 5) and ZERO on the 20 general-pool pages.  Pages carrying a real
#  M-Score go 15 -> 19, a real C-Score 20 -> 25.


def _score_deck(agg_df, forensic_df=None):
    """A builder with just enough state for `forensic_scores` / `forensic_validity`."""
    b = object.__new__(gp.PresentationBuilder)
    b.data = {'aggscore_df': agg_df, 'forensic_df': forensic_df, 'carveout_labels': None}
    b._validity_cache = {}
    b._forensic_score_cache = {}
    return b


def test_Q66_a_cohort_page_resolves_its_scores_from_the_FORENSIC_csv():
    """The hole: a cohort name is not in `AggScoreTop100` (the general pool's artifact), so the
    deck's aggscore-only read returned NaN for it and the page rendered a no-value icon on both
    forensic rows -- on 25 of 45 pages.

    The general pool must still resolve from `AggScoreTop100`, not from the forensic CSV: that
    file is the only one carrying the API-reconciled columns, and quietly re-sourcing 20 pages
    would be a change nobody asked for.  The -99.0 sentinel is what makes that assertion real
    rather than coincidental."""
    agg = pd.DataFrame([{'source': 'TNK', 'M-Score': -0.62, 'C-Score': 3.0,
                         'M_flag_gt_-1.78': False, 'M_drivers': 'AQI(+0.16)'}])
    fdf = pd.DataFrame([
        {'source': 'TNK', 'M_score_mean': -99.0, 'C_score_mean': -99.0,
         'M_flag_gt_-1.78': True, 'M_drivers': 'WRONG', 'forensicValid': True,
         'forensicReason': '', 'forensicNote': ''},
        {'source': 'TXG.TO', 'M_score_mean': 3.5957, 'C_score_mean': 3.0,
         'M_flag_gt_-1.78': True, 'M_drivers': 'DSRI(+4.00, ratio>neut)',
         'forensicValid': True, 'forensicReason': '', 'forensicNote': 'cohort caveat here'},
    ])
    d = _score_deck(agg, fdf)
    #  the cohort name now HAS a score, and it is the forensic table's
    tx = d.forensic_scores('TXG.TO')
    assert tx['M-Score'] == 3.5957 and tx['C-Score'] == 3.0, tx
    assert tx['source'] == 'ForensicFlagsTop100'
    assert tx['M_flag'] is True and 'DSRI' in tx['M_drivers']
    assert tx['note'] == 'cohort caveat here' and tx['reason'] == ''
    #  the general name still resolves from AggScore -- the -99.0 sentinel proves precedence
    tn = d.forensic_scores('TNK')
    assert tn['M-Score'] == -0.62 and tn['source'] == 'AggScoreTop100', tn
    assert tn['M_flag'] is False and tn['M_drivers'] == 'AQI(+0.16)', tn
    #  a name in neither frame is still absent, never a fabricated zero
    zz = d.forensic_scores('NOSUCH')
    assert np.isnan(zz['M-Score']) and np.isnan(zz['C-Score']) and zz['source'] == 'unresolved'
    print("PASS test_Q66_a_cohort_page_resolves_its_scores_from_the_FORENSIC_csv")


def test_Q66_a_REFUSED_cohort_page_says_WHY_instead_of_an_unexplained_dash():
    """A blank with no sentence beside it reads as a broken tool rather than as a refusal --
    the same "presentation must be correctly suggestive" constraint that put
    `M_abstain_reason` on the page.  Four of the five cohorts are refused by the Q-66 ruling,
    so this is what 20 of the 25 cohort pages render.

    Two-sided, because the failure mode is symmetric: a page that HAS a score must not render
    a refusal sentence, and a page carrying only a CAVEAT must render the caveat and not the
    refusal wording."""
    fdf = pd.DataFrame([
        {'source': 'HST', 'M_score_mean': np.nan, 'C_score_mean': np.nan,
         'M_flag_gt_-1.78': '', 'M_drivers': '', 'forensicValid': False,
         'forensicReason': 'the Beneish / Montier / Sloan models do not apply to a REIT',
         'forensicNote': ''},
        {'source': 'TXG.TO', 'M_score_mean': 3.5957, 'C_score_mean': 3.0,
         'M_flag_gt_-1.78': True, 'M_drivers': 'DSRI', 'forensicValid': True,
         'forensicReason': '', 'forensicNote': 'ramp-up caveat'},
    ])
    d = _score_deck(pd.DataFrame(columns=['source', 'M-Score', 'C-Score']), fdf)
    hst = d.forensic_scores('HST')
    assert np.isnan(hst['M-Score']) and hst['reason'].startswith('the Beneish'), hst
    assert hst['note'] == '', 'a caveat must never stand in for a refusal'
    #  ...and the blank flag reads as a NON-verdict, not as a manipulation flag
    assert hst['M_flag'] is False, hst
    txg = d.forensic_scores('TXG.TO')
    assert txg['reason'] == '' and txg['note'] == 'ramp-up caveat'
    #  the section that renders them keeps the two strings in mutually exclusive branches
    src = inspect.getsource(gp.PresentationBuilder.section_f_forensic)
    assert "if _fs['reason']:" in src and "elif _fs['note']:" in src, src
    assert 'Why there is no M / C score' in src
    assert 'Read the scores above with this caveat' in src
    #  and the em dash, not the literal string 'nan', is what an absent score renders as
    assert 'm_score = "—" if np.isnan(safe_float(_fs[\'M-Score\'])) else' in src, src
    print("PASS test_Q66_a_REFUSED_cohort_page_says_WHY_instead_of_an_unexplained_dash")


def test_Q66_the_LOW_CONFIDENCE_note_carries_the_cohorts_OWN_reason():
    """Q-44's note said "... do not apply to a financial" for every refused page.  That is
    right for a bank and wrong for the four different grounds the four refused cohorts carry --
    a REIT and a BDC are refused for different reasons and the reader is entitled to which.

    AND A CALIBRATION CAVEAT MUST NOT FIRE THE MARKER.  Mining IS scored, so its pages keep the
    honest value-based icon; the legend defines the amber marker here as "it is not established
    that the forensic models apply", which is false for a miner.  Amber on all five Mining
    pages forever would be the mirror image of the Q-44 guard that could never fire."""
    src = inspect.getsource(gp.PresentationBuilder.evaluate_page)
    assert "_why = _fs['reason'] if forensic_state is False and _fs['reason'] else None" in src
    assert "(_why or self._VALIDITY_NOTE[forensic_state])" in src
    #  the caveat is appended to the forensic RULES only, and only when their forensic limb fired
    assert "_caveat = ('  COHORT CAVEAT — ' + _fnote) if _fnote else ''" in src
    assert "_r1cav = _caveat if m_flag else ''" in src
    assert "_r5cav = _caveat if (mdriver_hit and not sloan_hi) else ''" in src
    #  ...and R3, the HIGH-tier rule, names the driver as well as the score: a HIGH flag that
    #  says only "M-Score 3.60 > 0" is a number with no mechanism, and on a Mining page the
    #  mechanism IS the finding.
    assert "_drv = ('  Driven by: ' + m_drivers)" in src
    #  the caveat must NOT be routed into the low-confidence marker: nothing between the
    #  marker's assignment and the verdict block mentions it.  (`_caveat` is constructed
    #  further down, immediately before the flag rules -- which is the only place it belongs.)
    between = src.split('low_conf_forensic = ')[1].split('# ---- verdicts')[0]
    assert '_caveat' not in between, between
    assert src.index('low_conf_forensic = ') < src.index("_caveat = ("), \
        'the caveat is built after the low-confidence marker, so it cannot reach it'
    print("PASS test_Q66_the_LOW_CONFIDENCE_note_carries_the_cohorts_OWN_reason")


def test_Q66_a_ROUND_TRIPPED_blank_reason_does_not_render_the_literal_nan():
    """FOUND BY RE-RENDERING THE DECK, NOT BY REASONING (2026-09-01).

    `str(x or '')` is not safe on a cell that has been through a CSV.  A blank `forensicReason`
    is `''` in memory and NaN after the round-trip, and `float('nan') or ''` short-circuits to
    the NaN because NaN is TRUTHY -- so `str(...)` yields the literal `'nan'`, which is not
    empty, so the caller's `if reason:` branch fires and the page renders

        Why there is no M / C score:  nan

    BESIDE A REAL M-SCORE.  Measured on the 2026-08-31 deck: all five Mining pages, i.e. every
    page the Q-66 extension actually scores -- the fix's own output, wrong on exactly the rows
    it was written to populate.  Third member of the same family as `_flag_true` and
    `published_forensic_validity`: a text column needs its own reader too.

    Both directions, because the useless half is easy to write: a REAL reason must survive."""
    fdf = pd.DataFrame([
        #  the blank-reason row: scored, so the page must say nothing about why it is not
        {'source': 'TXG.TO', 'M_score_mean': 3.5957, 'C_score_mean': 3.0,
         'M_flag_gt_-1.78': True, 'M_drivers': 'DSRI(+4.00)', 'forensicValid': True,
         'forensicReason': '', 'forensicNote': 'ramp-up caveat', 'forensicTag': 'single-flag'},
        #  a refused row whose NOTE is blank -- the mirror image
        {'source': 'HST', 'M_score_mean': np.nan, 'C_score_mean': np.nan,
         'M_flag_gt_-1.78': '', 'M_drivers': '', 'forensicValid': False,
         'forensicReason': 'the models do not apply to a REIT', 'forensicNote': '',
         'forensicTag': 'cohort REIT: forensic-inapplicable'},
    ])
    #  THROUGH A REAL CSV, not a hand-built NaN: the round trip is the mechanism, and a
    #  hand-placed np.nan would pass on a fix that only handled one of its two shapes.
    buf = io.StringIO()
    fdf.to_csv(buf, index=False)
    buf.seek(0)
    back = pd.read_csv(buf)
    d = _score_deck(pd.DataFrame(columns=['source', 'M-Score', 'C-Score']), back)

    scored = d.forensic_scores('TXG.TO')
    assert scored['reason'] == '', repr(scored['reason'])
    assert scored['note'] == 'ramp-up caveat', repr(scored['note'])
    assert scored['M_drivers'] == 'DSRI(+4.00)'
    refused = d.forensic_scores('HST')
    assert refused['reason'] == 'the models do not apply to a REIT'
    assert refused['note'] == '', repr(refused['note'])
    assert refused['M_drivers'] == '', repr(refused['M_drivers'])
    #  ...and nothing anywhere in the resolved payload is the string 'nan'
    for k, v in list(scored.items()) + list(refused.items()):
        assert not (isinstance(v, str) and v.strip().lower() == 'nan'), (k, v)
    #  ONE READER, not a private copy: the deck delegates to `forensicFlags.cell_text`, which
    #  sits beside `_flag_true` and `published_forensic_validity`.  Three near-identical
    #  absence readers in three files is how the BOOLEAN ones came to disagree.
    src = inspect.getsource(gp.PresentationBuilder.forensic_scores)
    assert '_text = _fflags.cell_text' in src, src
    assert 'def _text(' not in src, 'a private copy of the absence reader has come back'
    print("PASS test_Q66_a_ROUND_TRIPPED_blank_reason_does_not_render_the_literal_nan")
