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
