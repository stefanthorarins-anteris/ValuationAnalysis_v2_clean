"""
OFFLINE tests for the delisted-entity (survivorship) ingestion.  Synthetic
fixtures, deterministic, NO network (every network touch point is injected).

Run:  python baseline_tools/test_delisted_ingest.py

Proves the load-bearing findings + review items:
  * F-A  -- the datefail gate is BYPASSED on the dead path (a 2020-dead name that
            live would drop is KEPT)                        -> test_FA_datefail_bypass
  * F-B  -- <16q history ACCEPTED + tagged short_history    -> test_FB_short_history
  * None-trap -- a None mid-pagination ABORTS, never ends    -> test_registry_none_trap
  * bulk-absent branch -- weaker path + loud warn, no crash  -> test_bulk_absent_branch
  * ride-alongs (split basis, filing dates) PASS/FAIL logic  -> test_ridealong_*
  * API key NEVER in any log line                            -> test_api_key_never_logged
  * -ingest_delisted default OFF / config wiring / LOW-B     -> test_config_*
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import delisted_ingest as di
import getData_fmp as gdf
import failTests as ft
import configuration as cf
from run_logging import RunLogger


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def make_http_get(statement_map, status_map=None):
    """Return an http_get(url) that serves per-(symbol,endpoint) JSON from
    statement_map: {symbol: {'income-statement': [...], 'balance-sheet-statement':
    [...], ...}}.  Endpoints absent from the map return []."""
    status_map = status_map or {}
    endpoints = ["key-metrics", "ratios", "income-statement",
                 "balance-sheet-statement", "cash-flow-statement"]

    def _get(url):
        sym = None
        ep = None
        for e in endpoints:
            if f"/{e}/" in url:
                ep = e
                sym = url.split(f"/{e}/")[1].split("?")[0]
                break
        st = status_map.get(sym, 200)
        payload = statement_map.get(sym, {}).get(ep, [])
        return FakeResp(payload, status=st)

    return _get


def _stmt_rows(n_q, newest_year=2020):
    """n_q quarterly rows, newest first, dates descending from <newest_year>-12-31."""
    rows = []
    dates = pd.date_range(f"{newest_year}-12-31", periods=n_q, freq="-3MS")
    for d in dates:
        rows.append({"date": d.strftime("%Y-%m-%d"), "revenue": 100.0,
                     "netIncome": 10.0, "freeCashFlow": 8.0,
                     "weightedAverageShsOut": 50.0, "priceEarningsRatio": 12.0})
    return rows


def _full_statement_map(sym, n_q=20, newest_year=2020):
    rows = _stmt_rows(n_q, newest_year)
    return {sym: {ep: [dict(r) for r in rows] for ep in
                  ["key-metrics", "ratios", "income-statement",
                   "balance-sheet-statement", "cash-flow-statement"]}}


# --------------------------------------------------------------------------- #
# F-A : datefail gate bypassed on the dead path
# --------------------------------------------------------------------------- #
def test_FA_datefail_bypass():
    sm = _full_statement_map("DEADCO", n_q=20, newest_year=2020)
    http = make_http_get(sm)
    # LIVE gate: compyear=2025 > newest income year 2020 -> datefail -> dropped
    km, *_ = gdf.getFsData_fmp("DEADCO", "quarter", 160,
                               "https://x/api/", "KEY", 2025, [], [], [], [],
                               dead_path=False, http_get=http)
    assert isinstance(km, int) and km == -37707, "live gate must datefail-drop the 2020-dead name"
    # DEAD path: datefail BYPASSED -> the name is KEPT
    km2, fr2, inc2, bs2, cf2, *_ = gdf.getFsData_fmp(
        "DEADCO", "quarter", 160, "https://x/api/", "KEY", 2025, [], [], [], [],
        dead_path=True, http_get=http)
    assert not (isinstance(km2, int) and km2 == -37707), "dead path must KEEP the 2020-dead name (F-A)"
    assert len(inc2) == 20
    print("PASS F-A datefail bypass (live drops 2020-dead name; dead path keeps it)")


# --------------------------------------------------------------------------- #
# F-B : short history accepted + tagged
# --------------------------------------------------------------------------- #
def test_FB_short_history():
    sm = _full_statement_map("SHORTCO", n_q=8, newest_year=2020)
    http = make_http_get(sm)
    # LIVE gate with compyear<=2020 passes datefail but 8<16 -> lenfail -> dropped
    km, *_ = gdf.getFsData_fmp("SHORTCO", "quarter", 160,
                               "https://x/api/", "KEY", 2019, [], [], [], [],
                               dead_path=False, http_get=http)
    assert isinstance(km, int) and km == -37707, "live gate must lenfail-drop the 8q name"
    # DEAD loop: accepted + tagged short_history + filing_date_source
    order = [{"symbol": "SHORTCO", "entity_id": "SHORTCO", "delistedDate": "2020-12-31"}]
    results, meta, empty, failcode = di.fetch_dead_fundamentals(
        order, "https://x/api/", "KEY", limit=160, http_get=http, sleep=lambda s: None)
    assert "SHORTCO" in [r for r in results] or "SHORTCO" in results
    rec = results["SHORTCO"]
    assert rec["short_history"] is True, "8q must be tagged short_history (F-B)"
    assert rec["filing_date_source"] == "fixed_lag"
    assert (rec["inc"]["filing_date_source"] == "fixed_lag").all()
    assert meta["short_history"] == 1 and meta["emptyfail"] == 0
    print("PASS F-B short history accepted + tagged (short_history, fixed_lag)")


def test_dead_fund_emptyfail_artifact():
    # all endpoints return [] -> emptyfail -> skipped + recorded as first-class artifact
    http = make_http_get({"GONECO": {}})   # empty -> [] everywhere
    order = [{"symbol": "GONECO", "entity_id": "GONECO", "delistedDate": "2015-06-30"}]
    results, meta, empty, failcode = di.fetch_dead_fundamentals(
        order, "https://x/api/", "KEY", http_get=http, sleep=lambda s: None)
    assert results == {} and meta["emptyfail"] == 1
    assert meta["failcode"] == 0 and not failcode, "an empty response is NOT a failcode"
    assert empty and empty[0]["symbol"] == "GONECO"
    assert empty[0]["whyfail"] == "emptyfail"
    print("PASS dead-fund emptyfail is skipped + logged as a first-class artifact")


# --------------------------------------------------------------------------- #
# HIGH-1 : a transient 429/5xx after retries lands in RE-AUDIT, NOT emptyfail
# --------------------------------------------------------------------------- #
def test_dead_fund_failcode_reaudit():
    """A dead name whose endpoints return HTTP 429 (throttle) -- surviving the
    retry/backoff -- MUST be bucketed as a `failcode` (fetch-unknown / re-audit) and
    NOT mislabelled `emptyfail` ("no fundamentals").  This is the HIGH-1 bias hole."""
    # 429 on every endpoint for THROTTLED; a genuinely empty name for comparison.
    sm = _full_statement_map("THROTTLED", n_q=20, newest_year=2020)
    http = make_http_get(sm, status_map={"THROTTLED": 429})
    order = [{"symbol": "THROTTLED", "entity_id": "THROTTLED", "delistedDate": "2020-12-31"}]
    results, meta, empty, failcode = di.fetch_dead_fundamentals(
        order, "https://x/api/", "KEY", http_get=http, sleep=lambda s: None)
    assert results == {}, "a throttled name yields no fundamentals record"
    assert meta["emptyfail"] == 0 and not empty, "a 429 must NOT be counted as emptyfail"
    assert meta["failcode"] == 1 and failcode, "a 429 must be a first-class failcode/re-audit"
    assert failcode[0]["symbol"] == "THROTTLED" and failcode[0]["whyfail"] == "failcode"
    # A 5xx behaves the same way (transient/server error -> re-audit, not no-data).
    http5 = make_http_get(sm, status_map={"THROTTLED": 503})
    _, meta5, empty5, fc5 = di.fetch_dead_fundamentals(
        order, "https://x/api/", "KEY", http_get=http5, sleep=lambda s: None)
    assert meta5["failcode"] == 1 and not empty5, meta5
    print("PASS HIGH-1 dead-fund failcode/emptyfail split (429 & 5xx -> re-audit, not no-data)")


# --------------------------------------------------------------------------- #
# HIGH-1 : the retry/backoff getter (safe_http_get) retries transient failures
# --------------------------------------------------------------------------- #
def test_safe_http_get_retry_and_failcode():
    import getData_gen as gdg

    # (a) two 429s then a 200 -> retried, final status 200; backoff sleeps twice.
    calls = {"n": 0}
    slept = []

    def flaky(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        return FakeResp([{"ok": True}], status=200 if calls["n"] >= 3 else 429)

    resp = gdg.safe_http_get("http://x", _get=flaky, sleep=lambda s: slept.append(s))
    assert resp.status_code == 200 and calls["n"] == 3, (resp.status_code, calls["n"])
    assert len(slept) == 2, slept   # backed off before each retry

    # (b) a persistent 503 -> exhausts retries, hands back the FAILING response
    #     (status in the 400-599 failcode range), never a None/crash.
    def down(url, params=None, headers=None, timeout=None):
        return FakeResp([], status=503)

    r2 = gdg.safe_http_get("http://x", _get=down, sleep=lambda s: None)
    assert r2.status_code == 503 and 400 <= r2.status_code < 600

    # (c) every attempt RAISES (timeout/conn error) -> a _FailedResponse with a
    #     failing status_code, so the caller records a failcode instead of crashing.
    def raiser(url, params=None, headers=None, timeout=None):
        raise gdg.requests.RequestException("boom")

    r3 = gdg.safe_http_get("http://x", _get=raiser, sleep=lambda s: None)
    assert r3 is not None and 400 <= r3.status_code < 600, r3
    assert r3.json() == []
    print("PASS safe_http_get retries 429, exhausts 5xx to a failcode, never crashes")


# --------------------------------------------------------------------------- #
# None-trap : a None mid-pagination ABORTS (never end-of-list)
# --------------------------------------------------------------------------- #
def test_registry_none_trap():
    page_full = [{"symbol": f"S{i}", "companyName": "C", "exchange": "NYSE",
                  "ipoDate": "2000-01-01", "delistedDate": "2020-01-01"}
                 for i in range(100)]
    # page0 full, page1 -> None (a 429 after backoff).  MUST raise, NOT terminate.
    seq = [page_full, None, page_full]
    urls = []

    def fake_get(url):
        urls.append(url)
        return seq[len(urls) - 1] if len(urls) <= len(seq) else []

    raised = False
    try:
        di.fetch_delisted_registry("https://x/api/", "KEY", get=fake_get,
                                   sleep=lambda s: None)
    except di.DelistedRegistryError:
        raised = True
    assert raised, "a None mid-pagination MUST abort (bias trap), not end the pull"
    # single-slash URL form (double-slash cleanup)
    assert "api/v3/delisted-companies" in urls[0] and "api//v3" not in urls[0]
    print("PASS None-trap aborts pagination (no silent truncation) + single-slash URL")


def test_registry_terminates_on_short_and_empty():
    full = [{"symbol": f"A{i}", "companyName": "C", "exchange": "NYSE",
             "ipoDate": "2001-01-01", "delistedDate": f"{2005 + (i % 15)}-06-30"}
            for i in range(100)]
    short = [{"symbol": f"B{i}", "companyName": "C", "exchange": "NASDAQ",
              "ipoDate": "2001-01-01", "delistedDate": "2019-06-30"}
             for i in range(37)]
    # full page (size 100) then a short page (37) -> terminate on short_page
    seq = [full, short]
    it = iter(seq)
    df, meta = di.fetch_delisted_registry("https://x/api/", "KEY",
                                          get=lambda u: next(it, []),
                                          sleep=lambda s: None)
    assert meta["termination"] == "short_page"
    assert meta["total"] == 137
    assert set(["symbol", "companyName", "exchange", "ipoDate", "delistedDate"]).issubset(df.columns)
    # per-year coverage byproduct populated
    assert meta["coverage_by_year"] and sum(meta["coverage_by_year"].values()) == 137
    print("PASS registry terminates on short page + keeps fields + per-year coverage")


# --------------------------------------------------------------------------- #
# Entity-id assignment (both modes, F4 band)
# --------------------------------------------------------------------------- #
def test_entity_ids_recycled_dead_plus_live():
    """MEDIUM-1 (was MASKED): a GENUINE recycled ticker -- a DEAD registry entity
    whose symbol is ALSO a currently-LIVE occupant -- must SPLIT: the dead entity
    becomes SYM_2 while the live successor keeps the bare SYM.  The old test used two
    SEPARATE symbols (RCY/SOLO) and asserted only n_split>=n_merge, so it passed
    vacuously while the dead-vs-live split never fired (dead inherited the bare
    symbol in BOTH modes).  This test exercises the real boundary and FAILS against
    the pre-fix code (which merged the dead RCY into the bare 'RCY')."""
    reg = pd.DataFrame([
        {"symbol": "RCY", "companyName": "Old Recycle Co", "exchange": "NYSE",
         "ipoDate": "1995-01-01", "delistedDate": "2012-01-01"},
        {"symbol": "SOLO", "companyName": "Solo Inc", "exchange": "NASDAQ",
         "ipoDate": "2001-01-01", "delistedDate": "2016-01-01"},   # dead, NOT live
    ])
    out = di.assign_registry_entities(reg, live_symbols=["RCY"])
    assert "entity_id" in out.columns and "entity_id_split" in out.columns
    rcy = out[out["symbol"] == "RCY"].iloc[0]
    solo = out[out["symbol"] == "SOLO"].iloc[0]
    # THE FIX: the dead recycled entity is SYM_2 in BOTH modes (live keeps bare SYM,
    # and the live occupant is not emitted -- only the dead registry row is).
    assert rcy["entity_id"] == "RCY_2", ("recycled dead entity must be RCY_2 (merge), "
                                         f"got {rcy['entity_id']}")
    assert rcy["entity_id_split"] == "RCY_2", rcy["entity_id_split"]
    # a NON-recycled dead entity (no live occupant) keeps its bare symbol
    assert solo["entity_id"] == "SOLO" and solo["entity_id_split"] == "SOLO"
    # split-first floor still >= merge-mode ceiling
    assert out["entity_id_split"].nunique() >= out["entity_id"].nunique()
    print("PASS recycled dead+live SPLITS to SYM_2 in both modes (MEDIUM-1 un-masked)")


def test_dead_fundamentals_sliced_to_entity_life():
    """MEDIUM-1 aggravator: a recycled symbol's LIVE-successor quarters (returned by
    FMP under the reused ticker) must be SLICED OUT of the dead entity's record, so
    the dead entity's history does not carry post-delisting successor data."""
    # REUSE reports quarterly 2016..2023; dead entity delisted 2018-12-31.
    dates = pd.date_range("2016-03-31", "2023-12-31", freq="QE")
    rows = [{"date": d.strftime("%Y-%m-%d"), "revenue": 100.0, "netIncome": 10.0,
             "freeCashFlow": 8.0, "weightedAverageShsOut": 50.0,
             "priceEarningsRatio": 12.0} for d in dates]
    sm = {"REUSE": {ep: [dict(r) for r in rows] for ep in
                    ["key-metrics", "ratios", "income-statement",
                     "balance-sheet-statement", "cash-flow-statement"]}}
    http = make_http_get(sm)
    order = [{"symbol": "REUSE", "entity_id": "REUSE_2",
              "ipoDate": "2016-01-01", "delistedDate": "2018-12-31"}]
    results, meta, empty, failcode = di.fetch_dead_fundamentals(
        order, "https://x/api/", "KEY", http_get=http, sleep=lambda s: None)
    inc = results["REUSE_2"]["inc"]
    yrs = pd.to_datetime(inc["date"]).dt.year
    assert yrs.max() <= 2018, f"successor quarters leaked past delisting: {sorted(set(yrs))}"
    assert yrs.min() >= 2016 and (yrs > 2018).sum() == 0
    # exactly the pre-delisting rows survive (2016Q1..2018Q4 = 12 quarters)
    assert len(inc) == 12, len(inc)
    print("PASS dead fundamentals sliced to entity life-span (recycled successor removed)")


def test_dead_fetch_order_recent_first():
    reg = pd.DataFrame([
        {"symbol": "OLD", "entity_id": "OLD", "delistedDate": "2011-03-31"},
        {"symbol": "NEW", "entity_id": "NEW", "delistedDate": "2024-09-30"},
        {"symbol": "MID", "entity_id": "MID", "delistedDate": "2018-06-30"},
    ])
    order = di.dead_fetch_order(reg)
    assert [o["symbol"] for o in order] == ["NEW", "MID", "OLD"], order
    print("PASS dead-fetch order is recent-era-first (interrupted run keeps recent cohort)")


# --------------------------------------------------------------------------- #
# Bulk prices : present/absent two-outcome branch (never per-symbol)
# --------------------------------------------------------------------------- #
def test_bulk_absent_branch():
    dead = ["DEADX", "DEADY"]
    # bulk responses contain ONLY live names -> dead symbols ABSENT
    def fake_get(url):
        return [{"symbol": "AAPL", "close": 150.0, "adjClose": 150.0},
                {"symbol": "MSFT", "close": 300.0, "adjClose": 300.0}]
    dates = di.monthly_grid(trailing_months=3, end="2026-06-30")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prices, meta = di.fetch_bulk_prices(dates, dead, "https://x/api/", "KEY",
                                            get=fake_get, sleep=lambda s: None)
        assert meta["mode"] == "delistedDate_presence_only"
        assert any("BULK-PRICE ABSENT" in str(x.message) for x in w), "must warn LOUDLY"
    assert prices.empty
    # death classification on the weaker path -> presence_only, no drawdown
    reg = pd.DataFrame([{"symbol": "DEADX", "entity_id": "DEADX",
                         "ipoDate": "2000-01-01", "delistedDate": "2019-01-01"}])
    death = di.classify_death_signature(prices, reg, mode=meta["mode"])
    assert (death["classification"] == "delisted_presence_only").all()
    print("PASS bulk-absent branch (weaker path, loud warn, no per-symbol, no crash)")


def test_bulk_present_and_death_signature():
    dead = ["DEADX"]
    # DEADX present pre-delisting: peak 100 -> terminal 5 = -95% -> DEATH
    responses = {
        "2023-01-31": [{"symbol": "DEADX", "date": "2023-01-31", "close": 100.0, "adjClose": 100.0}],
        "2026-06-30": [{"symbol": "DEADX", "date": "2026-06-30", "close": 5.0, "adjClose": 5.0}],
    }
    def fake_get(url):
        d = url.split("date=")[1].split("&")[0]
        return responses.get(d, [{"symbol": "DEADX", "date": d, "close": 60.0, "adjClose": 60.0}])
    dates = di.monthly_grid(trailing_months=2, end="2026-06-30", extra_dates=["2023-01-31"])
    prices, meta = di.fetch_bulk_prices(dates, dead, "https://x/api/", "KEY",
                                        get=fake_get, sleep=lambda s: None,
                                        probe_symbol="DEADX", probe_date="2023-01-31")
    assert meta["mode"] == "full" and meta["probe_present"] is True
    reg = pd.DataFrame([{"symbol": "DEADX", "entity_id": "DEADX",
                         "ipoDate": "2000-01-01", "delistedDate": "2026-12-31"}])
    death = di.classify_death_signature(prices, reg, mode="full")
    assert death.iloc[0]["classification"] == "death", death.iloc[0].to_dict()
    print("PASS bulk-present + death-signature drawdown (-95% -> death), probe hit")


def test_bulk_endpoint_unavailable_vs_absent():
    """LOW-2: a TOTAL outage (every date None) must be distinguished from structural
    dead-absence -- it is 'bulk_endpoint_unavailable' (retry), NOT the presence-only
    degrade.  Contrast with test_bulk_absent_branch where the endpoint IS working."""
    dead = ["DEADX", "DEADY"]

    def outage(url):
        return None            # every date fails -> no live data ever returned

    dates = di.monthly_grid(trailing_months=3, end="2026-06-30")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prices, meta = di.fetch_bulk_prices(dates, dead, "https://x/api/", "KEY",
                                            get=outage, sleep=lambda s: None,
                                            live_symbols=["AAPL", "MSFT"])
        assert meta["mode"] == "bulk_endpoint_unavailable", meta
        assert meta["endpoint_working"] is False
        assert any("ENDPOINT UNAVAILABLE" in str(x.message) for x in w), "must warn OUTAGE"
    assert prices.empty
    # live-symbol confirmation: if a live symbol IS present but dead ones absent ->
    # the endpoint IS working -> the (weaker) presence-only mode, not outage.
    def live_only(url):
        return [{"symbol": "AAPL", "close": 150.0, "adjClose": 150.0}]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _, meta2 = di.fetch_bulk_prices(dates, dead, "https://x/api/", "KEY",
                                        get=live_only, sleep=lambda s: None,
                                        live_symbols=["AAPL"])
    assert meta2["mode"] == "delistedDate_presence_only" and meta2["endpoint_working"] is True
    print("PASS bulk outage != structural absence; live-symbol probe gates the branch (LOW-2)")


def test_resume_order_anchor_roundtrip():
    """MEDIUM-2: the persisted enumeration anchor round-trips EXACTLY (entity_id,
    symbol, dates incl. None) so `-startfromlastindex` replays the same order even if
    the freshly-pulled registry drifted between the interrupt and the resume."""
    order = [
        {"symbol": "NEW", "entity_id": "NEW", "ipoDate": "2001-01-01", "delistedDate": "2024-09-30"},
        {"symbol": "BBBY", "entity_id": "BBBY_2", "ipoDate": "1992-06-01", "delistedDate": "2023-05-01"},
        {"symbol": "NOIPO", "entity_id": "NOIPO", "ipoDate": None, "delistedDate": "2010-01-01"},
    ]
    path = os.path.join(REPO, "run_logs", "test_order_anchor.txt")
    di._write_order(order, path)
    back = di._read_order(path)
    os.remove(path)
    assert [o["entity_id"] for o in back] == ["NEW", "BBBY_2", "NOIPO"], back
    assert [o["symbol"] for o in back] == ["NEW", "BBBY", "NOIPO"], back
    assert back[1]["ipoDate"] == "1992-06-01" and back[1]["delistedDate"] == "2023-05-01"
    assert back[2]["ipoDate"] is None, "None date must round-trip as None, not ''"
    print("PASS resume order anchor round-trips exactly (MEDIUM-2 deterministic replay)")


# --------------------------------------------------------------------------- #
# Ride-along verifications
# --------------------------------------------------------------------------- #
def test_ridealong_split():
    # close 4:1 drop across split; adjClose continuous -> close is RAW -> PASS
    hist = {"historical": [
        {"date": "2020-08-28", "close": 500.0, "adjClose": 125.0},
        {"date": "2020-08-31", "close": 125.0, "adjClose": 125.0},
    ]}
    ok, vals = di.classify_split_adjustment(hist, split_date="2020-08-31")
    assert ok and vals["close_ratio"] > 3.5, vals
    # close == adjClose across split (both continuous) -> ADJUSTED -> FAIL
    hist_adj = {"historical": [
        {"date": "2020-08-28", "close": 125.0, "adjClose": 125.0},
        {"date": "2020-08-31", "close": 125.0, "adjClose": 125.0},
    ]}
    bad, vals2 = di.classify_split_adjustment(hist_adj, split_date="2020-08-31")
    assert not bad, vals2
    print("PASS ride-along split-adjustment (raw close -> PASS; adjusted -> FAIL)")


def test_ridealong_filing():
    rows = [{"date": f"{y}-12-31", "fillingDate": f"{y + 1}-02-15",
             "acceptedDate": f"{y + 1}-02-14"} for y in range(2015, 2025)]
    ok, vals = di.classify_filing_dates(rows, min_year=2023)
    assert ok and vals["earliest_availability_year"] == 2015, vals
    # none populated -> FAIL, fixed-lag fallback everywhere
    rows_np = [{"date": f"{y}-12-31", "fillingDate": None, "acceptedDate": None}
               for y in range(2015, 2025)]
    bad, vals2 = di.classify_filing_dates(rows_np, min_year=2023)
    assert not bad, vals2
    print("PASS ride-along filing-date availability (populated->PASS; null->FAIL)")


# --------------------------------------------------------------------------- #
# Security : API key NEVER in a log line
# --------------------------------------------------------------------------- #
def test_api_key_never_logged():
    key = "SUPERSECRETKEY123"
    tmp = os.path.join(REPO, "run_logs")
    logger = RunLogger("unittest_key", out_dir=tmp, secrets=[key], echo=False)
    # deliberately try to leak the key through a keyed URL + a nested field
    logger.data("registry_page_ok",
                url=f"https://x/api/v3/delisted-companies?page=0&apikey={key}",
                nested={"deep": [f"prefix-{key}-suffix"]})
    logger.write_manifest()
    logger.close()
    events = open(logger.events_path, encoding="utf-8").read()
    manifest = open(logger.manifest_path, encoding="utf-8").read()
    assert key not in events, "API KEY LEAKED into the event log"
    assert key not in manifest, "API KEY LEAKED into the manifest"
    assert "***" in events
    os.remove(logger.events_path)
    os.remove(logger.manifest_path)
    print("PASS API key never appears in any log line (scrubbed to ***)")


# --------------------------------------------------------------------------- #
# Config wiring + review-addendum LOW-B
# --------------------------------------------------------------------------- #
def test_config_ingest_flag_default_off():
    base = cf.getDataFetchConfiguration([])
    assert base["ingest_delisted"] == 0 and base["as_of"] is None
    on = cf.getDataFetchConfiguration(["-ingest_delisted"])
    assert on["ingest_delisted"] == 1 and on["as_of"] is None
    print("PASS -ingest_delisted default OFF; ON when passed; as_of untouched")


def test_config_asof_lastarg_no_indexerror():
    # LOW-B: -asof as the FINAL arg (no date) -> clear Exception, NOT IndexError
    try:
        cf.getDataFetchConfiguration(["-asof"])
        raise AssertionError("should have raised a clear error")
    except Exception as e:
        assert not isinstance(e, IndexError), "must not be a raw IndexError"
        assert "asof" in str(e).lower()
    # MEDIUM-B: a valid -asof warns PARTIAL-PIT
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = cf.getDataFetchConfiguration(["-asof", "2023-01-27"])
        assert c["as_of"] == "2023-01-27"
        assert any("PARTIAL-PIT" in str(x.message) for x in w), "must warn partial-PIT"
    print("PASS -asof last-arg raises clean error (LOW-B) + PARTIAL-PIT warn (MEDIUM-B)")


if __name__ == "__main__":
    test_FA_datefail_bypass()
    test_FB_short_history()
    test_dead_fund_emptyfail_artifact()
    test_dead_fund_failcode_reaudit()
    test_safe_http_get_retry_and_failcode()
    test_registry_none_trap()
    test_registry_terminates_on_short_and_empty()
    test_entity_ids_recycled_dead_plus_live()
    test_dead_fundamentals_sliced_to_entity_life()
    test_dead_fetch_order_recent_first()
    test_bulk_absent_branch()
    test_bulk_present_and_death_signature()
    test_bulk_endpoint_unavailable_vs_absent()
    test_resume_order_anchor_roundtrip()
    test_ridealong_split()
    test_ridealong_filing()
    test_api_key_never_logged()
    test_config_ingest_flag_default_off()
    test_config_asof_lastarg_no_indexerror()
    print("\nALL DELISTED-INGEST TESTS PASSED")
