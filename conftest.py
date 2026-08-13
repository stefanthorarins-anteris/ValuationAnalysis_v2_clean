"""pytest configuration for the repo root.

WHY THIS FILE EXISTS (ship gate, 2026-07-27)
--------------------------------------------
`test_pipeline_prereqs.py` is a STANDALONE pre-flight SCRIPT, not a pytest module.  It is
named `test_*` and its console helpers are named `test_pass` / `test_fail` / `test_warn`,
so pytest collects the file and then errors on all three ("fixture 'msg' not found") --
which is why a bare `pytest` at the repo root was never green even when every real suite
passed.

The collection is worse than cosmetic: the script does its work AT IMPORT TIME, and that
work includes

  * `open('fmpAPIkey.txt')` -- collection FAILS outright on any machine without the key
    file, and reads the secret on machines that have it, and
  * LIVE FMP API calls (profile / key-metrics / ratios / DCF probes)

so merely *collecting* the repo fired real network requests against the paid API and made
the outcome depend on the key being present.  Blocking collection is therefore the fix;
renaming the helpers would silence the errors but leave the import-time key read and the
network calls in place.

The script itself is UNCHANGED and still runs the way it is documented to:
`python test_pipeline_prereqs.py`.
"""
#  `test_cycleheat.py` ADDED 2026-08-13 — SAME DEFECT, MISSED FOR THE SAME REASON.
#  It is a SCRIPT wearing a test_ prefix: zero test functions, `api_key =
#  open('fmpAPIkey.txt').read()` at MODULE level (line 15), and three module-level
#  `for` loops that fire live paid `v3/profile` and `v4/*` calls at IMPORT time
#  (lines 75, 122, 147).  Collection imports it, so the calls ran on EVERY
#  `pytest .` from the repo root.
#
#  The blast radius is the part worth recording: the house has been running the full
#  suite as its standard verification for weeks, and repeatedly told the CEO that a
#  directory-scoped run is offline-safe.  It was not.  Roughly eight paid calls fired
#  per full-suite invocation.  Found 2026-08-13 by a reviewer who noticed the cost,
#  fired the calls itself before realising, and DISCLOSED it rather than burying it.
#
#  This list is the whole defence and it was one entry long while a second file with
#  the identical shape sat beside it.  Anything named `test_*.py` that is really a
#  script belongs here — or, better, gets renamed so it cannot be collected at all.
collect_ignore = ["test_pipeline_prereqs.py", "test_cycleheat.py"]


# --------------------------------------------------------------------------------- #
#  REPO-WIDE FX-STATE ISOLATION (moved here from test_fx_rates.py, reviewer F-6)     #
# --------------------------------------------------------------------------------- #
#  `carveOut` holds the run's FX source in MODULE-GLOBAL state (`_FX_STATE`/`_LIVE_FX`),
#  because it has to: every conversion call site reads it, and threading it through would
#  touch every consumer.  The cost is that a test which installs a table LEAKS into every
#  test that runs after it -- including tests in other FILES, since module state outlives
#  a module's tests.  A leaked 'live' table makes `_fx_to_usd` return None for currencies
#  the constants would have resolved, so the symptom is an unrelated suite failing on a
#  NaN market cap, which is about as hard to trace as this project gets.
#
#  It lived in `test_fx_rates.py` and therefore protected only that file.  It belongs
#  here: the state is global, so the guard must be too, and a future test author touching
#  carveOut will not know to isolate something they never see.  Blast radius today is
#  zero -- no other test installs FX -- which is exactly why it is cheap to fix now.
#
#  This root conftest is loaded for `baseline_tools/` too (it is under the rootdir), so
#  one fixture covers BOTH pytest invocations.  The import is inside the fixture so a
#  collection-time import cannot fail for suites that never touch carveOut.
import pytest


@pytest.fixture(autouse=True)
def _isolate_carveout_fx_state():
    """Reset carveOut's global FX source to 'unset' around EVERY test in the repo."""
    try:
        import carveOut as _co
    except Exception:
        yield
        return
    _co.clear_live_fx_rates()
    yield
    _co.clear_live_fx_rates()
