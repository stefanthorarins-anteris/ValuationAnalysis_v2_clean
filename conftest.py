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
collect_ignore = ["test_pipeline_prereqs.py"]


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
