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
