"""THE 2026-08-14 CAPTURE WAVE: every new field must be CAPTURED and read by NOTHING.

WHAT THIS IS.  A fetch is the only chance the pipeline ever gets to gain a column -- a SAVED
PICKLE CAN NEVER GAIN ONE -- and the 2026-08-14 fetch adds 22 fields that are already inside
payloads we are already paying for (confirmed PRESENT on a live probe 2026-08-13, recorded in
`APIcallsDocs/endpoint_fields.json`).  ZERO extra API calls.

  13 STATEMENT fields : 5 balance-sheet + 4 income + 4 cash-flow
   9 PROFILE fields   : PROFILE_EXTRA_CAPTURE_FIELDS
                                                       -- 22 total, asserted below rather
than stated, because an earlier version of this docstring said "19" and nothing checked it.

THE DISCIPLINE THESE PIN IS THE WHOLE POINT, and it is the same one the 2026-08-05/08/09 waves
were held to: CAPTURE ONLY.  Every field lands on the panel or in the profile map and is read by
NOTHING.  Anything that changes a score is out of scope and must be raised, not done.  So:

  * every new statement field is in `preReq_dict` and in NO scoring dict;
  * no new field creates a `BoMetric_df` column (that schema comes from the metric dicts);
  * `retainedEarnings` is captured and Altman-Z's x2 term is STILL the equity substitute --
    rewiring it is a separate decision with a real behaviour change behind it;
  * `commonStockRepurchased` and `commonStockIssued` are captured as a PAIR (capturing only the
    repurchase leg invites reading a buyback off a company that issued into one).

AND THE TRAP THAT HAS ALREADY COST A RUN.  A profile-field addition does not land unless the
map rebuild actually happens, and the gate could not see a capture-schema change: on 2026-08-10
all four maps were present, fresh and above the coverage floor, so the gate skipped -- correctly,
by its own rules -- and two already-shipped capture changes never landed.  Nothing would have
forced a rebuild until 2026-10-06.  `PROFILE_CAPTURE_FIELDS` + the fingerprint stamp make a
capture change trigger its OWN rebuild, with no flag and nobody remembering.
"""

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import createDicts as cdic
import findAllSectors as fas
import utils as utils

#  The endpoint each statement field must live on, and the wave it belongs to.
NEW_STATEMENT_FIELDS = {
    'bs': ('retainedEarnings', 'netReceivables', 'accountPayables',
           'goodwillAndIntangibleAssets', 'minorityInterest'),
    'inc': ('incomeTaxExpense', 'incomeBeforeTax',
            'researchAndDevelopmentExpenses', 'costOfRevenue'),
    'cf': ('capitalExpenditure', 'commonStockRepurchased', 'commonStockIssued',
           'stockBasedCompensation'),
}
_ENDPOINT = {'bs': 'balance-sheet-statement', 'inc': 'income-statement',
             'cf': 'cash-flow-statement', 'km': 'key-metrics', 'fr': 'ratios'}


def _inventory():
    path = os.path.join(_REPO, 'APIcallsDocs', 'endpoint_fields.json')
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


# =========================================================================== #
#  STATEMENT FIELDS                                                            #
# =========================================================================== #
def test_the_wave_size_this_docstring_claims_is_ASSERTED_not_stated():
    """L-9 (review, 2026-08-14): the docstring said 19; it is 22.  A count in prose that
    nothing checks is a number that goes stale on the next field added -- so it is a test."""
    n_statement = sum(len(v) for v in NEW_STATEMENT_FIELDS.values())
    n_profile = len(fas.PROFILE_EXTRA_CAPTURE_FIELDS)
    assert (n_statement, n_profile, n_statement + n_profile) == (13, 9, 22)


def test_every_new_statement_field_is_in_preReq_dict():
    preReq = cdic.getDicts()[0]
    for stmt, fields in NEW_STATEMENT_FIELDS.items():
        for f in fields:
            assert f in preReq[stmt], (stmt, f)


def test_every_preReq_field_actually_EXISTS_on_its_endpoint_payload():
    """MEASURED PRESENCE, NOT VENDOR DOCS.  A field that is not on the payload becomes a
    silent all-NaN column, which is worse than an absent one because it looks like coverage.
    Checked for the WHOLE dict, not just the new wave -- a stale entry anywhere is the same
    defect."""
    preReq, inv = cdic.getDicts()[0], _inventory()
    missing = [(stmt, f) for stmt, fields in preReq.items() if stmt in _ENDPOINT
               for f in fields if f not in inv[_ENDPOINT[stmt]]]
    assert missing == [], missing


def test_no_new_field_is_ALSO_a_scoring_criterion():
    """CAPTURE ONLY.  A field that appears as a scoring criterion's Upper/Lower would be
    WIRED, which is exactly what this wave does not do."""
    preReq, calc, base, mean, diff, unity, special = cdic.getDicts()
    scored_operands = set()
    for d in (base, mean, diff, unity):
        for spec in d.values():
            scored_operands.update({spec.get('Upper'), spec.get('Lower')})
    new = {f for fields in NEW_STATEMENT_FIELDS.values() for f in fields}
    assert new & scored_operands == set(), new & scored_operands


def test_no_new_field_creates_a_BoMetric_column():
    """`BoMetric_df`'s schema comes from the METRIC dicts, never from preReq_dict -- so a
    capture cannot change the Stage-1 panel schema, and `calcScore`'s column-exact schema gate
    cannot start refusing panels because of it."""
    cols = set(utils.initBoMetric_fromDict()['BoMetric_df'].columns)
    new = {f for fields in NEW_STATEMENT_FIELDS.values() for f in fields}
    assert new & cols == set(), new & cols


def test_the_buyback_pair_is_captured_TOGETHER():
    """Both legs or neither.  The net share count cannot tell a buyback from a company that
    issued into one, so capturing only `commonStockRepurchased` would invite exactly that
    misreading in whoever wires it."""
    cf = cdic.getDicts()[0]['cf']
    assert ('commonStockRepurchased' in cf) == ('commonStockIssued' in cf)


def test_the_tax_anomaly_operands_are_BOTH_present():
    """`createDicts`'s own `effectiveTaxRate` note names these two as the missing operands
    making the both-operands-negative case "NOT detectable here and NOT fixed".  One without
    the other closes nothing."""
    inc = cdic.getDicts()[0]['inc']
    assert 'incomeTaxExpense' in inc and 'incomeBeforeTax' in inc


def test_altman_z_is_NOT_rewired_to_the_published_retained_earnings():
    """CAPTURING IS NOT REWIRING.  Altman's x2 is defined as retainedEarnings/totalAssets and
    the code substitutes totalStockholdersEquity/totalAssets.  Moving it is a real behaviour
    change on a WEIGHTED metric and is explicitly a separate decision -- this test is what
    makes taking it a deliberate act rather than a tidy-up."""
    import inspect
    import stage2_metrics as sm
    src = inspect.getsource(sm.altman_z)
    assert 'retainedEarnings' not in src, (
        'altman_z now reads retainedEarnings. That is a scoring change on a weighted metric '
        'and was NOT the decision taken in the 2026-08-14 capture wave -- it needs a ruling '
        'and a measured before/after, then this test updates with it.')
    assert 'totalStockholdersEquity' in src


def test_preReq_dict_has_no_duplicates_across_statements():
    """A field requested from two endpoints collides on one cdx column; `dictCheckValid`
    already asserts it, so this is the regression pin for the new wave."""
    assert cdic.dictCheckValid() is True


# =========================================================================== #
#  PROFILE FIELDS + THE REBUILD TRIGGER                                        #
# =========================================================================== #
def test_every_declared_profile_field_exists_on_the_profile_payload():
    inv = _inventory()
    missing = [f for f in fas.PROFILE_CAPTURE_FIELDS if f not in inv['profile']]
    assert missing == [], missing


def test_the_new_profile_wave_is_declared():
    for f in ('mktCap', 'ipoDate', 'companyName', 'isAdr', 'isEtf', 'isFund',
              'cik', 'cusip', 'fullTimeEmployees'):
        assert f in fas.PROFILE_EXTRA_CAPTURE_FIELDS, f
        assert f in fas.PROFILE_CAPTURE_FIELDS, f


def test_the_declaration_covers_the_EARLIER_waves_too():
    """The fingerprint has to cover every field the writer pulls, or removing an old one would
    not trigger a rebuild -- the same 'derived from what the writer WRITES, never a subset of
    it frozen at some past revision' lesson the PRESENCE check already learned."""
    for f in ('isin', 'volAvg', 'price', 'currency', 'isActivelyTrading',
              'exchange', 'exchangeShortName', 'country', 'beta', 'sector', 'industry'):
        assert f in fas.PROFILE_CAPTURE_FIELDS, f


def test_an_ABSENT_stamp_counts_as_CHANGED(tmp_path):
    """UNKNOWN IS NOT UNCHANGED.  Every machine is in this state the first time this ships, and
    the safe direction for a gate whose only cost is ~30 batched profile calls is to rebuild."""
    changed, why = fas.capture_schema_changed(str(tmp_path))
    assert changed is True
    assert fas.PROFILE_CAPTURE_SCHEMA_FILE in why


def test_a_matching_stamp_counts_as_UNCHANGED(tmp_path):
    fas.write_capture_schema(str(tmp_path), verbose=False)
    changed, why = fas.capture_schema_changed(str(tmp_path))
    assert changed is False, why


def test_ADDING_a_field_changes_the_fingerprint__which_is_the_whole_mechanism(tmp_path):
    """THE DEFECT THIS CLOSES: on 2026-08-10 two shipped capture changes never landed because
    presence/age/coverage cannot see a code change to WHICH FIELDS are captured."""
    fas.write_capture_schema(str(tmp_path), verbose=False)
    path = os.path.join(str(tmp_path), fas.PROFILE_CAPTURE_SCHEMA_FILE)
    with open(path, 'r', encoding='utf-8') as fh:
        stamp = json.load(fh)
    stamp['fields'] = [f for f in stamp['fields'] if f != 'mktCap']
    stamp['fingerprint'] = fas.profile_capture_fingerprint(stamp['fields'])
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(stamp, fh)
    changed, why = fas.capture_schema_changed(str(tmp_path))
    assert changed is True
    assert 'mktCap' in why, why


def test_the_fingerprint_is_ORDER_INDEPENDENT():
    """Re-ordering the declaration must not trigger a spurious rebuild -- a gate that cries
    wolf gets a `-force_rebuild_maps` habit built around it, which is the failure mode."""
    a = fas.profile_capture_fingerprint(('b', 'a', 'c'))
    b = fas.profile_capture_fingerprint(('c', 'b', 'a'))
    assert a == b


def test_a_CORRUPT_stamp_counts_as_CHANGED(tmp_path):
    path = os.path.join(str(tmp_path), fas.PROFILE_CAPTURE_SCHEMA_FILE)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('{not json')
    changed, _why = fas.capture_schema_changed(str(tmp_path))
    assert changed is True


def test_EVERY_profile_key_the_writer_pulls_is_IN_the_fingerprint():
    """M-2 (review, 2026-08-14) -- THE HOLE THE FINGERPRINT EXISTS TO CLOSE, STILL OPEN.

    `PROFILE_CAPTURE_FIELDS` is a hand-written literal unioned with the loop-captured extras,
    but TWELVE earlier-wave keys are still pulled as individually-named `prof.get('...')` lines
    (beta, country, currency, exchange, exchangeShortName, industry, isActivelyTrading, isin,
    price, sector, symbol, volAvg).  Those named lines are fine in themselves -- each carries
    consumers and commentary, and renaming them would be a refactor riding a capture change.
    What is NOT fine is that ADDING A THIRTEENTH ONE would leave the fingerprint unmoved, the
    gate would skip, and the 2026-08-10 defect (two shipped capture changes that never landed)
    would recur EXACTLY.  A declaration that a writer can silently outgrow is the same
    "an operator remembering" mechanism this module's own header condemns.

    So the writer is checked against the declaration by READING IT, which closes the hole
    without touching the writer: every `prof.get('<literal>')` in `buildSectorIndustryMaps`
    must be a declared field.  Add a named line for an undeclared key and this fails, naming it.
    """
    import inspect
    import re
    src = inspect.getsource(fas.buildSectorIndustryMaps)
    pulled = set(re.findall(r"prof\.get\(\s*'([A-Za-z0-9_]+)'", src))
    assert pulled, 'the regex found no prof.get() literals -- it has stopped testing anything'
    undeclared = sorted(pulled - set(fas.PROFILE_CAPTURE_FIELDS))
    assert undeclared == [], (
        'buildSectorIndustryMaps pulls %r off the profile payload, and %s NOT in '
        'PROFILE_CAPTURE_FIELDS. The capture-schema fingerprint is built from that tuple, so '
        'the rebuild gate CANNOT SEE this field: a machine holding present, fresh, '
        'above-coverage maps would skip the rebuild and the new field would never land -- '
        'which is exactly the 2026-08-10 loss. Add it to PROFILE_CAPTURE_FIELDS (or to '
        'PROFILE_EXTRA_CAPTURE_FIELDS, which feeds it) in the same edit.'
        % (undeclared, 'it is' if len(undeclared) == 1 else 'they are'))


def test_the_writer_can_only_capture_what_is_DECLARED():
    """The extras are read in a loop over `PROFILE_EXTRA_CAPTURE_FIELDS`, so the writer cannot
    pull a field the fingerprint does not cover, nor omit one it does.  That equivalence is
    what makes the stamp an honest description rather than a second thing to maintain."""
    import inspect
    src = inspect.getsource(fas.buildSectorIndustryMaps)
    assert 'for _f in PROFILE_EXTRA_CAPTURE_FIELDS' in src, (
        'the extras must be captured FROM the declaration, not as individually-named '
        'prof.get() lines -- that pattern is what let the capture set drift out of sight of '
        'the rebuild gate in the first place.')
    for f in fas.PROFILE_EXTRA_CAPTURE_FIELDS:
        assert "prof.get('%s')" % f not in src, f


def test_isActivelyTrading_lesson_is_carried_forward_in_the_source():
    """`isActivelyTrading` was captured 2026-08-09 and measured True on 100/100 of a
    deliberately ADVERSE sample including all 39 names that failed the previous fetch -- it
    discriminates nothing here.  The three new booleans are captured with NO discrimination
    measurement (none can exist until a panel carries them), and that status has to survive in
    the file, not just in a report."""
    import inspect
    src = inspect.getsource(fas)
    assert 'isActivelyTrading' in src and '100/100' in src
    assert 'WITHOUT a discrimination' in src or 'without a discrimination' in src
