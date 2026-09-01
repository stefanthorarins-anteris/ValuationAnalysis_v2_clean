"""Tests for diagnostic row filtering and EVIDENCE_DIR path handling.

Tests nested structure exclusion in diagnostic rows (from scoring_compare),
and that EVIDENCE_DIR is anchored to module location, not CWD.
"""
import os
import sys
import tempfile
import shutil
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the actual function being tested
import scoring_compare as sc
import transfer_utils as tu


# ============================================================================
# Test: Diagnostic rows filter excludes nested structures
# ============================================================================
def test_build_diagnostic_rows_excludes_nested_structures():
    """Verify that _build_diagnostic_rows filters out nested structures.

    Tests the ACTUAL FUNCTION from scoring_compare._build_diagnostic_rows,
    not a replica. Ensures that nested dicts/lists (stage1_veto, top20_deduped,
    ranking) are excluded from the diagnostic CSV while scalar values are kept.
    """
    # Build a per_anchor dict mimicking what rank_all_anchors returns
    per_anchor = {
        'buy2021': {
            'buy': '2021-12-31',
            'ranking': ['AAPL', 'MSFT', 'GOOGL'],  # List - should be excluded
            'rank_depth': 3,
            'top20_deduped': ['AAPL', 'MSFT'],  # List - should be excluded
            'universe_size': 1000,
            'n_pit_scored': 950,
            'n_pit_live': 900,
            'n_pit_dead': 50,
            'basis': 'vetoed',
            'stage1_veto': {  # Dict - should be excluded
                'applies': True,
                'n_ejected': 5,
                'n_in': 950,
                'n_out': 945,
                'by_flag': {'solvency': 5}
            }
        },
        'buy2020': {
            'buy': '2020-12-31',
            'ranking': ['XYZ'],
            'rank_depth': 1,
            'top20_deduped': [],
            'universe_size': 800,
            'n_pit_scored': 750,
            'n_pit_live': 700,
            'n_pit_dead': 50,
            'basis': 'un-vetoed',
            'stage1_veto': None
        }
    }

    config = 'BASELINE'

    # Call the ACTUAL function from scoring_compare
    diag_rows = sc._build_diagnostic_rows(per_anchor, config)

    # Convert to DataFrame like the real code does
    df = pd.DataFrame(diag_rows)

    # Assertions: excluded keys should NOT be in the DataFrame columns
    assert 'ranking' not in df.columns, \
        "Column 'ranking' (list) should have been excluded"
    assert 'top20_deduped' not in df.columns, \
        "Column 'top20_deduped' (list) should have been excluded"
    assert 'stage1_veto' not in df.columns, \
        "Column 'stage1_veto' (dict) should have been excluded"

    # Assertions: scalar columns SHOULD be present
    expected_columns = {'config', 'anchor', 'buy', 'rank_depth', 'universe_size',
                        'n_pit_scored', 'n_pit_live', 'n_pit_dead', 'basis'}
    assert expected_columns.issubset(set(df.columns)), \
        f"Missing expected columns. Expected {expected_columns}, got {set(df.columns)}"

    # Verify row data is correct
    assert len(df) == 2
    row_2021 = df[df['anchor'] == 'buy2021'].iloc[0]
    row_2020 = df[df['anchor'] == 'buy2020'].iloc[0]

    # Check buy2021 row
    assert row_2021['config'] == config
    assert row_2021['basis'] == 'vetoed'
    assert row_2021['n_pit_scored'] == 950
    assert row_2021['rank_depth'] == 3

    # Check buy2020 row
    assert row_2020['config'] == config
    assert row_2020['basis'] == 'un-vetoed'
    assert row_2020['n_pit_scored'] == 750
    assert row_2020['rank_depth'] == 1


# ============================================================================
# Tests: EVIDENCE_DIR is absolute and CWD-independent
# ============================================================================
def test_evidence_dir_is_absolute_path():
    """Verify that transfer_utils.EVIDENCE_DIR is an absolute path."""
    assert os.path.isabs(tu.EVIDENCE_DIR), \
        f"EVIDENCE_DIR must be absolute, got: {tu.EVIDENCE_DIR}"


def test_evidence_dir_cwd_independent():
    """Verify that EVIDENCE_DIR doesn't change when CWD changes.

    With the fix, EVIDENCE_DIR should be anchored to the module's location
    and not drift when the current working directory changes.
    """
    # Store the original value
    original_evidence_dir = tu.EVIDENCE_DIR
    original_cwd = os.getcwd()

    try:
        # Create a temp directory and change to it
        tmpdir = tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)

            # Verify CWD actually changed
            assert os.getcwd() == tmpdir, "Failed to change CWD to temp directory"

            # The constant is already loaded, so check it hasn't changed
            current_evidence_dir = tu.EVIDENCE_DIR

            assert current_evidence_dir == original_evidence_dir, \
                f"EVIDENCE_DIR changed when CWD changed! " \
                f"Was: {original_evidence_dir}, Now: {current_evidence_dir}"

            # Verify it's still absolute
            assert os.path.isabs(current_evidence_dir), \
                f"EVIDENCE_DIR became relative: {current_evidence_dir}"

        finally:
            # Restore original CWD BEFORE cleanup (Windows can't delete current dir)
            os.chdir(original_cwd)
            # Now clean up the temp directory
            shutil.rmtree(tmpdir, ignore_errors=True)

    finally:
        # Extra safety: ensure we're back in original CWD
        if os.path.exists(original_cwd):
            os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
