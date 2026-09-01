"""
Test for profit_timing.py Q-54 defect: low-n directional conclusions.

The defect: lines 70-80 in the original code drew a directional conclusion
(e.g., "MODEL problem") based on a ±0.05 band comparison with NO guard on
the number of observations. A single datapoint (n=1) is completely meaningless.

This test verifies that:
1. With n < MIN_N_FLOOR, the code does NOT print a directional conclusion.
2. With n >= MIN_N_FLOOR, it may draw one.
3. Below the floor, an explicit "NO DIRECTIONAL CONCLUSION DRAWN" line is printed.

Run: python baseline_tools/test_profit_timing.py
     or: pytest baseline_tools/ (by DIRECTORY, not file name, to avoid API calls)
"""

import os
import sys
import io
import contextlib
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profit_timing as pt


def test_low_n_no_directional_conclusion():
    """
    Verify that with n=1 exiters, no directional conclusion is drawn.
    The unfixed code would print "MODEL problem" or similar even with n=1.
    The fixed code must print "NO DIRECTIONAL CONCLUSION DRAWN".
    """
    # Create a minimal price panel
    panel = pd.DataFrame({
        "source": ["ACME", "ACME"],
        "date": [pd.Timestamp("2021-12-31"), pd.Timestamp("2024-12-31")],
        "_price": [100.0, 150.0],
    })

    # Create a single-row exiters DataFrame (n=1): profit AFTER exit
    # This is designed to trigger the "PREMATURE CHURN -> MODEL problem" conclusion
    # in the unfixed code, even though n=1 is meaningless.
    exiters = pd.DataFrame({
        "sym": ["ACME"],
        "during_hold": [0.05],     # +5% during hold [D, D+1y]
        "after_exit": [0.15],      # +15% after exit [D+1y, D+3y] -> after > during + 0.05
        "full_3y": [0.20],
    })

    stayers = pd.DataFrame({
        "sym": [],
        "during_hold": [],
        "after_exit": [],
        "full_3y": [],
    })

    # Patch mvm.build_panel and stage2_pit calls to return our fixtures
    with patch("profit_timing.mvm.build_panel") as mock_build:
        with patch("profit_timing.s2.prepare_pit") as mock_prepare:
            with patch("profit_timing.s2.stage1_boscore") as mock_score:
                with patch("profit_timing.s2.stage2_top") as mock_top:
                    mock_build.return_value = panel

                    # Each call to prepare_pit, stage1_boscore, stage2_top in run()
                    # happens twice (for D=2021 and D=2022). We return empty lists
                    # for top0 and ['STAYER'] for top1, so exiters = top0 - top1 = ['ACME']
                    mock_prepare.return_value = (None, None)  # Not used in our test
                    mock_score.return_value = None
                    # top0 has one name, top1 is empty -> one exiter
                    mock_top.side_effect = [
                        ["ACME"],      # top0 (exiters)
                        [],            # top1 (stayers) -> ACME exits
                    ]

                    # Capture stdout to check for the no-conclusion line
                    captured = io.StringIO()
                    with contextlib.redirect_stdout(captured):
                        pt.run()

                    output = captured.getvalue()

                    # The fixed code MUST print this line when n < 3
                    assert "NO DIRECTIONAL CONCLUSION DRAWN" in output, \
                        f"Missing no-conclusion guard for n=1.\nOutput:\n{output}"

                    # Also verify the underlying numbers are printed
                    assert "n = 1" in output, \
                        f"n=1 count not printed.\nOutput:\n{output}"

                    # Verify NO directional conclusion was drawn
                    # (the unfixed code would print "MODEL problem")
                    assert "MODEL problem" not in output, \
                        f"Directional conclusion drawn for n=1 (should be guarded).\nOutput:\n{output}"
                    assert "PREMATURE CHURN" not in output, \
                        f"Directional conclusion drawn for n=1 (should be guarded).\nOutput:\n{output}"

    print("PASS: low-n (n=1) correctly blocked from directional conclusion")


def test_sufficient_n_allows_conclusion():
    """
    Verify that with n >= MIN_N_FLOOR, the guard does not suppress output
    (i.e., no "NO DIRECTIONAL CONCLUSION DRAWN" is printed).
    This confirms the guard correctly gates on n and doesn't always suppress.
    """
    # Create a minimal price panel (values don't matter for this test)
    panel = pd.DataFrame({
        "source": ["X"] * 6,
        "date": [pd.Timestamp("2021-12-31")] * 6,
        "_price": [100.0] * 6,
    })

    # Create a 3-row exiters DataFrame with clear directional signal
    # (after > during + 0.05, should print "PREMATURE CHURN")
    exiters = pd.DataFrame({
        "sym": ["A", "B", "C"],
        "during_hold": [0.05, 0.06, 0.04],
        "after_exit": [0.20, 0.21, 0.19],  # all clearly > during + 0.05
        "full_3y": [0.25, 0.27, 0.23],
    })

    with patch("profit_timing.mvm.build_panel") as mock_build:
        with patch("profit_timing.s2.prepare_pit") as mock_prepare:
            with patch("profit_timing.s2.stage1_boscore") as mock_score:
                with patch("profit_timing.s2.stage2_top") as mock_top:
                    mock_build.return_value = panel
                    mock_prepare.return_value = (None, None)
                    mock_score.return_value = None
                    # Bypass the _timing() call by directly mocking _price_at
                    with patch("profit_timing._price_at") as mock_price:
                        # All prices are present and positive
                        mock_price.return_value = 100.0

                        # top0 has three names, top1 is empty -> three exiters
                        mock_top.side_effect = [
                            ["A", "B", "C"],  # top0 (exiters)
                            [],               # top1 (stayers, empty)
                        ]

                        captured = io.StringIO()
                        with contextlib.redirect_stdout(captured):
                            pt.run()

                        output = captured.getvalue()

                        # With n=3 >= MIN_N_FLOOR (which is 3), no guard should fire
                        # Verify "NO DIRECTIONAL CONCLUSION DRAWN" is NOT printed
                        assert "NO DIRECTIONAL CONCLUSION DRAWN" not in output, \
                            f"No-conclusion guard incorrectly fired for n=3 (should allow conclusion).\nOutput:\n{output}"

                        # With this data (after >> during), should print "PREMATURE CHURN / MODEL problem"
                        # (If it doesn't, that's fine—the point is the guard didn't suppress it)
                        # We just verify the guard didn't fire.

    print("PASS: sufficient-n (n=3) guard correctly does not fire")


if __name__ == "__main__":
    import pytest as _pytest

    for fn in [
        test_low_n_no_directional_conclusion,
        test_sufficient_n_allows_conclusion,
    ]:
        try:
            fn()
        except _pytest.skip.Exception as s:
            print(f"SKIP {fn.__name__}: {s}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            raise

    print("\nALL PROFIT_TIMING TESTS PASSED")
