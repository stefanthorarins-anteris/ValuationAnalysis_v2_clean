"""CycleHeat's window must be a fixed CALENDAR span, not "whatever rows the fetch returned".

THE DEFECT THESE PIN (fixed 2026-07-30).  `cycleheat` was the only metric in the Stage-2 block
taking neither `nq` nor `rpy`, so its self-reference baseline was however much history existed:
  * a semi-annual filer's window spanned 11.04y (median 11.50, max 25.25) against a quarterly
    peer's 5.53y off the same ~22.5 rows;
  * a LONGER window RAISES h for an improving company (lower baseline to exceed) -- +0.246 sigma
    on the 1,065 deep-history names -- and w(CycleHeat) = -0.080, so it is a PENALTY on exactly
    the "recently became a value company" population the fix exists to protect;
  * at `-nrperiods 80` those spans would have become ~20y and ~40y, so the deep-history fetch
    would have INTENSIFIED the penalty it was run to help detect.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reporting_period as rp
import stage2_metrics as sm


def _frame(n, months, eps_path=None, shares=100.0, start="2026-01-01"):
    """n rows spaced `months` apart, ASCENDING (cdx storage order), newest last."""
    dates = pd.date_range(end=pd.Timestamp(start), periods=n, freq="-%dMS" % months)[::-1]
    ni = (np.asarray(eps_path, dtype=float) * shares if eps_path is not None
          else np.linspace(10.0, 10.0, n) * shares)
    return pd.DataFrame({"date": dates, "netIncome": ni,
                         "weightedAverageShsOut": np.full(n, shares)})


# --------------------------------------------------------------------------- #
#  The window                                                                 #
# --------------------------------------------------------------------------- #
def test_base_window_is_long_enough_to_hold_a_cycle_and_not_bind_on_todays_panel():
    """Two constraints fix the constant: >= a business cycle, and >= the ~24 rows a quarterly
    filer carries today so the change is a no-op for them."""
    assert 24 <= sm.CYCLEHEAT_BASE_NQ <= 32, sm.CYCLEHEAT_BASE_NQ
    assert rp.scale_window(sm.CYCLEHEAT_BASE_NQ, 4) >= 24
    assert rp.scale_window(sm.CYCLEHEAT_BASE_NQ, 2) == pytest.approx(
        sm.CYCLEHEAT_BASE_NQ / 2, abs=1)


def test_window_spans_the_same_calendar_time_for_both_frequencies():
    """The core requirement.  A quarterly filer with 40 quarterly rows and a semi-annual filer
    with 40 semi-annual rows must both be measured over ~the same elapsed years."""
    W = sm.CYCLEHEAT_BASE_NQ
    q_rows = rp.scale_window(W, 4)
    s_rows = rp.scale_window(W, 2)
    q_years = q_rows * 3 / 12.0
    s_years = s_rows * 6 / 12.0
    assert abs(q_years - s_years) <= 0.75, (q_years, s_years)
    assert 6.0 <= q_years <= 8.0 and 6.0 <= s_years <= 8.0


def test_quarterly_is_UNCHANGED_when_the_cap_cannot_bind():
    """THE regression that matters: on a 24-row quarterly panel the 28-row cap does not bind, so
    h must be bit-identical to the uncapped computation."""
    rng = np.random.default_rng(3)
    for seed in range(6):
        f = _frame(24, 3, eps_path=rng.normal(10, 3, 24))
        e = sm.prepare_eps_series(f)
        uncapped = sm.cycleheat_zscore(e, e.iloc[-1])
        capped = sm.cycleheat(f, rpy=4)
        assert capped == uncapped, (seed, capped, uncapped)


def test_semiannual_window_is_TRUNCATED_and_h_moves_toward_zero():
    """A semi-annual filer with 24 rows spans 12 years; the cap takes it to 14 rows / 7 years.
    Shortening the baseline moves h toward zero -- which for an improving company (h > 0, a
    negative-weight penalty) is a RELAXATION, the direction the fix is for."""
    n = 24
    rising = np.linspace(2.0, 20.0, n)          # steadily improving EPS
    f = _frame(n, 6, eps_path=rising)
    e = sm.prepare_eps_series(f)
    h_uncapped = sm.cycleheat_zscore(e, e.iloc[-1])
    h_capped = sm.cycleheat(f, rpy=2)
    assert h_uncapped > 0 and h_capped > 0
    assert h_capped < h_uncapped, (h_capped, h_uncapped)
    # and the rows actually used are the frequency-scaled window
    assert rp.scale_window(sm.CYCLEHEAT_BASE_NQ, 2) == 14


def test_deep_history_does_NOT_lengthen_the_window():
    """The fetch-depth guarantee: 80 rows must give the same window as 28+, not a 20-year one."""
    rng = np.random.default_rng(7)
    long_path = rng.normal(10, 3, 80)
    f80 = _frame(80, 3, eps_path=long_path)
    f28 = _frame(28, 3, eps_path=long_path[-28:])
    h80, h28 = sm.cycleheat(f80, rpy=4), sm.cycleheat(f28, rpy=4)
    assert h80 == pytest.approx(h28, abs=1e-12), (h80, h28)
    # and it is genuinely different from the uncapped 80-row answer (the cap is doing work)
    e = sm.prepare_eps_series(f80)
    assert sm.cycleheat(f80, rpy=4) != sm.cycleheat_zscore(e, e.iloc[-1])


def test_rising_eps_is_penalised_LESS_after_the_cap_on_a_deep_panel():
    """Directly the CEO's concern: a company that has recently become a value company must not be
    penalised harder simply because more history was fetched."""
    n = 80
    rising = np.concatenate([np.full(n - 12, 1.0), np.linspace(1.0, 8.0, 12)])
    f = _frame(n, 3, eps_path=rising)
    e = sm.prepare_eps_series(f)
    h_uncapped = sm.cycleheat_zscore(e, e.iloc[-1])     # long baseline -> hot
    h_capped = sm.cycleheat(f, rpy=4)
    assert h_uncapped > 0
    assert h_capped < h_uncapped, (h_capped, h_uncapped)


def test_short_history_is_untouched_and_under_two_observations_is_NaN():
    assert np.isnan(sm.cycleheat(_frame(1, 3), rpy=4))
    f = _frame(5, 3, eps_path=[1.0, 2.0, 3.0, 4.0, 9.0])
    e = sm.prepare_eps_series(f)
    assert sm.cycleheat(f, rpy=4) == sm.cycleheat_zscore(e, e.iloc[-1])


def test_current_row_is_still_the_most_recent_after_truncation():
    """Truncation must keep the NEWEST rows -- taking the oldest would invert the metric."""
    n = 40
    path = np.arange(1.0, n + 1.0)              # strictly increasing, newest = largest
    f = _frame(n, 3, eps_path=path)
    assert sm.cycleheat(f, rpy=4) > 0           # newest is the max -> hot, not cold


# --------------------------------------------------------------------------- #
#  Every caller passes rpy                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path,needle", [
    ("postBoRank.py", "sm.cycleheat(tempcdx, rpy=rpy)"),
    (os.path.join("baseline_tools", "stage2_pit.py"), "sm.cycleheat(tempcdx, rpy=_rpy)"),
    ("reviewReference.py", "sm.cycleheat(t, rpy=_rpy)"),
])
def test_every_caller_passes_rpy(path, needle):
    """Three call sites; all three were missing it.  A caller that omits `rpy` silently gets the
    quarterly window for a semi-annual filer, re-introducing the 2x-span divergence."""
    src = open(os.path.join(_REPO, path), encoding="utf-8").read()
    assert needle in src, "%s does not pass rpy to cycleheat" % path


def test_no_cdx_row_consumer_in_the_stage2_block_is_missing_rpy():
    """The sweep, as a standing check.  This is the third time a window has been anchored to the
    wrong frequency, so the whole block is asserted rather than the one metric that broke."""
    import inspect
    import re
    import postBoRank as pbr
    # The metric calls live in the per-ticker helper, not in postBoScoreRanking itself.
    src = inspect.getsource(pbr._compute_ticker_metrics)
    missing = []
    # one logical call per `sm.<fn>(...)`, balanced to its closing paren so a multi-line call
    # is read whole (free_cash_flow_per_share_growth spans two lines)
    for m in re.finditer(r"sm\.(\w+)\(", src):
        fn = m.group(1)
        f = getattr(sm, fn, None)
        if f is None or "rpy" not in inspect.signature(f).parameters:
            continue                       # takes no rpy -> nothing to pass
        k, depth = m.end() - 1, 0
        while k < len(src):
            if src[k] == "(":
                depth += 1
            elif src[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        call = src[m.start():k + 1]
        if "rpy=" not in call:
            missing.append((fn, call.replace("\n", " ")))
    assert not missing, ("Stage-2 metrics accepting `rpy` but not receiving it: %s" % missing)
