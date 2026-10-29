"""Smoke test on synthetic prices: checks the pipeline runs end to end and
that the trade arithmetic is consistent. It says nothing about performance."""

import numpy as np
import pandas as pd

from agent import backtest as bt
from agent.data import Panel


def synthetic_panel(n_days=700, n_syms=40, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n_days)
    syms = [f"S{i:02d}" for i in range(n_syms)]
    drift = rng.normal(0.0004, 0.0004, n_syms)
    rets = rng.normal(drift, 0.02, (n_days, n_syms))
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, close.shape)))
    vol = rng.lognormal(13, 0.5, close.shape)
    mk = lambda a: pd.DataFrame(a, index=dates, columns=syms)  # noqa: E731
    return Panel(mk(open_), mk(high), mk(low), mk(close), mk(vol))


def test_backtest_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "OUT", tmp_path)
    panel = synthetic_panel()
    uni = pd.DataFrame({
        "symbol": panel.symbols,
        "industry": [f"Ind{i % 6}" for i in range(len(panel.symbols))],
        "segment": ["large", "mid", "small", "micro"] * (len(panel.symbols) // 4),
    })
    index_close = panel.close.mean(axis=1)
    summary = bt.run(panel, uni, index_close, None, "2018-01-01")

    assert set(summary["results"]) >= {"next_day/always_top10", "next_week/always_top10"}
    day = pd.read_csv(tmp_path / "trades_next_day.csv.gz")
    t = day[day["triggered"]]
    if len(t):
        assert (t["stop"] < t["entry"]).all()
        assert (t["target"] > t["entry"]).all()
        assert t["outcome"].isin(["stop", "target", "expired"]).all()


def test_adjustment_factors_back_adjust_a_split():
    from agent.nse import adjustment_factors

    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 102.0, 51.5, 52.0]}, index=dates)
    # 1:2 split on day 3: NSE restates the previous close from 102 to 51.
    prev = pd.DataFrame({"A": [99.0, 100.0, 51.0, 51.5]}, index=dates)
    f = adjustment_factors(close, prev)
    assert np.allclose((close * f)["A"], [50.0, 51.0, 51.5, 52.0])


def test_adjustment_factors_detect_unrestated_bonus():
    from agent.nse import adjustment_factors

    dates = pd.bdate_range("2024-10-23", periods=4)
    # Bonus 1:1 where NSE left the previous close as it was.
    close = pd.DataFrame({"A": [2650.0, 2655.7, 1334.35, 1340.0]}, index=dates)
    prev = pd.DataFrame({"A": [2640.0, 2650.0, 2655.7, 1334.35]}, index=dates)
    adj = (close * adjustment_factors(close, prev))["A"]
    assert np.allclose(adj.iloc[:2], [1325.0, 1327.85])
    assert np.allclose(adj.iloc[2:], [1334.35, 1340.0])
