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


def test_trend_lines_find_rising_support():
    import numpy as np
    import pandas as pd
    from agent.trendlines import trend_lines
    n = 120
    i = np.arange(n)
    base = 100 + 0.2 * i + 5 * np.sin(i / 6)        # rising, wavy
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n),
                       "low": base - 1, "high": base + 1, "close": base})
    lines = {l["kind"]: l for l in trend_lines(df)}
    s = lines["support"]
    assert s["anchor_price"] > s["start_price"] and s["anchor_date"] > s["start_date"]
    assert not s["broken"]
    assert s["end_date"] == str(df["date"].iloc[-1].date())


def test_pick_by_segment_fills_a_short_size():
    from agent.selection import pick_by_segment

    syms = [f"L{i}" for i in range(12)] + [f"M{i}" for i in range(4)] + ["X0"]
    seg = pd.Series(["large"] * 12 + ["mid"] * 4 + ["micro"], index=syms)
    loose = pd.Series(np.arange(len(syms), dtype=float), index=syms)
    gated = loose.where(~loose.index.isin(["M2", "M3"]))   # two mid caps miss the filter
    picks = pick_by_segment(gated, loose, seg, n=3)
    large = [p for p in picks if p["segment"] == "large"]
    mid = [p for p in picks if p["segment"] == "mid"]
    assert [p["symbol"] for p in large] == ["L11", "L10", "L9"] and not any(p["filled"] for p in large)
    assert [p["symbol"] for p in mid] == ["M1", "M0", "M3"]
    assert [p["filled"] for p in mid] == [False, False, True]
    assert not any(p["symbol"] == "X0" for p in picks)        # microcaps never picked
    assert not [p for p in pick_by_segment(gated, None, seg, n=3) if p["filled"]]


def test_long_term_annual_signals_and_lag():
    from agent import longterm as lt

    years = pd.DatetimeIndex([f"{y}-03-31" for y in range(2016, 2024)])
    s = lambda v: pd.Series(v, index=years, dtype=float)  # noqa: E731
    d = dict(y_sales=s([100 * 1.15 ** i for i in range(8)]),
             y_profit=s([10 * 1.2 ** i for i in range(8)]),
             y_cfo=s([9 * 1.2 ** i for i in range(8)]),
             y_roce=s([0.22] * 8), equity=s([50 * 1.1 ** i for i in range(8)]),
             borrowings=s([5] * 8))
    t = lt.annual_signals(d, financial=False)
    last = t.iloc[-1]
    assert abs(last["sales_cagr3"] - 0.15) < 1e-9 and abs(last["profit_cagr3"] - 0.2) < 1e-9
    assert last["profit_up5"] == 5 and last["growth"] > 80 and last["quality"] > 80
    # a March year end only counts 60 days later
    assert t.index[-1] == pd.Timestamp("2023-05-30")
