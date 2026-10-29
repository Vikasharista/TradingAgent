"""Market regime: decides how many picks are shown and the minimum score."""

from __future__ import annotations

import pandas as pd

# regime -> (max picks per horizon, minimum score)
RULES = {"risk_on": (10, 70), "neutral": (5, 80), "risk_off": (3, 90)}


def market_regime(index_close: pd.Series, universe_close: pd.DataFrame,
                  vix: pd.Series | None = None,
                  fii_net_5d: pd.Series | None = None) -> pd.DataFrame:
    """Daily regime with the three conditions from the spec.

    FII flows have no free history, so the backtest runs without them.
    """
    idx = universe_close.index
    above50 = (index_close > index_close.rolling(50).mean()).reindex(idx, method="ffill")
    sma50 = universe_close.rolling(50, min_periods=50).mean()
    valid = sma50.notna().sum(axis=1)
    breadth = (universe_close > sma50).sum(axis=1) / valid.where(valid > 0)
    conds = [above50.fillna(False), breadth > 0.55]
    if vix is not None:
        conds.append((vix.reindex(idx, method="ffill") < 18).fillna(False))
    count = sum(c.astype(int) for c in conds)
    n = len(conds)

    regime = pd.Series("risk_off", index=idx)
    regime[count == n] = "risk_on"
    regime[count == n - 1] = "neutral"
    if fii_net_5d is not None:
        heavy_selling = fii_net_5d.reindex(idx, method="ffill") < -15000  # crore
        regime[heavy_selling.fillna(False)] = "risk_off"

    out = pd.DataFrame({"regime": regime, "breadth": breadth,
                        "index_above_50dma": above50})
    out["max_picks"] = out["regime"].map(lambda r: RULES[r][0])
    out["min_score"] = out["regime"].map(lambda r: RULES[r][1])
    return out
