"""Long-term (1 to 3 year) score: business quality and growth first, the chart
for timing.

60 of the 100 points come from the annual accounts on screener.in, 40 from
the price trend:

  Quality (25)   3-year average ROCE (ROE for banks and NBFCs) and debt/equity
  Growth (25)    3-year sales and profit CAGR, and how many of the last five
                 years profit grew
  Cash (10)      operating cash flow / net profit over 3 years (not banks)
  Trend (15)     above a rising 200 DMA
  Momentum (15)  12-month return excluding the latest month, ranked
  Near high (10) price as a share of its 52-week high

Annual figures count only from 60 days after the year end (when they must be
public), so the backtest never sees a number before it was published.
Screener shows today's restated figures rather than the ones first reported,
which is a small source of look-ahead the backtest cannot remove.

Valuation is not in the score: historical P/E needs split-consistent EPS and
share counts that the free pages do not give. P/E and PEG are shown beside each
pick instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import fundamentals as fu
from . import indicators as ind
from .scoring import Features

WEIGHTS = {"quality": 25, "growth": 25, "cash": 10, "trend": 15, "momentum": 15, "near_high": 10}


def _annual(s: pd.Series, months: set[int]) -> pd.Series:
    s = s.dropna()
    return s[s.index.month.isin(months)]


def _cagr(s: pd.Series, years: int = 3) -> pd.Series:
    prev = s.shift(years)
    ok = (prev > 0) & (s > 0)
    return ((s / prev) ** (1 / years) - 1).where(ok)


def annual_signals(d: dict[str, pd.Series], financial: bool) -> pd.DataFrame | None:
    """One row per annual report (indexed by the date it became public)."""
    sales, profit = d["y_sales"].dropna(), d["y_profit"].dropna()
    if len(profit) < 4:
        return None
    months = set(profit.index.month)
    idx = profit.index
    equity = _annual(d["equity"], months).reindex(idx)
    debt = _annual(d["borrowings"], months).reindex(idx)
    sales = sales.reindex(idx)
    roe = profit / ((equity + equity.shift(1)) / 2)
    roce = d["y_roce"].reindex(idx)
    cfo = d["y_cfo"].reindex(idx)
    p3 = profit.rolling(3).sum()
    t = pd.DataFrame({
        "roe3": roe.rolling(3).mean(),
        "roce3": roce.rolling(3, min_periods=2).mean(),
        "de": debt / equity,
        "sales_cagr3": _cagr(sales),
        "profit_cagr3": _cagr(profit),
        "profit_up5": (profit.diff() > 0).astype(float).rolling(5, min_periods=4).sum(),
        "cfo_ratio": (cfo.rolling(3).sum() / p3).where(p3 > 0),
        "profit_positive": profit > 0,
    })
    up_share = t["profit_up5"] / 5 * 100
    growth = (ind.ramp(t["sales_cagr3"], 0.05, 0.18) * 0.4
              + ind.ramp(t["profit_cagr3"], 0.05, 0.22) * 0.4 + up_share * 0.2)
    if financial:
        quality = ind.ramp(t["roe3"], 0.10, 0.20)
        cash = pd.Series(np.nan, index=idx)
    else:
        ret = t["roce3"].fillna(t["roe3"])
        quality = (ind.ramp(ret, 0.12, 0.25) * 0.6 + ind.ramp(t["de"], 1.0, 0.2).fillna(100) * 0.4)
        cash = ind.ramp(t["cfo_ratio"], 0.5, 1.0)
    t["quality"], t["growth"], t["cash"] = quality, growth, cash
    t.index = fu._available(t.index)
    return t[t[["quality", "growth"]].notna().all(axis=1)]


def fundamentals_panels(symbols: list[str], industry: pd.Series, dates: pd.DatetimeIndex
                        ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """quality/growth/cash/profit_positive as dates x symbols panels, plus each
    stock's annual rows for display."""
    cols: dict[str, dict] = {k: {} for k in ("quality", "growth", "cash", "profit_positive")}
    latest = {}
    for sym in symbols:
        d = fu.parse(sym)
        if d is None:
            continue
        t = annual_signals(d, industry.get(sym) in fu.FINANCIALS)
        if t is None or t.empty:
            continue
        t = t[~t.index.duplicated(keep="last")]
        full = t.reindex(t.index.union(dates)).ffill().reindex(dates)
        for k in cols:
            cols[k][sym] = full[k].astype(float)
        latest[sym] = t
    panels = {k: pd.DataFrame(v, index=dates).reindex(columns=symbols) for k, v in cols.items()}
    return panels, latest


def technical(f: Features) -> dict[str, pd.DataFrame]:
    c = f.p.close
    rising = f.sma200 > f.sma200.shift(21)
    trend = ind.flag(c > f.sma200) * 0.5 + ind.flag(rising) * 0.5
    mom = ind.pct_rank(c.shift(21) / c.shift(252) - 1)
    hi52 = f.p.high.rolling(252, min_periods=200).max()
    near = ind.ramp(c / hi52, 0.6, 0.9)
    valid = f.sma200.notna() & hi52.notna()
    return {k: v.where(valid) for k, v in
            {"trend": trend, "momentum": mom, "near_high": near}.items()}


def score(f: Features, fund: dict[str, pd.DataFrame],
          weights: dict[str, float] = WEIGHTS) -> tuple[pd.DataFrame, dict]:
    """0-100 long-term score (NaN without annual data or with a loss last year)."""
    parts = {**technical(f), **{k: fund[k].reindex_like(f.p.close) for k in ("quality", "growth", "cash")}}
    num = den = 0
    for k, w in weights.items():
        if w == 0:
            continue
        s = parts[k]
        num = num + s.fillna(0) * w
        den = den + s.notna() * w
    total = num / den.where(den > 0)
    need = parts["quality"].notna() & parts["growth"].notna() & parts["trend"].notna()
    profitable = fund["profit_positive"].reindex_like(f.p.close).fillna(0).astype(bool)
    return total.where(need & profitable), parts
