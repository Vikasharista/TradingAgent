"""Per-horizon scores, following the strategy and scoring spec.

Each signal is a 0-100 sub-score; a horizon's score is the weighted mean of
its available signals. Signals whose data is missing (delivery %, fundamentals
in the backtest) are dropped and the remaining weights are renormalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import Panel

CRORE = 1e7


@dataclass
class Features:
    """Indicators shared by scoring, regime and trade levels."""
    p: Panel
    ema9: pd.DataFrame
    ema20: pd.DataFrame
    ema50: pd.DataFrame
    sma50: pd.DataFrame
    sma150: pd.DataFrame
    sma200: pd.DataFrame
    rsi14: pd.DataFrame
    atr14: pd.DataFrame
    vol_avg20: pd.DataFrame
    turnover20: pd.DataFrame
    swing_low10: pd.DataFrame
    extra: dict = field(default_factory=dict)


def build_features(p: Panel) -> Features:
    c, h, l, v = p.close, p.high, p.low, p.volume
    return Features(
        p=p,
        ema9=ind.ema(c, 9), ema20=ind.ema(c, 20), ema50=ind.ema(c, 50),
        sma50=ind.sma(c, 50), sma150=ind.sma(c, 150), sma200=ind.sma(c, 200),
        rsi14=ind.rsi(c, 14),
        atr14=ind.atr(h, l, c, 14),
        vol_avg20=ind.sma(v, 20).shift(1),
        turnover20=ind.sma(c * v, 20),
        swing_low10=l.rolling(10, min_periods=10).min(),
    )


def eligible(f: Features, segment: pd.Series) -> pd.DataFrame:
    """Liquidity and price filter from the spec."""
    small = segment.reindex(f.p.close.columns).isin(["small", "micro"])
    min_turnover = pd.Series(np.where(small, 2 * CRORE, 5 * CRORE), index=f.p.close.columns)
    return (f.turnover20.ge(min_turnover, axis=1)) & (f.p.close >= 20)


def _combine(signals: dict[str, tuple[float, pd.DataFrame | None]]) -> pd.DataFrame:
    num, den = None, 0.0
    for w, s in signals.values():
        if s is None or w == 0:
            continue
        num = s * w if num is None else num + s * w
        den += w
    return num / den


def score_day(f: Features, delivery_pct: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    p = f.p
    c, o, h, l, v = p.close, p.open, p.high, p.low, p.volume
    rng = (h - l).replace(0, np.nan)
    close_loc = (c - l) / rng
    day_ret = c / c.shift(1) - 1
    vol_surge = v / f.vol_avg20
    nr7 = rng <= rng.rolling(7, min_periods=7).min()
    inside = (h <= h.shift(1)) & (l >= l.shift(1))
    contraction = nr7.shift(1, fill_value=False) | inside.shift(1, fill_value=False) | nr7
    short_trend = ((c > f.ema9).astype(float) + (c > f.ema20) + (f.ema9 > f.ema20)) / 3 * 100

    deliv = None
    if delivery_pct is not None:
        deliv = ind.flag(delivery_pct > ind.sma(delivery_pct, 20).shift(1))

    signals = {
        # Momentum and price action: 35
        "close_location": (15, ind.ramp(close_loc, 0.4, 0.85)),
        "day_return": (10, ind.band(day_ret, 0.01, 0.05, 0.03)),
        "rsi": (10, ind.band(f.rsi14, 55, 70, 15)),
        # Volume and delivery: 25
        "volume_surge": (15, ind.ramp(vol_surge, 1.0, 2.0)),
        "delivery": (10, deliv),
        # Setup: 20
        "range_contraction": (20, contraction.astype(float) * 70 + 30),
        # Trend: 10
        "short_trend": (10, short_trend),
        # Relative strength: 10
        "rs_1m": (10, ind.pct_rank(c.pct_change(21))),
    }
    return _combine(signals), {k: s for k, (_, s) in signals.items() if s is not None}


def score_week(f: Features) -> tuple[pd.DataFrame, dict]:
    p = f.p
    c, o, h, l, v = p.close, p.open, p.high, p.low, p.volume
    _, _, hist = ind.macd(c)
    macd_s = ind.flag(hist > 0) * 0.5 + ind.flag(hist > hist.shift(1)) * 0.5

    pivot = h.shift(1).rolling(50, min_periods=20).max()
    vol_surge = v / f.vol_avg20
    breakout = ind.flag((c > pivot) & (vol_surge >= 1.5))
    near_pivot = ind.ramp(c / pivot, 0.95, 1.0) * 0.6  # coiling under the pivot
    uptrend = f.ema20 > f.ema50
    pullback = ind.flag(uptrend & ((c / f.ema20 - 1).abs() <= 0.02) & (c > o))
    atr10 = ind.atr(h, l, c, 10)
    atr50 = ind.atr(h, l, c, 50)
    vcp = ind.ramp(atr10 / atr50, 1.0, 0.6)
    setup = np.maximum(np.maximum(breakout, near_pivot), pullback) * 0.8 + vcp * 0.2

    trend = (uptrend.astype(float) + (c > f.ema20) + (c > f.ema50)) / 3 * 100
    updown = ind.up_down_volume(c, v, 20)

    signals = {
        # Momentum: 20
        "macd": (10, macd_s),
        "rsi": (10, ind.band(f.rsi14, 50, 70, 15)),
        # Volume: 20
        "volume_surge": (10, ind.ramp(vol_surge, 1.0, 2.0)),
        "accumulation": (10, ind.ramp(updown, 0.8, 1.5)),
        # Setup: 25
        "setup": (25, setup),
        # Trend: 20
        "trend": (20, trend),
        # Relative strength: 15
        "rs_1m": (15, ind.pct_rank(c.pct_change(21))),
    }
    extra = {k: s for k, (_, s) in signals.items()}
    extra["is_breakout"] = breakout > 0
    return _combine(signals), extra


def trend_template(f: Features) -> pd.DataFrame:
    c = f.p.close
    hi52 = f.p.high.rolling(252, min_periods=200).max()
    lo52 = f.p.low.rolling(252, min_periods=200).min()
    return ((c > f.sma50) & (c > f.sma150) & (c > f.sma200)
            & (f.sma50 > f.sma150) & (f.sma150 > f.sma200)
            & (f.sma200 > f.sma200.shift(21))
            & (c >= 0.75 * hi52) & (c >= 1.30 * lo52))


def rs_rating(c: pd.DataFrame) -> pd.DataFrame:
    """O'Neil-style weighted 3/6/9/12-month return, ranked 1-99."""
    perf = (0.4 * c.pct_change(63) + 0.2 * c.pct_change(126)
            + 0.2 * c.pct_change(189) + 0.2 * c.pct_change(252))
    return ind.pct_rank(perf).clip(1, 99)


def sector_rank(c: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """Rank (1 = strongest) of each stock's industry by 3-month return."""
    ind_map = industry.reindex(c.columns)
    by_ind = c.pct_change(63).T.groupby(ind_map).mean().T
    ranks = by_ind.rank(axis=1, ascending=False)
    out = ranks.reindex(columns=ind_map.values)
    out.columns = c.columns
    return out


def score_month(f: Features, industry: pd.Series,
                fundamentals: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    c, v = f.p.close, f.p.volume
    gate = trend_template(f)
    rs = rs_rating(c)
    sec = sector_rank(c, industry)
    hi50 = f.p.high.rolling(50, min_periods=50).max()
    slope50 = f.sma50 / f.sma50.shift(21) - 1
    stretch = c / f.sma50 - 1
    trend = ind.ramp(slope50, 0.0, 0.05) * 0.5 + ind.band(stretch, 0.0, 0.15, 0.15) * 0.5

    signals = {
        "momentum": (10, ind.band(f.rsi14, 55, 75, 15)),
        "accumulation": (10, ind.ramp(ind.up_down_volume(c, v, 50), 0.8, 1.5)),
        "weekly_setup": (15, ind.ramp(c / hi50, 0.90, 0.98)),
        "trend": (25, trend),
        "rs_rating": (12, ind.ramp(rs, 50, 90)),
        "sector": (8, ind.ramp(sec, 12, 4)),
    }
    score = _combine(signals)
    if fundamentals is not None:
        # Fundamentals carry 20 of 100 points where they are known; elsewhere
        # (no results yet, or a stock outside the fetched list) the technical
        # signals stand alone.
        f20 = fundamentals.reindex_like(score)
        score = (score * 80 + f20 * 20).div(100).where(f20.notna(), score)
    score = score.where(gate)
    extra = {k: s for k, (_, s) in signals.items() if s is not None}
    if fundamentals is not None:
        extra["fundamentals"] = fundamentals
    extra["rs_rating"] = rs
    extra["trend_template"] = gate
    return score, extra
