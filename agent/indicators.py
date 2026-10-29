"""Technical indicators on wide panels (dates x symbols)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(loss > 0, 100.0).where(gain.notna())


def true_range(high, low, close) -> pd.DataFrame:
    prev = close.shift(1)
    return np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))


def atr(high, low, close, n: int = 14) -> pd.DataFrame:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(close: pd.DataFrame, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def up_down_volume(close, volume, n: int) -> pd.DataFrame:
    """Volume on up days divided by volume on down days over n sessions."""
    chg = close.diff()
    up = volume.where(chg > 0, 0).rolling(n, min_periods=n).sum()
    down = volume.where(chg < 0, 0).rolling(n, min_periods=n).sum()
    return up / down.replace(0, np.nan)


# ---- helpers that turn raw values into 0-100 sub-scores ----

def ramp(x: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """0 at lo, 100 at hi, linear in between (works for lo > hi too)."""
    return ((x - lo) / (hi - lo)).clip(0, 1) * 100


def band(x: pd.DataFrame, lo: float, hi: float, soft: float) -> pd.DataFrame:
    """100 inside [lo, hi], falling to 0 at `soft` outside either edge."""
    below = ((x - (lo - soft)) / soft).clip(0, 1)
    above = (((hi + soft) - x) / soft).clip(0, 1)
    return np.minimum(below, above) * 100


def pct_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile (0-100) on each date."""
    return df.rank(axis=1, pct=True) * 100


def flag(cond: pd.DataFrame) -> pd.DataFrame:
    return cond.astype(float) * 100
