"""Automatic trend lines through swing lows and highs.

A swing low is a bar whose low is the lowest of the `window` bars on each
side (a swing high likewise). The support line joins the two most recent
swing lows that rise, with no close more than `tol` below the line between
them; the resistance line joins the two most recent swing highs with no
close more than `tol` above it. Both are then projected to the last bar,
where a close beyond the line marks it broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pivots(values: np.ndarray, window: int, low: bool) -> list[int]:
    out = []
    for i in range(window, len(values) - window):
        seg = values[i - window:i + window + 1]
        v = values[i]
        if np.isnan(v):
            continue
        if (low and v == np.nanmin(seg)) or (not low and v == np.nanmax(seg)):
            if not out or i - out[-1] > window:
                out.append(i)
    return out


def _line(piv: list[int], values: np.ndarray, close: np.ndarray, support: bool,
          tol: float, min_gap: int) -> tuple[int, int] | None:
    """Most recent valid pair of pivots (i1 < i2), or None."""
    for b in reversed(range(len(piv))):
        i2 = piv[b]
        for a in reversed(range(b)):
            i1 = piv[a]
            if i2 - i1 < min_gap:
                continue
            v1, v2 = values[i1], values[i2]
            if support and v2 <= v1:
                continue
            slope = (v2 - v1) / (i2 - i1)
            between = np.arange(i1, i2 + 1)
            line = v1 + slope * (between - i1)
            c = close[between]
            ok = np.all(c >= line * (1 - tol)) if support else np.all(c <= line * (1 + tol))
            if ok:
                return i1, i2
    return None


def trend_lines(df: pd.DataFrame, window: int = 5, tol: float = 0.01,
                min_gap: int = 10) -> list[dict]:
    """Support and resistance for one stock.

    df has a date index (or column `date`) and columns low, high, close in
    date order. Returns up to two dicts: kind, start/anchor/end dates and prices,
    and whether the last close has broken the line.
    """
    df = df.reset_index(drop=False) if "date" not in df.columns else df.reset_index(drop=True)
    dates = df["date"].astype(str).str.slice(0, 10).to_numpy()
    close = df["close"].to_numpy(float)
    out = []
    for kind, col, support in (("support", "low", True), ("resistance", "high", False)):
        values = df[col].to_numpy(float)
        pair = _line(_pivots(values, window, support), values, close, support, tol, min_gap)
        if pair is None:
            continue
        i1, i2 = pair
        slope = (values[i2] - values[i1]) / (i2 - i1)
        last = len(df) - 1
        end = values[i1] + slope * (last - i1)
        broken = close[last] < end * (1 - tol) if support else close[last] > end * (1 + tol)
        out.append(dict(kind=kind, start_date=dates[i1], start_price=round(float(values[i1]), 2),
                        anchor_date=dates[i2], anchor_price=round(float(values[i2]), 2),
                        end_date=dates[last], end_price=round(float(end), 2),
                        slope_pct_per_day=round(float(slope / values[i2] * 100), 3),
                        broken=bool(broken)))
    return out
