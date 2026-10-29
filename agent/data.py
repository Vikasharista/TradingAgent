"""End-of-day price data stored as wide panels (rows = dates, columns = NSE
symbols). The source is NSE's own daily archive files (see agent/nse.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import nse

FIELDS = ["Open", "High", "Low", "Close", "Volume"]

# Index names as they appear in NSE's ind_close_all files.
NIFTY500 = "Nifty 500"
INDIA_VIX = "India VIX"


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for f in FIELDS:
            getattr(self, f.lower()).to_parquet(path / f"{f.lower()}.parquet")

    @classmethod
    def load(cls, path: Path) -> "Panel":
        return cls(**{f.lower(): pd.read_parquet(path / f"{f.lower()}.parquet") for f in FIELDS})


def load_prices(symbols: list[str] | None, start: str, refresh: bool = True,
                min_days: int = 60) -> tuple[Panel, pd.DataFrame, pd.DataFrame]:
    """Split- and bonus-adjusted daily OHLCV plus delivery % for NSE stocks.

    symbols=None keeps every listed equity (ISIN starting INE), which is
    used for the survivorship-free check. Returns (panel, delivery_pct, t2t),
    where t2t is True on days a stock traded in the trade-for-trade (BE)
    segment, which the spec excludes.
    """
    if refresh:
        nse.sync(start)
    long = nse.load_equities(start)
    if symbols is None:
        long = long[long["isin"].str.startswith("INE")]
    else:
        long = long[long["symbol"].isin(set(symbols))]

    wide = {c: long.pivot(index="date", columns="symbol", values=c)
            for c in ["open", "high", "low", "close", "prev_close", "volume", "deliv_pct"]}
    factor = nse.adjustment_factors(wide["close"], wide["prev_close"])
    panel = Panel(
        open=wide["open"] * factor, high=wide["high"] * factor,
        low=wide["low"] * factor, close=wide["close"] * factor,
        volume=wide["volume"] / factor,
    )
    keep = panel.close.notna().sum() > min_days
    panel = Panel(**{f.lower(): getattr(panel, f.lower()).loc[:, keep] for f in FIELDS})
    t2t = long.pivot(index="date", columns="symbol", values="series").eq("BE")
    return panel, wide["deliv_pct"].loc[:, keep], t2t.loc[:, keep]


def load_index(name: str, start: str) -> pd.DataFrame:
    return nse.load_index(name, start)
