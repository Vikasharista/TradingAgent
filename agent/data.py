"""End-of-day price data from Yahoo Finance, stored as wide panels
(rows = dates, columns = NSE symbols)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/cache")
FIELDS = ["Open", "High", "Low", "Close", "Volume"]

# Index tickers on Yahoo Finance.
NIFTY500 = "^CRSLDX"
INDIA_VIX = "^INDIAVIX"


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


def _yf_symbol(sym: str) -> str:
    return f"{sym}.NS"


def download_prices(symbols: list[str], start: str, end: str | None = None,
                    chunk: int = 80, refresh: bool = False) -> Panel:
    """Download split- and dividend-adjusted daily OHLCV for NSE symbols."""
    import yfinance as yf

    path = CACHE_DIR / "prices"
    if path.exists() and not refresh:
        return Panel.load(path)

    frames = {f: [] for f in FIELDS}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        tickers = [_yf_symbol(s) for s in batch]
        for attempt in range(3):
            try:
                raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                                  group_by="column", progress=False, threads=True)
                break
            except Exception:  # noqa: BLE001 - retry transient failures
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for f in FIELDS:
            part = raw[f] if isinstance(raw.columns, pd.MultiIndex) else raw[[f]]
            part = part.rename(columns=lambda t: t.removesuffix(".NS"))
            frames[f].append(part)
        print(f"downloaded {min(i + chunk, len(symbols))}/{len(symbols)}", flush=True)

    panel = Panel(**{f.lower(): pd.concat(frames[f], axis=1).sort_index() for f in FIELDS})
    # Drop symbols with almost no history.
    keep = panel.close.notna().sum() > 60
    panel = Panel(**{f.lower(): getattr(panel, f.lower()).loc[:, keep] for f in FIELDS})
    panel.save(path)
    return panel


def download_index(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()
