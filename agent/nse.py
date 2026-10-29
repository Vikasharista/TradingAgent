"""End-of-day data from NSE's public archives (archives.nseindia.com).

Each trading day has three files:
  * the equity bhavcopy (open, high, low, close, previous close, volume),
  * the MTO file with delivery quantity and delivery %,
  * ind_close_all with every NSE index, including Nifty 500 and India VIX.

Days are cached one parquet file each under data/cache/nse/, so a nightly run
only fetches the newest day. Prices are back-adjusted for splits, bonuses and
rights (see adjustment_factors).
"""

from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://archives.nseindia.com/"
UDIFF_FROM = date(2024, 7, 8)   # bhavcopy switched to the UDiFF format here
SERIES = {"EQ", "BE"}           # rolling and trade-for-trade equity
CACHE = Path("data/cache/nse")
INDICES = {"Nifty 500": "nifty500", "Nifty 50": "nifty50", "India VIX": "india_vix"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


class Holiday(Exception):
    pass


def _get(session: requests.Session, url: str) -> bytes | None:
    """Body of url, or None on 404 (no file for that day)."""
    for attempt in range(5):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}")


def _bhav_url(d: date) -> str:
    if d >= UDIFF_FROM:
        return f"{BASE}content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    return f"{BASE}content/historical/EQUITIES/{d.year}/{mon}/cm{d:%d}{mon}{d.year}bhav.csv.zip"


def _parse_bhav(raw: bytes, d: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip() for c in df.columns]
    if d >= UDIFF_FROM:
        df = df.rename(columns={
            "TckrSymb": "symbol", "SctySrs": "series", "OpnPric": "open",
            "HghPric": "high", "LwPric": "low", "ClsPric": "close",
            "PrvsClsgPric": "prev_close", "TtlTradgVol": "volume",
            "TtlTrfVal": "turnover", "ISIN": "isin",
        })
    else:
        df = df.rename(columns={
            "SYMBOL": "symbol", "SERIES": "series", "OPEN": "open", "HIGH": "high",
            "LOW": "low", "CLOSE": "close", "PREVCLOSE": "prev_close",
            "TOTTRDQTY": "volume", "TOTTRDVAL": "turnover", "ISIN": "isin",
        })
    df["series"] = df["series"].astype(str).str.strip()
    df = df[df["series"].isin(SERIES)]
    cols = ["symbol", "series", "isin", "open", "high", "low", "close", "prev_close",
            "volume", "turnover"]
    return df[cols].astype({"symbol": str, "isin": str})


def _parse_mto(raw: bytes) -> pd.DataFrame:
    rows = []
    for line in raw.decode("latin-1").splitlines():
        parts = line.split(",")
        if len(parts) == 7 and parts[0] == "20":
            rows.append((parts[2].strip(), parts[3].strip(), parts[5], parts[6]))
    df = pd.DataFrame(rows, columns=["symbol", "series", "deliv_qty", "deliv_pct"])
    df["deliv_qty"] = pd.to_numeric(df["deliv_qty"], errors="coerce")
    df["deliv_pct"] = pd.to_numeric(df["deliv_pct"], errors="coerce")
    return df


def _parse_indices(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df["Index Name"] = df["Index Name"].str.strip()
    df = df[df["Index Name"].isin(INDICES)]
    out = df.rename(columns={
        "Index Name": "index", "Open Index Value": "open", "High Index Value": "high",
        "Low Index Value": "low", "Closing Index Value": "close",
    })[["index", "open", "high", "low", "close"]]
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fetch_day(d: date, session: requests.Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _get(session, _bhav_url(d))
    if raw is None:
        raise Holiday(d)
    eq = _parse_bhav(raw, d)
    mto = _get(session, f"{BASE}archives/equities/mto/MTO_{d:%d%m%Y}.DAT")
    if mto is not None:
        eq = eq.merge(_parse_mto(mto), on=["symbol", "series"], how="left")
    else:
        eq["deliv_qty"], eq["deliv_pct"] = np.nan, np.nan
    idx = _get(session, f"{BASE}content/indices/ind_close_all_{d:%d%m%Y}.csv")
    idx = _parse_indices(idx) if idx is not None else pd.DataFrame(
        columns=["index", "open", "high", "low", "close"])
    eq.insert(0, "date", pd.Timestamp(d))
    idx.insert(0, "date", pd.Timestamp(d))
    return eq, idx


def _day_paths(d: date) -> tuple[Path, Path, Path]:
    return (CACHE / "eq" / f"{d:%Y%m%d}.parquet",
            CACHE / "idx" / f"{d:%Y%m%d}.parquet",
            CACHE / "holidays" / f"{d:%Y%m%d}")


def sync(start: str, end: str | None = None, workers: int = 16) -> None:
    """Download every missing weekday between start and end (default today)."""
    for sub in ("eq", "idx", "holidays"):
        (CACHE / sub).mkdir(parents=True, exist_ok=True)
    first = pd.Timestamp(start).date()
    last = pd.Timestamp(end).date() if end else date.today()
    days = [first + timedelta(i) for i in range((last - first).days + 1)]
    todo = [d for d in days if d.weekday() < 5
            and not any(p.exists() for p in (_day_paths(d)[0], _day_paths(d)[2]))]
    # A file for today may simply not be published yet, so never mark it a holiday.
    session = requests.Session()

    def one(d: date) -> str:
        eq_p, idx_p, hol_p = _day_paths(d)
        try:
            eq, idx = fetch_day(d, session)
        except Holiday:
            if d < last:
                hol_p.touch()
            return "holiday"
        eq.to_parquet(eq_p, index=False)
        idx.to_parquet(idx_p, index=False)
        return "ok"

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(one, todo):
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"nse: {done}/{len(todo)} days", flush=True)


def _symbol_map() -> dict[str, str]:
    """Old symbol -> current symbol, following chains of renames."""
    path = CACHE / "symbolchange.csv"
    if not path.exists():
        raw = _get(requests.Session(), f"{BASE}content/equities/symbolchange.csv")
        if raw is None:
            return {}
        path.write_bytes(raw)
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    m = dict(zip(df["OLD SYMBOL"].str.strip(), df["NEW SYMBOL"].str.strip()))
    out = {}
    for old in m:
        new, seen = m[old], {old}
        while new in m and new not in seen:
            seen.add(new)
            new = m[new]
        out[old] = new
    return out


def load_equities(start: str | None = None) -> pd.DataFrame:
    files = sorted((CACHE / "eq").glob("*.parquet"))
    if start:
        cut = pd.Timestamp(start).strftime("%Y%m%d")
        files = [f for f in files if f.stem >= cut]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["symbol"] = df["symbol"].replace(_symbol_map())
    # A stock can sit in EQ and BE on the same day only around a series move;
    # keep the line with more volume.
    df = df.sort_values("volume").drop_duplicates(["date", "symbol"], keep="last")
    return df


def load_index(name: str, start: str | None = None) -> pd.DataFrame:
    files = sorted((CACHE / "idx").glob("*.parquet"))
    if start:
        cut = pd.Timestamp(start).strftime("%Y%m%d")
        files = [f for f in files if f.stem >= cut]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["index"] == name].set_index("date").sort_index()
    df.columns = [c.capitalize() for c in df.columns]
    return df[["Open", "High", "Low", "Close"]].dropna()


# Price bands cap a normal session at +/-20%, so a bigger overnight jump is a
# split, bonus or consolidation. The ratio is snapped to the nearest p/q.
_RATIOS = np.array(sorted({p / q for p in range(1, 11) for q in range(1, 11)}))


def _snap(r: np.ndarray) -> np.ndarray:
    idx = np.abs(_RATIOS[None, :] - r[:, None]).argmin(axis=1)
    near = _RATIOS[idx]
    return np.where(np.abs(near / r - 1) < 0.05, near, r)


def adjustment_factors(close: pd.DataFrame, prev_close: pd.DataFrame,
                       tol: float = 0.005, jump: float = 0.25) -> pd.DataFrame:
    """Multiplier for each bar so prices line up with the latest share count.

    Two sources of corporate-action ratios:
      * NSE sometimes restates the previous close on an ex-date, so
        prev_close / last close is the ratio for that date;
      * otherwise a close-to-close move beyond the 20% price band (a fall of
        more than `jump`, or the matching rise) is taken as a split or bonus.
    Bonus issues smaller than 1:3 (ratio above 0.75) are missed by the second
    rule and show up as an ordinary down day.
    """
    last = close.ffill().shift(1)
    restated = prev_close / last
    restated = restated.where((restated - 1).abs() > tol).fillna(1.0)
    move = close / (last * restated)
    big = (move < 1 - jump) | (move > 1 / (1 - jump))
    snapped = move.to_numpy(copy=True)
    mask = big.to_numpy()
    if mask.any():
        snapped[mask] = _snap(snapped[mask])
    snapped = pd.DataFrame(snapped, index=move.index, columns=move.columns)
    ratio = restated * snapped.where(big, 1.0).fillna(1.0)
    # Bar s gets the product of every ratio on dates after s.
    rev = ratio.iloc[::-1].cumprod().iloc[::-1]
    return (rev / ratio).fillna(1.0)
