"""Stock universe: Nifty 500 (which already contains the Midcap 150 and
Smallcap 250) plus the Nifty Microcap 250 for small caps outside it."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

INDEX_FILES = {
    "nifty500": "ind_nifty500list.csv",
    "midcap150": "ind_niftymidcap150list.csv",
    "smallcap250": "ind_niftysmallcap250list.csv",
    "microcap250": "ind_niftymicrocap250_list.csv",
}

# NSE sometimes refuses cloud IPs, so try both official hosts.
BASE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/",
    "https://www.niftyindices.com/IndexConstituent/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}

CACHE = Path("data/universe.csv")


def _fetch_index(file_name: str) -> pd.DataFrame:
    errors = []
    for base in BASE_URLS:
        try:
            resp = requests.get(base + file_name, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as exc:  # noqa: BLE001 - try the next host
            errors.append(f"{base}{file_name}: {exc}")
    raise RuntimeError("Could not download index list:\n" + "\n".join(errors))


def load_universe(refresh: bool = False) -> pd.DataFrame:
    """Return one row per stock: symbol, company, industry, segment.

    segment is "large" (Nifty 100), "mid" (Midcap 150), "small"
    (Smallcap 250) or "micro" (Microcap 250).
    """
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE)

    lists = {name: _fetch_index(f) for name, f in INDEX_FILES.items()}
    mid = set(lists["midcap150"]["Symbol"])
    small = set(lists["smallcap250"]["Symbol"])

    rows = []
    for _, r in lists["nifty500"].iterrows():
        sym = r["Symbol"]
        seg = "mid" if sym in mid else "small" if sym in small else "large"
        rows.append((sym, r["Company Name"], r["Industry"], seg))
    in500 = {r[0] for r in rows}
    for _, r in lists["microcap250"].iterrows():
        if r["Symbol"] not in in500:
            rows.append((r["Symbol"], r["Company Name"], r["Industry"], "micro"))

    uni = pd.DataFrame(rows, columns=["symbol", "company", "industry", "segment"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(CACHE, index=False)
    return uni
