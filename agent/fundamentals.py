"""Quarterly fundamentals from screener.in company pages.

Each page carries about 13 quarters of results, 12 years of annual P&L and
balance sheet, and 12 quarters of shareholding. Pages are cached under
data/cache/screener/ and fetched politely (one request every 1.5 seconds,
paths allowed by the site's robots.txt), so a full refresh of ~750 stocks
takes about 20 minutes and is only needed once a week.

Results are treated as public only some time after the period ends (SEBI
allows 45 days after a quarter, 60 after the March year end), so every number
enters the score on its availability date and never before.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from . import indicators as ind

BASE = "https://www.screener.in/company/"
CACHE = Path("data/cache/screener")
HEADERS = {"User-Agent": "Mozilla/5.0 (TradingAgent personal research)"}
DELAY = 1.5
MAX_AGE_DAYS = 7
FINANCIALS = {"Financial Services"}   # banks and NBFCs: no debt/equity test


def _section(html: str, name: str) -> pd.DataFrame | None:
    i = html.find(f'<section id="{name}"')
    if i < 0:
        return None
    j = html.find("</section>", i)
    try:
        t = pd.read_html(io.StringIO(html[i:j]), flavor="lxml")[0]
    except ValueError:
        return None
    t = t.rename(columns={t.columns[0]: "item"})
    t["item"] = t["item"].astype(str).str.replace("+", "", regex=False).str.strip()
    return t.set_index("item")


def _has_quarters(html: str) -> bool:
    q = _section(html, "quarters")
    return q is not None and q.shape[1] >= 5


def fetch(symbols: list[str], refresh: bool = False) -> None:
    """Cache each symbol's page: consolidated if it has results, else standalone."""
    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    todo = []
    for s in symbols:
        p = CACHE / f"{s}.html"
        if refresh or not p.exists() or time.time() - p.stat().st_mtime > MAX_AGE_DAYS * 86400:
            todo.append(s)
    for n, sym in enumerate(todo, 1):
        html = None
        for path in (f"{quote(sym)}/consolidated/", f"{quote(sym)}/"):
            try:
                r = session.get(BASE + path, headers=HEADERS, timeout=30)
            except requests.RequestException:
                time.sleep(DELAY * 4)
                continue
            time.sleep(DELAY)
            if r.status_code == 429:
                time.sleep(60)
                continue
            if r.status_code == 200 and _has_quarters(r.text):
                html = r.text
                break
        if html:
            (CACHE / f"{sym}.html").write_text(html)
        if n % 50 == 0 or n == len(todo):
            print(f"screener: {n}/{len(todo)}", flush=True)


def _num(x) -> float:
    if isinstance(x, str):
        x = x.replace(",", "").replace("%", "").strip()
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _series(t: pd.DataFrame | None, item: str) -> pd.Series:
    if t is None or item not in t.index:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    row = t.loc[item]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    out = {}
    for col, v in row.items():
        d = pd.to_datetime(col, format="%b %Y", errors="coerce")
        if pd.notna(d):
            out[d + pd.offsets.MonthEnd(0)] = _num(v)
    return pd.Series(list(out.values()), index=pd.DatetimeIndex(list(out.keys())), dtype=float).sort_index()


def _available(period_end: pd.DatetimeIndex) -> pd.DatetimeIndex:
    lag = np.where(period_end.month == 3, 60, 45)
    return period_end + pd.to_timedelta(lag, unit="D")


def parse(sym: str) -> dict[str, pd.Series] | None:
    p = CACHE / f"{sym}.html"
    if not p.exists():
        return None
    html = p.read_text()
    q, pl = _section(html, "quarters"), _section(html, "profit-loss")
    bs, sh = _section(html, "balance-sheet"), _section(html, "shareholding")
    equity = _series(bs, "Equity Capital").add(_series(bs, "Reserves"), fill_value=0)
    return dict(
        q_sales=_series(q, "Sales").combine_first(_series(q, "Revenue")),
        q_profit=_series(q, "Net Profit"),
        q_eps=_series(q, "EPS in Rs"),
        y_profit=_series(pl, "Net Profit"),
        equity=equity,
        borrowings=_series(bs, "Borrowings"),
        promoters=_series(sh, "Promoters"),
        fii=_series(sh, "FIIs"),
        dii=_series(sh, "DIIs"),
    )


def _yoy(s: pd.Series) -> pd.Series:
    prev = s.shift(4)
    return (s - prev) / prev.abs().where(prev.abs() > 0)


def stock_signals(d: dict[str, pd.Series], financial: bool) -> pd.DataFrame:
    """One row per date a new number became public, with the 0-100 sub-scores."""
    rows = {}
    eps_g = _yoy(d["q_eps"])
    sales_g = _yoy(d["q_sales"])
    q = pd.DataFrame({"eps_g": eps_g, "eps_g_prev": eps_g.shift(1), "sales_g": sales_g})
    q.index = _available(q.index)
    rows["q"] = q

    eq_avg = (d["equity"] + d["equity"].shift(1)) / 2
    y = pd.DataFrame({"roe": d["y_profit"].reindex(eq_avg.index) / eq_avg,
                      "de": d["borrowings"] / d["equity"]})
    y.index = _available(y.index)
    rows["y"] = y

    sh = pd.DataFrame({"promoter_chg": d["promoters"].diff(),
                       "inst_chg": d["fii"].add(d["dii"], fill_value=0).diff()})
    sh.index = sh.index + pd.Timedelta(days=21)   # filed within 21 days of quarter end
    rows["sh"] = sh

    out = pd.concat(rows.values(), axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")].ffill()
    eps = (ind.ramp(out["eps_g"], 0.0, 0.20) * 0.7
           + ((out["eps_g"] > out["eps_g_prev"]) & (out["eps_g"] > 0)).astype(float) * 30)
    sales = ind.ramp(out["sales_g"], 0.0, 0.15)
    roe = ind.ramp(out["roe"], 0.05, 0.15)
    quality = roe if financial else roe * 0.6 + ind.ramp(out["de"], 2.0, 1.0) * 0.4
    holding = (out["promoter_chg"].ge(-0.25).astype(float) * 50
               + ind.ramp(out["inst_chg"], 0.0, 1.0) * 0.5)
    parts = pd.DataFrame({"eps": eps, "sales": sales, "quality": quality, "holding": holding})
    weights = pd.Series({"eps": 35, "sales": 25, "quality": 25, "holding": 15})
    have = parts.notna()
    score = (parts.fillna(0) * weights).sum(axis=1) / (have * weights).sum(axis=1).replace(0, np.nan)
    # Without any quarterly growth figure the score says little: leave it out.
    score = score.where(have["eps"] | have["sales"])
    return pd.concat([parts, out[["eps_g", "sales_g", "roe", "de"]], score.rename("score")], axis=1)


def fundamentals_panel(symbols: list[str], industry: pd.Series,
                       dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """0-100 fundamentals score as a dates x symbols panel (NaN where unknown),
    plus each stock's latest signal row for display."""
    cols, latest = {}, {}
    for sym in symbols:
        d = parse(sym)
        if d is None or d["q_sales"].empty:
            continue
        sig = stock_signals(d, industry.get(sym) in FINANCIALS)
        if sig["score"].notna().any():
            cols[sym] = sig["score"].reindex(sig.index.union(dates)).ffill().reindex(dates)
            latest[sym] = sig
    panel = pd.DataFrame(cols, index=dates).reindex(columns=symbols)
    return panel, latest


def _top_ratios(html: str) -> dict[str, float]:
    import re
    i = html.find('id="top-ratios"')
    if i < 0:
        return {}
    block = html[i:html.find("</ul>", i)]
    out = {}
    for name, value in re.findall(r'<span class="name">\s*(.*?)\s*</span>.*?<span class="number">([^<]*)</span>', block, re.S):
        out[name.strip()] = _num(value)
    return out


def snapshot(sym: str) -> dict | None:
    """Everything shown on the dashboard's company card, as of the latest page.

    Valuation comes from the page's current figures; growth, margins, cash
    flow and holdings from the latest reported period. Promoter pledges are not
    on the public page, so they are not included.
    """
    p = CACHE / f"{sym}.html"
    if not p.exists():
        return None
    html = p.read_text()
    top = _top_ratios(html)
    d = parse(sym)
    q, pl = _section(html, "quarters"), _section(html, "profit-loss")
    cf, ra = _section(html, "cash-flow"), _section(html, "ratios")
    opm = _series(q, "OPM %").combine_first(_series(q, "Financing Margin %"))
    eps_ttm = d["q_eps"].rolling(4).sum()
    eps_ttm_g = (eps_ttm / eps_ttm.shift(4) - 1).dropna()
    cfo, np_y = _series(cf, "Cash from Operating Activity"), _series(pl, "Net Profit")
    last = lambda s: float(s.dropna().iloc[-1]) if s.notna().any() else None  # noqa: E731
    pe, price, book = top.get("Stock P/E"), top.get("Current Price"), top.get("Book Value")
    g = last(eps_ttm_g)
    return dict(
        market_cap_cr=top.get("Market Cap"), pe=pe,
        peg=round(pe / (g * 100), 2) if pe and g and g > 0 else None,
        pb=round(price / book, 2) if price and book else None,
        dividend_yield=top.get("Dividend Yield"),
        roe=top.get("ROE"), roce=top.get("ROCE"),
        opm_latest=last(opm), opm_year_ago=float(opm.iloc[-5]) if len(opm) >= 5 else None,
        eps_ttm_growth=round(g * 100, 1) if g is not None else None,
        cfo_to_profit=round(last(cfo) / last(np_y), 2) if last(cfo) is not None and last(np_y) else None,
        promoters=last(d["promoters"]), fii=last(d["fii"]), dii=last(d["dii"]),
        promoters_chg_1y=float(d["promoters"].iloc[-1] - d["promoters"].iloc[-5]) if len(d["promoters"]) >= 5 else None,
        inst_chg_1y=float((d["fii"] + d["dii"]).iloc[-1] - (d["fii"] + d["dii"]).iloc[-5]) if len(d["fii"]) >= 5 else None,
        quarters=pd.DataFrame({"sales": d["q_sales"], "net_profit": d["q_profit"], "opm": opm, "eps": d["q_eps"]}),
        holding=pd.DataFrame({"promoters": d["promoters"], "fii": d["fii"], "dii": d["dii"]}),
        roce_history=_series(ra, "ROCE %"),
    )
