"""Build the dashboard's data files from the latest NSE close.

    python -m agent.dashboard --out dashboard_data

Writes one JSON file per dashboard dataset (watchlist, price history, market
regime, company cards, backtest results) into --out, plus manifest.json with
the as-of date. The nightly run uploads these to the dashboard.

Only the last few years of prices are needed for today's scores, so a fresh
machine downloads about three years of NSE files (a few minutes). Fundamentals
are fetched only for the stocks whose technical score puts them in contention,
since the fundamentals pillar carries 20 of the 100 points.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import fundamentals as fu
from . import regime as rg
from . import scoring as sc
from .data import INDIA_VIX, NIFTY500, load_index, load_prices
from .universe import load_universe

TOP = 10
CONTENDERS = 60          # stocks whose fundamentals are refreshed each night
HISTORY_SESSIONS = 250   # candles shown per stock
REGIME_SESSIONS = 500
EXIT_LABELS = {"stop": "Stopped out", "expired": "Held to 60 sessions",
               "trail": "Trailed out (50 DMA)",
               "breakeven_after_booking": "Booked half, rest at break-even"}


def r2(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 2) if np.isfinite(x) else None


def _dump(out: Path, name: str, rows) -> int:
    (out / f"{name}.json").write_text(json.dumps(rows))
    return len(rows)


def build(out: Path, start: str, offline: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    uni = load_universe()
    panel, deliv, t2t = load_prices(uni["symbol"].tolist(), start, refresh=not offline)
    idx = load_index(NIFTY500, start)["Close"]
    vix = load_index(INDIA_VIX, start)["Close"]
    info = uni.set_index("symbol")
    seg = info["segment"].reindex(panel.symbols).fillna("small")
    industry = info["industry"].reindex(panel.symbols).fillna("Unclassified")

    f = sc.build_features(panel)
    ok = sc.eligible(f, seg) & ~t2t.reindex_like(panel.close).fillna(False).astype(bool)
    reg = rg.market_regime(idx, panel.close, vix)
    c = panel.close
    last = c.index[-1]

    # Technical score first, then fundamentals for the contenders only.
    tech, _ = sc.score_month(f, industry)
    contenders = tech.loc[last].where(ok.loc[last]).dropna().nlargest(CONTENDERS).index.tolist()
    if not offline:
        fu.fetch(contenders)
    fund, flatest = fu.fundamentals_panel(contenders, industry, c.index)
    fund = fund.reindex(columns=c.columns)
    score, extra = sc.score_month(f, industry, fund)
    s = score.loc[last].where(ok.loc[last]).dropna().sort_values(ascending=False)
    top = s.head(TOP).index

    # 1. Watchlist (picks when the market is not risk-off) with trade levels.
    atr, sma50 = f.atr14.loc[last], f.sma50.loc[last]
    rows = []
    for rank, sym in enumerate(top, 1):
        cl = c.at[last, sym]
        stop = cl - 3 * atr[sym]
        risk = cl - stop
        fl = flatest.get(sym)
        rows.append(dict(
            rank=rank, symbol=sym,
            company=info.at[sym, "company"] if sym in info.index else sym,
            industry=info.at[sym, "industry"] if sym in info.index else "",
            segment=info.at[sym, "segment"] if sym in info.index else "",
            score=r2(s[sym]), close=r2(cl), stop=r2(stop), book_at=r2(cl + 1.5 * risk),
            trail_50dma=r2(sma50[sym]), risk_pct=r2(risk / cl * 100),
            **{k: r2(extra[k].at[last, sym]) for k in
               ("rs_rating", "trend", "momentum", "accumulation", "weekly_setup", "sector")},
            fundamentals=r2(fund.at[last, sym]),
            eps_growth=r2(fl["eps_g"].iloc[-1] * 100) if fl is not None else None,
            sales_growth=r2(fl["sales_g"].iloc[-1] * 100) if fl is not None else None,
            roe=r2(fl["roe"].iloc[-1] * 100) if fl is not None else None,
            debt_equity=r2(fl["de"].iloc[-1]) if fl is not None else None,
            ret_1m=r2((cl / c[sym].iloc[-22] - 1) * 100),
            ret_3m=r2((cl / c[sym].iloc[-64] - 1) * 100),
            delivery_pct=r2(deliv[sym].iloc[-20:].mean()),
            turnover_cr=r2(f.turnover20.at[last, sym] / sc.CRORE)))
    counts = {"watchlist": _dump(out, "watchlist", rows)}

    # 2. Candles with the 50 DMA and 20 EMA.
    ph = []
    for sym in top:
        sub = pd.DataFrame({"open": panel.open[sym], "high": panel.high[sym], "low": panel.low[sym],
                            "close": c[sym], "volume": panel.volume[sym],
                            "sma50": f.sma50[sym], "ema20": f.ema20[sym]})
        for d, r in sub.dropna(subset=["close"]).iloc[-HISTORY_SESSIONS:].iterrows():
            ph.append(dict(symbol=sym, date=str(d.date()), open=r2(r.open), high=r2(r.high),
                           low=r2(r.low), close=r2(r.close),
                           volume=int(r.volume) if np.isfinite(r.volume) else None,
                           sma50=r2(r.sma50), ema20=r2(r.ema20)))
    counts["price_history"] = _dump(out, "price_history", ph)

    # 3. Market regime.
    i5 = idx.reindex(c.index, method="ffill")
    rgm = pd.DataFrame({"nifty500": i5, "nifty500_50dma": i5.rolling(50).mean(),
                        "breadth_pct": reg.breadth * 100,
                        "india_vix": vix.reindex(c.index, method="ffill"),
                        "regime": reg.regime}).iloc[-REGIME_SESSIONS:]
    counts["regime"] = _dump(out, "regime", [
        dict(date=str(d.date()), **{k: (v if k == "regime" else r2(v)) for k, v in r.items()})
        for d, r in rgm.iterrows()])

    # 4. Company cards.
    cm, qh, hh = [], [], []
    for sym in top:
        sn = fu.snapshot(sym)
        if sn is None:
            continue
        cm.append(dict(symbol=sym, **{k: r2(v) for k, v in sn.items() if not hasattr(v, "shape")}))
        for d, r in sn["quarters"].iterrows():
            qh.append(dict(symbol=sym, quarter=d.strftime("%Y-%m"), sales_cr=r2(r.sales),
                           net_profit_cr=r2(r.net_profit), margin_pct=r2(r.opm), eps=r2(r.eps)))
        for d, r in sn["holding"].iterrows():
            hh.append(dict(symbol=sym, quarter=d.strftime("%Y-%m"), promoters=r2(r.promoters),
                           fii=r2(r.fii), dii=r2(r.dii)))
    counts["company_metrics"] = _dump(out, "company_metrics", cm)
    counts["quarterly_results"] = _dump(out, "quarterly_results", qh)
    counts["shareholding"] = _dump(out, "shareholding", hh)

    counts.update(_backtest_files(out))
    manifest = dict(as_of=str(last.date()), regime=reg.regime.loc[last],
                    breadth_pct=r2(reg.breadth.loc[last] * 100),
                    built_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    picks=list(top), rows=counts)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _backtest_files(out: Path) -> dict:
    """Backtest tab, from the committed results of the last backtest run."""
    counts, by, summ = {}, [], []
    for name, path in (("Index members", Path("results")), ("All listed stocks", Path("results/all"))):
        if not (path / "backtest_by_year.csv").exists():
            continue
        b = pd.read_csv(path / "backtest_by_year.csv")
        for r in b[b.variant == "with_regime"].itertuples():
            by.append(dict(universe=name, horizon=r.horizon.replace("_", " ").capitalize(),
                           year=int(r.year), trades=int(r.trades), hit_rate=r2(r.hit_rate * 100),
                           win_rate=r2(r.win_rate * 100), avg_r=round(r.avg_r, 3),
                           avg_net_return=r2(r.avg_net_return * 100),
                           avg_nifty500_return=r2(r.avg_nifty500_return * 100),
                           avg_excess_return=r2(r.avg_excess_return * 100)))
        js = json.loads((path / "backtest_summary.json").read_text())
        for k, o in js["results"].items():
            h, variant = k.split("/")
            if variant != "with_regime" or not o.get("trades"):
                continue
            summ.append(dict(universe=name, horizon=h.replace("_", " ").capitalize(),
                             trades=o["trades"], hit_rate=r2(o["hit_rate"] * 100),
                             win_rate=r2(o["win_rate"] * 100), avg_r=round(o["avg_r"], 3),
                             avg_net_return=r2(o["avg_net_return"] * 100),
                             avg_excess_return=r2(o["avg_excess_return"] * 100),
                             years_beating_pct=r2(o["years_beating_nifty500"] * 100),
                             passes=o["passes"]))
    counts["backtest_by_year"] = _dump(out, "backtest_by_year", by)
    counts["backtest_summary"] = _dump(out, "backtest_summary", summ)

    t = pd.read_csv("results/trades_next_month.csv.gz")
    t = t[t.triggered]
    t["exit_month"] = pd.to_datetime(t.exit_date).dt.to_period("M").astype(str)
    m = t.groupby("exit_month").agg(trades=("r_multiple", "size"), sum_r=("r_multiple", "sum"),
                                    avg_net=("net_return", "mean"),
                                    avg_bench=("nifty500_return", "mean")).reset_index()
    counts["month_by_exit_month"] = _dump(out, "month_by_exit_month", [
        dict(month=r.exit_month, trades=int(r.trades), avg_r=round(r.sum_r / r.trades, 3),
             avg_net_pct=r2(r.avg_net * 100), avg_nifty500_pct=r2(r.avg_bench * 100))
        for r in m.itertuples()])
    ex = t.groupby("outcome").agg(trades=("r_multiple", "size"), avg_r=("r_multiple", "mean"),
                                  avg_net=("net_return", "mean")).reset_index()
    counts["month_exits"] = _dump(out, "month_exits", [
        dict(exit=EXIT_LABELS.get(r.outcome, r.outcome), trades=int(r.trades),
             share_pct=r2(r.trades / len(t) * 100), avg_r=round(r.avg_r, 3),
             avg_net_pct=r2(r.avg_net * 100)) for r in ex.itertuples()])
    rec = t.sort_values("date").drop_duplicates("symbol", keep="last").sort_values("date").tail(40)
    counts["recent_trades"] = _dump(out, "recent_trades", [
        dict(signal_date=r.date, symbol=r.symbol, score=r.score, entry=r2(r.entry),
             stop=r2(r.stop), book_at=r2(r.target), exit_date=r.exit_date,
             exit=EXIT_LABELS.get(r.outcome, r.outcome), booked_half=bool(r.booked),
             net_pct=r2(r.net_return * 100), r_multiple=round(r.r_multiple, 2))
        for r in rec.itertuples()])
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard_data")
    ap.add_argument("--years", type=float, default=3.0, help="years of price history to load")
    ap.add_argument("--offline", action="store_true", help="use cached files only")
    args = ap.parse_args()
    start = (pd.Timestamp.today() - pd.DateOffset(days=int(args.years * 365))).strftime("%Y-%m-%d")
    print(json.dumps(build(Path(args.out), start, args.offline), indent=2))


if __name__ == "__main__":
    main()
