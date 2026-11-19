"""Build the dashboard's data files from the latest NSE close.

    python -m agent.dashboard --out dashboard_data

Writes one JSON file per dashboard dataset (watchlist, price history, market
regime, picks history, company cards, backtest results) into --out, plus
manifest.json with the as-of date. The nightly run uploads these to the
dashboard.

The dashboard's date-range filter reaches back 3 years of prices, so a fresh
machine downloads about four years of NSE files (the extra year warms up the
200 DMA). The picks history rebuilds each of the last 250 sessions' lists from
data known on that day, so it needs no stored state between runs.
Fundamentals are fetched only for stocks whose technical score put them in
contention on one of those days, since the fundamentals pillar carries 20 of
the 100 points.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import fundamentals as fu
from . import longterm as lt
from . import regime as rg
from . import scoring as sc
from .trendlines import trend_lines
from .data import INDIA_VIX, NIFTY500, load_index, load_prices
from .selection import SEGMENTS, pick_by_segment
from .universe import load_universe

CONTENDERS = 30          # per size: stocks whose quarterly results are refreshed each night
LONG_MAX_AGE_DAYS = 30   # annual accounts for the long-term list change slowly
LONG_FETCH_LIMIT = 80    # pages refreshed per night for the long-term list
HISTORY_SESSIONS = 750   # candles shown per stock (about 3 years)
REGIME_SESSIONS = 750
PICKS_SESSIONS = 250     # sessions of rebuilt daily picks (about 1 year)
SIZE_LABEL = {"large": "Large cap", "mid": "Mid cap", "small": "Small cap"}
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


def pct(x):
    v = r2(x)
    return None if v is None else round(v * 100, 1)


def _dump(out: Path, name: str, rows) -> int:
    (out / f"{name}.json").write_text(json.dumps(rows))
    return len(rows)


def _contenders(tech: pd.Series, loose: pd.Series, seg: pd.Series) -> list[str]:
    """Per size, the best technical scores that pass the trend template plus
    the best that miss it (candidates to fill a short size)."""
    out = []
    for name in SEGMENTS:
        mine = seg.reindex(tech.index).eq(name)
        out += tech[mine].dropna().nlargest(CONTENDERS).index.tolist()
        out += loose[mine].dropna().nlargest(CONTENDERS // 2).index.tolist()
    return list(dict.fromkeys(out))


def build(out: Path, start: str, offline: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    uni = load_universe()
    panel, deliv, t2t = load_prices(uni["symbol"].tolist(), start, refresh=not offline)
    idx = load_index(NIFTY500, start)["Close"]
    vix = load_index(INDIA_VIX, start)["Close"]
    info = uni.set_index("symbol")
    seg = info["segment"].reindex(panel.symbols).fillna("small")
    industry = info["industry"].reindex(panel.symbols).fillna("Unclassified")
    sized = [s for s in panel.symbols if seg[s] in SEGMENTS]   # microcaps left out

    f = sc.build_features(panel)
    ok = (sc.eligible(f, seg) & ~t2t.reindex_like(panel.close).fillna(False).astype(bool)
          & seg.isin(SEGMENTS).to_numpy()[None, :])
    reg = rg.market_regime(idx, panel.close, vix)
    c = panel.close
    last = c.index[-1]
    okl = ok.loc[last]

    # Mid term: technical score first, then quarterly fundamentals for the
    # contenders only (they carry 20 of the 100 points).
    tech, tech_extra = sc.score_month(f, industry)
    contenders = _contenders(tech.loc[last].where(okl), tech_extra["ungated"].loc[last].where(okl), seg)
    # The picks history rebuilds each past day's lists the same way, so it
    # needs the quarterly results of every stock that was in contention on
    # any of those days.
    hdates = c.index[-PICKS_SESSIONS:]
    ht, hu = tech.loc[hdates].where(ok.loc[hdates]), tech_extra["ungated"].loc[hdates].where(ok.loc[hdates])
    hist_contenders = list(dict.fromkeys(contenders + [s for d in hdates for s in _contenders(ht.loc[d], hu.loc[d], seg)]))
    contenders = hist_contenders
    if not offline:
        fu.fetch(contenders)
        fu.fetch(sized, max_age_days=LONG_MAX_AGE_DAYS, limit=LONG_FETCH_LIMIT)
    fund, flatest = fu.fundamentals_panel(contenders, industry, c.index)
    fund = fund.reindex(columns=c.columns)
    score, extra = sc.score_month(f, industry, fund)
    mid = pick_by_segment(score.loc[last].where(okl), extra["ungated"].loc[last].where(okl), seg)

    # Long term: annual accounts for every large, mid and small cap.
    lfund, llatest = lt.fundamentals_panels(sized, industry, c.index)
    lscore, lparts = lt.score(f, lfund)
    long_ = pick_by_segment(lscore.loc[last].where(okl), None, seg)

    def base(sym):
        return dict(symbol=sym, company=info.at[sym, "company"] if sym in info.index else sym,
                    industry=info.at[sym, "industry"] if sym in info.index else "",
                    segment=seg[sym], size=SIZE_LABEL.get(seg[sym], seg[sym]))

    # 1. Mid-term list (picks when the market is not risk-off) with trade levels.
    atr, sma50, sma200 = f.atr14.loc[last], f.sma50.loc[last], f.sma200.loc[last]
    rows = []
    for pk in mid:
        sym = pk["symbol"]
        cl = c.at[last, sym]
        stop = cl - 3 * atr[sym]
        risk = cl - stop
        fl = flatest.get(sym)
        rows.append(dict(
            rank=pk["rank"], **base(sym), filled=pk["filled"],
            passes_trend_template=not pk["filled"],
            score=r2(pk["score"]), close=r2(cl), stop=r2(stop), book_at=r2(cl + 1.5 * risk),
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

    # 1b. Long-term list.
    lrows = []
    for pk in long_:
        sym = pk["symbol"]
        cl = c.at[last, sym]
        a = llatest[sym].iloc[-1]
        sn = fu.snapshot(sym) or {}
        lrows.append(dict(
            rank=pk["rank"], **base(sym), score=r2(pk["score"]), close=r2(cl),
            sma200=r2(sma200[sym]), review_below=r2(0.9 * sma200[sym]),
            **{k: r2(lparts[k].at[last, sym]) for k in lt.WEIGHTS},
            roce_3y=pct(a["roce3"]), roe_3y=pct(a["roe3"]), debt_equity=r2(a["de"]),
            sales_cagr_3y=pct(a["sales_cagr3"]), profit_cagr_3y=pct(a["profit_cagr3"]),
            profit_up_years_5=r2(a["profit_up5"]), cfo_to_profit_3y=r2(a["cfo_ratio"]),
            pe=r2(sn.get("pe")), peg=r2(sn.get("peg")), market_cap_cr=r2(sn.get("market_cap_cr")),
            ret_6m=r2((cl / c[sym].iloc[-127] - 1) * 100),
            ret_12m=r2((cl / c[sym].iloc[-253] - 1) * 100),
            pct_of_52w_high=r2(cl / panel.high[sym].iloc[-252:].max() * 100),
            turnover_cr=r2(f.turnover20.at[last, sym] / sc.CRORE)))
    counts["longterm"] = _dump(out, "longterm", lrows)

    # 1c. Picks history: each past session's lists, rebuilt with the data
    # known on that day (fundamentals are lagged to their publication date).
    hrows = []
    sc_h, ug_h, ls_h = score.loc[hdates].where(ok.loc[hdates]), extra["ungated"].loc[hdates].where(ok.loc[hdates]), lscore.loc[hdates].where(ok.loc[hdates])
    for d in hdates:
        ds = str(d.date())
        for lst, picks in (("mid", pick_by_segment(sc_h.loc[d], ug_h.loc[d], seg)),
                           ("long", pick_by_segment(ls_h.loc[d], None, seg))):
            for pk in picks:
                hrows.append(dict(date=ds, list=lst, symbol=pk["symbol"], size=SIZE_LABEL[pk["segment"]],
                                  rank=pk["rank"], score=r2(pk["score"]), filled=pk["filled"],
                                  close=r2(c.at[d, pk["symbol"]]), close_now=r2(c.at[last, pk["symbol"]])))
    counts["picks_history"] = _dump(out, "picks_history", hrows)

    shown = list(dict.fromkeys([r["symbol"] for r in rows] + [r["symbol"] for r in lrows]))

    # 2. Candles with the 50 and 200 DMA and 20 EMA.
    ph = []
    for sym in shown:
        sub = pd.DataFrame({"open": panel.open[sym], "high": panel.high[sym], "low": panel.low[sym],
                            "close": c[sym], "volume": panel.volume[sym],
                            "sma50": f.sma50[sym], "sma200": f.sma200[sym], "ema20": f.ema20[sym]})
        for d, r in sub.dropna(subset=["close"]).iloc[-HISTORY_SESSIONS:].iterrows():
            ph.append(dict(symbol=sym, date=str(d.date()), open=r2(r.open), high=r2(r.high),
                           low=r2(r.low), close=r2(r.close),
                           volume=int(r.volume) if np.isfinite(r.volume) else None,
                           sma50=r2(r.sma50), sma200=r2(r.sma200), ema20=r2(r.ema20)))
    counts["price_history"] = _dump(out, "price_history", ph)

    # 2b. Automatic support and resistance over the same sessions.
    tl = []
    for sym in shown:
        sub = pd.DataFrame({"low": panel.low[sym], "high": panel.high[sym], "close": c[sym]})
        sub = sub.dropna().iloc[-HISTORY_SESSIONS:].rename_axis("date").reset_index()
        tl += [dict(symbol=sym, **line) for line in trend_lines(sub)]
    counts["trendlines"] = _dump(out, "trendlines", tl)

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
    for sym in shown:
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
                    mid_term={n: [r["symbol"] + ("*" if r["filled"] else "") for r in rows
                                  if r["segment"] == n] for n in SEGMENTS},
                    long_term={n: [r["symbol"] for r in lrows if r["segment"] == n] for n in SEGMENTS},
                    rows=counts)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


MID_VARIANTS = {"with_regime": "", "by_segment": ", 10 per size",
                "by_segment_filled": ", 10 per size, gaps filled"}
EXIT_RULES = {"hold_12m": "Hold 12 months",
              "exit_below_200dma": "Hold 12 months, sell 10% under the 200 DMA"}


def _long_row(row, period, size, rule, o, passes=None) -> dict:
    return dict(row=row, period=period, size=size, exit_rule=rule, trades=o.get("trades", 0),
                win_rate=pct(o.get("win_rate")), avg_net_return=pct(o.get("avg_net_return")),
                avg_segment_return=pct(o.get("avg_segment_return")),
                avg_nifty500_return=pct(o.get("avg_nifty500_return")),
                avg_excess_segment=pct(o.get("avg_excess_segment")),
                years_beating_pct=pct(o.get("years_beating_segment")), passes=passes)


def _long_backtest_files(out: Path) -> dict:
    """Long Term tab, from the committed results of the last long-term backtest."""
    path = Path("results/backtest_long_summary.json")
    if not path.exists():
        return {"long_backtest_summary": _dump(out, "long_backtest_summary", []),
                "long_backtest_by_year": _dump(out, "long_backtest_by_year", [])}
    js = json.loads(path.read_text())
    best = js["chosen_exit"]
    other = next(e for e in EXIT_RULES if e != best)
    res = js["results"]
    m, alt = res[f"model/{best}"], res[f"model/{other}"]
    p_ok = lambda o: bool(o.get("trades") and o["avg_excess_segment"] >= 0.03  # noqa: E731
                          and o["years_beating_segment"] >= 0.6)
    rows = [
        _long_row("tune-all", "2017–2021 (tuning)", "All sizes", EXIT_RULES[best], m["tune_2017_2021"], p_ok(m["tune_2017_2021"])),
        _long_row("check-all", "2022 on (unseen)", "All sizes", EXIT_RULES[best], m["check_2022_on"], p_ok(m["check_2022_on"])),
    ]
    rows += [_long_row(f"all-{k}", "All years", SIZE_LABEL[k] + "s", EXIT_RULES[best], m["by_segment"][k])
             for k in SEGMENTS if k in m["by_segment"]]
    rows += [
        _long_row("tune-alt", "2017–2021 (tuning)", "All sizes", EXIT_RULES[other], alt["tune_2017_2021"], p_ok(alt["tune_2017_2021"])),
        _long_row("check-alt", "2022 on (unseen)", "All sizes", EXIT_RULES[other], alt["check_2022_on"], p_ok(alt["check_2022_on"])),
    ]
    for v, label in (("fundamentals_only", "fundamentals only"), ("technical_only", "chart only")):
        if f"{v}/hold_12m" in res:
            rows.append(_long_row(f"check-{v}", f"2022 on, {label}", "All sizes", EXIT_RULES["hold_12m"],
                                  res[f"{v}/hold_12m"]["check_2022_on"]))
    by = pd.read_csv("results/backtest_long_by_year.csv")
    by = by[by.exit_rule == best]
    years = [dict(year=int(r.year), size=SIZE_LABEL[r.segment] + "s", trades=int(r.trades),
                  avg_net_return=pct(r.avg_net_return), avg_segment_return=pct(r.avg_segment_return),
                  avg_nifty500_return=pct(r.avg_nifty500_return), avg_excess_segment=pct(r.avg_excess_segment))
             for r in by.itertuples()]
    return {"long_backtest_summary": _dump(out, "long_backtest_summary", rows),
            "long_backtest_by_year": _dump(out, "long_backtest_by_year", years)}


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
            if variant not in MID_VARIANTS or not o.get("trades"):
                continue
            summ.append(dict(universe=name, horizon=h.replace("_", " ").capitalize() + MID_VARIANTS[variant],
                             trades=o["trades"], hit_rate=r2(o["hit_rate"] * 100),
                             win_rate=r2(o["win_rate"] * 100), avg_r=round(o["avg_r"], 3),
                             avg_net_return=r2(o["avg_net_return"] * 100),
                             avg_excess_return=r2(o["avg_excess_return"] * 100),
                             years_beating_pct=r2(o["years_beating_nifty500"] * 100),
                             passes=o["passes"]))
    counts["backtest_by_year"] = _dump(out, "backtest_by_year", by)
    counts["backtest_summary"] = _dump(out, "backtest_summary", summ)
    counts.update(_long_backtest_files(out))

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
    ap.add_argument("--years", type=float, default=4.0, help="years of price history to load")
    ap.add_argument("--offline", action="store_true", help="use cached files only")
    args = ap.parse_args()
    start = (pd.Timestamp.today() - pd.DateOffset(days=int(args.years * 365))).strftime("%Y-%m-%d")
    print(json.dumps(build(Path(args.out), start, args.offline), indent=2))


if __name__ == "__main__":
    main()
