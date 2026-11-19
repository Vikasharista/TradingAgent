"""Backtest of the long-term list: 10 large, 10 mid and 10 small caps bought
on the first session of each month and held for up to 12 months.

    python -m agent.backtest_long --start 2016-01-01 --test-from 2017-01-01

Two exit rules are compared:
  hold_12m           sell at the close 252 sessions after entry
  exit_below_200dma  also sell at the next open after a close more than 10%
                     below the 200 DMA (a broken long-term trend)

Each pick is compared with the Nifty 500 and, more fairly, with an equal
weight basket of every eligible stock of the same size over the same days.
Today's index members are used, which flatters returns (stocks that fell out
are missing); the same-size basket has the same bias, so the excess over it
is the cleaner number.

The exit rule is chosen on 2017-2021 signals only and then checked on 2022
onwards. Pass bar, set before the first run: in both periods the average
excess over the same-size basket is at least 3 percentage points per trade
and the picks beat it in at least 60% of signal years.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import longterm as lt
from . import scoring as sc
from .backtest import COMMISSION, SLIPPAGE
from .data import NIFTY500, load_index, load_prices
from .selection import SEGMENTS, pick_by_segment
from .universe import load_universe

HOLD = 252
TUNE_END = "2021-12-31"
PASS_EXCESS = 0.03
PASS_YEARS = 0.6
EXITS = ("hold_12m", "exit_below_200dma")
VARIANTS = {
    "model": lt.WEIGHTS,
    "fundamentals_only": {**lt.WEIGHTS, "trend": 0, "momentum": 0, "near_high": 0},
    "technical_only": {**lt.WEIGHTS, "quality": 0, "growth": 0, "cash": 0},
}


def month_starts(dates: pd.DatetimeIndex, start: str, room: int) -> pd.DatetimeIndex:
    d = dates[(dates >= pd.Timestamp(start))]
    d = d[: max(len(d) - room, 0)]
    s = pd.Series(d, index=d)
    return pd.DatetimeIndex(s.groupby([d.year, d.month]).first().values)


def segment_basket(close: pd.DataFrame, ok: pd.DataFrame, segment: pd.Series) -> pd.DataFrame:
    """Equal-weight index per size, rebuilt daily from eligible stocks."""
    ret = close.pct_change(fill_method=None).where(ok.shift(1, fill_value=False))
    out = {}
    for name in SEGMENTS:
        cols = segment.index[segment.eq(name)]
        r = ret[cols].clip(-0.5, 1.0).mean(axis=1).fillna(0)
        out[name] = (1 + r).cumprod()
    return pd.DataFrame(out)


def simulate(picks: list[dict], f: sc.Features, basket: pd.DataFrame,
             index_close: pd.Series, exit_rule: str) -> pd.DataFrame:
    p = f.p
    dates = p.close.index
    pos = {d: i for i, d in enumerate(dates)}
    col = {s: j for j, s in enumerate(p.close.columns)}
    O, C, SMA200 = p.open.to_numpy(), p.close.to_numpy(), f.sma200.to_numpy()
    idx = index_close.reindex(dates, method="ffill").to_numpy()
    bk = {k: basket[k].to_numpy() for k in basket}
    n = len(dates)
    rows = []
    for pk in picks:
        t, j = pos[pk["date"]], col[pk["symbol"]]
        e = t + 1
        if e >= n or not np.isfinite(O[e, j]):
            continue
        fill = O[e, j]
        last = min(e + HOLD - 1, n - 1)
        k, px, outcome = last, None, "held_12m"
        if exit_rule == "exit_below_200dma":
            for k2 in range(e, last):
                c, m = C[k2, j], SMA200[k2, j]
                if np.isfinite(c) and np.isfinite(m) and c < 0.9 * m:
                    k = k2 + 1
                    px = O[k, j] if np.isfinite(O[k, j]) else c
                    outcome = "broke_200dma"
                    break
        if px is None:
            col_c = C[: k + 1, j]
            good = np.where(np.isfinite(col_c))[0]
            if not len(good):
                continue
            k = int(good[-1])
            px = C[k, j]
        cost = 2 * (COMMISSION + SLIPPAGE.get(pk["segment"], 0.0025))
        net = px / fill - 1 - cost
        b = bk[pk["segment"]]
        seg_ret = b[k] / b[e - 1] - 1           # basket from the signal close
        nifty = idx[k] / idx[t] - 1 if np.isfinite(idx[t]) and np.isfinite(idx[k]) else np.nan
        rows.append(dict(date=pk["date"], symbol=pk["symbol"], segment=pk["segment"],
                         rank=pk["rank"], score=round(pk["score"], 1), entry=fill, exit=px,
                         exit_date=dates[k], outcome=outcome, days=k - e + 1,
                         net_return=net, segment_return=seg_ret, nifty500_return=nifty,
                         excess_segment=net - seg_ret, excess_nifty=net - nifty))
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    yr = t.assign(year=pd.to_datetime(t["date"]).dt.year).groupby("year")
    by_year = yr[["net_return", "segment_return", "nifty500_return", "excess_segment"]].mean()
    return dict(
        trades=int(len(t)),
        win_rate=float((t["net_return"] > 0).mean()),
        avg_net_return=float(t["net_return"].mean()),
        median_net_return=float(t["net_return"].median()),
        avg_segment_return=float(t["segment_return"].mean()),
        avg_nifty500_return=float(t["nifty500_return"].mean()),
        avg_excess_segment=float(t["excess_segment"].mean()),
        avg_excess_nifty=float(t["excess_nifty"].mean()),
        beat_segment_share=float((t["excess_segment"] > 0).mean()),
        years_beating_segment=float((by_year["excess_segment"] > 0).mean()),
        avg_days_held=float(t["days"].mean()),
    )


def passes(s: dict) -> bool:
    return bool(s.get("trades") and s["avg_excess_segment"] >= PASS_EXCESS
                and s["years_beating_segment"] >= PASS_YEARS)


def run(start: str, test_from: str, out: Path, offline: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    uni = load_universe()
    uni = uni[uni["segment"].isin(SEGMENTS)]
    panel, _, t2t = load_prices(uni["symbol"].tolist(), start, refresh=not offline)
    index_close = load_index(NIFTY500, start)["Close"]
    info = uni.set_index("symbol")
    segment = info["segment"].reindex(panel.symbols)
    industry = info["industry"].reindex(panel.symbols).fillna("Unclassified")
    f = sc.build_features(panel)
    ok = sc.eligible(f, segment) & ~t2t.reindex_like(panel.close).fillna(False).astype(bool)
    fund, _ = lt.fundamentals_panels(panel.symbols, industry, panel.close.index)
    basket = segment_basket(panel.close, ok, segment)
    rebal = month_starts(panel.close.index, test_from, HOLD + 1)

    results, yearly = {}, []
    for vname, w in VARIANTS.items():
        s, _ = lt.score(f, fund, w)
        s = s.where(ok)
        picks = []
        for dt in rebal:
            for pk in pick_by_segment(s.loc[dt], None, segment):
                picks.append(dict(date=dt, **pk))
        for rule in EXITS if vname == "model" else ("hold_12m",):
            t = simulate(picks, f, basket, index_close, rule)
            key = f"{vname}/{rule}"
            tune = t[pd.to_datetime(t["date"]) <= TUNE_END]
            check = t[pd.to_datetime(t["date"]) > TUNE_END]
            r = dict(all=summarise(t), tune_2017_2021=summarise(tune),
                     check_2022_on=summarise(check),
                     by_segment={g: summarise(x) for g, x in t.groupby("segment")})
            r["passes"] = passes(r["tune_2017_2021"]) and passes(r["check_2022_on"])
            results[key] = r
            if vname == "model":
                t.to_csv(out / f"trades_long_{rule}.csv.gz", index=False)
                y = t.assign(year=pd.to_datetime(t["date"]).dt.year).groupby(["year", "segment"]).agg(
                    trades=("net_return", "size"), win_rate=("net_return", lambda x: (x > 0).mean()),
                    avg_net_return=("net_return", "mean"), avg_segment_return=("segment_return", "mean"),
                    avg_nifty500_return=("nifty500_return", "mean"),
                    avg_excess_segment=("excess_segment", "mean")).reset_index()
                yearly.append(y.assign(exit_rule=rule))
            print(key, json.dumps({k: r[k] for k in ("tune_2017_2021", "check_2022_on", "passes")}), flush=True)

    tune_pick = max(EXITS, key=lambda e: results[f"model/{e}"]["tune_2017_2021"].get("avg_excess_segment", -1))
    summary = dict(test_from=str(rebal[0].date()), last_signal=str(rebal[-1].date()),
                   rebalances=len(rebal), universe_size=int(panel.close.shape[1]),
                   hold_sessions=HOLD, chosen_exit=tune_pick,
                   pass_bar=dict(avg_excess_segment=PASS_EXCESS, years_beating_segment=PASS_YEARS),
                   results=results)
    (out / "backtest_long_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.concat(yearly).round(4).to_csv(out / "backtest_long_by_year.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--test-from", default="2017-01-01")
    ap.add_argument("--out", default="results")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    run(args.start, args.test_from, Path(args.out), args.offline)


if __name__ == "__main__":
    main()
