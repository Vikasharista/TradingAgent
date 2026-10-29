"""Backtest of the three horizon scores.

The weights come straight from the spec and are not fitted to any data, so
every calendar year in the test is out of sample. Results are reported per
year so we can see whether the edge holds across different markets.

Run:  python -m agent.backtest --start 2017-01-01 --test-from 2019-01-01
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import regime as rg
from . import scoring as sc
from .data import INDIA_VIX, NIFTY500, Panel, load_index, load_prices
from .universe import load_universe

OUT = Path("results")
COMMISSION = 0.001          # brokerage + STT + charges, per side
SLIPPAGE = {"large": 0.001, "mid": 0.001, "small": 0.0025, "micro": 0.0025}


@dataclass(frozen=True)
class Horizon:
    name: str
    hold: int          # sessions before the pick expires
    reward: float | None  # full target as a multiple of risk (None: no full target)
    entry: str         # "buy_stop" above the signal day's high, or "open"
    # Partial profit booking: sell book_frac at book_at x risk, move the stop
    # to break-even, then let the rest run until a close below the 50 DMA,
    # the stop, or the end of the holding period.
    book_at: float | None = None
    book_frac: float = 0.0
    stop_atr: float | None = None   # stop this many ATRs below entry
    regime_filter: str = "spec"     # "spec": count + score floor; "skip_risk_off": top N unless risk-off


HORIZONS = [
    Horizon("next_day", 1, 1.5, "buy_stop"),
    Horizon("next_week", 5, 2.0, "buy_stop"),
    # Tuned on 2019-2022 only, then checked on 2023-2026 (see README).
    Horizon("next_month", 60, None, "open", book_at=1.5, book_frac=0.5,
            stop_atr=3.0, regime_filter="skip_risk_off"),
]
PASS_AVG_R = 0.2
PASS_YEARS_BEATING = 0.6


def pick(score: pd.DataFrame, ok: pd.DataFrame, regime: pd.DataFrame | None,
         default_n: int = 10) -> list[tuple[pd.Timestamp, str, float]]:
    """Top picks per date, honouring the regime's count and score floor."""
    s = score.where(ok)
    out = []
    for dt, row in s.iterrows():
        if regime is not None:
            n, floor = regime.at[dt, "max_picks"], regime.at[dt, "min_score"]
        else:
            n, floor = default_n, 0
        row = row[row >= floor].nlargest(n)
        out.extend((dt, sym, val) for sym, val in row.items())
    return out


def simulate(h: Horizon, picks, f: sc.Features, segment: pd.Series,
             index_close: pd.Series) -> pd.DataFrame:
    p = f.p
    dates = p.close.index
    pos = {d: i for i, d in enumerate(dates)}
    col = {s: j for j, s in enumerate(p.close.columns)}
    O, H, L, C = (x.to_numpy() for x in (p.open, p.high, p.low, p.close))
    ATR, SMA50, SWING = f.atr14.to_numpy(), f.sma50.to_numpy(), f.swing_low10.to_numpy()
    idx = index_close.reindex(dates, method="ffill").to_numpy()
    n_dates = len(dates)

    rows = []
    for dt, sym, score in picks:
        t, j = pos[dt], col[sym]
        if t + 1 >= n_dates:
            continue
        atr = ATR[t, j]
        if not np.isfinite(atr) or atr <= 0:
            continue
        e = t + 1
        if h.entry == "buy_stop":
            trigger = H[t, j] * 1.001
            if not (H[e, j] >= trigger):
                rows.append(dict(date=dt, symbol=sym, score=score, triggered=False))
                continue
            fill = max(O[e, j], trigger)
        else:
            fill = O[e, j]
        if not np.isfinite(fill):
            continue

        if h.stop_atr is not None:
            stop = fill - h.stop_atr * atr
        elif h.name == "next_day":
            stop = fill - atr
        elif h.name == "next_week":
            stop = max(fill - 1.5 * atr, SWING[t, j] * 0.995)
        else:
            stop = max(SMA50[t, j], fill * 0.92) if np.isfinite(SMA50[t, j]) else fill * 0.92
        stop = min(stop, fill - 0.5 * atr)   # never a stop tighter than half an ATR
        risk = fill - stop
        target = fill + h.reward * risk if h.reward else np.inf
        book_px = fill + h.book_at * risk if h.book_at else np.inf

        # gross return is built up from the parts sold: rem is the share still held
        gross, rem, booked, outcome, k = 0.0, 1.0, False, "expired", e
        last = min(e + h.hold - 1, n_dates - 1)
        for k in range(e, last + 1):
            o, hi, lo, c = O[k, j], H[k, j], L[k, j], C[k, j]
            if not np.isfinite(lo):
                continue
            if k > e and o <= stop:          # gapped through the stop
                gross, rem, outcome = gross + rem * (o / fill - 1), 0.0, "stop"
                break
            if k > e and o >= target:        # gapped through the target
                gross, rem, outcome = gross + rem * (o / fill - 1), 0.0, "target"
                break
            if lo <= stop:                   # both touched in one bar: assume the stop
                gross, rem, outcome = gross + rem * (stop / fill - 1), 0.0, "stop"
                break
            if not booked and hi >= book_px:
                px = max(o, book_px) if k > e else book_px
                gross, rem, booked = gross + h.book_frac * (px / fill - 1), rem - h.book_frac, True
                stop = max(stop, fill)
            if hi >= target:
                gross, rem, outcome = gross + rem * (target / fill - 1), 0.0, "target"
                break
            if booked and k < last and np.isfinite(SMA50[k, j]) and c < SMA50[k, j]:
                k += 1                       # trail: sell the rest at the next open
                px = O[k, j] if np.isfinite(O[k, j]) else c
                gross, rem, outcome = gross + rem * (px / fill - 1), 0.0, "trail"
                break
        if rem > 0:
            k = last
            if not np.isfinite(C[k, j]):
                continue
            gross += rem * (C[k, j] / fill - 1)
        if booked and outcome == "stop":
            outcome = "breakeven_after_booking"
        exit_px = fill * (1 + gross)         # blended exit price of the whole position

        seg = segment.get(sym, "small")
        cost = 2 * (COMMISSION + SLIPPAGE.get(seg, 0.0025))
        net = gross - cost
        bench = idx[k] / idx[t] - 1 if np.isfinite(idx[t]) and np.isfinite(idx[k]) else np.nan
        rows.append(dict(
            date=dt, symbol=sym, score=round(float(score), 1), triggered=True,
            segment=seg, entry=fill, stop=fill - risk,
            target=target if np.isfinite(target) else book_px, exit=exit_px,
            exit_date=dates[k], outcome=outcome, booked=booked or outcome == "target",
            net_return=net, r_multiple=net * fill / risk,
            nifty500_return=bench, excess_return=net - bench,
        ))
    return pd.DataFrame(rows)


def summarise(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    t = trades[trades["triggered"]].copy()
    t["booked"] = t["booked"].astype(bool)
    t["year"] = pd.to_datetime(t["date"]).dt.year
    by_year = t.groupby("year").agg(
        trades=("net_return", "size"),
        hit_rate=("booked", "mean"),
        win_rate=("net_return", lambda s: (s > 0).mean()),
        avg_r=("r_multiple", "mean"),
        avg_net_return=("net_return", "mean"),
        avg_nifty500_return=("nifty500_return", "mean"),
        avg_excess_return=("excess_return", "mean"),
    )
    cum_r = t.sort_values("exit_date")["r_multiple"].cumsum()
    max_dd_r = float((cum_r.cummax() - cum_r).max()) if len(cum_r) else 0.0
    years_beating = float((by_year["avg_excess_return"] > 0).mean()) if len(by_year) else 0.0
    overall = dict(
        signals=int(len(trades)),
        trades=int(len(t)),
        trigger_rate=float(trades["triggered"].mean()) if len(trades) else 0.0,
        hit_rate=float(t["booked"].mean()) if len(t) else 0.0,
        win_rate=float((t["net_return"] > 0).mean()) if len(t) else 0.0,
        avg_r=float(t["r_multiple"].mean()) if len(t) else 0.0,
        avg_net_return=float(t["net_return"].mean()) if len(t) else 0.0,
        avg_excess_return=float(t["excess_return"].mean()) if len(t) else 0.0,
        years_beating_nifty500=years_beating,
        max_drawdown_r=max_dd_r,
    )
    overall["passes"] = bool(overall["avg_r"] >= PASS_AVG_R
                             and years_beating >= PASS_YEARS_BEATING)
    return by_year, overall


def run(panel: Panel, uni: pd.DataFrame, index_close: pd.Series,
        vix: pd.Series | None, test_from: str,
        delivery_pct: pd.DataFrame | None = None,
        t2t: pd.DataFrame | None = None) -> dict:
    # Stocks outside the index lists (the "all" universe) count as small caps
    # in their own unclassified industry.
    segment = uni.set_index("symbol")["segment"].reindex(panel.symbols).fillna("small")
    industry = uni.set_index("symbol")["industry"].reindex(panel.symbols).fillna("Unclassified")
    f = sc.build_features(panel)
    ok = sc.eligible(f, segment)
    if t2t is not None:
        ok &= ~t2t.reindex_like(ok).fillna(False).astype(bool)
    regime = rg.market_regime(index_close, panel.close, vix)

    scores = {
        "next_day": sc.score_day(f, delivery_pct)[0],
        "next_week": sc.score_week(f)[0],
        "next_month": sc.score_month(f, industry)[0],
    }
    test_dates = panel.close.index[panel.close.index >= pd.Timestamp(test_from)]
    # The last signal date must leave room for the longest holding period.
    room = max(h.hold for h in HORIZONS) + 1
    test_dates = test_dates[:-room] if len(test_dates) > room else test_dates

    OUT.mkdir(exist_ok=True)
    results, yearly = {}, []
    for h in HORIZONS:
        s = scores[h.name].loc[test_dates]
        for variant, reg in (("with_regime", regime), ("always_top10", None)):
            okv = ok.loc[test_dates]
            if reg is not None and h.regime_filter == "skip_risk_off":
                okv = okv & reg.loc[test_dates, "regime"].ne("risk_off").to_numpy()[:, None]
                reg = None
            picks = pick(s, okv, reg)
            trades = simulate(h, picks, f, segment, index_close)
            if trades.empty or not trades["triggered"].any():
                results[f"{h.name}/{variant}"] = {"signals": len(trades), "trades": 0, "passes": False}
                continue
            by_year, overall = summarise(trades)
            results[f"{h.name}/{variant}"] = overall
            yearly.append(by_year.assign(horizon=h.name, variant=variant).reset_index())
            if variant == "with_regime":
                trades.to_csv(OUT / f"trades_{h.name}.csv.gz", index=False)
            print(h.name, variant, json.dumps(overall), flush=True)

    regime_share = regime.loc[test_dates, "regime"].value_counts(normalize=True).round(3).to_dict()
    summary = dict(
        test_from=str(test_dates[0].date()), test_to=str(test_dates[-1].date()),
        universe_size=int(panel.close.shape[1]), regime_share=regime_share,
        delivery_data=delivery_pct is not None,
        pass_bar=dict(avg_r=PASS_AVG_R, years_beating_nifty500=PASS_YEARS_BEATING),
        results=results,
    )
    (OUT / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if yearly:
        pd.concat(yearly).round(4).to_csv(OUT / "backtest_by_year.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01", help="first date of price history")
    ap.add_argument("--test-from", default="2019-01-01", help="first signal date tested")
    ap.add_argument("--universe", choices=["index", "all"], default="index",
                    help="index: today's Nifty 500 + Microcap 250 members; all: every "
                         "NSE-listed stock that passes the liquidity filter on each day "
                         "(no survivorship bias)")
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--offline", action="store_true", help="use cached NSE files only")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    global OUT
    OUT = Path(args.out)
    uni = load_universe(refresh=args.refresh_universe)
    symbols = uni["symbol"].tolist() if args.universe == "index" else None
    panel, delivery, t2t = load_prices(symbols, start=args.start, refresh=not args.offline)
    index_close = load_index(NIFTY500, start=args.start)["Close"]
    vix = load_index(INDIA_VIX, start=args.start)["Close"]
    run(panel, uni, index_close, vix, args.test_from, delivery, t2t)


if __name__ == "__main__":
    main()
