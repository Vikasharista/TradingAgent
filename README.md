# TradingAgent

A nightly agent that scores Indian stocks (Nifty 500 plus the Nifty Microcap 250)
and recommends the ones most likely to rise over the next day, week and month.
It makes recommendations only and never places trades.

The rules are set out in the [strategy and scoring spec](https://claude.ai/artifact/C36EnYQJhZy8yGp3UqtAbV).

## Layout

| Path | What it does |
| --- | --- |
| `agent/universe.py` | Downloads the index constituent lists from NSE |
| `agent/nse.py` | Downloads and caches NSE's daily archive files: bhavcopy, delivery %, index closes |
| `agent/fundamentals.py` | Quarterly results, annual ROE and debt, and shareholding from screener.in, scored point in time |
| `agent/data.py` | Turns the NSE files into split-adjusted date x symbol panels |
| `agent/indicators.py` | EMAs, RSI, ATR, MACD and the 0-100 scoring helpers |
| `agent/scoring.py` | The next-day, next-week and next-month scores |
| `agent/regime.py` | Market regime: how many picks to show and the minimum score |
| `agent/backtest.py` | Simulates every pick with entry, stop, target and costs |
| `agent/selection.py` | Splits a day's scores into 10 large, 10 mid and 10 small caps |
| `agent/longterm.py` | The long-term (1 to 3 year) score from annual accounts and the 200 DMA trend |
| `agent/backtest_long.py` | Backtest of the long-term list: monthly picks held up to 12 months |
| `agent/dashboard.py` | Builds the dashboard's data files for the nightly refresh |
| `results/` | Backtest output, written by the Backtest workflow |

## Running the backtest

The **CI** workflow runs the tests on every pull request and push to main. The
**Backtest** workflow (about an hour, it downloads years of NSE files) runs only
when started by hand from the Actions tab. Locally:

```
pip install -r requirements.txt
python -m agent.backtest --start 2017-01-01 --test-from 2019-01-01
# every NSE-listed stock that passes the liquidity filter, no survivorship bias
python -m agent.backtest --universe all --out results/all
```

All data comes from `archives.nseindia.com`: one equity bhavcopy, one delivery
(MTO) file and one index-close file per trading day, cached in `data/cache/nse/`.
The first run downloads about ten years of files (roughly 30 minutes); later runs
only fetch new days. Prices are back-adjusted for splits, bonuses and rights using
NSE's restated previous close on each ex-date. Dividends are not adjusted.

## Lists by company size

Both lists show 10 large caps (Nifty 100), 10 mid caps (Midcap 150) and 10
small caps (Smallcap 250). Microcaps are scored for market breadth but never
listed.

- **Mid term (1 to 3 months)** is the next-month score below. Stocks that pass
  the trend template come first; when fewer than 10 of a size pass, the gap is
  filled with that size's next best scores, marked `filled` in the data.
- **Long term (1 to 3 years)** is `agent/longterm.py`: 60 points from the annual
  accounts (3-year ROCE or ROE and debt for quality, 3-year sales and profit
  growth, operating cash flow against profit) and 40 from the chart (above a
  rising 200 DMA, 12-month momentum, nearness to the 52-week high). Loss-making
  companies are left out. It is shown whatever the market regime. Valuation is
  shown but not scored, because historical P/E needs share counts the free pages
  do not give.

```
python -m agent.backtest_long --start 2016-01-01 --test-from 2017-01-01
```

buys the long-term list on the first session of each month and holds each
stock up to 12 months, comparing it with an equal-weight basket of every
eligible stock of the same size. The exit rule is chosen on 2017-2021 signals
and checked on 2022 onwards. NSE's archive has no bhavcopies before 2016.

## Next-month exit rule (partial profit booking)

The next-month horizon books half the position at 1.5x the risk, moves the stop
to break-even, and lets the rest run until a close below the 50 DMA, the stop,
or 60 sessions. The initial stop is 3 ATR below entry. Picks are the top 10
scores on any day the market regime is not risk-off.

These settings were chosen on 2019-2022 data only and then checked on
2023-2026, which the tuning never saw (avg +0.40R in the tuning years, +0.32R in
the unseen years).

## Fundamentals

`--fundamentals` adds the spec's 20-point fundamentals pillar to the next-month
score: EPS growth and acceleration, sales growth, ROE and debt/equity (banks and
NBFCs skip debt/equity), and promoter and FII/DII holding changes. Pages come
from screener.in (paths its robots.txt allows, one request every 1.5 seconds,
cached for a week). A quarter's numbers count only from 45 days after it ends
(60 for March), when the results have to be public. Screener pages hold about 13
quarters, so year-on-year growth exists from about August 2024; before that the
technical signals stand alone.

## Known limits of the backtest

- `--universe index` uses today's index members, so stocks that were dropped or
  delisted are missing. That flatters the results (survivorship bias).
  `--universe all` avoids this by taking every listed stock that passed the
  liquidity filter on each day, including ones delisted since.
- Fundamentals only reach back to about August 2024, and FII flows are not
  used. Delivery % is used.
- When a stop and a target are both touched in the same session, the stop is
  assumed to have been hit first.

This is a probability tool, not investment advice.
