# TradingAgent

A nightly agent that scores Indian stocks (Nifty 500 plus the Nifty Microcap 250)
and recommends the ones most likely to rise over the next day, week and month.
It makes recommendations only and never places trades.

The rules are set out in the [strategy and scoring spec](https://claude.ai/artifact/C36EnYQJhZy8yGp3UqtAbV).

## Layout

| Path | What it does |
| --- | --- |
| `agent/universe.py` | Downloads the index constituent lists from NSE |
| `agent/data.py` | Daily prices from Yahoo Finance as date x symbol panels |
| `agent/indicators.py` | EMAs, RSI, ATR, MACD and the 0-100 scoring helpers |
| `agent/scoring.py` | The next-day, next-week and next-month scores |
| `agent/regime.py` | Market regime: how many picks to show and the minimum score |
| `agent/backtest.py` | Simulates every pick with entry, stop, target and costs |
| `results/` | Backtest output, written by the Backtest workflow |

## Running the backtest

The **Backtest** GitHub Actions workflow runs on every push to a `claude/` branch
and can be started by hand from the Actions tab. Locally:

```
pip install -r requirements.txt
python -m agent.backtest --start 2017-01-01 --test-from 2019-01-01
```

## Known limits of the backtest

- It uses today's index members, so stocks that were dropped or delisted are
  missing. That flatters the results (survivorship bias).
- Delivery %, fundamentals and FII flows have no free point-in-time history, so
  their weights are spread over the other signals in the backtest.
- When a stop and a target are both touched in the same session, the stop is
  assumed to have been hit first.

This is a probability tool, not investment advice.
