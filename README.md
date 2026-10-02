# t212bot

A swing-trading bot for the [Trading212 API](https://docs.trading212.com/api). It screens a
watchlist on daily bars and indicators, asks Claude to rank the setups, and places orders
through Trading212 under hard risk limits. Developed on Windows, run on a Raspberry Pi.

It runs against the Trading212 **demo (practice) account** by default, in **dry-run** mode,
and refuses to place live orders unless `allow_live = true` is set. No strategy here has a
proven edge. Backtest and paper-trade before risking money.

## Status

Phase 1 (foundations) of the [implementation plan](https://claude.ai/code/artifact/817c0264-b8d0-408c-a974-0b785479aeb5):

- [x] Project layout, config, CI
- [x] Trading212 client: account, positions, orders, metadata, rate-limit pacing and 429 retry
- [x] Daily price store: Parquet per symbol per year, queried with DuckDB
- [x] Indicators: SMA, EMA, RSI, MACD, ATR, Bollinger, ADX, ROC, volume ratio, 52-week high, relative strength
- [ ] Phase 2: screener, risk manager, backtest
- [ ] Phase 3: Claude analyst, scheduling on the Pi, alerts

## Setup (Windows or Raspberry Pi)

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/edharve-prog/trading212.git
cd trading212
uv sync
cp .env.example .env               # Windows: copy .env.example .env
cp config.example.toml config.toml
```

Put your Trading212 **demo** API key and secret in `.env` (Trading212 app: Settings > API),
and your Alpaca API key ID and secret for price data. To use Yahoo Finance instead, set
`provider = "yfinance"` under `[data]` in `config.toml`.
On the Pi, run `chmod 600 .env`.

## Usage

```bash
uv run t212bot account                 # demo account summary and open positions
uv run t212bot instruments             # save Trading212 tickers to data/t212_instruments.parquet
uv run t212bot update-data             # fetch daily bars for the configured universe
uv run t212bot update-data AAPL MSFT   # or for specific symbols
uv run t212bot indicators AAPL         # print recent indicator values
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

## Layout

```
src/t212bot/
  broker/t212_client.py   Trading212 REST client
  data/market_data.py     price provider interface + yfinance
  data/alpaca.py          Alpaca market data provider (default)
  data/store.py           Parquet + DuckDB price store
  indicators.py           technical indicators
  config.py               config.toml + .env loading
  cli.py                  command line
tests/                    unit tests (no network)
```

## Things to confirm on the demo account

The Trading212 API is in beta and some details are not on the overview pages:

- Exact rate limits for order placement (paced conservatively in `MIN_INTERVAL`).
- Whether a resting stop and a limit sell can both be placed on the same shares.
- Fractional quantities per instrument.
