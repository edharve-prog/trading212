# t212bot

A swing-trading bot for the [Trading212 API](https://docs.trading212.com/api). It screens a
watchlist on daily bars and indicators, uses an LLM to rank the setups, and places orders
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
- [x] OpenAI "Sign in with ChatGPT" OAuth client for eligible Codex/OpenAI models
- [ ] Phase 2: screener, risk manager, backtest
- [ ] Phase 3: analyst workflow, scheduling on the Pi, alerts

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

OpenAI OAuth does **not** require an OpenAI API key. It uses OpenAI's open-source
"Sign in with ChatGPT" flow and stores the OAuth credentials outside this repository.

## Usage

```bash
uv run t212bot account                 # demo account summary and open positions
uv run t212bot instruments             # save Trading212 tickers to data/t212_instruments.parquet
uv run t212bot update-data             # fetch daily bars for the configured universe
uv run t212bot update-data AAPL MSFT   # or for specific symbols
uv run t212bot indicators AAPL         # print recent indicator values
```

### OpenAI / Codex via ChatGPT OAuth

Sign in once on the machine running t212bot:

```bash
uv run t212bot llm-login
uv run t212bot llm-status
uv run t212bot llm-models
uv run t212bot llm-test
```

The login flow uses Authorization Code + PKCE with an HTTP loopback callback on
`127.0.0.1`. It validates the OpenAI ID-token signature, issuer, audience, expiry and nonce,
then saves the issued client ID, access token and rotating refresh token under
`~/.config/t212bot/openai_oauth.json` by default. On Unix, the credential file is written
with owner-only permissions.

The client uses only OpenAI's public endpoints:

- `GET https://api.openai.com/v1/models` for the account-specific model catalog.
- `POST https://api.openai.com/v1/responses` for inference, always with `store=false` and
  `stream=true`.

It does not read Codex's private auth files or call ChatGPT private backend endpoints.

To select a model, run `llm-models` and put its slug in `[llm].model`, or pass
`--model <slug>` to `llm-test`. Leaving `model = ""` makes `llm-test` use the first
model returned for the signed-in account.

To sign out and revoke the renewable OAuth session:

```bash
uv run t212bot llm-logout
```

#### Raspberry Pi / headless login

Because the OAuth callback must be `127.0.0.1`, a browser on another machine needs an SSH
port forward. The default callback port is 1455.

From your laptop:

```bash
ssh -L 1455:127.0.0.1:1455 pi@YOUR_PI
```

Then, inside that SSH session:

```bash
uv run t212bot llm-login --no-browser --port 1455
```

Open the printed authorization URL in the laptop browser. The browser callback to
`127.0.0.1:1455` will be forwarded to t212bot on the Pi.

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
  llm/openai_oauth.py     ChatGPT OAuth + OpenAI Responses client
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
