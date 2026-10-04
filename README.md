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
uv run t212bot orders                  # open (pending) orders
uv run t212bot instruments             # save Trading212 tickers to data/t212_instruments.parquet
uv run t212bot update-data             # fetch daily bars for the configured universe
uv run t212bot update-data AAPL MSFT   # or for specific symbols
uv run t212bot indicators AAPL         # print recent indicator values
```

Every command except `dashboard` is recorded in `state/journal.sqlite`, along with each
order the bot places or cancels (dry-run ones included).

### Testing buy and sell orders on the demo account

CI covers order placement with a mocked Trading212 API. To check it against the real
**demo** account, run:

```bash
uv run t212bot demo-order-test AAPL_US_EQ
```

It asks for confirmation, then places a limit buy 10% below the current price and, if you
hold at least `--quantity` shares, a limit sell 10% above. Each order is checked in the
open-orders list and cancelled straight away, so nothing should fill. Each step prints
PASS or FAIL.

- It always uses `T212_DEMO_API_KEY` / `T212_DEMO_API_SECRET` and the demo URL, and refuses
  to run if `broker.environment = "live"`. It sends real demo orders even when
  `dry_run = true`.
- If you do not hold the ticker on demo, pass `--buy-price`; the sell step is reported as
  skipped (and the run as failed) until you hold some.
- `--fill` instead sends a market buy, waits for it to fill, then a market sell of the same
  size. It needs the market to be open; an order that does not fill within
  `--fill-timeout` seconds is cancelled.

### Using it on the Raspberry Pi

**Command line over Tailscale.** With Tailscale on both machines, from Windows:

```bash
ssh pi@YOUR_PI_TAILSCALE_NAME
cd trading212 && uv run t212bot account
```

**Read-only web dashboard.** Shows the demo/live and dry-run mode, account summary,
positions, open orders, recent runs and the orders the bot placed. It has no buttons and
cannot place or cancel anything; API keys stay on the Pi.

```bash
uv run t212bot dashboard               # http://127.0.0.1:8212 on the Pi
```

To open it from Windows, either publish it on your tailnet only (HTTPS, recommended):

```bash
sudo tailscale serve --bg 8212         # then browse https://YOUR_PI.YOUR_TAILNET.ts.net
```

or bind it to the Pi's Tailscale address and browse `http://YOUR_PI:8212`:

```bash
uv run t212bot dashboard --host "$(tailscale ip -4)"
```

It refuses to bind anything other than loopback or a Tailscale (100.64.0.0/10) address, so
it is never exposed on your home LAN, and it only answers requests whose Host header is
this machine, `localhost` or a `*.ts.net` name (add others with `--allow-host`). Broker
data is cached for 30 seconds to stay inside Trading212's rate limits.

To run it at boot, see [`deploy/t212bot-dashboard.service`](deploy/t212bot-dashboard.service).

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
  broker/demo_order_check.py  place-and-cancel check against the demo account
  data/market_data.py     price provider interface + yfinance
  data/alpaca.py          Alpaca market data provider (default)
  data/store.py           Parquet + DuckDB price store
  llm/openai_oauth.py     ChatGPT OAuth + OpenAI Responses client
  indicators.py           technical indicators
  config.py               config.toml + .env loading
  journal.py              SQLite journal of runs and orders (state/journal.sqlite)
  dashboard.py            read-only local web dashboard
  cli.py                  command line
deploy/                   systemd unit for the dashboard
tests/                    unit tests (no network)
```

## Things to confirm on the demo account

The Trading212 API is in beta and some details are not on the overview pages:

- Exact rate limits for order placement (paced conservatively in `MIN_INTERVAL`).
- Whether a resting stop and a limit sell can both be placed on the same shares.
- Fractional quantities per instrument.
