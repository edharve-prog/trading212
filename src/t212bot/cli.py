"""Command line entry point: ``t212bot <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from .broker.t212_client import T212Client
from .config import Settings, load_settings
from .data.alpaca import AlpacaProvider
from .data.market_data import MarketData, YFinanceProvider
from .data.store import PriceStore
from .llm.openai_oauth import LLMError, OAuthError, OpenAIOAuthClient

log = logging.getLogger("t212bot")


def _client(settings: Settings) -> T212Client:
    key, secret = settings.t212_credentials()
    return T212Client(
        key,
        secret,
        settings.broker.environment,
        allow_live=settings.broker.allow_live,
        dry_run=settings.broker.dry_run,
    )


def _provider(settings: Settings) -> MarketData:
    if settings.data.provider == "yfinance":
        return YFinanceProvider()
    key, secret = settings.alpaca_credentials()
    return AlpacaProvider(key, secret, feed=settings.data.alpaca_feed)


def _openai(settings: Settings) -> OpenAIOAuthClient:
    return OpenAIOAuthClient(credential_path=settings.llm.openai_credential_path)


def cmd_account(settings: Settings, args: argparse.Namespace) -> int:
    with _client(settings) as client:
        summary = client.account_summary()
        positions = client.positions()
    print(f"Environment: {settings.broker.environment}")
    print(json.dumps(summary, indent=2))
    print(f"Open positions: {len(positions)}")
    for p in positions:
        print(f"  {p.get('ticker', p)}")
    return 0


def cmd_instruments(settings: Settings, args: argparse.Namespace) -> int:
    import pandas as pd

    with _client(settings) as client:
        instruments = client.instruments()
    out = settings.data_dir / "t212_instruments.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instruments).to_parquet(out, index=False)
    print(f"Saved {len(instruments)} instruments to {out}")
    return 0


def cmd_update_data(settings: Settings, args: argparse.Namespace) -> int:
    symbols = args.symbols or [*settings.universe.symbols, settings.universe.benchmark]
    store = PriceStore(settings.data_dir)
    provider = _provider(settings)
    start = date.fromisoformat(settings.universe.history_start)
    failures = 0
    for symbol in symbols:
        try:
            n = store.update(provider, symbol, start)
            print(f"{symbol}: {n} new bars")
        except Exception as exc:  # one bad symbol should not stop the run
            failures += 1
            log.error("%s: update failed: %s", symbol, exc)
    return 1 if failures else 0


def cmd_indicators(settings: Settings, args: argparse.Namespace) -> int:
    from .indicators import add_indicators

    store = PriceStore(settings.data_dir)
    bars = store.read_bars([args.symbol])
    if bars.empty:
        print(f"No stored bars for {args.symbol}; run update-data first.")
        return 1
    bench = store.read_bars([settings.universe.benchmark])
    bench_close = bench.set_index("date")["close"] if not bench.empty else None
    df = add_indicators(bars.reset_index(drop=True), bench_close)
    cols = ["date", "close", "sma_50", "sma_200", "rsi_14", "macd_hist", "atr_14", "adx"]
    if "rs_63" in df:
        cols.append("rs_63")
    print(df[cols].tail(args.rows).to_string(index=False))
    return 0


def cmd_llm_login(settings: Settings, args: argparse.Namespace) -> int:
    port = args.port or settings.llm.openai_callback_port
    with _openai(settings) as client:
        status = client.login(
            callback_port=port,
            open_browser=not args.no_browser,
            url_callback=(lambda url: print(f"Open this URL in a browser:\n{url}"))
            if args.no_browser
            else None,
        )
    account = status.email or status.client_id or "ChatGPT account"
    print(f"Signed in: {account}")
    if status.plan_enabled:
        print("ChatGPT plan usage: enabled")
        return 0
    print("ChatGPT plan usage: not authorized")
    return 1


def cmd_llm_status(settings: Settings, args: argparse.Namespace) -> int:
    with _openai(settings) as client:
        status = client.status()
    if not status.signed_in:
        print("OpenAI OAuth: signed out")
        return 1
    print(f"OpenAI OAuth: signed in as {status.email or status.client_id}")
    print(f"ChatGPT plan usage: {'enabled' if status.plan_enabled else 'disabled'}")
    if status.expires_at:
        print(f"Access token expires: {status.expires_at.isoformat()}")
    return 0


def cmd_llm_models(settings: Settings, args: argparse.Namespace) -> int:
    with _openai(settings) as client:
        models = client.list_models()
    for model in models:
        print(f"{model.slug}\t{model.display_name}")
    return 0


def cmd_llm_test(settings: Settings, args: argparse.Namespace) -> int:
    with _openai(settings) as client:
        model = args.model or settings.llm.model
        if not model:
            models = client.list_models()
            if not models:
                raise LLMError("No models are available for the signed-in ChatGPT account")
            model = models[0].slug
        output = client.respond(args.prompt, model=model)
    print(output)
    return 0


def cmd_llm_logout(settings: Settings, args: argparse.Namespace) -> int:
    with _openai(settings) as client:
        revoked = client.logout()
    print("Signed out locally.")
    if not revoked:
        print(
            "Remote token revocation could not be confirmed; "
            "disconnect the app in ChatGPT Settings."
        )
        return 1
    print("Remote OAuth session revoked.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="t212bot")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("account", help="Show account summary and positions").set_defaults(
        func=cmd_account
    )
    sub.add_parser("instruments", help="Save the Trading212 instrument list").set_defaults(
        func=cmd_instruments
    )
    p = sub.add_parser("update-data", help="Fetch new daily bars into the price store")
    p.add_argument("symbols", nargs="*", help="Defaults to the configured universe + benchmark")
    p.set_defaults(func=cmd_update_data)
    p = sub.add_parser("indicators", help="Print recent indicator values for one symbol")
    p.add_argument("symbol")
    p.add_argument("--rows", type=int, default=10)
    p.set_defaults(func=cmd_indicators)

    p = sub.add_parser("llm-login", help="Sign in with ChatGPT for OpenAI/Codex model access")
    p.add_argument("--port", type=int, help="Loopback callback port (default from config)")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the authorization URL instead of opening a browser",
    )
    p.set_defaults(func=cmd_llm_login)
    sub.add_parser("llm-status", help="Show OpenAI OAuth status").set_defaults(func=cmd_llm_status)
    sub.add_parser("llm-models", help="List models available to the ChatGPT account").set_defaults(
        func=cmd_llm_models
    )
    p = sub.add_parser("llm-test", help="Run a small streamed Responses API request")
    p.add_argument("--model", help="Model slug; defaults to [llm].model or the first listed model")
    p.add_argument("--prompt", default="Reply with exactly: OK")
    p.set_defaults(func=cmd_llm_test)
    sub.add_parser("llm-logout", help="Revoke and clear the OpenAI OAuth session").set_defaults(
        func=cmd_llm_logout
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config, args.env_file)
    try:
        return int(args.func(settings, args))
    except (OAuthError, LLMError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
