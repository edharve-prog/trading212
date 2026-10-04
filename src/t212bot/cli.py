"""Command line entry point: ``t212bot <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import date
from typing import Any

from .broker.demo_order_check import NotDemoError, run_demo_order_check
from .broker.t212_client import T212Client
from .config import ConfigError, Settings, load_settings
from .data.alpaca import AlpacaProvider
from .data.market_data import MarketData, YFinanceProvider
from .data.store import PriceStore
from .journal import Journal
from .llm.openai_oauth import LLMError, OAuthError, OpenAIOAuthClient

log = logging.getLogger("t212bot")

# Commands not written to the run journal (long-running or read-only noise).
UNJOURNALED = {"dashboard"}


def _journal(settings: Settings) -> Journal:
    return Journal(settings.state_dir / "journal.sqlite")


def _order_journaler(
    settings: Settings, run_id: int | None, environment: str, dry_run: bool
) -> Callable[[str, dict[str, Any], Any], None]:
    journal = _journal(settings)

    def record(kind: str, payload: dict[str, Any], result: Any) -> None:
        journal.record_order(run_id, environment, dry_run, kind, payload, result)

    return record


def _client(settings: Settings, run_id: int | None = None) -> T212Client:
    key, secret = settings.t212_credentials()
    broker = settings.broker
    return T212Client(
        key,
        secret,
        broker.environment,
        allow_live=broker.allow_live,
        dry_run=broker.dry_run,
        on_order=_order_journaler(settings, run_id, broker.environment, broker.dry_run),
    )


def _demo_client(settings: Settings, run_id: int | None = None) -> T212Client:
    """A client that can only reach the demo account and really sends orders."""
    key, secret = settings.t212_demo_credentials()
    return T212Client(
        key,
        secret,
        "demo",
        allow_live=False,
        dry_run=False,
        on_order=_order_journaler(settings, run_id, "demo", False),
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


def cmd_orders(settings: Settings, args: argparse.Namespace) -> int:
    with _client(settings) as client:
        orders = client.orders()
    print(f"Environment: {settings.broker.environment}")
    print(f"Open orders: {len(orders)}")
    for o in orders:
        print("  " + json.dumps(o))
    return 0


def cmd_demo_order_test(settings: Settings, args: argparse.Namespace) -> int:
    if settings.broker.environment != "demo":
        print(
            "Refusing: broker.environment is 'live'. The order test only runs against the "
            'demo account; set environment = "demo" in config.toml first.'
        )
        return 2
    if args.fill:
        plan = f"market BUY then market SELL {args.quantity:g} {args.ticker} (fills on demo)"
    else:
        plan = (
            f"limit BUY {args.quantity:g} {args.ticker} below market, and a limit SELL above "
            "market if you hold it; each is cancelled straight away"
        )
    print(f"Trading212 DEMO account: {plan}.")
    if not args.yes and input("Type 'yes' to continue: ").strip().lower() != "yes":
        print("Cancelled.")
        return 1
    with _demo_client(settings, args.run_id) as client:
        result = run_demo_order_check(
            client,
            args.ticker,
            args.quantity,
            fill=args.fill,
            buy_price=args.buy_price,
            sell_price=args.sell_price,
            offset=args.offset,
            fill_timeout=args.fill_timeout,
        )
    for step in result.steps:
        print(f"[{'PASS' if step.ok else 'FAIL'}] {step.name}: {step.detail}")
    print("All steps passed." if result.ok else "Some steps failed.")
    return 0 if result.ok else 1


def cmd_dashboard(settings: Settings, args: argparse.Namespace) -> int:
    from .dashboard import BindError, DashboardData, serve

    data = DashboardData(settings, lambda: _client(settings), _journal(settings), ttl=args.ttl)
    try:
        serve(data, args.host, args.port, args.allow_host)
    except BindError as exc:
        print(exc)
        return 2
    except KeyboardInterrupt:
        pass
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
    sub.add_parser("orders", help="List open (pending) orders").set_defaults(func=cmd_orders)
    p = sub.add_parser(
        "demo-order-test",
        help="Place and cancel a small buy and sell on the Trading212 DEMO account only",
    )
    p.add_argument("ticker", help="Trading212 ticker, e.g. AAPL_US_EQ")
    p.add_argument("--quantity", type=float, default=1.0, help="Shares per order (default 1)")
    p.add_argument("--buy-price", type=float, help="Limit buy price (default: 10%% below)")
    p.add_argument("--sell-price", type=float, help="Limit sell price (default: 10%% above)")
    p.add_argument(
        "--offset", type=float, default=0.10, help="Distance from current price (default 0.10)"
    )
    p.add_argument(
        "--fill",
        action="store_true",
        help="Market buy then market sell instead (fills on demo; needs the market open)",
    )
    p.add_argument("--fill-timeout", type=float, default=60.0, help="Seconds to wait for a fill")
    p.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt")
    p.set_defaults(func=cmd_demo_order_test)
    p = sub.add_parser("dashboard", help="Serve the read-only web dashboard")
    p.add_argument("--host", default="127.0.0.1", help="Loopback or Tailscale IP")
    p.add_argument("--port", type=int, default=8212)
    p.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="Extra hostname accepted in the Host header (repeatable)",
    )
    p.add_argument("--ttl", type=float, default=30.0, help="Seconds to cache broker data")
    p.set_defaults(func=cmd_dashboard)
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
    journal = None if args.command in UNJOURNALED else _journal(settings)
    environment = "demo" if args.command == "demo-order-test" else settings.broker.environment
    dry_run = False if args.command == "demo-order-test" else settings.broker.dry_run
    args.run_id = None
    if journal is not None:
        try:
            args.run_id = journal.start_run(args.command, environment, dry_run)
        except Exception as exc:  # never let the journal stop a command
            log.warning("could not write run journal: %s", exc)
            journal = None
    code, error = 1, None
    try:
        code = int(args.func(settings, args))
    except (OAuthError, LLMError, ConfigError, NotDemoError) as exc:
        error = str(exc)
        log.error("%s", exc)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if journal is not None and args.run_id is not None:
            try:
                journal.finish_run(args.run_id, code, error)
            except Exception as exc:
                log.warning("could not update run journal: %s", exc)
    return code


if __name__ == "__main__":
    sys.exit(main())
