"""Command line entry point: ``t212bot <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from .broker.t212_client import T212Client
from .config import Settings, load_settings
from .data.market_data import YFinanceProvider
from .data.store import PriceStore

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
    provider = YFinanceProvider()
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config, args.env_file)
    return int(args.func(settings, args))


if __name__ == "__main__":
    sys.exit(main())
