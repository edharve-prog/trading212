"""Settings loaded from config.toml plus secrets from the environment (.env)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

Environment = Literal["demo", "live"]


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class BrokerConfig:
    environment: Environment = "demo"
    allow_live: bool = False
    dry_run: bool = True


@dataclass(frozen=True)
class RiskConfig:
    sleeve_fraction: float = 0.20
    risk_per_trade: float = 0.005
    max_position_fraction: float = 0.25
    max_open_positions: int = 5
    daily_loss_cap: float = 0.02
    drawdown_kill: float = 0.10


@dataclass(frozen=True)
class UniverseConfig:
    symbols: tuple[str, ...] = ()
    benchmark: str = "SPY"
    history_start: str = "2010-01-01"


@dataclass(frozen=True)
class DataConfig:
    provider: Literal["alpaca", "yfinance"] = "alpaca"
    alpaca_feed: Literal["sip", "iex"] = "sip"


@dataclass(frozen=True)
class Settings:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    data_dir: Path = Path("data")
    state_dir: Path = Path("state")

    def t212_credentials(self) -> tuple[str, str]:
        prefix = "T212_LIVE" if self.broker.environment == "live" else "T212_DEMO"
        key = os.environ.get(f"{prefix}_API_KEY", "")
        secret = os.environ.get(f"{prefix}_API_SECRET", "")
        if not key or not secret:
            raise ConfigError(f"{prefix}_API_KEY and {prefix}_API_SECRET must be set in .env")
        return key, secret

    def alpaca_credentials(self) -> tuple[str, str]:
        key = os.environ.get("ALPACA_API_KEY_ID", "")
        secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        if not key or not secret:
            raise ConfigError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set in .env")
        return key, secret


def load_settings(
    config_path: Path | str = "config.toml", env_path: Path | str = ".env"
) -> Settings:
    """Load settings. A missing config.toml gives safe defaults (demo, dry run)."""
    load_dotenv(env_path, override=False)
    path = Path(config_path)
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)

    broker_raw = raw.get("broker", {})
    environment = broker_raw.get("environment", "demo")
    if environment not in ("demo", "live"):
        raise ConfigError(f"broker.environment must be 'demo' or 'live', got {environment!r}")
    broker = BrokerConfig(
        environment=environment,
        allow_live=bool(broker_raw.get("allow_live", False)),
        dry_run=bool(broker_raw.get("dry_run", True)),
    )

    universe_raw = raw.get("universe", {})
    universe = UniverseConfig(
        symbols=tuple(universe_raw.get("symbols", ())),
        benchmark=universe_raw.get("benchmark", "SPY"),
        history_start=universe_raw.get("history_start", "2010-01-01"),
    )

    data_raw = raw.get("data", {})
    provider = data_raw.get("provider", "alpaca")
    if provider not in ("alpaca", "yfinance"):
        raise ConfigError(f"data.provider must be 'alpaca' or 'yfinance', got {provider!r}")
    feed = data_raw.get("alpaca_feed", "sip")
    if feed not in ("sip", "iex"):
        raise ConfigError(f"data.alpaca_feed must be 'sip' or 'iex', got {feed!r}")
    data = DataConfig(provider=provider, alpaca_feed=feed)

    risk = RiskConfig(**raw.get("risk", {}))
    paths = raw.get("paths", {})
    base = path.parent if path.exists() else Path(".")
    return Settings(
        broker=broker,
        risk=risk,
        universe=universe,
        data=data,
        data_dir=base / paths.get("data_dir", "data"),
        state_dir=base / paths.get("state_dir", "state"),
    )
