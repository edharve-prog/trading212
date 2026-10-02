import pytest

from t212bot.config import ConfigError, load_settings


def test_defaults_are_safe_without_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings(tmp_path / "missing.toml", tmp_path / "missing.env")
    assert s.broker.environment == "demo"
    assert s.broker.allow_live is False
    assert s.broker.dry_run is True


def test_example_config_loads(tmp_path):
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "config.example.toml"
    s = load_settings(example, tmp_path / "missing.env")
    assert s.universe.benchmark == "SPY"
    assert s.risk.max_open_positions == 5


def test_credentials_by_environment(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[broker]\nenvironment = "live"\n')
    monkeypatch.setenv("T212_LIVE_API_KEY", "k")
    monkeypatch.setenv("T212_LIVE_API_SECRET", "s")
    assert load_settings(cfg, tmp_path / "x.env").t212_credentials() == ("k", "s")
    monkeypatch.delenv("T212_LIVE_API_SECRET")
    with pytest.raises(ConfigError):
        load_settings(cfg, tmp_path / "x.env").t212_credentials()


def test_rejects_unknown_environment(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[broker]\nenvironment = "paper"\n')
    with pytest.raises(ConfigError):
        load_settings(cfg, tmp_path / "x.env")
