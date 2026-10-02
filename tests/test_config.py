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
    assert s.llm.provider == "openai_oauth"


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


def test_data_provider_defaults_to_alpaca(tmp_path, monkeypatch):
    s = load_settings(tmp_path / "missing.toml", tmp_path / "missing.env")
    assert s.data.provider == "alpaca"
    assert s.data.alpaca_feed == "sip"
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    with pytest.raises(ConfigError):
        s.alpaca_credentials()
    monkeypatch.setenv("ALPACA_API_KEY_ID", "id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "sk")
    assert s.alpaca_credentials() == ("id", "sk")


def test_rejects_unknown_data_provider(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[data]\nprovider = "bloomberg"\n')
    with pytest.raises(ConfigError):
        load_settings(cfg, tmp_path / "x.env")


def test_llm_config_can_select_openai_oauth(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[llm]\nprovider = "openai_oauth"\nmodel = "gpt-test"\nopenai_callback_port = 1555\n'
    )
    s = load_settings(cfg, tmp_path / "x.env")
    assert s.llm.provider == "openai_oauth"
    assert s.llm.model == "gpt-test"
    assert s.llm.openai_callback_port == 1555


def test_rejects_unknown_llm_provider_and_bad_callback_port(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[llm]\nprovider = "local"\n')
    with pytest.raises(ConfigError):
        load_settings(cfg, tmp_path / "x.env")

    cfg.write_text('[llm]\nopenai_callback_port = 70000\n')
    with pytest.raises(ConfigError):
        load_settings(cfg, tmp_path / "x.env")
