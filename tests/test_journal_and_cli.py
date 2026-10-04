import httpx
import pytest

from t212bot import cli
from t212bot.broker.t212_client import T212Client
from t212bot.journal import Journal

from .fake_t212 import FakeT212


def test_order_listener_sees_dry_run_orders_and_cancels():
    seen = []
    client = T212Client(
        "k",
        "s",
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        dry_run=True,
        on_order=lambda kind, payload, result: seen.append((kind, payload)),
    )
    client.place_market_order("AAPL_US_EQ", 1)
    client.cancel_order(7)
    assert seen == [("market", {"ticker": "AAPL_US_EQ", "quantity": 1}), ("cancel", {"orderId": 7})]


def test_failing_listener_does_not_break_orders():
    def boom(*args):
        raise RuntimeError("disk full")

    client = T212Client(
        "k",
        "s",
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        dry_run=True,
        on_order=boom,
    )
    assert client.place_market_order("AAPL_US_EQ", 1).kind == "market"


def test_journal_round_trip(tmp_path):
    journal = Journal(tmp_path / "state" / "journal.sqlite")
    assert journal.recent_runs() == []
    run_id = journal.start_run("account", "demo", True)
    journal.record_order(run_id, "demo", False, "limit", {"ticker": "X", "quantity": 1}, {"id": 9})
    journal.record_order(run_id, "demo", False, "cancel", {"orderId": 9}, None)
    journal.finish_run(run_id, 0)
    [run] = journal.recent_runs()
    assert run["command"] == "account" and run["exit_code"] == 0 and run["dry_run"] == 1
    orders = journal.recent_orders()
    assert [(o["kind"], o["order_id"]) for o in orders] == [("cancel", "9"), ("limit", "9")]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("T212_DEMO_API_KEY", "dk")
    monkeypatch.setenv("T212_DEMO_API_SECRET", "ds")
    return tmp_path


def patch_transport(monkeypatch, fake):
    real_init = T212Client.__init__

    def init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        kwargs["sleep"] = lambda s: None
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(T212Client, "__init__", init)


def test_demo_order_test_command_sends_to_demo_and_journals(workdir, monkeypatch, capsys):
    fake = FakeT212(positions=[{"ticker": "AAPL_US_EQ", "quantity": 2, "currentPrice": 100.0}])
    patch_transport(monkeypatch, fake)
    # Default config is dry_run = true; the order test must still really send to demo.
    assert cli.main(["demo-order-test", "AAPL_US_EQ", "--yes"]) == 0
    assert "All steps passed." in capsys.readouterr().out
    assert fake.hosts == {"demo.trading212.com"} and len(fake.placed) == 2
    journal = Journal(workdir / "state" / "journal.sqlite")
    [run] = journal.recent_runs()
    assert (run["command"], run["environment"], run["dry_run"], run["exit_code"]) == (
        "demo-order-test",
        "demo",
        0,
        0,
    )
    assert [o["kind"] for o in journal.recent_orders()] == ["cancel", "limit", "cancel", "limit"]


def test_demo_order_test_refuses_live_config(workdir, monkeypatch, capsys):
    (workdir / "config.toml").write_text('[broker]\nenvironment = "live"\nallow_live = true\n')
    fake = FakeT212()
    patch_transport(monkeypatch, fake)
    assert cli.main(["demo-order-test", "AAPL_US_EQ", "--yes"]) == 2
    assert "Refusing" in capsys.readouterr().out
    assert fake.hosts == set()


def test_demo_order_test_needs_confirmation(workdir, monkeypatch):
    fake = FakeT212()
    patch_transport(monkeypatch, fake)
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert cli.main(["demo-order-test", "AAPL_US_EQ"]) == 1
    assert fake.hosts == set()


def test_missing_credentials_are_reported_and_journaled(workdir, monkeypatch):
    monkeypatch.delenv("T212_DEMO_API_KEY")
    assert cli.main(["orders"]) == 1
    [run] = Journal(workdir / "state" / "journal.sqlite").recent_runs()
    assert run["exit_code"] == 1 and "T212_DEMO_API_KEY" in run["error"]
