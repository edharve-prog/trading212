import numpy as np
import pandas as pd
import pytest

from t212bot import indicators as ind


def make_bars(close, spread=1.0):
    close = pd.Series(close, dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=len(close)),
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
        }
    )


def reference_wilder_rsi(close, period=14):
    """Independent loop implementation of Wilder's RSI (seeded with the first period's EMA)."""
    deltas = np.diff(close)
    gains, losses = np.clip(deltas, 0, None), np.clip(-deltas, 0, None)
    out = [np.nan] * len(close)
    avg_g = avg_l = None
    for i, (g, loss) in enumerate(zip(gains, losses, strict=True), start=1):
        if avg_g is None:
            avg_g, avg_l = g, loss
        else:
            avg_g = (avg_g * (period - 1) + g) / period
            avg_l = (avg_l * (period - 1) + loss) / period
        if i >= period:
            out[i] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return np.array(out)


def test_sma_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype="float64")
    assert ind.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert ind.sma(s, 3).isna().sum() == 2


def test_ema_matches_recursive_formula():
    s = pd.Series([10, 11, 12, 11, 13, 14], dtype="float64")
    alpha = 2 / (3 + 1)
    expected = [s[0]]
    for x in s[1:]:
        expected.append(alpha * x + (1 - alpha) * expected[-1])
    out = ind.ema(s, 3)
    assert out.isna().sum() == 2
    np.testing.assert_allclose(out[2:], expected[2:])


def test_rsi_matches_reference_loop():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    out = ind.rsi(pd.Series(close), 14).to_numpy()
    ref = reference_wilder_rsi(close, 14)
    np.testing.assert_allclose(out[14:], ref[14:], rtol=1e-10)
    assert np.isnan(out[:14]).all()


def test_rsi_extremes():
    assert ind.rsi(pd.Series(np.arange(1, 40, dtype=float))).iloc[-1] == 100.0
    assert ind.rsi(pd.Series(np.arange(40, 1, -1, dtype=float))).iloc[-1] == pytest.approx(0.0)
    assert ind.rsi(pd.Series([5.0] * 30)).iloc[-1] == 50.0


def test_atr_constant_range():
    bars = make_bars([100.0] * 30, spread=2.0)
    out = ind.atr(bars["high"], bars["low"], bars["close"], 14)
    assert out.iloc[-1] == pytest.approx(4.0)


def test_true_range_uses_gap_from_previous_close():
    high = pd.Series([10.0, 15.0])
    low = pd.Series([9.0, 14.0])
    close = pd.Series([9.5, 14.5])
    assert ind.true_range(high, low, close).tolist() == [1.0, 5.5]


def test_macd_columns_and_sign_in_uptrend():
    close = pd.Series(np.linspace(100, 200, 100))
    out = ind.macd(close)
    assert list(out.columns) == ["macd", "macd_signal", "macd_hist"]
    assert out["macd"].iloc[-1] > 0


def test_bollinger_flat_series_has_zero_width():
    out = ind.bollinger(pd.Series([50.0] * 25))
    last = out.iloc[-1]
    assert last["bb_upper"] == last["bb_mid"] == last["bb_lower"] == 50.0


def test_adx_strong_uptrend():
    bars = make_bars(np.linspace(100, 160, 120), spread=0.5)
    out = ind.adx(bars["high"], bars["low"], bars["close"])
    last = out.iloc[-1]
    assert last["plus_di"] > last["minus_di"]
    assert last["adx"] > 50


def test_volume_ratio_and_distance_from_high():
    vol = pd.Series([100.0] * 20 + [300.0])
    assert ind.volume_ratio(vol, 20).iloc[-1] == pytest.approx(3.0)
    close = pd.Series([10.0, 12.0, 9.0])
    assert ind.distance_from_high(close, 3).iloc[-1] == pytest.approx(-0.25)


def test_add_indicators_aligns_benchmark_by_date():
    bars = make_bars(np.linspace(100, 200, 260))
    bench = pd.Series(np.linspace(100, 110, 260), index=pd.DatetimeIndex(bars["date"]))
    out = ind.add_indicators(bars, bench)
    for col in ["sma_200", "rsi_14", "atr_14", "adx", "bb_upper", "dist_52w_high", "rs_63"]:
        assert col in out
        assert pd.notna(out[col].iloc[-1])
    assert out["rs_63"].iloc[-1] > 0
    assert len(out) == len(bars)
