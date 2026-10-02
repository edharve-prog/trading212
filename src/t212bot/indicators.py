"""Technical indicators on daily bars, in plain pandas so they run anywhere (including ARM).

Smoothed indicators (RSI, ATR, ADX) use Wilder's smoothing, an EMA with alpha = 1/period.
Every function returns a Series (or DataFrame) aligned to the input index, NaN until there
is enough history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), period)
    loss = _wilder(-delta.clip(upper=0), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + gain / loss)
    # No losses in the window means RSI 100; no movement at all means neutral 50.
    out = out.where(loss != 0, 100.0)
    out = out.where(~((gain == 0) & (loss == 0)), 50.0)
    return out.where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1, skipna=True)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), period)


def bollinger(close: pd.Series, period: int = 20, width: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + width * std, "bb_lower": mid - width * std}
    )


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(high, low, close, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * _wilder(plus_dm, period) / atr_
        minus_di = 100 * _wilder(minus_dm, period) / atr_
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": _wilder(dx, period)})


def roc(close: pd.Series, period: int) -> pd.Series:
    return close / close.shift(period) - 1


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Today's volume relative to the average of the previous ``period`` days."""
    avg = volume.shift(1).rolling(period, min_periods=period).mean()
    return volume / avg.replace(0, np.nan)


def distance_from_high(close: pd.Series, period: int = 252) -> pd.Series:
    """Fractional distance below the rolling ``period``-day closing high (0 = at the high)."""
    high = close.rolling(period, min_periods=period).max()
    return close / high - 1


def relative_strength(close: pd.Series, benchmark: pd.Series, period: int = 63) -> pd.Series:
    """Return over ``period`` days minus the benchmark's return over the same days."""
    aligned = benchmark.reindex(close.index)
    return roc(close, period) - roc(aligned, period)


def add_indicators(bars: pd.DataFrame, benchmark_close: pd.Series | None = None) -> pd.DataFrame:
    """Return a copy of single-symbol bars (date-sorted) with the standard indicator set.

    ``benchmark_close`` is optional and must be indexed by date.
    """
    df = bars.copy()
    close, high, low = df["close"], df["high"], df["low"]
    for period in (20, 50, 200):
        df[f"sma_{period}"] = sma(close, period)
    df["ema_21"] = ema(close, 21)
    df["rsi_14"] = rsi(close, 14)
    df = df.join(macd(close))
    df["atr_14"] = atr(high, low, close, 14)
    df["atr_pct"] = df["atr_14"] / close
    df = df.join(bollinger(close))
    df = df.join(adx(high, low, close, 14))
    df["roc_10"] = roc(close, 10)
    df["roc_20"] = roc(close, 20)
    df["vol_ratio_20"] = volume_ratio(df["volume"], 20)
    df["dist_52w_high"] = distance_from_high(close, 252)
    if benchmark_close is not None:
        # benchmark_close is indexed by date; line it up with this frame's date column.
        bench = pd.Series(
            benchmark_close.reindex(pd.DatetimeIndex(df["date"])).to_numpy(), index=df.index
        )
        df["rs_63"] = relative_strength(close, bench, 63)
    return df
