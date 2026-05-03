# indicators.py
# ============================================================
# All technical indicator calculations live here.
#
# By centralising them in one file we ensure that the Chart
# Terminal page, the Signals tab, and alerts.py all use the
# EXACT same math — no risk of subtle differences causing
# different signals on the same data.
#
# Every function here takes a pandas Series (a single column
# of price or volume data) and returns a new Series of the
# same length, with NaN values at the front where there is not
# yet enough historical data to compute the indicator.
# ============================================================

import pandas as pd
import numpy as np


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) using Wilder's smoothing method.

    RSI measures how overbought or oversold a stock is on a scale of 0 to 100.
    Above 80: the stock has risen fast and may be due for a pullback (overbought).
    Below 30: the stock has fallen fast and may be due for a bounce (oversold).

    HOW THE MATH WORKS:
    1. Find each day's price change (today minus yesterday).
    2. Separate gains (positive changes) from losses (negative changes, made positive).
    3. Smooth both using exponential weighted moving averages (EWM).
       Alpha = 1/period means each new day gets weight 1/14, and older days get
       progressively less weight — recent price action matters more.
    4. RS = average gain / average loss.
    5. RSI = 100 − (100 / (1 + RS)).
       If gains dominate RS is large → RSI approaches 100.
       If losses dominate RS is small → RSI approaches 0.

    Args:
        series (pd.Series): Daily closing prices indexed by date.
        period (int): Look-back window in trading days. Standard is 14.

    Returns:
        pd.Series: RSI values, same index as input. First `period` rows will be NaN.
    """
    delta    = series.diff()                          # day-over-day price change
    gains    = delta.clip(lower=0)                    # keep only positive moves
    losses   = (-delta).clip(lower=0)                 # flip losses to positive numbers

    # alpha=1/period is Wilder's original smoothing constant
    alpha    = 1 / period
    avg_gain = gains.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD shows the relationship between two exponential moving averages of price.
    It is used to spot changes in momentum, direction, and duration of a trend.

    HOW THE MATH WORKS:
    1. MACD Line = 12-day EMA minus 26-day EMA.
       When the short EMA is above the long EMA the MACD is positive → bullish momentum.
    2. Signal Line = 9-day EMA of the MACD line itself, acting as a trigger.
    3. Histogram = MACD Line minus Signal Line.
       When the histogram crosses from negative to positive, traders see it as a buy signal.

    Args:
        series (pd.Series): Daily closing prices.
        fast (int): Short EMA period. Default 12.
        slow (int): Long EMA period. Default 26.
        sig (int): Signal line EMA period. Default 9.

    Returns:
        tuple: (macd_line, signal_line, histogram) — three pd.Series of same length as input.
    """
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands around a simple moving average.

    Bollinger Bands create an envelope around price. When price touches or breaks
    the upper band the stock may be overextended. When it touches the lower band
    it may be at a short-term extreme — often used as a mean-reversion signal.

    HOW THE MATH WORKS:
    1. Middle Band = simple moving average (SMA) of the last `period` closes.
    2. Standard Deviation = how far prices have spread from the SMA in that window.
    3. Upper Band = SMA + (num_std × std). With 2 standard deviations roughly
       95% of all price action historically falls inside the bands.
    4. Lower Band = SMA − (num_std × std).

    Args:
        series (pd.Series): Daily closing prices.
        period (int): Rolling window in days. Default 20.
        num_std (float): Number of standard deviations for band width. Default 2.0.

    Returns:
        tuple: (upper_band, middle_band, lower_band) — three pd.Series.
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    return upper, sma, lower


def calculate_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """
    Calculate VWAP (Volume Weighted Average Price).

    VWAP is the average price a stock has traded at throughout the day,
    weighted by how much volume occurred at each price. Institutions use it
    as a benchmark — buying below VWAP is considered a good fill.

    Note: True intraday VWAP resets each day. Since we are working with daily
    data, this is a cumulative VWAP from the first day in the dataset forward,
    which acts more like a long-run fair-value reference line.

    HOW THE MATH WORKS:
    1. Typical Price = (High + Low + Close) / 3.
       This single number represents the average price for that trading day.
    2. Multiply typical price by volume to get dollar volume traded.
    3. Divide cumulative dollar volume by cumulative volume.
       The result is the average price per share, weighted by how many shares
       traded at each level.

    Args:
        high   (pd.Series): Daily high prices.
        low    (pd.Series): Daily low prices.
        close  (pd.Series): Daily closing prices.
        volume (pd.Series): Daily volume (number of shares traded).

    Returns:
        pd.Series: VWAP values, same index as input.
    """
    typical_price    = (high + low + close) / 3
    dollar_volume    = typical_price * volume          # total dollars traded each day
    cumulative_dv    = dollar_volume.cumsum()          # running total of dollar volume
    cumulative_vol   = volume.cumsum()                 # running total of shares traded
    return cumulative_dv / cumulative_vol


def calculate_historical_volatility(close: pd.Series, period: int = 30) -> float:
    """
    Calculate annualised historical volatility over the last `period` trading days.

    Historical volatility measures how much a stock's price has fluctuated
    in the recent past. It is used to compare against implied volatility (IV)
    from options to judge whether options are cheap or expensive.

    HOW THE MATH WORKS:
    1. Log return = ln(today's price / yesterday's price). Log returns are used
       instead of simple returns because they are symmetric and easier to work with.
    2. Standard deviation of log returns = how spread out the daily moves are.
    3. Annualise by multiplying by √252 because there are ~252 trading days per year
       and volatility scales with the square root of time.
    4. The result is expressed as a decimal (0.30 = 30% annualised vol).

    Args:
        close (pd.Series): Daily closing prices.
        period (int): How many recent trading days to measure. Default 30.

    Returns:
        float: Annualised historical volatility as a decimal (e.g. 0.30 for 30%).
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    recent_returns = log_returns.tail(period)
    # 252 trading days per year; std() gives daily vol; multiply by sqrt(252) to annualise
    return float(recent_returns.std() * np.sqrt(252))


def evaluate_leap_conditions(
    df: pd.DataFrame,
    conditions: dict,
    rsi_series: pd.Series,
    macd_line: pd.Series,
    signal_line: pd.Series,
    bb_lower: pd.Series,
) -> pd.Series:
    """
    Evaluate all active signal conditions on every row of historical data.

    Returns a boolean Series where True means ALL active conditions were
    simultaneously satisfied on that date — a signal hit.

    This function is the core of the Signals tab scanner and the alerts engine.
    By running it on the full 5-year dataset we can see every time in history
    this setup would have triggered.

    HOW IT WORKS:
    Each condition is checked independently against every date.
    We start with all True, then AND each active condition in.
    If any condition is False on a given date, that date gets False overall.
    Think of it as stacking filters: a date must pass every filter to survive.

    Args:
        df (pd.DataFrame): OHLCV dataframe from yfinance with Close, High, Low, Volume.
        conditions (dict): The condition configuration dict from the database.
        rsi_series (pd.Series): Pre-calculated RSI values, same index as df.
        macd_line (pd.Series): Pre-calculated MACD line values.
        signal_line (pd.Series): Pre-calculated MACD signal line values.
        bb_lower (pd.Series): Pre-calculated Bollinger lower band values.

    Returns:
        pd.Series: Boolean Series (True = all conditions met on that date).
    """
    close    = df["Close"].squeeze()
    volume   = df["Volume"].squeeze()

    # Start with every day as a candidate — we will filter down from here
    mask = pd.Series(True, index=df.index)

    # ── RSI below threshold ─────────────────────────────────
    cfg = conditions.get("rsi_below", {})
    if cfg.get("enabled"):
        # True only on days where RSI was below the threshold (oversold territory)
        mask &= rsi_series < float(cfg.get("value", 35))

    # ── RSI above threshold ─────────────────────────────────
    cfg = conditions.get("rsi_above", {})
    if cfg.get("enabled"):
        mask &= rsi_series > float(cfg.get("value", 70))

    # ── Price near 200-day moving average ───────────────────
    cfg = conditions.get("near_200ma", {})
    if cfg.get("enabled"):
        ma200      = close.rolling(200).mean()
        pct_thresh = float(cfg.get("value", 2.0)) / 100
        # Distance from MA as a percentage of the MA value
        pct_away   = abs(close - ma200) / ma200
        mask &= pct_away <= pct_thresh

    # ── Price near 50-day moving average ────────────────────
    cfg = conditions.get("near_50ma", {})
    if cfg.get("enabled"):
        ma50       = close.rolling(50).mean()
        pct_thresh = float(cfg.get("value", 2.0)) / 100
        pct_away   = abs(close - ma50) / ma50
        mask &= pct_away <= pct_thresh

    # ── Price down X% or more from 52-week high ─────────────
    cfg = conditions.get("down_from_52w", {})
    if cfg.get("enabled"):
        high_52w   = close.rolling(252).max()
        pct_thresh = float(cfg.get("value", 10.0)) / 100
        # Percentage drawdown from the rolling 52-week high
        drawdown   = (high_52w - close) / high_52w
        mask &= drawdown >= pct_thresh

    # ── Volume X times above 20-day average ─────────────────
    cfg = conditions.get("vol_multiplier", {})
    if cfg.get("enabled"):
        avg_vol    = volume.rolling(20).mean()
        multiplier = float(cfg.get("value", 1.5))
        mask &= volume >= avg_vol * multiplier

    # ── Bollinger Band lower band touch ─────────────────────
    cfg = conditions.get("bb_lower_touch", {})
    if cfg.get("enabled"):
        # True when price is at or below the lower Bollinger Band
        mask &= close <= bb_lower

    # ── MACD bullish crossover ───────────────────────────────
    cfg = conditions.get("macd_bullish", {})
    if cfg.get("enabled"):
        # A bullish crossover is when MACD crosses ABOVE the signal line.
        # Yesterday MACD was below signal, today it is above or equal.
        prev_macd   = macd_line.shift(1)
        prev_signal = signal_line.shift(1)
        crossover   = (prev_macd < prev_signal) & (macd_line >= signal_line)
        mask &= crossover

    return mask
