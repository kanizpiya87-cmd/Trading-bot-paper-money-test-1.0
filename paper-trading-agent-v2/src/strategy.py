"""
Ensemble strategy: multiple independent technical signals are computed and
combined into a single confidence-weighted decision, instead of relying on
one MA/RSI crossover. This is the same "more opinions, one auditable
decision" philosophy as the crew (see crew.py), applied at the indicator
level.

Indicators included:
  - MA crossover (trend direction)          -> trend_signal()
  - RSI (mean-reversion timing)              -> compute_rsi()
  - MACD (momentum/trend confirmation)       -> compute_macd()
  - Bollinger Bands (volatility/breakout)    -> compute_bollinger()
  - Stochastic RSI (faster overbought/oversold) -> compute_stoch_rsi()
  - ATR (volatility sizing + stop-loss/take-profit distance) -> compute_atr()
  - Trend strength / ADX-lite (is there a trend worth following at all)
        -> compute_trend_strength()

ensemble_signal() combines all of the above into one (action, confidence,
breakdown) tuple. Confidence is 0-1 and represents how many independent
indicators agree, weighted by how reliable each indicator is in the current
regime (e.g. Bollinger squeeze -> breakout signals get more weight; strong
existing trend -> trend/MACD signals get more weight).

Params are intentionally kept in one dataclass so the learning loop
(agent.py's tune_params / mutate) has a single, serializable place to adjust.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class StrategyParams:
    # --- MA crossover / trend ---
    fast_ma: int = 5
    slow_ma: int = 15

    # --- RSI (mean reversion timing) ---
    rsi_period: int = 10
    rsi_buy_max: float = 65.0
    rsi_sell_min: float = 35.0

    # --- MACD ---
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- Bollinger Bands ---
    bb_period: int = 20
    bb_std: float = 2.0

    # --- Stochastic RSI ---
    stoch_rsi_period: int = 14
    stoch_buy_max: float = 20.0
    stoch_sell_min: float = 80.0

    # --- ATR / risk management ---
    atr_period: int = 14
    stop_loss_atr_mult: float = 1.5    # exit if price moves this many ATRs against us
    take_profit_atr_mult: float = 3.0  # exit if price moves this many ATRs in our favor

    # --- Position sizing / decision threshold ---
    position_size_pct: float = 0.05   # fraction of equity per trade, adjusted by learner
    min_confidence: float = 0.35      # ensemble must clear this to act (lower = more trades)

    def mutate(self, rng: np.random.Generator, scale: float = 0.15):
        """Return a mutated copy of these params (used when spawning child agents)."""
        def jitter(val, lo, hi, is_int=False):
            new = val * (1 + rng.uniform(-scale, scale))
            new = max(lo, min(hi, new))
            return int(round(new)) if is_int else round(new, 3)

        return StrategyParams(
            fast_ma=jitter(self.fast_ma, 3, 50, is_int=True),
            slow_ma=jitter(self.slow_ma, 10, 200, is_int=True),
            rsi_period=jitter(self.rsi_period, 5, 30, is_int=True),
            rsi_buy_max=jitter(self.rsi_buy_max, 30, 70),
            rsi_sell_min=jitter(self.rsi_sell_min, 30, 70),
            macd_fast=jitter(self.macd_fast, 5, 20, is_int=True),
            macd_slow=jitter(self.macd_slow, 15, 40, is_int=True),
            macd_signal=jitter(self.macd_signal, 5, 15, is_int=True),
            bb_period=jitter(self.bb_period, 10, 40, is_int=True),
            bb_std=jitter(self.bb_std, 1.2, 3.0),
            stoch_rsi_period=jitter(self.stoch_rsi_period, 7, 21, is_int=True),
            stoch_buy_max=jitter(self.stoch_buy_max, 10, 30),
            stoch_sell_min=jitter(self.stoch_sell_min, 70, 90),
            atr_period=jitter(self.atr_period, 7, 21, is_int=True),
            stop_loss_atr_mult=jitter(self.stop_loss_atr_mult, 0.8, 3.0),
            take_profit_atr_mult=jitter(self.take_profit_atr_mult, 1.5, 5.0),
            position_size_pct=jitter(self.position_size_pct, 0.01, 0.15),
            min_confidence=jitter(self.min_confidence, 0.15, 0.6),
        )


# ---------------------------------------------------------------- indicators

def compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(prices: pd.Series, fast: int, slow: int, signal: int):
    """Returns (macd_line, signal_line, histogram) as Series."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger(prices: pd.Series, period: int, num_std: float):
    """Returns (mid, upper, lower, pct_b, bandwidth) as Series.
    pct_b: where price sits within the bands (0 = lower band, 1 = upper band).
    bandwidth: (upper-lower)/mid, a cheap "is volatility contracting" proxy.
    """
    mid = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_range = (upper - lower).replace(0, np.nan)
    pct_b = ((prices - lower) / band_range).fillna(0.5)
    bandwidth = (band_range / mid.replace(0, np.nan)).fillna(0)
    return mid, upper, lower, pct_b, bandwidth


def compute_stoch_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Stochastic RSI: RSI's own position within its recent range (0-100).
    Reacts faster than plain RSI, useful for catching earlier reversals."""
    rsi = compute_rsi(prices, period)
    lo = rsi.rolling(period).min()
    hi = rsi.rolling(period).max()
    rng = (hi - lo).replace(0, np.nan)
    stoch = ((rsi - lo) / rng * 100).fillna(50)
    return stoch


def compute_atr(prices: pd.Series, period: int) -> pd.Series:
    """True-range proxy using close-to-close moves (we only have close prices
    from get_recent_bars, not full OHLC), which is a standard simplification
    ('close-to-close volatility') when high/low aren't available."""
    tr = prices.diff().abs()
    atr = tr.rolling(period).mean()
    return atr.bfill().fillna(0)


def compute_trend_strength(prices: pd.Series, fast: int, slow: int) -> float:
    """0-1 score for 'is there an actual trend worth following'. Uses the
    normalized separation between fast/slow MAs (wider gap = stronger trend),
    an ADX-lite proxy that doesn't need full OHLC data."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    if slow_ma.iloc[-1] == 0 or pd.isna(slow_ma.iloc[-1]):
        return 0.0
    gap = abs(fast_ma.iloc[-1] - slow_ma.iloc[-1]) / abs(slow_ma.iloc[-1])
    return float(min(1.0, gap * 25))  # scaled so a ~4% gap maxes out the score


# ------------------------------------------------------------- legacy signal

def generate_signal(prices: pd.Series, params: StrategyParams) -> str:
    """Original MA-crossover + RSI signal. Kept as one voice in the ensemble
    (and for backward compatibility with anything importing it directly)."""
    if len(prices) < max(params.slow_ma, params.rsi_period) + 2:
        return "hold"

    fast = prices.rolling(params.fast_ma).mean()
    slow = prices.rolling(params.slow_ma).mean()
    rsi = compute_rsi(prices, params.rsi_period)

    trend_up = fast.iloc[-1] > slow.iloc[-1]
    trend_down = fast.iloc[-1] < slow.iloc[-1]
    last_rsi = rsi.iloc[-1]

    if trend_up and last_rsi <= params.rsi_buy_max:
        return "buy"
    if trend_down and last_rsi >= params.rsi_sell_min:
        return "sell"
    return "hold"


# --------------------------------------------------------------- ensemble

def ensemble_signal(prices: pd.Series, params: StrategyParams) -> dict:
    """
    Combines MA-crossover, RSI, MACD, Bollinger %B, and Stochastic RSI into
    one vote. Each indicator casts a vote in {-1, 0, +1} (sell/hold/buy);
    votes are weighted (trend indicators weighted up when trend_strength is
    high, mean-reversion indicators weighted up when it's low/range-bound)
    and averaged into a confidence score in [0, 1] per direction.

    Returns: {action, confidence, votes: {name: vote}, trend_strength, atr}
    """
    min_needed = max(params.slow_ma, params.macd_slow, params.bb_period,
                      params.rsi_period, params.stoch_rsi_period, params.atr_period) + 2
    if len(prices) < min_needed:
        return {"action": "hold", "confidence": 0.0, "votes": {}, "trend_strength": 0.0, "atr": 0.0}

    votes = {}

    # 1. MA crossover (trend)
    fast_ma = prices.rolling(params.fast_ma).mean().iloc[-1]
    slow_ma = prices.rolling(params.slow_ma).mean().iloc[-1]
    votes["ma_crossover"] = 1 if fast_ma > slow_ma else (-1 if fast_ma < slow_ma else 0)

    # 2. RSI (mean reversion timing)
    rsi = compute_rsi(prices, params.rsi_period).iloc[-1]
    if rsi <= params.rsi_buy_max and rsi < 50:
        votes["rsi"] = 1
    elif rsi >= params.rsi_sell_min and rsi > 50:
        votes["rsi"] = -1
    else:
        votes["rsi"] = 0

    # 3. MACD histogram (momentum confirmation)
    _, _, hist = compute_macd(prices, params.macd_fast, params.macd_slow, params.macd_signal)
    votes["macd"] = 1 if hist.iloc[-1] > 0 else (-1 if hist.iloc[-1] < 0 else 0)
    macd_rising = len(hist) > 1 and hist.iloc[-1] > hist.iloc[-2]

    # 4. Bollinger %B (breakout / mean reversion depending on bandwidth)
    _, _, _, pct_b, bandwidth = compute_bollinger(prices, params.bb_period, params.bb_std)
    pb = pct_b.iloc[-1]
    if pb <= 0.05:
        votes["bollinger"] = 1   # pinned to lower band -> bounce candidate
    elif pb >= 0.95:
        votes["bollinger"] = -1  # pinned to upper band -> pullback candidate
    else:
        votes["bollinger"] = 0

    # 5. Stochastic RSI (faster overbought/oversold)
    stoch = compute_stoch_rsi(prices, params.stoch_rsi_period).iloc[-1]
    if stoch <= params.stoch_buy_max:
        votes["stoch_rsi"] = 1
    elif stoch >= params.stoch_sell_min:
        votes["stoch_rsi"] = -1
    else:
        votes["stoch_rsi"] = 0

    trend_strength = compute_trend_strength(prices, params.fast_ma, params.slow_ma)
    atr = compute_atr(prices, params.atr_period).iloc[-1]

    # Weighting: trending regime -> trust trend/momentum voices more;
    # range-bound regime -> trust mean-reversion voices more.
    if trend_strength >= 0.5:
        weights = {"ma_crossover": 1.4, "macd": 1.3, "rsi": 0.7, "bollinger": 0.6, "stoch_rsi": 0.6}
    else:
        weights = {"ma_crossover": 0.7, "macd": 0.8, "rsi": 1.3, "bollinger": 1.3, "stoch_rsi": 1.2}

    weighted_sum = sum(votes[k] * weights[k] for k in votes)
    total_weight = sum(weights.values())
    score = weighted_sum / total_weight  # in [-1, 1]

    # small bonus if MACD momentum is rising in the same direction (reduces
    # false starts / catches moves slightly earlier)
    if score > 0 and macd_rising:
        score = min(1.0, score + 0.05)
    elif score < 0 and not macd_rising:
        score = max(-1.0, score - 0.05)

    confidence = abs(score)
    if confidence < params.min_confidence:
        action = "hold"
    else:
        action = "buy" if score > 0 else "sell"

    return {
        "action": action,
        "confidence": round(confidence, 3),
        "score": round(score, 3),
        "votes": votes,
        "trend_strength": round(trend_strength, 3),
        "atr": round(float(atr), 5),
        "regime": "trending" if trend_strength >= 0.5 else "range_bound",
    }


def stop_loss_take_profit_hit(entry_price: float, current_price: float, side: str,
                               atr: float, params: StrategyParams) -> str | None:
    """Checks an open virtual position against ATR-based stop-loss/take-profit
    distances. Returns 'stop_loss', 'take_profit', or None."""
    if atr <= 0 or entry_price <= 0:
        return None
    move = current_price - entry_price
    if side == "buy":
        if move <= -params.stop_loss_atr_mult * atr:
            return "stop_loss"
        if move >= params.take_profit_atr_mult * atr:
            return "take_profit"
    elif side == "sell":
        if move >= params.stop_loss_atr_mult * atr:
            return "stop_loss"
        if move <= -params.take_profit_atr_mult * atr:
            return "take_profit"
    return None
