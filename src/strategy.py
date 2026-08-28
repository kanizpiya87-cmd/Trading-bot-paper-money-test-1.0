"""
Momentum + Mean-Reversion blended strategy.

Signal logic:
- Trend filter: fast MA vs slow MA (momentum direction)
- Entry timing: RSI (mean-reversion within the trend)
- Only go long when trend is up AND RSI shows a pullback (not overbought)
- Only go short/flat when trend is down AND RSI shows a bounce (not oversold)

This is intentionally simple and transparent so the learning loop has a small,
interpretable set of parameters to adjust: fast_ma, slow_ma, rsi_period,
rsi_buy_max, rsi_sell_min.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class StrategyParams:
    fast_ma: int = 5          # shorter windows react faster to price moves (was 10)
    slow_ma: int = 15         # (was 30) - trend flips are detected sooner
    rsi_period: int = 10      # (was 14) - more responsive RSI
    rsi_buy_max: float = 65.0   # only buy if RSI below this (was 55 - now allows more room before "overbought")
    rsi_sell_min: float = 35.0  # only sell if RSI above this (was 45 - now allows more room before "oversold")
    position_size_pct: float = 0.05  # fraction of equity per trade, adjusted by learner

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
            position_size_pct=jitter(self.position_size_pct, 0.01, 0.15),
        )


def compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def generate_signal(prices: pd.Series, params: StrategyParams) -> str:
    """
    Returns one of: 'buy', 'sell', 'hold'
    prices: recent close price series, most recent last, indexed oldest->newest.
    """
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
