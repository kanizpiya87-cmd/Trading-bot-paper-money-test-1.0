"""
Lessons: a small, hand-written knowledge base the bot starts with instead of
a single blank-slate default config. Two things live here:

1. LESSONS - 20 short, well-known trading heuristics. These aren't magic and
   don't guarantee profit; they're standard risk/behavior rules that a
   beginner trader gets told repeatedly. Each lesson maps to something
   concrete the code already enforces or a starter parameter choice below,
   so the "knowledge" isn't just flavor text - see `enforced_by` on each one.

2. STARTER_PRESETS - instead of spawning ONE root agent with arbitrary
   default parameters and waiting weeks for it to randomly mutate into
   something reasonable, we spawn a handful of root agents pre-seeded with
   different, sane starting configurations (trend-follower, mean-reverter,
   fast-scalper, conservative). This means:
     - the bot has several different "opinions" running from day one
     - it naturally trades more often (more agents x more indicators, see
       strategy.ensemble_signal) without weakening any risk guardrail
     - the self-tuning loop (agent.tune_params) still adapts each one from
       there based on its own live win rate
"""

from dataclasses import asdict
from strategy import StrategyParams

LESSONS = [
    {"id": 1, "title": "The trend is your friend",
     "rule": "Weight trend-confirming indicators (MA crossover, MACD) more heavily when trend_strength is high.",
     "enforced_by": "strategy.ensemble_signal() regime weighting"},
    {"id": 2, "title": "Don't fight a strong trend with mean-reversion signals",
     "rule": "RSI/Bollinger/StochRSI are down-weighted in a trending regime so they don't call premature reversals.",
     "enforced_by": "strategy.ensemble_signal() regime weighting"},
    {"id": 3, "title": "Buy dips in an uptrend, don't buy strength",
     "rule": "Only buy when RSI is below the buy threshold, not simply because price is rising.",
     "enforced_by": "strategy.ensemble_signal() rsi vote"},
    {"id": 4, "title": "Sell bounces in a downtrend, don't sell weakness",
     "rule": "Only sell when RSI is above the sell threshold.",
     "enforced_by": "strategy.ensemble_signal() rsi vote"},
    {"id": 5, "title": "Never risk more than a small slice of capital on one trade",
     "rule": "Hard per-trade notional ceiling regardless of what the strategy asks for.",
     "enforced_by": "crew.risk_checker() 15% ceiling"},
    {"id": 6, "title": "Cut losses, let winners run",
     "rule": "Use an ATR-based stop-loss tighter than the take-profit target (default 1.5x vs 3x ATR).",
     "enforced_by": "strategy.stop_loss_take_profit_hit()"},
    {"id": 7, "title": "Size down when volatility spikes",
     "rule": "Reduce position size under abnormal recent volatility instead of blocking the trade outright.",
     "enforced_by": "crew.sentiment_scanner() + manager_decide() size_multiplier"},
    {"id": 8, "title": "Confirm signals across more than one indicator",
     "rule": "Require several independent indicators to agree (confidence threshold) before acting.",
     "enforced_by": "strategy.ensemble_signal() min_confidence"},
    {"id": 9, "title": "A move at the edge of its recent range often reverses",
     "rule": "Price pinned at the Bollinger Band edges is treated as a reversal candidate, not a breakout, by default.",
     "enforced_by": "strategy.ensemble_signal() bollinger vote"},
    {"id": 10, "title": "Momentum should confirm price, not just follow it",
     "rule": "A small confidence bonus is given only when MACD histogram is moving the same direction as the score.",
     "enforced_by": "strategy.ensemble_signal() macd_rising bonus"},
    {"id": 11, "title": "Protect capital first, grow it second",
     "rule": "An agent stops trading entirely if its drawdown from peak hits 20%.",
     "enforced_by": "agent.Agent._check_death()"},
    {"id": 12, "title": "Let good performers grow, but cap the colony",
     "rule": "Agents may spawn a mutated child once up 15%, but the total population is capped so risk can't compound unbounded.",
     "enforced_by": "agent.Agent.should_spawn() / MAX_AGENTS"},
    {"id": 13, "title": "Diversify starting assumptions instead of guessing one config",
     "rule": "Start with several differently-tuned presets (trend, mean-reversion, scalper, conservative) rather than one.",
     "enforced_by": "lessons.STARTER_PRESETS"},
    {"id": 14, "title": "Adapt position size to real, recent performance",
     "rule": "Position size nudges up after a stretch of good win rate, down after a stretch of poor win rate.",
     "enforced_by": "agent.Agent.tune_params()"},
    {"id": 15, "title": "Don't oversize into a market with no clear direction",
     "rule": "In a range-bound regime, base position sizing leans on the more conservative preset's smaller pct.",
     "enforced_by": "lessons.STARTER_PRESETS['mean_reversion']"},
    {"id": 16, "title": "Faster timeframes need faster (not slower) confirmation",
     "rule": "The scalper preset uses a shorter MA/RSI/MACD lookback so it doesn't lag on 5-minute bars.",
     "enforced_by": "lessons.STARTER_PRESETS['scalper']"},
    {"id": 17, "title": "A wide RSI band with no other filter is not a strategy",
     "rule": "RSI alone never triggers a trade - it's one vote among five, weighted by regime.",
     "enforced_by": "strategy.ensemble_signal()"},
    {"id": 18, "title": "Every decision should be explainable after the fact",
     "rule": "Every specialist's reasoning and every indicator vote is logged, not just the final action.",
     "enforced_by": "crew.manager_decide() reasoning[] + main.py log_event()"},
    {"id": 19, "title": "Track risk-adjusted return, not just raw P&L",
     "rule": "Report Sharpe-lite and max drawdown per agent, not just win rate.",
     "enforced_by": "agent.Agent.performance_stats()"},
    {"id": 20, "title": "A daily loss limit prevents one bad day from becoming a bad month",
     "rule": "If an agent's realized loss in a rolling 24h window exceeds a threshold, it pauses new entries until the window rolls off.",
     "enforced_by": "agent.Agent.daily_loss_breaker()"},
    {"id": 21, "title": "Correlated symbols shouldn't all fire the same bet",
     "rule": "Crypto majors (BTC/ETH/SOL) are treated as one correlation cluster for exposure purposes so one agent can't stack near-identical risk five times over.",
     "enforced_by": "agent.py CORRELATION_CLUSTERS"},
    {"id": 22, "title": "Backtest before you trust a config",
     "rule": "Starter presets were chosen based on standard, well-documented technical-analysis defaults, not arbitrary numbers.",
     "enforced_by": "lessons.STARTER_PRESETS (see comments)"},
]


def starter_presets() -> dict:
    """Named, pre-tuned StrategyParams. Values are standard/textbook defaults
    for each style, not the result of a private backtest - treat them as a
    reasonable starting point for the self-tuning loop, not a guarantee."""
    return {
        "trend_follower": StrategyParams(
            fast_ma=8, slow_ma=21, rsi_period=14, rsi_buy_max=70, rsi_sell_min=30,
            macd_fast=12, macd_slow=26, macd_signal=9,
            bb_period=20, bb_std=2.0, stoch_rsi_period=14,
            position_size_pct=0.06, min_confidence=0.30,
        ),
        "mean_reversion": StrategyParams(
            fast_ma=5, slow_ma=15, rsi_period=10, rsi_buy_max=35, rsi_sell_min=65,
            macd_fast=8, macd_slow=17, macd_signal=9,
            bb_period=14, bb_std=1.8, stoch_rsi_period=10,
            position_size_pct=0.04, min_confidence=0.40,
        ),
        "scalper": StrategyParams(
            fast_ma=3, slow_ma=9, rsi_period=7, rsi_buy_max=60, rsi_sell_min=40,
            macd_fast=6, macd_slow=13, macd_signal=5,
            bb_period=10, bb_std=1.6, stoch_rsi_period=7,
            position_size_pct=0.03, min_confidence=0.25,
        ),
        "conservative": StrategyParams(
            fast_ma=10, slow_ma=30, rsi_period=14, rsi_buy_max=60, rsi_sell_min=40,
            macd_fast=12, macd_slow=26, macd_signal=9,
            bb_period=20, bb_std=2.2, stoch_rsi_period=14,
            position_size_pct=0.03, min_confidence=0.50,
            stop_loss_atr_mult=1.2, take_profit_atr_mult=2.5,
        ),
    }


def starter_preset_dicts() -> dict:
    return {name: asdict(p) for name, p in starter_presets().items()}
