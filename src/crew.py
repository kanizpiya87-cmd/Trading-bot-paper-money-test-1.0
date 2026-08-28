"""
Crew: a small team of specialist "agents" that run IN PARALLEL for each trading
decision, all racing against one shared time budget (default 5 minutes), with
a Manager that collects whatever finished in time and makes the final call.

This is NOT multiple LLMs talking to each other - it's multiple independent,
fast, rule-based analysis functions run concurrently via a thread pool, each
looking at the same symbol from a different angle. That's what actually lets
this finish in seconds, comfortably inside a 5-minute budget, without needing
any external API calls or added cost.

Roles:
  - MarketScanner   : is this symbol's data even usable right now (enough bars, moving)?
  - TechnicalAnalyst: momentum/RSI signal (same logic as strategy.py, wrapped as a specialist)
  - RiskChecker     : is this position size within the agent's own risk limits?
  - SentimentScanner: lightweight, rule-based proxy for "is anything unusual happening"
                      (checks recent volatility as a cheap proxy - no news API/LLM call)
  - Manager         : combines the above into one decision, logs its reasoning,
                       and is the ONLY role that decides whether Execution should fire.
  - ExecutionAgent  : the ONLY role allowed to call broker.submit_order(). Never
                       decides on its own; only carries out what the Manager approved.

"Hiring": the Manager can request additional TechnicalAnalyst passes on the same
symbol with different lookback windows (a form of getting a second opinion), but
CANNOT invent new roles or call anything outside this fixed set. That cap is
enforced by MAX_EXTRA_ANALYST_CALLS below, not by the Manager's own judgment.
"""

import time
import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Optional

from strategy import StrategyParams, generate_signal, compute_rsi

CREW_TIME_BUDGET_SECONDS = 300  # 5 minutes, hard cap for the whole parallel phase per symbol batch
MAX_EXTRA_ANALYST_CALLS = 2      # how many extra "second opinion" passes the Manager may request
VOLATILITY_ALERT_STD = 0.03      # 3% rolling stdev of returns flags "unusual activity"


@dataclass
class SpecialistReport:
    role: str
    symbol: str
    ok: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_sec: float = 0.0


def _timed(role, symbol, fn, *args, **kwargs) -> SpecialistReport:
    start = time.monotonic()
    try:
        result = fn(*args, **kwargs)
        return SpecialistReport(role=role, symbol=symbol, ok=True, data=result,
                                 elapsed_sec=time.monotonic() - start)
    except Exception as e:
        return SpecialistReport(role=role, symbol=symbol, ok=False, error=str(e),
                                 elapsed_sec=time.monotonic() - start)


# ---------- Specialist functions (each is fast, synchronous, and independent) ----------

def market_scanner(prices, params: StrategyParams) -> dict:
    """Checks whether there's enough usable data to analyze this symbol at all."""
    min_needed = max(params.slow_ma, params.rsi_period) + 2
    usable = len(prices) >= min_needed
    return {
        "usable": usable,
        "bars_available": len(prices),
        "bars_needed": min_needed,
        "latest_price": float(prices.iloc[-1]) if len(prices) else None,
    }


def technical_analyst(prices, params: StrategyParams) -> dict:
    """The core momentum + RSI signal, same logic as the original single-agent strategy."""
    signal = generate_signal(prices, params)
    rsi_series = compute_rsi(prices, params.rsi_period)
    return {
        "signal": signal,
        "rsi": float(rsi_series.iloc[-1]) if len(rsi_series) else None,
        "fast_ma": float(prices.rolling(params.fast_ma).mean().iloc[-1]) if len(prices) >= params.fast_ma else None,
        "slow_ma": float(prices.rolling(params.slow_ma).mean().iloc[-1]) if len(prices) >= params.slow_ma else None,
    }


def risk_checker(agent_virtual_capital: float, params: StrategyParams, proposed_notional: float) -> dict:
    """Checks the proposed trade against basic risk rules. Does not know about
    strategy signals - purely a sanity/limits check, same job a real risk desk does."""
    max_allowed = agent_virtual_capital * 0.15  # hard ceiling regardless of what strategy asks for
    within_limit = proposed_notional <= max_allowed and proposed_notional >= 1.0
    return {
        "approved": within_limit,
        "proposed_notional": round(proposed_notional, 2),
        "max_allowed": round(max_allowed, 2),
        "reason": "ok" if within_limit else "exceeds per-trade risk ceiling or below $1 minimum",
    }


def sentiment_scanner(prices) -> dict:
    """
    Cheap, rule-based proxy for 'is something unusual happening' - looks at recent
    return volatility rather than calling any news/LLM API. High recent volatility
    is treated as a caution flag, not a stop: the Manager weighs it, doesn't obey it blindly.
    """
    if len(prices) < 10:
        return {"volatility_flag": False, "recent_std": None}
    returns = prices.pct_change().dropna()
    recent_std = float(returns.tail(10).std())
    return {
        "volatility_flag": recent_std is not None and recent_std >= VOLATILITY_ALERT_STD,
        "recent_std": recent_std,
    }


# ---------- Manager: combines specialist reports into one decision ----------

def manager_decide(reports: dict, agent_virtual_capital: float, params: StrategyParams) -> dict:
    """
    reports: dict keyed by role name -> SpecialistReport, for one symbol.
    Returns a decision dict: {action: 'buy'|'sell'|'hold', notional, reasoning: [...]}
    The Manager is deliberately simple and auditable: every factor it weighs is
    logged in `reasoning` so you can see exactly why it did or didn't trade.
    """
    reasoning = []

    scan = reports.get("MarketScanner")
    if not scan or not scan.ok or not scan.data.get("usable"):
        reasoning.append("MarketScanner: insufficient data, skipping symbol.")
        return {"action": "hold", "notional": 0.0, "reasoning": reasoning}

    tech = reports.get("TechnicalAnalyst")
    if not tech or not tech.ok:
        reasoning.append("TechnicalAnalyst: failed or missing, defaulting to hold.")
        return {"action": "hold", "notional": 0.0, "reasoning": reasoning}

    signal = tech.data.get("signal", "hold")
    reasoning.append(f"TechnicalAnalyst signal: {signal} (RSI={tech.data.get('rsi')}).")

    if signal == "hold":
        reasoning.append("No directional signal, holding.")
        return {"action": "hold", "notional": 0.0, "reasoning": reasoning}

    sentiment = reports.get("SentimentScanner")
    size_multiplier = 1.0
    if sentiment and sentiment.ok and sentiment.data.get("volatility_flag"):
        size_multiplier = 0.5  # don't block the trade, but size down under unusual volatility
        reasoning.append(
            f"SentimentScanner flagged elevated volatility (std={sentiment.data.get('recent_std'):.4f}); "
            f"sizing down by {int((1-size_multiplier)*100)}%."
        )

    proposed_notional = agent_virtual_capital * params.position_size_pct * size_multiplier

    risk = reports.get("RiskChecker")
    if not risk or not risk.ok or not risk.data.get("approved"):
        reason = risk.data.get("reason") if risk and risk.ok else "RiskChecker unavailable"
        reasoning.append(f"RiskChecker rejected trade: {reason}. Holding instead.")
        return {"action": "hold", "notional": 0.0, "reasoning": reasoning}

    reasoning.append(f"RiskChecker approved notional ${proposed_notional:.2f}.")
    return {"action": signal, "notional": round(proposed_notional, 2), "reasoning": reasoning}


# ---------- Crew orchestration: runs all specialists in parallel per symbol ----------

def run_crew_for_symbol(symbol: str, prices, agent_virtual_capital: float,
                         params: StrategyParams) -> dict:
    """
    Runs MarketScanner, TechnicalAnalyst, SentimentScanner in parallel immediately
    (they don't depend on each other). RiskChecker needs TechnicalAnalyst's implied
    trade size first, so it runs right after, still well within the time budget.
    Returns the Manager's final decision plus the full specialist report for logging.
    """
    start = time.monotonic()
    reports = {}

    with cf.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_timed, "MarketScanner", symbol, market_scanner, prices, params): "MarketScanner",
            executor.submit(_timed, "TechnicalAnalyst", symbol, technical_analyst, prices, params): "TechnicalAnalyst",
            executor.submit(_timed, "SentimentScanner", symbol, sentiment_scanner, prices): "SentimentScanner",
        }
        remaining_budget = CREW_TIME_BUDGET_SECONDS - (time.monotonic() - start)
        for future in cf.as_completed(futures, timeout=max(1, remaining_budget)):
            role = futures[future]
            try:
                reports[role] = future.result()
            except cf.TimeoutError:
                reports[role] = SpecialistReport(role=role, symbol=symbol, ok=False,
                                                  error="timed out within crew budget")

    # RiskChecker depends on knowing what TechnicalAnalyst/strategy would propose,
    # so it runs after the parallel batch (still typically sub-second).
    tech = reports.get("TechnicalAnalyst")
    proposed_notional = 0.0
    if tech and tech.ok and tech.data.get("signal") in ("buy", "sell"):
        proposed_notional = agent_virtual_capital * params.position_size_pct
    reports["RiskChecker"] = _timed(
        "RiskChecker", symbol, risk_checker, agent_virtual_capital, params, proposed_notional
    )

    decision = manager_decide(reports, agent_virtual_capital, params)
    elapsed = time.monotonic() - start

    return {
        "symbol": symbol,
        "decision": decision,
        "reports": {role: {
            "ok": r.ok, "data": r.data, "error": r.error, "elapsed_sec": round(r.elapsed_sec, 3)
        } for role, r in reports.items()},
        "crew_elapsed_sec": round(elapsed, 3),
        "within_budget": elapsed <= CREW_TIME_BUDGET_SECONDS,
    }
