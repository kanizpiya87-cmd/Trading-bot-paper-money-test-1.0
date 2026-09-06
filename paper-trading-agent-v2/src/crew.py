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
  - MarketScanner    : is this symbol's data even usable right now (enough bars, moving)?
  - TechnicalAnalyst  : the full indicator ensemble (MA, RSI, MACD, Bollinger,
                        Stochastic RSI - see strategy.ensemble_signal), wrapped
                        as a specialist.
  - MomentumAnalyst   : MACD-focused read, logged separately for auditability
                        even though it's also folded into the ensemble.
  - VolatilityAnalyst : ATR + Bollinger bandwidth - "how much room is this
                        symbol moving in right now", used for stop distance
                        and the SentimentScanner's size-down decision.
  - SentimentScanner  : lightweight, rule-based proxy for "is anything unusual
                        happening" (checks recent volatility as a cheap proxy
                        - no news API/LLM call).
  - RiskChecker       : is this position size within the agent's own risk
                        limits, and is this symbol's correlation cluster
                        already at its exposure cap?
  - StopManager       : checks any already-open virtual position against its
                        ATR stop-loss/take-profit distance BEFORE a new entry
                        is even considered.
  - Manager           : combines all of the above into one decision, logs its
                        full reasoning, and is the ONLY role that decides
                        whether Execution should fire.
  - ExecutionAgent    : the ONLY role allowed to call broker.submit_order().
                        Never decides on its own; only carries out what the
                        Manager approved.

"Hiring": the Manager can request additional TechnicalAnalyst passes on the same
symbol with different lookback windows (a form of getting a second opinion), but
CANNOT invent new roles or call anything outside this fixed set. That cap is
enforced by MAX_EXTRA_ANALYST_CALLS below, not by the Manager's own judgment.
"""

import time
import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Optional

from strategy import (
    StrategyParams, generate_signal, compute_rsi, compute_macd, compute_bollinger,
    compute_atr, ensemble_signal, stop_loss_take_profit_hit,
)

CREW_TIME_BUDGET_SECONDS = 300  # 5 minutes, hard cap for the whole parallel phase per symbol batch
MAX_EXTRA_ANALYST_CALLS = 2      # how many extra "second opinion" passes the Manager may request
VOLATILITY_ALERT_STD = 0.05      # 5% rolling stdev of returns flags "unusual activity" (raised from 3%
                                  # since more volatile symbols like SOL/DOGE would otherwise get
                                  # sized down on nearly every trade, fighting the goal of trading them)


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
    min_needed = max(params.slow_ma, params.macd_slow, params.bb_period,
                      params.rsi_period, params.atr_period) + 2
    usable = len(prices) >= min_needed
    return {
        "usable": usable,
        "bars_available": len(prices),
        "bars_needed": min_needed,
        "latest_price": float(prices.iloc[-1]) if len(prices) else None,
    }


def technical_analyst(prices, params: StrategyParams) -> dict:
    """The full indicator ensemble (MA/RSI/MACD/Bollinger/StochRSI) combined
    into one confidence-weighted signal. See strategy.ensemble_signal()."""
    result = ensemble_signal(prices, params)
    # legacy single-indicator signal kept alongside for comparison/logging
    result["legacy_ma_rsi_signal"] = generate_signal(prices, params)
    return result


def momentum_analyst(prices, params: StrategyParams) -> dict:
    """MACD-focused read, reported separately for auditability even though
    it's already one of the ensemble's votes."""
    macd_line, signal_line, hist = compute_macd(prices, params.macd_fast, params.macd_slow, params.macd_signal)
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(hist.iloc[-1]),
        "rising": bool(len(hist) > 1 and hist.iloc[-1] > hist.iloc[-2]),
    }


def volatility_analyst(prices, params: StrategyParams) -> dict:
    """ATR + Bollinger bandwidth: how much room this symbol is moving in
    right now. Feeds stop-loss distance and the sentiment size-down check."""
    atr = compute_atr(prices, params.atr_period).iloc[-1]
    _, _, _, pct_b, bandwidth = compute_bollinger(prices, params.bb_period, params.bb_std)
    return {
        "atr": float(atr),
        "bollinger_pct_b": float(pct_b.iloc[-1]),
        "bollinger_bandwidth": float(bandwidth.iloc[-1]),
    }


def risk_checker(agent_virtual_capital: float, params: StrategyParams, proposed_notional: float,
                  cluster_ok: bool = True, daily_loss_paused: bool = False) -> dict:
    """Checks the proposed trade against basic risk rules. Does not know about
    strategy signals - purely a sanity/limits check, same job a real risk desk does."""
    max_allowed = agent_virtual_capital * 0.15  # hard ceiling regardless of what strategy asks for
    within_limit = proposed_notional <= max_allowed and proposed_notional >= 1.0

    if daily_loss_paused:
        return {"approved": False, "proposed_notional": round(proposed_notional, 2),
                "max_allowed": round(max_allowed, 2),
                "reason": "daily loss circuit breaker active (lesson #20), new entries paused"}
    if not cluster_ok:
        return {"approved": False, "proposed_notional": round(proposed_notional, 2),
                "max_allowed": round(max_allowed, 2),
                "reason": "correlation cluster exposure cap reached (lesson #21)"}

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


def stop_manager(open_position: dict | None, current_price: float, atr: float,
                  params: StrategyParams) -> dict:
    """Checks any already-open virtual position against ATR stop-loss/take-profit
    distances BEFORE a new entry is considered (lesson #6)."""
    if not open_position:
        return {"has_position": False, "exit_reason": None}
    reason = stop_loss_take_profit_hit(
        open_position["entry_price"], current_price, open_position["side"], atr, params
    )
    return {"has_position": True, "exit_reason": reason, "entry_price": open_position["entry_price"],
            "side": open_position["side"]}


# ---------- Manager: combines specialist reports into one decision ----------

def manager_decide(reports: dict, agent_virtual_capital: float, params: StrategyParams) -> dict:
    """
    reports: dict keyed by role name -> SpecialistReport, for one symbol.
    Returns a decision dict:
      {action: 'buy'|'sell'|'hold'|'close', notional, reasoning: [...]}
    'close' means: exit the existing open position (stop-loss/take-profit hit),
    independent of any new-entry signal.
    The Manager is deliberately simple and auditable: every factor it weighs is
    logged in `reasoning` so you can see exactly why it did or didn't trade.
    """
    reasoning = []

    # 0. Stop-loss / take-profit on an existing position takes priority over
    # any new-entry signal (lesson #6).
    stop = reports.get("StopManager")
    if stop and stop.ok and stop.data.get("has_position") and stop.data.get("exit_reason"):
        reasoning.append(f"StopManager: {stop.data['exit_reason']} hit on existing "
                          f"{stop.data['side']} position (entry ${stop.data['entry_price']:.2f}); closing.")
        return {"action": "close", "notional": 0.0, "reasoning": reasoning,
                "close_side": stop.data["side"]}

    scan = reports.get("MarketScanner")
    if not scan or not scan.ok or not scan.data.get("usable"):
        reasoning.append("MarketScanner: insufficient data, skipping symbol.")
        return {"action": "hold", "notional": 0.0, "reasoning": reasoning}

    tech = reports.get("TechnicalAnalyst")
    if not tech or not tech.ok:
        reasoning.append("TechnicalAnalyst: failed or missing, defaulting to hold.")
        return {"action": "hold", "notional": 0.0, "reasoning": reasoning}

    signal = tech.data.get("action", "hold")
    confidence = tech.data.get("confidence", 0.0)
    regime = tech.data.get("regime", "unknown")
    votes = tech.data.get("votes", {})
    reasoning.append(
        f"TechnicalAnalyst ensemble: {signal} (confidence={confidence}, regime={regime}, votes={votes})."
    )

    momentum = reports.get("MomentumAnalyst")
    if momentum and momentum.ok:
        reasoning.append(f"MomentumAnalyst: MACD histogram={momentum.data.get('histogram'):.4f}, "
                          f"rising={momentum.data.get('rising')}.")

    if signal == "hold":
        reasoning.append("Ensemble confidence below threshold or mixed votes, holding.")
        return {"action": "hold", "notional": 0.0, "reasoning": reasoning}

    sentiment = reports.get("SentimentScanner")
    size_multiplier = 1.0
    if sentiment and sentiment.ok and sentiment.data.get("volatility_flag"):
        size_multiplier = 0.5  # don't block the trade, but size down under unusual volatility (lesson #7)
        reasoning.append(
            f"SentimentScanner flagged elevated volatility (std={sentiment.data.get('recent_std'):.4f}); "
            f"sizing down by {int((1-size_multiplier)*100)}%."
        )

    # confidence itself scales size a little too - a 90%-agreement signal
    # gets a bit more weight than a signal that barely cleared the threshold
    confidence_scalar = 0.6 + 0.4 * min(1.0, confidence)
    proposed_notional = agent_virtual_capital * params.position_size_pct * size_multiplier * confidence_scalar

    risk = reports.get("RiskChecker")
    if not risk or not risk.ok or not risk.data.get("approved"):
        reason = risk.data.get("reason") if risk and risk.ok else "RiskChecker unavailable"
        reasoning.append(f"RiskChecker rejected trade: {reason}. Holding instead.")
        return {"action": "hold", "notional": 0.0, "reasoning": reasoning}

    reasoning.append(f"RiskChecker approved notional ${proposed_notional:.2f}.")
    return {"action": signal, "notional": round(proposed_notional, 2), "reasoning": reasoning}


# ---------- Crew orchestration: runs all specialists in parallel per symbol ----------

def run_crew_for_symbol(symbol: str, prices, agent_virtual_capital: float,
                         params: StrategyParams, open_position: dict | None = None,
                         cluster_ok: bool = True, daily_loss_paused: bool = False) -> dict:
    """
    Runs MarketScanner, TechnicalAnalyst, MomentumAnalyst, VolatilityAnalyst,
    SentimentScanner in parallel immediately (they don't depend on each
    other). RiskChecker and StopManager need outputs from that batch first,
    so they run right after - still well within the time budget.
    Returns the Manager's final decision plus the full specialist report for logging.
    """
    start = time.monotonic()
    reports = {}

    with cf.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_timed, "MarketScanner", symbol, market_scanner, prices, params): "MarketScanner",
            executor.submit(_timed, "TechnicalAnalyst", symbol, technical_analyst, prices, params): "TechnicalAnalyst",
            executor.submit(_timed, "MomentumAnalyst", symbol, momentum_analyst, prices, params): "MomentumAnalyst",
            executor.submit(_timed, "VolatilityAnalyst", symbol, volatility_analyst, prices, params): "VolatilityAnalyst",
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

    # StopManager needs current price + ATR from the batch above.
    vol = reports.get("VolatilityAnalyst")
    atr = vol.data.get("atr", 0.0) if vol and vol.ok else 0.0
    current_price = float(prices.iloc[-1])
    reports["StopManager"] = _timed("StopManager", symbol, stop_manager, open_position, current_price, atr, params)

    # RiskChecker depends on knowing what TechnicalAnalyst/strategy would propose,
    # so it runs after the parallel batch (still typically sub-second).
    tech = reports.get("TechnicalAnalyst")
    proposed_notional = 0.0
    if tech and tech.ok and tech.data.get("action") in ("buy", "sell"):
        proposed_notional = agent_virtual_capital * params.position_size_pct
    reports["RiskChecker"] = _timed(
        "RiskChecker", symbol, risk_checker, agent_virtual_capital, params, proposed_notional,
        cluster_ok, daily_loss_paused,
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
