"""
Main entrypoint. Run this on a schedule (e.g. GitHub Actions cron, or a VM cron job).

Each run:
  1. Loads existing agents (or creates a starter COLONY of pre-tuned agents on
     first run - see lessons.starter_presets() - instead of one blank-slate agent)
  2. Figures out the union of symbols every living agent needs, and fetches
     price bars for each symbol ONCE, concurrently, and shares that cache
     across every agent that trades it. (Previously: one fetch per
     agent-per-symbol, sequentially - with several agents this was doing the
     same network call over and over. This alone cuts network round-trips
     from agents*symbols down to symbols, and does them in parallel.)
  3. For each LIVING agent, for each symbol it trades:
       - checks any open virtual position against its ATR stop-loss/take-profit
       - runs the CREW (a small team of specialist checks - see crew.py) IN
         PARALLEL, all within a shared time budget, and lets the Manager decide
       - submits a REAL paper order via Alpaca if the Manager approved one,
         and updates the agent's open-position record
  4. Runs each agent's self-tuning step
  5. Checks spawn conditions and creates children if eligible (capped at MAX_AGENTS)
  6. Saves state, writes a leaderboard, appends a run summary + full crew
     reasoning to logs/, and (optionally) pings Discord/Slack/Telegram

NOTE ON PNL ATTRIBUTION: because all agents share ONE real Alpaca paper account,
this script tracks each agent's PnL *virtually*, based on its own recorded
entry price and notional for each symbol, rather than reading Alpaca's own
position PnL (which is account-wide, not per-agent). This is a simplification
that's fine for paper trading / research, but means the dashboard's per-agent
numbers are computed estimates, not numbers pulled directly from Alpaca.

NOTE ON THE "CREW": MarketScanner, TechnicalAnalyst, MomentumAnalyst,
VolatilityAnalyst, SentimentScanner, RiskChecker, StopManager, and the Manager
are fast, independent, rule-based functions (see crew.py) - not separate LLM
calls. They run concurrently via a thread pool so decisions stay fast and
free, while still giving you the multi-perspective, auditable reasoning trail.
"""

import os
import sys
import json
import time
import threading
import numpy as np
import concurrent.futures as cf
from dataclasses import asdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from broker import AlpacaBroker
from strategy import StrategyParams
from agent import (
    Agent, load_agents, save_agents, init_starter_colony, MAX_AGENTS, cluster_of,
)
from crew import run_crew_for_symbol, CREW_TIME_BUDGET_SECONDS

SYMBOLS_STOCKS = ["SPY", "QQQ", "AAPL"]
SYMBOLS_CRYPTO = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "SHIBUSD"]  # crypto trades 24/7; see the
                                                                        # weekend workflow for coverage
                                                                        # outside the weekday cron
ALL_SYMBOLS = SYMBOLS_STOCKS + SYMBOLS_CRYPTO
STARTING_CAPITAL = float(os.environ.get("STARTING_CAPITAL", "10000"))
MAX_BAR_FETCH_WORKERS = int(os.environ.get("MAX_BAR_FETCH_WORKERS", "8"))
MAX_AGENT_WORKERS = int(os.environ.get("MAX_AGENT_WORKERS", "4"))

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "run_log.jsonl")
LEADERBOARD_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "leaderboard.json")
_log_lock = threading.Lock()


def log_event(event: dict):
    event["time"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(event) + "\n"
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with _log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(line)


def notify(text: str):
    """Optional webhook notifications - set any of these env vars to enable.
    Silently no-ops (and never breaks a run) if unset or unreachable."""
    import requests
    try:
        discord = os.environ.get("DISCORD_WEBHOOK_URL")
        if discord:
            requests.post(discord, json={"content": text[:1900]}, timeout=5)
        slack = os.environ.get("SLACK_WEBHOOK_URL")
        if slack:
            requests.post(slack, json={"text": text[:3900]}, timeout=5)
        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
        if tg_token and tg_chat:
            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                           json={"chat_id": tg_chat, "text": text[:3900]}, timeout=5)
    except Exception:
        pass  # never let a notification failure break the trading run


def fetch_all_bars(broker: AlpacaBroker, symbols: list, timeframe: str, limit: int) -> dict:
    """Fetches price bars for every distinct symbol ONCE, concurrently, and
    returns {symbol: pd.Series}. This is the main speed win: with N agents
    all trading the same symbol list, this replaces N sequential fetches per
    symbol with 1 fetch per symbol, done in parallel."""
    bars = {}
    with cf.ThreadPoolExecutor(max_workers=MAX_BAR_FETCH_WORKERS) as executor:
        futures = {executor.submit(broker.get_recent_bars, sym, timeframe, limit): sym for sym in symbols}
        for future in cf.as_completed(futures):
            sym = futures[future]
            try:
                bars[sym] = future.result()
            except Exception as e:
                log_event({"type": "error", "symbol": sym, "error": f"bar fetch failed: {e}"})
                bars[sym] = None
    return bars


def close_position_pnl(entry_price: float, current_price: float, side: str, notional: float) -> float:
    """Realized pnl for closing a virtual position of `notional` dollars opened at
    entry_price, given the current price and which side it was opened on."""
    if entry_price <= 0:
        return 0.0
    pct_move = (current_price - entry_price) / entry_price
    return notional * pct_move if side == "buy" else notional * -pct_move


def process_agent(agent: Agent, bars_cache: dict, broker: AlpacaBroker, rng: np.random.Generator,
                   allowed_symbols: set):
    """Runs one full trading cycle for one agent across all its symbols.
    Safe to run concurrently with other agents (only touches this agent's
    own mutable state; bars_cache is read-only)."""
    if not agent.alive:
        return

    params = StrategyParams(**agent.params)
    daily_loss_paused = agent.daily_loss_breaker()
    if daily_loss_paused:
        log_event({"type": "daily_loss_pause", "agent": agent.id,
                   "message": "daily loss circuit breaker active, new entries paused this run"})

    for symbol in agent.symbols:
        if symbol not in allowed_symbols:
            continue  # filtered out this run (e.g. weekend crypto-only workflow)
        prices = bars_cache.get(symbol)
        if prices is None or prices.empty:
            continue

        current_price = float(prices.iloc[-1])
        open_position = agent.open_positions.get(symbol)
        cluster_ok = agent.cluster_exposure_ok(symbol)

        crew_result = run_crew_for_symbol(
            symbol, prices, agent.virtual_capital, params,
            open_position=open_position, cluster_ok=cluster_ok,
            daily_loss_paused=daily_loss_paused,
        )
        decision = crew_result["decision"]
        action = decision["action"]

        log_event({
            "type": "crew_decision", "agent": agent.id, "symbol": symbol,
            "action": action, "notional": decision.get("notional", 0.0),
            "reasoning": decision["reasoning"],
            "crew_elapsed_sec": crew_result["crew_elapsed_sec"],
            "within_budget": crew_result["within_budget"],
        })

        if action == "close" and open_position:
            pnl = close_position_pnl(open_position["entry_price"], current_price,
                                      open_position["side"], open_position.get("notional", 0.0))
            agent.record_trade(symbol, open_position["side"], pnl)
            try:
                broker.close_position(symbol)
            except Exception as e:
                log_event({"type": "close_error", "agent": agent.id, "symbol": symbol, "error": str(e)})
            del agent.open_positions[symbol]
            log_event({"type": "position_closed", "agent": agent.id, "symbol": symbol,
                       "pnl": round(pnl, 2), "capital_after": round(agent.virtual_capital, 2)})
            if not agent.alive:
                log_event({"type": "death", "agent": agent.id, "symbol": symbol,
                           "final_capital": agent.virtual_capital})
                notify(f":skull: Agent `{agent.id}` ({agent.preset}) hit its drawdown limit and stopped trading.")
                continue

        elif action in ("buy", "sell") and decision.get("notional", 0.0) >= 1.0:
            # if flipping an existing opposite-side position, realize it first
            if open_position and open_position["side"] != action:
                pnl = close_position_pnl(open_position["entry_price"], current_price,
                                          open_position["side"], open_position.get("notional", 0.0))
                agent.record_trade(symbol, open_position["side"], pnl)
                del agent.open_positions[symbol]
                if not agent.alive:
                    continue

            try:
                # ExecutionAgent role: this is the only place that calls the
                # broker to actually place an order, and it only ever acts
                # on a notional/side the Manager already approved above.
                order = broker.submit_order(symbol, decision["notional"], action)
                agent.open_positions[symbol] = {
                    "entry_price": current_price, "side": action, "notional": decision["notional"],
                }
                log_event({"type": "order", "agent": agent.id, "symbol": symbol,
                           "side": action, "notional": decision["notional"],
                           "order_id": order.get("id"), "executed_by": "ExecutionAgent"})
                notify(f":chart_with_upwards_trend: `{agent.id}` ({agent.preset}) {action.upper()} "
                       f"${decision['notional']:.2f} of {symbol} @ ${current_price:.2f}")
            except Exception as e:
                log_event({"type": "order_error", "agent": agent.id, "symbol": symbol, "error": str(e)})

    agent.tune_params(rng)


def build_leaderboard(agents: list[Agent]) -> list:
    rows = []
    for a in agents:
        stats = a.performance_stats()
        rows.append({
            "id": a.id, "preset": a.preset, "alive": a.alive,
            "virtual_capital": round(a.virtual_capital, 2),
            "gain_pct": round((a.virtual_capital - a.start_capital) / a.start_capital * 100, 2)
                        if a.start_capital else 0.0,
            "win_rate": round(a.win_rate * 100, 1),
            "trades": a.trade_count,
            "open_positions": len(a.open_positions),
            **stats,
        })
    rows.sort(key=lambda r: r["gain_pct"], reverse=True)
    return rows


def save_leaderboard(rows: list):
    os.makedirs(os.path.dirname(LEADERBOARD_FILE), exist_ok=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(rows, f, indent=2)


def run():
    run_start = time.monotonic()
    broker = AlpacaBroker()
    rng = np.random.default_rng()

    agents = load_agents()
    if not agents:
        agents = init_starter_colony(ALL_SYMBOLS, STARTING_CAPITAL)
        log_event({"type": "init", "message": f"created starter colony of {len(agents)} pre-tuned agents "
                                                f"(presets: {[a.preset for a in agents]})",
                   "capital": STARTING_CAPITAL})

    kill_id = os.environ.get("KILL_AGENT_ID", "").strip()
    if kill_id:
        target = next((a for a in agents if a.id == kill_id), None)
        if target and target.alive:
            target.alive = False
            log_event({"type": "kill", "agent": kill_id, "message": "killed manually from control app"})
            save_agents(agents)
            print(f"Agent {kill_id} killed manually. Skipping trading cycle this run.")
            return
        else:
            log_event({"type": "kill_error", "agent": kill_id,
                       "message": "agent not found or already dead"})
            print(f"Kill request for {kill_id} ignored: not found or already dead.")
            return

    revive_id = os.environ.get("REVIVE_AGENT_ID", "").strip()
    if revive_id:
        target = next((a for a in agents if a.id == revive_id), None)
        if target and not target.alive:
            target.revive()
            log_event({"type": "revive", "agent": revive_id,
                       "message": "revived manually from control app",
                       "new_capital": target.virtual_capital})
            save_agents(agents)
            print(f"Agent {revive_id} revived manually. Skipping trading cycle this run.")
            return
        else:
            log_event({"type": "revive_error", "agent": revive_id,
                       "message": "agent not found or already alive"})
            print(f"Revive request for {revive_id} ignored: not found or already alive.")
            return

    # Optional symbol filter, e.g. so a weekend-only workflow can trade just
    # crypto (which trades 24/7) without touching stock symbols that are
    # closed anyway. SYMBOL_FILTER: "all" (default), "crypto", or "stocks".
    symbol_filter = os.environ.get("SYMBOL_FILTER", "all").strip().lower()
    if symbol_filter == "crypto":
        allowed_symbols = set(SYMBOLS_CRYPTO)
    elif symbol_filter == "stocks":
        allowed_symbols = set(SYMBOLS_STOCKS)
    else:
        allowed_symbols = set(ALL_SYMBOLS)

    # ---- Speed win #1: fetch each distinct symbol's bars ONCE, concurrently,
    # instead of once per agent per symbol.
    living_agents = [a for a in agents if a.alive]
    needed_symbols = sorted({sym for a in living_agents for sym in a.symbols if sym in allowed_symbols})
    max_lookback = max(
        (max(StrategyParams(**a.params).slow_ma, StrategyParams(**a.params).macd_slow,
             StrategyParams(**a.params).bb_period) for a in living_agents),
        default=30,
    )
    bars_cache = fetch_all_bars(broker, needed_symbols, timeframe="5Min", limit=max_lookback + 15)

    # ---- Speed win #2: process independent agents concurrently. Each agent
    # only mutates its own state, so this is safe; log writes are lock-protected.
    new_children = []
    with cf.ThreadPoolExecutor(max_workers=MAX_AGENT_WORKERS) as executor:
        list(executor.map(lambda a: process_agent(a, bars_cache, broker, rng, allowed_symbols), living_agents))

    for agent in living_agents:
        if agent.should_spawn() and (len(agents) + len(new_children)) < MAX_AGENTS:
            child = agent.spawn_child(rng)
            new_children.append(child)
            log_event({"type": "spawn", "parent": agent.id, "child": child.id,
                       "child_capital": child.virtual_capital})
            notify(f":baby: `{agent.id}` spawned child `{child.id}` with ${child.virtual_capital:.2f}")

    agents.extend(new_children)
    save_agents(agents)
    leaderboard = build_leaderboard(agents)
    save_leaderboard(leaderboard)

    equity = broker.get_equity()
    total_elapsed = time.monotonic() - run_start
    log_event({"type": "summary", "account_equity": equity,
               "living_agents": sum(1 for a in agents if a.alive),
               "total_agents": len(agents),
               "symbols_fetched": len(needed_symbols),
               "total_run_seconds": round(total_elapsed, 2),
               "within_5min_budget": total_elapsed <= 300,
               "top_agent": leaderboard[0] if leaderboard else None})
    print(f"Run complete in {total_elapsed:.1f}s ({len(needed_symbols)} symbols, "
          f"{len(living_agents)} agents processed). Account equity: ${equity}. "
          f"Living agents: {sum(1 for a in agents if a.alive)}/{len(agents)}")


if __name__ == "__main__":
    run()
