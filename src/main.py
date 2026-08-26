"""
Main entrypoint. Run this on a schedule (e.g. GitHub Actions cron, or a VM cron job).

Each run:
  1. Loads existing agents (or creates the root agent on first run)
  2. For each LIVING agent, for each symbol it trades:
       - pulls recent price bars
       - generates a signal from the agent's own strategy params
       - sizes a trade as a % of the agent's virtual capital
       - submits a REAL paper order via Alpaca
       - estimates pnl of the previous open position (simplified, see note below)
  3. Runs each agent's self-tuning step
  4. Checks spawn conditions and creates children if eligible (capped at MAX_AGENTS)
  5. Saves state and appends a run summary to logs/

NOTE ON PNL ATTRIBUTION: because all agents share ONE real Alpaca paper account,
this script tracks each agent's PnL *virtually* based on the price move since its
last trade in that symbol, rather than reading Alpaca's own position PnL (which is
account-wide, not per-agent). This is a simplification that's fine for paper
trading / research, but means the dashboard's per-agent numbers are Claude-computed
estimates, not numbers pulled directly from Alpaca.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from broker import AlpacaBroker
from strategy import StrategyParams, generate_signal
from agent import Agent, load_agents, save_agents, init_root_agent, MAX_AGENTS

SYMBOLS_STOCKS = ["SPY", "QQQ", "AAPL"]
SYMBOLS_CRYPTO = ["BTCUSD", "ETHUSD"]
ALL_SYMBOLS = SYMBOLS_STOCKS + SYMBOLS_CRYPTO
STARTING_CAPITAL = float(os.environ.get("STARTING_CAPITAL", "10000"))
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "run_log.jsonl")

# tracks last known price per (agent_id, symbol) so we can estimate pnl on next signal flip
LAST_PRICE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "last_prices.json")


def load_last_prices():
    if not os.path.exists(LAST_PRICE_FILE):
        return {}
    with open(LAST_PRICE_FILE) as f:
        return json.load(f)


def save_last_prices(d):
    os.makedirs(os.path.dirname(LAST_PRICE_FILE), exist_ok=True)
    with open(LAST_PRICE_FILE, "w") as f:
        json.dump(d, f, indent=2)


def log_event(event: dict):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    event["time"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def run():
    broker = AlpacaBroker()
    rng = np.random.default_rng()

    agents = load_agents()
    if not agents:
        agents = [init_root_agent(ALL_SYMBOLS, STARTING_CAPITAL)]
        log_event({"type": "init", "message": "created root agent", "capital": STARTING_CAPITAL})

    # Manual kill request from the control app (Android/desktop), passed through
    # GitHub Actions workflow_dispatch inputs as an env var. If set, kill that
    # agent immediately and skip the trading cycle for this run.
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

    last_prices = load_last_prices()
    new_children = []

    for agent in agents:
        if not agent.alive:
            continue

        params = StrategyParams(**agent.params)

        for symbol in agent.symbols:
            try:
                prices = broker.get_recent_bars(symbol, limit=max(params.slow_ma, params.rsi_period) + 10)
            except Exception as e:
                log_event({"type": "error", "agent": agent.id, "symbol": symbol, "error": str(e)})
                continue

            if prices.empty:
                continue

            current_price = float(prices.iloc[-1])
            key = f"{agent.id}:{symbol}"

            # estimate pnl since our last recorded price for this agent+symbol
            if key in last_prices:
                prev_price, prev_side = last_prices[key]["price"], last_prices[key]["side"]
                trade_value = agent.virtual_capital * params.position_size_pct
                if prev_side == "buy":
                    pnl = trade_value * (current_price - prev_price) / prev_price
                elif prev_side == "sell":
                    pnl = trade_value * (prev_price - current_price) / prev_price
                else:
                    pnl = 0.0
                if pnl != 0.0:
                    agent.record_trade(symbol, prev_side, pnl)

            if not agent.alive:
                log_event({"type": "death", "agent": agent.id, "symbol": symbol,
                           "final_capital": agent.virtual_capital})
                break

            signal = generate_signal(prices, params)

            if signal in ("buy", "sell"):
                notional = round(agent.virtual_capital * params.position_size_pct, 2)
                if notional >= 1.0:
                    try:
                        order = broker.submit_order(symbol, notional, signal)
                        log_event({"type": "order", "agent": agent.id, "symbol": symbol,
                                   "side": signal, "notional": notional, "order_id": order.get("id")})
                    except Exception as e:
                        log_event({"type": "order_error", "agent": agent.id, "symbol": symbol, "error": str(e)})

            last_prices[key] = {"price": current_price, "side": signal if signal != "hold" else
                                 last_prices.get(key, {}).get("side", "hold")}

        agent.tune_params(rng)

        if agent.should_spawn() and (len(agents) + len(new_children)) < MAX_AGENTS:
            child = agent.spawn_child(rng)
            new_children.append(child)
            log_event({"type": "spawn", "parent": agent.id, "child": child.id,
                       "child_capital": child.virtual_capital})

    agents.extend(new_children)
    save_agents(agents)
    save_last_prices(last_prices)

    equity = broker.get_equity()
    log_event({"type": "summary", "account_equity": equity,
               "living_agents": sum(1 for a in agents if a.alive),
               "total_agents": len(agents)})
    print(f"Run complete. Account equity: ${equity}. "
          f"Living agents: {sum(1 for a in agents if a.alive)}/{len(agents)}")


if __name__ == "__main__":
    run()
