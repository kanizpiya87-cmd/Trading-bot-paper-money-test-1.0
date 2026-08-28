"""
Agent: owns a strategy config, tracks its own (virtual) sub-ledger of trades,
adjusts its own parameters over time, and can be marked dead or spawn children.

IMPORTANT: All agents trade through the SAME Alpaca paper account. There is no
way to have fully separate real sub-accounts per spawned agent without opening
multiple brokerage accounts. So "spawning" here means: a new agent config is
created and tracked in the ledger with its own virtual capital allocation and
its own trade history, but real orders all flow through one paper account.
Position sizing per agent is scaled down accordingly so they don't double up.
"""

import json
import os
import uuid
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from strategy import StrategyParams, generate_signal

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "agents.json")
MAX_DRAWDOWN_PCT = 0.20   # agent "dies" (stops trading) if its virtual equity drops 20% from peak
SPAWN_PROFIT_PCT = 0.15   # agent may spawn a child once up 15% from its start
MIN_TRADES_BEFORE_TUNE = 10
MAX_AGENTS = 8            # hard cap so this can't run away


@dataclass
class Agent:
    id: str
    parent_id: str | None
    symbols: list
    params: dict
    virtual_capital: float
    start_capital: float
    peak_capital: float
    trade_count: int = 0
    win_count: int = 0
    alive: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trade_log: list = field(default_factory=list)

    @property
    def win_rate(self):
        return self.win_count / self.trade_count if self.trade_count else 0.0

    def record_trade(self, symbol: str, side: str, pnl: float):
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        self.virtual_capital += pnl
        self.peak_capital = max(self.peak_capital, self.virtual_capital)
        self.trade_log.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "side": side, "pnl": round(pnl, 2),
            "capital_after": round(self.virtual_capital, 2),
        })
        self._check_death()

    def _check_death(self):
        drawdown = (self.peak_capital - self.virtual_capital) / self.peak_capital
        if drawdown >= MAX_DRAWDOWN_PCT:
            self.alive = False

    def revive(self, fresh_capital: float | None = None):
        """
        Bring a dead agent back to life. Resets its virtual capital (to either
        a specified fresh amount, or its original starting capital if not
        given), clears its peak/drawdown tracking, and keeps its strategy
        params and trade history intact so you can see how it performed
        before, and how it's tuned now, after reviving it.
        """
        new_capital = fresh_capital if fresh_capital is not None else self.start_capital
        self.virtual_capital = new_capital
        self.start_capital = new_capital
        self.peak_capital = new_capital
        self.alive = True

    def should_spawn(self):
        gain = (self.virtual_capital - self.start_capital) / self.start_capital
        return self.alive and gain >= SPAWN_PROFIT_PCT

    def tune_params(self, rng: np.random.Generator):
        """Simple self-tuning: nudge position size based on recent win rate."""
        if self.trade_count < MIN_TRADES_BEFORE_TUNE or self.trade_count % MIN_TRADES_BEFORE_TUNE != 0:
            return
        p = StrategyParams(**self.params)
        if self.win_rate > 0.55:
            p.position_size_pct = min(0.15, p.position_size_pct * 1.1)
        elif self.win_rate < 0.40:
            p.position_size_pct = max(0.01, p.position_size_pct * 0.8)
        self.params = asdict(p)

    def spawn_child(self, rng: np.random.Generator) -> "Agent":
        parent_params = StrategyParams(**self.params)
        child_params = parent_params.mutate(rng)
        child_capital = self.virtual_capital * 0.3  # allocate a slice, don't double-count
        self.virtual_capital -= child_capital
        return Agent(
            id=str(uuid.uuid4())[:8],
            parent_id=self.id,
            symbols=self.symbols,
            params=asdict(child_params),
            virtual_capital=child_capital,
            start_capital=child_capital,
            peak_capital=child_capital,
        )


def load_agents() -> list[Agent]:
    if not os.path.exists(STATE_FILE):
        return []
    with open(STATE_FILE) as f:
        raw = json.load(f)
    return [Agent(**a) for a in raw]


def save_agents(agents: list[Agent]):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump([asdict(a) for a in agents], f, indent=2)


def init_root_agent(symbols: list, starting_capital: float) -> Agent:
    return Agent(
        id="root",
        parent_id=None,
        symbols=symbols,
        params=asdict(StrategyParams()),
        virtual_capital=starting_capital,
        start_capital=starting_capital,
        peak_capital=starting_capital,
    )
