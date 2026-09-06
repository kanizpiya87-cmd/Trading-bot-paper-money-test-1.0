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
from datetime import datetime, timezone, timedelta

from strategy import StrategyParams

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "agents.json")
MAX_DRAWDOWN_PCT = 0.20        # agent "dies" (stops trading) if its virtual equity drops 20% from peak
SPAWN_PROFIT_PCT = 0.15        # agent may spawn a child once up 15% from its start
MIN_TRADES_BEFORE_TUNE = 10
MAX_AGENTS = 8                 # hard cap so this can't run away
DAILY_LOSS_LIMIT_PCT = 0.08    # pause new entries if realized loss in the last 24h exceeds this
                                # fraction of start_capital (lesson #20)

# Symbols that tend to move together get grouped so one agent can't stack
# near-identical directional risk across all of them at once (lesson #21).
CORRELATION_CLUSTERS = {
    "crypto_majors": {"BTCUSD", "ETHUSD", "SOLUSD"},
    "meme_crypto": {"DOGEUSD", "SHIBUSD"},
    "broad_equity": {"SPY", "QQQ"},
}
MAX_OPEN_PER_CLUSTER = 2  # at most this many simultaneously-open positions per cluster, per agent


def cluster_of(symbol: str) -> str | None:
    for cluster, members in CORRELATION_CLUSTERS.items():
        if symbol in members:
            return cluster
    return None


@dataclass
class Agent:
    id: str
    parent_id: str | None
    symbols: list
    params: dict
    virtual_capital: float
    start_capital: float
    peak_capital: float
    preset: str = "default"
    trade_count: int = 0
    win_count: int = 0
    alive: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trade_log: list = field(default_factory=list)
    # symbol -> {"entry_price": float, "side": "buy"/"sell"} for open virtual positions,
    # used for ATR stop-loss/take-profit checks (lesson #6)
    open_positions: dict = field(default_factory=dict)
    trading_paused_until: str | None = None  # set by daily_loss_breaker()

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
        self.trading_paused_until = None

    def should_spawn(self):
        gain = (self.virtual_capital - self.start_capital) / self.start_capital
        return self.alive and gain >= SPAWN_PROFIT_PCT

    def tune_params(self, rng: np.random.Generator):
        """Simple self-tuning: nudge position size and confidence threshold
        based on recent win rate (lesson #14)."""
        if self.trade_count < MIN_TRADES_BEFORE_TUNE or self.trade_count % MIN_TRADES_BEFORE_TUNE != 0:
            return
        p = StrategyParams(**self.params)
        if self.win_rate > 0.55:
            p.position_size_pct = min(0.15, p.position_size_pct * 1.1)
            p.min_confidence = max(0.15, p.min_confidence * 0.95)  # slightly more willing to act
        elif self.win_rate < 0.40:
            p.position_size_pct = max(0.01, p.position_size_pct * 0.8)
            p.min_confidence = min(0.65, p.min_confidence * 1.1)   # demand more agreement
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
            preset=f"{self.preset}_child",
        )

    # ------------------------------------------------------------ analytics

    def performance_stats(self) -> dict:
        """Risk-adjusted metrics, not just win rate (lesson #19)."""
        pnls = [t["pnl"] for t in self.trade_log]
        if not pnls:
            return {"sharpe_lite": 0.0, "max_drawdown_pct": 0.0, "avg_pnl": 0.0, "trades": 0}
        arr = np.array(pnls)
        mean, std = float(arr.mean()), float(arr.std())
        sharpe_lite = round(mean / std, 3) if std > 1e-9 else 0.0

        # max drawdown over the capital curve implied by the trade log
        capital_curve = [t["capital_after"] for t in self.trade_log]
        peak = capital_curve[0]
        max_dd = 0.0
        for c in capital_curve:
            peak = max(peak, c)
            if peak > 0:
                max_dd = max(max_dd, (peak - c) / peak)

        return {
            "sharpe_lite": sharpe_lite,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_pnl": round(mean, 2),
            "trades": len(pnls),
        }

    def daily_loss_breaker(self) -> bool:
        """Lesson #20: if realized losses in the trailing 24h exceed
        DAILY_LOSS_LIMIT_PCT of start_capital, pause new entries (existing
        stop-loss/take-profit checks still run) until the window rolls off.
        Returns True if currently paused."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_loss = sum(
            t["pnl"] for t in self.trade_log
            if t["pnl"] < 0 and datetime.fromisoformat(t["time"]) >= cutoff
        )
        limit = -abs(self.start_capital * DAILY_LOSS_LIMIT_PCT)
        paused = recent_loss <= limit
        self.trading_paused_until = "24h_rolling" if paused else None
        return paused

    def cluster_exposure_ok(self, symbol: str) -> bool:
        """Lesson #21: cap simultaneously-open positions within a correlated
        cluster (e.g. don't go long BTC, ETH, and SOL all at once as if
        that were three independent bets)."""
        cluster = cluster_of(symbol)
        if cluster is None:
            return True
        open_in_cluster = sum(
            1 for sym in self.open_positions
            if cluster_of(sym) == cluster and sym != symbol
        )
        return open_in_cluster < MAX_OPEN_PER_CLUSTER


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


def init_root_agent(symbols: list, starting_capital: float,
                     params: StrategyParams | None = None, preset: str = "default",
                     agent_id: str = "root") -> Agent:
    return Agent(
        id=agent_id,
        parent_id=None,
        symbols=symbols,
        params=asdict(params) if params else asdict(StrategyParams()),
        virtual_capital=starting_capital,
        start_capital=starting_capital,
        peak_capital=starting_capital,
        preset=preset,
    )


def init_starter_colony(symbols: list, total_starting_capital: float) -> list[Agent]:
    """Instead of one blank-slate root agent, seed several differently-tuned
    presets from lessons.py (lesson #13), splitting capital evenly between
    them. This gives the bot multiple live "opinions" from day one, which
    both diversifies risk and naturally increases trade frequency, without
    weakening any risk guardrail (drawdown/spawn caps still apply per-agent)."""
    from lessons import starter_presets
    presets = starter_presets()
    per_agent_capital = round(total_starting_capital / len(presets), 2)
    agents = []
    for name, params in presets.items():
        agents.append(init_root_agent(
            symbols, per_agent_capital, params=params, preset=name,
            agent_id=f"root_{name}",
        ))
    return agents
