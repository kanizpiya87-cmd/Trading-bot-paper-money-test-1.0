# Paper Trading Agent

A self-tuning, self-"spawning" trading agent that trades **paper money only**
via Alpaca's paper trading API. It cannot place real trades unless you
deliberately change the broker endpoint — which this project actively guards
against (see `src/broker.py`).

## Why it was only making 1-2 trades a month (and what changed)

Two separate problems were combining to make this feel painfully slow:

1. **The strategy itself was strict.** The original signal only fired when
   both an MA crossover AND an RSI condition agreed — a narrow gate that
   holds most of the time by design. v2 replaces this with a 5-indicator
   **confidence-weighted ensemble** (see "The Crew" below) with a tunable
   `min_confidence` threshold, so it acts on strong-enough agreement instead
   of requiring one specific combination.
2. **GitHub Actions' own cron scheduler is not reliable at 5-minute
   granularity** on the free tier — it's a known, documented limitation that
   scheduled runs get queued and delayed (sometimes by 30-60+ minutes) when
   GitHub's scheduler is under load, especially on quiet/low-activity repos.
   This is *not* something the trading code can fix by itself. v2 adds a
   `repository_dispatch` trigger so you can point a free external pinger
   (e.g. cron-job.org) at the GitHub API to trigger runs on a real 5-minute
   cadence instead of relying solely on GitHub's queue — see "Speeding this
   up further" below.

## What it does

- Trades a basket of stocks/ETFs (`SPY`, `QQQ`, `AAPL`) and crypto (`BTCUSD`,
  `ETHUSD`, `SOLUSD`, `DOGEUSD`, `SHIBUSD`) using a **team of specialist
  checks** rather than one monolithic strategy — see "The Crew" below.
- Starts as a **colony of 4 pre-tuned agents** (trend-follower, mean-reverter,
  scalper, conservative — see "Pre-loaded lessons" below) instead of one
  blank-slate agent, so it has several live opinions running from day one.
- Combines **5 independent technical indicators** (MA crossover, RSI, MACD,
  Bollinger %B, Stochastic RSI) into one confidence-weighted decision per
  symbol, instead of one rigid crossover rule.
- Manages **ATR-based stop-loss / take-profit** on every open virtual
  position, checked every cycle, before any new-entry signal is even
  considered.
- Enforces a **daily loss circuit breaker** (pauses new entries, not exits,
  if realized losses in a rolling 24h window get too large) and a
  **correlation-cluster exposure cap** (won't stack near-identical bets
  across BTC/ETH/SOL as if they were independent).
- Tracks its own virtual P&L, win rate, drawdown, and now **Sharpe-lite and
  max drawdown per agent** (see `data/leaderboard.json`).
- **Dies** (stops trading) if its virtual capital drawdown hits 20% from peak.
- Can be **revived** from the control app, restarting with fresh capital.
- **Spawns a child agent** with mutated strategy parameters once it's up 15%
  from its starting capital, capped at 8 total agents so it can't run away.
- Self-tunes position sizing *and* confidence threshold based on recent win rate.
- Fetches each symbol's price data **once per run, concurrently**, shared
  across every agent that trades it (previously: one fetch per agent per
  symbol, sequentially) — and processes independent agents concurrently too.
  With 4+ agents this cuts a run's network round-trips and wall time
  substantially; exact speedup depends on agent/symbol count and network
  latency.
- Sends optional **Discord / Slack / Telegram** notifications on trades,
  spawns, deaths, and daily-loss pauses.
- Runs automatically on a schedule via GitHub Actions (every 5 minutes on
  weekdays, plus a lighter weekend workflow for crypto) and commits its own
  state/logs back to the repo so you can watch it evolve.

## The Crew

Instead of one strategy function, each trading decision runs through a small
team of specialist checks (`src/crew.py`), all **in parallel**, all within a
hard 5-minute time budget per decision (in practice each one finishes in a
fraction of a second):

- **MarketScanner** — is there enough usable price data for this symbol right now?
- **TechnicalAnalyst** — the full 5-indicator ensemble (MA crossover, RSI,
  MACD, Bollinger %B, Stochastic RSI — see `strategy.ensemble_signal()`),
  combined into one confidence-weighted buy/sell/hold call
- **MomentumAnalyst** — MACD histogram/direction, logged separately for
  auditability even though it's also one of the ensemble's votes
- **VolatilityAnalyst** — ATR + Bollinger bandwidth ("how much room is this
  symbol moving in right now"), feeds stop-loss distance and sizing
- **SentimentScanner** — a cheap, rule-based volatility check (flags unusual
  price swings, sizes trades down under it — no news API or LLM call involved)
- **RiskChecker** — enforces a hard per-trade risk ceiling, the daily-loss
  circuit breaker, and the correlation-cluster exposure cap
- **StopManager** — checks any already-open virtual position against its
  ATR stop-loss/take-profit distance, *before* any new-entry signal is even
  considered
- **Manager** — combines all of the above into one final decision, and writes
  out its full reasoning so every trade is auditable (see the Event Log in the
  control app)
- **ExecutionAgent** — the only role allowed to actually call Alpaca's order
  API, and only ever on what the Manager already approved

These are fast, independent, rule-based Python functions run concurrently via
a thread pool — not separate LLM calls — which is what keeps this both free
and comfortably inside the 5-minute budget. The Manager can request a second
`TechnicalAnalyst` opinion on the same symbol (a capped, fixed capability),
but cannot invent new roles or capabilities on its own.

## Pre-loaded lessons (starter knowledge, not a blank slate)

`src/lessons.py` seeds the bot with 22 short, standard trading heuristics
(trend-following, risk sizing, stop discipline, diversification, etc.) —
each one maps to something concretely enforced in the code (see the
`enforced_by` field on each lesson), not just flavor text. The practical
effect is `init_starter_colony()`: instead of one agent with arbitrary
default parameters, the bot starts as **4 differently-tuned agents**
(`trend_follower`, `mean_reversion`, `scalper`, `conservative`), each seeded
from standard technical-analysis defaults for that style. They still
self-tune from there based on their own live results — this just means it
isn't starting from zero. Read the full list directly in `src/lessons.py`;
it's short and meant to be read, not just imported.

**Honesty check:** these are well-known heuristics and reasonable starting
parameters, not a backtested-and-proven edge. Nothing here promises
profitability — see "Known limitations" below.

## Speeding this up further

- **Use `repository_dispatch` for a real 5-minute cadence.** GitHub's own
  cron scheduler queues and delays runs under load. Point a free service
  like [cron-job.org](https://cron-job.org) at
  `POST https://api.github.com/repos/<you>/<repo>/dispatches` with header
  `Authorization: Bearer <a fine-grained PAT with contents+actions write>`
  and body `{"event_type": "run-trade"}` — this fires the workflow directly,
  bypassing GitHub's scheduler queue.
- **Self-hosted runner / VM cron job.** If you have any always-on machine
  (a Raspberry Pi, a free-tier VM, etc.), running `python src/main.py` on a
  real cron job there removes the GitHub Actions scheduling limitation
  entirely and can run every minute if you want.
- **Lower `min_confidence` per agent** in `data/agents.json` if you want more
  trades at the cost of more marginal ones — this is the main "trade more
  often" dial, and it's what the self-tuning loop already nudges based on
  win rate.

## Setup

1. **Create a free Alpaca account**: https://alpaca.markets/ → sign up → go to
   your dashboard → toggle to **Paper Trading** (should be default) → generate
   an API key + secret. No real money or identity verification is required for
   paper trading.

2. **Add secrets to this GitHub repo**:
   Repo → Settings → Secrets and variables → Actions → New repository secret
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`

3. **Push this repo to GitHub.** The workflow in `.github/workflows/trade.yml`
   runs every 5 minutes on weekdays automatically (subject to the GitHub scheduling caveat above), and you can also trigger it manually
   from the **Actions** tab (`workflow_dispatch`).

4. **Watch it run**: after each run, check `data/agents.json` for current
   agent states and `logs/run_log.jsonl` for a full event history (orders,
   spawns, deaths, errors).

## Control app (Android + desktop)

The `app/` folder is an installable web app (PWA) that lets you trigger runs,
watch the dashboard, and kill agents — from your phone or computer. It doesn't
run the trading logic itself (that stays on GitHub Actions, which is reliable
about running on schedule even when your phone is off); it's a remote control
and live view for it.

**Host it** (pick one):
- **GitHub Pages** (easiest): repo → Settings → Pages → Deploy from branch →
  select `main` and folder `/app` → save. You'll get a URL like
  `https://yourname.github.io/paper-trading-agent/`.
- Or open `app/index.html` locally in a browser for desktop-only use.

**Install it on Android:**
1. Open the GitHub Pages URL in Chrome on your phone.
2. Tap the menu (⋮) → **Add to Home screen** / **Install app**.
3. It now opens full-screen from your home screen like a native app.

**Install it on desktop:**
1. Open the same URL in Chrome or Edge.
2. Click the install icon in the address bar (or menu → **Install Agent Colony Control**).

**Connect it to your repo:**
1. Create a GitHub Personal Access Token: https://github.com/settings/tokens/new
   → check only the **repo** and **workflow** scopes → generate.
2. Open the app → enter your repo as `yourname/paper-trading-agent` and paste
   the token. It's stored only in the browser's local storage on that device,
   never sent anywhere except GitHub's API.
3. Tap **Run Agent Now** to trigger a real trading cycle on demand, or wait
   for the schedule (or, better, set up the repository_dispatch pinger above). Tap **Refresh Data** to pull the latest agent
   states and event log. Tap **Kill this agent** on any card to stop a
   specific agent's trading manually.

Note: a classic PAT with `repo`+`workflow` scope can trigger workflows and
push to any of your repos it has access to — treat it like a password. If
you'd rather scope it tighter, GitHub also supports fine-grained tokens
limited to just this one repository.

## Guardrails built in (please don't remove these)

- `broker.py` refuses to run if `ALPACA_BASE_URL` doesn't contain "paper".
- Hard cap of 8 total agents (`MAX_AGENTS` in `agent.py`).
- Per-agent drawdown kill-switch at 20%.
- Position sizing is capped at 15% of an agent's virtual capital per trade.
- ATR-based stop-loss/take-profit checked on every open position every cycle.
- Daily loss circuit breaker: pauses *new entries* (not exits) if an agent's
  realized loss in a rolling 24h window exceeds 8% of its starting capital.
- Correlation-cluster exposure cap: an agent can't hold more than 2
  simultaneous open positions within a correlated group (e.g. BTC/ETH/SOL).

These were kept (and, for the risk ones, extended) rather than loosened —
"make it faster/add more features" was read as "make it trade and adapt
more capably," not "remove the safety rails." If you genuinely want to
change any of these, they're plain constants at the top of `agent.py` /
`crew.py` / `strategy.py`.

## Going live later

When/if you decide to connect this to a real account, that should be a
deliberate, separate decision — not a config flag flip. At minimum:
review months of paper performance, add real risk controls (max daily loss,
circuit breakers, manual trade approval), and start with a very small amount
of real capital. This project intentionally does not include a "go live"
switch.

## Known limitations

- All agents share one Alpaca paper account. Per-agent P&L is estimated by
  this script from each agent's own recorded entry price/notional, not
  pulled directly from Alpaca, since Alpaca only tracks account-wide
  positions.
- The strategy ensemble (MA crossover, RSI, MACD, Bollinger, Stochastic RSI)
  is still a set of standard, publicly-known technical indicators combined
  with simple rules — it is for research/learning purposes and is **not** a
  proven profitable strategy, however many indicators feed into it.
- ATR here is computed from close-to-close price moves (the bars available
  from `get_recent_bars` are close-only), not full high/low true range —
  a standard simplification when OHLC isn't available, but it does make ATR
  somewhat narrower than a textbook ATR would be.
- GitHub Actions cron schedules are not guaranteed to the minute, even with
  the `repository_dispatch` workaround for the *scheduling* side of things —
  Actions runners themselves can still queue under load.
- More agents and more indicators mean more Alpaca API calls per run even
  with the fetch caching in v2; if you hit Alpaca's rate limits, reduce
  `MAX_BAR_FETCH_WORKERS`/`MAX_AGENT_WORKERS` (env vars) or trim the symbol list.

## What's new in this upgrade (summary)

**Strategy / decision-making:** 5-indicator confidence-weighted ensemble
(MA, RSI, MACD, Bollinger %B, Stochastic RSI) · regime detection
(trending vs range-bound) with indicator re-weighting · ATR-based
stop-loss/take-profit · tunable confidence threshold · momentum
confirmation bonus · legacy MA+RSI signal kept for comparison.

**Risk management:** daily loss circuit breaker · correlation-cluster
exposure cap · confidence-scaled position sizing · existing volatility
size-down and per-trade risk ceiling kept and reused by more roles.

**Crew:** 2 new specialist roles (MomentumAnalyst, VolatilityAnalyst) plus a
new StopManager role, all logged with full reasoning.

**Multi-agent / learning:** 4-agent pre-tuned starter colony instead of one
blank-slate agent · self-tuning now adjusts confidence threshold as well as
position size · Sharpe-lite and max-drawdown tracked per agent · sortable
leaderboard (`data/leaderboard.json`).

**Speed:** per-run bar-fetch caching+dedup across agents · concurrent bar
fetching · concurrent agent processing · pip dependency caching in the
workflow · `repository_dispatch` trigger to bypass GitHub's cron queue ·
weekend crypto workflow so BTC/ETH/etc. aren't idle Sat/Sun.

**Ops:** optional Discord/Slack/Telegram notifications on trades, spawns,
deaths, and daily-loss pauses · symbol filtering (`SYMBOL_FILTER` env var)
for the weekend workflow.

**Knowledge base:** 22 documented trading lessons in `src/lessons.py`, each
tied to a concrete enforcement point in the code.
