# Paper Trading Agent

A self-tuning, self-"spawning" trading agent that trades **paper money only**
via Alpaca's paper trading API. It cannot place real trades unless you
deliberately change the broker endpoint — which this project actively guards
against (see `src/broker.py`).

## What it does

- Trades a small basket of stocks/ETFs (`SPY`, `QQQ`, `AAPL`) and crypto
  (`BTCUSD`, `ETHUSD`) using a **team of specialist checks** rather than one
  monolithic strategy — see "The Crew" below.
- Tracks its own virtual P&L, win rate, and drawdown.
- **Dies** (stops trading) if its virtual capital drawdown hits 20% from peak.
- Can be **revived** from the control app, restarting with fresh capital.
- **Spawns a child agent** with mutated strategy parameters once it's up 15%
  from its starting capital, capped at 8 total agents so it can't run away.
- Self-tunes position sizing based on recent win rate.
- Runs automatically on a schedule via GitHub Actions (every 5 minutes on
  weekdays) and commits its own state/logs back to the repo so you can watch
  it evolve.

## The Crew

Instead of one strategy function, each trading decision runs through a small
team of specialist checks (`src/crew.py`), all **in parallel**, all within a
hard 5-minute time budget per decision (in practice each one finishes in a
fraction of a second):

- **MarketScanner** — is there enough usable price data for this symbol right now?
- **TechnicalAnalyst** — the momentum/RSI signal (buy/sell/hold)
- **SentimentScanner** — a cheap, rule-based volatility check (flags unusual
  price swings, sizes trades down under it — no news API or LLM call involved)
- **RiskChecker** — enforces a hard per-trade risk ceiling regardless of what
  the strategy asks for
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
   runs hourly on weekdays automatically, and you can also trigger it manually
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
   for the hourly schedule. Tap **Refresh Data** to pull the latest agent
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

## Going live later

When/if you decide to connect this to a real account, that should be a
deliberate, separate decision — not a config flag flip. At minimum:
review months of paper performance, add real risk controls (max daily loss,
circuit breakers, manual trade approval), and start with a very small amount
of real capital. This project intentionally does not include a "go live"
switch.

## Known limitations

- All agents share one Alpaca paper account. Per-agent P&L is estimated by
  this script (price move since the agent's last signal), not pulled directly
  from Alpaca, since Alpaca only tracks account-wide positions.
- The strategy (MA crossover + RSI) is simple and for research/learning
  purposes — it is not a proven profitable strategy.
- GitHub Actions cron schedules are not guaranteed to the minute; expect some
  drift.
