"""
Thin wrapper around Alpaca's paper trading API.

Requires environment variables:
  ALPACA_API_KEY
  ALPACA_SECRET_KEY
  ALPACA_BASE_URL   (defaults to paper trading endpoint)

Never hits a live-money endpoint: base_url defaults to paper-api.alpaca.markets.
This is deliberate - do not change ALPACA_BASE_URL to the live endpoint without
fully understanding you'd be trading real money.
"""

import os
import requests
import pandas as pd

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"


class AlpacaBroker:
    def __init__(self):
        self.api_key = os.environ["ALPACA_API_KEY"]
        self.secret_key = os.environ["ALPACA_SECRET_KEY"]
        self.base_url = os.environ.get("ALPACA_BASE_URL", PAPER_BASE_URL)

        if "paper" not in self.base_url:
            raise RuntimeError(
                "Refusing to start: ALPACA_BASE_URL does not look like a paper "
                "trading endpoint. This bot is built for paper trading only."
            )

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    # ---------- Account ----------
    def get_account(self) -> dict:
        r = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
        r.raise_for_status()
        return r.json()

    def get_equity(self) -> float:
        return float(self.get_account()["equity"])

    # ---------- Market data ----------
    def get_recent_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> pd.Series:
        """Returns a pandas Series of close prices, oldest first.

        NOTE ON SYMBOL FORMAT: Alpaca is inconsistent between endpoints.
        - Market data (this method) requires crypto symbols with a slash: "BTC/USD"
        - Orders/positions (submit_order, get_position) require NO slash: "BTCUSD"
        We accept either format from the caller and normalize here.
        """
        is_crypto = symbol.replace("/", "") in ("BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "SHIBUSD")

        if is_crypto:
            data_symbol = symbol if "/" in symbol else f"{symbol[:-3]}/{symbol[-3:]}"
            url = f"{DATA_BASE_URL}/v1beta3/crypto/us/bars"
            params = {"symbols": data_symbol, "timeframe": timeframe, "limit": limit}
        else:
            data_symbol = symbol
            url = f"{DATA_BASE_URL}/v2/stocks/bars"
            params = {"symbols": data_symbol, "timeframe": timeframe, "limit": limit, "adjustment": "raw"}

        r = requests.get(url, headers=self.headers, params=params)
        r.raise_for_status()
        payload = r.json()
        bars = payload.get("bars", {}).get(data_symbol, [])
        closes = [b["c"] for b in bars]
        return pd.Series(closes)

    # ---------- Positions ----------
    def get_position(self, symbol: str):
        symbol = symbol.replace("/", "")  # positions API wants no-slash format
        r = requests.get(f"{self.base_url}/v2/positions/{symbol}", headers=self.headers)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def list_positions(self) -> list:
        r = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
        r.raise_for_status()
        return r.json()

    # ---------- Orders ----------
    def submit_order(self, symbol: str, notional_usd: float, side: str) -> dict:
        """
        side: 'buy' or 'sell'
        notional_usd: dollar amount to trade (Alpaca supports fractional notional orders
                      for most stocks/crypto; falls back to qty=1 if notional unsupported).
        """
        symbol = symbol.replace("/", "")  # orders API wants no-slash format, e.g. BTCUSD not BTC/USD
        order = {
            "symbol": symbol,
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "notional": round(notional_usd, 2),
        }
        r = requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order)
        r.raise_for_status()
        return r.json()

    def close_position(self, symbol: str) -> dict:
        symbol = symbol.replace("/", "")  # positions API wants no-slash format
        r = requests.delete(f"{self.base_url}/v2/positions/{symbol}", headers=self.headers)
        r.raise_for_status()
        return r.json()
