"""fetch_market.py - Descarga el estado de mercado por REST publico (ccxt).

- Sin claves de exchange: solo datos publicos (order book + klines + funding).
- Todo anclado al timestamp REAL de ejecucion en UTC (no al del cron).
- Sin lookahead: solo se piden datos hasta "ahora".

Ejecutable de forma autonoma para inspeccion:
    python src/fetch_market.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import ccxt

from cfg import load_config


def get_exchange(cfg: dict):
    """Instancia el exchange ccxt segun config (spot o futures USDT-M)."""
    market = cfg["market"]
    klass = getattr(ccxt, market["exchange"])
    opts = {"enableRateLimit": True}
    if market["market_type"] == "future":
        opts["options"] = {"defaultType": "future"}
    return klass(opts)


def fetch_orderbook(ex, symbol: str, depth: int) -> dict:
    """Order book top-`depth` por lado. bids descendente, asks ascendente."""
    ob = ex.fetch_order_book(symbol, limit=depth)
    return {
        "bids": [[float(p), float(q)] for p, q in ob["bids"][:depth]],
        "asks": [[float(p), float(q)] for p, q in ob["asks"][:depth]],
        "exchange_ts": ob.get("timestamp"),
    }


def fetch_klines(ex, symbol: str, timeframes: list[str], limit: int) -> dict:
    """Devuelve {timeframe: DataFrame[open,high,low,close,volume]} indexado UTC.

    Nota: la ULTIMA vela de cada timeframe esta en formacion (incompleta).
    Es informacion actual y valida (no es lookahead), pero los indicadores
    sobre esa vela se moveran hasta su cierre.
    """
    out: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        raw = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(
            raw, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("dt").drop(columns="ts").astype(float)
        out[tf] = df
    return out


def fetch_funding(ex, symbol: str) -> dict:
    """Funding rate / mark / index para perp. Tolerante a fallos."""
    try:
        fr = ex.fetch_funding_rate(symbol)
        return {
            "funding_rate": fr.get("fundingRate"),
            "mark_price": fr.get("markPrice"),
            "index_price": fr.get("indexPrice"),
            "next_funding_ts": fr.get("fundingTimestamp")
            or fr.get("nextFundingTimestamp"),
        }
    except Exception as e:  # pragma: no cover - dependiente del exchange
        return {"error": f"{type(e).__name__}: {e}"}


def fetch_market(cfg: dict) -> dict:
    """Captura completa del estado de mercado anclada al timestamp real."""
    ex = get_exchange(cfg)
    market = cfg["market"]
    symbol = market["symbol"]

    exec_ts = datetime.now(timezone.utc)  # ancla temporal real

    ticker = ex.fetch_ticker(symbol)
    last_price = ticker.get("last") or ticker.get("close")

    ob = fetch_orderbook(ex, symbol, market["orderbook_depth"])
    klines = fetch_klines(
        ex, symbol, market["timeframes"], market["klines_limit"]
    )

    funding = (
        fetch_funding(ex, symbol) if market["market_type"] == "future" else None
    )

    return {
        "exec_ts": exec_ts,
        "symbol": symbol,
        "exchange": market["exchange"],
        "market_type": market["market_type"],
        "last_price": float(last_price) if last_price is not None else None,
        "orderbook": ob,
        "klines": klines,  # dict[tf -> DataFrame]
        "funding": funding,
    }


def _summary(m: dict) -> str:
    lines = []
    lines.append(f"exec_ts (UTC) : {m['exec_ts'].isoformat()}")
    lines.append(
        f"symbol        : {m['symbol']} @ {m['exchange']} ({m['market_type']})"
    )
    lines.append(f"last_price    : {m['last_price']:,.2f}")
    ob = m["orderbook"]
    bb, ba = ob["bids"][0], ob["asks"][0]
    lines.append(
        f"order book    : {len(ob['bids'])} bids / {len(ob['asks'])} asks | "
        f"best_bid={bb[0]:,.2f} ({bb[1]:.3f})  best_ask={ba[0]:,.2f} ({ba[1]:.3f})"
    )
    if m["funding"] and "error" not in m["funding"]:
        f = m["funding"]
        fr = f.get("funding_rate")
        mp = f.get("mark_price")
        lines.append(
            f"funding       : rate={fr:+.6f}  mark={mp:,.2f}"
            if fr is not None
            else f"funding       : {f}"
        )
    elif m["funding"]:
        lines.append(f"funding       : {m['funding']}")
    for tf, df in m["klines"].items():
        last = df.iloc[-1]
        lines.append(
            f"klines {tf:>3}    : {len(df)} velas | "
            f"[{df.index[0].date()} -> {df.index[-1]}] | "
            f"last close={last['close']:,.2f} vol={last['volume']:,.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = load_config()
    m = fetch_market(cfg)
    print("=" * 70)
    print("FETCH_MARKET - datos reales")
    print("=" * 70)
    print(_summary(m))
