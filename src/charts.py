"""charts.py - Genera graficos de velas por CODIGO (no capturas).

Reproducibilidad total: mismas velas -> mismo PNG. Cada grafico se guarda con
el timestamp real de ejecucion en el nombre. Por defecto 4h y 1h, ~100 velas,
con EMA20/EMA50 superpuestas para que el LLM lea estructura (soportes,
resistencias, tendencia).

Ejecutable de forma autonoma:
    python src/charts.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import mplfinance as mpf

from cfg import load_config, DATA_DIR
from fetch_market import fetch_market


_RENAME = {"open": "Open", "high": "High", "low": "Low",
           "close": "Close", "volume": "Volume"}

_STYLE = mpf.make_mpf_style(
    base_mpf_style="binance",
    rc={"font.size": 9, "axes.titlesize": 11},
)


def _ts_tag(exec_ts) -> str:
    """2026-06-23T14:11:05+00:00 -> 2026-06-23_1411 (UTC, minuto)."""
    return pd.Timestamp(exec_ts).strftime("%Y-%m-%d_%H%M")


def make_chart(df: pd.DataFrame, symbol: str, tf: str, candles: int,
               out_path: Path) -> Path:
    """Dibuja un grafico de velas + EMA20/EMA50 + volumen y lo guarda en PNG."""
    # EMAs sobre la serie COMPLETA (evita artefactos de borde), luego recorte
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    view = df.tail(candles).rename(columns=_RENAME)
    ema20, ema50 = ema20.tail(candles), ema50.tail(candles)

    addplots = [
        mpf.make_addplot(ema20, color="#2962FF", width=1.0),
        mpf.make_addplot(ema50, color="#FF6D00", width=1.0),
    ]
    last_close = float(view["Close"].iloc[-1])
    title = f"{symbol}  {tf}  (ultimas {len(view)} velas)  close={last_close:,.0f}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mpf.plot(
        view,
        type="candle",
        volume=True,
        addplot=addplots,
        style=_STYLE,
        title=title,
        ylabel="",
        ylabel_lower="",
        figratio=(16, 9),
        figscale=1.1,
        tight_layout=True,
        savefig=dict(fname=str(out_path), dpi=110, bbox_inches="tight"),
    )
    return out_path


def make_charts(market: dict, cfg: dict, out_dir: Path | None = None) -> list[Path]:
    """Genera un PNG por cada timeframe de cfg['chart']['timeframes']."""
    out_dir = out_dir or (DATA_DIR / "charts")
    symbol = market["symbol"]
    tag = _ts_tag(market["exec_ts"])
    sym_slug = symbol.replace("/", "").replace(":", "")
    candles = cfg["chart"]["candles"]

    paths = []
    for tf in cfg["chart"]["timeframes"]:
        df = market["klines"][tf]
        out_path = out_dir / f"{sym_slug}_{tf}_{tag}.png"
        paths.append(make_chart(df, symbol, tf, candles, out_path))
    return paths


if __name__ == "__main__":
    cfg = load_config()
    market = fetch_market(cfg)
    paths = make_charts(market, cfg)
    print("=" * 70)
    print("CHARTS - PNG generados")
    print("=" * 70)
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"  {p}  ({size_kb:.0f} KB)")
