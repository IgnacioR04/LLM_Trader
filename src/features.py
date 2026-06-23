"""features.py - Indicadores tecnicos, volatilidad, vol esperada y regimen.

Reglas:
- TODOS los indicadores son backward-looking. NUNCA se usa rolling(center=True)
  ni shift negativo: cero lookahead.
- Se calcula sobre la serie de klines (incluida la ultima vela en formacion,
  que es el estado actual, no informacion futura).
- Los numeros que aqui salen son los que vera el LLM: deben ser fiables.

Ejecutable de forma autonoma:
    python src/features.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cfg import load_config
from fetch_market import fetch_market


# --------------------------------------------------------------------------
# Indicadores base (pandas/numpy puro, sin ta-lib)
# --------------------------------------------------------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def true_range(high, low, close) -> pd.Series:
    prev = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)


def atr(high, low, close, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high, low, close, period: int = 14):
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)) * down.clip(lower=0)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_, plus_di, minus_di


def stochastic(high, low, close, k: int = 14, d: int = 3):
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    pk = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    return pk, pk.rolling(d).mean()


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    return 100 * (close / close.shift(period) - 1)


def bollinger(close: pd.Series, period: int = 20, n: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper, lower = ma + n * sd, ma - n * sd
    width = (upper - lower) / ma
    return ma, upper, lower, width


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def vwap_anchored(high, low, close, volume, window: int) -> pd.Series:
    tp = (high + low + close) / 3
    pv = tp * volume
    return pv.rolling(window).sum() / volume.rolling(window).sum()


# --------------------------------------------------------------------------
# Volatilidad
# --------------------------------------------------------------------------
TF_HOURS = {"1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5,
            "1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24}


def realized_vol_annual(close: pd.Series, tf: str, window: int = 100) -> float:
    """Vol realizada anualizada: std de log-returns * sqrt(periodos/ano)."""
    ret = np.log(close / close.shift(1)).dropna().iloc[-window:]
    if len(ret) < 5:
        return float("nan")
    ppy = (365 * 24) / TF_HOURS[tf]
    return float(ret.std(ddof=1) * math.sqrt(ppy))


def ewma_expected_vol(close: pd.Series, tf: str, lam: float, horizon_hours: float):
    """Vol esperada forward via EWMA (RiskMetrics) sobre log-returns.

    Devuelve (sigma_por_periodo_pct, sigma_horizonte_pct).
    """
    ret = np.log(close / close.shift(1)).dropna()
    var = ret.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    sigma_period = float(np.sqrt(var.iloc[-1]))
    n_periods = horizon_hours / TF_HOURS[tf]
    sigma_h = sigma_period * math.sqrt(n_periods)
    return sigma_period * 100, sigma_h * 100


# --------------------------------------------------------------------------
# Features por timeframe
# --------------------------------------------------------------------------
def _last(s: pd.Series) -> float:
    v = s.iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def timeframe_features(df: pd.DataFrame, tf: str) -> dict:
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])
    price = _last(c)

    ema20, ema50, ema200 = ema(c, 20), ema(c, 50), ema(c, 200)
    ema20_slope = (_last(ema20) - float(ema20.iloc[-6])) / float(ema20.iloc[-6]) * 100 \
        if len(ema20) > 6 else float("nan")
    adx_, pdi, mdi = adx(h, l, c)
    macd_line, macd_sig, macd_hist = macd(c)
    pk, pd_ = stochastic(h, l, c)
    atr_ = atr(h, l, c)
    _, _, _, bb_width = bollinger(c)
    vwap = vwap_anchored(h, l, c, v, min(len(df), 96))
    vol_rel = price and _last(v) / float(v.rolling(20).mean().iloc[-1])

    return {
        "price": price,
        "ema20": _last(ema20),
        "ema50": _last(ema50),
        "ema200": _last(ema200),
        "price_vs_ema20_pct": (price / _last(ema20) - 1) * 100,
        "price_vs_ema50_pct": (price / _last(ema50) - 1) * 100,
        "price_vs_ema200_pct": (price / _last(ema200) - 1) * 100,
        "ema20_slope_pct": ema20_slope,
        "adx": _last(adx_),
        "plus_di": _last(pdi),
        "minus_di": _last(mdi),
        "rsi": _last(rsi(c)),
        "macd_line": _last(macd_line),
        "macd_signal": _last(macd_sig),
        "macd_hist": _last(macd_hist),
        "roc_12": _last(roc(c)),
        "stoch_k": _last(pk),
        "stoch_d": _last(pd_),
        "atr": _last(atr_),
        "atr_pct": _last(atr_) / price * 100,
        "bb_width_pct": _last(bb_width) * 100,
        "realized_vol_annual_pct": realized_vol_annual(c, tf) * 100,
        "vwap": _last(vwap),
        "price_vs_vwap_pct": (price / _last(vwap) - 1) * 100,
        "obv": _last(obv(c, v)),
        "vol_rel_20": float(vol_rel) if vol_rel else float("nan"),
        "last_candle_close": df.index[-1].isoformat(),
    }


# --------------------------------------------------------------------------
# Order book
# --------------------------------------------------------------------------
def orderbook_features(ob: dict, bands_pct: list[float]) -> dict:
    bids = np.array(ob["bids"], dtype=float)  # precio descendente
    asks = np.array(ob["asks"], dtype=float)  # precio ascendente
    best_bid, bid0 = bids[0]
    best_ask, ask0 = asks[0]
    mid = (best_bid + best_ask) / 2
    microprice = (best_bid * ask0 + best_ask * bid0) / (bid0 + ask0)
    spread = best_ask - best_bid

    def imb(n: int) -> float:
        b, a = bids[:n, 1].sum(), asks[:n, 1].sum()
        return float((b - a) / (b + a)) if (b + a) else 0.0

    def band(pct: float) -> dict:
        lo, hi = mid * (1 - pct), mid * (1 + pct)
        b = float(bids[bids[:, 0] >= lo][:, 1].sum())
        a = float(asks[asks[:, 0] <= hi][:, 1].sum())
        return {"bid_depth": b, "ask_depth": a,
                "imbalance": (b - a) / (b + a) if (b + a) else 0.0}

    def walls(side: np.ndarray, n: int = 3) -> list:
        sizes = side[:, 1]
        med = np.median(sizes)
        mad = np.median(np.abs(sizes - med)) or 1e-9
        idx = np.where(sizes > med + 4 * mad)[0]
        idx = idx[np.argsort(sizes[idx])[::-1]][:n]
        return [{"price": float(side[i, 0]), "size": float(side[i, 1])} for i in idx]

    # cobertura real del libro (BTC perp ~ +-0.2% del mid via REST)
    coverage_pct = float((mid - bids[-1, 0]) / mid * 100)
    return {
        "mid": float(mid),
        "microprice": float(microprice),
        "spread_abs": float(spread),
        "spread_bps": float(spread / mid * 1e4),
        "levels_per_side": int(len(bids)),
        "coverage_pct": coverage_pct,
        "imbalance_top5": imb(5),
        "imbalance_top20": imb(20),
        "bands": {f"{p*100:g}%": band(p) for p in bands_pct},
        "bid_walls": walls(bids),
        "ask_walls": walls(asks),
    }


# --------------------------------------------------------------------------
# Regimen
# --------------------------------------------------------------------------
def regime(df: pd.DataFrame, tf_feats: dict, cfg: dict) -> dict:
    h, l, c = df["high"], df["low"], df["close"]
    adx_val = tf_feats["adx"]
    if adx_val >= 25:
        trend = "tendencia"
    elif adx_val < 20:
        trend = "rango"
    else:
        trend = "transicion"

    # direccion por estructura de EMAs
    if tf_feats["price_vs_ema50_pct"] > 0 and tf_feats["ema50"] > tf_feats["ema200"]:
        direction = "alcista"
    elif tf_feats["price_vs_ema50_pct"] < 0 and tf_feats["ema50"] < tf_feats["ema200"]:
        direction = "bajista"
    else:
        direction = "mixta"

    # vol por percentil de ATR
    atr_series = atr(h, l, c)
    win = cfg["volatility"]["atr_percentile_window"]
    recent = atr_series.dropna().iloc[-win:]
    cur = atr_series.iloc[-1]
    pct = float((recent < cur).mean() * 100) if len(recent) else float("nan")
    vol_label = "alta" if pct >= 70 else "baja" if pct <= 30 else "media"

    return {
        "trend": trend,
        "direction": direction,
        "adx": adx_val,
        "atr_percentile": pct,
        "vol": vol_label,
        "label": f"{trend} {direction} / vol {vol_label}",
    }


# --------------------------------------------------------------------------
# Ensamblado
# --------------------------------------------------------------------------
def compute_features(market: dict, cfg: dict) -> dict:
    base_tf = cfg["market"]["base_timeframe"]
    use_closed = cfg["market"].get("use_closed_candles", True)

    # Indicadores sobre velas CERRADAS: descartamos la ultima vela en formacion.
    def closed(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[:-1] if use_closed and len(df) > 1 else df

    klines = {tf: closed(df) for tf, df in market["klines"].items()}
    tf_feats = {tf: timeframe_features(df, tf) for tf, df in klines.items()}

    sig_p, sig_h = ewma_expected_vol(
        klines[base_tf]["close"],
        base_tf,
        cfg["volatility"]["ewma_lambda"],
        cfg["prediction"]["horizon_hours"],
    )

    ob = orderbook_features(market["orderbook"], cfg["market"]["orderbook_bands_pct"])
    # precio LIVE (order book) vs ultimo cierre del timeframe base
    live_price = ob["mid"]
    base_close = tf_feats[base_tf]["price"]

    return {
        "exec_ts": market["exec_ts"].isoformat(),
        "symbol": market["symbol"],
        "live_price": live_price,
        "indicators_on": "velas_cerradas" if use_closed else "incluye_vela_actual",
        "live_vs_base_close_pct": (live_price / base_close - 1) * 100,
        "funding": market["funding"],
        "orderbook": ob,
        "timeframes": tf_feats,
        "expected_vol": {
            "method": "EWMA(RiskMetrics)",
            "base_timeframe": base_tf,
            "sigma_per_period_pct": sig_p,
            "horizon_hours": cfg["prediction"]["horizon_hours"],
            "sigma_horizon_pct": sig_h,
        },
        "regime": regime(klines[base_tf], tf_feats[base_tf], cfg),
    }


def _print(feats: dict) -> None:
    ob = feats["orderbook"]
    print("=" * 70)
    print("FEATURES - datos reales")
    print("=" * 70)
    print(f"exec_ts        : {feats['exec_ts']}")
    print(f"live_price     : {feats['live_price']:,.2f}  "
          f"(vs cierre base {feats['live_vs_base_close_pct']:+.2f}%) | "
          f"indicadores: {feats['indicators_on']}")
    print(f"\n[ORDER BOOK]  {ob['levels_per_side']} niveles/lado, "
          f"cubre +-{ob['coverage_pct']:.2f}% del mid")
    print(f"  mid={ob['mid']:,.2f}  micro={ob['microprice']:,.2f}  "
          f"spread={ob['spread_bps']:.2f} bps")
    print(f"  imbalance top5={ob['imbalance_top5']:+.3f}  "
          f"top20={ob['imbalance_top20']:+.3f}")
    for k, b in ob["bands"].items():
        print(f"  banda +-{k:>4}: imb={b['imbalance']:+.3f}  "
              f"bid={b['bid_depth']:.1f}  ask={b['ask_depth']:.1f}")
    print(f"  bid_walls={ob['bid_walls']}")
    print(f"  ask_walls={ob['ask_walls']}")
    for tf, f in feats["timeframes"].items():
        print(f"\n[TECNICO {tf}]  (cierre vela: {f['last_candle_close']})")
        print(f"  EMA20={f['ema20']:,.0f} EMA50={f['ema50']:,.0f} "
              f"EMA200={f['ema200']:,.0f} | px vs EMA50={f['price_vs_ema50_pct']:+.2f}%")
        print(f"  ADX={f['adx']:.1f} (+DI={f['plus_di']:.1f} -DI={f['minus_di']:.1f}) "
              f"| RSI={f['rsi']:.1f} | ROC12={f['roc_12']:+.2f}%")
        print(f"  MACD hist={f['macd_hist']:+.2f} | Stoch k={f['stoch_k']:.1f} "
              f"d={f['stoch_d']:.1f}")
        print(f"  ATR={f['atr']:,.1f} ({f['atr_pct']:.2f}%) | BBwidth={f['bb_width_pct']:.2f}% "
              f"| RV30={f['realized_vol_annual_pct']:.1f}%")
        print(f"  VWAP={f['vwap']:,.0f} (px {f['price_vs_vwap_pct']:+.2f}%) | "
              f"vol_rel={f['vol_rel_20']:.2f}")
    ev = feats["expected_vol"]
    print(f"\n[VOL ESPERADA] {ev['method']} base={ev['base_timeframe']}: "
          f"{ev['sigma_per_period_pct']:.2f}%/periodo -> "
          f"{ev['sigma_horizon_pct']:.2f}% en {ev['horizon_hours']}h")
    r = feats["regime"]
    print(f"\n[REGIMEN] {r['label']}  "
          f"(ADX={r['adx']:.1f}, ATR pctil={r['atr_percentile']:.0f})")


if __name__ == "__main__":
    cfg = load_config()
    market = fetch_market(cfg)
    feats = compute_features(market, cfg)
    _print(feats)
