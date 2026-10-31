import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="US Stock Institutional & Trend Scanner", layout="wide")

# ---------------------------------------------------------------- settings
MIN_CAP = 4_000_000_000      # market cap > $4B
MIN_FLOAT = 450_000_000      # float > 450M shares

# Built-in large-cap universe (yfinance has no "whole market" list).
# Add or remove tickers here; the cap/float filter trims it automatically.
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "NFLX", "AMD",
    "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU", "ORCL", "CRM", "ADBE", "IBM",
    "PLTR", "UBER", "PYPL", "SOFI", "SNAP", "F", "GM", "T", "VZ", "KO",
    "PEP", "WMT", "COST", "DIS", "BAC", "WFC", "C", "JPM", "XOM", "CVX",
    "PFE", "MRK", "ABBV", "JNJ", "BMY", "GILD", "CMCSA", "NKE", "SBUX", "MCD",
    "HD", "LOW", "TGT", "RIVN", "LCID", "HOOD", "COIN", "MARA", "RIOT", "NIO",
    "BABA", "TSM", "ABEV", "ITUB", "KVUE", "CCL", "AAL", "DAL", "UAL", "KMI",
    "WBD", "PARA", "WMB", "OXY", "SLB", "HAL", "BSX", "MDT", "CSX", "NEE",
]

TEXTS = {
    "en": {
        "title": "🇺🇸 US Stock Institutional Flow & Trend Scanner",
        "sidebar_title": "⚙️ Scan Settings",
        "mode_select": "Stock Universe",
        "mode_auto": "Auto scan US large caps (Market Cap > $4B, Float > 450M shares)",
        "mode_manual": "Custom stock tickers",
        "manual_input": "Enter tickers (comma separated, e.g. AAPL, TSLA, NVDA)",
        "period_select": "Scan Timeframe",
        "daily": "Daily",
        "weekly": "Weekly",
        "run_btn": "Run Scanner",
        "keias_buy": "🟢 KEIAS Buy Signals (Breakout 60)",
        "keias_sell": "🔴 KEIAS Sell Signals (Breakdown -60)",
        "kxkx_buy": "🟢 KXKX Institutional Accumulation",
        "kxkx_sell": "🔴 KXKX Institutional Distribution",
        "no_data": "No matching stocks found for this signal",
        "filtering": "Checking market cap & float",
        "downloading": "Downloading price data...",
        "done": "Scan completed",
        "scanned": "Stocks scanned",
        "empty": "No stocks to scan (none passed the filter, or no tickers entered).",
        "caption": "For research only. Not financial advice.",
    },
    "zh": {
        "title": "🇺🇸 美股机构资金与趋势量化扫描系统",
        "sidebar_title": "⚙️ 扫描设置",
        "mode_select": "股票池来源",
        "mode_auto": "自动扫描美股大盘（市值 > 40亿美元，流通股 > 4.5亿）",
        "mode_manual": "自定义输入股票代码",
        "manual_input": "输入美股代码（用逗号分隔，例如：AAPL, TSLA, NVDA）",
        "period_select": "扫描周期",
        "daily": "日线 (Daily)",
        "weekly": "周线 (Weekly)",
        "run_btn": "开始扫描",
        "keias_buy": "🟢 KEIAS 买入信号股 (突破60)",
        "keias_sell": "🔴 KEIAS 卖出信号股 (跌破-60)",
        "kxkx_buy": "🟢 KXKX 机构建仓信号股",
        "kxkx_sell": "🔴 KXKX 机构出货信号股",
        "no_data": "暂无符合条件的股票",
        "filtering": "正在筛选市值与流通股",
        "downloading": "正在下载行情数据...",
        "done": "扫描完成",
        "scanned": "已扫描股票数",
        "empty": "没有可扫描的股票（未通过筛选，或未输入代码）。",
        "caption": "仅供研究参考，不构成投资建议。",
    },
}

# ---------------------------------------------------------------- data
@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {
        "cap": info.get("marketCap") or 0,
        "float": info.get("floatShares") or info.get("sharesOutstanding") or 0,
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers, interval):
    period = "2y" if interval == "1d" else "5y"
    return yf.download(
        list(tickers), period=period, interval=interval,
        group_by="ticker", auto_adjust=True, progress=False, threads=True,
    )


def get_frame(raw, ticker):
    """Pull one ticker's OHLCV out of a (possibly multi-ticker) download."""
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            df = raw[ticker]
        else:
            df = raw.copy()
            df.columns = df.columns.get_level_values(0)
    else:
        df = raw
    if not {"Close", "High", "Low", "Volume"}.issubset(df.columns):
        return None
    return df.dropna(subset=["Close"])


# ---------------------------------------------------------------- indicators
def keias_ias(df, lookback):
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    ll = low.rolling(lookback, min_periods=20).min()
    hh = high.rolling(lookback, min_periods=20).max()
    pos_score = ((close - ll) / (hh - ll).replace(0, np.nan) * 200 - 100).fillna(0)

    ma50 = close.rolling(50).mean()
    trend_score = pd.Series(np.where(close > ma50, 100, -100), index=df.index)

    coef = ((2 * close - high - low) / (high - low).replace(0, np.nan)).fillna(0.001)
    raw_ev = vol * coef
    sd_ev = raw_ev.rolling(30).std().replace(0, np.nan)
    vol_score = ((raw_ev - raw_ev.rolling(30).mean()) / sd_ev * 100).clip(-100, 100).fillna(0)

    raw_ias = pos_score * 0.3 + trend_score * 0.4 + vol_score * 0.3
    return raw_ias.rolling(3).mean()


def kxkx_signals(df):
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    vma20 = vol.rolling(20).mean()
    vma60 = vol.rolling(60).mean()
    amp = (high - low).replace(0, 0.001)
    net_inst = ((close - low) - (high - close)) / amp * vol

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    safe_zone = (close > ema50) | (ema50 > ema200)
    quiet_before = vol.rolling(30).min() < vma60 * 0.5
    near_low = close <= close.rolling(60).min() * 1.15
    surge_buy = (vol > vma20 * 2.0) & (net_inst > 0)
    buy = safe_zone & quiet_before & near_low & surge_buy

    near_high = close >= high.rolling(60).max() * 0.95
    sell = near_high & (vol > vma20 * 1.8) & (net_inst < 0)

    return bool(buy.iloc[-1]), bool(sell.iloc[-1])


# ---------------------------------------------------------------- sidebar
lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"], index=0)
t = TEXTS["en" if lang == "English" else "zh"]

st.title(t["title"])
st.caption(t["caption"])

st.sidebar.header(t["sidebar_title"])
scan_mode = st.sidebar.radio(
    t["mode_select"], ["auto", "manual"],
    format_func=lambda x: t["mode_auto"] if x == "auto" else t["mode_manual"],
)

if scan_mode == "manual":
    raw_input = st.sidebar.text_input(
        t["manual_input"], "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, NFLX, INTC"
    )
    tickers = [s.strip().upper() for s in raw_input.split(",") if s.strip()]
else:
    tickers = UNIVERSE

interval = st.sidebar.selectbox(
    t["period_select"], ["1d", "1wk"],
    format_func=lambda x: t["daily"] if x == "1d" else t["weekly"],
)
lookback = 252 if interval == "1d" else 52  # "52-week" high/low window

# ---------------------------------------------------------------- scan
if st.sidebar.button(t["run_btn"], type="primary"):
    status = st.empty()
    progress = st.progress(0.0)

    universe = list(dict.fromkeys(tickers))  # de-duplicate, keep order

    if scan_mode == "auto":
        kept = []
        for i, tk in enumerate(universe):
            status.info(f"{t['filtering']}: {tk} ({i + 1}/{len(universe)})")
            try:
                f = get_fundamentals(tk)
                if f["cap"] >= MIN_CAP and f["float"] >= MIN_FLOAT:
                    kept.append(tk)
            except Exception:
                pass
            progress.progress((i + 1) / len(universe))
        universe = kept
        progress.progress(0.0)

    if not universe:
        status.empty()
        progress.empty()
        st.session_state.pop("results", None)
        st.warning(t["empty"])
        st.stop()

    status.info(t["downloading"])
    raw = load_prices(tuple(universe), interval)

    results = {"keias_buy": [], "keias_sell": [], "kxkx_buy": [], "kxkx_sell": []}
    for i, tk in enumerate(universe):
        progress.progress((i + 1) / len(universe))
        try:
            df = get_frame(raw, tk)
            if df is None or len(df) < 60:
                continue

            ias = keias_ias(df, lookback)
            if len(ias) > 2:
                now, prev = ias.iloc[-1], ias.iloc[-2]
                if now > 60 and prev <= 60:
                    results["keias_buy"].append(tk)
                elif now < -60 and prev >= -60:
                    results["keias_sell"].append(tk)

            kx_buy, kx_sell = kxkx_signals(df)
            if kx_buy:
                results["kxkx_buy"].append(tk)
            if kx_sell:
                results["kxkx_sell"].append(tk)
        except Exception:
            continue

    status.empty()
    progress.empty()
    st.session_state["results"] = results
    st.session_state["count"] = len(universe)

# ---------------------------------------------------------------- output
results = st.session_state.get("results")
if results:
    st.success(f"{t['done']} — {t['scanned']}: {st.session_state['count']}")

    def show(title, items):
        st.subheader(title)
        st.write(", ".join(items) if items else t["no_data"])

    col1, col2 = st.columns(2)
    with col1:
        show(t["keias_buy"], results["keias_buy"])
        show(t["kxkx_buy"], results["kxkx_buy"])
    with col2:
        show(t["keias_sell"], results["keias_sell"])
        show(t["kxkx_sell"], results["kxkx_sell"])
