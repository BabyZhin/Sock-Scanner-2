import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page Configuration (Optimized for iPad & Mobile browsers)
st.set_page_config(
    page_title="US Stock Institutional & Trend Quantitative Scanner",
    layout="wide"
)

# 1. Multi-language Dictionary
TEXTS = {
    "en": {
        "title": "🇺🇸 US Stock Institutional Flow & Trend Scanner",
        "sidebar_title": "⚙️ Scan Settings",
        "lang_select": "Language / 语言",
        "mode_select": "Stock Universe",
        "mode_auto": "Auto Scan US Market (Market Cap > $4B, Volume > 100M)",
        "mode_manual": "Custom Stock Tickers",
        "manual_input": "Enter tickers (comma separated, e.g., AAPL, TSLA, NVDA)",
        "period_select": "Scan Timeframe",
        "daily": "Daily",
        "weekly": "Weekly",
        "run_btn": "Run Scanner",
        "keias_buy": "🟢 KEIAS Buy Signals (Breakout 60)",
        "keias_sell": "🔴 KEIAS Sell Signals (Breakdown -60)",
        "kxkx_buy": "🟢 KXKX Institutional Accumulation",
        "kxkx_sell": "🔴 KXKX Institutional Distribution",
        "no_data": "No matching stocks found for this signal",
    },
    "zh": {
        "title": "🇺🇸 美股机构资金与趋势量化扫描系统",
        "sidebar_title": "⚙️ 扫描设置",
        "lang_select": "选择语言 / Language",
        "mode_select": "股票池来源",
        "mode_auto": "自动扫描美股大盘（市值 > 40亿，流通股 > 1亿）",
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
    }
}

# Language Selector
lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"], index=0)
lang_key = "en" if lang == "English" else "zh"
t = TEXTS[lang_key]

st.title(t["title"])

# 2. Sidebar Controls
st.sidebar.header(t["sidebar_title"])
scan_mode = st.sidebar.radio(t["mode_select"], [t["mode_auto"], t["mode_manual"]])

if scan_mode == t["mode_manual"]:
    custom_tickers = st.sidebar.text_input(
        t["manual_input"], 
        "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, NFLX, INTC"
    )
    tickers = [s.strip().upper() for s in custom_tickers.split(",") if s.strip()]
else:
    # Preset representative list of large-cap US stocks (Market Cap > $4B)
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
        "AVGO", "COST", "PEP", "TMUS", "CSCO", "TM", "CRM", "QCOM", "TXN", "AMAT"
    ]

timeframe = st.sidebar.selectbox(t["period_select"], [t["daily"], t["weekly"]])
interval_val = "1d" if timeframe == t["daily"] else "1wk"

if st.sidebar.button(t["run_btn"]):
    st.info(f"Scanning {len(tickers)} stocks... Timeframe: {timeframe}")
    
    keias_buy_list, keias_sell_list = [], []
    kxkx_buy_list, kxkx_sell_list = [], []
    
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        try:
            # Download stock data via yfinance
            df = yf.download(ticker, period="1y", interval=interval_val, progress=False)
            if df.empty or len(df) < 60:
                continue
            
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            close = df['Close']
            high = df['High']
            low = df['Low']
            vol = df['Volume']
            
            # --- Indicator 1: KEIAS Logic ---
            ll52 = low.rolling(252).min() if len(df) >= 252 else low.min()
            hh52 = high.rolling(252).max() if len(df) >= 252 else high.max()
            pos_score = np.where(hh52 == ll52, 0, (close - ll52) / (hh52 - ll52) * 200 - 100)
            
            ma50 = close.rolling(50).mean()
            trend_score = np.where(close > ma50, 100, -100)
            
            coef = np.where(high == low, 0.001, (2 * close - high - low) / (high - low))
            raw_ev = vol * coef
            sd_ev = raw_ev.rolling(30).std()
            vol_score = np.where(sd_ev == 0, 0, (raw_ev - raw_ev.rolling(30).mean()) / sd_ev * 100)
            vol_score_lim = np.clip(vol_score, -100, 100)
            
            raw_ias = pos_score * 0.3 + trend_score * 0.4 + vol_score_lim * 0.3
            ias = pd.Series(raw_ias, index=df.index).rolling(3).mean()
            
            # KEIAS Signal Checking
            if len(ias) > 2:
                if ias.iloc[-1] > 60 and ias.iloc[-2] <= 60:
                    keias_buy_list.append(ticker)
                elif ias.iloc[-1] < -60 and ias.iloc[-2] >= -60:
                    keias_sell_list.append(ticker)
            
            # --- Indicator 2: KXKX Logic ---
            vma20 = vol.rolling(20).mean()
            vma60 = vol.rolling(60).mean()
            amp = np.where((high - low) == 0, 0.001, high - low)
            pos_kx = ((close - low) - (high - close)) / amp
            net_inst = pos_kx * vol
            
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()
            safe_zone = (close > ema50) | (ema50 > ema200)
            
            zqxsl = (vol.rolling(30).min() < vma60 * 0.5)
            zcxdw = (close <= close.rolling(60).min() * 1.15)
            szjbf = (vol > vma20 * 2.0) & (net_inst > 0)
            
            inst_buy_signal = safe_zone & zqxsl & zcxdw & szjbf
            if len(inst_buy_signal) > 0 and inst_buy_signal.iloc[-1]:
                kxkx_buy_list.append(ticker)
                
            zcxgw = (close >= high.rolling(60).max() * 0.95)
            inst_sell_signal = zcxgw & (vol > vma20 * 1.8) & (net_inst < 0)
            if len(inst_sell_signal) > 0 and inst_sell_signal.iloc[-1]:
                kxkx_sell_list.append(ticker)
                
        except Exception:
            pass
            
        progress_bar.progress((idx + 1) / len(tickers))
        
    # 3. Output Panels
    st.success("Scan Completed!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t["keias_buy"])
        st.write(keias_buy_list if keias_buy_list else t["no_data"])
        
        st.subheader(t["kxkx_buy"])
        st.write(kxkx_buy_list if kxkx_buy_list else t["no_data"])
        
    with col2:
        st.subheader(t["keias_sell"])
        st.write(keias_sell_list if keias_sell_list else t["no_data"])
        
        st.subheader(t["kxkx_sell"])
        st.write(kxkx_sell_list if kxkx_sell_list else t["no_data"])