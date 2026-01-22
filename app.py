import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# ページ設定
st.set_page_config(page_title="日経225 総合分析ツール", layout="wide")

# ==========================================
# 0. 免責事項
# ==========================================
def display_disclaimer():
    with st.sidebar:
        st.warning("⚠️ **免責事項**")
        st.caption("""
        本ツールは統計データおよびAI予測を表示するもので、投資助言ではありません。
        判断は自己責任で行ってください。
        """)
        return st.checkbox("内容に同意して利用する")

# ==========================================
# 1. 共通データ取得・計算エンジン
# ==========================================
def get_latest(df):
    if df is None or df.empty: return None
    df = df.T.copy()
    try:
        df.index = pd.to_datetime(df.index)
        return df.sort_index().iloc[-1]
    except: return df.iloc[-1]

def safe_val(row, col):
    if row is None or col not in row or pd.isna(row[col]): return np.nan
    return row[col]

def analyze_ticker(symbol):
    """株価統計と財務指標を1セットで取得"""
    t = yf.Ticker(symbol)
    res = {"Ticker": symbol}
    
    # --- A. 株価統計（幾何平均ベース） ---
    try:
        hist = t.history(period="1y")
        if len(hist) > 10:
            p_start = hist['Close'].iloc[0]
            p_end = hist['Close'].iloc[-1]
            g_mean = np.sqrt(p_start * p_end)
            e_profit = p_end - g_mean
            res.update({
                "現在価格": round(p_end, 1),
                "1年後想定価格": round(p_end + e_profit, 1),
                "想定倍率": round(1 + (e_profit / p_end), 3)
            })
    except: pass

    # --- B. 財務分析 ---
    try:
        bs = get_latest(t.balance_sheet)
        is_ = get_latest(t.income_stmt)
        cf = get_latest(t.cash_flow)
        info = t.info
        
        m_cap = info.get("marketCap", np.nan)
        beta = info.get("beta", np.nan)

        if bs is not None and is_ is not None and cf is not None:
            equity = safe_val(bs, "Total Equity Gross Minority Interest")
            assets = safe_val(bs, "Total Assets")
            debt = safe_val(bs, "Total Debt")
            cash = safe_val(bs, "Cash And Cash Equivalents")
            op_inc = safe_val(is_, "Operating Income")
            rev = safe_val(is_, "Total Revenue")
            int_exp = abs(safe_val(is_, "Interest Expense"))
            fcf = safe_val(cf, "Free Cash Flow")

            # 指標計算
            res["自己資本比率"] = round(equity / assets, 3) if assets else np.nan
            res["営業利益率"] = round(op_inc / rev, 3) if rev else np.nan
            res["FCF利回り"] = round(fcf / m_cap, 3) if m_cap else np.nan
            
            # ROIC/WACC
            tax = 0.3
            roic = (op_inc * (1 - tax)) / (equity + debt - cash) if (equity + debt - cash) else np.nan
            cost_e = 0.01 + beta * 0.06 if not np.isnan(beta) else np.nan
            cost_d = (int_exp / debt) * (1 - tax) if debt else 0
            wacc = ((equity / (equity + debt)) * cost_e + (debt / (equity + debt)) * cost_d) if (equity + debt) else np.nan
            
            res["ROIC"] = round(roic, 3)
            res["WACC"] = round(wacc, 3)
            res["ROIC-WACC"] = round(roic - wacc, 3) if not np.isnan(roic) and not np.isnan(wacc) else np.nan
    except: pass
    
    return res

# ==========================================
# 2. AI予測エンジン
# ==========================================
def forecast_mlp(ticker):
    df = yf.download(ticker, period="1y", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df = df.xs("Close", level=0, axis=1)
    else: df = df[["Close"]]
    
    target = df.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(target)
    
    window = 30
    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window].flatten())
        y.append(scaled[i+window])
    
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=0)
    model.fit(np.array(X), np.array(y).ravel())
    
    curr = scaled[-window:].flatten()
    preds = []
    for _ in range(60):
        p = model.predict([curr])[0]
        preds.append(p)
        curr = np.append(curr[1:], p)
    
    return df, pd.DataFrame(scaler.inverse_transform(np.array(preds).reshape(-1, 1)), 
                            index=pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=60))

# ==========================================
# UI部
# ==========================================
st.title("🏛️ 日経225 総合分析ダッシュボード")

if display_disclaimer():
    tab1, tab2, tab3 = st.tabs(["📈 株価予測一覧", "💎 財務分析一覧", "🔍 個別AI詳細"])

    CSV_FILE = "Nikkei225.csv"
    if os.path.exists(CSV_FILE):
        # データの保持（Session Stateを使用して再計算を防ぐ）
        if "master_df" not in st.session_state:
            st.session_state.master_df = None

        if st.sidebar.button("日経225 全銘柄を解析開始"):
            base = pd.read_csv(CSV_FILE)
            tickers = [f"{str(n)}.T" for n in base.iloc[:, 0]]
            
            rows = []
            prog = st.sidebar.progress(0)
            status = st.sidebar.empty()
            
            for i, tk in enumerate(tickers):
                prog.progress((i+1)/len(tickers))
                status.text(f"解析中: {tk}")
                res = analyze_ticker(tk)
                # 元のCSVデータ（社名など）と結合
                full_row = base.iloc[i].to_dict()
                full_row.update(res)
                rows.append(full_row)
            
            st.session_state.master_df = pd.DataFrame(rows)
            status.empty()
            prog.empty()
            st.sidebar.success("解析完了！")

        if st.session_state.master_df is not None:
            df = st.session_state.master_df
            
            # --- Tab 1: 株価予測 ---
            with tab1:
                st.subheader("統計モデルによる価格予測ランキング")
                price_cols = ["Ticker", df.columns[1], "現在価格", "1年後想定価格", "想定倍率"]
                st.dataframe(df[price_cols].sort_values("想定倍率", ascending=False), use_container_width=True)

            # --- Tab 2: 財務分析 ---
            with tab2:
                st.subheader("財務クオリティ（ROIC/WACC/FCF利回り）一覧")
                fin_cols = ["Ticker", df.columns[1], "自己資本比率", "営業利益率", "FCF利回り", "ROIC", "WACC", "ROIC-WACC"]
                st.dataframe(df[fin_cols].sort_values("ROIC-WACC", ascending=False), use_container_width=True)
        else:
            st.info("サイドバーの「解析開始」ボタンを押してください。225銘柄のデータを取得します（数分かかります）。")
            
        # --- Tab 3: 個別AI ---
        with tab3:
            st.subheader("AI（ニューラルネット）による個別銘柄推移予測")
            target_tk = st.text_input("銘柄コードを入力 (例: 7203.T)", "4974.T")
            if st.button("AI詳細分析を実行"):
                with st.spinner("AI学習中..."):
                    hist_df, fore_df = forecast_mlp(target_tk)
                    c1, c2 = st.columns(2)
                    c1.metric("1ヶ月後予測値", f"{fore_df.iloc[20,0]:,.1f}円")
                    
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'NanumGothic']
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(hist_df.index, hist_df.values, label="実績価格")
                    ax.plot(fore_df.index, fore_df.values, label="AI予測", linestyle="--")
                    ax.set_title(f"{target_tk} AI Forecast")
                    ax.legend()
                    st.pyplot(fig)
    else:
        st.error(f"{CSV_FILE} が見つかりません。")