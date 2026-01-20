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
st.set_page_config(page_title="株価分析＆AI予測ツール", layout="wide")

# --- 共通関数：データ取得 ---
def get_stock_raw_data(ticker, current_date):
    start_date = current_date - datetime.timedelta(days=365)
    try:
        data = yf.download(ticker, start=start_date, end=current_date, progress=False)
        if data.empty: return None
        # マルチインデックス対策
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs("Close", level=0, axis=1)
        else:
            data = data[["Close"]]
        return data
    except:
        return None

# --- AI予測ロジック：MLPモデル ---
def forecast_mlp(df, window=30, steps=60):
    target = df.iloc[:, 0].values.reshape(-1, 1)
    
    # 正規化
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(target)
    
    # 学習データ作成
    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window].flatten())
        y.append(scaled[i+window])
    
    X, y = np.array(X), np.array(y).ravel()
    
    # MLPモデル構築・学習
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0)
    model.fit(X, y)
    
    # 未来予測
    preds = []
    current_seq = scaled[-window:].flatten()
    for _ in range(steps):
        p = model.predict([current_seq])[0]
        preds.append(p)
        current_seq = np.append(current_seq[1:], p)
    
    # スケールを元に戻す
    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.DataFrame({"Forecast": forecast.flatten()}, index=future_index)

# --- メイン UI ---
st.title("📈 株価分析＆AI予測システム")

tab1, tab2 = st.tabs(["日経225一括分析", "個別銘柄AI予測"])

# --- Tab 1: 日経225一括分析（幾何平均） ---
with tab1:
    st.header("日経225 幾何平均シミュレーション")
    CSV_FILE = "Nikkei225.csv"

    if os.path.exists(CSV_FILE):
        if st.button("一括計算を開始"):
            df_base = pd.read_csv(CSV_FILE)
            ticker_col = df_base.columns[0]
            tickers = [f"{str(num)}.T" for num in df_base[ticker_col]]
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            now = datetime.datetime.now()

            for idx, t in enumerate(tickers):
                progress_bar.progress((idx + 1) / len(tickers))
                status_text.text(f"処理中: {t}")
                
                data = get_stock_raw_data(t, now)
                if data is not None and len(data) > 10:
                    p_start = float(data.iloc[0])
                    p_end = float(data.iloc[-1])
                    g_mean = np.sqrt(p_start * p_end)
                    exp_price = p_end + (p_end - g_mean)
                    results.append({
                        "コード": t,
                        "現在価格": round(p_end, 1),
                        "1年後想定": round(exp_price, 1),
                        "想定倍率": round(exp_price / p_end, 3)
                    })
            
            status_text.empty()
            progress_bar.empty()
            st.dataframe(pd.DataFrame(results).sort_values("想定倍率", ascending=False), height=500)
    else:
        st.error(f"{CSV_FILE} が見つかりません。")

# --- Tab 2: 個別銘柄AI予測（ニューラルネット） ---
with tab2:
    st.header("AI (ニューラルネットワーク) 詳細予測")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        target_ticker = st.text_input("銘柄コードを入力してください (例: 7203.T, 4974.T)", "4974.T")
        predict_button = st.button("AI予測を実行")

    if predict_button:
        with st.spinner("AIモデルを生成・学習中..."):
            now = datetime.datetime.now()
            df = get_stock_raw_data(target_ticker, now)
            
            if df is not None and len(df) > 50:
                # 予測実行
                forecast_df = forecast_mlp(df)
                
                # 指標計算
                today_p = df.iloc[-1, 0]
                fut_p = forecast_df["Forecast"].iloc[19] # 20ステップ後≒1ヶ月
                change_rate = (fut_p - today_p) / today_p * 100
                
                # 統計の表示
                c1, c2, c3 = st.columns(3)
                c1.metric("現在株価", f"{today_p:,.1f}円")
                c2.metric("1ヶ月後予測", f"{fut_p:,.1f}円")
                c3.metric("予測騰落率", f"{change_rate:+.2f}%")

                # グラフ作成
                st.subheader("予測チャート")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df.index, df.iloc[:, 0], label="Actual (実績)", color="royalblue")
                ax.plot(forecast_df.index, forecast_df["Forecast"], label="Forecast (AI予測)", color="orange", linestyle="--")
                ax.set_title(f"{target_ticker} - AI Prediction Model")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("データの取得に失敗したか、データ量が不足しています。")