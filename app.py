import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# ページ設定（結果を見やすくするためワイドモードに設定）
st.set_page_config(page_title="株価予測・分析ツール", layout="wide")

# ==========================================
# 投資助言業に該当しないためのガイドライン・注意喚起
# ==========================================
def display_disclaimer():
    with st.sidebar:
        st.warning("⚠️ **利用上の注意・免責事項**")
        st.caption("""
        本ツールは統計的手法（ニューラルネットワーク）を用いたデータ解析結果を表示するものであり、
        特定の銘柄の売買を推奨する「投資助言」ではありません。
        
        以下の点に同意の上、参考情報としてご利用ください。
        1. **自己責任の原則**: 実際の投資判断はご自身の責任で行ってください。
        2. **正確性の非保証**: 過去のデータに基づく計算であり、将来の成果を保証しません。
        3. **非助言性**: 投資の時期、価格、銘柄の選択について個別具体的な助言は行いません。
        """)
        
        if st.checkbox("上記の内容を理解し、同意します"):
            st.success("ツールをご利用いただけます")
            return True
        else:
            st.info("同意いただける場合のみ、計算結果を参考にしてください。")
            return False

# ==========================================
# 1. 一括計算用ロジック（幾何平均）
# ==========================================
def get_stock_data_stats(ticker, current_date):
    end_date = current_date
    start_date = end_date - datetime.timedelta(days=365)
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(stock_data) < 2:
            return None
        
        # Closeデータの抽出
        if isinstance(stock_data.columns, pd.MultiIndex):
            price_series = stock_data['Close'][ticker]
        else:
            price_series = stock_data['Close']

        price_start = float(price_series.iloc[0])
        price_end = float(price_series.iloc[-1])

        # 幾何平均計算
        geometric_mean = np.sqrt(price_start * price_end)
        expected_profit = price_end - geometric_mean
        expected_price = price_end + expected_profit
        expected_interest_rate = 1 + (expected_profit / price_end)

        return {
            "始値（1年前）": round(price_start, 1),
            "終値（現在）": round(price_end, 1),
            "1年後の想定価格": round(expected_price, 1),
            "想定倍率": round(expected_interest_rate, 3)
        }
    except:
        return None

# ==========================================
# 2. AI予測用ロジック（MLP）
# ==========================================
def get_stock_raw_data(ticker, current_date):
    start_date = current_date - datetime.timedelta(days=365)
    data = yf.download(ticker, start=start_date, end=current_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs("Close", level=0, axis=1)
    else:
        data = data[["Close"]]
    return data

def forecast_mlp(df, window=30, steps=60):
    target = df.iloc[:, 0].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(target)
    
    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window].flatten())
        y.append(scaled[i+window])

    X, y = np.array(X), np.array(y).ravel()
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0)
    model.fit(X, y)
    
    preds = []
    current_seq = scaled[-window:].flatten()
    for _ in range(steps):
        p = model.predict([current_seq])[0]
        preds.append(p)
        current_seq = np.append(current_seq[1:], p)
        
    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.DataFrame({"Forecast": forecast.flatten()}, index=future_index)

# ==========================================
# メインUI
# ==========================================
st.title("📈 株価予測・分析ツール")

tab1, tab2 = st.tabs(["日経225一括分析表示", "個別銘柄AI詳細予測"])

# --- Tab 1: 日経225一括分析（表示のみ） ---
with tab1:
    st.header("日経225 銘柄別予測一覧")
    CSV_FILE = "Nikkei225.csv"

    if os.path.exists(CSV_FILE):
        if st.button("全銘柄の計算を実行"):
            df_base = pd.read_csv(CSV_FILE)
            ticker_col = df_base.columns[0]
            tickers = [f"{str(num)}.T" for num in df_base[ticker_col]]

            # 結果格納用
            results_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_date = datetime.datetime.now()

            for idx, ticker in enumerate(tickers):
                progress_bar.progress((idx + 1) / len(tickers))
                status_text.text(f"計算中... ({idx + 1}/{len(tickers)}): {ticker}")
                
                res = get_stock_data_stats(ticker, current_date)
                if res:
                    # 元のCSV情報と計算結果を結合
                    row_data = df_base.iloc[idx].to_dict()
                    row_data.update(res)
                    results_list.append(row_data)

            status_text.empty()
            progress_bar.empty()

            # データフレーム化して表示
            results_df = pd.DataFrame(results_list)
            
            st.subheader("📊 予測結果ランキング（想定倍率順）")
            # 想定倍率で降順ソートして表示
            st.dataframe(
                results_df.sort_values(by="想定倍率", ascending=False), 
                height=600, 
                use_container_width=True
            )
    else:
        st.error(f"エラー: `{CSV_FILE}` が見つかりません。")

# --- Tab 2: 個別銘柄AI予測 ---
with tab2:
    st.header("AI（ニューラルネット）詳細チャート")
    ticker_input = st.text_input("銘柄コードを入力 (例: 7203.T)", value="4974.T")
    
    if st.button("AI予測チャートを表示"):
        with st.spinner('AIが学習・分析中...'):
            current_date = datetime.datetime.now()
            df = get_stock_raw_data(ticker_input, current_date)
            
            if not df.empty:
                forecast_df = forecast_mlp(df)
                
                today_price = float(df.iloc[-1, 0])
                future_price = float(forecast_df["Forecast"].iloc[19]) 
                future_change = (future_price - today_price) / today_price * 100

                # 指標をタイル表示
                c1, c2, c3 = st.columns(3)
                c1.metric("現在の株価", f"{today_price:,.1f}円")
                c2.metric("1ヶ月後予測価格", f"{future_price:,.1f}円")
                c3.metric("予測騰落率", f"{future_change:+.2f}%")

                # グラフ（日本語対応）
               # グラフ表示の部分
                st.subheader(f"【{ticker_input}】 実績とAI予測の推移")
                
                # フォントを明示的に指定（Streamlit Cloud環境用）
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Ubuntu', 'NanumGothic', 'Arial'] 
                
                # もし上記でも化ける場合は、日本語対応フォントを直接指定
                # plt.rcParams['font.family'] = 'Noto Sans CJK JP' 

# グラフの描画
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(df.index, df.iloc[:, 0], label="Actual Price", color="#1f77b4", linewidth=2)
                ax.plot(forecast_df.index, forecast_df["Forecast"], label="AI Forecast", color="#ff7f0e", linestyle="--", linewidth=2)
                
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (JPY)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.error("データを取得できませんでした。コードが正しいか確認してください。")