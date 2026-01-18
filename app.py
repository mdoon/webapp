import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os

# 1. ページの設定
st.set_page_config(page_title="日経225予測表示", layout="wide")

def get_stock_data(ticker, current_date):
    end_date = current_date
    start_date = end_date - datetime.timedelta(days=365)
    try:
        # yfinanceでデータ取得
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(stock_data) < 2:
            return None
        
        price_start = float(stock_data['Close'].iloc[0])
        price_end = float(stock_data['Close'].iloc[-1])

        # 幾何平均を用いた予測計算
        geometric_mean = np.sqrt(price_start * price_end)
        expected_profit = price_end - geometric_mean
        expected_price = price_end + expected_profit
        expected_interest_rate = 1 + (expected_profit / price_end)

        return {
            "始値（1年前）": round(price_start, 2),
            "終値（現在）": round(price_end, 2),
            "1年後の想定価格": round(expected_price, 2),
            "想定倍率": round(expected_interest_rate, 3)
        }
    except:
        return None

# --- UI部分 ---
st.title("📈 日経225 株価予測シミュレーター")

CSV_FILE = "Nikkei225.csv"

# ファイルが自動読み込みできるか確認
if os.path.exists(CSV_FILE):
    # ボタンのみ表示
    if st.button("全銘柄の予測計算を開始する"):
        df_base = pd.read_csv(CSV_FILE)
        
        # 銘柄コードの取得（1列目）
        ticker_col = df_base.columns[0]
        tickers = [f"{str(num)}.T" for num in df_base[ticker_col]]

        # 結果を格納するデータフレームの準備
        results_df = df_base.copy()
        
        # 進捗バーの設定
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_date = datetime.datetime.now()

        # ループで計算
        for idx, ticker in enumerate(tickers):
            pct = (idx + 1) / len(tickers)
            progress_bar.progress(pct)
            status_text.text(f"計算中... ({idx + 1}/{len(tickers)})")
            
            res = get_stock_data(ticker, current_date)
            if res:
                results_df.at[idx, "始値（1年前）"] = res["始値（1年前）"]
                results_df.at[idx, "終値（現在）"] = res["終値（現在）"]
                results_df.at[idx, "1年後の想定価格"] = res["1年後の想定価格"]
                results_df.at[idx, "想定倍率"] = res["想定倍率"]

        # 完了後の表示（バーを消して結果を表示）
        status_text.empty()
        progress_bar.empty()
        
        st.subheader("📊 予測計算結果")
        # 想定倍率が高い順に並び替えて表示
        st.dataframe(results_df.sort_values(by="想定倍率", ascending=False), height=600)
else:
    st.error(f"エラー: `{CSV_FILE}` が見つかりません。")