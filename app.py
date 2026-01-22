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
st.set_page_config(page_title="株価・財務複合分析ツール", layout="wide")

# ==========================================
# 0. 投資助言業に該当しないためのガイドライン・注意喚起
# ==========================================
def display_disclaimer():
    with st.sidebar:
        st.warning("⚠️ **利用上の注意・免責事項**")
        st.caption("""
        本ツールは統計的手法および機械学習を用いたデータ解析結果を表示するものであり、
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
            st.info("同意いただける場合のみ、計算結果を表示します。")
            return False

# ==========================================
# 1. 財務指標計算ロジック（追加分）
# ==========================================
def latest_financial(df):
    if df is None or df.empty: return None
    df = df.T.copy()
    try:
        df.index = pd.to_datetime(df.index)
        return df.sort_index().iloc[-1]
    except: return df.iloc[-1]

def val(row, col):
    if row is None or col not in row or pd.isna(row[col]): return np.nan
    return row[col]

def calc_comprehensive_metrics(ticker_symbol):
    """株価統計と財務指標をまとめて計算"""
    t = yf.Ticker(ticker_symbol)
    
    # --- 1. 株価統計（もともとの機能） ---
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    hist = t.history(start=start_date, end=end_date)
    
    stock_stats = {"始値（1年前）": np.nan, "終値（現在）": np.nan, "想定倍率转换": np.nan}
    if len(hist) >= 2:
        p_start = hist['Close'].iloc[0]
        p_end = hist['Close'].iloc[-1]
        g_mean = np.sqrt(p_start * p_end)
        e_profit = p_end - g_mean
        stock_stats = {
            "始値（1年前）": round(p_start, 1),
            "終値（現在）": round(p_end, 1),
            "1年後の想定価格": round(p_end + e_profit, 1),
            "想定倍率": round(1 + (e_profit / p_end), 3)
        }

    # --- 2. 財務指標（追加機能） ---
    try:
        bs = latest_financial(t.balance_sheet)
        is_ = latest_financial(t.income_stmt)
        cf = latest_financial(t.cash_flow)
        info = t.info
        
        m_cap = info.get("marketCap", np.nan)
        beta = info.get("beta", np.nan)

        if bs is not None and is_ is not None and cf is not None:
            # BS/PL/CF値の抽出
            equity = val(bs, "Total Equity Gross Minority Interest")
            assets = val(bs, "Total Assets")
            debt = val(bs, "Total Debt")
            cash = val(bs, "Cash And Cash Equivalents")
            op_inc = val(is_, "Operating Income")
            rev = val(is_, "Total Revenue")
            int_exp = abs(val(is_, "Interest Expense"))
            fcf = val(cf, "Free Cash Flow")

            # 指標計算
            tax_rate = 0.30
            equity_ratio = equity / assets if assets else np.nan
            op_margin = op_inc / rev if rev else np.nan
            fcf_yield = fcf / m_cap if m_cap else np.nan
            roic = (op_inc * (1 - tax_rate)) / (equity + debt - cash) if (equity + debt - cash) else np.nan
            
            # WACC計算
            cost_equity = 0.01 + beta * 0.06 if not np.isnan(beta) else np.nan
            cost_debt = (int_exp / debt) * (1 - tax_rate) if debt else 0
            wacc = ((equity / (equity + debt)) * cost_equity + (debt / (equity + debt)) * cost_debt) if (equity + debt) else np.nan
            
            stock_stats.update({
                "自己資本比率": round(equity_ratio, 3),
                "営業利益率": round(op_margin, 3),
                "FCF利回り": round(fcf_yield, 3),
                "ROIC": round(roic, 3),
                "WACC": round(wacc, 3),
                "ROIC-WACC": round(roic - wacc, 3) if not np.isnan(roic) and not np.isnan(wacc) else np.nan
            })
    except:
        pass # 財務データが取れない場合は株価統計のみ
        
    return stock_stats

# ==========================================
# 2. AI予測用ロジック（もともとの機能）
# ==========================================
def get_stock_raw_data(ticker, current_date):
    data = yf.download(ticker, start=current_date - datetime.timedelta(days=365), end=current_date, progress=False)
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
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0)
    model.fit(np.array(X), np.array(y).ravel())
    preds = []
    curr = scaled[-window:].flatten()
    for _ in range(steps):
        p = model.predict([curr])[0]
        preds.append(p)
        curr = np.append(curr[1:], p)
    return pd.DataFrame({"Forecast": scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()}, 
                        index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D"))

# ==========================================
# メインUI
# ==========================================
st.title("📈 株価予測 × 財務クオリティ分析")

if display_disclaimer():
    tab1, tab2, tab3 = st.tabs(["日経225一括分析", "個別銘柄AI予測", "財務クオリティ詳細"])

    # --- Tab 1: 日経225一括分析（統合版） ---
    with tab1:
        st.header("日経225 予測＆財務スコア一覧")
        CSV_FILE = "Nikkei225.csv"
        if os.path.exists(CSV_FILE):
            if st.button("全銘銘柄の統合計算を実行"):
                df_base = pd.read_csv(CSV_FILE)
                tickers = [f"{str(num)}.T" for num in df_base.iloc[:, 0]]
                
                results_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, ticker in enumerate(tickers):
                    progress_bar.progress((idx + 1) / len(tickers))
                    status_text.text(f"分析中: {ticker}")
                    res = calc_comprehensive_metrics(ticker)
                    if res:
                        row = df_base.iloc[idx].to_dict()
                        row.update(res)
                        results_list.append(row)
                
                status_text.empty()
                progress_bar.empty()
                
                res_df = pd.DataFrame(results_list)
                st.subheader("📊 総合分析ランキング (FCF利回り順)")
                st.dataframe(res_df.sort_values(by="FCF利回り", ascending=False), use_container_width=True)
        else:
            st.error("Nikkei225.csv が見つかりません。")

    # --- Tab 2: 個別銘柄AI予測（据え置き） ---
    with tab2:
        st.header("AI詳細チャート予測")
        t_input = st.text_input("銘柄コードを入力", value="4974.T")
        if st.button("AI予測を実行"):
            df = get_stock_raw_data(t_input, datetime.datetime.now())
            if not df.empty:
                f_df = forecast_mlp(df)
                c1, c2, c3 = st.columns(3)
                c1.metric("現在株価", f"{df.iloc[-1,0]:,.1f}円")
                c2.metric("1ヶ月後予測", f"{f_df['Forecast'].iloc[19]:,.1f}円")
                
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'NanumGothic']
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index, df.iloc[:,0], label="実績")
                ax.plot(f_df.index, f_df["Forecast"], label="AI予測", linestyle="--")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

    # --- Tab 3: 財務クオリティ詳細 ---
    with tab3:
        st.header("ROIC-WACC スプレッド分析")
        t_input_fin = st.text_input("分析する銘柄コード", value="7203.T", key="fin_input")
        if st.button("財務詳細を表示"):
            with st.spinner("財務データを解析中..."):
                f_res = calc_comprehensive_metrics(t_input_fin)
                if "ROIC" in f_res:
                    cols = st.columns(4)
                    cols[0].metric("ROIC (投下資本利益率)", f"{f_res['ROIC']*100:.2f}%")
                    cols[1].metric("WACC (資本コスト)", f"{f_res['WACC']*100:.2f}%")
                    cols[2].metric("ROIC-WACC スプレッド", f"{f_res['ROIC-WACC']*100:.2f}%")
                    cols[3].metric("FCF利回り", f"{f_res['FCF利回り']*100:.2f}%")
                    
                    st.write("---")
                    st.info("""
                    **分析のヒント:**
                    - **ROIC-WACC > 0**: 企業が資本コスト以上に利益を生み出しており、価値を創造している状態です。
                    - **FCF利回り**: 時価総額に対して自由に使える現金がどれだけあるかを示します。高いほど割安かつキャッシュ創出力が強い傾向にあります。
                    """)
                else:
                    st.error("財務データを取得できませんでした。")