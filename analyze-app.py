import os
import yfinance as yf
import pandas as pd
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from datetime import datetime, time
import pytz  # 加入 pytz 模組
import re

def is_stock_code(text: str) -> bool:
    """判斷輸入是否為台灣股票代碼 (上市 .TW 或 上櫃 .TWO)"""
    return bool(re.match(r"^\d{4}(\.(TW|TWO))?$", text.strip()))

def is_market_open():
    """檢查台灣股市是否開盤（09:00–13:30，週一至週五，UTC+8）"""
    tz = pytz.timezone('Asia/Taipei')  # 使用台灣時區
    now = datetime.now(tz)
    if now.weekday() >= 5:  # 週六週日不開盤
        return False
    market_start = time(9, 0)
    market_end = time(13, 30)
    return market_start <= now.time() <= market_end

app = Flask(__name__)

# ---- 環境變數 (Heroku /Render 上要設定) ----
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    print("WARNING: LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET not set in environment variables.")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ---- 分析函式 ----
def _ensure_single_ticker_df(df, code):
    if isinstance(df.columns, pd.MultiIndex):
        try:
            tickers = list(df.columns.get_level_values(1).unique())
        except Exception:
            tickers = []
        if code in tickers:
            return df.xs(code, axis=1, level=1)
        for t in tickers:
            if code == t or code in t or t in code:
                return df.xs(t, axis=1, level=1)
        if tickers:
            print(f"[DEBUG] {code}: MultiIndex returned but exact ticker not found. Using first ticker {tickers[0]} as fallback.")
            return df.xs(tickers[0], axis=1, level=1)
    return df

def calculate_kd_safe(df, n=9):
    low_min = df['Low'].rolling(window=n, min_periods=1).min()
    high_max = df['High'].rolling(window=n, min_periods=1).max()
    denom = (high_max - low_min)
    rsv = ((df['Close'] - low_min) / denom * 100).where(denom > 0, 50).fillna(50)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    df = df.copy()
    df['K'] = k
    df['D'] = d
    return df

def analyze_stock(stock_code):
    stock_code = stock_code.strip()
    if '.' not in stock_code:
        stock_code += '.TW'
    print(f"[INFO] Attempting to download: {stock_code}")

    df = None
    data_source = None
    max_retries = 3

    if is_market_open():
        # --- 盤中先試 1d 30m ---
        data_source = "1d 5m"
        for attempt in range(max_retries):
            try:
                df = yf.download(stock_code, period="1d", interval="5m", prepost=True, progress=False)
                if df is not None and not df.empty and len(df) >= 3:  # 至少要有3筆
                    print(f"[INFO] {stock_code} downloaded with 1d 5m")
                    break
            except Exception as e:
                print(f"[ERROR] {stock_code} yf.download (1d 5m) attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    break

        # --- 如果 1d 5m 不夠，改 5d 1d ---
        if df is None or df.empty or len(df) < 3:
            print(f"[WARN] {stock_code} 1d 5m insufficient, falling back to 5d 1d")
            data_source = "5d 1d"
            for attempt in range(max_retries):
                try:
                    df = yf.download(stock_code, period="5d", interval="1d", progress=False)
                    if df is not None and not df.empty:
                        print(f"[INFO] {stock_code} downloaded with 5d 1d")
                        break
                except Exception as e:
                    print(f"[ERROR] {stock_code} yf.download (5d 1d) attempt {attempt+1}/{max_retries} failed: {e}")
                    if attempt == max_retries - 1:
                        return f"{stock_code} 資料下載失敗: {e}"

    else:
        # --- 非開盤時間，直接用 5d 1d ---
        data_source = "5d 1d"
        for attempt in range(max_retries):
            try:
                df = yf.download(stock_code, period="5d", interval="1d", progress=False)
                if df is not None and not df.empty:
                    print(f"[INFO] {stock_code} downloaded with 5d 1d (non-market hours)")
                    break
            except Exception as e:
                print(f"[ERROR] {stock_code} yf.download (5d 1d) attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return f"{stock_code} 資料下載失敗: {e}"

    if df is None or df.empty:
        print(f"[WARN] {stock_code} download empty after fallbacks")
        return f"{stock_code} 無法取得資料 (市場可能未開或延遲，請稍後重試)"

    df = _ensure_single_ticker_df(df, stock_code)
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df = calculate_kd_safe(df, n=9)

    df_clean = df.dropna(subset=['Close', 'MA5', 'MA20', 'K', 'D'])
    if df_clean.empty or len(df_clean) < 3:  # 確保至少3筆有效數據
        print(f"[WARN] {stock_code} insufficient cleaned data rows: {len(df_clean)}")
        note = "資料筆數不足，建議開盤後重試" if data_source == "5d 1d" else "開盤資料數<3筆數不足，建議稍後重試"
        return f"{stock_code} 資料不足，無法分析（有效列數 {len(df_clean)}，{note})"

    try:
        last_close = float(df_clean['Close'].iloc[-1])
        ma5 = float(df_clean['MA5'].iloc[-1])
        ma20 = float(df_clean['MA20'].iloc[-1])
        last_k = float(df_clean['K'].iloc[-1])
        last_d = float(df_clean['D'].iloc[-1])
    except Exception as e:
        print(f"[ERROR] {stock_code} value extraction failed: {e}")
        return f"{stock_code} 資料解析失敗: {e}"

    recent_n = min(5, len(df_clean))
    support = float(df_clean['Low'].tail(recent_n).median())
    resistance = float(df_clean['High'].tail(recent_n).median())
    support = round(support, 2)  # 改進：保留兩位小數，提升精確度
    resistance = round(resistance, 2)

    ma_signal = "短期均線突破長期均線，趨勢轉強" if ma5 > ma20 else "短期均線在長期均線下方，趨勢偏弱"
    if last_k > last_d:
        kd_signal = f"黃金交叉，偏多 (K={last_k:.1f}, D={last_d:.1f})"
        if last_k > 80:  # 改進：添加超買警告
            kd_signal += "（K值超買，短期可能回檔）"
    elif last_k < last_d:
        kd_signal = f"死亡交叉，偏空 (K={last_k:.1f}, D={last_d:.1f})"
    else:
        kd_signal = f"持平 (K={last_k:.1f}, D={last_d:.1f})"

    buy_signal = (last_close >= resistance * 0.995) or (ma5 > ma20 and last_k > last_d)
    sell_signal = (last_close <= support * 1.005) or (ma5 < ma20 and last_k < last_d)
    if buy_signal and not sell_signal:
        advice = "建議: BUY ✅"
        # 改進：對於 BUY，設更高目標
        next_target = resistance * 1.02  # 假設上漲至壓力位 +2%
        expected_return = (next_target - last_close) / last_close * 100
    elif sell_signal and not buy_signal:
        advice = "建議: SELL ❌"
        expected_return = (last_close - support) / last_close * 100
    else:
        advice = "建議: HOLD ⏸"
        expected_return = 0.0

    # 說明邏輯
    note = f"目前是以 {data_source} 當沖條件" if data_source == "1d 5m" else f"開盤資料數<3或目前非盤中，故沒有資料，將以 {data_source} 使用近 5 日日線資料進行分析"

    report = (
        f"📊 {stock_code}\n"
        f"收盤價: {last_close:.2f}\n"
        f"支撐: {support:.2f}, 壓力: {resistance:.2f}\n"
        f"預期報酬率: {expected_return:.2f}%\n"
        f"MA 判斷: {ma_signal}\n"
        f"KD 判斷: {kd_signal}\n"
        f"{advice}\n"
        f"備註: {note}"
    )

    print(f"[INFO] {stock_code} processed: rows={len(df_clean)}, last_close={last_close}, ma5={ma5}, ma20={ma20}, K={last_k:.2f}, D={last_d:.2f}")

    return report

# ---- Flask / LINE webhook ----
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text.strip()

    # 檢查是不是至少有一個股票代碼
    codes = [c.strip() + '.TW' if '.' not in c.strip() else c.strip()
             for c in user_text.split(",") if is_stock_code(c.strip())]

    if not codes:  # 如果不是股票代碼，就當一般聊天
        #line_bot_api.reply_message(
            #event.reply_token,
            #TextSendMessage(text="我可以幫你查股票哦～請輸入 2330 或 2330.TW 試試！")
        #)
        return

    # 原本的股票分析流程
    results = []
    for code in codes:
        try:
            res = analyze_stock(code)
        except Exception as e:
            res = f"{code} 分析失敗: {e}"
        results.append(res)

    reply_text = "\n\n".join(results)
    if len(reply_text) > 4900:
        reply_text = reply_text[:4900] + "\n\n(結果過長，已截斷)"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))


if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=10000)