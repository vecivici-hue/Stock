# pip install pykrx pandas numpy tqdm openpyxl

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykrx import stock
from tqdm import tqdm
import time

# =========================
# 설정값
# =========================
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 3)   # 주봉 지표 계산용 3년치
RECENT_DAYS = 31                                  # 최근 한 달

RSI_PERIOD = 14
RSI_SIGNAL_PERIOD = 9

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

SLEEP_SEC = 0.15   # KRX 과도 호출 방지

# =========================
# 날짜 포맷
# =========================
start = START_DATE.strftime("%Y%m%d")
end = END_DATE.strftime("%Y%m%d")
recent_cutoff = END_DATE - timedelta(days=RECENT_DAYS)

# =========================
# 지표 함수
# =========================
def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist


def is_golden_cross(series, signal):
    """
    직전 주에는 series <= signal,
    이번 주에는 series > signal 이면 골든크로스
    """
    return (series.shift(1) <= signal.shift(1)) & (series > signal)


# =========================
# 주봉 변환 함수
# =========================
def to_weekly(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    weekly = pd.DataFrame()
    weekly["시가"] = df["시가"].resample("W-FRI").first()
    weekly["고가"] = df["고가"].resample("W-FRI").max()
    weekly["저가"] = df["저가"].resample("W-FRI").min()
    weekly["종가"] = df["종가"].resample("W-FRI").last()
    weekly["거래량"] = df["거래량"].resample("W-FRI").sum()

    weekly = weekly.dropna()
    return weekly


# =========================
# 종목 스캔 함수
# =========================
def scan_market(market):
    results = []

    tickers = stock.get_market_ticker_list(end, market=market)

    for ticker in tqdm(tickers, desc=f"Scanning {market}"):
        try:
            name = stock.get_market_ticker_name(ticker)

            df = stock.get_market_ohlcv_by_date(start, end, ticker)

            if df.empty or len(df) < 200:
                continue

            weekly = to_weekly(df)

            if len(weekly) < 60:
                continue

            close = weekly["종가"]

            # RSI
            weekly["RSI"] = calculate_rsi(close, RSI_PERIOD)
            weekly["RSI_SIGNAL"] = weekly["RSI"].rolling(RSI_SIGNAL_PERIOD).mean()
            weekly["RSI_GC"] = is_golden_cross(
                weekly["RSI"],
                weekly["RSI_SIGNAL"]
            )

            # MACD
            weekly["MACD"], weekly["MACD_SIGNAL"], weekly["MACD_HIST"] = calculate_macd(
                close,
                MACD_FAST,
                MACD_SLOW,
                MACD_SIGNAL
            )
            weekly["MACD_GC"] = is_golden_cross(
                weekly["MACD"],
                weekly["MACD_SIGNAL"]
            )

            # 동시 발생
            weekly["BOTH_GC"] = weekly["RSI_GC"] & weekly["MACD_GC"]

            matched = weekly[
                (weekly["BOTH_GC"]) &
                (weekly.index >= recent_cutoff)
            ]

            if not matched.empty:
                last = matched.iloc[-1]
                signal_date = matched.index[-1]

                results.append({
                    "시장": market,
                    "종목코드": ticker,
                    "종목명": name,
                    "신호일": signal_date.strftime("%Y-%m-%d"),
                    "종가": int(last["종가"]),
                    "RSI": round(last["RSI"], 2),
                    "RSI_SIGNAL": round(last["RSI_SIGNAL"], 2),
                    "MACD": round(last["MACD"], 2),
                    "MACD_SIGNAL": round(last["MACD_SIGNAL"], 2),
                    "거래량": int(last["거래량"])
                })

            time.sleep(SLEEP_SEC)

        except Exception as e:
            continue

    return results


# =========================
# 실행
# =========================
kospi_results = scan_market("KOSPI")
kosdaq_results = scan_market("KOSDAQ")

result_df = pd.DataFrame(kospi_results + kosdaq_results)

if result_df.empty:
    print("최근 한 달 내 RSI + MACD 동시 골든크로스 종목이 없습니다.")
else:
    result_df = result_df.sort_values(["신호일", "시장"], ascending=[False, True])
    print(result_df)

    file_name = "kospi_kosdaq_weekly_rsi_macd_golden_cross.xlsx"
    result_df.to_excel(file_name, index=False)
    print(f"\n저장 완료: {file_name}")

    from google.colab import files
    files.download(file_name)
