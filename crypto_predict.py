#!/usr/bin/env python3
"""
Crypto ML Predictor — Daily Prediction Script
Untuk di-run di Termux via cron job
"""

import requests
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# ============================================================
# CONFIGURATION
# ============================================================
TELEGRAM_TOKEN = "PASTE_TOKEN_LO_DISINI"
TELEGRAM_CHAT_ID = "PASTE_CHAT_ID_LO_DISINI"

MODEL_PATH = os.path.expanduser("~/.hermes/crypto_model.pkl")
SCALER_PATH = os.path.expanduser("~/.hermes/crypto_scaler.pkl")
FEATURES_PATH = os.path.expanduser("~/.hermes/feature_cols.pkl")
LOG_PATH = os.path.expanduser("~/.hermes/crypto_predict.log")

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            log("✅ Telegram sent!")
        else:
            log(f"⚠️ Telegram failed: {response.text}")
    except Exception as e:
        log(f"⚠️ Telegram error: {e}")

# ============================================================
# LOGGING
# ============================================================
def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, 'a') as f:
            f.write(line + '\n')
    except:
        pass

# ============================================================
# DATA FETCHER
# ============================================================
def fetch_latest_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {'vs_currency': 'usd', 'days': 365, 'interval': 'daily'}
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    vol = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    df['volume'] = vol['volume']
    df = df.set_index('date').drop('timestamp', axis=1)
    
    # F&G
    try:
        fng_url = "https://api.alternative.me/fng/?limit=365"
        fng_data = requests.get(fng_url, timeout=30).json()['data']
        fng_df = pd.DataFrame(fng_data)
        fng_df['date'] = pd.to_datetime(fng_df['timestamp'], unit='s')
        fng_df['fng'] = fng_df['value'].astype(float)
        fng_df = fng_df.set_index('date').sort_index().resample('D').ffill()
        df = df.join(fng_df[['fng']], how='left')
    except:
        df['fng'] = 50
    
    # Macro
    try:
        import yfinance as yf
        dxy = yf.download('DX-Y.NYB', period='1y', progress=False)['Close']
        sp = yf.download('^GSPC', period='1y', progress=False)['Close']
        df = df.join(dxy.rename('dxy'), how='left')
        df = df.join(sp.rename('sp500'), how='left')
    except:
        pass
    
    df = df.ffill().bfill()
    return df

def add_features(df):
    for d in [1, 3, 7, 14, 30]:
        df[f'ret_{d}d'] = df['price'].pct_change(d)
    df['log_ret_1d'] = np.log(df['price'] / df['price'].shift(1))
    
    for period in [7, 14, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['price'].rolling(period).mean()
        df[f'price_vs_sma_{period}'] = (df['price'] - df[f'sma_{period}']) / df[f'sma_{period}']
    
    df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = calc_rsi(df['price'], 14)
    df['rsi_7'] = calc_rsi(df['price'], 7)
    
    df['bb_mid'] = df['price'].rolling(20).mean()
    bb_std = df['price'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    high_low = df['price'].rolling(2).max() - df['price'].rolling(2).min()
    df['atr_7'] = high_low.rolling(7).mean()
    df['atr_14'] = high_low.rolling(14).mean()
    
    df['vol_sma_7'] = df['volume'].rolling(7).mean()
    df['vol_sma_30'] = df['volume'].rolling(30).mean()
    df['vol_ratio_7'] = df['volume'] / df['vol_sma_7']
    df['vol_ratio_30'] = df['volume'] / df['vol_sma_30']
    
    df['volatility_7d'] = df['ret_1d'].rolling(7).std()
    df['volatility_30d'] = df['ret_1d'].rolling(30).std()
    
    for d in [7, 14, 30]:
        df[f'momentum_{d}d'] = df['price'] / df['price'].shift(d) - 1
    
    if 'dxy' in df.columns:
        df['dxy_ret_1d'] = df['dxy'].pct_change(1)
        df['dxy_ret_7d'] = df['dxy'].pct_change(7)
    if 'sp500' in df.columns:
        df['sp500_ret_1d'] = df['sp500'].pct_change(1)
        df['sp500_ret_7d'] = df['sp500'].pct_change(7)
    
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    return df

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 50)
    log("🤖 CRYPTO ML PREDICTOR — Starting...")
    log("=" * 50)
    
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, 'rb') as f:
            feature_cols = pickle.load(f)
        log(f"✅ Model loaded: {type(model).__name__}")
    except Exception as e:
        log(f"❌ Model load failed: {e}")
        log("   Run Notebook 2 in Colab first to train model!")
        sys.exit(1)
    
    # Fetch data
    log("📡 Fetching latest data...")
    try:
        df = fetch_latest_data()
        df = add_features(df)
        log(f"✅ Data fetched: {len(df)} rows")
    except Exception as e:
        log(f"❌ Data fetch failed: {e}")
        sys.exit(1)
    
    # Prepare features
    latest = df[feature_cols].iloc[-1:]
    missing = set(feature_cols) - set(latest.columns)
    for m in missing:
        latest[m] = 0
    latest = latest[feature_cols]
    
    # Predict
    latest_scaled = scaler.transform(latest)
    pred = model.predict(latest_scaled)[0]
    proba = model.predict_proba(latest_scaled)[0]
    
    btc_price = df['price'].iloc[-1]
    direction = "🟢 NAIK" if pred == 1 else "🔴 TURUN"
    confidence = max(proba) * 100
    
    # Additional info
    rsi = df['rsi_14'].iloc[-1]
    fng = df['fng'].iloc[-1]
    macd_val = df['macd_hist'].iloc[-1]
    bb_pos = df['bb_position'].iloc[-1]
    
    # Smart DCA
    threshold = 77000
    if btc_price < threshold:
        dca = 100000
        dca_mode = "agresif 🔥"
    else:
        dca = 50000
        dca_mode = "pelan 🐢"
    
    emoji_fng = "😱" if fng < 25 else "😰" if fng < 50 else "😐" if fng < 75 else "🤑"
    
    # Format message
    message = f"""
╔══════════════════════════════════╗
║  📊 <b>BTC DAILY SIGNAL</b>          ║
╠══════════════════════════════════╣
║  💰 BTC: <b>${btc_price:,.2f}</b>
║  📈 Prediksi 7d: <b>{direction}</b>
║  🎯 Confidence: <b>{confidence:.1f}%</b>
╠══════════════════════════════════╣
║  TEKNIKAL:
║  RSI 14: {rsi:.1f}
║  MACD: {'Bullish 📈' if macd_val > 0 else 'Bearish 📉'}
║  BB Position: {bb_pos:.1f}%
║  F&G: {fng:.0f} {emoji_fng}
╠══════════════════════════════════╣
║  💰 SMART DCA: <b>Rp {dca:,}</b>
║  Mode: {dca_mode}
║  Strategi: &lt;$77k→Rp100k | ≥$77k→Rp50k
╠══════════════════════════════════╣
║  📊 Probabilitas:
║  NAIK: {proba[1]*100:.1f}%
║  TURUN: {proba[0]*100:.1f}%
╚══════════════════════════════════╝

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')} WIB
🤖 ML Model: {type(model).__name__}
⚠️ Bukan saran finansial. DYOR!
"""
    
    log(f"📊 Prediction: {direction} ({confidence:.1f}%)")
    log(f"💰 BTC: ${btc_price:,.2f}")
    log(f"💰 DCA: Rp {dca:,} ({dca_mode})")
    
    # Send to Telegram
    send_telegram(message)
    log("✅ Done!")

if __name__ == "__main__":
    main()
