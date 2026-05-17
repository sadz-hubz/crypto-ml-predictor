#!/usr/bin/env python3
"""
crypto_ml_pipeline.py
Full pipeline: fetch data → features → train → predict → save model
Run via GitHub Actions atau Termux
"""

import pandas as pd
import numpy as np
import requests
import joblib
import pickle
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# ============================================================
# LOG
# ============================================================
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")

# ============================================================
# FETCH DATA
# ============================================================
def fetch_data():
    log("Fetching BTC data...")
    
    # Try CoinGecko first
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {'vs_currency': 'usd', 'days': 730, 'interval': 'daily'}
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        
        if 'prices' in data and len(data['prices']) > 0:
            df = pd.DataFrame(data['prices'], columns=['ts', 'price'])
            df['date'] = pd.to_datetime(df['ts'], unit='ms')
            vol = pd.DataFrame(data['total_volumes'], columns=['ts', 'volume'])
            df['volume'] = vol['volume']
            df = df.set_index('date').drop('ts', axis=1)
            log(f"CoinGecko OK: {len(df)} rows")
        else:
            raise ValueError("CoinGecko returned empty data")
    except Exception as e:
        log(f"CoinGecko failed: {e}, trying Binance...")
        # Fallback to Binance
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': 'BTCUSDT', 'interval': '1d', 'limit': 730}
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df['price'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df = df.set_index('date')[['price', 'volume']]
        log(f"Binance OK: {len(df)} rows")
    
    # F&G
    try:
        fng = requests.get("https://api.alternative.me/fng/?limit=730", timeout=30).json()['data']
        fng_df = pd.DataFrame(fng)
        fng_df['date'] = pd.to_datetime(fng_df['timestamp'], unit='s')
        fng_df['fng'] = fng_df['value'].astype(float)
        fng_df = fng_df.set_index('date').sort_index().resample('D').ffill()
        df = df.join(fng_df[['fng']], how='left')
    except:
        df['fng'] = 50
    
    # Macro
    try:
        import yfinance as yf
        dxy = yf.download('DX-Y.NYB', period='2y', progress=False)['Close']
        sp = yf.download('^GSPC', period='2y', progress=False)['Close']
        df = df.join(dxy.rename('dxy'), how='left')
        df = df.join(sp.rename('sp500'), how='left')
    except:
        pass
    
    df = df.ffill().bfill()
    log(f"Data: {len(df)} rows")
    return df

# ============================================================
# FEATURES
# ============================================================
def add_features(df):
    for d in [1, 3, 7, 14, 30]:
        df[f'ret_{d}d'] = df['price'].pct_change(d)
    df['log_ret_1d'] = np.log(df['price'] / df['price'].shift(1))
    
    for p in [7, 14, 20, 50, 100, 200]:
        df[f'sma_{p}'] = df['price'].rolling(p).mean()
        df[f'price_vs_sma_{p}'] = (df['price'] - df[f'sma_{p}']) / df[f'sma_{p}']
    
    df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    def rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = rsi(df['price'], 14)
    df['rsi_7'] = rsi(df['price'], 7)
    
    df['bb_mid'] = df['price'].rolling(20).mean()
    bb_std = df['price'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    hl = df['price'].rolling(2).max() - df['price'].rolling(2).min()
    df['atr_7'] = hl.rolling(7).mean()
    df['atr_14'] = hl.rolling(14).mean()
    
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
# TARGET
# ============================================================
def add_target(df):
    df['target_ret_7d'] = df['price'].shift(-7) / df['price'] - 1
    df['target_dir_7d'] = (df['target_ret_7d'] > 0).astype(int)
    return df

# ============================================================
# TRAIN
# ============================================================
def train_model(df):
    exclude = ['price', 'volume',
               'sma_7','sma_14','sma_20','sma_50','sma_100','sma_200',
               'ema_12','ema_26','bb_mid','bb_upper','bb_lower',
               'vol_sma_7','vol_sma_30','atr_7','atr_14',
               'dxy','sp500','target_ret_7d','target_dir_7d']
    
    features = [c for c in df.columns if c not in exclude]
    df_clean = df.dropna()
    
    X = df_clean[features]
    y = df_clean['target_dir_7d']
    
    # Walk-forward
    tscv = TimeSeriesSplit(n_splits=5)
    best_acc = 0
    best_model = None
    best_scaler = None
    
    models = {
        'RF': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        'GB': GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
        'XGB': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42,
                                   use_label_encoder=False, eval_metric='logloss')
    }
    
    for name, model in models.items():
        accs = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)
            accs.append(accuracy_score(y_te, preds))
        
        avg = np.mean(accs)
        log(f"  {name}: {avg:.4f}")
        if avg > best_acc:
            best_acc = avg
            best_model = model
            best_name = name
    
    # Final train on all data
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    best_model.fit(X_s, y)
    
    log(f"Best: {best_name} ({best_acc:.4f})")
    return best_model, scaler, features, best_name, best_acc

# ============================================================
# PREDICT
# ============================================================
def predict(model, scaler, features, df):
    latest = df[features].iloc[-1:]
    missing = set(features) - set(latest.columns)
    for m in missing:
        latest[m] = 0
    latest = latest[features]
    
    latest_s = scaler.transform(latest)
    pred = model.predict(latest_s)[0]
    proba = model.predict_proba(latest_s)[0]
    
    btc = df['price'].iloc[-1]
    direction = "NAIK" if pred == 1 else "TURUN"
    conf = max(proba) * 100
    
    return {
        'btc_price': btc,
        'direction': direction,
        'confidence': conf,
        'prob_up': proba[1],
        'prob_down': proba[0],
        'rsi': df['rsi_14'].iloc[-1],
        'fng': df['fng'].iloc[-1],
        'macd': df['macd_hist'].iloc[-1],
        'bb_pos': df['bb_position'].iloc[-1]
    }

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(token, chat_id, result, model_name, acc):
    if not token or not chat_id:
        log("No Telegram config, skip")
        return
    
    btc = result['btc_price']
    direction = result['direction']
    conf = result['confidence']
    rsi = result['rsi']
    fng = result['fng']
    macd = result['macd']
    bb_pos = result['bb_pos']
    prob_up = result['prob_up']
    prob_down = result['prob_down']
    
    dca = 100000 if btc < 77000 else 50000
    dca_mode = "agresif" if btc < 77000 else "pelan"
    emoji_fng = "😱" if fng < 25 else "😰" if fng < 50 else "😐" if fng < 75 else "🤑"
    emoji_dir = "🟢" if direction == "NAIK" else "🔴"
    
    msg = f"""📊 <b>BTC DAILY SIGNAL</b>

💰 BTC: <b>${btc:,.2f}</b>
📈 Prediksi 7d: <b>{emoji_dir} {direction}</b>
🎯 Confidence: <b>{conf:.1f}%</b>

TEKNIKAL:
RSI 14: {rsi:.1f}
MACD: {'Bullish' if macd > 0 else 'Bearish'}
BB%: {bb_pos:.1f}%
F&G: {fng:.0f} {emoji_fng}

💰 SMART DCA: <b>Rp {dca:,}</b> ({dca_mode})
📊 NAIK: {prob_up*100:.1f}% | TURUN: {prob_down*100:.1f}%

🤖 Model: {model_name} ({acc:.1f}% acc)
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')} WIB
⚠️ Bukan saran finansial. DYOR!"""
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'}, timeout=30)
    if resp.status_code == 200:
        log("Telegram sent!")
    else:
        log(f"Telegram failed: {resp.text[:100]}")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 40)
    log("CRYPTO ML PIPELINE START")
    log("=" * 40)
    
    # 1. Fetch
    df = fetch_data()
    
    # 2. Features
    df = add_features(df)
    df = add_target(df)
    log(f"Features: {len(df.columns)}")
    
    # 3. Train
    log("Training models...")
    model, scaler, features, model_name, acc = train_model(df)
    
    # 4. Predict
    result = predict(model, scaler, features, df)
    log(f"Prediction: {result['direction']} ({result['confidence']:.1f}%)")
    log(f"BTC: ${result['btc_price']:,.2f}")
    
    # 5. Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, f'{OUTPUT_DIR}/crypto_model.pkl')
    joblib.dump(scaler, f'{OUTPUT_DIR}/crypto_scaler.pkl')
    with open(f'{OUTPUT_DIR}/feature_cols.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    # Save prediction result
    result['model_name'] = model_name
    result['accuracy'] = acc
    result['timestamp'] = datetime.now().isoformat()
    with open(f'{OUTPUT_DIR}/latest_prediction.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    log(f"Model saved to {OUTPUT_DIR}/")
    
    # 6. Telegram
    send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, result, model_name, acc)
    
    log("DONE!")
    return result

if __name__ == "__main__":
    main()
