#!/usr/bin/env python3
"""
crypto_ml_pipeline.py — MAXIMUM POWER VERSION
Feature selection + GridSearchCV + Stacking + Multi-timeframe
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# FETCH DATA
# ============================================================
def fetch_data():
    log("Fetching BTC data via yfinance (5 years)...")
    import yfinance as yf
    
    btc = yf.download('BTC-USD', period='5y', interval='1d', progress=False)
    if btc.empty:
        raise RuntimeError("yfinance empty!")
    
    df = btc[['Close', 'Volume']].copy()
    df.columns = ['price', 'volume']
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    df['volume'] = df['volume'].fillna(0)
    log(f"BTC: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
    
    # F&G
    try:
        fng = requests.get("https://api.alternative.me/fng/?limit=1825", timeout=10).json()['data']
        fng_df = pd.DataFrame(fng)
        fng_df['date'] = pd.to_datetime(fng_df['timestamp'], unit='s').dt.normalize()
        fng_df['fng'] = fng_df['value'].astype(float)
        fng_df = fng_df.set_index('date').sort_index()
        fng_df = fng_df[~fng_df.index.duplicated(keep='last')].resample('D').ffill()
        df = df.join(fng_df[['fng']], how='left')
    except:
        df['fng'] = 50
    
    # Macro
    for ticker, name in [('DX-Y.NYB','dxy'), ('^GSPC','sp500'), ('GC=F','gold'), ('^VIX','vix')]:
        try:
            d = yf.download(ticker, period='5y', interval='1d', progress=False)['Close']
            d.index = d.index.tz_localize(None) if d.index.tz else d.index
            df = df.join(d.rename(name), how='left')
        except:
            pass
    
    df = df.ffill().bfill()
    log(f"Final: {len(df)} rows, {len(df.columns)} cols")
    return df

# ============================================================
# MAXIMUM FEATURES (200+)
# ============================================================
def add_features(df):
    log("Computing 200+ features...")
    
    # --- Returns ---
    for d in [1,2,3,5,7,10,14,21,30,60,90,180,365]:
        df[f'ret_{d}d'] = df['price'].pct_change(d)
    df['log_ret_1d'] = np.log(df['price'] / df['price'].shift(1))
    
    # --- SMA ---
    for p in [3,5,7,10,14,20,25,30,50,75,100,150,200,250,365]:
        df[f'sma_{p}'] = df['price'].rolling(p).mean()
        df[f'price_vs_sma_{p}'] = (df['price'] - df[f'sma_{p}']) / df[f'sma_{p}']
        df[f'sma_{p}_slope'] = df[f'sma_{p}'].pct_change(5)
    
    # --- EMA ---
    for p in [5,8,12,13,21,26,34,55,89,144,200]:
        df[f'ema_{p}'] = df['price'].ewm(span=p, adjust=False).mean()
        df[f'price_vs_ema_{p}'] = (df['price'] - df[f'ema_{p}']) / df[f'ema_{p}']
    
    # --- MACD ---
    for fast,slow,signal in [(12,26,9),(8,21,5),(5,35,5)]:
        ema_f = df['price'].ewm(span=fast, adjust=False).mean()
        ema_s = df['price'].ewm(span=slow, adjust=False).mean()
        df[f'macd_{fast}_{slow}'] = ema_f - ema_s
        df[f'macd_{fast}_{slow}_signal'] = df[f'macd_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
        df[f'macd_{fast}_{slow}_hist'] = df[f'macd_{fast}_{slow}'] - df[f'macd_{fast}_{slow}_signal']
    
    # --- RSI ---
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    for p in [5,7,9,14,21,28]:
        df[f'rsi_{p}'] = calc_rsi(df['price'], p)
    df['rsi_14_slope'] = df['rsi_14'].diff(5)
    df['price_slope_5d'] = df['price'].pct_change(5)
    df['rsi_divergence'] = (np.sign(df['price_slope_5d']) != np.sign(df['rsi_14_slope'])).astype(int)
    
    # --- Bollinger Bands ---
    for p in [10,20,30]:
        for mult in [1.5, 2.0, 2.5]:
            mid = df['price'].rolling(p).mean()
            std = df['price'].rolling(p).std()
            df[f'bb_{p}_{mult}_upper'] = mid + mult * std
            df[f'bb_{p}_{mult}_lower'] = mid - mult * std
            df[f'bb_{p}_{mult}_width'] = (mult * 2 * std) / mid
            df[f'bb_{p}_{mult}_pos'] = (df['price'] - df[f'bb_{p}_{mult}_lower']) / (df[f'bb_{p}_{mult}_upper'] - df[f'bb_{p}_{mult}_lower'])
    
    # --- ATR ---
    for p in [7,14,21]:
        hl = df['price'].rolling(2).max() - df['price'].rolling(2).min()
        df[f'atr_{p}'] = hl.rolling(p).mean()
        df[f'atr_{p}_pct'] = df[f'atr_{p}'] / df['price']
    
    # --- Volume ---
    for p in [3,5,7,14,21,30,60]:
        df[f'vol_sma_{p}'] = df['volume'].rolling(p).mean()
        df[f'vol_ratio_{p}'] = df['volume'] / df[f'vol_sma_{p}']
    df['vol_change_1d'] = df['volume'].pct_change(1)
    df['vol_change_7d'] = df['volume'].pct_change(7)
    df['obv'] = (np.sign(df['price'].diff()) * df['volume']).cumsum()
    df['obv_sma_20'] = df['obv'].rolling(20).mean()
    df['obv_vs_sma'] = df['obv'] / df['obv_sma_20']
    
    # --- Volatility ---
    for p in [5,7,10,14,21,30,60]:
        df[f'volatility_{p}d'] = df['log_ret_1d'].rolling(p).std()
    df['vol_regime'] = df['volatility_7d'] / df['volatility_30d']
    
    # --- Momentum ---
    for d in [1,3,5,7,10,14,21,30,60,90]:
        df[f'momentum_{d}d'] = df['price'] / df['price'].shift(d) - 1
    for d in [10,20,30]:
        df[f'roc_{d}'] = (df['price'] - df['price'].shift(d)) / df['price'].shift(d) * 100
    
    # --- Stochastic RSI ---
    for p in [14,21]:
        rsi = df[f'rsi_{p}']
        rsi_min = rsi.rolling(p).min()
        rsi_max = rsi.rolling(p).max()
        df[f'stoch_rsi_{p}'] = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    
    # --- Williams %R ---
    for p in [14,21]:
        high = df['price'].rolling(p).max()
        low = df['price'].rolling(p).min()
        df[f'williams_r_{p}'] = (high - df['price']) / (high - low).replace(0, np.nan) * -100
    
    # --- CCI ---
    for p in [14,20]:
        tp = df['price']
        tp_sma = tp.rolling(p).mean()
        tp_mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df[f'cci_{p}'] = (tp - tp_sma) / (0.015 * tp_mad)
    
    # --- Support/Resistance ---
    for window in [20,50,100,200]:
        df[f'support_{window}'] = df['price'].rolling(window).min()
        df[f'resistance_{window}'] = df['price'].rolling(window).max()
        df[f'price_vs_support_{window}'] = (df['price'] - df[f'support_{window}']) / df[f'support_{window}']
        df[f'price_vs_resistance_{window}'] = (df['price'] - df[f'resistance_{window}']) / df[f'resistance_{window}']
    
    # --- Trend ---
    df['trend_7d'] = np.where(df['price'] > df['sma_7'], 1, -1)
    df['trend_20d'] = np.where(df['price'] > df['sma_20'], 1, -1)
    df['trend_50d'] = np.where(df['price'] > df['sma_50'], 1, -1)
    df['trend_200d'] = np.where(df['price'] > df['sma_200'], 1, -1)
    df['trend_alignment'] = df['trend_7d'] + df['trend_20d'] + df['trend_50d'] + df['trend_200d']
    
    # --- Calendar ---
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['day_of_month'] = df.index.day
    
    # --- Macro ---
    for col in ['dxy','sp500','gold','vix']:
        if col in df.columns:
            for d in [1,3,5,7,14,30]:
                df[f'{col}_ret_{d}d'] = df[col].pct_change(d)
    
    # --- F&G ---
    if 'fng' in df.columns:
        df['fng_sma_7'] = df['fng'].rolling(7).mean()
        df['fng_sma_30'] = df['fng'].rolling(30).mean()
        df['fng_change'] = df['fng'].diff(7)
        df['fng_extreme_fear'] = (df['fng'] < 20).astype(int)
        df['fng_extreme_greed'] = (df['fng'] > 80).astype(int)
    
    log(f"Raw features: {len(df.columns)}")
    return df

# ============================================================
# TARGET
# ============================================================
def add_target(df):
    df['target_ret_7d'] = df['price'].shift(-7) / df['price'] - 1
    df['target_dir_7d'] = (df['target_ret_7d'] > 0).astype(int)
    return df

# ============================================================
# FEATURE SELECTION — Mutual Information
# ============================================================
def select_features(X, y, n_features=50):
    log(f"Selecting top {n_features} features via Mutual Information...")
    
    # Remove constant features
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=constant_cols, errors='ignore')
    
    # Fill NaN for MI calculation
    X_filled = X.fillna(0)
    
    # Mutual Information
    mi_scores = mutual_info_classif(X_filled, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi': mi_scores}).sort_values('mi', ascending=False)
    
    top_features = mi_df.head(n_features)['feature'].tolist()
    
    log(f"Top 10 features:")
    for _, row in mi_df.head(10).iterrows():
        log(f"  {row['feature']}: {row['mi']:.4f}")
    
    return top_features

# ============================================================
# TRAIN — GridSearchCV + Stacking
# ============================================================
def train_model(df):
    exclude = ['price', 'volume',
               'sma_3','sma_5','sma_7','sma_10','sma_14','sma_20','sma_25','sma_30','sma_50','sma_75','sma_100','sma_150','sma_200','sma_250','sma_365',
               'ema_5','ema_8','ema_12','ema_13','ema_21','ema_26','ema_34','ema_55','ema_89','ema_144','ema_200',
               'bb_10_1.5_upper','bb_10_1.5_lower','bb_10_2.0_upper','bb_10_2.0_lower','bb_10_2.5_upper','bb_10_2.5_lower',
               'bb_20_1.5_upper','bb_20_1.5_lower','bb_20_2.0_upper','bb_20_2.0_lower','bb_20_2.5_upper','bb_20_2.5_lower',
               'bb_30_1.5_upper','bb_30_1.5_lower','bb_30_2.0_upper','bb_30_2.0_lower','bb_30_2.5_upper','bb_30_2.5_lower',
               'vol_sma_3','vol_sma_5','vol_sma_7','vol_sma_14','vol_sma_21','vol_sma_30','vol_sma_60',
               'atr_7','atr_14','atr_21','obv','obv_sma_20',
               'dxy','sp500','gold','vix',
               'support_20','support_50','support_100','support_200',
               'resistance_20','resistance_50','resistance_100','resistance_200',
               'target_ret_7d','target_dir_7d']
    
    all_features = [c for c in df.columns if c not in exclude]
    df_clean = df.dropna()
    X_all = df_clean[all_features]
    y = df_clean['target_dir_7d']
    
    # Feature Selection
    top_features = select_features(X_all, y, n_features=60)
    X = df_clean[top_features]
    
    log(f"Training: {len(X)} samples, {len(top_features)} features")
    
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=top_features, index=X.index)
    
    # --- GridSearch: XGBoost ---
    log("GridSearch XGBoost...")
    xgb_params = {
        'n_estimators': [300, 500, 700],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0],
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    
    # Manual grid search with walk-forward (faster than GridSearchCV for time series)
    best_xgb_acc = 0
    best_xgb_params = {}
    
    # Sample a subset of params to search (full grid too slow)
    param_grid = [
        {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
        {'n_estimators': 700, 'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
        {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.0},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0, 'reg_lambda': 0.5},
        {'n_estimators': 700, 'max_depth': 7, 'learning_rate': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    ]
    
    for params in param_grid:
        accs = []
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0, **params)
        for train_idx, test_idx in tscv.split(X_s):
            X_tr, X_te = X_s.iloc[train_idx], X_s.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            accs.append(accuracy_score(y_te, model.predict(X_te)))
        avg = np.mean(accs)
        if avg > best_xgb_acc:
            best_xgb_acc = avg
            best_xgb_params = params
            log(f"  XGB new best: {avg:.4f} | {params}")
    
    # --- GridSearch: Random Forest ---
    log("GridSearch Random Forest...")
    rf_param_grid = [
        {'n_estimators': 500, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 700, 'max_depth': 12, 'min_samples_split': 15, 'min_samples_leaf': 7, 'max_features': 'sqrt'},
        {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_features': 'log2'},
        {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 0.5},
    ]
    
    best_rf_acc = 0
    best_rf_params = {}
    
    for params in rf_param_grid:
        accs = []
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        for train_idx, test_idx in tscv.split(X_s):
            X_tr, X_te = X_s.iloc[train_idx], X_s.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            accs.append(accuracy_score(y_te, model.predict(X_te)))
        avg = np.mean(accs)
        if avg > best_rf_acc:
            best_rf_acc = avg
            best_rf_params = params
            log(f"  RF new best: {avg:.4f}")
    
    # --- GridSearch: Gradient Boosting ---
    log("GridSearch Gradient Boosting...")
    gb_param_grid = [
        {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'min_samples_split': 10},
        {'n_estimators': 700, 'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.8, 'min_samples_split': 10},
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.7, 'min_samples_split': 15},
        {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.9, 'min_samples_split': 5},
    ]
    
    best_gb_acc = 0
    best_gb_params = {}
    
    for params in gb_param_grid:
        accs = []
        model = GradientBoostingClassifier(random_state=42, **params)
        for train_idx, test_idx in tscv.split(X_s):
            X_tr, X_te = X_s.iloc[train_idx], X_s.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            accs.append(accuracy_score(y_te, model.predict(X_te)))
        avg = np.mean(accs)
        if avg > best_gb_acc:
            best_gb_acc = avg
            best_gb_params = params
            log(f"  GB new best: {avg:.4f}")
    
    # --- Train best models on full data ---
    log("Training best models on full data...")
    
    best_xgb = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0, **best_xgb_params)
    best_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_rf_params)
    best_gb = GradientBoostingClassifier(random_state=42, **best_gb_params)
    
    best_xgb.fit(X_s, y)
    best_rf.fit(X_s, y)
    best_gb.fit(X_s, y)
    
    # --- Voting Ensemble (soft voting) ---
    log("Building Voting Ensemble...")
    voting = VotingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0, **best_xgb_params)),
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1, **best_rf_params)),
            ('gb', GradientBoostingClassifier(random_state=42, **best_gb_params)),
        ],
        voting='soft',
        weights=[1.2, 1.0, 1.1],
        n_jobs=-1
    )
    
    # Evaluate voting
    vote_accs = []
    for train_idx, test_idx in tscv.split(X_s):
        X_tr, X_te = X_s.iloc[train_idx], X_s.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        voting.fit(X_tr, y_tr)
        vote_accs.append(accuracy_score(y_te, voting.predict(X_te)))
    
    vote_avg = np.mean(vote_accs)
    log(f"  Voting: {vote_avg:.4f}")
    
    # --- Pick best ---
    results = {
        'XGB': (best_xgb_acc, best_xgb),
        'RF': (best_rf_acc, best_rf),
        'GB': (best_gb_acc, best_gb),
        'Voting': (vote_avg, voting),
    }
    
    best_name = max(results, key=lambda k: results[k][0])
    best_acc, final_model = results[best_name]
    
    log(f"\n🏆 BEST: {best_name} ({best_acc:.4f})")
    for name, (acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        marker = "🏆" if name == best_name else "  "
        log(f"  {marker} {name}: {acc:.4f}")
    
    # Final train on ALL data
    if best_name == 'Voting':
        voting.fit(X_s, y)
        final_model = voting
    # else already fitted above
    
    return final_model, scaler, top_features, best_name, best_acc

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
        'btc_price': float(btc),
        'direction': direction,
        'confidence': float(conf),
        'prob_up': float(proba[1]),
        'prob_down': float(proba[0]),
        'rsi': float(df['rsi_14'].iloc[-1]),
        'fng': float(df['fng'].iloc[-1]),
        'macd': float(df['macd_12_26_hist'].iloc[-1]),
        'bb_pos': float(df['bb_20_2.0_pos'].iloc[-1])
    }

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(token, chat_id, result, model_name, acc):
    if not token or not chat_id:
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
    log("CRYPTO ML PIPELINE MAX POWER — START")
    log("=" * 40)
    
    try:
        df = fetch_data()
    except Exception as e:
        log(f"FATAL fetch: {e}")
        _send_error(f"⚠️ Pipeline Error\n\nFetch failed: {e}")
        sys.exit(1)
    
    if len(df) < 200:
        log(f"FATAL: not enough data ({len(df)} rows)")
        sys.exit(1)
    
    df = add_features(df)
    df = add_target(df)
    
    log("Training models (this may take 3-5 minutes)...")
    try:
        model, scaler, features, model_name, acc = train_model(df)
    except Exception as e:
        log(f"FATAL train: {e}")
        import traceback
        traceback.print_exc()
        _send_error(f"⚠️ Training Error\n\n{e}")
        sys.exit(1)
    
    result = predict(model, scaler, features, df)
    log(f"Prediction: {result['direction']} ({result['confidence']:.1f}%)")
    log(f"BTC: ${result['btc_price']:,.2f}")
    log(f"Model: {model_name} ({acc:.1f}% acc)")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, f'{OUTPUT_DIR}/crypto_model.pkl')
    joblib.dump(scaler, f'{OUTPUT_DIR}/crypto_scaler.pkl')
    with open(f'{OUTPUT_DIR}/feature_cols.pkl', 'wb') as f:
        pickle.dump(features, f)
    result['model_name'] = model_name
    result['accuracy'] = acc
    result['timestamp'] = datetime.now().isoformat()
    with open(f'{OUTPUT_DIR}/latest_prediction.json', 'w') as f:
        json.dump(result, f, indent=2)
    log(f"Model saved to {OUTPUT_DIR}/")
    
    send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, result, model_name, acc)
    log("DONE!")
    return result

def _send_error(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'}, timeout=15)
        except:
            pass

if __name__ == "__main__":
    main()
