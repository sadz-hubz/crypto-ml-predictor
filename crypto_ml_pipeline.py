#!/usr/bin/env python3
"""
crypto_ml_pipeline.py — MAXIMUM POWER v2
On-chain + Sentiment + Technical + Macro
VotingClassifier with walk-forward CV (StackingClassifier bug fixed)
"""

import pandas as pd
import numpy as np
import requests
import joblib
import pickle
import json
import os
import sys
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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
    log(f"BTC: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})")
    
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
# ON-CHAIN DATA (via free public APIs)
# ============================================================
def fetch_onchain(df):
    log("Fetching on-chain data...")
    
    # 1. Hashrate via Blockchain.com
    try:
        r = requests.get("https://api.blockchain.info/charts/hash-rate?timespan=5years&format=json&cors=true", timeout=15)
        data = r.json()
        if 'values' in data:
            hr_df = pd.DataFrame(data['values'])
            hr_df['date'] = pd.to_datetime(hr_df['x'], unit='s').dt.normalize()
            hr_df = hr_df.set_index('date').sort_index()
            hr_df = hr_df[~hr_df.index.duplicated(keep='last')]
            df = df.join(hr_df[['y']].rename(columns={'y': 'hashrate'}), how='left')
            log(f"  Hashrate: OK ({len(hr_df)} points)")
    except Exception as e:
        log(f"  Hashrate failed: {e}")
    
    # 2. Active addresses
    try:
        r = requests.get("https://api.blockchain.info/charts/n-unique-addresses?timespan=5years&format=json&cors=true", timeout=15)
        data = r.json()
        if 'values' in data:
            aa_df = pd.DataFrame(data['values'])
            aa_df['date'] = pd.to_datetime(aa_df['x'], unit='s').dt.normalize()
            aa_df = aa_df.set_index('date').sort_index()
            aa_df = aa_df[~aa_df.index.duplicated(keep='last')]
            df = df.join(aa_df[['y']].rename(columns={'y': 'active_addresses'}), how='left')
            log(f"  Active addresses: OK")
    except Exception as e:
        log(f"  Active addresses failed: {e}")
    
    # 3. Transaction count
    try:
        r = requests.get("https://api.blockchain.info/charts/n-transactions?timespan=5years&format=json&cors=true", timeout=15)
        data = r.json()
        if 'values' in data:
            tx_df = pd.DataFrame(data['values'])
            tx_df['date'] = pd.to_datetime(tx_df['x'], unit='s').dt.normalize()
            tx_df = tx_df.set_index('date').sort_index()
            tx_df = tx_df[~tx_df.index.duplicated(keep='last')]
            df = df.join(tx_df[['y']].rename(columns={'y': 'tx_count'}), how='left')
            log(f"  TX count: OK")
    except Exception as e:
        log(f"  TX count failed: {e}")
    
    # 4. BTC dominance via CoinGecko
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=15)
        data = r.json()
        if 'data' in data:
            dom = data['data'].get('market_cap_percentage', {}).get('btc', None)
            if dom:
                df['btc_dominance'] = float(dom)
                log(f"  BTC dominance: {dom:.1f}%")
    except Exception as e:
        log(f"  BTC dominance failed: {e}")
    
    df = df.ffill().bfill()
    return df

# ============================================================
# SENTIMENT DATA
# ============================================================
def fetch_sentiment(df):
    log("Fetching sentiment data...")
    
    # Social metrics via CoinGecko
    try:
        r = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&community_data=true&developer_data=true", timeout=15)
        data = r.json()
        comm = data.get('community_data', {})
        dev = data.get('developer_data', {})
        if comm:
            df['twitter_followers'] = comm.get('twitter_followers', 0) or 0
            df['reddit_subscribers'] = comm.get('reddit_subscribers', 0) or 0
            df['reddit_active'] = comm.get('reddit_accounts_active_48h', 0) or 0
            df['reddit_posts_48h'] = comm.get('reddit_average_posts_48h', 0) or 0
            df['reddit_comments_48h'] = comm.get('reddit_average_comments_48h', 0) or 0
            log(f"  Social metrics: OK")
        if dev:
            df['github_stars'] = dev.get('stars', 0) or 0
            df['github_forks'] = dev.get('forks', 0) or 0
            df['github_commits_4w'] = dev.get('commit_count_4_weeks', 0) or 0
            log(f"  Developer metrics: OK")
    except Exception as e:
        log(f"  Social/dev metrics failed: {e}")
    
    df = df.ffill().bfill()
    return df

# ============================================================
# FEATURE ENGINEERING (200+ features)
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
    
    # --- On-chain features ---
    if 'hashrate' in df.columns:
        for p in [7,14,30,60]:
            df[f'hashrate_sma_{p}'] = df['hashrate'].rolling(p).mean()
            df[f'hashrate_change_{p}d'] = df['hashrate'].pct_change(p)
        df['hashrate_price_ratio'] = df['hashrate'] / df['hashrate'].rolling(30).mean()
        df['hashrate_price_divergence'] = df['hashrate_price_ratio'] - (df['price'] / df['price'].rolling(30).mean())
    
    if 'active_addresses' in df.columns:
        for p in [7,14,30]:
            df[f'aa_sma_{p}'] = df['active_addresses'].rolling(p).mean()
            df[f'aa_change_{p}d'] = df['active_addresses'].pct_change(p)
        df['nvt_proxy'] = df['price'] / df['active_addresses'].rolling(30).mean()
        df['nvt_proxy_sma_14'] = df['nvt_proxy'].rolling(14).mean()
    
    if 'tx_count' in df.columns:
        for p in [7,14,30]:
            df[f'tx_sma_{p}'] = df['tx_count'].rolling(p).mean()
            df[f'tx_change_{p}d'] = df['tx_count'].pct_change(p)
        if 'active_addresses' in df.columns:
            df['tx_per_address'] = df['tx_count'] / df['active_addresses'].replace(0, np.nan)
    
    if 'btc_dominance' in df.columns:
        for p in [7,14,30]:
            df[f'dom_sma_{p}'] = df['btc_dominance'].rolling(p).mean()
            df[f'dom_change_{p}d'] = df['btc_dominance'].pct_change(p)
    
    # --- Sentiment features ---
    for col in ['twitter_followers', 'reddit_subscribers', 'reddit_active', 
                'reddit_posts_48h', 'reddit_comments_48h', 'github_commits_4w']:
        if col in df.columns:
            for p in [7,14,30]:
                df[f'{col}_sma_{p}'] = df[col].rolling(p).mean()
                df[f'{col}_change_{p}d'] = df[col].pct_change(p)
    
    # --- Cross-signal ---
    if 'fng' in df.columns:
        df['fng_price_divergence'] = df['fng'].diff(7) - df['price'].pct_change(7) * 100
    df['volume_price_divergence'] = df['vol_ratio_7'] - df['ret_7d']
    
    if 'hashrate' in df.columns and 'fng' in df.columns:
        df['onchain_sentiment_composite'] = (
            df['hashrate_change_30d'].fillna(0) * 0.5 +
            (df['fng'] - 50) / 50 * 0.5
        )
    
    log(f"Total features: {len(df.columns)}")
    return df

# ============================================================
# TARGET
# ============================================================
def add_target(df):
    df['target_ret_7d'] = df['price'].shift(-7) / df['price'] - 1
    df['target_dir_7d'] = (df['target_ret_7d'] > 0).astype(int)
    return df

# ============================================================
# FEATURE SELECTION
# ============================================================
def select_features(X, y, n_features=60):
    log(f"Selecting top {n_features} features via MI...")
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=constant_cols, errors='ignore')
    X_filled = X.fillna(0)
    mi_scores = mutual_info_classif(X_filled, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi': mi_scores}).sort_values('mi', ascending=False)
    top_features = mi_df.head(n_features)['feature'].tolist()
    log(f"Top 10: {[(r['feature'], f"{r['mi']:.4f}") for _, r in mi_df.head(10).iterrows()]}")
    return top_features

# ============================================================
# TRAIN
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
    
    top_features = select_features(X_all, y, n_features=60)
    X = df_clean[top_features]
    
    log(f"Training: {len(X)} samples, {len(top_features)} features")
    
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=top_features, index=X.index)
    
    # XGBoost grid search
    log("GridSearch XGBoost...")
    xgb_param_grid = [
        {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
        {'n_estimators': 700, 'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
        {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.0},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0, 'reg_lambda': 0.5},
        {'n_estimators': 700, 'max_depth': 7, 'learning_rate': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    ]
    
    best_xgb_acc, best_xgb_params = 0, {}
    for params in xgb_param_grid:
        accs = []
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0, **params)
        for train_idx, test_idx in tscv.split(X_s):
            model.fit(X_s.iloc[train_idx], y.iloc[train_idx])
            accs.append(accuracy_score(y.iloc[test_idx], model.predict(X_s.iloc[test_idx])))
        avg = np.mean(accs)
        if avg > best_xgb_acc:
            best_xgb_acc, best_xgb_params = avg, params
            log(f"  XGB best: {avg:.4f}")
    
    # RF grid search
    log("GridSearch RF...")
    rf_param_grid = [
        {'n_estimators': 500, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 700, 'max_depth': 12, 'min_samples_split': 15, 'min_samples_leaf': 7, 'max_features': 'sqrt'},
        {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_features': 'log2'},
    ]
    
    best_rf_acc, best_rf_params = 0, {}
    for params in rf_param_grid:
        accs = []
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        for train_idx, test_idx in tscv.split(X_s):
            model.fit(X_s.iloc[train_idx], y.iloc[train_idx])
            accs.append(accuracy_score(y.iloc[test_idx], model.predict(X_s.iloc[test_idx])))
        avg = np.mean(accs)
        if avg > best_rf_acc:
            best_rf_acc, best_rf_params = avg, params
            log(f"  RF best: {avg:.4f}")
    
    # GB grid search
    log("GridSearch GB...")
    gb_param_grid = [
        {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'min_samples_split': 10},
        {'n_estimators': 700, 'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.8, 'min_samples_split': 10},
        {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.9, 'min_samples_split': 5},
    ]
    
    best_gb_acc, best_gb_params = 0, {}
    for params in gb_param_grid:
        accs = []
        model = GradientBoostingClassifier(random_state=42, **params)
        for train_idx, test_idx in tscv.split(X_s):
            model.fit(X_s.iloc[train_idx], y.iloc[train_idx])
            accs.append(accuracy_score(y.iloc[test_idx], model.predict(X_s.iloc[test_idx])))
        avg = np.mean(accs)
        if avg > best_gb_acc:
            best_gb_acc, best_gb_params = avg, params
            log(f"  GB best: {avg:.4f}")
    
    # Voting Ensemble
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
    
    vote_accs = []
    for train_idx, test_idx in tscv.split(X_s):
        voting.fit(X_s.iloc[train_idx], y.iloc[train_idx])
        vote_accs.append(accuracy_score(y.iloc[test_idx], voting.predict(X_s.iloc[test_idx])))
    vote_avg = np.mean(vote_accs)
    log(f"  Voting: {vote_avg:.4f}")
    
    results = {
        'XGB': (best_xgb_acc, xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0, **best_xgb_params)),
        'RF': (best_rf_acc, RandomForestClassifier(random_state=42, n_jobs=-1, **best_rf_params)),
        'GB': (best_gb_acc, GradientBoostingClassifier(random_state=42, **best_gb_params)),
        'Voting': (vote_avg, voting),
    }
    
    best_name = max(results, key=lambda k: results[k][0])
    best_acc, _ = results[best_name]
    
    log(f"BEST: {best_name} ({best_acc:.4f})")
    for name, (acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        log(f"  {'[BEST]' if name == best_name else '     '} {name}: {acc:.4f}")
    
    # Final train on all data
    final_model = results[best_name][1]
    final_model.fit(X_s, y)
    
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
    direction = "UP" if pred == 1 else "DOWN"
    conf = max(proba) * 100
    return {
        'btc_price': float(btc),
        'direction': direction,
        'confidence': float(conf),
        'prob_up': float(proba[1]),
        'prob_down': float(proba[0]),
        'rsi': float(df['rsi_14'].iloc[-1]) if 'rsi_14' in df.columns else 0,
        'fng': float(df['fng'].iloc[-1]) if 'fng' in df.columns else 50,
        'macd': float(df['macd_12_26_hist'].iloc[-1]) if 'macd_12_26_hist' in df.columns else 0,
        'bb_pos': float(df['bb_20_2.0_pos'].iloc[-1]) if 'bb_20_2.0_pos' in df.columns else 0.5,
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
    emoji_dir = "🟢" if direction == "UP" else "🔴"
    
    msg = f"""📊 <b>BTC DAILY SIGNAL v2</b>

💰 BTC: <b>${btc:,.2f}</b>
📈 Prediksi 7d: <b>{emoji_dir} {direction}</b>
🎯 Confidence: <b>{conf:.1f}%</b>

TEKNIKAL:
RSI 14: {rsi:.1f}
MACD: {'Bullish' if macd > 0 else 'Bearish'}
BB%: {bb_pos:.1f}%
F&G: {fng:.0f} {emoji_fng}

💰 SMART DCA: <b>Rp {dca:,}</b> ({dca_mode})
📊 UP: {prob_up*100:.1f}% | DOWN: {prob_down*100:.1f}%

🤖 Model: {model_name} ({acc:.1f}% acc) + On-chain + Sentiment
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
    log("CRYPTO ML PIPELINE v2 — ONCHAIN + SENTIMENT")
    log("=" * 40)
    
    try:
        df = fetch_data()
    except Exception as e:
        log(f"FATAL fetch: {e}")
        _send_error(f"Pipeline Error: {e}")
        sys.exit(1)
    
    if len(df) < 200:
        log(f"FATAL: not enough data ({len(df)} rows)")
        sys.exit(1)
    
    try:
        df = fetch_onchain(df)
    except Exception as e:
        log(f"On-chain failed (continuing): {e}")
    
    try:
        df = fetch_sentiment(df)
    except Exception as e:
        log(f"Sentiment failed (continuing): {e}")
    
    df = add_features(df)
    df = add_target(df)
    
    log("Training models (3-5 min)...")
    try:
        model, scaler, features, model_name, acc = train_model(df)
    except Exception as e:
        log(f"FATAL train: {e}")
        import traceback
        traceback.print_exc()
        _send_error(f"Training Error: {e}")
        sys.exit(1)
    
    result = predict(model, scaler, features, df)
    log(f"Prediction: {result['direction']} ({result['confidence']:.1f}%)")
    log(f"BTC: ${result['btc_price']:,.2f}")
    log(f"Model: {model_name} ({acc:.1f}% acc)")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, f'{OUTPUT_DIR}/crypto_model_v2.pkl')
    joblib.dump(scaler, f'{OUTPUT_DIR}/crypto_scaler_v2.pkl')
    with open(f'{OUTPUT_DIR}/feature_cols_v2.pkl', 'wb') as f:
        pickle.dump(features, f)
    result['model_name'] = model_name
    result['accuracy'] = acc
    result['timestamp'] = datetime.now().isoformat()
    result['version'] = 'v2_onchain_sentiment'
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
