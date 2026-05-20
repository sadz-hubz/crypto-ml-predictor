#!/usr/bin/env python3
###"""
Crypto ML Predictor — Daily Prediction Script
Untuk di-run di Termux via cron job
""""

import requests
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# ====================================================
# CONFIGURATION
# ====================================================
TELEGRAM_TOKEN = "8342453374:AAHAKRnTK246-4LCXrrfmZ0tEhagPe31AD4"  TELEGRAM_CHAT_ID = "255013006"

MODEL_PATH = os.path.expanduser("~/.hermes/crypto_model.pkl")
SCALER_PATH = os.path.expanduser("~/.hermes/crypto_scaler.pkl")
FEATURES_PATH = os.path.expanduser("~/.hermes/feature_cols.pkl")  LOG_PATH = os.path.expanduser("~/.hermes/crypto_predict.log")