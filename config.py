"""
Configuration file for DGA Analysis Project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
STATIC_PATH = os.path.join(PROJECT_ROOT, 'static')

# Data files
DGA_DATA_FILE = os.path.join(DATA_PATH, 'dga_data.csv')

# Model configuration
MODEL_CONFIG = {
    'max_length': 64,          # Maximum domain length
    'vocab_size': 40,          # Character vocabulary size
    'embedding_dim': 128,      # Embedding dimension
    'lstm_units': 256,         # LSTM units
    'cnn_filters': 128,        # CNN filters
    'dense_units': 512,        # Dense layer units
    'dropout_rate': 0.3,       # Dropout rate
    'learning_rate': 0.001,    # Learning rate
    'batch_size': 128,         # Batch size
    'epochs': 10,              # Training epochs
    'validation_split': 0.2    # Validation split
}

# Feature engineering parameters
FEATURE_CONFIG = {
    'ngram_range': (1, 3),     # N-gram range
    'min_domain_length': 3,    # Minimum domain length
    'max_domain_length': 100,  # Maximum domain length
}

# Alert thresholds for SOC
ALERT_CONFIG = {
    'high_risk_threshold': 0.8,    # High risk DGA probability
    'medium_risk_threshold': 0.6,  # Medium risk DGA probability
    'batch_alert_count': 10,       # Alert if more than X DGA domains in batch
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'DGA Analysis - AIOps SOC',
    'page_icon': '🛡️',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}

# Model file names
MODEL_FILES = {
    'main_model': 'dga_classifier.h5',
    'tokenizer': 'tokenizer.pkl',
    'scaler': 'feature_scaler.pkl',
    'label_encoder': 'label_encoder.pkl'
}