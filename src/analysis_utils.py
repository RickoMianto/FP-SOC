import numpy as np
from collections import Counter
from sklearn.ensemble import IsolationForest

def calculate_entropy(s):
    """Menghitung Shannon Entropy dari sebuah string."""
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * np.log2(count/lns) for count in p.values())

def feature_engineering(df):
    """
    Menerima DataFrame log DNS dan menambahkan kolom fitur baru.
    Fitur: panjang query, entropy, dan jumlah subdomain.
    """
    df_copy = df.copy()
    df_copy['query_length'] = df_copy['query_name'].str.len()
    df_copy['query_entropy'] = df_copy['query_name'].apply(calculate_entropy)
    df_copy['subdomain_count'] = df_copy['query_name'].str.count('\\.')
    return df_copy

def train_and_predict(df_features, feature_cols):
    """
    Melatih model Isolation Forest dan memprediksi anomali.
    """
    X = df_features[feature_cols]
    
    # Inisialisasi model. Contamination 'auto' adalah standar yang baik.
    model = IsolationForest(contamination='auto', random_state=42)
    
    # Latih model
    model.fit(X)
    
    # Lakukan prediksi (-1 untuk anomali, 1 untuk normal)
    predictions = model.predict(X)
    
    return predictions