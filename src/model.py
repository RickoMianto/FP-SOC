# LOKASI: src/model.py
# TUJUAN: Satu-satunya tempat untuk logika machine learning.

import pandas as pd
from sklearn.ensemble import IsolationForest
# Import fungsi dari file tetangganya, analysis_utils.py
from .analysis_utils import feature_engineering

class DNSAnomalyDetector:
    """
    Kelas untuk mendeteksi anomali DNS.
    Dilatih dengan whitelist (semi-supervised).
    """
    def __init__(self, contamination=0.01, random_state=42):
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=random_state,
            n_jobs=-1 
        )
        self.feature_cols = ['query_length', 'query_entropy', 'subdomain_count']
        self.is_fitted = False

    def fit(self, df_whitelist: pd.DataFrame):
        """
        Melatih model HANYA dengan data whitelist yang aman.
        """
        print("INFO: Memulai pelatihan model dengan whitelist...")
        
        # Membuat fitur dari data whitelist menggunakan fungsi dari analysis_utils
        df_featured = feature_engineering(df_whitelist)
        
        X_train = df_featured[self.feature_cols]
        
        self.model.fit(X_train)
        self.is_fitted = True
        print("INFO: Pelatihan model selesai.")
        return self

    def predict(self, df_new_logs: pd.DataFrame) -> pd.Series:
        """
        Memprediksi data log baru menggunakan model yang sudah dilatih.
        """
        if not self.is_fitted:
            raise RuntimeError("Model belum dilatih. Panggil .fit() dengan whitelist terlebih dahulu.")
            
        # Membuat fitur dari data log baru
        df_featured = feature_engineering(df_new_logs)
        X_predict = df_featured[self.feature_cols]
        
        # Mengembalikan prediksi (-1 untuk anomali, 1 untuk normal)
        return self.model.predict(X_predict)