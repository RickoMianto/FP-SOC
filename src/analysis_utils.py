import pandas as pd
import numpy as np
from collections import Counter

def calculate_entropy(s: str) -> float:
    """
    Menghitung Shannon Entropy dari sebuah string.
    """
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * np.log2(count/lns) for count in p.values())

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menerima DataFrame dan menambahkan kolom fitur baru.
    """
    df_copy = df.copy()
    # Mengganti nama kolom jika ada 'domain' menjadi 'query_name' untuk konsistensi
    if 'domain' in df_copy.columns:
        df_copy.rename(columns={'domain': 'query_name'}, inplace=True)

    df_copy['query_length'] = df_copy['query_name'].str.len()
    df_copy['query_entropy'] = df_copy['query_name'].apply(calculate_entropy)
    df_copy['subdomain_count'] = df_copy['query_name'].str.count('\\.')
    return df_copy