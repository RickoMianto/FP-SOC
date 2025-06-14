import streamlit as st
import pandas as pd
from src.analysis_utils import feature_engineering, train_and_predict

# Konfigurasi halaman
st.set_page_config(page_title="AIOps Deteksi Anomali DNS", layout="wide")

# --- JUDUL APLIKASI ---
st.title("🔎 AIOps: Deteksi Anomali pada Log DNS")
st.write("""
Aplikasi ini adalah implementasi sederhana dari konsep AIOps untuk keamanan siber. 
Unggah file log DNS Anda (dalam format CSV) untuk mendeteksi potensi aktivitas mencurigakan 
seperti koneksi ke server malware atau DNS tunneling.
""")

# --- UPLOAD FILE ---
uploaded_file = st.file_uploader(
    "Unggah file log DNS Anda (format .csv dengan kolom 'timestamp', 'source_ip', 'query_name')", 
    type="csv"
)

if uploaded_file is not None:
    # Baca file yang diunggah
    df = pd.read_csv(uploaded_file)
    
    st.write("---")
    st.subheader("✅ Data Berhasil Diunggah")
    st.write(f"Nama File: `{uploaded_file.name}`")
    st.write(f"Jumlah Baris: `{df.shape[0]}`")
    st.write("**Pratinjau Data:**")
    st.dataframe(df.head())

    # Tombol untuk memulai analisis
    if st.button("🚀 Mulai Analisis Anomali!", type="primary"):
        with st.spinner('Sedang melakukan analisis, mohon tunggu...'):
            
            # 1. Feature Engineering
            df_featured = feature_engineering(df)
            
            # 2. Pemodelan dan Prediksi
            feature_cols = ['query_length', 'query_entropy', 'subdomain_count']
            predictions = train_and_predict(df_featured, feature_cols)
            df_featured['is_anomaly'] = predictions
            
            # 3. Ambil data anomali
            anomalies = df_featured[df_featured['is_anomaly'] == -1]

        st.success("Analisis Selesai!")
        
        # 4. Tampilkan Hasil
        st.write("---")
        st.subheader("🚨 Hasil Deteksi Anomali")
        
        if anomalies.empty:
            st.info("Tidak ada anomali signifikan yang terdeteksi pada dataset ini.")
        else:
            st.warning(f"Ditemukan **{len(anomalies)}** potensi anomali:")
            st.dataframe(anomalies[['timestamp', 'source_ip', 'query_name', 'query_length', 'query_entropy']])
            
            # Tampilkan dalam format yang bisa di-copy
            st.write("**Log Anomali (format text):**")
            st.code('\n'.join(anomalies['query_name'].tolist()), language='text')

else:
    st.info("Silakan unggah file log DNS untuk memulai.")

st.write("---")
st.markdown("Dibuat untuk Final Project Mata Kuliah SOC di ITS - Juni 2025")