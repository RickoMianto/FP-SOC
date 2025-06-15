import streamlit as st
import pandas as pd
import os
# Import kelas dari "mesin" model kita
from src.model import DNSAnomalyDetector

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="AIOps Deteksi Anomali DNS", layout="wide")
st.title("🔎 AIOps: Deteksi Anomali DNS")
st.write("Aplikasi ini menggunakan model Machine Learning untuk mendeteksi anomali pada log DNS secara proaktif.")

# --- Logika Aplikasi ---

# Cek apakah file whitelist ada, jika tidak, hentikan aplikasi
whitelist_path = 'data/whitelist.csv'
if not os.path.exists(whitelist_path):
    st.error(f"PENTING: File whitelist tidak ditemukan di `{whitelist_path}`. Silakan unduh dan letakkan file tersebut terlebih dahulu.")
    st.stop()

# Fungsi untuk memuat data (dengan cache agar lebih cepat)
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Muat data whitelist
df_whitelist = load_data(whitelist_path)

# --- Antarmuka Pengguna ---
st.subheader("1. Unggah File Log DNS Anda")
uploaded_file = st.file_uploader(
    "Pilih file .csv yang berisi log DNS Anda", 
    type="csv"
)

if uploaded_file is not None:
    df_logs = load_data(uploaded_file)
    st.success(f"File `{uploaded_file.name}` berhasil diunggah dengan {len(df_logs)} baris.")
    
    st.subheader("2. Atur Sensitivitas Model")
    contamination_rate = st.slider(
        "Sensitivitas Deteksi (Contamination Rate)", 
        min_value=0.001, max_value=0.1, value=0.025, step=0.001, format="%.3f",
        help="Nilai lebih tinggi = lebih sensitif (lebih banyak anomali). Nilai lebih rendah = lebih selektif."
    )

    st.subheader("3. Jalankan Analisis")
    if st.button("🚀 Analisis Sekarang!", type="primary"):
        with st.spinner('Proses... Model sedang dilatih dengan whitelist dan menganalisis log Anda...'):
            try:
                # 1. Buat instance dari "mesin" model
                detector = DNSAnomalyDetector(contamination=contamination_rate)
                
                # 2. Latih "mesin" dengan data whitelist
                detector.fit(df_whitelist)
                
                # 3. Gunakan "mesin" untuk prediksi
                predictions = detector.predict(df_logs)
                
                # Tambahkan hasil prediksi ke dataframe
                df_logs['is_anomaly'] = predictions
                anomalies = df_logs[df_logs['is_anomaly'] == -1]

                st.success("Analisis Selesai!")
                
                # Tampilkan hasil
                st.subheader("🚨 Hasil Deteksi Anomali")
                if anomalies.empty:
                    st.info("Tidak ada anomali yang terdeteksi dengan tingkat sensitivitas ini.")
                else:
                    st.warning(f"Ditemukan **{len(anomalies)}** potensi anomali:")
                    st.dataframe(anomalies[['timestamp', 'source_ip', 'query_name']])

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")