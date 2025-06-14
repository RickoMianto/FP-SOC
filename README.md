# FP-SOC

Tentu, berikut adalah penjelasan setiap fitur dari proyek deteksi anomali DNS ini dalam format Markdown, seperti yang Anda minta.

### Penjelasan Fitur (Features)

Berikut adalah penjelasan untuk setiap kolom (fitur) yang ada di dalam dataset `dns_log.csv` dan fitur-fitur yang dihasilkan oleh proses *feature engineering* dalam skrip Python.

#### **Fitur Awal (dari `dns_log.csv`)**

* **`timestamp`**:
    * Waktu kapan permintaan (query) DNS dicatat.
    * Penting untuk analisis forensik dan korelasi waktu antar peristiwa.

* **`source_ip`**:
    * Alamat IP dari perangkat yang melakukan permintaan DNS.
    * Membantu mengidentifikasi sumber aktivitas, apakah itu dari server internal, workstation, atau perangkat lain.

* **`query_name`**:
    * Nama domain yang diminta oleh klien. Ini adalah fitur paling penting karena namanya sering kali menjadi indikator utama aktivitas berbahaya.
    * **Contoh Normal**: `google.com`, `integra.its.ac.id`.
    * **Contoh Anomali**: `9a8f7c6e5d4b3a2c1b0a9f8e7d6c5b4a.com` (mirip DGA), `dGhpc2lzc29tZXNlY3JldGRhdGE=.tunnel.example.org` (potensi tunneling).

* **`tld` (Top-Level Domain)**:
    * Bagian akhir dari nama domain (misalnya, `.com`, `.org`, `.ru`, `.xyz`).
    * Beberapa TLD seperti `.xyz`, `.top`, atau `.online` sering disalahgunakan untuk aktivitas berbahaya karena biaya registrasinya yang murah.

* **`numeric_ratio`**:
    * Rasio atau perbandingan antara jumlah karakter angka dengan total panjang `query_name`.
    * Nama domain yang dibuat oleh algoritma (DGA) sering kali memiliki rasio numerik yang tinggi.

* **`has_hyphen`**:
    * Sebuah flag (1 jika ada, 0 jika tidak) yang menandakan apakah `query_name` mengandung tanda hubung (`-`).
    * Phishing sering menggunakan tanda hubung untuk meniru domain yang sah (contoh: `paypal-secure.com`).

---

#### **Fitur Hasil Rekayasa (Feature Engineering)**

Fitur-fitur ini dibuat dari data mentah untuk membantu model *machine learning* mengenali pola anomali dengan lebih baik. Definisi fungsi-fungsi ini dapat ditemukan di `src/analysis_utils.py`.

* **`query_length`**:
    * Panjang total dari `query_name`.
    * Nama domain yang sangat panjang sering kali menjadi ciri khas dari DNS tunneling atau DGA, di mana data atau ID unik disisipkan ke dalam nama subdomain.
    * **Rumus**: $panjang('query\_name')$.

* **`query_entropy`**:
    * Mengukur tingkat keacakan karakter dalam `query_name` menggunakan **Shannon Entropy**.
    * Nama domain yang dihasilkan oleh DGA (*Domain Generation Algorithm*) malware cenderung memiliki tingkat keacakan (entropi) yang tinggi karena tidak mengikuti pola bahasa manusia. Sebaliknya, domain normal seperti `google.com` memiliki entropi rendah.
    * **Rumus**: $H(S) = -\sum_{i} p(x_i) \log_{2} p(x_i)$, di mana $S$ adalah string query dan $p(x_i)$ adalah probabilitas kemunculan karakter $x_i$.

* **`subdomain_count`**:
    * Jumlah titik (`.`) dalam `query_name`, yang merepresentasikan jumlah tingkatan subdomain.
    * DNS tunneling sering kali menggunakan banyak subdomain untuk menyandikan data yang akan dieksfiltrasi.
    * **Contoh**: `data.payload.internal.attacker.com` memiliki 4 subdomain.
    * **Rumus**: $jumlah(' . ')$.

Model `IsolationForest` yang digunakan dalam proyek ini akan memanfaatkan kombinasi fitur-fitur di atas untuk mengisolasi dan mendeteksi data yang perilakunya menyimpang dari mayoritas data normal.
