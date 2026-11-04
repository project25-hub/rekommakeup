# Sistem Rekomendasi Produk Makeup — Aplikasi Web Streamlit

Aplikasi web interaktif untuk sistem rekomendasi produk makeup berbasis
**content-based filtering** (TF-IDF + cosine similarity), dikembangkan dari notebook
*Makeup Recommendation System*. Data: Kaggle `zarasarkar/makeup-insights-customer-reviews`
(file `cleaned_makeup_products.csv`).

## Fitur aplikasi

- **Rekomendasi**: pilih produk (bisa dicari dengan mengetik), tampilkan top-*k* produk mirip
  beserta skor, kemiripan teks, harga, rating, dan tautan produk.
- **Pengaturan bobot**: kemiripan teks, kesamaan kategori, dan kesamaan merek.
- **Filter**: hanya merek lain (mencari alternatif), hanya kategori yang sama,
  harga maksimum, dan rating minimum.
- **Tab Data**: laporan pembersihan data, distribusi kategori dan merek.
- **Tab Evaluasi**: Precision@k terhadap baseline acak, lift, dan cakupan katalog.
- **Unggah CSV** dari sidebar bila file data belum disertakan.

## Struktur

```
makeup-recsys/
├── app.py                 # Aplikasi Streamlit
├── recommender.py         # Pembersihan data, model, evaluasi
├── prepare_data.py        # CSV mentah Kaggle -> data/products_clean.csv
├── evaluate.py            # Evaluasi offline untuk laporan
├── requirements.txt
├── .streamlit/config.toml
├── data/                  # Tempat products_clean.csv
└── tests/                 # Uji otomatis (memakai data SINTETIS, bukan data asli)
```

## Menjalankan secara lokal

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Unduh dataset dari halaman Kaggle (manual, atau Kaggle API di komputer sendiri),
#    lalu bersihkan:
python prepare_data.py --input cleaned_makeup_products.csv --output data/products_clean.csv

# 2. (Opsional) Evaluasi offline:
python evaluate.py --data data/products_clean.csv --k 5 10

# 3. Jalankan aplikasi:
streamlit run app.py
```

Uji otomatis: `pip install pytest` lalu `python -m pytest -q`.

## Deploy ke Streamlit Community Cloud

1. Buat repositori GitHub dan unggah seluruh isi folder ini, **termasuk**
   `data/products_clean.csv` (lihat catatan lisensi di bawah).
2. Pastikan `kaggle.json` **tidak** ikut ter-commit (sudah tercantum di `.gitignore`).
3. Masuk ke Streamlit Community Cloud dengan akun GitHub, pilih *Create app*,
   pilih repositori dan branch, isi *Main file path* dengan `app.py`, lalu *Deploy*.
4. Setiap `git push` ke branch tersebut akan memperbarui aplikasi.

Jika data tidak boleh diunggah ke repositori publik, aplikasi tetap bisa dipakai:
pengguna mengunggah CSV melalui sidebar.

## Perbaikan dibanding notebook asli

| Masalah di notebook | Perbaikan |
|---|---|
| `products_df = products_df.drop_duplicates(subset='brand')` di sel EDA menimpa data utama; produk menyusut dari 1.373 menjadi 35 | EDA tidak mengubah data utama; hanya duplikat ID, baris tanpa nama/deskripsi, dan varian nama+merek yang sama yang dibuang |
| `dropna()` pada `pros`, `cons`, `best_uses` membuang produk yang valid | Nilai kosong diisi string kosong; kategori kosong menjadi `Unknown` |
| Brand berisi artefak scraping ("Find your shade", "Ask A Question") | Merek diekstraksi ulang dari teks deskripsi |
| TF-IDF hanya dari kolom kategori: semua produk sekategori skor 1,0 sehingga peringkat ditentukan urutan data | TF-IDF (unigram+bigram, stop words Inggris) atas nama, ringkasan deskripsi, best uses, dan pros; dikombinasikan berbobot dengan kategori dan merek; seri dipecah dengan rating |
| Evaluasi Davies-Bouldin (mengukur clustering, bukan rekomendasi); narasi 0,85 vs output 0,13 | Precision@k (proxy relevansi = kategori sama, tanpa kategori sebagai fitur) dibandingkan baseline acak, lift, cakupan katalog |
| API key Kaggle tercetak di output notebook | Tidak ada kredensial di kode; `kaggle.json` di-ignore |

## Catatan metodologis untuk laporan

- Precision@k berbasis kategori adalah **proxy** relevansi, bukan relevansi menurut pengguna.
  Karena itu evaluasi sengaja tidak memakai kategori sebagai fitur (kalau dipakai, nilainya
  trivial mendekati 1).
- Bandingkan Precision@k dengan baseline acak (nilai harapan jika rekomendasi diambil acak);
  lift > 1 menunjukkan model lebih baik dari acak.
- Validasi yang lebih kuat membutuhkan data interaksi per pengguna (file
  `cleaned_makeup_reviews.csv` bisa dijajaki untuk collaborative filtering/hybrid)
  atau uji pengguna.
- Angka hasil di `tests/` berasal dari data sintetis dan **tidak boleh** dilaporkan.

## Keamanan

API key Kaggle yang pernah tercetak di output notebook harus segera dicabut/diganti
di pengaturan akun Kaggle, dan output sel tersebut dihapus sebelum notebook dibagikan.
