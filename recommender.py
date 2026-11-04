"""
Sistem rekomendasi produk makeup berbasis konten (content-based filtering).

Versi perbaikan dari notebook "Makeup Recommendation System":
  1. Pembersihan data tidak lagi membuang produk secara tidak sengaja
     (notebook asli menyusut dari 1.373 menjadi 35 produk karena
     drop_duplicates(subset='brand') menimpa dataframe utama).
  2. Kemiripan dihitung dari teks gabungan (nama, ringkasan deskripsi,
     best_uses, pros), bukan hanya kolom kategori, sehingga skor bervariasi
     dan peringkat bermakna (tidak lagi seri 1,0 untuk semua produk sekategori).
  3. Skor akhir = kombinasi berbobot kemiripan teks, kesamaan kategori,
     dan kesamaan merek; bobot dapat diatur dari aplikasi.
  4. Evaluasi memakai Precision@k terhadap baseline acak dan cakupan katalog,
     bukan Davies-Bouldin (yang mengukur clustering, bukan rekomendasi).

Skema kolom mengikuti file `cleaned_makeup_products.csv` dari dataset Kaggle
zarasarkar/makeup-insights-customer-reviews.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = ["product_link_id", "product_name", "description"]
OPTIONAL_COLUMNS = [
    "product_link", "category", "brand", "price", "average_rating",
    "rating_count", "review_count", "pros", "best_uses",
]
# Nilai "brand" yang sebenarnya artefak scraping halaman Ulta, bukan merek.
INVALID_BRANDS = {"", "nan", "none", "find your shade", "ask a question"}
UNKNOWN = "Unknown"

# Kolom teks yang digabung menjadi representasi konten tiap produk.
TEXT_FIELDS = ["product_name", "summary", "best_uses", "pros"]


# --------------------------------------------------------------------------
# Pembersihan data
# --------------------------------------------------------------------------
def _extract_brand(description: str, product_name: str) -> str | None:
    """Ambil merek dari deskripsi hasil scraping.

    Pola deskripsi: '... navigate each product image <Merek> <Nama Produk> Item ...'
    """
    pattern = r"product image\s+(.+?)\s+" + re.escape(product_name)
    match = re.search(pattern, description)
    if not match:
        return None
    brand = match.group(1).strip()
    if 0 < len(brand) <= 40 and brand.lower() not in INVALID_BRANDS:
        return brand
    return None


def summarize_description(description: str) -> str:
    """Ambil bagian deskripsi yang informatif (Summary s.d. sebelum How To Use).

    Bagian navigasi halaman, daftar ingredients lengkap, dan info pengiriman
    dibuang karena menambah noise pada TF-IDF.
    """
    text = str(description)
    start = text.find(" Summary ")
    if start != -1:
        text = text[start + len(" Summary "):]
    end = text.find(" How To Use ")
    if end == -1:
        m = re.search(r"(?<!Key) Ingredients ", text)
        end = m.start() if m else -1
    if end != -1:
        text = text[:end]
    cut = text.find(" Shipping & Coupon")
    if cut != -1:
        text = text[:cut]
    text = text.replace("Conscious Beauty at Ulta Beauty™", " ")
    text = re.sub(
        r"\b(Details|Benefits|Key Ingredients|Formulated Without|Research Results)\b",
        " ", text,
    )
    return re.sub(r"\s+", " ", text).strip()


def clean_products(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bersihkan data produk tanpa membuang produk yang masih valid.

    Mengembalikan (dataframe bersih, laporan pembersihan).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    report: dict[str, int] = {"baris_awal": len(raw)}
    df = raw.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # 1) Baris tanpa nama atau deskripsi tidak bisa direkomendasikan.
    df = df.dropna(subset=["product_name", "description"])
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["description"] = df["description"].astype(str)
    report["dibuang_tanpa_nama_atau_deskripsi"] = report["baris_awal"] - len(df)

    # 2) Duplikat ID produk.
    before = len(df)
    df = df.drop_duplicates(subset="product_link_id")
    report["dibuang_duplikat_id"] = before - len(df)

    # 3) Merek: perbaiki artefak scraping ("Find your shade", "Ask A Question", NaN).
    brands, repaired = [], 0
    for brand, desc, name in zip(df["brand"], df["description"], df["product_name"]):
        b = "" if pd.isna(brand) else str(brand).strip()
        if b.lower() in INVALID_BRANDS:
            extracted = _extract_brand(desc, name)
            if extracted:
                repaired += 1
            b = extracted or UNKNOWN
        brands.append(b)
    df["brand"] = brands
    report["merek_diperbaiki_dari_deskripsi"] = repaired
    report["merek_tetap_unknown"] = int((df["brand"] == UNKNOWN).sum())

    # 4) Kategori kosong -> "Unknown" (produk tetap dipertahankan).
    df["category"] = df["category"].fillna(UNKNOWN).astype(str).str.strip()
    df.loc[df["category"] == "", "category"] = UNKNOWN
    report["kategori_unknown"] = int((df["category"] == UNKNOWN).sum())

    # 5) Numerik. Rating 0 dengan jumlah rating 0 berarti "belum dinilai".
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    rc = pd.to_numeric(df["rating_count"], errors="coerce")
    df.loc[(rc == 0) & (df["average_rating"] == 0), "average_rating"] = np.nan
    df["rating_count"] = rc.fillna(0).astype(int)

    # 6) Varian produk yang sama (nama + merek sama): simpan yang ratingnya terbanyak.
    before = len(df)
    df = (df.sort_values("rating_count", ascending=False)
            .drop_duplicates(subset=["product_name", "brand"]))
    report["dibuang_varian_nama_merek_sama"] = before - len(df)

    # 7) Teks fitur.
    df["summary"] = df["description"].map(summarize_description)
    for col in ["pros", "best_uses"]:
        df[col] = df[col].fillna("").astype(str)

    df = df.sort_values("product_link_id").reset_index(drop=True)
    report["baris_akhir"] = len(df)
    return df, report


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class Recommender:
    df: pd.DataFrame
    text_matrix: sparse.csr_matrix
    vectorizer: TfidfVectorizer

    @classmethod
    def fit(cls, df: pd.DataFrame, text_fields: list[str] | None = None) -> "Recommender":
        fields = text_fields or TEXT_FIELDS
        soup = df[fields].fillna("").astype(str).agg(" ".join, axis=1)
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2 if len(df) >= 50 else 1,
            max_df=0.85,
            sublinear_tf=True,
        )
        # Baris dinormalisasi L2 (default), sehingga hasil kali titik = cosine similarity.
        matrix = vectorizer.fit_transform(soup).tocsr()
        return cls(df=df.reset_index(drop=True), text_matrix=matrix, vectorizer=vectorizer)

    # -- utilitas ---------------------------------------------------------
    def index_of(self, product_link_id) -> int:
        hits = np.flatnonzero(self.df["product_link_id"].to_numpy() == product_link_id)
        if hits.size == 0:
            raise KeyError(f"product_link_id {product_link_id} tidak ditemukan")
        return int(hits[0])

    def scores(self, idx: int, w_text: float = 0.7, w_cat: float = 0.3,
               w_brand: float = 0.0) -> dict[str, np.ndarray]:
        """Skor kemiripan produk ke-idx terhadap semua produk."""
        text_sim = (self.text_matrix @ self.text_matrix[idx].T).toarray().ravel()

        cats = self.df["category"].to_numpy()
        cat_sim = ((cats == cats[idx]) & (cats[idx] != UNKNOWN)).astype(float)

        brands = self.df["brand"].to_numpy()
        brand_sim = ((brands == brands[idx]) & (brands[idx] != UNKNOWN)).astype(float)

        total = w_text + w_cat + w_brand
        if total <= 0:
            w_text, total = 1.0, 1.0
        combined = (w_text * text_sim + w_cat * cat_sim + w_brand * brand_sim) / total
        return {"score": combined, "text": text_sim, "category": cat_sim, "brand": brand_sim}

    def recommend(
        self,
        product_link_id,
        k: int = 5,
        w_text: float = 0.7,
        w_cat: float = 0.3,
        w_brand: float = 0.0,
        exclude_same_brand: bool = False,
        categories: list[str] | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
    ) -> pd.DataFrame:
        """Top-k rekomendasi untuk satu produk, dengan filter opsional."""
        idx = self.index_of(product_link_id)
        s = self.scores(idx, w_text, w_cat, w_brand)
        df = self.df

        mask = np.ones(len(df), dtype=bool)
        mask[idx] = False
        if exclude_same_brand:
            mask &= df["brand"].to_numpy() != df.at[idx, "brand"]
        if categories:
            mask &= df["category"].isin(categories).to_numpy()
        if max_price is not None:
            mask &= (df["price"].fillna(np.inf) <= max_price).to_numpy()
        if min_rating is not None and min_rating > 0:
            mask &= (df["average_rating"].fillna(0) >= min_rating).to_numpy()

        cand = np.flatnonzero(mask)
        if cand.size == 0:
            return df.iloc[[]].assign(score=[], text_similarity=[])
        # Urutkan: skor tertinggi dulu; seri dipecah dengan rating tertinggi.
        rating = df["average_rating"].fillna(0).to_numpy()
        order = np.lexsort((-rating[cand], -s["score"][cand]))
        top = cand[order[:k]]

        out = df.iloc[top].copy()
        out.insert(0, "score", s["score"][top])
        out.insert(1, "text_similarity", s["text"][top])
        return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Evaluasi offline
# --------------------------------------------------------------------------
def evaluate(rec: Recommender, k: int = 5, w_text: float = 1.0,
             w_cat: float = 0.0, w_brand: float = 0.0) -> dict:
    """Evaluasi offline berbasis proxy relevansi = kategori sama.

    Agar tidak bocor, evaluasi default memakai kemiripan teks saja (w_cat=0):
    kalau kategori ikut dipakai sebagai fitur, Precision@k kategori menjadi
    trivial mendekati 1 (seperti pada notebook asli).

    Metrik:
      - precision_at_k : rata-rata proporsi rekomendasi berkategori sama
      - baseline_random: nilai harapan precision@k jika rekomendasi diambil acak
      - lift           : precision_at_k / baseline_random
      - catalog_coverage: proporsi produk yang pernah muncul di daftar top-k
      - mean_rating_rec : rata-rata rating produk yang direkomendasikan
    """
    df = rec.df
    n = len(df)
    cats = df["category"].to_numpy()
    cat_counts = df["category"].value_counts()
    rating = df["average_rating"].to_numpy(dtype=float)

    precisions, baselines, rec_ratings = [], [], []
    per_cat: dict[str, list[float]] = {}
    seen: set[int] = set()

    for idx in range(n):
        s = rec.scores(idx, w_text, w_cat, w_brand)["score"].copy()
        s[idx] = -np.inf
        top = np.argsort(-s, kind="stable")[:k]
        seen.update(top.tolist())
        rec_ratings.extend(rating[top][~np.isnan(rating[top])].tolist())

        c = cats[idx]
        # Hanya produk dengan kategori diketahui dan punya >= k "tetangga" sekategori.
        if c == UNKNOWN or cat_counts[c] - 1 < k:
            continue
        p = float(np.mean(cats[top] == c))
        precisions.append(p)
        baselines.append((cat_counts[c] - 1) / (n - 1))
        per_cat.setdefault(c, []).append(p)

    precision = float(np.mean(precisions)) if precisions else float("nan")
    baseline = float(np.mean(baselines)) if baselines else float("nan")
    per_category = (
        pd.DataFrame(
            [{"kategori": c, "jumlah_query": len(v), f"precision@{k}": float(np.mean(v))}
             for c, v in per_cat.items()]
        ).sort_values(f"precision@{k}", ascending=False).reset_index(drop=True)
        if per_cat else pd.DataFrame(columns=["kategori", "jumlah_query", f"precision@{k}"])
    )
    return {
        "k": k,
        "n_products": n,
        "n_queries_evaluated": len(precisions),
        "precision_at_k": precision,
        "baseline_random": baseline,
        "lift": precision / baseline if baseline and baseline > 0 else float("nan"),
        "catalog_coverage": len(seen) / n if n else float("nan"),
        "mean_rating_rec": float(np.mean(rec_ratings)) if rec_ratings else float("nan"),
        "mean_rating_catalog": float(np.nanmean(rating)) if np.any(~np.isnan(rating)) else float("nan"),
        "per_category": per_category,
    }
