"""
Aplikasi web Streamlit: Sistem Rekomendasi Produk Makeup (Content-Based Filtering).

Jalankan lokal:
    streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from recommender import UNKNOWN, Recommender, clean_products, evaluate

DATA_PATH = Path(__file__).parent / "data" / "products_clean.csv"

st.set_page_config(page_title="Rekomendasi Produk Makeup", page_icon="💄", layout="wide")


# --------------------------------------------------------------------------
# Data & model (di-cache agar tidak dihitung ulang setiap interaksi)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat dan membersihkan data...")
def load_from_bytes(content: bytes) -> tuple[pd.DataFrame, dict]:
    return clean_products(pd.read_csv(io.BytesIO(content)))


@st.cache_data(show_spinner="Memuat dan membersihkan data...")
def load_from_path(path: str) -> tuple[pd.DataFrame, dict]:
    return clean_products(pd.read_csv(path))


@st.cache_resource(show_spinner="Membangun model TF-IDF...")
def get_model(df: pd.DataFrame) -> Recommender:
    return Recommender.fit(df)


@st.cache_data(show_spinner="Menghitung evaluasi untuk seluruh katalog...")
def run_evaluation(df: pd.DataFrame, k: int) -> dict:
    return evaluate(Recommender.fit(df), k=k)


# --------------------------------------------------------------------------
# Sidebar: sumber data dan pengaturan
# --------------------------------------------------------------------------
st.sidebar.title("💄 Pengaturan")

uploaded = st.sidebar.file_uploader(
    "Unggah CSV produk (opsional)",
    type="csv",
    help="File `cleaned_makeup_products.csv` dari Kaggle atau hasil `prepare_data.py`. "
         "Jika kosong, aplikasi memakai data/products_clean.csv.",
)
if uploaded is not None:
    df, clean_report = load_from_bytes(uploaded.getvalue())
    source_label = f"Unggahan: {uploaded.name}"
elif DATA_PATH.exists():
    df, clean_report = load_from_path(str(DATA_PATH))
    source_label = "data/products_clean.csv"
else:
    st.title("Sistem Rekomendasi Produk Makeup")
    st.warning(
        "Data belum tersedia. Unggah `cleaned_makeup_products.csv` lewat sidebar, "
        "atau jalankan `python prepare_data.py --input cleaned_makeup_products.csv` "
        "lalu simpan hasilnya di folder `data/`."
    )
    st.stop()

if df.empty:
    st.error("Tidak ada produk valid setelah pembersihan data.")
    st.stop()

model = get_model(df)

k = st.sidebar.slider("Jumlah rekomendasi (k)", 3, 15, 5)

with st.sidebar.expander("Bobot kemiripan", expanded=False):
    st.caption("Skor = rata-rata berbobot dari ketiga komponen di bawah.")
    w_text = st.slider("Kemiripan teks (nama, deskripsi, kegunaan)", 0.0, 1.0, 0.7, 0.05)
    w_cat = st.slider("Kategori sama", 0.0, 1.0, 0.3, 0.05)
    w_brand = st.slider("Merek sama", 0.0, 1.0, 0.0, 0.05)

st.sidebar.subheader("Filter hasil")
exclude_same_brand = st.sidebar.toggle(
    "Hanya merek lain", value=False,
    help="Berguna untuk mencari alternatif produk dari merek berbeda.",
)
same_category_only = st.sidebar.toggle("Hanya kategori yang sama", value=False)

prices = df["price"].dropna()
max_price = None
if not prices.empty:
    hi = float(np.ceil(prices.max()))
    chosen = st.sidebar.slider("Harga maksimum (USD)", 0.0, hi, hi, 1.0)
    max_price = None if chosen >= hi else chosen

min_rating = st.sidebar.slider("Rating minimum", 0.0, 5.0, 0.0, 0.1)
st.sidebar.caption(f"Sumber data: {source_label} · {len(df):,} produk")


# --------------------------------------------------------------------------
# Halaman utama
# --------------------------------------------------------------------------
st.title("Sistem Rekomendasi Produk Makeup")
st.caption(
    "Content-based filtering: TF-IDF atas nama produk, ringkasan deskripsi, kegunaan "
    "(best uses) dan kelebihan (pros), dikombinasikan dengan kesamaan kategori dan merek."
)

tab_rec, tab_data, tab_eval = st.tabs(["🔎 Rekomendasi", "📊 Data", "✅ Evaluasi"])

# ---- Tab Rekomendasi ------------------------------------------------------
with tab_rec:
    c1, c2 = st.columns([1, 2])
    with c1:
        cat_options = ["Semua kategori"] + sorted(df["category"].unique().tolist())
        cat_pick = st.selectbox("Saring daftar produk berdasarkan kategori", cat_options)
    pool = df if cat_pick == "Semua kategori" else df[df["category"] == cat_pick]
    labels = {
        row.product_link_id: f"{row.product_name} — {row.brand}"
        for row in pool.itertuples()
    }
    with c2:
        pid = st.selectbox(
            "Pilih produk yang disukai (ketik untuk mencari)",
            options=list(labels.keys()),
            format_func=labels.get,
        )

    item = df.loc[df["product_link_id"] == pid].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Merek", item["brand"])
    m2.metric("Kategori", item["category"])
    m3.metric("Harga", "-" if pd.isna(item["price"]) else f"${item['price']:.2f}")
    m4.metric("Rating", "-" if pd.isna(item["average_rating"]) else f"{item['average_rating']:.2f} ★")
    with st.expander("Ringkasan deskripsi produk"):
        st.write(item["summary"] or "_Tidak ada ringkasan._")
        if isinstance(item.get("product_link"), str) and item["product_link"].startswith("http"):
            st.markdown(f"[Buka halaman produk]({item['product_link']})")

    categories = [item["category"]] if same_category_only and item["category"] != UNKNOWN else None
    result = model.recommend(
        pid, k=k, w_text=w_text, w_cat=w_cat, w_brand=w_brand,
        exclude_same_brand=exclude_same_brand, categories=categories,
        max_price=max_price, min_rating=min_rating,
    )

    st.subheader(f"Top-{k} rekomendasi")
    if result.empty:
        st.info("Tidak ada produk yang lolos filter. Longgarkan filter di sidebar.")
    else:
        if len(result) < k:
            st.caption(f"Hanya {len(result)} produk yang lolos filter.")
        st.dataframe(
            result[["score", "text_similarity", "product_name", "brand", "category",
                    "price", "average_rating", "product_link"]],
            hide_index=True,
            width="stretch",
            column_config={
                "score": st.column_config.ProgressColumn(
                    "Skor", min_value=0.0, max_value=1.0, format="%.3f",
                    help="Skor gabungan berbobot (0–1)"),
                "text_similarity": st.column_config.NumberColumn(
                    "Kemiripan teks", format="%.3f", help="Cosine similarity TF-IDF"),
                "product_name": "Produk",
                "brand": "Merek",
                "category": "Kategori",
                "price": st.column_config.NumberColumn("Harga", format="$%.2f"),
                "average_rating": st.column_config.NumberColumn("Rating", format="%.2f ★"),
                "product_link": st.column_config.LinkColumn("Tautan", display_text="Lihat"),
            },
        )
        st.caption("Seri skor dipecah dengan rating tertinggi.")

# ---- Tab Data -------------------------------------------------------------
with tab_data:
    st.subheader("Hasil pembersihan data")
    rep = pd.DataFrame(
        {"Langkah": [key.replace("_", " ") for key in clean_report],
         "Jumlah": list(clean_report.values())}
    )
    st.dataframe(rep, hide_index=True, width="content")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Produk per kategori")
        st.bar_chart(df["category"].value_counts(), horizontal=True)
    with col_b:
        st.subheader("10 merek dengan produk terbanyak")
        st.bar_chart(df["brand"].value_counts().head(10), horizontal=True)

    with st.expander("Lihat tabel produk"):
        st.dataframe(
            df[["product_link_id", "product_name", "brand", "category", "price",
                "average_rating", "rating_count"]],
            hide_index=True, width="stretch",
        )

# ---- Tab Evaluasi ---------------------------------------------------------
with tab_eval:
    st.subheader("Evaluasi offline")
    st.markdown(
        "Relevansi didekati dengan **kategori yang sama**. Agar tidak bocor, evaluasi "
        "memakai **kemiripan teks saja** (kategori tidak dipakai sebagai fitur). "
        "Jika kategori ikut menjadi fitur, Precision@k pasti mendekati 1 sehingga tidak informatif."
    )
    eval_k = st.select_slider("k untuk evaluasi", options=[3, 5, 10, 15], value=5)
    if st.button("Hitung evaluasi", type="primary"):
        r = run_evaluation(df, eval_k)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"Precision@{eval_k}", f"{r['precision_at_k']:.3f}")
        e2.metric("Baseline acak", f"{r['baseline_random']:.3f}")
        e3.metric("Lift", f"{r['lift']:.2f}×")
        e4.metric("Cakupan katalog", f"{r['catalog_coverage']:.1%}")
        st.caption(
            f"{r['n_queries_evaluated']} dari {r['n_products']} produk dievaluasi "
            "(kategori diketahui dan punya cukup produk sekategori). "
            f"Rata-rata rating produk yang direkomendasikan: {r['mean_rating_rec']:.2f} "
            f"(rata-rata katalog: {r['mean_rating_catalog']:.2f})."
        )
        st.dataframe(r["per_category"], hide_index=True, width="stretch")
        st.info(
            "Keterbatasan: kategori sama hanyalah proxy relevansi. Validasi yang lebih kuat "
            "membutuhkan data interaksi pengguna (pembelian/ulasan per pengguna) atau "
            "uji pengguna (user study)."
        )
