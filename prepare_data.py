"""
Bersihkan CSV mentah Kaggle menjadi data siap pakai untuk aplikasi.

Contoh:
    python prepare_data.py --input cleaned_makeup_products.csv --output data/products_clean.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from recommender import clean_products

KEEP = [
    "product_link_id", "product_link", "product_name", "brand", "category",
    "price", "average_rating", "rating_count", "review_count",
    "description", "summary", "pros", "best_uses",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="cleaned_makeup_products.csv dari Kaggle")
    ap.add_argument("--output", default="data/products_clean.csv")
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    df, report = clean_products(raw)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[[c for c in KEEP if c in df.columns]].to_csv(out, index=False)

    print("Laporan pembersihan:")
    for key, val in report.items():
        print(f"  {key:35s}: {val}")
    print(f"\nDisimpan ke {out} ({len(df)} produk)")


if __name__ == "__main__":
    main()
