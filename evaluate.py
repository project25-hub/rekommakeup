"""
Evaluasi offline sistem rekomendasi (untuk dilaporkan di laporan/artikel).

Contoh:
    python evaluate.py --data data/products_clean.csv --k 5 10
"""
import argparse
import json

import pandas as pd

from recommender import Recommender, clean_products, evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/products_clean.csv")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--json", help="Opsional: simpan ringkasan ke file JSON")
    args = ap.parse_args()

    df, _ = clean_products(pd.read_csv(args.data))
    model = Recommender.fit(df)

    summary = []
    for k in args.k:
        r = evaluate(model, k=k)
        row = {key: val for key, val in r.items() if key != "per_category"}
        summary.append(row)
        print(f"\n=== k = {k} (kemiripan teks saja, kategori tidak dipakai sebagai fitur) ===")
        print(f"Produk                 : {r['n_products']}")
        print(f"Query dievaluasi       : {r['n_queries_evaluated']}")
        print(f"Precision@{k:<3d}          : {r['precision_at_k']:.4f}")
        print(f"Baseline acak          : {r['baseline_random']:.4f}")
        print(f"Lift                   : {r['lift']:.2f}x")
        print(f"Cakupan katalog        : {r['catalog_coverage']:.2%}")
        print(f"Rata-rata rating rekom.: {r['mean_rating_rec']:.2f} "
              f"(katalog: {r['mean_rating_catalog']:.2f})")
        print("\nPer kategori:")
        print(r["per_category"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nRingkasan disimpan ke {args.json}")


if __name__ == "__main__":
    main()
