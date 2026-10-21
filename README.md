# Analisis Computational Overhead pada Pipeline Inferensi Deteksi Fraud Berbasis XGBoost Tanpa dan dengan TreeSHAP

Program ini mengukur tambahan waktu dan sumber daya untuk menjalankan
TreeSHAP pada inferensi XGBoost. Pengukuran dilakukan pada dua model dengan
ukuran batch 1, 10, 50, dan 100 transaksi.

- **A:** penyesuaian masukan dan inferensi XGBoost.
- **B:** tahapan A ditambah TreeSHAP.

Metrik yang dicatat meliputi latency, throughput, CPU, puncak RSS, dan
Top-5 SHAP Coverage. Perbandingan A/B dilakukan pada setiap model untuk
melihat konsistensi pola overhead.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
adalah dataset transaksi e-commerce Vesta yang digunakan dalam kompetisi
Kaggle tahun 2019. Data berlabel berisi 590.540 transaksi, termasuk 20.663 fraud.

CSV lokal diperoleh dari [mirror Hugging Face](https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/tree/c908df307713c748cc348e3049369f16333e2b5c).
Checksum dicatat dalam [models.json](models.json); kesamaan byte dengan CSV
resmi Kaggle belum diperiksa. Data uji bersama terdiri atas 13.361 transaksi
yang berada dalam test kedua model, termasuk 475 fraud.

## Model

| Model | Sumber | Pohon | Kedalaman maksimum | Fitur |
| --- | --- | ---: | ---: | ---: |
| M1 | XGBoost teacher Kang dan Kim (2026), seed 42 | 3.000 | 7 | 154 |
| M2 | XGBoost FraudLens | 400 | 6 | 42 |

**M1** berasal dari [artikel Kang dan Kim (2026)](https://doi.org/10.3390/app16125809).
Model tersedia di [lampiran resmi jurnal](https://mdpi-res.com/d_attachment/applsci/applsci-16-05809/article_deploy/applsci-16-05809-s001.zip),
pada `main_out_of_time/seed_42/teacher_model.json`. Penelitian ini memakai
XGBoost teacher yang disertakan penulis, sedangkan artikel tersebut
mengembangkan hybrid EBM.

**M2** berasal dari [proyek FraudLens](https://github.com/praju120056/FraudLens/tree/74162be78808aa2dd3a0def581d2a28c626929c4).
[Model sumber](https://github.com/praju120056/FraudLens/blob/74162be78808aa2dd3a0def581d2a28c626929c4/backend/model/model.pkl)
dikonversi dari bundle PKL ke JSON native XGBoost tanpa pelatihan ulang;
preprocessing disimpan terpisah. Hasil prediksi sebelum dan sesudah konversi
identik pada 88.581 transaksi test sumber. Publikasi jurnal dan lisensi
eksplisit untuk model ini belum ditemukan. Repo ini menyediakan tautan sumber
tanpa mengunggah ulang bobot model dan preprocessing.

## Hasil pilot

Uji pendahuluan menggunakan 128 transaksi per replikasi, dua pasangan A/B
untuk setiap kombinasi model dan batch, serta satu warm-up per skenario.
Latency adalah waktu seluruh 128 transaksi; overhead dihitung per transaksi.

| Model | Batch | Latency A (s) | Latency B (s) | Overhead (ms/transaksi) |
| --- | ---: | ---: | ---: | ---: |
| M1 | 1 | 0.2187 | 5.7319 | 43.0724 |
| M1 | 10 | 0.0571 | 2.8140 | 21.5386 |
| M1 | 50 | 0.0137 | 2.6558 | 20.6416 |
| M1 | 100 | 0.0135 | 2.7557 | 21.4235 |
| M2 | 1 | 0.0592 | 0.5772 | 4.0470 |
| M2 | 10 | 0.0093 | 0.2577 | 1.9407 |
| M2 | 50 | 0.0031 | 0.2300 | 1.7726 |
| M2 | 100 | 0.0022 | 0.2522 | 1.9533 |

[Ringkasan pilot CSV](results/pilot/Ringkasan%20Pilot.csv).
Hasil ini digunakan untuk memeriksa jalannya program. Pengambilan data utama
akan dilakukan pada sesi terpisah dengan replikasi yang lebih banyak.

## Menjalankan

Lengkapi repo dengan model, resource preprocessing, dan paket fitur uji
bersama yang cocok dengan checksum dalam `models.json`. Folder `models/`
dan `resources/` tersedia di workspace penelitian dan tidak disertakan dalam
repo. Mengunduh kode saja belum cukup untuk menjalankan benchmark.

```text
Benchmark TreeSHAP/
  models/M1/teacher_model.json
  models/M2/fraudlens_native.json
  resources/
Dataset + Model/
  Dataset IEEE-CIS/
    Uji Bersama Dua Model/
```

Dengan Docker Desktop aktif, jalankan dari folder kode:

```powershell
docker build -f Dockerfile.runtime -t fraud-benchmark:primary-20261016 .
.\run-docker.ps1 -DataPath "D:\Data\Uji Bersama Dua Model" -RunKind pilot
```

Pilot bawaan perintah memakai 1.000 transaksi. Untuk pengambilan data utama,
ganti `-RunKind pilot` menjadi `-RunKind formal`. Protokol formal memakai
1.000 transaksi, 30 pasangan per model-batch, dan lima warm-up per skenario.
Container diberi kuota dua CPU dan 4 GiB tanpa swap. Alokasi WSL/host perlu
lebih besar daripada kuota container.

Pembacaan dataset, pembentukan fitur, pemuatan model, dan penyimpanan hasil
berada di luar timer. A/B berbagi allocator, sehingga puncak RSS tidak
mengisolasi tambahan memori TreeSHAP. Fitur dan pelatihan M1/M2 berbeda;
selisih overhead antarmodel tidak hanya dipengaruhi jumlah pohon.
Preprocessing M1 mempertahankan fallback median seluruh data dari sumber,
sedangkan M2 memakai split acak. Hasilnya terbatas pada model, data historis,
dan lingkungan pengukuran ini.
