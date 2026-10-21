# Analisis Computational Overhead pada Pipeline Inferensi Deteksi Fraud Berbasis XGBoost Tanpa dan dengan TreeSHAP

Penelitian skripsi ini mengukur tambahan waktu dan sumber daya ketika TreeSHAP
menjelaskan prediksi XGBoost. Perbandingan A/B dilakukan di dalam setiap model:

- **A:** penyesuaian masukan dan inferensi XGBoost.
- **B:** tahapan A ditambah perhitungan TreeSHAP.

Metrik meliputi latency, throughput, penggunaan CPU, puncak RSS dan Top-5
SHAP Coverage. Model kedua dipakai untuk memeriksa konsistensi pola overhead,
bukan untuk menentukan classifier terbaik.

**Status per 21 Oktober 2026:** proposal dan implementasi sudah disiapkan;
validasi fungsi dan pilot pasangan model sekarang sudah dilakukan. Data utama
pasangan model sekarang belum diambil. Proposal Word dan CSV pilot dibagikan
sebagai lampiran bimbingan.

## Dataset dan model

| Komponen | Sumber | Karakteristik |
| --- | --- | --- |
| Dataset | IEEE-CIS Fraud Detection, kompetisi 2019 | Data transaksi e-commerce Vesta; 590.540 transaksi berlabel, 20.663 fraud |
| M1 | XGBoost teacher Kang dan Kim (2026), seed 42 | 3.000 pohon, kedalaman maksimum 7, 154 fitur; kasus utama dari lampiran jurnal |
| M2 | XGBoost FraudLens, revisi tetap | 400 pohon, kedalaman maksimum 6, 42 fitur; kasus tambahan dari repositori publik |

Dataset IEEE-CIS bukan dataset terbaru. Paket uji bersama berisi 13.361 ID,
termasuk 475 fraud, yang berada dalam test kedua sumber. ID dan label sama,
tetapi matriks fitur serta riwayat pelatihan kedua model berbeda. Kedua model
dibekukan tanpa pelatihan ulang.

### Sumber dataset

- [Kompetisi resmi IEEE-CIS di Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
- [Mirror asal CSV lokal, pada revisi yang digunakan](https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/tree/c908df307713c748cc348e3049369f16333e2b5c).

Checksum CSV lokal dicatat dalam [models.json](models.json). CSV lokal belum
dibandingkan langsung secara byte dengan unduhan resmi Kaggle.

### Sumber M1

Kang, J. dan Kim, K. (2026). *Deployment-Oriented Interpretable Fraud Detection
via Hybrid Explainable Boosting Machines with Concept–Raw Fusion on the
IEEE-CIS Benchmark*. Applied Sciences, 16(12), 5809.
[Artikel jurnal](https://doi.org/10.3390/app16125809) ·
[Lampiran resmi berisi model dan kode](https://mdpi-res.com/d_attachment/applsci/applsci-16-05809/article_deploy/applsci-16-05809-s001.zip).

Model diambil dari `main_out_of_time/seed_42/teacher_model.json` dalam ZIP.
Yang digunakan adalah XGBoost teacher, bukan hybrid EBM yang menjadi kontribusi
utama artikel. Identitas model dan checksum terdapat dalam registry.

### Sumber M2

[FraudLens pada revisi 74162be78808aa2dd3a0def581d2a28c626929c4](https://github.com/praju120056/FraudLens/tree/74162be78808aa2dd3a0def581d2a28c626929c4).
[Berkas model asli](https://github.com/praju120056/FraudLens/blob/74162be78808aa2dd3a0def581d2a28c626929c4/backend/model/model.pkl).

Belum ada jurnal yang terverifikasi untuk M2. Bundle PKL sumber dipulihkan
menjadi JSON native XGBoost tanpa pelatihan ulang; preprocessing disimpan
terpisah. Probabilitas pada 88.581 transaksi test sumber identik secara bitwise
sebelum dan sesudah perpindahan runtime XGBoost 3.2.0 ke 3.1.3.
Lisensi eksplisit FraudLens belum ditemukan. Bobot model dan preprocessing
sumber tidak diunggah ulang dalam repositori ini.

## Pilot pasangan model sekarang

Sesi asli: `Pilot_Integrasi`, 19 Oktober 2026. Pilot memakai 128 transaksi per
replikasi, dua pasangan A/B per model-batch dan satu warm-up per skenario.
Terdapat 32 pengamatan dan 16 pasangan. Latency berikut adalah waktu seluruh
128 transaksi; overhead adalah rerata selisih B − A per transaksi.

| Model | Batch (transaksi) | Latency A (s) | Latency B (s) | Overhead (ms/transaksi) |
| --- | ---: | ---: | ---: | ---: |
| M1 | 1 | 0.2187 | 5.7319 | 43.0724 |
| M1 | 10 | 0.0571 | 2.8140 | 21.5386 |
| M1 | 50 | 0.0137 | 2.6558 | 20.6416 |
| M1 | 100 | 0.0135 | 2.7557 | 21.4235 |
| M2 | 1 | 0.0592 | 0.5772 | 4.0470 |
| M2 | 10 | 0.0093 | 0.2577 | 1.9407 |
| M2 | 50 | 0.0031 | 0.2300 | 1.7726 |
| M2 | 100 | 0.0022 | 0.2522 | 1.9533 |

Angka tabel dibulatkan menjadi empat desimal. [Unduh ringkasan untuk Excel](results/pilot/Ringkasan%20Pilot.csv).
CSV ringkasan memakai UTF-8, pemisah titik koma dan desimal koma.

- [CSV pengamatan A/B](results/pilot/benchmark_raw.csv).
- [CSV pasangan A/B](results/pilot/benchmark_pairs.csv).
- [Manifest sesi dan identitas kode saat pilot](results/pilot/run_manifest.json).
- [Manifest M1](results/pilot/M1_manifest.json) dan [M2](results/pilot/M2_manifest.json):
  struktur model serta lingkungan pengukuran.

Pilot belum menjadi data utama atau bukti kestabilan. Benchmark formal
17 Oktober menggunakan dua seed model jurnal lama dan tidak menjadi hasil
pasangan M1 jurnal–M2 FraudLens. Kode dalam repositori ini dapat lebih baru
daripada kode saat pilot; identitas kode pilot tetap mengikuti manifest sesi.

## Kode

| Berkas | Fungsi |
| --- | --- |
| `models.json` | Registry sumber, checksum, fitur dan struktur model |
| `protocol.json` | Rancangan workload, batch, replikasi dan batas pengukuran |
| `scripts/prepare_ieee.py`, `scripts/prepare_shared.py` | Pembentukan fitur di luar timer |
| `scripts/model_registry.py`, `scripts/ieee_data.py` | Pemuatan dan pemeriksaan model serta data siap |
| `scripts/validate_functional.py` | Pemeriksaan prediksi A/B dan aditivitas SHAP |
| `scripts/benchmark.py` | Pengukuran berpasangan A/B |
| `scripts/analyze_results.py`, `scripts/quality_checks.py` | Analisis serta pemeriksaan kelengkapan dan diagnostik |
| `scripts/export_readable.py` | Ekspor ringkasan dari hasil tersimpan |
| `run-docker.ps1` | Pelaksanaan melalui Docker dan penyimpanan bukti konfigurasi |

## Menjalankan pengukuran

Repo memuat kode dan hasil pilot, sedangkan dataset, bobot dan resource
preprocessing disimpan terpisah. Clone repo saja belum cukup untuk melakukan
pengukuran. Pada workspace penelitian lengkap, letakkan:

```text
Benchmark TreeSHAP/
  models/M1/teacher_model.json
  models/M2/fraudlens_native.json
  resources/                     # preprocessing, ID test dan template beku
Dataset + Model/
  Dataset IEEE-CIS/
    Uji Bersama Dua Model/        # matriks M1/M2, label, ID dan manifest
```

Seluruh artefak harus cocok dengan checksum registry. Resource M2 memerlukan
bundle sumber dan hasil pemulihan yang sudah divalidasi; tidak cukup hanya
mengunduh JSON model M1. Folder `resources` dan `models` tersedia di workspace
lokal penelitian, tetapi tidak dilacak Git.

Dari folder kode, dengan Docker Desktop aktif:

```powershell
docker build -f Dockerfile.runtime -t fraud-benchmark:primary-20261016 .
.\run-docker.ps1 -DataPath "D:\Data\Uji Bersama Dua Model" -RunKind pilot
```

Pilot bawaan perintah menggunakan 1.000 transaksi, berbeda dari pilot integrasi
128 transaksi yang disertakan. Untuk sesi formal yang terpisah:

```powershell
.\run-docker.ps1 -DataPath "D:\Data\Uji Bersama Dua Model" -RunKind formal
```

Protokol formal: 1.000 transaksi, batch 1/10/50/100, 30 pasangan per
model-batch, 15 urutan AB dan 15 BA, serta lima warm-up per skenario.
Kuota container dua CPU dan 4 GiB tanpa swap; dua thread inferensi.
Alokasi WSL/host harus lebih besar daripada kuota container. Data utama
memerlukan sesi pengukuran baru, bukan penggabungan sesi dari model berbeda.

## Batas pengukuran

Pembacaan CSV, pembentukan fitur, sampling, pemuatan model, inisialisasi
explainer dan penyimpanan hasil berada di luar timer. TreeSHAP menggunakan
`tree_path_dependent` pada raw margin. RSS A/B berbagi allocator, sehingga
bukan ukuran tambahan memori TreeSHAP yang terisolasi. Coverage mengukur
konsentrasi kontribusi fitur, bukan kualitas penjelasan menurut pengguna.

M1 mempertahankan fallback median seluruh data dari preprocessing sumber,
sehingga klaim prediksi temporal harus dibatasi. M2 memakai split acak.
Perbedaan overhead antarmodel tidak dapat dikaitkan hanya dengan jumlah pohon.
Data historis ini tidak memvalidasi kemampuan deteksi seluruh modus fraud terkini.
