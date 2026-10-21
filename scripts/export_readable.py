"""Readable CSV/HTML exports from one completed run, outside measurement timers.

The run's own manifest supplies model identities. Existing raw data and analysis
are read only; no model, inference, statistical test or benchmark is run here.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}")
    return value


def mean(rows, key):
    return statistics.mean(number(r, key) for r in rows)


def show(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Ya" if value else "Tidak"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return html.escape(str(value))


def time_wib(value):
    if not value:
        return "Tidak tercatat"
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone(timedelta(hours=7))).strftime("%d-%m-%Y %H:%M:%S WIB")


def summarize(manifest, raw, pairs):
    if manifest.get("status") != "complete":
        raise ValueError("Only a completed run can be exported")
    args = manifest["arguments"]
    models = manifest["models"]
    if set(models) != {"M1", "M2"}:
        raise ValueError("Require identities of both models from this run")
    expected = {(m, int(b), i) for m in models for b in args["batch_sizes"]
                for i in range(1, args["replications"] + 1)}
    key = lambda r: (r["model_id"], int(r["batch_size"]), int(r["replication"]))
    if len(pairs) != len(expected) or {key(r) for r in pairs} != expected:
        raise ValueError("Incomplete or duplicate pairs")
    if len(raw) != 2 * len(expected) or {(key(r), r["scenario"]) for r in raw} != {
            (k, s) for k in expected for s in "AB"}:
        raise ValueError("Incomplete or duplicate raw observations")
    if any(r["workload_sha256"] != manifest["workload_sha256"] for r in raw + pairs):
        raise ValueError("Different workloads")
    indexed = {(key(r), r["scenario"]): r for r in raw}
    for p in pairs:
        if p["predictions_match"].lower() != "true":
            raise ValueError("A/B predictions do not match")
        a, b = [indexed[key(p), s] for s in "AB"]
        for metric, ca, cb in [("latency_s", "latency_A_s", "latency_B_s"),
                               ("cpu_utilization_pct", "cpu_A_pct", "cpu_B_pct"),
                               ("peak_rss_mib", "peak_rss_A_mib", "peak_rss_B_mib")]:
            for row, col in [(a, ca), (b, cb)]:
                if not math.isclose(number(row, metric), number(p, col), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("Raw and paired metrics differ")
        delta = number(b, "latency_s") - number(a, "latency_s")
        rows = int(a["workload_rows"])
        for field, value in [("latency_difference_B_minus_A_s", delta),
                             ("normalized_overhead_ms_per_transaction", delta / rows * 1000),
                             ("overhead_ratio_pct", delta / number(a, "latency_s") * 100)]:
            if not math.isclose(number(p, field), value, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError("Invalid paired overhead")
    summary = []
    for m in sorted(models):
        for batch in sorted(args["batch_sizes"]):
            g = [r for r in pairs if r["model_id"] == m and int(r["batch_size"]) == batch]
            a, b = [[r for r in raw if r["model_id"] == m and int(r["batch_size"]) == batch
                     and r["scenario"] == s] for s in "AB"]
            d = dict(model=m, label=models[m]["label"], batch=batch, n=len(g),
                     rows=args["rows"], latency_a=mean(a, "latency_s"), latency_b=mean(b, "latency_s"),
                     overhead=mean(g, "normalized_overhead_ms_per_transaction"),
                     overhead_pct=mean(g, "overhead_ratio_pct"),
                     throughput_a=mean(a, "throughput_txn_s"), throughput_b=mean(b, "throughput_txn_s"),
                     cpu_a=mean(a, "cpu_utilization_pct"), cpu_b=mean(b, "cpu_utilization_pct"),
                     rss_a=mean(a, "peak_rss_mib"), rss_b=mean(b, "peak_rss_mib"),
                     coverage=mean(b, "top5_shap_coverage_ratio"))
            for scenario, group in [("a", a), ("b", b)]:
                values = [number(r, "latency_s") for r in group]
                d[f"cv_{scenario}"] = statistics.stdev(values) / statistics.mean(values) * 100 if len(values) > 1 else None
            summary.append(d)
    return summary


def chart(rows):
    """Small separate chart per model; equal category spacing, zero-based y axis."""
    w, h, left, top, width, height = 600, 285, 64, 25, 510, 205
    peak = max(max(r["overhead"] for r in rows), 0.001)
    floor = min(min(r["overhead"] for r in rows), 0)
    span = peak - floor
    y = lambda value: top + height * (1 - (value - floor) / span)
    xs = [left + width * i / max(len(rows) - 1, 1) for i in range(len(rows))]
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Overhead per transaksi menurut ukuran batch">']
    for i in range(5):
        v = floor + span * i / 4
        out.append(f'<path d="M{left} {y(v):.2f}H{left+width}" stroke="#dce4e9"/><text x="{left-8}" y="{y(v)+4:.2f}" text-anchor="end">{show(v,2)}</text>')
    out.append('<polyline fill="none" stroke="#176c79" stroke-width="3" points="' +
               ' '.join(f'{x:.2f},{y(r["overhead"]):.2f}' for x,r in zip(xs,rows)) + '"/>')
    for x, r in zip(xs, rows):
        out.append(f'<circle cx="{x:.2f}" cy="{y(r["overhead"]):.2f}" r="4" fill="#176c79"><title>Batch {r["batch"]}: {show(r["overhead"],4)} ms/transaksi</title></circle><text x="{x}" y="252" text-anchor="middle">{r["batch"]}</text>')
    out.append('<text x="65" y="16">ms/transaksi</text><text x="320" y="279" text-anchor="middle">Ukuran batch (transaksi)</text></svg>')
    return ''.join(out)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def export_report(run_dir, analysis_dir=None, output_dir=None):
    run_dir = Path(run_dir).resolve()
    analysis_dir = Path(analysis_dir) if analysis_dir else run_dir / "analysis"
    output_dir = Path(output_dir) if output_dir else analysis_dir / "Laporan Terbaca"
    manifest = read_json(run_dir / "run_manifest.json")
    raw, pairs = records(run_dir / "benchmark_raw.csv"), records(run_dir / "benchmark_pairs.csv")
    summary = summarize(manifest, raw, pairs)
    analysis = read_json(analysis_dir / "analysis_status.json") if (analysis_dir / "analysis_status.json").exists() else {}
    quality = read_json(analysis_dir / "quality_status.json") if (analysis_dir / "quality_status.json").exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "Ringkasan Benchmark.csv"
    html_path = output_dir / "Laporan Benchmark.html"
    model_meta = manifest["models"]
    journal_seeds = (all(v.get("source_doi") == "10.3390/app16125809" for v in model_meta.values())
                     and len({v.get("seed") for v in model_meta.values()}) > 1)
    pilot = manifest["run_kind"] != "formal"
    fixture = manifest.get("test_fixture", False)
    if fixture:
        status, note = "Uji perangkat lunak sintetis", "Bukan pengukuran pada transaksi penelitian dan bukan data utama."
    elif pilot:
        status, note = "Uji pendahuluan (pilot)", "Hasil ini hanya memeriksa fungsi dan memperkirakan durasi. Belum menjadi data utama atau bukti kestabilan benchmark."
    else:
        status, note = "Sesi formal selesai", "Penyelesaian sesi tidak otomatis menyatakan seluruh pengukuran stabil. Diagnostik dan keterbatasannya tetap perlu ditinjau."
    if journal_seeds:
        note += " Sesi ini memakai dua seed model jurnal, bukan pasangan jurnal–FraudLens yang digunakan pada rancangan sekarang."
    cols = [("model", "Model"), ("label", "Identitas model pada sesi ini"), ("batch", "Batch (transaksi)"),
            ("n", "Jumlah pasangan"), ("rows", "Transaksi per replikasi"),
            ("latency_a", "Rerata latency A (detik)"), ("latency_b", "Rerata latency B (detik)"),
            ("overhead", "Rerata overhead (ms/transaksi)"), ("overhead_pct", "Rerata overhead pasangan (%)"),
            ("throughput_a", "Rerata throughput A (transaksi/detik)"), ("throughput_b", "Rerata throughput B (transaksi/detik)"),
            ("cpu_a", "Rerata CPU A (%)"), ("cpu_b", "Rerata CPU B (%)"),
            ("rss_a", "Rerata puncak RSS A (MiB)"), ("rss_b", "Rerata puncak RSS B (MiB)"),
            ("coverage", "Rerata Top-5 Coverage B (rasio 0–1)"),
            ("cv_a", "CV latency A (%)"), ("cv_b", "CV latency B (%)")]
    # An explicit Excel separator directive supports direct opening on Windows.
    # It is an export convenience; canonical comma-delimited raw CSVs are unchanged.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("sep=;\n")
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Status sesi", "Catatan sesi", "ID sesi"] + [title for _, title in cols])
        for row in summary:
            values = []
            for key, _ in cols:
                value = row[key]
                if value is None: values.append("")
                elif isinstance(value, float): values.append(format(value, ".15g").replace(".", ","))
                else: values.append(value)
            writer.writerow([status, note, run_dir.name] + values)
    pieces = [f'<header><p class="eyebrow">IEEE-CIS · {html.escape(run_dir.name)}</p><h1>Hasil benchmark TreeSHAP</h1><p>Perbandingan inferensi XGBoost tanpa dan dengan TreeSHAP</p></header>',
              f'<section class="notice"><h2>{status}</h2><p>{html.escape(note)}</p></section>',
              '<section><h2>Sesi pengukuran</h2><dl>' + ''.join(f'<dt>{k}</dt><dd>{html.escape(str(v))}</dd>' for k,v in [
                  ("Mulai", time_wib(manifest.get("started_utc"))), ("Selesai", time_wib(manifest.get("finished_utc"))),
                  ("Workload", f'{manifest["arguments"]["rows"]} transaksi per replikasi'),
                  ("Data", f'{len(raw)} pengamatan · {len(pairs)} pasangan'),
                  ("Skenario A", "Penyesuaian format masukan + inferensi XGBoost"),
                  ("Skenario B", "Tahapan A + perhitungan TreeSHAP")]) + '</dl></section>',
              '<section><h2>Model yang benar-benar digunakan pada sesi ini</h2>']
    model_rows = []
    for key,v in sorted(model_meta.items()):
        state_path = run_dir / f"{key}_manifest.json"
        state = read_json(state_path) if state_path.exists() else {}
        structure = state.get("structure", {})
        features = state.get("input_features", v.get("input_features"))
        count = len(features) if isinstance(features,list) else manifest.get("dataset_source",{}).get("feature_count", "Tidak tercatat")
        model_rows.append([key, html.escape(v["label"]), str(structure.get("trees", "—")), str(structure.get("max_depth", "—")), str(count)])
    pieces += [table(["ID", "Identitas pada manifest sesi", "Pohon", "Kedalaman maksimum", "Fitur"], model_rows), '</section>',
               '<section><h2>Tambahan waktu TreeSHAP</h2><p>Nilai merupakan rerata selisih waktu B − A per transaksi pada setiap pasangan. Grafik tiap model memakai skala vertikalnya sendiri.</p><div class="charts">']
    for model in sorted(model_meta):
        group = [r for r in summary if r["model"] == model]
        pieces.append(f'<div><h3>{model} · {html.escape(model_meta[model]["label"])}</h3>{chart(group)}</div>')
    pieces += ['</div>', table(["Model", "Batch", "Pasangan", "Latency A (s)", "Latency B (s)", "Overhead (ms/transaksi)", "Overhead (%)"],
                  [[r["model"], str(r["batch"]), str(r["n"]), show(r["latency_a"]), show(r["latency_b"]), show(r["overhead"]), show(r["overhead_pct"],2)] for r in summary]),
               '<p class="small">Persentase merupakan rerata persentase tiap pasangan, sehingga dapat berbeda dari persentase yang dihitung dari dua rerata latency.</p></section>',
               '<section><h2>Throughput dan sumber daya</h2>',
               table(["Model", "Batch", "Throughput A (txn/s)", "Throughput B (txn/s)", "CPU A (%)", "CPU B (%)", "Puncak RSS A (MiB)", "Puncak RSS B (MiB)", "Coverage B"],
                   [[r["model"],str(r["batch"])] + [show(r[k]) for k in ["throughput_a","throughput_b","cpu_a","cpu_b","rss_a","rss_b","coverage"]] for r in summary]),
               '<p class="small">Semua nilai dalam tabel adalah rerata replikasi. CPU dinormalisasi terhadap kuota dua CPU. RSS adalah rerata puncak proses yang teramati; penggunaan ulang memori A/B membatasi penafsirannya sebagai tambahan memori TreeSHAP. Coverage merupakan rasio konsentrasi kontribusi fitur, bukan skor pemahaman pengguna.</p></section>',
               '<section><h2>Kestabilan dan hasil analisis</h2>',
               table(["Model", "Batch", "CV latency A (%)", "CV latency B (%)"],[[r["model"],str(r["batch"]),show(r["cv_a"],2),show(r["cv_b"],2)] for r in summary])]
    if pilot:
        pieces.append('<p>Pilot berukuran kecil belum cukup untuk menilai kestabilan atau menguji hipotesis. Tidak ada p-value yang dibuat pada laporan pilot ini.</p>')
    elif not analysis:
        pieces.append('<p>Analisis statistik belum tersedia. Laporan ini hanya menyajikan ringkasan pengukuran.</p>')
    else:
        review = quality.get("review_required", analysis.get("quality_review_required"))
        pieces.append(f'<p>Jumlah uji dari analisis tersimpan: {analysis.get("inferential_tests", "Tidak tercatat")}. Penanda diagnostik yang memerlukan peninjauan: {show(review)}. Pencilan tidak dihapus oleh laporan ini.</p>')
        tests_file = analysis_dir / "paired_tests_holm24.csv"
        if tests_file.exists():
            tests = records(tests_file)
            pieces.append('<details><summary>Lihat hasil uji A/B dan interval kepercayaan</summary>' + table(
                ["Model","Batch","Metrik","Metode","Rerata B − A","CI 95% bawah","CI 95% atas","p Holm"],
                [[html.escape(r[k]) for k in ["model_id","batch_size","metric","test"]] +
                 [show(float(r[k]),6) for k in ["mean_difference_B_minus_A","paired_mean_ci95_low","paired_mean_ci95_high","p_holm"]] for r in tests]) +
                '<p class="small">Satuan selisih dan CI: latency dalam detik, CPU dalam poin persentase, RSS dalam MiB. CI bootstrap bersifat per perbandingan; koreksi Holm berlaku pada p-value. Metode dan nilai ditampilkan dari hasil analisis tersimpan, tanpa pengujian ulang.</p></details>')
    hashes = {name: hashlib.sha256((run_dir/name).read_bytes()).hexdigest()
              for name in ["run_manifest.json", "benchmark_raw.csv", "benchmark_pairs.csv"]}
    pieces += ['</section><section><h2>Membaca dan menggunakan hasil</h2><ul><li>Latency mengukur seluruh workload dari fitur siap inferensi. Pembacaan dataset, pembentukan fitur, dan pemuatan model berada di luar timer.</li><li>Perbandingan utama adalah A/B di dalam model dan ukuran batch yang sama. Selisih antarmodel juga dipengaruhi fitur dan pelatihan masing-masing.</li><li>CSV ringkasan dapat dibuka di Excel. Jika kolom tidak terpisah, impor sebagai UTF-8 dengan pemisah titik koma dan desimal koma.</li></ul>',
               '<a class="download" href="Ringkasan%20Benchmark.csv" download>Unduh ringkasan CSV</a></section>',
               '<details><summary>Sumber data dan checksum</summary><p>Ringkasan dihitung dari berkas sesi ini; identitas model berasal dari manifest sesi, bukan registry model terbaru.</p><pre>' + html.escape(json.dumps(hashes,indent=2)) + '</pre></details>']
    css = '''*{box-sizing:border-box}body{font:16px/1.6 system-ui,Arial,sans-serif;color:#233343;background:#f3f6f8;margin:0}main{max-width:1260px;margin:auto;padding:36px 26px 70px}header{margin-bottom:30px}h1{font-size:34px;line-height:1.2;margin:8px 0}h2{font-size:23px;line-height:1.3;margin:0 0 16px}h3{font-size:17px}p{margin:10px 0}section{background:white;border:1px solid #dce4e9;border-radius:9px;padding:25px;margin:22px 0}.notice{background:#fff9eb;border-left:5px solid #b37c16}.eyebrow,.small{font-size:14px;color:#536373}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th{background:#eaf1f4;text-align:left;vertical-align:bottom;white-space:normal}td,th{padding:11px 13px;border-bottom:1px solid #dce4e9}td{font-variant-numeric:tabular-nums;white-space:nowrap}td:first-child{font-weight:600}tr:nth-child(even){background:#f8fafb}dl{display:grid;grid-template-columns:160px 1fr;gap:10px 20px}dt{font-weight:600}dd{margin:0}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:25px}svg{width:100%;font:13px system-ui}pre{white-space:pre-wrap;word-break:break-all;font-size:12px}summary{cursor:pointer;font-weight:600}.download{display:inline-block;background:#176c79;color:white;padding:10px 18px;border-radius:5px;text-decoration:none}@media(max-width:750px){main{padding:20px 12px}.charts{grid-template-columns:1fr}section{padding:18px}h1{font-size:28px}dl{grid-template-columns:1fr;gap:4px}dd{margin-bottom:10px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid}.scroll{overflow:visible}table{font-size:10px}td,th{padding:5px;white-space:normal}.download{display:none}}'''
    html_path.write_text('<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hasil benchmark TreeSHAP</title><style>' + css + '</style><main>' + ''.join(pieces) + '</main></html>',encoding="utf-8")
    return {"html":str(html_path), "csv":str(csv_path), "summary_rows":len(summary), "raw_rows":len(raw), "pairs":len(pairs), "status":status}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--analysis-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(export_report(args.run_dir,args.analysis_dir,args.output_dir),ensure_ascii=False,indent=2))
