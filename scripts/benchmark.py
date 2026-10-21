import argparse
import gc
import hashlib
import os
import subprocess
import sys
import uuid
import shutil
from ieee_data import sample_workload
from datetime import datetime, timezone
from model_registry import DATA_SHA256, registry, load_model, validate, runtime_versions, sha256
import json
import math
import random
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT.parent / "Hasil Benchmark"


def append_csv_rows(rows, path):
    """Append completed results to CSV immediately."""
    if not rows:
        return

    pd.DataFrame(rows).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def run_workload(X, batch_size, scenario, preprocessor, classifier, explainer):
    """Process the same ordered workload in batches."""
    probabilities = []
    shap_chunks = []

    for start in range(0, len(X), batch_size):
        batch = X.iloc[start : start + batch_size]

        transformed = preprocessor.transform(batch)
        batch_probabilities = classifier.predict_proba(transformed)[:, 1]
        probabilities.append(batch_probabilities)

        if scenario == "B":
            values = explainer.shap_values(
                transformed,
                check_additivity=False,
            )
            shap_chunks.append(np.asarray(values))

    all_probabilities = np.concatenate(probabilities)
    return all_probabilities, shap_chunks



def measure_scenario(
    X,
    batch_size,
    scenario,
    preprocessor,
    classifier,
    explainer,
    cpu_limit,
):
    """Measure workload latency, process CPU, and peak RSS."""
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop_event = threading.Event()
    ready_event = threading.Event()

    def sample_memory():
        try:
            peak_rss[0] = max(
                peak_rss[0],
                process.memory_info().rss,
            )
        except psutil.Error:
            pass

        ready_event.set()

        while not stop_event.wait(0.01):
            try:
                current_rss = process.memory_info().rss
                peak_rss[0] = max(peak_rss[0], current_rss)
            except psutil.Error:
                pass

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    ready_event.wait()

    # Start wall-clock timing before taking the CPU snapshot.
    start_time = time.perf_counter()
    cpu_before = time.process_time_ns()

    probabilities, shap_chunks = run_workload(
        X,
        batch_size,
        scenario,
        preprocessor,
        classifier,
        explainer,
    )

    cpu_after = time.process_time_ns()
    end_time = time.perf_counter()

    try:
        peak_rss[0] = max(
            peak_rss[0],
            process.memory_info().rss,
        )
    except psutil.Error:
        pass

    stop_event.set()
    monitor.join()

    latency = end_time - start_time

    # Process-wide user + system CPU time, with nanosecond API precision.
    cpu_seconds = (cpu_after - cpu_before) / 1e9

    cpu_utilization = (
        cpu_seconds / latency / cpu_limit * 100
        if latency > 0
        else float("nan")
    )

    return {
        "latency_s": latency,
        "cpu_time_s": cpu_seconds,
        "cpu_utilization_pct": cpu_utilization,
        "peak_rss_mib": peak_rss[0] / (1024 ** 2),
        "probabilities": probabilities,
        "shap_chunks": shap_chunks,
    }

def top5_coverage(shap_chunks):
    """
    Mean per-transaction share of absolute SHAP attribution
    represented by the five largest absolute feature values.
    """
    if not shap_chunks:
        return float("nan")

    values = np.concatenate(shap_chunks, axis=0)
    absolute_values = np.abs(values)

    total = absolute_values.sum(axis=1)
    top5 = np.sort(absolute_values, axis=1)[:, -5:].sum(axis=1)

    ratios = np.divide(
        top5,
        total,
        out=np.zeros_like(total, dtype=float),
        where=total > 0,
    )

    return float(np.mean(ratios))



def environment():
    info = {"versions": runtime_versions(), "pid": os.getpid(),
            "cpu_clock":{"name":"time.process_time_ns","resolution_s":time.get_clock_info('process_time').resolution,"implementation":time.get_clock_info('process_time').implementation},
            "thread_environment": {k: os.environ.get(k) for k in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_WAIT_POLICY")},
            "host_logical_cpus": psutil.cpu_count(), "host_memory_bytes": psutil.virtual_memory().total,
            "cpu_affinity": psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None}
    for key, path in {"cpu_max":"/sys/fs/cgroup/cpu.max", "memory_max":"/sys/fs/cgroup/memory.max",
                      "memory_swap_max":"/sys/fs/cgroup/memory.swap.max"}.items():
        info[key] = Path(path).read_text().strip() if Path(path).is_file() else None
    info["container_marker"] = Path("/.dockerenv").exists()
    return info


def worker(args, run_dir):
    env = environment()
    if args.run_kind == "formal":
        cpu = (env["cpu_max"] or "").split()
        if (not env["container_marker"] or len(cpu) != 2 or cpu[0] == "max"
                or float(cpu[0])/float(cpu[1]) != args.cpu_limit
                or env["memory_max"] != str(4 * 1024**3) or env["memory_swap_max"] != "0"):
            raise RuntimeError("Formal run requires verified cgroup v2: 2 CPUs, 4 GiB memory, zero swap. Use run-docker.ps1.")
    X, workload_hash, selection = sample_workload(args.data,args.rows,args.seed,args.worker)
    (run_dir/f"{args.worker}_workload.json").write_text(json.dumps(selection,indent=2),encoding='utf8')
    preprocessor, classifier, explainer, features, actual = load_model(args.worker, threads=2)
    calibration = validate(preprocessor, classifier, explainer, X.iloc[:min(128,len(X))])
    generic=json.loads(classifier.get_booster().save_config())["learner"]["generic_param"]
    runtime_booster={"nthread":int(generic["nthread"]),"device":generic["device"]}
    if runtime_booster!={"nthread":2,"device":"cpu"}:
        raise RuntimeError("Actual Booster runtime must use CPU with two threads")
    gc.collect()
    state = {"model_id":args.worker,"status":"running","prepared_manifest_sha256":DATA_SHA256,
             "workload_sha256":workload_hash, "structure":actual,"environment":env,
             "calibration":calibration,"runtime_n_jobs":2,"input_features":features,
             "feature_matrix_sha256":hashlib.sha256(np.asarray(X,dtype='<f4').tobytes()).hexdigest(),
             "runtime_booster":runtime_booster,"started_utc":datetime.now(timezone.utc).isoformat()}
    state_path = run_dir / f"{args.worker}_manifest.json"
    state_path.write_text(json.dumps(state,indent=2),encoding="utf8")
    raw_path, pairs_path = run_dir / f"{args.worker}_raw.csv", run_dir / f"{args.worker}_pairs.csv"
    rng = random.Random(args.seed)
    warmup_orders = {}
    warmup_measurements = []
    try:
        for batch_size in args.batch_sizes:
            number_of_batches = math.ceil(len(X)/batch_size)
            warmup_orders[str(batch_size)] = []
            state['progress']={'phase':'warmup','batch_size':batch_size}
            state_path.write_text(json.dumps(state,indent=2),encoding='utf8')
            for warmup_index in range(1,args.warmups+1):
                order = ["A","B"]; rng.shuffle(order)
                warmup_orders[str(batch_size)].append("".join(order))
                for scenario in order:
                    warmup_start=time.perf_counter()
                    run_workload(X,batch_size,scenario,preprocessor,classifier,explainer)
                    warmup_latency=time.perf_counter()-warmup_start
                    warmup_measurements.append({'batch_size':batch_size,'warmup_index':warmup_index,
                        'scenario':scenario,'latency_s':warmup_latency})
                    print(f"{args.worker}: warmup batch={batch_size}, index={warmup_index}/{args.warmups}, scenario={scenario}, seconds={warmup_latency:.3f}",flush=True)
            state['progress']={'phase':'measurement','batch_size':batch_size}
            state_path.write_text(json.dumps(state,indent=2),encoding='utf8')
            # Exact 15 AB / 15 BA for the planned 30 pairs. Odd pilot counts differ by at most one.
            orders = ["AB"]*((args.replications+1)//2) + ["BA"]*(args.replications//2)
            rng.shuffle(orders)
            for replication, order in enumerate(orders,1):
                gc.collect()
                measured = {}
                for scenario in order:
                    measured[scenario] = measure_scenario(X,batch_size,scenario,preprocessor,classifier,explainer,args.cpu_limit)
                a,b = measured["A"], measured["B"]
                if not np.allclose(a["probabilities"],b["probabilities"],rtol=1e-6,atol=1e-7):
                    raise RuntimeError("A/B predictions differ")
                common = {"model_id":args.worker,"run_kind":args.run_kind,"batch_size":batch_size,
                          "replication":replication,"scenario_order":order,"workload_sha256":workload_hash}
                raw_rows=[]
                for scenario in "AB":
                    r=measured[scenario]; latency=r["latency_s"]
                    raw_rows.append({**common,"scenario":scenario,"position_in_pair":order.index(scenario)+1,
                        "workload_rows":len(X),"number_of_batches":number_of_batches,
                        "latency_s":latency,"average_batch_time_s":latency/number_of_batches,
                        "throughput_txn_s":len(X)/latency,"cpu_time_s":r["cpu_time_s"],"cpu_utilization_pct":r["cpu_utilization_pct"],
                        "peak_rss_mib":r["peak_rss_mib"],
                        "top5_shap_coverage_ratio":top5_coverage(r["shap_chunks"]) if scenario=="B" else float("nan")})
                delta=b["latency_s"]-a["latency_s"]
                pair={**common,"latency_A_s":a["latency_s"],"latency_B_s":b["latency_s"],
                      "latency_difference_B_minus_A_s":delta,"overhead_ratio_pct":delta/a["latency_s"]*100,
                      "normalized_overhead_ms_per_transaction":delta/len(X)*1000,
                      "cpu_time_A_s":a["cpu_time_s"],"cpu_time_B_s":b["cpu_time_s"],
                      "cpu_A_pct":a["cpu_utilization_pct"],"cpu_B_pct":b["cpu_utilization_pct"],
                      "peak_rss_A_mib":a["peak_rss_mib"],"peak_rss_B_mib":b["peak_rss_mib"],
                      "predictions_match":True}
                append_csv_rows(raw_rows,raw_path); append_csv_rows([pair],pairs_path)
                print(f"{args.worker}: batch={batch_size}, pair={replication}/{args.replications}, order={order}",flush=True)
                # Discard retained outputs before the next pair, outside the timing interval.
                del measured, a, b, r
        state["status"]="complete"
    except Exception as exc:
        state.update(status="failed",error=repr(exc)); raise
    finally:
        state.update(warmup_orders=warmup_orders,warmup_measurements=warmup_measurements,
            cgroup_end={name:Path('/sys/fs/cgroup/'+name).read_text().strip() if Path('/sys/fs/cgroup/'+name).is_file() else None
                        for name in ['cpu.stat','memory.events','memory.peak']},
            finished_utc=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(state,indent=2),encoding="utf8")


def main():
    parser=argparse.ArgumentParser(description="Paired TreeSHAP benchmark on two frozen XGBoost artifacts")
    parser.add_argument("--data",required=True)
    parser.add_argument("--rows",type=int,default=1000)
    parser.add_argument("--batch-sizes",type=int,nargs="+",default=[1,10,50,100])
    parser.add_argument("--replications",type=int,default=30)
    parser.add_argument("--warmups",type=int,default=5)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--cpu-limit",type=float,default=2)
    parser.add_argument("--run-kind",choices=["formal","pilot"],default="formal")
    parser.add_argument("--output-dir")
    parser.add_argument("--worker",choices=["M1","M2"],help=argparse.SUPPRESS)
    args=parser.parse_args()
    if (args.rows<1 or args.replications<1 or args.warmups<0 or args.cpu_limit!=2
        or not args.batch_sizes or min(args.batch_sizes)<1 or len(set(args.batch_sizes))!=len(args.batch_sizes)):
        raise ValueError("Invalid settings: positive sizes, no duplicate batches, cpu-limit fixed at 2")
    if args.run_kind=="formal" and (args.rows,args.replications,args.warmups,args.seed,args.batch_sizes)!=(1000,30,5,42,[1,10,50,100]):
        raise ValueError("Formal protocol is fixed: 1000 rows, 30 pairs, 5 warmups, seed42, batches1/10/50/100")
    if args.worker:
        worker(args,Path(args.output_dir)); return
    run_dir=Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]+"_"+args.run_kind)
    run_dir.mkdir(parents=True,exist_ok=False)
    order=["M1","M2"]; random.Random(args.seed).shuffle(order)
    manifest={"schema_version":5,"research_dataset":"IEEE-CIS","status":"running","run_kind":args.run_kind,"arguments":vars(args),
              "models":registry()["models"],"model_execution_order":order,
              "dataset_source":registry()["dataset"],
              "code_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in
                 [ROOT/"Dockerfile",ROOT/"Dockerfile.runtime",ROOT/"protocol.json",ROOT/"requirements.txt",ROOT/"models.json",ROOT/"run-docker.ps1",*sorted((ROOT/"resources").glob("*")),
                  *sorted((ROOT/"scripts").glob("*.py"))] if p.name!="benchmark_backup.py"},
              "started_utc":datetime.now(timezone.utc).isoformat(),
              "measurement_boundary":registry()["measurement_boundary"],
              "measurement_note":"A fresh sequential child process per model; A/B reuse that process. CPU time is recorded separately from normalized utilization. Retained SHAP outputs and allocator state affect RSS. Native pilot is not a controlled Docker benchmark."}
    snapshot=run_dir/'execution_snapshot'
    snapshot.mkdir()
    for name in ['scripts','resources','models','models.json','protocol.json','requirements.txt','Dockerfile','Dockerfile.runtime','run-docker.ps1']:
        source=ROOT/name; target=snapshot/name
        if source.is_dir():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
        else:shutil.copy2(source,target)
    manifest_path=run_dir/"run_manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("OUTPUT_DIR:",run_dir,flush=True)
    try:
        env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",OMP_WAIT_POLICY="PASSIVE")
        for model_id in order:
            command=[sys.executable,str(Path(__file__).resolve()),"--data",str(Path(args.data).resolve()),
                     "--rows",str(args.rows),"--replications",str(args.replications),"--warmups",str(args.warmups),
                     "--seed",str(args.seed),"--cpu-limit",str(args.cpu_limit),"--run-kind",args.run_kind,
                     "--output-dir",str(run_dir),"--worker",model_id,"--batch-sizes",*[str(s) for s in args.batch_sizes]]
            subprocess.run(command,env=env,check=True)
        model_states=[json.loads((run_dir/f"{m}_manifest.json").read_text()) for m in order]
        if len({s["workload_sha256"] for s in model_states})!=1:
            raise RuntimeError("Model workloads differ")
        for kind in ("raw","pairs"):
            frames=[pd.read_csv(run_dir/f"{m}_{kind}.csv") for m in ["M1","M2"]]
            pd.concat(frames,ignore_index=True).to_csv(run_dir/f"benchmark_{kind}.csv",index=False)
        manifest.update(status="complete",raw_rows=2*2*len(args.batch_sizes)*args.replications,
                        pairs=2*len(args.batch_sizes)*args.replications,
                        workload_sha256=model_states[0]["workload_sha256"])
    except Exception as exc:
        manifest.update(status="failed",error=repr(exc)); raise
    finally:
        manifest["finished_utc"]=datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
