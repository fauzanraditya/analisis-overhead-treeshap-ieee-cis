"""Paired analysis gated to complete formal runs; pilots are descriptive only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from quality_checks import validate_measurements,validate_formal_evidence,stability
from export_readable import export_report


def holm(pvalues):
    values=np.asarray(pvalues,dtype=float)
    order=np.argsort(values)
    adjusted=np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values))))
    result=np.empty(len(values)); result[order]=np.minimum(adjusted,1)
    return result


def qq_svg(d,path,title):
    theoretical,observed=stats.probplot(d,dist="norm",fit=False)
    x=np.asarray(theoretical); y=np.asarray(observed)
    lo,hi=float(y.min()),float(y.max())
    if lo==hi: lo-=0.5; hi+=0.5
    xx=50+450*(x-x.min())/(x.max()-x.min()); yy=330-270*(y-lo)/(hi-lo)
    circles=''.join(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="3" fill="#164c80"/>' for a,b in zip(xx,yy))
    fit=stats.linregress(x,y)
    reference=330-270*((fit.intercept+fit.slope*x)-lo)/(hi-lo)
    circles=f'<line x1="{xx[0]:.2f}" y1="{reference[0]:.2f}" x2="{xx[-1]:.2f}" y2="{reference[-1]:.2f}" stroke="#888"/>'+circles
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="560" height="390"><rect width="100%" height="100%" fill="white"/><text x="30" y="25" font-size="16">{title}</text><path d="M50 50V330H520" fill="none" stroke="black"/>{circles}<text x="150" y="370">Theoretical normal quantiles</text><text x="55" y="48">Ordered B-A: [{lo:.6g}, {hi:.6g}]</text></svg>',encoding="utf8")


def analyze(run_dir,output_dir=None):
    manifest=json.loads((run_dir/"run_manifest.json").read_text())
    if manifest["status"]!="complete" or manifest["schema_version"] not in [3,4,5] or manifest.get("research_dataset")!="IEEE-CIS":
        raise ValueError("Require a completed two-model run with manifest")
    raw=pd.read_csv(run_dir/"benchmark_raw.csv"); pairs=pd.read_csv(run_dir/"benchmark_pairs.csv")
    args=manifest["arguments"]; keys=["model_id","batch_size","replication"]
    if raw.duplicated(keys+["scenario"]).any() or pairs.duplicated(keys).any():
        raise ValueError("Duplicate measurements")
    if set(raw.model_id)!={"M1","M2"} or set(pairs.model_id)!={"M1","M2"}:
        raise ValueError("Both model IDs are required")
    expected=pd.MultiIndex.from_product([["M1","M2"],args["batch_sizes"],range(1,args["replications"]+1)],names=keys)
    if set(map(tuple,pairs[keys].to_numpy()))!=set(expected):
        raise ValueError("Missing or unexpected pairs")
    if len(raw)!=2*len(pairs) or not pairs.predictions_match.eq(True).all():
        raise ValueError("Invalid A/B pairs")
    if (raw.workload_sha256.nunique()!=1 or pairs.workload_sha256.nunique()!=1
        or raw.workload_sha256.iloc[0]!=manifest["workload_sha256"]
        or pairs.workload_sha256.iloc[0]!=manifest["workload_sha256"]):
        raise ValueError("Different workloads")
    a=raw[raw.scenario=="A"].set_index(keys); b=raw[raw.scenario=="B"].set_index(keys); p=pairs.set_index(keys)
    if set(a.index)!=set(p.index) or set(b.index)!=set(p.index):
        raise ValueError("Invalid raw scenario keys")
    for metric,acol,bcol in [("latency_s","latency_A_s","latency_B_s"),("cpu_utilization_pct","cpu_A_pct","cpu_B_pct"),("peak_rss_mib","peak_rss_A_mib","peak_rss_B_mib")]:
        if not (np.allclose(a.loc[p.index,metric],p[acol]) and np.allclose(b.loc[p.index,metric],p[bcol])):
            raise ValueError("Raw and paired values differ")
    validate_measurements(manifest,raw,pairs)
    metrics=["latency_s","average_batch_time_s","throughput_txn_s","cpu_utilization_pct","peak_rss_mib","top5_shap_coverage_ratio"]
    if not np.isfinite(raw[metrics[:-1]].to_numpy()).all():
        raise ValueError("Non-finite measurements")
    analysis_dir=Path(output_dir) if output_dir else run_dir/"analysis"; analysis_dir.mkdir(parents=True,exist_ok=False)
    quality=stability(raw,pairs,analysis_dir)
    if "cpu_time_s" in raw:
        raw.groupby(["model_id","batch_size","scenario"])["cpu_time_s"].agg(["mean","median","std","min","max"]).to_csv(analysis_dir/"cpu_time_descriptive.csv")
    raw.groupby(["model_id","batch_size","scenario"])[metrics].agg(["mean","median","std","min","max"]).to_csv(analysis_dir/"descriptive.csv")
    pairs.groupby(["model_id","batch_size"])[["overhead_ratio_pct","normalized_overhead_ms_per_transaction","latency_difference_B_minus_A_s"]].agg(["mean","median","std","min","max"]).to_csv(analysis_dir/"overhead_descriptive.csv")
    frontier=raw.groupby(["model_id","scenario","batch_size"])[["average_batch_time_s","throughput_txn_s"]].mean().reset_index()
    frontier["pareto_nondominated"]=False
    for _,g in frontier.groupby(["model_id","scenario"]):
        for i,r in g.iterrows():
            dom=(g.average_batch_time_s<=r.average_batch_time_s)&(g.throughput_txn_s>=r.throughput_txn_s)&((g.average_batch_time_s<r.average_batch_time_s)|(g.throughput_txn_s>r.throughput_txn_s))
            frontier.loc[i,"pareto_nondominated"]=not dom.any()
    frontier.to_csv(analysis_dir/"pareto.csv",index=False)
    if manifest["run_kind"]!="formal":
        (analysis_dir/"analysis_status.json").write_text(json.dumps({"status":"pilot_descriptive_only","inferential_tests":0,"note":"Small or native pilot is not a formal benchmark."},indent=2))
        export_report(run_dir, analysis_dir)
        print("Pilot: descriptive only; no hypothesis tests. Readable CSV/HTML exported."); return
    if (args["rows"],args["replications"],args["warmups"],args["seed"],args["batch_sizes"])!=(1000,30,5,42,[1,10,50,100]):
        raise ValueError("Incomplete formal design")
    for m in ["M1","M2"]:
        state=json.loads((run_dir/f"{m}_manifest.json").read_text()); env=state["environment"]
        cpu=(env["cpu_max"] or "").split()
        if (state["status"]!="complete" or not env["container_marker"] or len(cpu)!=2 or cpu[0]=="max"
            or float(cpu[0])/float(cpu[1])!=2 or env["memory_max"]!=str(4*1024**3) or env["memory_swap_max"]!="0"):
            raise ValueError("Formal container limits not verified")
    validate_formal_evidence(run_dir,manifest)
    tests=[]
    for (model,batch),g in pairs.groupby(["model_id","batch_size"]):
        for metric,aa,bb in [("latency","latency_A_s","latency_B_s"),("cpu","cpu_A_pct","cpu_B_pct"),("rss","peak_rss_A_mib","peak_rss_B_mib")]:
            d=(g[bb]-g[aa]).to_numpy(); sd=float(np.std(d,ddof=1)); mean=float(d.mean())
            shapiro=float(stats.shapiro(d).pvalue) if sd>0 else None
            qq_svg(d,analysis_dir/f"qq_{model}_{batch}_{metric}.svg",f"{model}, batch {batch}, {metric}")
            if np.all(d==0): method,pval,effect="all_zero",1.0,0.0
            elif sd==0:
                method,pval="wilcoxon",float(stats.wilcoxon(d,zero_method="wilcox",alternative="two-sided",method="auto").pvalue)
                effect=float(np.sign(mean))
            elif shapiro>=0.05:
                method,pval,effect="paired_t",float(stats.ttest_1samp(d,0).pvalue),mean/sd
            else:
                method,pval="wilcoxon",float(stats.wilcoxon(d,zero_method="wilcox",alternative="two-sided",method="auto").pvalue)
                nz=d[d!=0]; ranks=stats.rankdata(np.abs(nz))
                effect=float((ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum())
            rng=np.random.default_rng(42)
            means=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(axis=1); low,high=np.percentile(means,[2.5,97.5])
            tests.append(dict(model_id=model,batch_size=int(batch),metric=metric,n=len(d),mean_difference_B_minus_A=mean,
                shapiro_p=shapiro,test=method,p_value=pval,effect_size=effect,
                effect_definition="Cohen dz" if method=="paired_t" else ("reporting convention: all differences zero" if method=="all_zero" else "signed rank-biserial (zero differences excluded)"),
                paired_mean_ci95_low=float(low),paired_mean_ci95_high=float(high),ci_method="percentile bootstrap, 5000 resamples, seed42"))
    if len(tests)!=24: raise ValueError("Expected global family of 24 tests")
    for t,adjusted in zip(tests,holm([t["p_value"] for t in tests])):
        t.update(p_holm=float(adjusted),reject_h0=bool(adjusted<0.05))
    pd.DataFrame(tests).to_csv(analysis_dir/"paired_tests_holm24.csv",index=False)
    (analysis_dir/"analysis_status.json").write_text(json.dumps({"status":"synthetic_software_test_only" if manifest.get("test_fixture") else "formal_analysis_complete_review_required","quality_review_required":quality["review_required"],"inferential_tests":24,"alpha":0.05,"caveats":["Inspect Q-Q plots and outliers; Shapiro screening does not prove normality.","Wilcoxon location interpretation assumes approximately symmetric differences.","Bootstrap intervals are pointwise, not simultaneous family-wise intervals.","A/B share allocator state; sampled RSS is not isolated extra memory.","Sequential model blocks can confound exploratory cross-model differences with time drift."]},indent=2))
    print("Synthetic software test only: 24 analysis branches checked." if manifest.get("test_fixture") else "Formal analysis complete: 24 tests, global Holm adjustment.")
    export_report(run_dir, analysis_dir)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",required=True); parser.add_argument("--output-dir")
    args=parser.parse_args(); analyze(Path(args.run_dir),args.output_dir)
