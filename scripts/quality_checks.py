"""Integrity and descriptive stability checks, never automatic outlier removal."""
import json,math,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def require(condition,message):
    if not condition:raise ValueError(message)

def validate_measurements(manifest,raw,pairs):
    args=manifest['arguments'];keys=['model_id','batch_size','replication']
    require(set(raw.scenario)=={'A','B'},'Unexpected or missing scenario')
    require(set(raw.run_kind)=={manifest['run_kind']} and set(pairs.run_kind)=={manifest['run_kind']},'Run-kind mismatch')
    require(raw.workload_rows.eq(args['rows']).all(),'Wrong workload size in CSV')
    require(np.all(raw.number_of_batches==np.ceil(args['rows']/raw.batch_size)),'Wrong batch count')
    require((raw.latency_s>0).all() and (raw.peak_rss_mib>0).all() and (raw.cpu_utilization_pct>=0).all(),'Invalid physical metric')
    require(raw[raw.scenario=='B'].top5_shap_coverage_ratio.between(0,1).all(),'Invalid SHAP Coverage')
    require(raw[raw.scenario=='A'].top5_shap_coverage_ratio.isna().all(),'Coverage must be absent in A')
    require(set(pairs.scenario_order)<= {'AB','BA'},'Invalid scenario order')
    indexed=pairs.set_index(keys)
    for row in raw.itertuples():
        paired=indexed.loc[(row.model_id,row.batch_size,row.replication)]
        require(row.scenario_order==paired.scenario_order,'Raw/paired order mismatch')
        require(row.position_in_pair==row.scenario_order.index(row.scenario)+1,'Wrong position in pair')
    for _,group in pairs.groupby(['model_id','batch_size']):
        counts=group.scenario_order.value_counts()
        require(abs(counts.get('AB',0)-counts.get('BA',0))<=1,'Unbalanced scenario orders')
    require(np.allclose(raw.throughput_txn_s,args['rows']/raw.latency_s,rtol=1e-8,atol=1e-8),'Throughput formula mismatch')
    require(np.allclose(raw.average_batch_time_s,raw.latency_s/raw.number_of_batches,rtol=1e-8,atol=1e-10),'Average batch formula mismatch')
    d=pairs.latency_B_s-pairs.latency_A_s
    for observed,expected in [(pairs.latency_difference_B_minus_A_s,d),(pairs.overhead_ratio_pct,d/pairs.latency_A_s*100),(pairs.normalized_overhead_ms_per_transaction,d/args['rows']*1000)]:
        require(np.isfinite(observed).all() and np.allclose(observed,expected,rtol=1e-8,atol=1e-8),'Derived overhead formula mismatch')
    if 'cpu_time_s' in raw:
        require(np.isfinite(raw.cpu_time_s).all() and (raw.cpu_time_s>=0).all(),'Invalid CPU time')
        require(np.allclose(raw.cpu_utilization_pct,raw.cpu_time_s/raw.latency_s/2*100,rtol=1e-8,atol=1e-8),'CPU normalization mismatch')
        for scenario in 'AB':
            actual=raw[raw.scenario==scenario].set_index(keys).loc[indexed.index,'cpu_time_s']
            require(np.allclose(actual,indexed[f'cpu_time_{scenario}_s'],rtol=1e-8,atol=1e-8),'Raw/paired CPU time mismatch')
    elif manifest['schema_version']>=4:raise ValueError('Missing measured CPU time')

def validate_formal_evidence(run_dir,manifest):
    require(manifest['schema_version']>=4,'Formal analysis requires manifest schema 4 or later')
    args=manifest['arguments']
    snapshot=(run_dir/'execution_snapshot').resolve()
    for relative,expected in manifest['code_sha256'].items():
        file=(snapshot/relative).resolve()
        require(file.is_relative_to(snapshot) and file.is_file(),'Missing execution snapshot file')
        h=hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
        require(h.hexdigest()==expected,'Execution snapshot checksum mismatch: '+relative)
    for mid in ['M1','M2']:
        state=json.loads((run_dir/f'{mid}_manifest.json').read_text())
        require(state['model_id']==mid and state['workload_sha256']==manifest['workload_sha256'],'Worker identity/workload mismatch')
        require(state['prepared_manifest_sha256']==manifest['dataset_source']['prepared_manifest_sha256'],'Prepared data provenance mismatch')
        if manifest['schema_version']>=5:
            require(state['input_features']==manifest['models'][mid]['input_features'],'Worker feature order mismatch')
            require(len(state.get('feature_matrix_sha256',''))==64,'Missing model-specific workload hash')
        require(state['structure']==manifest['models'][mid]['actual_structure'] or all(state['structure'][k]==manifest['models'][mid]['actual_structure'][k] for k in ['trees','nodes','leaves','max_depth']),'Worker model structure mismatch')
        warm=pd.DataFrame(state['warmup_measurements'])
        require(len(warm)==2*len(args['batch_sizes'])*args['warmups'],'Incomplete warmups')
        require(not warm.duplicated(['batch_size','warmup_index','scenario']).any(),'Duplicate warmup')
        expected={(b,i,s) for b in args['batch_sizes'] for i in range(1,args['warmups']+1) for s in 'AB'}
        require(set(map(tuple,warm[['batch_size','warmup_index','scenario']].to_numpy()))==expected,'Missing warmup combinations')
        require(np.isfinite(warm.latency_s).all() and (warm.latency_s>0).all(),'Invalid warmup timing')
        events=state['cgroup_end'].get('memory.events') or ''
        counters=dict(line.split() for line in events.splitlines())
        require(counters.get('oom_kill')=='0' and counters.get('oom')=='0','OOM or missing memory-event evidence')
        require(state['runtime_n_jobs']==2 and state['runtime_booster']['nthread']==2 and state['runtime_booster']['device']=='cpu','Wrong inference runtime')
        env=state['environment']['thread_environment']
        require(all(env.get(k)=='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']) and env.get('OMP_WAIT_POLICY')=='PASSIVE','Thread environment mismatch')
        clock=state['environment'].get('cpu_clock',{})
        require(clock.get('name')=='time.process_time_ns' and clock.get('resolution_s',0)>0,'Missing high-resolution CPU clock evidence')
    require(len(manifest['model_execution_order'])==2 and set(manifest['model_execution_order'])=={'M1','M2'},'Wrong model execution order')
    selections=[json.loads((run_dir/f'{mid}_workload.json').read_text()) for mid in ['M1','M2']]
    require(selections[0]==selections[1] and len(set(selections[0]['transaction_ids']))==args['rows'],'Workload selection mismatch')

def relative_change(first,last):
    return float((last-first)/abs(first)*100) if first!=0 else None

def stability(raw,pairs,out):
    rows=[];outliers=[];series=[]
    for (mid,batch,scenario),g in raw.groupby(['model_id','batch_size','scenario']):
        g=g.sort_values('replication');v=g.latency_s.to_numpy();n=len(v)
        cv=float(np.std(v,ddof=1)/np.mean(v)*100) if n>1 else None
        early_late=relative_change(float(np.median(v[:10])),float(np.median(v[-10:]))) if n>=20 else None
        median=float(np.median(v));mad=float(np.median(np.abs(v-median)))
        z=.6744897501960817*(v-median)/mad if mad>0 else np.zeros(n)
        flags=np.abs(z)>3.5 if mad>0 else np.zeros(n,dtype=bool)
        rows.append(dict(model_id=mid,batch_size=int(batch),scenario=scenario,n=n,
            latency_cv_pct=cv,early10_to_late10_median_change_pct=early_late,
            mad_s=mad,mad_outliers=int(flags.sum()),latency_lag1_correlation=float(np.corrcoef(v[:-1],v[1:])[0,1]) if n>=4 and np.std(v[:-1])>0 and np.std(v[1:])>0 else None,
            review_flag=bool((cv is not None and cv>10) or (early_late is not None and abs(early_late)>10) or flags.any())))
        for rep,val,score,flag in zip(g.replication,v,z,flags):
            outliers.append(dict(model_id=mid,batch_size=int(batch),scenario=scenario,replication=int(rep),latency_s=float(val),modified_z_score=float(score) if mad>0 else None,flagged=bool(flag),excluded=False))
    for (mid,batch),g in pairs.groupby(['model_id','batch_size']):
        g=g.sort_values('replication');n=len(g)
        for metric,aa,bb in [('latency','latency_A_s','latency_B_s'),('cpu','cpu_A_pct','cpu_B_pct'),('rss','peak_rss_A_mib','peak_rss_B_mib')]:
            d=(g[bb]-g[aa]).to_numpy();ab=d[g.scenario_order=='AB'];ba=d[g.scenario_order=='BA']
            effect=relative_change(float(np.median(ab)),float(np.median(ba))) if len(ab)>=2 and len(ba)>=2 else None
            lag=float(np.corrcoef(d[:-1],d[1:])[0,1]) if n>=4 and np.std(d[:-1])>0 and np.std(d[1:])>0 else None
            rho=float(stats.spearmanr(g.replication,d).statistic) if n>=3 and np.ptp(d)>0 else None
            q1,q2,q3=np.percentile(d,[25,50,75]);bowley=float((q3+q1-2*q2)/(q3-q1)) if q3>q1 else None
            series.append(dict(model_id=mid,batch_size=int(batch),metric=metric,n=n,
                median_difference_AB=float(np.median(ab)) if len(ab) else None,median_difference_BA=float(np.median(ba)) if len(ba) else None,
                AB_to_BA_median_change_pct=effect,lag1_correlation=lag,replication_spearman_rho=rho,bowley_skewness=bowley,
                review_flag=bool((effect is not None and abs(effect)>10) or (lag is not None and abs(lag)>=.5) or (rho is not None and abs(rho)>=.5) or (bowley is not None and abs(bowley)>=.5))))
    pd.DataFrame(rows).to_csv(out/'stability_latency.csv',index=False)
    pd.DataFrame(outliers).to_csv(out/'outlier_review.csv',index=False)
    pd.DataFrame(series).to_csv(out/'paired_diagnostics.csv',index=False)
    summary={'review_required':any(r['review_flag'] for r in rows+series),
        'exclusion_count':0,'main_session_stability_assessable':bool(raw.groupby(['model_id','batch_size','scenario']).size().min()>=30),
        'thresholds':{'latency_cv_pct':10,'early_late_or_order_change_pct':10,'modified_z_score':3.5,'absolute_correlation_or_bowley':.5},
        'interpretation':'Descriptive review flags only, not universal validity criteria or additional hypothesis tests. No observations are removed. Small pilots cannot establish stability or independence.'}
    (out/'quality_status.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    return summary
