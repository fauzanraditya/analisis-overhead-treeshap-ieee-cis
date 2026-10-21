import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from ieee_data import load_prepared
from model_registry import DATA_SHA256,load_model,validate,runtime_versions
from benchmark import run_workload

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    X,y,ids,_=load_prepared(a.data);rng=np.random.default_rng(42)
    ix=np.concatenate([rng.choice(np.flatnonzero(y==c),64,replace=False) for c in [0,1]]);rng.shuffle(ix)
    report={'purpose':'functional validation only; separate from unbalanced benchmark workload',
            'prepared_manifest_sha256':DATA_SHA256,'runtime':runtime_versions(),'probe_ids':ids[ix].tolist(),'models':{}}
    for mid in ['M1','M2']:
        X,yy,other_ids,manifest=load_prepared(a.data,mid)
        if not np.array_equal(yy,y) or not np.array_equal(other_ids,ids):raise ValueError('Different test IDs or labels')
        frame=pd.DataFrame(np.asarray(X[ix]),columns=manifest['features'][mid])
        pre,clf,exp,features,actual=load_model(mid);checks=validate(pre,clf,exp,frame)
        full=clf.predict_proba(pre.transform(frame))[:,1]
        for batch in [1,10,50,100]:
            pa,_=run_workload(frame,batch,'A',pre,clf,exp);pb,chunks=run_workload(frame,batch,'B',pre,clf,exp)
            if not np.array_equal(pa,pb) or not np.allclose(pa,full,rtol=1e-6,atol=1e-7):raise ValueError('A/B batch predictions differ')
            if np.concatenate(chunks).shape!=(128,len(features)):raise ValueError('Invalid batched SHAP shape')
        report['models'][mid]={**checks,'structure':actual,'batch_sizes_passed':[1,10,50,100],'predictions_match':True}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
