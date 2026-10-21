"""Rebuild common held-out features using frozen preprocessing; never fit a model."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from model_registry import registry,sha256,ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--journal-prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reg=registry();source=reg['source_prepared_dataset']
    if a.output.exists():raise FileExistsError('Use a new output directory')
    for name,h in source['raw_csv_sha256'].items():
        if sha256(a.raw/name)!=h:raise ValueError('Raw source checksum mismatch: '+name)
    for name in ['X_test.npy','y_test.npy','test_ids.npy']:
        if sha256(a.journal_prepared/name)!=source['prepared_files'][name]:raise ValueError('Journal pack checksum mismatch')
    oldids=np.load(a.journal_prepared/'test_ids.npy',allow_pickle=False)
    publicids=np.load(ROOT/'resources/fraudlens_test_ids.npy',allow_pickle=False)
    ids=np.intersect1d(oldids,publicids)
    if len(ids)!=13361:raise ValueError('Unexpected test intersection')
    pos=pd.Index(oldids).get_indexer(ids)
    X1=np.asarray(np.load(a.journal_prepared/'X_test.npy',allow_pickle=False,mmap_mode='r')[pos],dtype='<f4')
    y=np.load(a.journal_prepared/'y_test.npy',allow_pickle=False)[pos]
    meta=json.loads((ROOT/'resources/fraudlens_preprocessing.json').read_text())
    pd.options.future.infer_string=False
    df=pd.read_csv(a.raw/'train_transaction.csv',usecols=['TransactionID','isFraud']+meta['feature_names']).set_index('TransactionID').loc[ids]
    if not np.array_equal(df.isFraud.to_numpy(),y):raise ValueError('Labels differ')
    parts=[]
    for col in meta['numeric_features']:
        parts.append(pd.to_numeric(df[col],errors='coerce').fillna(meta['medians'][col]).to_numpy(dtype=np.float32))
    for col in meta['categorical_features']:
        mapping=meta['category_maps'][col]
        parts.append(df[col].fillna('__MISSING__').astype(str).map(mapping).fillna(mapping['__UNKNOWN__']).to_numpy(dtype=np.float32))
    X2=np.column_stack(parts).astype('<f4',copy=False)
    a.output.mkdir(parents=True)
    for name,arr in [('M1_X_test.npy',X1),('M2_X_test.npy',X2),('y_test.npy',y),('test_ids.npy',ids.astype('<i8'))]:
        np.save(a.output/name,arr,allow_pickle=False)
    (a.output/'M2_preprocessing.json').write_bytes((ROOT/'resources/fraudlens_preprocessing.json').read_bytes())
    checks={name:sha256(a.output/name)==expected for name,expected in reg['dataset']['prepared_files'].items()}
    if not all(checks.values()):raise ValueError('Reconstructed files differ: '+str(checks))
    template=ROOT/'resources/shared_manifest_template.json'
    if sha256(template)!=reg['dataset']['prepared_manifest_sha256']:raise ValueError('Shared manifest template mismatch')
    (a.output/'prepared_manifest.json').write_bytes(template.read_bytes())
    print(json.dumps({'all_files_identical':True,'rows':len(ids),'fraud_rows':int(y.sum()),'checks':checks}))

if __name__=='__main__':main()
