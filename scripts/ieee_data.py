"""Shared held-out TransactionIDs with a frozen feature matrix per model."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from model_registry import registry,sha256,DATA_SHA256

def load_prepared(data_dir,model_id='M1'):
    data_dir=Path(data_dir);manifest_path=data_dir/'prepared_manifest.json'
    spec=registry()['dataset'];features=registry()['models'][model_id]['input_features']
    if sha256(manifest_path)!=DATA_SHA256:raise ValueError('Prepared manifest checksum mismatch')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest['features'][model_id]!=features or manifest['dataset']!='IEEE-CIS':raise ValueError('Wrong model-specific dataset')
    if manifest['partition']!='intersection_of_published_held_out_test_ids':raise ValueError('Expected common held-out transactions')
    for name,expected in spec['prepared_files'].items():
        if sha256(data_dir/name)!=expected:
            raise ValueError('Prepared file checksum mismatch: '+name)
    X=np.load(data_dir/f'{model_id}_X_test.npy',allow_pickle=False,mmap_mode='r')
    y=np.load(data_dir/'y_test.npy',allow_pickle=False)
    ids=np.load(data_dir/'test_ids.npy',allow_pickle=False)
    if X.shape!=(spec['rows'],len(features)) or X.dtype!=np.dtype('float32') or not np.isfinite(X).all():raise ValueError('Invalid prepared matrix')
    if y.shape!=(spec['rows'],) or ids.shape!=y.shape or int(y.sum())!=spec['fraud_rows']:raise ValueError('Invalid common test partition')
    if len(np.unique(ids))!=len(ids) or not np.isin(y,[0,1]).all():raise ValueError('Invalid IDs or labels')
    return X,y,ids,manifest

def sample_workload(data_dir,rows,seed,model_id):
    X,y,ids,manifest=load_prepared(data_dir,model_id)
    if rows>len(X):raise ValueError('Workload exceeds source test partition')
    index=np.random.RandomState(seed).choice(len(X),size=rows,replace=False)
    values=np.ascontiguousarray(X[index],dtype='<f4')
    # Common identity covers IDs, labels, and a manifest hashing both feature matrices.
    fingerprint=hashlib.sha256(ids[index].astype('<i8').tobytes()+y[index].astype('i1').tobytes()+DATA_SHA256.encode('ascii')).hexdigest()
    return pd.DataFrame(values,columns=manifest['features'][model_id]),fingerprint,{
        'test_indices':index.tolist(),'transaction_ids':ids[index].tolist(),
        'fraud_rows':int(y[index].sum()),'rows':rows,'seed':seed,
        'sampling':'without replacement from common published held-out IDs; no class balancing',
        'prepared_manifest_sha256':DATA_SHA256,'measurement_boundary':manifest['measurement_boundary']}
