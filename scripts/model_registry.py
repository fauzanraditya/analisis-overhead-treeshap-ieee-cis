"""Frozen XGBoost cases with model-specific, offline prepared features."""
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def registry():
    return json.loads((ROOT/'models.json').read_text(encoding='utf8'))

DATA_SHA256=registry()['dataset']['prepared_manifest_sha256']

def structure(classifier):
    trees=[json.loads(t) for t in classifier.get_booster().get_dump(dump_format='json')]
    def walk(node,depth=0):
        if 'leaf' in node:return 1,1,depth
        children=[walk(c,depth+1) for c in node['children']]
        return 1+sum(c[0] for c in children),sum(c[1] for c in children),max(c[2] for c in children)
    counts=[walk(t) for t in trees]
    return dict(trees=len(trees),nodes=sum(c[0] for c in counts),leaves=sum(c[1] for c in counts),
                max_depth=max(c[2] for c in counts),mean_depth=float(np.mean([c[2] for c in counts])))

class PreparedInput:
    def __init__(self,features):
        self.features=list(features)

    def transform(self,frame):
        if list(frame.columns)!=self.features:raise ValueError('Prepared feature order changed')
        return frame.to_numpy(dtype=np.float32,copy=False)

def load_model(model_id,threads=2):
    spec=registry()['models'][model_id];path=ROOT/spec['artifact'];features=spec['input_features']
    if sha256(path)!=spec['sha256']:raise ValueError('Model checksum mismatch: '+model_id)
    classifier=xgb.XGBClassifier();classifier.load_model(path)
    classifier.set_params(device='cpu',n_jobs=threads)
    booster=classifier.get_booster()
    if booster.num_features()!=len(features):raise ValueError('Native feature count mismatch')
    if booster.feature_names is not None and booster.feature_names!=features:
        raise ValueError('Native feature order mismatch')
    actual=structure(classifier)
    if any(actual[k]!=spec['actual_structure'][k] for k in ['trees','nodes','leaves','max_depth']):
        raise ValueError('Model structure mismatch')
    explainer=shap.TreeExplainer(classifier,feature_perturbation='tree_path_dependent',model_output='raw')
    return PreparedInput(features),classifier,explainer,features,actual

def validate(preprocessor,classifier,explainer,frame):
    X=preprocessor.transform(frame);p=classifier.predict_proba(X)[:,1]
    named=classifier.get_booster().predict(xgb.DMatrix(X,feature_names=preprocessor.features))
    if not np.allclose(p,named,rtol=1e-6,atol=1e-7):raise ValueError('Named input prediction mismatch')
    phi=np.asarray(explainer.shap_values(X,check_additivity=True))
    margin=classifier.predict(X,output_margin=True)
    reconstructed=phi.sum(axis=1)+float(np.asarray(explainer.expected_value).item())
    if phi.shape!=(len(X),len(preprocessor.features)) or not np.isfinite(phi).all():raise ValueError('Invalid SHAP values')
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid probabilities')
    if not np.allclose(margin,reconstructed,rtol=1e-4,atol=2e-4):raise ValueError('SHAP additivity failed')
    return {'rows':len(X),'shap_shape':list(phi.shape),'max_additivity_error':float(np.max(np.abs(margin-reconstructed)))}

def runtime_versions():
    return {'python':platform.python_version(),'platform':platform.platform(),
        **{p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','shap','psutil','joblib']}}
