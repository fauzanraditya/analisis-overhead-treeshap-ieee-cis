"""Offline IEEE-CIS feature reconstruction; never trains or overwrites a model.
Reviewed functions adapted from the verified feasibility tool, with original
feature arithmetic preserved. Requires the pinned pandas 3 runtime.
"""
from __future__ import annotations

import argparse

import ast

import gc

import hashlib

import json

import platform

import time

from datetime import datetime, timezone

from pathlib import Path

from types import SimpleNamespace

import numpy as np

import pandas as pd

import psutil

import shap

import sklearn

import xgboost as xgb

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

SOURCE_SHA = 'a5bc3fae1607dfa0d12fa72151ed1501e705516e837ce80ed8f13244e9d9c492'

ALLOWED = {'safe_div', 'optimize_memory', 'load_ieee_data', 'build_entity_key', 'chronological_split', 'add_safe_keys', 'add_time_buckets', 'add_history_stats', 'add_missingness_concepts', 'add_novelty_reuse_concepts', 'add_causal_concepts', 'frequency_encode', 'select_base_numeric_features', 'build_feature_matrices'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, payload):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)

def source_functions(source, shallow_copy=True):
    assert digest(source) == SOURCE_SHA, 'Official preprocessing source changed'
    tree = ast.parse(source.read_text(encoding='utf8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ALLOWED]
    assert {n.name for n in nodes} == ALLOWED
    if shallow_copy:
        assert int(pd.__version__.split('.')[0]) >= 3, 'Shallow copies require mandatory pandas 3 copy-on-write'

        class CopyOnWrite(ast.NodeTransformer):

            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'copy' and (not node.args) and (not node.keywords):
                    node.keywords = [ast.keyword(arg='deep', value=ast.Constant(value=False))]
                return node
        nodes = [CopyOnWrite().visit(n) for n in nodes]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    selected = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    scope = {'pd': pd, 'np': np, 'log': log}
    exec(compile(selected, str(source), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in ALLOWED})

def downcast_in_place(frame):
    """Same per-column rules as source optimize_memory, without frame.copy()."""
    for name in frame:
        s = frame[name]
        if pd.api.types.is_integer_dtype(s):
            frame[name] = pd.to_numeric(s, downcast='integer')
        elif pd.api.types.is_float_dtype(s):
            frame[name] = pd.to_numeric(s, downcast='float')
        elif s.dtype == 'object':
            if s.nunique(dropna=True) <= max(100, int(len(frame) * 0.25)):
                frame[name] = s.astype('category')
    return frame

def bounded_load(cfg):
    """Move loss-controlled source downcasting before merge to bound peak RAM.

    The left join is one-to-one, with sorting by unchanged integer keys. A final
    source-rule downcast handles identity integer columns promoted by missing
    matches. No filtering, sampling, rounding to float16 or feature rule changes.
    Reconstructed feature groups and reported metrics are checked downstream.
    """
    log('Read transaction CSV, then downcast its columns in place')
    transactions = pd.read_csv(cfg.train_transaction_path)
    assert transactions.TransactionID.is_unique
    downcast_in_place(transactions)
    gc.collect()
    log('Read identity CSV, downcast, then one-to-one left join')
    identity = pd.read_csv(cfg.train_identity_path)
    assert identity.TransactionID.is_unique
    downcast_in_place(identity)
    merged = transactions.merge(identity, on='TransactionID', how='left', validate='one_to_one')
    del transactions, identity
    gc.collect()
    merged.sort_values(['TransactionDT', 'TransactionID'], inplace=True, ignore_index=True)
    downcast_in_place(merged)
    gc.collect()
    return merged

def prepare(args, fn, out):
    cfg_dict = json.loads((args.source / 'config.json').read_text(encoding='utf8'))
    cfg_dict.update(train_transaction_path=str(args.data / 'train_transaction.csv'), train_identity_path=str(args.data / 'train_identity.csv'))
    cfg = SimpleNamespace(**cfg_dict)
    input_hashes = {name: digest(args.data / name) for name in ['train_transaction.csv', 'train_identity.csv']}
    log('Memory-bounded loading; source feature formulas preserved with pandas 3 copy-on-write')
    df = bounded_load(cfg)
    assert df.shape == (590540, 434), f'Unexpected source table shape {df.shape}'
    assert df.TransactionID.is_unique and set(df.isFraud.unique()) == {0, 1}
    assert int(df.isFraud.sum()) == 20663
    data_audit = {'rows': len(df), 'merged_columns': len(df.columns), 'fraud_rows': int(df.isFraud.sum()), 'unique_transaction_ids': True, 'identity_matched_rows': int(pd.read_csv(args.data / 'train_identity.csv', usecols=['TransactionID']).shape[0]), 'input_sha256': input_hashes, 'loader': 'Downcast with source per-column rules before merge; one-to-one left join and integer-key sorting; repeated in-place downcast after merge.', 'identity_note': 'Schema and counts are consistent; these checks do not establish byte identity with the inaccessible official download.'}
    df['entity_key'] = fn.build_entity_key(df, cfg.entity_cols)
    log(f'Merged data RSS {psutil.Process().memory_info().rss / 2 ** 30:.2f} GiB')
    amount_median = float(df.TransactionAmt.median())
    training_boundary = df.TransactionDT.quantile(cfg.train_ratio)
    train_amount_median = float(df.loc[df.TransactionDT <= training_boundary, 'TransactionAmt'].median())
    log('Build source history features; keep global-median behavior for reproduction')
    df, concepts = fn.add_causal_concepts(df)
    log(f'History features ready; RSS {psutil.Process().memory_info().rss / 2 ** 30:.2f} GiB')
    train, valid, test = fn.chronological_split(df, cfg)
    del df
    gc.collect()
    assert [len(train), len(valid), len(test)] == [413378, 88581, 88581]
    test_ids = test.TransactionID.to_numpy(copy=True)
    y_valid = valid.isFraud.to_numpy(copy=True)
    y_test = test.isFraud.to_numpy(copy=True)
    expected = json.loads((args.source / 'feature_groups.json').read_text(encoding='utf8'))
    selection_audit = {}
    select_original = fn.select_base_numeric_features

    def select_frozen(train_fe, exclude_cols, max_features, max_missing_ratio):
        rerun = select_original(train_fe, exclude_cols, max_features, max_missing_ratio)
        frozen = expected['numeric_base_cols']
        assert all((c in train_fe for c in frozen))
        selection_audit.update(rerun_numeric_base_cols=rerun, frozen_numeric_base_cols=frozen, rerun_equals_frozen=rerun == frozen, rerun_only=sorted(set(rerun) - set(frozen)), frozen_only=sorted(set(frozen) - set(rerun)), policy='Use published frozen feature list, not a new feature selection fit.')
        medians = train_fe[expected['final_cols']].apply(pd.to_numeric, errors='coerce').median(numeric_only=True)
        selection_audit['training_imputation_medians'] = {k: float(v) if np.isfinite(v) else None for k, v in medians.items()}
        return list(frozen)
    fn.build_feature_matrices.__globals__['select_base_numeric_features'] = select_frozen
    log('Build source frequency encoding and imputation, using published frozen feature list')
    X_train, X_valid, X_test, groups = fn.build_feature_matrices(train, valid, test, concepts, cfg)
    write_json(out / 'feature_selection_audit.json', selection_audit)
    assert groups == expected, 'Reconstructed feature selection differs from source'
    names = json.loads((Path(__file__).resolve().parents[1] / 'models/M1/teacher_model.json').read_text(encoding='utf8'))['learner']['feature_names']
    assert list(X_test.columns) == names and len(names) == 154
    assert np.isfinite(X_test.to_numpy()).all() and np.isfinite(X_valid.to_numpy()).all()
    state = {'features': names, 'source_global_amount_median': amount_median, 'training_only_amount_median': train_amount_median, 'training_imputation_medians': selection_audit['training_imputation_medians'], 'feature_selection_policy': 'Published frozen feature list is authoritative. A rerun under current library versions is recorded only as a diagnostic.', 'frequency_maps_training_only': {c: {str(k): int(v) for k, v in train[c].astype('string').fillna('__NA__').value_counts(dropna=False).items()} for c in cfg.cat_freq_cols if c in train}, 'scope': 'Offline reproduction of source feature construction. These frequency maps alone are not a complete online history preprocessor. No full raw-transaction A/B benchmark is implemented here.'}
    del X_train, train, valid, test
    gc.collect()
    np.save(out / 'X_valid.npy', X_valid.to_numpy(dtype=np.float32), allow_pickle=False)
    np.save(out / 'X_test.npy', X_test.to_numpy(dtype=np.float32), allow_pickle=False)
    np.save(out / 'y_valid.npy', y_valid, allow_pickle=False)
    np.save(out / 'y_test.npy', y_test, allow_pickle=False)
    np.save(out / 'test_ids.npy', test_ids, allow_pickle=False)
    write_json(out / 'feature_state.json', state)
    write_json(out / 'data_audit.json', data_audit)
    log('Prepared 154 features; no model was trained')
    return (X_valid, y_valid, X_test, y_test, test_ids)

def main():
    parser=argparse.ArgumentParser(description='Rebuild source-compatible prepared IEEE-CIS features outside benchmark timing')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output folder; the active pack must not be overwritten')
    args.output.mkdir(parents=True)
    args.source=Path(__file__).resolve().parents[1]/'resources'
    from model_registry import registry,sha256
    spec=registry().get('source_prepared_dataset',registry()['dataset'])
    for name,h in spec['raw_csv_sha256'].items():
        if sha256(args.data/name)!=h:raise ValueError('Raw dataset checksum mismatch: '+name)
    fn=source_functions(args.source/'run_ieee_cis_distilled_additive_v7_2_revision.py')
    prepare(args,fn,args.output)
    result={name:{'expected':h,'actual':sha256(args.output/name)} for name,h in spec['prepared_files'].items()}
    core=['X_valid.npy','X_test.npy','y_valid.npy','y_test.npy','test_ids.npy']
    passed=all(result[name]['actual']==result[name]['expected'] for name in core)
    template=args.source/'prepared_manifest_template.json'
    if sha256(template)!=spec['prepared_manifest_sha256']:
        raise ValueError('Frozen manifest template checksum mismatch')
    manifest=json.loads(template.read_text(encoding='utf8'))
    manifest['files']={name:item['actual'] for name,item in result.items()}
    # Preserve the original bytes only when every referenced file matches.
    # Otherwise publish an honest manifest for the new pack; the frozen registry
    # will reject it until a separate provenance review explicitly registers it.
    if all(item['expected']==item['actual'] for item in result.values()):
        (args.output/'prepared_manifest.json').write_bytes(template.read_bytes())
    else:
        write_json(args.output/'prepared_manifest.json',manifest)
    registry_compatible=sha256(args.output/'prepared_manifest.json')==spec['prepared_manifest_sha256']
    write_json(args.output/'reconstruction_check.json',{'core_arrays_match':passed,'frozen_registry_compatible':registry_compatible,'prepared_manifest_sha256':sha256(args.output/'prepared_manifest.json'),'files':result,'note':'Diagnostic JSON may differ. Every manifest hash describes the actual new file. A differing manifest requires review and explicit registration; the active registry is never changed automatically.'})
    if not passed:raise ValueError('Rebuilt core arrays differ; investigate before use')
    print('All five core arrays match the active frozen pack; no model training occurred.')
    print('Frozen registry compatible:',registry_compatible)
    if not registry_compatible:
        print('The new pack has a different manifest. Review diagnostic metadata before explicitly registering it for a new session.')

if __name__=='__main__':main()
