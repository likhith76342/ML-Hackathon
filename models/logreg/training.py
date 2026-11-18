"""Shared training routines for Zubed's logistic regression and random forest."""
import json,time
import joblib
import numpy as np
import pandas as pd
from pathsnap.config import CFG,repo_path
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import candidate_metrics,top1_accuracy,candidate_miss_rate
from pathsnap.quality import flagged_trip_ids, QUALITY_DATASET_VERSION

def run(name,family,make_model):
    base=repo_path(CFG['processed_dir']); out=repo_path(CFG['results_dir'])/name; out.mkdir(parents=True,exist_ok=True)
    train=pd.read_parquet(base/'train_candidates.parquet'); test=pd.read_parquet(base/'test_candidates.parquet'); refs=pd.read_parquet(base/'point_refs.parquet'); folds=pd.read_csv(base/'folds.csv')
    if set(train.trip_id.unique())!=set(folds.trip_id): raise ValueError('folds.csv must cover training trips exactly')
    excluded=flagged_trip_ids()
    if not set(excluded).issubset(set(folds.trip_id)): raise ValueError('Quality file contains a non-training trip')
    train=train[~train.trip_id.isin(excluded)].copy()
    if set(train.trip_id.unique()) != (set(folds.trip_id)-set(excluded)): raise ValueError('Quality filtering did not match folds')
    if CFG.get('dataset_version') != QUALITY_DATASET_VERSION:
        print('WARNING: shared config dataset_version is not 2; results use the v2 exclusion policy. Ask the config owner to update it.',flush=True)
    fold_map=dict(zip(folds.trip_id,folds.fold)); assignments=train.trip_id.map(fold_map)
    def fit(model,df):
        valid=df.arc_id.ge(0); model.fit(df.loc[valid,FEATURE_COLUMNS],df.loc[valid,'label'])
    def score(model,df):
        valid=df.arc_id.ge(0); probs=np.zeros(len(df)); start=time.perf_counter(); probs[valid]=model.predict_proba(df.loc[valid,FEATURE_COLUMNS])[:,1]; elapsed=time.perf_counter()-start
        selected=refs[refs.trip_id.isin(df.trip_id.unique())]
        return candidate_metrics(df.loc[valid,'label'],probs[valid]),top1_accuracy(df,probs,selected),candidate_miss_rate(df,selected),elapsed,len(selected)
    rows=[]
    for fold in sorted(folds.fold.unique()):
        model=make_model(); fit(model,train[assignments!=fold]); cand,top,miss,_,_=score(model,train[assignments==fold])
        rows.append({'fold':int(fold),'macro_f1':cand['macro_f1'],'top1':top,'log_loss':cand['log_loss'],'candidate_miss_rate':miss,'accuracy':cand['accuracy'],'precision':cand['precision'],'recall':cand['recall'],'confusion_matrix':json.dumps(cand['confusion_matrix'])}); print(name,'fold',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/'cv_folds.csv',index=False)
    model=make_model(); start=time.perf_counter(); fit(model,train); train_time=time.perf_counter()-start
    tr,train_top,_,_,_=score(model,train); tc,test_top,miss,elapsed,n=score(model,test)
    joblib.dump(model,out/'model.joblib')
    params={key:value for key,value in model.get_params(deep=True).items() if isinstance(value,(str,int,float,bool,type(None)))}
    params.update({'excluded_trips':excluded,'dataset_version':QUALITY_DATASET_VERSION})
    d={'model':name,'family':family,'config':params,'cv_macro_f1_mean':float(np.mean([r['macro_f1'] for r in rows])),'cv_macro_f1_std':float(np.std([r['macro_f1'] for r in rows])),'cv_top1_mean':float(np.mean([r['top1'] for r in rows])),'cv_top1_std':float(np.std([r['top1'] for r in rows])),'cv_log_loss_mean':float(np.mean([r['log_loss'] for r in rows])),'train_top1':train_top,'train_macro_f1':tr['macro_f1'],'test_accuracy':tc['accuracy'],'test_precision':tc['precision'],'test_recall':tc['recall'],'test_macro_f1':tc['macro_f1'],'test_log_loss':tc['log_loss'],'test_top1':test_top,'test_candidate_miss_rate':miss,'test_route_continuity_argmax':None,'test_confusion_matrix':tc['confusion_matrix'],'train_time_s':train_time,'inference_ms_per_point':1000*elapsed/n,'seed':CFG['seed']}
    (out/'metrics.json').write_text(json.dumps(d,indent=2),encoding='utf-8'); print(name,json.dumps(d),flush=True)

