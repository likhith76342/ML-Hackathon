"""Tune v2 LightGBM with Optuna on trip-grouped training folds only.

Run from repository root: python -m optimisation.tune_lightgbm
The study resumes from SQLite and writes trials.csv after each completed trial.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import GroupShuffleSplit
from models.lightgbm.train import make_pipeline, score, SEED, INNER_VAL_FRACTION, EARLY_STOPPING_ROUNDS
from pathsnap.config import CFG, repo_path
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.quality import flagged_trip_ids

MIN_TRIALS = 50

def digest_files(paths):
    out={}
    for path in paths:
        with path.open('rb') as f: out[str(path.name)]=hashlib.file_digest(f,'sha256').hexdigest()
    return out

def load_train():
    base=repo_path(CFG['processed_dir']); folds=pd.read_csv(base/'folds.csv'); excluded=flagged_trip_ids()
    train=pd.read_parquet(base/'train_candidates.parquet'); refs=pd.read_parquet(base/'point_refs.parquet')
    if CFG['dataset_version']!=2 or excluded!=[33,54]: raise ValueError('Unexpected v2 configuration or exclusions')
    if set(train.trip_id.unique())!=set(folds.trip_id): raise ValueError('Training trips do not match folds')
    train=train[~train.trip_id.isin(excluded)].reset_index(drop=True)
    refs=refs[refs.trip_id.isin(train.trip_id.unique())].reset_index(drop=True)
    if len(set(train.trip_id))!=78 or folds.fold.nunique()!=5: raise ValueError('Expected 78 retained trips and 5 folds')
    return train,refs,folds,excluded

def params_for(trial):
    return {'num_leaves':trial.suggest_int('num_leaves',15,127,log=True),
      'learning_rate':trial.suggest_float('learning_rate',0.02,0.2,log=True),
      'min_child_samples':trial.suggest_int('min_child_samples',10,100,log=True),
      'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
      'subsample':trial.suggest_float('subsample',0.6,1.0),
      'reg_alpha':trial.suggest_float('reg_alpha',1e-8,10.0,log=True),
      'reg_lambda':trial.suggest_float('reg_lambda',1e-8,10.0,log=True),
      'n_estimators':1000, 'n_jobs':4}

def fold_data(train,refs,folds):
    fold_lookup=dict(zip(folds.trip_id,folds.fold)); assignment=train.trip_id.map(fold_lookup)
    if assignment.isna().any(): raise ValueError('A training trip lacks a fold')
    for fold in sorted(folds.fold.unique()):
        outer=train[assignment!=fold]; validation=train[assignment==fold].reset_index(drop=True)
        real=outer[outer.arc_id>=0].reset_index(drop=True)
        group=GroupShuffleSplit(n_splits=1,test_size=INNER_VAL_FRACTION,random_state=SEED)
        fit_idx,inner_idx=next(group.split(real,groups=real.trip_id)); fit=real.iloc[fit_idx]; inner=real.iloc[inner_idx]
        assert not (set(fit.trip_id)&set(inner.trip_id)) and not (set(outer.trip_id)&set(validation.trip_id))
        yield int(fold),fit,inner,validation,refs[refs.trip_id.isin(validation.trip_id.unique())]

def run_target(target, out_dir):
    if target<1: raise ValueError('Target trials must be positive')
    out_dir.mkdir(parents=True,exist_ok=True); train,refs,folds,excluded=load_train()
    base=repo_path(CFG['processed_dir']); fingerprint=digest_files([base/'train_candidates.parquet',base/'point_refs.parquet',base/'folds.csv',base/'trip_quality.csv',Path(__file__),repo_path('models/lightgbm/train.py')])
    db=(out_dir/'study_v2.sqlite3').resolve(); sampler_path=out_dir/'sampler_v2.joblib'
    sampler=joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=SEED)
    study=optuna.create_study(study_name='pathsnap_lightgbm_v2',storage='sqlite:///'+db.as_posix(),direction='maximize',sampler=sampler,load_if_exists=True)
    if any(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials) and not sampler_path.exists(): raise ValueError('Sampler state is missing; use a fresh study directory to avoid duplicate suggestions')
    if study.user_attrs.get('fingerprint') not in (None,fingerprint): raise ValueError('Input/code changed since this study began; use a new study directory')
    study.set_user_attr('fingerprint',fingerprint); study.set_user_attr('excluded_trips',excluded); study.set_user_attr('dataset_version',2)
    splits=list(fold_data(train,refs,folds)); start=time.perf_counter()
    def objective(trial):
        params=params_for(trial)
        def evaluate_fold(split):
            fold,fit,inner,valid,ref=split
            pipe=make_pipeline(**params)
            pipe.fit(fit[FEATURE_COLUMNS],fit.label,model__eval_set=[(inner[FEATURE_COLUMNS],inner.label)],model__eval_metric='binary_logloss',model__callbacks=[early_stopping(EARLY_STOPPING_ROUNDS,verbose=False),log_evaluation(0)])
            result=score(pipe,valid,ref)
            return result['top1'],result['cand']['macro_f1'],result['cand']['log_loss'],int(pipe.named_steps['model'].best_iteration_)
        # Independent folds run in parallel; map preserves their original order.
        with ThreadPoolExecutor(max_workers=2) as executor:
            per_fold=list(executor.map(evaluate_fold,splits))
        scores,macros,losses,iterations=map(list,zip(*per_fold))
        trial.set_user_attr('fold_top1',scores); trial.set_user_attr('fold_macro_f1',macros); trial.set_user_attr('fold_log_loss',losses); trial.set_user_attr('best_iterations',iterations)
        trial.set_user_attr('cv_macro_f1_mean',float(np.mean(macros))); trial.set_user_attr('cv_log_loss_mean',float(np.mean(losses)))
        return float(np.mean(scores))
    def checkpoint(study,trial):
        study.trials_dataframe().to_csv(out_dir/'trials.csv',index=False)
        sampler_tmp=out_dir/'sampler_v2.tmp'; joblib.dump(study.sampler,sampler_tmp); sampler_tmp.replace(sampler_path)
        print(f'trial {trial.number}: CV top1={trial.value:.6f}; completed={len([x for x in study.trials if x.state==optuna.trial.TrialState.COMPLETE])}; elapsed_s={time.perf_counter()-start:.1f}',flush=True)
    complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    if complete<target: study.optimize(objective,n_trials=target-complete,callbacks=[checkpoint],gc_after_trial=True)
    study.trials_dataframe().to_csv(out_dir/'trials.csv',index=False)
    print('completed trials:',sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),flush=True)
    if sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)<MIN_TRIALS:
        print('Search is incomplete; not selecting or evaluating a final model.',flush=True); return
    best=study.best_trial; best_params={**best.params,'n_estimators':int(np.median(best.user_attrs['best_iterations'])),'n_jobs':4}
    real=train[train.arc_id>=0]; model=make_pipeline(**best_params); t0=time.perf_counter(); model.fit(real[FEATURE_COLUMNS],real.label); train_time=time.perf_counter()-t0
    # Student-test files are opened only after the search and final choice.
    test=pd.read_parquet(base/'test_candidates.parquet'); all_refs=pd.read_parquet(base/'point_refs.parquet'); test_refs=all_refs[all_refs.trip_id.isin(test.trip_id.unique())]; final=score(model,test,test_refs)
    artifacts=repo_path(CFG['artifacts_dir']); artifacts.mkdir(parents=True,exist_ok=True); joblib.dump(model,artifacts/'final_model.joblib')
    (artifacts/'feature_columns.json').write_text(json.dumps(FEATURE_COLUMNS,indent=2),encoding='utf-8')
    baseline=json.loads((repo_path(CFG['results_dir'])/'lightgbm'/'metrics.json').read_text(encoding='utf-8-sig'))
    summary={'model':'lightgbm','dataset_version':2,'excluded_trips':excluded,'n_completed_trials':sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),'best_trial':best.number,'best_cv_top1':best.value,'best_cv_macro_f1':best.user_attrs['cv_macro_f1_mean'],'best_cv_log_loss':best.user_attrs['cv_log_loss_mean'],'best_params':best_params,'test_top1':final['top1'],'test_macro_f1':final['cand']['macro_f1'],'test_log_loss':final['cand']['log_loss'],'test_candidate_miss_rate':final['miss'],'inference_ms_per_point':1000*final['predict_s']/final['n_points'],'refit_time_s':train_time,'baseline':{k:baseline[k] for k in ('cv_top1_mean','cv_macro_f1_mean','test_top1','test_macro_f1','test_log_loss','inference_ms_per_point')},'search_elapsed_s_this_invocation':time.perf_counter()-start,'search_trial_time_s_total':sum(t.duration.total_seconds() for t in study.trials if t.duration is not None),'fold_results':{'top1':best.user_attrs['fold_top1'],'macro_f1':best.user_attrs['fold_macro_f1'],'log_loss':best.user_attrs['fold_log_loss'],'best_iterations':best.user_attrs['best_iterations']}}
    (out_dir/'best_params.json').write_text(json.dumps(best_params,indent=2),encoding='utf-8'); (out_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    figdir=repo_path(CFG['figures_dir']); figdir.mkdir(parents=True,exist_ok=True)
    done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]; ys=np.array([t.value for t in done]); fig,ax=plt.subplots(figsize=(8,4)); ax.plot([t.number for t in done],ys,'.',label='trial'); ax.plot([t.number for t in done],np.maximum.accumulate(ys),label='best so far'); ax.set(xlabel='Trial',ylabel='Mean grouped-CV Top-1',title='LightGBM Bayesian optimisation'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'model_lightgbm_optuna_history.png',dpi=150); plt.close(fig)
    importance=optuna.importance.get_param_importances(study); fig,ax=plt.subplots(figsize=(8,4)); ax.barh(list(importance)[::-1],list(importance.values())[::-1]); ax.set(xlabel='Relative importance',title='LightGBM search parameter importance'); fig.tight_layout(); fig.savefig(figdir/'model_lightgbm_optuna_importance.png',dpi=150); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--trials',type=int,default=50,help='Target total completed trials, at least 50 for final output'); p.add_argument('--study-dir',default='optimisation'); args=p.parse_args(); run_target(args.trials,repo_path(args.study_dir))
if __name__=='__main__': main()
