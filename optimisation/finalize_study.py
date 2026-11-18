"""Finalize the interrupted v2 search with the user's approved >10-trial budget.

Run after stopping the study: python -m optimisation.finalize_study
The CV-selected trial is fixed before reading or evaluating student-test candidates.
"""
import json,time
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from models.lightgbm.train import make_pipeline,score
from optimisation.tune_lightgbm import digest_files,load_train
from pathsnap.config import CFG,repo_path
from pathsnap.features import FEATURE_COLUMNS

def main():
    out=repo_path('optimisation'); base=repo_path(CFG['processed_dir'])
    study=optuna.load_study(study_name='pathsnap_lightgbm_v2',storage='sqlite:///'+(out/'study_v2.sqlite3').resolve().as_posix())
    complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    if len(complete)<=10: raise ValueError('More than 10 completed trials are required for this user-approved run')
    if any(t.state==optuna.trial.TrialState.RUNNING for t in study.trials): raise ValueError('Stop the running search before finalizing')
    fingerprint=digest_files([base/'train_candidates.parquet',base/'point_refs.parquet',base/'folds.csv',base/'trip_quality.csv',repo_path('optimisation/tune_lightgbm.py'),repo_path('models/lightgbm/train.py')])
    if study.user_attrs.get('fingerprint') != fingerprint: raise ValueError('Training data or search code changed since tuning')
    train,refs,folds,excluded=load_train(); best=study.best_trial
    params={**best.params,'n_estimators':int(np.median(best.user_attrs['best_iterations'])),'n_jobs':4}
    real=train[train.arc_id>=0]; model=make_pipeline(**params)
    start=time.perf_counter(); model.fit(real[FEATURE_COLUMNS],real.label); fit_seconds=time.perf_counter()-start
    # Open the student-test candidates only after the winning CV configuration is fixed.
    test=pd.read_parquet(base/'test_candidates.parquet'); all_refs=pd.read_parquet(base/'point_refs.parquet'); test_refs=all_refs[all_refs.trip_id.isin(test.trip_id.unique())]; test_score=score(model,test,test_refs)
    artifacts=repo_path(CFG['artifacts_dir']); artifacts.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,artifacts/'final_model.joblib'); (artifacts/'feature_columns.json').write_text(json.dumps(FEATURE_COLUMNS,indent=2),encoding='utf-8')
    baseline=json.loads((repo_path(CFG['results_dir'])/'lightgbm'/'metrics.json').read_text(encoding='utf-8-sig'))
    total=sum(t.duration.total_seconds() for t in study.trials if t.duration is not None)
    report={'model':'lightgbm','dataset_version':2,'excluded_trips':excluded,'n_completed_trials':len(complete),'n_failed_trials':sum(t.state==optuna.trial.TrialState.FAIL for t in study.trials),'best_trial':best.number,'best_cv_top1':best.value,'best_cv_macro_f1':best.user_attrs['cv_macro_f1_mean'],'best_cv_log_loss':best.user_attrs['cv_log_loss_mean'],'best_params':params,'test_top1':test_score['top1'],'test_macro_f1':test_score['cand']['macro_f1'],'test_log_loss':test_score['cand']['log_loss'],'test_candidate_miss_rate':test_score['miss'],'inference_ms_per_point':1000*test_score['predict_s']/test_score['n_points'],'refit_time_s':fit_seconds,'search_trial_time_s_total':total,'fold_results':{'top1':best.user_attrs['fold_top1'],'macro_f1':best.user_attrs['fold_macro_f1'],'log_loss':best.user_attrs['fold_log_loss'],'best_iterations':best.user_attrs['best_iterations']},'baseline':{k:baseline[k] for k in ('cv_top1_mean','cv_macro_f1_mean','test_top1','test_macro_f1','test_log_loss','inference_ms_per_point')},'budget_note':'Search stopped at the user\'s request after more than 10 completed trials; earlier team brief asked for at least 50.'}
    study.trials_dataframe().to_csv(out/'trials.csv',index=False); (out/'best_params.json').write_text(json.dumps(params,indent=2),encoding='utf-8'); (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    figures=repo_path(CFG['figures_dir']); figures.mkdir(parents=True,exist_ok=True)
    y=np.array([t.value for t in complete]); x=[t.number for t in complete]; fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True); ax.plot(x,y,'.',label='Completed trial'); ax.plot(x,np.maximum.accumulate(y),label='Best so far'); ax.set(xlabel='Trial',ylabel='Mean grouped-CV Top-1',title='LightGBM Bayesian optimisation'); ax.legend(); fig.savefig(figures/'model_lightgbm_optuna_history.png',dpi=150); plt.close(fig)
    importance=optuna.importance.get_param_importances(study); fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True); ax.barh(list(importance)[::-1],list(importance.values())[::-1]); ax.set(xlabel='Relative importance',title='Search parameter importance'); fig.savefig(figures/'model_lightgbm_optuna_importance.png',dpi=150); plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
