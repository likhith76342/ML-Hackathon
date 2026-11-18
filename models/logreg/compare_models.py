"""Create a comparison from existing metrics; missing models stay missing.

Run: python -m models.logreg.compare_models
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathsnap.config import CFG,repo_path

def main():
    root=repo_path(CFG['results_dir']); figures=repo_path(CFG['figures_dir']); figures.mkdir(parents=True,exist_ok=True)
    names=['logreg','random_forest','knn','mlp','lightgbm']; rows=[]; missing=[]
    for name in names:
        p=root/name/'metrics.json'
        if not p.exists(): missing.append(name); continue
        data=json.loads(p.read_text(encoding='utf-8-sig')); rows.append({'model':name,'family':data.get('family'),'cv_top1_mean':data.get('cv_top1_mean'),'cv_top1_std':data.get('cv_top1_std'),'cv_macro_f1_mean':data.get('cv_macro_f1_mean'),'cv_macro_f1_std':data.get('cv_macro_f1_std'),'test_top1':data.get('test_top1'),'test_macro_f1':data.get('test_macro_f1'),'test_log_loss':data.get('test_log_loss'),'test_route_continuity_argmax':data.get('test_route_continuity_argmax'),'inference_ms_per_point':data.get('inference_ms_per_point')})
    if not rows: raise ValueError('No model metrics found')
    df=pd.DataFrame(rows); df.to_csv(root/'comparison.csv',index=False)
    subtitle='Preliminary: missing '+', '.join(missing) if missing else 'All five model reports available; verify common data and folds'
    x=np.arange(len(df)); fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes[0,0].bar(x,df.cv_macro_f1_mean,yerr=df.cv_macro_f1_std,capsize=4,color='#3d7192'); axes[0,0].set_title('Grouped CV macro F1 (fold SD)'); axes[0,0].set_ylim(0,1)
    axes[0,1].bar(x-0.18,df.test_top1,width=.36,label='Top-1'); axes[0,1].bar(x+0.18,df.test_macro_f1,width=.36,label='Macro F1'); axes[0,1].set_title('Student-test scores'); axes[0,1].set_ylim(0,1); axes[0,1].legend()
    axes[1,0].bar(x,df.test_log_loss,color='#ab6a43'); axes[1,0].set_title('Student-test candidate log loss')
    axes[1,1].bar(x,df.inference_ms_per_point,color='#687b54'); axes[1,1].set_title('Inference time (ms per point)')
    for ax in axes.flat: ax.set_xticks(x,df.model,rotation=15,ha='right'); ax.grid(axis='y',alpha=.2)
    fig.suptitle(subtitle,fontsize=12); fig.savefig(figures/'model_comparison_preliminary.png',dpi=150); plt.close(fig)
    (root/'comparison_status.json').write_text(json.dumps({'included':df.model.tolist(),'missing':missing,'status':'preliminary' if missing else 'complete','metrics_source':'results/<model>/metrics.json'},indent=2),encoding='utf-8')
    print('included:',', '.join(df.model),'missing:',', '.join(missing) if missing else 'none')
if __name__=='__main__': main()
