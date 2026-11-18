"""Shared candidate and observation metrics; sequence metrics belong to Keshav."""
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,log_loss,confusion_matrix

def candidate_metrics(y_true,p_pred,threshold=0.5):
    y=np.asarray(y_true); p=np.asarray(p_pred,dtype=float)
    if len(y)!=len(p) or not len(y): raise ValueError('Candidate arrays must have equal nonzero length')
    pred=p>=threshold
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'log_loss':float(log_loss(y,p,labels=[0,1])),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def top1_accuracy(cands,probs,refs):
    if len(cands)!=len(probs): raise ValueError('One score per candidate row is required')
    if refs.duplicated(['trip_id','point_idx']).any(): raise ValueError('Duplicate references')
    df=cands[['trip_id','point_idx','arc_id']].copy().reset_index(drop=True); df['score']=np.asarray(probs); df.loc[df.arc_id<0,'score']=-np.inf
    # Only real candidates may match; a placeholder is always counted wrong.
    chosen=df.loc[df.groupby(['trip_id','point_idx'],sort=False).score.idxmax()]
    joined=refs[['trip_id','point_idx','ref_arc']].merge(chosen,on=['trip_id','point_idx'],how='left',validate='one_to_one')
    return float(((joined.arc_id>=0)&(joined.arc_id==joined.ref_arc)).mean()) if len(joined) else float('nan')

def candidate_miss_rate(cands,refs):
    real=cands[cands.arc_id>=0][['trip_id','point_idx','arc_id']].drop_duplicates()
    match=refs[['trip_id','point_idx','ref_arc']].merge(real,left_on=['trip_id','point_idx','ref_arc'],right_on=['trip_id','point_idx','arc_id'],how='left',indicator=True)
    return float((match['_merge']=='left_only').mean()) if len(match) else float('nan')


