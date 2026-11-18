"""Reproduce a diagnostic decomposition of approximate candidate misses.

Run from the repository root with the team-pinned Python environment:
    python -m models.logreg.miss_diagnostics
"""
import json
import numpy as np
import pandas as pd
from pathsnap.config import CFG, repo_path

def main():
    data=repo_path(CFG['processed_dir']); out=repo_path(CFG['results_dir'])/'logreg'; out.mkdir(parents=True,exist_ok=True)
    refs=pd.read_parquet(data/'point_refs.parquet'); folds=pd.read_csv(data/'folds.csv'); train_ids=set(folds.trip_id)
    all_rows=[]; summaries={}
    for split,filename in [('train','train_candidates.parquet'),('student_test','test_candidates.parquet')]:
        cand=pd.read_parquet(data/filename,columns=['trip_id','point_idx','arc_id','dist_m'])
        subset=refs[refs.trip_id.isin(train_ids) if split=='train' else ~refs.trip_id.isin(train_ids)].copy()
        groups=cand.groupby(['trip_id','point_idx'],sort=False).agg(n_candidates=('arc_id',lambda a:int((a>=0).sum())),nearest_candidate_m=('dist_m','min')).reset_index()
        points=subset.merge(groups,on=['trip_id','point_idx'],how='left',validate='one_to_one')
        if points.n_candidates.isna().any(): raise ValueError('A reference point has no candidate or placeholder row')
        missed=~points.ref_in_candidates.astype(bool); within=points.ref_dist_m<=CFG['candidate_radius_m']
        points['miss_no_candidates']=missed & (points.n_candidates==0)
        points['miss_ref_outside_radius']=missed & (points.ref_dist_m>CFG['candidate_radius_m'])
        points['miss_ref_within_radius']=missed & within
        points['ref_farther_100m']=points.ref_dist_m>100
        points['missed']=missed
        # The two independent descriptions (no candidates and distance to reference)
        # can overlap; within-radius misses have a candidate ranking/tie or numerical cause.
        summaries[split]={'points':len(points),'misses':int(missed.sum()),'candidate_miss_rate':float(missed.mean()),'miss_ref_outside_radius':int(points.miss_ref_outside_radius.sum()),'miss_ref_within_radius':int(points.miss_ref_within_radius.sum()),'miss_no_candidates':int(points.miss_no_candidates.sum()),'ref_farther_100m':int(points.ref_farther_100m.sum()),'max_candidates':int(CFG['max_candidates']),'candidate_radius_m':float(CFG['candidate_radius_m'])}
        summary=points.groupby('trip_id',as_index=False).agg(n_points=('point_idx','size'),misses=('missed','sum'),miss_ref_outside_radius=('miss_ref_outside_radius','sum'),miss_ref_within_radius=('miss_ref_within_radius','sum'),miss_no_candidates=('miss_no_candidates','sum'),ref_farther_100m=('ref_farther_100m','sum'),median_ref_distance_m=('ref_dist_m','median'),max_ref_distance_m=('ref_dist_m','max'))
        summary.insert(0,'split',split); summary['candidate_miss_rate']=summary.misses/summary.n_points; all_rows.append(summary)
    pd.concat(all_rows,ignore_index=True).to_csv(out/'miss_diagnostics_by_trip.csv',index=False)
    (out/'miss_diagnostics.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__': main()
