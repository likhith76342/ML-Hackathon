"""Build shared parquet data and trip-based folds. Run: python -m pathsnap.build_dataset"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from .config import CFG,repo_path
from .io import list_trips,trip_dir,load_trip,load_route
from .candidates import build_candidates
from .labels import align_points_to_route,add_labels
from .metrics import candidate_miss_rate

def main():
    root=repo_path(CFG['data_root']); out=repo_path(CFG['processed_dir']); out.mkdir(parents=True,exist_ok=True)
    train=list_trips(root/'train'); test=list_trips(root/'student_test')
    if len(train)<CFG['n_cv_folds'] or not test: raise ValueError('Training and student_test trips are required under data_root')
    folds=np.empty(len(train),dtype=int); ids=np.asarray(train)
    for i,(_,valid) in enumerate(GroupKFold(n_splits=CFG['n_cv_folds']).split(ids,groups=ids)): folds[valid]=i
    fold_file=pd.DataFrame({'trip_id':ids,'fold':folds})
    writers={}; temp=[]
    try:
        for split,trips,dataset in [('train',train,'train_candidates.parquet'),('student_test',test,'test_candidates.parquet')]:
            for trip_id in trips:
                folder=trip_dir(root/split,trip_id); trip=load_trip(folder); cands=build_candidates(trip); refs=align_points_to_route(trip,load_route(folder))
                labelled=add_labels(cands,refs); refs['ref_in_candidates']=False
                pairs=set(zip(cands.point_idx,cands.arc_id)); refs['ref_in_candidates']=[(int(i),int(a)) in pairs for i,a in zip(refs.point_idx,refs.ref_arc)]
                for name,df in [(dataset,labelled),('point_refs.parquet',refs)]:
                    table=pa.Table.from_pandas(df,preserve_index=False)
                    if name not in writers:
                        tmp=out/(name+'.partial.parquet'); temp.append(tmp); writers[name]=pq.ParquetWriter(tmp,table.schema)
                    writers[name].write_table(table)
                print(f'{split}/{trip_id:08d} points={len(refs)} candidates={len(cands)} miss={candidate_miss_rate(cands,refs):.2%} distance_over_100m={(refs.ref_dist_m>100).sum()}',flush=True)
        for w in writers.values(): w.close()
        writers.clear()
        for tmp in temp: tmp.replace(out/tmp.name.removesuffix('.partial.parquet'))
        fold_file.to_csv(out/'folds.csv',index=False)
    finally:
        for w in writers.values(): w.close()
        for tmp in temp:
            if tmp.exists(): tmp.unlink()
if __name__=='__main__': main()

