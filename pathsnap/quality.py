"""Trip-quality policy for dataset v2; never filter inference or student test."""
from pathlib import Path
import pandas as pd
from .config import CFG, repo_path

QUALITY_DATASET_VERSION = 2

def make_trip_quality(refs, folds):
    """Flag training trips when more than half of reference distances exceed 50 m."""
    ids=set(pd.to_numeric(folds['trip_id'],errors='raise').astype(int))
    r=refs[refs.trip_id.isin(ids)].copy()
    if set(r.trip_id.astype(int)) != ids: raise ValueError('References must cover every training trip in folds.csv')
    if r.ref_dist_m.isna().any(): raise ValueError('Reference distances contain missing values')
    r['far']=r.ref_dist_m>50.0
    out=r.groupby('trip_id',as_index=False,sort=True).agg(far_fraction=('far','mean'))
    out['flagged']=out.far_fraction>0.5
    return out[['trip_id','far_fraction','flagged']]

def flagged_trip_ids(path=None):
    """Return flagged training trip IDs from the shared, reviewed CSV."""
    p=Path(path) if path is not None else repo_path(CFG['processed_dir'])/'trip_quality.csv'
    q=pd.read_csv(p,dtype={'trip_id':'int64','far_fraction':'float64','flagged':'boolean'})
    if not {'trip_id','far_fraction','flagged'}.issubset(q.columns): raise ValueError('Invalid trip-quality columns')
    if q.trip_id.duplicated().any() or q[['far_fraction','flagged']].isna().any().any(): raise ValueError('Invalid or duplicate quality rows')
    if ((q.far_fraction<0)|(q.far_fraction>1)).any() or not q.flagged.equals(q.far_fraction.gt(0.5).astype('boolean')): raise ValueError('Quality flag disagrees with the >50% rule')
    return sorted(q.loc[q.flagged,'trip_id'].astype(int).tolist())

def main():
    d=repo_path(CFG['processed_dir']); quality=make_trip_quality(pd.read_parquet(d/'point_refs.parquet'),pd.read_csv(d/'folds.csv'))
    quality.to_csv(d/'trip_quality.csv',index=False)
    print(quality[quality.flagged].to_string(index=False))
if __name__=='__main__': main()
