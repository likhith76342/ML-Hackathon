"""Inference and training candidate generation share this exact code path."""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from .config import CFG
from .features import FEATURE_COLUMNS
from .geo import add_xy,point_segment_distance,bearing_deg,angle_diff_deg

def build_candidates(trip, radius_m=None, k=None):
    radius_m=CFG['candidate_radius_m'] if radius_m is None else radius_m
    k=CFG['max_candidates'] if k is None else k
    if radius_m<=0 or k<1: raise ValueError('radius_m and k must be positive')
    add_xy(trip); xy=trip.nodes[['x','y']].to_numpy(); arcs=trip.arcs.to_numpy(dtype=int)
    if trip.track.empty or len(arcs)==0: raise ValueError('Track and road arcs must not be empty')
    if (arcs<0).any() or (arcs>=len(xy)).any(): raise ValueError('Invalid road node ID')
    start=xy[arcs[:,0]]; end=xy[arcs[:,1]]
    tree=STRtree(shapely.linestrings(np.stack([start,end],axis=1))); gps=trip.track[['x','y']].to_numpy(); rows=[]
    for i,p in enumerate(gps):
        ids=tree.query(shapely.box(p[0]-radius_m,p[1]-radius_m,p[0]+radius_m,p[1]+radius_m))
        if len(ids):
            d,t=point_segment_distance(p[0],p[1],start[ids,0],start[ids,1],end[ids,0],end[ids,1]); keep=np.flatnonzero(d<=radius_m); keep=keep[np.lexsort((ids[keep],d[keep]))][:k]
        else: keep=[]
        if len(keep)==0: rows.append((trip.trip_id,i,float(trip.track.t.iloc[i]),-1,np.nan,np.nan))
        else:
            for j in keep: rows.append((trip.trip_id,i,float(trip.track.t.iloc[i]),int(ids[j]),float(d[j]),float(t[j])))
    frame=pd.DataFrame(rows,columns=['trip_id','point_idx','t','arc_id','dist_m','proj_t']); valid=frame.arc_id.ge(0)
    idx=frame.point_idx.to_numpy(dtype=int); prev=np.maximum(idx-1,0); nxt=np.minimum(idx+1,len(gps)-1); ts=trip.track.t.to_numpy(); vec=gps[nxt]-gps[prev]; span=ts[nxt]-ts[prev]
    speed=np.divide(np.linalg.norm(vec,axis=1),span,out=np.zeros(len(frame)),where=span>0)
    for col in FEATURE_COLUMNS:
        if col not in frame: frame[col]=np.nan
    minimum=frame.loc[valid].groupby('point_idx').dist_m.transform('min')
    frame.loc[valid,'dist_rank']=frame.loc[valid].groupby('point_idx').cumcount()
    frame.loc[valid,'dist_minus_min']=frame.loc[valid,'dist_m']-minimum; frame.loc[valid,'dist_ratio']=frame.loc[valid,'dist_m']/(minimum+1)
    cand_ids=frame.loc[valid,'arc_id'].to_numpy(dtype=int); edge=end[cand_ids]-start[cand_ids]
    frame.loc[valid,'bearing_diff']=angle_diff_deg(bearing_deg(0,0,vec[valid,0],vec[valid,1]),bearing_deg(0,0,edge[:,0],edge[:,1]))
    frame.loc[valid,'speed_mps']=speed[valid]; frame.loc[valid,'heading_valid']=(speed[valid]>=1).astype(float)
    frame.loc[valid,'dt_prev']=(ts[idx]-ts[prev])[valid]; frame.loc[valid,'dt_next']=(ts[nxt]-ts[idx])[valid]
    frame.loc[valid,'n_candidates']=frame.loc[valid].groupby('point_idx').arc_id.transform('size').astype(float)
    frame.loc[valid,'arc_length_m']=np.linalg.norm(edge,axis=1)
    return frame[['trip_id','point_idx','t','arc_id']+FEATURE_COLUMNS].reset_index(drop=True)
