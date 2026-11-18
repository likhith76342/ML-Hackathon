"""Approximate route-position labels, never used to construct inference features."""
import numpy as np
import pandas as pd
from .geo import add_xy,point_segment_distance

def align_points_to_route(trip,route):
    add_xy(trip); route=np.asarray(route,dtype=int); arcs=trip.arcs.to_numpy(dtype=int)
    if len(route)==0 or (route<0).any() or (route>=len(arcs)).any(): raise ValueError('Invalid reference route')
    nodes=trip.nodes[['x','y']].to_numpy(); start=nodes[arcs[route,0]]; end=nodes[arcs[route,1]]; gps=trip.track[['x','y']].to_numpy(); n=len(gps); m=len(route)
    if n==0: raise ValueError('Empty track')
    dtype=np.uint16 if m<=65535 else np.uint32; back=np.empty((n,m),dtype=dtype)
    def dist(p): return point_segment_distance(p[0],p[1],start[:,0],start[:,1],end[:,0],end[:,1])[0]
    cost=dist(gps[0]); back[0]=0
    for i in range(1,n):
        best=np.minimum.accumulate(cost); change=np.r_[True,cost[1:]<best[:-1]]
        back[i]=np.maximum.accumulate(np.where(change,np.arange(m),0)); cost=best+dist(gps[i])
    pos=np.empty(n,dtype=np.int64); pos[-1]=np.argmin(cost)
    for i in range(n-1,0,-1): pos[i-1]=back[i,pos[i]]
    d,_=point_segment_distance(gps[:,0],gps[:,1],start[pos,0],start[pos,1],end[pos,0],end[pos,1])
    return pd.DataFrame({'trip_id':trip.trip_id,'point_idx':np.arange(n),'ref_arc':route[pos],'ref_route_pos':pos,'ref_dist_m':d})

def add_labels(cands,refs):
    merged=cands.merge(refs[['trip_id','point_idx','ref_arc']],on=['trip_id','point_idx'],how='left',validate='many_to_one',sort=False)
    if merged.ref_arc.isna().any(): raise ValueError('Missing point reference')
    merged['label']=((merged.arc_id==merged.ref_arc)&(merged.arc_id>=0)).astype(int)
    return merged.drop(columns='ref_arc').reset_index(drop=True)
