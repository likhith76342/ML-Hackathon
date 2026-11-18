"""Spatial spot-check for the two largest training miss contributors."""
import json
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from pathsnap.config import CFG,repo_path
from pathsnap.io import load_trip,load_route,trip_dir
from pathsnap.geo import add_xy

def nearest_distances(points,segments):
    tree=STRtree(shapely.linestrings(segments)); pairs,dist=tree.query_nearest(shapely.points(points),return_distance=True)
    result=np.full(len(points),np.inf); np.minimum.at(result,pairs[0],dist)
    return result

def main():
    data=repo_path(CFG['processed_dir']); root=repo_path(CFG['data_root'])/'train'; out=repo_path(CFG['results_dir'])/'logreg'
    detail=pd.read_csv(out/'miss_diagnostics_by_trip.csv'); ids=detail[detail.split=='train'].nlargest(2,'misses').trip_id.tolist(); refs=pd.read_parquet(data/'point_refs.parquet'); results=[]
    for id in ids:
        folder=trip_dir(root,int(id)); trip=add_xy(load_trip(folder)); arcs=trip.arcs.to_numpy(dtype=int); xy=trip.nodes[['x','y']].to_numpy(); segments=np.stack([xy[arcs[:,0]],xy[arcs[:,1]]],axis=1); selected=np.arange(0,len(trip.track),20); points=trip.track[['x','y']].to_numpy()[selected]
        route=np.asarray(load_route(folder),dtype=int); nearest_road=nearest_distances(points,segments); nearest_ref=nearest_distances(points,segments[route]); aligned=refs[refs.trip_id==id].set_index('point_idx').loc[selected,'ref_dist_m'].to_numpy()
        results.append({'trip_id':int(id),'sampling':'every 20th point in track order','sample_points':len(selected),'median_nearest_any_road_m':float(np.median(nearest_road)),'fraction_nearest_any_road_over_50m':float(np.mean(nearest_road>50)),'median_nearest_reference_arc_m':float(np.median(nearest_ref)),'fraction_nearest_reference_arc_over_50m':float(np.mean(nearest_ref>50)),'fraction_alignment_over_10m_worse_than_nearest_reference':float(np.mean(aligned-nearest_ref>10))})
    (out/'miss_spotcheck.json').write_text(json.dumps(results,indent=2),encoding='utf-8'); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
