"""Small contract checks, independent of private data files."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pathsnap.io import Trip
from pathsnap.geo import project,point_segment_distance,angle_diff_deg,add_xy
from pathsnap.candidates import build_candidates
from pathsnap.labels import align_points_to_route,add_labels
from pathsnap.metrics import top1_accuracy,candidate_miss_rate

class SharedPipelineTest(unittest.TestCase):
    def setUp(self):
        # Eastbound and westbound arcs share their geometry; point 1 is far away.
        self.trip=Trip(1,Path('.'),pd.DataFrame({'lon':[0.00005,0.01],'lat':[0.,0.],'t':[0.,1.]}),pd.DataFrame({'lon':[0.,0.0001],'lat':[0.,0.]}),pd.DataFrame({'from_node':[0,1],'to_node':[1,0]}))
    def test_geometry(self):
        self.assertAlmostEqual(project(1,0,0,0)[0],111194.9266,places=3)
        d,t=point_segment_distance(5,3,0,0,10,0); self.assertEqual(d,3); self.assertEqual(t,.5)
        self.assertEqual(angle_diff_deg(359,1),2)
    def test_candidates_labels_and_metrics(self):
        c=build_candidates(self.trip); self.assertEqual(c.point_idx.nunique(),2)
        self.assertEqual(c[c.point_idx==1].arc_id.tolist(),[-1]); self.assertTrue(c[c.arc_id<0].dist_m.isna().all()); self.assertNotIn('label',c)
        refs=align_points_to_route(self.trip,[0,1]); self.assertTrue((np.diff(refs.ref_route_pos)>=0).all())
        labeled=add_labels(c,refs); self.assertEqual(len(labeled),len(c)); self.assertEqual(candidate_miss_rate(c,refs),.5)
        probs=np.zeros(len(c)); positive=np.flatnonzero((labeled.point_idx==0)&(labeled.label==1)); probs[positive]=1
        self.assertEqual(top1_accuracy(c,probs,refs),.5)
if __name__=='__main__': unittest.main()
