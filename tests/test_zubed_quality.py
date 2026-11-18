import tempfile
import unittest
from pathlib import Path
import pandas as pd
from pathsnap.quality import make_trip_quality, flagged_trip_ids

class QualityTest(unittest.TestCase):
    def test_threshold_training_only_and_csv_roundtrip(self):
        refs=pd.DataFrame({'trip_id':[1,1,2,2,3,3,99,99], 'ref_dist_m':[51,50,51,52,1,100,99,99]})
        folds=pd.DataFrame({'trip_id':[1,2,3], 'fold':[0,1,2]})
        q=make_trip_quality(refs,folds)
        self.assertEqual(q.trip_id.tolist(),[1,2,3])
        self.assertEqual(q.flagged.tolist(),[False,True,False])
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'trip_quality.csv'; q.to_csv(p,index=False)
            self.assertEqual(flagged_trip_ids(p),[2])
if __name__=='__main__': unittest.main()
