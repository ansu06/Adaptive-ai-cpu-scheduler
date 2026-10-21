import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.predict import predict_scheduler, extract_features

class TestPredictionPipeline(unittest.TestCase):
    def test_pipeline_validity(self):
        wl = pd.DataFrame({
            "Job Id": [1, 2, 3, 4],
            "Arrival Time": [0.0, 1.0, 2.0, 3.0],
            "Burst time": [10, 5, 2, 8]
        })
        pred, conf, feats = predict_scheduler(wl)
        self.assertIn(pred, ["FCFS", "SJF", "SRTF"])
        self.assertEqual(set(conf.keys()), {"FCFS", "SJF", "SRTF"})
        self.assertAlmostEqual(sum(conf.values()), 1.0, places=4)
        for p in conf.values():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
        self.assertEqual(len(feats), 11)

    def test_diverse_patterns(self):
        # Workload 1: Sequential arrivals
        wl_fcfs = pd.DataFrame({
            "Job Id": [1, 2, 3, 4],
            "Arrival Time": [0, 10, 20, 30],
            "Burst time": [8, 8, 8, 8]
        })
        pred1, conf1, _ = predict_scheduler(wl_fcfs)
        print("\nSequential Workload:", pred1, {k: f"{v*100:.1f}%" for k, v in conf1.items()})

        # Workload 2: Simultaneous arrivals
        wl_sjf = pd.DataFrame({
            "Job Id": [1, 2, 3, 4, 5],
            "Arrival Time": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Burst time": [1, 3, 5, 10, 20]
        })
        pred2, conf2, _ = predict_scheduler(wl_sjf)
        print("Simultaneous Workload:", pred2, {k: f"{v*100:.1f}%" for k, v in conf2.items()})

        # Workload 3: Heavy preemption
        wl_srtf = pd.DataFrame({
            "Job Id": [1, 2, 3, 4],
            "Arrival Time": [0.0, 2.0, 4.0, 6.0],
            "Burst time": [40, 2, 2, 2]
        })
        pred3, conf3, _ = predict_scheduler(wl_srtf)
        print("Heavy Preemption Workload:", pred3, {k: f"{v*100:.1f}%" for k, v in conf3.items()})

if __name__ == "__main__":
    unittest.main()
