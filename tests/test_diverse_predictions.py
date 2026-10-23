import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.predict import predict_scheduler, extract_features, FEATURES
from scheduler.schedulers import load_processes
from model.decision_engine import make_decision
from model.adaptive_scheduler import adaptive_schedule


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

    def test_feature_order_and_consistency(self):
        wl = pd.DataFrame({
            "Job Id": [1, 2, 3],
            "Arrival Time": [0.0, 5.0, 10.0],
            "Burst time": [20, 15, 30]
        })
        feats = extract_features(wl)
        self.assertEqual(list(feats.keys()), FEATURES)
        for k, v in feats.items():
            self.assertFalse(np.isnan(v), f"Feature {k} is NaN")
            self.assertFalse(np.isinf(v), f"Feature {k} is Inf")

    def test_missing_and_edge_case_features(self):
        # Single process: std and correlation must be safely 0.0
        single_df = pd.DataFrame({
            "Job Id": [1],
            "Arrival Time": [0.0],
            "Burst time": [10]
        })
        feats = extract_features(single_df)
        self.assertEqual(feats["num_processes"], 1)
        self.assertEqual(feats["std_burst"], 0.0)
        self.assertEqual(feats["std_arrival"], 0.0)
        self.assertEqual(feats["arrival_burst_corr"], 0.0)
        self.assertEqual(feats["avg_arrival_gap"], 0.0)

        # Simultaneous arrivals with varied burst: arrival std is 0.0 -> corr is 0.0
        simult_df = pd.DataFrame({
            "Job Id": [1, 2, 3],
            "Arrival Time": [0.0, 0.0, 0.0],
            "Burst time": [10, 20, 30]
        })
        feats_simult = extract_features(simult_df)
        self.assertEqual(feats_simult["std_arrival"], 0.0)
        self.assertEqual(feats_simult["arrival_burst_corr"], 0.0)

    def test_prediction_confidence_monotonicity(self):
        wl = pd.DataFrame({
            "Job Id": [1, 2, 3, 4],
            "Arrival Time": [0.0, 1.0, 2.0, 3.0],
            "Burst time": [10, 5, 2, 8]
        })
        pred, conf, _ = predict_scheduler(wl)
        # The predicted class must have the highest confidence score
        self.assertEqual(pred, max(conf, key=conf.get))
        self.assertAlmostEqual(sum(conf.values()), 1.0, places=4)

    def test_historical_sjf_prediction_regression(self):
        # Default dashboard workload (N=5 from load_processes)
        procs = load_processes(n=5)
        df = pd.DataFrame({
            "Job Id": [int(str(p["pid"]).replace("P", "")) for p in procs],
            "Burst time": [p["burst"] for p in procs],
            "Arrival Time": [p["arrival"] for p in procs]
        })
        pred, conf, _ = predict_scheduler(df)
        self.assertEqual(pred, "SJF", f"Expected SJF for N=5 normal workload, got {pred}")
        self.assertGreaterEqual(conf["SJF"], 0.50, f"Expected SJF confidence >= 50%, got {conf['SJF']}")

    def test_separation_of_prediction_and_fallback(self):
        # High drift workload (large bursts, large arrivals)
        drift_df = pd.DataFrame({
            "Job Id": [1, 2, 3, 4, 5],
            "Arrival Time": [200.0, 225.0, 250.0, 275.0, 300.0],
            "Burst time": [800, 900, 1000, 1100, 1200]
        })
        dec = make_decision(drift_df)
        # AI prediction is retained honestly as an ML classification
        self.assertIn(dec["ai_prediction"], ["FCFS", "SJF", "SRTF"])
        # Drift must be detected
        self.assertEqual(dec["drift_status"], "DRIFT DETECTED")
        self.assertFalse(dec["trust_ai"])
        self.assertEqual(dec["decision"], "FALLBACK")
        self.assertIsNone(dec["selected_scheduler"])

        # Adaptive scheduler must run fallback empirical evaluation
        res = adaptive_schedule(drift_df)
        self.assertEqual(res["method"], "DRIFT FALLBACK")
        self.assertIsNotNone(res["selected_scheduler"])
        self.assertIn(res["selected_scheduler"], ["FCFS", "SJF", "SRTF", "Round Robin"])

    def test_diverse_patterns(self):
        # Workload 1: Sequential arrivals
        wl_fcfs = pd.DataFrame({
            "Job Id": [1, 2, 3, 4],
            "Arrival Time": [0, 10, 20, 30],
            "Burst time": [8, 8, 8, 8]
        })
        pred1, conf1, _ = predict_scheduler(wl_fcfs)
        self.assertEqual(pred1, max(conf1, key=conf1.get))

        # Workload 2: Simultaneous arrivals
        wl_sjf = pd.DataFrame({
            "Job Id": [1, 2, 3, 4, 5],
            "Arrival Time": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Burst time": [1, 3, 5, 10, 20]
        })
        pred2, conf2, _ = predict_scheduler(wl_sjf)
        self.assertEqual(pred2, max(conf2, key=conf2.get))

        # Workload 3: Heavy preemption
        wl_srtf = pd.DataFrame({
            "Job Id": [1, 2, 3, 4],
            "Arrival Time": [0.0, 2.0, 4.0, 6.0],
            "Burst time": [40, 2, 2, 2]
        })
        pred3, conf3, _ = predict_scheduler(wl_srtf)
        self.assertEqual(pred3, max(conf3, key=conf3.get))


if __name__ == "__main__":
    unittest.main()
