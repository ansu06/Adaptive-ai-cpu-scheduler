import sys
import os
import unittest
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.predict import predict_scheduler, extract_features, FEATURES
from model.decision_engine import make_decision
from model.adaptive_scheduler import adaptive_schedule
from model.process_count_analysis import run_process_count_analysis, evaluate_ground_truth_scheduler
from scheduler.schedulers import load_processes


class TestModelPerformanceAndSensitivity(unittest.TestCase):
    def test_ui_accuracy_matches_evaluation_artifact(self):
        """Verify that evaluation metrics stored in the artifact match verified values."""
        artifact_path = Path(__file__).resolve().parent.parent / "model" / "model_evaluation_artifact.json"
        self.assertTrue(artifact_path.exists(), "Evaluation artifact JSON must exist")

        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertAlmostEqual(data["test_accuracy"], 0.7250, places=3)
        self.assertGreaterEqual(data["test_accuracy"], 0.70, "Test accuracy must exceed 70% target")
        self.assertTrue(data["target_accuracy_achieved"])
        self.assertAlmostEqual(data["test_macro_f1"], 0.7258, places=3)
        self.assertAlmostEqual(data["test_balanced_accuracy"], 0.8454, places=3)
        self.assertAlmostEqual(data["cv_macro_f1_mean"], 0.7204, places=3)
        self.assertEqual(data["test_samples"], 80)
        self.assertEqual(data["train_samples"], 320)
        self.assertEqual(data["total_samples"], 400)
        self.assertIn("SVM", data["selected_model"])

    def test_process_count_change_refreshes_all_values(self):
        """Verify changing process count regenerates workload, features, and model prediction."""
        procs_4 = load_processes(n=4)
        df_4 = pd.DataFrame({
            "Job Id": [int(str(p["pid"]).replace("P", "")) for p in procs_4],
            "Burst time": [p["burst"] for p in procs_4],
            "Arrival Time": [p["arrival"] for p in procs_4]
        })
        pred_4, conf_4, feats_4 = predict_scheduler(df_4)

        procs_10 = load_processes(n=10)
        df_10 = pd.DataFrame({
            "Job Id": [int(str(p["pid"]).replace("P", "")) for p in procs_10],
            "Burst time": [p["burst"] for p in procs_10],
            "Arrival Time": [p["arrival"] for p in procs_10]
        })
        pred_10, conf_10, feats_10 = predict_scheduler(df_10)

        # Features must reflect different process count and statistics
        self.assertEqual(feats_4["num_processes"], 4)
        self.assertEqual(feats_10["num_processes"], 10)
        self.assertNotEqual(feats_4["avg_burst"], feats_10["avg_burst"])

        # Predictions differ properly according to trained decision boundaries
        self.assertEqual(pred_4, "SJF")
        self.assertEqual(pred_10, "SRTF")
        self.assertNotEqual(conf_4["SJF"], conf_10["SJF"])

    def test_features_recomputed_for_every_workload(self):
        """Verify all 11 features are extracted without NaNs or Infinities."""
        workloads = [
            pd.DataFrame({"Job Id": [1, 2], "Arrival Time": [0, 5], "Burst time": [10, 20]}),
            pd.DataFrame({"Job Id": [1, 2, 3, 4], "Arrival Time": [0, 0, 0, 0], "Burst time": [5, 5, 5, 5]}),
            pd.DataFrame({"Job Id": [1, 2, 3], "Arrival Time": [10, 20, 30], "Burst time": [50, 40, 30]}),
        ]
        for wl in workloads:
            feats = extract_features(wl)
            self.assertEqual(len(feats), 11)
            self.assertEqual(list(feats.keys()), FEATURES)
            for k, v in feats.items():
                self.assertFalse(np.isnan(v), f"Feature {k} returned NaN")
                self.assertFalse(np.isinf(v), f"Feature {k} returned Inf")

    def test_prediction_probabilities_valid_and_sum_to_one(self):
        """Verify probability calibration produces valid distributions summing to ~1.0."""
        for n in [3, 5, 7, 10]:
            procs = load_processes(n=n)
            df = pd.DataFrame({
                "Job Id": [int(str(p["pid"]).replace("P", "")) for p in procs],
                "Burst time": [p["burst"] for p in procs],
                "Arrival Time": [p["arrival"] for p in procs]
            })
            pred, conf, _ = predict_scheduler(df)
            self.assertAlmostEqual(sum(conf.values()), 1.0, places=4)
            for c, p in conf.items():
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)
            # The predicted class must be the highest probability class
            self.assertEqual(pred, max(conf, key=conf.get))

    def test_accuracy_calculated_only_against_ground_truth(self):
        """Verify accuracy is strictly evaluated against objective simulation ground-truth."""
        wl = pd.DataFrame({
            "Job Id": [1, 2, 3, 4],
            "Arrival Time": [0.0, 2.0, 4.0, 6.0],
            "Burst time": [40, 2, 3, 2]
        })
        gt = evaluate_ground_truth_scheduler(wl)
        self.assertIn(gt, ["FCFS", "SJF", "SRTF", "Round Robin"])
        # In this heavy preemption case, SRTF is the verified mathematical ground truth
        self.assertEqual(gt, "SRTF")

    def test_drift_fallback_does_not_overwrite_raw_ai_prediction(self):
        """Verify drift fallback preserves raw ML recommendation in decision output."""
        drift_df = pd.DataFrame({
            "Job Id": [1, 2, 3, 4, 5],
            "Arrival Time": [200.0, 225.0, 250.0, 275.0, 300.0],
            "Burst time": [800, 900, 1000, 1100, 1200]
        })
        dec = make_decision(drift_df)
        # Raw AI recommendation must be retained as the ML classifier output
        self.assertIn(dec["ai_prediction"], ["FCFS", "SJF", "SRTF"])
        self.assertEqual(dec["drift_status"], "DRIFT DETECTED")
        self.assertEqual(dec["decision"], "FALLBACK")
        self.assertIsNone(dec["selected_scheduler"], "Selected scheduler in decision must be None on fallback")

        res = adaptive_schedule(drift_df)
        self.assertEqual(res["method"], "DRIFT FALLBACK")
        self.assertIsNotNone(res["selected_scheduler"])

    def test_deterministic_results_for_fixed_seeds(self):
        """Verify process-count sensitivity analysis produces identical results with fixed seeds."""
        df1, _ = run_process_count_analysis(process_counts=(3, 5, 8), samples_per_count=3, base_seed=42)
        df2, _ = run_process_count_analysis(process_counts=(3, 5, 8), samples_per_count=3, base_seed=42)
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    unittest.main()
