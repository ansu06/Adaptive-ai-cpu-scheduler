import sys
import os
import unittest
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.decision_engine import make_decision

class TestDecisionEngine(unittest.TestCase):
    def test_trust_ai_branch(self):
        # Mock drift detection as stable and high confidence
        with patch("model.decision_engine.detect_drift") as mock_drift, \
             patch("model.decision_engine.predict_scheduler") as mock_pred:
            mock_drift.return_value = {
                "status": "WORKLOAD STABLE",
                "drift_score": 0.85,
                "trust_ai": True,
                "features": {}
            }
            mock_pred.return_value = (
                "SRTF",
                {"FCFS": 0.1, "SJF": 0.1, "SRTF": 0.8},
                {}
            )
            df = pd.DataFrame({"Job Id": [1], "Arrival Time": [0], "Burst time": [5]})
            res = make_decision(df)
            self.assertEqual(res["decision"], "TRUST AI")
            self.assertEqual(res["selected_scheduler"], "SRTF")
            self.assertTrue(res["trust_ai"])
            self.assertAlmostEqual(res["ai_confidence"], 0.8)

    def test_low_confidence_re_evaluate_branch(self):
        # Mock drift detection as stable, but low confidence (< 0.50)
        with patch("model.decision_engine.detect_drift") as mock_drift, \
             patch("model.decision_engine.predict_scheduler") as mock_pred:
            mock_drift.return_value = {
                "status": "WORKLOAD STABLE",
                "drift_score": 0.95,
                "trust_ai": True,
                "features": {}
            }
            mock_pred.return_value = (
                "SJF",
                {"FCFS": 0.35, "SJF": 0.40, "SRTF": 0.25},
                {}
            )
            df = pd.DataFrame({"Job Id": [1], "Arrival Time": [0], "Burst time": [5]})
            res = make_decision(df)
            self.assertEqual(res["decision"], "RE-EVALUATE")
            self.assertIsNone(res["selected_scheduler"])
            self.assertIn("below decision threshold", res["reason"])

    def test_drift_fallback_branch(self):
        # Mock drift detection as drift detected
        with patch("model.decision_engine.detect_drift") as mock_drift, \
             patch("model.decision_engine.predict_scheduler") as mock_pred:
            mock_drift.return_value = {
                "status": "DRIFT DETECTED",
                "drift_score": 3.45,
                "trust_ai": False,
                "features": {}
            }
            mock_pred.return_value = (
                "SRTF",
                {"FCFS": 0.05, "SJF": 0.05, "SRTF": 0.90},
                {}
            )
            df = pd.DataFrame({"Job Id": [1], "Arrival Time": [0], "Burst time": [5]})
            res = make_decision(df)
            self.assertEqual(res["decision"], "FALLBACK")
            self.assertIsNone(res["selected_scheduler"])
            self.assertFalse(res["trust_ai"])
            self.assertIn("Workload drift detected beyond safety threshold", res["reason"])

    def test_threshold_boundary_1_50(self):
        # Exactly at threshold 1.50
        with patch("model.decision_engine.detect_drift") as mock_drift, \
             patch("model.decision_engine.predict_scheduler") as mock_pred:
            mock_drift.return_value = {
                "status": "DRIFT DETECTED",
                "drift_score": 1.50,
                "trust_ai": False,
                "features": {}
            }
            mock_pred.return_value = (
                "SRTF",
                {"FCFS": 0.1, "SJF": 0.1, "SRTF": 0.8},
                {}
            )
            df = pd.DataFrame({"Job Id": [1], "Arrival Time": [0], "Burst time": [5]})
            res = make_decision(df)
            self.assertEqual(res["decision"], "FALLBACK")

if __name__ == "__main__":
    unittest.main()
