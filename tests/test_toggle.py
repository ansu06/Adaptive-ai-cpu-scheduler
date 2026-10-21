import unittest
import pandas as pd
from drift.workload_detector import detect_drift, calculate_training_statistics, calculate_drift_score
from scheduler.schedulers import load_processes
from model.adaptive_scheduler import adaptive_schedule

class TestWorkloadAndDrift(unittest.TestCase):
    def test_normal_workload_drift(self):
        # 1. Normal workload with 5 processes
        processes = load_processes(n=5)
        df_normal = pd.DataFrame({
            "Job Id": [int(str(p["pid"]).replace("P", "")) for p in processes],
            "Burst time": [p["burst"] for p in processes],
            "Arrival Time": [p["arrival"] for p in processes]
        })
        
        result_normal = detect_drift(df_normal)
        self.assertEqual(result_normal["status"], "WORKLOAD STABLE")
        self.assertTrue(result_normal["trust_ai"])
        self.assertLess(result_normal["drift_score"], 1.5)
        
        adaptive_norm = adaptive_schedule(df_normal)
        self.assertEqual(adaptive_norm["decision"]["drift_status"], "WORKLOAD STABLE")
        self.assertLess(adaptive_norm["decision"]["drift_score"], 1.5)

    def test_drift_test_workload_drift(self):
        # 2. Drift test workload with 5 processes
        processes = load_processes(n=5)
        drift_processes = []
        for i, p in enumerate(processes):
            drift_processes.append({
                "pid": p["pid"],
                "arrival": 200 + (i * 25),
                "burst": 800 + (i * 100)
            })
            
        df_drift = pd.DataFrame({
            "Job Id": [int(str(p["pid"]).replace("P", "")) for p in drift_processes],
            "Burst time": [p["burst"] for p in drift_processes],
            "Arrival Time": [p["arrival"] for p in drift_processes]
        })
        
        result_drift = detect_drift(df_drift)
        self.assertEqual(result_drift["status"], "DRIFT DETECTED")
        self.assertFalse(result_drift["trust_ai"])
        self.assertGreaterEqual(result_drift["drift_score"], 1.5)
        
        adaptive_drift = adaptive_schedule(df_drift)
        self.assertEqual(adaptive_drift["decision"]["drift_status"], "DRIFT DETECTED")
        self.assertGreaterEqual(adaptive_drift["decision"]["drift_score"], 1.5)
        self.assertEqual(adaptive_drift["decision"]["decision"], "FALLBACK")

if __name__ == "__main__":
    unittest.main()
