import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler.schedulers import fcfs, sjf, srtf, round_robin_metrics, summarize


class TestSchedulers(unittest.TestCase):

    def test_normal_workload_fcfs(self):
        """Test FCFS on a standard 3-process workload."""
        processes = [
            {"pid": "P1", "arrival": 0, "burst": 5},
            {"pid": "P2", "arrival": 1, "burst": 3},
            {"pid": "P3", "arrival": 2, "burst": 8},
        ]
        order, results, trace = fcfs(processes)
        self.assertEqual(order, ["P1", "P2", "P3"])
        self.assertEqual(results[0]["completion"], 5)
        self.assertEqual(results[0]["waiting"], 0)
        self.assertEqual(results[0]["response"], 0)
        self.assertEqual(results[1]["completion"], 8)
        self.assertEqual(results[1]["waiting"], 4)
        self.assertEqual(results[1]["response"], 4)
        self.assertEqual(results[2]["completion"], 16)
        self.assertEqual(results[2]["waiting"], 6)
        self.assertEqual(results[2]["response"], 6)

    def test_sjf_non_preemptive(self):
        """Test SJF on simultaneous and staggered arrival."""
        processes = [
            {"pid": "P1", "arrival": 0, "burst": 7},
            {"pid": "P2", "arrival": 2, "burst": 4},
            {"pid": "P3", "arrival": 4, "burst": 1},
            {"pid": "P4", "arrival": 5, "burst": 4},
        ]
        order, results, trace = sjf(processes)
        self.assertEqual(order, ["P1", "P3", "P2", "P4"])
        p3_res = next(r for r in results if r["pid"] == "P3")
        self.assertEqual(p3_res["completion"], 8)
        self.assertEqual(p3_res["response"], 3)
        self.assertEqual(p3_res["waiting"], 3)
        self.assertEqual(p3_res["turnaround"], 4)

    def test_srtf_preemption_and_response_time(self):
        """Test SRTF preemptive scheduling and ensure response time is measured at first execution."""
        processes = [
            {"pid": "P1", "arrival": 0, "burst": 8},
            {"pid": "P2", "arrival": 1, "burst": 4},
            {"pid": "P3", "arrival": 2, "burst": 9},
            {"pid": "P4", "arrival": 3, "burst": 5},
        ]
        order, results, trace = srtf(processes)
        p1_res = next(r for r in results if r["pid"] == "P1")
        self.assertEqual(p1_res["response"], 0)
        self.assertEqual(p1_res["completion"], 17)
        self.assertEqual(p1_res["turnaround"], 17)
        self.assertEqual(p1_res["waiting"], 9)

        p2_res = next(r for r in results if r["pid"] == "P2")
        self.assertEqual(p2_res["response"], 0)
        self.assertEqual(p2_res["completion"], 5)

    def test_idle_cpu_period_jump(self):
        """Test that idle intervals are correctly fast-forwarded without hanging."""
        processes = [
            {"pid": "P1", "arrival": 0, "burst": 3},
            {"pid": "P2", "arrival": 50, "burst": 4},
        ]
        _, fcfs_res, _ = fcfs(processes)
        self.assertEqual(fcfs_res[1]["completion"], 54)
        self.assertEqual(fcfs_res[1]["waiting"], 0)
        self.assertEqual(fcfs_res[1]["response"], 0)

        _, sjf_res, _ = sjf(processes)
        self.assertEqual(sjf_res[1]["completion"], 54)
        self.assertEqual(sjf_res[1]["waiting"], 0)

        _, srtf_res, _ = srtf(processes)
        self.assertEqual(srtf_res[1]["completion"], 54)
        self.assertEqual(srtf_res[1]["waiting"], 0)
        self.assertEqual(srtf_res[1]["response"], 0)

        _, rr_res, _ = round_robin_metrics(processes, quantum=4)
        self.assertEqual(rr_res[1]["completion"], 54)
        self.assertEqual(rr_res[1]["waiting"], 0)
        self.assertEqual(rr_res[1]["response"], 0)

    def test_zero_burst_handling(self):
        """Test zero-burst processes complete without infinite loops."""
        processes = [
            {"pid": "P0", "arrival": 0, "burst": 0},
            {"pid": "P1", "arrival": 0, "burst": 4},
        ]
        order, results, trace = srtf(processes)
        self.assertEqual(len(results), 2)
        p0_res = next(r for r in results if r["pid"] == "P0")
        self.assertEqual(p0_res["completion"], 0)
        self.assertEqual(p0_res["waiting"], 0)

        order_rr, rr_results, _ = round_robin_metrics(processes, quantum=2)
        self.assertEqual(len(rr_results), 2)

    def test_duplicate_pid_handling(self):
        """Test that duplicate process IDs do not overwrite dictionary entries."""
        processes = [
            {"pid": "P1", "arrival": 0, "burst": 4},
            {"pid": "P1", "arrival": 2, "burst": 2},
        ]
        _, sjf_res, _ = sjf(processes)
        self.assertEqual(len(sjf_res), 2)

        _, srtf_res, _ = srtf(processes)
        self.assertEqual(len(srtf_res), 2)

        _, rr_res, _ = round_robin_metrics(processes, quantum=2)
        self.assertEqual(len(rr_res), 2)

    def test_round_robin_metrics_and_quantum(self):
        """Test Round Robin metrics and quantum switching."""
        processes = [
            {"pid": "P1", "arrival": 0, "burst": 5},
            {"pid": "P2", "arrival": 0, "burst": 3},
        ]
        order, results, trace = round_robin_metrics(processes, quantum=2)
        p1 = results[0]
        p2 = results[1]
        self.assertEqual(p1["completion"], 8)
        self.assertEqual(p1["response"], 0)
        self.assertEqual(p1["turnaround"], 8)
        self.assertEqual(p1["waiting"], 3)

        self.assertEqual(p2["completion"], 7)
        self.assertEqual(p2["response"], 2)
        self.assertEqual(p2["turnaround"], 7)
        self.assertEqual(p2["waiting"], 4)

    def test_every_process_completes_all_schedulers(self):
        """Verify that every process completes across all 4 scheduling algorithms on a complex workload."""
        processes = [
            {"pid": f"P{i}", "arrival": i * 1.5, "burst": (i * 3) + 2}
            for i in range(8)
        ]
        for name, func in [("FCFS", fcfs), ("SJF", sjf), ("SRTF", srtf)]:
            order, results, trace = func(processes)
            self.assertEqual(len(results), len(processes))
            for r in results:
                self.assertGreaterEqual(r["completion"], 0)
                self.assertGreaterEqual(r["waiting"], 0)
                self.assertGreaterEqual(r["turnaround"], 0)
                self.assertGreaterEqual(r["response"], 0)

        rr_order, rr_results, _ = round_robin_metrics(processes, quantum=4)
        self.assertEqual(len(rr_results), len(processes))
        for r in rr_results:
            self.assertGreaterEqual(r["completion"], 0)
            self.assertGreaterEqual(r["waiting"], 0)
            self.assertGreaterEqual(r["turnaround"], 0)
            self.assertGreaterEqual(r["response"], 0)

    def test_summarize_metrics(self):
        """Verify summarize accurately aggregates waiting, turnaround, and response averages."""
        order = ["P1", "P2"]
        results = [
            {"pid": "P1", "waiting": 10.0, "turnaround": 20.0, "response": 5.0},
            {"pid": "P2", "waiting": 20.0, "turnaround": 30.0, "response": 15.0},
        ]
        summary = summarize("TestScheduler", order, results)
        self.assertEqual(summary["scheduler"], "TestScheduler")
        self.assertEqual(summary["avg_waiting"], 15.0)
        self.assertEqual(summary["avg_turnaround"], 25.0)
        self.assertEqual(summary["avg_response"], 10.0)


if __name__ == "__main__":
    unittest.main()
