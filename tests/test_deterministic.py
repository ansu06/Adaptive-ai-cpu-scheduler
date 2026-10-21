import unittest
from scheduler.schedulers import fcfs, sjf, srtf, round_robin_metrics, summarize

class TestDeterministicSchedulers(unittest.TestCase):
    def test_single_process(self):
        procs = [{"pid": "P1", "arrival": 5.0, "burst": 10}]
        for name, fn in [("FCFS", fcfs), ("SJF", sjf), ("SRTF", srtf)]:
            order, res, trace = fn(procs)
            self.assertEqual(order, ["P1"])
            self.assertEqual(res[0]["completion"], 15.0)
            self.assertEqual(res[0]["waiting"], 0.0)
            self.assertEqual(res[0]["response"], 0.0)
            self.assertEqual(res[0]["turnaround"], 10.0)
            self.assertEqual(trace, [{"pid": "P1", "start": 5.0, "end": 15.0}])

        order_rr, res_rr, trace_rr = round_robin_metrics(procs, quantum=4)
        self.assertEqual(res_rr[0]["completion"], 15.0)
        self.assertEqual(res_rr[0]["waiting"], 0.0)
        self.assertEqual(res_rr[0]["response"], 0.0)
        self.assertEqual(res_rr[0]["turnaround"], 10.0)

    def test_all_arrival_zero(self):
        # P1: 6, P2: 2, P3: 8, P4: 3. All arrive at 0.
        procs = [
            {"pid": "P1", "arrival": 0, "burst": 6},
            {"pid": "P2", "arrival": 0, "burst": 2},
            {"pid": "P3", "arrival": 0, "burst": 8},
            {"pid": "P4", "arrival": 0, "burst": 3},
        ]
        # FCFS order: P1(0-6), P2(6-8), P3(8-16), P4(16-19)
        _, f_res, _ = fcfs(procs)
        self.assertEqual([r["completion"] for r in f_res], [6, 8, 16, 19])

        # SJF order: P2(0-2), P4(2-5), P1(5-11), P3(11-19)
        _, s_res, _ = sjf(procs)
        self.assertEqual([r["completion"] for r in s_res], [11, 2, 19, 5])

        # SRTF order: same as SJF because all arrive at 0
        _, sr_res, _ = srtf(procs)
        self.assertEqual([r["completion"] for r in sr_res], [11, 2, 19, 5])

    def test_preemption_fractional_arrival(self):
        # P1 arrives at 0, burst 10.
        # P2 arrives at 1.5, burst 1.
        # Preemption should occur exactly at 1.5.
        procs = [
            {"pid": "P1", "arrival": 0.0, "burst": 10},
            {"pid": "P2", "arrival": 1.5, "burst": 1},
        ]
        order, res, trace = srtf(procs)
        p1 = res[0]
        p2 = res[1]
        self.assertEqual(p2["response"], 0.0, f"P2 response should be 0.0 but got {p2['response']}")
        self.assertEqual(p2["completion"], 2.5, f"P2 completion should be 2.5 but got {p2['completion']}")
        self.assertEqual(p1["completion"], 11.0, f"P1 completion should be 11.0 but got {p1['completion']}")

    def test_simultaneous_arrivals_tie_breaking(self):
        # P1 and P2 both arrive at 2.0 with identical burst 4.
        procs = [
            {"pid": "P1", "arrival": 2.0, "burst": 4},
            {"pid": "P2", "arrival": 2.0, "burst": 4},
        ]
        order_f, res_f, _ = fcfs(procs)
        self.assertEqual(order_f, ["P1", "P2"])
        self.assertEqual(res_f[0]["completion"], 6.0)
        self.assertEqual(res_f[1]["completion"], 10.0)

        order_s, res_s, _ = sjf(procs)
        self.assertEqual(order_s, ["P1", "P2"])
        self.assertEqual(res_s[0]["completion"], 6.0)
        self.assertEqual(res_s[1]["completion"], 10.0)

    def test_long_idle_period_advancement(self):
        procs = [
            {"pid": "P1", "arrival": 0.0, "burst": 2},
            {"pid": "P2", "arrival": 100.0, "burst": 3},
        ]
        for name, fn in [("FCFS", fcfs), ("SJF", sjf), ("SRTF", srtf)]:
            order, res, trace = fn(procs)
            self.assertEqual(res[0]["completion"], 2.0)
            self.assertEqual(res[1]["completion"], 103.0)
            self.assertEqual(res[1]["waiting"], 0.0)
            self.assertEqual(res[1]["response"], 0.0)

        order_rr, res_rr, trace_rr = round_robin_metrics(procs, quantum=4)
        self.assertEqual(res_rr[0]["completion"], 2.0)
        self.assertEqual(res_rr[1]["completion"], 103.0)
        self.assertEqual(res_rr[1]["waiting"], 0.0)
        self.assertEqual(res_rr[1]["response"], 0.0)

    def test_short_jobs_arriving_during_long_job(self):
        # Long job P1 (burst 20) arrives at 0.
        # Short job P2 (burst 2) arrives at 3.
        # Short job P3 (burst 1) arrives at 4.
        # In SRTF:
        # 0-3: P1 runs (rem 17)
        # At 3: P2 arrives (burst 2 < 17). P2 preempts!
        # 3-4: P2 runs (rem 1)
        # At 4: P3 arrives (burst 1 <= rem 1 of P2). By arrival tie-breaker, P2 continues.
        # 4-5: P2 finishes (completion 5.0).
        # At 5: P3 (rem 1 < rem 17 of P1) runs from 5-6 (completion 6.0).
        # At 6: P1 finishes remaining 17 units (completion 23.0).
        procs = [
            {"pid": "P1", "arrival": 0.0, "burst": 20},
            {"pid": "P2", "arrival": 3.0, "burst": 2},
            {"pid": "P3", "arrival": 4.0, "burst": 1},
        ]
        order, res, trace = srtf(procs)
        self.assertEqual(res[1]["completion"], 5.0)  # P2 completes at 5
        self.assertEqual(res[2]["completion"], 6.0)  # P3 completes at 6
        self.assertEqual(res[0]["completion"], 23.0) # P1 completes at 23
        self.assertEqual(res[1]["waiting"], 0.0)
        self.assertEqual(res[2]["waiting"], 1.0)     # arrived at 4, started at 5

    def test_zero_burst_processes(self):
        procs = [
            {"pid": "P1", "arrival": 0.0, "burst": 0},
            {"pid": "P2", "arrival": 2.0, "burst": 5},
        ]
        for name, fn in [("FCFS", fcfs), ("SJF", sjf), ("SRTF", srtf)]:
            order, res, trace = fn(procs)
            self.assertEqual(res[0]["completion"], 0.0)
            self.assertEqual(res[0]["waiting"], 0.0)
            self.assertEqual(res[0]["turnaround"], 0.0)
            self.assertEqual(res[1]["completion"], 7.0)

        order_rr, res_rr, trace_rr = round_robin_metrics(procs, quantum=4)
        self.assertEqual(res_rr[0]["completion"], 0.0)
        self.assertEqual(res_rr[1]["completion"], 7.0)

    def test_empty_workload(self):
        procs = []
        for name, fn in [("FCFS", fcfs), ("SJF", sjf), ("SRTF", srtf)]:
            order, res, trace = fn(procs)
            self.assertEqual(order, [])
            self.assertEqual(res, [])
            self.assertEqual(trace, [])
            summ = summarize(name, order, res)
            self.assertEqual(summ["avg_waiting"], 0.0)

        order_rr, res_rr, trace_rr = round_robin_metrics(procs, quantum=4)
        self.assertEqual(order_rr, [])
        self.assertEqual(res_rr, [])

if __name__ == "__main__":
    unittest.main()
