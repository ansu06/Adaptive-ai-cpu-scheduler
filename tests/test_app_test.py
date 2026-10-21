import unittest
from streamlit.testing.v1 import AppTest

class TestAppStreamlitState(unittest.TestCase):
    def test_bidirectional_workload_toggles(self):
        # 1. Initial render (Default: Normal Workload)
        at = AppTest.from_file("../app.py", default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App exception on initial load: {at.exception}")
        
        # Check initial radio and badge
        wl_radio = [r for r in at.radio if r.key == "wl_radio"][0]
        self.assertEqual(wl_radio.value, "Normal Workload")
        
        norm_badge = [m.value for m in at.markdown if "NORMAL PROFILE" in m.value or "DRIFT TEST" in m.value]
        self.assertTrue(len(norm_badge) > 0)
        self.assertIn("NORMAL PROFILE", norm_badge[0])
        self.assertNotIn("DRIFT TEST ACTIVE", norm_badge[0])
        
        # Check AI decision panel status
        ai_badges = [m.value for m in at.markdown if "STABLE WORKLOAD" in m.value or "DRIFT DETECTED" in m.value]
        self.assertTrue(len(ai_badges) > 0)
        self.assertIn("STABLE WORKLOAD", ai_badges[0])

        # ----------------------------------------------------
        # 2. Toggle to Drift Test
        # ----------------------------------------------------
        wl_radio.set_value("Drift Test").run()
        self.assertFalse(at.exception, f"App exception after toggle to Drift Test: {at.exception}")
        
        wl_radio = [r for r in at.radio if r.key == "wl_radio"][0]
        self.assertEqual(wl_radio.value, "Drift Test")
        
        drift_badge = [m.value for m in at.markdown if "NORMAL PROFILE" in m.value or "DRIFT TEST" in m.value]
        self.assertTrue(len(drift_badge) > 0)
        self.assertIn("DRIFT TEST ACTIVE", drift_badge[0])
        self.assertNotIn("NORMAL PROFILE", drift_badge[0])
        
        ai_badges_drift = [m.value for m in at.markdown if "STABLE WORKLOAD" in m.value or "DRIFT DETECTED" in m.value]
        self.assertTrue(len(ai_badges_drift) > 0)
        self.assertIn("DRIFT DETECTED", ai_badges_drift[0])

        # ----------------------------------------------------
        # 3. Toggle BACK to Normal Workload
        # ----------------------------------------------------
        wl_radio.set_value("Normal Workload").run()
        self.assertFalse(at.exception, f"App exception after toggle back to Normal Workload: {at.exception}")
        
        wl_radio = [r for r in at.radio if r.key == "wl_radio"][0]
        self.assertEqual(wl_radio.value, "Normal Workload")
        
        norm_badge_2 = [m.value for m in at.markdown if "NORMAL PROFILE" in m.value or "DRIFT TEST" in m.value]
        self.assertTrue(len(norm_badge_2) > 0)
        self.assertIn("NORMAL PROFILE", norm_badge_2[0])
        self.assertNotIn("DRIFT TEST ACTIVE", norm_badge_2[0])
        
        ai_badges_norm_2 = [m.value for m in at.markdown if "STABLE WORKLOAD" in m.value or "DRIFT DETECTED" in m.value]
        self.assertTrue(len(ai_badges_norm_2) > 0)
        self.assertIn("STABLE WORKLOAD", ai_badges_norm_2[0])

        # ----------------------------------------------------
        # 4. Toggle AGAIN to Drift Test
        # ----------------------------------------------------
        wl_radio.set_value("Drift Test").run()
        self.assertFalse(at.exception, f"App exception after second toggle to Drift Test: {at.exception}")
        
        wl_radio = [r for r in at.radio if r.key == "wl_radio"][0]
        self.assertEqual(wl_radio.value, "Drift Test")
        
        drift_badge_2 = [m.value for m in at.markdown if "NORMAL PROFILE" in m.value or "DRIFT TEST" in m.value]
        self.assertTrue(len(drift_badge_2) > 0)
        self.assertIn("DRIFT TEST ACTIVE", drift_badge_2[0])
        self.assertNotIn("NORMAL PROFILE", drift_badge_2[0])
        
        ai_badges_drift_2 = [m.value for m in at.markdown if "STABLE WORKLOAD" in m.value or "DRIFT DETECTED" in m.value]
        self.assertTrue(len(ai_badges_drift_2) > 0)
        self.assertIn("DRIFT DETECTED", ai_badges_drift_2[0])

if __name__ == "__main__":
    unittest.main()
