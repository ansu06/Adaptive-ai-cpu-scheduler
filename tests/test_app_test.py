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

    def test_process_slider_driven_workload(self):
        # 1. Initial render at N=5
        at = AppTest.from_file("../app.py", default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App exception: {at.exception}")

        # Check slider exists and starts at default 5
        slider = [s for s in at.slider if s.key == "proc_slider"][0]
        self.assertEqual(slider.value, 5)

        # In Balanced profile at N=5: raw recommendation is SJF
        ai_pred_texts = [m.value for m in at.markdown if "Raw AI:" in m.value]
        self.assertTrue(len(ai_pred_texts) > 0)
        self.assertIn("SJF", ai_pred_texts[0])

        # 2. Move slider to N=10 processes
        slider.set_value(10).run()
        self.assertFalse(at.exception, f"App exception after moving slider: {at.exception}")

        # In Balanced profile at N=10: raw recommendation dynamically updates to SRTF
        ai_pred_texts_10 = [m.value for m in at.markdown if "Raw AI:" in m.value]
        self.assertTrue(len(ai_pred_texts_10) > 0)
        self.assertIn("SRTF", ai_pred_texts_10[0])

        # Drift status remains stable within threshold
        ai_badges = [m.value for m in at.markdown if "STABLE WORKLOAD" in m.value or "DRIFT DETECTED" in m.value]
        self.assertTrue(len(ai_badges) > 0)
        self.assertIn("STABLE WORKLOAD", ai_badges[0])

        # 3. Switch profile to "Mixed Arrivals (High Load)"
        profile_sel = [s for s in at.selectbox if s.key == "wl_profile_select"][0]
        profile_sel.select("Mixed Arrivals (High Load)").run()
        self.assertFalse(at.exception, f"App exception after selecting high load profile: {at.exception}")

        # Mathematical drift detected authentically from features
        ai_badges_drift = [m.value for m in at.markdown if "STABLE WORKLOAD" in m.value or "DRIFT DETECTED" in m.value]
        self.assertTrue(len(ai_badges_drift) > 0)
        self.assertIn("DRIFT DETECTED", ai_badges_drift[0])

        # Explanatory text informs the user of the drift
        drift_expl = [m.value for m in at.markdown if "Drift detected because the current workload features differ" in m.value]
        self.assertTrue(len(drift_expl) > 0)


if __name__ == "__main__":
    unittest.main()
