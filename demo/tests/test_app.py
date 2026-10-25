"""Tests de régression légers pour le rendu et les actifs de la démo."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))

import app as demo_app  # noqa: E402
from state import DemoState  # noqa: E402


class DemoRenderingTests(unittest.TestCase):
    def setUp(self):
        # Conserver l'objet singleton importé par les services et remettre son état à zéro.
        demo_app.demo_state.__dict__.clear()
        demo_app.demo_state.__dict__.update(DemoState().__dict__)
        self.client = demo_app.app.test_client()

    def _set_analyzed_heart_case(self, maintenance_visible=False):
        demo_app.demo_state.current_dataset = "heart"
        demo_app.demo_state.current_case = object()
        demo_app.demo_state.last_before = 238
        demo_app.demo_state.last_after = 238
        demo_app.demo_state.cbm_applied["heart"] = maintenance_visible
        demo_app.demo_state.last_results = {
            "linear": {"query_time_ms": 1.25, "nearest_distance": 0.2, "redundant": True},
            "kdtree": {"query_time_ms": 0.48, "nearest_distance": 0.2, "redundant": True, "agreement_with_reference": 5},
            "lsh": {"query_time_ms": 0.31, "nearest_distance": 0.21, "redundant": True, "agreement_with_reference": 4},
            "_after_added": False,
            "_fastest": "lsh",
        }

    def test_home_uses_local_frontend_assets_without_alpine(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("vendor/htmx-1.9.12.min.js", html)
        self.assertIn("vendor/chart-4.5.1.umd.min.js", html)
        self.assertNotIn("alpinejs", html)
        htmx_asset = self.client.get("/static/vendor/htmx-1.9.12.min.js")
        chart_asset = self.client.get("/static/vendor/chart-4.5.1.umd.min.js")
        self.assertEqual(htmx_asset.status_code, 200)
        self.assertEqual(chart_asset.status_code, 200)
        htmx_asset.close()
        chart_asset.close()

    def test_maintenance_route_renders_svg_and_benchmark_data(self):
        self._set_analyzed_heart_case(maintenance_visible=True)

        response = self.client.get("/demo/maintenance")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-chart-data="benchmark"', html)
        self.assertIn("maintenance-comparison", html)
        self.assertNotIn("maintenanceChart", html)
        self.assertNotIn("new Chart", html)
        self.assertNotIn("💡", html)
        self.assertNotIn("🌟", html)

    def test_chart_manager_uses_the_htmx_settled_target(self):
        manager = (ROOT / "demo" / "static" / "js" / "charts.js").read_text()

        self.assertIn("htmx:afterSettle", manager)
        self.assertIn("event.detail.target || event.target", manager)

    def test_missing_parquet_engine_returns_a_clear_fragment(self):
        with patch.object(demo_app.dataset_bridge, "load_dataset", side_effect=ImportError("pyarrow is required")):
            response = self.client.post("/demo/select-dataset", data={"dataset": "chestxray14"})

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Jeu de données indisponible", html)
        self.assertIn("./venv/bin/python demo/app.py", html)


if __name__ == "__main__":
    unittest.main()
