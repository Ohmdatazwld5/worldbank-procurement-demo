from __future__ import annotations

import unittest
from pathlib import Path

from worldbank_demo_backend.api import app


class WebUiTests(unittest.TestCase):
    def test_home_route_serves_local_q_and_a_page(self) -> None:
        route = next(route for route in app.routes if getattr(route, "path", None) == "/")
        response = route.endpoint()
        self.assertEqual(response.media_type, "text/html")
        self.assertTrue(Path(response.path).is_file())
        content = Path(response.path).read_text(encoding="utf-8")
        self.assertIn('id="query-form"', content)
        self.assertIn('fetch("/v1/query"', content)
        self.assertIn("fictional test fixtures", content)


if __name__ == "__main__":
    unittest.main()
