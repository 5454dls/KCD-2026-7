import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site" / "index.html"


class SiteScriptTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_inline_script_parses(self):
        scripts = re.findall(r"<script[^>]*>([\s\S]*?)</script>", SITE.read_text(encoding="utf-8"))
        for i, js in enumerate(scripts):
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
                f.write(js)
            r = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"script {i}: {r.stderr[:500]}")


if __name__ == "__main__":
    unittest.main()
