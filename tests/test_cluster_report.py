import tempfile
import unittest

import numpy as np
import pandas as pd

from kcd_liver import cluster_report as cr


def synthetic(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    centers = np.array([[4, 3.5, .5, .5], [1, .5, 2, 1], [.2, .5, .2, 2.5]])
    X = 2 ** (centers[g] + rng.normal(0, .35, (n, 4)))
    f = pd.DataFrame(X, columns=cr.FEATS)
    f["R_ratio"] = f.ALT_ratio / f.ALP_ratio
    f["case_id"] = [f"C{i}" for i in range(n)]
    f["split"] = np.where(rng.random(n) < .7, "development", "evaluation")
    died = rng.random(n) < np.array([.4, .15, .25])[g]
    rec = np.where(rng.random(n) < .2, "unassessable_", np.where(rng.random(n) < np.array([.1, .4, .2])[g], "recovered_", "unrecovered_"))
    o = pd.DataFrame({"case_id": f.case_id, "died_icu": died, "observed_path": [r + ("died_icu" if d else "alive_icu") for r, d in zip(rec, died)]})
    return f, o


class ClusterReportTest(unittest.TestCase):
    def test_run_and_render(self):
        f, o = synthetic()
        R = cr.run(f, o, n_boot=3, ks=range(2, 5), log=lambda *_: None)
        self.assertEqual(len(R["profiles"]), R["selected"]["k"])
        self.assertIn("R_ratio", cr.PROFILE)
        self.assertNotIn("R_ratio", R["features"])
        self.assertIsNotNone(R["added_value"]["icu_death"])
        for r in R["outcomes"]["evaluation"]["by_cluster"]:
            v = r["icu_death"]
            self.assertTrue(v["k"] is None or v["k"] >= 10)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(cr.render(R, d)), 5)


if __name__ == "__main__":
    unittest.main()
