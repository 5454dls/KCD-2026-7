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
    f["LACTATE_latest"] = rng.lognormal(np.array([1.5, .5, .7])[g], .4)
    f["age"] = rng.normal(60, 12, n)
    f["history_liver"] = (rng.random(n) < np.array([.05, .1, .4])[g]).astype(float)
    o = pd.DataFrame({"case_id": f.case_id, "died_icu": died, "observed_path": [r + ("died_icu" if d else "alive_icu") for r, d in zip(rec, died)]})
    return f, o


class ClusterReportTest(unittest.TestCase):
    def test_run_and_render(self):
        f, o = synthetic()
        extra = pd.DataFrame({"case_id": f.case_id, "group": np.where(f.index % 2, "incident", "initial_abnormal"),
                              "flag_hypoxic_hepatitis": (f.ALT_ratio > 8).astype(float), "icd_hypoxic_hepatitis": (f.ALT_ratio > 12).astype(float)})
        R = cr.run(f, o, extra=extra, n_boot=3, ks=range(2, 5), log=lambda *_: None)
        self.assertTrue(any(r["variable"] == "LACTATE_latest" for r in R["characterization"]["variables"]))
        self.assertEqual(len(R["profiles"]), R["selected"]["k"])
        self.assertIn("R_ratio", cr.PROFILE)
        self.assertNotIn("R_ratio", R["features"])
        self.assertIsNotNone(R["added_value"]["icu_death"])
        self.assertIn("cluster_added_to_severity", R["added_value"]["icu_death"])
        self.assertEqual({r["source"] for r in R["causes"]["rows"]}, {"rule", "icd"})

    def test_zero_values_still_logged(self):
        f, o = synthetic()
        f.loc[0, "ALT_ratio"] = 0.0
        R = cr.run(f, o, n_boot=2, ks=range(2, 4), log=lambda *_: None)
        self.assertLess(max(e["mean"][0] for e in R["ellipses_alt_alp"]), 10)
        for r in R["outcomes"]["evaluation"]["by_cluster"]:
            v = r["icu_death"]
            self.assertTrue(v["k"] is None or v["k"] >= 10)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(cr.render(R, d)), 7)


if __name__ == "__main__":
    unittest.main()
