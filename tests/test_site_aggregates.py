import unittest

import pandas as pd

from kcd_liver.site_aggregates import aalen_johansen, auroc, build_site_aggregates, suppress


class SiteAggregateTests(unittest.TestCase):
    def test_aalen_johansen_matches_notebook_competing_cif(self):
        # Notebook self-check: two patients, both events at t=1 -> 0.5 / 0.5
        r = aalen_johansen([1, 1], ["resolution", "death"], horizon=4)
        self.assertAlmostEqual(r["cif"]["resolution"][-1], .5)
        self.assertAlmostEqual(r["cif"]["death"][-1], .5)

    def test_naive_censoring_overestimates_resolution(self):
        times = [2, 3, 5, 8, 10, 48]
        events = ["death", "hospital_discharge", "resolution", "resolution", "death", "censor"]
        r = aalen_johansen(times, events, horizon=48)
        self.assertGreater(r["naive_resolution"][-1], r["cif"]["resolution"][-1])
        total = sum(r["cif"][c][-1] for c in r["cif"])
        self.assertLessEqual(total, 1 + 1e-9)

    def test_auroc(self):
        self.assertEqual(auroc([0, 0, 1, 1], [.1, .2, .8, .9]), 1.0)
        self.assertEqual(auroc([0, 1, 0, 1], [.5, .5, .5, .5]), .5)
        self.assertIsNone(auroc([1, 1], [.2, .3]))

    def test_suppression_removes_small_counts_and_rates(self):
        out = suppress({"rows": [{"n": 4, "rate": .5}, {"n": 40, "rate": .2}], "at_risk": [30, 5]}, 10)
        self.assertIsNone(out["rows"][0]["n"])
        self.assertIsNone(out["rows"][0]["rate"])
        self.assertEqual(out["rows"][1]["n"], 40)
        self.assertEqual(out["at_risk"], [30, None])

    def test_build_has_no_patient_level_rows(self):
        events = pd.DataFrame({"case_id": ["a", "b", "c"], "split": ["development"] * 3,
                               "group": ["incident", "initial_abnormal", "incident"],
                               "trigger_analyte": ["ALT", "TBIL", "ALT"], "trigger_ratio": [2.5, 7, 30],
                               "subject_id": [1, 2, 3]})
        agg = build_site_aggregates(source="mimic-iv", events=events, min_cell=None)
        text = str(agg)
        self.assertNotIn("subject_id", text)
        self.assertEqual(sum(r["n"] for r in agg["event_groups"]), 3)


if __name__ == "__main__":
    unittest.main()


class DevelopV2Tests(unittest.TestCase):
    def test_noncrossing_and_group_shapley_efficiency(self):
        import numpy as np
        import pandas as pd
        from kcd_liver.develop import _noncrossing, group_shapley
        P = _noncrossing({0.1: np.array([2.0, 0.0]), 0.5: np.array([1.0, 0.5]), 0.9: np.array([3.0, 1.0])})
        self.assertTrue((P[0.1] <= P[0.5]).all() and (P[0.5] <= P[0.9]).all())
        cols = ["ALT_latest", "MAP_latest", "age"]
        f = lambda A: 0.1 * A[:, 0] + 0.2 * A[:, 1] + 0.05 * A[:, 0] * A[:, 1]
        X = pd.DataFrame([[3.0, 2.0, 70.0]], columns=cols)
        B = pd.DataFrame([[1.0, 1.0, 50.0], [0.0, 0.0, 60.0]], columns=cols)
        phi, groups = group_shapley(f, X, B, cols)
        total = f(X.to_numpy())[0] - f(B.to_numpy()).mean()
        self.assertAlmostEqual(phi.sum(), total, places=9)  # efficiency: contributions add up exactly
        self.assertAlmostEqual(phi[0][groups.index("background")], 0.0, places=12)  # unused feature gets zero
