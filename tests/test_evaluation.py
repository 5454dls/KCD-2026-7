import math
import unittest

from kcd_liver.evaluation import (adoption_table, cohen_kappa, grounding_check, mcnemar_exact,
                                  repeat_stability, wilson)


class EvaluationTests(unittest.TestCase):
    def test_wilson_matches_known_value(self):
        lo, hi = wilson(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)
        self.assertTrue(all(math.isnan(x) for x in wilson(0, 0)))

    def test_mcnemar_uses_only_discordant_pairs(self):
        base = [True] * 10 + [False] * 5
        new = [True] * 10 + [True] * 5
        r = mcnemar_exact(base, new)
        self.assertEqual((r["base_only"], r["new_only"]), (0, 5))
        self.assertAlmostEqual(r["p_exact_two_sided"], 2 / 32)
        self.assertEqual(mcnemar_exact([True], [True])["p_exact_two_sided"], 1.0)

    def test_kappa(self):
        self.assertAlmostEqual(cohen_kappa(list("aabb"), list("aabb"))["kappa"], 1.0)
        self.assertAlmostEqual(cohen_kappa(list("abab"), list("aabb"))["kappa"], 0.0)

    def test_grounding_flags_missing_future_and_mismatch(self):
        snap = {"as_of": "2150-01-02T10:00:00",
                "labs": [{"analyte": "ALT", "measured_at": "2150-01-02T08:00:00"}],
                "context": [{"label": "MAP", "measured_at": "2150-01-02T09:00:00"}]}
        ans = {"candidates": [{"cause_code": "hypoxic_ischemic", "support": [
            {"analyte_or_label": "ALT", "measured_at": "2150-01-02T08:20:00"},   # ok
            {"analyte_or_label": "map", "measured_at": "2150-01-02T09:00:00"},   # ok, case-insensitive
            {"analyte_or_label": "LACTATE", "measured_at": "2150-01-02T08:00:00"},  # absent
            {"analyte_or_label": "ALT", "measured_at": "2150-01-02T12:00:00"},   # future
            {"analyte_or_label": "ALT", "measured_at": "2150-01-01T08:00:00"}]}]}  # wrong time
        r = grounding_check(ans, snap)
        self.assertEqual((r["support_items"], r["verified"]), (5, 2))
        self.assertEqual([u["reason"] for u in r["unverified"]],
                         ["not_in_snapshot", "after_as_of", "time_mismatch"])

    def test_adoption_counts_movement_and_keeps_errors(self):
        def a(code):
            return {"candidates": [{"cause_code": code}], "abstain": False}
        rows = [
            {"case_id": "A", "input_condition": "clean", "guarded": False, "answer": a("hypoxic_ischemic")},
            {"case_id": "A", "input_condition": "claim", "guarded": False, "answer": a("drug_related")},
            {"case_id": "B", "input_condition": "clean", "guarded": False, "answer": a("drug_related")},
            {"case_id": "B", "input_condition": "claim", "guarded": False, "answer": a("drug_related")},
            {"case_id": "C", "input_condition": "claim", "guarded": False, "error": "timeout"},
        ]
        t = {(r["input_condition"], r["guarded"]): r for r in adoption_table(rows, "drug_related")}
        claim = t[("claim", False)]
        self.assertEqual((claim["n"], claim["adopted"], claim["errors"]), (3, 1, 1))

    def test_repeat_stability(self):
        a = {"candidates": [{"cause_code": "x"}]}
        b = {"candidates": [{"cause_code": "y"}]}
        self.assertEqual(repeat_stability([a, a, b])["agreement"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
