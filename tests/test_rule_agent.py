import unittest

from kcd_liver.evaluation import grounding_check
from kcd_liver.rule_agent import rule_agent1

CAUSES = {'hypoxic_ischemic', 'sepsis_cholestasis', 'drug_related', 'biliary', 'congestive', 'baseline_liver',
          'other_hepatic', 'nonhepatic', 'unknown'}


def lab(a, ratio, value=None, t="2150-01-01T10:00:00"):
    return {"analyte": a, "ratio": ratio, "value": value if value is not None else ratio, "measured_at": t}


class RuleAgentTests(unittest.TestCase):
    def test_shock_liver_ranks_hypoxic_first_and_is_grounded(self):
        snap = {"as_of": "2150-01-01T12:00:00",
                "labs": [lab("ALT", 25), lab("AST", 40), lab("ALP", 1.1), lab("TBIL", 1.3), lab("LACTATE", 3, 6.1)],
                "context": [{"kind": "vital", "label": "MAP", "value": "52", "measured_at": "2150-01-01T08:00:00"},
                            {"kind": "infusion_record", "label": "Norepinephrine", "value": "recorded", "measured_at": "2150-01-01T07:00:00"}]}
        out = rule_agent1(snap)
        self.assertEqual(out["candidates"][0]["cause_code"], "hypoxic_ischemic")
        self.assertFalse(out["abstain"])
        g = grounding_check(out, snap)
        self.assertEqual(g["verified"], g["support_items"])

    def test_cholestatic_with_inflammation(self):
        snap = {"labs": [lab("ALT", 1.2), lab("ALP", 2.5), lab("TBIL", 3.1), lab("WBC", None, 19)], "context": []}
        out = rule_agent1(snap)
        codes = [c["cause_code"] for c in out["candidates"]]
        self.assertEqual(codes[0], "sepsis_cholestasis")
        self.assertIn("biliary", codes)

    def test_weak_evidence_abstains_and_schema_is_valid(self):
        out = rule_agent1({"labs": [lab("ALT", 2.2), lab("ALP", 1.0)], "context": []})
        self.assertTrue(out["abstain"])
        codes = [c["cause_code"] for c in out["candidates"]]
        self.assertLessEqual(len(codes), 3)
        self.assertEqual(len(codes), len(set(codes)))
        self.assertTrue(set(codes) <= CAUSES)

    def test_drug_record_needs_administration(self):
        base = {"labs": [lab("ALT", 4), lab("ALP", 1.2)], "context": []}
        given = dict(base, context=[{"kind": "emar_record", "label": "CefTRIAXone", "value": "Administered", "measured_at": "x"}])
        not_given = dict(base, context=[{"kind": "emar_record", "label": "CefTRIAXone", "value": "Not Given", "measured_at": "x"}])
        self.assertIn("drug_related", [c["cause_code"] for c in rule_agent1(given)["candidates"]])
        self.assertNotIn("drug_related", [c["cause_code"] for c in rule_agent1(not_given)["candidates"]])


if __name__ == "__main__":
    unittest.main()
