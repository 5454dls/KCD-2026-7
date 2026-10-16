import unittest

from kcd_liver.evaluation import grounding_check
from kcd_liver.rule_agent import rule_agent1

CAUSES = {'hypoxic_hepatitis', 'congestive_hepatopathy', 'sepsis_cholestasis', 'dili', 'pn_associated',
          'biliary_obstruction_cholangitis', 'cholecystitis', 'hemolysis', 'viral_hepatitis', 'others', 'unknown'}


def lab(a, ratio, value=None, t="2150-01-01T10:00:00"):
    return {"analyte": a, "ratio": ratio, "value": value if value is not None else ratio, "measured_at": t}


class RuleAgentTests(unittest.TestCase):
    def test_shock_liver_ranks_hypoxic_first_and_is_grounded(self):
        snap = {"as_of": "2150-01-01T12:00:00",
                "labs": [lab("ALT", 25), lab("AST", 40), lab("ALP", 1.1), lab("TBIL", 1.3), lab("LACTATE", 3, 6.1)],
                "context": [{"kind": "vital", "label": "MAP", "value": "52", "measured_at": "2150-01-01T08:00:00"},
                            {"kind": "infusion_record", "label": "Norepinephrine", "value": "recorded", "measured_at": "2150-01-01T07:00:00"}]}
        out = rule_agent1(snap)
        self.assertEqual(out["candidates"][0]["cause_code"], "hypoxic_hepatitis")
        self.assertFalse(out["abstain"])
        g = grounding_check(out, snap)
        self.assertEqual(g["verified"], g["support_items"])

    def test_cholestatic_with_inflammation(self):
        snap = {"labs": [lab("ALT", 1.2), lab("ALP", 2.5), lab("TBIL", 3.1), lab("WBC", None, 19)], "context": []}
        out = rule_agent1(snap)
        codes = [c["cause_code"] for c in out["candidates"]]
        self.assertEqual(set(codes[:2]), {"sepsis_cholestasis", "biliary_obstruction_cholangitis"})

    def test_weak_evidence_abstains_and_schema_is_valid(self):
        out = rule_agent1({"labs": [lab("ALT", 2.2), lab("ALP", 1.0)], "context": []})
        self.assertTrue(out["abstain"])
        codes = [c["cause_code"] for c in out["candidates"]]
        self.assertLessEqual(len(codes), 3)
        self.assertEqual(len(codes), len(set(codes)))
        self.assertTrue(set(codes) <= CAUSES)

    def test_flags_are_independent_binary_and_can_overlap(self):
        snap = {"labs": [lab("ALT", 1.2), lab("ALP", 2.5), lab("TBIL", 3.1), lab("WBC", None, 19)], "context": []}
        f = rule_agent1(snap)["flags"]
        self.assertEqual(set(f), CAUSES - {"unknown"})
        self.assertTrue(all(v["present"] in (0, 1) for v in f.values()))
        self.assertEqual((f["sepsis_cholestasis"]["present"], f["biliary_obstruction_cholangitis"]["present"]), (1, 1))
        self.assertEqual(f["cholecystitis"]["present"], 0)
        self.assertFalse(f["cholecystitis"]["assessable"])

    def test_others_when_unexplained_hepatocellular(self):
        out = rule_agent1({"labs": [lab("ALT", 8), lab("ALP", 1.0)], "context": []})
        self.assertEqual(out["flags"]["others"]["present"], 1)

    def test_hemolysis_indirect_bilirubin_and_haptoglobin(self):
        snap = {"labs": [lab("TBIL", 3.0, 3.6), lab("DBIL", None, 0.4), lab("HAPTO", None, 10), lab("LDH", 2.5), lab("ALT", 1.0),
                         lab("ALP", 1.0)], "context": []}
        f = rule_agent1(snap)["flags"]
        self.assertEqual(f["hemolysis"]["present"], 1)

    def test_viral_serology_and_pn_exposure(self):
        from kcd_liver.rule_agent import exposures, serology_positive
        self.assertTrue(serology_positive("POSITIVE"))
        self.assertFalse(serology_positive("NON-REACTIVE"))
        self.assertFalse(serology_positive("NOT DETECTED"))
        self.assertFalse(serology_positive("NEGATIVE.  RECOMMENDED FOLLOW-UP FOR POSITIVE HCV ANTIBODY: HCV VIRAL LOAD."))
        self.assertTrue(serology_positive("POSITIVE*.  RECOMMENDED FOLLOW-UP FOR POSITIVE HCV ANTIBODY: HCV VIRAL LOAD."))
        self.assertFalse(serology_positive("NEG."))
        self.assertFalse(serology_positive("___"))
        ctx = [{"kind": "serology_record", "label": "Hepatitis B Surface Antigen", "value": "POSITIVE",
                "charttime": "2150-01-01T00:00:00", "available_at": "2150-01-01T02:00:00"},
               {"kind": "infusion_record", "label": "Parenteral Nutrition", "value": "recorded",
                "charttime": "2149-12-20T00:00:00", "available_at": "2149-12-20T00:00:00"}]
        ex = exposures(ctx, "2150-01-01T12:00:00")
        self.assertGreater(ex["pn_days"], 7)
        f = rule_agent1({"labs": [lab("ALT", 12), lab("ALP", 2.5), lab("TBIL", 2.4, 2.8)], "context": [], "exposures": ex})["flags"]
        self.assertEqual(f["viral_hepatitis"]["present"], 1)
        self.assertEqual(f["pn_associated"]["present"], 1)
        late = exposures(ctx, "2150-01-01T01:00:00")
        self.assertEqual(late["serology"], [])

    def test_drug_record_needs_administration(self):
        base = {"labs": [lab("ALT", 4), lab("ALP", 1.2)], "context": []}
        given = dict(base, context=[{"kind": "emar_record", "label": "CefTRIAXone", "value": "Administered", "measured_at": "x"}])
        not_given = dict(base, context=[{"kind": "emar_record", "label": "CefTRIAXone", "value": "Not Given", "measured_at": "x"}])
        self.assertIn("dili", [c["cause_code"] for c in rule_agent1(given)["candidates"]])
        self.assertNotIn("dili", [c["cause_code"] for c in rule_agent1(not_given)["candidates"]])


if __name__ == "__main__":
    unittest.main()


class DataErrorTests(unittest.TestCase):
    def test_text_value_classes(self):
        from kcd_liver.data_errors import classify_text_value, malformed
        self.assertEqual(classify_text_value("37..5"), "malformed_number")
        self.assertEqual(classify_text_value("<5*."), "censored")
        self.assertEqual(classify_text_value("___"), "deidentified")
        self.assertEqual(classify_text_value("NEG."), "text")
        self.assertEqual(malformed(37.5), "37..5")

    def test_guard_range_and_partner_discordance(self):
        import pandas as pd
        from kcd_liver.data_errors import guard_lab
        t = pd.Timestamp("2150-01-01 10:00")
        lab = pd.DataFrame({"stay_id": [1, 1, 1, 1], "analyte": ["ALT", "AST", "TBIL", "CREAT"],
                            "value": [4500.0, 40.0, 1.0, 88.4], "ratio": [100.0, 1.0, 1.0, 70.0],
                            "charttime": [t] * 4, "available_at": [t] * 4})
        kept, q = guard_lab(lab)
        self.assertEqual(set(q.analyte), {"ALT", "CREAT"})
        self.assertEqual(dict(zip(q.analyte, q.reason))["ALT"], "discordant_partner")


class CongestiveTests(unittest.TestCase):
    def test_ntprobnp_supports_congestive(self):
        snap = {"labs": [lab("TBIL", 2.5, 3.0), lab("ALP", 1.5), lab("ALT", 1.2), lab("NTPROBNP", None, 5200)], "context": [],
                "demographics": {"age": 70}}
        self.assertEqual(rule_agent1(snap)["flags"]["congestive_hepatopathy"]["present"], 1)
        snap["labs"][-1] = lab("NTPROBNP", None, 300)
        self.assertEqual(rule_agent1(snap)["flags"]["congestive_hepatopathy"]["present"], 0)
