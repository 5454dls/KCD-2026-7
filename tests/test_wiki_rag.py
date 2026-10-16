import json
import tempfile
import unittest
from pathlib import Path


PAGE = """---
type: concept
title: Hypoxic hepatitis
---
# Hypoxic hepatitis

Massive transient aminotransferase rise after cardiac or circulatory failure (HH review p.1).

## Diagnosis

Three criteria: clinical setting, aminotransferase rise, exclusion of other causes (HH review p.2).
"""


class WikiRagTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "wiki" / "concepts").mkdir(parents=True)
        (self.root / "wiki" / "concepts" / "hypoxic-hepatitis.md").write_text(PAGE, encoding="utf-8")
        (self.root / "wiki" / "concepts" / "dili.md").write_text("---\ntype: concept\ntitle: DILI\n---\n# DILI\n\nRUCAM causality, drug latency.\n", encoding="utf-8")

    def test_chunks_follow_headings_and_ids(self):
        from kcd_liver.wiki_rag import load_chunks
        ch = load_chunks(self.root)
        ids = [c["chunk_id"] for c in ch]
        self.assertIn("concepts/hypoxic-hepatitis#0", ids)
        self.assertTrue(any(c["heading_path"] == "Hypoxic hepatitis > Diagnosis" for c in ch))

    def test_bm25_finds_and_citation_check(self):
        from kcd_liver.wiki_rag import WikiRetriever, build_index, wiki_citation_check
        m = build_index(self.root, with_embeddings=False)
        self.assertIsNone(m["embedding"])
        r = WikiRetriever(self.root / "index")
        top = r.search("shock liver aminotransferase circulatory failure", k=1)[0]
        self.assertEqual(top["page_id"], "concepts/hypoxic-hepatitis")
        ev = [{"id": top["chunk_id"]}]
        ans = {"candidates": [{"support": [{"wiki": top["chunk_id"]}, {"wiki": "made-up#9"}]}]}
        self.assertEqual(wiki_citation_check(ans, ev), {"wiki_citations": 2, "valid": 1, "invalid": ["made-up#9"]})

    def test_experiment_runs_with_mock_llm(self):
        from kcd_liver.wiki_rag import WikiRetriever, build_index
        from kcd_liver.wiki_experiment import run
        build_index(self.root, with_embeddings=False)
        r = WikiRetriever(self.root / "index")
        snap = {"labs": [{"analyte": "ALT", "ratio": 30, "value": 900, "measured_at": "2150-01-01T10:00:00"},
                         {"analyte": "ALP", "ratio": 1.0, "value": 100, "measured_at": "2150-01-01T10:00:00"}], "context": []}
        def mock(system, payload):
            ev = payload.get("guideline_evidence") or []
            sup = [{"analyte_or_label": "ALT", "measured_at": "2150-01-01T10:00:00", "reason": "ALT 30x"}]
            if ev:
                sup.append({"wiki": ev[0]["id"], "reason": "guideline"})
            return {"flags": {"hypoxic_hepatitis": {"present": 1, "support": sup}}, "candidates": [{"cause_code": "hypoxic_hepatitis", "support": sup}], "abstain": False}
        out = run([{"payload": snap, "hadm_id": 1}], {1: {"hypoxic_hepatitis"}}, retriever=r, call=mock, log=lambda *a: None)
        self.assertEqual(out["arms"]["B_wiki"]["wiki_citation_valid"], 1.0)
        self.assertEqual(out["arms"]["A_no_wiki"]["patient_citation_valid"], 1.0)
        self.assertIn("B_wiki", out["paired_vs_no_wiki"])


if __name__ == "__main__":
    unittest.main()
