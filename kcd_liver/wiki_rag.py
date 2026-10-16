"""Guideline wiki retrieval for Agent 1 (llm_wiki-compatible project folder)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

CHUNK_CHARS, OVERLAP_CHARS = 1000, 200
META_HEADINGS = re.compile(r"근거 공백|검색 방법|이 페이지의 범위|evidence gap", re.I)
META_WEIGHT = 0.3
_FM = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _frontmatter(text: str) -> tuple[dict, str]:
    m = _FM.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def _split(text: str, size=CHUNK_CHARS, overlap=OVERLAP_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        cut = text.rfind("\n", start + size // 2, end) if end < len(text) else end
        end = cut if cut > start else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def chunk_page(page_id: str, text: str) -> list[dict]:
    """Heading-aware chunks: a new section starts at every markdown heading (code fences respected)."""
    meta, body = _frontmatter(text)
    chunks, stack, section, fence = [], [], [], False

    def flush():
        t = "\n".join(section).strip()
        if t:
            path = " > ".join(h for _, h in stack)
            for piece in _split(t):
                chunks.append({"page_id": page_id, "chunk_index": len(chunks), "heading_path": path,
                               "chunk_text": piece, "title": meta.get("title", page_id), "type": meta.get("type", "")})
        section.clear()

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = [(l, h) for l, h in stack if l < level] + [(level, m.group(2).strip())]
        else:
            section.append(line)
    flush()
    for c in chunks:
        c["chunk_id"] = f"{page_id}#{c['chunk_index']}"
    return chunks


def load_chunks(root) -> list[dict]:
    root = Path(root)
    out = []
    for p in sorted((root / "wiki").rglob("*.md")):
        if p.name.startswith("_frag_") or p.name == "log.md":
            continue
        out += chunk_page(str(p.relative_to(root / "wiki").with_suffix("")).replace("\\", "/"), p.read_text(encoding="utf-8"))
    return out


def _tokens(s: str) -> list[str]:
    s = s.lower()
    words = re.findall(r"[a-z][a-z0-9\-]+|\d+(?:\.\d+)?", s)
    hangul = re.findall(r"[가-힣]+", s)
    grams = [h[i:i + 2] for h in hangul for i in range(max(1, len(h) - 1))]
    return words + grams


class BM25:
    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.toks = [_tokens(d) for d in docs]
        self.n, self.k1, self.b = len(docs), k1, b
        self.avg = sum(map(len, self.toks)) / max(1, self.n)
        df = Counter(t for d in self.toks for t in set(d))
        self.idf = {t: math.log(1 + (self.n - f + .5) / (f + .5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.toks]

    def scores(self, query: str) -> list[float]:
        q = _tokens(query)
        out = []
        for tf, d in zip(self.tf, self.toks):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out


def embed(texts: list[str], base_url=None, model=None, api_key=None, batch=32) -> list[list[float]]:
    import urllib.request
    base_url = (base_url or os.environ.get("EMBED_BASE_URL", "")).rstrip("/")
    model = model or os.environ.get("EMBED_MODEL", "")
    api_key = api_key or os.environ.get("EMBED_API_KEY", "")
    if not base_url or not model:
        raise RuntimeError("Set EMBED_BASE_URL and EMBED_MODEL (OpenAI-compatible /v1/embeddings)")
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": model, "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(base_url + "/embeddings", data=body, method="POST",
                                     headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + api_key} if api_key else {})})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)["data"]
        out += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
    return out


def wiki_hash(root) -> str:
    h = hashlib.sha256()
    for p in sorted((Path(root) / "wiki").rglob("*.md")):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def build_index(root, out_dir=None, with_embeddings=None) -> dict:
    """Write chunks.jsonl (+ embeddings.json when an endpoint is set) and index_manifest.json."""
    root = Path(root)
    out = Path(out_dir or root / "index")
    out.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(root)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    manifest = {"wiki_sha256": wiki_hash(root), "chunks": len(chunks), "pages": len({c["page_id"] for c in chunks}),
                "chunking": {"chars": CHUNK_CHARS, "overlap": OVERLAP_CHARS, "heading_aware": True},
                "lexical": "BM25 (k1=1.5, b=0.75), Korean bigrams", "embedding": None}
    want = with_embeddings if with_embeddings is not None else bool(os.environ.get("EMBED_BASE_URL"))
    if want:
        vecs = embed([f"{c['title']} | {c['heading_path']}\n{c['chunk_text']}" for c in chunks])
        (out / "embeddings.json").write_text(json.dumps(vecs), encoding="utf-8")
        manifest["embedding"] = {"model": os.environ.get("EMBED_MODEL"), "dim": len(vecs[0]) if vecs else None,
                                 "endpoint_kind": "openai-compatible /v1/embeddings"}
    (out / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


class WikiRetriever:
    """Hybrid retrieval: BM25 always; cosine over stored embeddings when present and an endpoint is configured."""

    def __init__(self, index_dir):
        d = Path(index_dir)
        self.chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.manifest = json.loads((d / "index_manifest.json").read_text(encoding="utf-8"))
        self.bm25 = BM25([f"{c['title']} {c['heading_path']} {c['chunk_text']}" for c in self.chunks])
        ep = d / "embeddings.json"
        self.vecs = json.loads(ep.read_text()) if ep.exists() and os.environ.get("EMBED_BASE_URL") else None
        if self.vecs and self.manifest.get("embedding", {}).get("model") != os.environ.get("EMBED_MODEL"):
            raise RuntimeError("Query embedding model differs from the index model; rebuild the index or set EMBED_MODEL")
        self.mode = "hybrid" if self.vecs else "bm25"

    def search(self, query: str, k=5, exclude_pages=()) -> list[dict]:
        lex = self.bm25.scores(query)
        top = max(lex) or 1.0
        score = [s / top for s in lex]
        if self.vecs:
            q = embed([query])[0]
            qn = math.sqrt(sum(x * x for x in q)) or 1
            cos = [sum(a * b for a, b in zip(q, v)) / (qn * (math.sqrt(sum(x * x for x in v)) or 1)) for v in self.vecs]
            score = [.5 * s + .5 * c for s, c in zip(score, cos)]
        score = [sc * (META_WEIGHT if META_HEADINGS.search(self.chunks[i]["heading_path"]) else 1) for i, sc in enumerate(score)]
        order = sorted(range(len(self.chunks)), key=lambda i: -score[i])
        out = []
        for i in order:
            c = self.chunks[i]
            if c["page_id"] in exclude_pages or score[i] <= 0:
                continue
            out.append({**c, "score": round(float(score[i]), 4)})
            if len(out) >= k:
                break
        return out


CAUSE_QUERIES = {
    "hypoxic_hepatitis": "hypoxic hepatitis shock liver ischemic hepatitis aminotransferase rise hypotension cardiac failure",
    "congestive_hepatopathy": "congestive hepatopathy heart failure right heart congestion liver tests",
    "sepsis_cholestasis": "sepsis cholestasis bilirubin organ dysfunction septic shock liver",
    "dili": "drug induced liver injury causality RUCAM latency R ratio implicated drug",
    "pn_associated": "parenteral nutrition liver dysfunction cholestasis enteral nutrition",
    "biliary_obstruction_cholangitis": "biliary obstruction cholangitis alkaline phosphatase imaging duct dilation source control",
    "cholecystitis": "acute cholecystitis gallbladder source control imaging",
    "hemolysis": "hemolysis unconjugated hyperbilirubinemia isolated bilirubin",
    "viral_hepatitis": "viral hepatitis serology HBsAg HCV antibody HAV IgM HEV testing",
    "others": "hepatocellular injury evaluation autoimmune alcohol fatty liver muscle injury",
}


def snapshot_queries(payload: dict, rule_answer: dict | None = None, max_q=4) -> list[str]:
    """Queries from the as-of snapshot only (pattern, drugs, flagged causes). No identifiers are included."""
    q = []
    labs = {r.get("analyte"): r for r in payload.get("labs", [])}
    rat = lambda a: (labs.get(a) or {}).get("ratio")
    alt, alp, tb = rat("ALT"), rat("ALP"), rat("TBIL")
    if alt and alp:
        r = alt / alp
        q.append(("hepatocellular" if r >= 5 else "cholestatic" if r <= 2 else "mixed") + " pattern liver injury evaluation R ratio")
    elif tb:
        q.append("isolated hyperbilirubinemia evaluation conjugated unconjugated")
    drugs = sorted({str(c.get("label")) for c in payload.get("context", []) if c.get("kind") == "emar_record"})[:3]
    if drugs:
        q.append("drug induced liver injury " + " ".join(drugs))
    flagged = [c for c, v in ((rule_answer or {}).get("flags") or {}).items() if v.get("present")]
    for c in flagged[:2]:
        q.append(CAUSE_QUERIES.get(c, c))
    if any(c.get("kind") == "infusion_record" and re.search("norepinephrine|vasopressin", str(c.get("label")), re.I) for c in payload.get("context", [])):
        q.append(CAUSE_QUERIES["hypoxic_hepatitis"])
    return q[:max_q] or [CAUSE_QUERIES["others"]]


def evidence_for(retriever: WikiRetriever, payload: dict, rule_answer=None, k_per_query=2, max_chunks=6, placebo=False, seed=0):
    """Guideline chunks for the LLM payload. placebo=True returns the same number of unrelated chunks (control arm)."""
    if placebo:
        import random
        rnd = random.Random(seed)
        pool = [c for c in retriever.chunks if c.get("type") not in ("source",)]
        pick = rnd.sample(pool, min(max_chunks, len(pool)))
        return [{"id": c["chunk_id"], "page": c["page_id"], "heading": c["heading_path"], "text": c["chunk_text"]} for c in pick]
    seen, out = set(), []
    for q in snapshot_queries(payload, rule_answer):
        for c in retriever.search(q, k=k_per_query):
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                out.append({"id": c["chunk_id"], "page": c["page_id"], "heading": c["heading_path"], "text": c["chunk_text"]})
    return out[:max_chunks]


def wiki_citation_check(answer: dict, evidence: list[dict]) -> dict:
    """A wiki citation is valid only if it names a chunk that was actually supplied in this payload."""
    given = {e["id"] for e in evidence}
    cited = []
    for cand in (answer.get("candidates") or []):
        cited += [s.get("wiki") for s in (cand.get("support") or []) if s.get("wiki")]
    for f in (answer.get("flags") or {}).values():
        cited += [s.get("wiki") for s in (f.get("support") or []) if isinstance(s, dict) and s.get("wiki")]
    valid = [c for c in cited if c in given]
    return {"wiki_citations": len(cited), "valid": len(valid), "invalid": sorted(set(cited) - given)}
