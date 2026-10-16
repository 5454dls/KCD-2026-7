"""Build the retrieval index for the guideline wiki (llm_wiki project folder).

python scripts/build_wiki_index.py --wiki wiki_kb                 # BM25 only (no model needed)
EMBED_BASE_URL=http://localhost:11434/v1 EMBED_MODEL=bge-m3 python scripts/build_wiki_index.py --wiki wiki_kb
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kcd_liver.wiki_rag import build_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--wiki", type=Path, default=Path("wiki_kb"))
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()
print(json.dumps(build_index(a.wiki, a.out), ensure_ascii=False, indent=1))
