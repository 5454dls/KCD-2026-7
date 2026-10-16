# ICU 간 검사 이상 원인 근거 위키 (llm_wiki 프로젝트)

원문 4편(ACG 2017, AASLD DILI 2023, 저산소성 간염 리뷰 2016, SSC 2026)으로 만든 43쪽 위키입니다. 모든 문장에 원문과 PDF 쪽수가 붙어 있고, 원문에 없는 내용은 '근거 공백'으로 적었습니다.

## 구조 (llm_wiki 규격)
- `purpose.md`, `schema.md`: 목적과 작성 규칙
- `raw/sources/*.pdf`: 원문 (불변)
- `wiki/`: sources 4, concepts 17, entities 16, comparisons 1, queries 1, synthesis 1, overview·index·log
- `index/`: 검색용 청크(llm_wiki와 같은 방식: 제목 경계, 1,000자, 200자 겹침)와 manifest

시작 페이지: `wiki/overview.md`, `wiki/synthesis/icu-liver-cause-evidence-map.md`

## llm_wiki 앱에서 열기
1. llm_wiki에서 이 폴더(`wiki_kb`)를 프로젝트로 엽니다(또는 프로젝트를 만들고 이 폴더 내용을 복사).
2. Settings에서 벡터 검색(Embedding)을 켜고 OpenAI 호환 임베딩 엔드포인트를 지정합니다(예: 로컬 Ollama `http://localhost:11434/v1`, 모델 `bge-m3`).
3. 앱이 페이지를 임베딩해 `.llm-wiki/lancedb`에 저장합니다. 원문 PDF를 다시 인제스트할 필요는 없습니다(이미 위키 페이지가 있음).
4. 답변을 "Save to Wiki" 하지 마세요. 환자 정보가 위키에 들어갈 수 있습니다.

## 파이프라인에서 쓰기 (앱 없이)
```
python scripts/build_wiki_index.py --wiki wiki_kb                       # 키워드(BM25)만
EMBED_BASE_URL=http://localhost:11434/v1 EMBED_MODEL=bge-m3 python scripts/build_wiki_index.py --wiki wiki_kb   # 임베딩 포함
```
질의도 같은 임베딩 모델로 만들어야 합니다(manifest에 모델명이 기록되고 다르면 오류).

## 전후 비교 실험
```
LLM_BASE_URL=... LLM_MODEL=... python scripts/wiki_experiment.py --out-dir <분석 결과 폴더> --wiki wiki_kb --mimic-root <MIMIC 폴더> --n 200
```
같은 사례·같은 모델로 A 위키 없음, B 위키, C 무관한 위키 조각(위약), D 규칙 모델, E 거짓 문장 하나를 심은 위키를 비교합니다. 전체 MIMIC-IV 스냅샷은 승인된 무보존 엔드포인트에만 보냅니다.

## 원문 간 불일치
`wiki/queries/threshold-disagreements.md`: R 비 경계(>5 대 ≥5), 저산소성 간염 기준(20× 대 2.5-10× 대 15×), Hy's law 빌리루빈(>2.5 mg/dl), DILI 기준과 사건 정의 차이.
