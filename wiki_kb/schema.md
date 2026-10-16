# Wiki Schema: ICU 간 검사 이상 원인 근거

## Page Types

| Type | Directory | Purpose |
|------|-----------|---------|
| entity | wiki/entities/ | 약물, 검사, 점수 체계 같은 이름 있는 대상 |
| concept | wiki/concepts/ | 질환 개념, 진단 기준, 평가 절차 |
| source | wiki/sources/ | 원문 한 편의 요약 |
| query | wiki/queries/ | 원문 사이 모순이나 근거 공백으로 남은 질문 |
| comparison | wiki/comparisons/ | 감별 비교 |
| synthesis | wiki/synthesis/ | 원인 10개별 근거 지도 |
| overview | wiki/ | 전체 요약 (overview.md) |

## Naming
- Files: `kebab-case.md`
- Sources: `author-year-slug.md`

## Frontmatter
모든 페이지:
```yaml
---
type: entity | concept | source | query | comparison | synthesis | overview
title: 제목
tags: []
related: []
sources: ["원문 파일명.pdf"]
created: 2026-10-16
updated: 2026-10-16
---
```
Source 페이지는 `authors`, `year`, `url`, `venue`를 추가합니다.

## Cross-references
- `[[page-slug]]` 로 연결합니다. 모든 entity·concept는 `wiki/index.md`에 올립니다.
- 본문의 근거는 `(원문 약칭 p.쪽)` 형식으로 적습니다. 예: `(ACG 2017 p.5)`

## Contradictions and gaps
원문끼리 다르면 concept 페이지에 적고 `wiki/queries/`에 질문 페이지를 만듭니다. 원문에 근거가 없으면 '근거 공백'으로 적고 synthesis에서 모읍니다.

## Page references
`p.N`은 PDF 파일의 N번째 쪽(인쇄된 쪽 번호가 아님)입니다.

## Style
- 원문 문장은 길게 옮기지 않고 요약합니다(인용은 한 문장 이내).
- 줄표(—) 문자는 쓰지 않습니다. index 항목은 `- [[slug]] : 설명` 형식입니다.
