# KCD 2026 간 감시 에이전트

MIMIC-IV ICU 환자의 ALT·ALP·총빌리루빈을 한 시간 단위로 재생하면서, 2×ULN 이상 사건이 생기면 원인 후보와 근거(Agent 1), 해소·사망·퇴원의 경쟁 경로(Agent 2), 아직 정상인 환자의 24시간 이상 발생 참고값(Agent 3)을 제시하는 연구용 시연입니다. 후향적 재생 시뮬레이션이며 임상 도구가 아닙니다.

- **시연 페이지**: GitHub Pages (`site/`). 저장소 Settings → Pages → Source: GitHub Actions 설정 후 `main`에 병합하면 배포됩니다.
- **검토 문서**: [docs/REVIEW.md](docs/REVIEW.md) (계획 검토, 수정 내역, 발표 전략, 오늘 밤 실행 순서)
- **실행 순서**: [docs/RUNBOOK.md](docs/RUNBOOK.md) (전체 MIMIC-IV 모델 개발 → 결과 표 → 사이트·발표 자료 교체)
- **한 번에 실행**: `notebooks/KCD_RUN_ALL.ipynb` 하나를 기관 PC에서 끝까지 실행하면 돌려줄 `kcd_return_package_*.zip`이 만들어집니다 (모델·환자 단위 파일 제외)

## 현재 상태

- 시연 페이지의 환자 재생과 그림은 **MIMIC-IV Demo 100명 실데이터**로 노트북을 실제 실행한 결과입니다.
- Agent 1은 기관 LLM이 연결되기 전까지 **규칙 기반 기준 모델(`kcd_liver/rule_agent.py`, LLM 아님)**로 동작하며, 화면에도 그렇게 표시됩니다.
- 전문의 참조 판정 기반 성능 수치와 전체 MIMIC-IV 결과는 아직 없습니다. 자세한 내용은 [docs/REVIEW.md](docs/REVIEW.md) 0절.

데모 데이터 재생성:

```bash
python scripts/fhir_demo_to_csv.py --out data/mimic-iv-demo-csv   # PhysioNet 공개 S3에서 FHIR 판 내려받아 변환
# 노트북: ROOT=data/mimic-iv-demo-csv, DATA_SOURCE='mimic-iv-demo' 로 실행
```

## 구성

| 경로 | 내용 |
|---|---|
| `notebooks/KCD_MIMICIV_liver_agents_v2.ipynb` | 분석 노트북 (코호트, 사건, 세 에이전트, 평가, replay.sqlite, 마지막 셀에서 그림과 사이트 데이터 생성) |
| `kcd_liver/evaluation.py` | 근거 검증, 짝지은 McNemar, kappa, 입력 오류 채택률, 반복 안정성 |
| `kcd_liver/site_aggregates.py` | 집단 수준 집계(그림용), Aalen-Johansen, 작은 칸 가림 |
| `kcd_liver/figures.py` | 노트북용 논문 형식 Figure 2–10 |
| `scripts/export_site_data.py` | 노트북 출력 → `site/data/*.json` (데이터 출처 검사 포함) |
| `scripts/check_public_data.py` | 공개 전 자격 인증 데이터 노출 검사 (CI) |
| `scripts/make_synthetic_site_data.py` | 실제 실행 전 화면 확인용 합성 데이터 |
| `scripts/fhir_demo_to_csv.py` | MIMIC-IV Demo FHIR 판 → 노트북 CSV 형식 변환 |
| `kcd_liver/rule_agent.py` | Agent 1 규칙 기반 기준 모델 (LLM 비교군) |
| `kcd_liver/develop.py`, `scripts/develop_models.py` | 전체 MIMIC-IV 모델 개발·평가, Table 1·2, 전과 후 실험 결과 |
| `reference/mimic_iv_demo_subject_ids.csv` | 학습에서 제외하는 공개 데모 환자 100명 |
| `site/index.html` | 정적 시연 페이지 (외부 라이브러리 없음, 오프라인 실행 가능) |

## 실행

```bash
pip install pandas numpy scikit-learn matplotlib ipywidgets m4-infra
m4 init mimic-iv-demo                     # 공개 데모 데이터
# 노트북: ROOT=데모 폴더, DATA_SOURCE='mimic-iv-demo' 로 실행 → 마지막 셀이 site/data 갱신
python scripts/check_public_data.py       # 커밋 전 확인
python -m http.server -d site 8000        # 로컬 미리보기
python -m unittest discover -s tests -t . # 테스트
```

## 데이터 사용 원칙

- 공개 페이지의 환자 단위 재생은 **MIMIC-IV Clinical Database Demo (ODbL v1.0)** 또는 합성 데이터만 사용합니다.
- 전체 MIMIC-IV(PhysioNet 자격 인증)는 10 미만 칸을 가린 집계만 공개하며, `kcd_liver_v2_outputs/`와 `*.sqlite`는 커밋하지 않습니다.
- 전체 MIMIC-IV를 LLM에 보낼 때는 PhysioNet의 LLM 사용 지침(데이터 무보존, 학습 미사용, 사람 검토 없음 확인 또는 로컬 모델)을 따릅니다.

데이터 계층은 [M4](https://github.com/hannesill/m4)와 MIT-LCP mimic-code 정의를 참고했습니다.
