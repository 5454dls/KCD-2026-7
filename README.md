# KCD 2026 ICU 간기능 이상 감시 에이전트

MIMIC-IV ICU 환자의 ALT·ALP·총빌리루빈을 한 시간 단위로 재생하면서 네 가지를 내는 후향적 연구 시스템입니다. 임상 도구가 아닙니다.

| 구성 | 실행 시점 | 출력 |
|---|---|---|
| Agent 3 조기 예측 | 정상일 때 매시간 | 24시간 안 ≥5×ULN 위험 |
| 궤적 예측 | 매시간 | ALT·AST·ALP·TBIL 6-24시간 뒤 10·50·90% 구간 |
| Agent 1 원인 | 첫 이상(≥2×ULN) 시 1회 | 원인 10개 각각 0/1과 근거 |
| Agent 2 예후 | 첫 이상 시 1회 | ICU 퇴실 시 회복 여부 × ICU 사망(퇴실 후 24시간 포함) |

## 실행

```bash
pip install pandas numpy scikit-learn matplotlib joblib nbformat nbclient ipykernel
```

`notebooks/KCD_ONE.ipynb` 하나(코드 내장)를 열고 첫 셀의 `MIMIC_ROOT`를 확인한 뒤 끝까지 실행합니다. 노트북 옆에 결과 묶음(`kcd_return_package_*.zip`)과 모델 묶음(`kcd_model_package_*.zip`)이 생깁니다. 자세한 순서는 [docs/RUNBOOK.md](docs/RUNBOOK.md).

```bash
python -m pytest -q tests                 # 테스트
python scripts/check_public_data.py       # 공개 전 자료 검사
python -m http.server -d site 8000        # 사이트 미리보기
python tools/make_one_notebook.py         # 코드 수정 후 KCD_ONE.ipynb 재생성
```

## 구성

| 경로 | 내용 |
|---|---|
| `notebooks/KCD_MIMICIV_liver_agents_v2.ipynb` | 코호트, 검사 추출·품질 관리, 사건, 원인 규칙, 재생 기록 |
| `notebooks/KCD_ONE.ipynb`, `KCD_RUN_ALL.ipynb` | 전체 실행기 |
| `notebooks/KCD_EXTERNAL.ipynb` | 다른 병원 외부 검증 (모델 묶음 안에서 실행) |
| `notebooks/KCD_CLUSTER.ipynb` | 개선된 클러스터링 (KCD_ONE 실행 뒤 같은 폴더에서) |
| `kcd_liver/develop.py` | 모델 개발·평가, Table 1·2, 그림자 실험, 작은 칸 가림 |
| `kcd_liver/rule_agent.py` | Agent 1 규칙 모델 |
| `kcd_liver/data_errors.py` | 입력 오류 가드와 오류 주입 실험 |
| `kcd_liver/wiki_rag.py`, `wiki_experiment.py` | 가이드라인 위키 검색, 위키 전후 비교 실험 |
| `kcd_liver/external.py` | 고정 모델 외부 검증 |
| `scripts/` | 명령행 도구 (모델 개발, 사이트 내보내기, 공개 검사, 위키 색인) |
| `site/` | 정적 시연 페이지와 공개 데이터 |
| `wiki_kb/` | 가이드라인 위키 (llm_wiki 형식, 원문 13편) |
| `docs/` | 변수 설명, 실행 순서, 명세서, 결과 보고서 |
| `reference/mimic_iv_demo_subject_ids.csv` | 학습에서 제외한 공개 데모 환자 100명 |

## 데이터 원칙

- 공개 사이트의 환자 단위 재생은 MIMIC-IV Clinical Database Demo(ODbL v1.0)만 씁니다.
- 전체 MIMIC-IV는 10 미만 칸을 가린 집계만 공개합니다. 환자 단위 파일, 모델 학습 중간 산출물, `*.sqlite`는 커밋하지 않습니다.
- 전체 MIMIC-IV를 LLM에 보낼 때는 PhysioNet의 LLM 사용 지침(무보존·학습 미사용 엔드포인트 또는 로컬 모델)을 따릅니다.
