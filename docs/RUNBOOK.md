# 실행 순서

전체 MIMIC-IV는 기관 PC에서만 실행합니다.

## 1. 준비

```powershell
pip install pandas numpy scikit-learn matplotlib joblib nbformat nbclient ipykernel
```

## 2. 실행

`notebooks/KCD_ONE.ipynb`를 아무 폴더에 두고 엽니다. 첫 셀에서 확인할 것:

| 변수 | 값 |
|---|---|
| `MIMIC_ROOT` | `hosp/`, `icu/`가 있는 MIMIC-IV 3.1 폴더 |
| `MAX_PATIENTS` | `None`(전체). 시간이 부족하면 `20000` 같은 숫자 |
| `DEMO_FHIR_DIR` | 인터넷이 막힌 PC면 미리 받은 FHIR 데모의 `fhir` 폴더 |
| LLM (선택) | 환경변수 `LLM_BASE_URL`, `LLM_MODEL`. 없으면 Agent 1은 규칙 모델 |

위에서부터 끝까지 실행합니다. 단계별 진행이 출력됩니다. 코드와 중간 산출물은 노트북 옆 `kcd_work/`에 생기며 이 PC에만 남습니다.

| 단계 | 하는 일 |
|---|---|
| 1 | 전체 MIMIC-IV 분석 노트북 (코호트, 검사, 사건, 원인 규칙) |
| 2 | 모델 개발·평가 (공개 데모 환자 제외) |
| 3 | 공개 데모 100명에 학습된 모델을 적용해 시연 화면 생성 |
| 4 | 결과 묶음 `kcd_return_package_*.zip` (집계, 공개 데모, 모델 카드) |
| 5 | 모델 묶음 `kcd_model_package_*.zip` (모델, 외부 검증 노트북) |

한 단계가 실패해도 나머지와 압축은 진행되고, 실패 이유는 `run_report.json`에 적힙니다. 추출 항목이 바뀌지 않은 새 버전이면 `RUN_STEPS['full'] = False`로 1단계를 건너뛸 수 있습니다.

## 3. 공유 범위

- 결과 묶음의 `full_aggregates/`에는 10 미만 칸이 가려지지 않은 집계가 있을 수 있으므로 팀 밖으로 보내지 않습니다. 공개용은 `site_data/results_full.json`입니다.
- 모델 묶음에는 학습 자료와 환자 단위 값이 없습니다. 외부 기관은 그 안의 `KCD_EXTERNAL.ipynb`를 실행합니다.

## 4. 사이트 갱신

```powershell
copy <결과 묶음>\site_data\* site\data\
python scripts/check_public_data.py
python -m http.server -d site 8000
```
