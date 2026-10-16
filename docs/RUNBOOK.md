# 실행 순서 (전체 MIMIC-IV 개발 → 결과 → 발표)

전체 MIMIC-IV는 기관 PC에서만 실행합니다. 이 저장소에는 집계 결과만 들어옵니다.

## 가장 쉬운 방법: 노트북 파일 하나

```powershell
pip install pandas numpy scikit-learn matplotlib joblib nbformat nbclient ipykernel
```

`notebooks/KCD_ONE.ipynb` 파일 하나만 있으면 됩니다(코드가 안에 들어 있어 저장소가 필요 없습니다). 아무 폴더에 두고 열어 첫 셀의 `MIMIC_ROOT`만 확인한 뒤 위에서부터 끝까지 실행합니다. 코드와 중간 산출물은 노트북 옆 `kcd_work/`에 생기고, 돌려줄 zip은 노트북 옆에 생깁니다. 새 버전 노트북을 받으면 `RUN_STEPS['full'] = False`로 가장 오래 걸리는 1단계를 건너뛸 수 있습니다(단, 추출 항목이 바뀐 버전이면 1단계부터 다시). 인터넷이 막힌 PC라면 FHIR 데모를 미리 받아 `DEMO_FHIR_DIR`에 그 `fhir` 폴더를 적습니다.

끝나면 저장소 폴더에 `kcd_return_package_<날짜>.zip`이 생깁니다. 이 zip 하나를 대화에 올려주시면 사이트와 발표 자료 숫자를 교체합니다.

- 들어가는 것: `results_full.json`(10 미만 칸 가림), 논문 표 CSV, 집계 그림, 공개 데모 환자 시연 JSON, `run_report.json`(단계별 성공/실패와 소요 시간)
- 들어가지 않는 것: 학습된 모델(`run_all/models/*.joblib`), 전체 MIMIC-IV 환자 단위 파일, 실행된 노트북 사본
- 한 단계가 실패해도 나머지와 압축은 진행되고, 실패 이유가 `run_report.json`에 적힙니다.

아래 1~4는 같은 일을 손으로 나눠 하는 방법입니다.

## 0. 준비 (5분)

```powershell
pip install pandas numpy scikit-learn matplotlib ipywidgets joblib
git clone https://github.com/5454dls/KCD-2026-7   # 또는 받은 zip / bundle
```

bundle로 받은 경우: `git clone KCD-2026-7.bundle KCD-2026-7` 후 `git remote set-url origin https://github.com/5454dls/KCD-2026-7.git` 그리고 `git push -u origin claude/sweet-faraday-yxaf4j`.

## 1. 노트북을 전체 MIMIC-IV로 실행

`notebooks/KCD_MIMICIV_liver_agents_v2.ipynb` 설정 셀에서:

| 변수 | 값 |
|---|---|
| `ROOT` | 공유 드라이브 MIMIC-IV 3.1 경로 (기본값 그대로) |
| `DATA_SOURCE` | `'mimic-iv'` |
| `MAX_PATIENTS` | `None` (시간이 부족하면 `20000`부터) |
| LLM | 기관 승인 엔드포인트가 있으면 환경변수 `LLM_BASE_URL`, `LLM_MODEL`. 없으면 규칙 모델로 자동 실행 |

처음부터 끝까지 실행합니다. 실행 시간은 이 환경에서 측정하지 못했습니다(데모 100명 약 2분). labevents 전체 스캔이 가장 오래 걸립니다.

## 2. 모델 개발과 결과 표 (한 줄)

```powershell
python scripts/develop_models.py --out-dir notebooks/kcd_liver_v2_outputs --mimic-root "\\10.7.12.146\데이터 및 소스\MIMIC\MIMIC4-V3.1" --model-dir models
```

생성물:

- `notebooks/kcd_liver_v2_outputs/results_full.json`: 사이트와 발표 자료가 읽는 집계 결과 (10 미만 칸 가림)
- `notebooks/kcd_liver_v2_outputs/results_tables/table1_baseline.csv`, `table2_performance.csv`, `shadow_before_after.json`
- `models/agent3_gbm.joblib`, `models/agent2_gbm.joblib`: **커밋하지 않습니다** (자격 인증 데이터로 학습)

## 3. 시연 화면을 학습된 모델로 다시 만들기

노트북을 공개 데모 데이터로 한 번 더 실행합니다.

```powershell
python scripts/fhir_demo_to_csv.py --out data/mimic-iv-demo-csv
set KCD_MODEL_DIR=models
```

설정: `ROOT = Path('data/mimic-iv-demo-csv')`, `DATA_SOURCE = 'mimic-iv-demo'`, `MAX_PATIENTS = None`. 마지막 셀이 `site/data/`를 갱신합니다. 데모 환자 100명은 2단계 학습에서 제외되어 있으므로 시연은 처음 보는 환자에 대한 예측입니다.

## 4. 결과를 사이트에 넣고 공개 전 확인

```powershell
copy notebooks\kcd_liver_v2_outputs\results_full.json site\data\results_full.json
python scripts/check_public_data.py
python -m http.server -d site 8000
```

`results_full.json`이 있으면 사이트의 연구 설계·성능·전과 후가 자동으로 전체 MIMIC 값으로 바뀝니다.

## 5. 발표 자료 숫자 교체

발표 자료의 숫자는 지금 데모 100명 값입니다. 아래 키를 `results_full.json`에서 찾아 바꿉니다. `[__]`는 비어 있는 칸입니다.

| 슬라이드 | 바꿀 값 | JSON 키 |
|---|---|---|
| 연구 설계 | 140, 100명, 70/30, 6,504, 1,520, 281, 16 | `flow`, `table1[0]`, `agent3_development`, `agent2_development` |
| 기초 특성 | 표 전체 | `table1` (또는 `table1_baseline.csv`) |
| 모델 성능 | AUROC·AUPRC·Brier·보정 | `agent3_performance.models`, `agent2_performance`, `agent1_performance` |
| 빛 | 환자 수, 판단 시점, 사건 수, AUROC | `table1[0].overall`, `agent3_development.landmarks_total`, `agent2_development.events` |
| 그림자 1 | 1.00 → 0.78 | `shadow_leakage.before_row_split`, `after_patient_split` |
| 그림자 2 | 6,484 → 1,520, 77% | `shadow_missing_as_normal` |
| 그림자 3 | [__]%p | `shadow_competing.evaluation.naive_resolution_48h − aj_resolution_48h` |
| 그림자 4 | [__]% → [__]% | `shadow_input` (`combined`, `guarded` false/true의 `adoption_rate`). LLM 없이 실행했다면 결과 없음이라고 말함 |
| 그림자 5 | 78%, 1/3 | `agent3_development.label_counts`, `agent1_performance.n_with_proxy / n_eval_events` |
| 시스템 | 177 → 16 | `alert_aggregation` |

`results_full.json`을 이 대화에 올려주시면 사이트와 발표 자료 숫자를 바로 교체합니다.
