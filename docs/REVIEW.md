# 계획·노트북 검토와 github.io 시연 방향

대상: `KCD_liver_agent_project_plan_1.docx` (v2.1), `KCD_MIMICIV_liver_agents_v2.ipynb`, [hannesill/m4](https://github.com/hannesill/m4) (commit `f3c4342`, 노트북이 검토한 commit과 동일)
작성일: 2026-10-16. 코드 동결 목표 10월 17일 12:30 기준.

## 1. 결론

1. **계획의 골격은 유지합니다.** 시점 누출 차단, 환자 단위 분할, 판단 보류 허용, 실패 분모 유지, 경쟁위험 처리, Wilson CI는 모두 방법론적으로 맞습니다. 바꿀 것은 방향이 아니라 "보이게 만드는 층"과 몇 가지 구현 결함입니다.
2. **github.io에는 Gradio를 올릴 수 없습니다.** GitHub Pages는 정적 파일만 제공하므로 Python 서버(Gradio)가 돌지 않습니다. 그래서 노트북이 만든 `replay.sqlite`와 집계표를 JSON으로 내보내고, 정적 웹 페이지(`site/index.html`)가 그것을 재생하도록 바꿨습니다. LLM 출력은 사전 계산 캐시이므로 발표장 네트워크가 불안정해도 시연이 멈추지 않습니다.
3. **공개 페이지의 환자 화면은 MIMIC-IV Demo로만 만듭니다.** 저장소가 public이고, 전체 MIMIC-IV는 자격 인증 데이터입니다. MIMIC-IV Clinical Database Demo(100명)는 ODbL v1.0 공개 데이터라 환자 단위 재생을 올릴 수 있습니다. 전체 MIMIC-IV 결과는 10 미만 칸을 가린 집계 그림만 올립니다. 이 규칙은 내보내기 스크립트와 CI 검사로 강제했습니다.

## 2. 노트북에서 확인한 문제와 조치

근거는 원본 노트북의 셀 번호(원본 기준)입니다. 수정은 `notebooks/` 커밋 diff에서 확인할 수 있습니다.

| # | 문제 | 근거 | 영향 | 조치 |
|---|---|---|---|---|
| 1 | 시연 환자를 평가 집단에서 무작위 6명 추출 | 셀 56 `cohort[...].sample(DEMO_CASES)` | 이상 사건이 없는 환자만 뽑히면 Agent 1·2가 화면에 한 번도 안 나옴 | 결과를 보기 전 고정 시드로 집단별(초기·신규·불명·정상) 2명씩 사전 지정 (`select_demo_stays`) |
| 2 | 시연 환자의 Agent 1 캐시가 없을 수 있음 | 셀 22는 평가 사건 40건만 LLM 실행, 셀 56은 다른 표본 | 화면에 `model_not_run`만 표시 | LLM 실행 대상에 시연 환자를 합집합으로 포함 |
| 3 | 재계산 서명에 매시간 활력징후 포함 | 셀 56 signature가 `ss['context']` 전체 사용 | `PRECOMPUTE_DEMO_LLM=True`이면 거의 매시간 LLM 재호출(환자당 최대 120×2회) | 간 패널 검사와 투약·주입 기록이 바뀔 때만 재호출 |
| 4 | 기본 vs 군집 비교를 독립 CI 두 개로만 제시 | 셀 26 `score_predictions` | 같은 환자를 두 조건으로 본 짝지은 자료인데 짝을 무시 | 정확 McNemar 추가 (`35_paired_mcnemar.csv`) |
| 5 | 근거 인용의 사실성을 자동으로 확인하지 않음 | 셀 22 `validate_answer`는 형식만 검사 | 환각 근거가 전문의 검토 전까지 드러나지 않음 | 인용 항목이 스냅샷에 실제 있는지 대조 (`grounding_check`, `36_grounding_detail.csv`) |
| 6 | 입력 오류 시험이 사례 1건 | 셀 36 `e=next(selected.itertuples())` | 일화 하나로는 비율·CI를 낼 수 없음 | 시연 사건 + 평가 20건으로 확장, 주입 원인 기록 |
| 7 | 실행해도 그림이 거의 없음 | 결과가 CSV로만 저장됨 | "돌기는 하는데 제대로 하는지 모름" | 마지막 셀에서 Figure 2–10을 노트북에 바로 표시하고 PNG 저장 |
| 8 | Agent 2 화면에 곡선이 없음 | 셀 34는 48시간 끝값만 반환 | 경쟁위험의 의미를 그림으로 설명하기 어려움 | Aalen-Johansen 곡선을 함께 저장 |

검증: MIMIC-IV와 같은 구조의 합성 CSV(240명)와 OpenAI 호환 모의 LLM 서버로, 모든 옵션(`RUN_LLM`, `RUN_PERTURBATION`, `BUILD_AGENT3_REFERENCE`, `RUN_FORECAST_VALIDATION`, `PRECOMPUTE_DEMO_LLM`, `ENABLE_REVIEW_SCORING`)을 켠 채 65개 셀을 끝까지 실행했습니다(약 4분, 오류 0). 모의 LLM이 일부러 섞은 가짜 근거 27건은 근거 검증에서 정확히 27건 모두 미확인으로 잡혔습니다. **실제 MIMIC 자료와 실제 LLM으로는 실행하지 않았습니다.** 이 환경에서는 PhysioNet 접속이 막혀 있습니다.

### 고치지 않은 것 (판단이 필요하거나 범위 밖)

- **M4 백엔드 모드**: `file_for()`가 CSV 파일 존재를 먼저 요구하고 `columns()`가 CSV 헤더를 읽습니다. `DATA_BACKEND='m4'`로 바꿔도 CSV가 같은 경로에 없으면 실패합니다. 오늘은 CSV 모드로 진행하고, M4는 데모 데이터 준비(`m4 init mimic-iv-demo`)와 SQL 확인 용도로 쓰는 편이 안전합니다.
- **사망 판정**: `deathtime <= dischtime`으로 판정합니다. `hospital_expire_flag`와 교차 확인을 권합니다.
- **주입 원인이 실제 정답인 사례**: 항생제 주장을 모든 사례에 같은 문장으로 넣습니다. 전문의 판정이 `drug_related`인 사례는 "채택"이 오류가 아니므로 Figure 9에서 빼야 합니다.
- **과잉 방어**: 방어 문구가 참인 정보까지 무시하게 만들 수 있습니다. 실제 투약 기록과 맞는 "참인 주장" 조건을 하나 추가하면, 잘못된 주장 거부율과 참인 주장 활용률을 함께 보고할 수 있습니다.
- **반복 안정성**: temperature 0도 결정론을 보장하지 않습니다. 평가 10건을 3회 반복해 `repeat_stability`(Top-1 일치율)를 보고하면 재현성 항목의 근거가 됩니다.

## 3. 통계적으로 미리 말해 둘 것

- 평가 40건, 일치율 50%이면 95% Wilson CI는 약 35–65%입니다(폭 ±15%p). 기본 vs 군집의 차이를 "검정"하기에는 검정력이 낮습니다. 발표에서는 차이의 유의성보다 추정치, 불일치 쌍 수, 실패 사례를 보여주는 편이 정직하고 질문에도 강합니다.
- Agent 2·3은 지도학습 모델이 아닌 유사 사례 비율입니다. MIMIC-IV Demo(100명)에서는 개발 집단이 작아 "유사 50명"이 사실상 개발 집단 전체가 될 수 있습니다. github.io 시연은 **작동 방식**을, 전체 MIMIC-IV 집계 그림은 **결과**를 보여준다고 구분해 말하십시오.
- Figure 8의 성능은 다음 24시간 안에 재검된 환자에서만 계산됩니다(선택 편향). 전체 대비 확인 가능 비율을 그림에 함께 표시했습니다.
- 데모 데이터의 사건 수는 실행해 봐야 압니다. 사건이 너무 적으면 군집(k=4, 개발 사건 12건 이상 필요)은 자동으로 꺼지고 "미실행"으로 표시됩니다. 이는 의도된 동작입니다.

## 4. 데이터 거버넌스

| 자료 | github.io 공개 | LLM 전송 |
|---|---|---|
| MIMIC-IV Demo (ODbL v1.0) | 환자 단위 재생 가능, 출처 표기 | 제한 없음 |
| 전체 MIMIC-IV (PhysioNet 자격 인증) | 집계만, n<10 칸 가림 | PhysioNet 2025년 지침: 제3자 공유 금지, 데이터 무보존·학습 미사용·사람 검토 없음을 직접 확인한 서비스만, 로컬 모델 권장 |

`scripts/export_site_data.py`는 `--source mimic-iv`일 때 환자 단위 내보내기를 거부하고, `scripts/check_public_data.py`는 배포 전에 sqlite·private 파일·미가림 집계를 찾으면 CI를 실패시킵니다. `.gitignore`는 `kcd_liver_v2_outputs/`를 통째로 막습니다.

## 5. 시연 구성과 발표 점수 전략

공식 기준(안내자료 7–8쪽 기준, 계획서 11단계): 임상적 가치, 데이터 사이언스 창의성, 발표 및 전달력, 재현 가능성, 멘토 팀 활동. 발표 평가는 참가자·멘토·조교 1인 1표이므로 비전공 청중도 첫 장면에서 이해해야 합니다. 배점은 공개되지 않았으므로 점수 효과는 추정하지 않습니다.

### 시연 1: 세 에이전트 (약 2분 30초)

1. **Figure 1 (20초)**: 다섯 단계와 점선 상자 두 개(시점 고정, 평가 금고)만 짚습니다. "미래를 보지 않는 구조"가 핵심 메시지입니다.
2. **입실 초기 이상 환자 (60초)**: 재생 → ALT가 오르면 2×ULN 순간 자동 정지 → Agent 1 후보와 근거·반대 근거 → `+군집` 토글 → Agent 2의 해소·사망·퇴원 막대. 끝으로 "실제 관찰 결과 공개"를 눌러 예측과 관찰을 분리합니다.
3. **관찰 중 신규 이상 환자 (50초)**: 정상일 때는 Agent 3만 켜져 있다가, 사건이 생기면 Agent 3이 꺼지고 Agent 1·2가 이어받는 장면. 세 에이전트가 한 환자를 시간순으로 넘겨받는다는 점이 이 프로젝트만의 장면입니다.
4. **정상 환자 (20초)**: 검사 공백 구간에서 상태가 "판정 불가"로 바뀌는 것. 측정 없음을 정상으로 보지 않는다는 원칙(강연의 "데이터가 만들어진 과정")을 화면으로 보여줍니다.

### 시연 2: 빛과 그림자 (약 2분)

| 탭 | 보여줄 장면 | 연결되는 기준 |
|---|---|---|
| 잘못된 입력 | 한 환자의 8칸 격자(빨간 칸 = 주입 원인 채택) → 집단 비율 그림 | 임상적 가치, AI 비판적 사용 |
| 경쟁 사건 | 점선(단순 중도절단)이 실선(경쟁위험)보다 높은 그림 | 데이터 사이언스 창의성 |
| 근거 검증 | "인용 근거 중 몇 %가 실제 기록에 있었나" | 안전성, 재현 가능 지표 |
| 알림 통합 | 검사 기록 수 vs 환자당 알림 수, 임계값 민감도 | 업무 결합, "왜 2배인가" 질문 대비 |

우리 에이전트의 실패를 직접 보여주는 장면을 최소 하나 넣으십시오(예: 근거 미확인 사례, 방어 후에도 남은 채택). 같은 기준으로 만든 다른 팀과 구별되는 지점이 됩니다.

### 마무리 (30초)

단계별 그림 한 장(Figure 6 또는 9)과 github.io QR 코드를 띄웁니다. 청중이 자기 휴대폰으로 같은 재생을 돌려볼 수 있게 하면 재현 가능성과 전달력을 한 번에 보여줄 수 있습니다.

### 대비책

- 발표장 오프라인 대비: `python -m http.server -d site 8000` 로컬 실행 + 화면 녹화. 녹화를 쓸 때는 녹화라고 말합니다.
- 페이지 단축키: Space 재생/정지, ←/→ 한 시간, 1–4 환자 선택.

## 6. 오늘 밤 실행 순서

1. MIMIC-IV Demo 내려받기: `m4 init mimic-iv-demo` 또는 PhysioNet에서 `mimic-iv-demo/2.2` CSV.
2. 노트북 설정: `ROOT`=데모 폴더, `DATA_SOURCE='mimic-iv-demo'`, `MAX_PATIENTS=None`. LLM 엔드포인트 연결 후 `RUN_LLM=True`, `RUN_PERTURBATION=True`. 시간이 되면 `BUILD_AGENT3_REFERENCE=True`, `PRECOMPUTE_DEMO_LLM=True`.
3. 끝까지 실행하면 마지막 셀이 그림을 표시하고 `site/data/`를 데모 데이터로 교체합니다. `python scripts/check_public_data.py`로 확인 후 커밋합니다.
4. 전체 MIMIC-IV는 별도 OUT 폴더에서 `DATA_SOURCE='mimic-iv'`로 실행합니다. 마지막 셀이 가림 처리된 집계만 `site/data/summary.json`에 씁니다. 이 파일은 커밋 전에 눈으로 확인하십시오.
5. 저장소 Settings → Pages → Source를 **GitHub Actions**로 설정하고 `main`에 병합하면 배포됩니다.

## 7. 확인하지 못한 것

- 실제 MIMIC-IV Demo에서의 사건 수, 군집 활성 여부, 실행 시간.
- 기관 LLM 엔드포인트의 데이터 보존 조건(전체 MIMIC 사용 시 필수 확인).
- 공식 발표 시간과 배점. 시연 시간 배분은 운영진 확인 후 조정이 필요합니다.
- MIMIC-IV Demo의 ODbL 표기는 AWS Open Data Registry 등 2차 목록으로 확인했습니다. 이 환경에서 PhysioNet 원문 LICENSE 파일은 열지 못했으니, 발표 자료에 넣기 전에 원문을 한 번 확인하십시오.

참고: [PhysioNet, Use of MIMIC data with LLMs and online services (2025)](https://physionet.org/news/post/llm-responsible-use/), [MIMIC-IV Demo on AWS Open Data](https://registry.opendata.aws/mimic-iv-demo/), [M4](https://github.com/hannesill/m4).
