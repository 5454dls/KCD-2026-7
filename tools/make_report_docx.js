const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, BorderStyle, Footer, PageNumber, TableOfContents, PageBreak,
} = require("docx");

const FONT = "Malgun Gothic";
const W = 9638; // A4 text width in DXA with 2 cm margins
const INK = "1F2A30", MUTED = "5A6B70", ACC = "1F6F64", HEAD = "E8F0EE", LINE = "B9C8C4";

const P = (text, o = {}) => new Paragraph({ spacing: { after: 100, line: 300 }, ...o,
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === "string" ? new TextRun({ text: t }) : t) });
const B = (t) => new TextRun({ text: t, bold: true });
const H1 = (t, br = false) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: br, spacing: { before: 360, after: 200 }, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const L = (text, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, spacing: { after: 60, line: 290 },
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === "string" ? new TextRun({ text: t }) : t) });
let NUM_INST = 1;
const newList = () => { NUM_INST += 1; };
const N = (text) => new Paragraph({ numbering: { reference: "num", level: 0, instance: NUM_INST }, spacing: { after: 60, line: 290 },
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === "string" ? new TextRun({ text: t }) : t) });

function table(head, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const ws = widths.map(w => Math.round(w / total * W));
  ws[ws.length - 1] += W - ws.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: LINE };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, i, isHead) => new TableCell({
    width: { size: ws[i], type: WidthType.DXA }, borders,
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: HEAD } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: String(t).split("\n").map(line => new Paragraph({ spacing: { after: 0, line: 260 },
      children: [new TextRun({ text: line, bold: isHead, size: 18 })] })),
  });
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: ws,
    rows: [new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, i, true)) }),
           ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))] });
}
const gap = () => new Paragraph({ spacing: { after: 60 }, children: [] });
const path = require("path");
const R = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "site", "data", "results_full.json"), "utf8"));
const AGG = process.argv[3] || null; // optional: full_aggregates folder from the return package (counts < 10 are hidden here)
const MIN = R.meta.min_cell_suppression || 10;
const csv = (f) => { if (!AGG || !fs.existsSync(path.join(AGG, f))) return null;
  const [h, ...rows] = fs.readFileSync(path.join(AGG, f), "utf8").replace(/^﻿/, "").trim().split(/\r?\n/);
  const k = h.split(","); return rows.map(r => Object.fromEntries(r.split(",").map((v, i) => [k[i], v]))); };
const n = (v) => v == null ? "<10" : Number(v).toLocaleString("en-US");
const sm = (v) => (v == null || v === "") ? "–" : (+v > 0 && +v < MIN ? `<${MIN}` : Number(v).toLocaleString("en-US"));
const pc = (v, d = 1) => v == null ? "–" : (v * 100).toFixed(d) + "%";
const f = (v, d = 3) => v == null ? "–" : Number(v).toFixed(d);
const ci = (v, c, d = 3) => v == null ? "–" : `${f(v, d)} (${c ? f(c[0], d) + "-" + f(c[1], d) : "–"})`;
const flowN = Object.fromEntries(R.flow.map(r => [r.step, r.n]));
const t1n = R.table1.find(r => r.variable === "n");
const A3 = R.agent3_development, P3 = R.agent3_performance, ET = R.event_thresholds, BT = R.agent3_by_threshold;
const PD = R.prognosis_development, PP = R.prognosis_performance, DI = R.death_icu_performance, AF = R.agent1_flags, A1 = R.agent1_performance;
const PROG = { recovered_alive_icu: "회복·생존 퇴실", recovered_died_icu: "회복·ICU 사망", unrecovered_alive_icu: "미회복·생존 퇴실", unrecovered_died_icu: "미회복·ICU 사망", unassessable_alive_icu: "판정 불가·생존 퇴실", unassessable_died_icu: "판정 불가·ICU 사망" };
const CAUSE = { hypoxic_hepatitis: "저산소성 간염", congestive_hepatopathy: "울혈성 간병증", sepsis_cholestasis: "패혈증 유발 담즙정체", dili: "약물 유발 간손상 (DILI)", pn_associated: "정맥영양 관련", biliary_obstruction_cholangitis: "담도폐쇄·담관염", cholecystitis: "담낭염", hemolysis: "용혈", viral_hepatitis: "바이러스 간염", others: "기타" };
const gbm = P3.models.find(m => m.model === "그래디언트 부스팅"), lr = P3.models.find(m => m.model === "로지스틱 회귀");
const body = [];

body.push(new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: "KCD 2026 Datathon", color: ACC, bold: true, size: 26 })] }));
body.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "ICU 간기능 이상 감시 에이전트", bold: true, size: 48 })] }));
body.push(new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "결과 보고서: 데이터 전처리부터 성능과 그림자 실험까지", size: 30, color: MUTED })] }));
body.push(P([B("데이터  "), "MIMIC-IV v3.1 전체 (개발·내부 검증)"]));
body.push(P([B("실행  "), `${R.meta.built_at.slice(0, 10)} 생성, 실행 시간 ${R.meta.runtime_minutes}분 (모델 개발 단계), 규칙 모델 ${R.meta.rule_model}`]));
body.push(P([B("공개 기준  "), `모든 수는 집계값이며 ${MIN} 미만 칸은 가렸습니다(<${MIN}). 빼기로 되살릴 수 있는 칸은 함께 가렸습니다.`]));
body.push(P([B("성격  "), "후향적 연구용 시스템의 내부 검증 결과입니다. 임상 의사결정에 쓰지 않습니다."]));
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("목차")] }));
body.push(new TableOfContents("목차", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(P([new TextRun({ text: "목차가 비어 보이면 Word에서 목차를 오른쪽 클릭한 뒤 '필드 업데이트'를 누르세요.", color: MUTED, size: 18 })]));

body.push(H1("1. 요약", true));
body.push(L(`대상: 첫 ICU 입실 성인 ${n(t1n.overall)}명(개발 ${n(t1n.development)}, 평가 ${n(t1n.evaluation)}). 이 가운데 ${n(t1n.event)}명(${pc(t1n.event / t1n.overall)})에서 ICU 중 첫 간 검사 이상(ALT·ALP·총빌리루빈 중 하나라도 ≥2×ULN)이 생겼습니다.`));
body.push(L(`경과: 2×ULN 첫 사건 중 ICU 퇴실 때 회복 ${pc(ET[0].recovered_rate)}, ICU 사망(퇴실 후 24시간 포함) ${pc(ET[0].died_icu_rate)}. 5×ULN에서는 회복 ${pc(ET[3].recovered_rate)}, 사망 ${pc(ET[3].died_icu_rate)}로 기준이 높을수록 나빴습니다.`));
body.push(L(`조기 예측(Agent 3, 24시간 안 ≥5×ULN): AUROC ${f(gbm.auroc)}, AUPRC ${f(gbm.auprc)}(발생률 ${pc(gbm.prevalence)}). 판별은 좋지만 그래디언트 부스팅은 보정이 나쁩니다(절편 ${f(gbm.cal_intercept, 2)}, 기울기 ${f(gbm.cal_slope, 2)}). 로지스틱 회귀는 판별이 거의 같고(AUROC ${f(lr.auroc)}) 보정이 낫습니다(절편 ${f(lr.cal_intercept, 2)}, 기울기 ${f(lr.cal_slope, 2)}).`));
body.push(L(`예후(Agent 2): ICU 사망 AUROC ${f(DI.auroc)}. 회복×사망 4경로 분류는 정확도 ${pc(PP.models[1].accuracy, 0)}, 경로별 AUROC 0.68-0.81로 중간 수준입니다.`));
body.push(L(`궤적: ALT·AST·ALP·총빌리루빈 모두 6-24시간 뒤 값을 '마지막 값 유지'보다 정확하게 맞혔고, 80% 구간의 실제 포함률은 71-77%로 구간이 약간 좁습니다.`));
body.push(L(`원인(Agent 1, 규칙 모델): 퇴원 진단 대리 기준과 Top-1 ${pc(A1.models[0].top1.rate, 0)}, Top-3 ${pc(A1.models[0].top3.rate, 0)} 일치. 담도폐쇄는 과다 판정(특이도 32%), DILI(민감도 8%)와 바이러스 간염은 거의 잡지 못했습니다. 대리 기준 자체의 한계는 11장에 정리했습니다.`));
body.push(L(`그림자: 행 단위 분할은 AUROC를 ${f(R.shadow_leakage.after_patient_split)}에서 ${f(R.shadow_leakage.before_row_split)}로 부풀렸고, 측정 안 된 시간을 정상으로 두면 정상 시간의 ${pc(R.shadow_missing_as_normal.unmeasured_share_of_naive_normal, 0)}가 실제로는 측정되지 않은 시간이었습니다. ALT 소수점 오류 하나로 규칙 원인 판정 ${R.shadow_data_error.scenarios[1].before.agent1_flag_changed}/${R.shadow_data_error.scenarios[1].before.agent1_rows}건이 바뀌었고, 입력 검사(가드)가 그중 ${pc(R.shadow_data_error.scenarios[1].catch_rate_after, 0)}를 잡았습니다.`));

body.push(H1("2. 데이터 원천과 사용 테이블"));
body.push(P("MIMIC-IV v3.1의 hosp·icu 모듈을 CSV로 직접 읽었습니다. 환자 단위 원자료와 중간 파일은 실행 PC 밖으로 나가지 않았고, 이 보고서의 수는 모두 집계입니다."));
body.push(table(["테이블", "모듈", "쓰임"], [
  ["patients", "hosp", "나이, 성별, 사망일(dod)"],
  ["admissions", "hosp", "입원 시각(admittime), 입원 유형, 병원 내 사망 시각(deathtime)"],
  ["icustays", "icu", "ICU 입실·퇴실 시각, 첫 ICU 유형"],
  ["labevents, d_labitems", "hosp", "간 검사와 혈액·생화학 검사, 참고범위 상한(ULN), 결과 보고 시각(storetime). ICU 안에서 채혈한 검사도 labevents에 있음"],
  ["chartevents, d_items", "icu", "활력징후(평균동맥압, 심박, 호흡수, SpO2, 체온)"],
  ["inputevents", "icu", "승압제, 정맥영양 주입"],
  ["emar", "hosp", "약물 투여 기록(간독성 의심 약물 노출)"],
  ["outputevents", "icu", "소변량 등 보조 문맥"],
  ["diagnoses_icd", "hosp", "이전 입원 진단(기저질환 변수)과 같은 입원의 퇴원 진단(원인 대리 기준). 시각 정보 없음"],
], [2.2, 1, 6]));

body.push(H1("3. 전처리"));
body.push(H2("3.1 시점 원칙"));
body.push(L("모든 값은 그 시각까지 보고된 결과만 씁니다. 검사의 사용 가능 시각 = max(채혈 시각 charttime, 보고 시각 storetime)."));
body.push(L("검사는 같은 입원의 입원 시각(admittime)부터 반영합니다. 감시, 결과 판정, 궤적은 ICU 퇴실까지입니다."));
body.push(L("최근값은 검사 24시간, 활력징후 6시간 안의 값만 쓰고, 그보다 오래되면 결측으로 둡니다."));
body.push(H2("3.2 단위, 범위, ULN 배수"));
body.push(L("단위를 통일하고(예: µmol/L를 mg/dL로), 생리적으로 불가능한 값은 버립니다. 간 검사는 행마다 기록된 참고범위 상한으로 나눠 ULN 배수로 바꿉니다."));
body.push(L("NT-proBNP와 합토글로빈은 d_labitems 이름으로 항목을 찾았고, '<5' 같은 검출 한계 표기는 한계값으로 읽었습니다."));
const LQ = csv("02_lab_quality.csv");
if (LQ) { body.push(P("항목별 측정 수와 단위 오류로 버린 수(전체 MIMIC-IV, ICU 대상자):"));
  body.push(table(["항목", "측정 수", "단위 오류 제외", "ULN 없음"], LQ.map(r => [r.analyte, sm(r.n), sm(r.unit_invalid), sm(r.missing_uln)]), [2, 2, 2, 2])); }
body.push(H2("3.3 값 형식 감사"));
const FA = csv("02c_value_format_audit.csv");
body.push(P("숫자 칸에 문자가 들어간 경우를 종류별로 셌습니다. text = 설명문, censored = '<5' 같은 한계 표기, deidentified = 비식별 처리('___'), malformed_number = '37..5'처럼 깨진 숫자. 숫자로 읽을 수 없는 값은 버리고 그 수를 보고합니다."));
if (FA) { const big = FA.filter(r => +r.n >= MIN); const small = FA.length - big.length;
  body.push(table(["출처", "항목", "형식", "건수"], big.map(r => [r.source, r.analyte, r.kind, sm(r.n)]), [2, 2, 2.4, 1.6]));
  body.push(P([new TextRun({ text: `그 밖에 ${small}개 조합은 건수가 ${MIN} 미만이라 표에서 뺐습니다.`, color: MUTED, size: 18 })])); }
body.push(H2("3.4 입력 오류 가드 (모델 앞단)"));
body.push(L("R1 생리적 범위: 예) ALT 1-20,000, 총빌리루빈 0.05-60 mg/dL, 젖산 0.1-30 mmol/L, 체온 25-45°C, 나이 18-120세."));
body.push(L("R2 급변: 같은 환자의 48시간 안 직전 값보다 정해진 배수 이상 뛰면 보류(ALT·AST 20배, 총빌리루빈 8배, 크레아티닌 5배, INR 6배, 알부민 2.5배). 재검으로 확인되면 풀어줍니다."));
body.push(L("R3 짝 불일치: ALT와 AST 중 하나만 크게 튀면 보류."));
const GC = R.shadow_data_error.guard_cost_on_real_data;
body.push(P(`실제 자료에 적용한 비용: 검사값 ${n(GC.lab_values)}개 중 ${n(GC.quarantined)}개(${(GC.quarantined_share * 100).toFixed(3)}%)를 보류했고, 실제 첫 이상 사건 ${n(GC.true_events)}건 중 가드 때문에 알림이 막힌 사건은 ${GC.true_events_trigger_held}건이었습니다.`));
body.push(H2("3.5 모델 입력 처리"));
for (const r of R.preprocessing.rules) body.push(L(r));
body.push(P(`Agent 3 특징은 ${A3.n_features}개입니다: 간 검사 ULN 배수·24시간 변화·측정 횟수·경과 시간, 혈액·생화학, 활력징후, 승압제, 나이·성별, 이전 입원 진단, ICU 입실 전 병동 검사(마지막 값·최댓값, 입원부터 ICU까지 시간). 변수별 결측률·중앙값은 VARIABLES.md와 preprocessing_variables.csv에 있습니다.`));

body.push(H1("4. 대상자 선정"));
body.push(table(["단계", "n", "설명"], [
  ["전체 ICU 입실", n(flowN.all_icu_stays), "MIMIC-IV icustays"],
  ["환자별 첫 ICU 입실", n(flowN.first_icu_per_patient), `제외 ${n(flowN.all_icu_stays - flowN.first_icu_per_patient)} (같은 환자의 두 번째 이후 입실)`],
  ["성인·유효 입실", n(flowN.adult_valid_stay), `제외 ${n(flowN.first_icu_per_patient - flowN.adult_valid_stay)}`],
  ["ICU 입실 전 이상 없음", n(flowN.no_pre_icu_abnormality), `제외 ${n(flowN.excluded_pre_icu_abnormal)} (같은 입원의 병동에서 이미 ≥2×ULN)`],
  ["분석 코호트", n(t1n.overall), `공개 시연용 MIMIC-IV Demo 환자 ${R.meta.excluded_demo_subjects_in_cohort}명 제외`],
  ["개발 / 평가 집단", `${n(t1n.development)} / ${n(t1n.evaluation)}`, "환자 단위 70:30 무작위 분할 (seed 20261016)"],
  ["ICU 중 첫 이상 사건", n(t1n.event), `시연 환자 제외 전 ${n(flowN.first_abnormal_event)}`],
], [2.6, 1.6, 5]));
body.push(P("사건 집단: 입실 초기 이상(입실 직후 첫 측정부터 이상), 관찰 중 신규(정상 측정 뒤 이상), 시점 불명(입실 뒤 첫 측정이 이미 이상이지만 입실 초기로 보기 어려움)."));

body.push(H1("5. 결과 정의"));
body.push(table(["항목", "정의"], [
  ["이상 사건", "ALT, ALP, 총빌리루빈 중 하나라도 ≥2×ULN. 기준별 분석은 2·3·4·5×ULN"],
  ["첫 사건 시점 실행", "Agent 1(원인 10개), Agent 2(예후), 보조 군집은 2×ULN 첫 사건 시점에 한 번만 실행"],
  ["회복", "ICU 퇴실 시점, 사건 이후 마지막 ALT·ALP·총빌리루빈이 모두 2×ULN 미만"],
  ["미회복", "마지막 값 중 하나라도 2×ULN 이상"],
  ["회복 판정 불가", "사건 이후 세 항목 중 다시 재지 않은 항목이 있음. 사망은 그대로 판정"],
  ["ICU 사망", "ICU 퇴실 후 24시간 이내 사망까지 포함 (deathtime, 없으면 dod 날짜 기준)"],
  ["조기 예측 목표", "세 항목이 모두 24시간 안에 측정되고 모두 2×ULN 미만인 시각에서, 다음 24시간 안 ≥5×ULN 관찰"],
], [2.2, 7]));

body.push(H1("6. 기초 특성 (Table 1)"));
const T1 = (r, k) => { const v = r[k]; if (v == null) return "–"; if (r.kind === "count") return n(v);
  if (r.kind === "median_iqr") return `${v[0]} [${v[1]}-${v[2]}]`; if (r.kind === "n_pct") return v[0] == null ? "<10" : `${n(v[0])}${v[1] != null ? " (" + pc(v[1]) + ")" : ""}`; return String(v); };
body.push(table(["변수", "전체", "이상 사건", "사건 없음", "SMD", "개발", "평가"],
  R.table1.map(r => [r.variable, T1(r, "overall"), T1(r, "event"), T1(r, "no_event"), r.smd_event == null ? "–" : f(r.smd_event, 2), T1(r, "development"), T1(r, "evaluation")]), [3.2, 1.6, 1.5, 1.5, 0.8, 1.5, 1.4]));
body.push(P([new TextRun({ text: "연속형은 중앙값 [사분위범위], 범주형은 n (%). SMD 0.1 미만이면 비슷합니다. 개발·평가 SMD는 모두 0.02 이하로 분할이 균형적입니다.", color: MUTED, size: 18 })]));

body.push(H1("7. 기준별 사건과 경과"));
body.push(table(["기준", "첫 사건", "회복", "미회복", "판정 불가", "ICU 사망", "생존 퇴실"], ET.map(r => [
  `${r.threshold_x_uln}×ULN`, n(r.patients_with_event), `${n(r.recovered_before_icu_discharge)} (${pc(r.recovered_rate)})`, n(r.unrecovered_before_icu_discharge),
  n(r.recovery_unassessable), `${n(r.icu_death)} (${pc(r.died_icu_rate)})`, n(r.alive_icu_discharge)]), [1.2, 1.2, 1.7, 1.2, 1.3, 1.7, 1.3]));
body.push(P("기준이 높아질수록 회복률은 떨어지고(19.9%에서 8.9%) ICU 사망률은 올라갑니다(21.7%에서 29.3%). 회복 판정 불가가 2×ULN에서 23%로 커서, 퇴실 전 재검 여부가 결과 해석에 큰 영향을 줍니다."));

body.push(H1("8. 모델 개발과 성능"));
body.push(H2("8.1 Agent 3 조기 예측 (매시간)"));
const L3 = A3.label_counts;
body.push(table(["단계", "수"], [
  ["매시간 랜드마크", `${n(A3.landmarks_total)} 시점 · ${n(A3.patients_total)}명`],
  ["세 항목 24시간 안 측정 · 모두 2×ULN 미만", `${n(A3.eligible)} 시점`],
  ["정답 확인 가능", `${n(A3.known)} 시점 · ${n(A3.known_patients)}명 (재검으로 정상 확인 ${n(L3.no_abnormality_observed)}, 이상 관찰 ${n(L3.abnormal_observed)})`],
  ["정답 확인 불가 (제외)", `재검 없음 ${n(L3.insufficient_measurement)}, 24시간 전에 사망·퇴실 ${n(L3.death_or_icu_discharge_before_observation_end)}`],
  ["학습 / 평가", `${n(A3.train_rows)} 시점 · ${n(A3.train_patients)}명 / ${n(A3.test_rows)} 시점 · ${n(A3.test_patients)}명`],
  ["하이퍼파라미터", `개발 집단 환자 단위 3겹 교차검증, ${A3.gbm_grid.length}개 조합 중 학습률 ${A3.gbm_selected.learning_rate}, 잎 ${A3.gbm_selected.max_leaf_nodes}`],
], [3.5, 6]));
body.push(gap());
const m3 = [...P3.models, Object.assign({ model: "그래디언트 부스팅 · 환자당 한 시점" }, P3.one_landmark_per_patient_gbm)];
body.push(table(["모델", "평가 시점/환자", "사건", "AUROC (95% CI)", "AUPRC (95% CI)", "Brier", "보정 절편/기울기", "상위 10% PPV"], m3.map(m => [
  m.model, `${n(m.n)} / ${n(m.patients)}`, n(m.events), ci(m.auroc, m.auroc_ci), ci(m.auprc, m.auprc_ci), f(m.brier), `${f(m.cal_intercept, 2)} / ${f(m.cal_slope, 2)}`, pc(m.top10_ppv)]), [2.4, 1.6, 0.9, 1.7, 1.7, 0.9, 1.4, 1.1]));
body.push(P("해석: 두 모델의 판별은 같은 수준이고, 그래디언트 부스팅은 위험을 체계적으로 낮게 예측합니다(절편 2.1). 실제 경보 문턱을 확률로 정하려면 보정(예: 개발 집단 교차검증 예측으로 등위 회귀)을 하거나 로지스틱 회귀를 기본으로 써야 합니다. 신뢰구간은 환자 군집 부트스트랩입니다."));
body.push(H3("기준별 조기 예측 (같은 방법, 목표만 바꿈)"));
body.push(table(["목표", "평가 시점", "사건 (발생률)", "AUROC", "AUPRC", "Brier", "보정 절편/기울기"], BT.map(r => [
  `24시간 안 ≥${r.threshold_x_uln}×ULN${r.threshold_x_uln === 5 ? " (주 모델)" : ""}`, n(r.n), `${n(r.events)} (${pc(r.prevalence)})`, ci(r.auroc, r.auroc_ci), ci(r.auprc, r.auprc_ci), f(r.brier), `${f(r.cal_intercept, 2)} / ${f(r.cal_slope, 2)}`]), [2.4, 1.2, 1.6, 1.8, 1.8, 0.9, 1.4]));
body.push(H3("무엇을 보고 예측했나 (그룹 Shapley)"));
body.push(table(["그룹", "평균 |기여| (확률 %p)", "비중"], (() => { const im = R.agent3_explanation.importance, tot = im.reduce((a, g) => a + g.mean_abs, 0);
  return im.map(g => [g.label, (g.mean_abs * 100).toFixed(2), pc(g.mean_abs / tot)]); })(), [3, 2.5, 2]));
body.push(P(`${R.agent3_explanation.method}, 평가 시점 ${R.agent3_explanation.n_rows}개. '측정 패턴'(얼마나 자주, 얼마나 오래전에 쟀나)이 9개 중 ${R.shadow_shortcut.rank}위(${pc(R.shadow_shortcut.measurement_share)})로, 모델이 질병뿐 아니라 검사 행태에도 기대고 있습니다.`));

body.push(H2("8.2 Agent 2 예후 (첫 사건 시 1회)"));
body.push(P(`첫 사건 ${n(PD.events)}건. 회복×ICU 사망 4경로 모델은 회복 판정이 가능한 ${n(PD.assessable)}건으로, ICU 사망 모델은 전체 사건으로 학습했습니다.`));
body.push(table(["경로", "전체 사건 수"], Object.entries(PD.label_counts).map(([k, v]) => [PROG[k] || k, n(v)]), [4, 2]));
body.push(gap());
const cls = ["recovered_alive_icu", "recovered_died_icu", "unrecovered_alive_icu", "unrecovered_died_icu"];
body.push(table(["모델", "평가 사건", ...cls.map(c => PROG[c] + " AUROC"), "정확도"], PP.models.map(m => [m.model, n(m.n_test),
  ...cls.map(c => { const x = m.per_class.find(p => p.class === c) || {}; return ci(x.auroc, x.auroc_ci); }), pc(m.accuracy, 0)]), [2, 1, 1.7, 1.7, 1.7, 1.7, 1]));
body.push(gap());
body.push(table(["ICU 사망 모델", "평가 사건", "사망", "AUROC (95% CI)", "AUPRC (95% CI)", "Brier", "보정 절편/기울기"], [[
  "그래디언트 부스팅", n(DI.n), `${n(DI.events)} (${pc(DI.prevalence)})`, ci(DI.auroc, DI.auroc_ci), ci(DI.auprc, DI.auprc_ci), f(DI.brier), `${f(DI.cal_intercept, 2)} / ${f(DI.cal_slope, 2)}`]], [2, 1.2, 1.4, 1.9, 1.9, 0.9, 1.5]));
body.push(P("해석: 사망은 비교적 잘 가려내지만(0.83), 회복 여부는 첫 사건 시점 정보만으로 예측이 어렵습니다(0.67-0.75). '회복·ICU 사망' 경로는 사건 수가 적어 혼동행렬은 공개하지 않습니다(작은 칸이 행 합계로 되살아나기 때문)."));

body.push(H2("8.3 24시간 궤적 (상태와 무관하게 매시간)"));
const TD = R.trajectory_development;
body.push(P(`평가 환자의 ${TD.step_hours}시간 간격 시점에서 ${TD.horizons.join("·")}시간 뒤 실제 측정값(±3시간)과 비교했습니다. 오차는 log2(값/ULN)의 평균 절대 오차이고 1.0은 두 배 차이입니다. 분위 그래디언트 부스팅(10·50·90%)에 교차 방지 재배열을 적용했습니다.`));
const TP = R.trajectory_performance;
body.push(table(["항목", "+6h 모델/유지", "+12h", "+18h", "+24h", "80% 구간 포함률(24h)"], TD.analytes.map(a => { const g = h => TP.find(r => r.analyte === a && r.horizon_h === h) || {};
  return [a, ...TD.horizons.map(h => `${f(g(h).mae_model, 2)} / ${f(g(h).mae_persistence, 2)} (n=${n(g(h).n)})`), pc(g(24).coverage_80, 0)]; }), [1, 2, 2, 2, 2, 1.6]));

body.push(H2("8.4 Agent 1 원인 10개 (첫 사건 시 1회)"));
body.push(P(`규칙 모델 ${AF.model}은 평가 전에 고정했습니다. 평가 사건 ${n(AF.n_events)}건에서 원인 10개를 각각 0/1로 판정하고, 같은 입원의 퇴원 진단 대리 기준(11장)과 비교했습니다. 2개 이상 동시 해당 ${n(AF.events_with_multiple_flags)}건, 해당 없음 ${n(AF.events_with_no_flag)}건.`));
body.push(table(["원인", "판정 1", "대리 양성", "민감도 (95% CI)", "특이도 (95% CI)"], AF.per_cause.map(r => {
  if (r._suppressed) return [CAUSE[r.cause], "가림", "가림", "가림", "가림"];
  if (r.assessable === false) return [CAUSE[r.cause], n(r.flagged), n(r.proxy_positive), "판단 불가 (영상·국소 소견 필요)", "–"];
  if (r.proxy_available === false) return [CAUSE[r.cause], n(r.flagged), "대리 기준 없음", "–", pc(r.specificity, 0)];
  return [CAUSE[r.cause], n(r.flagged), n(r.proxy_positive), `${pc(r.sensitivity, 0)} (${pc(r.sensitivity_ci[0], 0)}-${pc(r.sensitivity_ci[1], 0)})`, `${pc(r.specificity, 0)} (${pc(r.specificity_ci[0], 0)}-${pc(r.specificity_ci[1], 0)})`]; }), [2.6, 1.1, 1.3, 2, 2]));
body.push(P(`용혈과 바이러스 간염은 칸 하나가 ${MIN} 미만이라, 빼기로 되살아나지 않도록 판정 수와 민감도·특이도를 함께 가렸습니다. Top-1 일치 ${A1.models[0].top1.k}/${A1.models[0].top1.n}(${pc(A1.models[0].top1.rate, 0)}), Top-3 ${A1.models[0].top3.k}/${A1.models[0].top3.n}(${pc(A1.models[0].top3.rate, 0)}), 판단 보류 ${pc(A1.models[0].abstain_rate, 0)}.`));

body.push(H2("8.5 보조 군집"));
body.push(P("ALT·ALP·총빌리루빈 배수와 R 비(log2)로 가우시안 혼합(대각 공분산)을 맞추고 BIC로 군집 수를 골랐습니다. 후보 2-8 중 8이 선택되었고 부트스트랩 ARI 중앙값은 0.83(사분위 0.70-0.90)이었습니다. BIC가 후보 끝(8)까지 계속 줄어 경계에서 선택된 것이므로, 군집 수는 확정값이 아니라 탐색 결과로 봐야 합니다. 군집은 원인이 아니라 패턴 묶음입니다."));

body.push(H1("9. 그림자 실험 (전과 후)"));
const SL = R.shadow_leakage, SM = R.shadow_missing_as_normal, SC = R.shadow_competing.evaluation;
body.push(table(["그림자", "잘못된 방법 (전)", "바른 방법 (후)", "의미"], [
  ["누수: 행 단위 분할", `AUROC ${f(SL.before_row_split)}, AUPRC ${f(SL.before_auprc)}`, `환자 단위 분할 AUROC ${f(SL.after_patient_split)}, AUPRC ${f(SL.after_auprc)}`, "같은 환자의 다른 시각이 학습·평가에 섞이면 성능이 거의 완벽해 보임"],
  ["결측을 정상으로", `정상 시간 ${n(SM.naive_normal_hours)}`, `실제 판정 가능 ${n(SM.truly_eligible_hours)}`, `정상으로 센 시간의 ${pc(SM.unmeasured_share_of_naive_normal, 0)}는 측정 안 된 시간`],
  ["경쟁 위험 무시 (평가 집단)", `48시간 해소 ${pc(SC.naive_resolution_48h)}`, `Aalen-Johansen ${pc(SC.aj_resolution_48h)} (사망 ${pc(SC.aj_death_48h)}, 퇴실 ${pc(SC.aj_discharge_48h)})`, "사망·퇴실을 검열로 두면 해소를 과대 추정"],
  ["알림 폭주", `이상 검사 기록 ${n(R.alert_aggregation[0].n)}건`, `환자별 첫 사건 ${n(R.alert_aggregation[1].n)}건`, "기록마다 알리면 같은 환자에게 반복 알림"],
  ["지름길 학습", "간 수치로 예측한다고 믿음", `측정 패턴이 기여 ${R.shadow_shortcut.rank}위 (${pc(R.shadow_shortcut.measurement_share)})`, "검사 빈도라는 진료 행태를 배움"],
], [2, 2.4, 2.6, 3]));
body.push(H2("9.1 값 하나가 틀리면 (입력 오류 실험)"));
const DE = R.shadow_data_error;
body.push(P(`평가 환자 ${n(DE.design.patients)}명에서 무작위로 한 시점씩 골라(규칙 모델은 ${DE.design.rule_rows}건) 흔한 입력 오류 하나를 심고, 가드 없이와 가드 적용 후를 비교했습니다. 고위험 = 깨끗한 자료 위험 상위 9%.`));
const s = (v) => v == null ? "<10" : n(v);
body.push(table(["오류", "해당 시점", "가드가 잡음", "새 고위험 (전→후)", "원인 판정 변경 (전→후)"], DE.scenarios.map(x => [
  x.label, n(x.applicable), pc(x.catch_rate_after, 0), `${s(x.before.new_high_risk)} → ${s(x.after.new_high_risk)}`, x.before.agent1_rows ? `${s(x.before.agent1_flag_changed)} → ${s(x.after.agent1_flag_changed)} / ${x.before.agent1_rows}` : "–"]), [3, 1.2, 1.3, 1.8, 2.2]));
body.push(P("핵심: 범위 밖 오류(나이, ALT×100, 체온 °F)는 가드가 거의 다 잡지만, 그럴듯한 범위 안에 떨어지는 오류(빌리루빈 µmol/L 42%, 젖산×10 8%)는 대부분 통과합니다. 체온 형식 오류('37..5')는 가드가 없으면 조용히 '측정 안 됨'으로 바뀝니다."));

body.push(H1("10. 진단 코드(ICD) 매핑"));
body.push(H2("10.1 원인 10개의 대리 기준"));
body.push(P("MIMIC-IV에는 ICD-9-CM(대략 2015년 9월까지)과 ICD-10-CM(그 뒤)이 섞여 있어 두 버전을 모두 매핑했습니다. 코드는 앞자리 일치(접두어)로 찾습니다. 예: 'I50'은 I50.0-I50.9 전체."));
body.push(table(["원인", "ICD-9-CM", "ICD-10-CM", "문제점"], [
  ["저산소성 간염", "570 급성·아급성 간괴사\n785.5x 쇼크(785.50 상세불명, 785.51 심인성, 785.52 패혈성, 785.59 기타)", "K72.0 급성·아급성 간부전\nK76.2 중심 출혈성 간괴사\nK76.3 간경색\nR57.x 쇼크", "쇼크 코드는 저산소성 간염이 아님. 쇼크가 있었다는 뜻일 뿐이라 대리 양성이 과다"],
  ["울혈성 간병증", "573.0 간의 만성 수동 울혈\n428.x 심부전", "K76.1 간의 만성 수동 울혈\nI50.x 심부전", "심부전 코드만으로 간 울혈을 뜻하지 않음"],
  ["패혈증 유발 담즙정체", "038.x 패혈증\n995.91 패혈증\n995.92 중증 패혈증\n785.52 패혈성 쇼크", "A40.x 연쇄구균 패혈증\nA41.x 기타 패혈증\nR65.2x 중증 패혈증", "담즙정체 코드는 요구하지 않음. 패혈증이 있었다는 뜻"],
  ["약물 유발 간손상", "573.3 상세불명 간염(독성 간염 포함)\nE930-E949 치료 목적 약물의 부작용(외인 코드)", "K71.x 독성 간질환\nT36-T50 약물·생물학적 제제의 중독·부작용", "E·T 코드는 간이 아닌 부작용(발진, 출혈 등)도 포함. 간 특이적인 것은 K71뿐"],
  ["정맥영양 관련", "없음", "없음", "해당 코드가 없어 대리 기준 없음 (민감도 계산 안 함)"],
  ["담도폐쇄·담관염", "576.1 담관염\n576.2 담관 폐쇄\n574.3-574.9 담관 결석이 들어간 범주", "K83.0 담관염\nK83.1 담관 폐쇄\nK80.3-K80.7 담관 결석(담관염·담낭염 동반 포함)", "결석이 있어도 이번 간 이상의 원인인지는 모름"],
  ["담낭염", "575.0 급성 담낭염\n575.1 기타 담낭염\n574.0-574.1, 574.3-574.4, 574.6-574.8 결석+담낭염", "K81.x 담낭염\nK80.0, K80.1, K80.4, K80.6 결석+담낭염", "구조화 자료로는 판정하지 않음(규칙 모델 항상 0)"],
  ["용혈", "282.x 유전성 용혈 빈혈\n283.x 후천성 용혈 빈혈", "D55-D59 용혈 빈혈", "겸상적혈구·지중해빈혈 같은 만성 질환도 포함"],
  ["바이러스 간염", "070.x 바이러스 간염", "B15-B19 급성·만성 바이러스 간염\nB25.1 거대세포바이러스 간염\nB00.81 단순포진 간염", "만성 B·C형 보유(B18)도 포함. 이번 급성 이상의 원인이 아닐 수 있음"],
  ["기타", "571.x 만성 간질환·간경변(알코올 포함)\n728.88 횡문근융해", "K70 알코올 간질환\nK74 섬유화·간경변\nK75 기타 염증 간질환(자가면역 간염 포함)\nK76.0 지방간\nM62.82 횡문근융해", "기저 간질환은 원인이라기보다 배경"],
], [1.6, 3, 3, 2.6]));
body.push(P("코드는 kcd_liver/develop.py의 ICD_PROXY에 정규식으로 있습니다. 한 입원에 여러 원인 코드가 있으면 여러 원인이 동시에 양성입니다(담관 결석+담낭염 코드는 두 원인에 모두 들어감)."));
body.push(H2("10.2 기저질환 변수 (Table 1과 모델 입력)"));
body.push(table(["변수", "ICD-9-CM", "ICD-10-CM"], [
  ["이전 간질환", "570-573, 070", "B18, K70-K74, K76"],
  ["이전 바이러스 간염", "070", "B15-B19"],
  ["이전 심부전", "428", "I50, I43, I11.0, I13.0, I13.2"],
  ["이전 만성 신질환", "585, 586, V56", "N18, N19, Z99.2, Z94.0"],
  ["이전 당뇨", "250", "E10-E14"],
  ["이전 암", "140-208", "C"],
], [2.4, 3, 4]));
body.push(P("기저질환은 이번 ICU 입실 전에 퇴원이 끝난 이전 입원의 진단만 씁니다. 같은 입원의 코드는 퇴원 뒤에 붙으므로 모델 입력에 쓰면 미래 정보 누수가 됩니다."));

body.push(H1("11. 대리 기준의 시간 문제"));
body.push(H2("11.1 질문: ICU 진단 시점이 없지 않은가, 병원과 ICU를 구분하는가"));
body.push(L([B("시각이 없습니다. "), "diagnoses_icd에는 subject_id, hadm_id, seq_num, icd_code, icd_version만 있고 진단 시각이 없습니다. 코드는 퇴원 뒤 청구용으로 붙습니다. seq_num은 중요도 순서이지 시간 순서가 아닙니다."]));
body.push(L([B("병원과 ICU를 구분하지 못합니다. "), "코드는 입원(hadm_id) 단위입니다. ICU 재원은 그 입원 안에 들어 있으므로, 같은 입원의 코드가 ICU 전 병동, ICU 중, ICU 퇴실 뒤 병동 중 언제 생긴 문제인지 알 수 없습니다. 입원 시 이미 있던 질환인지 표시(present on admission)도 MIMIC-IV에는 없습니다."]));
body.push(L([B("그래서 대리 기준은 '정답'이 아닙니다. "), "'이 입원 중 언젠가 이 진단이 붙었다'는 뜻일 뿐입니다. 예) ICU 퇴실 뒤 병동에서 생긴 담낭염도, 입원 전부터 있던 만성 B형 간염도 첫 사건의 원인 양성으로 셉니다. 반대로 실제 원인이었어도 코드가 안 붙으면 음성입니다."]));
body.push(L([B("영향. "), "8.4의 민감도·특이도는 '대리 기준과의 일치도'이며 원인 판정 정확도의 하한도 상한도 아닙니다. 그래서 전문의 눈가림 판정을 별도 참조 기준으로 두었습니다(진행 중)."]));
body.push(H2("11.2 개선안: 시각이 있는 근거로 대리 기준 보강"));
body.push(table(["원인", "시각이 있는 근거 (MIMIC-IV)", "창"], [
  ["패혈증", "microbiologyevents 혈액배양 양성(charttime) + 항생제 투여(emar) : Sepsis-3 의심 감염 정의", "첫 사건 -48h ~ +24h"],
  ["저산소성 간염", "승압제 주입(inputevents), 평균동맥압 <65(chartevents), 젖산 ≥2 + AST/ALT 급상승 후 빠른 하강", "-24h ~ 0"],
  ["담도폐쇄·담관염·담낭염", "procedures_icd의 ERCP, 담관 배액, 담낭 절제·경피 담낭조루술(chartdate, 날짜 단위), 영상 판독문(MIMIC-IV-Note, 별도 승인 필요)", "같은 날 ± 2일"],
  ["DILI", "emar 투여 시각 + 약물별 잠복기, 다른 원인 배제 후", "투여 후 잠복기 안"],
  ["정맥영양", "inputevents 정맥영양 주입 시작·지속 시간", "사건 전 누적 기간"],
  ["용혈", "합토글로빈 저하, LDH 상승, 간접 빌리루빈 우세(labevents)", "-24h ~ +24h"],
], [2, 5.5, 2]));
body.push(P("이 방식은 같은 정보를 규칙 모델도 쓰기 때문에 순환 평가가 될 수 있습니다. 따라서 ICD 대리 기준, 시각 근거 대리 기준, 전문의 판정 세 가지를 나란히 보고하는 것을 권합니다."));

body.push(H1("12. 한계와 다음 단계"));
body.push(L("Agent 3 그래디언트 부스팅의 보정이 나쁩니다. 확률 문턱을 쓰기 전에 재보정하거나 로지스틱 회귀를 기본 모델로 둡니다."));
body.push(L("원인 판정은 대리 기준의 시간 문제(11장) 때문에 해석에 한계가 있습니다. 담도폐쇄 과다 판정, DILI·바이러스 간염 저판정을 규칙에서 고쳐야 합니다."));
body.push(L("회복 판정 불가가 23%입니다. 퇴실 전 재검 여부가 결과를 좌우합니다."));
body.push(L("군집 수가 후보 끝에서 선택되었습니다."));
body.push(L("한 기관 자료입니다. 모델 패키지(원자료 없음)로 외부 기관 검증이 필요합니다."));
body.push(L("과거 자료 재생입니다. 실제 배포 전 전향적 검증이 필요합니다."));

body.push(H1("13. 공유 범위"));
body.push(table(["묶음", "들어 있는 것", "들어 있지 않은 것"], [
  ["결과 묶음 (return package)", "집계 표, 그림, 결과 JSON, 변수 설명, 모델 카드", "환자 단위 자료, ID"],
  ["모델 묶음 (model package)", "학습된 모델 파일 9개, 특징 명세, 외부 검증 노트북(KCD_EXTERNAL.ipynb), 코드, 공개 Demo 환자 ID 목록", "학습 자료, 환자 단위 값 (노트북 출력 없음)"],
  ["공개 사이트", `전체 MIMIC-IV 집계(${MIN} 미만 가림), MIMIC-IV Demo 환자 4명 재생`, "전체 MIMIC-IV 환자 단위 자료"],
], [2.4, 4.5, 3]));

const doc = new Document({
  creator: "KCD 2026 team", title: "ICU 간기능 이상 감시 에이전트 결과 보고서",
  styles: {
    default: { document: { run: { font: { ascii: FONT, eastAsia: FONT, hAnsi: FONT }, size: 20, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: ACC, font: { ascii: FONT, eastAsia: FONT, hAnsi: FONT } }, paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: INK, font: { ascii: FONT, eastAsia: FONT, hAnsi: FONT } }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 21, bold: true, font: { ascii: FONT, eastAsia: FONT, hAnsi: FONT } }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 240 } } } },
                                 { level: 1, format: LevelFormat.BULLET, text: "-", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 240 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 440, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "KCD 2026 · ICU 간기능 감시 에이전트 결과 보고서 · ", size: 16, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(process.argv[2], b); console.log("wrote", process.argv[2]); });
