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

const body = [];
body.push(new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: "KCD 2026 Datathon", color: ACC, bold: true, size: 26 })] }));
body.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "ICU 간기능 이상 감시 에이전트", bold: true, size: 48 })] }));
body.push(new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "전체 명세서: 설계와 흐름", size: 32, color: MUTED })] }));
body.push(P([B("데이터  "), "MIMIC-IV v3.1 (개발·내부 검증), MIMIC-IV Clinical Database Demo (공개 시연)"]));
body.push(P([B("주제  "), "Agentic AI: Light and Shadow"]));
body.push(P([B("성격  "), "후향적 연구용 시스템. 임상 의사결정에 쓰지 않습니다."]));
body.push(P([B("기준 코드  "), "저장소 브랜치 claude/sweet-faraday-yxaf4j, 실행 노트북 KCD_ONE.ipynb"]));
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("목차")] }));
body.push(new TableOfContents("목차", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(P([new TextRun({ text: "목차가 비어 보이면 Word에서 목차를 오른쪽 클릭한 뒤 '필드 업데이트'를 누르세요.", color: MUTED, size: 18 })]));

body.push(H1("1. 개요", true));
body.push(H2("1.1 목적"));
body.push(P("ICU 환자의 간 검사(ALT, ALP, 총빌리루빈)를 시간 순서대로 따라가며 세 가지 질문에 답하는 시스템을 만듭니다. 지금 정상인 환자가 24시간 안에 크게 나빠질까(조기 예측), 이상이 처음 생겼을 때 원인은 무엇이고 어떻게 될까(원인과 예후), 앞으로 24시간 간 수치는 어떻게 움직일까(궤적). 동시에 이 시스템이 어디서 틀릴 수 있는지를 직접 실험해 전과 후로 보여줍니다(빛과 그림자)."));
body.push(H2("1.2 범위"));
body.push(L("대상: 환자별 첫 ICU 입실, 성인, 같은 입원의 ICU 입실 전 병동에서 이미 이상이 있던 환자는 제외"));
body.push(L("관찰: 같은 입원의 병동 검사를 입원 시점부터 반영하고, 감시·결과 판정·궤적은 ICU 퇴실까지"));
body.push(L("시간 단위: 매시간 재생(실시간 시스템이 아니라 과거 자료를 시간순으로 재생)"));
body.push(L("공개 범위: 공개 사이트에는 MIMIC-IV Demo 환자만 환자 단위로, 전체 MIMIC-IV는 10 미만 칸을 가린 집계만"));
body.push(H2("1.3 구성 요소 한눈에"));
body.push(table(["구성", "실행 시점", "출력", "방식"], [
  ["Agent 3 조기 예측", "정상일 때 매시간", "24시간 안에 ≥5×ULN 이상 위험 + 근거(그룹 Shapley)", "그래디언트 부스팅"],
  ["궤적 예측", "상태와 무관하게 매시간", "ALT·AST·ALP·TBIL의 6·12·18·24시간 뒤 10·50·90% 구간", "분위 그래디언트 부스팅"],
  ["Agent 1 원인", "첫 이상 시점 1회", "원인 10개 각각 0/1 + 근거 + 판단 보류 여부", "규칙 기준 모델 (LLM은 기관 승인 시 선택)"],
  ["Agent 2 예후", "첫 이상 시점 1회", "회복 여부 × ICU 사망 4경로 확률 + ICU 사망 확률", "그래디언트 부스팅 2개"],
  ["보조 군집", "첫 이상 시점 1회", "간 패턴·임상 표현형 군집 번호 (원인 아님)", "가우시안 혼합 + BIC + 부트스트랩 ARI"],
], [2.2, 2, 4, 2.6]));

body.push(H1("2. 전체 흐름"));
body.push(H2("2.1 실행 흐름 (KCD_ONE.ipynb)"));
body.push(table(["단계", "하는 일", "주요 산출물"], [
  ["0 코드 풀기", "노트북 안에 들어 있는 코드를 kcd_work/code에 풂", "실행 환경"],
  ["1 분석 노트북 (전체 MIMIC-IV)", "코호트, 검사·활력징후·투약 추출, 품질 관리, 사건 탐지, 결과 판정, 원인 규칙, 군집, 시뮬레이션 기록", "환자 단위 중간 파일(로컬), 집계 표"],
  ["2 모델 개발·평가", "Agent 3(2·3·4·5배), 예후 4경로, ICU 사망, 궤적, Shapley, 그림자 실험, 잘못된 값 실험", "results_full.json, 표 CSV, 모델 파일"],
  ["3 시연 화면", "공개 데모 100명에 학습된 모델을 적용해 매시간 화면 기록", "site/data 환자별 JSON"],
  ["4 결과 묶기", "집계·표·그림·시연 화면·실행 보고서·문서를 zip으로", "kcd_return_package_*.zip"],
  ["5 모델 패키지", "모델 4종 + 코드 + 외부 검증 노트북 + 모델 카드", "kcd_model_package_*.zip (결과 zip에도 모델 포함)"],
], [2.4, 5, 3.4]));
body.push(gap());
body.push(P("실행 중에는 '[full] 8/30 완료 (0.1분, 누적 0.2분)', 'hosp/labevents: 2,000,000행 읽음' 같은 진행 상황이 출력되고, 셀마다 걸린 시간이 run_report.json에 기록됩니다."));
body.push(H2("2.2 한 환자의 시간 흐름"));
body.push(N("ICU 입실. 같은 입원의 병동 검사가 있으면 입실 전 변수(마지막·최고 배수, 입원부터 입실까지 시간)로 반영합니다."));
body.push(N("정상 구간(세 항목이 모두 최근 24시간 안에 측정되고 모두 2×ULN 미만): 매시간 Agent 3이 24시간 안 ≥5×ULN 위험을 계산하고, 궤적 모델이 24시간 궤적을 그립니다. 하나라도 측정되지 않았으면 '판정 불가'로 표시합니다."));
body.push(N("첫 이상 시점(세 항목 중 하나라도 ≥2×ULN 결과가 보고된 시각): Agent 1이 원인 10개를 0/1로, Agent 2가 예후 4경로와 ICU 사망 확률을 한 번 계산합니다. 보조 군집 번호도 이때 한 번 붙습니다."));
body.push(N("이후: 궤적 모델은 매시간 계속 돕니다. 원인·예후는 다시 계산하지 않습니다."));
body.push(N("ICU 퇴실: 정답을 판정합니다. 회복은 퇴실 시점의 마지막 검사로, 사망은 ICU 퇴실 후 24시간 이내 사망으로 판정합니다."));

body.push(H1("3. 코호트와 흐름도"));
body.push(table(["단계", "기준"], [
  ["전체 ICU 입실", "icu/icustays 전체"],
  ["환자별 첫 ICU 입실", "같은 환자의 첫 입실만"],
  ["성인·유효 입실", "나이 ≥18세(anchor_age에 연도 차이 보정), 입실·퇴실 시각이 있고 퇴실 > 입실"],
  ["분석 표본", "기본은 전체 환자 (숫자를 넣으면 무작위 표본, 시험용)"],
  ["제외: 입실 전 이미 이상", "같은 입원에서 ICU 입실 전에 ALT·ALP·TBIL 중 하나라도 ≥2×ULN"],
  ["입실 전 이상 없음", "분석 코호트"],
  ["첫 이상 사건", "ICU 감시 구간에서 처음으로 ≥2×ULN 결과가 보고된 환자"],
], [3, 7.8]));
body.push(gap());
body.push(P("개발·평가 분할은 환자 단위 무작위 70:30(시드 20261016)입니다. 공개 데모 100명은 모델 학습에서 제외해, 시연은 처음 보는 환자에 대한 예측이 됩니다."));

body.push(H1("4. 핵심 정의"));
body.push(table(["항목", "정의"], [
  ["공개 시각", "max(charttime, storetime). 모든 입력은 그 시각까지 보고된 값만 사용"],
  ["ULN 배수", "검사값 ÷ 그 행의 참고범위 상한(ref_range_upper). 단위는 표준 단위로 통일"],
  ["이상 (첫 사건)", "ALT·ALP·총빌리루빈 중 하나라도 ≥2×ULN"],
  ["정상 (Agent 3 판정 대상)", "세 항목 모두 최근 24시간 안에 측정되고 모두 <2×ULN"],
  ["조기 예측 목표", "다음 24시간 안에 세 항목 중 하나라도 ≥5×ULN. 2·3·4배 목표 모델은 비교용"],
  ["회복", "ICU 퇴실 시점의 마지막 ALT·ALP·총빌리루빈(사건 이후 측정)이 모두 <2×ULN"],
  ["미회복", "그 마지막 값 중 하나라도 ≥2×ULN"],
  ["회복 판정 불가", "사건 이후 다시 재지 않은 항목이 있음 (재 항목은 모두 <2×). 사망은 판정"],
  ["ICU 사망", "ICU 퇴실 후 24시간 이내 사망. 원내 사망 시각(deathtime), 없으면 사망 날짜(dod)를 날짜 단위로 비교"],
  ["예후 4경로", "회복/미회복 × ICU 사망/생존 퇴실 (회복 판정 불가 사건은 4경로 학습·평가에서 빼고 ICU 사망 모델에는 포함)"],
  ["48시간 경쟁 사건", "첫 사건 후 48시간 안의 첫 회복(처음으로 세 항목 모두 <2×) · ICU 사망 · 생존 ICU 퇴실 (Aalen-Johansen)"],
  ["이상 기준 민감도", "2·3·4·5×ULN 각각 첫 사건 환자 수, 회복·미회복·판정 불가, ICU 사망, 생존 퇴실"],
], [3, 7.8]));

body.push(H1("5. 자료 추출과 품질 관리"));
body.push(H2("5.1 사용하는 표"));
body.push(table(["모듈", "표", "용도"], [
  ["hosp", "patients, admissions", "나이·성별·사망 기록, 입원·퇴원 시각"],
  ["hosp", "labevents, d_labitems", "검사값, 참고범위, 혈청 검사(문자 결과)"],
  ["hosp", "diagnoses_icd", "이전 입원 과거력, 퇴원 진단 대리 기준(평가 전용)"],
  ["hosp", "emar", "실제 투여 기록(처방만으로는 노출로 보지 않음)"],
  ["icu", "icustays, chartevents, d_items", "ICU 입·퇴실, 활력징후"],
  ["icu", "inputevents, outputevents", "승압제·정맥영양 주입, 소변량"],
], [1.4, 3.4, 6]));
body.push(H2("5.2 검사 항목"));
body.push(P("ALT, AST, ALP, 총·직접 빌리루빈, INR, 젖산, CK, LDH, 크레아티닌, GGT, 알부민, BUN, 나트륨, 칼륨, 중탄산, 혈당, 혈소판, 백혈구, 혈색소, PT, PTT, 피브리노겐, pH, PO2는 itemid로, 합토글로빈과 NT-proBNP와 간염 혈청 검사는 d_labitems의 검사명으로 찾습니다(기관마다 itemid가 달라도 동작하도록)."));
body.push(H2("5.3 단위·범위·형식"));
body.push(L("단위 통일: 빌리루빈·크레아티닌 µmol/L → mg/dL, 알부민·혈색소 g/L → g/dL, NT-proBNP pg/mL(=ng/L). 기대 단위가 아니면 무효"));
body.push(L("생리적 범위 검사(예: 크레아티닌 0.05–150, 알부민 0.1–10)로 단위·입력 오류 표시"));
body.push(L("합토글로빈 '<5' 같은 검출 한계 표기는 그 한계값으로 읽음(보수적)"));
body.push(L("혈청 결과는 문자로 해석: 'NEGATIVE ... FOR POSITIVE HCV' 같은 안내 문구가 있어도 음성 우선 판정"));
body.push(L("형식 감사: 숫자 칸인데 숫자로 읽히지 않은 값을 형식 오류(37..5), 검출 한계(<5), 비식별(___), 기타 문자로 나눠 집계"));

body.push(H1("6. 변수"));
body.push(H2("6.1 예측 모델 변수"));
body.push(P([B("간 검사 (ALT, AST, ALP, 총빌리루빈). "), "ULN 배수로 최근값(최근 24시간 안 채혈, 없으면 결측), 측정 후 경과 시간, 24시간 측정 횟수, 24시간 변화량(최근값 − 24시간 전 시점에 알려진 값) 네 가지."]));
body.push(P([B("그 밖의 검사 (INR, 알부민, 젖산, 크레아티닌, BUN, 혈소판, 백혈구, 나트륨, 혈당, 중탄산, 혈색소, NT-proBNP). "), "표준 단위 원값으로 같은 네 가지."]));
body.push(P([B("활력징후 (평균동맥압, 심박수, 호흡수, 산소포화도, 체온). "), "6시간 안 가장 최근 값. 화씨 체온은 섭씨로."]));
body.push(P([B("승압제. "), "최근 6시간 안 노르에피네프린·에피네프린·바소프레신·페닐에프린·도파민 주입 시작 기록 0/1."]));
body.push(P([B("환자. "), "나이, 성별, ICU 입실 후 경과 시간."]));
body.push(P([B("과거력. "), "이번 ICU 입실 전에 끝난 이전 입원 ICD 코드로 간질환, 바이러스 간염, 심부전, 만성 신장병, 당뇨, 암 0/1. 이전 입원이 없으면 결측."]));
body.push(P([B("입실 전 병동 변수. "), "같은 입원 병동에서 잰 간 검사 4종의 마지막·최고 배수, 입원부터 ICU 입실까지 시간."]));
body.push(P([B("모델별 차이. "), "궤적 모델은 예측 시점(6·12·18·24시간)을 추가하고 목표는 그 시각 ±3시간 안의 log2 ULN 배수. 예후·ICU 사망 모델은 첫 이상 시점의 같은 변수."]));
body.push(H2("6.2 군집 변수 (보조, 원인 판정에 쓰지 않음)"));
body.push(P([B("간 패턴 군집. "), "ALT·ALP·총빌리루빈 배수와 R 비(ALT 배수 ÷ ALP 배수, 6시간 안 동시 측정일 때만)."]));
body.push(P([B("임상 표현형 군집. "), "나이, 성별, 세 간 수치 배수와 24시간 기울기(최소 2회·1시간 이상 간격 측정에 맞춘 직선 기울기), 알부민·INR·젖산·크레아티닌·BUN·혈소판·백혈구·나트륨·혈당, 24시간 최저 평균동맥압, 심박수, 호흡수, 24시간 최저 산소포화도, 체온, 6시간 소변량 합, 승압제, 과거력 5종."]));
body.push(P("군집 전처리와 선택: 관측률 50% 미만 변수 제외, 양수 연속 변수 log2, 개발 집단 중앙값 대체, 개발 집단 평균·표준편차 표준화, K=2–8 중 BIC 최소, 부트스트랩 ARI 중앙값 <0.75이면 '불안정, 사용 안 함'."));

body.push(H1("7. 전처리 (결측·정규화, 누수 방지)"));
body.push(table(["항목", "방법"], [
  ["시점 누수", "모든 변수는 공개 시각 ≤ t인 값만. 채혈 시각 기준(미보고 결과 사용)은 그림자 실험으로만 비교"],
  ["환자 누수", "환자 단위 분할. 같은 환자의 시간들이 학습과 평가에 동시에 들어가지 않음"],
  ["오래된 값", "검사 24시간, 활력징후 6시간보다 오래되면 결측"],
  ["그래디언트 부스팅", "결측을 그대로 두고 분기에서 처리, 정규화 불필요. 개발 집단에서 한 번도 관측되지 않은 변수는 제외"],
  ["로지스틱 회귀 (비교)", "개발 집단 중앙값 대체 + 변수별 결측 지시자 + 개발 집단 평균·표준편차 표준화 (scikit-learn 파이프라인, 개발 집단에서만 적합)"],
  ["평가·외부 병원", "개발 집단에서 정한 값과 모델을 그대로 적용, 다시 맞추지 않음"],
  ["보고", "변수별 개발·평가 결측률, 중앙값·사분위, 대체값, 표준화 값 (preprocessing_variables.csv)"],
], [3, 7.8]));

body.push(H1("8. 모델", true));
body.push(H2("8.0 모델 4종 한눈에"));
body.push(table(["#", "모델", "실행 시점", "알고리즘", "파일"], [
  ["1", "조기 예측 (Agent 3)", "정상일 때 매시간", "그래디언트 부스팅 분류 (≥5×ULN, 24시간). 2·3·4·5배 비교 모델 별도", "agent3_gbm.joblib\nagent3_gbm_2x/3x/4x/5x.joblib"],
  ["2", "예후 (Agent 2)", "첫 이상 시점 1회", "그래디언트 부스팅 분류 2개: 4경로(회복 × ICU 사망), ICU 사망", "prognosis_gbm.joblib (두 모델 함께)"],
  ["3", "원인 (Agent 1) + 군집", "첫 이상 시점 1회", "원인 10개 규칙(학습 없음) + 보조 가우시안 혼합 군집 2개", "kcd_liver/rule_agent.py\ncluster_models.joblib"],
  ["4", "24시간 궤적", "매시간, 상태 무관", "분위 회귀 그래디언트 부스팅: 간 검사 4종 × 10·50·90% = 회귀 모델 12개, 예측 시점은 입력 변수", "trajectory_qgbm.joblib"],
  ["+", "48시간 첫 사건 (보조)", "첫 이상 시점 1회", "회복·ICU 사망·생존 퇴실 다항 분류 (경쟁 사건 분석용)", "agent2_gbm.joblib"],
], [0.4, 2.2, 1.8, 4.2, 2.8]));
body.push(H2("8.1 Agent 3: 24시간 조기 예측"));
body.push(L("시점: 정상 구간의 매시간 (세 항목 모두 24시간 안 측정, 모두 <2×)"));
body.push(L("목표: 다음 24시간 안에 하나라도 ≥5×ULN. 24시간 끝 6시간 안에 세 항목이 다시 측정되지 않았거나 그 전에 퇴실·사망하면 학습·평가에서 제외(정답 미확인)"));
body.push(L("모델: 그래디언트 부스팅(학습률·잎 수는 개발 집단 환자 단위 3겹 교차검증, AUPRC 기준), 로지스틱 회귀와 발생률 기준선을 함께 보고"));
body.push(L("비교: 같은 설정으로 2·3·4·5배 목표 각각 학습해 성능 표"));
body.push(L("설명: 9개 임상 그룹(현재 간 수치, 간 수치 변화, 측정 패턴, 관류·쇼크, 염증, 신장·대사, 합성 기능·혈액, 호흡, 배경)의 정확한 그룹 Shapley. 기준 행은 변수별 백분위로 만든 합성 행(실제 환자 행 아님)"));
body.push(H2("8.2 궤적 예측"));
body.push(L("시점: 상태와 무관하게 ICU 안 매시간(개발은 4시간 간격 시점)"));
body.push(L("목표: ALT·AST·ALP·TBIL의 6·12·18·24시간 뒤 log2 ULN 배수, 10·50·90% 분위"));
body.push(L("분위 교차 방지: 단조 재배열. 기준선: 마지막 값 유지와 오차 비교, 80% 구간 포함률 보고"));
body.push(H2("8.3 Agent 2: 예후"));
body.push(L("시점: 첫 이상 시점 1회"));
body.push(L("4경로 모델: 회복/미회복 × ICU 사망/생존 퇴실, 회복 판정 가능한 사건으로 학습·평가 (일대다 AUROC, 정확도, 혼동표)"));
body.push(L("ICU 사망 모델: 회복 판정 불가 사건까지 모든 사건으로 학습·평가"));
body.push(L("48시간 경쟁 사건: 회복·ICU 사망·생존 퇴실의 누적발생(Aalen-Johansen)과 단순 중도절단 비교"));
body.push(H2("8.4 Agent 1: 원인 10개 (규칙 기준 모델 v2.1)"));
body.push(P("원인마다 독립적으로 근거 점수를 더해 1.0 이상이면 1입니다. 여러 원인이 동시에 1일 수 있습니다. 근거는 스냅샷 안의 관측만 인용하므로 기계적으로 검증됩니다. 가중치는 평가 전에 고정했습니다."));
body.push(table(["원인", "주요 근거 (점수)"], [
  ["저산소성 간염", "AST/ALT ≥10×(1) 또는 5–10×(0.5) + MAP <65(1), 젖산 ≥4(1)/≥2(0.5), 승압제(1)"],
  ["울혈성 간병증", "간 수치 상승 + 이전 입원 심부전 코드(1.2), NT-proBNP 나이별 기준 이상(1.0; 450/900/1800 pg/mL, 나이 모르면 900)"],
  ["패혈증 유발 담즙정체", "TBIL ≥2× 또는 ALP ≥2×·R ≤2(0.5) + 발열·백혈구 이상 각 0.5, 혈소판 <100(0.5), 승압제(0.5)"],
  ["약물 유발 간손상", "간독성 약물의 실제 투여 기록, 약물당 0.9 (최대 2개)"],
  ["정맥영양 관련", "입원 중 정맥영양 노출 ≥7일(1)/3–7일(0.5) + 담즙정체·혼합 패턴(0.5)"],
  ["담도폐쇄·담관염", "TG18 의심: 담즙정체(TBIL ≥2 mg/dL 또는 효소 >1.5×) 0.5 + 전신 염증 0.5, 폐쇄 패턴 0.5, 직접 빌리루빈 ≥50% 0.25. 영상 없음"],
  ["담낭염", "구조화 자료에 국소 소견·영상이 없어 판단 불가 (항상 0, '판단 불가' 표시)"],
  ["용혈", "TBIL ≥1.5 mg/dL에서 간접 우세(직접 <30%) 1, 합토글로빈 <30 1, LDH ≥1.5× 0.5, 48시간 혈색소 ≥2 g/dL 감소 0.5"],
  ["바이러스 간염", "그 시각까지 보고된 혈청 양성: 활동성 표지(HBsAg, IgM, 바이러스량) 1.5, 노출 표지(HCV 항체, core 항체) 0.5, 만성 간염 과거력 0.5, ALT ≥10× 0.5"],
  ["기타", "근육 기원(CK ≥5×·AST ≥0.8×ALT) 1.5, 기저 간질환 1, 설명되지 않는 간세포형(R ≥5, ALT ≥3×) 1"],
], [2.6, 8.2]));
body.push(gap());
body.push(P("LLM을 쓰는 경우: 기관이 승인한 무보존 엔드포인트(환경변수 LLM_BASE_URL, LLM_MODEL)만 사용하고, 같은 10개 키를 0/1로 답하도록 요구합니다. 출력은 스키마 검증과 인용 근거 검증을 거칩니다. 설정이 없으면 규칙 모델로 자동 실행됩니다."));
body.push(H2("8.5 보조 군집"));
body.push(P("원인 판정에는 군집 번호가 아니라 고정 범주인 ACG 2017 R 비 패턴(간세포형 R ≥5, 담즙정체형 R ≤2, 그 사이 혼합형, 빌리루빈 단독)만 들어갑니다. 군집은 6.2의 변수로 따로 실행하고 안정성을 보고하며, 저장된 군집 모델은 외부 병원에서 같은 패턴이 나타나는지 확인하는 데 씁니다."));

body.push(H1("9. 평가"));
body.push(L("Agent 3·사망: AUROC, AUPRC, Brier, 보정 절편·기울기, 상위 10% 양성예측도·민감도. 95% 신뢰구간은 환자 단위 군집 부트스트랩(같은 환자의 시간들을 한 묶음으로)"));
body.push(L("Agent 3 민감도 분석: 평가 환자당 미리 정한 한 시점(입실 24시간 이후 첫 판정 가능 시점)"));
body.push(L("궤적: 평균 절대 오차(log2 배수) 대 마지막 값 유지, 80% 구간 포함률"));
body.push(L("원인 10개: 같은 입원의 퇴원 진단 코드로 만든 대리 범주와의 민감도·특이도(Wilson 구간). 대리 기준은 전문의 판정이 아니며, 정맥영양 관련은 해당 코드가 없어 대리 기준 없음"));
body.push(L("전문의 판정: 평가 사례 일부에 대해 모델 출력 없이 사전 판정하는 양식과 눈가림 자료를 제공 (결과는 판정 전까지 비워 둠)"));

body.push(H1("10. 빛과 그림자 실험 (전과 후)"));
body.push(table(["그림자", "전 (잘못된 방법)", "후 (적용한 방법)"], [
  ["데이터 누출", "시점을 환자 구분 없이 무작위 분할", "환자 단위 분할"],
  ["미보고 결과", "채혈 시각 기준 특징", "공개 시각 기준 특징"],
  ["측정 없음 = 정상", "이상 수치가 안 보이면 정상", "세 항목 측정된 시간만 판정, 나머지 '판정 불가'"],
  ["경쟁 사건", "퇴실·사망을 중도절단", "Aalen-Johansen 경쟁 위험"],
  ["알림 홍수", "기준 넘는 검사마다 알림", "환자당 첫 사건 알림"],
  ["지름길 학습", "간 수치를 본다고 가정", "그룹 Shapley로 '측정 패턴' 의존도 확인"],
  ["복잡한 모델의 가치", "모델만 보고", "마지막 값 유지 기준선과 비교"],
  ["잘못된 값 하나", "현재 파이프라인(일부 범위 검사만)", "값 검증 규칙(범위, 급변, 짝 효소 불일치) + 규칙 비용 보고"],
], [2.6, 4, 4.2]));
body.push(gap());
body.push(P("잘못된 값 실험은 평가 환자마다 정상 구간의 한 시각을 무작위로 정해 값 하나만 바꿉니다: 나이 ×10(19→190), ALT ×100, 빌리루빈 µmol/L 입력(×17.1), 크레아티닌 µmol/L 입력(×88.4), 젖산 ×10, 체온 °F를 °C 칸에, 체온 형식 오류(37..5), ALT 형식 오류(4..5). 오류 차단율, 가짜 이상 사건, Agent 1 판정 변화, 위험도·궤적 변화와 함께, 오류가 없는 실제 값을 잘못 보류한 비율(규칙의 비용)도 보고합니다. 젖산 ×10처럼 그럴듯한 범위 안의 오류는 규칙으로도 잡지 못한다는 한계를 그대로 보여줍니다."));

body.push(H1("11. 산출물"));
body.push(H2("11.1 결과 zip (kcd_return_package_*.zip)"));
body.push(table(["경로", "내용"], [
  ["site_data/results_full.json", "전체 결과 (10 미만 칸과 그 비율 가림)"],
  ["full_results_tables/", "기초 특성표, 성능표, 흐름도, 2–5배 기본 값, Agent 3 기준별 성능, ICU 사망 성능, 전처리 표, 전과 후 요약"],
  ["full_figures/", "집계 그림 PNG"],
  ["full_aggregates/", "환자 식별 열이 없는 모든 집계 CSV, 사이트 집계, 군집 점검, 실행 설정"],
  ["site_data/cases/", "공개 데모 환자 시연 화면"],
  ["models/, MODEL_CARD.md, feature_spec.json", "학습된 모델(4종 + 2·3·4배 비교 모델 + 군집), 모델 카드, 변수 명세"],
  ["VARIABLES.md, run_report.json", "변수 설명, 단계·셀별 실행 시간과 오류"],
], [3.6, 7.2]));
body.push(H2("11.2 모델 패키지와 외부 검증"));
body.push(P("kcd_model_package_*.zip에는 모델, 코드, KCD_EXTERNAL.ipynb, 모델 카드, requirements.txt(학습 때 버전 고정)가 들어 있습니다. 외부 병원은 MIMIC-IV와 같은 폴더·열 구조의 자료를 준비하고 노트북을 실행하면, 다시 학습하지 않고 고정 모델의 성능(조기 예측, 예후·사망, 궤적, 원인 대리 기준 일치, 군집 이식)을 계산해 집계 zip을 돌려줍니다. itemid가 다르면 대응표를 먼저 맞춥니다."));
body.push(H2("11.3 남기지 않고 보내지 않는 것"));
body.push(L("kcd_work/: 원자료와 환자 단위 중간 파일, 실행된 노트북 사본. 이 PC에만"));

body.push(H1("12. 시연 사이트 구성"));
newList();
body.push(N("논문형 앞부분: 흐름도, 기초 특성표, 모델 개발 과정, 성능(Table 2, 2B, 3, 4, 그림)"));
body.push(N("시연: 공개 데모 환자를 매시간 재생. Agent 3 위험과 근거, 24시간 궤적 띠, 첫 이상 시 원인 10개 0/1, 예후 4경로와 ICU 사망 확률"));
body.push(N("빛과 그림자: 전과 후 카드(뒤집기), 환자 한 명으로 보는 잘못된 값 실험, 경쟁 사건, 근거 검증, 알림 통합"));
body.push(N("근거와 재현: 검증 항목, 참고문헌, 실행 방법"));

body.push(H1("13. 공개와 보안"));
body.push(L("공개 사이트: MIMIC-IV Demo(ODbL)만 환자 단위, 전체 MIMIC-IV는 집계만"));
body.push(L("소표본 가림: 10 미만 칸과 같은 행의 비율·신뢰구간을 함께 가림 (가린 수를 비율로 역산할 수 없게)"));
body.push(L("공개 전 자동 점검: 금지 파일(*_private, *.sqlite, EVALUATION_ONLY), 데모가 아닌 환자 화면, 가림 없는 전체 결과를 막음"));
body.push(L("LLM: PhysioNet 지침에 따라 무보존·무학습이 확인된 기관 엔드포인트만. 없으면 규칙 모델"));
body.push(L("모델 공유: 원자료를 제외한 결과·모델 공유는 팀이 승인. 모델 파일에는 실제 환자 행을 넣지 않음(Shapley 기준 행은 합성)"));

body.push(H1("14. 한계"));
body.push(L("원인 정답이 없습니다. 퇴원 진단 코드는 대리 기준이며 전문의 판정 전까지 원인 성능은 참고값입니다."));
body.push(L("노트·영상·배양 결과를 쓰지 않아 담낭염은 판단하지 않고 담관염은 의심 수준까지만 판정합니다."));
body.push(L("Agent 3은 재검으로 결과가 확인된 시점만 평가하므로 측정 빈도에 따른 선택 편향이 남습니다."));
body.push(L("ICU 사망 판정은 퇴원 후 사망의 경우 날짜 단위라 최대 하루 정도 더 포함될 수 있습니다."));
body.push(L("실시간 운영이 아니라 과거 자료 재생입니다. 실제 배포에는 전향적 검증이 필요합니다."));
body.push(L("한 기관(MIMIC-IV) 개발이므로 외부 검증 전까지 다른 병원 성능은 알 수 없습니다."));

body.push(H1("15. 실행 방법"));
newList();
body.push(N("pip install pandas numpy scikit-learn matplotlib joblib nbformat nbclient ipykernel"));
body.push(N("KCD_ONE.ipynb를 아무 폴더에 두고 엽니다."));
body.push(N("첫 셀의 MIMIC_ROOT(hosp/, icu/가 있는 폴더)를 확인합니다. 인터넷이 막힌 PC면 DEMO_FHIR_DIR에 미리 받은 FHIR 데모 폴더를 적습니다."));
body.push(N("위에서부터 끝까지 실행합니다. 진행 상황이 출력됩니다."));
body.push(N("노트북 옆의 kcd_return_package_*.zip을 전달합니다. 실패한 단계가 있어도 zip이 만들어지고 이유가 run_report.json에 적힙니다."));
body.push(N("새 버전을 받았을 때 추출 항목이 바뀌지 않았다면 RUN_STEPS['full']=False로 가장 오래 걸리는 1단계를 건너뛸 수 있습니다."));

const doc = new Document({
  creator: "KCD 2026 team", title: "ICU 간기능 이상 감시 에이전트 전체 명세서",
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
                                 { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 240 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 440, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "KCD 2026 · ICU 간기능 감시 에이전트 명세서 · ", size: 16, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(process.argv[2], b); console.log("wrote", process.argv[2]); });
