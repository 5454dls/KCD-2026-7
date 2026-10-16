const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, BorderStyle, Footer, PageNumber, TableOfContents, PageBreak, ImageRun,
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
const [, , OUTF, JSONF, FIGDIR] = process.argv;
const R = JSON.parse(fs.readFileSync(JSONF, "utf8"));
const f3 = (v, d = 3) => v == null ? "–" : Number(v).toFixed(d);
const pc = (v, d = 0) => v == null ? "<10" : (v * 100).toFixed(d) + "%";
const n = (v) => v == null ? "<10" : Number(v).toLocaleString("en-US");
const ci = (o) => o && o.ci ? `${o.difference >= 0 ? "+" : ""}${o.difference.toFixed(3)} (${o.ci[0].toFixed(3)} ~ ${o.ci[1].toFixed(3)})` : "–";
function img(name, widthCm, caption) {
  const b = fs.readFileSync(path.join(FIGDIR, name + ".png"));
  const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  const px = widthCm / 2.54 * 96;
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
            children: [new ImageRun({ type: "png", data: b, transformation: { width: Math.round(px), height: Math.round(px * h / w) } })] }),
          new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: caption, size: 18, color: MUTED })] })];
}
const S = R.selected, PR = R.profiles, AV = R.added_value;
const NAME = ["허혈성 간세포 손상형", "경도 혼합형", "순수 담즙정체형", "황달 동반 혼합형", "기저 간질환형"];
const ev = Object.fromEntries(R.outcomes.evaluation.by_cluster.map(r => [r.group, r]));
const dv = Object.fromEntries(R.outcomes.development.by_cluster.map(r => [r.group, r]));
const V = {}; for (const r of R.characterization.variables) V[`${r.cluster}|${r.variable}`] = r;
const med = (k, v, d = 1) => { const r = V[`${k}|${v}`]; if (!r) return "–"; if (r.binary) return r.share == null ? "<10" : pc(r.share); return Number(r.median).toFixed(d); };
const body = [];

body.push(new Paragraph({ spacing: { before: 2000, after: 200 }, children: [new TextRun({ text: "KCD 2026 Datathon", color: ACC, bold: true, size: 26 })] }));
body.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "첫 간 검사 이상 사건의 군집 분석", bold: true, size: 44 })] }));
body.push(new Paragraph({ spacing: { after: 500 }, children: [new TextRun({ text: "개선판 결과 보고 (전체 MIMIC-IV)", size: 28, color: MUTED })] }));
body.push(P_([B("대상  "), `ICU 중 첫 이상 사건(ALT·ALP·총빌리루빈 중 하나라도 ≥2×ULN) 개발 ${n(R.n_dev)}건, 평가 ${n(R.n_eval)}건`]));
body.push(P_([B("공개 기준  "), `모든 수는 집계이며 ${R.min_cell_suppression} 미만 칸은 가렸습니다.`]));
body.push(P_([B("성격  "), "후향적 탐색 분석입니다. 군집은 원인 진단이 아닙니다."]));
function P_(t) { return P(t); }

body.push(H1("1. 요약", true));
body.push(L(`군집 수는 미리 정한 규칙으로 K=${S.k}(완전 공분산)가 선택되었습니다. 부트스트랩 ARI 중앙값 ${f3(S.ari_median, 2)}로 안정적이고, 가장 작은 군집도 ${n(S.smallest)}건입니다.`));
body.push(L("다섯 군집은 임상적으로 해석됩니다: 허혈성 간세포 손상형, 경도 혼합형, 순수 담즙정체형, 황달 동반 혼합형, 기저 간질환형."));
body.push(L("개발 집단과 평가 집단에서 군집별 ICU 사망률과 회복률이 거의 같아 재현됩니다."));
body.push(L(`R 비 패턴과 비교하면 군집이 예후 판별을 크게 올리는 것처럼 보이지만(회복 AUROC +${(AV.recovered.cluster_vs_pattern.difference).toFixed(3)}), 수치 크기를 함께 넣은 공정 비교에서는 더하는 정보가 미미했습니다(ICU 사망 ${ci(AV.icu_death.cluster_added_to_severity)}, 회복 ${ci(AV.recovered.cluster_added_to_severity)}).`));
body.push(L("결론: 군집은 환자 유형을 사람이 이해할 수 있게 요약하는 도구로는 쓸 만하지만, 예후를 더 잘 맞히는 도구는 아닙니다."));

body.push(H1("2. 방법"));
body.push(table(["항목", "내용"], [
  ["입력 변수", "첫 사건 시점의 ALT·AST·ALP·총빌리루빈 ULN 배수 4개. 모두 log2(최소 0.05)로 바꾸고 개발 집단 평균·표준편차로 표준화. R 비는 log ALT − log ALP와 같아 중복이므로 제외"],
  ["결측", "AST가 드물게 빠진 경우 개발 집단 중앙값으로 채움. 해석용 변수는 채우지 않고 측정된 값만 요약"],
  ["모형", "가우시안 혼합 모형. 대각·완전 공분산 × 군집 수 2-12, 22개 조합"],
  ["선택 규칙 (사전 고정)", "부트스트랩 30회 ARI 중앙값 ≥ 0.75이고 가장 작은 군집 ≥ 30건인 조합 중, 최저 BIC에서 1% 안에 드는 가장 작은 K"],
  ["학습과 적용", "개발 집단에서만 학습, 평가 집단에는 같은 모형을 그대로 적용"],
  ["해석", "정리된 모든 변수(나이, 성별, 승압제, 이전 진단, 검사 최신값, 24시간 변화)에 대해 군집 대 나머지 표준화 차이(SMD)"],
  ["예후", "ICU 사망(퇴실 후 24시간 포함), ICU 퇴실 시 회복(판정 가능한 사건 중). 95% Wilson 신뢰구간"],
  ["추가 정보 검정", "평가 집단 AUROC. 로지스틱 회귀 4개: 패턴, 패턴+수치 크기(log2 4개), 패턴+군집, 셋 모두. 차이는 부트스트랩 500회 95% CI"],
  ["이전 실행과 달라진 점", "이전 실행은 ALT에 0 값이 섞여 로그 변환이 빠졌고, 간세포형이 수치 크기만으로 5개로 쪼개졌습니다. 이번에는 네 변수를 모두 같은 방식으로 변환했습니다"],
], [2.2, 7]));

body.push(H1("3. 군집 수 선택"));
body.push(...img("c1_selection", 14, "그림 A. 군집 수와 BIC. 채운 점은 부트스트랩 ARI 0.75 이상(안정), 별은 선택된 모형."));
body.push(table(["공분산", "K", "BIC", "ARI 중앙값", "가장 작은 군집"], R.grid.filter(g => g.k <= 12).map(g => [g.covariance === "full" ? "완전" : "대각", String(g.k), Math.round(g.bic).toLocaleString("en-US"), f3(g.ari_median, 2), n(g.smallest)]), [1.5, 1, 2, 1.5, 2]));
body.push(P("K가 8 이상이면 BIC는 더 낮지만 안정성이 0.58-0.71로 떨어집니다. 표본을 바꿀 때마다 군집이 달라진다는 뜻이라 선택하지 않았습니다."));

body.push(H1("4. 다섯 군집의 정체"));
body.push(...img("c2_profiles", 15, "그림 B. 군집별 간 수치 중앙값 (개발 집단, ×ULN)."));
body.push(table(["군집", "해석 이름", "개발 / 평가", "ALT", "AST", "ALP", "TBIL", "R 비 주 패턴"], PR.map(p => [
  `C${p.cluster}`, NAME[p.cluster] || p.name, `${n(p.n_dev)} / ${n(p.n_eval)}`, f3(p.median.ALT_ratio, 1), f3(p.median.AST_ratio, 1), f3(p.median.ALP_ratio, 1), f3(p.median.TBIL_ratio, 1),
  `${({ hepatocellular: "간세포형", mixed: "혼합형", cholestatic: "담즙정체형", isolated_bilirubin: "빌리루빈 단독", unclassifiable: "분류 불가" })[p.main_pattern]} ${pc(p.main_pattern_share)}`]), [0.8, 2.2, 1.6, 0.9, 0.9, 0.9, 0.9, 1.9]));
body.push(gap());
body.push(table(["군집", "나이", "젖산", "INR", "혈소판", "크레아티닌", "LDH", "혈색소", "알부민", "이전 간질환", "관찰 중 신규"], PR.map(p => [
  `C${p.cluster}`, med(p.cluster, "age", 0), med(p.cluster, "LACTATE_latest"), med(p.cluster, "INR_latest"), med(p.cluster, "PLATELET_latest", 0), med(p.cluster, "CREAT_latest"),
  med(p.cluster, "LDH_latest", 0), med(p.cluster, "HEMOGLOBIN_latest"), med(p.cluster, "ALBUMIN_latest"), med(p.cluster, "history_liver"), med(p.cluster, "group_incident")]), [0.8, 0.8, 0.8, 0.8, 1, 1.1, 0.9, 0.9, 0.9, 1.1, 1.1]));
body.push(P([new TextRun({ text: "연속형은 중앙값, 이전 간질환과 관찰 중 신규는 비율. 이전 간질환은 이전 입원 진단이 있는 사건 중 비율입니다.", size: 18, color: MUTED })]));
body.push(H2("군집별 해석"));
const desc = [
  "AST가 ALT보다 높고 ALP는 정상이며 젖산·LDH·크레아티닌이 높고 pH가 낮습니다. 저산소성 간염(쇼크 간)의 전형에 가깝습니다. 사망이 가장 많고 회복은 드뭅니다.",
  "가장 흔한 군집입니다. 효소가 3-4배로 가볍게 오르고 빌리루빈은 정상이며 INR도 낮습니다. 회복이 가장 많습니다.",
  "ALP만 오르고 효소는 정상입니다. 젖산이 낮고 혈색소·알부민이 낮습니다. 사망이 가장 적습니다.",
  "빌리루빈 4배에 ALP·AST가 함께 오릅니다. INR이 높고 혈소판이 낮으며 입실 초기에 이미 이상인 경우가 많습니다. 회복이 가장 드뭅니다.",
  "빌리루빈만 2.5배 오르고 효소는 정상이며 혈소판이 가장 낮고 이전 간질환 비율이 가장 높습니다. 기저 간경변 악화에 가깝습니다.",
];
PR.forEach(p => body.push(L([B(`C${p.cluster} ${NAME[p.cluster]}. `), desc[p.cluster] || ""])));
body.push(P("같은 빌리루빈 상승이라도 효소를 동반한 C3(회복 10%)와 기저 간질환형 C4(회복 33%)로 나뉩니다. R 비 패턴은 이 둘을 구분하지 못합니다. 자동 이름표는 C3와 C4가 모두 '빌리루빈 단독'이지만, C3에서 실제 빌리루빈 단독 패턴은 36%뿐이라 해석 이름을 따로 붙였습니다."));
body.push(...img("c3_map", 12, "그림 C. ALT·ALP 평면의 군집. 회색은 사건 밀도(10 미만 칸 제외), 타원은 군집의 50% 영역, 점선 사선은 R=5와 R=2."));
body.push(...img("c6_characterization", 13, "그림 F. 군집을 가르는 변수. 군집 대 나머지의 표준화 차이(주황 = 더 높음, 보라 = 더 낮음)."));

body.push(H1("5. 군집별 예후"));
body.push(...img("c4_outcomes", 15, "그림 D. 평가 집단의 군집별 ICU 사망과 ICU 퇴실 시 회복 (95% Wilson CI)."));
body.push(table(["군집", "ICU 사망 개발", "ICU 사망 평가", "회복 개발", "회복 평가"], PR.map(p => {
  const d = dv[p.cluster], e = ev[p.cluster];
  return [`C${p.cluster} ${NAME[p.cluster]}`, pc(d.icu_death.rate, 1), pc(e.icu_death.rate, 1), pc(d.recovered.rate, 1), pc(e.recovered.rate, 1)];
}), [3, 1.5, 1.5, 1.5, 1.5]));
body.push(P("개발과 평가 집단의 값이 거의 같아 군집별 예후 차이는 재현됩니다."));

body.push(H1("6. 군집이 예후 정보를 더하는가"));
body.push(...img("c5_added_value", 12, "그림 E. 평가 집단 AUROC. 회색 = R 비 패턴, 연한 파랑 = 패턴 + 수치 크기, 파랑 = 패턴 + 군집, 검정 = 셋 모두."));
const row = (k, lab) => [lab, f3(AV[k].auroc_pattern), f3(AV[k].auroc_pattern_severity), f3(AV[k].auroc_pattern_cluster), f3(AV[k].auroc_pattern_severity_cluster)];
body.push(table(["결과", "패턴", "패턴 + 수치", "패턴 + 군집", "셋 모두"], [row("icu_death", `ICU 사망 (${n(AV.icu_death.events)}/${n(AV.icu_death.n)})`), row("recovered", `퇴실 시 회복 (${n(AV.recovered.events)}/${n(AV.recovered.n)})`)], [3, 1.5, 1.5, 1.5, 1.5]));
body.push(gap());
body.push(table(["비교 (평가 집단 AUROC 차이, 95% CI)", "ICU 사망", "퇴실 시 회복"], [
  ["군집 vs 패턴만", ci(AV.icu_death.cluster_vs_pattern), ci(AV.recovered.cluster_vs_pattern)],
  ["군집 vs 수치 크기", ci(AV.icu_death.cluster_vs_severity), ci(AV.recovered.cluster_vs_severity)],
  ["수치 크기에 군집을 더하면", ci(AV.icu_death.cluster_added_to_severity), ci(AV.recovered.cluster_added_to_severity)],
], [3.5, 2.5, 2.5]));
body.push(L("패턴만과 비교하면 군집이 크게 좋아 보입니다. 그러나 회복은 '퇴실 때 2×ULN 미만'으로 정의되어 처음 수치가 클수록 기계적으로 회복이 어렵습니다. 그래서 수치 크기를 함께 넣어 비교했습니다."));
body.push(L("수치 크기와 비교하면 회복은 수치 크기 쪽이 더 좋고(군집이 0.026 낮음), 사망은 차이가 유의하지 않습니다."));
body.push(L("수치 크기에 군집을 더하면 두 결과 모두 약 0.01 오릅니다. 통계적으로 유의하지만 임상적으로는 미미합니다."));
body.push(L("간 수치만으로 ICU 사망을 판별하는 능력은 AUROC 0.59 수준입니다. 모든 변수를 쓴 ICU 사망 모델(0.83)과 비교하면, 사망은 간 수치 모양보다 전신 상태가 결정합니다."));

body.push(H1("7. 한계와 다음 단계"));
body.push(L("군집은 원인 진단이 아닙니다. 규칙 원인 판정(Agent 1)과 군집의 연결은 이번 실행에 원인 판정 기록이 붙지 않아 확인하지 못했습니다."));
body.push(L("간 수치 4개만 썼습니다. 임상 표현형 군집(20개 변수)은 이전 실행에서 불안정했습니다(ARI 0.34)."));
body.push(L("한 기관 자료입니다. 모델 묶음의 군집 모형을 외부 기관에 그대로 적용해 군집 비율과 군집별 예후가 재현되는지 확인할 수 있습니다."));
body.push(L("발표에서는 '패턴만과 비교하면 좋아 보이지만 공정하게 비교하면 대부분 수치 크기 효과'라는 결과를 그림자 사례로 쓸 수 있습니다."));

const doc = new Document({
  creator: "KCD 2026 team", title: "첫 간 검사 이상 사건의 군집 분석",
  styles: {
    default: { document: { run: { font: { ascii: FONT, eastAsia: FONT, hAnsi: FONT }, size: 20, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: ACC, font: { ascii: FONT, eastAsia: FONT, hAnsi: FONT } }, paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: INK, font: { ascii: FONT, eastAsia: FONT, hAnsi: FONT } }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
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
      children: [new TextRun({ text: "KCD 2026 · 군집 분석 보고 · ", size: 16, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUTF, b); console.log("wrote", OUTF); });
