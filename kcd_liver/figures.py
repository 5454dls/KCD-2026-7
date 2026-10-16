"""Paper-style matplotlib figures from the site aggregates dict.

    from kcd_liver.figures import render_all
    paths = render_all(aggregates, OUT / "figures", show=True)

Each figure mirrors the web page (Fig 2-10) so that the notebook shows, step by
step, whether the pipeline did what it should. Missing inputs produce a
labelled placeholder instead of an invented plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
from matplotlib import font_manager

PALETTE = {"ALT": "#0e7a84", "ALP": "#6a4cb0", "TBIL": "#c0850b", "res": "#2e7a4c", "death": "#b53636",
           "dis": "#3d6fb0", "muted": "#56686d", "accent": "#a8660e", "crit": "#b53636", "ok": "#2e7a4c", "line": "#d3dedb"}
GROUP_KO = {"initial_abnormal": "입실 초기", "incident": "신규", "onset_unknown": "시점 불명"}
PATH_KO = {"recovered_alive_1y": "회복 · 1년 생존",
           "unrecovered_alive_1y": "회복 미확인 · 1년 생존",
           "unrecovered_died_1y": "회복 미확인 · 1년 내 사망", "recovered_died_1y": "회복 · 1년 내 사망"}
STEP_KO = {"all_icu_stays": "전체 ICU 입실", "first_icu_per_patient": "환자별 첫 ICU 입실", "adult_valid_stay": "성인·유효 입실",
           "analysis_sample": "분석 표본", "first_abnormal_event": "첫 이상 사건"}
COND_KO = {"claim": "잘못된 주장", "rationale": "잘못된 설명", "combined": "주장 + 설명"}


def _style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR",
                          "Noto Sans KR"] if f in have]
    matplotlib.rcParams.update({
        "font.family": korean + ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": PALETTE["line"],
        "grid.linewidth": .6, "axes.axisbelow": True, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def _na(ax, text):
    ax.axis("off")
    ax.text(.5, .5, f"아직 집계되지 않음\n{text}", ha="center", va="center", color=PALETTE["muted"], fontsize=10)


def _stamp(fig, agg):
    if agg.get("source") == "synthetic":
        fig.text(.99, .99, "합성 예시 · 실제 결과 아님", ha="right", va="top", color="#ad6300", fontsize=8,
                 bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#ad6300"))
    if agg.get("min_cell_suppression"):
        fig.text(.01, .005, f"n<{agg['min_cell_suppression']} 칸 가림", fontsize=7, color=PALETTE["muted"])


def fig2_flow(agg, ax):
    rows = agg.get("cohort_flow")
    if not rows:
        return _na(ax, "04_cohort_flow.csv")
    ax.axis("off")
    n = len(rows)
    for i, r in enumerate(rows):
        y = 1 - (i + .5) / n
        count = f"{r['n']:,}" if isinstance(r["n"], int) else "<10"
        ax.text(.5, y, f"{STEP_KO.get(r['step'], r['step'])}\nn = {count}", ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=.5", fc="#e9efed", ec=PALETTE["line"]))
        if i < n - 1:
            ax.annotate("", xy=(.5, 1 - (i + 1.5) / n + .06), xytext=(.5, y - .06),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]))
            nxt = rows[i + 1]["n"]
            if isinstance(r["n"], int) and isinstance(nxt, int):
                ax.text(.56, y - .5 / n, f"− {r['n'] - nxt:,}", fontsize=8, color=PALETTE["muted"], va="center")
    ax.set_title("Figure 2. 코호트 흐름")


def fig3_qc(agg, ax):
    q = agg.get("lab_qc")
    if not q:
        return _na(ax, "02_lab_quality.csv")
    q = q[::-1]
    names = [r["analyte"] for r in q]
    ax.barh(names, [r["n"] or 0 for r in q], color=[PALETTE.get(a, PALETTE["muted"]) for a in names])
    for i, r in enumerate(q):
        if r["n"]:
            share = lambda k: "<10건" if r.get(k) is None else f"{r[k] / r['n']:.1%}"  # small cells are suppressed
            ax.text(r["n"], i, f"  단위오류 {share('unit_invalid')} · ULN없음 {share('missing_uln')}",
                    va="center", fontsize=7, color=PALETTE["muted"])
    ax.set_xlabel("추출 건수")
    ax.set_title("Figure 3. 검사 품질 관리")


def fig4_events(agg, axes):
    a, b = axes
    eg = agg.get("event_groups")
    if eg:
        splits = ["development", "evaluation"]
        left = [0, 0]
        for g, col in zip(GROUP_KO, [PALETTE["crit"], PALETTE["accent"], PALETTE["muted"]]):
            vals = [next((r["n"] or 0 for r in eg if r["split"] == s and r["group"] == g), 0) for s in splits]
            a.barh(["개발", "평가"], vals, left=left, color=col, label=GROUP_KO[g])
            left = [x + y for x, y in zip(left, vals)]
        a.legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
        a.set_title("Figure 4a. 첫 이상 사건 집단")
    else:
        _na(a, "05_events_private.csv")
    h = agg.get("trigger_ratio_hist")
    if h:
        bins = list(dict.fromkeys(r["bin"] for r in h))
        w = .27
        for j, an in enumerate(["ALT", "ALP", "TBIL"]):
            vals = [next((r["count"] or 0 for r in h if r["bin"] == bn and r["analyte"] == an), 0) for bn in bins]
            b.bar([i + (j - 1) * w for i in range(len(bins))], vals, width=w, color=PALETTE[an], label=an)
        b.set_xticks(range(len(bins)), [x + "×" for x in bins])
        b.set_xlabel("사건 시점 ULN 배수")
        b.legend(frameon=False, fontsize=7)
        b.set_title("Figure 4b. 사건 크기 분포")
    else:
        _na(b, "trigger_ratio")


PATTERN_KO = {"hepatocellular": "간세포형", "mixed": "혼합형", "cholestatic": "담즙정체형", "isolated_bilirubin": "빌리루빈 단독"}


def fig5_clusters(agg, ax):
    cp = agg.get("cluster_profile")
    if not cp or not cp.get("rows"):
        return _na(ax, "08_features.csv (cluster)")
    feats = cp["features"]
    z = [[(cp["z"][i].get(f) or 0) for f in feats] for i in range(len(cp["rows"]))]
    im = ax.imshow(z, cmap="BrBG_r", vmin=-2, vmax=2, aspect="auto")
    for i, r in enumerate(cp["rows"]):
        for j, f in enumerate(feats):
            v = r.get(f)
            ax.text(j, i, "–" if v is None else f"{v:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(feats)), [f.replace("_ratio", "").replace("_slope24", "\n기울기").replace("_delta24", "\nΔ") for f in feats], fontsize=7)
    ax.set_yticks(range(len(cp["rows"])), [(f"{PATTERN_KO.get(r['cluster'], r['cluster'])}" if cp.get("grouping") == "liver_pattern" else f"군집 {r['cluster']}") + f" (n={r['n'] if r['n'] is not None else '<10'})" for r in cp["rows"]])
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.04, label="패턴 간 z")
    ck = agg.get("clustering_check") or {}
    note = f" · GMM K={ck.get('k_selected')}, ARI {ck.get('ari_median', float('nan')):.2f}, {'사용' if ck.get('used') else '불안정, 미사용'}" if ck.get("status") == "ok" else ""
    ax.set_title("Figure 5. 간 손상 패턴 (ACG R 비, 개발 집단 중앙값)" + note, fontsize=9)


def fig6_validation(agg, ax):
    cv = agg.get("cause_validation")
    if not cv:
        return _na(ax, "12_cause_validation.csv (전문의 판정 후)")
    for i, r in enumerate(cv):
        if r.get("rate") is None:
            continue
        c = PALETTE["ALT"] if r["condition"] == "cluster" else PALETTE["muted"]
        ax.errorbar(r["rate"], i, xerr=[[r["rate"] - r["ci_low"]], [r["ci_high"] - r["rate"]]], fmt="s", color=c, capsize=3)
    ax.set_yticks(range(len(cv)), [f"{'Top-1' if r['metric'] == 'top1' else 'Top-3'} · {'+패턴' if r['condition'] == 'cluster' else '기본'}  {r.get('k')}/{r.get('n')}" for r in cv])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    p = agg.get("paired_test")
    if p:
        ax.set_xlabel(f"일치율 (95% Wilson CI) · McNemar 불일치 {p['discordant']}쌍, 정확 p={p['p_exact_two_sided']:.3f}")
    ax.set_title("Figure 6. 전문의 참조 대비 원인 후보 일치")


def fig7_outcomes(agg, axes):
    a, b = axes
    op = agg.get("outcome_paths")
    if op:
        op = op[::-1]
        cols = [PALETTE["death"] if "death" in r["path"] else PALETTE["res"] if r["path"].startswith("resolved_then_hosp") else PALETTE["dis"] for r in op]
        a.barh([PATH_KO.get(r["path"], r["path"]) for r in op], [r["n"] or 0 for r in op], color=cols)
        a.set_title("Figure 7a. 관찰 경로 (ICU 퇴실·ICU 사망까지)")
    else:
        _na(a, "10_outcomes_EVALUATION_ONLY.csv")
    cif = agg.get("cif_development")
    if cif:
        g = cif["grid_hours"]
        b.step(g, cif["cif"]["resolution"], where="post", color=PALETTE["res"], label="해소 (경쟁위험)")
        b.step(g, cif["naive_resolution"], where="post", color=PALETTE["res"], ls="--", label="해소 (단순 중도절단, 1−KM)")
        b.step(g, cif["cif"]["death"], where="post", color=PALETTE["death"], label="사망")
        b.step(g, cif["cif"]["icu_discharge"], where="post", color=PALETTE["dis"], label="ICU 퇴실")
        b.set_xlabel("첫 이상 사건 이후 시간 (h)")
        b.set_ylabel("누적발생")
        b.set_ylim(0, 1)
        b.legend(frameon=False, fontsize=7, loc="upper left")
        b.set_title(f"Figure 7b. Aalen-Johansen 누적발생 (개발 n={cif['n']})")
    else:
        _na(b, "risk_reference")


def fig8_agent3(agg, ax):
    a3 = agg.get("agent3")
    if not a3 or not a3.get("calibration"):
        return _na(ax, "22_agent3_predictions.csv (BUILD_AGENT3_REFERENCE)")
    cal = [r for r in a3["calibration"] if r.get("observed") is not None]
    mx = max(.5, max(max(r["ci_high"], r["mean_predicted"]) for r in cal))
    ax.plot([0, mx], [0, mx], ls=":", color=PALETTE["muted"])
    for r in cal:
        ax.errorbar(r["mean_predicted"], r["observed"], yerr=[[r["observed"] - r["ci_low"]], [r["ci_high"] - r["observed"]]],
                    fmt="o", color=PALETTE["ALP"], ms=3 + min(6, (r["n"] or 0) ** .5 / 4), capsize=2)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("예측 위험")
    ax.set_ylabel("관찰 비율")
    f = lambda v: "–" if v is None else f"{v:.3f}"
    ax.text(.98, .04, f"AUROC {f(a3['auroc'])}\nAUPRC {f(a3['auprc'])}\nBrier {f(a3['brier'])}\n확인 {a3['n_known']}/{a3['n_all']}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_title("Figure 8. 정상 환자 24시간: 보정")


def fig9_stress(agg, ax):
    sa = agg.get("stress_adoption")
    if not sa:
        return _na(ax, "17_input_stress.jsonl")
    for i, c in enumerate(["claim", "rationale", "combined"]):
        for j, g in enumerate([False, True]):
            r = next((x for x in sa if x["input_condition"] == c and x["guarded"] == g), None)
            if not r or r.get("adoption_rate") is None:
                continue
            y = i + (j - .5) * .3
            col = PALETTE["ok"] if g else PALETTE["crit"]
            ax.errorbar(r["adoption_rate"], y, xerr=[[r["adoption_rate"] - r["ci_low"]], [r["ci_high"] - r["adoption_rate"]]],
                        fmt="o", color=col, capsize=3, label=("방어 문구" if g else "방어 없음") if i == 0 else None)
            ax.text(r["ci_high"] + .02, y, f"{r['adopted']}/{r['n']}", va="center", fontsize=7)
    ax.set_yticks(range(3), [COND_KO[c] for c in ["claim", "rationale", "combined"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel(f"주입 원인({agg.get('injected_cause')})이 1순위가 된 비율 (95% CI)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Figure 9. 잘못된 입력 취약성과 방어")


def fig10_alerts(agg, axes):
    a, b = axes
    al = agg.get("alert_aggregation")
    if al:
        a.barh(["환자별 첫 사건 알림", "기준 이상 검사 기록"],
               [next((r["n"] or 0 for r in al if r["comparison"] == k), 0) for k in ["first_patient_event_alerts", "raw_abnormal_lab_records"]],
               color=[PALETTE["accent"], PALETTE["muted"]])
        a.set_title("Figure 10a. 알림 통합 (기계적 건수)")
    else:
        _na(a, "18_alert_aggregation.csv")
    th = agg.get("threshold_sensitivity")
    if th:
        b.bar([f"{r['cutoff_fold_uln']}×" for r in th], [r["n_patients_flagged"] or 0 for r in th],
              color=[PALETTE["crit"] if r["cutoff_fold_uln"] == 2 else PALETTE["muted"] for r in th])
        b.set_ylabel("알림 대상 환자 수")
        b.set_title("Figure 10b. 임계값 민감도")
    else:
        _na(b, "33_threshold_sensitivity.csv")


def render_all(agg: dict, out_dir: Path, show: bool = False) -> list[Path]:
    _style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [("fig02_cohort_flow", (5, 5), 1, fig2_flow), ("fig03_lab_qc", (7, 4), 1, fig3_qc),
            ("fig04_events", (10, 3.6), 2, fig4_events), ("fig05_clusters", (8, 3.6), 1, fig5_clusters),
            ("fig06_validation", (7, 3.2), 1, fig6_validation), ("fig07_outcomes", (11, 3.8), 2, fig7_outcomes),
            ("fig08_agent3", (4.6, 4.2), 1, fig8_agent3), ("fig09_stress", (7, 3.2), 1, fig9_stress),
            ("fig10_alerts", (10, 3.2), 2, fig10_alerts)]
    paths = []
    for name, size, ncols, fn in plan:
        fig, axes = plt.subplots(1, ncols, figsize=size)
        try:
            fn(agg, axes if ncols > 1 else axes)
        except Exception as ex:  # one broken figure must not stop the run; the reason is printed on the figure
            for a in (axes if ncols > 1 else [axes]):
                a.clear()
            _na(axes[0] if ncols > 1 else axes, f"{name}: {type(ex).__name__}")
            print(f"figure {name} failed: {type(ex).__name__}: {ex}")
        _stamp(fig, agg)
        fig.tight_layout()
        p = out_dir / f"{name}.png"
        fig.savefig(p)
        paths.append(p)
        if show:
            plt.show()
        plt.close(fig)
    return paths
