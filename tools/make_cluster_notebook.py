"""Build notebooks/KCD_CLUSTER.ipynb: improved clustering on the outputs of an earlier KCD_ONE run (code embedded)."""
import base64
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
mods = {p.name: base64.b64encode(p.read_bytes()).decode() for p in sorted((ROOT / "kcd_liver").glob("*.py")) if p.name != "__init__.py"}
demo = (ROOT / "reference" / "mimic_iv_demo_subject_ids.csv").read_text(encoding="utf-8")
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
cells = [md("""# KCD 2026 · 개선된 클러스터링

KCD_ONE.ipynb를 이미 실행한 PC에서, **그 노트북과 같은 폴더**에 두고 위에서부터 실행합니다. 원자료를 다시 읽지 않고 1단계 산출물(`kcd_work/run_all/full/kcd_liver_v2_outputs`)만 씁니다. 원인 판정을 사건마다 새로 계산하므로 15-30분 걸립니다.

바뀐 점: 군집 수 2-12, 대각·완전 공분산 비교, R 비 제외(ALT·ALP와 중복), AST 추가, 선택 규칙 고정(부트스트랩 ARI ≥ 0.75이고 BIC가 최저에서 1% 안인 가장 작은 K), 정리된 모든 변수로 군집 해석, 군집별 예후, 군집이 예후 정보를 더하는지 검정, 군집별 원인 10개(규칙 판정과 퇴원 진단 코드).

끝나면 노트북 옆에 `kcd_cluster_package_*.zip`(집계와 그림만, 10 미만 칸 가림)이 생깁니다."""),
         code("""# ===== 설정 =====
from pathlib import Path
OUT = Path.cwd() / 'kcd_work' / 'run_all' / 'full' / 'kcd_liver_v2_outputs'   # 1단계 산출물 폴더
MIMIC_ROOT = Path(r'\\\\10.7.12.146\\데이터 및 소스\\MIMIC\\MIMIC4-V3.1')   # 원인 판정에 쓸 이전 진단·퇴원 진단 (hosp/)
N_BOOT = 30
print('산출물 폴더 확인:', OUT.exists(), OUT)
print('MIMIC 폴더 확인:', (MIMIC_ROOT / 'hosp').exists(), MIMIC_ROOT)"""),
         code(f"""# ===== 코드 풀기 (수정하지 마세요) =====
import base64, sys
CODE = Path.cwd() / 'kcd_cluster_code' / 'kcd_liver'
CODE.mkdir(parents=True, exist_ok=True)
(CODE / '__init__.py').write_text('', encoding='utf-8')
for name, b in {mods!r}.items():
    (CODE / name).write_bytes(base64.b64decode(b))
sys.path.insert(0, str(CODE.parent))
for m in [k for k in sys.modules if k.startswith('kcd_liver')]:
    del sys.modules[m]
from kcd_liver import cluster_report as cr
DEMO_IDS = set(int(x) for x in {demo!r}.split() if x.strip().isdigit())
print('코드 준비 완료')"""),
         code("""# ===== 입력 읽기 =====
import json, time
import pandas as pd
feat = pd.read_csv(OUT / '08_features.csv', low_memory=False)
outc = pd.read_csv(OUT / '10_outcomes_EVALUATION_ONLY.csv', low_memory=False)
ev = pd.read_csv(OUT / '05_events_private.csv', low_memory=False, usecols=lambda c: c in {'case_id', 'subject_id', 'group'})
extra = ev[['case_id', 'group']].copy() if 'group' in ev else ev[['case_id']].copy()
pred = OUT / '09_agent_predictions.jsonl'
if pred.exists():
    fl, conds = [], {}
    for line in pred.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        a = r.get('answer') or {}
        flags = a.get('flags') or next((v.get('flags') for v in a.values() if isinstance(v, dict) and v.get('flags')), None) or {}
        conds[r.get('condition')] = conds.get(r.get('condition'), 0) + 1
        if flags:
            fl.append({'case_id': r.get('case_id'), 'base': r.get('condition', 'base') == 'base',
                       **{f'flag_{k}': float(bool((v or {}).get('present'))) for k, v in flags.items()}})
    print('규칙 판정 기록:', conds, '· 원인 플래그가 있는 기록', len(fl))
    if fl:
        fl = pd.DataFrame(fl).sort_values('base', ascending=False).drop(columns='base').to_dict('records')
        extra = extra.merge(pd.DataFrame(fl).drop_duplicates('case_id'), on='case_id', how='left')
exclude = set(ev.loc[ev.subject_id.isin(DEMO_IDS), 'case_id']) if 'subject_id' in ev else set()
from kcd_liver.develop import load_private, history_flags, snapshot_payload, icd_proxy
from kcd_liver.rule_agent import rule_agent1, CAUSE_FLAGS
D = load_private(OUT)
evs = D['events']
adm = dx = None
if (MIMIC_ROOT / 'hosp').exists():
    hosp = MIMIC_ROOT / 'hosp'
    rd = lambda n, cols: pd.read_csv(next(p for p in [hosp / f'{n}.csv.gz', hosp / f'{n}.csv'] if p.exists()), usecols=cols, low_memory=False)
    subj = set(D['cohort'].subject_id)
    adm = rd('admissions', ['subject_id', 'hadm_id', 'dischtime'])
    dx = rd('diagnoses_icd', ['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
    adm, dx = adm[adm.subject_id.isin(subj)], dx[dx.subject_id.isin(subj)]
    print(f'진단 코드 {len(dx):,}행 읽음')
else:
    print('MIMIC_ROOT를 찾지 못해 이전 진단 없이 규칙 판정만 합니다 (퇴원 진단 비교 생략)')
hist = history_flags(D['cohort'], adm, dx) if adm is not None else None
hist_i = hist.set_index('stay_id').to_dict('index') if hist is not None else {}
proxy = icd_proxy(evs, dx) if dx is not None else {}
lab_g = {k: g for k, g in D['lab'][D['lab'].stay_id.isin(set(evs.stay_id))].groupby('stay_id')}
ctx_g = {k: g for k, g in D['context'][D['context'].stay_id.isin(set(evs.stay_id))].groupby('stay_id')}
rows, t = [], time.time()
for i, e in enumerate(evs.itertuples(), 1):
    a = rule_agent1(snapshot_payload(e.stay_id, e.t0, lab_g, ctx_g, hist_i.get(e.stay_id)))
    r = {'case_id': e.case_id, **{f'flag_{c}': float(a['flags'][c]['present']) for c in CAUSE_FLAGS}}
    if dx is not None:
        r.update({f'icd_{c}': float(c in proxy.get(int(e.hadm_id), set())) for c in CAUSE_FLAGS})
    rows.append(r)
    if i % 1000 == 0:
        print(f'원인 판정 {i:,}/{len(evs):,} ({time.time() - t:.0f}초)')
causes = pd.DataFrame(rows)
extra = extra[[c for c in extra.columns if not c.startswith('flag_')]].merge(causes, on='case_id', how='left')
print(f'원인 판정 완료: 사건 {len(causes):,}건 · 규칙 판정 {sum(c.startswith("flag_") for c in causes)}개 · 퇴원 진단 {sum(c.startswith("icd_") for c in causes)}개')
print(f'첫 사건 {len(feat):,}건, 결과 {len(outc):,}건, 공개 데모 제외 {len(exclude)}건, 추가 변수 {extra.shape[1] - 1}개')"""),
         code("""# ===== 군집 분석 =====
t = time.time()
R = cr.run(feat, outc, extra=extra, exclude_cases=exclude, n_boot=N_BOOT, log=print)
print(f'완료 {(time.time() - t) / 60:.1f}분 · 선택:', {k: R['selected'][k] for k in ['covariance', 'k', 'ari_median']})
for p in R['profiles']:
    print(f"C{p['cluster']} {p['name']} n={p['n_dev']} · 주요 변수:", [cr.var_label(v) for v in R['characterization']['top_by_cluster'].get(p['cluster'], [])])"""),
         code("""# ===== 그림과 묶음 =====
import zipfile
from datetime import datetime
from IPython.display import Image, display
FIG = OUT / 'cluster_v2'
FIG.mkdir(exist_ok=True)
cr.save(R, FIG / 'cluster_report.json')
files = cr.render(R, FIG)
for p in files:
    display(Image(filename=str(p)))
zp = Path.cwd() / f"kcd_cluster_package_{datetime.now():%Y%m%d_%H%M}.zip"
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(FIG / 'cluster_report.json', 'cluster_report.json')
    for p in files:
        z.write(p, 'figures/' + p.name)
print('돌려줄 파일:', zp)""")]
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbformat.write(nb, ROOT / "notebooks" / "KCD_CLUSTER.ipynb")
print("wrote notebooks/KCD_CLUSTER.ipynb")
