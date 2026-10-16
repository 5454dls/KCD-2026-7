"""Build notebooks/KCD_EXTERNAL.ipynb: runs at another hospital from inside the model package (no retraining)."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
runner = nbformat.read(ROOT / "notebooks" / "KCD_RUN_ALL.ipynb", as_version=4)
helper = next(c.source for c in runner.cells if "def run_main" in c.source)
envchk = next(c.source for c in runner.cells if "환경 확인" in c.source)

md = """# KCD 2026 · 외부 검증 (다른 병원)

이 노트북은 **모델 패키지 폴더 안에서** 실행합니다(`kcd_liver/`, `models/`가 옆에 있어야 함). 학습은 하지 않고, MIMIC-IV에서 학습해 고정한 4개 모델을 이 병원 자료에 그대로 적용해 성능을 계산합니다.

| 모델 | 실행 시점 | 출력 |
|---|---|---|
| 1 Agent 3 | 정상일 때 매시간 | 24시간 이내 간기능 이상 위험 |
| 2 Agent 2 | 첫 이상 시점 | 예후 4경로 (해소 × ICU 사망, 추적은 ICU 퇴실까지) |
| 3 Agent 1 | 첫 이상 시점 | 원인 10개 0/1 (규칙 코드) + 보조 군집 이식 |
| 4 궤적 | 상태와 무관, 매시간 | ALT·AST·ALP·TBIL 24시간 예측 (10·50·90%) |

**자료 형식**: MIMIC-IV와 같은 폴더·열 구조(`hosp/patients, admissions, labevents, d_labitems, diagnoses_icd, emar`, `icu/icustays, chartevents, inputevents, d_items, outputevents`). 검사·활력징후 itemid가 다르면 `notebooks/KCD_MIMICIV_liver_agents_v2.ipynb`의 `LABS`·`VITAL_ITEMS` 대응표를 이 병원 코드로 바꿔야 합니다.

**돌려주는 것**: `kcd_external_return_*.zip` (집계만, 10 미만 칸 가림). 환자 단위 파일은 `external_work/`에만 남습니다.

**버전**: 모델 파일은 학습 때와 같은 scikit-learn 버전에서 읽어야 합니다. `MODEL_CARD.md`의 버전을 확인하세요."""

cfg = """# ===== 설정 =====
from pathlib import Path
import os, sys, json, time, shutil, zipfile, traceback, subprocess, platform
from datetime import datetime
REPO = Path.cwd()
assert (REPO / 'kcd_liver').is_dir() and (REPO / 'models').is_dir(), '모델 패키지 폴더 안에서 실행하세요'
DATA_ROOT = Path(r'D:/hospital_data/mimic_format')   # 이 병원 자료 (hosp/ icu/)
SITE_NAME = 'external_site'
MAX_PATIENTS = None
MODEL_DIR = REPO / 'models'
WORK = REPO / 'external_work'
sys.path.insert(0, str(REPO)); WORK.mkdir(exist_ok=True)
REPORT = {'started_at': datetime.now().isoformat(timespec='seconds'), 'steps': {}, 'python': platform.python_version(), 'site': SITE_NAME}
print('DATA_ROOT 접근 가능:', DATA_ROOT.exists())
card = REPO / 'MODEL_CARD.md'
if card.exists(): print([l for l in card.read_text(encoding='utf-8').splitlines() if 'scikit-learn' in l][:1])"""

s1 = """# ===== 1단계: 이 병원 자료로 분석 노트북 실행 (코호트·사건·원인 규칙; 사이트 내보내기 끔) =====
EXT_OUT = WORK / 'external' / 'kcd_liver_v2_outputs'
def e1():
    out, m = run_main('external', DATA_ROOT, 'external', MAX_PATIENTS, extra={'EXPORT_SITE=True': 'EXPORT_SITE=False'},
                      env={'KCD_MODEL_DIR': MODEL_DIR})
    return {'notebook_minutes': m, 'cohort_flow': pd.read_csv(out / '04_cohort_flow.csv').to_dict('records')}
step('1_external_notebook', e1)"""

s2 = """# ===== 2단계: 고정 모델 적용, 성능 계산 =====
def e2():
    from scripts.external_validate import main as ev
    ev(['--out-dir', str(EXT_OUT), '--model-dir', str(MODEL_DIR), '--data-root', str(DATA_ROOT), '--site', SITE_NAME])
    R = json.loads((EXT_OUT / 'results_external.json').read_text(encoding='utf-8'))
    return {'agent3_auroc': (R.get('agent3') or {}).get('auroc'), 'prognosis_n': (R.get('prognosis') or {}).get('n')}
step('2_external_validation', e2)"""

s3 = """# ===== 3단계: 돌려줄 파일 (집계만) =====
FORBIDDEN_KEYS = {'subject_id', 'hadm_id', 'stay_id', 'labevent_id', 'case_id'}
def scan_ids(o, hits=None):
    hits = [] if hits is None else hits
    if isinstance(o, dict):
        for k, v in o.items():
            if k in FORBIDDEN_KEYS: hits.append(k)
            scan_ids(v, hits)
    elif isinstance(o, list):
        for v in o: scan_ids(v, hits)
    return hits
def e3():
    R = json.loads((EXT_OUT / 'results_external.json').read_text(encoding='utf-8'))
    if scan_ids(R): raise RuntimeError('환자 식별 키가 있어 압축하지 않습니다')
    zp = Path.cwd() / f"kcd_external_return_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(EXT_OUT / 'results_external.json', 'results_external.json')
        for n in ['04_cohort_flow.csv', '02_lab_quality.csv']:
            if (EXT_OUT / n).exists(): z.write(EXT_OUT / n, 'aggregates/' + n)
        z.writestr('run_report.json', json.dumps(REPORT, ensure_ascii=False, indent=1, default=str))
    print('완성:', zp); return {'package': str(zp)}
step('3_package', e3)
print(json.dumps({k: v.get('status') for k, v in REPORT['steps'].items()}, ensure_ascii=False))"""

cells = [nbformat.v4.new_markdown_cell(md), nbformat.v4.new_code_cell(cfg), nbformat.v4.new_code_cell(envchk),
         nbformat.v4.new_code_cell(helper), nbformat.v4.new_code_cell(s1), nbformat.v4.new_code_cell(s2), nbformat.v4.new_code_cell(s3)]
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbformat.write(nb, ROOT / "notebooks" / "KCD_EXTERNAL.ipynb")
print("wrote notebooks/KCD_EXTERNAL.ipynb")
