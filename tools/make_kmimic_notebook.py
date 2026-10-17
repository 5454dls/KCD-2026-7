"""Build notebooks/KCD_KMIMIC_EXTERNAL.ipynb: K-MIMIC 7 sites -> MIMIC layout -> frozen-model external validation, one notebook."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
C = []

C.append(md("""# K-MIMIC 7개 병원 외부 검증 (한 노트북)

**모델 묶음 폴더(`kcd_liver/`, `models/`, `notebooks/`가 있는 곳) 안에 두고** 위에서부터 끝까지 실행합니다. 중간에 끊기면 **그냥 처음부터 다시 실행**하면 끝난 단계는 건너뛰고 멈춘 곳부터 이어갑니다.

| 단계 | 하는 일 | 이어하기 단위 |
|---|---|---|
| 1 변환 | 앞 노트북 결과(코호트, 간 검사·기타 검사·활력징후·승압제 parquet)와 원본 입원·ICU·환자·진단 표를 MIMIC-IV 형식으로 | 병원 × 테이블 |
| 2 분석 | 병원마다 분석 노트북을 코호트·사건·결과 판정까지만 실행 (그 뒤 셀은 외부 검증에 필요 없어 건너뜀) | 병원 |
| 3 통합 | 7개 병원 결과를 하나로 합침 (번호가 겹치지 않게 병원마다 번호대를 나눔) | 한 번 |
| 4 검증 | 고정 모델 4개 적용: 병원별 7번 + 전체 1번 | 병원 |
| 5 묶기 | 집계만 zip (10 미만 칸 가림, 환자 번호 검사) | 한 번 |

**임상의 기준 반영**: 간 검사 ULN 고정(ALT·AST 40, ALP 115, 총빌리루빈 1.2, 직접빌리 0.5, GGT 35, INR 1.2), 간 검사는 '주' 항목만(009 총·직접빌리 승격), 보고 시각은 채혈 뒤 72시간 안일 때만 쓰고 아니면 채혈 시각.

**진행 표시**: 모든 단계의 시작·끝·소요 시간, 오래 걸리는 쿼리는 10초마다 진행률, 분석 노트북은 셀마다 출력합니다. 같은 내용이 `kcd_external/progress.log`에도 쌓입니다.

**반출**: `kcd_external/kcd_kmimic_external_*.zip` (집계만). 변환 자료와 번호 대응표는 서버 안 `kcd_external/private/`에만 남습니다."""))

C.append(code("""# ===== 0. 설정 (필요하면 여기만 고칩니다) =====
import os, sys, json, time, glob, re, codecs, shutil, zipfile, traceback, threading, platform, subprocess
from pathlib import Path
BASE = os.environ.get('KMIMIC_BASE') or next((p for p in ('/home/khdp-user/workspace/datasets', '/mnt/dataset/EMR') if os.path.isdir(p)), '/home/khdp-user/workspace/datasets')
OUT = Path(os.environ.get('KMIMIC_OUT') or '/home/khdp-user/workspace/liver_out_7site')   # 앞 노트북 결과 폴더
SITES = ['001', '003', '004', '005', '006', '009', '202']
PKG = Path(os.environ.get('KCD_PKG') or Path.cwd())   # 모델 묶음 폴더
WORK = Path(os.environ.get('KCD_WORK') or (OUT / 'kcd_external'))
ULN = {'ALT': 40.0, 'AST': 40.0, 'ALP': 115.0, 'TBIL': 1.2, 'DBIL': 0.5, 'GGT': 35.0, 'INR': 1.2}   # 임상의 기준
PROMOTE = {('009', '009LM000030'): '주', ('009', '009LM000020'): '주'}   # 009 총빌리, 직접빌리 승격
STORE_MAX_H = 72      # 보고 시각이 채혈 뒤 이 시간 안일 때만 사용
MEM, THREADS = '4GB', 4
HEARTBEAT = 10        # 오래 걸리는 쿼리 진행 표시 간격 (초)
MIN_CELL, BOOT = 10, 200
RUN = {'convert': True, 'analyze': True, 'validate': True, 'pooled': True, 'package': True}
REDO = set()          # 다시 할 단계, 예: {'convert:001:labevents', 'analyze:009'}
assert os.path.isdir(BASE), '원본 폴더 없음: ' + BASE
assert (PKG / 'kcd_liver').is_dir() and (PKG / 'models').is_dir() and (PKG / 'notebooks').is_dir(), '모델 묶음 폴더 안에서 실행하거나 KCD_PKG 를 지정하세요: ' + str(PKG)
for d in ('private', 'analysis', 'results'):
    (WORK / d).mkdir(parents=True, exist_ok=True)
print('원본', BASE, '| 앞 노트북 결과', OUT, '| 모델 묶음', PKG, '| 작업 폴더', WORK)"""))

C.append(code("""# ===== 1. 도우미 (진행 기록, 이어하기, DuckDB) =====
try:
    import duckdb
except ImportError:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'duckdb==1.3.2'], capture_output=True, text=True)
    raise SystemExit('duckdb 설치를 시도했습니다. 커널 재시작 뒤 다시 실행하세요. ' + r.stderr[-300:])
import pandas as pd, numpy as np
LOG, STATE_F = WORK / 'progress.log', WORK / 'state.json'
T0 = time.time()

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [{(time.time() - T0) / 60:6.1f}분] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\\n')

def _write_json(p, obj):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, p)

STATE = json.loads(STATE_F.read_text(encoding='utf-8')) if STATE_F.exists() else {}

def task(key, fn, outputs=(), after=()):
    \"\"\"Run fn once; skip if done, outputs exist and no upstream step (after) ran later. Up to 3 tries; failures are recorded, not fatal.\"\"\"
    st = STATE.get(key, {})
    stale = [d for d in after if STATE.get(d, {}).get('ts', 0) > st.get('ts', 0)]
    if st.get('status') == 'ok' and key not in REDO and not stale and all(Path(o).exists() for o in outputs):
        log(f'  건너뜀 (이미 끝남): {key}')
        return st.get('result')
    for attempt in (1, 2, 3):
        t = time.time()
        log(f'▶ 시작: {key}' + (f' (재시도 {attempt - 1})' if attempt > 1 else '') + (f' (앞 단계가 다시 돌아 새로 함: {stale})' if stale and attempt == 1 else ''))
        try:
            r = fn()
            STATE[key] = {'status': 'ok', 'sec': round(time.time() - t, 1), 'result': r, 'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'ts': time.time()}
            _write_json(STATE_F, STATE)
            log(f'✔ 끝: {key} ({time.time() - t:.0f}초)' + (f' → {str(r)[:160]}' if r is not None else ''))
            return r
        except Exception as e:
            log(f'✘ 실패: {key}: {type(e).__name__}: {str(e)[:300]}')
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write(traceback.format_exc() + '\\n')
            reset()
            if attempt == 3:
                STATE[key] = {'status': 'failed', 'error': f'{type(e).__name__}: {str(e)[:500]}', 'at': time.strftime('%Y-%m-%d %H:%M:%S')}
                _write_json(STATE_F, STATE)
            else:
                time.sleep(3 * attempt)
    return None

def ok(key):
    return STATE.get(key, {}).get('status') == 'ok'

def connect():
    c = duckdb.connect()
    (WORK / 'duckdb_tmp').mkdir(exist_ok=True)
    c.execute(f"SET memory_limit='{MEM}'"); c.execute(f'SET threads={THREADS}')
    c.execute('SET preserve_insertion_order=false'); c.execute(f"SET temp_directory='{WORK / 'duckdb_tmp'}'")
    return c
con = connect()

def reset():
    global con
    try:
        con.close()
    except Exception:
        pass
    con = connect()

def q(sql, label='쿼리'):
    \"\"\"Run SQL in a worker thread; print progress every HEARTBEAT seconds so a long query is visibly alive.\"\"\"
    res = {}
    def work():
        try:
            res['r'] = con.execute(sql).fetchall()
        except Exception as e:
            res['e'] = e
    th = threading.Thread(target=work, daemon=True); th.start(); t = time.time()
    while th.is_alive():
        th.join(HEARTBEAT)
        if th.is_alive():
            try:
                p = con.query_progress()
            except Exception:
                p = -1
            log(f'    … {label}: ' + (f'{p:.0f}%' if p is not None and p >= 0 else '진행 중') + f' ({time.time() - t:.0f}초)')
    if 'e' in res:
        raise res['e']
    return res.get('r')

def copy_csv(sql, path, label):
    \"\"\"COPY to a temp file, then rename (an interrupted write never looks finished). Returns the row count.\"\"\"
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    r = q(f"COPY ({sql}) TO '{tmp}' (FORMAT CSV, HEADER)", label)
    os.replace(tmp, path)
    n = int(r[0][0]) if r else None
    log(f'    저장 {path.parent.name}/{path.name}: {n:,}행')
    return n

def copy_pq(sql, path, label):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    r = q(f"COPY ({sql}) TO '{tmp}' (FORMAT PARQUET)", label)
    os.replace(tmp, path)
    return int(r[0][0]) if r else None

def reg(name, df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) in ('str', 'string'):
            df[c] = df[c].astype(object)
    con.register(name, df)

# ---- 원본 파일 찾기와 인코딩 (앞 노트북과 같은 규칙: UTF-8 사본이 있으면 그것을 씀)
def site_root(site):
    p = os.path.join(BASE, 'kmimic-emr-' + site)
    return p if os.path.isdir(p) else os.path.join(BASE, site)
OVERRIDE = {}
for s in SITES:
    for sub, cut in (('utf8', False), ('utf8f', True)):
        for d in glob.glob(str(OUT / sub / s / '*')):
            fs = sorted(glob.glob(os.path.join(d, '*.csv')))
            if fs:
                OVERRIDE[(s, os.path.basename(d).rsplit('_', 1)[0] if cut else os.path.basename(d))] = fs
_FILES = {}
def table_files(site, table, raw=False, maxdepth=3):
    if not raw and (site, table) in OVERRIDE:
        return OVERRIDE[(site, table)]
    if (site, table) not in _FILES:
        root, hits = site_root(site), []
        for d, sub, files in os.walk(root):
            if d[len(root):].count(os.sep) >= maxdepth:
                sub[:] = []
            for fn in files:
                low = fn.lower()
                for e in ('.csv', '.csv.gz'):
                    if low.endswith(e) and low[:-len(e)] == table.lower() and os.path.getsize(os.path.join(d, fn)) > 0:
                        hits.append(os.path.join(d, fn))
        _FILES[(site, table)] = sorted(hits)
    return _FILES[(site, table)]

def is_utf8(site, table, nbytes=8 << 20):
    for f in table_files(site, table, raw=True)[:3]:
        if f.endswith('.gz'):
            return True
        with open(f, 'rb') as fi:
            b = fi.read(nbytes)
        b = b[:b.rfind(b'\\n') + 1] or b
        try:
            b.decode('utf-8')
        except UnicodeDecodeError:
            return False
    return True

def to_utf8(site, table, enc='cp949'):
    outd = OUT / 'utf8' / site / table; outd.mkdir(parents=True, exist_ok=True); outs = []
    for i, f in enumerate(table_files(site, table, raw=True)):
        o = outd / ('%04d.csv' % i)
        if not (o.exists() and o.stat().st_size > 0):
            log(f'    {site} {table} UTF-8 사본 만드는 중: {os.path.basename(f)} ({os.path.getsize(f) / 1e9:.2f} GB)')
            dec = codecs.getincrementaldecoder(enc)(errors='replace')
            with open(f, 'rb') as fi, open(str(o) + '.tmp', 'w', encoding='utf-8', newline='') as fo:
                while True:
                    b = fi.read(1 << 24)
                    if not b:
                        fo.write(dec.decode(b'', final=True)); break
                    fo.write(dec.decode(b))
            os.replace(str(o) + '.tmp', o)
        outs.append(str(o))
    OVERRIDE[(site, table)] = outs

def src(site, table):
    if (site, table) not in OVERRIDE and not is_utf8(site, table):
        log(f'    {site} {table}: UTF-8 아님 → 사본으로 읽음'); to_utf8(site, table)
    fs = table_files(site, table)
    assert fs, f'{site} 병원에 {table} 파일 없음'
    return f"read_csv_auto({fs!r}, all_varchar=true, union_by_name=true, ignore_errors=true)"

def ts_sql(col):
    return f"coalesce(try_cast({col} AS TIMESTAMP), try_strptime({col}, '%Y-%m-%d %H:%M:%S'))"

def shifted(col, sh='sh.yrs'):
    t = f'({ts_sql(col)} - to_years(CAST({sh} AS INTEGER)))'
    return f"CASE WHEN year({t}) BETWEEN 1700 AND 2260 THEN strftime({t}, '%Y-%m-%d %H:%M:%S') END"

def store_ok(ch, st):   # 보고 시각은 채혈 뒤 STORE_MAX_H 안일 때만
    return f"({ts_sql(st)} >= {ts_sql(ch)} AND {ts_sql(st)} <= {ts_sql(ch)} + INTERVAL {STORE_MAX_H} HOUR)"

def pq_count(p):
    try:
        return con.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0]
    except Exception:
        return None
log(f'준비 끝 · duckdb {duckdb.__version__} · python {platform.python_version()} · 끝난 단계 {sum(v.get(\"status\") == \"ok\" for v in STATE.values())}개')"""))

C.append(code("""# ===== 2. 사전 점검 (몇 초) =====
COH = OUT / 'cohort_first_icu.parquet'
IMAP = OUT / 'item_map_7site.csv'
assert COH.exists() and IMAP.exists(), '코호트 parquet 또는 항목표가 없습니다 (앞 노트북 먼저)'
coh_n = dict(con.execute(f"SELECT site, count(*) FROM read_parquet('{COH}') GROUP BY 1").fetchall())
rows = []
for s in SITES:
    have = lambda pat: [f for f in sorted(glob.glob(str(OUT / pat))) if '.part' not in f and not f.endswith('.tmp')]
    r = {'병원': s, '코호트': coh_n.get(s, 0)}
    for name, pat in [('간검사', f'lab_liver_{s}.parquet'), ('기타검사', f'lab_other_{s}.parquet'), ('추가검사', f'lab_extra_{s}.parquet'),
                      ('활력', f'vital_{s}*.parquet'), ('소변', f'urine_{s}*.parquet'), ('승압제', f'emar_vaso_{s}.parquet')]:
        fs = have(pat)
        r[name] = sum(pq_count(f) or 0 for f in fs) if fs else '없음'
    for t in ('ADMISSIONS', 'ICUSTAYS', 'PATIENTS', 'DIAGNOSES_ICD'):
        r[t] = 'O' if table_files(s, t) else '없음'
    rows.append(r)
PRE = pd.DataFrame(rows)
print(PRE.to_string(index=False))
READY = [r['병원'] for r in rows if r['코호트'] and r['간검사'] != '없음' and r['활력'] != '없음' and all(r[t] == 'O' for t in ('ADMISSIONS', 'ICUSTAYS', 'PATIENTS', 'DIAGNOSES_ICD'))]
log(f'진행할 병원: {READY} · 빠진 필수 자료가 있어 건너뛸 병원: {[s for s in SITES if s not in READY]}')"""))

C.append(code("""# ===== 3. 1단계: MIMIC-IV 형식 변환 (병원 × 테이블, 이어하기) =====
LAB_ID = {'ALT': 50861, 'AST': 50878, 'ALP': 50863, 'TBIL': 50885, 'DBIL': 50883, 'GGT': 50927, 'INR': 51237, 'LACT': 50813,
          'CREAT': 50912, 'ALB': 50862, 'BUN': 51006, 'NA': 50983, 'HCO3': 50882, 'GLU': 50931, 'PLT': 51265, 'WBC': 51301,
          'HB': 51222, 'NTPROBNP': 50963, 'CK': 50910, 'LDH': 50954, 'K': 50971, 'PT': 51274, 'PTT': 51275, 'FIB': 51214, 'PH': 50820, 'PO2': 50821}
LAB_LABEL = {'ALT': 'Alanine Aminotransferase (ALT)', 'AST': 'Asparate Aminotransferase (AST)', 'ALP': 'Alkaline Phosphatase',
             'TBIL': 'Bilirubin, Total', 'DBIL': 'Bilirubin, Direct', 'GGT': 'Gamma Glutamyltransferase', 'INR': 'INR(PT)', 'LACT': 'Lactate',
             'CREAT': 'Creatinine', 'ALB': 'Albumin', 'BUN': 'Urea Nitrogen', 'NA': 'Sodium', 'HCO3': 'Bicarbonate', 'GLU': 'Glucose',
             'PLT': 'Platelet Count', 'WBC': 'White Blood Cells', 'HB': 'Hemoglobin', 'NTPROBNP': 'NTproBNP', 'CK': 'Creatine Kinase (CK)',
             'LDH': 'Lactate Dehydrogenase (LD)', 'K': 'Potassium', 'PT': 'PT', 'PTT': 'PTT', 'FIB': 'Fibrinogen, Functional', 'PH': 'pH', 'PO2': 'pO2'}
UNIT = {'ALT': 'IU/L', 'AST': 'IU/L', 'ALP': 'IU/L', 'GGT': 'IU/L', 'CK': 'IU/L', 'LDH': 'IU/L', 'TBIL': 'mg/dL', 'DBIL': 'mg/dL',
        'CREAT': 'mg/dL', 'GLU': 'mg/dL', 'BUN': 'mg/dL', 'ALB': 'g/dL', 'HB': 'g/dL', 'NA': 'mEq/L', 'K': 'mEq/L', 'HCO3': 'mEq/L',
        'PLT': 'K/uL', 'WBC': 'K/uL', 'LACT': 'mmol/L', 'NTPROBNP': 'pg/mL', 'PT': 'sec', 'PTT': 'sec', 'PO2': 'mm Hg', 'INR': '', 'PH': 'units', 'FIB': 'mg/dL'}
VIT_ID = {'MAP': 220052, 'HR': 220045, 'RR': 220210, 'SPO2': 220277, 'TEMP': 223762, 'SBP': 220050, 'DBP': 220051}
VIT_LABEL = {220052: 'Arterial Blood Pressure mean', 220045: 'Heart Rate', 220210: 'Respiratory Rate', 220277: 'O2 saturation pulseoxymetry',
             223762: 'Temperature Celsius', 220050: 'Arterial Blood Pressure systolic', 220051: 'Arterial Blood Pressure diastolic', 226559: 'Foley'}
VASO_ID = {'NE': (221906, 'Norepinephrine'), 'EPI': (221289, 'Epinephrine'), 'VASO': (222315, 'Vasopressin'), 'PHE': (221749, 'Phenylephrine'), 'DOPA': (221662, 'Dopamine')}
EXC = r'dobut|도부|tropicamid|eye|점안|bosmin|보스민|lido|리도|atropin|아트로핀|soln|apply|inhal|patch'
def vaso_class(n):
    n = str(n).lower()
    if re.search(r'norepi|norpin|노르에피|노르핀|noradren|levophed|q-?phrine', n): return 'NE'
    if re.search(r'vasopressin|바소프레', n): return 'VASO'
    if re.search(r'phenylephr|neosynephr|페닐레프|페닐에프', n): return 'PHE'
    if re.search(r'dopamin|도파민', n): return 'DOPA'
    if re.search(r'epineph|adrenalin|에피네프', n): return 'EPI'
    return None
IM_ALL = pd.read_csv(IMAP, dtype=str)
MF = WORK / 'private' / 'mimic_format'

def convert_site(site, idx):
    root, stg = MF / site, WORK / 'private' / 'stage' / site
    H, I = root / 'hosp', root / 'icu'
    off = (idx + 1) * 100_000_000   # 병원마다 번호대를 나눠 합쳐도 겹치지 않게
    P = {k: stg / f'{k}.parquet' for k in ('C', 'ADM0', 'ICU0', 'PAT0', 'SH', 'K', 'KH', 'KS', 'KP')}
    def views():
        for k, p in P.items():
            con.execute(f"CREATE OR REPLACE TEMP VIEW {k} AS SELECT * FROM read_parquet('{p}')")

    def keys():
        con.execute(f"CREATE OR REPLACE TEMP TABLE C0 AS SELECT DISTINCT trim(SUBJECT_ID) AS sid, trim(HADM_ID) AS hid, trim(STAY_ID) AS stid FROM read_parquet('{COH}') WHERE site = '{site}'")
        copy_pq('SELECT * FROM C0', P['C'], f'{site} 코호트')
        copy_pq(f"SELECT DISTINCT trim(SUBJECT_ID) AS sid, trim(HADM_ID) AS hid, * EXCLUDE (SUBJECT_ID, HADM_ID) FROM {src(site, 'ADMISSIONS')} WHERE trim(SUBJECT_ID) IN (SELECT sid FROM C0)", P['ADM0'], f'{site} 입원')
        copy_pq(f"SELECT trim(SUBJECT_ID) AS sid, trim(HADM_ID) AS hid, trim(STAY_ID) AS stid, * EXCLUDE (SUBJECT_ID, HADM_ID, STAY_ID) FROM {src(site, 'ICUSTAYS')} WHERE trim(STAY_ID) IN (SELECT stid FROM C0)", P['ICU0'], f'{site} ICU')
        copy_pq(f"SELECT trim(SUBJECT_ID) AS sid, * EXCLUDE (SUBJECT_ID) FROM {src(site, 'PATIENTS')} WHERE trim(SUBJECT_ID) IN (SELECT sid FROM C0)", P['PAT0'], f'{site} 환자')
        for k in ('ADM0', 'ICU0', 'PAT0'):
            con.execute(f"CREATE OR REPLACE TEMP VIEW {k}_v AS SELECT * FROM read_parquet('{P[k]}')")
        copy_pq(f'''WITH y AS (SELECT sid, min(year({ts_sql('INTIME')})) AS yi FROM ICU0_v GROUP BY 1),
                         a AS (SELECT sid, min(year({ts_sql('ADMITTIME')})) AS ymin, max(year({ts_sql('DISCHTIME')})) AS ymax FROM ADM0_v GROUP BY 1)
                    SELECT y.sid, y.yi, CASE WHEN y.yi > 2250 THEN 400 * CAST(ceil((y.yi - 2250) / 400.0) AS INTEGER) ELSE 0 END AS yrs, a.ymax, a.ymin
                    FROM y LEFT JOIN a USING (sid)''', P['SH'], f'{site} 날짜 당김')
        copy_pq(f"SELECT c.sid, c.hid, c.stid, {off} + dense_rank() OVER (ORDER BY c.sid) AS subject_id FROM C0 c", P['K'], f'{site} 번호')
        copy_pq(f"SELECT sid, hid, {off} + 20000000 + dense_rank() OVER (ORDER BY sid, hid) AS hadm_id FROM (SELECT DISTINCT sid, hid FROM ADM0_v)", P['KH'], f'{site} 입원 번호')
        copy_pq(f"SELECT stid, {off} + 30000000 + row_number() OVER (ORDER BY stid) AS stay_id FROM (SELECT DISTINCT stid FROM C0)", P['KS'], f'{site} ICU 번호')
        copy_pq("SELECT DISTINCT sid, subject_id FROM read_parquet('" + str(P['K']) + "')", P['KP'], f'{site} 환자 번호')
        reset(); views()
        r = con.execute('SELECT count(*), count(*) FILTER (WHERE yrs > 0), count(*) FILTER (WHERE ymax - yrs > 2260 OR ymin - yrs < 1700) FROM SH').fetchone()
        return {'patients': r[0], 'shifted': r[1], 'out_of_range': r[2]}

    def patients():
        views()
        return copy_csv(f'''SELECT kp.subject_id, CASE WHEN upper(trim(p.SEX)) IN ('M', '1', 'MALE') THEN 'M' WHEN upper(trim(p.SEX)) IN ('F', '2', 'FEMALE') THEN 'F' END AS gender,
                    try_cast(regexp_extract(p.ANCHOR_AGE, '([0-9]+)', 1) AS INTEGER) AS anchor_age,
                    CASE WHEN abs(try_cast(p.ANCHOR_YEAR AS INTEGER) - sh.yi) <= 120 THEN try_cast(p.ANCHOR_YEAR AS INTEGER) - sh.yrs ELSE sh.yi - sh.yrs END AS anchor_year,
                    strftime(CAST({ts_sql('p.DOD')} - to_years(CAST(sh.yrs AS INTEGER)) AS DATE), '%Y-%m-%d') AS dod
             FROM (SELECT * FROM PAT0 QUALIFY row_number() OVER (PARTITION BY sid ORDER BY sid) = 1) p JOIN KP kp USING (sid) JOIN SH sh USING (sid)''', H / 'patients.csv', f'{site} patients')

    def admissions():
        views()
        return copy_csv(f'''SELECT kp.subject_id, kh.hadm_id, {shifted('a.ADMITTIME')} AS admittime, {shifted('a.DISCHTIME')} AS dischtime,
                    {shifted('a.DEATHTIME')} AS deathtime, a.ADMISSION_TYPE AS admission_type, try_cast(a.HOSPITAL_EXPIRE_FLAG AS INTEGER) AS hospital_expire_flag
             FROM (SELECT * FROM ADM0 QUALIFY row_number() OVER (PARTITION BY sid, hid ORDER BY {ts_sql('ADMITTIME')}) = 1) a
             JOIN KP kp USING (sid) JOIN KH kh USING (sid, hid) JOIN SH sh USING (sid)''', H / 'admissions.csv', f'{site} admissions')

    def icustays():
        views()
        return copy_csv(f'''SELECT kp.subject_id, kh.hadm_id, ks.stay_id, upper(trim(i.FIRST_CAREUNIT)) AS first_careunit, upper(trim(i.LAST_CAREUNIT)) AS last_careunit,
                    {shifted('i.INTIME')} AS intime, {shifted('i.OUTTIME')} AS outtime
             FROM ICU0 i JOIN KP kp USING (sid) JOIN KH kh USING (sid, hid) JOIN KS ks USING (stid) JOIN SH sh USING (sid)''', I / 'icustays.csv', f'{site} icustays')

    def diagnoses():
        views()
        return copy_csv(f'''SELECT kp.subject_id, kh.hadm_id, try_cast(d.SEQ_NUM AS INTEGER) AS seq_num, regexp_replace(upper(trim(d.ICD_CODE)), '[^A-Z0-9]', '', 'g') AS icd_code,
                    coalesce(try_cast(regexp_extract(d.ICD_VERSION, '([0-9]+)', 1) AS INTEGER), 10) AS icd_version
             FROM {src(site, 'DIAGNOSES_ICD')} d JOIN KH kh ON kh.hid = trim(d.HADM_ID) JOIN KP kp ON kp.sid = kh.sid''', H / 'diagnoses_icd.csv', f'{site} diagnoses_icd')

    def labevents():
        views()
        im = IM_ALL[(IM_ALL['site'] == site) & (IM_ALL['src'] == 'lab') & (IM_ALL['role'] == '주')][['ITEMID', 'var']]
        reg('im', im); reg('pm', pd.DataFrame([(k[1], v) for k, v in PROMOTE.items() if k[0] == site] or [('-', '-')], columns=['ITEMID', 'r2']))
        reg('lid', pd.DataFrame(list(LAB_ID.items()), columns=['var', 'itemid'])); reg('uu', pd.DataFrame(list(UNIT.items()), columns=['var', 'unit']))
        reg('ul', pd.DataFrame(list(ULN.items()), columns=['var', 'uln_fixed']))
        parts = [f'''SELECT trim(x.HADM_ID) AS hid, x.ITEMID, x.kind AS var, CAST(x.CHARTTIME AS VARCHAR) AS CHARTTIME, CAST(x.STORETIME AS VARCHAR) AS STORETIME,
                            try_cast(coalesce(x.valuenum, x.value_num) AS DOUBLE) AS v, x.uom, try_cast(x.uln_rec AS DOUBLE) AS uln_rec
                     FROM read_parquet('{OUT / f"lab_liver_{site}.parquet"}', union_by_name=true) x LEFT JOIN pm ON pm.ITEMID = x.ITEMID
                     WHERE coalesce(pm.r2, x.role) = '주' AND x.kind IN ('ALT', 'AST', 'ALP', 'TBIL', 'DBIL', 'GGT')''']
        for f in (f'lab_other_{site}.parquet', f'lab_extra_{site}.parquet'):
            if (OUT / f).exists() and pq_count(OUT / f):
                parts.append(f'''SELECT trim(x.HADM_ID) AS hid, x.ITEMID, im.var, CAST(x.CHARTTIME AS VARCHAR), CAST(x.STORETIME AS VARCHAR), try_cast(x.v AS DOUBLE), x.uom, try_cast(x.uln_rec AS DOUBLE)
                                 FROM read_parquet('{OUT / f}', union_by_name=true) x JOIN im ON im.ITEMID = x.ITEMID''')
        con.execute('CREATE OR REPLACE TEMP TABLE LB0 AS SELECT * FROM (' + ' UNION ALL '.join(parts) + ') WHERE v IS NOT NULL AND CHARTTIME IS NOT NULL AND hid IN (SELECT hid FROM KH)')
        lact = "(LB0.var = 'LACT' AND lower(coalesce(LB0.uom, '')) LIKE '%mg%')"
        n = copy_csv(f'''WITH m AS (SELECT ITEMID, mode(uln_rec) AS uln_item FROM LB0 WHERE uln_rec > 0 GROUP BY 1),
                 z AS (SELECT DISTINCT kh.hadm_id, kp.subject_id, lid.itemid, {shifted('LB0.CHARTTIME')} AS charttime,
                              CASE WHEN {store_ok('LB0.CHARTTIME', 'LB0.STORETIME')} THEN {shifted('LB0.STORETIME')} END AS storetime,
                              round(CASE WHEN {lact} THEN LB0.v / 9.008 ELSE LB0.v END, 4) AS valuenum, uu.unit AS valueuom,
                              round(coalesce(ul.uln_fixed, CASE WHEN LB0.uln_rec > 0 THEN LB0.uln_rec ELSE m.uln_item END * CASE WHEN {lact} THEN 1 / 9.008 ELSE 1 END), 4) AS ref_range_upper
                       FROM LB0 JOIN KH kh ON kh.hid = LB0.hid JOIN KP kp ON kp.sid = kh.sid JOIN SH sh ON sh.sid = kh.sid
                       JOIN lid ON lid.var = LB0.var JOIN uu ON uu.var = LB0.var LEFT JOIN ul ON ul.var = LB0.var LEFT JOIN m ON m.ITEMID = LB0.ITEMID)
             SELECT row_number() OVER () AS labevent_id, subject_id, hadm_id, itemid, charttime, storetime, CAST(valuenum AS VARCHAR) AS value, valuenum, valueuom,
                    ref_range_upper, NULL AS flag
             FROM z WHERE charttime IS NOT NULL''', H / 'labevents.csv', f'{site} labevents')
        reg('dli', pd.DataFrame([(LAB_ID[k], LAB_LABEL[k], 'Blood', 'Chemistry') for k in LAB_ID], columns=['itemid', 'label', 'fluid', 'category']))
        copy_csv('SELECT * FROM dli', H / 'd_labitems.csv', f'{site} d_labitems')
        return n

    def chartevents():
        views()
        reg('imv', IM_ALL[(IM_ALL['site'] == site) & IM_ALL['src'].isin(['vital', 'urine']) & (IM_ALL['role'] == '주')][['ITEMID', 'var']])
        reg('vid', pd.DataFrame(list(VIT_ID.items()), columns=['var', 'itemid']))
        fv = [f for f in sorted(glob.glob(str(OUT / f'vital_{site}*.parquet'))) if '.part' not in f and not f.endswith('.tmp')]
        fu = [f for f in sorted(glob.glob(str(OUT / f'urine_{site}*.parquet'))) if '.part' not in f and not f.endswith('.tmp')]
        assert fv, '활력징후 parquet 없음'
        n = copy_csv(f'''SELECT DISTINCT kp.subject_id, kh.hadm_id, ks.stay_id, vid.itemid, {shifted('x.CHARTTIME')} AS charttime,
                    CASE WHEN {store_ok('x.CHARTTIME', 'x.STORETIME')} THEN {shifted('x.STORETIME')} END AS storetime,
                    CAST(round(CASE WHEN imv.var = 'TEMP' AND try_cast(x.v AS DOUBLE) > 45 THEN (try_cast(x.v AS DOUBLE) - 32) * 5 / 9 ELSE try_cast(x.v AS DOUBLE) END, 2) AS VARCHAR) AS value,
                    round(CASE WHEN imv.var = 'TEMP' AND try_cast(x.v AS DOUBLE) > 45 THEN (try_cast(x.v AS DOUBLE) - 32) * 5 / 9 ELSE try_cast(x.v AS DOUBLE) END, 2) AS valuenum,
                    CASE imv.var WHEN 'TEMP' THEN '°C' WHEN 'SPO2' THEN '%' WHEN 'HR' THEN 'bpm' WHEN 'RR' THEN 'insp/min' ELSE 'mmHg' END AS valueuom
             FROM read_parquet({fv!r}, union_by_name=true) x JOIN imv ON imv.ITEMID = trim(CAST(x.ITEMID AS VARCHAR))
             JOIN KH kh ON kh.hid = trim(CAST(x.HADM_ID AS VARCHAR)) JOIN K k ON k.hid = kh.hid JOIN KS ks ON ks.stid = k.stid JOIN KP kp ON kp.sid = kh.sid JOIN SH sh ON sh.sid = kh.sid
             JOIN vid ON vid.var = imv.var WHERE try_cast(x.v AS DOUBLE) IS NOT NULL AND x.CHARTTIME IS NOT NULL AND {shifted('x.CHARTTIME')} IS NOT NULL''', I / 'chartevents.csv', f'{site} chartevents')
        cols = 'subject_id, hadm_id, stay_id, charttime, storetime, itemid, value, valueuom'
        if fu:
            copy_csv(f'''SELECT * FROM (SELECT kp.subject_id, kh.hadm_id, ks.stay_id, {shifted('x.CHARTTIME')} AS charttime,
                        CASE WHEN {store_ok('x.CHARTTIME', 'x.STORETIME')} THEN {shifted('x.STORETIME')} END AS storetime, 226559 AS itemid,
                        round(try_cast(x.v AS DOUBLE), 1) AS value, 'mL' AS valueuom
                 FROM read_parquet({fu!r}, union_by_name=true) x JOIN imv ON imv.ITEMID = trim(CAST(x.ITEMID AS VARCHAR)) AND imv.var = 'URINE'
                 JOIN KH kh ON kh.hid = trim(CAST(x.HADM_ID AS VARCHAR)) JOIN K k ON k.hid = kh.hid JOIN KS ks ON ks.stid = k.stid JOIN KP kp ON kp.sid = kh.sid JOIN SH sh ON sh.sid = kh.sid
                 WHERE try_cast(x.v AS DOUBLE) >= 0 AND x.CHARTTIME IS NOT NULL) WHERE charttime IS NOT NULL''', I / 'outputevents.csv', f'{site} outputevents')
        else:
            copy_csv(f"SELECT * FROM (SELECT 1::BIGINT AS subject_id, 1::BIGINT AS hadm_id, 1::BIGINT AS stay_id, ''::VARCHAR AS charttime, ''::VARCHAR AS storetime, 226559 AS itemid, 0.0 AS value, 'mL' AS valueuom) WHERE false", I / 'outputevents.csv', f'{site} outputevents (빈 표)')
        return n

    def drugs():
        views()
        pe = OUT / f'emar_vaso_{site}.parquet'
        empty_in = "SELECT * FROM (SELECT 1::BIGINT AS subject_id, 1::BIGINT AS hadm_id, 1::BIGINT AS stay_id, 1 AS itemid, ''::VARCHAR AS starttime, ''::VARCHAR AS endtime, ''::VARCHAR AS storetime, NULL AS amount, NULL AS amountuom, NULL AS rate, NULL AS rateuom, ''::VARCHAR AS statusdescription, NULL AS orderid) WHERE false"
        empty_em = "SELECT * FROM (SELECT 1::BIGINT AS subject_id, 1::BIGINT AS hadm_id, ''::VARCHAR AS charttime, ''::VARCHAR AS storetime, ''::VARCHAR AS medication, ''::VARCHAR AS event_txt) WHERE false"
        n = 0
        if pe.exists() and pq_count(pe):
            e = con.execute(f"SELECT trim(CAST(HADM_ID AS VARCHAR)) AS hid, CAST(MEDICATION AS VARCHAR) AS MEDICATION, CAST(EVENT_TXT AS VARCHAR) AS EVENT_TXT, CAST(CHARTTIME AS VARCHAR) AS CHARTTIME, CAST(STORETIME AS VARCHAR) AS STORETIME FROM read_parquet('{pe}') WHERE trim(CAST(HADM_ID AS VARCHAR)) IN (SELECT hid FROM KH)").df()
            e = e[~e['MEDICATION'].astype(str).str.lower().str.contains(EXC, regex=True)].copy()
            e['c'] = e['MEDICATION'].map(vaso_class)
            bag = e['MEDICATION'].astype(str).str.lower().str.contains(r'bag|premix|프리믹스|/(?:100|200|250|500) ?ml', regex=True)
            e['start'] = e['c'].isin(['NE', 'VASO', 'DOPA']) | (e['c'].isin(['EPI', 'PHE']) & bag)
            e['itemid'] = e['c'].map(lambda c: VASO_ID[c][0] if c in VASO_ID else None)
            reg('e', e[['hid', 'MEDICATION', 'EVENT_TXT', 'CHARTTIME', 'STORETIME', 'start', 'itemid']])
            n = copy_csv(f'''SELECT * FROM (SELECT kp.subject_id, kh.hadm_id, ks.stay_id, CAST(e.itemid AS INTEGER) AS itemid, {shifted('e.CHARTTIME')} AS starttime,
                        {shifted('e.CHARTTIME')} AS endtime, CASE WHEN {store_ok('e.CHARTTIME', 'e.STORETIME')} THEN {shifted('e.STORETIME')} END AS storetime,
                        NULL AS amount, NULL AS amountuom, NULL AS rate, NULL AS rateuom, 'FinishedRunning' AS statusdescription, NULL AS orderid
                 FROM e JOIN KH kh ON kh.hid = e.hid JOIN K k ON k.hid = kh.hid JOIN KS ks ON ks.stid = k.stid JOIN KP kp ON kp.sid = kh.sid JOIN SH sh ON sh.sid = kh.sid
                 WHERE e.start AND e.EVENT_TXT = 'Administered' AND e.itemid IS NOT NULL) WHERE starttime IS NOT NULL''', I / 'inputevents.csv', f'{site} inputevents')
            copy_csv(f'''SELECT * FROM (SELECT kp.subject_id, kh.hadm_id, {shifted('e.CHARTTIME')} AS charttime,
                        CASE WHEN {store_ok('e.CHARTTIME', 'e.STORETIME')} THEN {shifted('e.STORETIME')} END AS storetime, e.MEDICATION AS medication, e.EVENT_TXT AS event_txt
                 FROM e JOIN KH kh ON kh.hid = e.hid JOIN KP kp ON kp.sid = kh.sid JOIN SH sh ON sh.sid = kh.sid) WHERE charttime IS NOT NULL''', H / 'emar.csv', f'{site} emar')
        else:
            log(f'    {site}: 승압제 투여 기록 없음 → 빈 표 (승압제 변수는 0)')
            copy_csv(empty_in, I / 'inputevents.csv', f'{site} inputevents (빈 표)'); copy_csv(empty_em, H / 'emar.csv', f'{site} emar (빈 표)')
        rows = [(k, v) for k, v in VIT_LABEL.items()] + list(VASO_ID.values())
        reg('dit', pd.DataFrame(rows, columns=['itemid', 'label']).assign(abbreviation='', linksto='', category='', unitname=''))
        copy_csv('SELECT * FROM dit', I / 'd_items.csv', f'{site} d_items')
        return n

    def check():
        r = {}
        for d, t in [('icu', 'icustays'), ('hosp', 'labevents'), ('icu', 'chartevents')]:
            col = 'intime' if t == 'icustays' else 'charttime'
            n, bad = con.execute(f"SELECT count(*), count(*) FILTER (WHERE try_cast({col} AS TIMESTAMP) IS NULL) FROM read_csv_auto('{root / d / (t + '.csv')}', all_varchar=true)").fetchone()
            r[t] = {'rows': n, 'bad_time': bad}
        lab = con.execute(f'''SELECT itemid, count(*) n, round(100.0 * avg((storetime IS NOT NULL)::INT), 1) AS store_pct, round(median(try_cast(valuenum AS DOUBLE)), 2) AS p50
                              FROM read_csv_auto('{root / 'hosp' / 'labevents.csv'}', all_varchar=true) GROUP BY 1 ORDER BY 1''').df()
        lab['var'] = lab.itemid.astype(int).map({v: k for k, v in LAB_ID.items()})
        print(lab[['var', 'n', 'store_pct', 'p50']].to_string(index=False))
        return r

    if not task(f'convert:{site}:keys', keys, [P['SH'], P['K'], P['KH'], P['KS'], P['KP']]):
        if not ok(f'convert:{site}:keys'):
            return False
    steps = [('patients', patients, [H / 'patients.csv']), ('admissions', admissions, [H / 'admissions.csv']), ('icustays', icustays, [I / 'icustays.csv']),
             ('diagnoses_icd', diagnoses, [H / 'diagnoses_icd.csv']), ('labevents', labevents, [H / 'labevents.csv', H / 'd_labitems.csv']),
             ('chartevents', chartevents, [I / 'chartevents.csv', I / 'outputevents.csv']), ('drugs', drugs, [I / 'inputevents.csv', H / 'emar.csv', I / 'd_items.csv'])]
    for name, fn, outs in steps:
        task(f'convert:{site}:{name}', fn, outs, after=[f'convert:{site}:keys'])
    if all(ok(f'convert:{site}:{n}') for n, _, _ in steps):
        task(f'convert:{site}:check', check, after=[f'convert:{site}:{n}' for n, _, _ in steps])
        return True
    return False

CONVERTED = []
if RUN['convert']:
    for i, s in enumerate(SITES):
        if s not in READY:
            continue
        log(f'===== 변환 {s} ({READY.index(s) + 1}/{len(READY)}) =====')
        reset()
        if convert_site(s, i):
            CONVERTED.append(s)
    log(f'변환 끝난 병원: {CONVERTED}')
else:
    CONVERTED = [s for s in READY if ok(f'convert:{s}:check')]"""))

C.append(code("""# ===== 4. 2단계: 병원별 분석 (코호트·사건·결과 판정까지만) =====
import nbformat
from nbclient import NotebookClient
MAIN_NB = PKG / 'notebooks' / 'KCD_MIMICIV_liver_agents_v2.ipynb'
ORIG_ROOT = "ROOT = Path(r'\\\\\\\\10.7.12.146\\\\데이터 및 소스\\\\MIMIC\\\\MIMIC4-V3.1')"
STOP_MARK = "10_outcomes_EVALUATION_ONLY.csv')"
NEED = ['01_cohort_private.csv', '03_labs_private.csv', '05_events_private.csv', '07_context_private.csv', '08_features.csv', '10_outcomes_EVALUATION_ONLY.csv']

def analyze_site(site):
    final = WORK / 'analysis' / site
    work = WORK / 'analysis' / (site + '_running')
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    nb = nbformat.read(MAIN_NB, as_version=4)
    stop = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and STOP_MARK in c.source)
    nb.cells = nb.cells[:stop + 1]
    subs = [(ORIG_ROOT, f"ROOT = Path(r'{MF / site}')"), ("DATA_SOURCE = 'mimic-iv'", "DATA_SOURCE = 'external'"),
            ("RUN_LLM=True", "RUN_LLM=False")]
    for a, b in subs:
        hit = 0
        for c in nb.cells:
            if a in c.source:
                c.source = c.source.replace(a, b); hit += 1
        if not hit and a == ORIG_ROOT:
            raise RuntimeError('분석 노트북에서 ROOT 줄을 찾지 못했습니다')
    code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code']
    t, last = time.time(), [time.time()]
    def done(cell, cell_index, **kw):
        dt = time.time() - last[0]; last[0] = time.time()
        head = next((l.strip() for l in cell.source.splitlines() if l.strip()), '')[:60]
        k = code_idx.index(cell_index) + 1 if cell_index in code_idx else cell_index
        log(f'    [{site} 분석] 셀 {k}/{len(code_idx)} ({dt:.0f}초): {head}')
    os.environ['KCD_REPO'] = str(PKG)
    plog = work / 'progress.log'
    halt = threading.Event()
    def tail():
        pos = 0
        while not halt.wait(HEARTBEAT):
            try:
                with open(plog, encoding='utf-8') as f:
                    f.seek(pos); new = f.read(); pos = f.tell()
                for l in new.splitlines():
                    log(f'    [{site} 분석] {l}')
            except OSError:
                pass
    th = threading.Thread(target=tail, daemon=True); th.start()
    try:
        try:
            client = NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': str(work)}}, on_cell_executed=done)
        except TypeError:
            client = NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': str(work)}})
        client.execute()
    finally:
        halt.set(); th.join(timeout=2)
        nbformat.write(nb, work / 'executed_PRIVATE.ipynb')
    out = work / 'kcd_liver_v2_outputs'
    miss = [n for n in NEED if not (out / n).exists()]
    if miss:
        raise RuntimeError(f'분석 결과 파일 없음: {miss}')
    if final.exists():
        shutil.rmtree(final)
    os.replace(work, final)
    ev = pd.read_csv(final / 'kcd_liver_v2_outputs' / '05_events_private.csv', usecols=['case_id'])
    fl = pd.read_csv(final / 'kcd_liver_v2_outputs' / '04_cohort_flow.csv')
    return {'minutes': round((time.time() - t) / 60, 1), 'events': int(len(ev)), 'flow': dict(zip(fl.step, fl.n))}

ANALYZED = []
if RUN['analyze']:
    for s in CONVERTED:
        log(f'===== 분석 {s} ({CONVERTED.index(s) + 1}/{len(CONVERTED)}) =====')
        task(f'analyze:{s}', lambda s=s: analyze_site(s), [WORK / 'analysis' / s / 'kcd_liver_v2_outputs' / n for n in NEED], after=[f'convert:{s}:check'])
        if ok(f'analyze:{s}'):
            ANALYZED.append(s)
else:
    ANALYZED = [s for s in CONVERTED if ok(f'analyze:{s}')]
log(f'분석 끝난 병원: {ANALYZED}')"""))

C.append(code("""# ===== 5. 3단계: 7개 병원 결과 합치기 (전체 검증용) =====
POOL = WORK / 'analysis' / 'pooled' / 'kcd_liver_v2_outputs'
POOL_ROOT = WORK / 'private' / 'mimic_format' / 'pooled'
def pool():
    reset()
    POOL.mkdir(parents=True, exist_ok=True); (POOL_ROOT / 'hosp').mkdir(parents=True, exist_ok=True)
    dirs = [WORK / 'analysis' / s / 'kcd_liver_v2_outputs' for s in ANALYZED]
    for n in NEED:
        fs = [str(d / n) for d in dirs if (d / n).exists()]
        copy_csv(f"SELECT * FROM read_csv_auto({fs!r}, union_by_name=true, all_varchar=true)", POOL / n, f'합치기 {n}')
    fs = [str(d / '02c_value_format_audit.csv') for d in dirs if (d / '02c_value_format_audit.csv').exists()]
    if fs:
        copy_csv(f"SELECT source, analyte, kind, sum(try_cast(n AS BIGINT)) AS n FROM read_csv_auto({fs!r}, union_by_name=true, all_varchar=true) GROUP BY 1, 2, 3", POOL / '02c_value_format_audit.csv', '합치기 형식 감사')
    fs = [str(d / '04_cohort_flow.csv') for d in dirs]
    copy_csv(f"SELECT step, sum(try_cast(n AS BIGINT)) AS n FROM read_csv_auto({fs!r}, all_varchar=true) GROUP BY 1", POOL / '04_cohort_flow.csv', '합치기 흐름')
    for t in ('admissions', 'diagnoses_icd'):
        fs = [str(MF / s / 'hosp' / f'{t}.csv') for s in ANALYZED]
        copy_csv(f"SELECT * FROM read_csv_auto({fs!r}, union_by_name=true, all_varchar=true)", POOL_ROOT / 'hosp' / f'{t}.csv', f'합치기 {t}')
    return {'sites': ANALYZED}
if RUN['pooled'] and len(ANALYZED) > 1:
    if (STATE.get('pool', {}).get('result') or {}).get('sites') != ANALYZED:
        REDO.add('pool')
    task('pool', pool, [POOL / n for n in NEED], after=[f'analyze:{s}' for s in ANALYZED])"""))

C.append(code("""# ===== 6. 4단계: 고정 모델 검증 (병원별 + 전체) =====
sys.path.insert(0, str(PKG))
from kcd_liver.external import validate
from scripts.develop_models import read_hosp

def public_clean(R):
    \"\"\"Close subtraction leaks before anything leaves the server (cause 2x2 tables, confusion matrices).\"\"\"
    cells = ('tp', 'fp', 'fn', 'tn')
    for r in (R.get('agent1_flags') or {}).get('per_cause') or []:
        hid = [c for c in cells if c in r and r[c] is None]
        if hid:
            if len(hid) == 1:
                rest = sorted((c for c in cells if isinstance(r.get(c), int) and r[c] > 0), key=lambda c: r[c])
                if rest:
                    r[rest[0]] = None
            for k in ('flagged', 'proxy_positive', 'sensitivity', 'sensitivity_ci', 'specificity', 'specificity_ci'):
                r[k] = None
    P = R.get('prognosis') or {}
    if isinstance(P.get('confusion'), dict) and any(v is None for v in P['confusion'].values()):
        P['confusion'] = None
    return R

def validate_site(name, out_dir, data_root):
    adm = read_hosp(data_root, 'admissions', ['subject_id', 'hadm_id', 'dischtime'])
    dx = read_hosp(data_root, 'diagnoses_icd', ['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
    R = public_clean(validate(out_dir, PKG / 'models', adm, dx, boot=BOOT, min_cell=MIN_CELL, site=f'kmimic_{name}', log=lambda m: log(f'    [{name} 검증] {m}')))
    _write_json(WORK / 'results' / f'results_external_{name}.json', R)
    a3, dth = R.get('agent3') or {}, R.get('death_icu') or {}
    return {'stays': R['meta']['icu_stays'], 'events': R['meta']['events'], 'agent3_auroc': a3.get('auroc'), 'death_auroc': dth.get('auroc')}

VALIDATED = []
if RUN['validate']:
    targets = [(s, WORK / 'analysis' / s / 'kcd_liver_v2_outputs', MF / s, [f'analyze:{s}']) for s in ANALYZED]
    if RUN['pooled'] and len(ANALYZED) > 1 and ok('pool'):
        targets.append(('pooled', POOL, POOL_ROOT, ['pool']))
    for name, od, dr, dep in targets:
        log(f'===== 검증 {name} ({[t[0] for t in targets].index(name) + 1}/{len(targets)}) =====')
        task(f'validate:{name}', lambda name=name, od=od, dr=dr: validate_site(name, od, dr), [WORK / 'results' / f'results_external_{name}.json'], after=dep)
        if ok(f'validate:{name}'):
            VALIDATED.append(name)
log(f'검증 끝: {VALIDATED}')"""))

C.append(code("""# ===== 7. 5단계: 반출용 묶음 (집계만) =====
FORBIDDEN = {'subject_id', 'hadm_id', 'stay_id', 'labevent_id', 'case_id', 'sid', 'hid', 'stid'}
def ids_in(o, hits=None):
    hits = [] if hits is None else hits
    if isinstance(o, dict):
        for k, v in o.items():
            if k in FORBIDDEN: hits.append(k)
            ids_in(v, hits)
    elif isinstance(o, list):
        for v in o: ids_in(v, hits)
    return hits

def package():
    rows = []
    for name in VALIDATED:
        R = json.loads((WORK / 'results' / f'results_external_{name}.json').read_text(encoding='utf-8'))
        if ids_in(R):
            raise RuntimeError(f'{name}: 환자 번호 키가 있어 묶지 않습니다')
        g = lambda k, f='auroc': (R.get(k) or {}).get(f)
        tr = [r for r in (R.get('trajectory') or {}).get('performance') or [] if r.get('horizon_h') == 24]
        rows.append({'site': name, 'icu_stays': R['meta']['icu_stays'], 'events': R['meta']['events'],
                     'agent3_auroc': g('agent3'), 'agent3_auroc_ci': g('agent3', 'auroc_ci'), 'agent3_cal_slope': g('agent3', 'cal_slope'),
                     'death_auroc': g('death_icu'), 'death_auroc_ci': g('death_icu', 'auroc_ci'), 'prognosis_accuracy': g('prognosis', 'accuracy'),
                     'trajectory_24h_mae_model': np.mean([r['mae_model'] for r in tr]) if tr else None,
                     'trajectory_24h_mae_last_value': np.mean([r['mae_persistence'] for r in tr]) if tr else None})
    S = pd.DataFrame(rows)
    for c in ('icu_stays', 'events'):
        S[c] = [('<' + str(MIN_CELL)) if isinstance(v, (int, np.integer)) and 0 < v < MIN_CELL else v for v in S[c]]
    S.to_csv(WORK / 'results' / 'summary_by_site.csv', index=False)
    print(S.to_string(index=False))
    zp = WORK / f"kcd_kmimic_external_{time.strftime('%Y%m%d_%H%M')}.zip"
    with zipfile.ZipFile(str(zp) + '.tmp', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in VALIDATED:
            z.write(WORK / 'results' / f'results_external_{name}.json', f'results_external_{name}.json')
        z.write(WORK / 'results' / 'summary_by_site.csv', 'summary_by_site.csv')
        for s in ANALYZED + (['pooled'] if 'pooled' in VALIDATED else []):
            fl = WORK / 'analysis' / s / 'kcd_liver_v2_outputs' / '04_cohort_flow.csv'
            if fl.exists():
                f = pd.read_csv(fl)
                f['n'] = [('<' + str(MIN_CELL)) if 0 < v < MIN_CELL else int(v) for v in f['n']]
                z.writestr(f'aggregates/{s}_04_cohort_flow.csv', f.to_csv(index=False))
        z.writestr('run_state.json', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'result'} for k, v in STATE.items()}, ensure_ascii=False, indent=1, default=str))
        z.writestr('settings.json', json.dumps({'sites': SITES, 'uln': ULN, 'promote': [list(k) + [v] for k, v in PROMOTE.items()], 'store_max_h': STORE_MAX_H,
                                               'min_cell': MIN_CELL, 'boot': BOOT}, ensure_ascii=False, indent=1))
    os.replace(str(zp) + '.tmp', zp)
    return str(zp)
if RUN['package'] and VALIDATED:
    REDO.add('package')
    zp = task('package', package)
    log(f'반출 파일: {zp}')
failed = {k: v.get('error') for k, v in STATE.items() if v.get('status') == 'failed'}
log('실패한 단계: ' + (json.dumps(failed, ensure_ascii=False)[:1500] if failed else '없음'))"""))

nb = nbformat.v4.new_notebook(cells=C, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbformat.write(nb, ROOT / "notebooks" / "KCD_KMIMIC_EXTERNAL.ipynb")
print("wrote notebooks/KCD_KMIMIC_EXTERNAL.ipynb")
