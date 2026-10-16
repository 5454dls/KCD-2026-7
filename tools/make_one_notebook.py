"""Build notebooks/KCD_ONE.ipynb: a single self-contained notebook (code embedded as base64 tar.gz)."""
import base64, io, subprocess, tarfile, textwrap
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
PATHS = ["kcd_liver", "scripts", "reference", "site", "notebooks/KCD_MIMICIV_liver_agents_v2.ipynb", "notebooks/KCD_EXTERNAL.ipynb", "docs/VARIABLES.md", ".gitignore"]
runner = nbformat.read(ROOT / "notebooks" / "KCD_RUN_ALL.ipynb", as_version=4)

buf = io.BytesIO()
files = subprocess.run(["git", "ls-files", *PATHS], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
    for f in files:
        tar.add(ROOT / f, arcname=f)
payload = base64.b64encode(buf.getvalue()).decode()
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

md = nbformat.v4.new_markdown_cell(runner.cells[0].source.replace("# KCD 2026 · 한 번에 전부 실행 (RUN ALL)",
    f"# KCD 2026 · 노트북 하나로 전부 실행\n\n코드 버전 `{commit}`. 이 파일 하나에 필요한 코드가 모두 들어 있습니다. 첫 셀의 `MIMIC_ROOT`만 확인하고 위에서부터 끝까지 실행하세요.")
    + "\n\n**다시 실행할 때**: 코드만 바뀐 새 노트북을 받았다면 `RUN_STEPS['full'] = False`로 두면 가장 오래 걸리는 1단계를 건너뛰고 이전 결과(`kcd_work/run_all`)를 재사용합니다.")

cfg = runner.cells[1].source
cfg = cfg.replace("REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'kcd_liver').is_dir())\n",
                  "BASE = Path.cwd() / 'kcd_work'         # 코드와 중간 산출물 (이 PC에만 남음)\nREPO = BASE / 'code'                   # 아래 셀이 코드를 여기에 풉니다\n")
cfg = cfg.replace("WORK = REPO / 'run_all'    # 환자 단위 중간 산출물 (이 PC에만 남음)", "WORK = BASE / 'run_all'    # 환자 단위 중간 산출물 (이 PC에만 남음)")
cfg = cfg.replace("DEMO_CSV = REPO / 'data' / 'mimic-iv-demo-csv'", "DEMO_CSV = BASE / 'mimic-iv-demo-csv'")
cfg = cfg.replace("sys.path.insert(0, str(REPO))\nWORK.mkdir(exist_ok=True)\n", "")
assert "BASE = Path.cwd()" in cfg and "WORK = BASE" in cfg and "DEMO_CSV = BASE" in cfg
cfg += "\nOUTBOX = Path.cwd()   # 돌려줄 zip이 이 노트북 옆에 생깁니다\n"

unpack = nbformat.v4.new_code_cell(textwrap.dedent(f'''\
    # ===== 코드 풀기 (수정하지 마세요) =====
    import base64, io, tarfile
    PAYLOAD = "{payload}"
    if REPO.exists(): shutil.rmtree(REPO)
    REPO.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode='r:gz') as t:
        try: t.extractall(REPO, filter='data')\n        except TypeError: t.extractall(REPO)
    sys.path.insert(0, str(REPO)); WORK.mkdir(parents=True, exist_ok=True)
    for m in [k for k in sys.modules if k.split('.')[0] in ('kcd_liver', 'scripts')]: del sys.modules[m]
    REPORT['code_version'] = '{commit}'
    print('코드 버전 {commit}:', sum(1 for _ in REPO.rglob('*.py')), '개 파이썬 파일 ->', REPO)
    '''))
unpack.metadata["jupyter"] = {"source_hidden": True}

cells = [md, nbformat.v4.new_code_cell(cfg), unpack]
for c in runner.cells[2:]:
    s = c.source.replace("zp = REPO / f'kcd_return_package_{stamp}.zip'", "zp = OUTBOX / f'kcd_return_package_{stamp}.zip'")
    cells.append(nbformat.v4.new_code_cell(s) if c.cell_type == "code" else nbformat.v4.new_markdown_cell(s))
assert any("zp = OUTBOX" in c.source for c in cells)
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbformat.write(nb, ROOT / "notebooks" / "KCD_ONE.ipynb")
print("wrote notebooks/KCD_ONE.ipynb", round(len(payload) / 1e6, 2), "MB payload,", len(files), "files, commit", commit)
