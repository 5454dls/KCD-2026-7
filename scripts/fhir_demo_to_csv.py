"""Convert the public MIMIC-IV Clinical Database Demo on FHIR (ODbL v1.0) to the
MIMIC-IV CSV layout the notebook reads (hosp/*.csv, icu/*.csv).

Why FHIR: physionet.org may be unreachable from some networks, while the same
100 demo patients are mirrored in PhysioNet's open S3 bucket as FHIR NDJSON.
FHIR keeps the fields this project needs: lab itemid, charttime
(effectiveDateTime), storetime (issued) and ref_range_upper (referenceRange.high).

    python scripts/fhir_demo_to_csv.py --out data/mimic-iv-demo-csv          # downloads ~52 MB
    python scripts/fhir_demo_to_csv.py --fhir-dir path/to/fhir --out ...      # already unzipped

Known differences from the original CSV demo (documented, not hidden):
- patients.anchor_year = birth year and anchor_age = 0, so the notebook's
  age = anchor_age + intime.year - anchor_year equals intime.year - birth year.
- admissions.deathtime = dischtime when the discharge disposition is DIED.
- FHIR timestamps carry a UTC offset; it is dropped to match MIMIC's local naive times.
- emar has no storetime in FHIR; availability falls back to charttime.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://physionet-open.s3.amazonaws.com/mimic-iv-fhir-demo/mimic-iv-fhir-demo-2.1.0.zip"
VITAL_ITEMS = {220045, 220052, 220181, 225312, 220050, 220179, 225309, 220051, 220180, 225310, 220210, 224690,
               220277, 223761, 223762, 223835, 220339, 224700}


def t(s):
    return None if not s else str(s)[:19].replace("T", " ")


def ident(res, suffix):
    for i in res.get("identifier", []):
        if i.get("system", "").endswith(suffix):
            return i.get("value")
    return None


def ref(res, key):
    r = (res.get(key) or {}).get("reference", "")
    return r.split("/", 1)[1] if "/" in r else None


def coding(res, key="code"):
    c = (res.get(key) or {}).get("coding") or [{}]
    return c[0]


def read(fhir_dir: Path, name: str):
    p = fhir_dir / f"{name}.ndjson.gz"
    if not p.exists():
        return
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def locate_fhir(args) -> Path:
    if args.fhir_dir:
        return Path(args.fhir_dir)
    cache = Path(args.out) / "_fhir"
    hit = next(cache.glob("**/fhir/MimicPatient.ndjson.gz"), None) if cache.exists() else None
    if hit is None:
        print("downloading", URL)
        data = urllib.request.urlopen(URL, timeout=300).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(cache)
        hit = next(cache.glob("**/fhir/MimicPatient.ndjson.gz"))
    return hit.parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fhir-dir")
    args = ap.parse_args(argv)
    fd = locate_fhir(args)
    out = Path(args.out)
    (out / "hosp").mkdir(parents=True, exist_ok=True)
    (out / "icu").mkdir(parents=True, exist_ok=True)

    # patients
    pid, pts = {}, []
    for p in read(fd, "MimicPatient"):
        sid = int(ident(p, "identifier/patient"))
        pid[p["id"]] = sid
        sex = next((e.get("valueCode") for e in p.get("extension", []) if e["url"].endswith("us-core-birthsex")), None)
        by = int(p["birthDate"][:4])
        pts.append({"subject_id": sid, "gender": sex or {"female": "F", "male": "M"}.get(p.get("gender")),
                    "anchor_age": 0, "anchor_year": by, "anchor_year_group": "", "dod": p.get("deceasedDateTime")})
    # hospital admissions
    hadm_of, adm = {}, []
    for e in read(fd, "MimicEncounter"):
        hid = ident(e, "encounter-hosp")
        if hid is None:
            continue
        hadm_of[e["id"]] = int(hid)
        disp = coding(e.get("hospitalization", {}), "dischargeDisposition").get("code")
        died = disp == "DIED"
        adm.append({"subject_id": pid.get(ref(e, "subject")), "hadm_id": int(hid), "admittime": t(e["period"].get("start")),
                    "dischtime": t(e["period"].get("end")), "deathtime": t(e["period"].get("end")) if died else None,
                    "admission_type": e.get("class", {}).get("code"), "discharge_location": disp,
                    "hospital_expire_flag": int(died)})
    # ICU stays
    stay_of, stays = {}, []
    for e in read(fd, "MimicEncounterICU"):
        sid_ = int(ident(e, "encounter-icu"))
        stay_of[e["id"]] = sid_
        loc = None
        stays.append({"subject_id": pid.get(ref(e, "subject")), "hadm_id": hadm_of.get(ref(e, "partOf")), "stay_id": sid_,
                      "first_careunit": loc, "last_careunit": loc, "intime": t(e["period"].get("start")),
                      "outtime": t(e["period"].get("end"))})
    st = pd.DataFrame(stays)
    st["los"] = (pd.to_datetime(st.outtime) - pd.to_datetime(st.intime)).dt.total_seconds() / 86400

    # labs + dictionary
    labs, dlab = [], {}
    for o in read(fd, "MimicObservationLabevents"):
        c = coding(o)
        item = int(c["code"])
        dlab.setdefault(item, c.get("display", ""))
        q = o.get("valueQuantity") or {}
        rr = (o.get("referenceRange") or [{}])[0]
        labs.append({"labevent_id": int(ident(o, "observation-labevents")), "subject_id": pid.get(ref(o, "subject")),
                     "hadm_id": hadm_of.get(ref(o, "encounter")), "itemid": item, "charttime": t(o.get("effectiveDateTime")),
                     "storetime": t(o.get("issued")), "value": o.get("valueString", q.get("value")), "valuenum": q.get("value"),
                     "valueuom": q.get("unit"), "ref_range_lower": (rr.get("low") or {}).get("value"),
                     "ref_range_upper": (rr.get("high") or {}).get("value")})
    # charted vitals (only the items the notebook uses), output, ICU infusions
    d_items, ce, uo, ie = {}, [], [], []
    for o in read(fd, "MimicObservationChartevents"):
        c = coding(o)
        item = int(c["code"])
        if item not in VITAL_ITEMS:
            continue
        d_items.setdefault(item, c.get("display", ""))
        q = o.get("valueQuantity") or {}
        enc = ref(o, "encounter")
        ce.append({"subject_id": pid.get(ref(o, "subject")), "stay_id": stay_of.get(enc), "itemid": item,
                   "charttime": t(o.get("effectiveDateTime")), "storetime": t(o.get("issued")),
                   "valuenum": q.get("value"), "valueuom": q.get("unit")})
    for o in read(fd, "MimicObservationOutputevents"):
        c = coding(o)
        item = int(c["code"])
        d_items.setdefault(item, c.get("display", ""))
        q = o.get("valueQuantity") or {}
        uo.append({"subject_id": pid.get(ref(o, "subject")), "stay_id": stay_of.get(ref(o, "encounter")), "itemid": item,
                   "charttime": t(o.get("effectiveDateTime")), "storetime": t(o.get("issued")),
                   "value": q.get("value"), "valueuom": q.get("unit")})
    for m in read(fd, "MimicMedicationAdministrationICU"):
        c = coding(m, "medicationCodeableConcept")
        item = int(c["code"])
        d_items.setdefault(item, c.get("display", ""))
        per = m.get("effectivePeriod") or {}
        start = t(per.get("start") or m.get("effectiveDateTime"))
        dose = (m.get("dosage") or {}).get("dose") or {}
        rate = (m.get("dosage") or {}).get("rateQuantity") or {}
        ie.append({"subject_id": pid.get(ref(m, "subject")), "stay_id": stay_of.get(ref(m, "context")), "itemid": item,
                   "starttime": start, "endtime": t(per.get("end")) or start, "storetime": start,
                   "amount": dose.get("value"), "amountuom": dose.get("unit"), "rate": rate.get("value"), "rateuom": rate.get("unit"),
                   "statusdescription": {"completed": "FinishedRunning", "stopped": "Stopped", "in-progress": "ChangeDose/Rate"}.get(m.get("status"), m.get("status")),
                   "orderid": None})
    # eMAR (hospital medication administrations)
    names = {}
    for med in read(fd, "MimicMedication"):
        fc, nm = ident(med, "formulary-drug-cd"), ident(med, "medication-name")
        if fc and nm:
            names[fc] = nm
    em = []
    for m in read(fd, "MimicMedicationAdministration"):
        c = coding(m, "medicationCodeableConcept")
        code, system = c.get("code"), c.get("system", "")
        med = code if system.endswith("medication-name") else names.get(code, code)
        em.append({"subject_id": pid.get(ref(m, "subject")), "hadm_id": hadm_of.get(ref(m, "context")),
                   "charttime": t(m.get("effectiveDateTime")), "storetime": None, "medication": med,
                   "event_txt": coding((m.get("dosage") or {}), "method").get("code")})
    # diagnoses (used only from completed PREVIOUS admissions by the notebook)
    dx = []
    for c_ in read(fd, "MimicCondition"):
        c = coding(c_)
        system = c.get("system", "")
        dx.append({"subject_id": pid.get(ref(c_, "subject")), "hadm_id": hadm_of.get(ref(c_, "encounter")), "seq_num": None,
                   "icd_code": c.get("code"), "icd_version": 10 if "icd10" in system else 9})

    w = lambda rows, path: pd.DataFrame(rows).to_csv(out / path, index=False)
    w(pts, "hosp/patients.csv")
    w(adm, "hosp/admissions.csv")
    st.to_csv(out / "icu/icustays.csv", index=False)
    w(labs, "hosp/labevents.csv.gz")
    w([{"itemid": k, "label": v, "fluid": "", "category": ""} for k, v in sorted(dlab.items())], "hosp/d_labitems.csv")
    w(em, "hosp/emar.csv.gz")
    w(dx, "hosp/diagnoses_icd.csv")
    w(ce, "icu/chartevents.csv.gz")
    w(uo, "icu/outputevents.csv")
    w(ie, "icu/inputevents.csv.gz")
    w([{"itemid": k, "label": v} for k, v in sorted(d_items.items())], "icu/d_items.csv")
    print(f"patients {len(pts)} admissions {len(adm)} icustays {len(st)} labevents {len(labs)} "
          f"chartevents(vitals) {len(ce)} emar {len(em)} inputevents {len(ie)} -> {out}")


if __name__ == "__main__":
    main()
