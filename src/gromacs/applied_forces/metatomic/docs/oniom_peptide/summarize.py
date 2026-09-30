"""Collect jobs/*/metrics.json into summary.csv and summary.md, with means over seeds."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
jobs = [l.split() for l in (HERE / "jobs.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
KEYS = ["atoms", "ml_atoms", "nve_drift_kj_mol_ps", "nve_drift_per_atom", "nve_residual_std_kj_mol",
        "nve_temperature_k", "npt_density_kg_m3", "ring_bond_nm", "uncertainty_warnings", "nve_ms_per_step"]

rows, groups = [], defaultdict(list)
for name, box, ml, seed in jobs:
    job = HERE / "jobs" / name
    if (job / "metrics.json").exists():
        m = json.loads((job / "metrics.json").read_text())
        status = "done"
        groups[(float(box), ml)].append(m)
    else:
        m = {}
        tries = (job / ".tries").read_text().strip() if (job / ".tries").exists() else "0"
        stages = sorted(p.name[1:-5] for p in job.glob(".*.done")) if job.exists() else []
        status = f"pending (tries {tries}, done: {','.join(stages) or '-'})"
    rows.append({"job": name, "box_nm": box, "ml_region": ml, "seed": seed, "status": status,
                 **{k: m.get(k, "") for k in KEYS}})

with open(HERE / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

fmt = lambda v, d: "" if v == "" else f"{v:.{d}f}" if isinstance(v, float) else str(v)
md = ["# ONIOM campaign summary", "", f"{sum(r['status'] == 'done' for r in rows)}/{len(rows)} jobs done.", "",
      "## Means over seeds", "",
      "| box (nm) | ML region | seeds | atoms | ML atoms | NVE drift (kJ/mol/ps) | NVE residual std (kJ/mol) | T NVE (K) | density NpT (kg/m3) | ms/step |",
      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for (box, ml), ms in sorted(groups.items()):
    a = lambda k: np.array([m[k] for m in ms], float)
    pm = lambda k, d: f"{a(k).mean():.{d}f}" + (f" ± {a(k).std(ddof=1):.{d}f}" if len(ms) > 1 else "")
    md.append(f"| {box:.1f} | {ml} | {len(ms)} | {int(a('atoms').mean())} | {pm('ml_atoms', 0)} | {pm('nve_drift_kj_mol_ps', 2)} "
              f"| {pm('nve_residual_std_kj_mol', 2)} | {pm('nve_temperature_k', 1)} | {pm('npt_density_kg_m3', 1)} | {pm('nve_ms_per_step', 1)} |")
md += ["", "## Jobs", "", "| job | status | ML atoms | drift | std | ring bond (nm) | uncertainty warnings |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
md += [f"| {r['job']} | {r['status']} | {r['ml_atoms']} | {fmt(r['nve_drift_kj_mol_ps'], 2)} | {fmt(r['nve_residual_std_kj_mol'], 2)} "
       f"| {fmt(r['ring_bond_nm'], 3)} | {r['uncertainty_warnings']} |" for r in rows]
(HERE / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
