"""Metrics of one campaign job: analyze.py JOB_DIR GMX -> JOB_DIR/metrics.json."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

job, gmx = Path(sys.argv[1]), sys.argv[2]


def energy(edr, terms):
    out = job / f"_{edr}.xvg"
    subprocess.run([gmx, "energy", "-f", f"{edr}.edr", "-o", out.name], cwd=job, input="\n".join(terms) + "\n0\n",
                   capture_output=True, text=True, check=True)
    lines = out.read_text().splitlines()
    names = [l.split('"')[1] for l in lines if l.startswith("@ s") and "legend" in l]
    data = np.loadtxt([l for l in lines if l and l[0] not in "#@"])
    out.unlink()
    return data[:, 0], {n: data[:, k + 1] for k, n in enumerate(names)}


def performance(log):
    m = re.search(r"Performance:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", (job / log).read_text())
    return {"ns_per_day": float(m[1]), "ms_per_step": float(m[3])} if m else {}


def ml_atoms():
    text = (job / "ml.ndx").read_text()
    return len(text.split("]", 1)[1].split())


t, e = energy("nve", ["Total-Energy", "Temperature", "Metatomic-Potential"])
etot = e["Total Energy"]
fit = np.polyfit(t, etot, 1)
_, p = energy("npt", ["Density", "Volume", "Temperature"])
half = len(p["Density"]) // 2  # second half of the NpT run
gro = (job / "nve.gro").read_text().splitlines()
natoms = int(gro[1])
xyz = lambda i: np.array([float(gro[2 + i][20 + 8 * k:28 + 8 * k]) for k in range(3)])
L = float(gro[-1].split()[0])
ring = xyz(85) - xyz(0)  # ALA6 C - GLY1 N, the bond that closes the ring
ring -= L * np.round(ring / L)

metrics = {
    "atoms": natoms,
    "ml_atoms": ml_atoms(),
    "box_nm": float(gro[-1].split()[0]),
    "ring_bond_nm": float(np.linalg.norm(ring)),
    "nve_drift_kj_mol_ps": float(fit[0]),
    "nve_drift_per_atom": float(fit[0] / natoms),
    "nve_residual_std_kj_mol": float((etot - np.polyval(fit, t)).std()),
    "nve_temperature_k": float(e["Temperature"].mean()),
    "npt_density_kg_m3": float(p["Density"][half:].mean()),
    "npt_temperature_k": float(p["Temperature"][half:].mean()),
    "uncertainty_warnings": sum((job / f).read_text().count("uncertainty on atomic") for f in ("npt.out", "nve.out")),
    **{f"nve_{k}": v for k, v in performance("nve.log").items()},
}
(job / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps(metrics, indent=1))
