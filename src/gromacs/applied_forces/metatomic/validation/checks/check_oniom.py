"""End-to-end check of the ONIOM link-atom path through the metatomic C API.

Runs `gmx mdrun -rerun` with the analytic pair model (pair_model.cpp) on a
7-atom chain whose ML group (atoms 2-4) has four hydrogen link caps, two of
them on the same MM atom, with the molecule wrapped across the periodic
boundary, and checks

1. the Metatomic energy against an independent evaluation of the model on
   the ML atoms plus caps (cap positions, element and charges);
2. the total forces against central finite differences of the energy;
3. the virial against 0.5 dE/d(strain), by central finite differences on
   rectangular and sheared boxes.

usage:
    python check_oniom.py --gmx path/to/gmx --plugin path/to/libpair_model.so
                          [--ranks 2] [--no-links] [--system-energy]

`--ranks 2` runs with a 2 x 1 x 1 domain decomposition (thread-MPI);
`--no-links` disables ONIOM and link atoms; `--system-energy` makes the
model return one energy per system instead of per-atom energies (which is
only correct without domain decomposition).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L = 3.0
BASE = np.array(
    [
        [2.85, 1.50, 1.50],
        [2.98, 1.58, 1.50],
        [0.10, 1.51, 1.52],
        [0.22, 1.59, 1.50],
        [0.36, 1.53, 1.51],
        [0.50, 1.60, 1.49],
        [0.10, 1.72, 1.55],
    ]
)
NAMES = ["C0", "C1", "O2", "C3", "C4", "C5", "C6"]
Z = [6, 6, 8, 6, 6, 6, 6]
Q = [0.30, -0.20, -0.40, 0.25, 0.15, -0.30, 0.20]  # as in chain.top
ML = [1, 2, 3]
ALL_LINKS = [(1, 0), (3, 4), (1, 6), (3, 6)]  # (embedded, mm), in topology bond order
RC, A, LAMBDA, D_LINK = 0.45, 20.0, 50.0, 0.1  # model parameters and cap distance (nm)
H, DELTA = 5e-4, 5e-4  # displacement (nm) and strain for finite differences
STRAINS = [(0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)]  # GROMACS boxes are lower triangular
MOVED = [0, 1, 2, 3, 4, 6]

MDP = """integrator              = md
nsteps                  = 0
cutoff-scheme           = Verlet
pbc                     = xyz
verlet-buffer-tolerance = -1
rlist                   = 1.2
coulombtype             = Reaction-Field
epsilon-rf              = 1
rcoulomb                = 1.0
vdwtype                 = Cut-off
vdw-modifier            = Potential-shift
rvdw                    = 1.0
nstcalcenergy           = 1
nstenergy               = 1
nstfout                 = 1
; pressure coupling only makes the zero-step run write the virial
pcoupl                  = Parrinello-Rahman
pcoupltype              = anisotropic
tau-p                   = 5.0
compressibility         = 4.5e-5 4.5e-5 4.5e-5 4.5e-5 4.5e-5 4.5e-5
ref-p                   = 1 1 1 0 0 0
nstpcouple              = 1
metatomic-active        = yes
metatomic-input-group   = ML
metatomic-model         = {model}
metatomic-extensions    = {plugin}
metatomic-oniom         = {oniom}
metatomic-link-atoms    = {oniom}
"""


def minimum_image(d, box):
    frac = d @ np.linalg.inv(box)
    return (frac - np.round(frac)) @ box


def model_energy(x, box, links):
    rows = [(x[i], Z[i], Q[i]) for i in ML]
    for emb, mm in links:
        bond = minimum_image(x[mm] - x[emb], box)
        rows.append((x[emb] + D_LINK * bond / np.linalg.norm(bond), 1, Q[mm]))
    energy = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r2 = np.sum(minimum_image(rows[j][0] - rows[i][0], box) ** 2)
            if r2 < RC**2:
                w = A * (1 + 0.1 * (rows[i][1] + rows[j][1])) + LAMBDA * rows[i][2] * rows[j][2]
                energy += w * (RC**2 - r2) ** 2 / RC**4
    return energy


def gro_frame(x, box, title):
    lines = [title, str(len(x))]
    for i, (name, r) in enumerate(zip(NAMES, x)):
        lines.append(f"{1:5d}{'MOL':<5s}{name:>5s}{i + 1:5d}{r[0]:12.7f}{r[1]:12.7f}{r[2]:12.7f}")
    b = box
    lines.append(" ".join(f"{v:.9f}" for v in [b[0, 0], b[1, 1], b[2, 2], b[0, 1], b[0, 2], b[1, 0], b[1, 2], b[2, 0], b[2, 1]]))
    return "\n".join(lines) + "\n"


def xvg(path):
    text = Path(path).read_text().splitlines()
    names = [l.split('"')[1] for l in text if l.startswith("@ s") and "legend" in l]
    data = np.loadtxt([l for l in text if l and l[0] not in "#@"]).reshape(-1, len(names) + 1)
    return {name: data[:, k + 1] for k, name in enumerate(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gmx", required=True)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--no-links", action="store_true")
    parser.add_argument("--system-energy", action="store_true")
    args = parser.parse_args()

    links = [] if args.no_links else ALL_LINKS
    work = Path(tempfile.mkdtemp(prefix="oniom_check_"))
    ranks = ["-ntmpi", str(args.ranks)] + (["-dd", str(args.ranks), "1", "1"] if args.ranks > 1 else [])

    def run(cmd, **kw):
        out = subprocess.run([args.gmx, *cmd], cwd=work, capture_output=True, text=True, **kw)
        if out.returncode != 0:
            sys.exit(f"gmx {' '.join(cmd)} failed in {work}:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}")
        return out

    (work / "model.pair").write_text(f"{RC} {A} {LAMBDA} {0 if args.system_energy else 1}\n")
    oniom = "no" if args.no_links else "yes"
    (work / "rerun.mdp").write_text(MDP.format(model=work / "model.pair", plugin=args.plugin.resolve(), oniom=oniom))

    box0 = np.diag([L, L, L])
    frames = [("base", BASE, box0)]
    for atom in MOVED:
        for dim in range(3):
            for sign in (+1, -1):
                x = BASE.copy()
                x[atom, dim] += sign * H
                frames.append((f"x{atom}{dim}{sign:+d}", x, box0))
    for a, b in STRAINS:
        for sign in (+1, -1):
            eps = np.eye(3)
            eps[a, b] += sign * DELTA
            frames.append((f"e{a}{b}{sign:+d}", BASE @ eps, box0 @ eps))

    (work / "base.gro").write_text(gro_frame(BASE, box0, "base"))
    (work / "frames.gro").write_text("".join(gro_frame(x, b, t) for t, x, b in frames))
    run(["grompp", "-f", "rerun.mdp", "-c", "base.gro", "-p", HERE / "chain.top", "-n", HERE / "chain.ndx",
         "-o", "topol.tpr", "-maxwarn", "4"])
    run(["mdrun", "-s", "topol.tpr", "-rerun", "frames.gro", *ranks, "-ntomp", "1",
         "-o", "rerun.trr", "-e", "rerun.edr", "-g", "rerun.log"])
    run(["energy", "-f", "rerun.edr", "-o", "energy.xvg", "-dp"], input="Potential\nMetatomic-Potential\n\n")
    energies = xvg(work / "energy.xvg")
    potential = dict(zip([t for t, _, _ in frames], energies["Potential"]))

    first = run(["dump", "-f", "rerun.trr"]).stdout.split("f (")[1].split("frame")[0]
    forces = np.array([[float(v) for v in l.split("{")[1].split("}")[0].split(",")]
                       for l in first.splitlines() if l.strip().startswith("f[")])

    # rerun does not write the virial: take it from a zero-step run of the base frame
    run(["mdrun", "-s", "topol.tpr", "-nsteps", "0", *ranks, "-ntomp", "1", "-deffnm", "step0"])
    virial_terms = [f"Vir-{'XYZ'[a]}{'XYZ'[b]}" for a, b in STRAINS]
    run(["energy", "-f", "step0.edr", "-o", "virial.xvg", "-dp"], input="\n".join(virial_terms) + "\n\n")
    virial = {name: values[0] for name, values in xvg(work / "virial.xvg").items()}

    ok = True
    reference = model_energy(BASE, box0, links)
    energy = energies["Metatomic Potential"][0]
    print(f"{len(links)} link caps, {args.ranks} rank(s), work dir {work}")
    print(f"1. Metatomic energy {energy:.5f}, reference {reference:.5f} kJ/mol")
    ok &= abs(energy - reference) < 1e-3 * max(1.0, abs(reference))

    # single-precision energies give ~0.1 kJ/mol/nm of finite-difference noise
    worst, largest = 0.0, 0.0
    for atom in MOVED:
        for dim in range(3):
            fd = -(potential[f"x{atom}{dim}+1"] - potential[f"x{atom}{dim}-1"]) / (2 * H)
            worst, largest = max(worst, abs(fd - forces[atom, dim])), max(largest, abs(fd))
            ok &= abs(fd - forces[atom, dim]) < 0.5 + 1e-3 * abs(fd)
    print(f"2. forces: max |GROMACS - finite difference| {worst:.3f} kJ/mol/nm (largest force {largest:.0f})")

    print("3. virial      GROMACS    0.5 dE/d(strain)")
    for (a, b), name in zip(STRAINS, virial_terms):
        fd = 0.5 * (potential[f"e{a}{b}+1"] - potential[f"e{a}{b}-1"]) / (2 * DELTA)
        print(f"   {name}  {virial[name]:10.4f}  {fd:10.4f}")
        ok &= abs(virial[name] - fd) < 1.0 + 2e-3 * abs(fd)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
