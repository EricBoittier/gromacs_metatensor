"""End-to-end check of metatomic-oniom with an ML region wider than rcoulomb.

The ML atoms are mutually excluded, and the non-bonded kernels only treat
excluded pairs within rcoulomb. `rod.top` has a 6-atom ML rod (1.3 nm long)
and four MM ions; rcoulomb is set just below the distance of ML atoms 0 and 3,
so the finite-difference and strain frames move that pair across the cut-off.
With `gmx mdrun -rerun` and the analytic pair model (pair_model.cpp) this checks

1. with PME, that the Coulomb energy (SR + reciprocal) does not depend on
   rcoulomb: the ML-ML pairs beyond it must not keep their MM interaction;
2. the forces against central finite differences of the potential energy;
3. the virial against 0.5 dE/d(strain).

usage:
    python check_wide_ml.py --gmx path/to/gmx --plugin path/to/libpair_model.so
                            [--coulomb pme|rf] [--ranks 2]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from check_oniom import xvg

HERE = Path(__file__).resolve().parent
L = 3.4
ROD = [[1.0 + 0.26 * i, 1.6 + 0.04 * (-1) ** i, 1.6 + 0.01 * i] for i in range(6)]
IONS = [[1.3, 2.2, 1.6], [2.0, 1.0, 1.7], [1.6, 2.3, 1.2], [1.1, 1.0, 1.9]]
BASE = np.array(ROD + IONS)
NAMES = [("ROD", n) for n in ["C0", "O1", "C2", "O3", "C4", "O5"]] + [("ION", "C")] * 2 + [("ANI", "O")] * 2
RC_MODEL, A, LAMBDA = 0.45, 20.0, 50.0
RC_LARGE = 1.6  # every ML pair within the cut-off: the kernels handle all exclusions
H, DELTA = 2e-3, 2e-3  # larger than for check_oniom.py: single-precision PME energies are noisier
STRAINS = [(0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)]

MDP = """integrator              = md
nsteps                  = 0
cutoff-scheme           = Verlet
pbc                     = xyz
verlet-buffer-tolerance = -1
rlist                   = {rc}
coulombtype             = {coulombtype}
epsilon-rf              = 0
rcoulomb                = {rc}
ewald-rtol              = 1e-6
fourierspacing          = 0.04
pme-order               = 8
vdwtype                 = Cut-off
rvdw                    = {rc}
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
metatomic-oniom         = yes
"""


def gro_frame(x, box, title):
    atoms = [f"{i + 1:5d}{res:<5s}{name:>5s}{i + 1:5d}{r[0]:12.7f}{r[1]:12.7f}{r[2]:12.7f}"
             for i, ((res, name), r) in enumerate(zip(NAMES, x))]
    b = box
    cell = " ".join(f"{v:.9f}" for v in [b[0, 0], b[1, 1], b[2, 2], b[0, 1], b[0, 2], b[1, 0], b[1, 2], b[2, 0], b[2, 1]])
    return "\n".join([title, str(len(x)), *atoms, cell]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gmx", required=True)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--coulomb", choices=["pme", "rf"], default="pme")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="wide_ml_check_"))
    ranks = ["-ntmpi", str(args.ranks)] + (["-dd", str(args.ranks), "1", "1"] if args.ranks > 1 else [])

    def run(cmd, **kw):
        out = subprocess.run([args.gmx, *cmd], cwd=work, capture_output=True, text=True, **kw)
        if out.returncode != 0:
            sys.exit(f"gmx {' '.join(cmd)} failed in {work}:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}")
        return out

    (work / "model.pair").write_text(f"{RC_MODEL} {A} {LAMBDA} 1\n")
    coulombtype = {"pme": "PME", "rf": "Reaction-Field"}[args.coulomb]

    def rerun(name, rc, frames):
        (work / f"{name}.mdp").write_text(MDP.format(rc=rc, coulombtype=coulombtype, model=work / "model.pair",
                                                     plugin=args.plugin.resolve()))
        (work / f"{name}.gro").write_text(frames[0][1])
        (work / f"{name}_frames.gro").write_text("".join(f for _, f in frames))
        run(["grompp", "-f", f"{name}.mdp", "-c", f"{name}.gro", "-p", HERE / "rod.top", "-n", HERE / "rod.ndx",
             "-o", f"{name}.tpr", "-maxwarn", "4"])
        run(["mdrun", "-s", f"{name}.tpr", "-rerun", f"{name}_frames.gro", *ranks, "-ntomp", "1",
             "-o", f"{name}.trr", "-e", f"{name}.edr", "-g", f"{name}.log"])
        run(["energy", "-f", f"{name}.edr", "-o", f"{name}.xvg", "-dp"],
            input="Potential\nCoulomb-(SR)\n" + ("Coul.-recip.\n" if args.coulomb == "pme" else "") + "\n")
        energies = xvg(work / f"{name}.xvg")
        coulomb = energies["Coulomb (SR)"] + energies.get("Coul. recip.", 0)
        return dict(zip([t for t, _ in frames], energies["Potential"])), dict(zip([t for t, _ in frames], coulomb))

    box0 = np.diag([L, L, L])
    rc = np.linalg.norm(BASE[3] - BASE[0]) - 2e-4
    frames = [("base", gro_frame(BASE, box0, "base"))]
    for atom in range(len(BASE)):
        for dim in range(3):
            for sign in (+1, -1):
                x = BASE.copy()
                x[atom, dim] += sign * H
                frames.append((f"x{atom}{dim}{sign:+d}", gro_frame(x, box0, "x")))
    for a, b in STRAINS:
        for sign in (+1, -1):
            eps = np.eye(3)
            eps[a, b] += sign * DELTA
            frames.append((f"e{a}{b}{sign:+d}", gro_frame(BASE @ eps, box0 @ eps, "e")))

    potential, coulomb = rerun("cross", rc, frames)
    first = run(["dump", "-f", "cross.trr"]).stdout.split("f (")[1].split("frame")[0]
    forces = np.array([[float(v) for v in l.split("{")[1].split("}")[0].split(",")]
                       for l in first.splitlines() if l.strip().startswith("f[")])
    run(["mdrun", "-s", "cross.tpr", "-nsteps", "0", *ranks, "-ntomp", "1", "-deffnm", "step0"])
    virial_terms = [f"Vir-{'XYZ'[a]}{'XYZ'[b]}" for a, b in STRAINS]
    run(["energy", "-f", "step0.edr", "-o", "virial.xvg", "-dp"], input="\n".join(virial_terms) + "\n\n")
    virial = {name: values[0] for name, values in xvg(work / "virial.xvg").items()}

    ok = True
    width = max(np.linalg.norm(BASE[i] - BASE[j]) for i in range(6) for j in range(6))
    print(f"{coulombtype}, {args.ranks} rank(s), ML width {width:.3f} nm, rcoulomb {rc:.4f} nm, work dir {work}")
    if args.coulomb == "pme":
        _, reference = rerun("large", RC_LARGE, frames[:1])
        diff = coulomb["base"] - reference["base"]
        print(f"1. Coulomb energy {coulomb['base']:.4f} with rcoulomb {rc:.4f}, "
              f"{reference['base']:.4f} with {RC_LARGE} nm (diff {diff:.4f} kJ/mol)")
        ok &= abs(diff) < 0.05

    # single-precision energies give ~0.1 kJ/mol/nm of finite-difference noise
    worst, largest = 0.0, 0.0
    for atom in range(len(BASE)):
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
