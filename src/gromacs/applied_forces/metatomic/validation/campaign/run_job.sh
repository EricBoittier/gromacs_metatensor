#!/usr/bin/env bash
# Run one campaign job: run_job.sh NAME BOX ML SEED
# Stages are skipped once done (.<stage>.done) and mdrun resumes from its
# checkpoint, so the script can be restarted at any point.
set -euo pipefail
CAMPAIGN=$(cd "$(dirname "$0")" && pwd)
GMX=${GMX:-/home/boittier/metawork/gromacs-oniom-torch/build/bin/gmx}
PY=${PY:-/home/boittier/metawork/.venv/bin/python}
NT=${NT:-16}
name=$1 box=$2 ml=$3 seed=$4
job=${JOBS:-$CAMPAIGN/jobs}/$name
mkdir -p "$job" && cd "$job"
T=${TEMPLATES:-$CAMPAIGN/templates}

done_() { [[ -f .$1.done ]]; }
mark() { touch ".$1.done"; echo "$(date -Is) $name: $1 done"; }
gmx() { "$GMX" "$@"; }
mdrun() {  # mdrun DEFFNM: resume from the checkpoint when there is one
    local cpi=(); [[ -f $1.cpt ]] && cpi=(-cpi "$1.cpt")
    gmx mdrun -deffnm "$1" -nt "$NT" "${cpi[@]}" > "$1.out" 2>&1
}

if ! done_ build; then
    cp "$CAMPAIGN"/base/{segetalinA_heavy.pdb,specbond.dat} .
    printf "0\n0\n" | gmx pdb2gmx -f segetalinA_heavy.pdb -ff amber99sb-ildn -water tip3p -ignh -ter \
        -o pep.gro -p topol.top -i posre.itp > build.log 2>&1
    gmx editconf -f pep.gro -o box.gro -box "$box" -c -bt cubic >> build.log 2>&1
    gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro >> build.log 2>&1
    mark build
fi

if ! done_ em; then
    gmx grompp -f "$T/em.mdp" -c solv.gro -p topol.top -o em.tpr > grompp_em.log 2>&1
    mdrun em && mark em
fi

if ! done_ nvt; then
    sed "s/^gen-seed .*/gen-seed                = $seed/" "$T/nvt.mdp" > nvt.mdp
    gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr > grompp_nvt.log 2>&1
    mdrun nvt && mark nvt
fi

if ! done_ index; then
    # the ML region is fixed from here on; "shell" takes whole waters near the peptide
    case $ml in
        peptide) sel='"ML" group Protein' ;;
        shell)   sel='"ML" group Protein or same residue as (resname SOL and within 0.5 of group Protein)' ;;
        *) echo "unknown ML region $ml" >&2; exit 2 ;;
    esac
    printf "q\n" | gmx make_ndx -f nvt.gro -o groups.ndx > index.log 2>&1
    gmx select -s nvt.tpr -f nvt.gro -select "$sel" -on ml.ndx >> index.log 2>&1
    sed -i "1s/.*/[ ML ]/" ml.ndx  # gmx select appends the frame to the name
    cat groups.ndx ml.ndx > index.ndx
    mark index
fi

for stage in npt nve; do
    if ! done_ $stage; then
        [[ $stage == npt ]] && prev=nvt || prev=npt
        [[ -f $stage.tpr ]] || gmx grompp -f "$T/$stage.mdp" -c $prev.gro -t $prev.cpt -p topol.top \
            -n index.ndx -o $stage.tpr > grompp_$stage.log 2>&1
        mdrun $stage && mark $stage
    fi
done

if ! done_ analysis; then
    "$PY" "$CAMPAIGN/analyze.py" "$job" "$GMX" > analysis.log 2>&1
    mark analysis
fi
