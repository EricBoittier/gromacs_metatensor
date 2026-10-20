#!/usr/bin/env bash
# Called by runner.sh when the queue is finished: copy the campaign inputs,
# scripts and summary (no trajectories) into the GROMACS fork's oniom-capi
# branch and push it to the fork. Skips, with a log line, unless the worktree
# is on oniom-capi and in sync with fork/oniom-capi, so it never pushes
# anything unexpected. Then commits summary.md/csv to metawork.
set -euo pipefail
CAMPAIGN=$(cd "$(dirname "$0")" && pwd)
REPO=/home/boittier/metawork/gromacs-oniom-capi
DEST=src/gromacs/applied_forces/metatomic/docs/oniom_peptide
log() { echo "$(date -Is) finish: $*"; }

push_gromacs() {
    cd "$REPO"
    [[ $(git rev-parse --abbrev-ref HEAD) == oniom-capi ]] || { log "skip: $REPO is not on oniom-capi"; return; }
    git fetch -q fork oniom-capi
    [[ $(git rev-parse HEAD) == $(git rev-parse fork/oniom-capi) ]] || { log "skip: oniom-capi differs from fork/oniom-capi"; return; }
    [[ -z $(git status --porcelain -- "$DEST") ]] || { log "skip: $DEST has local changes"; return; }

    mkdir -p "$DEST/templates" "$DEST/base"
    cp "$CAMPAIGN"/{README.md,jobs.txt,run_job.sh,runner.sh,install_cron.sh,finish.sh,analyze.py,summarize.py,summary.md,summary.csv} "$DEST/"
    cp "$CAMPAIGN"/templates/*.mdp "$DEST/templates/"
    cp "$CAMPAIGN"/base/* "$CAMPAIGN"/../build_peptide.py "$DEST/base/"
    git add -- "$DEST"
    done=$(grep -c "| done |" "$DEST/summary.md" || true)
    git commit -q -m "Add the ONIOM cyclic-peptide campaign: box sizes, ML water shell, repeats

    Segetalin A in TIP3P with PET-MAD xs through metatomic-oniom: cubic boxes
    of 4, 5 and 6 nm, the peptide or the peptide plus a 0.5 nm water shell as
    the ML region, three seeds each, 20 ps NpT then 20 ps NVE at rcoulomb 1.0
    nm. The scripts run unattended under cron; summary.md has the results of
    the $done finished jobs.

    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$DEST"
    git push -q fork oniom-capi
    touch "$CAMPAIGN/.pushed"
    log "pushed $(git rev-parse --short HEAD) to fork/oniom-capi"
}

# The campaign summary in metawork, on its current branch, when that branch
# is in sync with origin (so nothing but this commit is pushed).
push_metawork() {
    local repo=/home/boittier/metawork
    cd "$repo"
    local branch; branch=$(git rev-parse --abbrev-ref HEAD)
    git fetch -q origin "$branch"
    [[ $(git rev-parse HEAD) == $(git rev-parse "origin/$branch") ]] || { log "skip metawork: $branch differs from origin"; return; }
    git add -- gromacs-oniom-cyclic/campaign/summary.md gromacs-oniom-cyclic/campaign/summary.csv
    git diff --cached --quiet -- gromacs-oniom-cyclic/campaign/summary.md gromacs-oniom-cyclic/campaign/summary.csv && { log "skip metawork: nothing to commit"; return; }
    git commit -q -m "gromacs-oniom-cyclic: ONIOM campaign results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- gromacs-oniom-cyclic/campaign/summary.md gromacs-oniom-cyclic/campaign/summary.csv
    git push -q origin "$branch"
    touch "$CAMPAIGN/.pushed-metawork"
    log "pushed $(git rev-parse --short HEAD) to metawork origin/$branch"
}

[[ -f $CAMPAIGN/.pushed ]] || push_gromacs
[[ -f $CAMPAIGN/.pushed-metawork ]] || push_metawork
