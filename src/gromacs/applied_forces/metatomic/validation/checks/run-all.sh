#!/usr/bin/env bash
# Run every C-API check, serial and with domain decomposition. Each script
# prints its own numbers and ends in PASS or FAIL; this one reports the tally.
#
#   usage: ./run-all.sh <gmx> <libpair_model.so> [<python>]
set -uo pipefail

GMX=$(realpath "${1:?usage: run-all.sh <gmx> <plugin.so> [python]}")
PLUGIN=$(realpath "${2:?usage: run-all.sh <gmx> <plugin.so> [python]}")
PY=${3:-python3}
cd "$(dirname "${BASH_SOURCE[0]}")"

run() {
    local name=$1; shift
    printf '%-44s ' "$name"
    if out=$("$PY" "$@" --gmx "$GMX" --plugin "$PLUGIN" 2>&1); then
        if [ "$(tail -1 <<<"$out")" = PASS ]; then echo PASS; return 0; fi
    fi
    echo FAIL; printf '%s\n' "$out" | tail -20 | sed 's/^/    /'; return 1
}

fails=0
run "oniom, 4 caps, serial"        check_oniom.py                     || ((fails++))
run "oniom, 4 caps, 2 ranks"       check_oniom.py --ranks 2           || ((fails++))
run "oniom off (additive)"         check_oniom.py --no-links          || ((fails++))
run "ML wider than rcoulomb, PME"  check_wide_ml.py --coulomb pme     || ((fails++))
run "ML wider than rcoulomb, RF"   check_wide_ml.py --coulomb rf      || ((fails++))
run "4 ML sites, 1-4 ranks"        check_sites.py                     || ((fails++))

echo
[ "$fails" -eq 0 ] && echo "all checks passed" || echo "$fails check(s) failed"
exit "$fails"
