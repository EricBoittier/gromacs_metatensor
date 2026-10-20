#!/usr/bin/env bash
# Compile pair_model.cpp into a metatomic C-API plugin for the checks in
# ../checks. It is an analytic model, not a trained one: its energy is a closed
# form over pairs, so an independent evaluation can be compared against it.
#
#   usage: ./build-model.sh <prefix>[:<prefix>...] [<output.so>]
#
# One prefix if build-deps.sh installed both libraries together, or several
# colon-separated if metatomic and metatensor live apart (pip wheels do).
set -euo pipefail

PREFIXES=${1:?usage: build-model.sh <prefix>[:<prefix>...] [output.so]}
OUT=${2:-$(pwd)/libpair_model.so}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# metatomic-core installs its vendored nlohmann/json under third-party, and
# libraries land in lib or lib64 depending on the platform.
FLAGS=()
IFS=: read -ra dirs <<< "$PREFIXES"
for p in "${dirs[@]}"; do
    p=$(realpath "$p")
    FLAGS+=("-I$p/include" "-I$p/include/metatomic/third-party")
    for d in "$p/lib64" "$p/lib"; do
        [ -d "$d" ] && FLAGS+=("-L$d" "-Wl,-rpath,$d")
    done
done

c++ -std=c++17 -O2 -fPIC -shared "$HERE/pair_model.cpp" -o "$OUT" \
    "${FLAGS[@]}" -Wl,--disable-new-dtags -lmetatomic -lmetatensor

echo "built $OUT"
