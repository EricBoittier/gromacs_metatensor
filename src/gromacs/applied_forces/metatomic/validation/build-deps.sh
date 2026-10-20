#!/usr/bin/env bash
# Build and install the two C libraries the metatomic interface needs, at the
# revisions this branch is known to work with. See ../docs/dependencies.md for
# why these are pinned and not tracked.
#
#   usage: ./build-deps.sh <install-prefix> [<work-dir>]
set -euo pipefail

# -m so the paths need not exist yet
PREFIX=$(realpath -m "${1:?usage: build-deps.sh <install-prefix> [work-dir]}")
WORK=$(realpath -m "${2:-$PREFIX/src}")

METATENSOR_URL=https://github.com/metatensor/metatensor
METATENSOR_REV=a9fc36e7          # metatensor-core 0.2.4
METATOMIC_URL=https://github.com/metatensor/metatomic
METATOMIC_REV=a245aed8           # last revision before the requested_* API change
METATOMIC_BRANCH=metatomic-core  # unreleased; lives on a branch

mkdir -p "$WORK" "$PREFIX"

clone_at() {  # url rev dir [branch]
    local url=$1 rev=$2 dir=$3 branch=${4:-}
    if [ ! -d "$dir/.git" ]; then
        git clone --quiet ${branch:+--branch "$branch"} "$url" "$dir"
    fi
    git -C "$dir" fetch --quiet origin ${branch:+"$branch"}
    git -C "$dir" checkout --quiet "$rev"
}

echo "==> metatensor-core @ $METATENSOR_REV"
clone_at "$METATENSOR_URL" "$METATENSOR_REV" "$WORK/metatensor"
cmake -S "$WORK/metatensor/metatensor-core" -B "$WORK/metatensor/build" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" >/dev/null
cmake --build "$WORK/metatensor/build" --target install -j "$(nproc)" >/dev/null

echo "==> metatomic-core @ $METATOMIC_REV"
clone_at "$METATOMIC_URL" "$METATOMIC_REV" "$WORK/metatomic" "$METATOMIC_BRANCH"
cmake -S "$WORK/metatomic/metatomic-core" -B "$WORK/metatomic/build" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_PREFIX_PATH="$PREFIX" >/dev/null
cmake --build "$WORK/metatomic/build" --target install -j "$(nproc)" >/dev/null

echo
echo "installed into $PREFIX"
echo "configure GROMACS with: -DGMX_METATOMIC=ON -DCMAKE_PREFIX_PATH=$PREFIX"
