# Build dependencies of the metatomic interface

The force provider talks to metatomic through its **C API**, so this module
does not link libtorch. `GMX_TORCH` stays with NNPot; `GMX_METATOMIC` resolves
`metatomic` and `metatensor` on its own (`cmake/gmxManageMetatomic.cmake`).

| package | requirement | state |
| --- | --- | --- |
| metatomic-core (`libmetatomic`) | any (see below) | **unreleased** — branch `metatomic-core` of `metatensor/metatomic`, PR [#232](https://github.com/metatensor/metatomic/pull/232) (draft). `Cargo.toml` has `publish = false`. |
| metatensor-core (`libmetatensor`) | >= 0.2.4 | released 2026-07-30. Also metatomic-core's own `REQUIRED_METATENSOR_VERSION`. |

Both must be on `CMAKE_PREFIX_PATH`, as separate prefixes when they come from
wheels. `metatomic-config.cmake` calls `find_package(metatensor REQUIRED)`, so
GROMACS resolves metatensor first to keep a missing dependency a soft failure
under `GMX_METATOMIC=AUTO`.

## Why there is no version floor on metatomic

Asking `find_package` for a metatomic version is a trap here, and the trap is
worth stating because it looks like an obvious thing to add.

metatomic-core's config file is written with `COMPATIBILITY SameMinorVersion`,
so a request is satisfied only by the *same* `0.x`. The version it compares
against is not the one in `metatomic-core/Cargo.toml` (`0.1.0`): the build runs
that through `create_development_version()`, which bumps the minor for an
untagged tree. A source build of the 0.1.0 branch therefore installs a package
reporting `0.2.0-dev<N>`. Asking for `0.1.0` rejects every such build; asking
for `0.2.0` would reject `0.1.0` once it is released. Neither is right while
the dependency is a moving branch, so the build asks for no version and prints
what it found.

## The metatensor floor has to track metatomic-core's, and does not

`metatomic-config.cmake` runs its own `find_package(metatensor
${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)`, and that number is not the
same everywhere:

| where | required metatensor |
| --- | --- |
| this file (`gmxManageMetatomic.cmake`) | `0.2.4` |
| metatomic-core, branch tip upstream | `0.2.4` |
| metatomic-core, as built and installed in the metawork venv | `0.3` |

When GROMACS's floor is *looser* than metatomic-config's, metatensor passes
here, `find_package(metatomic)` runs, and metatomic-config aborts on its own
`REQUIRED` — a hard configure error, including under `GMX_METATOMIC=AUTO`,
where the comment above intends a soft failure. Resolving metatensor first only
guards against metatensor being *missing*, not against it being the wrong
version for metatomic-core.

Nothing is changed here for it, because the right floor depends on which
metatomic-core you install and upstream has not settled. If you hit
`Could not find a configuration file for package "metatensor" ... version:
0.3` from inside metatomic-config, that is this, and the fix is to match the
metatensor your metatomic-core was built against rather than to touch GROMACS.

## API surface in use

An API break upstream shows up in these names only:

- `<metatomic.hpp>` — `ExternalModel`, `load_model`, `load_plugin`,
  `execute_model`, `ModelCapabilities`, `PairListOptions`, `System`,
  `SampleKind`, `Quantity`, `Gradients`, `DLPackTensor`,
  `unit_conversion_factor`, `Error`
- `<metatensor.hpp>` — `Labels`, `TensorMap`, `TensorBlock`, `SimpleDataArray`,
  `DataArrayBase`
- `<metatensor/dlpack/dlpack.h>` — the DLPack structs the model inputs are
  handed over as

`tests/c_api_contract.cpp` covers the ones with non-obvious ownership or
lifetime rules; `docs/oniom_check/` exercises the whole path against an
analytic model loaded as a C-API plugin.

## Pinning

Upstream moves, so record the revision you built against rather than tracking
the branch tip. At the time of writing, `metatensor/metatomic` branch
`metatomic-core` is at `e9f59e7`. Related upstream work:
[#359](https://github.com/metatensor/metatomic/pull/359) rebuilds
metatomic-torch on top of metatomic-core, so the C core is the direction of
travel rather than a side branch.

`docs/oniom_check/README.md` has the compiler invocation for building a C-API
model plugin against these two prefixes.
