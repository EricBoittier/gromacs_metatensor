# Validating the metatomic interface

Everything here is runnable. The unit tests under `../tests/` cover the option
parsing and the small helpers; this directory covers the parts that only show
up in a real `mdrun`: energies, forces and the virial of an ONIOM-embedded
model, across periodic boundaries, sheared boxes and domain decomposition.

| | |
| --- | --- |
| `checks/` | The C-API checks. Minutes to run, no cluster, no trained model. **Start here.** |
| `model/` | `pair_model.cpp`, the analytic model the checks drive, plus its build script. |
| `campaign/` | The cyclic-peptide campaign: production-size runs over box sizes, ML regions and seeds. Needs a cluster and days. |
| `build-deps.sh` | Builds the two C libraries at the revisions this branch works with. |

The checks do not use a trained model. `model/pair_model.cpp` is an *analytic*
metatomic model: its energy is a closed form over pairs, so the same energy can
be worked out independently in Python and the two compared. That is what makes
these checks assertions rather than plots.

## Validate in four commands

```bash
V=src/gromacs/applied_forces/metatomic/validation

# 1. the C libraries, pinned (clones from GitHub on first run, ~5 min)
$V/build-deps.sh ~/opt/metatomic

# 2. GROMACS against them
cmake -S . -B build -DGMX_METATOMIC=ON -DCMAKE_PREFIX_PATH=~/opt/metatomic
cmake --build build -j$(nproc) --target gmx

# 3. the analytic model as a C-API plugin
$V/model/build-model.sh ~/opt/metatomic /tmp/libpair_model.so

# 4. every check, serial and with domain decomposition
$V/checks/run-all.sh build/bin/gmx /tmp/libpair_model.so
```

Step 4 should print:

```
oniom, 4 caps, serial                        PASS
oniom, 4 caps, 2 ranks                       PASS
oniom off (additive)                         PASS
ML wider than rcoulomb, PME                  PASS
ML wider than rcoulomb, RF                   PASS
4 ML sites, 1-4 ranks                        PASS

all checks passed
```

`run-all.sh` takes an optional third argument, the Python to use; it needs
NumPy and nothing else. Each script can also be run on its own with `--gmx`
and `--plugin`, which prints the numbers rather than just the verdict — see
`checks/README.md` for the per-script options, including `--ranks`,
`--no-links` and `--coulomb`.

## What each check establishes

| check | asserts | against |
| --- | --- | --- |
| `check_oniom.py` | ONIOM energy, total forces, all six virial components, with four hydrogen caps on a chain wrapped across a periodic boundary and two caps on one MM atom | an independent Python evaluation, finite differences, and `0.5 dE/d(strain)` on rectangular and sheared boxes |
| `check_wide_ml.py` | that an ML region wider than `rcoulomb` is still right, with PME and with reaction field | a run with `rcoulomb` large enough to contain every ML pair |
| `check_sites.py` | that four ML sites are evaluated as separate systems with their own charge and spin, cross-site pairs excluded, under 1 to 4 ranks and a 2x2x1 grid | a site-by-site Python evaluation, finite differences, and the site-to-rank assignment `mdrun` logs |

A failure is informative because each check has a known amount to lose. Drop
the cap virial correction and `check_oniom.py`'s Vir-XX moves from 37.8 to
-1.1. Drop the embedded Coulomb correction and `check_wide_ml.py`'s PME energy
is off by 51 kJ/mol and its forces by 8000 kJ/mol/nm. Those are the numbers the
checks exist to hold down.

Residuals are single-precision finite-difference noise. Last run on one
workstation, GROMACS at `GMX_SIMD=SSE2`, metatensor-core 0.2.4 and
metatomic-core `a245aed8`:

| | energy | max force error | max virial error |
| --- | --- | --- | --- |
| `check_oniom.py` | 449.69269 vs 449.69265 kJ/mol | 0.101 kJ/mol/nm, of forces up to 1453 | 0.11 kJ/mol |
| `check_sites.py` | 2084.39111 kJ/mol | 0.481 kJ/mol/nm, of forces up to 3410 | 0.275 kJ/mol |

## Why the dependencies are pinned

`build-deps.sh` pins both revisions, and that is not caution. metatomic-core is
unreleased, and the C API has already moved under this module: the commit after
the pinned one adds a parameter to `requested_pair_lists()` and
`requested_inputs()`, which does not compile here.
`../docs/dependencies.md` has the detail, the error, and what adapting would
take.

If `build-deps.sh` is not what you want — you have the libraries already, or
from wheels — any metatensor and metatomic will do as long as they agree with
each other and with the module. The configure line records what was found:

```
-- Found metatomic 0.2.0-dev1217+git.a245aed (metatensor 0.2.4): Metatomic potential support enabled.
```

Keep that line from a build that worked. While the dependency is a branch it is
the only thing that says which revision produced a binary.

## Known rough edges

- **The campaign is not a check.** `campaign/` reports numbers (NVE drift,
  density, ms/step) and has no pass criterion. Read `campaign/README.md` and
  `campaign/summary.md` for what it covers and what it found.
- **`GMX_SIMD` affects the unit tests, not these.** A low-SIMD build skips two
  `PlainPairlistTest` variants that need the 2xNN kernels. The checks here run
  through `mdrun` and are unaffected.
- **`pair_model.cpp` tracks the C API too.** It is a metatomic model, so it
  implements the same interface the library defines, and an upstream change
  breaks it the same way it breaks the force provider. If it stops compiling,
  compare its overrides against `metatomic/model.hpp` in the prefix.
