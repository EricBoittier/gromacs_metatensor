# ONIOM link-atom check through the metatomic C API

An end-to-end check of `metatomic-oniom` / `metatomic-link-atoms` on the C-API
force provider, without libtorch: an analytic metatomic model
(`pair_model.cpp`) is loaded as a C-API plugin, and `gmx mdrun -rerun` is
compared against an independent evaluation and finite differences.

The system (`chain.top`, `chain.ndx`) is a 7-atom chain in a 3 nm box, wrapped
across the x boundary. The ML group (atoms 2-4) has four hydrogen caps, two of
them on the same MM atom (bonded to two ML atoms). The model's pair weights
depend on the element and on the `charge` input, so the cap element and the
cap charge (its MM atom's) both change the energy.

`check_oniom.py` checks the Metatomic energy against a Python evaluation of the
model on the ML atoms and caps, the total forces against finite differences
of the potential energy, and the virial against `0.5 dE/d(strain)` on
rectangular and sheared boxes.

## Build and run

```bash
# metatomic-core (libmetatomic) and its metatensor, e.g. from a CMake install
PREFIX=/path/to/metatomic/install
METATENSOR=/path/to/metatensor   # prefix with include/metatensor.h and lib/libmetatensor.so
c++ -std=c++17 -O2 -fPIC -shared pair_model.cpp -o libpair_model.so \
    -I$PREFIX/include -I$METATENSOR/include -I/path/to/nlohmann-json/include \
    -L$PREFIX/lib -L$METATENSOR/lib -lmetatomic -lmetatensor \
    -Wl,--disable-new-dtags,-rpath,$PREFIX/lib,-rpath,$METATENSOR/lib

python check_oniom.py --gmx build/bin/gmx --plugin libpair_model.so              # serial
python check_oniom.py --gmx build/bin/gmx --plugin libpair_model.so --ranks 2    # 2 x 1 x 1 DD
python check_oniom.py --gmx build/bin/gmx --plugin libpair_model.so --no-links   # no ONIOM
```

## Results (single precision, one workstation)

| configuration | Metatomic energy vs reference | max force error | virial |
| --- | --- | --- | --- |
| 4 caps, serial | 449.69269 / 449.69265 | 0.085 kJ/mol/nm (forces up to 1453) | all 6 components within 0.12 |
| 4 caps, 2 ranks | 449.69269 / 449.69265 | 0.111 | within 0.15 |
| no ONIOM, serial | 98.06195 / 98.06194 | 0.90 (forces up to 10803) | within 0.8 |
| no ONIOM, 2 ranks | 98.06194 / 98.06194 | 0.87 | within 0.8 |

The residuals are the finite-difference noise of single-precision energies.
This check found two bugs, fixed on this branch: the lattice shift of pairs in
sheared boxes (`box * n` instead of `box^T * n`), and missing backward ghosts
with two domain-decomposition cells. Without the cap virial correction, the
virial is off by up to 39 kJ/mol here (Vir-XX -1.13 instead of 37.75).
