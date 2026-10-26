.. _metatomic:

Metatomic potentials
====================

The Metatomic interface evaluates an atomistic model through the C API of
metatomic-core and the applied-forces framework. The selected atoms are
provided to the model as a Metatomic ``System`` with positions, cell, periodic
boundary conditions, atom types, and pair lists built from the |Gromacs|
pairlist, in the model's own length unit.

By default, the Metatomic contribution is added on top of the unmodified
molecular-mechanics topology. This additive mode is useful as a control
calculation and for models that augment, rather than replace, force-field terms.

For embedded subsystem simulations where the Metatomic model replaces the
molecular-mechanics description of the selected atoms, :mdp:`metatomic-oniom`
enables the ONIOM-style subtractive scheme:

.. math::

   E = E_\mathrm{MM}(\mathrm{full}) + E_\mathrm{ML}(\mathrm{embedded})
       - E_\mathrm{MM}(\mathrm{embedded})

The subtractive term is applied through embedded-system topology preprocessing:
bonded and non-bonded molecular-mechanics terms represented by the model are
removed from the classical force-field contribution, while the model energy is
added through the Metatomic force provider. Boundary bonded terms involving
both Metatomic and MM atoms remain in the molecular-mechanics topology.

All embedded atoms are mutually excluded from the classical non-bonded
interactions. The non-bonded kernels treat excluded pairs within
:mdp:`rcoulomb` only, so the remaining classical Coulomb interaction between
embedded atoms is removed separately, for pairs at any distance: with Ewald
and PME electrostatics, the reciprocal-space interaction
:math:`q_i q_j \mathrm{erf}(\beta r)/r` of pairs beyond :mdp:`rcoulomb`;
with reaction-field and plain cut-off electrostatics, the reaction-field term
of excluded pairs within :mdp:`rcoulomb`. The embedded region may therefore
be larger than :mdp:`rcoulomb`. This correction is reported in the
Coulomb (SR) energy term; it evaluates all embedded pairs on every rank,
which is intended for embedded regions of up to a few thousand atoms.

Boundary link atoms
-------------------

The :mdp:`metatomic-link-atoms` option enables hydrogen caps at bonds that cross
the Metatomic/MM boundary and requires :mdp:`metatomic-oniom`. The cut bond, and
the angles and dihedrals through it, stay in the molecular-mechanics topology,
as do the interactions of the boundary MM atom. During model evaluation, the
boundary MM atom is replaced by a hydrogen placed on the boundary bond, 0.1 nm
from the embedded atom. The cap position is a function of the two atoms of the
bond, so the force on the cap is passed on to both of them with the chain rule,
and the virial is corrected for the cap's fixed distance.

This option requires a model trained for link-atom-capped boundary chemistry.
If the same boundary MM atom is bonded to multiple selected Metatomic atoms,
each cut bond is represented by a distinct hydrogen cap in the model input.

Model-requested charges
-----------------------

Metatomic models can declare additional inputs through their
``requested_inputs``. When a model requests a per-atom ``charge`` or
``charge/<variant>`` input, |Gromacs| provides the topology charges, converted
to the unit the model asks for; a link cap gets the charge of the MM atom it
replaces. Charge inputs are model-controlled; no :ref:`mdp` option enables or
disables them. A model requesting any other input is rejected when mdrun
starts.
