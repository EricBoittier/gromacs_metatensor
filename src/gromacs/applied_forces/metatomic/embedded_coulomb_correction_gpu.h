/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must be properly licensed. Other licenses are available.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * CUDA version of computeEmbeddedCoulombCorrection(), for large ML regions.
 *
 * The CPU pair loop is O(N^2) over the embedded atoms: with the whole solute as ML (thousands
 * of atoms) it costs more than the model. Here each GPU thread takes one atom and loops over
 * all others in single precision, like the non-bonded kernels, with the minimum image of
 * pbc_dx_aiuc(); the host adds the per-atom sums in double. Used in CUDA builds unless
 * GMX_METATOMIC_ECC_CPU is set.
 *
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_METATOMIC_EMBEDDED_COULOMB_CORRECTION_GPU_H
#define GMX_APPLIED_FORCES_METATOMIC_EMBEDDED_COULOMB_CORRECTION_GPU_H

#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

struct interaction_const_t;
struct t_pbc;

namespace gmx
{

struct EmbeddedCoulombCorrection;

//! Device buffers and stream for the embedded Coulomb correction of a fixed set of atoms.
class EmbeddedCoulombCorrectionGpu
{
public:
    //! Takes the per-atom data that stays fixed; allocates on the current device.
    EmbeddedCoulombCorrectionGpu(ArrayRef<const real> charges,
                                 ArrayRef<const int>  energyGroups,
                                 int                  numEnergyGroups,
                                 ArrayRef<const int>  sites);
    ~EmbeddedCoulombCorrectionGpu();

    /*! \brief Computes the correction for positions \p x into \p result.
     *
     * \returns false, leaving \p result untouched, for the cases left to the CPU:
     * electrostatics other than Ewald-family, reaction-field and cut-off, and PBC other
     * than full 3D (rectangular or triclinic) or none.
     */
    bool compute(ArrayRef<const RVec>       x,
                 const t_pbc&               pbc,
                 const interaction_const_t& ic,
                 EmbeddedCoulombCorrection* result);

private:
    int                 n_;
    int                 numEnergyGroups_;
    std::vector<int>    groups_;
    std::vector<float>  hostX_, hostForces_, hostVirial_, hostEnergies_;
    std::vector<int>    hostPairs_;
    void*               stream_ = nullptr;
    float *dX_ = nullptr, *dQ_ = nullptr, *dForces_ = nullptr, *dVirial_ = nullptr, *dEnergies_ = nullptr;
    int *dSites_ = nullptr, *dGroups_ = nullptr, *dPairs_ = nullptr;
};

} // namespace gmx

#endif
