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
 * Implements EmbeddedCoulombCorrectionGpu, see embedded_coulomb_correction_gpu.h.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "embedded_coulomb_correction_gpu.h"

#include <cuda_runtime.h>

#include <cmath>

#include <algorithm>

#include "gromacs/utility/vec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/stringutil.h"

#include "embedded_coulomb_correction.h"

namespace gmx
{

namespace
{

void check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
    {
        GMX_THROW(InternalError(formatString(
                "Embedded Coulomb correction on the GPU: %s failed: %s", what, cudaGetErrorString(status))));
    }
}

// t_pbc::pbcTypeDX values, from the enum private to pbc.cpp
constexpr int c_pbcDxRectangular = 1;
constexpr int c_pbcDxTriclinic   = 2;
constexpr int c_pbcDxNone        = 9;

//! The parts of t_pbc that pbc_dx_aiuc() uses for full 3D PBC, in single precision like pbc_dx_aiuc()
struct DevicePbc
{
    int    type;
    float box[DIM][DIM];
    float fbox[DIM], hbox[DIM], mhbox[DIM];
    float maxCutoff2;
    int    numTricVec;
    float tricVec[MAX_NTRICVEC][DIM];
};

//! Minimum image of dx, as pbc_dx_aiuc() for epbcdxRECTANGULAR and epbcdxTRICLINIC
__device__ void minimumImage(const DevicePbc& pbc, float dx[DIM])
{
    if (pbc.type == c_pbcDxRectangular)
    {
        for (int d = 0; d < DIM; d++)
        {
            if (dx[d] > pbc.hbox[d])
            {
                dx[d] -= pbc.fbox[d];
            }
            else if (dx[d] <= pbc.mhbox[d])
            {
                dx[d] += pbc.fbox[d];
            }
        }
    }
    else if (pbc.type == c_pbcDxTriclinic)
    {
        for (int d = DIM - 1; d >= 1; d--)
        {
            if (dx[d] > pbc.hbox[d])
            {
                for (int e = d; e >= 0; e--)
                {
                    dx[e] -= pbc.box[d][e];
                }
            }
            else if (dx[d] <= pbc.mhbox[d])
            {
                for (int e = d; e >= 0; e--)
                {
                    dx[e] += pbc.box[d][e];
                }
            }
        }
        // two shifts in x at most
        for (int s = 0; s < 2; s++)
        {
            if (dx[XX] > pbc.hbox[XX])
            {
                dx[XX] -= pbc.fbox[XX];
            }
            else if (dx[XX] <= pbc.mhbox[XX])
            {
                dx[XX] += pbc.fbox[XX];
            }
        }
        float d2min = dx[XX] * dx[XX] + dx[YY] * dx[YY] + dx[ZZ] * dx[ZZ];
        if (d2min > pbc.maxCutoff2)
        {
            const float start[DIM] = { dx[XX], dx[YY], dx[ZZ] };
            for (int k = 0; k < pbc.numTricVec && d2min > pbc.maxCutoff2; k++)
            {
                float       trial[DIM];
                float       d2 = 0.0F;
                for (int d = 0; d < DIM; d++)
                {
                    trial[d] = start[d] + pbc.tricVec[k][d];
                    d2 += trial[d] * trial[d];
                }
                if (d2 < d2min)
                {
                    d2min = d2;
                    for (int d = 0; d < DIM; d++)
                    {
                        dx[d] = trial[d];
                    }
                }
            }
        }
    }
}

constexpr int c_tile = 128;

/*! \brief One thread per embedded atom i, looping over every j != i.
 *
 * Each unordered pair is visited twice, from i and from j, so the energy and virial of an
 * ordered pair get half weight; the force on i needs no atomics.
 */
__global__ void correctionKernel(int           n,
                                 const float* x,
                                 const float* q,
                                 const int*    sites,
                                 const int*    groups,
                                 int           numGroups,
                                 DevicePbc     pbc,
                                 bool          ewald,
                                 float        epsfac,
                                 float        rc2,
                                 float        beta,
                                 float        twoBetaOverSqrtPi,
                                 float        krf,
                                 float        crf,
                                 float*       forces,
                                 float*       virial,
                                 float*       energies,
                                 int*          pairs)
{
    __shared__ float sx[c_tile][DIM];
    __shared__ float sq[c_tile];
    __shared__ int    ssite[c_tile], sgroup[c_tile];

    const int    i      = blockIdx.x * blockDim.x + threadIdx.x;
    const bool   active = i < n;
    const float xi[DIM] = { active ? x[3 * i] : 0.0F, active ? x[3 * i + 1] : 0.0F, active ? x[3 * i + 2] : 0.0F };
    const float qi      = active ? q[i] : 0;
    const int    si      = active ? sites[i] : 0;

    float f[DIM] = { 0.0F, 0.0F, 0.0F }, vir[DIM][DIM] = { { 0.0F } }, energy = 0.0F;
    int    numPairs = 0;
    for (int start = 0; start < n; start += c_tile)
    {
        const int j = start + threadIdx.x;
        if (threadIdx.x < c_tile && j < n)
        {
            for (int d = 0; d < DIM; d++)
            {
                sx[threadIdx.x][d] = x[3 * j + d];
            }
            sq[threadIdx.x]     = q[j];
            ssite[threadIdx.x]  = sites[j];
            sgroup[threadIdx.x] = groups[j];
        }
        __syncthreads();
        const int end = min(c_tile, n - start);
        for (int k = 0; active && k < end; k++)
        {
            const float qq = epsfac * qi * sq[k];
            if (start + k == i || ssite[k] != si || qq == 0.0F)
            {
                continue;
            }
            float dx[DIM] = { xi[XX] - sx[k][XX], xi[YY] - sx[k][YY], xi[ZZ] - sx[k][ZZ] };
            minimumImage(pbc, dx);
            const float r2 = dx[XX] * dx[XX] + dx[YY] * dx[YY] + dx[ZZ] * dx[ZZ];
            // Ewald: the kernels remove the pair within the cut-off, we remove it beyond.
            // Reaction-field: the kernels add k_rf r^2 - c_rf within the cut-off, we take it out.
            if (ewald == (r2 < rc2))
            {
                continue;
            }
            float v, fscal;
            if (ewald)
            {
                const float r      = sqrtf(r2);
                const float erfBr  = erff(beta * r);
                v                   = -qq * erfBr / r;
                fscal = -qq * (erfBr / r - twoBetaOverSqrtPi * expf(-beta * beta * r2)) / r2;
            }
            else
            {
                v     = -qq * (krf * r2 - crf);
                fscal = 2 * qq * krf;
            }
            if (numGroups == 1)
            {
                energy += 0.5F * v;
            }
            else
            {
                energies[i * numGroups + sgroup[k]] += 0.5F * v;
            }
            for (int d = 0; d < DIM; d++)
            {
                f[d] += fscal * dx[d];
                for (int e = 0; e < DIM; e++)
                {
                    vir[d][e] -= 0.25F * dx[d] * fscal * dx[e];
                }
            }
            numPairs++;
        }
        __syncthreads();
    }
    if (active)
    {
        for (int d = 0; d < DIM; d++)
        {
            forces[3 * i + d] = f[d];
            for (int e = 0; e < DIM; e++)
            {
                virial[9 * i + 3 * d + e] = vir[d][e];
            }
        }
        if (numGroups == 1)
        {
            energies[i] = energy;
        }
        pairs[i] = numPairs;
    }
}

template<typename T>
T* deviceCopy(ArrayRef<const T> host, size_t n, cudaStream_t stream)
{
    T* device = nullptr;
    check(cudaMalloc(&device, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    if (host.empty())
    {
        check(cudaMemsetAsync(device, 0, n * sizeof(T), stream), "cudaMemsetAsync");
    }
    else
    {
        check(cudaMemcpyAsync(device, host.data(), n * sizeof(T), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync");
    }
    return device;
}

} // namespace

EmbeddedCoulombCorrectionGpu::EmbeddedCoulombCorrectionGpu(ArrayRef<const real> charges,
                                                           ArrayRef<const int>  energyGroups,
                                                           int                  numEnergyGroups,
                                                           ArrayRef<const int>  sites) :
    n_(charges.ssize()),
    numEnergyGroups_(numEnergyGroups),
    groups_(energyGroups.begin(), energyGroups.end()),
    hostX_(3 * n_),
    hostForces_(3 * n_),
    hostVirial_(9 * n_),
    hostEnergies_(n_ * numEnergyGroups_),
    hostPairs_(n_)
{
    if (groups_.empty())
    {
        groups_.assign(n_, 0);
    }
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    stream_ = stream;
    dQ_      = deviceCopy<float>(charges, n_, stream);
    dSites_  = deviceCopy<int>(sites, n_, stream);
    dGroups_ = deviceCopy<int>(groups_, n_, stream);
    for (float** buffer : { &dX_, &dForces_, &dVirial_, &dEnergies_ })
    {
        const size_t size = buffer == &dX_ || buffer == &dForces_ ? 3 * n_
                            : buffer == &dVirial_                 ? 9 * n_
                                                                  : n_ * numEnergyGroups_;
        check(cudaMalloc(buffer, std::max<size_t>(size, 1) * sizeof(float)), "cudaMalloc");
    }
    check(cudaMalloc(&dPairs_, std::max(n_, 1) * sizeof(int)), "cudaMalloc");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

EmbeddedCoulombCorrectionGpu::~EmbeddedCoulombCorrectionGpu()
{
    for (void* buffer : { static_cast<void*>(dX_), static_cast<void*>(dQ_),
                          static_cast<void*>(dForces_), static_cast<void*>(dVirial_),
                          static_cast<void*>(dEnergies_), static_cast<void*>(dSites_),
                          static_cast<void*>(dGroups_), static_cast<void*>(dPairs_) })
    {
        cudaFree(buffer);
    }
    if (stream_)
    {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    }
}

bool EmbeddedCoulombCorrectionGpu::compute(ArrayRef<const RVec>       x,
                                           const t_pbc&               pbc,
                                           const interaction_const_t& ic,
                                           EmbeddedCoulombCorrection* result)
{
    const auto& coulomb = ic.coulomb;
    const bool  ewald   = usingPmeOrEwald(coulomb.type);
    if ((!ewald && !usingRF(coulomb.type) && coulomb.type != CoulombInteractionType::Cut)
        || (pbc.pbcTypeDX != c_pbcDxRectangular && pbc.pbcTypeDX != c_pbcDxTriclinic
            && pbc.pbcTypeDX != c_pbcDxNone))
    {
        return false;
    }
    GMX_RELEASE_ASSERT(x.ssize() == n_, "Need the positions of every embedded atom");

    DevicePbc devicePbc;
    devicePbc.type       = pbc.pbcTypeDX;
    devicePbc.maxCutoff2 = pbc.max_cutoff2;
    devicePbc.numTricVec = pbc.ntric_vec;
    for (int d = 0; d < DIM; d++)
    {
        devicePbc.fbox[d]  = pbc.fbox_diag[d];
        devicePbc.hbox[d]  = pbc.hbox_diag[d];
        devicePbc.mhbox[d] = pbc.mhbox_diag[d];
        for (int e = 0; e < DIM; e++)
        {
            devicePbc.box[d][e] = pbc.box[d][e];
        }
        for (int k = 0; k < MAX_NTRICVEC; k++)
        {
            devicePbc.tricVec[k][d] = pbc.tric_vec[k][d];
        }
    }
    for (int i = 0; i < n_; i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            hostX_[3 * i + d] = x[i][d];
        }
    }

    auto stream = static_cast<cudaStream_t>(stream_);
    check(cudaMemcpyAsync(dX_, hostX_.data(), hostX_.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync");
    if (numEnergyGroups_ > 1)
    {
        check(cudaMemsetAsync(dEnergies_, 0, hostEnergies_.size() * sizeof(float), stream),
              "cudaMemsetAsync");
    }
    const int threads = c_tile;
    const int blocks  = (n_ + threads - 1) / threads;
    correctionKernel<<<blocks, threads, 0, stream>>>(n_,
                                                     dX_,
                                                     dQ_,
                                                     dSites_,
                                                     dGroups_,
                                                     numEnergyGroups_,
                                                     devicePbc,
                                                     ewald,
                                                     coulomb.epsfac,
                                                     coulomb.cutoff * coulomb.cutoff,
                                                     coulomb.ewaldCoeff,
                                                     float(2 * coulomb.ewaldCoeff / std::sqrt(M_PI)),
                                                     coulomb.reactionFieldCoefficient,
                                                     coulomb.reactionFieldShift,
                                                     dForces_,
                                                     dVirial_,
                                                     dEnergies_,
                                                     dPairs_);
    check(cudaGetLastError(), "the kernel launch");
    const auto fetch = [&](auto& host, const auto* device) {
        check(cudaMemcpyAsync(host.data(), device, host.size() * sizeof(host[0]), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync");
    };
    fetch(hostForces_, dForces_);
    fetch(hostVirial_, dVirial_);
    fetch(hostEnergies_, dEnergies_);
    fetch(hostPairs_, dPairs_);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    result->energy = 0;
    result->numPairs = 0;
    result->forces.assign(n_, { 0, 0, 0 });
    result->groupPairEnergies.assign(numEnergyGroups_ * numEnergyGroups_, 0.0);
    clear_mat(result->virial);
    for (int i = 0; i < n_; i++)
    {
        for (int g = 0; g < numEnergyGroups_; g++)
        {
            const double e = hostEnergies_[i * numEnergyGroups_ + g];
            result->energy += e;
            result->groupPairEnergies[GID(groups_[i], g, numEnergyGroups_)] += e;
        }
        for (int d = 0; d < DIM; d++)
        {
            result->forces[i][d] = hostForces_[3 * i + d];
            for (int e = 0; e < DIM; e++)
            {
                result->virial[d][e] += hostVirial_[9 * i + 3 * d + e];
            }
        }
        result->numPairs += hostPairs_[i];
    }
    result->numPairs /= 2;
    return true;
}

} // namespace gmx
