// A small analytic metatomic model, loaded through the metatomic C API, for
// checking the GROMACS ONIOM link-atom path against finite differences.
//
// E = sum over half-list pairs within rc of
//       w(Zi, Zj) (rc^2 - r^2)^2 / rc^4  +  lambda q_i q_j (rc^2 - r^2)^2 / rc^4
// with w = A (1 + 0.1 (Zi + Zj)), so the cap element and the cap charge both
// change the energy. The pair term and its derivative vanish at rc, so
// finite differences are clean. Returns the total energy with positions and
// strain gradients. With per_atom = 1, energies are per atom (half of each
// pair to each end) and only the selected atoms are returned, which is what
// GROMACS needs under domain decomposition.
//
// Parameters are read from the model file: "rc A lambda per_atom" (nm, kJ/mol, kJ/mol/e^2, 0/1).
// lambda != 0 makes the model request the per-atom `charge` input.

#include <array>
#include <map>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <metatensor.hpp>
#include <metatomic.hpp>

namespace {

class PairModel final : public metatomic::BaseModel
{
public:
    PairModel(double rc, double a, double lambda, bool perAtom) :
        rc_(rc), a_(a), lambda_(lambda), perAtom_(perAtom)
    {
    }

    metatomic::ModelCapabilities capabilities() const override
    {
        std::vector<int64_t> types;
        for (int64_t z = 1; z <= 118; ++z)
        {
            types.push_back(z);
        }
        return metatomic::ModelCapabilities::builder()
                .add_output(metatomic::Quantity::builder()
                                    .name("energy")
                                    .unit("kJ/mol")
                                    .sample_kind(perAtom_ ? metatomic::SampleKind::Atom
                                                          : metatomic::SampleKind::System)
                                    .build())
                .atomic_types(types)
                .interaction_range(rc_)
                .length_unit("nm")
                .supported_devices({ metatomic::ModelCapabilities::Device::CPU })
                .dtype(metatomic::ModelCapabilities::DType::Float64)
                .build();
    }

    metatomic::ModelMetadata metadata() const override
    {
        return metatomic::ModelMetadata::builder().name("oniom-test pair model").build();
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const override
    {
        return { pairs() };
    }

    std::vector<metatomic::Quantity> requested_inputs() const override
    {
        if (lambda_ == 0.0)
        {
            return {};
        }
        return { metatomic::Quantity::builder()
                         .name("charge")
                         .unit("e")
                         .sample_kind(metatomic::SampleKind::Atom)
                         .build() };
    }

    std::vector<metatensor::TensorMap> execute_inner(const std::vector<metatomic::System>& systems,
                                                     const metatensor::Labels* selected,
                                                     const std::vector<metatomic::Quantity>& /*outputs*/) override
    {
        std::vector<metatensor::TensorMap> result;
        for (size_t s = 0; s < systems.size(); ++s)
        {
            const auto& system = systems[s];
            const auto  n      = system.size();
            const auto  types  = system.types();
            const auto* z      = static_cast<const int32_t*>(types->dl_tensor.data);

            std::vector<double> charges(n, 0.0);
            if (lambda_ != 0.0)
            {
                auto chargeMap = system.custom_data("charge");
                auto block     = chargeMap.block_by_id(0);
                auto values    = block.template values<double>();
                for (size_t i = 0; i < n; ++i)
                {
                    charges[i] = values(i, 0);
                }
            }

            auto       block   = system.pairs(pairs());
            const auto samples = block.samples().values_cpu();
            auto       vectors = block.values<double>();

            // output rows: one per system, or one per selected atom
            std::vector<int32_t> rowAtoms;
            std::vector<int32_t> rowOfAtom(n, -1);
            if (perAtom_)
            {
                if (selected != nullptr)
                {
                    const auto values = selected->values_cpu();
                    for (size_t k = 0; k < selected->count(); ++k)
                    {
                        if (values(k, 0) == static_cast<int32_t>(s))
                        {
                            rowOfAtom[values(k, 1)] = static_cast<int32_t>(rowAtoms.size());
                            rowAtoms.push_back(values(k, 1));
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < n; ++i)
                    {
                        rowOfAtom[i] = static_cast<int32_t>(i);
                        rowAtoms.push_back(static_cast<int32_t>(i));
                    }
                }
            }
            const size_t nRows = perAtom_ ? rowAtoms.size() : 1;
            std::vector<double> energy(nRows, 0.0);
            std::vector<double> strain(nRows * 9, 0.0);
            std::map<std::pair<int32_t, int32_t>, std::array<double, 3>> grad; // (row, atom)

            const double rc2 = rc_ * rc_;
            for (size_t k = 0; k < block.samples().count(); ++k)
            {
                const int32_t i = samples(k, 0);
                const int32_t j = samples(k, 1);
                const double  d[3] = { vectors(k, 0, 0), vectors(k, 1, 0), vectors(k, 2, 0) };
                const double  r2   = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if (r2 >= rc2)
                {
                    continue;
                }
                const double w = a_ * (1.0 + 0.1 * (z[i] + z[j])) + lambda_ * charges[i] * charges[j];
                const double f = rc2 - r2;
                std::array<double, 3> g; // dE/dd = -4 w f d / rc^4, with d = r_j - r_i
                for (int c = 0; c < 3; ++c)
                {
                    g[c] = -4.0 * w * f * d[c] / (rc2 * rc2);
                }
                const auto add = [&](int32_t row, double share)
                {
                    energy[row] += share * w * f * f / (rc2 * rc2);
                    auto& gj = grad[{ row, j }];
                    auto& gi = grad[{ row, i }];
                    for (int c = 0; c < 3; ++c)
                    {
                        gj[c] += share * g[c];
                        gi[c] -= share * g[c];
                        for (int b = 0; b < 3; ++b)
                        {
                            strain[9 * row + 3 * b + c] += share * d[b] * g[c]; // dE/d(eps)_bc
                        }
                    }
                };
                if (!perAtom_)
                {
                    add(0, 1.0);
                }
                else
                {
                    for (const int32_t end : { i, j })
                    {
                        if (rowOfAtom[end] >= 0)
                        {
                            add(rowOfAtom[end], 0.5);
                        }
                    }
                }
            }

            const int32_t zero = 0;
            std::vector<int32_t> energySamples;
            for (size_t r = 0; r < nRows; ++r)
            {
                if (perAtom_)
                {
                    energySamples.insert(energySamples.end(), { static_cast<int32_t>(s), rowAtoms[r] });
                }
                else
                {
                    energySamples.push_back(static_cast<int32_t>(s));
                }
            }
            std::vector<metatensor::TensorBlock> blocks;
            blocks.emplace_back(std::make_unique<metatensor::SimpleDataArray<double>>(
                                        std::vector<uintptr_t>{ nRows, 1 }, energy),
                                perAtom_ ? metatensor::Labels({ "system", "atom" }, energySamples.data(), nRows)
                                         : metatensor::Labels({ "system" }, energySamples.data(), nRows),
                                std::vector<metatensor::Labels>{},
                                metatensor::Labels({ "energy" }, &zero, 1));

            std::vector<int32_t> gradSamples;
            std::vector<double>  gradValues;
            for (const auto& [key, value] : grad)
            {
                gradSamples.insert(gradSamples.end(), { key.first, static_cast<int32_t>(s), key.second });
                gradValues.insert(gradValues.end(), value.begin(), value.end());
            }
            std::vector<int32_t> strainSamples(nRows);
            for (size_t r = 0; r < nRows; ++r)
            {
                strainSamples[r] = static_cast<int32_t>(r);
            }
            const int32_t xyz[3] = { 0, 1, 2 };
            blocks[0].add_gradient(
                    "positions",
                    metatensor::TensorBlock(std::make_unique<metatensor::SimpleDataArray<double>>(
                                                    std::vector<uintptr_t>{ grad.size(), 3, 1 }, gradValues),
                                            metatensor::Labels({ "sample", "system", "atom" },
                                                               gradSamples.empty() ? nullptr : gradSamples.data(),
                                                               grad.size()),
                                            { metatensor::Labels({ "xyz" }, xyz, 3) },
                                            metatensor::Labels({ "energy" }, &zero, 1)));
            blocks[0].add_gradient(
                    "strain",
                    metatensor::TensorBlock(std::make_unique<metatensor::SimpleDataArray<double>>(
                                                    std::vector<uintptr_t>{ nRows, 3, 3, 1 }, strain),
                                            metatensor::Labels({ "sample" },
                                                               strainSamples.empty() ? nullptr : strainSamples.data(),
                                                               nRows),
                                            { metatensor::Labels({ "xyz_1" }, xyz, 3),
                                              metatensor::Labels({ "xyz_2" }, xyz, 3) },
                                            metatensor::Labels({ "energy" }, &zero, 1)));
            result.emplace_back(metatensor::Labels({ "_" }, &zero, 1), std::move(blocks));
        }
        return result;
    }

private:
    metatomic::PairListOptions pairs() const
    {
        return metatomic::PairListOptions::builder().cutoff(rc_).full_list(false).strict(true).build();
    }

    double rc_, a_, lambda_;
    bool   perAtom_;
};

mta_status_t load(const char* path, const char* /*options*/, mta_model_t* model)
{
    std::string   file(path);
    std::ifstream input(file);
    if (file.size() < 5 || file.substr(file.size() - 5) != ".pair" || !input)
    {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }
    double rc = 0, a = 0, lambda = 0;
    int    perAtom = 0;
    input >> rc >> a >> lambda >> perAtom;
    *model = metatomic::BaseModel::to_mta_model(std::make_unique<PairModel>(rc, a, lambda, perAtom != 0));
    return MTA_SUCCESS;
}

} // namespace

MTA_REGISTER_PLUGIN(register_plugin, register_plugin(mta_plugin_t{ MTA_ABI_VERSION, "oniom-test-pair", load }))
