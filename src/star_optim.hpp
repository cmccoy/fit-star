#ifndef STAROPTIM_STAROPTIM_H
#define STAROPTIM_STAROPTIM_H

#include <cstdlib>
#include <memory>
#include <vector>

struct Sequence;

namespace bpp
{
class SubstitutionModel;
class DiscreteDistribution;
}

namespace star_optim
{

/// \brief calculate the star-tree likelihood
double starLikelihood(const bpp::SubstitutionModel&,
                      const bpp::DiscreteDistribution&,
                      const std::vector<Sequence>&,
                      const size_t partition);

double starLikelihood(const std::vector<std::unique_ptr<bpp::SubstitutionModel>>&,
                      const std::vector<std::unique_ptr<bpp::DiscreteDistribution>>&,
                      const std::vector<Sequence>&);

/// \brief estimate branch lengths
void estimateBranchLengths(const std::vector<std::unique_ptr<bpp::SubstitutionModel>>&,
                           const std::vector<std::unique_ptr<bpp::DiscreteDistribution>>&,
                           std::vector<Sequence>&);

/// \brief Optimize the model & branch lengths distribution for a collection of sequences
size_t optimize(std::vector<std::unique_ptr<bpp::SubstitutionModel>>&,
                std::vector<std::unique_ptr<bpp::DiscreteDistribution>>&,
                std::vector<Sequence>&,
                const bool verbose = true);
}
#endif
