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

class StarTreeOptimizer
{
public:
    /// Minimum improvement in LL over a round
    double& threshold() { return threshold_; };
    const double& threshold() const { return threshold_; };
    void threshold(const double value) { threshold_ = value; }
    /// Maximum rounds of fitting
    size_t& max_rounds() { return max_rounds_; };
    const size_t& max_rounds() const { return max_rounds_; };
    void max_rounds(const size_t value) { max_rounds_ = value; }
    /// Maximum number of iterations
    size_t& max_iter() { return max_iter_; };
    const size_t& max_iter() const { return max_iter_; };
    void max_iter(const size_t value) { max_iter_ = value; }
    /// Minimum value for substitution parameters
    double& min_subs_param() { return min_subs_param_; };
    const double& min_subs_param() const { return min_subs_param_; };
    void min_subs_param(const double value) { min_subs_param_ = value; }
    /// Maximum value for substitution parameters
    double& max_subs_param() { return max_subs_param_; };
    const double& max_subs_param() const { return max_subs_param_; };
    void max_subs_param(const double value) { max_subs_param_ = value; }
    /// Bit tolerance for branch length optimization (via Brent)
    size_t& bit_tol() { return bit_tol_; };
    const size_t& bit_tol() const { return bit_tol_; };
    void bit_tol(const size_t value) { bit_tol_ = value; }
    /// HKY prior
    double& hky85_prior() { return hky85_prior_; };
    const double& hky85_prior() const { return hky85_prior_; };
    void hky85_prior(const double value) { hky85_prior_ = value; }

    StarTreeOptimizer(std::vector<std::unique_ptr<bpp::SubstitutionModel>>& models,
                      std::vector<std::unique_ptr<bpp::DiscreteDistribution>>& rates,
                      std::vector<Sequence>& sequences);
    ~StarTreeOptimizer();

    size_t optimize(const bool verbose = true);
    /// \brief calculate the star-tree likelihood
    double starLikelihood(const size_t partition);

    double starLikelihood();
    /// \brief estimate branch lengths
    void estimateBranchLengths();
private:
    std::vector<std::vector<int>> beagleInstances_;
    std::vector<std::unique_ptr<bpp::SubstitutionModel>>& models_;
    std::vector<std::unique_ptr<bpp::DiscreteDistribution>>& rates_;
    std::vector<Sequence>& sequences_;
    double threshold_;
    size_t max_rounds_, max_iter_, bit_tol_;
    double min_subs_param_, max_subs_param_, hky85_prior_;
};

}
#endif
