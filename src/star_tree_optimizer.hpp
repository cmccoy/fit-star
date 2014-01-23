#ifndef STAROPTIM_STAROPTIM_H
#define STAROPTIM_STAROPTIM_H

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>


namespace bpp
{
class SubstitutionModel;
class DiscreteDistribution;
}

namespace star_optim
{

struct AlignedPair;

struct PartitionModel
{
    bpp::SubstitutionModel* model;
    bpp::DiscreteDistribution* rateDist;
};

class StarTreeOptimizer
{
public:
    /// Minimum improvement in LL over a round
    double& threshold() { return threshold_; };
    const double& threshold() const { return threshold_; };
    void threshold(const double value) { threshold_ = value; }
    /// Maximum rounds of fitting
    size_t& maxRounds() { return maxRounds_; };
    const size_t& maxRounds() const { return maxRounds_; };
    void maxRounds(const size_t value) { maxRounds_ = value; }
    /// Maximum number of iterations
    size_t& maxIterations() { return maxIterations_; };
    const size_t& maxIterations() const { return maxIterations_; };
    void maxIterations(const size_t value) { maxIterations_ = value; }
    /// Minimum value for substitution parameters
    double& minSubsParam() { return minSubsParam_; };
    const double& minSubsParam() const { return minSubsParam_; };
    void minSubsParam(const double value) { minSubsParam_ = value; }
    /// Maximum value for substitution parameters
    double& maxSubsParam() { return maxSubsParam_; };
    const double& maxSubsParam() const { return maxSubsParam_; };
    void maxSubsParam(const double value) { maxSubsParam_ = value; }
    /// Bit tolerance for branch length optimization (via Brent)
    size_t& bitTolerance() { return bitTolerance_; };
    const size_t& bitTolerance() const { return bitTolerance_; };
    void bitTolerance(const size_t value) { bitTolerance_ = value; }
    /// HKY prior
    double& hky85KappaPrior() { return hky85KappaPrior_; };
    const double& hky85KappaPrior() const { return hky85KappaPrior_; };
    void hky85KappaPrior(const double value) { hky85KappaPrior_ = value; }
    /// Should the rate distribution be fit for each partition?
    std::unordered_map<std::string, bool>& fitRates() { return fitRates_; };
    const std::unordered_map<std::string, bool>& fitRates() const { return fitRates_; };
    void fitRates(const std::unordered_map<std::string, bool> value) { fitRates_ = value; }

    StarTreeOptimizer(const std::unordered_map<std::string, PartitionModel>& models,
                      std::vector<AlignedPair>& sequences);
    ~StarTreeOptimizer();

    size_t optimize();
    /// \brief calculate the star-tree likelihood
    double starLikelihood(const std::string& partition);

    double starLikelihood();
    /// \brief estimate branch lengths
    void estimateBranchLengths();
private:
    std::vector<std::unordered_map<std::string, int>> beagleInstances_;
    std::unordered_map<std::string, PartitionModel> partitionModels_;
    std::vector<AlignedPair>& sequences_;
    double threshold_;
    size_t maxRounds_, maxIterations_, bitTolerance_;
    double minSubsParam_, maxSubsParam_, hky85KappaPrior_;
    std::unordered_map<std::string, bool> fitRates_;
};

}
#endif
