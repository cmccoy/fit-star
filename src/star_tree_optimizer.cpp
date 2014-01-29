#include "star_tree_optimizer.hpp"
#include "aligned_pair.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "libhmsbeagle/beagle.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/tools/minima.hpp>

#include <Bpp/Numeric/Constraints.h>
#include <Bpp/Numeric/Prob/DiscreteDistribution.h>
#include <Bpp/Phyl/Model/SubstitutionModel.h>

#include <string>
#include <stdexcept>
#include <unordered_set>

#include <nlopt.hpp>

#include <cpplog.hpp>

namespace fit_star
{

cpplog::StdErrLogger log;

std::string nlOptSuccessCodeToString(const int r)
{
    switch(r) {
        case nlopt::SUCCESS: return "success";
        case nlopt::STOPVAL_REACHED: return "stopping value reached";
        case nlopt::FTOL_REACHED: return "tolerance in f reached";
        case nlopt::XTOL_REACHED: return "tolerance in x reached";
        case nlopt::MAXEVAL_REACHED: return "maximum evaluations reached";
        case nlopt::MAXTIME_REACHED: return "maximum time reached";
        default: return "unknown";
    }
}

void beagleCheck(const int value, const std::string& details = "")
{
    if(value >= 0)
        return;

    std::string s;
    switch(value) {
        case BEAGLE_ERROR_GENERAL: s = "BEAGLE_ERROR_GENERAL"; break;
        case BEAGLE_ERROR_OUT_OF_MEMORY: s = "BEAGLE_ERROR_OUT_OF_MEMORY"; break;
        case BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION: s = "BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION"; break;
        case BEAGLE_ERROR_UNINITIALIZED_INSTANCE: s = "BEAGLE_ERROR_UNINITIALIZED_INSTANCE"; break;
        case BEAGLE_ERROR_OUT_OF_RANGE: s = "BEAGLE_ERROR_OUT_OF_RANGE"; break;
        case BEAGLE_ERROR_NO_RESOURCE: s = "BEAGLE_ERROR_NO_RESOURCE"; break;
        case BEAGLE_ERROR_NO_IMPLEMENTATION: s = "BEAGLE_ERROR_NO_IMPLEMENTATION"; break;
        case BEAGLE_ERROR_FLOATING_POINT: s = "BEAGLE_ERROR_FLOATING_POINT"; break;
        default: break;
    }
    if(!s.empty()) {
        LOG_WARN(log) << s << " " << details << '\n';
        throw std::runtime_error(s + " " + details);
    }
}



/// Copy the contents of vec into arr
/// \param arr destination array, with length at least vec.size()
/// \param vec Vector to copy from
void blitVectorToArray(double* arr, const std::vector<double>& vec)
{
    for(std::vector<double>::const_iterator it = vec.begin(); it != vec.end(); ++it)
        *arr++ = *it;
}

/// Copy the contents of matrix into arr, in row-major order
/// \param arr destination array, with length at least nrows x ncols in length
/// \param matrix Vector to copy from
void blitMatrixToArray(double* arr, const bpp::Matrix<double>& matrix)
{
    const int cols = matrix.getNumberOfColumns(), rows = matrix.getNumberOfRows();
    for(int i = 0; i < rows; ++i) {
        blitVectorToArray(arr, matrix.row(i));
        arr += cols;
    }
}

/// \brief calculate the likelihood of a collection of substitutions using an initialized BEAGLE instance.
double pairLogLikelihood(const int beagleInstance, const Eigen::Matrix4d& substitutions, const double distance)
{
    const size_t nStates = 4;
    std::vector<double> patternWeights;
    patternWeights.reserve(nStates * nStates);
    for(size_t i = 0; i < nStates; i++) {
        for(size_t j = 0; j < nStates; j++) {
            patternWeights.push_back(substitutions(i, j));
        }
    }

    beagleCheck(beagleSetPatternWeights(beagleInstance, patternWeights.data()));

    std::vector<int> nodeIndices { 0, 1 };
    std::vector<double> edgeLengths { distance, 0 };
    const int rootIndex = 2;

    beagleCheck(beagleUpdateTransitionMatrices(beagleInstance,
                0,
                nodeIndices.data(),
                nullptr,
                nullptr,
                edgeLengths.data(),
                nodeIndices.size()),
                "updateTransitionMatrices");

    BeagleOperation op {rootIndex, BEAGLE_OP_NONE, BEAGLE_OP_NONE,
                        nodeIndices[0], nodeIndices[0],
                        nodeIndices[1], nodeIndices[1]
                       };

    beagleCheck(beagleUpdatePartials(beagleInstance,      // instance
                                     &op,
                                     1,  // # of ops
                                     rootIndex), // cumulative scaling index
                "updatePartials");

    int scaleIndex = op.destinationPartials;
    beagleCheck(beagleAccumulateScaleFactors(beagleInstance, &scaleIndex, 1, BEAGLE_OP_NONE),
                "accumulateScaleFactors");
    double logLike = 1;
    const int weightIndex = 0, freqIndex = 0;
    int returnCode = beagleCalculateRootLogLikelihoods(beagleInstance,               // instance
                     &rootIndex,
                     &weightIndex,
                     &freqIndex,
                     &rootIndex,
                     1,
                     &logLike);
    beagleCheck(returnCode, "rootLogLike");
    return logLike;
}

/// \brief Update a BEAGLE instance
void updateBeagleInstance(const int instance,
                          const bpp::SubstitutionModel& model,
                          const bpp::DiscreteDistribution& rates)
{
    const size_t nStates = static_cast<size_t>(model.getNumberOfStates());
    std::vector<double> r = rates.getCategories();
    if(model.hasParameter("rate")) {
        const double rate = model.getParameterValue("rate");
        std::transform(r.begin(), r.end(), r.begin(), [rate](double d) { return d * rate; });
    }

    std::vector<double> p = rates.getProbabilities();

    //const double normExpectation = std::inner_product(r.begin(), r.end(), p.begin(), 0.0);
    //if(std::abs(normExpectation - 1.0) > 1e-2) {
    //LOG_INFO(log) << "Expected rate: " << normExpectation << '\n';
    //auto pList = rates.getParameters();
    //for(int i = 0; i < pList.size(); i++) {
    //LOG_WARN(log) << pList[i].getName() << "\t=\t" << pList[i].getValue() << '\n';
    //}
    //assert(false && "Expected rate is not 1.0");
    //}

    // Fill rates
    beagleSetCategoryRates(instance, r.data());
    beagleSetCategoryWeights(instance, 0, p.data());

    // And states
    std::vector<int> ref(nStates * nStates), qry(nStates * nStates);
    for(size_t i = 0; i < nStates; i++) {
        for(size_t j = 0; j < nStates; j++) {
            ref[nStates * i + j] = i;
            qry[nStates * i + j] = j;
        }
    }
    beagleSetTipStates(instance, 0, ref.data());
    beagleSetTipStates(instance, 1, qry.data());

    // And eigen decomposition
    std::vector<double> evec(nStates * nStates),
        ivec(nStates * nStates),
        eval(nStates);
    blitMatrixToArray(evec.data(), model.getColumnRightEigenVectors());
    blitMatrixToArray(ivec.data(), model.getRowLeftEigenVectors());
    blitVectorToArray(eval.data(), model.getEigenValues());
    beagleSetEigenDecomposition(instance, 0, evec.data(), ivec.data(), eval.data());

    // And state frequencies
    beagleSetStateFrequencies(instance, 0, model.getFrequencies().data());
}

/// \brief Create a BEAGLE instance
int createBeagleInstance(const bpp::SubstitutionModel& model,
                         const bpp::DiscreteDistribution& rates)
{
    const int nStates = model.getNumberOfStates();
    const int nRates = rates.getNumberOfCategories();
    const int nTips = 2;
    const int nBuffers = 3;
    BeagleInstanceDetails deets;
    const int instance = beagleCreateInstance(nTips,                            /**< Number of tip data elements (input) */
                         nBuffers,       /**< Number of partials buffers to create (input) */
                         nTips,                    /**< Number of compact state representation buffers to create (input) */
                         nStates,           /**< Number of states in the continuous-time Markov chain (input) */
                         nStates * nStates,            /**< Number of site patterns to be handled by the instance (input) */
                         1,                    /**< Number of rate matrix eigen-decomposition buffers to allocate (input) */
                         nBuffers,                    /**< Number of rate matrix buffers (input) */
                         nRates,            /**< Number of rate categories (input) */
                         nBuffers,                       /**< Number of scaling buffers */
                         nullptr,                     /**< List of potential resource on which this instance is allowed (input, NULL implies no restriction */
                         0,                        /**< Length of resourceList list (input) */
                         BEAGLE_FLAG_VECTOR_SSE | BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_SCALING_AUTO, // Bit-flags indicating
                         0,                /**< Bit-flags indicating required implementation characteristics, see BeagleFlags (input) */
                         &deets);

    beagleCheck(instance);
    updateBeagleInstance(instance, model, rates);
    return instance;
}

struct NlOptParams {
    bpp::ParameterList* paramList;
    StarTreeOptimizer* optimizer;
};

double nlLogLike(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    assert(grad.empty() && "Expected no derivative");

    NlOptParams* params = reinterpret_cast<NlOptParams*>(data);
    bpp::ParameterList& pl = *params->paramList;

    assert(pl.size() == x.size());
    for(size_t i = 0; i < x.size(); i++) {
        pl[i].setValue(x[i]);
    }

    params->optimizer->matchParameters(pl);

    return params->optimizer->starLikelihood();
}


// StarTreeOptimizer
StarTreeOptimizer::StarTreeOptimizer(const std::unordered_map<std::string, PartitionModel>& models,
                                     std::vector<AlignedPair>& sequences) :
    partitionModels_(models),
    sequences_(sequences),
    threshold_(0.1),
    maxRounds_(30),
    maxIterations_(1000),
    bitTolerance_(50),
    maxTime_(30 * 60),
    minSubsParam_(1e-5),
    maxSubsParam_(20.0),
    hky85KappaPrior_(-1.0)
{
    // BEAGLE
    beagleInstances_.resize(1);
#ifdef _OPENMP
    beagleInstances_.resize(omp_get_max_threads());
#endif
    for(std::unordered_map<std::string, int>& m : beagleInstances_) {
        for(const auto& p : models) {
            m[p.first] = createBeagleInstance(*p.second.model, *p.second.rateDist);
        }
    }

    for(const auto& p : models)
        fitRates_[p.first] = true;
}

StarTreeOptimizer::~StarTreeOptimizer()
{
    for(const std::unordered_map<std::string, int>& v : beagleInstances_)
        for(const auto p : v)
            beagleFinalizeInstance(p.second);
}

/// \brief Optimize the model & branch lengths distribution for a collection of sequences
size_t StarTreeOptimizer::optimize()
{
    double lastLogLike = starLikelihood();

    LOG_INFO(log) << "initial: " << lastLogLike << "\n";

    size_t iter = 0;
    for(iter = 0; iter < maxRounds(); iter++) {
        bool anyImproved = false;
        double logLike;

        bpp::ParameterList toFit;
        for(std::pair<std::string, PartitionModel> p : partitionModels_) {
            bpp::SubstitutionModel* model = p.second.model;
            bpp::DiscreteDistribution* r = p.second.rateDist;
            toFit.includeParameters(model->getIndependentParameters());
            if(fitRates_[p.first]) {
                toFit.includeParameters(r->getIndependentParameters());
            }
        }

        const size_t nParam = toFit.size();
        LOG_INFO(log) << "Fitting " << nParam << " parameters";
        //for(size_t i = 0; i < toFit.size(); i++) {
        //LOG_INFO(log) << "  - " << toFit[i].getName() << '\t' << toFit[i].getValue();
        //}

        // Optimize
        nlopt::opt opt(nlopt::LN_BOBYQA, nParam);
        NlOptParams optParams { &toFit, this };
        opt.set_max_objective(nlLogLike, &optParams);
        opt.set_initial_step(std::vector<double>(nParam, 0.01));

        std::vector<double> lowerBounds(nParam, -std::numeric_limits<double>::max());
        std::vector<double> upperBounds(nParam, std::numeric_limits<double>::max());
        for(size_t i = 0; i < toFit.size(); i++) {
            bpp::Parameter& bp = toFit[i];
            // Rate-related hacks
            if(boost::algorithm::ends_with(bp.getName(), "Constant.value")) {
                lowerBounds[i] = 1e-6;
                upperBounds[i] = 20;
            } else if(!bp.hasConstraint())
                continue;
            else {
                bpp::IntervalConstraint* constraint = dynamic_cast<bpp::IntervalConstraint*>(bp.getConstraint());
                assert(constraint != nullptr);
                lowerBounds[i] = constraint->getLowerBound() + 1e-7;
                upperBounds[i] = std::min(constraint->getUpperBound(), 20.0);
            }
        }
        opt.set_lower_bounds(lowerBounds);
        opt.set_upper_bounds(upperBounds);
        opt.set_ftol_abs(threshold());
        opt.set_xtol_rel(0.001);
        opt.set_maxeval(maxIterations());
        opt.set_maxtime(maxTime());

        std::vector<double> x(nParam);
        for(size_t i = 0; i < nParam; i++) {
            x[i] = toFit[i].getValue();
        }

        try {
            const int nlOptResult = opt.optimize(x, logLike);
            LOG_INFO(log) << "Optimization finished with '" << nlOptSuccessCodeToString(nlOptResult) << "'\n";
        } catch(std::exception& e) {
            LOG_WARN(log) << "Optimization failed:  " << e.what() << '\n';
            for(size_t i = 0 ; i < toFit.size(); i++) {
                LOG_INFO(log) << " - " << toFit[i].getName() << '\t' << x[i] << "\t(" << lowerBounds[i] << ", " << upperBounds[i] << ")";
            }
            throw e;
        }

        LOG_TRACE(log) << "after fitting: ";
        for(size_t i = 0; i < nParam; i++) {
            LOG_TRACE(log) << "  - " << toFit[i].getName() << '\t' << toFit[i].getValue() << '\t' << x[i];
            toFit[i].setValue(x[i]);
        }
        matchParameters(toFit);

        LOG_INFO(log) << "iteration " << iter << ": " << lastLogLike << " ->\t" << logLike << '\t' << logLike - lastLogLike << '\n';

        anyImproved = anyImproved || std::abs(logLike - lastLogLike) > threshold();
        lastLogLike = logLike;


        // then, branch lengths
        estimateBranchLengths();
        logLike = starLikelihood();

        LOG_INFO(log) << "iteration " << iter << " (branch lengths): " << lastLogLike << " ->\t" << logLike << '\t' << logLike - lastLogLike << '\n';

        anyImproved = anyImproved || std::abs(logLike - lastLogLike) > threshold();
        lastLogLike = logLike;

        if(!anyImproved)
            break;
    }

    return iter;
}

/// \brief Calculate the likelihood of a collection of sequences under the star tree model
double StarTreeOptimizer::starLikelihood()
{
    double result = 0.0;

    for(const std::unordered_map<std::string, int>& partitionInstanceMap : beagleInstances_) {
        for(const std::pair<std::string, int>& p : partitionInstanceMap) {
            const PartitionModel& partModel = partitionModels_[p.first];
            updateBeagleInstance(p.second, *partModel.model, *partModel.rateDist);
        }
    }

    double prior = 0.0;
    for(const auto& m : partitionModels_) {
        result += starLikelihood(m.first);
        if(hky85KappaPrior_ > 0.0) {
            assert(m.second.model->hasParameter("kappa") && "Model does not have kappa?");
            boost::math::lognormal_distribution<double> distn(1, hky85KappaPrior_);
            prior += std::log(boost::math::pdf(distn, m.second.model->getParameterValue("kappa")));
        }
    }

    return result + prior;
}

/// \brief Calculate the likelihood of a collection of sequences under the star tree model
double StarTreeOptimizer::starLikelihood(const std::string& partition)
{
    double result = 0.0;
    auto finder = [&partition](const Partition & p) { return p.name == partition; };
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:result)
#endif
    for(size_t i = 0; i < sequences_.size(); i++) {
        const AlignedPair& sequence = sequences_[i];

        int instance_idx = 0;
#ifdef _OPENMP
        instance_idx = omp_get_thread_num();
#endif
        auto p = std::find_if(sequence.partitions.begin(), sequence.partitions.end(), finder);
        if(p != sequence.partitions.end()) {
            assert(beagleInstances_[instance_idx].count(partition) > 0 && "No beagle instance for partition.");
            const int instance = beagleInstances_[instance_idx][partition];
            result += pairLogLikelihood(instance, p->substitutions, sequence.distance);
        }
    }
    return result;
}

/// \brief estimate branch lengths
void StarTreeOptimizer::estimateBranchLengths()
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(size_t i = 0; i < sequences_.size(); i++) {
        int idx = 0;
#ifdef _OPENMP
        idx = omp_get_thread_num();
#endif
        AlignedPair& s = sequences_[i];
        std::unordered_map<std::string, int> inst = beagleInstances_[idx];

        auto f = [&inst, &s](const double d) {
            assert(!std::isnan(d) && "NaN distance?");
            s.distance = d;
            double result = 0.0;
            for(const Partition& p : s.partitions) {
                assert(inst.count(p.name) > 0 && "Missing partition.");
                result += pairLogLikelihood(inst[p.name], p.substitutions, s.distance);
            }
            return -result;
        };

        boost::uintmax_t maxIterations = 100;
        std::pair<double, double> res =
            boost::math::tools::brent_find_minima(f, 1e-6, 0.8, 50, maxIterations);
        assert(!std::isnan(res.first) && "NaN distance?");
        s.distance = res.first;
    }
}

/// Match model parameters in `pl` - should be namespaced
bool StarTreeOptimizer::matchParameters(const bpp::ParameterList& pl)
{
    std::unordered_set<bpp::Parametrizable*> toVisit;
    for(auto& pair : partitionModels_) {
        toVisit.insert(pair.second.model);
        toVisit.insert(pair.second.rateDist);
    }
    bool result = false;
    for(bpp::Parametrizable* p : toVisit) {
        bool updated = p->matchParametersValues(pl);
        result = result || updated;
    }

    return result;
}


} // namespace
