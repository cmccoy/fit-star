#include "star_tree_optimizer.hpp"
#include "sequence.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "libhmsbeagle/beagle.h"

#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/tools/minima.hpp>

#include <Bpp/Numeric/Constraints.h>
#include <Bpp/Numeric/Prob/DiscreteDistribution.h>
#include <Bpp/Phyl/Model/SubstitutionModel.h>

#include <string>
#include <stdexcept>

#include <nlopt.hpp>

#include <cpplog.hpp>

namespace star_optim
{

cpplog::StdErrLogger log;

void beagleCheck(const int value, const std::string& details = "")
{
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
    const int nStates = model.getNumberOfStates();
    std::vector<double> r = rates.getCategories();
    std::vector<double> p = rates.getProbabilities();
    //const double expectation = std::inner_product(r.begin(), r.end(), p.begin(), 0.0);

    //for(int i = 0; i < r.size(); i++)
        //r[i] = r[i] / expectation;

    const double normExpectation = std::inner_product(r.begin(), r.end(), p.begin(), 0.0);
    if(std::abs(normExpectation - 1.0) > 1e-2) {
        LOG_INFO(log) << "Expected rate: " << normExpectation << '\n';
        auto pList = rates.getParameters();
        for(int i = 0; i < pList.size(); i++) {
            LOG_WARN(log) << pList[i].getName() << "\t=\t" << pList[i].getValue() << '\n';
        }
        assert(false && "Expected rate is not 1.0");
    }

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

struct Parameter
{
    bpp::Parametrizable* source;
    std::string parameterName;

    void setValue(const double value)
    {
        source->setParameterValue(parameterName, value);
    }

    double getValue() const
    {
        return source->getParameterValue(parameterName);
    }
};


struct NlOptParams {
    std::vector<Parameter>* paramList;
    StarTreeOptimizer* optimizer;
};

double nlLogLike(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    assert(grad.empty() && "Expected no derivative");

    NlOptParams* params = reinterpret_cast<NlOptParams*>(data);

    for(size_t i = 0; i < x.size(); i++) {
        params->paramList->operator[](i).setValue(x[i]);
    }

    return params->optimizer->starLikelihood();
}


// StarTreeOptimizer
StarTreeOptimizer::StarTreeOptimizer(std::vector<std::unique_ptr<bpp::SubstitutionModel>>& models,
                                     std::vector<std::unique_ptr<bpp::DiscreteDistribution>>& rates,
                                     std::vector<Sequence>& sequences) :
        models_(models),
        rates_(rates),
        sequences_(sequences),
        threshold_(0.1),
        maxRounds_(30),
        maxIterations_(300),
        bitTolerance_(50),
        minSubsParam_(1e-5),
        maxSubsParam_(20.0),
        hky85KappaPrior_(-1.0)
{
    beagleInstances_.resize(1);
#ifdef _OPENMP
    beagleInstances_.resize(omp_get_max_threads());
#endif
    for(std::vector<int>& v : beagleInstances_) {
        for(size_t i = 0; i < models_.size(); i++) {
            v.push_back(star_optim::createBeagleInstance(*models[i], *rates[i]));
        }
    }
}

StarTreeOptimizer::~StarTreeOptimizer()
{
    for(const std::vector<int>& v : beagleInstances_)
        for(const int i : v)
            beagleFinalizeInstance(i);
}

/// \brief Optimize the model & branch lengths distribution for a collection of sequences
size_t StarTreeOptimizer::optimize()
{
    double lastLogLike = starLikelihood();

    LOG_INFO(log) << "initial: " << lastLogLike << "\n";

    size_t iter = 0;
    for(iter = 0; iter < maxRounds_; iter++) {
        bool anyImproved = false;
        double logLike;

        for(size_t idx = 0; idx < models_.size(); idx++) {
            std::vector<Parameter> params;
            bpp::SubstitutionModel* model = models_[idx].get();
            bpp::DiscreteDistribution* r = rates_[idx].get();
            for(const std::string& s : model->getIndependentParameters().getParameterNames()) {
                params.push_back(Parameter { static_cast<bpp::Parametrizable*>(model), model->getParameterNameWithoutNamespace(s) });
            }
            for(const std::string& s : r->getIndependentParameters().getParameterNames()) {
                // fix the rate in position 1 at 1.0
                if((idx != 0 || s != "Constant.value"))
                    params.push_back(Parameter { static_cast<bpp::Parametrizable*>(r), r->getParameterNameWithoutNamespace(s) });
                assert(s != "Gamma.beta" && "Gamma.beta should be aliased");
            }
            const size_t nParam = params.size();

            // First, substitution model
            nlopt::opt opt(nlopt::LN_BOBYQA, nParam);
            NlOptParams optParams { &params, this };
            opt.set_max_objective(nlLogLike, &optParams);

            std::vector<double> lowerBounds(nParam, -std::numeric_limits<double>::max());
            std::vector<double> upperBounds(nParam, std::numeric_limits<double>::max());
            for(size_t i = 0; i < params.size(); i++) {
                bpp::Parameter bp = params[i].source->getParameter(params[i].parameterName);
                // Rate-related hacks
                if(params[i].parameterName == "value") {
                    lowerBounds[i] = 1e-6;
                    upperBounds[i] = 20;
                } else if (params[i].parameterName == "alpha") {
                    // The Bio++ implementation fails at small alpha
                    lowerBounds[i] = 0.3;
                    upperBounds[i] = 20;
                } else if(!bp.hasConstraint())
                    continue;
                else {
                    // TODO: changes in bpp 2.1
                    bpp::Interval* constraint = dynamic_cast<bpp::Interval*>(bp.getConstraint());
                    assert(constraint != nullptr);
                    lowerBounds[i] = constraint->getLowerBound() + 1e-7;
                    upperBounds[i] = std::min(constraint->getUpperBound(), 20.0);
                }
            }
            opt.set_lower_bounds(lowerBounds);
            opt.set_upper_bounds(upperBounds);
            opt.set_ftol_abs(threshold());
            opt.set_maxeval(maxIterations());

            std::vector<double> x(nParam);
            for(size_t i = 0; i < nParam; i++) {
                x[i] = params[i].getValue();
            }

            try {
                const int nlOptResult = opt.optimize(x, logLike);
                LOG_INFO(log) << "Optimization finished with " << nlOptResult << '\n';
            } catch(std::exception& e) {
                LOG_WARN(log) << "Optimization failed:  " << e.what() << '\n';
            }

            for(size_t i = 0; i < nParam; i++) {
                params[i].setValue(x[i]);
            }

            LOG_INFO(log) << "iteration " << iter << "." << idx << ": " << lastLogLike << " ->\t" << logLike << '\t' << logLike - lastLogLike << '\n';

            anyImproved = anyImproved || std::abs(logLike - lastLogLike) > threshold();
            lastLogLike = logLike;
        }


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
///
/// \param instances - a vector, with one entry per thread, with a vector containing a BEAGLE instance ID for each
/// partition.
double StarTreeOptimizer::starLikelihood()
{
    double result = 0.0;

    for(const std::vector<int>& v : beagleInstances_) {
        for(size_t i = 0; i < v.size(); i++) {
            updateBeagleInstance(v[i], *models_[i], *rates_[i]);
        }
    }

    double prior = 0.0;
    for(size_t i = 0; i < models_.size(); i++) {
        result += starLikelihood(i);
        if(hky85KappaPrior_ > 0.0) {
            assert(models_[i]->hasParameter("kappa") && "Model does not have kappa?");
            boost::math::lognormal_distribution<double> distn(1, hky85KappaPrior_);
            prior += std::log(boost::math::pdf(distn, models_[i]->getParameterValue("kappa")));
        }
    }

    return result + prior;
}

/// \brief Calculate the likelihood of a collection of sequences under the star tree model
double StarTreeOptimizer::starLikelihood(const size_t partition)
{
    double result = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:result)
#endif
    for(size_t i = 0; i < sequences_.size(); i++) {
        int instance_idx = 0;
#ifdef _OPENMP
        instance_idx = omp_get_thread_num();
#endif
        const int instance = beagleInstances_[instance_idx][partition];
        result += pairLogLikelihood(instance, sequences_[i].substitutions[partition], sequences_[i].distance);
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
        Sequence& s = sequences_[i];
        std::vector<int> inst = beagleInstances_[idx];

        auto f = [&inst, &s](const double d) {
            assert(!std::isnan(d) && "NaN distance?");
            s.distance = d;
            double result = 0.0;
            for(size_t i = 0; i < inst.size(); i++)
                result += pairLogLikelihood(inst[i], s.substitutions[i], s.distance);
            return -result;
        };

        boost::uintmax_t maxIterations = 100;
        std::pair<double, double> res =
            boost::math::tools::brent_find_minima(f, 1e-6, 0.8, 50, maxIterations);
        assert(!std::isnan(res.first) && "NaN distance?");
        s.distance = res.first;
    }
}



} // namespace
