#include <algorithm>
#include <vector>
#include <iostream>
#include "gtest/gtest.h"
#include "star_tree_optimizer.hpp"
#include "aligned_pair.hpp"
#include "log_tricks.hpp"
#include "eigen_bpp.hpp"

#include <libhmsbeagle/beagle.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <Bpp/Phyl/Distance/DistanceEstimation.h>
#include <Bpp/Phyl/Likelihood/DiscreteRatesAcrossSitesTreeLikelihood.h>
#include <Bpp/Phyl/Likelihood/RHomogeneousTreeLikelihood.h>
#include <Bpp/Phyl/Model.all>
#include <Bpp/Phyl/TreeTemplate.h>
#include <Bpp/Seq/Alphabet/AlphabetTools.h>
#include <Bpp/Seq/Container/VectorSiteContainer.h>
#include <Bpp/Seq/Sequence.h>

using namespace fit_star;


bpp::VectorSiteContainer createSites(const AlignedPair& sequence,
                                     const bpp::Alphabet *alphabet = &bpp::AlphabetTools::DNA_ALPHABET)
{
    std::vector<std::string> names {"ref", "qry" };
    std::vector<std::string> seqs(2);
    const size_t nStates = alphabet->getSize();

    for(size_t i = 0; i < nStates; i++) {
        for(size_t j = 0; j < nStates; j++) {
            int count = static_cast<int>(sequence.partitions[0].substitutions(i, j));
            for(int k = 0; k < count; k++) {
                seqs[0] += alphabet->intToChar(i);
                seqs[1] += alphabet->intToChar(j);
            }
        }
    }

    bpp::VectorSiteContainer result(alphabet);
    for(size_t i = 0; i < 2; i++)
        result.addSequence(bpp::BasicSequence(names[i], seqs[i], alphabet));

    return result;
}

double starLogLike(std::vector<AlignedPair>& sequences,
                   std::unique_ptr<bpp::SubstitutionModel>& model,
                   std::unique_ptr<bpp::DiscreteDistribution>& rates,
                   bool fixRootFrequencies = false) {
    std::unordered_map<std::string, PartitionModel> models ;
    models[""] =  PartitionModel { model.get(), rates.get() };

    fit_star::StarTreeOptimizer optimizer(models, sequences);
    optimizer.fixRootFrequencies(fixRootFrequencies);
    return optimizer.starLikelihood("");
}

void checkAgainstBpp(std::vector<AlignedPair>& sequences,
                     std::unique_ptr<bpp::SubstitutionModel>& model,
                     std::unique_ptr<bpp::DiscreteDistribution>& rates,
                     double& logL)
{
    using namespace bpp;
    ASSERT_EQ(1, sequences.size());
    const double starLL = starLogLike(sequences, model, rates, false);

    VectorSiteContainer sites = createSites(sequences[0], model->getAlphabet());

    Node *root = new Node(0),
         *c1 = new Node(1, sites.getSequence(0).getName()),
         *c2 = new Node(2, sites.getSequence(1).getName());
    root->addSon(c1);
    root->addSon(c2);
    c1->setDistanceToFather(sequences[0].distance / 2);
    c2->setDistanceToFather(sequences[0].distance / 2);
    TreeTemplate<Node> tree(root);

    RHomogeneousTreeLikelihood calc(tree, sites, model.get(), rates.get(), true, false);
    calc.initialize();
    calc.computeTreeLikelihood();

    const double bppLL = calc.getLogLikelihood();
    EXPECT_NEAR(bppLL, starLL, 1e-3) << "Likelihood calculations do not match.";
    logL = starLL;
}

template<typename Scalar>
struct LogSumBinaryOp
{
    EIGEN_EMPTY_STRUCT_CTOR(LogSumBinaryOp)
    typedef Scalar result_type;
    Scalar operator()(const Scalar x, const Scalar y) const { return fit_star::logSum(x, y); }
};


template <typename T>
std::ostream& operator<<(ostream& o, const vector<T>& v) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(o,","));

    return o; // Edited
}


// Equivalent of checkAgainstBpp, but using fixed root, implemented in Eigen
void checkAgainstEigen(std::vector<AlignedPair>& sequences,
                       std::unique_ptr<bpp::SubstitutionModel>& model,
                       std::unique_ptr<bpp::DiscreteDistribution>& rateDist)
{
    using namespace Eigen;

    ASSERT_EQ(1, sequences.size());
    ASSERT_EQ(1, sequences[0].partitions.size());

    const VectorXd bppEval = bppToEigen(model->getEigenValues());

    const Eigen::MatrixXd bppIEvec = bppToEigen(model->getRowLeftEigenVectors());
    const Eigen::MatrixXd bppEvec = bppToEigen(model->getColumnRightEigenVectors());

    const std::vector<double> rates = rateDist->getCategories();
    const std::vector<double> rateProbs = rateDist->getProbabilities();
    std::vector<MatrixXd> pMatrices(rates.size());
    for(size_t i = 0; i < rates.size(); i++) {
        const VectorXd lambda = (ArrayXd(bppEval) * sequences[0].distance * rates[i]).exp();
        pMatrices[i] = bppEvec * lambda.asDiagonal() * bppIEvec;
    }
    auto f = [](const double d) { return std::log(d); };
    for(size_t i = 0; i < rates.size(); i++) {
        pMatrices[i] = pMatrices[i].unaryExpr(f);
        if(rates.size() > 1) {
            const Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(model->getNumberOfStates(),
                                                               model->getNumberOfStates());
            Eigen::MatrixXd r = ones * std::log(rateProbs[i]);
            pMatrices[i] += r;
        }
    }

    // Sum over mixture
    MatrixXd p = pMatrices[0];
    for(size_t i = 1; i < rates.size(); i++) {
        p = p.binaryExpr(pMatrices[i], LogSumBinaryOp<double>());
    }

    double expectedLL = p.cwiseProduct(sequences[0].partitions[0].substitutions).sum();
    double actualLL = starLogLike(sequences, model, rateDist, true);

    EXPECT_NEAR(expectedLL, actualLL, 1e-5);
}

TEST(FitStar, simple_jc) {
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::GTR(&bpp::AlphabetTools::DNA_ALPHABET));
    bpp::RateDistributionFactory fac(4);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Constant"));

    std::vector<AlignedPair> v { AlignedPair() };
    AlignedPair& s = v.front();

    s.partitions.resize(1);
    Eigen::MatrixXd& m = s.partitions[0].substitutions;
    m.resize(4, 4);
    m.fill(0.0);
    m.diagonal() = Eigen::VectorXd::Constant(4, 1.0);

    s.distance = 0.02;

    double ll = 0.0;
    checkAgainstBpp(v, model, rates, ll);
    checkAgainstEigen(v, model, rates);
    const double expll = -5.62490959465585; // from bppml
    EXPECT_NEAR(expll, ll, 1e-5);
}

TEST(FitStar, known_distance) {
    bpp::DNA dna;
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::GTR(&dna));
    model->setParameterValue("theta", 0.2);
    model->setParameterValue("theta2", 0.8);
    model->setParameterValue("a", 0.45);
    model->setParameterValue("b", 1.3);

    bpp::RateDistributionFactory fac(1);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Constant"));

    std::vector<AlignedPair> v { AlignedPair() };

    AlignedPair& s = v.front();
    s.partitions.resize(1);
    s.partitions[0].substitutions.resize(4, 4);
    s.partitions[0].substitutions <<
        94, 3, 2, 1,
        2, 95, 2, 1,
        2, 4, 89, 5,
        1, 3, 2, 94;
    s.distance = 0.02;

    double ll = 0.0;
    checkAgainstBpp(v, model, rates, ll);
    checkAgainstEigen(v, model, rates);
}


TEST(FitStar, gamma_variation) {
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::GTR(&bpp::AlphabetTools::DNA_ALPHABET));
    bpp::RateDistributionFactory fac(4);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Gamma"));
    rates->setParameterValue("alpha", 1.2);

    // Alpha and beta should be linked.
    EXPECT_NEAR(1.2, rates->getParameterValue("beta"), 1e-6);

    model->setParameterValue("a", 0.5);
    model->setParameterValue("theta", 0.6);
    model->setParameterValue("theta1", 0.4);
    model->setParameterValue("theta2", 0.4);

    std::vector<AlignedPair> v { AlignedPair() };

    AlignedPair& s = v.front();

    s.partitions.resize(1);
    s.partitions[0].substitutions.resize(4, 4);
    s.partitions[0].substitutions <<
        94, 3, 2, 1,
        2, 95, 2, 1,
        2, 4, 89, 5,
        1, 3, 2, 94;
    s.distance = 0.02;

    double ll = 0.0;
    checkAgainstBpp(v, model, rates, ll);
    checkAgainstEigen(v, model, rates);
    rates->setParameterValue("alpha", 0.10);
    checkAgainstBpp(v, model, rates, ll);
    checkAgainstEigen(v, model, rates);
}

TEST(FitStar, amino_acids) {
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::LG08(&bpp::AlphabetTools::PROTEIN_ALPHABET));
    bpp::RateDistributionFactory fac(1);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Constant"));

    std::vector<AlignedPair> v { AlignedPair() };

    AlignedPair& s = v.front();

    s.partitions.resize(1);
    s.partitions[0].substitutions.resize(model->getNumberOfStates(),
                                         model->getNumberOfStates());
    s.partitions[0].substitutions <<
        123, 2, 3, 4, 6, 4, 4, 6, 2, 6, 8, 1, 2, 4, 1, 7, 3, 4, 7, 3,
        5, 131, 2, 7, 3, 2, 2, 2, 3, 5, 7, 4, 2, 4, 3, 3, 5, 4, 0, 6,
        6, 1, 120, 4, 5, 2, 3, 6, 3, 6, 8, 4, 6, 4, 6, 3, 4, 2, 5, 2,
        5, 4, 5, 116, 4, 6, 2, 5, 7, 2, 3, 6, 3, 4, 5, 3, 5, 8, 5, 2,
        1, 4, 5, 7, 137, 1, 2, 5, 1, 4, 3, 8, 0, 1, 1, 4, 6, 3, 1, 6,
        4, 4, 3, 1, 5, 126, 2, 4, 8, 2, 3, 3, 2, 2, 8, 3, 6, 3, 5, 6,
        8, 4, 4, 2, 5, 1, 132, 4, 2, 3, 4, 5, 2, 5, 2, 2, 5, 2, 3, 5,
        2, 3, 4, 3, 5, 3, 1, 131, 3, 5, 3, 6, 5, 1, 6, 4, 6, 0, 5, 4,
        9, 4, 8, 3, 1, 3, 4, 6, 114, 2, 5, 4, 7, 6, 5, 4, 1, 5, 6, 3,
        5, 3, 6, 7, 5, 6, 5, 3, 5, 104, 7, 2, 5, 2, 8, 4, 1, 9, 6, 7,
        3, 3, 4, 3, 1, 4, 3, 1, 3, 5, 138, 4, 2, 3, 5, 2, 4, 5, 4, 3,
        6, 2, 4, 3, 5, 3, 3, 2, 5, 1, 3, 134, 3, 0, 7, 5, 2, 5, 3, 4,
        3, 3, 4, 3, 3, 2, 4, 5, 4, 3, 7, 5, 125, 4, 4, 3, 3, 6, 6, 3,
        5, 7, 1, 2, 1, 4, 2, 5, 6, 3, 1, 2, 4, 146, 3, 3, 1, 0, 3, 1,
        3, 10, 5, 3, 5, 3, 3, 4, 5, 6, 5, 9, 5, 3, 105, 6, 1, 4, 10, 5,
        2, 6, 6, 1, 2, 3, 6, 5, 0, 4, 6, 4, 5, 1, 3, 123, 8, 2, 2, 11,
        3, 8, 1, 3, 1, 2, 4, 4, 2, 3, 4, 2, 1, 1, 3, 2, 152, 1, 2, 1,
        4, 4, 5, 1, 6, 5, 7, 2, 3, 3, 4, 5, 5, 2, 6, 6, 4, 124, 2, 2,
        2, 7, 2, 3, 4, 7, 2, 4, 4, 4, 5, 3, 2, 4, 4, 6, 7, 3, 124, 3,
        0, 10, 3, 5, 4, 7, 5, 2, 1, 7, 3, 2, 3, 5, 4, 4, 3, 2, 4, 126;
    s.distance = 0.02;

    double ll = 0.0;
    checkAgainstBpp(v, model, rates, ll);
    checkAgainstEigen(v, model, rates);
}
