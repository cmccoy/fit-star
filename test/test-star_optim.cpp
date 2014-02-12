#include <algorithm>
#include <vector>
#include <iostream>
#include "gtest/gtest.h"
#include "star_tree_optimizer.hpp"
#include "aligned_pair.hpp"
#include "log_tricks.hpp"

#include <libhmsbeagle/beagle.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <Bpp/Phyl/Distance/DistanceEstimation.h>
#include <Bpp/Phyl/Likelihood/DiscreteRatesAcrossSitesTreeLikelihood.h>
#include <Bpp/Phyl/Likelihood/RHomogeneousTreeLikelihood.h>
#include <Bpp/Phyl/Model.all>
#include <Bpp/Phyl/TreeTemplate.h>
#include <Bpp/Seq/Alphabet/DNA.h>
#include <Bpp/Seq/Container/VectorSiteContainer.h>
#include <Bpp/Seq/Sequence.h>

using namespace fit_star;

bpp::DNA DNA;

bpp::VectorSiteContainer createSites(const AlignedPair& sequence)
{
    std::vector<std::string> names {"ref", "qry" };
    std::vector<std::string> seqs(2);
    const std::string bases = "ACGT";

    for(size_t i = 0; i < bases.size(); i++) {
        for(size_t j = 0; j < bases.size(); j++) {
            int count = static_cast<int>(sequence.partitions[0].substitutions(i, j));
            for(int k = 0; k < count; k++) {
                seqs[0] += bases[i];
                seqs[1] += bases[j];
            }
        }
    }

    bpp::VectorSiteContainer result(&DNA);
    for(size_t i = 0; i < 2; i++)
        result.addSequence(bpp::BasicSequence(names[i], seqs[i], &DNA));

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

    VectorSiteContainer sites = createSites(sequences[0]);

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

Eigen::MatrixXd bppToEigen(const bpp::Matrix<double>& m) {
    Eigen::MatrixXd result(m.getNumberOfRows(), m.getNumberOfColumns());
    for(size_t i = 0; i < m.getNumberOfRows(); i++)
        for(size_t j = 0; j < m.getNumberOfColumns(); j++)
            result(i, j) = m(i, j);
    return result;
}

Eigen::VectorXd bppToEigen(const std::vector<double>& v) {
    Eigen::VectorXd result(v.size());
    for(size_t i = 0; i < v.size(); i++)
        result[i] = v[i];
    return result;
}

// Equivalent of checkAgainstBpp, but using fixed root
void checkAgainstEigen(std::vector<AlignedPair>& sequences,
                       std::unique_ptr<bpp::SubstitutionModel>& model,
                       std::unique_ptr<bpp::DiscreteDistribution>& rateDist)
{
    using namespace Eigen;

    ASSERT_EQ(1, sequences.size());
    ASSERT_EQ(1, sequences[0].partitions.size());

    Matrix4d Q;
    const bpp::Matrix<double>& gen = model->getGenerator();
    for(size_t i = 0; i < 4; i++)
        for(size_t j = 0; j < 4; j++)
            Q(i, j) = gen(i, j);

    const EigenSolver<Matrix4d> decomp(Q);
    const Vector4d eval = decomp.eigenvalues().real();
    const Matrix4d evec = decomp.eigenvectors().real();
    const Matrix4d ievec = evec.inverse();

    const Vector4d bppEval = bppToEigen(model->getEigenValues());
    for(size_t i = 0; i < 4; i++) {
        EXPECT_NEAR(bppEval[i], eval[i], 1e-5);
    }

    const Eigen::MatrixXd bppIEvec = bppToEigen(model->getRowLeftEigenVectors());
    const Eigen::MatrixXd bppEvec = bppToEigen(model->getColumnRightEigenVectors());
    std::cout << "Eigen:\n" << ievec << '\n' 
        << eval << '\n'
        << evec << '\n';
    std::cout << "Bpp:\n" << bppIEvec << '\n' 
        << bppEval << '\n'
        << bppEvec  << '\n';
    for(size_t i = 0; i < 4; i++) {
        for(size_t j = 0; j < 4; j++) {
            EXPECT_NEAR(bppEvec(i, j), evec(i, j), 1e-5);
            EXPECT_NEAR(bppIEvec(i, j), ievec(i, j), 1e-5);
        }
    }

    const std::vector<double> rates = rateDist->getCategories();
    const std::vector<double> rateProbs = rateDist->getProbabilities();
    std::vector<Matrix4d> pMatrices(rates.size());
    for(size_t i = 0; i < rates.size(); i++) {
        const Vector4d lambda = (Array4d(eval) * sequences[0].distance * rates[i]).exp();
        pMatrices[i] = ievec * lambda.asDiagonal() * evec;
    }
    auto f = [](const double d) { return std::log(d); };
    for(size_t i = 0; i < rates.size(); i++) {
        Eigen::Matrix4d r;
        r.fill(std::log(rateProbs[i]));
        pMatrices[i] = pMatrices[i].unaryExpr(f) + r;
    }

    // Sum over mixture
    Matrix4d p = pMatrices[0];
    for(size_t i = 1; i < rates.size(); i++) {
        const Matrix4d& pi = pMatrices[i];
        for(size_t j = 0; j < 4; j++) {
            for(size_t k = 0; k < 4; k++) {
                p(j, k) = fit_star::logSum(p(j, k), pi(j, k));
            }
        }
    }

    double expectedLL = p.cwiseProduct(sequences[0].partitions[0].substitutions).sum();
    double actualLL = starLogLike(sequences, model, rateDist, true);

    EXPECT_NEAR(expectedLL, actualLL, 0.5);
}

TEST(FitStar, simple_jc) {
    bpp::DNA dna;
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::GTR(&dna));
    bpp::RateDistributionFactory fac(4);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Constant"));

    std::vector<AlignedPair> v { AlignedPair() };
    AlignedPair& s = v.front();

    s.partitions.resize(1);
    Eigen::Matrix4d& m = s.partitions[0].substitutions;
    m.fill(0.0);
    m.diagonal() = Eigen::Vector4d::Constant(1.0);

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
    bpp::DNA dna;
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::GTR(&dna));
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
