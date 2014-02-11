#include <algorithm>
#include <vector>
#include <iostream>
#include "gtest/gtest.h"
#include "star_tree_optimizer.hpp"
#include "aligned_pair.hpp"

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
    for(int i = 0; i < 2; i++)
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

// Equivalent of checkAgainstBpp, but using fixed root
void checkAgainstEigen(std::vector<AlignedPair>& sequences,
                       std::unique_ptr<bpp::SubstitutionModel>& model,
                       std::unique_ptr<bpp::DiscreteDistribution>& rates)
{
    using namespace Eigen;

    ASSERT_EQ(1, sequences.size());
    ASSERT_EQ(1, sequences[0].partitions.size());
    ASSERT_EQ(1, rates->getNumberOfCategories());

    Matrix4d Q;
    auto& gen = model->getGenerator();
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            Q(i, j) = gen(i, j);

    const SelfAdjointEigenSolver<Matrix4d> decomp(Q);
    const Vector4d lambda = (Array4d(decomp.eigenvalues().real()) * sequences[0].distance).exp();
    const Matrix4d P = decomp.eigenvectors() * lambda.asDiagonal() * decomp.eigenvectors().inverse();
    auto f = [](const double d) { return std::log(d); };
    const Matrix4d logP = P.unaryExpr(f);

    double expectedLL = logP.cwiseProduct(sequences[0].partitions[0].substitutions).sum();
    double actualLL = starLogLike(sequences, model, rates, true);

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
    // No eigen test here - only single category gamma supported
}
