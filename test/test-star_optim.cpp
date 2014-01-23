#include <algorithm>
#include <vector>
#include <iostream>
#include "gtest/gtest.h"
#include "star_tree_optimizer.hpp"
#include "aligned_pair.hpp"

#include "libhmsbeagle/beagle.h"

#include "Eigen/Core"

#include <Bpp/Phyl/Distance/DistanceEstimation.h>
#include <Bpp/Phyl/Likelihood/DiscreteRatesAcrossSitesTreeLikelihood.h>
#include <Bpp/Phyl/Likelihood/RHomogeneousTreeLikelihood.h>
#include <Bpp/Phyl/Model.all>
#include <Bpp/Phyl/TreeTemplate.h>
#include <Bpp/Seq/Alphabet/DNA.h>
#include <Bpp/Seq/Container/VectorSiteContainer.h>
#include <Bpp/Seq/Sequence.h>

bpp::DNA DNA;

bpp::VectorSiteContainer createSites(const AlignedPair& sequence)
{
    std::vector<std::string> names {"ref", "qry" };
    std::vector<std::string> seqs(2);
    const std::string bases = "ACGT";

    for(size_t i = 0; i < bases.size(); i++) {
        for(size_t j = 0; j < bases.size(); j++) {
            int count = static_cast<int>(sequence.substitutions[0](i, j));
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

void checkAgainstBpp(std::vector<AlignedPair>& sequences,
                     std::unique_ptr<bpp::SubstitutionModel>& model,
                     std::unique_ptr<bpp::DiscreteDistribution>& rates,
                     double& logL)
{
    using namespace bpp;
    ASSERT_EQ(1, sequences.size());

    std::vector<std::unique_ptr<bpp::SubstitutionModel>> models;
    models.emplace_back(model->clone());
    std::vector<std::unique_ptr<bpp::DiscreteDistribution>> rateDists;
    rateDists.emplace_back(rates->clone());

    star_optim::StarTreeOptimizer optimizer(models, rateDists, sequences);
    const size_t partition = 0;
    const double starLL = optimizer.starLikelihood(partition);

    VectorSiteContainer sites = createSites(sequences[0]);

    Node *root = new Node(0),
         *c1 = new Node(1, sites.getSequence(0).getName()),
         *c2 = new Node(2, sites.getSequence(1).getName());
    root->addSon(c1);
    root->addSon(c2);
    c1->setDistanceToFather(sequences[0].distance / 2);
    c2->setDistanceToFather(sequences[0].distance / 2);
    TreeTemplate<Node> tree(root);

    RHomogeneousTreeLikelihood calc(tree, sites, model.get(), rateDists[0].get(), true, false);
    calc.initialize();
    calc.computeTreeLikelihood();

    const double bppLL = calc.getLogLikelihood();
    EXPECT_NEAR(bppLL, starLL, 1e-3) << "Likelihood calculations do not match.";
    logL = starLL;
}

TEST(GTR, simple_jc) {
    bpp::DNA dna;
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::GTR(&dna));
    bpp::RateDistributionFactory fac(4);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Constant"));

    std::vector<AlignedPair> v { AlignedPair() };
    AlignedPair& s = v.front();

    s.substitutions.resize(1);
    Eigen::Matrix4d& m = s.substitutions[0];
    m.fill(0.0);
    m.diagonal() = Eigen::Vector4d::Constant(1.0);

    s.distance = 0.02;

    double ll = 0.0;
    checkAgainstBpp(v, model, rates, ll);
    const double expll = -5.62490959465585; // from bppml
    EXPECT_NEAR(expll, ll, 1e-5);
}

TEST(GTR, known_distance) {
    bpp::DNA dna;
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::GTR(&dna));
    bpp::RateDistributionFactory fac(4);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Constant"));

    std::vector<AlignedPair> v { AlignedPair() };

    AlignedPair& s = v.front();
    s.substitutions.resize(1);
    s.substitutions[0] <<
        94, 3, 2, 1,
        2, 95, 2, 1,
        2, 4, 89, 5,
        1, 3, 2, 94;
    s.distance = 0.02;

    double ll = 0.0;
    checkAgainstBpp(v, model, rates, ll);
}


TEST(GTR, gamma_variation) {
    bpp::DNA dna;
    std::unique_ptr<bpp::SubstitutionModel> model(new bpp::GTR(&dna));
    bpp::RateDistributionFactory fac(4);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Gamma"));
    rates->setParameterValue("alpha", 1.2);

    EXPECT_NEAR(1.2, rates->getParameterValue("beta"), 1e-6);

    model->setParameterValue("a", 0.5);
    model->setParameterValue("theta", 0.6);
    model->setParameterValue("theta1", 0.4);

    std::vector<AlignedPair> v { AlignedPair() };

    AlignedPair& s = v.front();

    s.substitutions.resize(1);
    s.substitutions[0] <<
        94, 3, 2, 1,
        2, 95, 2, 1,
        2, 4, 89, 5,
        1, 3, 2, 94;
    s.distance = 0.02;

    double ll = 0.0;
    checkAgainstBpp(v, model, rates, ll);
}
