#include <algorithm>
#include <vector>
#include <iostream>
#include "gtest/gtest.h"
#include "kmer_model.hpp"

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

TEST(KmerModel, silly) {
    //bpp::GTR gtr(&bpp::AlphabetTools::DNA_ALPHABET);
    //std::vector<bpp::SubstitutionModel*> v (2, new bpp::GTR(&bpp::AlphabetTools::DNA_ALPHABET));
    //KmerSubstitutionModel model(v);
    //bpp::WordSubstitutionModel model(v);
    //bpp::GTR model(&bpp::AlphabetTools::DNA_ALPHABET);
    KmerSubstitutionModel model(new bpp::GTR(&bpp::AlphabetTools::DNA_ALPHABET), 2);
    const auto& m = model.getPij_t(0.1);

    const bpp::Matrix<double>& gen = model.getGenerator();
    const bpp::Matrix<double>& exch = model.getExchangeabilityMatrix();
    //std::clog << "freq\t" << bpp::VectorTools::paste(model.getFrequencies()) << '\n';
    //std::clog << "gen\t";
    //bpp::MatrixTools::print(gen, std::clog);
    //std::clog << "exch\t";
    //bpp::MatrixTools::print(exch, std::clog);
    //std::clog << "p0.1\t";
    //bpp::MatrixTools::print(m, std::clog);
    //std::clog << "rowLeftEigen\t";
    //bpp::MatrixTools::print(model.getRowLeftEigenVectors(), std::clog);
}

double testHelper(bpp::SubstitutionModel* model) {
    bpp::BasicSequence first("A", "ACGGTACCGTAAC", model->getAlphabet()),
                      second("B", "ACTGTGGCGTCAT", model->getAlphabet());
    bpp::VectorSiteContainer sites(model->getAlphabet());
    sites.addSequence(first);
    sites.addSequence(second);

    bpp::Node *root = new bpp::Node(0),
        *c1 = new bpp::Node(1, sites.getSequence(0).getName()),
        *c2 = new bpp::Node(2, sites.getSequence(1).getName());
    root->addSon(c1);
    root->addSon(c2);
    c1->setDistanceToFather(0.04);
    c2->setDistanceToFather(0.10);
    bpp::TreeTemplate<bpp::Node> tree(root);

    bpp::RateDistributionFactory fac(1);
    std::unique_ptr<bpp::DiscreteDistribution> rates(fac.createDiscreteDistribution("Constant"));

    bpp::RHomogeneousTreeLikelihood calc(tree, sites, model, rates.get(), true, false);
    calc.initialize();
    calc.computeTreeLikelihood();

    return calc.getLogLikelihood();
}

TEST(KmerModel, single_nucleotide_equals_gtr) {
    bpp::GTR gtrModel(&bpp::AlphabetTools::DNA_ALPHABET);
    KmerSubstitutionModel oneWordModel(gtrModel.clone(), 1);

    EXPECT_NEAR(testHelper(&gtrModel), testHelper(&oneWordModel), 1e-3) << "Likelihood calculations do not match.";
}