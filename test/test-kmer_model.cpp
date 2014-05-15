#include <algorithm>
#include <vector>
#include <iostream>
#include "gtest/gtest.h"
#include "kmer_model.hpp"

#include <libhmsbeagle/beagle.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <Bpp/Numeric/VectorTools.h>
#include <Bpp/Phyl/Distance/DistanceEstimation.h>
#include <Bpp/Phyl/Likelihood/DiscreteRatesAcrossSitesTreeLikelihood.h>
#include <Bpp/Phyl/Likelihood/RHomogeneousTreeLikelihood.h>
#include <Bpp/Phyl/Model.all>
#include <Bpp/Phyl/TreeTemplate.h>
#include <Bpp/Seq/Alphabet/AlphabetTools.h>
#include <Bpp/Seq/Container/VectorSiteContainer.h>
#include <Bpp/Seq/Sequence.h>

using namespace fit_star;

double testSimpleLikelihood(bpp::SubstitutionModel* model) {
    bpp::BasicSequence first("A", "ACGGTACCGTAAC", model->getAlphabet()),
                      second("B", "ACTTTGGCGTCAT", model->getAlphabet());
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

    EXPECT_NEAR(testSimpleLikelihood(&gtrModel), testSimpleLikelihood(&oneWordModel), 1e-3) << "Likelihood calculations do not match.";

    // Extract Q, exchangeability matrices
    const bpp::Matrix<double>& gtrQ = gtrModel.getGenerator();
    const bpp::Matrix<double>& oneWordQ = oneWordModel.getGenerator();
    const bpp::Matrix<double>& gtrExch = gtrModel.getExchangeabilityMatrix();
    const bpp::Matrix<double>& oneWordExch = oneWordModel.getExchangeabilityMatrix();

    EXPECT_EQ(gtrQ.getNumberOfRows(), oneWordQ.getNumberOfRows());
    EXPECT_EQ(gtrQ.getNumberOfColumns(), oneWordQ.getNumberOfColumns());
    for(size_t i = 0; i < gtrQ.getNumberOfRows(); i++) {
        for(size_t j = 0; j < gtrQ.getNumberOfColumns(); j++) {
            EXPECT_NEAR(gtrQ(i, j), oneWordQ(i, j), 1e-4);
            EXPECT_NEAR(gtrExch(i, j), oneWordExch(i, j), 1e-4);
        }
    }

    // Next: eigenvectors, eigenvalues
    std::vector<double> gtrEval = gtrModel.getEigenValues(),
        oneWordEval = oneWordModel.getEigenValues();

    const bpp::Matrix<double>& gtrLeft = gtrModel.getRowLeftEigenVectors(),
          &wordLeft = oneWordModel.getRowLeftEigenVectors(),
          &gtrRight = gtrModel.getColumnRightEigenVectors(),
          &wordRight = oneWordModel.getColumnRightEigenVectors();

    bpp::RowMatrix<double> pGTR(gtrLeft.getNumberOfRows(), gtrLeft.getNumberOfColumns()),
        pWord(gtrLeft.getNumberOfRows(), gtrLeft.getNumberOfColumns());

    bpp::MatrixTools::mult(gtrRight, bpp::VectorTools::exp(bpp::operator*(gtrEval, 0.1)), gtrLeft, pGTR);
    bpp::MatrixTools::mult(wordRight, bpp::VectorTools::exp(bpp::operator*(oneWordEval, 0.1)), wordLeft, pWord);

    for(size_t i = 0; i < gtrQ.getNumberOfRows(); i++) {
        for(size_t j = 0; j < gtrQ.getNumberOfColumns(); j++) {
            EXPECT_NEAR(pGTR(i, j), pWord(i, j), 1e-4);
        }
    }

    bpp::MatrixTools::mult(gtrRight, bpp::VectorTools::exp(bpp::operator*(gtrEval, 0.001)), gtrLeft, pGTR);
    bpp::MatrixTools::mult(wordRight, bpp::VectorTools::exp(bpp::operator*(oneWordEval, 0.001)), wordLeft, pWord);

    for(size_t i = 0; i < gtrQ.getNumberOfRows(); i++) {
        for(size_t j = 0; j < gtrQ.getNumberOfColumns(); j++) {
            EXPECT_NEAR(pGTR(i, j), pWord(i, j), 1e-4);
        }
    }

    std::sort(gtrEval.begin(), gtrEval.end());
    std::sort(oneWordEval.begin(), oneWordEval.end());
    for(size_t i = 0; i < gtrEval.size(); i++)
        EXPECT_NEAR(gtrEval[i], oneWordEval[i], 1e-4);
}
