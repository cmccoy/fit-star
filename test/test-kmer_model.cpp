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

template <typename T>
void expectMatricesEqual(const bpp::Matrix<T>& x,
                         const bpp::Matrix<T>& y,
                         const T tol = 1e-4)
{
    ASSERT_EQ(x.getNumberOfRows(), y.getNumberOfRows());
    ASSERT_EQ(x.getNumberOfColumns(), y.getNumberOfColumns());

    for(size_t i = 0; i < x.getNumberOfRows(); i++) {
        for(size_t j = 0; j < x.getNumberOfColumns(); j++) {
            EXPECT_NEAR(x(i, j), y(i, j), tol) << "Matrices differ at (" << i << ", " << j << ")";
        }
    }
}

template <typename T>
void expectVectorsEqual(const std::vector<T>& x,
                        const std::vector<T>& y,
                        const T tol = 1e-4)
{
    ASSERT_EQ(x.size(), y.size());

    for(size_t i = 0; i < x.size(); i++) {
        EXPECT_NEAR(x[i], y[i], tol) << "Vectors differ at " << i;
    }
}

class GTRvs1merTest : public ::testing::Test {
protected:
    GTRvs1merTest() :
        gtrModel(&bpp::AlphabetTools::DNA_ALPHABET),
        oneWordModel(gtrModel.clone(), 1)
    {
        gtrModel.setParameterValue("a", 4);
        oneWordModel.setParameterValue("1_GTR.a", 4);
    };

  bpp::GTR gtrModel;
  KmerSubstitutionModel oneWordModel;

  void expectPMatricesEqual(const double t)
  {
    std::vector<double> gtrEval = gtrModel.getEigenValues(),
        oneWordEval = oneWordModel.getEigenValues();

    const bpp::Matrix<double>& gtrLeft = gtrModel.getRowLeftEigenVectors(),
          &wordLeft = oneWordModel.getRowLeftEigenVectors(),
          &gtrRight = gtrModel.getColumnRightEigenVectors(),
          &wordRight = oneWordModel.getColumnRightEigenVectors();

    bpp::RowMatrix<double> pGTR(gtrLeft.getNumberOfRows(), gtrLeft.getNumberOfColumns()),
        pWord(gtrLeft.getNumberOfRows(), gtrLeft.getNumberOfColumns());

    bpp::MatrixTools::mult(gtrRight, bpp::VectorTools::exp(bpp::operator*(gtrEval, t)), gtrLeft, pGTR);
    bpp::MatrixTools::mult(wordRight, bpp::VectorTools::exp(bpp::operator*(oneWordEval, t)), wordLeft, pWord);

    expectMatricesEqual(pGTR, pWord);
  }
};

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

TEST_F(GTRvs1merTest, compare_likelihood) {
    bpp::GTR gtrModel(&bpp::AlphabetTools::DNA_ALPHABET);
    KmerSubstitutionModel oneWordModel(gtrModel.clone(), 1);

    EXPECT_NEAR(testSimpleLikelihood(&gtrModel), testSimpleLikelihood(&oneWordModel), 1e-3) << "Likelihood calculations do not match.";
}


TEST_F(GTRvs1merTest, compare_generators) {
    // Extract Q, exchangeability matrices
    const bpp::Matrix<double>& gtrQ = gtrModel.getGenerator();
    const bpp::Matrix<double>& oneWordQ = oneWordModel.getGenerator();
    expectMatricesEqual(gtrQ, oneWordQ);
}

TEST_F(GTRvs1merTest, compare_exchangeability) {
    const bpp::Matrix<double>& gtrExch = gtrModel.getExchangeabilityMatrix();
    const bpp::Matrix<double>& oneWordExch = oneWordModel.getExchangeabilityMatrix();
    expectMatricesEqual(gtrExch, oneWordExch);
}

TEST_F(GTRvs1merTest, compare_eigenvalues) {
    std::vector<double> gtrEval = gtrModel.getEigenValues(),
        oneWordEval = oneWordModel.getEigenValues();
    std::sort(gtrEval.begin(), gtrEval.end());
    std::sort(oneWordEval.begin(), oneWordEval.end());
    expectVectorsEqual(gtrEval, oneWordEval);
}

TEST_F(GTRvs1merTest, compare_Psmall) {
    expectPMatricesEqual(0.001);
}

TEST_F(GTRvs1merTest, compare_P1) {
    expectPMatricesEqual(1.0);
}
