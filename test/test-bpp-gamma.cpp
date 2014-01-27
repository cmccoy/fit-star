#include <Bpp/Phyl/Model/RateDistributionFactory.h> 
#include "gtest/gtest.h"
#include <algorithm>
#include <memory>

class DiscreteGammaTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    bpp::RateDistributionFactory fac(4);
    dist.reset(fac.createDiscreteDistribution("Gamma"));
  }

  void TestExpectedRate() const {
    const std::vector<double> rates = dist->getCategories(),
        probs = dist->getProbabilities();
    const double expectation = std::inner_product(rates.begin(), rates.end(), probs.begin(), 0.0);
    ASSERT_NEAR(1.0, expectation, 1e-5);
  }

  // virtual void TearDown() {}
  std::unique_ptr<bpp::DiscreteDistribution> dist;
};

TEST_F(DiscreteGammaTest, alpha1) {
    ASSERT_EQ(dist->getIndependentParameters().size(), 1);
    dist->setParameterValue("alpha", 0.1);
    TestExpectedRate();
}

TEST_F(DiscreteGammaTest, alpha2) {
    ASSERT_EQ(dist->getIndependentParameters().size(), 1);
    dist->setParameterValue("alpha", 2);
    TestExpectedRate();
}

TEST_F(DiscreteGammaTest, alpha3) {
    ASSERT_EQ(dist->getIndependentParameters().size(), 1);
    dist->setParameterValue("alpha", 0.08);
    TestExpectedRate();
}

TEST_F(DiscreteGammaTest, alpha4) {
    ASSERT_EQ(dist->getIndependentParameters().size(), 1);
    dist->setParameterValue("alpha", 0.05);
    TestExpectedRate();
}

TEST_F(DiscreteGammaTest, alpha5) {
    ASSERT_EQ(dist->getIndependentParameters().size(), 1);
    dist->setParameterValue("alpha", 100);
    TestExpectedRate();
}
