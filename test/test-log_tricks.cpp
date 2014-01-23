#include "log_tricks.hpp"
#include "gtest/gtest.h"

TEST(logSum, both_large) {
    const double x = 0.1;
    const double y = 0.1;
    ASSERT_NEAR(std::log(0.2), logSum(std::log(x), std::log(y)), 1e-6);
}

TEST(logSum, one_large) {
    const double x = std::log(0.1);
    const double y = -500000;
    ASSERT_NEAR(x, logSum(x, y), 1e-6);
}
