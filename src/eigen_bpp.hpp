#ifndef FIT_STAR_EIGEN_BPP_H
#define FIT_STAR_EIGEN_BPP_H

#include <Bpp/Numeric/Matrix/Matrix.h>
#include <Eigen/Dense>

#include <vector>

namespace fit_star {

Eigen::MatrixXd bppToEigen(const bpp::Matrix<double>&);
Eigen::VectorXd bppToEigen(const std::vector<double>&);

bpp::RowMatrix<double> eigenToBpp(const Eigen::MatrixXd&);
std::vector<double> eigenToBpp(const Eigen::VectorXd&);

}

#endif
