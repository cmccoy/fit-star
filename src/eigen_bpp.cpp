#include "eigen_bpp.hpp"

namespace fit_star {

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

bpp::RowMatrix<double> eigenToBpp(const Eigen::MatrixXd& m) {
    bpp::RowMatrix<double> result(m.rows(), m.cols());

    for(long i = 0; i < m.rows(); i++)
        for(long j = 0; j < m.cols(); j++)
            result(i, j) = m(i, j);
    return result;
}


std::vector<double> eigenToBpp(const Eigen::VectorXd& v) {
    std::vector<double> result(v.size());
    for(long i = 0; i < v.size(); i++)
        result[i] = v[i];
    return result;
}

}
