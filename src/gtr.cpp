#include "gtr.hpp"
#include "sequence.hpp"
#include <boost/math/tools/minima.hpp>

using Eigen::Array4d;
using Eigen::Matrix4d;
using Eigen::Vector4d;

namespace gtr {

// GTRModel
Matrix4d GTRModel::buildPMatrix(const double t) const
{
    const Matrix4d& v = decomp.eigenvectors().real();
    Vector4d lambda = (Array4d(decomp.eigenvalues().real()) * t).exp();
    return v * Eigen::DiagonalMatrix<double, 4, 4>(lambda) * v.inverse();
}

double GTRModel::logLikelihood(const Sequence& s) const
{
    const Matrix4d p = buildPMatrix(s.distance);

    auto f = [](const double d) { return std::log(d); };
    const Matrix4d log_p = p.unaryExpr(f);

    return log_p.cwiseProduct(s.transitions).sum();
}

// GTRParameters

GTRParameters::GTRParameters()
{
    params.fill(1);
    pi.fill(0.25);
};

Matrix4d GTRParameters::buildQMatrix() const
{
    const double pi1 = pi[0], pi2 = pi[1], pi3 = pi[2], pi4 = pi[3];
    const double x1 = params[0], x2 = params[1], x3 = params[2],
          x4 = params[3], x5 = params[4], x6 = params[5];
    Matrix4d result;
    result << - (x1 + x2 + x3), (pi1 * x1 / pi2), (pi1 * x2 / pi3), pi1 * x3 / pi4,
           x1, -(pi1 * x1 / pi2 + x4 + x5), (pi2 * x4 / pi3), (pi2 * x5 / pi4),
           x2, x4, -(pi1 * x2 / pi3 + pi2 * x4 / pi3 + x6), pi3 * x6 / pi4,
           x3, x5, x6, -(pi1 * x3 / pi4 + pi2 * x5 / pi4 + pi3 * x6 / pi4);
    return result;
};

GTRModel GTRParameters::buildModel() const
{
    return GTRModel(buildQMatrix());
}

// Functions
double star_likelihood(const GTRModel& model,
                       const std::vector<Sequence>& sequences)
{
    double result = 0.0;

#pragma omp parallel for reduction(+:result)
    for(size_t i = 0; i < sequences.size(); i++) {
        result += model.logLikelihood(sequences[i]);
    }
    return result;
}

void estimate_branch_lengths(const GTRModel& model,
                             std::vector<Sequence>& sequences)
{
#pragma omp parallel for
    for(size_t i = 0; i < sequences.size(); i++) {
        Sequence& s = sequences[i];
        auto f = [&model, &s](const double d) {
            s.distance = d;
            const double result = -model.logLikelihood(s);
            return result;
        };
        boost::uintmax_t max_iter = 100;
        std::pair<double, double> res =
            boost::math::tools::brent_find_minima(f, 1e-9, 1.0, 50, max_iter);
        s.distance = res.first;
    }
}

Eigen::Vector4d count_base_frequences(const std::vector<Sequence>& sequences)
{
    Eigen::Vector4d result;
    result.fill(1);

    for(const Sequence& s : sequences)
        result += s.transitions.colwise().sum();

    result /= result.sum();
    return result;
}

void empirical_model(const std::vector<Sequence>& sequences,
                     gtr::GTRParameters& model)
{
    model.pi = count_base_frequences(sequences);

    Matrix4d result;
    result.fill(0);

    for(const Sequence& s : sequences) {
        result += s.transitions;
    }

    model.params << result(0, 1), result(0, 2), result(0, 3),
                    result(1, 2), result(1, 3), result(2, 3);
    model.params /= model.params[5];
}

double optimize_parameter(const std::vector<Sequence>& sequences,
                        const size_t index,
                        gtr::GTRParameters& params)
{
    auto f = [&sequences, &index, &params] (const double d) {
        params.params[index] = d;
        const GTRModel model = params.buildModel();
        return -star_likelihood(model, sequences);
    };

    boost::uintmax_t max_iter = 50;
    std::pair<double, double> result = boost::math::tools::brent_find_minima(f, 1e-9, 20.0, 50, max_iter);

    params.params[index] = result.first;
    params.params /= params.params[5];

    return -result.second;
}

// TODO: tolerance, no magic numbers, no printing.
void optimize(gtr::GTRParameters& params,
              std::vector<Sequence>& sequences)
{
    double last_log_like = star_likelihood(params.buildModel(), sequences);

    for(size_t iter = 0; iter < 20; iter++) {
        bool any_improved = false;
        for(size_t param_index = 0; param_index < 7; param_index++) {
            if(param_index == 6)
                estimate_branch_lengths(params.buildModel(),
                                        sequences);
            else
                optimize_parameter(sequences, param_index, params);

            const double log_like = star_likelihood(params.buildModel(), sequences);

            std::cerr << "p=" << params.params.transpose() << '\n';
            std::cerr << "iteration " << iter << " parameter " << param_index << ": " << last_log_like << " ->\t" << log_like << '\t' << log_like - last_log_like << '\n';
            std::cerr.flush();

            if(std::abs(log_like - last_log_like) > 1e-4)
                any_improved = true;
            last_log_like = log_like;
        }
        if(!any_improved)
            break;
    }
}

}
