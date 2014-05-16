#include "kmer_model.hpp"

#include <Bpp/Seq/Alphabet/WordAlphabet.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cassert>
#include <stdexcept>

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

KmerSubstitutionModel::KmerSubstitutionModel(const std::vector<bpp::SubstitutionModel*>& modelVector, const std::string& st) :
    bpp::AbstractParameterAliasable((st == "") ? "Word." : st),
    bpp::AbstractSubstitutionModel(bpp::AbstractWordSubstitutionModel::extractAlph(modelVector),
                                   (st == "") ? "Word." : st),
    bpp::WordSubstitutionModel(modelVector, (st == "") ? "Word." : st)
{
    KmerSubstitutionModel::updateMatrices();
}

KmerSubstitutionModel::KmerSubstitutionModel(bpp::SubstitutionModel* pmodel, unsigned int num, const std::string& st) :
    bpp::AbstractParameterAliasable((st == "") ? "Word." : st),
    bpp::AbstractSubstitutionModel(new bpp::WordAlphabet(pmodel->getAlphabet(), num),
                                   (st == "") ? "Word." : st),
    bpp::WordSubstitutionModel(pmodel, num, (st == "") ? "Word." : st)
{
    KmerSubstitutionModel::updateMatrices();
}

void KmerSubstitutionModel::updateMatrices()
{
    enableEigenDecomposition(false);
    bpp::WordSubstitutionModel::updateMatrices();

    // Fill in eigendecomposition
    Eigen::MatrixXd Q = bppToEigen(getGenerator());
    const Eigen::EigenSolver<Eigen::MatrixXd> decomp(Q);
    const Eigen::VectorXd eval = decomp.eigenvalues().real();
    const Eigen::MatrixXd evec = decomp.eigenvectors().real();
    const Eigen::MatrixXd ievec = evec.inverse();

    const size_t nbStates = getNumberOfStates();
    for(size_t i = 0; i < nbStates; i++) {
        eigenValues_[i] = eval(i);
        for(size_t j = 0; j < nbStates; j++) {
            rightEigenVectors_(i, j) = evec(i, j);
            leftEigenVectors_(i, j) = ievec(i, j);
        }
    }
}

/// Add parameters named START_END to model.
void KmerSubstitutionModel::completeMatrices()
{
    bpp::WordSubstitutionModel::completeMatrices();
    std::vector<const bpp::Matrix<double>*> subGenerators;
    for(const bpp::SubstitutionModel* m : VSubMod_) {
        subGenerators.push_back(&m->getGenerator());
    }

    size_t i, j;
    const size_t nbStates = getNumberOfStates();
    if(!dynamic_cast<const bpp::WordAlphabet*>(alphabet_)) {
        throw std::runtime_error("Expected word alphabet.");
    }

    const bpp::WordAlphabet* alpha = reinterpret_cast<const bpp::WordAlphabet*>(alphabet_);

    for (i = 0; i < nbStates; i++) {
        const std::vector<int> iState = alpha->getPositions(i);
        double sum = 0;
        for (j = 0; j < nbStates; j++) {
            if(i == j) continue;

            const std::vector<int> jState = alpha->getPositions(j);
            assert(iState.size() == jState.size() && "Position counts differ");
            size_t nDiff = 0;
            for(size_t k = 0; k < iState.size(); k++) {
                if(iState[k] != jState[k]) nDiff++;
            }
            if(nDiff == 1) {
                for(size_t k = 0; k < iState.size(); k++) {
                    if(iState[k] != jState[k]) {
                        generator_(i, j) = subGenerators[k]->operator()(iState[k], jState[k]);
                        break;
                    }
                }
                generator_(i, j) *= getFrequencies()[j];
            } else {
                generator_(i, j) = 0.0;
            }

            const std::string param = getNamespace() + alphabet_->intToChar(i) + "_" + alphabet_->intToChar(j);
            if(hasParameter(param)) {
                generator_(i, j) += getParameterValue(param);
            }

            sum += generator_(i, j);
        }
        generator_(i, i) = -sum;
    }

    // Scale
    double scale = 0;
    for(i = 0; i < nbStates; i++)
        scale += getFrequencies()[i] * generator_(i, i);
    scale = -1 / scale;

    for(i = 0; i < nbStates; i++) {
        for(j = 0; j < nbStates; j++) {
            generator_(i, j) *= scale;
        }
    }
}

}
