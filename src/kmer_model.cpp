#include "kmer_model.hpp"
#include "eigen_bpp.hpp"

#include <Bpp/Seq/Alphabet/WordAlphabet.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cassert>
#include <stdexcept>

namespace fit_star {

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
    std::vector<const bpp::Matrix<double>*> subExchangeabilities;
    for(const bpp::SubstitutionModel* m : VSubMod_) {
        subExchangeabilities.push_back(&m->getExchangeabilityMatrix());
    }

    size_t i, j;
    const size_t nbStates = getNumberOfStates();
    if(!dynamic_cast<const bpp::WordAlphabet*>(alphabet_)) {
        throw std::runtime_error("Expected word alphabet.");
    }

    const bpp::WordAlphabet* alpha = reinterpret_cast<const bpp::WordAlphabet*>(alphabet_);
    const std::vector<double> freqs = getFrequencies();

    for (i = 0; i < nbStates; i++) {
        // Find which states changed
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
                        generator_(i, j) = subExchangeabilities[k]->operator()(iState[k], jState[k]);
                        break;
                    }
                }

                //const std::string param = alphabet_->intToChar(i < j ? i : j) + "_" + alphabet_->intToChar(i < j ? j : i);
                const std::string param = alphabet_->intToChar(i < j ? i : j) + "_" + alphabet_->intToChar(i < j ? j : i);
                if(hasParameter(param)) {
                    generator_(i, j) += getParameterValue(param);
                }

                generator_(i, j) *= freqs[j];
            } else {
                generator_(i, j) = 0.0;
            }


            sum += generator_(i, j);
        }
        generator_(i, i) = -sum;
    }

    // Scale
    double scale = 0;
    for(i = 0; i < nbStates; i++)
        scale += freqs[i] * generator_(i, i);
    scale = -1 / scale;

    for(i = 0; i < nbStates; i++) {
        for(j = 0; j < nbStates; j++) {
            generator_(i, j) *= scale;
        }
    }
}

void KmerSubstitutionModel::addParameter(bpp::Parameter *p)
{
    this->addParameter_(p);
}

}
