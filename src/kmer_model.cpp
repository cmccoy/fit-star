#include "kmer_model.hpp"

#include <Bpp/Seq/Alphabet/WordAlphabet.h>
#include <cassert>

namespace fit_star {

KmerSubstitutionModel::KmerSubstitutionModel(const std::vector<bpp::SubstitutionModel*>& modelVector, const std::string& st) :
    bpp::AbstractParameterAliasable((st == "") ? "Word." : st),
    bpp::AbstractSubstitutionModel(bpp::AbstractWordSubstitutionModel::extractAlph(modelVector),
                                   (st == "") ? "Word." : st),
    bpp::WordSubstitutionModel(modelVector, (st == "") ? "Word." : st),
    k_(modelVector.size())
{
    KmerSubstitutionModel::updateMatrices();
}

KmerSubstitutionModel::KmerSubstitutionModel(bpp::SubstitutionModel* pmodel, unsigned int num, const std::string& st) :
    bpp::AbstractParameterAliasable((st == "") ? "Word." : st),
    bpp::AbstractSubstitutionModel(new bpp::WordAlphabet(pmodel->getAlphabet(), num),
                                   (st == "") ? "Word." : st),
    bpp::WordSubstitutionModel(pmodel, num, (st == "") ? "Word." : st),
    k_(num)
{
    KmerSubstitutionModel::updateMatrices();
}

void KmerSubstitutionModel::updateMatrices()
{
    bpp::WordSubstitutionModel::updateMatrices();

    // TODO: should we do the eigendecomposition ourselves?
}

/// Add parameters names NAMESPACE.START_END to model.
void KmerSubstitutionModel::completeMatrices()
{
    std::vector<const bpp::Matrix<double>*> subGenerators;
    for(const bpp::SubstitutionModel* m : VSubMod_) {
        subGenerators.push_back(&m->getGenerator());
    }

    size_t i, j;
    const size_t nbStates = getNumberOfStates();
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

            //const std::string param = getNamespace() + alphabet_->intToChar(i) + "_" + alphabet_->intToChar(j);
            //if(hasParameter(param)) {
                //generator_(i, j) += getParameterValue(param);
            //}

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

    //bpp::WordSubstitutionModel::completeMatrices();
    //for(size_t i = 0; i < generator_.getNumberOfRows(); i++) {
    //    for(size_t j = 0; j < generator_.getNumberOfColumns(); j++) {
    //        const std::string param = getNamespace() + alphabet_->intToChar(i) + "_" + alphabet_->intToChar(j);
    //        if(hasParameter(param)) {
    //            generator_(i, j) += getParameterValue(param);
    //        }
    //    }
    //}

    // Renormalize
    //for(size_t i = 0; i < generator_.getNumberOfRows(); i++) {
    //    double scale = 0.0;
    //    for(size_t j = 0; j < generator_.getNumberOfColumns(); j++) {
    //        if(i == j)
    //            continue;
    //        scale += generator_(i, j);
    //    }
    //    scale /= - generator_(i, i);
    //    std::clog << "Scaling by: " << scale << '\n';
    //    for(size_t j = 0; j < generator_.getNumberOfColumns(); j++) {
    //        if(i == j)
    //            continue;
    //        generator_(i, j) /= scale;
    //    }
    //    scale = 0;
    //    for(size_t j = 0; i < generator_.getNumberOfColumns(); j++) {
    //        scale += generator_(i, j);
    //    }
    //    assert(std::abs(scale) <= 1e-5 && "Row sum should be 0");
    //}
}

}
