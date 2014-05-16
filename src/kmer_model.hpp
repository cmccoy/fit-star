#ifndef FIT_STAR_KMER_MODEL_H
#define FIT_STAR_KMER_MODEL_H

#include <Bpp/Phyl/Model/WordSubstitutionModel.h>

namespace fit_star {

class KmerSubstitutionModel : public bpp::WordSubstitutionModel
{
public:
    KmerSubstitutionModel() = delete;
    KmerSubstitutionModel(const std::vector<bpp::SubstitutionModel*>& modelVector, const std::string& st = "");
    KmerSubstitutionModel(bpp::SubstitutionModel* pmodel, unsigned int num, const std::string& st = "");
    virtual ~KmerSubstitutionModel() {}

    KmerSubstitutionModel(const KmerSubstitutionModel&) = delete;
    KmerSubstitutionModel(KmerSubstitutionModel&&) = delete;
    KmerSubstitutionModel& operator=(const KmerSubstitutionModel&) = delete;
    KmerSubstitutionModel& operator=(KmerSubstitutionModel&&) = delete;

    void updateMatrices() override;

protected:
    void completeMatrices() override;
};
}

#endif
