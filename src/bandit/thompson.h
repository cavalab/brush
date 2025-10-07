#include "bandit_operator.h"

#include "../util/utils.h" // to use random generator

#ifndef THOMPSON_H
#define THOMPSON_H

namespace Brush {
namespace MAB {

class ThompsonSamplingBandit : public BanditOperator
{
public:
    ThompsonSamplingBandit(vector<string> arms, bool dynamic=false);
    ThompsonSamplingBandit(map<string, float> arms_probs, bool dynamic=false);
    ~ThompsonSamplingBandit(){};

    std::map<string, float> sample_probs(bool update);
    string choose();
    void update(string arm, float reward);
private:
    bool dynamic_update;
    float C = 250;

    std::map<string, float> alphas;
    std::map<string, float> betas;
};

} // MAB
} // Brush

#endif // THOMPSON_H
