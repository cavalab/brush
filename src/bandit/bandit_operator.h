#ifndef BANDIT_OPERATOR_H
#define BANDIT_OPERATOR_H

// virtual class. selection must be made with static methods

#include "../init.h"
#include "../data/data.h"
#include "../types.h"
#include "../params.h"

namespace Brush {
namespace MAB {
    
template<typename T>
class BanditOperator
{
public:
    BanditOperator(vector<T> arms);
    BanditOperator(map<T, float> arms_probs);
    ~BanditOperator() {};

    virtual std::map<T, float> sample_probs(bool update);

    virtual void update(T arm, float reward);
protected:    
    std::map<T, float> probabilities;
};

} // MAB
} // Brush

#endif // BANDIT_OPERATOR_H
