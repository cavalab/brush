#ifndef BANDIT_OPERATOR_H
#define BANDIT_OPERATOR_H

// virtual class. selection must be made with static methods

#include "../init.h"
#include "../data/data.h"
#include "../types.h"
#include "../params.h"

namespace Brush {
namespace MAB {
    
class BanditOperator
{
public:
    std::string name; 
    vector<float> probabilities;

    virtual ~BanditOperator() {}

    // TODO: rename to sample_new_probs and make it taking no arguments
    virtual std::vector<float> sample_probs(bool update);

    virtual void update_with_reward(std::vector<float> rewards);
};;

} // MAB
} // Brush

#endif // BANDIT_OPERATOR_H
