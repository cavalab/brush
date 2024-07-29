#ifndef BANDIT_OPERATOR_H
#define BANDIT_OPERATOR_H

#include "../init.h"
#include "../data/data.h"
#include "../types.h"
#include "../params.h"

namespace Brush {
namespace MAB {
    
/**
 * @brief A **virtual** class representing a bandit operator.
 * 
 * This class provides functionality for sampling probabilities and updating rewards for different arms.
 * 
 * @tparam T The type of the arms.
 */
template<typename T>
class BanditOperator
{
public:
    /**
     * @brief Constructs a BanditOperator object with a vector of arms.
     * 
     * @param arms The vector of arms.
     */
    BanditOperator(vector<T> arms);
    
    /**
     * @brief Constructs a BanditOperator object with a map of arms and their probabilities.
     * 
     * @param arms_probs The map of arms and their probabilities.
     */
    BanditOperator(map<T, float> arms_probs);
    
    ~BanditOperator() {};

    /**
     * @brief Samples the probabilities of the arms.
     * 
     * @param update A boolean indicating whether to update the probabilities.
     * @return A map of arms and their probabilities.
     */
    virtual std::map<T, float> sample_probs(bool update);

    /**
     * @brief Updates the reward for a specific arm.
     * 
     * @param arm The arm for which to update the reward.
     * @param reward The reward value.
     */
    virtual void update(T arm, float reward);
protected:    
    std::map<T, float> probabilities;
};

} // MAB
} // Brush

#endif // BANDIT_OPERATOR_H
