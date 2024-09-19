#ifndef BANDIT_OPERATOR_H
#define BANDIT_OPERATOR_H

#include "../init.h"
#include "../data/data.h"
#include "../types.h"
#include "../params.h"
#include "../program/tree_node.h"
#include "../ind/fitness.h"

namespace Brush {
namespace MAB {
    
/**
 * @brief A **virtual** class representing a bandit operator.
 * 
 * This class provides functionality for sampling probabilities and updating rewards for different arms. The bandit should somehow behave stocasticaly when called twice in a row without being rewarded, because some mutations may require multiple sampling from the search space. Also, the bandit should have a functionality to return smapling probabilities, instead of just the chosen arm.
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
     * @brief Chooses an arm based on the given tree and fitness. Should call sample_probs internally.
     * 
     * @param tree The tree structure used to choose the arm.
     * @param f The fitness value used to influence the choice.
     * @return The arm with highest probability.
     */
    virtual T choose(tree<Node>& tree, Fitness& f);

    /**
     * @brief Updates the reward for a specific arm.
     * 
     * @param arm The arm for which to update the reward.
     * @param reward The reward value.
     */
    virtual void update(T arm, float reward, tree<Node>* tree=nullptr, Fitness* f=nullptr); // TODO: this should not have a default value in the future
protected:    
    std::map<T, float> probabilities;
};

} // MAB
} // Brush

#endif // BANDIT_OPERATOR_H
