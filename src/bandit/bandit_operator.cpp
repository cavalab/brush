#include "bandit_operator.h"

namespace Brush {
namespace MAB {

template<typename T>
BanditOperator<T>::BanditOperator(vector<T> arms)
{
    // Initialize the map with the keys and uniform distributed values
    float uniform_prob = 1.0 / arms.size();

    for (const T& arm : arms) {
        probabilities[arm] = uniform_prob;
    }
}

template<typename T>
BanditOperator<T>::BanditOperator(map<T, float> arms_probs)
{
    for (const auto& arm_prob : arms_probs) {
        probabilities[arm_prob.first] = arm_prob.second;
    }
}

template<typename T>
std::map<T, float> BanditOperator<T>::sample_probs(bool update)
{
    // TODO: Implement the logic for sampling probabilities
    // based on the bandit operator's strategy

    // Throw an error if the select() operation is undefined
    HANDLE_ERROR_THROW("Undefined bandit sample_probs() operation");

    // Return an empty vector
    return std::map<T, float>();
}

template<typename T>
void BanditOperator<T>::update(T arm, float reward)
{
    // TODO: Implement the logic for updating the bandit operator's internal state
    // based on the received rewards

    // Throw an error if the update operation is undefined
    HANDLE_ERROR_THROW("Undefined bandit update_with_reward() operation");
}

} // MAB
} // Brush