#include "bandit_operator.h"

namespace Brush {
namespace MAB {

template<typename T>
BanditOperator<T>::BanditOperator(vector<T> arms)
{
    // Initialize the map with the keys and uniform distributed values
    float uniform_prob = 1.0 / arms.size();

    this->probabilities = std::map<T, float>();
    for (const T& arm : arms) {
        this->probabilities[arm] = uniform_prob;
    }
}

template<typename T>
BanditOperator<T>::BanditOperator(map<T, float> arms_probs)
{
    this->probabilities = std::map<T, float>();
    for (const auto& arm_prob : arms_probs) {
        this->probabilities[arm_prob.first] = arm_prob.second;
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
    return this->probabilities;
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