#include "bandit_operator.h"

namespace Brush {
namespace MAB {

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
void BanditOperator<T>::update_with_reward(std::vector<float> rewards)
{
    // TODO: Implement the logic for updating the bandit operator's internal state
    // based on the received rewards

    // Throw an error if the update operation is undefined
    HANDLE_ERROR_THROW("Undefined bandit update_with_reward() operation");
}

} // MAB
} // Brush