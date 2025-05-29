#include "bandit_operator.h"

namespace Brush {
namespace MAB {

BanditOperator::BanditOperator(vector<string> arms)
{
    // Initialize the map with the keys and uniform distributed values
    float uniform_prob = 1.0 / arms.size();

    this->probabilities = std::map<string, float>();
    for (const string& arm : arms) {
        this->probabilities[arm] = uniform_prob;
    }
}

BanditOperator::BanditOperator(map<string, float> arms_probs)
{
    this->probabilities = std::map<string, float>();
    for (const auto& arm_prob : arms_probs) {
        this->probabilities[arm_prob.first] = arm_prob.second;
    }
}

std::map<string, float> BanditOperator::sample_probs(bool update)
{
    // TODO: Implement the logic for sampling probabilities
    // based on the bandit operator's strategy

    // Throw an error if the select() operation is undefined
    HANDLE_ERROR_THROW("Undefined bandit sample_probs() operation");

    // Return an empty vector
    return this->probabilities;
}

string BanditOperator::choose()
{
    // TODO: Implement the logic for sampling probabilities
    // based on the bandit operator's strategy

    HANDLE_ERROR_THROW("Undefined bandit choose() operation");

    // Placeholder
    return this->probabilities.begin()->first;
}


void BanditOperator::update(string arm, float reward)
{
    // TODO: Implement the logic for updating the bandit operator's internal state
    // based on the received rewards

    // Throw an error if the update operation is undefined
    HANDLE_ERROR_THROW("Undefined bandit update() operation");
}

} // MAB
} // Brush