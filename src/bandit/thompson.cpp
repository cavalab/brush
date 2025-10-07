#include "thompson.h"

namespace Brush {
namespace MAB {

ThompsonSamplingBandit::ThompsonSamplingBandit(vector<string> arms, bool dynamic)
    : BanditOperator(arms)
    , dynamic_update(dynamic)
{
    for (const auto& arm : arms) {
        alphas[arm] = 2;
        betas[arm]  = 2;
    }
}

ThompsonSamplingBandit::ThompsonSamplingBandit(map<string, float> arms_probs, bool dynamic)
    : BanditOperator(arms_probs)
    , dynamic_update(dynamic)
{
    for (const auto& pair : arms_probs) {
        alphas[pair.first] = 2;
        betas[pair.first]  = 2;
    }
};
    

std::map<string, float> ThompsonSamplingBandit::sample_probs(bool update) {
    // gets sampling probabilities using the bandit

    if (update) {
        // 1. use a beta distribution based on alphas and betas to sample probabilities
        // 2. normalize probabilities so the sum is 1

        for (const auto& pair : this->probabilities) {
            string arm = pair.first;

            float prob = r.rnd_alpha_beta(alphas[arm], betas[arm]);

            // avoiding deadlocks when sampling from search space
            this->probabilities[arm] = std::max(std::min(prob, 1.0f), 0.001f);
        }

        // assert that the sum is not zero
        float totalProb = 0.0f;
        for (const auto& pair : this->probabilities) {
            totalProb += pair.second;
        }
        assert(totalProb != 0.0f && "Sum of probabilities is zero!");
    }

    return this->probabilities;
}

string ThompsonSamplingBandit::choose() {
    std::map<string, float> probs = this->sample_probs(true);

    return r.random_choice(probs);
}

void ThompsonSamplingBandit::update(string arm, float reward) {
    // reward must be either 0 or 1

    alphas[arm] += reward;
    betas[arm]  += 1.0f-reward;

    if (dynamic_update && alphas[arm] + betas[arm] >= C)
    {
        alphas[arm] *= C/(C+1.0f) ;
        betas[arm]  *= C/(C+1.0f) ;
    }
}

} // MAB
} // Brush