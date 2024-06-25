#include "thompson.h"

namespace Brush {
namespace MAB {

template <typename T>
ThompsonSamplingBandit<T>::ThompsonSamplingBandit(vector<T> arms)
    : BanditOperator<T>(arms)
{
    for (const auto& arm : arms) {
        alphas[arm] = 2;
        betas[arm]  = 2;
    }
}

template <typename T>
ThompsonSamplingBandit<T>::ThompsonSamplingBandit(map<T, float> arms_probs)
    : BanditOperator<T>(arms_probs)
{
    for (const auto& pair : arms_probs) {
        alphas[pair.first] = 2;
        betas[pair.first]  = 2;
    }
};
    

template <typename T>
std::map<T, float> ThompsonSamplingBandit<T>::sample_probs(bool update) {

    if (update) {
        for (const auto& pair : this->probabilities) {
            T arm = pair.first;
            float prob = static_cast<float>(alphas[arm] - 1) / static_cast<float>(alphas[arm] + betas[arm] - 2);
            this->probabilities[arm] = prob;
        }
    }

    return this->probabilities;
}

template <typename T>
void ThompsonSamplingBandit<T>::update(T arm, float reward) {
    // reward must be either 0 or 1
    alphas[arm] += reward;
    betas[arm]  += 1 - reward;
}

} // MAB
} // Brush