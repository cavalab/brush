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
    
    // from https://stackoverflow.com/questions/4181403/generate-random-number-based-on-beta-distribution-using-boost
    // You'll first want to draw a random number uniformly from the
    // range (0,1). Given any distribution, you can then plug that number
    // into the distribution's "quantile function," and the result is as
    // if a random value was drawn from the distribution. 

    // from https://stackoverflow.com/questions/10358064/random-numbers-from-beta-distribution-c
    // The beta distribution is related to the gamma distribution. Let X be a
    // random number drawn from Gamma(α,1) and Y from Gamma(β,1), where the
    // first argument to the gamma distribution is the shape parameter.
    // Then Z=X/(X+Y) has distribution Beta(α,β). 
    
    if (update) {
        // 1. use a beta distribution based on alphas and betas to sample probabilities
        // 2. normalize probabilities so the sum is 1?

        float alpha, beta, X, Y, prob;
        for (const auto& pair : this->probabilities) {
            T arm = pair.first;

            alpha = alphas[arm];
            beta  = betas[arm];
            
            // TODO: stop using boost and use std::gamma_distribution (first, search to see if it is faster)
            boost::math::gamma_distribution<> gammaX(alpha);
            boost::math::gamma_distribution<> gammaY(beta);

            X = boost::math::quantile(gammaX, Brush::Util::r.rnd_flt());
            Y = boost::math::quantile(gammaY, Brush::Util::r.rnd_flt());

            prob =  X/(X+Y);

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