#include "linear_thompson.h"

namespace Brush {
namespace MAB {

template <typename T>
LinearThompsonSamplingBandit<T>::LinearThompsonSamplingBandit(vector<T> arms, bool dynamic)
    : BanditOperator<T>(arms)
{
}

template <typename T>
LinearThompsonSamplingBandit<T>::LinearThompsonSamplingBandit(map<T, float> arms_probs, bool dynamic)
    : BanditOperator<T>(arms_probs)
{
};
    

template <typename T>
std::map<T, float> LinearThompsonSamplingBandit<T>::sample_probs(bool update) {
    return this->probabilities;
}

template <typename T>
T LinearThompsonSamplingBandit<T>::choose(VectorXf& context) {
    // TODO: use context here
    
    std::map<T, float> probs = this->sample_probs(true);

    return r.random_choice(probs);
}

template <typename T>
void LinearThompsonSamplingBandit<T>::update(T arm, float reward, VectorXf& context) {

}

} // MAB
} // Brush