#include "dummy.h"

namespace Brush {
namespace MAB {

template <typename T>
std::map<T, float> DummyBandit<T>::sample_probs(bool update) {
    return this->probabilities;
}

template <typename T>
T DummyBandit<T>::choose(tree<Node>& tree, Fitness& f) {
    // std::map<T, float> probs = this->sample_probs(false);

    return r.random_choice(this->probabilities);
}

template <typename T>
void DummyBandit<T>::update(T arm, float reward, tree<Node>* tree, Fitness* f) {
    // Do nothing
}

} // MAB
} // Brush