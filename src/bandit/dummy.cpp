#include "dummy.h"

namespace Brush {
namespace MAB {

template <typename T>
std::map<T, float> DummyBandit<T>::sample_probs(bool update) {
    return this->probabilities;
}

template <typename T>
void DummyBandit<T>::update(T arm, float reward) {
}

} // MAB
} // Brush