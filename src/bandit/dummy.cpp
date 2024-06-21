#include "dummy.h"

namespace Brush {
namespace MAB {

template <typename T>
DummyBandit<T>::DummyBandit() {
    // Constructor implementation
}

template <typename T>
map<T, float> DummyBandit<T>::sample_probs(bool update) {
    return probabilities; // TODO: return the probabilities
}

template <typename T>
void DummyBandit<T>::update_with_reward(std::vector<float> rewards) {
}

} // MAB
} // Brush