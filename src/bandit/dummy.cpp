#include "dummy.h"

namespace Brush {
namespace MAB {

template <typename T>
map<T, float> DummyBandit<T>::sample_probs(bool update) {
    return probabilities;
}

template <typename T>
void DummyBandit<T>::update(T arm, float reward) {
}

} // MAB
} // Brush