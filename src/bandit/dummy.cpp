#include "dummy.h"

namespace Brush {
namespace MAB {

std::map<string, float> DummyBandit::sample_probs(bool update) {
    return this->probabilities;
}

string DummyBandit::choose() {
    // std::map<T, float> probs = this->sample_probs(false);

    return r.random_choice(this->probabilities);
}

void DummyBandit::update(string arm, float reward) {
    // Do nothing
}

} // MAB
} // Brush