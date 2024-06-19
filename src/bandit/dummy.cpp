#include "dummy.h"

namespace Brush {
namespace MAB {

DummyBandit::DummyBandit() {
    // Constructor implementation
}

vector<float> DummyBandit::sample_probs(bool update) {
    return probabilities;
}

void DummyBandit::update_with_reward(std::vector<float> rewards) {
}

} // MAB
} // Brush