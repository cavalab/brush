#include "bandit.h"

namespace Brush {
namespace MAB {

Bandit::Bandit() : type("dummy"), arms(0) { 
    this->set_bandit();
}

Bandit::Bandit(std::string type, int arms) : type(type), arms(arms) {
    // Initialize the probability vector
    probabilities.resize(arms, 1.0f / arms);
}

void Bandit::set_bandit() {
    if (type == "dummy") {
        pbandit = std::make_unique<DummyBandit>();
    } else {
        HANDLE_ERROR_THROW("Undefined Selection Operator " + this->type + "\n");
    }
}

std::string Bandit::get_type() {
    return type;
}

void Bandit::set_type(std::string type) {
    this->type = type;
}

std::vector<float> Bandit::sample_probs(bool update) {
    // TODO: Implement the logic to sample probability distribution for each arm

    return probabilities;
}

void Bandit::update_with_reward(std::vector<float> rewards) {
    // TODO: Implement the logic to update the inner state of the bandit based on the rewards obtained from the arms
}

} // MAB
} // Brush