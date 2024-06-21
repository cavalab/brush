#include "bandit.h"

namespace Brush {
namespace MAB {

template <typename T>
Bandit<T>::Bandit() : type("dummy"), arms(0) { 
    this->set_bandit();
}

template <typename T>
Bandit<T>::Bandit(string type, int arms) : type(type), arms(arms) {
}

template <typename T>
void Bandit<T>::set_bandit() {
    if (type == "dummy") {
        pbandit = make_unique<DummyBandit<T>>();
    } else {
        HANDLE_ERROR_THROW("Undefined Selection Operator " + this->type + "\n");
    }
}

template <typename T>
string Bandit<T>::get_type() {
    return type;
}

template <typename T>
void Bandit<T>::set_type(string type) {
    this->type = type;
}

template <typename T>
map<T, float> Bandit<T>::Bandit::sample_probs(bool update) {
    return this->pbandit->sample_probs(update);
}

template <typename T>
void Bandit<T>::update_with_reward(vector<float> rewards) {
    // TODO: Implement the logic to update the inner state of the bandit based on the rewards obtained from the arms
}

} // MAB
} // Brush