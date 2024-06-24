#include "bandit.h"

namespace Brush {
namespace MAB {

template <typename T>
Bandit<T>::Bandit() { 
    set_type("dummy");
    set_arms({});
    set_bandit();
}

template <typename T>
Bandit<T>::Bandit(string type, vector<T> arms) : type(type) {
    this->set_arms(arms);

    map<T, float> arms_probs;
    float prob = 1.0 / arms.size();
    for (const auto& arm : arms) {
        arms_probs[arm] = prob;
    }
    this->set_probs(arms_probs);
}

template <typename T>
Bandit<T>::Bandit(string type, map<T, float> arms_probs) : type(type) {
    this->set_probs(arms_probs);

    vector<T> arms_names;
    for (const auto& pair : arms_probs) {
        arms_names.push_back(pair.first);
    }
    this->set_arms(arms_names);
}

template <typename T>
void Bandit<T>::set_bandit() {
    // TODO: a flag that is set to true when this function is called. make all
    // other methods to raise an error if bandit was not set
    if (type == "thompson") {
        pbandit = make_unique<ThompsonSamplingBandit<T>>(probabilities);
    } else if (type == "dummy") {
        pbandit = make_unique<DummyBandit<T>>(probabilities);
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
vector<T> Bandit<T>::get_arms() {
    return arms;
}

template <typename T>
void Bandit<T>::set_arms(vector<T> arms) {
    this->arms = arms;
}

template <typename T>
map<T, float> Bandit<T>::get_probs() {
    return probabilities;
}

template <typename T>
void Bandit<T>::set_probs(map<T, float> arms_probs) {
    probabilities = arms_probs;
}

template <typename T>
map<T, float> Bandit<T>::sample_probs(bool update) {
    return this->pbandit->sample_probs(update);
}

template <typename T>
void Bandit<T>::update(T arm, float reward) {
    this->pbandit->update(arm, reward);
}

} // MAB
} // Brush