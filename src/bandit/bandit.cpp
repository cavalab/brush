#include "bandit.h"
#include <typeinfo> // FOR DEBUGGING PURPOSES. TODO: remove it later

namespace Brush {
namespace MAB {

Bandit::Bandit() { 
    set_type("dynamic_thompson");
    set_arms({});
    set_probs({});
    set_bandit();
}

Bandit::Bandit(string type, vector<string> arms) : type(type) {
    this->set_arms(arms);

    map<string, float> arms_probs;
    float prob = 1.0 / arms.size();
    for (const auto& arm : arms) {
        arms_probs[arm] = prob;
    }
    this->set_probs(arms_probs);
    this->set_bandit();
}

Bandit::Bandit(string type, map<string, float> arms_probs) : type(type) {
    this->set_probs(arms_probs);

    vector<string> arms_names;
    for (const auto& pair : arms_probs) {
        arms_names.push_back(pair.first);
    }
    this->set_arms(arms_names);
    this->set_bandit();
}

void Bandit::set_bandit() {
    // TODO: a flag that is set to true when this function is called. make all
    // other methods to raise an error if bandit was not set
    if (type == "thompson") {
        pbandit = make_unique<ThompsonSamplingBandit>(probabilities);
    } else if (type == "dynamic_thompson") {
        pbandit = make_unique<ThompsonSamplingBandit>(probabilities, true);
    } else if (type == "dummy") {
        pbandit = make_unique<DummyBandit>(probabilities);
    } else {
        HANDLE_ERROR_THROW("Undefined Selection Operator " + this->type + "\n");
    }
}

string Bandit::get_type() {
    return type;
}

void Bandit::set_type(string type) {
    this->type = type;
}

vector<string> Bandit::get_arms() {
    return arms;
}

void Bandit::set_arms(vector<string> arms) {
    this->arms = arms;
}

map<string, float> Bandit::get_probs() {
    return probabilities;
}

void Bandit::set_probs(map<string, float> arms_probs) {
    probabilities = arms_probs;
}

map<string, float> Bandit::sample_probs(bool update) {
    map<string, float> new_probs = this->pbandit->sample_probs(update);

    // making all probabilities strictly positive
    float eps = 1e-6;

    for (auto& pair : new_probs) {
        if (pair.second <= 0.0f) {
            pair.second = eps;
        }
    }

    return new_probs; 
}

string Bandit::choose() {
    return this->pbandit->choose();
}

void Bandit::update(string arm, float reward) {
    this->pbandit->update(arm, reward);
}

} // MAB
} // Brush