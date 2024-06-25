/* Brush
copyright 2024 William La Cava
license: GNU/GPL v3
*/

#ifndef BANDIT_H
#define BANDIT_H

#include "bandit_operator.h"
#include "dummy.h"
#include "thompson.h"

namespace Brush {
namespace MAB {

using namespace Brush;

// TODO: add bandit to parameters and write doc in estimatorinterface

// TODO: all templates, or require some specific types?
template <typename T>
struct Bandit
{
public:
    /**
     * @brief A shared pointer to the bandit operator (policy).
    */
    std::shared_ptr<BanditOperator<T>> pbandit;
    // TODO: This should be a shared pointer to allow multiple instances of Bandit to share the same operator.
     
    std::string type;
    vector<T> arms;

    std::map<T, float> probabilities;

    Bandit();
    ~Bandit(){};

    Bandit(string type, vector<T> arms);
    Bandit(string type, map<T, float> arms_probs);

    void set_bandit();

    void set_arms(vector<T> arms);
    vector<T> get_arms();

    string get_type();
    void set_type(string type);

    map<T, float> get_probs();
    void set_probs(map<T, float> arms_probs);

    map<T, float> sample_probs(bool update=false);

    void update(T arm, float reward);
};

//TODO: serialization should save the type of bandit and its parameters

} // MAB
} // Brush
#endif