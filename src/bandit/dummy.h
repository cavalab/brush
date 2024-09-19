#include "bandit_operator.h"

#ifndef DUMMY_H
#define DUMMY_H

#include "bandit_operator.h"

namespace Brush {
namespace MAB {

// TODO: rename dummy to static or fixed

template <typename T>
class DummyBandit : public BanditOperator<T>
{
public:
    DummyBandit(vector<T> arms)           : BanditOperator<T>(arms) {};
    DummyBandit(map<T, float> arms_probs) : BanditOperator<T>(arms_probs) {};
    ~DummyBandit(){};

    std::map<T, float> sample_probs(bool update);
    T choose(tree<Node>& tree, Fitness& f);
    void update(T arm, float reward, tree<Node>* tree=nullptr, Fitness* f=nullptr);

private:
    // additional stuff should come here
};

} // MAB
} // Brush

#endif // DUMMY_H
