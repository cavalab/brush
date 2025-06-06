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
    T choose(const VectorXf& context);
    void update(T arm, float reward, VectorXf& context);

private:
    // additional stuff should come here
};

} // MAB
} // Brush

#endif // DUMMY_H
