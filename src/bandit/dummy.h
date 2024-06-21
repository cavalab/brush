#include "bandit_operator.h"

#ifndef DUMMY_H
#define DUMMY_H

#include "bandit_operator.h"

namespace Brush {
namespace MAB {

template <typename T>
class DummyBandit : public BanditOperator<T>
{
public:
    DummyBandit();
    ~DummyBandit(){};

    map<T, float> sample_probs(bool update);
    void update_with_reward(std::vector<float> rewards);

    std::map<T, float> probabilities;

    // additional stuff should come here
};

} // MAB
} // Brush

#endif // DUMMY_H
