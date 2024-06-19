#include "bandit_operator.h"

#ifndef DUMMY_H
#define DUMMY_H

#include "bandit_operator.h"

namespace Brush {
namespace MAB {

class DummyBandit : public BanditOperator {
public:
    DummyBandit();

    ~DummyBandit(){};

    vector<float> sample_probs(bool update);
    void update_with_reward(std::vector<float> rewards);

    // additional stuff should come here
};

} // MAB
} // Brush

#endif // DUMMY_H
