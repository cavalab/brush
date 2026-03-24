#include "bandit_operator.h"

#ifndef DUMMY_H
#define DUMMY_H

#include "bandit_operator.h"

namespace Brush {
namespace MAB {

class DummyBandit : public BanditOperator
{
public:
    DummyBandit(vector<string> arms)      : BanditOperator(arms) {};
    DummyBandit(map<string, float> arms_probs) : BanditOperator(arms_probs) {};
    ~DummyBandit(){};

    std::map<string, float> sample_probs(bool update);
    string choose();
    void update(string arm, float reward);

private:
    // additional stuff should come here
};

} // MAB
} // Brush

#endif // DUMMY_H
