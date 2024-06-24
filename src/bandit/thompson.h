#include "bandit_operator.h"

// https://www.boost.org/doc/libs/1_85_0/doc/html/boost/random/beta_distribution.html
#include <boost/random/beta_distribution.hpp>

#ifndef THOMPSON_H
#define THOMPSON_H

namespace Brush {
namespace MAB {

using namespace boost::random;

template <typename T>
class ThompsonSamplingBandit : public BanditOperator<T>
{
public:
    ThompsonSamplingBandit(vector<T> arms);
    ThompsonSamplingBandit(map<T, float> arms_probs);
    ~ThompsonSamplingBandit(){};

    map<T, float> sample_probs(bool update);
    void update(T arm, float reward);

private:
    // additional stuff should come here
    beta_distribution<> BetaDistribution;

    std::map<T, int> alphas;
    std::map<T, int> betas;
};

} // MAB
} // Brush

#endif // THOMPSON_H
