#include "bandit_operator.h"

// #include <boost/random.hpp>
// #include <boost/random/gamma_distribution.hpp>

#include <boost/math/distributions/gamma.hpp>

// // https://www.boost.org/doc/libs/1_85_0/doc/html/boost/random/beta_distribution.html
// #include <boost/random/beta_distribution.hpp>

#include "../util/utils.h" // to use random generator

#ifndef THOMPSON_H
#define THOMPSON_H

namespace Brush {
namespace MAB {

template <typename T>
class ThompsonSamplingBandit : public BanditOperator<T>
{
public:
    ThompsonSamplingBandit(vector<T> arms, bool dynamic=false);
    ThompsonSamplingBandit(map<T, float> arms_probs, bool dynamic=false);
    ~ThompsonSamplingBandit(){};

    std::map<T, float> sample_probs(bool update);
    T choose(const VectorXf& context);
    void update(T arm, float reward, VectorXf& context);
private:
    bool dynamic_update;
    float C = 250;

    std::map<T, float> alphas;
    std::map<T, float> betas;
};

} // MAB
} // Brush

#endif // THOMPSON_H
