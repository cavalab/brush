#include "bandit_operator.h"

// #include <boost/random.hpp>
// #include <boost/random/gamma_distribution.hpp>

#include <boost/math/distributions/gamma.hpp>

// // https://www.boost.org/doc/libs/1_85_0/doc/html/boost/random/beta_distribution.html
// #include <boost/random/beta_distribution.hpp>

#include "../util/utils.h" // to use random generator

#ifndef LINEAR_THOMPSON_H
#define LINEAR_THOMPSON_H

namespace Brush {
namespace MAB {

template <typename T>
class LinearThompsonSamplingBandit : public BanditOperator<T>
{
public:
    LinearThompsonSamplingBandit(vector<T> arms, int c_size=1);
    LinearThompsonSamplingBandit(map<T, float> arms_probs, int c_size=1);
    ~LinearThompsonSamplingBandit(){};

    std::map<T, float> sample_probs(bool update);
    T choose(const VectorXf& context);
    void update(T arm, float reward, VectorXf& context);
private:
    int n_arms;
    int context_size;

    vector<MatrixXf> B;
    vector<MatrixXf> B_inv;
    vector<MatrixXf> B_inv_sqrt;
    
    MatrixXf m2_r;
    MatrixXf mean;
    
    // we need to make sure indexes does not change
    std::map<int, T> arm_index_to_key;
};

} // MAB
} // Brush

#endif // LINEAR_THOMPSON_H
