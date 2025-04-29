#include "linear_thompson.h"

namespace Brush {
namespace MAB {

template <typename T>
LinearThompsonSamplingBandit<T>::LinearThompsonSamplingBandit(vector<T> arms)
    : BanditOperator<T>(arms)
{
    n_arms = arms.size();

    // Initialize Eigen matrices
    arm_index_to_key.resize(0);
    B.resize(0);
    B_inv.resize(0);
    B_inv_sqrt.resize(0);

    m2_r = MatrixXf::Zero(n_arms, 1);
    mean = MatrixXf::Zero(n_arms, 1);

    for (int i = 0; i < n_arms; ++i) { // one for each arm
        arm_index_to_key[i] = arms[i];
    }
}

template <typename T>
LinearThompsonSamplingBandit<T>::LinearThompsonSamplingBandit(map<T, float> arms_probs)
    : BanditOperator<T>(arms_probs)
{
    n_arms = arms_probs.size();

    B.resize(0);
    B_inv.resize(0);
    B_inv_sqrt.resize(0);

    m2_r = MatrixXf::Zero(n_arms, 1);
    mean = MatrixXf::Zero(n_arms, 1);

    int index = 0;
    for (const auto& pair : arms_probs) { // making sure we have the same order
        arm_index_to_key[index++] = pair.first;
    }
};
    

template <typename T>
std::map<T, float> LinearThompsonSamplingBandit<T>::sample_probs(bool update) {

    if (update && B.size()>0) // must be called after at least one choose
    {
        int context_size = B.at(0).rows();

        MatrixXf w(n_arms, context_size);
        MatrixXf r = MatrixXf::Random(n_arms, context_size); // TODO: use random generator here
        for (int i = 0; i < n_arms; ++i) {
            w.row(i) = B_inv_sqrt[i] * r.row(i); // mat mul
        }

        w = mean + w;

        VectorXf u(n_arms);
        
        VectorXf last_context = ArrayXf::Random(context_size);
        
        u = w * last_context; // mat mul

        float total_prob = 0.0f;
        for (int i = 0; i < n_arms; ++i) {
            float prob = std::exp(u(i)) / std::exp(u.maxCoeff());
            this->probabilities[arm_index_to_key[i]] = prob;
            total_prob += prob;
        }

        assert(total_prob > 0 && "Total probability must be greater than zero");

        // Normalize probabilities to ensure they sum to 1
        for (auto& [k, v] : this->probabilities) {
            this->probabilities[k] = std::min(this->probabilities[k], 1.0f); // / total_prob
        }
    }

    return this->probabilities;

}

template <typename T>
T LinearThompsonSamplingBandit<T>::choose(const VectorXf& context) {
    int context_size = context.size();

    if (B.size()==0){


        for (int i = 0; i < n_arms; ++i) { // one for each arm
            B.push_back( MatrixXf::Identity(context_size, context_size) );
            B_inv.push_back( MatrixXf::Identity(context_size, context_size) );
            B_inv_sqrt.push_back( MatrixXf::Identity(context_size, context_size) );
        }

        m2_r = MatrixXf::Zero(n_arms, context_size);
        mean = MatrixXf::Zero(n_arms, context_size);
    }

    MatrixXf w(n_arms, context_size);
    MatrixXf r = MatrixXf::Random(n_arms, context_size); // TODO: use random generator here
    for (int i = 0; i < n_arms; ++i) {
        w.row(i) = B_inv_sqrt[i] * r.row(i); // mat mul
    }

    w = mean + w;
        

    VectorXf u(n_arms);
    u = w * context; // mat mul


    Eigen::Index max_index;
    float max_value = u.maxCoeff(&max_index);



    return arm_index_to_key[max_index];
}

template <typename T>
void LinearThompsonSamplingBandit<T>::update(T arm, float reward, VectorXf& context) {
    int context_size = context.size();

    if (B.size()==0){

        for (int i = 0; i < n_arms; ++i) { // one for each arm
            B.push_back( MatrixXf::Identity(context_size, context_size) );
            B_inv.push_back( MatrixXf::Identity(context_size, context_size) );
            B_inv_sqrt.push_back( MatrixXf::Identity(context_size, context_size) );
        }

        m2_r = MatrixXf::Zero(n_arms, context_size);
        mean = MatrixXf::Zero(n_arms, context_size);
    }

    // TODO: have a more efficient way of doing this
    // Find the arm index using our mapping
    auto it = std::find_if(arm_index_to_key.begin(), arm_index_to_key.end(),
                           [&arm](const auto& pair) { return pair.second == arm; });

    if (it == arm_index_to_key.end()) {
        throw std::invalid_argument("Arm not found in the arm_index_to_key map");
    }

    int arm_index = it->first;






    B[arm_index] += context * context.transpose();


    m2_r.row(arm_index) += (context * reward).transpose();


    B_inv[arm_index] = B[arm_index].inverse();


    B_inv_sqrt[arm_index] = B_inv[arm_index].ldlt().matrixL();


    mean.row(arm_index) = B_inv[arm_index] * m2_r.row(arm_index).transpose(); // mat mul



}

} // MAB
} // Brush