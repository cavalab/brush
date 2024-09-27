#include "linear_thompson.h"

namespace Brush {
namespace MAB {

template <typename T>
LinearThompsonSamplingBandit<T>::LinearThompsonSamplingBandit(vector<T> arms, int c_size)
    : BanditOperator<T>(arms)
    , context_size(c_size)
{
    n_arms = arms.size();

    // Initialize Eigen matrices
    B = MatrixXf::Identity(context_size, context_size).replicate(1, n_arms);
    m2_r = MatrixXf::Zero(n_arms, context_size);
    B_inv = B;
    B_inv_sqrt = B;
    mean = MatrixXf::Zero(n_arms, context_size);

    arm_index_to_key.resize(n_arms);
    for (int i = 0; i < n_arms; ++i) {
        arm_index_to_key[i] = arms[i];
    }
}

template <typename T>
LinearThompsonSamplingBandit<T>::LinearThompsonSamplingBandit(map<T, float> arms_probs, int c_size)
    : BanditOperator<T>(arms_probs)
    , context_size(c_size)
{
    n_arms = arms_probs.size();

    // Initialize Eigen matrices
    B = MatrixXf::Identity(context_size, context_size).replicate(1, n_arms);
    m2_r = MatrixXf::Zero(n_arms, context_size);
    B_inv = B;
    B_inv_sqrt = B;
    mean = MatrixXf::Zero(n_arms, context_size);

    int index = 0;
    for (const auto& pair : arms_probs) {
        arm_index_to_key[index++] = pair.first;
    }
};
    

template <typename T>
std::map<T, float> LinearThompsonSamplingBandit<T>::sample_probs(bool update) {
    cout << "sampling probs started" << endl;
    if (update)
    {
        MatrixXf w = mean + (B_inv_sqrt * MatrixXf::Random(n_arms, context_size));

        // Calculate probabilities
        std::map<T, float> probs;
        float total_prob = 0.0f;
        
        for (int i = 0; i < n_arms; ++i) {
            float prob = exp(w(i)) / exp(w.maxCoeff());
            probs[arm_index_to_key[i]] = prob;
            total_prob += prob;
        }

        // Normalize probabilities to ensure they sum to 1
        for (auto& pair : probs) {
            pair.second /= total_prob;
        }

        this->probabilities = probs;
    }

    return this->probabilities;
    cout << "sampling probs finished" << endl;
}

template <typename T>
T LinearThompsonSamplingBandit<T>::choose(const VectorXf& context) {
    cout << "choose started" << endl;
    assert(context.size() == context_size);

    cout << "Context: " << context.transpose() << endl;

    MatrixXf w = mean + (B_inv_sqrt * MatrixXf::Random(n_arms, context_size));
    cout << "w: " << w << endl;

    VectorXf u = w * context;
    cout << "u: " << u.transpose() << endl;

    Eigen::Index max_index;
    float max_value = u.maxCoeff(&max_index);
    cout << "max_index: " << max_index << ", max_value: " << max_value << endl;

    cout << "choose finished" << endl;
    return arm_index_to_key[max_index];
}

template <typename T>
void LinearThompsonSamplingBandit<T>::update(T arm, float reward, VectorXf& context) {
    cout << "update started" << endl;

    assert(context.size() == context_size && "Context vector size mismatch");
    
    // Find the arm index using our mapping
    auto it = std::find_if(arm_index_to_key.begin(), arm_index_to_key.end(),
                           [&arm](const auto& pair) { return pair.second == arm; });
    
    if (it == arm_index_to_key.end()) {
        throw std::invalid_argument("Arm not found in the arm_index_to_key map");
    }
    
    int arm_index = it->first;

    B.row(arm_index) += context.transpose() * context;
    m2_r.row(arm_index) += reward * context.transpose();
    B_inv.row(arm_index) = B.row(arm_index).inverse();
    B_inv_sqrt.row(arm_index) = B_inv.row(arm_index).llt().matrixL();
    mean.row(arm_index) = B_inv.row(arm_index) * m2_r.row(arm_index);
    
    cout << "update finished" << endl;
}

} // MAB
} // Brush