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
    arm_index_to_key.resize(0);
    B.resize(0);
    B_inv.resize(0);
    B_inv_sqrt.resize(0);
    for (int i = 0; i < n_arms; ++i) { // one for each arm
        arm_index_to_key[i] = arms[i];

        B.push_back( MatrixXf::Identity(context_size, context_size) );
        B_inv.push_back( MatrixXf::Identity(context_size, context_size) );
        B_inv_sqrt.push_back( MatrixXf::Identity(context_size, context_size) );
    }

    m2_r = MatrixXf::Zero(n_arms, context_size);
    mean = MatrixXf::Zero(n_arms, context_size);
    
    last_context.resize(context_size);
    last_context.setOnes();
}

template <typename T>
LinearThompsonSamplingBandit<T>::LinearThompsonSamplingBandit(map<T, float> arms_probs, int c_size)
    : BanditOperator<T>(arms_probs)
    , context_size(c_size)
{
    n_arms = arms_probs.size();

    B.resize(0);
    B_inv.resize(0);
    B_inv_sqrt.resize(0);
    for (int i = 0; i < n_arms; ++i) { // one for each arm
        B.push_back( MatrixXf::Identity(context_size, context_size) );
        B_inv.push_back( MatrixXf::Identity(context_size, context_size) );
        B_inv_sqrt.push_back( MatrixXf::Identity(context_size, context_size) );
    }

    m2_r = MatrixXf::Zero(n_arms, context_size);
    mean = MatrixXf::Zero(n_arms, context_size);

    int index = 0;
    for (const auto& pair : arms_probs) { // making sure we have the same order
        arm_index_to_key[index++] = pair.first;
    }

    last_context.resize(context_size);
    last_context.setOnes();
};
    

template <typename T>
std::map<T, float> LinearThompsonSamplingBandit<T>::sample_probs(bool update) {
    // cout << "sampling probs started" << endl;
    if (update)
    {
        MatrixXf w(n_arms, context_size);
        MatrixXf r = MatrixXf::Random(n_arms, context_size); // TODO: use random generator here
        for (int i = 0; i < n_arms; ++i) {
            w.row(i) = B_inv_sqrt[i] * r.row(i); // mat mul
        }

        w = mean + w;

        VectorXf u(n_arms);
        u = w * last_context; // mat mul

        // for (int i = 0; i < n_arms; ++i) {
        //     // cout << "Dot product for row " << i;
        //     float dot_product = w.row(i).dot(last_context);
        //     if (std::isnan(dot_product))
        //     {
        //         dot_product = 0.0f;
        //         // cout << "(nan)";
        //     }
        //     // cout << "Dot product for row " << i << ": " << dot_product << endl;
                
        //     u(i) = dot_product;
        // }

        for (int i = 0; i < n_arms; ++i) {
            this->probabilities[arm_index_to_key[i]] = std::exp(u(i));
        }

        // // Calculate probabilities
        // std::map<T, float> probs;
        // float total_prob = 0.0f;
        
        // for (int i = 0; i < n_arms; ++i) {
        //     float prob = exp(u(i)) / exp(u.maxCoeff());
        //     probs[arm_index_to_key[i]] = prob;
        //     total_prob += prob;
        // }

        // // Normalize probabilities to ensure they sum to 1
        // for (auto& pair : probs) {
        //     pair.second /= total_prob;
        // }

        // this->probabilities = probs;
    }

    return this->probabilities;
    // cout << "sampling probs finished" << endl;
}

template <typename T>
T LinearThompsonSamplingBandit<T>::choose(const VectorXf& context) {
    // cout << "choose started" << endl;
    assert(context.size() == context_size && "Context vector size mismatch in choose");

    last_context = context;

    // cout << "Context: " << context.transpose() << endl;

    MatrixXf w(n_arms, context_size);
    MatrixXf r = MatrixXf::Random(n_arms, context_size); // TODO: use random generator here
    for (int i = 0; i < n_arms; ++i) {
        w.row(i) = B_inv_sqrt[i] * r.row(i); // mat mul
    }

    w = mean + w;
        
    // cout << "w: " << w << endl;
    VectorXf u(n_arms);
    u = w * context; // mat mul
    // cout << "u: " << u << endl;

    // for (int i = 0; i < n_arms; ++i) {
    //     // cout << "Dot product for row " << i;
    //     float dot_product = w.row(i).dot(context);
    //     if (std::isnan(dot_product))
    //     {
    //         dot_product = 0.0f;
    //         // cout << "(nan)";
    //     }

    //     // cout << "Dot product for row " << i << ": " << dot_product << endl;

    //     u(i) = dot_product;
    // }

    Eigen::Index max_index;
    float max_value = u.maxCoeff(&max_index);
    // cout << "max_index: " << max_index << ", max_value: " << max_value << endl;

    // cout << "choose finished" << endl;
    return arm_index_to_key[max_index];
}

template <typename T>
void LinearThompsonSamplingBandit<T>::update(T arm, float reward, VectorXf& context) {
    // cout << "update started" << endl;
    assert(context.size() == context_size && "Context vector size mismatch in update");

    // TODO: have a more efficient way of doing this
    // Find the arm index using our mapping
    auto it = std::find_if(arm_index_to_key.begin(), arm_index_to_key.end(),
                           [&arm](const auto& pair) { return pair.second == arm; });

    if (it == arm_index_to_key.end()) {
        throw std::invalid_argument("Arm not found in the arm_index_to_key map");
    }

    int arm_index = it->first;

    // cout << "Arm index: " << arm_index << endl;
    // cout << "Context: " << context.transpose() << endl;
    // cout << "B[arm_index] before update: " << B[arm_index] << endl;
    // cout << "m2_r.row(arm_index) before update: " << m2_r.row(arm_index) << endl;

    B[arm_index] += context * context.transpose();
    // cout << "B[arm_index] after update: " << B[arm_index] << endl;

    m2_r.row(arm_index) += context * reward;
    // cout << "m2_r.row(arm_index) after update: " << m2_r.row(arm_index) << endl;

    B_inv[arm_index] = B[arm_index].inverse();
    // cout << "B_inv[arm_index]: " << B_inv[arm_index] << endl;

    B_inv_sqrt[arm_index] = B_inv[arm_index].ldlt().matrixL();
    // cout << "B_inv_sqrt[arm_index]: " << B_inv_sqrt[arm_index] << endl;

    mean.row(arm_index) = B_inv[arm_index] * m2_r.row(arm_index); // mat mul
    // cout << "mean.row(arm_index): " << mean.row(arm_index) << endl;

    // cout << "update finished" << endl;
}

} // MAB
} // Brush