#include "bandit.h"

namespace Brush {
namespace MAB {

template <typename T>
Bandit<T>::Bandit() { 
    set_type("dynamic_thompson");
    set_arms({});
    set_probs({});
    set_context_size(1);
    set_bandit();
}

template <typename T>
Bandit<T>::Bandit(string type, vector<T> arms, int c_size) : type(type) {
    this->set_arms(arms);

    map<T, float> arms_probs;
    float prob = 1.0 / arms.size();
    for (const auto& arm : arms) {
        arms_probs[arm] = prob;
    }
    this->set_probs(arms_probs);
    this->set_context_size(c_size);
    this->set_bandit();
}

template <typename T>
Bandit<T>::Bandit(string type, map<T, float> arms_probs, int c_size) : type(type) {
    this->set_probs(arms_probs);

    vector<T> arms_names;
    for (const auto& pair : arms_probs) {
        arms_names.push_back(pair.first);
    }
    this->set_arms(arms_names);
    this->set_context_size(c_size);
    this->set_bandit();
}

template <typename T>
void Bandit<T>::set_bandit() {
    // TODO: a flag that is set to true when this function is called. make all
    // other methods to raise an error if bandit was not set
    if (type == "thompson") {
        pbandit = make_unique<ThompsonSamplingBandit<T>>(probabilities);
    } else if (type == "dynamic_thompson") {
        pbandit = make_unique<ThompsonSamplingBandit<T>>(probabilities, true);
    } else if (type == "linear_thompson") {
        pbandit = make_unique<LinearThompsonSamplingBandit<T>>(probabilities, context_size);
    } else if (type == "dummy") {
        pbandit = make_unique<DummyBandit<T>>(probabilities);
    } else {
        HANDLE_ERROR_THROW("Undefined Selection Operator " + this->type + "\n");
    }
}

template <typename T>
string Bandit<T>::get_type() {
    return type;
}

template <typename T>
void Bandit<T>::set_type(string type) {
    this->type = type;
}

template <typename T>
int Bandit<T>::get_context_size() {
    return context_size;
}

template <typename T>
void Bandit<T>::set_context_size(int new_context_size) {
    this->context_size = new_context_size;
}

template <typename T>
vector<T> Bandit<T>::get_arms() {
    return arms;
}

template <typename T>
void Bandit<T>::set_arms(vector<T> arms) {
    this->arms = arms;
}

template <typename T>
map<T, float> Bandit<T>::get_probs() {
    return probabilities;
}

template <typename T>
void Bandit<T>::set_probs(map<T, float> arms_probs) {
    probabilities = arms_probs;
}

template <typename T>
map<T, float> Bandit<T>::sample_probs(bool update) {
    return this->pbandit->sample_probs(update);
}

template <typename T>
T Bandit<T>::choose(const VectorXf& context) {
    return this->pbandit->choose(context);
}

template <typename T>
void Bandit<T>::update(T arm, float reward, VectorXf& context) {
    this->pbandit->update(arm, reward, context);
}

template <typename T>
VectorXf Bandit<T>::get_context(const tree<Node>& tree, Iter spot, const SearchSpace &ss) {
    // TODO: for better performance, get_context should calculate the context only if the 
    // pbandit is of a contextual type. otherwise, return empty stuff

    // context is 3 times the number of nodes in the search space.
    // it represents a label encoding of the tree structure, where
    // the first third represents number of nodes above the spot,
    // the second represents the spot, and the third represents
    // the number of nodes below the spot.
    // The vector below works as a reference of the nodes.

    // std::cout << "Tree: " << std::endl;
    // for (auto it = tree.begin(); it != tree.end(); ++it) {
    //     for (int i = 0; i < tree.depth(it); ++i) {
    //         std::cout << "  ";
    //     }
    //     std::cout << (*it).name << std::endl;
    // }

    // std::cout << "Spot name: " << (*spot).name << std::endl;

    size_t tot_operators = ss.op_names.size(); //NodeTypes::Count;
    size_t tot_features  = 0;

    for (const auto& pair : ss.terminal_map)
        tot_features += pair.second.size();

    size_t tot_symbols = tot_operators + tot_features;

    // Print the header with the operator names and terminal names
    std::cout << "Operators: ";
    for (const auto& op_name : ss.op_names) {
        std::cout << op_name << " ";
    }
    std::cout << std::endl;

    std::cout << "Terminals: ";
    for (const auto& pair : ss.terminal_map) {
        for (const auto& terminal : pair.second) {
            std::cout << terminal.name << " ";
        }
    }
    std::cout << std::endl;

    // Assert that tot_symbols is the same as context_size
    assert(tot_symbols == context_size);

    VectorXf context( 3 * tot_symbols );
    context.setZero();
    
    for (auto it = tree.begin(); it != tree.end(); ++it) {
        if (tree.is_valid(it)) {
            // std::cout << "Check succeeded for node: " << (*it).name << std::endl;
            // std::cout << "Depth of spot: " << tree.depth(spot) << std::endl;
            // std::cout << "Depth of it: " << tree.depth(it) << std::endl;
            // std::cout << "It is the spot, searching for it " << std::endl;
            
            // deciding if it is above or below the spot
            size_t pos_shift = 0; // above
            if (it == spot) { // spot
                pos_shift = 1;
            }
            else if (tree.is_in_subtree(it, spot)) // below
                pos_shift = 2;

            // std::cout << "Position shift: " << pos_shift << std::endl;
            if (Is<NodeType::Terminal, NodeType::Constant, NodeType::MeanLabel>((*it).node_type)){
                size_t feature_index = 0;

                // iterating using terminal_types since it is ordered
                for (const auto& terminal : ss.terminal_map.at((*it).ret_type)) {
                    if (terminal.name == (*it).name) {
                        context((tot_operators + feature_index) + pos_shift*tot_symbols) += 1.0;
                        // std::cout << "Below spot, terminal: " << terminal.name << " at feature index " << feature_index << std::endl;
                        break;
                    }
                    ++feature_index;
                }
            } else {
                auto it_op = std::find(ss.op_names.begin(), ss.op_names.end(), (*it).name);
                if (it_op != ss.op_names.end()) {
                    size_t op_index = std::distance(ss.op_names.begin(), it_op);
                    context(pos_shift * tot_symbols + op_index) += 1.0;
                    // std::cout << "Below spot, operator: " << (*it).name << " of index " << pos_shift*tot_symbols + op_index << std::endl;
                }
                else {
                    HANDLE_ERROR_THROW("Undefined operator " + (*it).name + "\n");
                }
            }
        }
    }

    std::cout << "Context part 1: " << context.head(tot_symbols).transpose() << std::endl;
    std::cout << "Context part 2: " << context.segment(tot_symbols, tot_symbols).transpose() << std::endl;
    std::cout << "Context part 3: " << context.tail(tot_symbols).transpose() << std::endl;

    return context;
}

} // MAB
} // Brush