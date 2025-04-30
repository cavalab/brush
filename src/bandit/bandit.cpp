#include "bandit.h"
#include <typeinfo> // FOR DEBUGGING PURPOSES. TODO: remove it later

namespace Brush {
namespace MAB {

template <typename T>
Bandit<T>::Bandit() { 
    set_type("dynamic_thompson");
    set_arms({});
    set_probs({});
    set_bandit();
}

template <typename T>
Bandit<T>::Bandit(string type, vector<T> arms) : type(type) {
    this->set_arms(arms);

    map<T, float> arms_probs;
    float prob = 1.0 / arms.size();
    for (const auto& arm : arms) {
        arms_probs[arm] = prob;
    }
    this->set_probs(arms_probs);
    this->set_bandit();
}

template <typename T>
Bandit<T>::Bandit(string type, map<T, float> arms_probs) : type(type) {
    this->set_probs(arms_probs);

    vector<T> arms_names;
    for (const auto& pair : arms_probs) {
        arms_names.push_back(pair.first);
    }
    this->set_arms(arms_names);
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
        pbandit = make_unique<LinearThompsonSamplingBandit<T>>(probabilities);
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

template <typename T> template <ProgramType PT>
VectorXf Bandit<T>::get_context(const Program<PT>& program, Iter spot,
                                const SearchSpace &ss, const Dataset &d) {
    // TODO: for better performance, get_context should calculate the context only if the 
    // pbandit is of a contextual type. otherwise, return empty stuff

    // cout << "Inside get_context" << endl;
    VectorXf context;
    // -------------------------------------------------------------------------
    //  SECOND APPROACH: prediction vector of the spot node
    // -------------------------------------------------------------------------
    if constexpr (PT==ProgramType::Regressor)
    {
        // cout << "RegressorProgram detected\n" << endl;
        
        // use the code below to work with the whole tree prediction -----------
        ArrayXf out = (*program.Tree.begin().node).template predict<ArrayXf>(d);
        context = out;

        // predicting the spot node --------------------------------------------
        // context = (*spot.node).template predict<ArrayXf>(d);
    }
    else if constexpr (PT==ProgramType::BinaryClassifier)
    {
        // cout << "ClassifierProgram detected\n" << endl;

        // use the code below to work with the whole tree prediction -----------
        ArrayXf out = (*program.Tree.begin().node).template predict<ArrayXf>(d);
        context = ArrayXf(out.template cast<float>());

        // predicting the spot node --------------------------------------------
        // ArrayXf logit = (*spot.node).template predict<ArrayXf>(d);
        // ArrayXb pred  = (logit > 0.5);
        // context = ArrayXf(pred.template cast<float>());
    }
    else if constexpr (PT==ProgramType::MulticlassClassifier)
    {
        // cout << "MulticlassClassifierProgram detected\n" << endl;

        // use the code below to work with the whole tree prediction -----------
        ArrayXXf out = (*program.Tree.begin().node).template predict<ArrayXXf>(d);
        auto argmax = Function<NodeType::ArgMax>{};
        context = ArrayXf(argmax(out).template cast<float>());
        
        // predicting the spot node --------------------------------------------
    }
    else if constexpr (PT==ProgramType::Representer)
    {
        cout << "MulticlassClassifierProgram detected, not implemented\n" << endl;
    }
    else
    {
        HANDLE_ERROR_THROW("No predict available for the class.");
    }

    // -------------------------------------------------------------------------
    // FIRST APPROACH: label encoding of nodes above/below/on the spot
    // -------------------------------------------------------------------------
    // context is 3 times the number of nodes in the search space.
    // it represents a label encoding of the Tree structure, where
    // the first third represents number of nodes above the spot,
    // the second represents the spot, and the third represents
    // the number of nodes below the spot.
    // The vector below works as a reference of the nodes.

    // cout << "Tree: " << std::endl;
    // for (auto it = Tree.begin(); it != Tree.end(); ++it) {
    //     for (int i = 0; i < Tree.depth(it); ++i) {
    //         std::cout << "  ";
    //     }
    //     std::cout << (*it).name << std::endl;
    // }

    // cout << "Spot name: " << (*spot).name << std::endl;

    // size_t tot_operators = ss.op_names.size(); //NodeTypes::Count;
    // size_t tot_features  = 0;

    // for (const auto& pair : ss.terminal_map)
    //     tot_features += pair.second.size();

    // size_t tot_symbols = tot_operators + tot_features;

    // VectorXf context( 3 * tot_symbols );
    // context.setZero();
    
    // for (auto it = Tree.begin(); it != Tree.end(); ++it) {
    //     if (Tree.is_valid(it)) {
    //         cout << "Check succeeded for node: " << (*it).name << std::endl;
    //         cout << "Depth of spot: " << Tree.depth(spot) << std::endl;
    //         cout << "Depth of it: " << Tree.depth(it) << std::endl;
    //         cout << "It is the spot, searching for it " << std::endl;
            
    //         // deciding if it is above or below the spot
    //         size_t pos_shift = 0; // above
    //         if (it == spot) { // spot
    //             pos_shift = 1;
    //         }
    //         else if (Tree.is_in_subTree(it, spot)) // below
    //             pos_shift = 2;

    //         cout << "Position shift: " << pos_shift << std::endl;
    //         if (Is<NodeType::Terminal, NodeType::Constant, NodeType::MeanLabel>((*it).node_type)){
    //             size_t feature_index = 0;

    //             // iterating using terminal_types since it is ordered
    //             for (const auto& terminal : ss.terminal_map.at((*it).ret_type)) {
    //                 if (terminal.name == (*it).name) {
    //                     // Just one hot encode --------------------------------------
    //                     context((tot_operators + feature_index) + pos_shift*tot_symbols) += 1.0;

    //                     // encode with weights --------------------------------------
    //                     // int Tree_complexity = operator_complexities.at((*it).node_type);
    //                     // if ((*it).get_is_weighted()
    //                     // &&  Isnt<NodeType::Constant, NodeType::MeanLabel>((*it).node_type) )
    //                     // {
    //                     //     if ((Is<NodeType::OffsetSum>((*it).node_type) && (*it).W != 0.0)
    //                     //     ||  ((*it).W != 1.0))
    //                     //         Tree_complexity = operator_complexities.at(NodeType::Mul) +
    //                     //                           operator_complexities.at(NodeType::Constant) + 
    //                     //                           Tree_complexity;
    //                     // }
    //                     // context((tot_operators + feature_index) + pos_shift*tot_symbols) += static_cast<float>(Tree_complexity);

    //                     // use recursive evaluation to get the complexity of the subTree
    //                     // linear complexity to avoid exponential increase of values
    //                     // int complexity = it.node->get_linear_complexity();
    //                     // context((tot_operators + feature_index) + pos_shift*tot_symbols) += static_cast<float>(complexity);

    //                     cout << "Below spot, terminal: " << terminal.name << " at feature index " << feature_index << std::endl;
    //                     break;
    //                 }
    //                 ++feature_index;
    //             }
    //         } else {
    //             auto it_op = std::find(ss.op_names.begin(), ss.op_names.end(), (*it).name);
    //             if (it_op != ss.op_names.end()) {
    //                 size_t op_index = std::distance(ss.op_names.begin(), it_op);
    //                 context(pos_shift * tot_symbols + op_index) += 1.0;
    //                 cout << "Below spot, operator: " << (*it).name << " of index " << pos_shift*tot_symbols + op_index << std::endl;
    //             }
    //             else {
    //                 HANDLE_ERROR_THROW("Undefined operator " + (*it).name + "\n");
    //             }
    //         }
    //     }
    // }

    // cout << "Context part 1: " << context.head(tot_symbols).transpose() << std::endl;
    // cout << "Context part 2: " << context.segment(tot_symbols, tot_symbols).transpose() << std::endl;
    // cout << "Context part 3: " << context.tail(tot_symbols).transpose() << std::endl;
    // -------------------------------------------------------------------------
    
    return context;
}

} // MAB
} // Brush