#include "search_space.h"
#include "program/program.h"
#include <iostream>

namespace Brush{

/// @brief generate terminals from the dataset features and random constants.
/// @param d a dataset
/// @return a vector of nodes 
vector<Node> generate_terminals(const Dataset& d)
{
    vector<Node> terminals;
    int i = 0;
    for ( const auto &[feature_name, value]: d.features ) 
    {
        std::visit(
            [&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                using Scalar = typename T::Scalar;
                constexpr bool weighted = std::is_same_v<Scalar, float>;
                // if constexpr (T::Scalar == float)
                terminals.push_back(Node(
                    NodeType::Terminal, 
                    Signature<T()>{}, 
                    weighted,
                    feature_name
                ));
            },
            value
        );
        ++i;
    };

    // add a constant
    terminals.push_back( Node(NodeType::Constant, Signature<ArrayXf()>{}, true, "C"));
    return terminals;
};

std::unordered_map<std::size_t, std::string> ArgsName; 

void SearchSpace::print() const { 
    std::cout << fmt::format("{}\n", *this) << std::flush; 
}

void SearchSpace::init(const Dataset& d, const unordered_map<string,float>& user_ops)
{
    // fmt::print("constructing search space...\n");
    this->node_map.clear();
    this->node_map_weights.clear();
    this->terminal_map.clear();
    this->terminal_types.clear();
    this->terminal_weights.clear();

    bool use_all = user_ops.size() == 0;
    vector<string> op_names;
    for (const auto& [op, weight] : user_ops)
        op_names.push_back(op);

    // create nodes based on data types 
    terminal_types = d.unique_data_types;

    vector<Node> terminals = generate_terminals(d);
    
    /* fmt::print("generate nodetype\n"); */
    GenerateNodeMap(user_ops, d.unique_data_types, 
                    std::make_index_sequence<NodeTypes::OpCount>());
    // map terminals
    /* fmt::print("looping through terminals...\n"); */
    for (const auto& term : terminals)
    {
        /* fmt::print("adding {} to search space...\n", term.get_name()); */
        if (terminal_map.find(term.ret_type) == terminal_map.end())
            terminal_map[term.ret_type] = vector<Node>();

        /* fmt::print("terminal ret_type: {}\n", DataTypeName[term.ret_type]); */
        terminal_map[term.ret_type].push_back(term);
        terminal_weights[term.ret_type].push_back(1.0);
    }
};

RegressorProgram SearchSpace::make_regressor(int max_d, int max_size)
{
    return make_program<RegressorProgram>(max_d, max_size);
};

ClassifierProgram SearchSpace::make_classifier(int max_d, int max_size)
{
    return make_program<ClassifierProgram>(max_d, max_size);
};

MulticlassClassifierProgram SearchSpace::make_multiclass_classifier(
    int max_d, int max_size)
{
    return make_program<MulticlassClassifierProgram>(max_d, max_size);
};

RepresenterProgram SearchSpace::make_representer(int max_d, int max_size)
{
    return make_program<RepresenterProgram>(max_d, max_size);
};

} //Brush

