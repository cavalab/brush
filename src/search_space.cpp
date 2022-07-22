#include "search_space.h"

namespace Brush{

/* template<DataType DT> */
/* Node make_terminal(const string& featurename){ */
/*     auto sig_hash=Signature<RetType()>::hash(); */
/*     return Node(NodeType::Terminal, featurename, DT, sig_hash); */
/* } */

vector<Node> generate_terminals(const Data& d)
{
    vector<Node> terminals;
    int i = 0;
    for ( const auto &[featurename, value]: d.features ) 
    {
        auto data_type = d.feature_types.at(i);
        /* using RetType = typename DataEnumType<data_type>::type; */
        /* auto sig_hash = typeid(tuple<RetType>); */
        auto sig_hash = DataTypeID.at(data_type).hash_code();
        /* fmt::print("generating Terminal name: {}, data_type: {}", featurename, data_type); */
        terminals.push_back(Node(NodeType::Terminal, featurename, data_type, sig_hash));
        ++i;
    };
    return terminals;
};

std::unordered_map<std::size_t, std::string> ArgsName; 

void SearchSpace::init(const Data& d, const unordered_map<string,float>& user_ops)
{
    fmt::print("constructing search space...\n");

    this->node_map.clear();
    this->weight_map.clear();
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
    
    /* fmt::print("generate nodemap\n"); */
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

    fmt::print("{}\n", *this);

};


} //Brush

