#include "search_space.h"

namespace Brush{

set<Node> generate_all_nodes(vector<string>& node_names, set<DataType> term_types)
{
    set<Node> nodes;

    for ( const auto &[name, v]: NodeSchema.items() ) 
    {
        if (!in(node_names, name) )
            continue;

        auto node_type = NodeNameType[name];

        for ( const auto &[ret_name, sig]: v["Signature"].items() ) 
        {
            // TODO: potentially check signature and filter out types
            auto ret_type = DataNameType[ret_name];
            nodes.insert(Node(node_type, ret_type, true));
            nodes.insert(Node(node_type, ret_type, false));
             
        }
    }
    return nodes;
}

NodeVector generate_terminals(const Data& d)
{
    NodeVector terminals;
    int i = 0;
    for ( const auto &[name, value]: d.features ) 
    {
        auto data_type = d.data_types.at(i);
        // note: structured bindings cannot be captured by lambdas until C++20
        /* std::cout << "generating terminal " << key << "of type " << value << endl; */
        // terminals.push_back(make_shared<Terminal>(var, d[var]));

        terminals.push_back(Node(NodeType::Terminal, data_type, name));
        ++i;
    };
    return terminals;
};

} //Brush
