#include "search_space.h"

namespace Brush{

template<typename T>
tuple<set<Node>,set<type_index>> generate_nodes(vector<string>& op_names)
{

    // NodeNameTypeMap name2ret2node;
    set<Node> nodes; 
    set<type_index> new_types;

    // auto binary_operators = make_binary_operators<ArrayXf>()(op_names);
    auto binary_operators = OpMaker<BinaryOp<T>>().make(op_names);
    auto unary_operators = OpMaker<UnaryOp<T>>().make(op_names);
    // auto binary_operators = make_binary_operators<T>()(op_names);
    // auto binary_operators = make_binary_operators(op_names);
    // auto unary_operators = make_unary_operators<T>()(op_names);
    // auto reduce_operators = make_reduction_operators<T>();

    for (const auto& op : binary_operators)
    {
        nodes.insert( new DxNode<T(T,T)>(op->name+"_W", op->f, op->df, true) );
        nodes.insert( new DxNode<T(T,T)>(op->name, op->f, op->df, false) );
    }

    for (const auto& op : unary_operators)
    {
        nodes.insert( new DxNode<T(T)>(op->name+"_W", op->f, op->df, true) );
        nodes.insert( new DxNode<T(T)>(op->name, op->f, op->df, false) );
    }

    // if ( in(op_names, "best_split"))
    //     nodes.insert(new SplitNode<T(T,T)>("best_split"));

    // if ( in(op_names, "arg_split"))
    // {
    //     nodes.insert( new SplitNode<T(T,T,T)>("arg_split"));
    // }

    return {nodes, new_types};
}
tuple<set<Node>,set<type_index>> generate_split_nodes(vector<string>& op_names)
{
    set<Node> nodes; 
    set<type_index> new_types;
    if ( in(op_names, "best_split"))
    {
        nodes.insert(new SplitNode<ArrayXf(ArrayXf,ArrayXf)>("best_split"));
        nodes.insert(new SplitNode<ArrayXi(ArrayXi,ArrayXi)>("best_split"));
        nodes.insert(new SplitNode<ArrayXb(ArrayXb,ArrayXb)>("best_split"));
    }

    if ( in(op_names, "arg_split"))
    {
        nodes.insert( new SplitNode<ArrayXf(ArrayXf,ArrayXf,ArrayXf)>("arg_split"));
        nodes.insert( new SplitNode<ArrayXf(ArrayXi,ArrayXf,ArrayXf)>("arg_split"));
        nodes.insert( new SplitNode<ArrayXf(ArrayXb,ArrayXf,ArrayXf)>("arg_split"));

        nodes.insert( new SplitNode<ArrayXi(ArrayXf,ArrayXi,ArrayXi)>("arg_split"));
        nodes.insert( new SplitNode<ArrayXi(ArrayXi,ArrayXi,ArrayXi)>("arg_split"));
        nodes.insert( new SplitNode<ArrayXi(ArrayXb,ArrayXi,ArrayXi)>("arg_split"));

        nodes.insert( new SplitNode<ArrayXb(ArrayXf,ArrayXb,ArrayXb)>("arg_split"));
        nodes.insert( new SplitNode<ArrayXb(ArrayXi,ArrayXb,ArrayXb)>("arg_split"));
        nodes.insert( new SplitNode<ArrayXb(ArrayXb,ArrayXb,ArrayXb)>("arg_split"));

    }

    return {nodes, new_types};

}

set<Node> generate_all_nodes(vector<string>& node_names, set<DataType> term_types)
{
    set<Node> nodes;

    for ( const auto &[name, v]: NodeSchema ) 
    {
        if !in(node_names, name) 
            continue;

        auto node_type = NodeNameType[name];

        for ( const auto &[ret_name, sig]: v["Signature"] ) 
        {
            // TODO: potentially check signature and filter out types
            auto ret_type = DataNameType[ret_name];
            nodes.insert(Node(node_type, ret_type, true));
            nodes.insert(Node(node_type, ret_type, false));
             
        }
    }
}

NodeVector generate_terminals(const Data& d)
{
    NodeVector terminals;
    int i = 0;
    for ( const auto &[key, value]: d.featues ) 
    {
        auto data_type = d.data_types.at(i);
        // note: structured bindings cannot be captured by lambdas until C++20
        std::cout << "generating terminal " << key << "of type " << value << endl;
        // terminals.push_back(make_shared<Terminal>(var, d[var]));

        terminals.push_back(Node(NodeType::Terminal, data_type, name));
        ++i;
    };
    return terminals;
};

} //Brush
