#include "search_space.h"

namespace Brush{

template<typename T>
tuple<set<NodeBase*>,set<type_index>> generate_nodes(vector<string>& op_names)
{

    // NodeNameTypeMap name2ret2node;
    set<NodeBase*> nodes; 
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
tuple<set<NodeBase*>,set<type_index>> generate_split_nodes(vector<string>& op_names)
{
    set<NodeBase*> nodes; 
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

set<NodeBase*> generate_all_nodes(vector<string>& node_names, 
                                  set<type_index> term_types)
{
    set<NodeBase*> nodes; 
    set<type_index> new_types;

    auto [new_nodes, nt] = generate_nodes<ArrayXf>(node_names);
    auto [new_nodes2, nt2] = generate_nodes<ArrayXXf>(node_names);
    auto [new_nodes3, nt3] = generate_split_nodes(node_names);
    nodes.merge(new_nodes);
    term_types.merge(nt);
    // term_types.erase(t);
    // generate nodes that act on the terminals, and on any new return 
    // types from the nodes encountered along the way.
    // while(term_types.size() > 0)
    // for (auto t: term_types)
    // {
    //     type_index t = *term_types.begin();
    //     string tn = type_names.at(t);

    //     // auto [new_nodes, nt] = generate_nodes<bool>(node_names);
    //     // nodes.merge(new_nodes);
    //     // term_types.merge(nt);
    //     // term_types.erase(t);

    //     // if (tn == "bool")
    //     // {
    //     //     auto [new_nodes, nt] = generate_nodes<bool>(node_names);
    //     //     nodes.merge(new_nodes);
    //     //     term_types.merge(nt);
    //     //     term_types.erase(t);
    //     // }
    //     // else if (tn == "int")
    //     // {
    //     //     auto [new_nodes, nt] = generate_nodes<int>(node_names);
    //     //     nodes.merge(new_nodes);
    //     //     term_types.merge(nt);
    //     //     term_types.erase(t);
    //     // }
    //     // else if (tn == "float")
    //     // {
    //     //     auto [new_nodes, nt] = generate_nodes<float>(node_names);
    //     //     nodes.merge(new_nodes);
    //     //     term_types.merge(nt);
    //     //     term_types.erase(t);
    //     // }
    //     // if (tn == "ArrayXb")
    //     // {
    //     //     // auto [new_nodes, nt] = generate_nodes<ArrayXb>(node_names);
    //     //     // nodes.merge(new_nodes);
    //     //     // term_types.merge(nt);
    //     //     // term_types.erase(t);
    //     // }
    //     // else if (tn == "ArrayXi")
    //     // {
    //     //     // auto [new_nodes, nt] = generate_nodes<ArrayXi>(node_names);
    //     //     // nodes.merge(new_nodes);
    //     //     // term_types.merge(nt);
    //     //     // term_types.erase(t);
    //     // }
    //     // else if (tn == "ArrayXf")
    //     // {
    //     //     auto [new_nodes, nt] = generate_nodes(node_names);
    //     //     nodes.merge(new_nodes);
    //     //     term_types.merge(nt);
    //     //     term_types.erase(t);
    //     // }
    //     // else if (tn == "Longitudinal")
    //     // {
    //     //     auto [new_nodes, nt] = generate_nodes<Longitudinal>(node_names);
    //     //     nodes.merge(new_nodes);
    //     //     term_types.merge(nt);
    //     //     term_types.erase(t);
    //     // }
    // }
    return nodes;
}

NodeVector generate_terminals(const Data& d)
{
    NodeVector terminals;
    for (const auto& kv : d.features)
    {
        // note: structured bindings cannot be captured by lambdas until C++20
        const string& name = kv.first;
        const State& value = kv.second;
        std::cout << "generating terminal " << name << endl;
        // terminals.push_back(make_shared<Terminal>(var, d[var]));

        std::visit([&](auto && v){
            std::cout << name << ":" << v << endl;
            },
            value
        );
        std::visit([&](auto && v){
            terminals.push_back(new Terminal(name, v));
            },
            value
        );

    };
    return terminals;
};

} //Brush