#include "search_space.h"

namespace Brush{

// template<typename T>
tuple<set<NodeBase*>,set<type_index>> generate_nodes(vector<string>& op_names)
{

    // NodeNameTypeMap name2ret2node;
    set<NodeBase*> nodes; 
    set<type_index> new_types;

    auto binary_operators = make_binary_operators<ArrayXf>()(op_names);
    // auto binary_operators = make_binary_operators(op_names);
    auto unary_operators = make_unary_operators<ArrayXf>()(op_names);
    // auto reduce_operators = make_reduction_operators<T>();


    for (const auto& op : binary_operators)
        nodes.insert( new WeightedDxNode<ArrayXf(ArrayXf,ArrayXf)>(op->name, 
                                                                   op->f, 
                                                                   op->df));

    for (const auto& op : unary_operators)
    {
        nodes.insert( new WeightedDxNode<ArrayXf(ArrayXf)>(op->name, 
                                                           op->f, 
                                                           op->df)
                    );
        // cout << "making TransformReduceDxNode " << op->name;
        // nodes.insert( new TransformReduceDxNode<ArrayXf(ArrayXf)>(op->name, 
        //                                                    op->f, 
        //                                                    op->df,
        //                                                    10)
        //             );
    }

    if ( in(op_names, "best_split"))
        nodes.insert(new SplitNode<ArrayXf(ArrayXf,ArrayXf)>("best_split"));

    if ( in(op_names, "arg_split"))
    {
        nodes.insert( new SplitNode<ArrayXf(ArrayXf,ArrayXf,ArrayXf)>("arg_split"));
    }

    return {nodes, new_types};
}

set<NodeBase*> generate_all_nodes(vector<string>& node_names, 
                                  set<type_index> term_types)
{
    set<NodeBase*> nodes; 
    set<type_index> new_types;

    auto [new_nodes, nt] = generate_nodes(node_names);
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
