/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef SEARCHSPACE_H 
#define SEARCHSPACE_H
//internal includes
#include "init.h"
#include "node.h"
#include "operators.h"
#include "util/utils.h"
#include "util/rnd.h"

/* TODO
* instead of specifying keys, values, just specify the values into a list of
* some kind, and then loop thru the list and construct a map using the node
* name as the key.
*/
/* template<typename R, typename Arg1, typename Arg2=Arg1> */
/* Defines the search space of Brush. 
 *  The search spaces consists of nodes and their accompanying probability
 *  distribution. 
 *  Nodes can be accessed by name using a string map. 
 *  Alternatively, the functions and terminal sets may be sampled separately
 *  or together. 
 *  You may also sample the search space by return type. 
 *  Sampling is done in proportion to the weight associated with 
 *  each node. By default, sampling is done uniform randomly.
*/
using namespace Brush::nodes;
using namespace Brush::data;

typedef vector<NodeBase*> NodeVector;

namespace Brush
{

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
        nodes.insert( new WeightedDxNode<ArrayXf(ArrayXf)>(op->name, 
                                                           op->f, 
                                                           op->df));

    if ( in(op_names, "best_split"))
        nodes.insert(new SplitNode<ArrayXf(ArrayXf,ArrayXf)>("best_split"));

    if ( in(op_names, "arg_split"))
    {
        nodes.insert( new SplitNode<ArrayXf(ArrayXf,ArrayXf,ArrayXf)>("arg_split"));
    }

    return {nodes, new_types};
};
//TODO: add reduction operators
/// generates nodes that act on T. return any non-T return types.
// template<typename T>
// tuple<set<NodeBase*>,set<type_index>> generate_nodes(vector<string>& op_names)
// {

//     // NodeNameTypeMap name2ret2node;
//     set<NodeBase*> nodes; 
//     set<type_index> new_types;

//     auto binary_operators = make_binary_operators<T>(op_names);
//     auto unary_operators = make_unary_operators<T>(op_names);
//     // auto reduce_operators = make_reduction_operators<T>();


//     for (const auto& op : binary_operators)
//         nodes.insert( new WeightedDxNode<T(T,T)>(op->name, op->f, op->df));

//     for (const auto& op : unary_operators)
//         nodes.insert( new WeightedDxNode<T(T)>(op->name, op->f, op->df));

//     if ( in(op_names, "best_split"))
//         nodes.insert(new SplitNode<T(T,T)>("best_split"));

//     if ( in(op_names, "arg_split"))
//     {
//         if (typeid(T) == typeid(ArrayXb))
//         {
//             nodes.insert(
//                     new SplitNode<ArrayXf(T,ArrayXf,ArrayXf)>("arg_split")
//             );
//             new_types.insert(typeid(ArrayXf));
//         }
//     }

//     return {nodes, new_types};
// };

// template<>
// tuple<set<NodeBase*>,set<type_index>> generate_nodes<Longitudinal>(vector<string>& op_names)
// {
//     return {{},{}};
// };
NodeVector generate_terminals(const Data& d)
{
    NodeVector terminals;
    for (const auto& var : d.var_names)
    {
        // terminals.push_back(make_shared<Terminal>(var, d[var]));
        terminals.push_back(new Terminal(var, d[var]));
    };
    return terminals;
};

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

struct SearchSpace
{
    TypeMap<TypeMap<map<string, NodeBase*>>> node_map;
    // NodeMap node_map; 
    TypeMap<NodeVector> terminal_map;
    set<type_index> terminal_types;
    // terminal weights
    TypeMap<vector<float>> terminal_weights;
    // map name to weights
    TypeMap<TypeMap<map<string, float>>> weight_map; 

    /* Construct a search space, consisting of operations and terminals
     * and functions that sample the space. 
     * The set of operators is a user controlled parameter; however, we can 
     * automate, to some extent, the set of possible operators based on the 
     * data types in the problem. 
     * Constraints on operators based on data types: 
     *  - only user specified operators are included. 
     *  - operators whose arguments are covered by terminal types are included
     *      first. Then, a second pass includes any operators whose arguments
     *      are covered by terminal_types + return types of the current set of 
     *      operators. One could imagine this continuing ad infinitum, but we
     *      just do two passes for simplicity. 
     *  - assertion check to make sure there is at least one operator that 
     *      returns the output type of the model. 
    */
    SearchSpace(){};

    void init(const Data& d, 
                const map<string,float>& user_ops = {}
                //add terminal_set
               )
    {
        cout << "constructing search space...\n";

        bool use_all = user_ops.size() == 0;
        vector<string> op_names;
        for (const auto& [op, weight] : user_ops)
            op_names.push_back(op);

        this->node_map.clear();

        // create nodes based on data types 
        for (const auto& dt : d.data_types)
            this->terminal_types.insert(dt);

        vector<NodeBase*> terminals = generate_terminals(d);
        set<NodeBase*> nodes = generate_all_nodes(op_names, terminal_types);

        for (const auto& n: nodes)
        {
            cout << "adding " << n->name << ") to search space...\n";
            // add the node to the nodemap
            this->node_map[n->ret_type()][n->args_type()][n->name] = n;
            
            // update weights
            float w = use_all? 1.0 : user_ops.at(n->name);
            this->weight_map[n->ret_type()][n->args_type()][n->name] = w;

            // this->ret_w_map[n->ret_type()] += w;
            // this->args_w_map[n->args_type()] += w;
            // this->name_w_map[n->name] = w;
            // this->weight_map[n->ret_type][typeid(n->args_type)] = 1.0;

        }
        // map terminals
        for (const auto& term : terminals)
        {
            if (terminal_map.find(term->ret_type()) == terminal_map.end())
                terminal_map[term->ret_type()] = NodeVector();

            terminal_map[term->ret_type()].push_back(term);
            terminal_weights[term->ret_type()].push_back(1.0);
        }

        cout << "done.\n";


    };

    // template<typename R>
    template<typename F> NodeBase* get(const string& name);
    /// get specific node by name and type.
    template<typename R, typename... Args>
    NodeBase* get(const string& name, R, Args...)
    {
         typedef std::tuple<Args...> TupleArgs;
         return node_map.at(typeid(R)).at(typeid(TupleArgs)).at(name);
    }

    ~SearchSpace()
    {
        for (auto it = node_map.begin(); it != node_map.end(); )
        {
            for (auto it2 = it->second.begin(); it2 != it->second.end(); )
            {
                for (auto it3 = it2->second.begin(); 
                     it3 != it2->second.end(); )
                {
                    // delete the NodeBase* pointer
                    delete it3->second;
                }
            }
            node_map.erase(it++);  
        }
        for (auto it = terminal_map.begin(); it != terminal_map.end(); )
        {
            for (auto it2 : it->second)
            {
                delete it2;
            }
            terminal_map.erase(it++);   
        }
    };

    /// get a terminal 
    NodeBase* get_terminal() const
    {
        //TODO: match terminal args_type (probably '{}' or something?)
        //  make a separate terminal_map
        auto match = *r.select_randomly(terminal_map.begin(), 
                                        terminal_map.end());
        return *r.select_randomly(match.second.begin(),
                                  match.second.end(), 
                                  terminal_weights.at(match.first).begin(), 
                                  terminal_weights.at(match.first).end());
    };
    /// get a typed terminal 
    NodeBase* get_terminal(type_index ret) const
    {
        //TODO: match terminal args_type (probably '{}' or something?)
        //  make a separate terminal_map
        return *r.select_randomly(terminal_map.at(ret).begin(), 
                                  terminal_map.at(ret).end(), 
                                  terminal_weights.at(ret).begin(),
                                  terminal_weights.at(ret).end());
    };

    vector<float> get_weights() const
    {
        // returns a weight vector, each element corresponding to a return type.
        vector<float> v;
        for (auto& [ret, arg_w_map]: weight_map) 
        {
            v.push_back(0);
            for (const auto& [arg, name_map] : arg_w_map)
            {
                for (const auto& [name, w]: name_map)
                {
                    v.back() += w; 
                }

            }
        }
        return v;
    };

    vector<float> get_weights(type_index ret) const
    {
        // returns a weight vector, each element corresponding to an args type.
        vector<float> v;
        for (const auto& [arg, name_map] : weight_map.at(ret))
        {
            v.push_back(0);
            for (const auto& [name, w]: name_map)
            {
                v.back() += w; 
            }

        }
        return v;
    };
    vector<float> get_weights(type_index ret, type_index args) const
    {
        // returns a weight vector, each element corresponding to an args type.
        vector<float> v;
        for (const auto& [name, w]: weight_map.at(ret).at(args))
            v.push_back(w); 

        return v;
    };
    /// get an operator 
    NodeBase* get_op(type_index ret) const
    {
        //TODO: match terminal args_type (probably '{}' or something?)
        auto ret_match = node_map.at(ret);

        vector<float> args_w = get_weights(ret);

        auto arg_match = *r.select_randomly(ret_match.begin(), 
                                            ret_match.end(), 
                                            args_w.begin(), 
                                            args_w.end());

        vector<float> name_w = get_weights(ret, arg_match.first);
        return (*r.select_randomly(arg_match.second.begin(), 
                                   arg_match.second.end(), 
                                   name_w.begin(), 
                                   name_w.end())).second;
    };

    // get operator with at least one argument matching arg 
    NodeBase* get_op_with_arg(type_index ret, type_index arg, 
                              bool terminal_compatible=true) const
    {
        // terminal_compatible: the other args the op takes must exist in the
        // terminal types. 

        auto args_map = node_map.at(ret);
        vector<NodeBase*> matches;
        vector<float> weights; 

        for (const auto& [args_type, name_map]: args_map)
        {
            for (const auto& [name, node]: name_map)
            {
                auto node_arg_types = node->arg_types();
                if ( in(node_arg_types, arg) )
                {
                    // if checking terminal compatibility, make sure there's
                    // a compatible terminal for the node's other arguments
                    if (terminal_compatible)
                    {
                        bool compatible = true;
                        for (const auto& arg_type: node_arg_types)
                        { 
                            if (arg_type != arg)
                            {
                                if ( ! in(terminal_types, arg_type) )
                                {
                                    compatible = false;
                                    break;
                                }
                            }
                        }
                        if (! compatible)
                            continue;

                    }
                    // if we made it this far, include the node as a match!
                    matches.push_back(node);
                    weights.push_back(weight_map.at(ret).at(args_type).at(name));
                }
            }
        }

        return (*r.select_randomly(matches.begin(), matches.end(), 
                                   weights.begin(), weights.end()));
    };

    /// get a node wth matching return type and argument types
    NodeBase* get_node_like(NodeBase* node) const
    {
        auto matches = node_map.at(node->ret_type()).at(node->args_type());
        auto match_weights = get_weights(node->ret_type(), node->args_type());
        return (*r.select_randomly(matches.begin(), 
                                   matches.end(), 
                                   match_weights.begin(), 
                                   match_weights.end())).second;
    };

    // NodeBase* operator[](const std::string& op)
    // { 
    //     if (node_map.find(op) == node_map.end())
    //         std::cerr << "ERROR: couldn't find " << op << endl;
        
    //     return this->node_map.at(op); 
    // };
};

static SearchSpace SS;


} // Brush
#endif

// Old nodes from FEAT:
    /* { "+",  new NodeAdd({1.0,1.0})}, */ 
    /* { "-",  new NodeSubtract({1.0,1.0})}, */ 
    /* { "*",  new NodeMultiply({1.0,1.0})}, */ 
    /* { "/",  new NodeDivide({1.0,1.0})}, */ 
    /* { "sqrt",  new NodeSqrt({1.0})}, */ 
    /* { "sin",  new NodeSin({1.0})}, */ 
    /* { "cos",  new NodeCos({1.0})}, */ 
    /* { "tanh",  new NodeTanh({1.0})}, */ 
    /* { "^2",  new NodeSquare({1.0})}, */ 
    /* { "^3",  new NodeCube({1.0})}, */ 
    /* { "^",  new NodeExponent({1.0})}, */ 
    /* { "exp",  new NodeExponential({1.0})}, */ 
    /* { "gauss",  new NodeGaussian({1.0})}, */ 
    /* { "gauss2d",  new Node2dGaussian({1.0, 1.0})}, */ 
    /* { "log", new NodeLog({1.0}) }, */   
    /* { "logit", new NodeLogit({1.0}) }, */
    /* { "relu", new NodeRelu({1.0}) }, */
    /* { "b2f", new NodeFloat<bool>() }, */
    /* { "c2f", new NodeFloat<int>() }, */
    
    /* // logical operators */
    /* { "and", new NodeAnd() }, */
    /* { "or", new NodeOr() }, */
    /* { "not", new NodeNot() }, */
    /* { "xor", new NodeXor() }, */
    /* { "=", new NodeEqual() }, */
    /* { ">", new NodeGreaterThan() }, */
    /* { ">=", new NodeGEQ() }, */        
    /* { "<", new NodeLessThan() }, */
    /* { "<=", new NodeLEQ() }, */
    /* { "split", new NodeSplit<float>() }, */
    /* { "fuzzy_split", new NodeFuzzySplit<float>() }, */
    /* { "fuzzy_fixed_split", new NodeFuzzyFixedSplit<float>() }, */
    /* { "split_c", new NodeSplit<int>() }, */
    /* { "fuzzy_split_c", new NodeFuzzySplit<int>() }, */
    /* { "fuzzy_fixed_split_c", new NodeFuzzyFixedSplit<int>() }, */
    /* { "if", new NodeIf() }, */   	    		
    /* { "ite", new NodeIfThenElse() }, */
    /* { "step", new NodeStep() }, */
    /* { "sign", new NodeSign() }, */
       
    /* // longitudinal nodes */
    /* { "mean", new NodeMean() }, */
    /* { "median", new NodeMedian() }, */
    /* { "max", new NodeMax() }, */
    /* { "min", new NodeMin() }, */
    /* { "variance", new NodeVar() }, */
    /* { "skew", new NodeSkew() }, */
    /* { "kurtosis", new NodeKurtosis() }, */
    /* { "slope", new NodeSlope() }, */
    /* { "count", new NodeCount() }, */
    /* { "recent", new NodeRecent() }, */
