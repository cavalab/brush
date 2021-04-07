/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODEMAP_H
#define NODEMAP_H
//internal includes
#include "node.h"
#include "operators.h"
#include "util/utils.h"
#include "util/rnd.h"
using std::vector;
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

typedef vector<NodeBase*> NodeVector;

namespace Brush
{


struct SearchSpace
{
    // using type_index = std::reference_wrapper<const std::type_info>; 

    // typedef std::pair<NodeBase*, float> NodeWeightPair;
    // typedef std::map<type_index, 
    //                  map<type_index, 
    //                      map<string, NodeBase*> Hasher, EqualTo>, Hasher, EqualTo> NodeMap;
    TypeMap<TypeMap<map<string, NodeBase*>>> node_map;
    // NodeMap node_map; 
    TypeMap<NodeVector> terminal_map;
    // terminal weights
    TypeMap<vector<float>> terminal_weights;
    // TypeMap<float> ret_w_map;
    // TypeMap<vector<float>> args_w_map; // maps return type to weights for args
    // TypeMap<TypeMap<map<string, float>>> name_w_map; // map name to weights
    TypeMap<TypeMap<map<string, float>>> weight_map; // map name to weights

    SearchSpace(set<NodeVector>& node_set, 
                NodeVector& terminals,
                vector<type_index>& data_types,
                const map<string,float>& user_ops = {}
                //add terminal_set
               )
    {
        cout << "constructing search space...\n";

        bool use_all = user_ops.size() == 0;

        this->node_map.clear();

        for (const auto& nodes : node_set)
        {
            for (const auto& n: nodes)
            {
                // the node name needs to be in user_ops
                // the node return or argument type needs to be in data_types
                if (! use_all)
                    if (user_ops.find(n->name) == user_ops.end() )
                        continue;
                // if(!in(data_types, n->ret_type())) 
                // {
                //     bool pass = false;
                //     for (const auto& a : n->arg_types()) 
                //     {
                //         pass = pass || in(data_types, a);
                //     }
                //     if (!pass)
                //         continue;
                // }
                // // initialize weights
                // if (ret_w_map.find(n->ret_type()) == ret_w_map.end())
                //     ret_w_map[n->ret_type()] = 0;
                // if (args_w_map.find(n->args_type()) == args_w_map.end())
                //     args_w_map[n->args_type()] = 0;
                // if (name_w_map.find(n->name) == name_w_map.end())
                //     name_w_map[n->name] = 0;

                cout << "adding " << n->name << " = " 
                     << n->ret_type().name() << "(" 
                     << n->args_type().name() << ") to search space...\n";
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
    /// get specific node by name.
    template<typename R, typename... Args>
    NodeBase* get(const string& name, R, Args...)
    {
         typedef std::tuple<Args...> TupleArgs;
         return node_map.at(typeid(R)).at(typeid(TupleArgs)).at(name);

        // auto matches = node_map[ret];
        // vector<NodeBase*> results;
        // for (const auto& [key, arg_map] : node_map[ret])
        // {
        //     for (const auto& [key, arg_map] : node_map[ret])
        //     {

        //     }
        //     f
        //     if (in(arg_map, name))

        // }

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

    /// get any node 
    // NodeBase* get()
    // {
    //     return r.random_choice(r.random_choice(r.random_choice(node_map, 
    //                                                            ret_weights), 
    //                                            args_weights), 
    //                            name_weights);
    // }

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

    // /// get a node with matching return type
    // NodeBase* get_ret_type_like(const type_info& ret)
    // {
    //     auto matches = node_map.at(ret);
    //     return r.random_choice(r.random_choice(matches, args_weights),
    //                            name_weights);
    // }

    // NodeBase* get_like(const NodeBase*& node)
    // {
    //     return get_like(static_cast<TypedNodeBase<)
    // }
    /// get a node wth matching return type and argument types
    NodeBase* get_node_like(const NodeBase*& node) const
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

extern SearchSpace SS;

map<string,  NodeBase*> mapify(NodeVector& nodes)
{
    map<string,  NodeBase*> node_map;
    for (const auto& n : nodes)
        node_map[n->name] = n;

    return node_map;
};

template<typename T>
NodeVector make_continuous_nodes()
{
    return NodeVector{

            new Node<T(T,T)>("+", std::plus<T>()),
            new Node<T(T,T)>("-", std::minus<T>()),
            new Node<T(T,T)>("*", std::multiplies<T>()) 
            /* { "/", Node<T(T,T)>(Op::safe_divide<T>, "DIV") }, */
            /* { "sqrt",  new Node<T(T)>(sqrt, "sqrt")}, */ 
            /* { "sin",  new Node<T(T)>(sin, "sin")}, */ 
            /* { "cos",  new Node<T(T)>(cos, "cos")}, */ 
            /* { "tanh",  new Node<T(T)>(tanh, "tanh")}, */ 
            /* { "^2",  new Node<T(T)>(square, "^2")}, */ 
            /* { "^3",  new Node<T(T)>(cube, "^3")}, */ 
            /* { "^",  new Node<T(T)>(^, "^")}, */ 
            /* { "exp",  new Node<T(T)>(exp, "exp")}, */ 
            /* { "gauss",  new Node<T(T)>(gauss, "gauss")}, */ 
            /* { "gauss2d",  new Node<T(T,T)>(gauss2d, "gauss2d")}, */ 
            /* { "log", new Node<T(T)>(log, "log") }, */   
            /* { "logit", new Node<T(T)>(logit, "logit") }, */
            /* { "relu", new Node<T(T)>(relu, "relu") } */
    };
};

//TODO: define argument packs that can feed into different node types. 
// something like map<T>(name, fn, d_fn, inv_fn, arity)

// template<typename T> //, template<typename> typename N>
// NodeVector make_weighted_dx_nodes()
// {
//     cout << "making weighted dx nodes...\n";
//     return {
//             new WeightedDxNode<T(T,T)>("ADD", 
//                     std::plus<T>(), 
//                     d_plus<T>()
//                     ),
//             new WeightedDxNode<T(T,T)>("MINUS", 
//                     std::minus<T>(),
//                     d_minus<T>()
//                     ),
//             new WeightedDxNode<T(T,T)>("TIMES", 
//                     std::multiplies<T>(),
//                     d_multiplies<T>()
//                     ),
//             new WeightedDxNode<T(T,T)>("DIV", 
//                     std::divides<T>(),
//                     d_divides<T>()
//                     ),
//             new WeightedDxNode<T(T)>("sin", 
//                     [](const T& x) -> T {return sin(x);},
//                     [](const T& x) -> array<T,1>{return {-cos(x)};}
//                     ),
//             new WeightedDxNode<T(T)>("cos", 
//                     [](const T& x) -> T {return cos(x);},
//                     [](const T& x) -> array<T,1>{return {sin(x)};}
//                     ),
//             new WeightedDxNode<T(T)>("tanh", 
//                     [](const T& x) -> T {return tanh(x);},
//                     [](const T& x) -> array<T,1>
//                         { return {1 - pow(tanh(x), 2)}; }
//                     ),
//             new WeightedDxNode<T(T)>("exp", 
//                     [](const T& x) -> T {return exp(x);},
//                     [](const T& x) -> array<T,1>{return {exp(x)};}
//                     ),
//             new WeightedDxNode<T(T)>("log", 
//                     safe_log<T>(),
//                     d_safe_log<T>()
//                     ),
//             new WeightedDxNode<T(T)>("sqrt", 
//                     [](const T& x) -> T { return sqrt(abs(x)); },
//                     [](const T& x) -> array<T,1> {
//                         return {x/(2*sqrt(abs(x)))}; }
//                     ),
//             new WeightedDxNode<T(T)>("square", 
//                     [](const T& x) -> T {return pow(x, 2);},
//                     [](const T& x) -> array<T,1> {return {2*x}; }
//                     ),
//             new WeightedDxNode<T(T)>("cube", 
//                     [](const T& x) -> T {return pow(x, 3);},
//                     [](const T& x) -> array<T,1> {return {3*pow(x, 2)}; }
//                     ),
//             new WeightedDxNode<T(T,T)>("pow", 
//                     [](const T& lhs, const T& rhs) -> T {return pow(lhs, rhs);},
//                     [](const T& lhs, const T& rhs) -> array<T,2> {
//                         return {rhs * pow(lhs, rhs-1), 
//                                 log(lhs) * pow(lhs, rhs)}; 
//                         }
//                     ),
//             new WeightedDxNode<T(T)>("logit", 
//                     [](const T& x) -> T {return 1/(1+exp(-x));},
//                     [](const T& x) -> array<T,1> {
//                         return { exp(-x)/pow(1+exp(-x),2) }; }
//                     ),
//             new WeightedDxNode<T(T)>("relu", 
//                     relu<T>(),
//                     d_relu<T>()
//                     ),
//        };
//        /* TODO / potential adds:*/
//             /* { "gauss",  new Node<T(T)>(gauss, "gauss")}, */ 
//             /* { "gauss2d",  new Node<T(T,T)>(gauss2d, "gauss2d")}, */ 
// };


// template<typename T>

template<typename T, typename U>
NodeVector make_logical_nodes()
{
    cout << "making logical nodes...\n";
    return {
            new Node<T(U,U)>("<", lt<T,U>) 
            /* { "and", new Node<T(U,U)>(Op::plus<T>, "AND") }, */
            /* { "or", new Node<T(U,U)>(Op::minus<T>, "OR") }, */
            /* { "not", new Node<T(U,U)>(Op::multiplies<T>, "NOT") }, */
            /* { "xor", new Node<T(U,U)>(Op::divides<T>, "XOR") }, */
            /* { "<=", new Node<T(U,U)>(Op::leq<U>, "LESS") }, */
            /* { "=",  new Node<T(U,U)>(Op::equal<U>, "EQUAL")}, */ 
            /* { ">",  new Node<T(U,U)>(Op::gt<U>, ">")}, */ 
            /* { ">=",  new Node<T(U,U)>(Op::geq<U>, ">")}, */ 
    };
};
// template<typename T, typename U>
// struct LogicalNodeMap : NodeMap
// {
//     LogicalNodeMap() 
//     {
//        this->node_map = {
//             { "<", new Node<T(U,U)>("<", lt<T,U>) },
//         };
//             /* { "and", new Node<T(U,U)>(Op::plus<T>, "AND") }, */
//             /* { "or", new Node<T(U,U)>(Op::minus<T>, "OR") }, */
//             /* { "not", new Node<T(U,U)>(Op::multiplies<T>, "NOT") }, */
//             /* { "xor", new Node<T(U,U)>(Op::divides<T>, "XOR") }, */
//             /* { "<=", new Node<T(U,U)>(Op::leq<U>, "LESS") }, */
//             /* { "=",  new Node<T(U,U)>(Op::equal<U>, "EQUAL")}, */ 
//             /* { ">",  new Node<T(U,U)>(Op::gt<U>, ">")}, */ 
//             /* { ">=",  new Node<T(U,U)>(Op::geq<U>, ">")}, */ 
//     };
// };

template<typename T, typename U = T>
NodeVector make_split_nodes()
{
    cout << "making split nodes...\n";
    return {
            // Split on best feature
            new SplitNode<T(T,T)>("best_split"),
            // Split on first argument
            new SplitNode<T(U,T,T)>("arg_split")
            };
};

// template<typename T>
// struct SplitNodeMap : NodeMap
// {
//     SplitNodeMap() 
//     { 
//        this->node_map = {
//                // Automatic thresholding
//             { "split_"+type_names[typeid(T)], new SplitNode<T(T,T)>(
//                      "split_"+type_names[typeid(T)] )},
//                 // Split on first argument
//             { "split_"+type_names[typeid(T)], new SplitNode<T(T,T,T)>(
//                      "split3_"+type_names[typeid(T)] )},
//        };
//     };
// };

// template<typename R, typename T>
// struct ReduceMap : NodeMap
// {
//     ReduceMap()
//     { 
//        this->node_map = {
//             // longitudinal nodes
//             { "mean", new Node<R(T)>("mean", &T::mean) }
//        };
//             /* { "median", new ReduceNode<float(ArrayXf,ArrayXf)>() }, */
//             /* { "max", new ReduceNode<float(ArrayXf,ArrayXf)>() }, */
//             /* { "min", new MinNode<ArrayXf(ArrayXf,ArrayXf)>() }, */
//             /* { "variance", new VarNode<ArrayXf(ArrayXf,ArrayXf)>() }, */
//             /* { "skew", new SkewNode<ArrayXf(ArrayXf,ArrayXf)>() }, */
//             /* { "kurtosis", new KurtosisNode<ArrayXf(ArrayXf,ArrayXf)>() }, */
//             /* { "slope", new SlopeNode<ArrayXf(ArrayXf,ArrayXf)>() }, */
//             /* { "count", new CountNode<ArrayXf(ArrayXf,ArrayXf)>() }, */
//             /* { "recent", new RecentNode<ArrayXf(ArrayXf,ArrayXf)>() }, */
//     };
// };

/* Declare node maps 
 *
 */
// ContinuousNodeMap<ArrayXf> VectorArithmeticMap; 
// WeightedNodeMap<ArrayXf> DxMap; 
// ContinuousNodeMap<float> FloatNodeMap; 
// LogicalNodeMap<ArrayXb, ArrayXf> VectorLogicMap;
// SplitNodeMap<ArrayXf> SNMfloat;
// ReduceMap<float, ArrayXf> ScalarReduceMap;
// auto CNM = make_continuous_nodes<ArrayXf>();
// auto CNM = make_weighted_dx_nodes<ArrayXf>();
// auto WCN = make_dx_nodes<WeightedDxNode,ArrayXf>();
// auto LN = make_logical_nodes<ArrayXb, ArrayXf>();
// auto SN = make_split_nodes<ArrayXf>();
// auto RM = make_reduced_nodes<float, ArrayXf>();

// ReduceMap<Eigen::VectorwiseOp<VectorXf,1>, MatrixXf> VectorReduceMap;
/* LogicalNodeMap<bool, float> BoolLogicMap; */
/* define the search space */
/* NM.node_map.insert(VectorArithmeticMap.node_map.begin(), */
/*                    VectorArithmeticMap.node_map.end()); */
/* NM.node_map.insert(VectorLogicMap.node_map.begin(), */
/*                    VectorLogicMap.node_map.end()); */

vector<type_index> data_types = { 
                                typeid(ArrayXf), 
                                // typeid(ArrayXb), 
                                // typeid(ArrayXi), 
                                };
ArrayXf x1, x2;
NodeVector terminals = { new Terminal("x1",x1), new Terminal("x2",x2) };

// std::set<NodeVector> node_set = {
//                         // make_weighted_dx_nodes<ArrayXf>(),
//                         // make_weighted_dx_nodes<ArrayXi>(),
//                         // make_split_nodes<ArrayXf>(),
//                         // make_split_nodes<ArrayXf,ArrayXi>(),
//                         // make_split_nodes<ArrayXf,ArrayXb>(),
//                         // make_split_nodes<ArrayXi>(),
//                         // make_split_nodes<ArrayXi,ArrayXf>(),
//                         // make_split_nodes<ArrayXi,ArrayXb>(),
//                         // make_split_nodes<ArrayXb>(),
//                         // make_split_nodes<ArrayXb,ArrayXf>(),
//                         // make_split_nodes<ArrayXb,ArrayXi>()
//                     };

// SearchSpace SS(node_set, terminals, data_types);

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
