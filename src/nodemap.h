/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODELIST_H
#define NODELIST_H
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

template<typename... Ts>
struct SearchSpace
{
    using TypeInfoRef = std::reference_wrapper<const std::type_info>; 

    // typedef std::pair<NodeBase*, float> NodeWeightPair;
    typedef std::map<TypeInfoRef, 
                     map<TypeInfoRef, 
                         map<string, NodeBase*>>> NodeMap;
    NodeMap node_map; 
    // three levels of weights:
    // weights at the return type level
    typedef std::map<TypeInfoRef, float> return_weights;
    // weights at the arg type level
    typedef std::map<TypeInfoRef, float> args_weights;
    // weights at the operator name level
    typedef std::map<string, float> name_weights;

    SearchSpace(set<vector<NodeBase*>>& node_set, 
                vector<TypeInfoRef>& data_types,
                const map<string,float>& user_ops = {},
               )
    {
        bool use_all = user_ops.size() == 0;

        this->node_map.clear();

        for (const auto& nodes : node_set)
        {
            for (const auto& n: nodes)
            {
                // the node name needs to be in user_ops
                // the node return or argument type needs to be in data_types
                if (use_all || user_ops.find(n->name) != user_ops.end() )
                {
                    if(!in(data_types, n->ret_type)) 
                    {
                        bool pass = false;
                        for (const auto& a : n->arg_types) 
                        {
                            pass = pass || in(data_types, a);
                        }
                        if (!pass)
                            continue;
                    }
                }
                // initialize weights
                if (return_weights.find(n->ret_type) == return_weights.end())
                    return_weights[n->ret_type] = 0;
                if (args_weights.find(n->args_type) == args_weights.end())
                    args_weights[n->args_type] = 0;
                if (name_weights.find(n->name) == name_weights.end())
                    name_weights[n->name] = 0;


                // add the node to the nodemap
                this->node_map[n->ret_type][typeid(n->arg_types)][n->name] = n;

                // update weights
                float w = use_all? 1.0 : user_ops.at(n->name);
                this->return_weights[n->ret_type] += w;
                this->args_weights[n->args_type] += w;
                this->name_weights[n->name] += w;
                // this->weight_map[n->ret_type][typeid(n->arg_types)] = 1.0;

            }
        }

    };

    ~SearchSpace()
    {
        for (auto it = node_map.cbegin(); it != node_map.cend(); )
        {
            delete it->second;
            node_map.erase(it++);    // or "it = m.erase(it)" since C++11
        }
    };


    /// get any node 
    NodeBase* get()
    {
        return r.random_choice(r.random_choice(r.random_choice(node_map, 
                                                               return_weights), 
                                               args_weights), 
                               name_weights);
    }

    /// get a node with matching return type
    NodeBase* get_ret_type_like(const type_info& ret)
    {
        auto matches = node_map[ret];
        return r.random_choice(r.random_choice(matches, args_weights),
                               name_weights);
    }

    // NodeBase* get_like(const NodeBase*& node)
    // {
    //     return get_like(static_cast<TypedNodeBase<)
    // }
    /// get a node wth matching return type and argument types
    NodeBase* get_node_like(const NodeBase*& node)
    {
        auto matches = node_map[node->ret_type][node->args_type];
        return r.random_sample(matches, name_weights);
    }

    // NodeBase* operator[](const std::string& op)
    // { 
    //     if (node_map.find(op) == node_map.end())
    //         std::cerr << "ERROR: couldn't find " << op << endl;
        
    //     return this->node_map.at(op); 
    // };
};

map<string,  NodeBase*> mapify(vector<NodeBase*>& nodes)
{
    map<string,  NodeBase*> node_map;
    for (const auto& n : nodes)
        node_map[n->name] = n;
    return node_map;
}

template<typename T>
vector<NodeBase*> make_continuous_nodes<T>()
{
    auto nodes = vector<NodeBase*>{

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
}

template<typename N, typename T>
vector<NodeBase*> make_dx_nodes<T>(bool weighted)
{
    return vector<NodeBase*> nodes = {
            new N<T(T,T)>("+", 
                    std::plus<T>(), 
                    d_plus<T>()
                    ),
            new N<T(T,T)>("-", 
                    std::minus<T>(),
                    d_minus<T>()
                    ),
            new N<T(T,T)>("*", 
                    std::multiplies<T>(),
                    d_multiplies<T>()
                    ),
            new N<T(T,T)>("/", 
                    std::divides<T>(),
                    d_divides<T>()
                    ),
            new N<T(T)>("sin", 
                    [](const T& x) -> T {return sin(x);},
                    [](const T& x) -> array<T,1>{return {-cos(x)};}
                    ),
            new N<T(T)>("cos", 
                    [](const T& x) -> T {return cos(x);},
                    [](const T& x) -> array<T,1>{return {sin(x)};}
                    ),
            new N<T(T)>("tanh", 
                    [](const T& x) -> T {return tanh(x);},
                    [](const T& x) -> array<T,1>
                        { return {1 - pow(tanh(x), 2)}; }
                    ),
            new N<T(T)>("exp", 
                    [](const T& x) -> T {return exp(x);},
                    [](const T& x) -> array<T,1>{return {exp(x)};}
                    ),
            new N<T(T)>("log", 
                    safe_log<T>(),
                    d_safe_log<T>()
                    ),
            new N<T(T)>("sqrt", 
                    [](const T& x) -> T { return sqrt(abs(x)); },
                    [](const T& x) -> array<T,1> {
                        return {x/(2*sqrt(abs(x)))}; }
                    ),
            new N<T(T)>("^2", 
                    [](const T& x) -> T {return pow(x, 2);},
                    [](const T& x) -> array<T,1> {return {2*x}; }
                    ),
            new N<T(T)>("^3", 
                    [](const T& x) -> T {return pow(x, 3);},
                    [](const T& x) -> array<T,1> {return {3*pow(x, 2)}; }
                    ),
            new N<T(T,T)>("^", 
                    [](const T& lhs, const T& rhs) -> T {return pow(lhs, rhs);},
                    [](const T& lhs, const T& rhs) -> array<T,2> {
                        return {rhs * pow(lhs, rhs-1), 
                                log(lhs) * pow(lhs, rhs)}; 
                        }
                    ),
            new N<T(T)>("logit", 
                    [](const T& x) -> T {return 1/(1+exp(-x));},
                    [](const T& x) -> array<T,1> {
                        return { exp(-x)/pow(1+exp(-x),2) }; }
                    ),
            new N<T(T)>("relu", 
                    relu<T>(),
                    d_relu<T>()
                    ),
       };
       /* TODO / potential adds:*/
            /* { "gauss",  new Node<T(T)>(gauss, "gauss")}, */ 
            /* { "gauss2d",  new Node<T(T,T)>(gauss2d, "gauss2d")}, */ 
};


// template<typename T>

template<typename T, typename U>
map<string, vector<NodeBase*>> make_logical_nodes<T,U>()
{
    return vector<NodeBase*> {
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
}
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

template<typename T>
vector<NodeBase*> make_split_nodes<T>()
{
    return vector<NodeBase*>{
            // Split on best feature
            new SplitNode<T(T,T)>("best_split"),
            // Split on first argument
            new SplitNode<T(T,T,T)>("arg_split")
            };
}

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
auto CNM = make_dx_nodes<DxNode,ArrayXf>();
auto WCN = make_dx_nodes<WeightedDxNode,ArrayXf>();
// auto LN = make_logical_nodes<ArrayXb, ArrayXf>();
auto SN = make_split_nodes<ArrayXf>();
auto RM = make_reduced_nodes<float, ArrayXf>();

// ReduceMap<Eigen::VectorwiseOp<VectorXf,1>, MatrixXf> VectorReduceMap;
/* LogicalNodeMap<bool, float> BoolLogicMap; */
/* define the search space */
vector<TypeInfoRef> data_types = { 
                                  typeid(ArrayXf), 
                                  typeid(ArrayXb), 
                                  typeid(ArrayXi), 
                                 };
static SearchSpace SS(std::set<vector<NodeBase*>>{
                        CNM, 
                        WCN, 
                        LN, 
                        SN, 
                        RM
                    },

                 );
/* NM.node_map.insert(VectorArithmeticMap.node_map.begin(), */
/*                    VectorArithmeticMap.node_map.end()); */
/* NM.node_map.insert(VectorLogicMap.node_map.begin(), */
/*                    VectorLogicMap.node_map.end()); */

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
