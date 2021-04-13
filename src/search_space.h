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

tuple<set<NodeBase*>,set<type_index>> generate_nodes(vector<string>& op_names);
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
NodeVector generate_terminals(const Data& d);

set<NodeBase*> generate_all_nodes(vector<string>& node_names, 
                                  set<type_index> term_types);


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
        for (const auto& dt : d.data_types)
            this->terminal_types.insert(dt);

        vector<NodeBase*> terminals = generate_terminals(d);
        set<NodeBase*> nodes = generate_all_nodes(op_names, terminal_types);

        int i = 0;
        for (const auto& n: nodes)
        {
            cout << "adding " << n->name << ") to search space...\n";
            // add the node to the nodemap
            this->node_map[n->ret_type()][n->args_type()][n->op_name] = n;
            
            // update weights
            float w = use_all? 1.0 : user_ops.at(op_names.at(i));
            this->weight_map[n->ret_type()][n->args_type()][n->op_name] = w;

            // this->ret_w_map[n->ret_type()] += w;
            // this->args_w_map[n->args_type()] += w;
            // this->name_w_map[n->name] = w;
            // this->weight_map[n->ret_type][typeid(n->args_type)] = 1.0;
            ++i;

        }
        // map terminals
        for (const auto& term : terminals)
        {
            cout << "adding " << term->get_name() << ") to search space...\n";
            if (terminal_map.find(term->ret_type()) == terminal_map.end())
                terminal_map[term->ret_type()] = NodeVector();
            cout << "terminal ret_type: " << type_names[term->ret_type()] << "\n";
            terminal_map[term->ret_type()].push_back(term);
            terminal_weights[term->ret_type()].push_back(1.0);
        }

        cout << "terminal map: " << terminal_map.size() << "\n";
        for (const auto& [k, v] : terminal_map)
        {
            cout << type_names[k] << ": ";
            print(v.begin(), v.end());
        }
        cout << "node map: " << node_map.size() << "\n";
        for (const auto& [ret_type, v] : node_map)
        {
            for (const auto& [args_type, v2] : v)
            {
                for (const auto& [name, nodeval] : v2)
                {
                    cout << "node_map[" << type_names[ret_type] 
                        << "][args_type][" << name << "] = " 
                        << nodeval->get_name() << endl;
                }

            }
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
            for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
            {
                for (auto it3 = it2->second.begin(); it3 != it2->second.end(); ++it3)
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
        cout << "get terminal of type " << type_names[ret] << "\n";
        cout << "terminal map: " << terminal_map.size() << "\n";
        for (const auto& [k, v] : terminal_map)
        {
            cout << type_names[k] << ": ";
            print(v.begin(), v.end());
        }
        print(terminal_weights.at(ret).begin(), terminal_weights.at(ret).end());
        //TODO: match terminal args_type (probably '{}' or something?)
        //  make a separate terminal_map
        auto rval =  *r.select_randomly(terminal_map.at(ret).begin(), 
                                  terminal_map.at(ret).end(), 
                                  terminal_weights.at(ret).begin(),
                                  terminal_weights.at(ret).end());
        cout << "returning " << rval->get_name() << endl;
        return rval;
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

extern SearchSpace SS;


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
