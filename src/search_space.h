/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef SEARCHSPACE_H 
#define SEARCHSPACE_H
//internal includes
#include "init.h"
#include "node.h"
#include "nodemap.h"
#include "operator.h"
#include "util/utils.h"
#include "util/rnd.h"
#include <utility>

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
/* using namespace Brush::nodes; */
using namespace Brush::data;
using Brush::Node;
using Brush::DataType;
using std::type_index; 

typedef vector<Node> NodeVector;

namespace Brush
{
////////////////////////////////////////////////////////////////////////////////
// node generation routines
/* template<typename T> */
/* tuple<set<Node>,set<type_index>> generate_nodes(vector<string>& op_names); */
/* tuple<set<Node>,set<type_index>> generate_split_nodes(vector<string>& op_names); */

NodeVector generate_terminals(const Data& d);

set<Node> generate_all_nodes(vector<string>& node_names, 
                             set<DataType> term_types);
////////////////////////////////////////////////////////////////////////////////


struct SearchSpace
{

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
     *
     *
     * Params
     *
     * node_map: Maps return types to argument types to operator names. 
     *  schema:
     *      { return_type : { arguments_type : {node_type : node } }}
     *
     * terminal_map: Maps return types to terminals. 
     *      { return_type : vector of Nodes } 
     *
     * terminal_types: A set of the available terminal types. 
    */
    
    map<DataType,           // return type
        map<size_t,         // hash of arg types
            map<NodeType,   // node type (name)
                Node        // the node!
                >>> node_map;
    // NodeMap node_map; 
    map<DataType, NodeVector> terminal_map;
    map<DataType, vector<float>> terminal_weights;
    set<DataType> terminal_types;
    // terminal weights
    // map name to weights
    map<DataType,           // return type
        map<size_t,         // hash of arg types
            map<NodeType,   // node type (name)
                float       // the weight
                >>> weight_map; 
    SearchSpace(){};

    void init(const Data& d, 
              const map<string,float>& user_ops = {}
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

        vector<Node> terminals = generate_terminals(d);
        /* set<Node> nodes = generate_all_nodes(op_names, terminal_types); */
        
        cout << "generate nodemap\n";
        GenerateNodeMap(user_ops, std::make_index_sequence<NodeTypes::Count>());
        // map terminals
        cout << "looping through terminals...\n";
        for (const auto& term : terminals)
        {
            cout << "adding " << term.get_name() << " to search space...\n";
            if (terminal_map.find(term.ret_type) == terminal_map.end())
                terminal_map[term.ret_type] = NodeVector();
            cout << "terminal ret_type: " << DataTypeName[term.ret_type] << "\n";
            terminal_map[term.ret_type].push_back(term);
            terminal_weights[term.ret_type].push_back(1.0);
        }

        cout << "terminal map: " << terminal_map.size() << "\n";
        for (const auto& [k, v] : terminal_map)
        {
            cout << DataTypeName[k] << ": ";
            print(v.begin(), v.end());
        }

        cout << "node map: " << node_map.size() << "\n";
        for (const auto& [ret_type, v] : node_map)
        {
            for (const auto& [args_type, v2] : v)
            {
                for (const auto& [name, nodeval] : v2)
                {
                    cout << "node_map[" << DataTypeName[ret_type] 
                        << "][args_type][" << NodeTypeName[name] << "] = " 
                        << nodeval.get_name() 
                        /* << nodeval.ID */
                        << endl;
                }

            }
        }
        cout << "done.\n";

    };

    // template<typename R>
    template<typename F> Node get(const string& name);

    Node get(const NodeType type, DataType R, vector<DataType>& arg_types)
    {
         auto arg_hash = uint32_vector_hasher()(arg_types);
         return node_map.at(R).at(arg_hash).at(type);
    };

    /// get a terminal 
    Node get_terminal() const
    {
        //TODO: match terminal args_type (probably '{}' or something?)
        //  make a separate terminal_map
        auto match = *r.select_randomly(terminal_map.begin(), terminal_map.end());
        return *r.select_randomly(
                match.second.begin(), match.second.end(), 
                terminal_weights.at(match.first).begin(), 
                terminal_weights.at(match.first).end()
                );
    };
    /// get a typed terminal 
    Node get_terminal(DataType ret) const
    {
        cout << "get terminal of type " << DataTypeName[ret] << "\n";
        cout << "terminal map: " << terminal_map.size() << "\n";
        for (const auto& [k, v] : terminal_map)
        {
            cout << DataTypeName[k] << ": ";
            print(v.begin(), v.end());
        }
        print(terminal_weights.at(ret).begin(), terminal_weights.at(ret).end());
        //TODO: match terminal args_type (probably '{}' or something?)
        //  make a separate terminal_map
        auto rval =  *r.select_randomly(terminal_map.at(ret).begin(), 
                                  terminal_map.at(ret).end(), 
                                  terminal_weights.at(ret).begin(),
                                  terminal_weights.at(ret).end());
        cout << "returning " << rval.get_name() << endl;
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

    vector<float> get_weights(DataType ret) const
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
    vector<float> get_weights(DataType ret, size_t args_hash) const
    {
        // returns a weight vector, each element corresponding to an args type.
        vector<float> v;
        for (const auto& [name, w]: weight_map.at(ret).at(args_hash))
            v.push_back(w); 

        return v;
    };
    /// get an operator 
    Node get_op(DataType ret) const
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
    Node get_op_with_arg(DataType ret, DataType arg, 
                              bool terminal_compatible=true) const
    {
        // terminal_compatible: the other args the op takes must exist in the
        // terminal types. 

        auto args_map = node_map.at(ret);
        vector<Node> matches;
        vector<float> weights; 

        for (const auto& [args_type, name_map]: args_map)
        {
            for (const auto& [name, node]: name_map)
            {
                auto node_arg_types = node.arg_types;
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
    Node get_node_like(Node node) const
    {
        auto matches = node_map.at(node.ret_type).at(node.args_type());
        auto match_weights = get_weights(node.ret_type, node.args_type());
        return (*r.select_randomly(matches.begin(), 
                                   matches.end(), 
                                   match_weights.begin(), 
                                   match_weights.end())
               ).second;
    };

    // Node operator[](const std::string& op)
    // { 
    //     if (node_map.find(op) == node_map.end())
    //         std::cerr << "ERROR: couldn't find " << op << endl;
        
    //     return this->node_map.at(op); 
    // };
    private:
        template<NodeType NT, typename S>
        static constexpr auto MakeNode(bool weighted)
        {
            using RetType = typename S::RetType; 
            DataType output_type = DataTypeEnum<RetType>::value;
            std::size_t sig_hash = typeid(S).hash_code();
            return Node(NT, sig_hash, output_type, weighted);

        }
       
        template<NodeType NT, typename S>
        constexpr void AddNode(const map<string,float>& user_ops)
        {
            bool use_all = user_ops.size() == 0;
            auto name = NodeTypeName.at(NT);
            auto n = MakeNode<NT,S>(false);
            if (n.IsWeighable())
            {
                n.is_weighted=true; // weighted
            }
            node_map[n.ret_type][n.args_type()][n.node_type] = n;
            float w = use_all? 1.0 : user_ops.at(name);
            weight_map[n.ret_type][n.args_type()][n.node_type] = w;
        }

        template<NodeType NT, typename Sigs, std::size_t... Is>
        constexpr void AddNodes(const map<string,float>& user_ops, std::index_sequence<Is...>)
        {
            (AddNode<NT,std::tuple_element_t<Is, Sigs>>(user_ops), ...);
        }

        template<NodeType NT>
        void MakeNodes(const map<string,float>& user_ops)
        {
            bool use_all = user_ops.size() == 0;
            auto name = NodeTypeName.at(NT);

            if (!use_all & user_ops.find(name) == user_ops.end())
                return;

            /* constexpr auto signatures = Signatures<NT>::value; */
            using signatures = Signatures<NT>::type;
            constexpr auto size = std::tuple_size<signatures>::value ;
            AddNodes<NT, signatures>(user_ops, std::make_index_sequence<size>()); 
        }

        template<std::size_t... Is>
        void GenerateNodeMap(const map<string,float>& user_ops, std::index_sequence<Is...> )
        {
            auto nt = [](auto i) { return static_cast<NodeType>(1UL << i); };
            (MakeNodes<nt(Is)>(user_ops), ...);
        }
};

extern SearchSpace SS;


} // Brush
#endif
