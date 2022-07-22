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
#include "tree.h"
#include "util/utils.h"
#include "util/rnd.h"
#include "params.h"
#include <utility>
#include <optional>


/* Defines the search space of Brush. 
 *  The search spaces consists of nodes and their accompanying probability
 *  distribution. 
 *  Nodes can be accessed by type, signature, or a combination.  
 *  You may also sample the search space by return type. 
 *  Sampling is done in proportion to the weight associated with 
 *  each node. By default, sampling is done uniform randomly.
*/
using namespace Brush::data;
using namespace Brush::Util; 
using Brush::Node;
using Brush::DataType;
using std::type_index; 


namespace Brush
{
////////////////////////////////////////////////////////////////////////////////
// node generation routines
/* template<typename T> */
/* tuple<set<Node>,set<type_index>> generate_nodes(vector<string>& op_names); */
/* tuple<set<Node>,set<type_index>> generate_split_nodes(vector<string>& op_names); */

// forward declarations
using TreeIter = tree<Node>::pre_order_iterator;
template<typename T> struct Program;

vector<Node> generate_terminals(const Data& d);

////////////////////////////////////////////////////////////////////////////////

extern std::unordered_map<std::size_t, std::string> ArgsName; 

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
    using ArgsHash = std::size_t; 

    template<typename T>
    using Map = unordered_map<DataType,           // return type
                    unordered_map<ArgsHash,         // hash of arg types
                        unordered_map<NodeType,   // node type 
                            T>>>;        // the data!
    Map<Node> node_map;
    Map<float> weight_map; 

    unordered_map<DataType, vector<Node>> terminal_map;
    unordered_map<DataType, vector<float>> terminal_weights;
    vector<DataType> terminal_types;
    
    template<typename T>
    Program<T> make_program(int max_d=0, int max_breadth=0, int max_size=0);
    
    
    void init(const Data& d, const unordered_map<string,float>& user_ops = {});

    // template<typename R>
    template<typename F> Node get(const string& name);

    Node get(const NodeType type, DataType R, size_t sig_hash)
    {
         /* auto arg_hash = uint32_vector_hasher()(arg_types); */
         return node_map.at(R).at(sig_hash).at(type);
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
        /* cout << "get terminal of type " << DataTypeName[ret] << "\n"; */
        /* /1* cout << "terminal map: " << terminal_map.size() << "\n"; *1/ */
        /* for (const auto& [k, nv] : terminal_map) */
        /* { */
        /*     fmt::print("{}: ", DataTypeName[k]); */
        /*     for (auto n : nv) */
        /*         fmt::print("{}, ", n.get_name()); */
        /*     fmt::print("\n"); */
        /* } */
        /* print(terminal_weights.at(ret).begin(), terminal_weights.at(ret).end()); */
        /* fmt::print("{}\n",terminal_weights.at(ret)); */
        //TODO: match terminal args_type (probably '{}' or something?)
        //  make a separate terminal_map
        auto rval =  *r.select_randomly(terminal_map.at(ret).begin(), 
                                  terminal_map.at(ret).end(), 
                                  terminal_weights.at(ret).begin(),
                                  terminal_weights.at(ret).end());
        /* cout << "returning " << rval.get_name() << endl; */
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
    vector<float> get_weights(DataType ret, ArgsHash sig_hash) const
    {
        // returns a weight vector, each element corresponding to an args type.
        vector<float> v;
        for (const auto& [name, w]: weight_map.at(ret).at(sig_hash))
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
    // thoughts (TODO):
    //  this could be templated by return type and arg. although the lookup in the map should be
    //  fairly fast. 
    Node get_op_with_arg(DataType ret, DataType arg, 
                              bool terminal_compatible=true) const
    {
        // terminal_compatible: the other args the returned operator takes must exist in the
        // terminal types. 

        auto args_map = node_map.at(ret);
        vector<Node> matches;
        vector<float> weights; 

        for (const auto& [args_type, name_map]: args_map) {
            for (const auto& [name, node]: name_map) {
                auto node_arg_types = node.arg_types;
                if ( in(node_arg_types, arg) ) {
                    // if checking terminal compatibility, make sure there's
                    // a compatible terminal for the node's other arguments
                    if (terminal_compatible) {
                        bool compatible = true;
                        for (const auto& arg_type: node_arg_types) { 
                            if (arg_type != arg) {
                                if ( ! in(terminal_types, arg_type) ) {
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

    private:
        /* template<typename T, > */
        /* static constexpr bool contains() { return is_one_of_v<T, Args...>; } */
        /* static constexpr auto MakeNode(bool weighted) */
        template<NodeType NT, typename S>
        requires (!is_one_of_v<NT, NodeType::Terminal, NodeType::Constant>)
        static constexpr std::optional<Node> CreateNode(const auto& unique_data_types, bool use_all, bool weighted)
        {
            // if we're using all, prune the operators out that don't have argument types that
            // overlap with feature data types
            if (use_all) {
                for (auto arg: S::get_arg_types()){
                    if (! in(unique_data_types,arg) ){
                        /* fmt::print("not adding {} because {} is not in unique_data_types\n", NT, arg); */
                        return {}; 
                    }
                }    
            }
            using RetType = typename S::RetType; 
            DataType output_type = DataTypeEnum<RetType>::value;
            /* auto args_hash = typeid(S).hash_code(); */
            auto sig_hash = S::hash();
            ArgsName[sig_hash] = fmt::format("{}", S::get_arg_types());
            auto arg_types = S::get_arg_types();
            return Node(NT, arg_types, sig_hash, output_type, weighted);
        }
       
        template<NodeType NT, typename S>
        constexpr void AddNode(const unordered_map<string,float>& user_ops, 
                const vector<DataType>& unique_data_types
                )
        {
            
            bool use_all = user_ops.size() == 0;
            auto name = NodeTypeName[NT];
            //TODO: address this (whether weights should be included by default)
            bool weighted = IsWeighable<NT>();
            auto n_maybe = CreateNode<NT,S>(unique_data_types, use_all, weighted);

            if (n_maybe){
                auto n = n_maybe.value();
                node_map[n.ret_type][n.args_type()][n.node_type] = n;
                // sampling probability map
                float w = use_all? 1.0 : user_ops.at(name);
                weight_map[n.ret_type][n.args_type()][n.node_type] = w;
            }
        }

        template<NodeType NT, typename Sigs, std::size_t... Is>
        constexpr void AddNodes(const unordered_map<string,float>& user_ops, 
                const vector<DataType>& unique_data_types,
                std::index_sequence<Is...>)
        {
            (AddNode<NT,std::tuple_element_t<Is, Sigs>>(user_ops, unique_data_types), ...);
        }

        template<NodeType NT>
        void MakeNodes(const unordered_map<string,float>& user_ops, 
                       const vector<DataType>& unique_data_types
                      ) 
        {
            if (Is<NodeType::Terminal, NodeType::Constant>(NT))
                return;
            bool use_all = user_ops.size() == 0;
            auto name = NodeTypeName.at(NT);

            if (!use_all & user_ops.find(name) == user_ops.end())
            {
                /* cout << "skipping " << name << ", not in user_ops\n"; */ 
                return;
            }

            using signatures = Signatures<NT>::type;
            constexpr auto size = std::tuple_size<signatures>::value;
            AddNodes<NT, signatures>(user_ops, unique_data_types, std::make_index_sequence<size>()); 
        }

        template<std::size_t... Is>
        void GenerateNodeMap(const unordered_map<string,float>& user_ops, 
                const vector<DataType>& unique_data_types, 
                std::index_sequence<Is...> 
                )
        {
            auto nt = [](auto i) { return static_cast<NodeType>(1UL << i); };
            (MakeNodes<nt(Is)>(user_ops, unique_data_types), ...);
        }
}; // SearchSpace

/// queue for make program
template<typename T>
T RandomDequeue(std::vector<T>& Q)
{
    int loc = r.rnd_int(0, Q.size()-1);
    std::swap(Q[loc], Q[Q.size()-1]);
    T val = Q.back();
    Q.pop_back();
    return val;
};
// constructs a tree using functions, terminals, and settings
template<typename T>
Program<T> SearchSpace::make_program(int max_d, int max_breadth, int max_size)
{
    /*
    * implementation of PTC2 for strongly typed GP from Luke et al. 
    * "Two fast tree-creation algorithms for genetic programming"
    *  
    */
    if (max_d == 0)
        max_d = r.rnd_int(1, params.max_depth);
    if (max_breadth == 0)
        max_breadth = r.rnd_int(1, params.max_breadth);
    if (max_size == 0)
        max_size = r.rnd_int(1, params.max_size);
    DataType root_type = DataTypeEnum<T>::value;

    auto prg = tree<Node>();

    /* fmt::print("building program with max size {}, max depth {}",max_size,max_d); */ 

    // Queue of nodes that need children
    vector<tuple<TreeIter, DataType, int>> queue; 

    if (max_size == 1)
    {
        auto root = prg.insert(prg.begin(), get_terminal(root_type));
    }
    else
    {
        /* cout << "getting op of type " << DataTypeName[root_type] << endl; */
        auto n = get_op(root_type);
        /* cout << "chose " << n.name << endl; */
        // auto spot = prg.set_head(n);
        /* cout << "inserting...\n"; */
        auto spot = prg.insert(prg.begin(), n);
        // node depth
        int d = 1;
        // current tree size
        int s = 1;
        //For each argument position a of n, Enqueue(a; g) 
        for (auto a : n.arg_types)
        { 
            /* cout << "queing a node of type " << DataTypeName[a] << endl; */
            queue.push_back(make_tuple(spot, a, d));
        }

        /* cout << "queue size: " << queue.size() << endl; */ 
        /* cout << "entering first while loop...\n"; */
        while (queue.size() + s < max_size && queue.size() > 0) 
        {
            /* cout << "queue size: " << queue.size() << endl; */ 
            auto [qspot, t, d] = RandomDequeue(queue);

            /* cout << "current depth: " << d << endl; */
            if (d == max_d)
            {
                /* cout << "getting " << DataTypeName[t] << " terminal\n"; */ 
                prg.append_child(qspot, get_terminal(t));
            }
            else
            {
                //choose a nonterminal of matching type
                /* cout << "getting op of type " << DataTypeName[t] << endl; */
                auto n = get_op(t);
                /* cout << "chose " << n.name << endl; */
                TreeIter new_spot = prg.append_child(qspot, n);
                // For each arg of n, add to queue
                for (auto a : n.arg_types)
                {
                    /* cout << "queing a node of type " << DataTypeName[a] << endl; */
                    queue.push_back(make_tuple(new_spot, a, d+1));
                }
            }
            ++s;
            /* cout << "current tree size: " << s << endl; */
        } 
        /* cout << "entering second while loop...\n"; */
        while (queue.size() > 0)
        {
            if (queue.size() == 0)
                break;

            /* cout << "queue size: " << queue.size() << endl; */ 

            auto [qspot, t, d] = RandomDequeue(queue);

            /* cout << "getting " << DataTypeName[t] << " terminal\n"; */ 
            prg.append_child(qspot, get_terminal(t));

        }
    }
    /* cout << "final tree:\n" */ 
    /*     << prg.begin().node->get_model() << "\n" */
    /*     << prg.begin().node->get_tree_model(true) << endl; */
         /* << prg.get_model() << "\n" */ 
         /* << prg.get_model(true) << endl; // pretty */

    return Program<T>(*this,prg);
};;

extern SearchSpace SS;

} // Brush

// format overload 
template <> struct fmt::formatter<Brush::SearchSpace>: formatter<string_view> {
  template <typename FormatContext>
  auto format(const Brush::SearchSpace& SS, FormatContext& ctx) const {
    string output = "Search Space\n===\n";
    output += fmt::format("terminal_map: {}\n", SS.terminal_map);
    output += fmt::format("terminal_weights: {}\n", SS.terminal_weights);
    for (const auto& [ret_type, v] : SS.node_map) {
        for (const auto& [args_type, v2] : v) {
            for (const auto& [node_type, node] : v2) {
                output += fmt::format("node_map[{}][{}][{}] = {}, weight = {}\n", 
                        ret_type,
                        ArgsName[args_type],
                        node_type,
                        node,
                        SS.weight_map.at(ret_type).at(args_type).at(node_type)
                        );
            }
        }
    }
    output += "===";
    return formatter<string_view>::format(output, ctx);
  }
};
#endif
