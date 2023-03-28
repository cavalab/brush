/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef SEARCHSPACE_H 
#define SEARCHSPACE_H
//internal includes
#include "init.h"
#include "program/node.h"
#include "program/nodetype.h"
#include "program/tree_node.h"
// #include "program/program.h"
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
using namespace Brush::Data;
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
// template<typename T> struct Program;
// enum class ProgramType: uint32_t;
// template<typename T> struct ProgramTypeEnum; 

vector<Node> generate_terminals(const Dataset& d);

////////////////////////////////////////////////////////////////////////////////

extern std::unordered_map<std::size_t, std::string> ArgsName; 

/*! @brief Holds a search space, consisting of operations and terminals
    * and functions, and methods to sample that space to create programs. 
    *
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
    * Parameters
    * ----------
    *
*/
struct SearchSpace
{
    using ArgsHash = std::size_t; 

    template<typename T>
    using Map = unordered_map<DataType,           // return type
                    unordered_map<ArgsHash,         // hash of arg types
                        unordered_map<NodeType,   // node type 
                            T>>>;        // the data!
    
    /**
     * @brief Maps return types to argument types to node types. 
     * 
     *  schema:
     * 
     *      { return_type : { arguments_type : {node_type : node } }}
     */
    Map<Node> node_map;

    /// @brief A map of weights corresponding to elements in @ref node_map, used to weight probabilities of each node being sampled from the map. 
    Map<float> weight_map; 

    /**
     * @brief Maps return types to terminals. 
     * 
     * schema:
     * 
     *      { return_type : vector of Nodes } 
     *
     * 
     */
    unordered_map<DataType, vector<Node>> terminal_map;

    /// @brief A map of weights corresponding to elements in @ref terminal_map, used to weight probabilities of each node being sampled from the map. 
    unordered_map<DataType, vector<float>> terminal_weights;

    /// @brief A vector storing the available return types of terminals. 
    vector<DataType> terminal_types;

    // serialization
#ifndef DOXYGEN_SKIP

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SearchSpace, 
        node_map,
        weight_map,
        terminal_map,
        terminal_weights,
        terminal_types
    ) 

#endif
    
    /**
     * @brief Makes a random program.
     * 
     * We use an implementation of PTC2 for strongly typed GP from 
     * 
     * Sean Luke. "Two fast tree-creation algorithms for genetic programming"
     * (https://doi.org/10.1109/4235.873237)
     * 
     * @tparam PT program type
     * @param max_d max depth of the program
     * @param max_size max size of the programd 
     * @return a program of type PTsize 
     * 
     */
    template<typename PT>
    PT make_program(int max_d=0, int max_size=0);

    /// @brief Makes a random regressor program. Convenience wrapper for @ref make_program
    /// @param max_d max depth of the program
    /// @param max_size max size of the program
    /// @return a regressor program 
    RegressorProgram make_regressor(int max_d = 0, int max_size = 0);

    /// @brief Makes a random classifier program. Convenience wrapper for @ref make_program
    /// @param max_d max depth of the program
    /// @param max_size max size of the program
    /// @return a classifier program 
    ClassifierProgram make_classifier(int max_d = 0, int max_size = 0);

    /// @brief Makes a random multiclass classifier program. Convenience wrapper for @ref make_program
    /// @param max_d max depth of the program
    /// @param max_size max size of the program
    /// @return a multiclass classifier program 
    MulticlassClassifierProgram make_multiclass_classifier(int max_d = 0, int max_size = 0);

    /// @brief Makes a random representer program. Convenience wrapper for @ref make_program
    /// @param max_d max depth of the program
    /// @param max_size max size of the program
    /// @return a representer program 
    RepresenterProgram make_representer(int max_d = 0, int max_size = 0);

    SearchSpace() = default;

    /// @brief Construct a search space
    /// @param d A dataset containing terminal definitions
    /// @param user_ops Optional user-provided dictionary of operators with their probability of being chosen
    SearchSpace(const Dataset& d, const unordered_map<string,float>& user_ops = {}){
        init(d,user_ops);
    }

    /// @brief Called by the constructor to initialize the search space
    /// @param d A dataset containing terminal definitions
    /// @param user_ops Optional user-provided dictionary of operators with their probability of being chosen
    void init(const Dataset& d, const unordered_map<string,float>& user_ops = {});

    /// @brief check if a return type is in the node map
    /// @param R data type
    /// @return true if it exists 
    bool check(DataType R) const {
        if (node_map.find(R) == node_map.end()){
            auto msg = fmt::format("{} not in node_map\n",R);
            HANDLE_ERROR_THROW(msg); 
        }
        return true;
    }

    /// @brief check if a function signature is in the search space
    /// @param R return type
    /// @param sig_hash signature hash
    /// @return true if it exists
    bool check(DataType R, size_t sig_hash) const
    {
        if (check(R)){
            if (node_map.at(R).find(sig_hash) == node_map.at(R).end()){
                auto msg = fmt::format("{} not in node_map.at({})\n", sig_hash, R);
                HANDLE_ERROR_THROW(msg); 
            }
        }
        return true;
    }

    /// @brief check if a typed Node is in the search space
    /// @param R return type
    /// @param sig_hash signature hash
    /// @param type the node type
    /// @return true if it exists
    bool check(DataType R, size_t sig_hash, NodeType type) const
    {
        if (check(R,sig_hash)){
            if (node_map.at(R).at(sig_hash).find(type) == node_map.at(R).at(sig_hash).end()){

                auto msg = fmt::format("{} not in node_map[{}][{}]\n",type, sig_hash, R);
                HANDLE_ERROR_THROW(msg); 
            }}
        return true;
    }

    template<typename F> Node get(const string& name);

    /// @brief get a typed node 
    /// @param type the node type 
    /// @param R the return type of the node
    /// @param sig_hash the signature hash of the node
    /// @return the matching [Node](@ref Node)
    Node get(NodeType type, DataType R, size_t sig_hash)
    {
        check(R, sig_hash, type);
        return node_map.at(R).at(sig_hash).at(type);
    };

    /// @brief get a typed node. 
    /// @tparam S the signature of the node, inferred.  
    /// @param type the node type 
    /// @param R the return type of the node
    /// @param sig the signature of the node 
    /// @return the matching Node 
    template<typename S>
    Node get(NodeType type, DataType R, S sig){ return get(type, R, sig.hash()); };

    /// @brief Get a specific node type that matches a return value. 
    /// @param type the node type
    /// @param R the return type
    /// @return A Node of type `type` with return type `R`. 
    Node get(NodeType type, DataType R)
    {
        check(R);
        auto ret_match = node_map.at(R);
        vector<Node> matches; 
        vector<float> weights; 
        for (const auto& kv: ret_match)
        {
            auto arg_hash = kv.first;
            auto node_type_map = kv.second;
            if (node_type_map.find(type) != node_type_map.end())
            {
                matches.push_back(node_type_map.at(type));
                weights.push_back(weight_map.at(R).at(arg_hash).at(type));
            }
        }

        return (*r.select_randomly(matches.begin(),
                                   matches.end(),
                                   weights.begin(),
                                   weights.end()));
    };

    /// get a random terminal 
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

    /// get a terminal with return type `R` 
    Node get_terminal(DataType R) const
    {
        if (terminal_map.find(R) == terminal_map.end()){
            auto msg = fmt::format("{} not in terminal_map\n",R);
            HANDLE_ERROR_THROW(msg); 
        }
        auto rval =  *r.select_randomly(terminal_map.at(R).begin(), 
                                  terminal_map.at(R).end(), 
                                  terminal_weights.at(R).begin(),
                                  terminal_weights.at(R).end());
        return rval;
    };

    /// @brief get weights of the return types 
    /// @return a weight vector, each element corresponding to a return type.
    vector<float> get_weights() const
    {
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

    /// @brief get weights of the argument types matching return type `ret`.
    /// @param ret return type
    /// @return a weight vector, each element corresponding to an args type. 
    vector<float> get_weights(DataType ret) const
    {
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

    /// @brief get the weights of nodes matching a signature. 
    /// @param ret return type
    /// @param sig_hash signature hash
    /// @return a weight vector, each element corresponding to a node.
    vector<float> get_weights(DataType ret, ArgsHash sig_hash) const
    {
        vector<float> v;
        for (const auto& [name, w]: weight_map.at(ret).at(sig_hash))
            v.push_back(w); 

        return v;
    };

    /// @brief get an operator matching return type `ret`. 
    /// @param ret return type
    /// @return a randomly chosen operator
    Node get_op(DataType ret) const
    {
        check(ret);
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

    
    /// @brief get operator with at least one argument matching arg 
    /// @param ret return type
    /// @param arg argument type to match
    /// @param terminal_compatible if true, the other args the returned operator takes must exist in the terminal types. 
    /// @return a matching operator. 
    Node get_op_with_arg(DataType ret, DataType arg, 
                              bool terminal_compatible=true) const
    {
        // thoughts (TODO):
        //  this could be templated by return type and arg. although the lookup in the map should be
        //  fairly fast. 
        //TODO: these needs to be overhauled 
        // fmt::print("get_op_with_arg");
        check(ret);

        auto args_map = node_map.at(ret);
        vector<Node> matches;
        vector<float> weights; 

        for (const auto& [args_type, name_map]: args_map) {
            for (const auto& [name, node]: name_map) {
                auto node_arg_types = node.get_arg_types();
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
    
    /// @brief get a node with a signature matching `node`
    /// @param node the node to match
    /// @return a Node 
    Node get_node_like(Node node) const
    {
        if (Is<NodeType::Terminal>(node.node_type)){
            return get_terminal(node.ret_type);
        }

        auto matches = node_map.at(node.ret_type).at(node.args_type());
        auto match_weights = get_weights(node.ret_type, node.args_type());
        return (*r.select_randomly(matches.begin(), 
                                   matches.end(), 
                                   match_weights.begin(), 
                                   match_weights.end())
               ).second;
    };

    /// @brief prints the search space map. 
    void print() const; 

    private:

        template<NodeType NT, typename S>
        requires (!is_in_v<NT, NodeType::Terminal, NodeType::Constant>)
        static constexpr std::optional<Node> CreateNode(
            const auto& unique_data_types, 
            bool use_all, 
            bool weighted
        )
        {
            // prune the operators out that don't have argument types that
            // overlap with feature data types
            for (auto arg: S::get_arg_types()){
                if (! in(unique_data_types,arg) ){
                    return {}; 
                }
            }    
            ArgsName[S::hash()] = fmt::format("{}", S::get_arg_types());
            return Node(NT, S{}, weighted);
        }
       
        template<NodeType NT, typename S>
        constexpr void AddNode(
            const unordered_map<string,float>& user_ops, 
            const vector<DataType>& unique_data_types
        )
        {
            
            bool use_all = user_ops.size() == 0;
            auto name = NodeTypeName[NT];
            //TODO: address this (whether weights should be included by default)
            bool weighted = (IsWeighable<NT>() && is_same_v<typename S::RetType::Scalar, float>);
            auto n_maybe = CreateNode<NT,S>(unique_data_types, use_all, weighted);

            if (n_maybe){
                auto n = n_maybe.value();
                node_map[n.ret_type][n.args_type()][n.node_type] = n;
                // sampling probability map
                float w = use_all? 1.0 : user_ops.at(name);
                weight_map[n.ret_type][n.args_type()][n.node_type] = w;
            }
        }

        template <NodeType NT, typename Sigs, std::size_t... Is>
        constexpr void AddNodes(const unordered_map<string, float> &user_ops,
                                const vector<DataType> &unique_data_types,
                                std::index_sequence<Is...>)
        {
            (AddNode<NT, std::tuple_element_t<Is, Sigs>>(user_ops, unique_data_types), ...);
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

            // skip operators not defined by user
            if (!use_all & user_ops.find(name) == user_ops.end())
                return;

            using signatures = Signatures<NT>::type;
            constexpr auto size = std::tuple_size<signatures>::value;
            AddNodes<NT, signatures>(
                user_ops,
                unique_data_types,
                std::make_index_sequence<size>()
            ); 
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

template<typename PT>
PT SearchSpace::make_program(int max_d, int max_size)
{
    if (max_d == 0)
        max_d = r.rnd_int(1, PARAMS["max_depth"].get<int>());
    if (max_size == 0)
        max_size = r.rnd_int(1, PARAMS["max_size"].get<int>());

    DataType root_type = DataTypeEnum<typename PT::TreeType>::value;
    ProgramType program_type = ProgramTypeEnum<PT>::value;

    auto Tree = tree<Node>();

    /* fmt::print("building program with max size {}, max depth {}",max_size,max_d); */ 

    // Queue of nodes that need children
    vector<tuple<TreeIter, DataType, int>> queue; 

    if (max_size == 1)
    {
        auto root = Tree.insert(Tree.begin(), get_terminal(root_type));
    }
    else
    {
        Node n;
        if (program_type == ProgramType::BinaryClassifier)
        {
            n = get(NodeType::Logistic, DataType::ArrayF, Signature<ArrayXf(ArrayXf)>());
            n.set_prob_change(0.0);
            n.fixed=true;
        }
        else if (program_type == ProgramType::MulticlassClassifier)
        {
            n = get(NodeType::Softmax, DataType::MatrixF);
            n.set_prob_change(0.0);
            n.fixed=true;
        }
        else
            n = get_op(root_type);

        /* cout << "chose " << n.name << endl; */
        // auto spot = Tree.set_head(n);
        /* cout << "inserting...\n"; */
        auto spot = Tree.insert(Tree.begin(), n);
        // node depth
        int d = 1;
        // current tree size
        int s = 1;
        //For each argument position a of n, Enqueue(a; g) 
        for (auto a : n.arg_types)
        { 
            /* cout << "queing a node of type " << DataTypeName[a] << endl; */
            auto child_spot = Tree.append_child(spot);
            queue.push_back(make_tuple(child_spot, a, d));
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
                // choose terminal of matching type
                /* cout << "getting " << DataTypeName[t] << " terminal\n"; */ 
                // qspot = get_terminal(t);
                Tree.replace(qspot, get_terminal(t));
                // Tree.append_child(qspot, get_terminal(t));
            }
            else
            {
                //choose a nonterminal of matching type
                /* cout << "getting op of type " << DataTypeName[t] << endl; */
                auto n = get_op(t);
                /* cout << "chose " << n.name << endl; */
                // TreeIter new_spot = Tree.append_child(qspot, n);
                // qspot = n;
                auto newspot = Tree.replace(qspot, n);
                // For each arg of n, add to queue
                for (auto a : n.arg_types)
                {
                    /* cout << "queing a node of type " << DataTypeName[a] << endl; */
                    // queue.push_back(make_tuple(new_spot, a, d+1));
                    auto child_spot = Tree.append_child(newspot);
                    queue.push_back(make_tuple(child_spot, a, d+1));
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
            // Tree.append_child(qspot, get_terminal(t));
            // qspot = get_terminal(t);
            auto newspot = Tree.replace(qspot, get_terminal(t));

        }
    }
    /* cout << "final tree:\n" */ 
    /*     << Tree.begin().node->get_model() << "\n" */
    /*     << Tree.begin().node->get_tree_model(true) << endl; */
         /* << Tree.get_model() << "\n" */ 
         /* << Tree.get_model(true) << endl; // pretty */

    return PT(*this,Tree);
};

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
