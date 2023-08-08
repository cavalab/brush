#include "search_space.h"
#include "program/program.h"
#include <iostream>

namespace Brush{


float calc_initial_weight(const ArrayXf& value, const ArrayXf& y)
{
    // weights are initialized as the slope of the z-score of x and y.

    // If y has different length from X, we get a core dump here. 
    // TODO: need to make SS (or Datasaet) check for this when loading the data

    vector<char> dtypes = {'f', 'f'};

    MatrixXf data(value.size(), 2);

    data.col(0) << value;
    data.col(1) << y;

    Normalizer n(true);
    n.fit_normalize(data, dtypes); // normalize works row-wise

    // In slope function, argument order matters (if not normalized with z-score)
    // The feature should be the first value, and the true value the second
    // (it will divide covar(arg1, arg2) by var(arg2)).
    // Since z-score normalizes so mean=0 and std=1, then order doesnt matter
    float prob_change = std::abs(slope(data.col(0).array(),   // x
                                       data.col(1).array())); // y

    return prob_change;
}


/// @brief generate terminals from the dataset features and random constants.
/// @param d a dataset
/// @return a vector of nodes 
vector<Node> generate_terminals(const Dataset& d)
{
    vector<Node> terminals;
    int i = 0;
    for ( const auto &[feature_name, value]: d.features ) 
    {
        std::visit(
            [&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                using Scalar = typename T::Scalar;
                constexpr bool weighted = std::is_same_v<Scalar, float>;
                // if constexpr (T::Scalar == float)
                auto n = Node(
                    NodeType::Terminal, 
                    Signature<T()>{}, 
                    weighted,
                    feature_name
                );

                float prob_change = 1.0; // default value
                
                // if the value can be casted to float array, we can calculate slope
                if (std::holds_alternative<ArrayXf>(value)) 
                {
                    prob_change = calc_initial_weight(std::get<ArrayXf>(value), d.y);
                }
                else if (std::holds_alternative<ArrayXi>(value))
                {
                    // for each variable we create a one-vs-all binary variable, then
                    // calculate slope. Final value will be the average of slopes

                    auto tmp = std::get<ArrayXi>(value);

                    //get number of unique values
                    std::map<float, bool> uniqueMap;
                    for(int i = 0; i < tmp.size(); i++)
                        uniqueMap[(float)tmp(i)] = true;

                    ArrayXf slopes = ArrayXf::Ones(uniqueMap.size());
                    int slopesIterator = 0;
                    for (const auto& pair : uniqueMap)
                    {
                        auto one_vs_all = ArrayXf::Ones(tmp.size()).array() * (tmp.array()==pair.first).cast<float>();

                        slopes[slopesIterator++] = calc_initial_weight(one_vs_all, d.y);
                    }
                    
                    prob_change = slopes.mean();
                }
                else if (std::holds_alternative<ArrayXb>(value))
                {
                    auto tmp = std::get<ArrayXb>(value).template cast<float>();
                    prob_change = calc_initial_weight(tmp, d.y);
                }
                else
                {
                    auto msg = fmt::format("Brush coudn't calculate the initial weight of variable {}\n",feature_name);
                    HANDLE_ERROR_THROW(msg);
                }
                
                n.set_prob_change( prob_change );

                terminals.push_back(n);
            },
            value
        );
        ++i;
    };

    // iterate through terminals and take the average of values of same signature
    auto signature_avg = [terminals](DataType ret_type){
        float sum = 0.0;
        int count = 0;

        for (const auto& n : terminals) {
            if (n.ret_type == ret_type) {
                sum += n.get_prob_change();
                count++;
            }
        }

        return sum / count;
    };

    auto cXf = Node(NodeType::Constant, Signature<ArrayXf()>{}, true, "C");
    cXf.set_prob_change(signature_avg(cXf.ret_type));
    terminals.push_back(cXf);

    auto cXi = Node(NodeType::Constant, Signature<ArrayXi()>{}, true, "C");
    cXi.set_prob_change(signature_avg(cXi.ret_type));
    terminals.push_back(cXi);

    auto cXb = Node(NodeType::Constant, Signature<ArrayXb()>{}, false, "C");
    cXb.set_prob_change(signature_avg(cXb.ret_type));
    terminals.push_back(cXb);

    return terminals;
};

std::unordered_map<std::size_t, std::string> ArgsName; 

void SearchSpace::print() const { 
    std::cout << fmt::format("{}\n", *this) << std::flush; 
}

void SearchSpace::init(const Dataset& d, const unordered_map<string,float>& user_ops)
{
    // fmt::print("constructing search space...\n");
    this->node_map.clear();
    this->node_map_weights.clear();
    this->terminal_map.clear();
    this->terminal_types.clear();
    this->terminal_weights.clear();

    bool use_all = user_ops.size() == 0;
    vector<string> op_names;
    for (const auto& [op, weight] : user_ops)
        op_names.push_back(op);

    // create nodes based on data types 
    terminal_types = d.unique_data_types;

    vector<Node> terminals = generate_terminals(d);
    
    /* fmt::print("generate nodetype\n"); */
    GenerateNodeMap(user_ops, d.unique_data_types, 
                    std::make_index_sequence<NodeTypes::OpCount>());
    // map terminals
    /* fmt::print("looping through terminals...\n"); */
    for (const auto& term : terminals)
    {
        /* fmt::print("adding {} to search space...\n", term.get_name()); */
        if (terminal_map.find(term.ret_type) == terminal_map.end())
            terminal_map[term.ret_type] = vector<Node>();

        /* fmt::print("terminal ret_type: {}\n", DataTypeName[term.ret_type]); */
        terminal_map[term.ret_type].push_back(term);
        terminal_weights[term.ret_type].push_back(term.get_prob_change());
    }
};

std::optional<tree<Node>> SearchSpace::sample_subtree(Node root, int max_d, int max_size) const 
{
    // public interface to use PTC2 algorithm

    // PTC is designed to not fail (it will persistently try to find nodes with
    // sampling functions). In pop initialization, this shoudnt be a problem, but
    // during evolution, due to dynamic changes in node weights by the learners, 
    // it now may fail. We need to check, before calling it, that it has elements
    // in search space to sample
    auto ret_match = node_map.at(root.ret_type);

    vector<float> args_w = get_weights(root.ret_type);

    // at least one operator that matches the weight must have positive probability
    if (!has_solution_space(args_w.begin(), args_w.end()))
        return std::nullopt;

    if ( (terminal_map.find(root.ret_type) == terminal_map.end())
    ||   (!has_solution_space(terminal_weights.at(root.ret_type).begin(), 
                              terminal_weights.at(root.ret_type).end())) )
        return std::nullopt;

    // we should notice the difference between size of a PROGRAM and a TREE.
    // program count weights in its size, while the TREE structure dont. Wenever
    // using size of a program/tree, make sure you use the function from the correct class
    return PTC2(root, max_d, max_size);
};

tree<Node> SearchSpace::PTC2(Node root, int max_d, int max_size) const
{
    // PTC2 is agnostic of program type

    // A comment about PTC2 method:            
    // PTC2 can work with depth or size restriction, but it does not strictly
    // satisfies these conditions all time. Given a `max_size` and `max_depth`
    // parameters, the real maximum size that can occur is `max_size` plus the
    // highest operator arity, and the real maximum depth is `max_depth` plus one.

    auto Tree = tree<Node>();

    /* fmt::print("building program with max size {}, max depth {}",max_size,max_d); */ 

    // Queue of nodes that need children
    vector<tuple<TreeIter, DataType, int>> queue; 

    /* cout << "chose " << n.name << endl; */
    // auto spot = Tree.set_head(n);
    /* cout << "inserting...\n"; */
    auto spot = Tree.insert(Tree.begin(), root);
    // node depth
    int d = 1;
    // current tree size
    int s = 1;
    //For each argument position a of n, Enqueue(a; g) 
    for (auto a : root.arg_types)
    { 
        /* cout << "queing a node of type " << DataTypeName[a] << endl; */
        auto child_spot = Tree.append_child(spot);
        queue.push_back(make_tuple(child_spot, a, d));
    }

    Node n;
    // Now we actually start the PTC2 procedure to create the program tree
    /* cout << "queue size: " << queue.size() << endl; */ 
    /* cout << "entering first while loop...\n"; */
    while ( 3*(queue.size()-1) + s < max_size && queue.size() > 0) 
    {            
        // by default, terminals are weighted (counts as 3 nodes in program size).
        // since every spot in queue has potential to be a terminal, we multiply
        // its size by 3. Subtracting one due to the fact that this loop will
        // always insert a non terminal (which by default has weights off).
        // this way, we can have PTC2 working properly.
        
        /* cout << "queue size: " << queue.size() << endl; */ 
        auto [qspot, t, d] = RandomDequeue(queue);

        /* cout << "current depth: " << d << endl; */
        if (d == max_d)
        {
            // choose terminal of matching type
            /* cout << "getting " << DataTypeName[t] << " terminal\n"; */ 
            // qspot = sample_terminal(t);
            // Tree.replace(qspot, sample_terminal(t));
            // Tree.append_child(qspot, sample_terminal(t));

            auto opt = sample_terminal(t);
            while (!opt)
                opt = sample_terminal(t);

            // If we successfully get a terminal, use it
            n = opt.value();

            Tree.replace(qspot, n);
        }
        else
        {
            //choose a nonterminal of matching type
            /* cout << "getting op of type " << DataTypeName[t] << endl; */
            auto opt = sample_op(t);
            /* cout << "chose " << n.name << endl; */
            // TreeIter new_spot = Tree.append_child(qspot, n);
            // qspot = n;

            while (!opt)
                opt = sample_op(t);

            n = opt.value();
            
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

        // increment is different based on node weights
        ++s;
        if  (n.get_is_weighted())
            s += 2;

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
        // Tree.append_child(qspot, sample_terminal(t));
        // qspot = sample_terminal(t);
        // auto newspot = Tree.replace(qspot, sample_terminal(t));

        auto opt = sample_terminal(t);
        while (!opt) {
            opt = sample_terminal(t);
        }

        n = opt.value();
        
        auto newspot = Tree.replace(qspot, n);
    }

    /* cout << "final tree:\n" */ 
    /*     << Tree.begin().node->get_model() << "\n" */
    /*     << Tree.begin().node->get_tree_model(true) << endl; */
         /* << Tree.get_model() << "\n" */ 
         /* << Tree.get_model(true) << endl; // pretty */

    return Tree;
};

RegressorProgram SearchSpace::make_regressor(int max_d, int max_size)
{
    return make_program<RegressorProgram>(max_d, max_size);
};

ClassifierProgram SearchSpace::make_classifier(int max_d, int max_size)
{
    return make_program<ClassifierProgram>(max_d, max_size);
};

MulticlassClassifierProgram SearchSpace::make_multiclass_classifier(
    int max_d, int max_size)
{
    return make_program<MulticlassClassifierProgram>(max_d, max_size);
};

RepresenterProgram SearchSpace::make_representer(int max_d, int max_size)
{
    return make_program<RepresenterProgram>(max_d, max_size);
};

} //Brush

