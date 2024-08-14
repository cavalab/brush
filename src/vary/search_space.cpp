#include "search_space.h"
#include "../program/program.h" // TODO: dont import this header here

namespace Brush{


float calc_initial_weight(const ArrayXf& value, const ArrayXf& y)
{
    // OBS: only for terminals!

    // weights are initialized as the slope of the z-score of x and y.

    // If y has different length from X, we get a core dump in this function.
    // That is why Dataset makes a check for this 
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
    float prob_change = std::abs(slope(data.col(0).array() ,   // x=variable
                                       data.col(1).array() )); // y=target

    // having a minimum feature weight if it was not set to zero
    if (std::abs(prob_change)<1e-4)
        prob_change = 1e-1;

    // prob_change will evaluate to nan if variance(x)==0. Features with
    // zero variance should not be used (as they behave just like a constant).
    if (std::isnan(prob_change))
        prob_change = 0.0;

    return prob_change;
}


/// @brief generate terminals from the dataset features and random constants.
/// @param d a dataset
/// @param weights_init whether the terminal prob_change should be estimated from correlations with the target value
/// @return a vector of nodes 
vector<Node> generate_terminals(const Dataset& d, const bool weights_init)
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
                
                if (d.y.size()>0 && weights_init) 
                {
                    // if the value can be casted to float array, we can calculate slope
                    if (std::holds_alternative<ArrayXf>(value) && d.y.size()>0) 
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

    // constants for each type
    auto cXf = Node(NodeType::Constant, Signature<ArrayXf()>{}, true, "Cf");
    float floats_avg_weights = signature_avg(cXf.ret_type);
    cXf.set_prob_change(floats_avg_weights);
    terminals.push_back(cXf);

    auto cXi = Node(NodeType::Constant, Signature<ArrayXi()>{}, true, "Ci");
    cXi.set_prob_change(signature_avg(cXi.ret_type));
    terminals.push_back(cXi);

    auto cXb = Node(NodeType::Constant, Signature<ArrayXb()>{}, false, "Cb");
    cXb.set_prob_change(signature_avg(cXb.ret_type));
    terminals.push_back(cXb);

    // mean label node
    auto meanlabel = Node(NodeType::MeanLabel, Signature<ArrayXf()>{}, true, "MeanLabel");
    meanlabel.set_prob_change(floats_avg_weights);
    terminals.push_back(meanlabel);

    return terminals;
};

std::unordered_map<std::size_t, std::string> ArgsName; 

void SearchSpace::print() const { 
    std::cout << fmt::format("{}\n", *this) << std::flush; 
}

void SearchSpace::init(const Dataset& d, const unordered_map<string,float>& user_ops,
                       bool weights_init)
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

    vector<Node> terminals = generate_terminals(d, weights_init);
    
    // If it is a classification problem, we need to add the fixed root nodes 
    // (logistic for binary classification, softmax for multiclassification).
    // Sometimes, the user may not specify these two nodes as candidates when 
    // sampling functions, so we check if they are already in the terminal set, and
    // we add them with zero prob if they are not. They need to be in the func set
    // when calling GenerateNodeMap, so the search_space will contain all the hashes
    // and signatures for them (and they can be used only in program root).
    // TODO: fix softmax and add it here

    // Copy the original map using the copy constructor
    std::unordered_map<std::string, float> extended_user_ops(user_ops);

    if (d.classification)
    {        
        // Convert ArrayXf to std::vector<float> for compatibility with std::set
        std::vector<float> vec(d.y.data(), d.y.data() + d.y.size());

        std::set<float> unique_classes(vec.begin(), vec.end());

        // We need some ops in the search space so we can have the logit and offset
        if (user_ops.find("OffsetSum") == user_ops.end())
            extended_user_ops.insert({"OffsetSum", 0.0f});

        if (unique_classes.size()==2 && (user_ops.find("Logistic") == user_ops.end())) {
            extended_user_ops.insert({"Logistic", 0.0f});
        }
        else if (user_ops.find("Softmax") == user_ops.end()) {
            extended_user_ops.insert({"Softmax", 0.0f});
        }
    }

    /* fmt::print("generate nodetype\n"); */
    GenerateNodeMap(extended_user_ops, d.unique_data_types, 
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

    // std::cout << "Function name: sample_subtree" << std::endl;

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

    auto Tree = tree<Node>();
    auto spot = Tree.insert(Tree.begin(), root);

    // we should notice the difference between size of a PROGRAM and a TREE.
    // program count weights in its size, while the TREE structure dont. Wenever
    // using size of a program/tree, make sure you use the function from the correct class
    PTC2(Tree, spot, max_d, max_size);
    
    return Tree;
};

tree<Node>& SearchSpace::PTC2(tree<Node>& Tree,
    tree<Node>::iterator spot, int max_d, int max_size) const
{
    // PTC2 is agnostic of program type

    // A comment about PTC2 method:            
    // PTC2 can work with depth or size restriction, but it does not strictly
    // satisfies these conditions all time. Given a `max_size` and `max_depth`
    // parameters, the real maximum size that can occur is `max_size` plus the
    // highest operator arity, and the real maximum depth is `max_depth` plus one.

    // Queue of nodes that need children
    vector<tuple<TreeIter, DataType, int>> queue; 

    // node depth
    int d = 1;
    // current tree size
    int s = 1;

    Node root = spot.node->data;

    // updating size accordingly to root node
    if (Is<NodeType::SplitBest>(root.node_type))
        s += 3;
    else if (Is<NodeType::SplitOn>(root.node_type))
        s += 2;
    
    if ( root.get_is_weighted()==true
    &&   Isnt<NodeType::Constant, NodeType::MeanLabel>(root.node_type) )
        s += 2;
        
    //For each argument position a of n, Enqueue(a; g) 
    for (auto a : root.arg_types)
    { 
        // cout << "queing a node of type " << DataTypeName[a] << endl;
        auto child_spot = Tree.append_child(spot);
        queue.push_back(make_tuple(child_spot, a, d));
    }

    int max_arity = 4;

    Node n;
    // Now we actually start the PTC2 procedure to create the program tree
    while ( queue.size() + s < max_size && queue.size() > 0) 
    {            
        // including the queue size in the max_size, since each element in queue
        // can grow up exponentially

        // by default, terminals are weighted (counts as 3 nodes in program size).
        // since every spot in queue has potential to be a terminal, we multiply
        // its size by 3. Subtracting one due to the fact that this loop will
        // always insert a non terminal (which by default has weights off).
        // this way, we can have PTC2 working properly.
        
        // cout << "queue size: " << queue.size() << endl;
        auto [qspot, t, d] = RandomDequeue(queue);

        // cout << "current depth: " << d << endl;
        if (d >= max_d || s >= max_size)
        {
            auto opt = sample_terminal(t);

            // if it returned optional, then there's nothing to sample based on weights.
            // We'll force sampling again with uniform probs
            if (!opt)
                opt = sample_terminal(t, true);

            // If we successfully get a terminal, use it
            n = opt.value();

            Tree.replace(qspot, n);
        }
        else
        {
            //choose a nonterminal of matching type
            auto opt = sample_op(t);

            if (!opt) { // there is no operator for this node. sample a terminal instead
                opt = sample_terminal(t);

                // didnt work the easy way, lets try the hard way
                if (!opt)
                    opt = sample_terminal(t, true);
            }

            if (!opt) { // no operator nor terminal. weird.
                auto msg = fmt::format("Failed to sample operator AND terminal of data type  {} during PTC2.\n", DataTypeName[t]);
                HANDLE_ERROR_THROW(msg);

                // queue.push_back(make_tuple(qspot, t, d));
                // continue;
            }

            n = opt.value();
            
            auto newspot = Tree.replace(qspot, n);

            // For each arg of n, add to queue
            for (auto a : n.arg_types)
            {
                auto child_spot = Tree.append_child(newspot);

                queue.push_back(make_tuple(child_spot, a, d+1));
            }
        }

        // increment is different based on node weights
        ++s;
        
        if (Is<NodeType::SplitBest>(n.node_type))
            s += 3;
        else if (Is<NodeType::SplitOn>(n.node_type))
            s += 2;

        if ( n.get_is_weighted()==true
        &&   Isnt<NodeType::Constant, NodeType::MeanLabel>(n.node_type) )
            s += 2;
    } 

    while (queue.size() > 0)
    {
        if (queue.size() == 0)
            break;

        auto [qspot, t, d] = RandomDequeue(queue);

        auto opt = sample_terminal(t);

        if (!opt)
            opt = sample_terminal(t, true);

        n = opt.value();
        
        auto newspot = Tree.replace(qspot, n);
    }
    return Tree;
};

// TODO: stop using params as a default argument and actually pass it (also update tests)
RegressorProgram SearchSpace::make_regressor(int max_d, int max_size, const Parameters& params)
{
    return make_program<RegressorProgram>(params, max_d, max_size);
};

ClassifierProgram SearchSpace::make_classifier(int max_d, int max_size, const Parameters& params)
{
    return make_program<ClassifierProgram>(params, max_d, max_size);
};

MulticlassClassifierProgram SearchSpace::make_multiclass_classifier(
    int max_d, int max_size, const Parameters& params)
{
    return make_program<MulticlassClassifierProgram>(params, max_d, max_size);
};

RepresenterProgram SearchSpace::make_representer(int max_d, int max_size, const Parameters& params)
{
    return make_program<RepresenterProgram>(params, max_d, max_size);
};

} //Brush

