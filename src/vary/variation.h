/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H

#include "../ind/individual.h"

#include "../bandit/bandit.h"
#include "../bandit/bandit_operator.h"
#include "../bandit/dummy.h"

#include "../pop/population.h"
#include "../eval/evaluation.h"
#include "../simplification/constants.h"
#include "../simplification/inexact.h"

#include <map>
#include <optional>

using namespace Brush::Pop;
using namespace Brush::MAB;
using namespace Brush::Eval;
using namespace Brush::Simpl;

/**
 * @brief Namespace for variation functions like crossover and mutation. 
 * 
 */
namespace Brush {
namespace Var {

/*!
 * @class Variation
 * @brief Class representing the variation operators in Brush.
 * 
 * The Variation class is responsible for performing individual-level variations
 * and handling the variation of a population in Brush. It contains methods for
 * crossing individuals, mutating individuals, and varying a population.
 */
template<ProgramType T>
class Variation {
public:
    /**
     * @brief Default constructor.
     */
    Variation() = default;
    
    /**
     * @brief Constructor that initializes the Variation object with parameters and search space.
     * 
     * @param params The parameters for the variation operator.
     * @param ss The search space for the variation operator.
     */
    Variation(Parameters& params, SearchSpace& ss, Dataset &d)
        : parameters(params)
        , search_space(ss)
        , data(d)
    {
        init();
    };

    /**
     * @brief Destructor.
     */
    ~Variation() {};

    /**
     * @brief Initializes the Variation object with parameters and search space.
     * 
     * @param params The parameters for the variation operator.
     * @param ss The search space for the variation operator.
     */
    void init(){
        // initializing variation bandit with the probabilities of non-null variation
        map<string, float> variation_probs;
        for (const auto& mutation : parameters.get_mutation_probs())
            // TODO: I need to figure out a better way of avoiding options that were not positive. Make sure that this does not break the code
            if (mutation.second > 0.0)
                variation_probs[mutation.first] = mutation.second;

        if (parameters.cx_prob > 0.0)
            variation_probs["cx"] = parameters.cx_prob;
        
        this->variation_bandit = Bandit<string>(parameters.bandit, variation_probs);

        // TODO: should I set C parameter based on pop size or leave it fixed?
        // TODO: update string comparisons to use .compare method
        // if (parameters.bandit.compare("dynamic_thompson")==0)
        //     this->variation_bandit.pbandit.set_C(parameters.pop_size);

        // initializing one bandit for each terminal type
        for (const auto& entry : this->search_space.terminal_weights) {
            // entry is a tuple <dataType, vector<float>> where the vector is the weights
            
            // TODO: I dont think we need this find here
            if (terminal_bandits.find(entry.first) == terminal_bandits.end())
            {
                map<string, float> terminal_probs;
                for (int i = 0; i < entry.second.size(); i++)
                    // TODO: do not insert if coefficient is smaller than zero. Fix bandits to work with that
                    if (entry.second[i] > 0.0)
                    {
                        auto node_name = search_space.terminal_map.at(entry.first).at(i).get_feature();
                        terminal_probs[node_name] = entry.second[i];
                    }
                    
                if (terminal_probs.size()>0)
                    terminal_bandits[entry.first] = Bandit<string>(parameters.bandit,
                                                                   terminal_probs);
            }
        }

        // one bandit for each return type. If we look at implementation of 
        // sample op, the thing that matters is the most nested probabilities, so we will
        // learn only that
        for (auto& [ret_type, arg_w_map]: search_space.node_map) 
        {
            // if (op_bandits.find(ret_type) == op_bandits.end())
            //     op_bandits.at(ret_type) = map<size_t, Bandit<string>>();

            for (const auto& [args_type, node_map] : arg_w_map)
            {
                // if (op_bandits.at(ret_type).find(args_type) != op_bandits.at(ret_type).end())
                //     continue

                // TODO: this could be made much easier using user_ops
                map<string, float> node_probs;

                for (const auto& [node_type, node]: node_map)
                {
                    auto weight = search_space.node_map_weights.at(ret_type).at(args_type).at(node_type);
                    
                    if (weight > 0.0)
                    {
                        // Attempt to emplace; if the key exists, do nothing
                        auto [it, inserted] = node_probs.try_emplace(node.name, weight);
                        // cout << node.name << ", " << weight << endl;
                        // If the key already existed, update its value
                        if (!inserted) {
                            it->second = weight;
                        }
                    }
                }

                if (node_probs.size() > 0)
                    op_bandits[ret_type][args_type] = Bandit<string>(parameters.bandit,
                                                                     node_probs);
            }
        }

        inexact_simplifier.initUniformPlanes(128, data.get_training_data().get_n_samples(), 1);
        // TODO: init simplification with terminals (including booleans?)
        // for (const auto& entry : this->search_space.terminal_weights) {
        //     map<string, float> terminal_probs;
        //     for (int i = 0; i < entry.second.size(); i++)
        //         if (entry.second[i] > 0.0)
        //         {
        //             Node node = search_space.terminal_map.at(entry.first).at(i);

        //             // non-wheightable nodes are not simplified. TODO: revisit this and see if they should (then implement it)
        //             // This is avoiding using booleans.
        //             if (IsWeighable(node.node_type))
        //             {
        //                 tree<Node> dummy_tree;
        //                 dummy_tree.insert(dummy_tree.begin(), node);
        //                 auto it = dummy_tree.begin();
        //                 inexact_simplifier.index(it, data.get_training_data());
        //             }
        //         }
        // }
    };

    /**
     * @brief Performs croearch_spaceover operation on two individuals.
     * 
     * @param mom The first parent individual.
     * @param dad The second parent individual.
     * @return An optional containing the offspring individual if the crossover 
     * is successful, or an empty optional otherwise.
     */
    std::tuple<std::optional<Individual<T>>, VectorXf> cross(
        const Individual<T>& mom, const Individual<T>& dad);

    /**
     * @brief Performs mutation operation on an individual.
     * 
     * @param parent The parent individual.
     * @return An optional containing the mutated individual if the mutation is
     * successful, or an empty optional otherwise.
     */
    std::tuple<std::optional<Individual<T>>, VectorXf> mutate(
        const Individual<T>& parent, string choice="");

    /**
     * @brief Handles variation of a population.
     * 
     * @param pop The population to be varied.
     * @param island The island index.
     * @param parents The indices of the parent individuals.
     * @param p The parameters for the variation operator.
     */
    void vary(Population<T>& pop, int island, const vector<size_t>& parents);

    /**
     * Updates the probability distribution sampling for variation and nodes based on the given rewards.
     *
     * @param pop The population to update the selection strategy for.
     * @param rewards The rewards obtained from the evaluation of individuals.
     */
    void update_ss();

    /**
     * @brief Varies a population and updates the selection strategy based on rewards.
     * 
     * This function performs variation on a population, calculates rewards, and updates
     * the selection strategy based on the obtained rewards.
     * 
     * @param pop The population to be varied and updated.
     * @param island The island index.
     * @param parents The indices of the parent individuals.
     */
    void vary_and_update(Population<T>& pop, int island, const vector<size_t>& parents,
                         const Dataset& data, Evaluation<T>& evaluator) {

        // TODO: move implementation to cpp file and keep only declarations here
        // TODO: rewrite this entire function to avoid repetition (this is a frankenstein)
        auto indices = pop.get_island_indexes(island);

        vector<std::shared_ptr<Individual<T>>> aux_individuals;
        for (unsigned i = 0; i < indices.size(); ++i)
        {
            if (pop.individuals.at(indices.at(i)) != nullptr)
            {
                // cout<< "Skipping individual at index " << indices.at(i) << std::endl;
                continue; // skipping if it is an individual --- we just want to fill invalid positions
            }

            // cout<< "------" << endl;
            // cout<< "Processing individual at index " << indices.at(i) << std::endl;

            // pass check for children undergoing variation     
            std::optional<Individual<T>> opt = std::nullopt; // new individual  
                
            // TODO: should this be randomly selected, or should I use each parent sequentially?
            // auto idx = *r.select_randomly(parents.begin(), parents.end());
            auto idx = parents.at(i % parents.size()); // use modulo to cycle through parents
            
            const Individual<T>& mom = pop[idx];

             // cout<< "orig:  " << mom.program.get_model() << endl;
        
            // if we got here, then the individual is not fully locked and we can proceed with mutation
            vector<Individual<T>> ind_parents;
            VectorXf root_context = get_context(mom.program, mom.program.Tree.begin());
            VectorXf context;
            string choice;
            
            // this assumes that islands do not share indexes before doing variation
            unsigned id = parameters.current_gen * parameters.pop_size + indices.at(i);

            Individual<T> ind; // the new individual

            // fully locked individuals should not be replaced by random ones. returning
            // a copy
            if (std::all_of(mom.program.Tree.begin(), mom.program.Tree.end(),
                [](const auto& n) { return n.get_prob_change()<=0.0; }))
            {
                 // cout<< "Fully locked individual, copying it" << std::endl;    
                ind = Individual<T>(mom);
                ind.variation = "born";
            }
            else
            {
                choice = this->variation_bandit.choose(root_context);
                if (choice == "cx")
                {
                    // const Individual<T>& dad = pop[
                    //     *r.select_randomly(parents.begin(), parents.end())];
                    const Individual<T>& dad = pop[parents.at((i+1) % parents.size())]; // use modulo to cycle through parents

                    // cout<< "Performing crossover" << std::endl;
                    auto variation_result = cross(mom, dad);
                    ind_parents = {mom, dad};
                    tie(opt, context) = variation_result;
                }
                else
                {
                    // cout<< "Performing mutation " << choice << std::endl;
                    auto variation_result = mutate(mom, choice);  
                    // cout<< "finished mutation" << endl;
                    ind_parents = {mom};
                    tie(opt, context) = variation_result;
                    // cout<< "unpacked" << endl;
                }

                if (opt) // variation worked, lets keep this
                {
                    // cout<< "Variation successful" << std::endl;
                    ind = opt.value();
                    ind.set_parents(ind_parents);
                }
                else {  // no optional value was returned. creating a new random individual
                    // cout<< "Variation failed, copying the individual" << std::endl;
                    
                    ind = Individual<T>(mom);
                    ind.variation = "born";
                    
                    // cout<< "Variation failed, creating a new random individual" << std::endl;
                    // ind.init(search_space, parameters); // ind.variation is born by default
                }
            }
            
            // cout<< "Setting objt" << std::endl;
            // ind.set_objectives(mom.get_objectives()); // it will have an invalid fitness

            ind.set_id(id);

            // cout<< "Setting fitness values" << std::endl;
            ind.fitness.set_loss(mom.fitness.get_loss());
            ind.fitness.set_loss_v(mom.fitness.get_loss_v());
            ind.fitness.set_size(mom.fitness.get_size());
            ind.fitness.set_complexity(mom.fitness.get_complexity());
            ind.fitness.set_linear_complexity(mom.fitness.get_linear_complexity());
            ind.fitness.set_depth(mom.fitness.get_depth());

            assert(ind.program.size() > 0);
            assert(ind.fitness.valid() == false);

            // cout<< "Fitting" << std::endl;
            auto data_aux = data.get_training_data();
            // cout<< "data aux" << std::endl;
            // cout<< "program:  " << ind.program.get_model() << endl;
            ind.program.fit(data_aux);
            // cout<< "done fitting." << std::endl;

            // simplify before calculating fitness (order matters, as they are not refitted and constants simplifier does not replace with the right value.)
            // TODO: constants_simplifier should set the correct value for the constant (so we dont have to refit).
            // simplify constants first to avoid letting the lsh simplifier to visit redundant branches
            // cout<< "after vary:  " << ind.program.get_model() << endl;
            if (parameters.constants_simplification)
            {
                constants_simplifier.simplify_tree<T>(ind.program, search_space, data.get_training_data());            
                // cout<< "const: " << ind.program.get_model() << endl;
            }
            if (parameters.inexact_simplification)
            {
                inexact_simplifier.simplify_tree<T>(ind.program, search_space, data.get_training_data());
                // cout<< "inext: " << ind.program.get_model() << endl;
            }
        
            // cout<< "before assign fit" << endl;
            evaluator.assign_fit(ind, data, parameters, false);
            // cout<< "after fit: " << ind.program.get_model() << endl;
            
            // vector<float> deltas(ind.get_objectives().size(), 0.0f);
            vector<float> deltas;
            // cout<< "Updating metrics for individual..." << endl;
            // cout<< "Objectives: ";
            // for (const auto& obj : ind.get_objectives()) {
            //     cout << obj << " ";
            // }
            // cout << endl;

            float delta  = 0.0f;
            float weight = 0.0f;

            // cout<< "updating metrics" << endl;
            for (const auto& obj : ind.get_objectives())
            {   
                // some objectives are unsigned int, which can have weird values if we
                // do subtractions. Instead, for these cases, we calculate a placeholder
                // value indicating only if it was greater or not, so we can deal with 
                // this issue.

                // cout<< "Processing objective: " << obj << endl;

                if (obj.compare(parameters.scorer) == 0) {
                    delta = ind.fitness.get_loss() - ind.fitness.get_prev_loss();
                    // cout<< "  delta (loss) = " << delta << endl;
                }
                else if (obj.compare("complexity") == 0) {
                    delta = ind.fitness.get_complexity() > ind.fitness.get_prev_complexity() ? 1.0 : -1.0 ;
                    // cout<< "  delta (complexity) = " << delta << endl;
                }
                else if (obj.compare("linear_complexity") == 0) {
                    delta = ind.fitness.get_linear_complexity() > ind.fitness.get_prev_linear_complexity() ? 1.0 : -1.0;
                    // cout<< "  delta (linear_complexity) = " << delta << endl;
                }
                else if (obj.compare("size") == 0) {
                    delta = ind.fitness.get_size() > ind.fitness.get_prev_size() ? 1.0 : -1.0;
                    // cout<< "  delta (size) = " << delta << endl;
                }
                else if (obj.compare("depth") == 0) {
                    delta = ind.fitness.get_depth() > ind.fitness.get_prev_depth() ? 1.0 : -1.0;
                    // cout<< "  delta (depth) = " << delta << endl;
                }
                else {
                    HANDLE_ERROR_THROW(obj + " is not a known objective");
                }

                auto it = Individual<T>::weightsMap.find(obj);
                if (it == Individual<T>::weightsMap.end()) {
                    HANDLE_ERROR_THROW("Weight not found for objective: " + obj);
                }

                weight = it->second;
                // cout<< "  weight = " << weight << endl;

                float weighted_delta = delta * weight;
                // cout<< "  weighted delta = " << weighted_delta << endl;

                deltas.push_back(weighted_delta);
            }

            // cout<< "Final deltas vector (size = " << deltas.size() << "): ";
            // for (const auto& d : deltas) cout << d << " ";
            // cout << endl;
            
            bool allPositive = true;
            bool allNegative = true;
            for (float d : deltas) {
                if (d < 0)
                    allPositive = false;
                if (d > 0)
                    allNegative = false;
            }

            float r = 0.0;
            if (allPositive && !allNegative)
                r = 1.0;

            // linear bandit can handle non-bernoulli-like rewards
            if (parameters.bandit.compare("linear_thompson") == 0)
            {
                // r = std::accumulate(deltas.begin(), deltas.end(), 0.0f,
                //                     [](float sum, float delta) {
                //                         if (delta > 0) return sum + 1.0f;
                //                         // if (delta < 0) return sum - 1.0f;
                //                         return sum; // For delta == 0
                //                     });
                
                // if (allPositive)
                //     r = std::accumulate(deltas.begin(), deltas.end(),
                //                         1.0f, std::multiplies<float>());

                if ( allPositive && !allNegative) r =  1.0;
                if (!allPositive &&  allNegative) r = -1.0;
            }

            // cout<< "Updating variation bandit with reward: " << r << std::endl;

            if (ind.get_variation().compare("born") != 0)
            {
                // cout<< "Updating variation bandit with variation: " << ind.get_variation() << " and reward: " << r << ". choosen variation was: " << choice << std::endl;
                this->variation_bandit.update(ind.get_variation(), r, root_context);
                
                // if (!ind.get_variation().compare("born") && !ind.get_variation().compare("cx")
                // &&  !ind.get_variation().compare("subtree"))
                // {                
                    if (ind.get_sampled_nodes().size() > 0) {
                        // cout<< "Updating terminal and operator bandits for sampled nodes" << std::endl;
                        const auto& changed_nodes = ind.get_sampled_nodes();
                        for (auto& node : changed_nodes) {
                            if (node.get_arg_count() == 0) {
                                auto datatype = node.get_ret_type();
                                // cout<< "Updating terminal bandit for node: " << node.name << std::endl;
                                this->terminal_bandits[datatype].update(node.get_feature(), r, context);
                            }
                            else {
                                auto ret_type = node.get_ret_type();
                                auto args_type = node.args_type();
                                auto name = node.name;
                                // cout<< "Updating operator bandit for node: " << name << std::endl;
                                this->op_bandits[ret_type][args_type].update(name, r, context);
                            }
                        }
                    }
                // }
            }
            else
            { // giving zero reward if the variation failed
                // cout<< "Variation failed, updating variation bandit with choice: " << choice << " and reward: 0.0" << std::endl;
                this->variation_bandit.update(choice, 0.0, root_context);
            }
            
            // aux_individuals.push_back(std::make_shared<Individual<T>>(ind));
            pop.individuals.at(indices.at(i)) = std::make_shared<Individual<T>>(ind);
            // cout<< "Individual at index " << indices.at(i) << " updated successfully" << std::endl;
        }
        
        // updating the population with the new individual
        // int aux_index = 0;
        // for (unsigned i = 0; i < indices.size(); ++i)
        // {
        //     if (pop.individuals.at(indices.at(i)) != nullptr)
        //     {
        //         // the nullptrs should be at the end of the vector
        //        pop.individuals.at(indices.at(i)) = aux_individuals.at(aux_index);
        //        aux_index++;
        //     }
        // }
    }
    
    // these functions below will extract context and use it to choose the nodes to replace
    // bandit_sample_terminal
    std::optional<Node> bandit_sample_terminal(DataType R, VectorXf& context)
    {
        // cout << "bandit_sample_terminal called with DataType: " << std::endl;

        if (terminal_bandits.find(R) == terminal_bandits.end()) {
            // cout << "No bandit found for DataType: " << std::endl;
            return std::nullopt;
        }

        auto& bandit = terminal_bandits.at(R);
        string terminal_name = bandit.choose(context);
        // cout << "Bandit chose terminal name: " << terminal_name << std::endl;

        auto it = std::find_if(
            search_space.terminal_map.at(R).begin(),
            search_space.terminal_map.at(R).end(), 
            [&](auto& node) { return node.get_feature() == terminal_name; });

        if (it != search_space.terminal_map.at(R).end()) {
            auto index = std::distance(search_space.terminal_map.at(R).begin(), it);
            // cout << "Terminal found at index: " << index << std::endl;
            return search_space.terminal_map.at(R).at(index);
        }

        // cout << "Terminal not found for name: " << terminal_name << std::endl;
        return std::nullopt;
    };

    // bandit_get_node_like
    std::optional<Node> bandit_get_node_like(Node node, VectorXf& context)
    {
        // cout << "bandit_get_node_like called with node: " << node.name << std::endl;

        // TODO: use search_space.terminal_types here (and in search_space get_node_like as well)
        if (Is<NodeType::Terminal, NodeType::Constant, NodeType::MeanLabel>(node.node_type)){
            // cout << "Node is of type Terminal, Constant, or MeanLabel" << std::endl;
            return bandit_sample_terminal(node.ret_type, context);
        }

        if (op_bandits.find(node.ret_type) == op_bandits.end()) {
            // cout << "No bandit found for return type: " << std::endl;
            return std::nullopt;
        }
        if (op_bandits.at(node.ret_type).find(node.args_type()) == op_bandits.at(node.ret_type).end()) {
            // cout << "No bandit found for arg type: " << std::endl;
            return std::nullopt;
        }

        auto& bandit = op_bandits[node.ret_type][node.args_type()];
        string node_name = bandit.choose(context);
        // cout << "Bandit chose node name: " << node_name << std::endl;

        auto entries = search_space.node_map[node.ret_type][node.args_type()];
        // cout << "Ret match size: " << entries.size() << std::endl;

        for (const auto& [node_type, node_value]: entries)
        {
            // cout << " - Node name: " << node_value.name << std::endl;
            if (node_value.name == node_name) {
                // cout << "Node name match: " << node_value.name << std::endl;
                return node_value;
            }
        }

        return std::nullopt;
    };

    // bandit_sample_op_with_arg
    std::optional<Node> bandit_sample_op_with_arg(DataType ret, DataType arg,
                                                  VectorXf& context, int max_args=0)
    {
        auto args_map = search_space.node_map.at(ret);
        vector<size_t> matches;
        vector<float> weights;

        for (const auto& [args_type, name_map]: args_map) {
            for (const auto& [name, node]: name_map) {
                auto node_arg_types = node.get_arg_types();
                
                auto within_size_limit = !(max_args) || (node.get_arg_count() <= max_args);
                
                if (in(node_arg_types, arg)
                && within_size_limit 
                && search_space.node_map_weights.at(ret).at(args_type).at(name) > 0.0f ) {
                    // if it can be sampled
                    matches.push_back(node.args_type());
                }
            }
        }

        if (matches.size()==0)
            return std::nullopt;

        // we randomly select args type. This is what determines which bandit to use
        auto args_type = *r.select_randomly(matches.begin(), 
                                            matches.end() );
        auto& bandit = op_bandits[ret][args_type];
        string node_name = bandit.choose(context);

        // TODO: this could be more efficient
        auto entries = search_space.node_map[ret][args_type];
        for (const auto& [node_type, node_value]: entries)
        {
            if (node_value.name == node_name) {
                return node_value;
            }
        }

        return std::nullopt;
    };

    // bandit_sample_op
    std::optional<Node> bandit_sample_op(DataType ret, VectorXf& context)
    {
        if (search_space.node_map.find(ret) == search_space.node_map.end())
            return std::nullopt;

        // any bandit to do the job
        auto& [args_type, bandit] = *r.select_randomly(op_bandits[ret].begin(), 
                                                       op_bandits[ret].end() );

        string node_name = bandit.choose(context);

        auto entries = search_space.node_map[ret][args_type];
        for (const auto& [node_type, node_value]: entries)
        {
            if (node_value.name == node_name) {
                return node_value;
            }
        }

        return std::nullopt;
    };

    // bandit_sample_subtree // TODO: should I implement this? (its going to be hard).
    // without this one being performed directly by the bandits, we then rely on
    // the sampled probabilities we update after every generation. Since there are lots
    // of samplings, I think it is ok to not update them and just use the distribution they learned.

    VectorXf get_context(const Program<T>& program, Iter spot) {
        return variation_bandit.get_context<T>(program, spot, search_space, data); }

    // they need to be references because we are going to modify them
    SearchSpace search_space; // The search space for the variation operator.
    Dataset& data;             // the data used to extract context and evaluate the models
    Parameters parameters;    // The parameters for the variation operator
private:
    // bandits will internaly work as an interface between variation and its searchspace.
    // they will sample from the SS (instead of letting the search space do it directly),
    // and also propagate what they learn back to the search space at the end of the execution.
    Bandit<string> variation_bandit;
    map<DataType, Bandit<string>> terminal_bandits; 
    map<DataType, map<size_t, Bandit<string>>> op_bandits;  
    
    // simplification methods
    Constants_simplifier constants_simplifier; 
    Inexact_simplifier inexact_simplifier;
};

// // Explicitly instantiate the template for brush program types
// template class Variation<ProgramType::Regressor>;
// template class Variation<ProgramType::BinaryClassifier>;
// template class Variation<ProgramType::MulticlassClassifier>;
// template class Variation<ProgramType::Representer>;

class MutationBase {    
public:
    using Iter = tree<Node>::pre_order_iterator;

    template<Brush::ProgramType T>
    static auto find_spots(Program<T>& program, Variation<T>& variator,
                            const Parameters& params)
    {
        vector<float> weights(program.Tree.size());

        // by default, mutation can happen anywhere, based on node weights
        std::transform(program.Tree.begin(), program.Tree.end(), weights.begin(),
                       [&](const auto& n){ return n.get_prob_change();});
        
        // Must have same size as tree, even if all weights <= 0.0
        return weights;
    }

    template<Brush::ProgramType T>
    static auto mutate(Program<T>& program, Iter spot, Variation<T>& variator,
                            const Parameters& params);
};

} //namespace Var
} //namespace Brush
#endif