#ifndef FITNESS_H
#define FITNESS_H

#include <functional>
#include "../init.h"
#include "../util/utils.h"

using namespace nlohmann;

namespace Brush{
    
struct Fitness {
    // the loss is used in evolutionary functions
    
    float loss;     ///< aggregate loss score
    float loss_v;   ///< aggregate validation loss score

    unsigned int complexity;
    unsigned int size;
    unsigned int depth;

    // these can be different depending on the island the individual is
    unsigned int dcounter;  ///< number of individuals this dominates
    vector<unsigned int> dominated; ///< individual indices this dominates
    unsigned int rank;             ///< pareto front rank
    float crowding_dist;   ///< crowding distance on the Pareto front

    void set_dominated(vector<unsigned int>& dom){ dominated=dom; };
    vector<unsigned int> get_dominated() const { return dominated; };

    void set_loss(float f){ loss=f; };
    float get_loss() const { return loss; };

    void set_loss_v(float f_v){ loss_v=f_v; };
    float get_loss_v() const { return loss_v; };
    
    void set_size(unsigned int new_s){ size=new_s; };
    unsigned int get_size() const { return size; };
    
    void set_complexity(unsigned int new_c){ complexity=new_c; };
    unsigned int get_complexity() const { return complexity; };
    
    void set_depth(unsigned int new_d){ depth=new_d; };
    unsigned int get_depth() const { return depth; };

    void set_dcounter(unsigned int d){ dcounter=d; };
    unsigned int get_dcounter() const { return dcounter; };

    void set_rank(unsigned r){ rank=r; };
    size_t get_rank() const { return rank; };

    void set_crowding_dist(float cd){ crowding_dist=cd; };
    float get_crowding_dist() const { return crowding_dist; };

    vector<float> values;
    vector<float> weights;

    // weighted values
    vector<float> wvalues;

    // Constructor with initializer list for weights
    Fitness(const vector<float>& w={}) : values(), wvalues(), weights(w) {
        dcounter = 0;
        set_rank(0);
        set_crowding_dist(0);
        dominated.resize(0);
    }
    
    // Hash function (deap requires individuals (and fitness by induction)
    // to be hashable)
    size_t hash() const {
        std::size_t h = std::hash<vector<float>>{}(wvalues);
        return h;
    }

    void set_weights(vector<float>& w) {
        weights = w;
    }
    vector<float> get_weights() const {
        return weights;
    }
    vector<float> get_values() const {
        return values;
    }
    vector<float> get_wvalues() const {
        return wvalues;
    }

    // Method to set values
    void set_values(vector<float>& v) {
        if (v.size() != weights.size()) {
            throw std::length_error("Assigned values have not the same length than current values");
        }
        // fmt::print("updated values\n");

        values.resize(0);
        for (const auto& element : v) {
            values.push_back(element);
        }

        // Minimizing/maximizing problem: negative/positive weight, respectively.
        wvalues.resize(weights.size());

        // Perform element-wise multiplication
        std::transform(v.begin(), v.end(), 
                       weights.begin(), wvalues.begin(),
                        [](double a, double b) {
                            return a * b;
                        });
    }

    // Method to clear values
    void clearValues() {
        wvalues.clear();
    }

    bool valid() const {
        return !wvalues.empty();
    }

    // Equality comparison
    bool operator==(const Fitness& other) const {
        return wvalues == other.wvalues;
    }

    // Inequality comparison
    bool operator!=(const Fitness& other) const {
        return !(*this == other);
    }

    // Less than comparison
    bool operator<(const Fitness& other) const {
        // because of the weights, every objective is a maximization problem
        return !std::lexicographical_compare(wvalues.begin(), wvalues.end(),
                                            other.wvalues.begin(), other.wvalues.end());
    }

    // Greater than comparison
    bool operator>(const Fitness& other) const {
        return other < *this;
    }

    // Less than or equal to comparison
    bool operator<=(const Fitness& other) const {
        return !(other < *this);
    }

    // Greater than or equal to comparison
    bool operator>=(const Fitness& other) const {
        return !(*this < other);
    }

    // String representation
    std::string toString() const {
        if (valid()) {
            string s = "Fitness(";
            for (auto& v : values)
                s += to_string(v) + " ";
            return s+")";
        } else {
            return "Fitness()";
        }
    }

    // Representation for debugging
    std::string repr() const {
        if (valid()) {
            string s = "Fitness(";
            for (auto& v : values)
                s += to_string(v) + " ";
            return s+")";
        } else {
            return "Fitness()";
        }
    }

    /// set obj vector given a string of objective names
    int dominates(const Fitness& b) const;
};

void to_json(json &j, const Fitness &f);
void from_json(const json &j, Fitness& f);

}
#endif