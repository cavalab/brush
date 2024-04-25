#ifndef POPULATION_H
#define POPULATION_H

#include "../util/error.h"
#include "../ind/individual.h"

// TODO: move this serialization elsewhere
// serializing vector of shared ptr: https://github.com/nlohmann/json/discussions/2377
// (this is used by population, which has a shared_ptr vector)
namespace nlohmann
{
template <typename T>
struct adl_serializer<std::shared_ptr<T>>
{
    static void to_json(json& j, const std::shared_ptr<T>& opt)
    {
        if (opt)
        {
            j = *opt;
        }
        else
        {
            j = nullptr;
        }
    }

    static void from_json(const json& j, std::shared_ptr<T>& opt)
    {
        if (j.is_null())
        {
            opt = nullptr;
        }
        else
        {
            opt.reset(new T(j.get<T>()));
        }
    }
};
}

namespace Brush {   
namespace Pop {

template<ProgramType T> 
class Population{
public:
    size_t pop_size;
    int num_islands;
    float mig_prob; // TODO: mig_prob should not be part of population

    vector<std::shared_ptr<Individual<T>>> individuals;
    vector<vector<size_t>> island_indexes;

    Population();
    ~Population(){};
    
    /// initialize population of programs with a starting model and/or from file 
    void init(SearchSpace& ss, const Parameters& params);

    // initialize based on list of individuals
    void init(vector<Individual<T>>& individuals, const Parameters& params);

    // save serialized population
    void save(string filename);
    // load serialized population
    void load(string filename);

    /// returns population size (the effective size of the individuals)
    int size() { return individuals.size(); };

    vector<size_t> get_island_indexes(int island){ return island_indexes.at(island); };

    /// update individual vector size, distributing the expressions in num_islands
    void add_offspring_indexes(int island);
    
    /// reduce programs to the indices in survivors. Not thread safe,as it removes elements
    void update(vector<vector<size_t>> survivors);
    
    /// setting and getting from individuals vector (will ignore islands)
    const Individual<T>& operator [](size_t i) const {return *individuals.at(i);}
    const Individual<T>& operator [](size_t i) {return *individuals.at(i);}

    /// return population equations. 
    string print_models(string sep="\n");

    /// return complexity-sorted Pareto front indices for each island
    vector<vector<size_t>> sorted_front(unsigned rank=1);

    // pareto front ignoring island divisions
    vector<size_t> hall_of_fame(unsigned rank=1);
    
    // perform a migration in the population. Individuals from sorted front or hall of fame will replace others by the
    // probability set in parameters. Expects a population without offspring
    void migrate();

    /// Sort each island in increasing complexity. This is not thread safe. I should set complexities of the whole population before calling it, and use get_complexity instead
    struct SortComplexity
    {
        Population& pop;
        SortComplexity(Population& p): pop(p){}
        bool operator()(size_t i, size_t j)
        { 
            return pop[i].get_complexity() < pop[j].get_complexity();
        }
    };
    
    /// check for same fitness and complexity to filter uniqueness. 
    struct SameFitComplexity
    {
        Population<T> & pop;
        SameFitComplexity(Population<T>& p): pop(p){}
        bool operator()(size_t i, size_t j)
        {
            return pop[i].get_complexity() == pop[j].get_complexity();
        }
    };
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Population<PT::Regressor>, individuals, island_indexes, pop_size, num_islands);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Population<PT::BinaryClassifier>, individuals, island_indexes, pop_size, num_islands);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Population<PT::MulticlassClassifier>, individuals, island_indexes, pop_size, num_islands);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Population<PT::Representer>, individuals, island_indexes, pop_size, num_islands);
}// Pop
}// Brush

#endif
