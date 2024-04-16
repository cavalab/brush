#include "population.h"

namespace Brush{   
namespace Pop{
        
template<ProgramType T>
Population<T>::Population()
{
    individuals.resize(0);
    mig_prob = 0.0;
    pop_size = 0;
    num_islands = 0;
}


template<ProgramType T>
void Population<T>::init(vector<Individual<T>>& new_individuals, const Parameters& params)
{
    if (new_individuals.size() != params.pop_size
    &&  new_individuals.size() != 2*params.pop_size ) {
        throw std::runtime_error("Individual vector has different number of individuals than pop_size. popsize is "+to_string(params.pop_size)+", number of individuals is " + to_string(new_individuals.size()));
    }

    this->mig_prob = params.mig_prob;
    this->pop_size = params.pop_size;
    this->num_islands=params.num_islands;

    island_indexes.resize(num_islands);
    
    // If the assert fails, execution stops, but for completeness, you can also throw an exception
    size_t p = pop_size;

    individuals.resize(2*p);
    std::fill(individuals.begin(), individuals.end(), nullptr);

    for (int i=0; i<num_islands; ++i)
    {
        size_t idx_start = std::floor(i*p/num_islands);
        size_t idx_end   = std::floor((i+1)*p/num_islands);

        auto delta = idx_end - idx_start;

        island_indexes.at(i).resize(delta);
        iota(island_indexes.at(i).begin(), island_indexes.at(i).end(), idx_start);
    
        if (new_individuals.size() == 2*params.pop_size) { // pop + offspring
            island_indexes.at(i).resize(delta*2);
            iota(
                island_indexes.at(i).begin() + delta, island_indexes.at(i).end(),
                p+idx_start);
        }
        else 
        {
            // // second half is space to the offspring (but we dont initialize them)
            // individuals.at(i) = std::make_shared<nullptr>;
        }
    };

    for (int j=0; j< new_individuals.size(); j++) {
        individuals.at(j) = std::make_shared<Individual<T>>(new_individuals.at(j));
    }
}

template<ProgramType T>
void Population<T>::init(SearchSpace& ss, const Parameters& params)
{
    this->mig_prob = params.mig_prob;
    this->pop_size = params.pop_size;
    this->num_islands=params.num_islands;
    
    // Tuples with start and end indexes for each island. Number of individuals
    // in each island can slightly differ if num_islands is not a divisor of p (popsize)
    island_indexes.resize(num_islands);
    
    size_t p = pop_size; // population size

    for (int i=0; i<num_islands; ++i)
    {
        size_t idx_start = std::floor(i*p/num_islands);
        size_t idx_end   = std::floor((i+1)*p/num_islands); // TODO: i should use this equation to restore the original number of individuals (so I can have the algorithm to work regardless of number of islands being a divisor of pop size)

        auto delta = idx_end - idx_start;

        island_indexes.at(i).resize(delta);
        iota(island_indexes.at(i).begin(), island_indexes.at(i).end(), idx_start);
    };

   // TODO: load file (like feat)

    // this calls the default constructor for the container template class 
    individuals.resize(2*p); // we will never increase or decrease the size during execution (because is not thread safe). this way, theres no need to sync between selecting and varying the population

    for (int i = 0; i< p; ++i)
    {          
        // first half will contain the initial population
        individuals.at(i) = std::make_shared<Individual<T>>();
        individuals.at(i)->init(ss, params);
        individuals.at(i)->set_objectives(params.objectives);
        
        // second half is space to the offspring (but we dont initialize them)
        individuals.at(p+i) = nullptr;
    }
}

template<ProgramType T>
void Population<T>::save(string filename)
{
    std::ofstream out;                      
    if (!filename.empty())
        out.open(filename);
    else
        out.open("pop.json");

    json j;
    to_json(j, *this);
    out << j ;
    out.close();
    logger.log("Saved population to file " + filename, 1);
}

template<ProgramType T>
void Population<T>::load(string filename)
{
    
    // TODO: if initializing from a population file, then this is where we should load previous models.
    // three behaviors: if we have only 1 ind, then replicate it trought the entire pop
    // if n_ind is the same as pop_size, load all models. if n_ind != pop_size, throw error

    //TODO: replace with from_json(j, this) call
    std::ifstream indata;
    indata.open(filename);
    if (!indata.good())
        HANDLE_ERROR_THROW("Invalid input file " + filename + "\n"); 

    std::string line;
    indata >> line; 

    json j = json::parse(line);
    from_json(j, *this);

    logger.log("Loaded population from " + filename + " of size = " 
            + to_string(this->size()),1);

    indata.close();
}

/// update individual vector size and island indexes
template<ProgramType T>
void Population<T>::add_offspring_indexes(int island)
{	   
    size_t p = pop_size; // population size. prep_offspring slots will douple the population, adding the new expressions into the islands
    
    // this is going to be tricky (pay attention to delta and p use)
    size_t idx_start = std::floor(island*p/num_islands);
    size_t idx_end   = std::floor((island+1)*p/num_islands);

    auto delta = idx_end - idx_start; // island size

    // inserting indexes of the offspring
    island_indexes.at(island).resize(island_indexes.at(island).size() + delta);
    iota(
        island_indexes.at(island).begin() + delta, island_indexes.at(island).end(),
        p+idx_start);

    // Im keeping the offspring and parents in the same population object, because we
    // have operations that require them together (archive, hall of fame.)
    // The downside is having to be aware that islands will create offsprings
    // intercalated with other islands
}

template<ProgramType T>
void Population<T>::update(vector<vector<size_t>> survivors)
{
    // this is the step that should end up cutting off half of the population
    vector<Individual<T>> new_pop;
    new_pop.resize(0);
    for (int j=0; j<num_islands; ++j)
    {
        for (int k=0; k<survivors.at(j).size(); ++k){
            new_pop.push_back(
                *individuals.at(survivors.at(j).at(k)) );
        }

        // need to make island point to original range
        size_t idx_start = std::floor(j*pop_size/num_islands);
        size_t idx_end   = std::floor((j+1)*pop_size/num_islands);

        // auto delta = survivors.at(j).size(); // should have the same size as idx_end - idx_start
        auto delta = idx_end - idx_start;

        assert(delta == survivors.at(j).size()
           && " migration ended up with a different popsize");

        // inserting indexes of the offspring
        island_indexes.at(j).clear();
        island_indexes.at(j).resize(delta);
        iota(island_indexes.at(j).begin(), island_indexes.at(j).end(), idx_start);
    }

    assert(new_pop.size() == pop_size
           && " update ended up with a different popsize");

    this->individuals.resize(0);
    for (auto ind : new_pop)
    {
        // making hard copies of the individuals
        json ind_copy = ind;

        // this will fill just half of the pop
        individuals.push_back(
            std::make_shared<Individual<T>>(ind_copy) );
    }

    assert(individuals.size() == pop_size
           && " number of new individuals is different from pop size");

    for (int i=0; i< pop_size; ++i)
    {
        // second half is space to the offspring (but we dont initialize them)
        individuals.push_back(nullptr);   
    }
}

template<ProgramType T>
string Population<T>::print_models(string sep)
{
    // not printing the island each individual belongs to
    string output = "";

    for (int j=0; j<num_islands; ++j)
    {
        output += "island " + to_string(j) + ":\n";

        for (int k=0; k<island_indexes.at(j).size(); ++k) {
            output += "ind index " + to_string(k);
            output += " pos " + to_string(island_indexes.at(j).at(k)) + ": ";
            Individual<T>& ind = *individuals.at(island_indexes.at(j).at(k)).get();
            output += ind.get_model() + sep;
        }
    }
    return output;
}

template<ProgramType T>
vector<vector<size_t>> Population<T>::sorted_front(unsigned rank)
{
    // this is used to migration and update archive at the end of a generation. expect islands without offspring

    /* Returns individuals on the Pareto front, sorted by increasign complexity. */
    vector<vector<size_t>> pf_islands;
    pf_islands.resize(num_islands);

    for (int j=0;j<num_islands; ++j)
    {
        auto idxs = island_indexes.at(j);
        vector<size_t> pf;

        for (int i=0; i<idxs.size(); ++i)
        {
            // this assumes that rank was previously calculated. It is set in selection (ie nsga2) if the information is useful to select/survive
            if (individuals.at(idxs.at(i))->fitness.rank == rank)
                pf.push_back(i);
        }

        std::sort(pf.begin(),pf.end(),SortComplexity(*this)); 
        auto it = std::unique(pf.begin(),pf.end(),SameFitComplexity(*this));
        
        pf.resize(std::distance(pf.begin(),it));
        pf_islands.at(j) = pf;
    }

    return pf_islands;
}

template<ProgramType T>
vector<size_t> Population<T>::hall_of_fame(unsigned rank)
{
    // TODO: remove this ignore offspring (things should work without it)
    // this is used to migration and update archive at the end of a generation. expect islands without offspring

    vector<size_t> pf(0);
    
    for (int j=0;j<num_islands; ++j)
    {
        auto idxs = island_indexes.at(j);
        for (int i=0; i<idxs.size(); ++i)
        {
            if (individuals.at(idxs.at(i))->fitness.rank == rank)
                pf.push_back(idxs.at(i));
        }
    }
    std::sort(pf.begin(),pf.end(),SortComplexity(*this)); 

    auto it = std::unique(pf.begin(),pf.end(),SameFitComplexity(*this));
    
    pf.resize(std::distance(pf.begin(),it));

    return pf;
}


// template<ProgramType T>
// void Population<T>::migrate()
// {
//     // changes where island points to based on HOF and pareto fronts

//     if (num_islands==1)
//         return; // skipping. this only work because update is fixing island indexes

//     // we cant use more than half of population here
//     // std::cout << "finding island sorted fronts" << std::endl;
//     auto island_fronts = sorted_front(1);
    
//     // std::cout << "finding global hall of fame" << std::endl;
//     auto global_hall_of_fame = hall_of_fame(1);

//     // This method is not thread safe (as it is now)
//     vector<vector<size_t>> new_island_indexes;
//     new_island_indexes.resize(num_islands);

//     // std::cout << "Looping" << std::endl;
//     for (int island=0; island<num_islands; ++island)
//     {
//         new_island_indexes.at(island).resize(0);

//         auto idxs = island_indexes.at(island);
//         for (unsigned int i=0; i<idxs.size(); ++i)
//         {
//             if (r() < mig_prob)
//             {
//                 // std::cout << "migrating in island " << island << std::endl;

//                 size_t migrating_idx;
//                 // determine if incoming individual comes from global or local hall of fame
//                 if (r() < 0.5) { // from global hall of fame
//                     // std::cout << "from hall of fame" << std::endl;
//                     migrating_idx = *r.select_randomly(
//                         global_hall_of_fame.begin(),
//                         global_hall_of_fame.end());
                        
//                     // std::cout << "mig idx" << migrating_idx << std::endl;
//                 }
//                 else { // from any other local hall of fame
//                     // std::cout << "from other island" << std::endl;
//                     // finding other island indexes
//                     vector<int> other_islands(num_islands-1);
//                     iota(other_islands.begin(), other_islands.end(), 0);

//                     // skipping current island
//                     auto it = other_islands.begin();
//                     std::advance(it, island);
//                     for (;it != other_islands.end(); ++it) {
//                         ++(*it); // TODO: is this really skipping the current island?
//                     }

//                     // picking other island
//                     int other_island = *r.select_randomly(
//                         other_islands.begin(),
//                         other_islands.end());

//                     migrating_idx = *r.select_randomly(
//                         island_fronts.at(other_island).begin(),
//                         island_fronts.at(other_island).end());
//                     // std::cout << "mig idx" << migrating_idx << std::endl;
//                 }

//                 // std::cout << "index " << i << " of island " << island;
//                 // std::cout << " is now" << migrating_idx << std::endl;
                
//                 new_island_indexes.at(island).push_back(migrating_idx);
//             }
//             else
//             {
//                 new_island_indexes.at(island).push_back(idxs.at(i));
//             }
//         }
//     }
//     // making hard copies (so the next generation starts with islands that does not share individuals 
//     // this is particularly important to avoid multiple threads assigning different rank/crowdist/dcounter 
//     // or different fitness)
    
//     // std::cout << "starting to consolidate pop" << std::endl;
//     vector<Individual<T>> new_pop;
//     new_pop.resize(0);
//     for (int j=0; j<num_islands; ++j)
//     {
//         for (int k=0; k<new_island_indexes.at(j).size(); ++k){
//             new_pop.push_back(
//                 *individuals.at(new_island_indexes.at(j).at(k)) );
//         }

//         // need to make island point to original range
//         size_t idx_start = std::floor(j*pop_size/num_islands);
//         size_t idx_end   = std::floor((j+1)*pop_size/num_islands);

//         auto delta = idx_end - idx_start;

//         assert(delta == new_island_indexes.at(j).size()
//             && " new pop has the wrong number of new individuals");

//         // inserting indexes of the offspring
//         island_indexes.at(j).clear();
//         island_indexes.at(j).resize(delta);
//         iota(island_indexes.at(j).begin(), island_indexes.at(j).end(), idx_start);
//     }

//     // std::cout << "finished making copies" << std::endl;
//     assert(new_pop.size() == pop_size
//            && " migration ended up with a different popsize");

    
//     // std::cout << "filling individuals" << std::endl;
//     this->individuals.resize(0);
//     for (auto ind : new_pop)
//     {
//         // making hard copies of the individuals
//         json ind_copy = ind;

//         // this will fill just half of the pop
//         individuals.push_back(
//             std::make_shared<Individual<T>>(ind_copy) );
//     }
//     for (int i=0; i< pop_size; ++i)
//     {
//         // second half is space to the offspring (but we dont initialize them)
//         individuals.push_back(nullptr);   
//     }
// }



template<ProgramType T>
void Population<T>::migrate()
{
    // changes where island points to by shuffling it

    if (num_islands==1)
        return; // skipping. this only work because update is fixing island indexes

    // This method is not thread safe (as it is now)
    vector<vector<size_t>> new_island_indexes;
    new_island_indexes.resize(num_islands);

    // std::cout << "Looping" << std::endl;
    for (int island=0; island<num_islands; ++island)
    {
        new_island_indexes.at(island).resize(0);

        auto idxs = island_indexes.at(island);
        for (unsigned int i=0; i<idxs.size(); ++i)
        {
            if (r() < mig_prob)
            {
                // std::cout << "migrating in island " << island << std::endl;

                size_t migrating_idx;
                
                vector<int> other_islands(num_islands-1);
                iota(other_islands.begin(), other_islands.end(), 0);

                // skipping current island
                auto it = other_islands.begin();
                std::advance(it, island);
                for (;it != other_islands.end(); ++it) {
                    ++(*it); // TODO: is this really skipping the current island?
                }

                // picking other island
                int other_island = *r.select_randomly(
                    other_islands.begin(),
                    other_islands.end());

                migrating_idx = *r.select_randomly(
                    island_indexes.at(other_island).begin(),
                    island_indexes.at(other_island).end());
                // std::cout << "mig idx" << migrating_idx << std::endl;
            

                // std::cout << "index " << i << " of island " << island;
                // std::cout << " is now" << migrating_idx << std::endl;
                
                new_island_indexes.at(island).push_back(migrating_idx);
            }
            else
            {
                new_island_indexes.at(island).push_back(idxs.at(i));
            }
        }
    }
    // making hard copies (so the next generation starts with islands that does not share individuals 
    // this is particularly important to avoid multiple threads assigning different rank/crowdist/dcounter 
    // or different fitness)
    
    // std::cout << "starting to consolidate pop" << std::endl;
    vector<Individual<T>> new_pop;
    new_pop.resize(0);
    for (int j=0; j<num_islands; ++j)
    {
        for (int k=0; k<new_island_indexes.at(j).size(); ++k){
            new_pop.push_back(
                *individuals.at(new_island_indexes.at(j).at(k)) );
        }

        // need to make island point to original range
        size_t idx_start = std::floor(j*pop_size/num_islands);
        size_t idx_end   = std::floor((j+1)*pop_size/num_islands);

        auto delta = idx_end - idx_start;

        assert(delta == new_island_indexes.at(j).size()
            && " new pop has the wrong number of new individuals");

        // inserting indexes of the offspring
        island_indexes.at(j).clear();
        island_indexes.at(j).resize(delta);
        iota(island_indexes.at(j).begin(), island_indexes.at(j).end(), idx_start);
    }

    // std::cout << "finished making copies" << std::endl;
    assert(new_pop.size() == pop_size
           && " migration ended up with a different popsize");

    // std::cout << "filling individuals" << std::endl;
    this->individuals.resize(0);
    for (auto ind : new_pop)
    {
        // making hard copies of the individuals
        json ind_copy = ind;

        // this will fill just half of the pop
        individuals.push_back(
            std::make_shared<Individual<T>>(ind_copy) );
    }
    for (int i=0; i< pop_size; ++i)
    {
        // second half is space to the offspring (but we dont initialize them)
        individuals.push_back(nullptr);   
    }
}

} // Pop
} // Brush
