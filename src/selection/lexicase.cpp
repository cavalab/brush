#include "lexicase.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;
using namespace Sel;

template<ProgramType T>
Lexicase<T>::Lexicase(bool surv)
{ 
    this->name = "lexicase"; 
    this->survival = surv; 
}

template<ProgramType T>
vector<size_t> Lexicase<T>::select(Population<T>& pop, int island, 
        const Parameters& params)
{
    // this one can be executed in parallel because it is just reading the errors. This 
    // method assumes that the expressions have been fitted previously, and their respective
    // error vectors are filled 

    auto island_pool = pop.get_island_indexes(island);

    // if this is first generation, just return indices to pop
    if (params.current_gen==0)
        return island_pool;

    //< number of samples
    unsigned int N = pop.individuals.at(0)->error.size(); 

    //< number of individuals
    unsigned int P = island_pool.size();          
       
    // define epsilon
    ArrayXf epsilon = ArrayXf::Zero(N);
  
    // if output is continuous, use epsilon lexicase            
    if (!params.classification || params.scorer_.compare("log")==0 
    ||  params.scorer_.compare("multi_log")==0)
    {
        // for each sample, calculate epsilon
        for (int i = 0; i<epsilon.size(); ++i)
        {
            VectorXf case_errors(island_pool.size());
            for (int j = 0; j<island_pool.size(); ++j)
            {
                case_errors(j) = pop.individuals.at(island_pool[j])->error(i);
            }
            epsilon(i) = mad(case_errors);
        }
    }

    // selection pool
    vector<size_t> starting_pool;
    for (int i = 0; i < island_pool.size(); ++i)
    {
        starting_pool.push_back(i);
    }
    assert(starting_pool.size() == P);     
    
    vector<size_t> selected(P,0); // selected individuals

    #pragma omp parallel for 
    for (unsigned int i = 0; i<P; ++i)  // selection loop
    {
        vector<size_t> cases; // cases (samples)
        if (params.classification && !params.class_weights.empty()) 
        {
            // for classification problems, weight case selection 
            // by class weights
            vector<size_t> choices(N);
            std::iota(choices.begin(), choices.end(),0);

            vector<float> sample_weights = params.sample_weights;

            for (unsigned i = 0; i<N; ++i)
            {
                vector<size_t> choice_idxs(N-i);
                std::iota(choice_idxs.begin(),choice_idxs.end(),0);

                size_t idx = *r.select_randomly(
                        choice_idxs.begin(), choice_idxs.end(),
                        sample_weights.begin(), sample_weights.end());

                cases.push_back(choices.at(idx));
                choices.erase(choices.begin() + idx);

                sample_weights.erase(sample_weights.begin() + idx);
            }
        }
        else
        {   // otherwise, choose cases randomly
            cases.resize(N); 
            std::iota(cases.begin(),cases.end(),0);
            r.shuffle(cases.begin(),cases.end());   // shuffle cases
        }
        vector<size_t> pool = starting_pool;    // initial pool   
        vector<size_t> winner;                  // winners

        bool pass = true;     // checks pool size and number of cases
        unsigned int h = 0;   // case count
        
        float epsilon_threshold;

        while(pass){    // main loop
            epsilon_threshold = 0;

            winner.resize(0);   // winners                  
            // minimum error on case
            float minfit = std::numeric_limits<float>::max();                     

            // get minimum
            for (size_t j = 0; j<pool.size(); ++j)
                if (pop.individuals.at(pool[j])->error(cases[h]) < minfit) 
                    minfit = pop.individuals.at(pool[j])->error(cases[h]);
            
            // criteria to stay in pool
            epsilon_threshold = minfit+epsilon[cases[h]];

            // select best
            for (size_t j = 0; j<pool.size(); ++j)
                if (pop.individuals.at(pool[j])->error(cases[h]) 
                        <= epsilon_threshold)
                winner.push_back(pool[j]);                 
            
            ++h; // next case
            // only keep going if needed
            pass = (winner.size()>1 && h<cases.size()); 
            
            if(winner.size() == 0)
            {
            if(h >= cases.size())
                winner.push_back(*r.select_randomly(
                        pool.begin(), pool.end()) );
            else
                pass = true;
            }
            else
            pool = winner;    // reduce pool to remaining individuals
        }       
    
        assert(winner.size()>0);

        //if more than one winner, pick randomly
        selected.at(i) = *r.select_randomly(
                         winner.begin(), winner.end() );   
    }               

    if (selected.size() != island_pool.size())
    {
        std::cout << "selected: " ;
        for (auto s: selected) std::cout << s << " "; std::cout << "\n";
        HANDLE_ERROR_THROW("Lexicase did not select correct number of \
                parents");
    }

    return selected;
}


template<ProgramType T>
vector<size_t> Lexicase<T>::survive(Population<T>& pop, int island,
        const Parameters& params)
{
    /* Lexicase survival */
    HANDLE_ERROR_THROW("Lexicase survival not implemented");
    return vector<size_t>();
}


}
}
