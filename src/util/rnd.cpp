/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "rnd.h"

namespace Brush { namespace Util{
    
    Rnd* Rnd::instance = NULL;
    
    Rnd::Rnd()
    {
        /*!
         * need a random generator for each core to do multiprocessing.
         * The constructor will resize the random generators based on 
         * the number of available cores. 
         */
        

        // TODO: stop using omp. this should be based on number of islands. make each island to use their respective
        // when we resize, the constructor of new elements are invoked.
        rg.resize(omp_get_max_threads());                      
    }

    Rnd* Rnd::initRand()
    {
        // creates the static random generator by calling the constructor
        if (!instance)
        {
            instance = new Rnd();
            instance->set_seed(0); // setting a random initial state
        }

        return instance;
    }

    void Rnd::destroy()
    {
        if (instance)
            delete instance;
            
        instance = NULL;
    }
    
    void Rnd::set_seed(unsigned int seed)
    { 
        /*!
         * set seeds for each core's random number generator.
         */
        if (seed == 0) {
            // use a non-deterministic random generator to seed the deterministics
            std::random_device rd; 
            seed = rd();
        }

        // generating a seed sequence
        std::seed_seq seq{seed};
        
        std::vector<std::uint32_t> seeds(rg.size());
        seq.generate(seeds.begin(), seeds.end());

        for (size_t i = 0; i < rg.size(); ++i) {
            rg[i].seed(seeds[i]);
        }
    }

    int Rnd::rnd_int( int lowerLimit, int upperLimit ) 
    {
        std::uniform_int_distribution<> dist( lowerLimit, upperLimit );
        return dist(rg[omp_get_thread_num()]);
    }

    float Rnd::rnd_flt(float min, float max)
    {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rg[omp_get_thread_num()]);
    }

    float Rnd::rnd_alpha_beta(float alpha, float beta)
    {
        // use a beta distribution based on alphas and betas to sample probabilities.

        // from https://stackoverflow.com/questions/4181403/generate-random-number-based-on-beta-distribution-using-boost
        // You'll first want to draw a random number uniformly from the
        // range (0,1). Given any distribution, you can then plug that number
        // into the distribution's "quantile function," and the result is as
        // if a random value was drawn from the distribution. 

        // from https://stackoverflow.com/questions/10358064/random-numbers-from-beta-distribution-c
        // The beta distribution is related to the gamma distribution. Let X be a
        // random number drawn from Gamma(α,1) and Y from Gamma(β,1), where the
        // first argument to the gamma distribution is the shape parameter.
        // Then Z=X/(X+Y) has distribution Beta(α,β). 
        
        std::gamma_distribution<float> distA(alpha, 1.0f);
        std::gamma_distribution<float> distB(beta, 1.0f);

        float X = distA(rg[omp_get_thread_num()]);
        float Y = distB(rg[omp_get_thread_num()]);

        float prob = X/(X+Y+0.001f);
        
        return prob;
    }
    float Rnd::rnd_dbl(float min, float max)
    {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rg[omp_get_thread_num()]);
    }
    
    float Rnd::operator()(unsigned i) 
    {
        return rnd_dbl(0.0,i);
    }
    
    float Rnd::operator()() { return rnd_flt(0.0,1.0); }

    float Rnd::gasdev()
    //Returns a normally distributed deviate with zero mean and unit variance
    {
        float ran = rnd_flt(-1,1);
        static int iset=0;
        static float gset;
        float fac,rsq,v1,v2;
        if (iset == 0) {// We don't have an extra deviate handy, so 
            do{
                v1=float(2.0*rnd_flt(-1,1)-1.0); //pick two uniform numbers in the square ex
                v2=float(2.0*rnd_flt(-1,1)-1.0); //tending from -1 to +1 in each direction,
                rsq=v1*v1+v2*v2;	   //see if they are in the unit circle,
            } while (rsq >= 1.0 || rsq == 0.0); //and if they are not, try again.
            fac=float(sqrt(-2.0*log(rsq)/rsq));
        //Now make the Box-Muller transformation to get two normal deviates. Return one and
        //save the other for next time.
        gset=v1*fac;
        iset=1; //Set flag.
        return v2*fac;
        } 
        else 
        {		//We have an extra deviate handy,
            iset=0;			//so unset the flag,
            return gset;	//and return it.
        }
    }

    /// returns a shuffled index vector of length n
    vector<size_t> Rnd::shuffled_index(size_t n)
    {
        vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        this->shuffle(idx.begin(), idx.end());
        return idx;
    } 
    Rnd::~Rnd() {}
} }
