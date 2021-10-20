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
         * need a random generator for each core to do multiprocessing
         */
        //cout << "Max threads are " <<omp_get_max_threads()<<"\n";
        rg.resize(omp_get_max_threads());                      
    }

    Rnd* Rnd::initRand()
    {
        if (!instance)
        {
            instance = new Rnd();
        }

        return instance;
    }

    void Rnd::destroy()
    {
        if (instance)
            delete instance;
            
        instance = NULL;
    }
    
    void Rnd::set_seed(int seed)
    { 
        /*!
         * set seeds for each core's random number generator
         */
        if (seed == 0)
        {
            std::random_device rd; 

            for (auto& r : rg)
                r.seed(rd());
        }
        else    // seed first rg with seed, then seed rest with random ints from rg[0]. 
        {
            rg[0].seed(seed);
            
            int imax = std::numeric_limits<int>::max();
            
            std::uniform_int_distribution<> dist(0, imax);

            for (size_t i = 1; i < rg.size(); ++i)
                rg[i].seed(dist(rg[0]));                     
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
