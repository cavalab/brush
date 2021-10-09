/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef RND_H
#define RND_H
//external includes
#include <random>
#include <limits>
#include <vector>

#include "../init.h"

using namespace std;
using std::swap;

namespace Brush { namespace Util{
    
    ////////////////////////////////////////////////////////// Declarations
    
    /*!
     * @class Rnd
     * @brief Defines a multi-core random number generator and its operators.
     */

    class Rnd
    {
        public:
            
            static Rnd* initRand();
            
            static void destroy();

            void set_seed(int seed);
            
            int rnd_int( int lowerLimit, int upperLimit );

            float rnd_flt(float min=0.0, float max=1.0);

            float rnd_dbl(float min=0.0, float max=1.0);
            
            float operator()(unsigned i);
            
            float operator()();

            template <class RandomAccessIterator>
            void shuffle (RandomAccessIterator first, RandomAccessIterator last)
            {
                for (auto i=(last-first)-1; i>0; --i) 
                {
                    std::uniform_int_distribution<decltype(i)> d(0,i);
                    swap (first[i], first[d(rg[omp_get_thread_num()])]);
                }
            }    

            vector<size_t> shuffled_index(size_t n);
             
            template<typename Iter>                                    
            Iter select_randomly(Iter start, Iter end)
            {
                std::uniform_int_distribution<> dis(0, distance(start, end) - 1);
                advance(start, dis(rg[omp_get_thread_num()]));
                return start;
            }

            /// select randomly with weighted distribution
            template<typename Iter, typename Iter2>                                    
            Iter select_randomly(Iter start, Iter end, Iter2 wstart, Iter2 wend)
            {
                // std::uniform_int_distribution<> dis(0, distance(start, end) - 1);
                std::discrete_distribution<size_t> dis(wstart, wend);
                advance(start, dis(rg[omp_get_thread_num()]));
                return start;
            }
           
            template<typename T>
            T random_choice(const map<T, float>& m)
            {
               /*!
                * return a weighted random key of a map, 
                * where the values are weights.
                */          
               
                assert(m.size()>0 
                    && " attemping to return random choice from an empty map");

                vector<T> keys;
                vector<float> w; 
                for (const auto& [k, v]: m)
                {
                    keys.push_back(k);
                    w.push_back(v);
                }
                return *select_randomly(keys.begin(),keys.end(),
                                        w.begin(), w.end());
            }
            template<class V, class T>
            T random_choice(const V& v)
            {
               /*!
                * return a random element of a vector.
                */          
                assert(v.size()>0 
                    && " attemping to return random choice from empty vector");
                return *select_randomly(v.begin(),v.end());
            }
 
           
            template<template<class, class> class C, class T>
            T random_choice(const C<T, std::allocator<T>>& v, const vector<float>& w )
            {
                /*!
                 * return a weighted random element of an STL container
                 */
                 
                if(w.size() == 0)
                {   
                    cout<<"random_choice() w.size() = 0 Calling random_choice(v)\n";
                    
                    cout << "W size is " << w.size() << "v size is " << v.size() << "\n"; 
                    return random_choice(v);
                }
                if(w.size() != v.size())
                {   
                    cout<<"WARN! random_choice() w.size() " << w.size() << "!= v.size() " 
                        << v.size() << ", Calling random_choice(v)\n";
                    return random_choice(v);
                }
                else
                {
                    assert(v.size() == w.size());
                    std::discrete_distribution<size_t> dis(w.begin(), w.end());
                    return v.at(dis(rg[omp_get_thread_num()])); 
                }
            }
            
            float gasdev();

        private:

            Rnd();
        
            ~Rnd();
            
            vector<std::mt19937> rg;
            
            static Rnd* instance;
     
    };
    
    static Rnd &r = *Rnd::initRand();
}
}
#endif
