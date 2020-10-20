/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef STATE_H
#define STATE_H

#ifdef USE_CUDA
    #include "../pop/cuda-op/state_utils.h"
#endif

#include <string>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <iostream>
#include <variant>

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;
using Eigen::ArrayXi;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using namespace std;

//#include "node/node.h"
//external includes

namespace BR { 
    /* typedef std::tuple<float, bool, int, ArrayXf, ArrayXi, ArrayXb> State; */ 
    typedef std::variant<float, bool, int, ArrayXf, ArrayXi, ArrayXb> State; 
}
    /*template<typename type>*/
    /*!
    /* * @class NodeData*/
    /* * @brief template stack class which holds various stack types for brush*/ 
    /*class NodeData*/
    /*{*/
    /*    private:*/
    /*        type dat;               ///< the data*/ 
            
    /*    public:*/
        
    /*        ///< set element*/ 
    /*        void set(type element){ std::swap(dat, element);  }*/
            
    /*        ///< return the data*/
    /*        type& get()*/
    /*        {*/
    /*            return dat;*/
    /*        }*/
            
    /*        ~NodeData(){}*/
    /*};*/
    
    /*!
    /* * @class State*/
    /* * @brief contains various types of State actually used by brush*/
    /*struct State*/
    /*{*/
    /*    ///< floating node stack*/
    /*    NodeData<ArrayXf> f;*/                   
    /*    ///< boolean node stack*/
    /*    NodeData<ArrayXb> b;*/                   
    /*    ///<categorical stack*/
    /*    NodeData<ArrayXi> c;*/                   
    /*    ///< longitudinal node stack*/
    /*    NodeData<std::pair<vector<ArrayXf>, vector<ArrayXf> > > z;*/             
    /*    ///< floating node string stack*/
    /*    NodeData<string> fs;*/                   
    /*    ///< boolean node string stack*/
    /*    NodeData<string> bs;*/                   
    /*    ///< categorical node string stack*/
    /*    NodeData<string> cs;*/                   
    /*    ///< longitudinal node string stack*/
    /*    NodeData<string> zs;*/                   
        
    /*    ///< checks if arity of node provided satisfies the elements in various value State*/
       /* bool check(std::map<char, unsigned int> &arity); */
        
    /*    ///< checks if arity of node provided satisfies the node names in various string State*/
    /*    bool check_s(std::map<char, unsigned int> &arity);*/
        
    /*    template <typename T> inline*/ 
    /*        NodeData<Eigen::Array<T,Eigen::Dynamic,1> >& get()*/
    /*    {*/
    /*        return get<Eigen::Array<T,Eigen::Dynamic,1> >();*/
    /*    }*/
        
    /*    template <typename T> inline Eigen::Array<T,Eigen::Dynamic,1>& get_data()*/
    /*    {*/
    /*        return get<Eigen::Array<T,Eigen::Dynamic,1>>().get();*/
    /*    }*/
    /*    template <typename T> void set(Eigen::Array<T,Eigen::Dynamic,1>  value)*/
    /*    {*/
    /*        get<T>().set(value);*/
    /*    }*/

       /* template <typename T> Eigen::Array<T,Eigen::Dynamic,1>  pop() */
       /* { */
       /*     return get<T>().pop(); */
       /* } */
        
    /*    template <typename T> inline NodeData<string>& getStr()*/
    /*    {*/
    /*        return getStr<T>();*/
    /*    }*/
        
    /*    template <typename T> void set(string value)*/
    /*    {*/
    /*        getStr<T>().set(value);*/
    /*    }*/
        
    /*    template <typename T> string popStr()*/
    /*    {*/
    /*        return getStr<T>().pop();*/
    /*    }*/
        
    /*    template <typename T> unsigned int size()*/
    /*    {*/
    /*        return get<T>().size();*/
    /*    }*/
        
    /*};*/
    
    /*template <> inline NodeData<ArrayXf>& State::get(){ return f; }*/
        
    /*template <> inline NodeData<ArrayXb>& State::get(){ return b; }*/
    
    /*template <> inline NodeData<ArrayXi>& State::get(){ return c; }*/
    
    /*template <> inline ArrayXf& State::get_data(){ return f.get(); }*/
        
    /*template <> inline ArrayXb& State::get_data(){ return b.get(); }*/
    
    /*template <> inline ArrayXi& State::get_data(){ return c.get(); }*/
    
/* } } */

#endif
