/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "data/data.h"
#include "nodemap.h"

namespace Brush{

/* typedef tree_node<Node> TreeNode;*/ 

/* template<ExecType E, typename T> GetKids;*/ 
/* template<ExecType E, typename T> GetKidsFit;*/ 
/* template<ExecType E, typename T> GetKidsPredict;*/ 

/* template<ExecType E, typename T>*/
/* struct GetKidsFit {*/
/*     auto operator(const Data& d){*/
/*         return GetKids<E,T>(d, &TreeNode::fit);*/
/*     };*/
/* };*/

/* template<ExecType E, typename T>*/
/* struct GetKidsPredict {*/
/*     auto operator(const Data& d) {*/
/*         return GetKids<E,T>(d, &TreeNode::predict);*/
/*     };*/
/* };*/

/* /* returns a fixed-sized array of arguments of the same type.*/
/*  */*/
/* template<typename T>*/
/* struct GetKids<ExecType::Applier, T>*/
/* {*/
/*     /* ArrayArgs */*/
/*     using TreeNode = tree_node_<Node>;*/
/*     template <std::size_t N>*/
/*     using NthType = typename std::tuple_element<N, T>::type;*/

/*     T operator()(const Data& d, State (TreeNode::*fn)(const Data&))*/
/*     {*/
/*         // why not make get kids return the tuple? because tuples suck with weights*/
/*         T kid_outputs;*/

/*         TreeNode* sib = first_kid;*/
/*         for (int i = 0; i < kid_outputs.size(); ++i)*/
/*         {*/
/*             kid_outputs.at(i) = (sib->*fn)(d);*/
/*             sib = sib->next_sibling;*/
/*         }*/
/*         return kid_outputs;*/
/*     };*/
/* };*/

/* /* returns a vector of arguments of the same type.*/
/*  */*/
/* template<typename T>*/
/* struct GetKids<ExecType::Transformer, T>*/
/* {*/
/*     auto operator()(const Data& d, auto (TreeNode::*fn)(const Data&) )*/
/*     {*/
/*         vector<T> kid_outputs;*/ 

/*         auto sib = first_kid;*/
/*         while(sib != last_kid)*/
/*         {*/
/*             kid_outputs.push_back((sib->*fn)(d));*/
/*             sib = sib->next_sibling;*/
/*         }*/
/*         return kid_outputs;*/
/*     };*/
/* };*/
/* template<typename T>*/
/* struct GetKids<ExecType::Reducer, T>*/
/* {*/
/*     auto operator()(const Data& d, auto (TreeNode::*fn)(const Data&) )*/
/*     {*/
/*         return GetKids<ExecType::Transformer, T>(d, fn);*/
/*     };*/
/* };*/



/* returns a fixed sized array of arguments of the same type.
 */



/* returns a vector of arguments of the same type.
 */


}
#endif
