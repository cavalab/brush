#ifndef OPERATOR_H
#define OPERATOR_H
#include "init.h"
#include "data/data.h"
#include "nodemap.h"
#include "tree.h"
#include "tree_node.h"
#include <type_traits>
#include <concepts>
using namespace std;

namespace Brush {

    /* template<NodeType NT, SigType S> */ 
    /* struct Operator */ 
    /* { */
    /*     // TODO */
    /*     using Args = typename Signature<S>::ArgTypes; */
    /*     using RetType = typename Signature<S>::RetType; */
    /*     static constexpr size_t ArgCount = Signature<S>::ArgCount; */
    /*     // get arg types from tuple by index */
    /*     template <std::size_t N> */
    /*     using NthType = conditional_t<is_array_v<Args>, */
    /*                                   typename Args::value_type, */
    /*                                   typename std::tuple_element<N, Args>::type */
    /*                                  >; */
        
    /*     static constexpr auto F = [](const auto& ...args){ */
    /*         Function<NT> f{}; */
    /*         return f(args...); */ 
    /*     }; */ 

    /*     Operator() = default; */
    /*     /1* Operator(NT node_type, RetType y, Args... args){}; *1/ */
    /*     //////////////////////////////////////////////////////////////////////////////// */
    /*     /// Apply weights */
    /*     template<typename T=Args> */
    /*     enable_if_t<is_array_v<T,void>> */ 
    /*     apply_weights(T& inputs, const Node& n) const */
    /*     { */
    /*         cout << "applying weights to " << n.name << " operator\n"; */
    /*         std::transform( */
    /*                     inputs.begin(), */ 
    /*                     inputs.end(), */
    /*                     n.W.begin(), */
    /*                     inputs.begin(), */ 
    /*                     std::multiplies<>() */
    /*                     ); */
    /*     }; */
    /*     //////////////////////////////////////////////////////////////////////////////// */
    /*     /// Utilities to grab child outputs. */
    /*     /1* template<typename T> T get_kids(const Data&, TreeNode&, bool fit) const; *1/ */

    /*     // get a std::array of kids */
    /*     template<bool Fit,typename T=Args> */
    /*     enable_if_t<is_array_v<T>, T> */ 
    /*     get_kids(const Data& d, TreeNode& tn) const */
    /*     { */
    /*         T child_outputs; */
    /*         using arg_type = typename T::value_type; */

    /*         TreeNode* sib = tn.first_child; */
    /*         for (int i = 0; i < this->get_arg_count(); ++i) */
    /*         { */
    /*             if constexpr (Fit) */
    /*                 child_outputs.at(i) = sib->fit<arg_type>(d) ; */
    /*             else */
    /*                 child_outputs.at(i) = sib->predict<arg_type>(d); */
    /*             sib = sib->next_sibling; */
    /*         } */
    /*         return child_outputs; */
    /*     }; */


    /*     // get a std::tuple of kids */
    /*     template<int I,bool Fit> */
    /*     auto get_kid(const Data& d,TreeNode& tn ) const */
    /*     { */
    /*         auto sib = tn.first_child; */ 
    /*         for (int i = 0 ; i < I; ++i) */
    /*             sib = sib->next_sibling; */
    /*         if constexpr(Fit) */
    /*             return sib->fit<NthType<I>>(d); */
    /*         else */
    /*             return sib->predict<NthType<I>>(d); */
    /*     }; */

    /*     template<typename T, bool Fit, size_t ...Is> */
    /*     requires (!is_array_v<T>) */
    /*     auto get_kids_seq(const Data& d, TreeNode& tn, std::index_sequence<Is...>) const */ 
    /*     { */ 
    /*         return std::make_tuple(get_kid<Is,Fit>(d,tn)...); */
    /*     }; */

    /*     // tuple get kids */
    /*     template<bool Fit, typename T=Args> */
    /*     requires (!is_array_v<T>) */
    /*     auto get_kids(const Data& d, TreeNode& tn) const */
    /*     { */
    /*         return get_kids_seq<T,Fit>(d, tn, std::make_index_sequence<ArgCount>{}); */
    /*     }; */

    /*     // fit and predict convenience functions */
    /*     auto get_kids_fit(const Data& d, TreeNode& tn) const { return get_kids<true>(d, tn); }; */
    /*     auto get_kids_predict(const Data& d, TreeNode& tn) const { return get_kids<false>(d, tn); }; */

    /*     /////////////////////////////////////////////////////////////////////////// */
    /*     // fit and predict */
    /*     template<bool Fit, typename T=Args> */
    /*     requires (is_array_v<T>) */
    /*     RetType eval(const Data& d, TreeNode& tn) const */
	    /* { */
    /*         auto inputs = get_kids<Fit>(d, tn); */
    /*         if (tn.n.is_weighted) */
    /*             this->apply_weights(inputs, tn.n); */
    /*         return std::apply(F, inputs); */
    /*     }; */

    /*     template<bool Fit, typename T=Args> */
    /*     requires (!is_array_v<T>) */
    /*     RetType eval(const Data& d, TreeNode& tn) const */
	    /* { */
    /*         auto inputs = get_kids<Fit>(d, tn); */
    /*         return std::apply(F, inputs); */
    /*     }; */

    /*     RetType fit(const Data& d, TreeNode& tn) const { return eval<true>(d,tn); }; */

    /*     RetType predict(const Data& d, TreeNode& tn) const { return eval<false>(d,tn); }; */
    /* }; */
    /* /// Terminal Overload */
    /* template<SigType S> */
    /* struct Operator<NodeType::Terminal, S> */
    /* { */
    /*     using RetType = typename Signature<S>::RetType; */
    /*     RetType eval(const Data& d, TreeNode& tn) const { return std::get<RetType>(d[tn.n.feature]); }; */
    /*     RetType fit(const Data& d, TreeNode& tn) const { return eval(d,tn); }; */
    /*     RetType predict(const Data& d, TreeNode& tn) const { return eval(d,tn); }; */
    /* }; */
    /* template<SigType S> */ 
    /* struct Operator<NodeType::Constant, S> */
    /* { */
    /*     using RetType = typename Signature<S>::RetType; */

    /*     template<typename T=RetType> */
    /*     requires same_as<T, ArrayXf> */
    /*     RetType eval(const Data& d, TreeNode& tn) const { */ 
    /*         return tn.n.W.at(0)*RetType(d.n_samples); */ 
    /*     }; */

    /*     template<typename T=RetType> */
    /*     requires same_as<T, ArrayXb> */
    /*     RetType eval(const Data& d, TreeNode& tn) const { */ 
    /*         return RetType(d.n_samples) > tn.n.W.at(0); */ 
    /*     }; */

    /*     template<typename T=RetType> */
    /*     requires same_as<T, ArrayXXf> */
    /*     RetType eval(const Data& d, TreeNode& tn) const { */ 
    /*         return tn.n.W.at(0)*RetType(d.n_samples, d.n_features); */ 
    /*     }; */

    /*     template<typename T=RetType> */
    /*     requires same_as<T, ArrayXXb> */
    /*     RetType eval(const Data& d, TreeNode& tn) const { */ 
    /*         return RetType(d.n_samples, d.n_features) > tn.n.W.at(0); */
    /*     }; */
        
    /*     RetType fit(const Data& d, TreeNode& tn) const { return eval(d,tn); }; */
    /*     RetType predict(const Data& d, TreeNode& tn) const { return eval(d,tn); }; */
    /* }; */
    /* //////////////////////////////////////////////////////////////////////////// */
    /* // fit and predict Dispatch functions */
    /* template<typename R, NodeType NT, SigType S> //, typename ...Args> */
    /* R DispatchFit(const Data& d, TreeNode& tn) */ 
    /* { */
    /*     const auto op = Operator<NT,S>{}; */
    /*     return op.fit(d, tn); */
    /* }; */

    /* template<typename R, NodeType NT, SigType S> */
    /* R DispatchPredict(const Data& d, TreeNode& tn) */ 

    /* { */
    /*     const auto op = Operator<NT,S>{}; */
    /*     return op.predict(d, tn); */
    /* }; */

} // Brush

#endif
