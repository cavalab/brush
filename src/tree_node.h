#ifndef tree_node_h
#define tree_node_h

#include "init.h"
#include "data/data.h"
#include "node.h"
/* #include "interpreter.h" */

using std::string;
using Brush::data::Data;
using Brush::ExecType;
using Brush::Node;

namespace Brush {
/// A node in the tree, combining links to other nodes as well as the actual data.
template<class T> class tree_node_; 


// /**
//  * @brief tree node specialization for Node.
//  * 
//  */
//
template<>
class tree_node_<Node> { // size: 5*4=20 bytes (on 32 bit arch), can be reduced by 8.
	public:
        tree_node_()
            : parent(0), first_kid(0), last_child(0), prev_sibling(0), next_sibling(0)
            {}

        tree_node_(const Node& val)
            : parent(0), first_kid(0), last_child(0), prev_sibling(0), next_sibling(0), n(val)
            {}

        tree_node_(Node&& val)
            : parent(0), first_kid(0), last_child(0), prev_sibling(0), next_sibling(0), n(val)
            {}

		tree_node_<Node> *parent;
	    tree_node_<Node> *first_kid, *last_child;
		tree_node_<Node> *prev_sibling, *next_sibling;
		Node n;

        inline auto fit(const Data& d) { return this->_fit<n.exec_type>(d); };
        inline auto predict(const Data& d) { return this->_predict<n.exec_type>(d); };
        /* void grad_descent(const ArrayXf&, const Data&); */
		string get_model(bool pretty=false);
		string get_tree_model(bool pretty=false, string offset="");
    private:
        template<ExecType E>
        auto _fit(const Data& d);

        template<ExecType E>
        auto _predict(const Data& d);

        template<ExecType E, typename T> struct GetKids; 
        template<ExecType E, typename T> struct GetKidsFit; 
        template<ExecType E, typename T> struct GetKidsPredict; 

        template<typename T>
        void apply_weights(T& inputs)
        {
            cout << "applying weights to " << this->n.name << " operator\n";
            std::transform(
                        inputs.begin(), 
                        inputs.end(),
                        n.W.begin(),
                        inputs.begin(), 
                        std::multiplies<>()
                        );
        }
}; 
typedef class tree_node_<Node> TreeNode; 


/* template<> */
/* void TreeNode::grad_descent(const ArrayXf& gradient, const Data& d) */
/* { */
/*     /1* _grad_descent(gradient, d); *1/ */
/* }; */

template<>
string TreeNode::get_model(bool pretty)
{
    return this->n.get_model(pretty, first_kid, last_child);
}

template<>
string TreeNode::get_tree_model(bool pretty, string offset)
{
    return this->n.get_tree_model(pretty, offset, first_kid, last_child);
}

template <> template <> 
auto TreeNode::_fit<ExecType::Binary>(const Data& d)
{
    if (this->n.is_weighted)
        return std::apply(Brush::Function<n.node_type>, 
                          n.W[0]*first_kid->fit(d), 
                          n.W[1]*last_kid->fit(d)
        );
    else
        return std::apply(Brush::Function<n.node_type>, first_kid->fit(d), last_kid->fit(d));
};

template <> template <> 
auto TreeNode::_fit<ExecType::Unary>(const Data& d)
{
    if (this->n.is_weighted)
        return std::apply(Brush::Function<n.node_type>, n.W[0]*first_kid->fit(d));
    else
        return std::apply(Brush::Function<n.node_type>, first_kid->fit(d));
};

template <> template <> 
auto TreeNode::_fit<ExecType::Applier>(const Data& d)
{
    /* auto signature = NodeSchema[n.node_type]["Signature"][n.ret_type]; */ 
    typedef decltype(n.signature()) signature;

    auto inputs = GetKids<n.exec_type,signature>(d);
    
    if (this->n.is_weighted){
        apply_weights(inputs);
    }
    // State out = Util::apply(this->op, inputs);
    // cout << "returning " << std::get<R>(out) << endl;
    return std::apply(Brush::Function<n.node_type>, inputs);
};

template <> template <> 
auto TreeNode::_fit<ExecType::Transformer>(const Data& d)
{
    typedef decltype(n.signature().at(0)) signature;

    auto outputs = GetKidsFit<n.exec_type,signature>(d);
    
    if (this->n.is_weighted) {
        apply_weights(inputs);
    }
    // State out = Util::apply(this->op, inputs);
    // cout << "returning " << std::get<R>(out) << endl;
    std::transform(
                outputs.begin(), 
                outputs.end(),
                outputs.begin(), 
                Function<n.node_type>
                );

    return outputs; 
};

template <> template <> 
auto TreeNode::_fit<ExecType::Reducer>(const Data& d)
{
    typedef decltype(n.signature().at(0)) signature;

    auto inputs = GetKidsFit<n.exec_type,signature>(d);
    
    if (this->n.is_weighted) {
        apply_weights(inputs);
    }
    // State out = Util::apply(this->op, inputs);
    // cout << "returning " << std::get<R>(out) << endl;
    signature output = std::reduce(inputs.begin(), inputs.end(), signature(0), 
                                   Function<n.node_type>);
    return output; 
};


template <> template <> 
auto TreeNode::_fit<ExecType::Splitter>(const Data& d)
{
    // need to handle kind with fixed var and kind without

    // set feature and threshold
    if (this->n.fixed_variable)
    {
        tie(this->n.threshold, ignore) = best_threshold(
                                                     d[this->n.feature],
                                                     d.y, 
                                                     d.classification
                                                     );
    }
    else
        set_variable_and_threshold(d);

    auto data_splits = Function<n.node_type>(d); 
    ArrayXb mask = this->threshold_mask(d);
    array<Data, 2> data_splits = d.split(mask);

    /* array<State, base::ArgCount> kid_outputs; */
    //TODO: type for kids. also, handle scenario where first kid is the variable to split on.
    //
    typedef decltype(n.signature()) signature;

    auto kid_outputs = GetKidsFit<ExecType::Splitter,signature>(data_splits); 
    // stitch together outputs
    State out = stitch(kid_outputs, d, mask);

    cout << "returning " << std::get<R>(out) << endl;

    return out;
};

template <> template <> 
auto TreeNode::_fit<ExecType::Terminal>(const Data& d)
{
    return this->predict(d);
};

////////////////////////////////////////////////////////////////////////////////
// children fetching functions for nary operators

/* returns a fixed-sized array of arguments of the same type.
 */
template<typename T>
struct TreeNode::GetKids<ExecType::Applier, T>
{
    /* ArrayArgs */
    template <std::size_t N>
    using NthType = typename std::tuple_element<N, T>::type;

    T operator()(const Data& d, State (TreeNode::*fn)(const Data&))
    {
        // why not make get kids return the tuple? because tuples suck with weights
        T kid_outputs;

        TreeNode* sib = first_kid;
        for (int i = 0; i < kid_outputs.size(); ++i)
        {
            kid_outputs.at(i) = (sib->*fn)(d);
            sib = sib->next_sibling;
        }
        return kid_outputs;
    };
};

/* returns a vector of arguments of the same type. for nary children.
 * should be used for ExectType::Transformer and Reducer.
 */
template<ExecType E, typename T>
struct TreeNode::GetKids
{
    auto operator()(const Data& d, auto (TreeNode::*fn)(const Data&) )
    {
        vector<T> kid_outputs; 

        auto sib = first_kid;
        while(sib != last_kid)
        {
            kid_outputs.push_back((sib->*fn)(d));
            sib = sib->next_sibling;
        }
        return kid_outputs;
    };
};
/* template<typename T> */
/* struct TreeNode::GetKids<ExecType::Reducer, T> */
/* { */
/*     auto operator()(const Data& d, auto (TreeNode::*fn)(const Data&) ) */
/*     { */
/*         return GetKids<ExecType::Transformer, T>(d, fn); */
/*     }; */
/* }; */

template<ExecType E, typename T>
struct TreeNode::GetKidsFit {
    auto operator(const Data& d){
        return GetKids<E,T>(d, &TreeNode::fit);
    };
};

template<ExecType E, typename T>
struct TreeNode::GetKidsPredict {
    auto operator(const Data& d) {
        return GetKids<E,T>(d, &TreeNode::predict);
    };
};
////////////////////////////////////////////////////////////////////////////////


}// Brush
#endif
