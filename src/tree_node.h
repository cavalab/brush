#ifndef tree_node_h
#define tree_node_h

#include "init.h"
#include "data/data.h"
#include "node.h"
#include "interpreter.h"

using std::string;
using Brush::data::Data;
using Brush::ExecType;
using Brush::Node;

namespace Brush {
/// A node in the tree, combining links to other nodes as well as the actual data.
template<class T>
class tree_node_ { // size: 5*4=20 bytes (on 32 bit arch), can be reduced by 8.
	public:
		tree_node_();
		tree_node_(const T&);
		tree_node_(T&&);

		tree_node_<T> *parent;
	    tree_node_<T> *first_child, *last_child;
		tree_node_<T> *prev_sibling, *next_sibling;
		T data;

        auto fit(const Data& d);
        auto predict(const Data& d);
        void grad_descent(const ArrayXf&, const Data&);
		string get_model(bool pretty=false);
		string get_tree_model(bool pretty=false, string offset="");
    private:
        template<ExecType E>
        auto _fit(const Data& d);

        template<ExecType E>
        auto _predict(const Data& d);
}; 

// /**
//  * @brief tree node specialization for NodeBase.
//  * 
//  */
template<class T>
auto tree_node_<T>::fit(const Data& d)
{
    return this->_fit<data.exec_type>(d);
}

template<class T>
auto tree_node_<T>::predict(const Data& d)
{
    return _predict<data.exec_type>(d);
}

template<class T>
void tree_node_<T>::grad_descent(const ArrayXf& gradient, const Data& d)
{
    /* _grad_descent(gradient, d); */
}

template<class T>
string tree_node_<T>::get_model(bool pretty)
{
    return this->data->get_model(pretty, first_child, last_child);
}

template<class T>
string tree_node_<T>::get_tree_model(bool pretty, string offset)
{
    return this->data->get_tree_model(pretty, offset, first_child, last_child);
}

template<class T>
tree_node_<T>::tree_node_()
	: parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0)
	{
	}

template<class T>
tree_node_<T>::tree_node_(const T& val)
	: parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0), data(val)
	{
	}

template<class T>
tree_node_<T>::tree_node_(T&& val)
	: parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0), data(val)
	{
	}


/* template<T> auto fit(const Data& d); */
/* template<ExecType T> fit(const Data& d); */
void tree_node_<Node>::apply_weights(auto& inputs)
{
    std::transform(
                inputs.begin(), 
                inputs.end(),
                data.W.begin(),
                inputs.begin(), 
                std::multiplies<>()
                );
};

template <> 
template <> 
auto tree_node_<Node>::_fit<ExecType::Mapper>(const Data& d)
{
    /* auto inputs = this->get_children_fit(d, first_child, last_child); */
    auto signature = NodeSchema[data.node_type]["Signature"][data.ret_type]; 
    auto inputs = GetChildren<signature>(d);
    
    /* this->store_gradients(inputs); */

    if (this->data.is_weighted)
    {
        cout << "applying weights to " << this->data.name << " operator\n";
        apply_weights(inputs);
    }

    // State out = Util::apply(this->op, inputs);
    // cout << "returning " << std::get<R>(out) << endl;
    return std::apply(Brush::Function<data.node_type>, inputs);

};

template <> 
template <> 
auto tree_node_<Node>::_fit<ExecType::Splitter>(const Data& d)
{
    /* inline auto operator()(const Data& d, Node& node) */
    /* { */
        // need to handle kind with fixed var and kind without

        // set feature and threshold
        if (this->data.fixed_variable)
        {
            tie(this->data.threshold, ignore) = best_threshold(
                                                         d[this->data.feature],
                                                         d.y, 
                                                         d.classification
                                                         );
        }
        else
            set_variable_and_threshold(d);

        auto data_splits = Function<data.node_type>(d); 
        ArrayXb mask = this->threshold_mask(d);
        array<Data, 2> data_splits = d.split(mask);

        /* array<State, base::ArgCount> child_outputs; */
        auto child_outputs = this->get_children_fit(data_splits); 
        // stitch together outputs
        State out = stitch(child_outputs, d, mask);

        cout << "returning " << std::get<R>(out) << endl;

        return out;
    /* } */
};

template <> 
template <> 
auto tree_node_<Node>::_fit<ExecType::Terminal>(const Data& d)
{
    return this->predict(d);

};



}// Brush
#endif
