/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef TERMINAL_H
#define TERMINAL_H

namespace Brush {
namespace nodes {

// defined nodes:
template<typename F> class Terminal; 

/* specialization of Node for terminals */
template<typename R>
class Terminal: public TypedNodeBase<R> 
{
    public:
        using base = TypedNodeBase<R>;
        string variable_name;

        Terminal(string variable_name, R& value) : base("Terminal")
        {
            /* @param variable_name: name of the variable
            *  @param value: the value, to grab its reference type
            */
            this->variable_name = variable_name;
            this->set_name(this->name+"("+this->variable_name+")");
        };

        State fit(const Data& d, TreeNode*& first_child=0, TreeNode*& last_child=0) override
        {
            return this->predict(d, first_child, last_child);
        }
        State predict(const Data& d, TreeNode*& first_child, 
                      TreeNode*& last_child) override
        {
            State out = d[variable_name];
            return out;
        };
        void grad_descent(const ArrayXf& gradient, const Data& d, 
                           TreeNode*& first_child, TreeNode*& last_child) override 
        {
            this->set_prob_change(gradient.matrix().norm());
        };

        string get_model(bool pretty=false, TreeNode*& first_child=0, 
                         TreeNode*& last_child=0) const override
        { 
            if (pretty)
                return this->variable_name;
            else
                return this->name;
        };
        string get_tree_model(bool pretty=false, string offset="", 
                              TreeNode*& first_child=0, 
                         TreeNode*& last_child=0) const override
        { 
            return this->get_model(pretty, first_child, last_child);
        };
        string get_name() const override {return this->variable_name;};
        string get_op_name() const override {return this->variable_name;};
};

} // nodes
} // Brush
#endif
