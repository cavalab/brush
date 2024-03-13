#include "module.h"

#include "../individual.h"

namespace nl = nlohmann;
namespace br = Brush;

using stream_redirect = py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>;

// TODO: unify PT or T
template <br::ProgramType T>
void bind_individual(py::module& m, string name)
{
    using Class = br::Pop::Individual<T>;

    py::class_<Class> ind(m, name.data() ); 
    ind.def(py::init<>())
       .def(py::init([](br::Program<T>& prg){ Class i(prg);
                                                    return i; })
       )
       .def(py::init([](const json& j){ br::Program<T> prg = j;
                                        Class i(prg);
                                        return i; })
       )
       .def("init", &Class::init)
       .def_property("objectives", &Class::get_objectives, &Class::set_objectives)
       .def_property_readonly("program", &Class::get_program)
       .def_property_readonly("fitness", &Class::get_fitness)
    //    .def_property("complexity", &Class::get_complexity, &Class::set_complexity)
       .def(py::pickle(
            [](const Class &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                // return py::make_tuple(p.value(), p.extra());
                nl::json j = p;
                return j;
            },
            [](nl::json j) { // __setstate__
                Class p = j;
                return p;
            }
            )
       )
       ;

    // if constexpr (std::is_same_v<T,Cls>)
    // {

    // }

}