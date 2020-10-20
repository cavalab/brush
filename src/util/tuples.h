/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef TUPLES_H
#define TUPLES_H

#include <variant>
using namespace std;
/* This code taken from 
https://changkun.de/modern-cpp/en-us/04-containers/index.html#4-3-Tuples 
to enable run-time indexing of tuples
*/

/* template <size_t n, typename... T> */
/* constexpr std::variant<T...> _tuple_index(const std::tuple<T...>& tpl, size_t i) */ 
/* { */
/*     if constexpr (n >= sizeof...(T)) */
/*         throw std::out_of_range("Tuple index out of range."); */
/*     if (i == n) */
/*         return std::variant<T...>{ std::in_place_index<n>, std::get<n>(tpl) }; */
/*     return _tuple_index<(n < sizeof...(T)-1 ? n+1 : 0)>(tpl, i); */
/* } */

/* template <typename... T> */
/* constexpr std::variant<T...> tuple_index(const std::tuple<T...>& tpl, size_t i) */ 
/* { */
/*     return _tuple_index<0>(tpl, i); */
/* } */

/* template <typename T0, typename ... Ts> */
/* std::ostream & operator<< (std::ostream & s, std::variant<T0, Ts...> const & v) */ 
/* { */ 
/*     std::visit([&](auto && x){ s << x;}, v); */ 
/*     return s; */
/* } */


// trying to implement constant index access for tuples

/* template <std::size_t... Is> */
/* struct indices {}; */

/* template <std::size_t N, std::size_t... Is> */
/* struct build_indices : build_indices<N-1, N-1, Is...> {}; */

/* template <std::size_t... Is> */
/* struct build_indices<0, Is...> : indices<Is...> {}; */

/* template <typename Tuple> */
/* using IndicesFor = build_indices<std::tuple_size<Tuple>::value>; */

/* template <typename Tuple, std::size_t... Indices> */
/* std::array<int, std::tuple_size<Tuple>::value> f_them_all(Tuple&& t, */ 
/* 														  indices<Indices...>) */ 
/* { */
/* 	return std::array<int, std::tuple_size<Tuple>::value> { */ 
/* 		{ f(std::get<Indices>(std::forward<Tuple>(t)))... } */ 
/* 		}; */ 
/* } */

/* template <typename Tuple> */
/* std::array<int, std::tuple_size<Tuple>::value> f_them_all(Tuple&& t) */ 
/* { */
/*     return f_them_all(std::forward<Tuple>(t), IndicesFor<Tuple> {}); */
/* } */

#endif
