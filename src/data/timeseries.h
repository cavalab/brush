#ifndef TIMESERIES_H
#define TIMESERIES_H
#include "../init.h"
#include "../util/utils.h"
#include "../util/error.h"
/* #include "../util/logger.h" */
/* #include "../util/rnd.h" */

namespace Brush::data{

template<class T>
struct TimeSeries
{
    using Scalar = T;
    using EntryType = Eigen::Array<T,Dynamic,1>;
    using ValType = std::vector<EntryType>;
    using TimeType = std::vector<Eigen::ArrayXi>;
    /*! Wraps time and value slices to matrices
    *  TODO: define begin() and end() iterators? figure out how to handle operators that just use values versus time 
    */
    TimeSeries() = default; 

    TimeType time;
    ValType value;

    TimeSeries(const TimeType& t, 
               const ValType& v): time(t), value(v) {}
    // array<TimeSeries, 2> TimeSeries::split(const ArrayXb& mask) const ;
    /// return a slice of the data using indices idx
    template<typename U>
    TimeSeries operator()(const U& idx) const
    {
        // TimeType t = this->time(idx, Eigen::all);
        TimeType t = Util::slice(this->time, idx);
        // ValType v = this->value(idx, Eigen::all);
        ValType v = Util::slice(this->value, idx); //(idx, Eigen::all);
        // std::visit([&](auto&& arg) { arg = arg(idx, Eigen::all); }, v);
        return TimeSeries(t, v);
    };

    inline auto size() -> size_t { return value.size(); };
    inline auto rows() -> size_t { return value.size(); };
    inline auto cols(int i = 0) -> size_t { return value.at(i).size(); };
    // /// return a slice of the data by row or colum/n
    // template<typename R, typename C>
    // TimeSeries operator()(const R& rows, 
    //                       const C& cols) const
    // {
    //     ArrayXXf t = this->time(rows, cols);
    //     ValType v = this->value(rows, cols);
    //     return TimeSeries(t, v);
    // };
    // TODO: from_json and to_json

    // TODO: custom iterator that iterates over pairs of time and value vectors.
    // for now these only iterate over values.
    
    typename ValType::iterator begin() { return this->value.begin(); };
    typename ValType::iterator end() { return this->value.end(); };
    auto cbegin() const { return this->value.cbegin(); };
    auto cend() const { return this->value.cend(); };
    typename TimeType::iterator tbegin() { return this->time.begin(); };
    typename TimeType::iterator tend() { return this->time.end(); };
    auto ctbegin() const { return this->time.cbegin(); };
    auto ctend() const { return this->time.cend(); };
    // auto rowwise() const { return this->value.rowwise(); };
    // auto colwise() const { return this->value.colwise(); };
    // return std::visit([](auto v){return v.rowwise().end();}, this->value); }

    //TODO: define overloaded operations?
    /* operators on values */
    /* transform takes a unary function, and applies it to each entry.  */
    auto transform(std::function<EntryType(EntryType)> op) const -> TimeSeries<T>
    {
        ValType dest(this->value.size());
        std::transform(cbegin(), cend(), 
                       dest.begin(),
                       op
        );
        return TimeSeries<T>(this->time, dest);
    }
    /* reduce takes a unary aggregating function, applies it to each entry, and returns an Array.*/
    template<typename R=T>
    auto reduce(const auto& op) const 
    {
        using RetType = Array<R,Dynamic,1>;
        // output "dest" has one entry for each sample. 
        RetType dest(this->value.size());
        std::transform(cbegin(), cend(), 
                       dest.begin(),
                       [&](const EntryType& i){return R(op(i));}
        );
        return dest;
    }; 

    // transformation overloads
    inline auto abs() { return this->transform([](const EntryType& i){ return i.abs(); }); };
    inline auto pow() { return this->transform([](const EntryType& i){ return i.pow(); } ); };
    inline auto log() { return this->transform([](const EntryType& i){ return i.log(); } ); };
    inline auto logabs() { return this->transform([](const EntryType& i){ return i.abs().log(); } ); };
    inline auto log1p() { return this->transform([](const EntryType& i){ return i.log1p(); } ); };
    inline auto ceil() { return this->transform([](const EntryType& i){ return i.ceil(); } ); };
    inline auto floor() { return this->transform([](const EntryType& i){ return i.floor(); } ); };
    inline auto exp() { return this->transform([](const EntryType& i){ return i.exp(); } ); };
    inline auto sin() { return this->transform([](const EntryType& i){ return i.sin(); } ); };
    inline auto cos() { return this->transform([](const EntryType& i){ return i.cos(); } ); };
    inline auto tan() { return this->transform([](const EntryType& i){ return i.tan(); } ); };
    inline auto asin() { return this->transform([](const EntryType& i){ return i.asin(); } ); };
    inline auto acos() { return this->transform([](const EntryType& i){ return i.acos(); } ); };
    inline auto atan() { return this->transform([](const EntryType& i){ return i.atan(); } ); };
    inline auto sinh() { return this->transform([](const EntryType& i){ return i.sinh(); } ); };
    inline auto cosh() { return this->transform([](const EntryType& i){ return i.cosh(); } ); };
    inline auto tanh() { return this->transform([](const EntryType& i){ return i.tanh(); } ); };
    inline auto sqrt() { return this->transform([](const EntryType& i){ return i.sqrt(); } ); };
    inline auto sqrtabs() { return this->transform([](const EntryType& i){ return i.abs().sqrt(); } ); };
    inline auto square() { return this->transform([](const EntryType& i){ return i.square(); } ); };
    // reduction overloads
    inline auto median() { return this->reduce([](const EntryType& i){ return Util::median(i); } ); };
    inline auto mean() { return this->reduce([](const EntryType& i){ return i.mean(); } ); };
    inline auto std() { return this->reduce([](const EntryType& i){ return i.std(); } ); };
    inline auto max() { return this->reduce([](const EntryType& i){ return i.maxCoeff(); } ); };
    inline auto min() { return this->reduce([](const EntryType& i){ return i.minCoeff(); } ); };
    inline auto sum() { return this->reduce<float>([](const EntryType& i){ return i.sum(); } ); };
    inline auto count() { return this->reduce<float>([](const EntryType& i){ return i.size(); } ); };

    /* template<typename V=T> */
    /* enable_if_t<is_same_v<V,float>,TimeSeries<float>> */
    
    inline auto operator*(const float& v) requires(is_same_v<Scalar,float>) { 
        return this->transform([&](const EntryType& i){ return i*v; } ); 
    };

    template<typename T2>
    inline auto before(const TimeSeries<T2>& t2) const { 
        // return elements of this that are before elements in t2
        // TODO
        return (*this); 
    };
    template<typename T2>
    inline auto after(const TimeSeries<T2>& t2) const { 
        // return elements of this that are after elements in t2
        // TODO
        return (*this); 
    };
    template<typename T2>
    inline auto during(const TimeSeries<T2>& t2) const { 
        // return elements of this that occur within the window in which elements of t2 occur
        // TODO
        return (*this); 
    };

    std::string print() const
    { 
        /**
         * @brief Print the time series.
         * 
         * @param output ostream object
         * @param ts time series object
         * @return ostream& 
         */
        size_t m = 40;
        size_t max_len = std::min(m, this->value.size()); 
        string output = "[";
        for (int i = 0; i < max_len; ++i){
            size_t max_width = std::min(m, size_t(this->value.at(i).size())); 
            string dots = max_width < m ? "" : "...";
            auto val = this->value.at(i)(Eigen::seqN(0,max_width));
            auto t = this->time.at(i)(Eigen::seqN(0,max_width));
            output += fmt::format("[value: {}{},\n time: {}{}]\n", val, dots, t, dots); 
        }
        output += "]\n";

        return output;            
    };
};

/**
 * @brief TimeSeries convenience typedefs.
 * 
 */
typedef TimeSeries<bool> TimeSeriesb;
typedef TimeSeries<int> TimeSeriesi;
typedef TimeSeries<float> TimeSeriesf;

} // namespace Brush::data

namespace Brush{
std::ostream &operator<<( std::ostream& output, const Brush::data::TimeSeries<bool>& ts ) ;
std::ostream &operator<<( std::ostream& output, const Brush::data::TimeSeries<float>& ts ) ;
std::ostream &operator<<( std::ostream& output, const Brush::data::TimeSeries<int>& ts ) ;
} // namespace Brush

template <> struct fmt::formatter<Brush::data::TimeSeriesi>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(Brush::data::TimeSeriesi x, FormatContext& ctx) const {
    /* return formatter<string_view>::format(x.value, ctx); */
    return formatter<string_view>::format(x.print(), ctx);
  }
};

template <> struct fmt::formatter<Brush::data::TimeSeriesb>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(Brush::data::TimeSeriesb x, FormatContext& ctx) const {
    /* return formatter<string_view>::format(print_time_series(x), ctx); */
    return formatter<string_view>::format(x.print(), ctx);
  }
};

template <> struct fmt::formatter<Brush::data::TimeSeriesf>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(Brush::data::TimeSeriesf x, FormatContext& ctx) const {
    /* return formatter<string_view>::format(print_time_series(x), ctx); */
    return formatter<string_view>::format(x.print(), ctx);
    /* return formatter<string_view>::format(x.value, ctx); */
  }
};

#endif