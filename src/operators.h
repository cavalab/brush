using namespace std;

namespace Op{

template<typename T>
std::function<bool(T,T)> less = std::less<T>(); 

template<typename T>
std::function<T(T,T)> plus = std::plus<T>();

template<typename T>
std::function<T(T,T)> minus = std::minus<T>();

template<typename T>
std::function<T(T,T)> multiplies = std::multiplies<T>();

}
