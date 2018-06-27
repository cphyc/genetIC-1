#ifndef _UTILS_HPP_INCLUDED
#define _UTILS_HPP_INCLUDED

#include <vector>
#include <cmath>
#include <stdexcept>
/*!
    \namespace tools
    \brief Defines useful functions and tools used throughout the code

 */
namespace tools {
  // Argsort function from http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
  template<typename T>
  std::vector<size_t> argsort(const std::vector<T> &v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
  }

  template<typename T>
  int getRatioAndAssertInteger(T p, T q, T tolerance = 1e-8) {
    T ratio = p / q;
    int rounded_ratio = int(round(ratio));
    if (!(std::abs(T(rounded_ratio) - ratio) < tolerance)){
      throw std::runtime_error("The ratio is not an integer within tolerance");
    }
    return rounded_ratio;
  }

  template<typename T>
  size_t getRatioAndAssertPositiveInteger(T p, T q, T tolerance = 1e-8) {
    assert(p > 0);
    assert(q > 0);
    return (size_t) getRatioAndAssertInteger(p, q, tolerance);
  }

  //! Solves n^x = p for integers x
  size_t findPowerOf(size_t n, size_t p){
    return getRatioAndAssertPositiveInteger(log(p), log(n));
  }

  template<typename T>
  void set_zero(T &item) {
    item = 0;
  }

  template<typename T>
  void set_zero(std::complex<T> &item) {
    item.real(0);
    item.imag(0);
  }

  // helper function for dumps in mapper
  void indent(std::ostream &s, int level = 0) {
    for (int i = 0; i < level; i++) s << "| ";
    s << "+ ";

  }

  template<typename T>
  std::vector<T> linspace(T start, T end, int n) {
    std::vector<T> array;
    if ((n == 0) || (n == 1) || (start == end))
      array.push_back(end);
    else if (n > 1) {
      double step = (end - start) / (n);
      int count = 0;
      while (count < n + 1) {
        T elem = start + count * step;
        array.push_back(elem);
        ++count;
      }
    }
    return array;
  }

  template<typename T>
  void sortAndEraseDuplicate( std::vector<T> & vector){
    std::sort(vector.begin(), vector.end());
    vector.erase(std::unique(vector.begin(), vector.end()), vector.end());
  }


}

#endif
