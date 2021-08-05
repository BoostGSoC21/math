#include "math_unit_test.hpp"

#include <boost/config.hpp>
#include <boost/multiprecision/number.hpp>
#ifdef BOOST_MATH_USE_FLOAT128
#include <boost/multiprecision/complex128.hpp>
#endif
#include <boost/multiprecision/cpp_bin_float.hpp>
//#include <boost/multiprecision/mpfr.hpp>
//#include <boost/multiprecision/mpfi.hpp>

#include <boost/math/fft/bsl_backend.hpp>
#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>
#include <boost/math/fft/abstract_ring.hpp>
#include <vector>
#include <iterator>
#include <iostream>
#include <boost/core/demangle.hpp>

using namespace boost::math::fft;

template<class T>
void print(const std::vector<T>& V)
{
  std::cout << "Size: " << V.size() << '\n';
  std::cout << "[";
  for(auto x: V)
    std::cout << x << ", ";
  std::cout<< "]\n";
}

template<class Backend>
void test_r2c()
{
  using Real    = typename Backend::value_type;
  using Complex = typename detail::select_complex<Real>::type;
  
  std::vector<Real> A{1.,2.,3.};
  std::vector<Complex> B;
    
  Backend plan(A.size());
  plan.real_to_complex(A.begin(),A.end(),std::back_inserter(B));
  print(B);
  
  A.push_back(4);
  B.resize(A.size());
  
  plan.real_to_complex(A.begin(),A.end(),B.begin());
  print(B);
}

template<typename Real>
void test_bsl_fftw_gsl() {
//test_r2c< bsl_rfft <Real> >();
  test_r2c< fftw_rfft<Real> >();
  test_r2c< gsl_rfft <Real> >();
}

template<typename Real>
void test_bsl_fftw () {
//test_r2c< bsl_rfft <Real> >();
  test_r2c< fftw_rfft<Real> >();
}

template<typename Real>
void test_bsl() {
//test_r2c< bsl_rfft <Real> >();
}

int main()
{
  test_bsl_fftw    <float      >();
  test_bsl_fftw_gsl<double     >();
  test_bsl_fftw    <long double>();

// TODO
#ifdef BOOST_MATH_USE_FLOAT128
//test_bsl_fftw    <boost::multiprecision::float128>();
#endif
//test_bsl<boost::multiprecision::cpp_bin_float_100>();
//test_bsl<boost::multiprecision::cpp_bin_float_quad>();
//test_bsl<boost::multiprecision::mpfr_float_100>();
//test_bsl<boost::multiprecision::mpfi_float_100>();

  return boost::math::test::report_errors();
}

