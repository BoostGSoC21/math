///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp>
#include <boost/multiprecision/number.hpp>
#ifdef BOOST_MATH_USE_FLOAT128
#include <boost/multiprecision/complex128.hpp>
#endif
#include <boost/math/fft/fftw_backend.hpp>

#include <array>
#include <complex>

#include <boost/core/demangle.hpp>
#include <iostream>

template<class Backend, int N >
void test()
{
  using T = typename Backend::value_type;
  std::cout << "Testing: " << boost::core::demangle(typeid(T).name()) << "\n";
  Backend P(N);    
  alignas(1)  std::array<T,N> A;
  alignas(16) std::array<T,N> B;
  
  P.forward(A.data(),A.data()+N,B.data());
  P.backward(A.data(),A.data()+N,B.data());
  
  P.forward(A.data(),A.data()+N,A.data());
  P.backward(A.data(),A.data()+N,A.data());
  
  P.forward(B.data(),B.data()+N,B.data());
  P.backward(B.data(),B.data()+N,B.data());
}
template<class Backend, int N >
void test_real()
{
  using T = typename Backend::value_type;
  std::cout << "Testing: " << boost::core::demangle(typeid(T).name()) << "\n";
  Backend P(N);    
  alignas(1)  std::array<T,N> A;
  alignas(16) std::array<T,N> B;
  
  P.real_to_halfcomplex(A.data(),A.data()+N,B.data());
  P.halfcomplex_to_real(A.data(),A.data()+N,B.data());
  
  P.real_to_halfcomplex(A.data(),A.data()+N,A.data());
  P.halfcomplex_to_real(A.data(),A.data()+N,A.data());
  
  P.real_to_halfcomplex(B.data(),B.data()+N,B.data());
  P.halfcomplex_to_real(B.data(),B.data()+N,B.data());
}

int main()
{
  using boost::math::fft::fftw_dft;
  using boost::math::fft::fftw_rdft;
  test<fftw_dft<std::complex<float>>,      16 >();
  test<fftw_dft<std::complex<double>>,     16 >();
  test<fftw_dft<std::complex<long double>>,16 >();
#ifdef BOOST_MATH_USE_FLOAT128
  test<fftw_dft<boost::multiprecision::complex128>,16 >();
#endif
  
  test_real<fftw_rdft<float>,      16 >();
  test_real<fftw_rdft<double>,     16 >();
  test_real<fftw_rdft<long double>,16 >();
#ifdef BOOST_MATH_USE_FLOAT128
  test_real<fftw_rdft<boost::multiprecision::float128>,16 >();
#endif
  return 0;
}

