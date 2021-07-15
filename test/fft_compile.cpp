///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/fft.hpp>
#include <boost/math/fft/bsl_backend.hpp>
#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

#include <complex>
#include <vector>
#include <array>

template<template<class ...Args> class backend_t, class T, int N >
void transform_api()
{   
  using boost::math::fft::transform;
  
  // test same type of iterator
  std::vector<T> A(N),B(A.size());
  transform<backend_t>::forward(A.begin(),A.end(),B.begin());
  transform<backend_t>::backward(A.begin(),A.end(),B.begin());
  
  // test with raw pointers
  transform<backend_t>::forward(A.data(),A.data()+A.size(),B.data());
  transform<backend_t>::backward(A.data(),A.data()+A.size(),B.data());

  const auto & cA = A;
  // const iterator as input
  transform<backend_t>::forward(cA.begin(),cA.end(),B.begin());
  transform<backend_t>::backward(cA.begin(),cA.end(),B.begin());
  
  // const pointer as input
  transform<backend_t>::forward(cA.data(),cA.data()+cA.size(),B.data());
  transform<backend_t>::backward(cA.data(),cA.data()+cA.size(),B.data());
  
  std::array<T,N> C;
  // input as vector::iterator, output as array::iterator
  transform<backend_t>::forward(A.begin(),A.end(),C.begin());
  transform<backend_t>::backward(A.begin(),A.end(),C.begin());
  transform<backend_t>::forward(A.data(),A.data()+A.size(),C.data());
  transform<backend_t>::backward(A.data(),A.data()+A.size(),C.data());
  
  // input as array::iterator, output as vector::iterator
  transform<backend_t>::forward(C.begin(),C.end(),B.begin());
  transform<backend_t>::backward(C.begin(),C.end(),B.begin());
  transform<backend_t>::forward(C.data(),C.data()+C.size(),B.data());
  transform<backend_t>::backward(C.data(),C.data()+C.size(),B.data());
}

template<class Backend >
void plan_api(int N)
{
  using T = typename Backend::value_type;
  Backend P(N);    
  std::vector<T> A(N),B(N);
  P.forward(A.data(),A.data()+N,B.data());
  P.backward(A.data(),A.data()+N,B.data());
  
  P.forward(A.begin(),A.end(),B.begin());
  P.backward(A.begin(),A.end(),B.begin());
}

struct my_type{};

void test_traits()
{
  using boost::math::fft::detail::is_complex;
  static_assert(is_complex< std::complex<float> >::value,"");
  static_assert(is_complex< std::complex<double> >::value,"");
  static_assert(is_complex< std::complex<long double> >::value,"");
  static_assert(is_complex< float >::value==false,"");
  static_assert(is_complex< my_type >::value==false,"");
  static_assert(is_complex< my_type >::value==false,"");
  static_assert(is_complex<
    std::complex<boost::multiprecision::cpp_bin_float_50> >::value,"");
  static_assert(is_complex< boost::multiprecision::cpp_complex_quad >::value,"");
}

int main()
{
  using boost::math::fft::fftw_dft;
  using boost::math::fft::gsl_dft;
  using boost::math::fft::bsl_dft;
  
  test_traits();
  
  transform_api<fftw_dft,std::complex<float>,      4 >();
  transform_api<fftw_dft,std::complex<double>,     4 >();
  transform_api<fftw_dft,std::complex<long double>,4 >();
  
  transform_api<gsl_dft,std::complex<double>,4 >();
  
  transform_api<bsl_dft,std::complex<float>,      4 >();
  transform_api<bsl_dft,std::complex<double>,     4 >();
  transform_api<bsl_dft,std::complex<long double>,4 >();
  
  plan_api<fftw_dft<std::complex<double>> >(5);
  plan_api<fftw_dft<std::complex<float>> >(5);
  plan_api<fftw_dft<std::complex<long double>> >(5);
  
  plan_api<gsl_dft<std::complex<double>> >(5);
  
  plan_api<bsl_dft<std::complex<float>> >(5);
  plan_api<bsl_dft<std::complex<double>> >(5);
  plan_api<bsl_dft<std::complex<long double>> >(5);
  
  return 0;
}
