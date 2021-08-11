///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/fft/bsl_backend.hpp>
#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

#include <complex>
#include <vector>
#include <array>

template< class fft_engine,int N >
void transform_api()
{   
  using T = typename fft_engine::value_type;
  using boost::math::fft::transform;
  
  // test same type of iterator
  std::vector<T> A(N),B(A.size());
  transform<fft_engine>::forward(A.begin(),A.end(),B.begin());
  transform<fft_engine>::backward(A.begin(),A.end(),B.begin());
  
  // experimental
  // backend_t<T,std::allocator<T>>::static_forward(A.begin(),A.end(),B.begin());
  
  // test with raw pointers
  transform<fft_engine>::forward(A.data(),A.data()+A.size(),B.data());
  transform<fft_engine>::backward(A.data(),A.data()+A.size(),B.data());

  const auto & cA = A;
  // const iterator as input
  transform<fft_engine>::forward(cA.begin(),cA.end(),B.begin());
  transform<fft_engine>::backward(cA.begin(),cA.end(),B.begin());
  
  // const pointer as input
  transform<fft_engine>::forward(cA.data(),cA.data()+cA.size(),B.data());
  transform<fft_engine>::backward(cA.data(),cA.data()+cA.size(),B.data());
  
  std::array<T,N> C; // lets temporarily align this array here to avoid seg. fault
  // input as vector::iterator, output as array::iterator
  transform<fft_engine>::forward(A.begin(),A.end(),C.begin());
  transform<fft_engine>::backward(A.begin(),A.end(),C.begin());
  transform<fft_engine>::forward(A.data(),A.data()+A.size(),C.data());
  transform<fft_engine>::backward(A.data(),A.data()+A.size(),C.data());
  
  // input as array::iterator, output as vector::iterator
  transform<fft_engine>::forward(C.begin(),C.end(),B.begin());
  transform<fft_engine>::backward(C.begin(),C.end(),B.begin());
  transform<fft_engine>::forward(C.data(),C.data()+C.size(),B.data());
  transform<fft_engine>::backward(C.data(),C.data()+C.size(),B.data());
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
  using boost::multiprecision::is_boost_complex;
  static_assert(is_boost_complex< std::complex<float> >::value,"");
  static_assert(is_boost_complex< std::complex<double> >::value,"");
  static_assert(is_boost_complex< std::complex<long double> >::value,"");
  static_assert(is_boost_complex< float >::value==false,"");
  static_assert(is_boost_complex< my_type >::value==false,"");
  static_assert(is_boost_complex< my_type >::value==false,"");
  static_assert(is_boost_complex<
    std::complex<boost::multiprecision::cpp_bin_float_50> >::value==false,"");
  static_assert(is_boost_complex< boost::multiprecision::cpp_complex_quad >::value,"");
}

int main()
{
  #if defined(__GNUC__)
  using boost::math::fft::fftw_dft;
  using boost::math::fft::gsl_dft;
  #endif
  using boost::math::fft::bsl_dft;

  test_traits();

  #if defined(__GNUC__)
  transform_api<fftw_dft<std::complex<float>>,      4 >();
  transform_api<fftw_dft<std::complex<double>>,     4 >();
  transform_api<fftw_dft<std::complex<long double>>,4 >();
  #endif

  #if defined(__GNUC__)
  transform_api<gsl_dft<std::complex<double>>,4 >();
  #endif

  transform_api<bsl_dft<std::complex<float>>,      4 >();
  transform_api<bsl_dft<std::complex<double>>,     4 >();
  transform_api<bsl_dft<std::complex<long double>>,4 >();

  #if defined(__GNUC__)
  plan_api<fftw_dft<std::complex<double>> >(5);
  plan_api<fftw_dft<std::complex<float>> >(5);
  plan_api<fftw_dft<std::complex<long double>> >(5);
  #endif

  #if defined(__GNUC__)
  plan_api<gsl_dft<std::complex<double>> >(5);
  #endif

  plan_api<bsl_dft<std::complex<float>> >(5);
  plan_api<bsl_dft<std::complex<double>> >(5);
  plan_api<bsl_dft<std::complex<long double>> >(5);

  return 0;
}
