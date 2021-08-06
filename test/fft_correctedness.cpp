#include "math_unit_test.hpp"
#include <boost/math/fft/multiprecision_complex.hpp>
#include <boost/math/fft/bsl_backend.hpp>
#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>
#include <boost/math/fft/algorithms.hpp>
#include <boost/math/constants/constants.hpp>
#ifdef BOOST_MATH_USE_FLOAT128
#include <boost/multiprecision/complex128.hpp>
#endif
#include <boost/multiprecision/cpp_complex.hpp>
// TODO:
//#include <boost/multiprecision/mpfr.hpp>
//#include <boost/multiprecision/mpc.hpp>
#include <boost/random.hpp>

#include "fft_test_helpers.hpp"

#include <type_traits>
#include <vector>
#include <limits>

using namespace boost::math::fft;

template<class T>
void convolution_brute_force(
  const T* first1, const T* last1, 
  const T* first2,
  T* out)
{
  long N = std::distance(first1,last1);
  for(long i=0;i<N;++i)
  {
    T sum{0};
    for(int j=0;j<N;++j)
    {
      sum += first1[j] * first2[(i-j+N) % N];
    }
    out[i] = sum;
  }
}

template<class T>
void dft_forward_bruteForce(
  const T* in_beg, const T* in_end,
  T* out)
{
  ::boost::math::fft::detail::complex_dft_prime_bruteForce(
    in_beg,in_end,out,
    1,
    std::allocator<T>{});
}

template<template<class ...Args> class backend_t, class T>
void test_directly(unsigned int N, int tolerance)
{
  using Complex = typename detail::select_complex<T>::type;
  const T tol = tolerance*std::numeric_limits<T>::epsilon();
  
  // ...
  boost::random::mt19937 rng;
  boost::random::uniform_real_distribution<T> U(0.0,1.0);
  {
    std::vector<Complex> A(N),B(N),C(N);
    
    for(auto& x: A)
    {
        x.real( U(rng) );
        x.imag( U(rng) );
    }
    backend_t<Complex> plan(N);
    plan.forward(A.begin(),A.end(),B.begin());
    dft_forward_bruteForce(A.data(),A.data()+N,C.data());
    
    T diff{0.0};
    
    for(size_t i=0;i<N;++i)
    {
        using std::norm;
        diff += norm(B[i]-C[i]);
    }
    using std::sqrt;
    diff = sqrt(diff)/N;
    CHECK_MOLLIFIED_CLOSE(T{0.0},diff,tol);
  }
}

template<class backend_t>
void test_convolution(unsigned int N, int tolerance)
{
  using Complex = typename backend_t::value_type;
  using T = typename Complex::value_type;
  
  // using Complex = typename detail::select_complex<T>::type;
  const T tol = tolerance*std::numeric_limits<T>::epsilon();
  
  // ...
  boost::random::mt19937 rng;
  boost::random::uniform_real_distribution<T> U(0.0,1.0);
  {
    std::vector<Complex> A(N),B(N),C(N);
    
    for(auto& x: A)
    {
        x.real( U(rng) );
        x.imag( U(rng) );
    }
    for(auto& x: B)
    {
        x.real( U(rng) );
        x.imag( U(rng) );
    }
    convolution_brute_force(A.data(),A.data()+N,B.data(),C.data());
    
    std::vector<Complex> C_candidate;
    transform<backend_t>::convolution(A.begin(),A.end(),B.begin(),std::back_inserter(C_candidate));
    
    T diff{0.0};
    
    for(size_t i=0;i<N;++i)
    {
        using std::norm;
        diff += norm(C[i]-C_candidate[i]);
    }
    using std::sqrt;
    diff = sqrt(diff)/N;
    CHECK_MOLLIFIED_CLOSE(T{0.0},diff,tol);
  }
}

template<class Backend>
void test_fixed_transforms(int tolerance)
{
  // using Complex = typename detail::select_complex<T>::type;
  using Complex = typename Backend::value_type;
  using real_value_type = typename Complex::value_type;
  const real_value_type tol = tolerance*std::numeric_limits<real_value_type>::epsilon();
  {
    std::vector< Complex > A{1.0},B(1);
    Backend plan(A.size());
    plan.forward(A.data(),A.data()+A.size(),B.data());
    CHECK_MOLLIFIED_CLOSE(real_value_type{1.0},B[0].real(),0);
    CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},B[0].imag(),0);
  }
  {
    std::vector< Complex > A{1.0,1.0},B(2);
    Backend plan(A.size());
    plan.forward(A.data(),A.data()+A.size(),B.data());
    CHECK_MOLLIFIED_CLOSE(real_value_type{2.0},B[0].real(),tol);
    CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},B[0].imag(),tol);
    
    CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},B[1].real(),tol);
    CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},B[1].imag(),tol);
  }
  {
    std::vector< Complex > A{1.0,1.0,1.0},B(3);
    Backend plan(A.size());
    plan.forward(A.data(),A.data()+A.size(),B.data());
    CHECK_MOLLIFIED_CLOSE(real_value_type{3.0},B[0].real(),tol);
    CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},B[0].imag(),tol);
    
    CHECK_MOLLIFIED_CLOSE(
        real_value_type{0.0},B[1].real(),tol);
    CHECK_MOLLIFIED_CLOSE(
        real_value_type{0.0},B[1].imag(),tol);
    
    CHECK_MOLLIFIED_CLOSE(
        real_value_type{0.0},B[2].real(),tol);
    CHECK_MOLLIFIED_CLOSE(
        real_value_type{0.0},B[2].imag(),tol);
  }
  {
    std::vector< Complex > A{1.0,1.0,1.0};
    Backend plan(A.size());
    plan.forward(A.data(),A.data()+A.size(),A.data());
    CHECK_MOLLIFIED_CLOSE(real_value_type{3.0},A[0].real(),tol);
    CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},A[0].imag(),tol);
    
    CHECK_MOLLIFIED_CLOSE(
        real_value_type{0.0},A[1].real(),tol);
    CHECK_MOLLIFIED_CLOSE(
        real_value_type{0.0},A[1].imag(),tol);
    
    CHECK_MOLLIFIED_CLOSE(
        real_value_type{0.0},A[2].real(),tol);
    CHECK_MOLLIFIED_CLOSE(
        real_value_type{0.0},A[2].imag(),tol);
  }
}


template<class Backend>
void test_inverse(int N, int tolerance)
{
  using Complex = typename Backend::value_type;
  using real_value_type = typename Complex::value_type;
  const real_value_type tol = tolerance*std::numeric_limits<real_value_type>::epsilon();
  
  boost::random::mt19937 rng;
  boost::random::uniform_real_distribution<real_value_type> U(0.0,1.0);
  {
    std::vector<Complex> A(N),B(N),C(N);
    
    for(auto& x: A)
    {
        x.real( U(rng) );
        x.imag( U(rng) );
    }
    Backend plan(N);
    plan.forward(A.data(),A.data()+A.size(),B.data());
    plan.backward(B.data(),B.data()+B.size(),C.data());
    
    const real_value_type inverse_N = real_value_type{1.0}/N;
    for(auto &x : C)
      x *= inverse_N;
    
    real_value_type diff{0.0};
    
    for(size_t i=0;i<A.size();++i)
    {
        using std::norm;
        diff += norm(A[i]-C[i]);
    }
    using std::sqrt;
    diff = sqrt(diff)*inverse_N;
    CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},diff,tol);
  }
}

template<class T>
using complex_fftw_dft = fftw_dft< typename detail::select_complex<T>::type  >;

template<class T>
using complex_gsl_dft = gsl_dft< typename detail::select_complex<T>::type  >;

template<class T>
using complex_bsl_dft = bsl_dft< typename detail::select_complex<T>::type  >;

//template<class T>
//using complex_rader_dft = rader_dft< typename detail::select_complex<T>::type  >;
//
//template<class T>
//using complex_bruteForce_dft = bruteForce_dft< typename detail::select_complex<T>::type  >;
//
//template<class T>
//using complex_bruteForce_cdft = bruteForce_cdft< typename detail::select_complex<T>::type  >;
//
//template<class T>
//using complex_composite_dft = composite_dft< typename detail::select_complex<T>::type  >;
//
//template<class T>
//using complex_composite_cdft = composite_cdft< typename detail::select_complex<T>::type  >;
//
//template<class T>
//using complex_power2_dft = power2_dft< typename detail::select_complex<T>::type  >;
//
//template<class T>
//using complex_power2_cdft = power2_cdft< typename detail::select_complex<T>::type  >;

int main()
{
  test_fixed_transforms<complex_fftw_dft<float>>(1);
  test_fixed_transforms<complex_fftw_dft<double>>(1);
  test_fixed_transforms<complex_fftw_dft<long double>>(1);
  
   
#ifdef BOOST_MATH_USE_FLOAT128
  test_fixed_transforms<complex_fftw_dft<boost::multiprecision::float128>>(1);
#endif
  
  test_fixed_transforms<complex_gsl_dft<double>>(1);

//  test_fixed_transforms<complex_bruteForce_dft<float> >(4);
//  test_fixed_transforms<complex_bruteForce_dft<double> >(4);
//  test_fixed_transforms<complex_bruteForce_dft<long double> >(4);
  
//  test_fixed_transforms<complex_bruteForce_cdft<float> >(4);
//  test_fixed_transforms<complex_bruteForce_cdft<double> >(4);
//  test_fixed_transforms<complex_bruteForce_cdft<long double> >(4);
//  
//  
//  test_fixed_transforms<complex_composite_dft<float>>(4);
//  test_fixed_transforms<complex_composite_dft<double>>(4);
//  test_fixed_transforms<complex_composite_dft<long double>>(4);
//  
//  test_fixed_transforms<complex_composite_cdft<float>>(4);
//  test_fixed_transforms<complex_composite_cdft<double>>(4);
//  test_fixed_transforms<complex_composite_cdft<long double>>(4);
  
  test_fixed_transforms<complex_bsl_dft<float>>(2);
  test_fixed_transforms<complex_bsl_dft<double>>(2);
  test_fixed_transforms<complex_bsl_dft<long double>>(2);
#ifdef BOOST_MATH_USE_FLOAT128
  test_fixed_transforms<complex_bsl_dft< boost::multiprecision::float128 >>(1);
#endif
  test_fixed_transforms<complex_bsl_dft< boost::multiprecision::cpp_bin_float_50>>(2);
  test_fixed_transforms<complex_bsl_dft< boost::multiprecision::cpp_bin_float_100 >>(2);
  test_fixed_transforms<complex_bsl_dft< boost::multiprecision::cpp_bin_float_quad >>(2);
  // TODO:
  //test_fixed_transforms<complex_bsl_dft< boost::multiprecision::mpfr_float_100 >>(1);
  
  for(int i=1;i<=(1<<10); i*=2)
  {
    test_directly<fftw_dft,double>(i,i*8);
    test_directly<gsl_dft,double>(i,i*8);
    test_directly<bsl_dft,double>(i,i*8);
    
    test_inverse<complex_fftw_dft<float>>(i,1);
    test_inverse<complex_fftw_dft<double>>(i,1);
    test_inverse<complex_fftw_dft<long double>>(i,1);
#ifdef BOOST_MATH_USE_FLOAT128
    test_inverse<complex_fftw_dft<boost::multiprecision::float128>>(i,1);
#endif
    test_inverse<complex_gsl_dft<double>>(i,1);
    test_inverse<complex_bsl_dft<float>>(i,1);
    test_inverse<complex_bsl_dft<double>>(i,1);
    test_inverse<complex_bsl_dft<long double>>(i,1);
#ifdef BOOST_MATH_USE_FLOAT128
    test_inverse<complex_bsl_dft<boost::multiprecision::float128>>(i,1);
#endif
    test_inverse<complex_bsl_dft<boost::multiprecision::cpp_bin_float_50>>(i,1);
    
//    test_inverse<complex_power2_dft<float>>(i,32);
//    test_inverse<complex_power2_dft<double>>(i,32);
//    test_inverse<complex_power2_dft<long double>>(i,32);
//    test_inverse<complex_power2_dft<boost::multiprecision::cpp_bin_float_50>>(i,32);
//#ifdef BOOST_MATH_USE_FLOAT128
//    test_inverse<complex_power2_dft<boost::multiprecision::float128>>(i,32);
//#endif
//    
//    test_inverse<complex_power2_cdft<float>>(i,1);
//    test_inverse<complex_power2_cdft<double>>(i,1);
//    test_inverse<complex_power2_cdft<long double>>(i,1);
//    test_inverse<complex_power2_cdft<boost::multiprecision::cpp_bin_float_50>>(i,1);
//#ifdef BOOST_MATH_USE_FLOAT128
//    test_inverse<complex_power2_cdft<boost::multiprecision::float128>>(i,1);
//#endif
  }
  for(int i=1;i<=1000; i*=10)
  {
    test_directly<fftw_dft,double>(i,i*8);
    test_directly<gsl_dft,double>(i,i*8);
    test_directly<bsl_dft,double>(i,i*8);
    
    test_inverse<complex_fftw_dft<float>>(i,1);
    test_inverse<complex_fftw_dft<double>>(i,1);
    test_inverse<complex_fftw_dft<long double>>(i,1);
#ifdef BOOST_MATH_USE_FLOAT128
    test_inverse<complex_fftw_dft<boost::multiprecision::float128>>(i,1);
#endif
    test_inverse<complex_gsl_dft<double>>(i,1);
    test_inverse<complex_bsl_dft<float>>(i,1);
    test_inverse<complex_bsl_dft<double>>(i,1);
    test_inverse<complex_bsl_dft<long double>>(i,1);
#ifdef BOOST_MATH_USE_FLOAT128
    test_inverse<complex_bsl_dft<boost::multiprecision::float128>>(i,1);
#endif
    test_inverse<complex_bsl_dft<boost::multiprecision::cpp_bin_float_50>>(i,1);
  }
  for(auto i : std::vector<int>{2,3,5,7,11,13,17,23,29,31})
  {
    test_directly<fftw_dft,double>(i,i*8);
    test_directly<gsl_dft,double>(i,i*8);
    test_directly<bsl_dft,double>(i,i*8);
    
    test_inverse<complex_fftw_dft<float>>(i,1);
    test_inverse<complex_fftw_dft<double>>(i,1);
    test_inverse<complex_fftw_dft<long double>>(i,1);
#ifdef BOOST_MATH_USE_FLOAT128
    test_inverse<complex_fftw_dft<boost::multiprecision::float128>>(i,1);
#endif
    test_inverse<complex_gsl_dft<double>>(i,1);
    test_inverse<complex_bsl_dft<float>>(i,2);
    test_inverse<complex_bsl_dft<double>>(i,2);
    test_inverse<complex_bsl_dft<long double>>(i,2);
#ifdef BOOST_MATH_USE_FLOAT128
    test_inverse<complex_bsl_dft<boost::multiprecision::float128>>(i,2);
#endif
    test_inverse<complex_bsl_dft<boost::multiprecision::cpp_bin_float_50>>(i,2);

    
//    if(i>2)
//    {
//      test_inverse<complex_rader_dft<float> >(i,2);
//      test_inverse<complex_rader_dft<double> >(i,2);
//      test_inverse<complex_rader_dft<long double> >(i,2);
//#ifdef BOOST_MATH_USE_FLOAT128
//      test_inverse<complex_rader_dft<boost::multiprecision::float128> >(i,2);
//#endif
//      test_inverse<complex_rader_dft<boost::multiprecision::cpp_bin_float_50> >(i,2);
//    }
  }
  
  for(int i=1;i<=100;++i)
  {
    test_directly<fftw_dft,double>(i,i*8);
    test_directly<gsl_dft,double>(i,i*8);
    test_directly<bsl_dft,double>(i,i*8);
    
    test_inverse<complex_fftw_dft<float>>(i,1);
    test_inverse<complex_fftw_dft<double>>(i,1);
    test_inverse<complex_fftw_dft<long double>>(i,1);
#ifdef BOOST_MATH_USE_FLOAT128
    test_inverse<complex_fftw_dft<boost::multiprecision::float128>>(i,1);
#endif
    test_inverse<complex_gsl_dft<double>>(i,1);
    test_inverse<complex_bsl_dft<float>>(i,2);
    test_inverse<complex_bsl_dft<double>>(i,2);
    test_inverse<complex_bsl_dft<long double>>(i,2);
#ifdef BOOST_MATH_USE_FLOAT128
    test_inverse<complex_bsl_dft<boost::multiprecision::float128>>(i,2);
#endif
    test_inverse<complex_bsl_dft<boost::multiprecision::cpp_bin_float_50>>(i,2);
    
//    if(i<=20)
//    {
//      test_inverse<complex_bruteForce_dft<float> >(i,i*8);
//      test_inverse<complex_bruteForce_dft<double> >(i,i*8);
//      test_inverse<complex_bruteForce_dft<long double> >(i,i*8);
//      test_inverse<complex_bruteForce_dft<boost::multiprecision::cpp_bin_float_50> >(i,i*8);
//#ifdef BOOST_MATH_USE_FLOAT128
//      test_inverse<complex_bruteForce_dft<boost::multiprecision::float128> >(i,i*8);
//#endif
//      
//      test_inverse<complex_bruteForce_cdft<float> >(i,i*8);
//      test_inverse<complex_bruteForce_cdft<double> >(i,i*8);
//      test_inverse<complex_bruteForce_cdft<long double> >(i,i*8);
//      test_inverse<complex_bruteForce_cdft<boost::multiprecision::cpp_bin_float_50> >(i,i*8);
//#ifdef BOOST_MATH_USE_FLOAT128
//      test_inverse<complex_bruteForce_cdft<boost::multiprecision::float128> >(i,i*8);
//#endif
//    }
    
    test_convolution<fftw_dft<std::complex<double>>>(i,i*8);
    test_convolution<gsl_dft<std::complex<double>>>(i,i*8);
    test_convolution<bsl_dft<std::complex<double>>>(i,i*8);
    
//    test_inverse<complex_composite_dft<float>>(i,i*8);
//    test_inverse<complex_composite_dft<double>>(i,i*8);
//    test_inverse<complex_composite_dft<long double>>(i,i*8);
//    
//    test_inverse<complex_composite_cdft<float>>(i,2);
//    test_inverse<complex_composite_cdft<double>>(i,2);
//    test_inverse<complex_composite_cdft<long double>>(i,2);
  }
  // TODO: can we print a useful compilation error message for the following
  // illegal case?
  // dft<std::complex<int>> P(3);   
  return boost::math::test::report_errors();
}
