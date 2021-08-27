#include "math_unit_test.hpp"

#include <boost/config.hpp>
#include <boost/multiprecision/number.hpp>
#include <boost/math/fft/multiprecision_complex.hpp>
#ifdef BOOST_MATH_USE_FLOAT128
#include <boost/multiprecision/complex128.hpp>
#endif
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

#include <boost/math/fft/bsl_backend.hpp>
#if defined(__GNUC__)
#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>
#endif
#include <vector>
#include <iterator>
#include <iostream>
#include <boost/core/demangle.hpp>
#include <boost/random.hpp>

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

template<class Backend, class Complex = std::complex<typename Backend::value_type>>
void test_r2c(int n,int tolerance=1)
{
  using Real    = typename Backend::value_type;
  // using Complex = boost::multiprecision::complex<Real>;
  const Real tol = tolerance*std::numeric_limits<Real>::epsilon();
  
  boost::random::mt19937 rng;
  boost::random::uniform_real_distribution<double> U(0.0,1.0);
  
  std::vector<Real> A(n);
  std::vector<Complex> B(A.size());
  
  for(unsigned int i=0;i<A.size();++i)
  {
    A[i] = U(rng);
    B[i] = Complex{A[i],0.0};
  }
  
  std::vector<Real> HC,iHC;
  Backend rplan(A.size());
  rplan.real_to_halfcomplex(A.begin(),A.end(),std::back_inserter(HC));
  rplan.halfcomplex_to_real(HC.begin(),HC.end(),std::back_inserter(iHC));
  //print(A);
  //print(HC);
  //print(iHC);
  
  std::vector<Complex> TB;
  bsl_dft<Complex> cplan(B.size());
  cplan.forward(B.begin(),B.end(),std::back_inserter(TB));
  
  using std::abs;
  
  // check if the inverse recovers the original array
  {
    Real diff{0.0};
    const Real inv_n = Real{1.0}/n;
    for(unsigned int i=0;i<A.size();++i)
    {
      diff += abs(A[i]-iHC[i]*inv_n);
    }
    diff /= A.size();
    CHECK_MOLLIFIED_CLOSE(Real{0.0},diff,tol);
  }
  // check if the halfcomplex contains the non-redundant complex components
  {
    Real diff{0.0};
    
    diff += abs(TB[0]-HC[0]);
    for(int i=1,j=n-1;i<=j;++i,--j)
    {
      diff += abs(TB[i].real()-HC[i]);
      
      if(i<j)
      diff += abs(TB[j].imag()-HC[j]);
    }
    diff /= n;
    CHECK_MOLLIFIED_CLOSE(Real{0.0},diff,tol);
  }
}

int main()
{
  // corner cases
#if defined(__GNUC__)
  test_r2c<fftw_rdft<double>>(1);
  test_r2c<gsl_rdft<double>>(1);
#endif
  test_r2c<bsl_rdft<double>>(1);
  
#ifdef BOOST_MATH_USE_FLOAT128
  test_r2c< bsl_rdft<boost::multiprecision::float128>,
            boost::multiprecision::complex128 >(1);
#endif
    test_r2c< bsl_rdft<boost::multiprecision::cpp_bin_float_100>,
              boost::multiprecision::cpp_complex_100 >(1);
    test_r2c< bsl_rdft<boost::multiprecision::cpp_bin_float_quad>,
              boost::multiprecision::cpp_complex_quad >(1);
  
  // primes 
  for(auto n: std::vector<int>{2,3,5,7,11,13,17,19})
  {
#if defined(__GNUC__)
    test_r2c<fftw_rdft<double>>(n,4);
    test_r2c<gsl_rdft<double>>(n,16);
#endif
    test_r2c<bsl_rdft<double>>(n,4);
    
#ifdef BOOST_MATH_USE_FLOAT128
    test_r2c< bsl_rdft<boost::multiprecision::float128>,
              boost::multiprecision::complex128 >(n,4);
#endif
    test_r2c< bsl_rdft<boost::multiprecision::cpp_bin_float_100>,
              boost::multiprecision::cpp_complex_100 >(n,4);
    test_r2c< bsl_rdft<boost::multiprecision::cpp_bin_float_quad>,
              boost::multiprecision::cpp_complex_quad >(n,4);
  }
  
  // powers of two
  for(auto n: std::vector<int>{2,4,8,16,32,64,128})
  {
#if defined(__GNUC__)
    test_r2c<fftw_rdft<double>>(n,4);
    test_r2c<gsl_rdft<double>>(n,4);
#endif
    test_r2c<bsl_rdft<double>>(n,4);
    
#ifdef BOOST_MATH_USE_FLOAT128
    test_r2c< bsl_rdft<boost::multiprecision::float128>,
              boost::multiprecision::complex128 >(n,4);
#endif
    test_r2c< bsl_rdft<boost::multiprecision::cpp_bin_float_100>,
              boost::multiprecision::cpp_complex_100 >(n,4);
    test_r2c< bsl_rdft<boost::multiprecision::cpp_bin_float_quad>,
              boost::multiprecision::cpp_complex_quad >(n,4);
  }
  // composite
  for(auto n: std::vector<int>{6,9,10,12,14,15,18,20,21,22,24,25,26,27,28,30})
  {
#if defined(__GNUC__)
    test_r2c<fftw_rdft<double>>(n,8);
    test_r2c<gsl_rdft<double>>(n,16);
#endif
    test_r2c<bsl_rdft<double>>(n,4);
    
#ifdef BOOST_MATH_USE_FLOAT128
    test_r2c< bsl_rdft<boost::multiprecision::float128>,
              boost::multiprecision::complex128 >(n,4);
#endif
    test_r2c< bsl_rdft<boost::multiprecision::cpp_bin_float_100>,
              boost::multiprecision::cpp_complex_100 >(n,4);
    test_r2c< bsl_rdft<boost::multiprecision::cpp_bin_float_quad>,
              boost::multiprecision::cpp_complex_quad >(n,4);
  }  

  // TODO: fix this
  // test_r2c< bsl_rdft<boost::multiprecision::mpfr_float_100>,
  //           boost::multiprecision::mpc_complex_100 >(n,4;
  return boost::math::test::report_errors();
}

