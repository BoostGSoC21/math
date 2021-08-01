#include "math_unit_test.hpp"
#include <boost/math/fft/bsl_backend.hpp>
#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>
#include <vector>
#include <iterator>
#include <iostream>

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
  using Complex = typename Backend::value_type;
  using Real    = typename Complex::value_type;
  
  std::vector<Real> A{1.,2.,3.};
  std::vector<Complex> B;
    
  Backend plan(A.size());
  plan.r2c(A.begin(),A.end(),std::back_inserter(B));

  print(B);
}

int main()
{
  test_r2c< fftw_dft<std::complex<double> > >();
  //test_r2c< gsl_dft<std::complex<double> > >();
  //test_r2c< bsl_dft<std::complex<double> > >();
  return boost::math::test::report_errors();
}

