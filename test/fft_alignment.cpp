///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/fft/fftw_backend.hpp>

#include <array>
#include <complex>

template<class Backend, int N >
void test()
{
  using T = typename Backend::value_type;
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

int main()
{
  using boost::math::fft::fftw_dft;
  test<fftw_dft<std::complex<float>>,      16 >();
  test<fftw_dft<std::complex<double>>,     16 >();
  test<fftw_dft<std::complex<long double>>,16 >();
  return 0;
}

