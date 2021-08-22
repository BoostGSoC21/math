///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
/*
    boost::math::fft example 07
    
    Real transforms with GSL, FFTW and Boost backends
*/

#include <boost/random.hpp>
#include <boost/math/fft/bsl_backend.hpp>
#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>
#include <boost/math/fft/real_algorithms.hpp>
#include <iostream>
#include <vector>
#include <complex>

template<class T>
void print(const std::vector<T>& V)
{
  std::cout << "size(V) = " << V.size() << "\n";
  std::cout << "[";
  for(auto x: V)
  {
    std::cout << x << ", ";
  }
  std::cout << "]\n";
}

void check(int n)
{
  boost::random::mt19937 rng;
  boost::random::uniform_real_distribution<double> U(0.0,1.0);
  
  std::vector< double > V(n);
  for(auto& x: V)
      x = U(rng);
  std::cout << "Original data:\n";
  print(V);
  
  std::vector<double> A,B,C;
  
  boost::math::fft::fftw_rdft< double > P1(V.size());
  P1.real_to_halfcomplex(V.begin(),V.end(),std::back_inserter(A));
  //std::cout << "FFTW:\n";
  //print(A);
  
  boost::math::fft::gsl_rdft< double > P2(V.size());
  P2.real_to_halfcomplex(V.begin(),V.end(),std::back_inserter(B));
  //std::cout << "GSL:\n";
  //print(B);
  
  
  boost::math::fft::bsl_rdft< double > P3(V.size());
  P3.real_to_halfcomplex(V.begin(),V.end(),std::back_inserter(C));
  //std::cout << "Boost:\n";
  //print(C);
  
  std::cout << "Boost inverse:\n";
  P3.halfcomplex_to_real(C.begin(),C.end(),C.begin());
  const double inv_n = 1.0/C.size();
  for(auto &x : C) x *= inv_n;
  print(C);
  
  std::vector< std::complex<double> > cplx_V(V.begin(),V.end()),cplx_C;
  boost::math::fft::bsl_dft< std::complex<double> > P4(V.size());
  P4.forward(cplx_V.begin(),cplx_V.end(),std::back_inserter(cplx_C));
  //std::cout << "Boost complex:\n";
  //print(cplx_C);
}

int main()
{
  check(1);
  
  check(2);
  check(3);
  check(5);
  check(7);
  
  check(4);
  check(8);
  check(16);
  check(32);
  
  check(6);
  check(9);
  check(10);
  check(12);
  return 0;
}



