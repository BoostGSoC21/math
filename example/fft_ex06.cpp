///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
/*
    boost::math::fft example 06
    
    Fast Polynomial Multiplication
*/

#include <boost/math/fft/fftw_backend.hpp>
#include <iostream>
#include <vector>
#include <complex>

template<class T>
std::vector<T> multiply(const std::vector<T>& A, const std::vector<T>& B)
{
  const std::size_t N = A.size();
  std::vector<T> TA(N),TB(N);
  boost::math::fft::fftw_rfft<T> P(N); 
  P.real_to_halfcomplex(A.begin(),A.end(),TA.begin());
  P.real_to_halfcomplex(B.begin(),B.end(),TB.begin());
  
  std::vector<T> C(N);
  
  C[0]=TA[0]*TB[0];
  for(unsigned int i=1;i+1<N;i+=2)
  {
    C[i] = TA[i]*TB[i]-TA[i+1]*TB[i+1];
    C[i+1] = TA[i]*TB[i+1] + TA[i+1]*TB[i];
  }
  if(N%2==0)
  {
    C.back() = TA.back()*TB.back();
  }
  
  P.halfcomplex_to_real(C.begin(),C.end(),C.begin());
  std::transform(C.begin(), C.end(), C.begin(),
                 [N](T x) { return x / N; });
  return C;
}

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

int main()
{
  std::vector<double> A{1.,4.,-5.,1.,0.,0.,0.,0.};
  std::vector<double> B{-1.,1.,2.,3.,0.,0.,0.,0.};
  
  auto C = multiply(A,B);
  print(C);
  return 0;
}


