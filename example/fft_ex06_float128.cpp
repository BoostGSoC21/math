///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
/*
    boost::math::fft example 06
    
    Fast Polynomial Multiplication with quadmath
*/

#include <boost/math/fft/bsl_backend.hpp>
#include <boost/multiprecision/float128.hpp>
#include <iostream>
#include <vector>
#include <exception>

template<class T>
std::vector<T> multiply_complex(const std::vector<T>& A, const std::vector<T>& B)
{
  const std::size_t N = A.size();
  std::vector< boost::multiprecision::complex<T> > TA(N),TB(N);
  boost::math::fft::bsl_rdft<T> P(N); 
  P.real_to_complex(A.begin(),A.end(),TA.begin());
  P.real_to_complex(B.begin(),B.end(),TB.begin());
  
  std::vector<T> C(N);
  
  for(unsigned int i=0;i<N;++i)
  {
    TA[i]*=TB[i];
  }
  
  P.complex_to_real(TA.begin(),TA.end(),C.begin());
  std::transform(C.begin(), C.end(), C.begin(),
                 [N](T x) { return x / N; });
  
  return C;
}

template<class T>
T difference(const std::vector<T>& A, const std::vector<T>& B)
{
  using std::abs;
  T diff{};
  if(A.size()!=B.size()) return -1;
  for(unsigned int i=0;i<A.size();++i)
  {
    diff += abs(A[i]-B[i]);
  }
  return diff;
}

template<typename Real>
void multiply() {
  using std::abs;
  std::vector<Real> A{1.,4.,-5.,1.,0.,0.,0.,0.};
  std::vector<Real> B{-1.,1.,2.,3.,0.,0.,0.,0.};
  std::vector<Real> C{-1,-3,11,5,3,-13,3,0};
  
  std::vector<Real> result;
  Real diff;
  
  result = multiply_complex(A,B);
  diff = difference(result,C);
  if(abs(diff)>1e-3) 
    throw std::runtime_error("wrong result");
}

int main()
{
  multiply<::boost::multiprecision::float128>();
  return 0;
}


