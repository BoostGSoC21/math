///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
/*
  boost::math::fft 
  Example use of the algebraic DFT for complex numbers.
*/
#include <boost/math/fft/bsl_backend.hpp>
namespace fft = boost::math::fft;

#include <iostream>
#include <vector>

int algebraic_dft_example()
/*
    Use the Algebraic DFT algorithm to compute a complex DFT
*/
{
  using Real = double;
  using Complex = std::complex<Real>;
  
  std::vector<Complex> A{1.,-1.,3.,.5,9.},B,C;
  const int N = A.size();
  const Real w_phase = 2*boost::math::constants::pi<Real>()/N;
  const Complex w{cos(w_phase), -sin(w_phase)};

  fft::bsl_algebraic_transform::forward(A.cbegin(),A.cend(),std::back_inserter(B), w);
  fft::bsl_transform::forward(A.cbegin(),A.cend(),std::back_inserter(C));
  
  Real diff =0;
  for (int i=0;i<N;++i)
  {
    diff += abs(B[i]-C[i]);
  }
  return diff < 1e-6 ? 0 : 1;
}
int main()
{
  return algebraic_dft_example();
}

