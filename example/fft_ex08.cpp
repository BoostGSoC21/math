///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
/*
  boost::math::fft Example for non-complex types
  Use of DFT for Number Theoretical Transform.
*/
#include <boost/math/fft/bsl_backend.hpp>
namespace fft = boost::math::fft;

#include <iostream>
#include <vector>
#include "fft_test_helpers.hpp"

class Z337
{
public:
  typedef int integer;
  static constexpr integer mod{337}; // 337 = 2*2*2*2*3*7 + 1
};

int convolution()
/*
  product of two integer by means of the NTT,
  using the convolution theorem
*/
{
  int errors = 0;
  using M_int = fft::my_modulo_lib::mint<Z337>;
  // 85 is a primitive root of 337,
  // ie. the smallest number k for such that 85^k mod 337 = 1 is k=phi(337)=336
  const M_int w{85};
  const M_int inv_8{M_int{8}.inverse()};

  // Multiplying 1234 times 5678 = 7006652
  std::vector<M_int> A{4, 3, 2, 1, 0, 0, 0, 0};
  std::vector<M_int> B{8, 7, 6, 5, 0, 0, 0, 0};

  // forward FFT
  fft::bsl_algebraic_transform::forward(A.cbegin(),A.cend(),A.begin(), w);
  fft::bsl_algebraic_transform::forward(B.cbegin(),B.cend(),B.begin(), w);

  // convolution in Fourier space
  std::vector<M_int> AB;
  std::transform(A.begin(), A.end(), B.begin(),
                 std::back_inserter(AB),
                 [](M_int x, M_int y) { return x * y; });

  // backwards FFT
  fft::bsl_algebraic_transform::backward(AB.cbegin(),AB.cend(),AB.begin(),w);
  std::transform(AB.begin(), AB.end(), AB.begin(),
                 [&inv_8](M_int x) { return x * inv_8; });

  // carry the remainders in base 10
  std::vector<int> C;
  M_int r{0};
  for (auto x : AB)
  {
    auto y = x + r;
    C.emplace_back(int(y) % 10);
    r = M_int(int(y) / 10);
  }
  // yields 7006652
  if(static_cast<int>(C.size())!=8) errors++;
  if(C[0]!=2) errors++;
  if(C[1]!=5) errors++;
  if(C[2]!=6) errors++;
  if(C[3]!=6) errors++;
  if(C[4]!=0) errors++;
  if(C[5]!=0) errors++;
  if(C[6]!=7) errors++;
  if(C[7]!=0) errors++;
  return errors;
}
int main()
{
  return convolution();
}

