///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example using custom allocators
#include <boost/math/fft/bsl_backend.hpp>
#include <array>
#include <numeric>

#if __cplusplus >= 201700L
#include <memory>
#include <memory_resource>
#endif

using namespace boost::math::fft;

int main()
{
#if __cplusplus >= 201700L
  const std::size_t N = 100;
  std::array<char,30000> buf;
  std::pmr::monotonic_buffer_resource
    pool{buf.data(),buf.size(),std::pmr::null_memory_resource()};
  
  using Real = double;
  using Complex = std::complex<Real>;
  using allocator_type = std::pmr::polymorphic_allocator<Complex>;
  
  std::vector<Complex,allocator_type> A(N,Complex(),&pool);
  std::vector<Complex,allocator_type> B(N,Complex(),&pool);
  // ...
  bsl_dft<Complex,allocator_type> plan(N,&pool);
  plan.forward(A.begin(),A.end(),B.begin());
#endif
  return 0;
}

