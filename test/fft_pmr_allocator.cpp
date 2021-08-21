///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

int global_error_count{0};

#include <cstdlib>
#include <boost/math/fft/bsl_backend.hpp>
#include <array>
#include <numeric>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/container/pmr/global_resource.hpp>
#include "fft_test_helpers.hpp"


// Poison the call to global new and new[]
bool new_is_on{true};

void * operator new(size_t size)
{   
  if(new_is_on==false)
    throw std::bad_alloc{};
  void * p = std::malloc(size);
  return p;
}
void * operator new[](size_t size)
{
  if(new_is_on==false)
   throw std::bad_alloc{};
  void * p = std::malloc(size);
  return p;
}

#if __cplusplus >= 201700
void * operator new(size_t size, size_t align)
{
  if(new_is_on==false)
    throw std::bad_alloc{};
  void * p = std::aligned_alloc(size,align);
  return p;
}
void * operator new[](size_t size, size_t align)
{
  if(new_is_on==false)
   throw std::bad_alloc{};
  void * p = std::aligned_alloc(size,align);
  return p;
}
#endif

using namespace boost::math::fft;

namespace local
{
  std::vector<char> static_buf(200000U);

  boost::container::pmr::monotonic_buffer_resource
    pool{local::static_buf.data(),local::static_buf.size(),boost::container::pmr::null_memory_resource()};
}

template<class Backend>
void test_inverse(int N, int tolerance)
// Try the execution of complex FFT
{
  using allocator_type = typename Backend::allocator_type;
  using Complex = typename Backend::value_type;
  using real_value_type = typename Complex::value_type;
  const real_value_type tol = tolerance*std::numeric_limits<real_value_type>::epsilon();
  {
    std::vector<Complex,allocator_type> A(N,Complex(),&local::pool);
    std::vector<Complex,allocator_type> B(N,Complex(),&local::pool);
    std::vector<Complex,allocator_type> C(N,Complex(),&local::pool);
    for(auto& x: A)
    {
      x.real( 1. );
      x.imag( 2. );
    }
    Backend plan(N,&local::pool);
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
    using std::max;
    using std::abs;
    
    const real_value_type expected = 0.0;
    const real_value_type computed = diff;
    
    real_value_type denom = (max)(abs(expected), real_value_type(1));
    real_value_type mollified_relative_error = abs(expected - computed)/denom;
    if (mollified_relative_error > tol)
        ++global_error_count;
    // CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},diff,tol);
  }
}
template<class Backend>
void test_inverse_real(int N, int tolerance)
// Try the execution of real-to-halfcomplex 
{
  using allocator_type = typename Backend::allocator_type;
  using real_value_type = typename Backend::value_type;
  using plan_type = typename Backend::plan_type;
  
  const real_value_type tol = tolerance*std::numeric_limits<real_value_type>::epsilon();
  {
    std::vector<real_value_type,allocator_type> A(N,real_value_type(),&local::pool);
    std::vector<real_value_type,allocator_type> B(N,real_value_type(),&local::pool);
    std::vector<real_value_type,allocator_type> C(N,real_value_type(),&local::pool);

    std::iota(A.begin(),A.end(),1);

    plan_type plan(N,&local::pool);
    plan.real_to_halfcomplex(A.data(),A.data()+A.size(),B.data());
    plan.halfcomplex_to_real(B.data(),B.data()+B.size(),C.data());

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
    using std::max;
    using std::abs;
    
    const real_value_type expected = 0.0;
    const real_value_type computed = diff;
    
    real_value_type denom = (max)(abs(expected), real_value_type(1));
    real_value_type mollified_relative_error = abs(expected - computed)/denom;
    if (mollified_relative_error > tol)
        ++global_error_count;
    // CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},diff,tol);
  }
}
template<class Backend>
void test_inverse_real_complex(int N, int tolerance)
// Try the execution of real-to-complex 
{
  using allocator_type = typename Backend::allocator_type;
  using Complex = typename Backend::Complex;
  using complex_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<Complex>;
  using real_value_type = typename Backend::value_type;
  using plan_type = typename Backend::plan_type;
  
  const real_value_type tol = tolerance*std::numeric_limits<real_value_type>::epsilon();
  {
    std::vector<real_value_type,allocator_type> A(N,real_value_type(),&local::pool);
    std::vector<Complex,complex_allocator_type> B(N,Complex(),        &local::pool);
    std::vector<real_value_type,allocator_type> C(N,real_value_type(),&local::pool);

    std::iota(A.begin(),A.end(),1);

    plan_type plan(N,&local::pool);
    plan.real_to_complex(A.data(),A.data()+A.size(),B.data());
    plan.complex_to_real(B.data(),B.data()+B.size(),C.data());

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
    using std::max;
    using std::abs;

    const real_value_type expected = 0.0;
    const real_value_type computed = diff;

    real_value_type denom = (max)(abs(expected), real_value_type(1));
    real_value_type mollified_relative_error = abs(expected - computed)/denom;
    if (mollified_relative_error > tol)
        ++global_error_count;
    // CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},diff,tol);
  }
}

template<class T>
struct complex_bsl_dft
// Since new and new[] is forbidend, we use boost's Polymorphic Memory Resource
// to allocate internal buffers.
{
  using Complex = boost::multiprecision::complex<T> ;
  using type = bsl_dft< Complex, boost::container::pmr::polymorphic_allocator<Complex> >;
};
template<class T>
struct test_bsl_rdft_traits
// Since new and new[] is forbidend, we use boost's Polymorphic Memory Resource
// to allocate internal buffers.
{
  using value_type = T ;
  using Complex = boost::multiprecision::complex<T>;
  using allocator_type = boost::container::pmr::polymorphic_allocator<value_type>;
  using plan_type = bsl_rdft< value_type, allocator_type >;
};

class Z337
{
public:
  typedef int integer;
  static constexpr integer mod{337};
};

void test_inverse_algebraic_fft()
{
  std::array<char,200> my_buf;
  boost::container::pmr::monotonic_buffer_resource
    my_pool{my_buf.data(),my_buf.size(),boost::container::pmr::null_memory_resource()};

  constexpr int N = 8;
  using M_int = boost::math::fft::my_modulo_lib::mint<Z337>;
  using mint_allocator = boost::container::pmr::polymorphic_allocator<M_int> ;
  const M_int w{85};
  const M_int inv_N{M_int{N}.inverse()};

  std::vector<M_int,mint_allocator> A(N,M_int(),&my_pool);
  std::iota(A.begin(),A.end(),1);

  std::vector<M_int,mint_allocator> FT_A(&my_pool),FT_FT_A(&my_pool);

  boost::math::fft::bsl_algebraic_dft<M_int,mint_allocator> plan(N,w,&my_pool);

  plan.forward(A.cbegin(),A.cend(),std::back_inserter(FT_A));
  plan.backward(FT_A.cbegin(),FT_A.cend(),std::back_inserter(FT_FT_A));

  std::transform(FT_FT_A.begin(), FT_FT_A.end(), FT_FT_A.begin(),
                 [&inv_N](M_int x) { return x * inv_N; });

  int diff = 0;
  for (size_t i = 0; i < A.size(); ++i)
      diff += A[i] == FT_FT_A[i] ? 0 : 1;
  if ( diff != 0 )
      ++global_error_count;
}

int main()
{
  new_is_on=false;
  for(int i=1;i<=100;++i)
  {
    test_inverse<complex_bsl_dft<float>::type>(i,2);
    test_inverse<complex_bsl_dft<double>::type>(i,2);
    test_inverse<complex_bsl_dft<long double>::type>(i,2);
    
    test_inverse_real< test_bsl_rdft_traits<float>>(i,32);
    test_inverse_real< test_bsl_rdft_traits<double>>(i,32);
    test_inverse_real< test_bsl_rdft_traits<long double>>(i,32);
    
    test_inverse_real_complex< test_bsl_rdft_traits<float>>(i,32);
    test_inverse_real_complex< test_bsl_rdft_traits<double>>(i,32);
    test_inverse_real_complex< test_bsl_rdft_traits<long double>>(i,32);
  }
  test_inverse_algebraic_fft();
  return global_error_count;
}
