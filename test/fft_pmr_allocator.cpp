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

bool new_is_on{true};

#if __cplusplus >= 201700L
#include <memory>
#include <memory_resource>

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

template<class Backend>
void test_inverse(int N, int tolerance)
{
#if __cplusplus >= 201700
  std::array<char,200000> buf;
  std::pmr::monotonic_buffer_resource
    pool{buf.data(),buf.size(),std::pmr::null_memory_resource()};
#endif
  
  using allocator_type = typename Backend::allocator_type;
  using Complex = typename Backend::value_type;
  using real_value_type = typename Complex::value_type;
  const real_value_type tol = tolerance*std::numeric_limits<real_value_type>::epsilon();
  {
#if __cplusplus >= 201700
    std::vector<Complex,allocator_type> A(N,Complex(),&pool);
    std::vector<Complex,allocator_type> B(N,Complex(),&pool);
    std::vector<Complex,allocator_type> C(N,Complex(),&pool);
    Backend plan(N,&pool);
#else
    std::vector<Complex,allocator_type> A(N,Complex());
    std::vector<Complex,allocator_type> B(N,Complex());
    std::vector<Complex,allocator_type> C(N,Complex());
    Backend plan(N);
#endif
    
    for(auto& x: A)
    {
      x.real( 1. );
      x.imag( 2. );
    }
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

template<class T>
struct complex_bsl_dft
{
  using Complex = boost::multiprecision::complex<T> ;
#if __cplusplus >= 201700
  using type = bsl_dft< Complex, boost::container::pmr::polymorphic_allocator<Complex> >;
#else
  using type = bsl_dft< Complex, std::allocator<Complex> >;
#endif
};
int main()
{
  new_is_on=false;
  for(int i=1;i<=100;++i)
  {
    test_inverse<complex_bsl_dft<float>::type>(i,2);
    test_inverse<complex_bsl_dft<double>::type>(i,2);
    test_inverse<complex_bsl_dft<long double>::type>(i,2);
  }
  return global_error_count;
}
