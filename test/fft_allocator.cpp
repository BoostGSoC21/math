///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <boost/math/fft/bsl_backend.hpp>
#include "fft_test_helpers.hpp"
#include <boost/random.hpp>

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

template<class T>
struct my_allocator
{
    using value_type = T;
    
    my_allocator()
    {
    }
    
    template<class U>
    my_allocator(const my_allocator<U>&)
    {
    }
    
    ~my_allocator()
    {
    }
    T* allocate(size_t n)
    {
      void * p = std::malloc(n*sizeof(T));
      if(p==nullptr)
        throw std::bad_alloc{};
      return reinterpret_cast<T*>(p);
    }
    void deallocate(T* p,size_t /*n*/)
    {
      std::free(p);
    }
    
    template<class U>
    struct rebind { using other = my_allocator<U>;};
};

using namespace boost::math::fft;

template<class Backend>
void test_inverse(int N, int tolerance)
{
  using allocator_type = typename Backend::allocator_type;
  using Complex = typename Backend::value_type;
  using real_value_type = typename Complex::value_type;
  const real_value_type tol = tolerance*std::numeric_limits<real_value_type>::epsilon();
  
  // int *_ = new int[45]; //  std::bad_alloc
  // int *_ = new int; // std::bad_alloc
  
  boost::random::mt19937 rng;
  boost::random::uniform_real_distribution<real_value_type> U(0.0,1.0);
  {
    std::vector<Complex,allocator_type> A(N),B(N),C(N);
    for(auto& x: A)
    {
      x.real( U(rng) );
      x.imag( U(rng) );
    }
    Backend plan(N);
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
        ++::boost::math::test::detail::global_error_count;
    // CHECK_MOLLIFIED_CLOSE(real_value_type{0.0},diff,tol);
  }
}

template<class T>
struct complex_bsl_dft
{
  using Complex = typename detail::select_complex<T>::type ;
  using type = bsl_dft< Complex, my_allocator<Complex> >;
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
  return boost::math::test::report_errors();
}
