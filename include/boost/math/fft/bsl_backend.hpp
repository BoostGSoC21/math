///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_FFT_BSLBACKEND_HPP
  #define BOOST_MATH_FFT_BSLBACKEND_HPP

  #include <algorithm>
  #include <cmath>
  #include <type_traits>
  #include <boost/math/fft/multiprecision_complex.hpp>

  #include <boost/math/fft/algorithms.hpp>
  #include <boost/math/fft/real_algorithms.hpp>
  #include <boost/math/fft/dft_api.hpp>

  namespace boost { namespace math {  namespace fft { 
  
  namespace detail {


  /*
    Boost DFT backend:
    It handles RingTypes and it calls the appropriate specialized functions if
    the type is complex.
    
    A type is considered "complex" if is_boost_complex::value == true. A user-defined
    type can become complex by specializing that trait.
    
    We have specialized algorithms for complex numbers and general purpose DFT
    that need the specification of a root of unity.
    The general purpose DFT work with complex and non-complex types, but its
    performance and precision could be lower than the specialized complex
    versions.
    
    This interface selects general purpose DFT for non-complex types,
    and for complex types the default behaviour is to use the specialized
    complex algorithms, unless the user provides a root of unity 'W', in which case
    the interface will execute general purpose DFT using W.
  */
  template<class RingType, class allocator_t = std::allocator<RingType> >
  class bsl_backend
  {
  public:
    using value_type     = RingType;
    using allocator_type = allocator_t;
    
  private:
    enum plan_type { forward_plan , backward_plan};
    
    template<typename U = RingType>
    typename std::enable_if< boost::multiprecision::is_boost_complex<U>::value==true  >::type
    execute(plan_type plan, const RingType * in, RingType* out)const
    {
      const long N = static_cast<long>(size());
      const int sign = (plan == forward_plan ? 1 : -1);
      
      // select the implementation according to the DFT size
      switch(N)
      {
        case 0:
          return;
        case 1:
          out[0]=in[0];
          return;
        case 2:
          detail::complex_dft_2(in,out,sign);
          return;
      }
      
      if( detail::is_power2(N) )
      {
        detail::complex_dft_power2(in,in+N,out,sign);
      }
      else if(detail::is_prime(N))
      {
        // detail::complex_dft_prime_bruteForce(in,in+N,out,sign);
        detail::complex_dft_prime_rader(in,in+N,out,sign,alloc);
      }
      else
      {
        detail::complex_dft_composite(in,in+N,out,sign,alloc);
      }
    }
    
  public:
    
    // the provided root of unity is used instead of exp(-i 2 pi/n)
    constexpr bsl_backend(std::size_t n, const allocator_type& in_alloc = allocator_type{}):
        alloc{in_alloc},
        my_size{n}
    { 
    }

    ~bsl_backend()
    {
    }
    
    void resize(std::size_t new_size)
    {
      my_size = new_size;
    }
    RingType inverse_root(RingType root) const
    {
      return detail::power(root,size()-1);
    }
    
    constexpr std::size_t size() const { return my_size; }

    void forward(const RingType* in, RingType* out) const
    {
      execute(forward_plan,in,out);   
    }

    void backward(const RingType* in, RingType* out) const
    {
      execute(backward_plan,in,out);   
    }
    void dft(const RingType* in, RingType* out, RingType w) const
    {
      const long N = static_cast<long>(size());
      // select the implementation according to the DFT size
      if( detail::is_power2(N))
      {
        detail::dft_power2(in,in+N,out,w);
      }
      else
      {
        detail::dft_composite(in,in+N,out,w,alloc);
      }
    }

  private:
    allocator_type alloc;
    std::size_t my_size{};
  };
  
  
  template<class T, class allocator_t = std::allocator<T> >
  class bsl_rfft_backend
  {
  public:
    using value_type     = T;
    using allocator_type = allocator_t;
    
    // the provided root of unity is used instead of exp(-i 2 pi/n)
    constexpr bsl_rfft_backend(std::size_t n, const allocator_type& in_alloc = allocator_type{}):
        alloc{in_alloc},
        my_size{n}
    { 
    }

    ~bsl_rfft_backend()
    {
    }
    
    void resize(std::size_t new_size)
    {
      my_size = new_size;
    }
    
    constexpr std::size_t size() const { return my_size; }
    constexpr std::size_t unique_complex_size() const { return my_size/2 + 1;}

    void real_to_halfcomplex(const value_type* in, value_type* out) const
    {
      const long N = static_cast<long>(size());
      // select the implementation according to the DFT size
      switch(N)
      {
        case 0:
          return;
        case 1:
          out[0]=in[0];
          return;
        case 2:
          detail::real_dft_2(in,out,1);
          return;
      }
      if( detail::is_power2(N))
      {
        detail::real_dft_power2(in,in+N,out,1);
      }else
      {
        detail::real_dft_prime_bruteForce(in,in+N,out,1,alloc);
      }
    }
    void halfcomplex_to_real(const value_type* in, value_type* out) const
    {
      const long N = static_cast<long>(size());
      // select the implementation according to the DFT size
      switch(N)
      {
        case 0:
          return;
        case 1:
          out[0]=in[0];
          return;
        case 2:
          detail::real_inverse_dft_2(in,out,1);
          return;
      }
      if( detail::is_power2(N))
      { 
        detail::real_inverse_dft_power2(in,in+N,out,1);
      }else
      {
        detail::real_inverse_dft_prime_bruteForce(in,in+N,out,1,alloc);
      }
    }

  private:
    allocator_type alloc;
    std::size_t my_size{};
  };
  
  } // namespace detail
  
  template<class RingType = std::complex<double>, class Allocator_t = std::allocator<RingType> >
  using bsl_dft = detail::complex_dft<detail::bsl_backend,RingType,Allocator_t>;
  
  template<class T = double, class Allocator_t = std::allocator<T> >
  using bsl_rfft = detail::real_dft<detail::bsl_rfft_backend,T,Allocator_t>;
  
  template<class RingType = std::complex<double>, class Allocator_t = std::allocator<RingType> >
  using bsl_algebraic_dft = detail::algebraic_dft<detail::bsl_backend,RingType,Allocator_t>;
  
  using bsl_transform = transform< bsl_dft<> >;
  
  using bsl_algebraic_transform = transform< bsl_algebraic_dft<> >;
  
  } } } // namespace boost::math::fft

#endif // BOOST_MATH_FFT_BSLBACKEND_HPP
