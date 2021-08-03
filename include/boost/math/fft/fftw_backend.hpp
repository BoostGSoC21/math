///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_FFT_FFTWBACKEND_HPP
  #define BOOST_MATH_FFT_FFTWBACKEND_HPP
  
  #include <memory>

  #include <fftw3.h>
  #include <boost/math/fft/dft_api.hpp>
  #include <boost/math/fft/abstract_ring.hpp>

  namespace boost { namespace math {  namespace fft {

  namespace detail {
  
  template<typename T>
  struct fftw_traits_c_interface;

  template<>
  struct fftw_traits_c_interface<float>
  {
    using plan_type = fftwf_plan;

    using real_value_type = float;

    using complex_value_type = real_value_type[2U];

    static plan_type plan_construct(
      int n, complex_value_type* in, complex_value_type* out, int sign, unsigned int flags) 
    { 
      return ::fftwf_plan_dft_1d(n, in, out, sign, flags); 
    }
    static plan_type plan_construct_r2c(
      int n, real_value_type* in, complex_value_type* out, unsigned int flags) 
    { 
      return ::fftwf_plan_dft_r2c_1d(n, in, out, flags); 
    }
    static plan_type plan_construct_c2r(
      int n, complex_value_type* in, real_value_type* out, unsigned int flags) 
    { 
      return ::fftwf_plan_dft_c2r_1d(n, in, out, flags); 
    }
    
    static void plan_execute(
      plan_type plan, complex_value_type* in, complex_value_type* out) 
    { 
      ::fftwf_execute_dft(plan, in, out); 
    }
    static void plan_execute_r2c(
      plan_type plan, real_value_type* in, complex_value_type* out) 
    { 
      ::fftwf_execute_dft_r2c(plan, in, out); 
    }
    static void plan_execute_c2r(
      plan_type plan, complex_value_type* in, real_value_type* out) 
    { 
      ::fftwf_execute_dft_c2r(plan, in, out); 
    }

    static void plan_destroy(plan_type p) { ::fftwf_destroy_plan(p); }
    
    static int alignment_of(real_value_type* p) { return ::fftwf_alignment_of(p); }
  };

  template<>
  struct fftw_traits_c_interface<double>
  {
    using plan_type = fftw_plan;

    using real_value_type = double;

    using complex_value_type = real_value_type[2U];

    static plan_type plan_construct(
      int n, complex_value_type* in, complex_value_type* out, int sign, unsigned int flags) 
    { 
      return ::fftw_plan_dft_1d(n, in, out, sign, flags); 
    }
    static plan_type plan_construct_r2c(
      int n, real_value_type* in, complex_value_type* out, unsigned int flags) 
    { 
      return ::fftw_plan_dft_r2c_1d(n, in, out, flags); 
    }
    static plan_type plan_construct_c2r(
      int n, complex_value_type* in, real_value_type* out, unsigned int flags) 
    { 
      return ::fftw_plan_dft_c2r_1d(n, in, out, flags); 
    }

    static void plan_execute(
      plan_type plan, complex_value_type* in, complex_value_type* out) 
    { 
      ::fftw_execute_dft(plan, in, out); 
    }
    static void plan_execute_r2c(
      plan_type plan, real_value_type* in, complex_value_type* out) 
    { 
      ::fftw_execute_dft_r2c(plan, in, out); 
    }
    static void plan_execute_c2r(
      plan_type plan, complex_value_type* in, real_value_type* out) 
    { 
      ::fftw_execute_dft_c2r(plan, in, out); 
    }

    static void plan_destroy(plan_type p) { ::fftw_destroy_plan(p); }
    
    static int alignment_of(real_value_type* p) { return ::fftw_alignment_of(p); }
  };

  template<>
  struct fftw_traits_c_interface<long double>
  {
    using plan_type = fftwl_plan;

    using real_value_type = long double;

    using complex_value_type = real_value_type[2U];

    static plan_type plan_construct(
      int n, complex_value_type* in, complex_value_type* out, int sign, unsigned int flags) 
    { 
      return ::fftwl_plan_dft_1d(n, in, out, sign, flags); 
    }
    static plan_type plan_construct_r2c(
      int n, real_value_type* in, complex_value_type* out, unsigned int flags) 
    { 
      return ::fftwl_plan_dft_r2c_1d(n, in, out, flags); 
    }
    static plan_type plan_construct_c2r(
      int n, complex_value_type* in, real_value_type* out, unsigned int flags) 
    { 
      return ::fftwl_plan_dft_c2r_1d(n, in, out, flags); 
    }

    static void plan_execute(
      plan_type plan, complex_value_type* in, complex_value_type* out) 
    { 
      ::fftwl_execute_dft(plan, in, out); 
    }
    static void plan_execute_r2c(
      plan_type plan, real_value_type* in, complex_value_type* out) 
    { 
      ::fftwl_execute_dft_r2c(plan, in, out); 
    }
    static void plan_execute_c2r(
      plan_type plan, complex_value_type* in, real_value_type* out) 
    { 
      ::fftwl_execute_dft_c2r(plan, in, out); 
    }

    static void plan_destroy(plan_type p) { ::fftwl_destroy_plan(p); }
    
    static int alignment_of(real_value_type* p) { return ::fftwl_alignment_of(p); }
  };
  #ifdef BOOST_MATH_USE_FLOAT128
  template<>
  struct fftw_traits_c_interface<boost::multiprecision::float128>
  {
    using plan_type = fftwq_plan;

    // Type casting for fftw:
    using real_value_type = boost::float128_t;

    using complex_value_type = boost::multiprecision::complex128;

    static plan_type plan_construct(
      int n, complex_value_type* in, complex_value_type* out, int sign, unsigned int flags)
    {
      return ::fftwq_plan_dft_1d(n, (real_value_type(*)[2])in, (real_value_type(*)[2])out, sign, flags);
    }
    static plan_type plan_construct_r2c(
      int n, real_value_type* in, complex_value_type* out, unsigned int flags) 
    { 
      return ::fftwq_plan_dft_r2c_1d(n, in, (real_value_type(*)[2])out, flags); 
    }
    static plan_type plan_construct_c2r(
      int n, complex_value_type* in, real_value_type* out, unsigned int flags) 
    { 
      return ::fftwq_plan_dft_c2r_1d(n, (real_value_type(*)[2])in, out, flags); 
    }

    static void plan_execute(
      plan_type plan, complex_value_type* in, complex_value_type* out)
    {
      ::fftwq_execute_dft(plan, (real_value_type(*)[2])in, (real_value_type(*)[2])out);
    }
    static void plan_execute_r2c(
      plan_type plan, real_value_type* in, complex_value_type* out) 
    { 
      ::fftwq_execute_dft_r2c(plan, in, (real_value_type(*)[2]) out); 
    }
    static void plan_execute_c2r(
      plan_type plan, complex_value_type* in, real_value_type* out) 
    { 
      ::fftwq_execute_dft_c2r(plan,(real_value_type(*)[2]) in, out); 
    }

    static void plan_destroy(plan_type p) { ::fftwq_destroy_plan(p); }
    
    static int alignment_of(real_value_type* p) { return ::fftwq_alignment_of(p); }
  };
  #endif


  template<class NativeComplexType, class Allocator_t >
  class fftw_backend
  {
  public:
    using value_type     = NativeComplexType;
    using allocator_type = Allocator_t;
  
  private:
    using real_value_type    = typename NativeComplexType::value_type;
    using plan_type          = typename detail::fftw_traits_c_interface<real_value_type>::plan_type;
    using complex_value_type = typename detail::select_complex<real_value_type>::type;
    using fftw_real_value_type = typename detail::fftw_traits_c_interface<real_value_type>::real_value_type;
   
    void execute(plan_type plan, plan_type unaligned_plan, const complex_value_type* in, complex_value_type* out) const
    {
      using local_complex_type = typename detail::fftw_traits_c_interface<real_value_type>::complex_value_type;
      
      if(in!=out) // We have to copy, because fftw plan is forced to be in-place: from: nullptr, to: nullptr
        std::copy(in,in+size(),out);
      
      const int out_alignment = detail::fftw_traits_c_interface<real_value_type>::alignment_of(
                reinterpret_cast<fftw_real_value_type*>(out));
                
      if(out_alignment==ref_alignment)
        detail::fftw_traits_c_interface<real_value_type>::plan_execute
        (
          plan,
          reinterpret_cast<local_complex_type*>(out),
          reinterpret_cast<local_complex_type*>(out)
        );
      else
        detail::fftw_traits_c_interface<real_value_type>::plan_execute
        (
          unaligned_plan,
          reinterpret_cast<local_complex_type*>(out),
          reinterpret_cast<local_complex_type*>(out)
        );
    }
    void execute(plan_type plan, plan_type unaligned_plan, const real_value_type* in, complex_value_type* out) const
    {
      using local_complex_type = typename detail::fftw_traits_c_interface<real_value_type>::complex_value_type;
      
      std::memcpy(reinterpret_cast<void*>(out),reinterpret_cast<const void*>(in),sizeof(local_complex_type)*(1+size()/2));
      
      const int out_alignment = detail::fftw_traits_c_interface<real_value_type>::alignment_of(
                reinterpret_cast<fftw_real_value_type*>(out));
                
      if(out_alignment==ref_alignment)
        detail::fftw_traits_c_interface<real_value_type>::plan_execute_r2c
        (
          plan,
          reinterpret_cast<fftw_real_value_type*>(out),
          reinterpret_cast<local_complex_type*>(out)
        );
      else
        detail::fftw_traits_c_interface<real_value_type>::plan_execute_r2c
        (
          unaligned_plan,
          reinterpret_cast<fftw_real_value_type*>(out),
          reinterpret_cast<local_complex_type*>(out)
        );
    }
    
    void free()
    {
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_forward_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_backward_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_forward_unaligned_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_backward_unaligned_plan);
      
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_r2c_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_c2r_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_r2c_unaligned_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_c2r_unaligned_plan);
    }
    void alloc()
    {
      my_forward_plan = 
        detail::fftw_traits_c_interface<real_value_type>::plan_construct
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_FORWARD,  
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT
        );
      my_backward_plan =
        detail::fftw_traits_c_interface<real_value_type>::plan_construct
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_BACKWARD, 
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT
        );
      my_forward_unaligned_plan = 
        detail::fftw_traits_c_interface<real_value_type>::plan_construct
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_FORWARD,  
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT | FFTW_UNALIGNED
        );
      my_backward_unaligned_plan =
        detail::fftw_traits_c_interface<real_value_type>::plan_construct
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_BACKWARD, 
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT | FFTW_UNALIGNED
        );
      
      my_r2c_plan = 
        detail::fftw_traits_c_interface<real_value_type>::plan_construct_r2c
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT
        );
      my_c2r_plan =
        detail::fftw_traits_c_interface<real_value_type>::plan_construct_c2r
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT
        );
      my_r2c_unaligned_plan = 
        detail::fftw_traits_c_interface<real_value_type>::plan_construct_r2c
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT | FFTW_UNALIGNED
        );
      my_c2r_unaligned_plan =
        detail::fftw_traits_c_interface<real_value_type>::plan_construct_c2r
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT | FFTW_UNALIGNED
        );
    }

  public:
    fftw_backend(std::size_t n, const allocator_type& = allocator_type{} )
      : my_size{ n },
        ref_alignment{
            detail::fftw_traits_c_interface<real_value_type>::alignment_of(nullptr)}
    {
      // For C++11, this line needs to be constexpr-ified.
      // Then we could restore the constexpr-ness of this constructor.
      alloc();
    }

    ~fftw_backend()
    {
      free();
    }
    
    void resize(std::size_t new_size)
    {
      if(size()!=new_size)
      {
        free();
        my_size = new_size;
        alloc();
      }
    }

    constexpr std::size_t size() const { return my_size; }
    
    void forward(const complex_value_type* in, complex_value_type* out) const
    {
      execute(my_forward_plan, my_forward_unaligned_plan, in, out);  
    }

    void backward(const complex_value_type* in, complex_value_type* out) const
    {
      execute(my_backward_plan, my_backward_unaligned_plan, in, out);  
    }
    void r2c(const real_value_type* in, complex_value_type* out) const
    {
      execute(my_r2c_plan, my_r2c_unaligned_plan, in, out);  
    }

  private:
    std::size_t my_size;
    const int   ref_alignment;
    
    plan_type   my_forward_plan;
    plan_type   my_backward_plan;
    plan_type   my_forward_unaligned_plan;
    plan_type   my_backward_unaligned_plan;
    
    plan_type   my_r2c_plan;
    plan_type   my_c2r_plan;
    plan_type   my_r2c_unaligned_plan;
    plan_type   my_c2r_unaligned_plan;
  };
  
  template<class T, class Allocator_t >
  class fftw_rfft_backend
  {
  public:
    // using value_type     = T;
    using allocator_type = Allocator_t;
  
  private:
    using real_value_type    = T;
    using plan_type          = typename detail::fftw_traits_c_interface<real_value_type>::plan_type;
    using fftw_real_value_type     = typename detail::fftw_traits_c_interface<real_value_type>::real_value_type;
    using fftw_complex_value_type  = typename detail::fftw_traits_c_interface<real_value_type>::complex_value_type;
      
    template<class U>
    using vector_t = std::vector<U, typename std::allocator_traits<allocator_type>::template rebind_alloc<U> >;
   
    void execute_r2c(plan_type plan, plan_type unaligned_plan, const real_value_type* in, real_value_type* out) const
    // precondition:
    // size(in)  >= size()
    // size(out) >= unique_complex_size()*2
    {
      const int out_alignment = detail::fftw_traits_c_interface<real_value_type>::alignment_of(
                reinterpret_cast<fftw_real_value_type*>(out));
                
      if(in!=out) // We have to copy, because fftw plan is forced to be in-place: from: nullptr, to: nullptr
        std::copy(in,in+size(),out);
                
      if(out_alignment==ref_alignment)
        detail::fftw_traits_c_interface<real_value_type>::plan_execute_r2c
        (
          plan,
          reinterpret_cast<fftw_real_value_type*>(out),
          reinterpret_cast<fftw_complex_value_type*>(out)
        );
      else
        detail::fftw_traits_c_interface<real_value_type>::plan_execute_r2c
        (
          unaligned_plan,
          reinterpret_cast<fftw_real_value_type*>(out),
          reinterpret_cast<fftw_complex_value_type*>(out)
        );
    }
    void execute_c2r(plan_type plan, plan_type unaligned_plan, const real_value_type* in, real_value_type* out) const
    // precondition:
    // size(in)  >= unique_complex_size()*2
    // size(out) >= unique_complex_size()*2
    {
      const int out_alignment = detail::fftw_traits_c_interface<real_value_type>::alignment_of(
                reinterpret_cast<fftw_real_value_type*>(out));
                
      if(in!=out) // We have to copy, because fftw plan is forced to be in-place: from: nullptr, to: nullptr
        std::copy(in,in+unique_complex_size()*2,out);
                
      if(out_alignment==ref_alignment)
        detail::fftw_traits_c_interface<real_value_type>::plan_execute_c2r
        (
          plan,
          reinterpret_cast<fftw_complex_value_type*>(out),
          reinterpret_cast<fftw_real_value_type*>(out)
        );
      else
        detail::fftw_traits_c_interface<real_value_type>::plan_execute_c2r
        (
          unaligned_plan,
          reinterpret_cast<fftw_complex_value_type*>(out),
          reinterpret_cast<fftw_real_value_type*>(out)
        );
    }
    
    void free()
    {
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_r2c_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_c2r_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_r2c_unaligned_plan);
      detail::fftw_traits_c_interface<real_value_type>::plan_destroy(my_c2r_unaligned_plan);
    }
    void alloc()
    {
      my_r2c_plan = 
        detail::fftw_traits_c_interface<real_value_type>::plan_construct_r2c
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_ESTIMATE
        );
      my_c2r_plan =
        detail::fftw_traits_c_interface<real_value_type>::plan_construct_c2r
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_ESTIMATE
        );
      my_r2c_unaligned_plan = 
        detail::fftw_traits_c_interface<real_value_type>::plan_construct_r2c
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_ESTIMATE | FFTW_UNALIGNED
        );
      my_c2r_unaligned_plan =
        detail::fftw_traits_c_interface<real_value_type>::plan_construct_c2r
        (
          size(), 
          nullptr, 
          nullptr, 
          FFTW_ESTIMATE | FFTW_UNALIGNED
        );
    }

  public:
    fftw_rfft_backend(std::size_t n, const allocator_type& A = allocator_type{} )
      : my_size{ n },
        ref_alignment{
            detail::fftw_traits_c_interface<real_value_type>::alignment_of(nullptr)},
        my_allocator{A}
    {
      // For C++11, this line needs to be constexpr-ified.
      // Then we could restore the constexpr-ness of this constructor.
      alloc();
    }

    ~fftw_rfft_backend()
    {
      free();
    }
    
    void resize(std::size_t new_size)
    {
      if(size()!=new_size)
      {
        free();
        my_size = new_size;
        alloc();
      }
    }

    constexpr std::size_t size() const { return my_size; }
    constexpr std::size_t unique_complex_size() const {return my_size/2 + 1;}
     
    void pack_halfcomplex(const real_value_type* in, real_value_type* out) const
    {
      const std::size_t N = size();
      out[0]=in[0];
      for(unsigned int i=1;i<N;++i)
      {
        out[i] = in[i+1];
      }
    }
    void unpack_halfcomplex(const real_value_type* in, real_value_type* out) const
    {
      const std::size_t N = size();
      const std::size_t M = unique_complex_size();
      out[1]=0;
      out[2*M-1]=0;
      
      out[0]=in[0];
      for(unsigned int i=1;i<N;++i)
      {
        out[i+1] = in[i];
      }
    }
    void real_to_halfcomplex(const real_value_type* in, real_value_type* out) const
    {
      const std::size_t N = size();
      const std::size_t M = unique_complex_size();
      vector_t<real_value_type> tmp(2*M,my_allocator);
      std::copy(in,in+N,tmp.begin());
      execute_r2c(my_r2c_plan,my_r2c_unaligned_plan,tmp.data(),tmp.data());
      pack_halfcomplex(tmp.data(),out);
    }
    void halfcomplex_to_real(const real_value_type* in, real_value_type* out) const
    {
      const std::size_t N = size();
      const std::size_t M = unique_complex_size();
      vector_t<real_value_type> tmp(2*M,my_allocator);
      unpack_halfcomplex(in,tmp.data()); 
      execute_c2r(my_c2r_plan,my_c2r_unaligned_plan,tmp.data(),tmp.data());
      std::copy(tmp.begin(),tmp.begin()+N,out);
    }
    template<class Complex>
    void real_to_complex(const real_value_type* in, Complex *out)const
    {
      const std::size_t N = size();
      const std::size_t M = unique_complex_size();
      vector_t<real_value_type> tmp(2*M,my_allocator);
      std::copy(in,in+N,tmp.begin());
      execute_r2c(my_r2c_plan,my_r2c_unaligned_plan,tmp.data(),tmp.data());
      for(unsigned int i=0,j=0;i<M;++i,j+=2)
      {
        out[i].real(tmp[j]);
        out[i].imag(tmp[j+1]);
      }
    }
    template<class Complex>
    void complex_to_real(const Complex* in, real_value_type* out) const
    {
      const std::size_t N = size();
      const std::size_t M = unique_complex_size();
      vector_t<real_value_type> tmp(2*M,my_allocator);
      for(unsigned int i=0,j=0;i<M;++i,j+=2)
      {
        tmp[j]=out[i].real();
        tmp[j]=out[i].imag();
      }
      execute_c2r(my_c2r_plan,my_c2r_unaligned_plan,tmp.data(),tmp.data());
      std::copy(tmp.begin(),tmp.begin()+N,out);
    }

  private:
    std::size_t my_size;
    const int   ref_alignment;
    allocator_type my_allocator;
    
    plan_type   my_r2c_plan;
    plan_type   my_c2r_plan;
    plan_type   my_r2c_unaligned_plan;
    plan_type   my_c2r_unaligned_plan;
  };

  } // namespace detail
  
  template<class RingType = std::complex<double>, class Allocator_t = std::allocator<RingType> >
  using fftw_dft = detail::complex_dft<detail::fftw_backend,RingType,Allocator_t>;
  
  template<class T = double, class Allocator_t = std::allocator<T> >
  using fftw_rfft = detail::real_dft<detail::fftw_rfft_backend,T,Allocator_t>;
  
  using fftw_transform = transform< fftw_dft<> >;
  
  } } } // namespace boost::math::fft

#endif // BOOST_MATH_FFT_FFTWBACKEND_HPP
