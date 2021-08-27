///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_FFT_GSLBACKEND_HPP
#define BOOST_MATH_FFT_GSLBACKEND_HPP

#include <complex>

#if defined(__GNUC__)
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#endif

#include <boost/math/fft/dft_api.hpp>

namespace boost { namespace math { 
namespace fft { namespace detail {

    #if defined(__GNUC__)
    template<class T, class A>
    class gsl_backend;

    template<class T, class A>
    class gsl_rfft_backend;

    template<class Allocator_t>
    class gsl_backend< std::complex<double>, Allocator_t >
    {
    public:
      using value_type     = std::complex<double>;
      using allocator_type = Allocator_t;
      
    private:
      using real_value_type    = double;
      using complex_value_type = std::complex<real_value_type>;
      enum plan_type { forward_plan, backward_plan };
      
      std::size_t my_size; 
      allocator_type my_alloc;
      
      // complex fft
      gsl_fft_complex_wavetable *wtable;
      gsl_fft_complex_workspace *wspace;
      
      void execute(plan_type p, const complex_value_type* in, complex_value_type* out) const
      {
        if(in!=out)
        {
          // we avoid this extra step for in-place transforms
          // notice that if in==out, the following code has
          // undefined-behavior
          std::copy(in,in+size(),out);
        }
        
        if(p==forward_plan)
        gsl_fft_complex_forward(
          reinterpret_cast<real_value_type*>(std::addressof(*out)),
          1, my_size, wtable, wspace);
        else
        gsl_fft_complex_backward(
          reinterpret_cast<real_value_type*>(std::addressof(*out)),
          1, my_size, wtable, wspace);
      }
      void free()
      {
        gsl_fft_complex_wavetable_free(wtable);
        gsl_fft_complex_workspace_free(wspace);
      }
      void alloc()
      {
        wtable = gsl_fft_complex_wavetable_alloc(size());
        wspace = gsl_fft_complex_workspace_alloc(size());
      }
   public:
      
      gsl_backend(std::size_t n, const allocator_type& = allocator_type{}):
          my_size{n}
      {
        alloc();
      }
        
      ~gsl_backend()
      {
        free();
      }
      std::size_t size() const {return my_size;}
      
      void resize(std::size_t new_size)
      {
        if(size()!=new_size)
        {
          free();
          my_size = new_size;
          alloc();
        }
      }
        
      void forward(const complex_value_type* in, complex_value_type* out) const
      {
        execute(forward_plan,in,out);
      }
      void backward(const complex_value_type* in, complex_value_type* out) const
      {
        execute(backward_plan,in,out);
      }
    };
    
    template<class Allocator_t>
    class gsl_rfft_backend< double, Allocator_t >
    {
    public:
      // using value_type     = double;
      using allocator_type = Allocator_t;
      
    private:
      enum plan_type { forward_plan, backward_plan };
      using real_value_type    = double;
      
      template<class U>
      using vector_t = std::vector<U, typename std::allocator_traits<allocator_type>::template rebind_alloc<U> >;
      
      std::size_t my_size; 
      allocator_type my_alloc;
      
      gsl_fft_real_wavetable        *real_wtable;
      gsl_fft_halfcomplex_wavetable *halfcomplex_wtable;
      gsl_fft_real_workspace        *real_wspace;
      
      void pack_halfcomplex(real_value_type* out) const
      // precondition:
      // -> size(out) >= N
      {
        const std::size_t N = size();
        vector_t<real_value_type> tmp(out,out+N);
        out[0]=tmp[0];
        for(unsigned int i=1,j=1;j<N;++i,j+=2)
        {
          out[j] = tmp[i];
          if(j+1<N)
            out[j+1] = -tmp[N-i];
        }
      }
      void unpack_halfcomplex(real_value_type* out) const
      // precondition:
      // -> size(out) >= N
      {
        const std::size_t N = size();
        vector_t<real_value_type> tmp(out,out+N);
        out[0]=tmp[0];
        for(unsigned int i=1,j=1;j<N;++i,j+=2)
        {
          out[i] = tmp[j];
          if(j+1<N)
            out[N-i] = -tmp[j+1];
        }
      }
      template<plan_type p>
      void execute(const real_value_type* in, 
                   real_value_type* out,
                   const typename std::enable_if<p==forward_plan>::type* = nullptr) const
      {
        const std::size_t N = size();
        if(in!=out)
        {
          // we avoid this extra step for in-place transforms
          // notice that if in==out, the following code has
          // undefined-behavior
          std::copy(in,in+N,out);
        }
        gsl_fft_real_transform(
          out,1, N, real_wtable, real_wspace);
        unpack_halfcomplex(out);
      }
      template<plan_type p>
      void execute(const real_value_type* in, 
                   real_value_type* out,
                   const typename std::enable_if<p==backward_plan>::type* = nullptr) const
      {
        const std::size_t N = size();
        if(in!=out)
        {
          // we avoid this extra step for in-place transforms
          // notice that if in==out, the following code has
          // undefined-behavior
          std::copy(in,in+N,out);
        }
        pack_halfcomplex(out);
        gsl_fft_halfcomplex_transform(
          out,1, N, halfcomplex_wtable, real_wspace);
      }
      
      void free()
      {
        gsl_fft_real_wavetable_free(real_wtable);
        gsl_fft_halfcomplex_wavetable_free(halfcomplex_wtable);
        gsl_fft_real_workspace_free(real_wspace);
      }
      void alloc()
      {
        const std::size_t N = size();
        real_wtable        = gsl_fft_real_wavetable_alloc(N);
        halfcomplex_wtable = gsl_fft_halfcomplex_wavetable_alloc(N);
        real_wspace        = gsl_fft_real_workspace_alloc(N);
      }
   public:
      
      gsl_rfft_backend(std::size_t n, const allocator_type& A= allocator_type{}):
          my_size{n},
          my_alloc{A}
      {
        alloc();
      }
        
      ~gsl_rfft_backend()
      {
        free();
      }
      constexpr std::size_t size() const {return my_size;}
      constexpr std::size_t unique_complex_size() const {return my_size/2 + 1;}
      
      void resize(std::size_t new_size)
      {
        if(size()!=new_size)
        {
          free();
          my_size = new_size;
          alloc();
        }
      }
        
      void real_to_halfcomplex(const real_value_type* in, real_value_type* out) const
      {
        execute<forward_plan>(in,out);
      }
      void halfcomplex_to_real(const real_value_type* in, real_value_type* out) const
      {
        execute<backward_plan>(in,out);
      }
    };
    #endif
  } // namespace detail    

  #if defined(__GNUC__)
  template<class RingType = std::complex<double>, class Allocator_t = std::allocator<RingType> >
  using gsl_dft = detail::complex_dft<detail::gsl_backend,RingType,Allocator_t>;

  template<class T = double, class Allocator_t = std::allocator<T> >
  using gsl_rdft = detail::real_dft<detail::gsl_rfft_backend,T,Allocator_t>;

  using gsl_transform = transform< gsl_dft<> >;
  using gsl_real_transform = transform< gsl_rdft<> >;
  #endif

} // namespace fft
} // namespace math
} // namespace boost

#endif // BOOST_MATH_FFT_GSLBACKEND_HPP


