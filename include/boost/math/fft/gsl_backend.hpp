///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_FFT_GSLBACKEND_HPP
#define BOOST_MATH_FFT_GSLBACKEND_HPP

#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <complex>
#include <boost/math/fft/dft_api.hpp>

namespace boost { namespace math { 
namespace fft { namespace detail {
    
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
      
      // real fft
      gsl_fft_real_wavetable        *real_wtable;
      gsl_fft_halfcomplex_wavetable *halfcomplex_wtable;
      gsl_fft_real_workspace        *real_wspace;
        
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
      void execute(const real_value_type* in, complex_value_type* out) const
      {
        if(reinterpret_cast<const void*>(in)!=reinterpret_cast<void*>(out))
        {
          // we avoid this extra step for in-place transforms
          // notice that if in==out, the following code has
          // undefined-behavior
          std::memcpy(reinterpret_cast<void*>(out),
                      reinterpret_cast<const void*>(in),
                      sizeof(real_value_type)*size());
        }
        
        gsl_fft_real_transform(
          reinterpret_cast<real_value_type*>(out),
          1, size(), real_wtable, real_wspace);
        
        halfcomplex_to_complex(reinterpret_cast<real_value_type*>(out));
      }
      void halfcomplex_to_complex(real_value_type* data)const
      {
        std::size_t N = size();
        for(std::size_t i=N;i>0;--i)
          data[i] = data[i-1];
        data[1] = 0;
        
        if(N%2==0)
        {
          data[N+1] = 0;
        }
      }
      void free()
      {
        gsl_fft_complex_wavetable_free(wtable);
        gsl_fft_complex_workspace_free(wspace);
        
        gsl_fft_real_wavetable_free(real_wtable);
        gsl_fft_halfcomplex_wavetable_free(halfcomplex_wtable);
        gsl_fft_real_workspace_free(real_wspace);
      }
      void alloc()
      {
        wtable = gsl_fft_complex_wavetable_alloc(size());
        wspace = gsl_fft_complex_workspace_alloc(size());
        
        real_wtable        = gsl_fft_real_wavetable_alloc(size());
        halfcomplex_wtable = gsl_fft_halfcomplex_wavetable_alloc(size());
        real_wspace        = gsl_fft_real_workspace_alloc(size());
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
      void r2c(const real_value_type* in, complex_value_type* out) const
      {
        execute(in, out);  
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
      template<class Complex>
      void real_to_complex(const real_value_type* in, Complex *out)const
      {
        const std::size_t N = size();
        const std::size_t M = unique_complex_size();
        
        vector_t<real_value_type> tmp(N,my_alloc);
        real_to_halfcomplex(in,tmp.data());
        
        out[0].real(tmp.front());
        out[0].imag(0);
        
        for(unsigned int i=1,j=1;i<M && j+1<N;++i,j+=2)
        {
          out[i].real(tmp[j]);
          out[i].imag(tmp[j+1]);
        }
        
        if(N%2==0)
        {
          out[M-1].real(tmp.back());
          out[M-1].imag(0);
        }
      }
      template<class Complex>
      void complex_to_real(const Complex* in, real_value_type* out) const
      {
        const std::size_t N = size();
        const std::size_t M = unique_complex_size();
        
        vector_t<real_value_type> tmp(N,my_alloc);
        
        tmp.front() = in[0].real();
        
        for(unsigned int i=1,j=1;i<M && j+1<N;++i,j+=2)
        {
          tmp[j]   = in[i].real();
          tmp[j+1] = in[i].imag();
        }
        
        if(N%2==0)
        {
          tmp.back() = in[M-1].real();
        }
        halfcomplex_to_real(tmp.data(),out);
      }
    };
} // namespace detail    
  
  template<class RingType = std::complex<double>, class Allocator_t = std::allocator<RingType> >
  using gsl_dft = detail::complex_dft<detail::gsl_backend,RingType,Allocator_t>;
  
  template<class T = double, class Allocator_t = std::allocator<T> >
  using gsl_rfft = detail::real_dft<detail::gsl_rfft_backend,T,Allocator_t>;
  
  using gsl_transform = transform< gsl_dft<> >;
  
    
} // namespace fft
} // namespace math
} // namespace boost

#endif // BOOST_MATH_FFT_GSLBACKEND_HPP


