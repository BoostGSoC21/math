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
#include <complex>
#include <boost/math/fft/dft_api.hpp>

namespace boost { namespace math { 
namespace fft { namespace detail {
    
    template<class T, class A>
    class gsl_backend;
    
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
} // namespace detail    
  
  template<class RingType = std::complex<double>, class Allocator_t = std::allocator<RingType> >
  using gsl_dft = detail::dft<detail::gsl_backend,RingType,Allocator_t>;
  
  using gsl_transform = transform< gsl_dft<> >;
  
    
} // namespace fft
} // namespace math
} // namespace boost

#endif // BOOST_MATH_FFT_GSLBACKEND_HPP


