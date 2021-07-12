///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

//  What's in this header: 
//  A simple transform-like FFT interface, powered by a our boost backend.

#ifndef BOOST_MATH_FFT_HPP
  #define BOOST_MATH_FFT_HPP

  #include <algorithm>
  #include <iterator>
  #include <vector>
  #include <boost/math/fft/bsl_backend.hpp>

  namespace boost { namespace math { namespace fft { 
  
  // std::transform-like Fourier Transform API
  // for complex types
  template<typename InputIterator,
           typename OutputIterator>
  void dft_forward(InputIterator  input_begin,
                   InputIterator  input_end,
                   OutputIterator output)
  {
    using input_value_type  = typename std::iterator_traits<InputIterator >::value_type;
    bsl_dft<input_value_type> plan(static_cast<unsigned int>(std::distance(input_begin, input_end)));
    plan.forward(input_begin, input_end, output);
  }

  // std::transform-like Fourier Transform API
  // for complex types
  template<typename InputIterator,
           typename OutputIterator>
  void dft_backward(InputIterator  input_begin,
                    InputIterator  input_end,
                    OutputIterator output)
  {
    using input_value_type  = typename std::iterator_traits<InputIterator >::value_type;
    bsl_dft<input_value_type> plan(static_cast<unsigned int>(std::distance(input_begin, input_end)));
    plan.backward(input_begin, input_end, output);
  }
  
  // std::transform-like Fourier Transform API
  // for Ring types
  template<typename InputIterator,
           typename OutputIterator,
           typename value_type>
  void dft_forward(InputIterator  input_begin,
                   InputIterator  input_end,
                   OutputIterator output,
                   value_type w)
  {
    using input_value_type  = typename std::iterator_traits<InputIterator >::value_type;
    bsl_dft<input_value_type> plan(static_cast<unsigned int>(std::distance(input_begin, input_end)),w);
    plan.forward(input_begin, input_end, output);
  }

  // std::transform-like Fourier Transform API
  // for Ring types
  template<typename InputIterator,
           typename OutputIterator,
           typename value_type>
  void dft_backward(InputIterator  input_begin,
                    InputIterator  input_end,
                    OutputIterator output,
                    value_type w)
  {
    using input_value_type  = typename std::iterator_traits<InputIterator >::value_type;
    bsl_dft<input_value_type> plan(static_cast<unsigned int>(std::distance(input_begin, input_end)),w);
    plan.backward(input_begin, input_end, output);
  }
  
  template<typename InputIterator1,
           typename InputIterator2,
           typename OutputIterator>
  void convolution(InputIterator1 input1_begin,
                   InputIterator1 input1_end,
                   InputIterator2 input2_begin,
                   OutputIterator output)
  {
    using input_value_type  = typename std::iterator_traits<InputIterator1>::value_type;
    using allocator_type    = std::allocator<input_value_type>;
    detail::raw_convolution(input1_begin,input1_end,input2_begin,output,allocator_type{});
  }
  
  
  } } } // namespace boost::math::fft

#endif // BOOST_MATH_FFT_HPP
