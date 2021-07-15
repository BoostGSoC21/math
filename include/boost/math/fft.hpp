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
