///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file provides two traits:
//  boost::multiprecision::complex<Real>                 - creates a coplex type supported by Boost
//  boost::multiprecision::is_boost_complex<T>::value    - a bool telling whether the type is a complex number supported by Boost
//                                                         it has nothing to do with boost::is_complex<T> which recognizes only std::complex<T>
//
// TODO: this file should be split apart and each specializetion should go to the end of the relevant file in Boost.Multiprecision
//   --> specialization for float128      --> complex128    goes to <boost/multiprecision/complex128.hpp>
//   --> specialization for cpp_bin_float --> cpp_complex   goes to <boost/multiprecision/cpp_complex.hpp>
//   --> specialization for mpfr_float    --> mpc_complex   goes to <boost/multiprecision/mpc.hpp>

#ifndef BOOST_MATH_FFT_MULTIPRECISION_COMPLEX
  #define BOOST_MATH_FFT_MULTIPRECISION_COMPLEX
  #include <boost/config.hpp>
  #include <boost/cstdfloat.hpp>
  #if defined(BOOST_MATH_USE_FLOAT128)
  #include <boost/multiprecision/float128.hpp>
  #endif
  #include <boost/multiprecision/cpp_bin_float.hpp>
  #include <boost/multiprecision/cpp_dec_float.hpp>
  #if defined(__GNUC__)
  //#include <boost/multiprecision/mpfr.hpp>
  #endif
  #include <boost/core/demangle.hpp>
  #include <boost/core/enable_if.hpp>
  // now include complex types
  #if defined(BOOST_MATH_USE_FLOAT128)
  #include <boost/multiprecision/complex128.hpp>
  #endif
  #include <boost/multiprecision/cpp_complex.hpp>
  #if defined(__GNUC__)
  //#include <boost/multiprecision/mpc.hpp>
  #endif

  namespace boost { namespace multiprecision { namespace detail {

    template<typename T>
    struct make_boost_complex {
      using type = std::complex<T>;
    };

    // float128 --> <boost/multiprecision/complex128.hpp>
    #if defined(BOOST_MATH_USE_FLOAT128)
    template<>
    struct make_boost_complex<boost::multiprecision::float128> {
      using type = boost::multiprecision::complex128;
    };
    #endif

    // cpp_bin_float --> <boost/multiprecision/cpp_complex.hpp>
    template <unsigned Digits, backends::digit_base_type DigitBase, class Allocator, class Exponent, Exponent MinExponent, Exponent MaxExponent, expression_template_option ExpressionTemplates>
    struct make_boost_complex< number< cpp_bin_float<Digits, DigitBase, Allocator, Exponent, MinExponent, MaxExponent>, ExpressionTemplates> > {
      using type = number<complex_adaptor<cpp_bin_float<Digits, DigitBase, Allocator, Exponent, MinExponent, MaxExponent>>, ExpressionTemplates>;
    };

    #if defined(__GNUC__)
    // mpc_complex --> <boost/multiprecision/mpc.hpp>
    //template <unsigned Digits, mpfr_allocation_type AllocationType, expression_template_option ExpressionTemplates>
    //struct make_boost_complex< number< backends::mpfr_float_backend<Digits, AllocationType>, ExpressionTemplates> > {
    //  using type = number<mpc_complex_backend<Digits>, ExpressionTemplates>;
    //};
    #endif

    #if __cplusplus < 201700L
    template<typename... Ts> struct make_void { typedef void type;};
    template<typename... Ts> using void_t = typename make_void<Ts...>::type;
    #else
    using std::void_t;
    #endif

    template< class, class = void > struct has_value_type : std::false_type { };
    template< class T > struct has_value_type<T, void_t<typename T::value_type>> : std::true_type { };

    template <typename T, bool = has_value_type<T>::value > struct is_boost_complex {
      static constexpr bool value = false;
    };

    template <typename T> struct is_boost_complex<T, true> {
      static constexpr bool value = std::is_same<
                                      typename make_boost_complex<typename T::value_type>::type
                                    , typename std::decay<T>::type
                                    >::value;
    };

  }  // boost::multiprecision::detail
  template<typename T> using complex          = typename detail::make_boost_complex<T>::type;
  template<typename T> using is_boost_complex = detail::is_boost_complex<T>; // may be a duplicate of https://github.com/BoostGSoC21/math/pull/8#discussion_r660549430

  }} // boost::multiprecision

#endif
