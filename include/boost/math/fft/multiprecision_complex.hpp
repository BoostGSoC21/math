///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file provides two traits:
//  boost::multiprecision::make_boost_complex<Real>         - creates a coplex type supported by Boost
//  boost::multiprecision::is_boost_complex<T>::value       - a bool telling whether the type is a complex number supported by Boost
//                                                            it has nothing to do with boost::is_complex<T> which recognizes only std::complex<T>
//
// TODO: this file should be split apart and each specializetion should go to the end of the relevant file in Boost.Multiprecision
//   → specialization for float128      → complex128    goes to <boost/multiprecision/complex128.hpp>
//   → specialization for cpp_bin_float → cpp_complex   goes to <boost/multiprecision/cpp_complex.hpp>
//   → specialization for mpfr_float    → mpc_complex   goes to <boost/multiprecision/mpc.hpp>

#ifndef BOOST_MATH_FFT_MULTIPRECISION_COMPLEX
#define BOOST_MATH_FFT_MULTIPRECISION_COMPLEX
#include <boost/config.hpp>
#include <boost/cstdfloat.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpfi.hpp>
#include <boost/core/demangle.hpp>
// now include complex types
#include <boost/multiprecision/complex128.hpp>
#include <boost/multiprecision/cpp_complex.hpp>
#include <boost/multiprecision/mpc.hpp>

namespace boost { namespace math {  namespace fft {  namespace detail {
//namespace boost { namespace multiprecision {

    #if __cplusplus < 201700L
    template<typename ...> using void_t = void;
    #else
    using std::void_t;
    #endif


    template<class T, typename = void_t<> >
    struct is_boost_complex : std::false_type
    {};

    template<class T >
    struct is_boost_complex<T, 
        void_t<
            typename T::value_type,
            decltype(sin(std::declval<typename T::value_type>())),
            decltype(cos(std::declval<typename T::value_type>()))
         > > : std::true_type
    {};



  template<typename RealType> struct select_complex {
    using R    = typename std::decay<RealType>::type;

    static constexpr bool is_fundamental_fpnumber =
      (   std::is_same<R,float>::value
       || std::is_same<R,double>::value
       || std::is_same<R,long double>::value);

  #ifdef BOOST_MATH_USE_FLOAT128
    static constexpr bool is_float128             = std::is_same<R,boost::multiprecision::float128>::value;
  #else
    static constexpr bool is_float128             = false;
  #endif

    template<typename Real>
    struct is_multiprecision_number {
      static constexpr bool value = false;
      using make_complex = void;
    };
    template<typename Real, boost::multiprecision::expression_template_option Et>
    struct is_multiprecision_number <boost::multiprecision::number<Real,Et>> {
      static constexpr bool value = true;
      using make_complex = boost::multiprecision::number<boost::multiprecision::complex_adaptor<Real>,Et>;
    };

    static constexpr bool is_acceptable_number =
      (   is_fundamental_fpnumber
       || is_float128
       || is_multiprecision_number<R>::value);

    static_assert(is_acceptable_number , "Error: cannot create complex for given real type.");

    using type = typename std::conditional< is_fundamental_fpnumber,
        std::complex<R>,
  #ifdef BOOST_MATH_USE_FLOAT128
        typename std::conditional< is_float128,
          boost::multiprecision::complex128,
  #endif
          // only the boost::multiprecision::number remain, thanks to static_assert above.
          typename is_multiprecision_number<R>::make_complex
  #ifdef BOOST_MATH_USE_FLOAT128
        >::type
  #endif
      >::type;
  };

//}} // boost::multiprecision
} } } } // namespace boost::math::fft::detail

#endif

