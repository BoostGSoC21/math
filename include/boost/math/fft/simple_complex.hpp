///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_FFT_REAL_SCOMPLEX_HPP
  #define BOOST_MATH_FFT_REAL_SCOMPLEX_HPP

  namespace boost { namespace math {  namespace fft {
  
  namespace detail {
  
  template<class T>
  struct simple_complex
  // Simple std::complex-like class to perform computations of the real
  // transforms when complex structure is needed, so that we don't need to
  // deduce a complex type.
  {
    using value_type = T;
    T x{},y{};
    
    constexpr simple_complex(T c1 = T{},T c2 = T{}):
        x{c1}, y{c2}
    {
    }
    constexpr simple_complex(const simple_complex<T>& c):
        x{c.x}, y{c.y}
    {
    }
    
    simple_complex<T>& operator *= (const simple_complex<T>& that)
    // old school multiplication
    {
      T x_new = x * that.x - y * that.y;
      y = x * that.y + y * that.x;
      x = x_new;
      return *this;
    }
    // simple_complex<T>& operator *= (const simple_complex<T>& that)
    // // Gauss multiplication
    // {
    //   T k1 = that.x*(x+y),
    //     k2 = x*(that.y-that.x),
    //     k3 = y*(that.x+that.y);
    //   x = k1-k3,
    //   y = k1+k2;
    //   return *this;
    // }
    simple_complex<T>& operator += (const simple_complex<T>& that)
    {
      x += that.x;
      y += that.y;
      return *this;
    }
    simple_complex<T>& operator -= (const simple_complex<T>& that)
    {
      x -= that.x;
      y -= that.y;
      return *this;
    }
    
    T real()const{ return x; }
    T imag()const{ return y; }
    
    void real(T v){ x=v; }
    void imag(T v){ y=v; }
    
  };
  
  template<class T>
  simple_complex<T> operator * (
    const simple_complex<T>& A, 
    const simple_complex<T>& B)
  {
    simple_complex<T> C{A};
    return C*=B;
  }
  template<class T>
  simple_complex<T> operator + (
    const simple_complex<T>& A, 
    const simple_complex<T>& B)
  {
    simple_complex<T> C{A};
    return C+=B;
  }
  template<class T>
  simple_complex<T> operator - (
    const simple_complex<T>& A, 
    const simple_complex<T>& B)
  {
    simple_complex<T> C{A};
    return C-=B;
  }

  } // namespace detail

  } } } // namespace boost::math::fft

#endif // BOOST_MATH_FFT_REAL_SCOMPLEX_HPP
