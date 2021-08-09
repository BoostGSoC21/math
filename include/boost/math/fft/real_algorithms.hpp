///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_FFT_REAL_ALGORITHMS_HPP
  #define BOOST_MATH_FFT_REAL_ALGORITHMS_HPP

  #include <algorithm>
  #include <numeric>
  #include <cmath>
  
  namespace boost { namespace math {  namespace fft {
  
  namespace detail {
  
  template<class RealType>
  RealType complex_root_of_unity_real(long n,long p=1)
  /*
    Computes cos(-2 pi p/n)
  */
  {
    p = modulo(p,n);
    
    if(p==0)
      return RealType(1);
    
    long g = gcd(p,n); 
    n/=g;
    p/=g;
    switch(n)
    {
      case 1:
        return RealType(1);
      case 2:
        return p==0 ? RealType(1) : RealType(-1);
      case 4:
        return p==0 ? RealType(1) : 
               p==1 ? RealType(0) :
               p==2 ? RealType(-1) :
                      RealType(0) ;
    }
    using std::cos;
    RealType phase = -2*p*boost::math::constants::pi<RealType>()/n;
    return RealType(cos(phase));
  }
  template<class RealType>
  RealType complex_root_of_unity_imag(long n,long p=1)
  /*
    Computes sin(-2 pi p/n)
  */
  {
    p = modulo(p,n);
    
    if(p==0)
      return RealType(0);
    
    long g = gcd(p,n); 
    n/=g;
    p/=g;
    switch(n)
    {
      case 1:
        return RealType(0);
      case 2:
        return p==0 ? RealType(0) : RealType(0);
      case 4:
        return p==0 ? RealType(0) : 
               p==1 ? RealType(-1) :
               p==2 ? RealType(0) :
                      RealType(1) ;
    }
    using std::sin;
    RealType phase = -2*p*boost::math::constants::pi<RealType>()/n;
    return RealType(sin(phase));
  }
  
  template <class T>
  void real_dft_power2(const T *in_first, const T *in_last, T* out, int sign)
  {
    /*
      Cooley-Tukey mapping, in-place Decimation in Time 
    */
    const long ptrdiff = static_cast<long>(std::distance(in_first,in_last));
    if(ptrdiff <=0 )
      return;
    const long n = lower_bound_power2(ptrdiff);
    
    if(in_first!=out)
      std::copy(in_first,in_last,out);
    
    if (n == 1)
        return;

    // Gold-Rader bit-reversal algorithm.
    for(int i=0,j=0;i<n-1;++i)
    { 
      if(i<j)
        std::swap(out[i],out[j]);
      for(int k=n>>1;!( (j^=k)&k );k>>=1);
    }
    
    for (int len = 2, prev_len = 1; len <= n; len <<= 1,prev_len<<=1)
    {
      for (int i = 0; i < n; i += len)
      {
        {
          // j=0;
          T* u = out + i, *v = out + i + prev_len;
          T Bu = *u, Bv = *v;
          *u = Bu + Bv;
          *v = Bu - Bv;
        }
        for(int j=1;j < prev_len/2;++j)
        {
          T cos{ complex_root_of_unity_real<T>(len,j*sign) }, 
            sin{ complex_root_of_unity_imag<T>(len,j*sign) };
          
          T *ux = out + i + j, 
            *uy = out + i + len - j;
            
          T *vx = out + i + prev_len - j, 
            *vy = out + i + prev_len + j;
          
          T prev_ux = *ux, 
            prev_uy = *uy;
            
          T prev_vx = *vx, 
            prev_vy = *vy;
          
          *ux = prev_ux + cos * prev_vy + sin * prev_uy;
          *uy = prev_vx + cos * prev_uy - sin * prev_vy;
          *vx = prev_ux - cos * prev_vy - sin * prev_uy;
          *vy =-prev_vx + cos * prev_uy - sin * prev_vy;
        }
      }
    }
  }
  template <class T>
  void real_inverse_dft_power2(const T *in_first, const T *in_last, T* out, int sign)
  {
    /*
      Cooley-Tukey mapping, in-place Decimation in Time 
      Reverse flow graph of the real_dft_power2
    */
    const long ptrdiff = static_cast<long>(std::distance(in_first,in_last));
    if(ptrdiff <=0 )
      return;
    const long n = lower_bound_power2(ptrdiff);
    
    if(in_first!=out)
      std::copy(in_first,in_last,out);
    
    if (n == 1)
        return;
    
    for (int len = n, prev_len = len/2; len >= 2; len >>= 1,prev_len>>=1)
    {
      for (int i = 0; i < n; i += len)
      {
        {
          // j=0;
          T* u = out + i, *v = out + i + prev_len;
          T Bu = *u, Bv = *v;
          *u = (Bu + Bv)*0.5;
          *v = (Bu - Bv)*0.5;
        }
        for(int j=1;j < prev_len/2;++j)
        {
          T cos{ complex_root_of_unity_real<T>(len,j*sign) }, 
            sin{ complex_root_of_unity_imag<T>(len,j*sign) };
          
          T *ux = out + i + j, 
            *uy = out + i + len - j;
            
          T *vx = out + i + prev_len - j, 
            *vy = out + i + prev_len + j;
          
          T prev_ux = *ux, 
            prev_uy = *uy;
            
          T prev_vx = *vx, 
            prev_vy = *vy;
          
          T sum_x = (prev_ux + prev_vx) * 0.5,
            dif_x = (prev_ux - prev_vx) * 0.5,
            sum_y = (prev_uy + prev_vy) * 0.5,
            dif_y = (prev_uy - prev_vy) * 0.5;
            
          *ux = sum_x;
          *vx = dif_y;
          
          *uy = cos * sum_y + sin*dif_x;
          *vy = cos * dif_x - sin*sum_y;
        }
      }
    }
    
    // Gold-Rader bit-reversal algorithm.
    for(int i=0,j=0;i<n-1;++i)
    { 
      if(i<j)
        std::swap(out[i],out[j]);
      for(int k=n>>1;!( (j^=k)&k );k>>=1);
    }
  }
  
  template<class T>
  void real_dft_prime_bruteForce_outofplace(
    const T* in_first, 
    const T* in_last, 
    T* out, int sign)
  /*
    assumptions: 
    - allocated memory in out is enough to hold distance(in_first,in_last) element,
    - out!=in
  */
  {
    const long N = static_cast<long>(std::distance(in_first,in_last));
    if(N<=0)
      return;
    
    out[0] = std::accumulate(in_first+1,in_last,in_first[0]);
    
    for(long i=1,j=N-1;i<j;++i,--j)
    {
      T sum_x{in_first[0]},sum_y = 0;
      for(long l=1;l<N; ++l)
      {
        sum_x += in_first[l] * complex_root_of_unity_real<T>(N,i*l*sign);
        sum_y += in_first[l] * complex_root_of_unity_imag<T>(N,i*l*sign);
      }
      // if(i<j) // i==j never happens for odd sizes
      out[j] = -sum_y;
      out[i] = sum_x;
    }
  }
  
  template<class T,class Allocator_t>
  void real_dft_prime_bruteForce_inplace(
    T* in_first, 
    T* in_last, 
    int sign,
    const Allocator_t& alloc)
  {
    std::vector<T,Allocator_t> work_space(in_first,in_last,alloc);
    real_dft_prime_bruteForce_outofplace(in_first,in_last,work_space.data(),sign);
    std::copy(work_space.begin(),work_space.end(),in_first);
  }
  template<class T, class Allocator_t>
  void real_dft_prime_bruteForce(
    const T* in_first, 
    const T* in_last, 
    T* out, 
    int sign,
    const Allocator_t& alloc)
  {
    if(in_first==out)
      real_dft_prime_bruteForce_inplace(out,out+std::distance(in_first,in_last),sign,alloc);
    else
      real_dft_prime_bruteForce_outofplace(in_first,in_last,out,sign);
  }
  
  } // namespace detail

  } } } // namespace boost::math::fft

#endif // BOOST_MATH_FFT_REAL_ALGORITHMS_HPP


