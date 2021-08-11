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
  
  template<class T>
  inline void real_dft_2(
    const T* in, 
    T* out, int)
  {
    T o1 = in[0]+in[1], o2 = in[0]-in[1] ;
    out[0] = o1;
    out[1] = o2;
  }
  template<class T>
  inline void real_inverse_dft_2(
    const T* in, 
    T* out, int)
  {
    T o1 = in[0]+in[1], o2 = in[0]-in[1] ;
    out[0] = o1;
    out[1] = o2;
  }
  
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
    
    for(int i=0;i<n;++i) out[i] *= n;
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
  
  template<class T>
  void real_inverse_dft_prime_bruteForce_outofplace(
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
    
    {
      T sum_x{0.};
      for(long i=1,j=N-1;i<j;++i,--j)
      {
        sum_x += in_first[i];
      }
      // if(i<j) // i==j never happens for odd sizes
      out[0] = ( 2*sum_x + in_first[0]);
    } 
    for(long l=1;l<N; ++l)
    {
      T sum_x{0.},sum_y{0.};
      for(long i=1,j=N-1;i<j;++i,--j)
      {
        sum_x += in_first[i] * complex_root_of_unity_real<T>(N,i*l*sign);
        sum_y += in_first[j] * complex_root_of_unity_imag<T>(N,i*l*sign);
      }
      // if(i<j) // i==j never happens for odd sizes
      out[l] = (2*sum_x - 2*sum_y + in_first[0]);
    }
  }
  
  template<class T,class Allocator_t>
  void real_inverse_dft_prime_bruteForce_inplace(
    T* in_first, 
    T* in_last, 
    int sign,
    const Allocator_t& alloc)
  {
    std::vector<T,Allocator_t> work_space(in_first,in_last,alloc);
    real_inverse_dft_prime_bruteForce_outofplace(in_first,in_last,work_space.data(),sign);
    std::copy(work_space.begin(),work_space.end(),in_first);
  }
  template<class T, class Allocator_t>
  void real_inverse_dft_prime_bruteForce(
    const T* in_first, 
    const T* in_last, 
    T* out, 
    int sign,
    const Allocator_t& alloc)
  {
    if(in_first==out)
      real_inverse_dft_prime_bruteForce_inplace(out,out+std::distance(in_first,in_last),sign,alloc);
    else
      real_inverse_dft_prime_bruteForce_outofplace(in_first,in_last,out,sign);
  }
  
  template <class T, class allocator_t>
  void real_dft_composite_outofplace(
            const T *in_first, 
            const T *in_last, 
            T* out, 
            int /*sign*/,
            const allocator_t& alloc)
  {
    /*
      Cooley-Tukey mapping, intrinsically out-of-place, Decimation in Time
      composite sizes.
    */
    using allocator_type = allocator_t;
    using ComplexType = ::boost::multiprecision::complex<T>; 
    using ComplexAllocator = typename std::allocator_traits<allocator_type>::template rebind_alloc<ComplexType>;
    
    const long n = static_cast<long>(std::distance(in_first,in_last));
    if(n <=0 )
      return;
    
    if (n == 1)
    {
        out[0]=in_first[0];
        return;
    }
    std::array<int,32> prime_factors;
    const int nfactors = prime_factorization(n,prime_factors.begin());
    
    // reorder input
    for (long i = 0; i < n; ++i)
    {
        long j = 0, k = i;
        for (int ip=0;ip<nfactors;++ip)
        {
            int p = prime_factors[ip];
            j = j * p + k % p;
            k /= p;
        }
        out[j] = in_first[i];
    }
    
    std::reverse(prime_factors.begin(), prime_factors.begin()+nfactors);
    
    //auto show = [&] (const T* beg, const T* end)
    //{
    //  for(;beg!=end;++beg)
    //  {
    //    std::cout << *beg << ", ";
    //  }
    //  std::cout << "\n";
    //};
    
    // butterfly pattern
    long len = 1;
    for (int ip=0;ip<nfactors;++ip)
    {
      int p = prime_factors[ip];
      long len_old = len;
      len *= p;
      
      std::vector<ComplexType,ComplexAllocator> tmp(p,alloc);
      //std::cout << "pass " << ip << "\n";
      for (long i = 0; i < n; i += len)
      {
        //std::cout << "    i = " << i << "\n";
        for(long k=0;2*k<=len_old;++k)
        {
          if(k==0)
          {
            tmp[0] = ComplexType{out[i],0.};
            for(long j=1;j<p;++j)
              tmp[j] = ComplexType{out[i + j*len_old],0.};
          }else if(2*k == len_old)
          {
            tmp[0] = ComplexType{out[i + k ],0.};
            for(long j=1;j<p;++j)
              tmp[j] = ComplexType{out[i + j*len_old +k ],0.}
                * complex_root_of_unity<ComplexType>(len,k*j);
          }else
          {
            tmp[0] = ComplexType{out[i + k ],-out[i+len_old-k]};
            for(long j=1;j<p;++j)
              tmp[j] = ComplexType{out[i + j*len_old +k ],-out[i+j*len_old + len_old-k]}
                * complex_root_of_unity<ComplexType>(len,k*j);
          }
          if(p==2)
          {
            complex_dft_2(tmp.data(),tmp.data(),1);
          }
          else
          {
            complex_dft_prime_rader<ComplexType,ComplexAllocator>(tmp.data(),tmp.data()+p,tmp.data(),1,alloc);
          }
          for(long j=0;j<p;++j)
          {
            int posx = j*len_old + k, posy = len - k - j*len_old;
            
            if(posx==0)
            {
              out[i+posx] = tmp[j].real();
            }else if(posx>posy)
            {
              out[i+posx] = tmp[j].imag();
              out[i+posy] = tmp[j].real();
            }else
            {
              out[i+posy] = -tmp[j].imag();
              out[i+posx] = tmp[j].real();
            }
          }
        }
        //show(out+i,out+i+len);
      }
    }
  }
  
  template<class T,class Allocator_t>
  void real_dft_composite_inplace(
          T* in_first, 
          T* in_last, 
          int sign,
          const Allocator_t& alloc)
  {
    std::vector<T,Allocator_t> work_space(in_first,in_last,alloc);
    real_dft_composite_outofplace(in_first,in_last,work_space.data(),sign,alloc);
    std::copy(work_space.begin(),work_space.end(),in_first);
  }
  template<class T, class Allocator_t>
  void real_dft_composite(
          const T* in_first, 
          const T* in_last, 
          T* out, 
          int sign,
          const Allocator_t& alloc)
  {
    if(in_first==out)
      real_dft_composite_inplace(out,out+std::distance(in_first,in_last),sign,alloc);
    else
      real_dft_composite_outofplace(in_first,in_last,out,sign,alloc);
  }
  
  
  template <class T, class allocator_t>
  void real_inverse_dft_composite_outofplace(
            const T *in_first, 
            const T *in_last, 
            T* out, 
            int /*sign*/,
            const allocator_t& alloc)
  {
    /*
      Cooley-Tukey mapping, intrinsically out-of-place, Decimation in Time
      composite sizes.
      Reverse graph.
    */
    using allocator_type = allocator_t;
    using ComplexType = ::boost::multiprecision::complex<T>; 
    using ComplexAllocator = typename std::allocator_traits<allocator_type>::template rebind_alloc<ComplexType>;
    
    const long n = static_cast<long>(std::distance(in_first,in_last));
    if(n <=0 )
      return;
    
    std::copy(in_first,in_last,out);
    if (n == 1)
        return;
        
    std::array<int,32> prime_factors;
    const int nfactors = prime_factorization(n,prime_factors.begin());
    
    // butterfly pattern
    for (long ip=0,len=n,prev_len = len;ip<nfactors;++ip,len = prev_len)
    {
      int p = prime_factors[ip];
      prev_len = len/p;
      
      std::vector<ComplexType,ComplexAllocator> tmp(p,alloc);
      for (long i = 0; i < n; i += len)
      {
        for(long k=0;2*k<=prev_len;++k)
        {
          for(long j=0;j<p;++j)
          {
            int posx = j*prev_len + k, posy = len - k - j*prev_len;
            
            if(posx==0 || posx==posy)
            {
              tmp[j] = ComplexType {out[i+posx],0.};
            }else if(posx>posy)
            {
              tmp[j] = ComplexType{out[i+posy],out[i+posx]};
            }else // if(posx<posy)
            {
              tmp[j] = ComplexType{out[i+posx],-out[i+posy]};
            }
          }
          if(p==2)
          {
            complex_dft_2(tmp.data(),tmp.data(),-1);
          }
          else
          {
            complex_dft_prime_rader<ComplexType,ComplexAllocator>(tmp.data(),tmp.data()+p,tmp.data(),-1,alloc);
          }
          
          if(k==0)
          {
            out[i] = tmp[0].real();
            for(long j=1;j<p;++j)
              out[i + j*prev_len] = tmp[j].real();
          }else if(2*k == prev_len)
          {
            out[i + k ] = tmp[0].real();
            for(long j=1;j<p;++j)
              out[i + j*prev_len +k ] = (tmp[j]* complex_root_of_unity<ComplexType>(len,-k*j) ). real();
          }else
          {
            out[i+k]          =  tmp[0].real();
            out[i+prev_len-k] = -tmp[0].imag();
            for(long j=1;j<p;++j)
            {
              ComplexType cplx = tmp[j] * complex_root_of_unity<ComplexType>(len,-k*j);
              out[i + j*prev_len +k ]        = cplx.real();
              out[i+j*prev_len + prev_len-k] = -cplx.imag();
            }
          }
        }
      }
    }
    
    std::vector<T,allocator_type> tmp(out,out+n);
    // reorder
    for (long i = 0; i < n; ++i)
    {
        long j = 0, k = i;
        for (int ip=0;ip<nfactors;++ip)
        {
            int p = prime_factors[ip];
            j = j * p + k % p;
            k /= p;
        }
        out[i] = tmp[j];
    }
  }
  
  template<class T,class Allocator_t>
  void real_inverse_dft_composite_inplace(
          T* in_first, 
          T* in_last, 
          int sign,
          const Allocator_t& alloc)
  {
    std::vector<T,Allocator_t> work_space(in_first,in_last,alloc);
    real_inverse_dft_composite_outofplace(in_first,in_last,work_space.data(),sign,alloc);
    std::copy(work_space.begin(),work_space.end(),in_first);
  }
  template<class T, class Allocator_t>
  void real_inverse_dft_composite(
          const T* in_first, 
          const T* in_last, 
          T* out, 
          int sign,
          const Allocator_t& alloc)
  {
    if(in_first==out)
      real_inverse_dft_composite_inplace(out,out+std::distance(in_first,in_last),sign,alloc);
    else
      real_inverse_dft_composite_outofplace(in_first,in_last,out,sign,alloc);
  }
  
  } // namespace detail

  } } } // namespace boost::math::fft

#endif // BOOST_MATH_FFT_REAL_ALGORITHMS_HPP


