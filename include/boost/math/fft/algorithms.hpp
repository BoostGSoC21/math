///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_FFT_ALGORITHMS_HPP
  #define BOOST_MATH_FFT_ALGORITHMS_HPP

  #include <algorithm>
  #include <numeric>
  #include <cmath>
  #include <vector>
  #include <boost/math/constants/constants.hpp>
  #include <boost/math/fft/discrete_maths.hpp>
  #include <boost/container/static_vector.hpp>
  


  namespace boost { namespace math {  namespace fft {
  
  namespace detail {
  
  template<typename InputIterator1,
           typename InputIterator2,
           typename OutputIterator,
           typename allocator_t >
  void raw_convolution(InputIterator1 input1_begin,
                   InputIterator1 input1_end,
                   InputIterator2 input2_begin,
                   OutputIterator output,
                   const allocator_t& alloc);
  
  
  template<class ComplexType>
  ComplexType complex_root_of_unity(long n,long p=1)
  /*
    Computes exp(-i 2 pi p/n)
  */
  {
    using real_value_type = typename ComplexType::value_type;
    p = modulo(p,n);
    
    if(p==0)
      return ComplexType(1,0);
    
    long g = gcd(p,n); 
    n/=g;
    p/=g;
    switch(n)
    {
      case 1:
        return ComplexType(1,0);
      case 2:
        return p==0 ? ComplexType(1,0) : ComplexType(-1,0);
      case 4:
        return p==0 ? ComplexType(1,0) : 
               p==1 ? ComplexType(0,-1) :
               p==2 ? ComplexType(-1,0) :
                      ComplexType(0,1) ;
    }
    using std::sin;
    using std::cos;
    real_value_type phase = -2*p*boost::math::constants::pi<real_value_type>()/n;
    return ComplexType(cos(phase),sin(phase));
  }
  template<class ComplexType>
  ComplexType complex_inverse_root_of_unity(long n,long p=1)
  /*
    Computes exp(i 2 pi p/n)
  */
  {
    return complex_root_of_unity<ComplexType>(n,-p);
  }
  
  template<class complex_value_type>
  inline void complex_dft_2(
    const complex_value_type* in, 
    complex_value_type* out, int)
  {
    complex_value_type 
        o1 = in[0]+in[1], o2 = in[0]-in[1] ;
    out[0] = o1;
    out[1] = o2;
  }
  
  template<class T>
  void dft_prime_bruteForce_outofplace(const T* in_first, const T* in_last, T* out, const T w)
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
    
    T wi=w;
    for(long i=1;i<N;++i, wi*=w)
    {
      T wij{wi};
      T sum{in_first[0]};
      for(long j=1;j<N; ++j, wij*=wi)
      {
        sum += in_first[j]*wij;
      }
      out[i] = sum;
    }
  }
  template<class T,class Allocator_t>
  void dft_prime_bruteForce_inplace(T* in_first, T* in_last, const T w, Allocator_t& alloc)
  {
    std::vector<T,Allocator_t> work_space(in_first,in_last,alloc);
    dft_prime_bruteForce_outofplace(in_first,in_last,work_space.data(),w);
    std::copy(work_space.begin(),work_space.end(),in_first);
  }
  template<class T, class Allocator_t>
  void dft_prime_bruteForce(const T* in_first, const T* in_last, T* out, const T w, Allocator_t& alloc)
  {
    if(in_first==out)
      dft_prime_bruteForce_inplace(out,out+std::distance(in_first,in_last),w,alloc);
    else
      dft_prime_bruteForce_outofplace(in_first,in_last,out,w);
  }
  
  template<class complex_value_type>
  void complex_dft_prime_bruteForce_outofplace(
    const complex_value_type* in_first, 
    const complex_value_type* in_last, 
    complex_value_type* out, int sign)
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
    
    for(long i=1;i<N;++i)
    {
      complex_value_type sum{in_first[0]};
      for(long j=1;j<N; ++j)
      {
        sum += in_first[j] * complex_root_of_unity<complex_value_type>(N,i*j*sign);
      }
      out[i] = sum;
    }
  }
  
  template<class complex_value_type,class Allocator_t>
  void complex_dft_prime_bruteForce_inplace(
    complex_value_type* in_first, 
    complex_value_type* in_last, 
    int sign,
    Allocator_t& alloc)
  {
    std::vector<complex_value_type,Allocator_t> work_space(in_first,in_last,alloc);
    complex_dft_prime_bruteForce_outofplace(in_first,in_last,work_space.data(),sign);
    std::copy(work_space.begin(),work_space.end(),in_first);
  }
  template<class complex_value_type, class Allocator_t>
  void complex_dft_prime_bruteForce(
    const complex_value_type* in_first, 
    const complex_value_type* in_last, 
    complex_value_type* out, 
    int sign,
    Allocator_t& alloc)
  {
    if(in_first==out)
      complex_dft_prime_bruteForce_inplace(out,out+std::distance(in_first,in_last),sign,alloc);
    else
      complex_dft_prime_bruteForce_outofplace(in_first,in_last,out,sign);
  }
  
  /*
    Rader's FFT on prime sizes
  */
  template<class complex_value_type, class allocator_t>
  void complex_dft_prime_rader(
    const complex_value_type *in_first, 
    const complex_value_type *in_last, 
    complex_value_type* out, 
    int sign,
    const allocator_t& alloc = allocator_t{})
  // precondition: distance(in_first,in_last) is prime > 2
  {
    using allocator_type = allocator_t;
    const long my_n = static_cast<long>(std::distance(in_first,in_last));
    
    std::vector<complex_value_type,allocator_type> A(my_n-1,alloc),W(my_n-1,alloc),B(my_n-1,alloc);
    
    const long g = primitive_root(my_n);
    const long g_inv = power_mod(g,my_n-2,my_n);
    
    for(long i=0;i<my_n-1;++i)
    {
      W[i] = complex_root_of_unity<complex_value_type>(my_n,sign*power_mod(g_inv,i,my_n));
      A[i] = in_first[ power_mod(g,i+1,my_n) ];
    }
    
    raw_convolution(A.begin(),A.end(),W.begin(),B.begin(),alloc);
    
    complex_value_type a0 = in_first[0];
    complex_value_type sum_a {a0};
    for(long i=1;i<my_n;++i)
        sum_a += in_first[i];
    
    out[0] = sum_a;
    for(long i=1;i<my_n;++i)
    {
      out[i]=a0;
    }
    for(long i=1;i<my_n;++i)
    {
      out[ power_mod(g_inv,i,my_n) ] += B[i-1];
    }
  }
  
  
  template <class T, class allocator_t>
  void dft_composite(const T *in_first, 
                     const T *in_last, 
                     T* out, 
                     const T e,
                     const allocator_t& alloc)
  {
    /*
      Cooley-Tukey mapping, intrinsically out-of-place, Decimation in Time
      composite sizes.
    */
    using allocator_type = allocator_t;
    
    const long n = static_cast<unsigned int>(std::distance(in_first,in_last));
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
    
    // butterfly pattern
    long len = 1;
    for (int ip=0;ip<nfactors;++ip)
    {
      int p = prime_factors[ip];
      long len_old = len;
      len *= p;
      T w_len = power(e, n / len);
      T w_p = power(e,n/p);
      
      std::vector<T,allocator_type> tmp(p,alloc);
      for (long i = 0; i < n; i += len)
      {
        for(long k=0;k<len_old;++k)
        {
          for(long j=0;j<p;++j)
            if(j==0 || k==0)
              tmp[j] = out[i + j*len_old +k ];
            else
              tmp[j] = out[i + j*len_old +k ] * power(w_len,k*j);
          
          dft_prime_bruteForce_inplace(tmp.data(),tmp.data()+p,w_p,alloc);
          
          for(long j=0;j<p;++j)
            out[i+ j*len_old + k] = tmp[j];
        }
      }
    }
  }
  
  template <class ComplexType, class allocator_t>
  void complex_dft_composite(const ComplexType *in_first, 
                             const ComplexType *in_last, 
                             ComplexType* out, 
                             int sign,
                             const allocator_t& alloc)
  {
    /*
      Cooley-Tukey mapping, intrinsically out-of-place, Decimation in Time
      composite sizes.
    */
    using allocator_type = allocator_t;
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
    
    // butterfly pattern
    long len = 1;
    for (int ip=0;ip<nfactors;++ip)
    {
      int p = prime_factors[ip];
      long len_old = len;
      len *= p;
      
      std::vector<ComplexType,allocator_type> tmp(p,alloc);
      for (long i = 0; i < n; i += len)
      {
        for(long k=0;k<len_old;++k)
        {
          for(long j=0;j<p;++j)
            if(j==0 || k==0)
              tmp[j] = out[i + j*len_old +k ];
            else
              tmp[j] = out[i + j*len_old +k ] * complex_root_of_unity<ComplexType>(len,k*j*sign);
          
          if(p==2)
            complex_dft_2(tmp.data(),tmp.data(),sign);
          else
          {
          //  complex_dft_prime_bruteForce(tmp.data(),tmp.data()+p,tmp.data(),sign);
            complex_dft_prime_rader(tmp.data(),tmp.data()+p,tmp.data(),sign,alloc);
          }
          for(long j=0;j<p;++j)
            out[i+ j*len_old + k] = tmp[j];
        }
      }
    }
  }
  
  template <class T>
  void dft_power2(const T *in_first, const T *in_last, T* out, const T e)
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

    // auto _1 = T{1};
    
    int nbits = 0;
    ::boost::container::static_vector<T,32> e2{e};
    for (int m = n / 2; m > 0; m >>= 1, ++nbits)
      e2.push_back(e2.back() * e2.back());

    std::reverse(e2.begin(), e2.end());

    // Gold-Rader bit-reversal algorithm.
    for(int i=0,j=0;i<n-1;++i)
    { 
      if(i<j)
        std::swap(out[i],out[j]);
      for(int k=n>>1;!( (j^=k)&k );k>>=1);
    }
    
    
    for (int len = 2, k = 1; len <= n; len <<= 1, ++k)
    {
      for (int i = 0; i < n; i += len)
      {
        {
          int j=0;
          T* u = out + i + j, *v = out + i + j + len / 2;
          T Bu = *u, Bv = *v;
          *u = Bu + Bv;
          *v = Bu - Bv;
        }
        
        T ej = e2[k];
        for (int j = 1; j < len / 2; ++j)
        {
          T* u = out + i + j, *v = out + i + j + len / 2;
          T Bu = *u, Bv = *v * ej;
          *u = Bu + Bv;
          *v = Bu - Bv;
          ej *= e2[k];
        }
      }
    }
  }
  
  template<class complex_value_type>
  void complex_dft_power2(
    const complex_value_type *in_first, 
    const complex_value_type *in_last, 
    complex_value_type* out, 
    int sign)
  {
    // Naive in-place complex DFT.
    const long ptrdiff = static_cast<long>(std::distance(in_first,in_last));
    if(ptrdiff <=0 )
      return;
    const long my_n = lower_bound_power2(ptrdiff);
    
    if(in_first!=out)
      std::copy(in_first,in_last,out);
    
    if (my_n == 1)
        return;
    
    if(in_first!=out)
      std::copy(in_first, in_last, out);

    // Recursive decimation in frequency.
    for(long m = my_n; m > 1; m /= 2)
    {
      long mh = m / 2;
      
      for(long j = 0; j < mh; ++j)
      {
        complex_value_type cs{complex_root_of_unity<complex_value_type>(m,j*sign)};

        for (long t1=j; t1 < j + my_n; t1 += m)
        {
          complex_value_type u = out[t1];
          complex_value_type v = out[t1 + mh];

          out[t1]      =  u + v;
          out[t1 + mh] = (u - v) * cs;
        }
      }
    }

    // data reordering:
    for(long m = 1, j = 0; m < my_n - 1; ++m)
    {
      for(long k = my_n >> 1; (!((j^=k)&k)); k>>=1)
      {
        ;
      }

      if(j > m)
      {
        std::swap(out[m], out[j]);
      }
    }

    // Normalize for backwards transform (done externally).
  }
  
  template<typename InputIterator1,
           typename InputIterator2,
           typename OutputIterator,
           typename allocator_t  >
  void raw_convolution(InputIterator1 input1_begin,
                   InputIterator1 input1_end,
                   InputIterator2 input2_begin,
                   OutputIterator output,
                   const allocator_t& alloc)
  {
    using input_value_type = typename std::iterator_traits<InputIterator1>::value_type;
    using real_value_type  = typename input_value_type::value_type;
    using allocator_type   = allocator_t;
    
    const long N = std::distance(input1_begin,input1_end);
    const long N_extended = detail::is_power2(N) ? N : detail::upper_bound_power2(2*N-1);
    
    std::vector<input_value_type, allocator_type> In1(N_extended,alloc),In2(N_extended,alloc),Out(N_extended,alloc);
    
    std::copy(input1_begin,input1_end,In1.begin());
    
    InputIterator2 input2_end{input2_begin};
    std::advance(input2_end,N);
    std::copy(input2_begin,input2_end,In2.begin());
    
    // padding
    for(long i=N;i<N_extended;++i)
      In1[i]=In2[i]=input_value_type{0};
    
    // fake N-periodicity
    if(N!=N_extended)
    for(long i=1;i<N;++i)
      In2[N_extended-N+i] = In2[i];
    
    complex_dft_power2(In1.data(),In1.data()+In1.size(),In1.data(),1);
    complex_dft_power2(In2.data(),In2.data()+In1.size(),In2.data(),1);
    
    // direct convolution
    std::transform(In1.begin(),In1.end(),In2.begin(),Out.begin(),std::multiplies<input_value_type>()); 
    
    complex_dft_power2(Out.data(),Out.data()+Out.size(),Out.data(),-1);
    
    const real_value_type inv_N = real_value_type{1}/N_extended;
    for(auto & x : Out)
        x *= inv_N;
    
    std::copy(Out.begin(),Out.begin() + N,output);
  }
  

  } // namespace detail

  } } } // namespace boost::math::fft

#endif // BOOST_MATH_FFT_ALGORITHMS_HPP

