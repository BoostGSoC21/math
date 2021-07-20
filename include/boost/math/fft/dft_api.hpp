///////////////////////////////////////////////////////////////////
//  Copyright Eduardo Quintana 2021
//  Copyright Janek Kozicki 2021
//  Copyright Christopher Kormanyos 2021
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

//  What's in this header: 
//  The frontend of a plan-like FFT interface.

#ifndef BOOST_MATH_DFTAPI_HPP
  #define BOOST_MATH_DFTAPI_HPP

#include <algorithm>

  namespace boost { namespace math { namespace fft { 
  namespace detail {

  // fftw_plan-like Fourier Transform API
  
  /*
    RingType axioms:
    1. Abelian group addition (operator+)
      -> closure
      -> associativity
      -> neutral element (0)
      -> inverse (operator-)
      -> commutativity
    2. Monoid multiplication (operator*)
      -> closure
      -> associativity
      -> neutral element (1)
    3. addition and multiplication compatibility
      -> left distributivity, ie. a*(b+c) == a*b + a*c
      -> right distributivity, ie. (b+c)*a == b*a + c*a
  */
  
  template< template<class ... Args> class BackendType, class T, class allocator_t >
  class dft : public BackendType<T,allocator_t>
  {
    public:
    using base_type       = BackendType<T,allocator_t>;
    using value_type      = typename base_type::value_type;
    using allocator_type  = typename base_type::allocator_type;
    
    template<class U, class A>
    using other = dft<BackendType,U,A>;
    
    private:
    using RingType = value_type;
    allocator_type alloc;
    std::vector<value_type,allocator_type> my_mem;
    enum class execution_type { forward, backward };
    
    template<typename InputIteratorType,
             typename OutputIteratorType>
    void execute(
      execution_type ex,
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const RingType*>::value == true)
                               && (std::is_convertible<OutputIteratorType,       RingType*>::value == true))>::type* = nullptr)
    {
      resize(std::distance(in_first,in_last));
      
      if(ex==execution_type::backward)
        base_type::backward(in_first,out);
      else
        base_type::forward(in_first,out);
    }
    
    template<typename InputIteratorType,
             typename OutputIteratorType>
    void execute(
      execution_type ex,
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const RingType*>::value == false)
                               && (std::is_convertible<OutputIteratorType,       RingType*>::value == true))>::type* = nullptr)
    {
      resize(std::distance(in_first,in_last));
      std::copy(in_first, in_last, out);
      
      if(ex==execution_type::backward)
        base_type::backward(out,out);
      else
        base_type::forward(out,out);
    }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void execute(
      execution_type ex,
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const RingType*>::value == true)
                               && (std::is_convertible<OutputIteratorType,       RingType*>::value == false))>::type* = nullptr)
    {
      resize(std::distance(in_first,in_last));
      my_mem.resize(size());

      if(ex==execution_type::backward)
        base_type::backward(in_first,my_mem.data());
      else
        base_type::forward(in_first,my_mem.data());

      std::copy(std::begin(my_mem), std::end(my_mem), out);
    }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void execute(
      execution_type ex,
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const RingType*>::value == false)
                               && (std::is_convertible<OutputIteratorType,       RingType*>::value == false))>::type* = nullptr)
    {
      resize(std::distance(in_first,in_last));
      my_mem.resize(size());
      std::copy(in_first, in_last, std::begin(my_mem));

      if(ex==execution_type::backward)
        base_type::backward(my_mem.data(),my_mem.data());
      else
        base_type::forward(my_mem.data(),my_mem.data());
        
      std::copy(std::begin(my_mem),std::end(my_mem), out);
    }

  public:
    using base_type::size;
    using base_type::resize;

    // complex types ctor. n: the size of the dft
    constexpr dft(unsigned int n, const allocator_type& in_alloc = allocator_type{} )
      : base_type(n,in_alloc), alloc{in_alloc}, my_mem{in_alloc} { }

    // ring types ctor. n: the size of the dft, w: an n-root of unity
    constexpr dft(unsigned int n, RingType w, const allocator_type& in_alloc = allocator_type{} ) 
      : base_type( n, w, in_alloc ), alloc{in_alloc}, my_mem{in_alloc} { }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void forward(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      execute(execution_type::forward,in_first,in_last,out);
    }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void backward(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      execute(execution_type::backward,in_first,in_last,out);
    }
    
    // experimental
    // template<typename InputIterator,
    //          typename OutputIterator>
    // static void static_forward(InputIterator  input_begin,
    //                  InputIterator  input_end,
    //                  OutputIterator output)
    // {
    //   base_type plan(static_cast<unsigned int>(std::distance(input_begin, input_end)));
    //   plan.forward(input_begin, output);
    // }
  };
  
  } // namespace detail
  
  template< class dft_plan_t >
  struct transform
  {
    
    template<class RingType, class allocator_t = std::allocator<RingType> >
    using plan_type = typename dft_plan_t::template other< RingType,allocator_t > ;
    
    
    // std::transform-like Fourier Transform API
    // for complex types
    template<typename InputIterator,
             typename OutputIterator>
    static void forward(InputIterator  input_begin,
                     InputIterator  input_end,
                     OutputIterator output)
    {
      using input_value_type  = typename std::iterator_traits<InputIterator >::value_type;
      plan_type<input_value_type> plan(static_cast<unsigned int>(std::distance(input_begin, input_end)));
      plan.forward(input_begin, input_end, output);
    }
  
    // std::transform-like Fourier Transform API
    // for complex types
    template<typename InputIterator,
             typename OutputIterator>
    static void backward(InputIterator  input_begin,
                      InputIterator  input_end,
                      OutputIterator output)
    {
      using input_value_type  = typename std::iterator_traits<InputIterator >::value_type;
      plan_type<input_value_type> plan(static_cast<unsigned int>(std::distance(input_begin, input_end)));
      plan.backward(input_begin, input_end, output);
    }
    
    // std::transform-like Fourier Transform API
    // for Ring types
    template<typename InputIterator,
             typename OutputIterator,
             typename value_type>
    static void forward(InputIterator  input_begin,
                     InputIterator  input_end,
                     OutputIterator output,
                     value_type w)
    {
      using input_value_type  = typename std::iterator_traits<InputIterator >::value_type;
      plan_type<input_value_type> plan(static_cast<unsigned int>(std::distance(input_begin, input_end)),w);
      plan.forward(input_begin, input_end, output);
    }
  
    // std::transform-like Fourier Transform API
    // for Ring types
    template<typename InputIterator,
             typename OutputIterator,
             typename value_type>
    static void backward(InputIterator  input_begin,
                      InputIterator  input_end,
                      OutputIterator output,
                      value_type w)
    {
      using input_value_type  = typename std::iterator_traits<InputIterator >::value_type;
      plan_type<input_value_type> plan(static_cast<unsigned int>(std::distance(input_begin, input_end)),w);
      plan.backward(input_begin, input_end, output);
    }
  
    template<typename InputIterator1,
             typename InputIterator2,
             typename OutputIterator>
    static void convolution(
        InputIterator1 input1_begin,
        InputIterator1 input1_end,
        InputIterator2 input2_begin,
        OutputIterator output)
    {
      using input_value_type  = typename std::iterator_traits<InputIterator1>::value_type;
      using real_value_type  = typename input_value_type::value_type;
      // using allocator_type    = std::allocator<input_value_type>;
      const long N = std::distance(input1_begin,input1_end);
      plan_type<input_value_type> plan(static_cast<unsigned int>(N));
      
      std::vector<input_value_type> In1(N),In2(N),Out(N);
      
      std::copy(input1_begin,input1_end,In1.begin());
      
      InputIterator2 input2_end{input2_begin};
      std::advance(input2_end,N);
      std::copy(input2_begin,input2_end,In2.begin());
      
      plan.forward(In1.begin(),In1.end(),In1.begin());
      plan.forward(In2.begin(),In2.end(),In2.begin());
      
      // direct convolution
      std::transform(In1.begin(),In1.end(),In2.begin(),Out.begin(),std::multiplies<input_value_type>()); 
      
      plan.backward(Out.begin(),Out.end(),Out.begin());
      
      const real_value_type inv_N = real_value_type{1}/N;
      for(auto & x : Out)
          x *= inv_N;
      
      std::copy(Out.begin(),Out.end(),output);
    }
    
  };
  
  } } } // namespace boost::math::fft


#endif // BOOST_MATH_DFTAPI_HPP
