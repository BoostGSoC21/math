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
#include <vector>
#include <complex>

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
  
  /*
    Type A to type B execution API.
  */
  template<typename A_t, typename B_t, typename allocator_t>
  class asymmetric_executor
  {
    public:
    using value_type1 = A_t;
    using value_type2 = B_t;
    //using allocator_type1 = allocator_t;
    //using allocator_type2 = allocator_t;
    using allocator_type1 = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_type1>;
    using allocator_type2 = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_type2>;
    
    private:
    
    using buffer_type1 = std::vector<value_type1,allocator_type1> ;
    using buffer_type2 = std::vector<value_type2,allocator_type2> ;
    buffer_type1 my_mem_1;
    buffer_type2 my_mem_2;
    
    public:
    constexpr asymmetric_executor(const allocator_t& in_alloc = allocator_t{})
      : my_mem_1(in_alloc), my_mem_2(in_alloc) 
    { }
    
    template<typename InputIteratorType,
             typename OutputIteratorType,
             typename EngineType >
    void execute(
      InputIteratorType in_first, InputIteratorType /* in_last */,
      OutputIteratorType out,
      EngineType engine, 
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const value_type1*>::value == true)
                               && (std::is_convertible<OutputIteratorType,       value_type2*>::value == true))>::type* = nullptr)
    {
      engine(in_first,out);
    }
    
    template<typename InputIteratorType,
             typename OutputIteratorType,
             typename EngineType>
    void execute(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      EngineType engine,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const value_type1*>::value == false)
                               && (std::is_convertible<OutputIteratorType,       value_type2*>::value == true))>::type* = nullptr)
    {
      my_mem_1.resize(std::distance(in_first,in_last));
      std::copy(in_first, in_last, std::begin(my_mem_1));
      engine(my_mem_1.data(),out);
    }

    template<typename InputIteratorType,
             typename OutputIteratorType,
             typename EngineType>
    void execute(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      EngineType engine,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const value_type1*>::value == true)
                               && (std::is_convertible<OutputIteratorType,       value_type2*>::value == false))>::type* = nullptr)
    {
      my_mem_2.resize(std::distance(in_first,in_last));
      engine(in_first,my_mem_2.data());
      std::copy(std::begin(my_mem_2), std::end(my_mem_2), out);
    }

    template<typename InputIteratorType,
             typename OutputIteratorType,
             typename EngineType>
    void execute(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      EngineType engine,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const value_type1*>::value == false)
                               && (std::is_convertible<OutputIteratorType,       value_type2*>::value == false))>::type* = nullptr)
    {
      my_mem_1.resize(std::distance(in_first,in_last));
      my_mem_2.resize(my_mem_1.size());
      std::copy(in_first, in_last, std::begin(my_mem_1));
      engine(my_mem_1.data(),my_mem_2.data());
      std::copy(std::begin(my_mem_2),std::end(my_mem_2), out);
    }
    
  };
  
  /*
    Type T to type T execution API.
  */
  template<typename A_t, typename allocator_t>
  class symmetric_executor
  {
    public:
    using value_type = A_t;
    using allocator_type = allocator_t;
    
    private:
    using buffer_type = std::vector<value_type,allocator_type> ;
    buffer_type my_mem;
    
    public:
    constexpr symmetric_executor(const allocator_type& in_alloc = allocator_type{})
      : my_mem(in_alloc)
    { }
    
    template<typename InputIteratorType,
             typename OutputIteratorType,
             typename EngineType >
    void execute(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      EngineType engine, 
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const value_type*>::value == true)
                               && (std::is_convertible<OutputIteratorType,       value_type*>::value == true))>::type* = nullptr)
    {
      engine(in_first,out);
    }
    
    template<typename InputIteratorType,
             typename OutputIteratorType,
             typename EngineType>
    void execute(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      EngineType engine,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const value_type*>::value == false)
                               && (std::is_convertible<OutputIteratorType,       value_type*>::value == true))>::type* = nullptr)
    {
      std::copy(in_first, in_last, out);
      engine(out,out);
    }

    template<typename InputIteratorType,
             typename OutputIteratorType,
             typename EngineType>
    void execute(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      EngineType engine,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const value_type*>::value == true)
                               && (std::is_convertible<OutputIteratorType,       value_type*>::value == false))>::type* = nullptr)
    {
      my_mem.resize(std::distance(in_first,in_last));
      engine(in_first,my_mem.data());
      std::copy(std::begin(my_mem), std::end(my_mem), out);
    }

    template<typename InputIteratorType,
             typename OutputIteratorType,
             typename EngineType>
    void execute(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out,
      EngineType engine,
      typename std::enable_if<(   (std::is_convertible<InputIteratorType,  const value_type*>::value == false)
                               && (std::is_convertible<OutputIteratorType,       value_type*>::value == false))>::type* = nullptr)
    {
      my_mem.resize(std::distance(in_first,in_last));
      std::copy(in_first, in_last, std::begin(my_mem));
      engine(my_mem.data(),my_mem.data());
      std::copy(std::begin(my_mem),std::end(my_mem), out);
    }
    
  };
  
  template< template<class ... Args> class BackendType, class T, class allocator_t >
  class algebraic_dft : 
        public BackendType<T,allocator_t> , 
        public symmetric_executor<T,allocator_t>
  {
    public:
    using value_type      = T;
    using allocator_type  = allocator_t;
    
    using backend         = BackendType<value_type,allocator_type>;
    using executor        = symmetric_executor<value_type,allocator_type>;
    
    template<class U, class A>
    using other = algebraic_dft<BackendType,U,A>;
    
    private:
    allocator_type alloc;
    value_type root,inverse_root;
    
  public:
    using backend::size;
    using backend::resize;
    
    // complex types ctor. n: the size of the dft
    constexpr algebraic_dft(unsigned int n, value_type w, const allocator_type& in_alloc = allocator_type{} )
      : backend(n,in_alloc), 
        executor(in_alloc), 
        alloc{in_alloc} ,
        root{w},
        inverse_root{ backend::inverse_root(w) }
    { }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void forward(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      resize(std::distance(in_first,in_last));
      executor::execute(in_first,in_last,out,
        [this](const value_type* i, value_type* o)
        {
          backend::dft(i,o,root);
        });
    }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void backward(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      resize(std::distance(in_first,in_last));
      executor::execute(in_first,in_last,out,
        [this](const value_type* i, value_type* o)
        {
          backend::dft(i,o,inverse_root);
        });
    }
  };
  
  template< template<class ... Args> class BackendType, class T, class allocator_t >
  class complex_dft : 
        public BackendType<T,allocator_t> , 
        public symmetric_executor<T,allocator_t>,
        public asymmetric_executor<typename T::value_type,T,allocator_t>,
        public asymmetric_executor<T,typename T::value_type,allocator_t>
  {
    public:
    using complex_type    = T;
    using real_type       = typename T::value_type;
    using allocator_type  = allocator_t;
    
    using value_type      = complex_type;
    
    using backend         = BackendType<complex_type,allocator_type>;
    using executor_C2C    = symmetric_executor<complex_type,allocator_type>;
    using executor_R2C    = asymmetric_executor<real_type,complex_type,allocator_type>;
    using executor_C2R    = asymmetric_executor<complex_type,real_type,allocator_type>;
    
    template<class U, class A>
    using other = complex_dft<BackendType,U,A>;
    
    private:
    allocator_type alloc;
    
  public:
    using backend::size;
    using backend::resize;

    // complex types ctor. n: the size of the dft
    constexpr complex_dft(unsigned int n, const allocator_type& in_alloc = allocator_type{} )
      : backend(n,in_alloc), 
        executor_C2C(in_alloc), 
        executor_R2C(in_alloc), 
        executor_C2R(in_alloc), 
        alloc{in_alloc} { }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void forward(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      resize(std::distance(in_first,in_last));
      executor_C2C::execute(in_first,in_last,out,
        [this](const complex_type* i, complex_type* o)
        {
          backend::forward(i,o);
        });
    }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void backward(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      resize(std::distance(in_first,in_last));
      executor_C2C::execute(in_first,in_last,out,
        [this](const complex_type* i, complex_type* o)
        {
          backend::backward(i,o);
        });
    }
  };
  
  template< template<class ... Args> class BackendType, class T, class allocator_t >
  class real_dft : 
        public BackendType<T,allocator_t> , 
        public symmetric_executor<T,allocator_t>,
        public asymmetric_executor<T,std::complex<T>,allocator_t>,
        public asymmetric_executor<std::complex<T>,T,allocator_t>
  {
    public:
    using value_type      = T;
    using real_type       = T;
    using complex_type    = std::complex<T>; // TODO: a complex selector, or the complex could be choosen by the user
    using allocator_type  = allocator_t;
    
    using backend         = BackendType<real_type,allocator_type>;
    using executor_halfcomplex = symmetric_executor<real_type,allocator_type>;
    using executor_r2c = asymmetric_executor<real_type,complex_type,allocator_type>;
    using executor_c2r = asymmetric_executor<complex_type,real_type,allocator_type>;
    
    template<class U, class A>
    using other = real_dft<BackendType,U,A>;
    
    private:
    allocator_type alloc;
    
  public:
    using backend::size;
    using backend::unique_complex_size;
    using backend::resize;

    // complex types ctor. n: the size of the dft
    constexpr real_dft(unsigned int n, const allocator_type& in_alloc = allocator_type{} )
      : backend(n,in_alloc), 
        executor_halfcomplex(in_alloc), 
        alloc{in_alloc} { }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void real_to_halfcomplex(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      resize(std::distance(in_first,in_last));
      executor_halfcomplex::execute(in_first,in_last,out,
        [this](const real_type* i, real_type* o)
        {
          backend::real_to_halfcomplex(i,o);
        });
    }
    template<typename InputIteratorType,
             typename OutputIteratorType>
    void halfcomplex_to_real(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      resize(std::distance(in_first,in_last));
      executor_halfcomplex::execute(in_first,in_last,out,
        [this](const real_type* i, real_type* o)
        {
          backend::halfcomplex_to_real(i,o);
        });
    }
    
    template<typename InputIteratorType,
             typename OutputIteratorType>
    void real_to_complex(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      resize(std::distance(in_first,in_last));
      executor_r2c::execute(in_first,in_last,out,
        [this](const real_type* i, complex_type* o)
        {
          backend::template real_to_complex<complex_type>(i,o);
        });
    }

    template<typename InputIteratorType,
             typename OutputIteratorType>
    void complex_to_real(
      InputIteratorType in_first, InputIteratorType in_last,
      OutputIteratorType out)
    {
      resize(std::distance(in_first,in_last));
      executor_c2r::execute(in_first,in_last,out,
        [this](const complex_type* i, real_type* o)
        {
          backend::template complex_to_real<complex_type>(i,o);
        });
    }
    
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
