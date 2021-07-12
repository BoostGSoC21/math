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
  
  template< class BackendType >
  class dft : public BackendType
  {
    public:
    using value_type     = typename BackendType::value_type;
    using allocator_type = typename BackendType::allocator_type;
    
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
        backend_t::backward(in_first,out);
      else
        backend_t::forward(in_first,out);
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
        backend_t::backward(out,out);
      else
        backend_t::forward(out,out);
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
        backend_t::backward(in_first,my_mem.data());
      else
        backend_t::forward(in_first,my_mem.data());

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
        backend_t::backward(my_mem.data(),my_mem.data());
      else
        backend_t::forward(my_mem.data(),my_mem.data());
        
      std::copy(std::begin(my_mem),std::end(my_mem), out);
    }

  public:
    using backend_t = BackendType;
    using backend_t::size;
    using backend_t::resize;

    // complex types ctor. n: the size of the dft
    constexpr dft(unsigned int n, const allocator_type& in_alloc = allocator_type{} )
      : backend_t(n,in_alloc), alloc{in_alloc}, my_mem{in_alloc} { }

    // ring types ctor. n: the size of the dft, w: an n-root of unity
    constexpr dft(unsigned int n, RingType w, const allocator_type& in_alloc = allocator_type{} ) 
      : backend_t( n, w, in_alloc ), alloc{in_alloc}, my_mem{in_alloc} { }

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
  };
  
  } // namespace detail
  } } } // namespace boost::math::fft


#endif // BOOST_MATH_DFTAPI_HPP
