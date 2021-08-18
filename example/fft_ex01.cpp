/*
    boost::math::fft example 01.
    
    FFT transform-like API,
    default fft engine
*/

#include <boost/math/fft/bsl_backend.hpp>

#include <iostream>
#include <vector>
#include <complex>

namespace fft = boost::math::fft;

template<class T>
void print(const std::vector< std::complex<T> >& V)
{
    for(auto i=0UL;i<V.size();++i)
        std::cout << "V[" << i << "] = " 
            << V[i].real() << ", " << V[i].imag() << '\n';
}
int main()
{
    std::vector< std::complex<double> > A{1.0,2.0,3.0,4.0},B(A.size());
    using fft_transform = ::boost::math::fft::bsl_transform;
    
    // default fft engine, forward transform, out-of-place
    // following is called, but with an alias bsl_transform
    // fft::transform<fft::bsl_dft<std::complex<double>>>::forward(A.cbegin(),A.cend(),B.begin());
    fft_transform::forward(A.cbegin(),A.cend(),B.begin());
    
    print(B);
    
    // default fft engine, backward transform, in-place
    // fft::transform<fft::bsl_dft<std::complex<double>>>::backward(B.cbegin(),B.cend(),B.begin());
    fft_transform::backward(B.cbegin(),B.cend(),B.begin());
    
    print(B);
    
    return 0;
}
