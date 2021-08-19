/*
    boost::math::fft example 01.
    FFT transform-like API,
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
    
    // Boost fft engine, forward transform, out-of-place.
    // The following call bsl_transform is an alias for fft::transform< fft::bsl_dft<> >
    fft_transform::forward(A.cbegin(),A.cend(),B.begin());
    
    print(B);
    
    // default fft engine, backward transform, in-place
    fft_transform::backward(B.cbegin(),B.cend(),B.begin());
    
    print(B);
    
    return 0;
}
