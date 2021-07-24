/*
    boost::math::fft example 02.
    
    FFT transform-like API,
    custom engine
*/

#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>

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
    
    // gsl engine, forward transform, out-of-place
    fft::transform<fft::gsl_dft<std::complex<double>>>::forward(A.cbegin(),A.cend(),B.begin());
    
    print(B);
    
    // fftw engine, backward transform, in-place
    fft::transform<fft::fftw_dft<std::complex<double>>>::backward(B.cbegin(),B.cend(),B.begin());
    
    print(B);
    return 0;
}

