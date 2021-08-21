/*
    boost::math::fft example 03.
    
    FFT plan-like API,
    default engine
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
    
    // default engine, create plan
    fft::bsl_dft<std::complex<double>> P(A.size());
    
    // forward transform, out-of-place
    P.forward(A.cbegin(),A.cend(),B.begin());
    
    print(B);
    
    // backward transform, in-place
    P.backward(B.cbegin(),B.cend(),B.begin());
    
    print(B);
    return 0;
}


