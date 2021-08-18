#include <benchmark/benchmark.h>
#include <random>
#include <complex>
#include <boost/math/fft/bsl_backend.hpp>
#include <boost/math/fft/fftw_backend.hpp>
#include <boost/math/fft/gsl_backend.hpp>
#include <boost/math/constants/constants.hpp>

#include "fft_test_helpers.hpp"

std::default_random_engine gen;
std::uniform_real_distribution<double> distribution;

using Real = double;
std::vector<Real> random_vec(size_t N)
{
    std::vector<Real> V(N);
    for (auto& x : V)
      x = distribution(gen);
    return V;
}

void bench_bsl(benchmark::State& state)
{
    using fft_plan = boost::math::fft::bsl_rdft<Real>;
    auto A = random_vec(state.range(0));
    fft_plan P(A.size());
    for (auto _ : state)
    {
        P.real_to_halfcomplex(A.data(),A.data()+A.size(),A.data());
    }
    state.SetComplexityN(state.range(0));
}
void bench_gsl(benchmark::State& state)
{
    using fft_plan = boost::math::fft::gsl_rdft<Real>;
    auto A = random_vec(state.range(0));
    fft_plan P(A.size());
    for (auto _ : state)
    {
        P.real_to_halfcomplex(A.data(),A.data()+A.size(),A.data());
    }
    state.SetComplexityN(state.range(0));
}
void bench_fftw(benchmark::State& state)
{
    using fft_plan = boost::math::fft::fftw_rdft<Real>;
    auto A = random_vec(state.range(0));
    fft_plan P(A.size());
    for (auto _ : state)
    {
        P.real_to_halfcomplex(A.data(),A.data()+A.size(),A.data());
    }
    state.SetComplexityN(state.range(0));
}

//// powers of 2
BENCHMARK(bench_bsl)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 20)
    ->Complexity(benchmark::oNLogN);

BENCHMARK(bench_gsl)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 20)
    ->Complexity(benchmark::oNLogN);

BENCHMARK(bench_fftw)
    ->RangeMultiplier(4)
    ->Range(1 << 8, 1 << 20)
    ->Complexity(benchmark::oNLogN);

// powers of 10
//BENCHMARK(bench_bsl)
//    ->RangeMultiplier(10)
//    ->Range(100, 1000000)
//    ->Complexity(benchmark::oNLogN);
//
//BENCHMARK(bench_gsl)
//    ->RangeMultiplier(10)
//    ->Range(100, 1000000)
//    ->Complexity(benchmark::oNLogN);
//
//BENCHMARK(bench_fftw)
//    ->RangeMultiplier(10)
//    ->Range(100, 1000000)
//    ->Complexity(benchmark::oNLogN);

// primes
//BENCHMARK(bench_bsl)
//    ->Arg(109)
//    ->Arg(1009)
//    ->Arg(10009)
//    ->Arg(100003)
//    ->Complexity(benchmark::oNLogN);
//
//BENCHMARK(bench_gsl)
//    ->Arg(109)
//    ->Arg(1009)
//    ->Arg(10009)
//    ->Arg(100003)
//    ->Complexity(benchmark::oNLogN);
//
//BENCHMARK(bench_fftw)
//    ->Arg(109)
//    ->Arg(1009)
//    ->Arg(10009)
//    ->Arg(100003)
//    ->Complexity(benchmark::oNLogN);

BENCHMARK_MAIN();
