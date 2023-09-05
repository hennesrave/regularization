#include "regularizer.hpp"
#include <iostream>
#include <random>

int main( int argc, char** argv )
{
    const uint32_t pointCount { 1'000'000 };
    const uint32_t kernelRadius { 8 };
    const uint32_t iterations { 8 };

    // --- Generate random points --- //
    std::mt19937_64 engine { 42 };
    std::normal_distribution<float> distribution { 0.0f, 1.0f };

    std::vector<Regularizer::Point> points( pointCount );
    float absmax { 0.0f };

    for( auto& point : points )
    {
        point = Regularizer::Point {
            distribution( engine ),
            distribution( engine ),
        };
        absmax = std::max( absmax, std::max( std::abs( point.x ), std::abs( point.y ) ) );
    }

    // --- Normalize points to [-1, 1]² --- //
    for( auto& point : points )
    {
        point = Regularizer::Point {
            point.x / absmax,
            point.y / absmax
        };
    }

    // --- Perform regularization --- //
    const uint32_t measurementCount { 100 };
    std::vector<double> measurements( measurementCount );

    Regularizer regularizer {};
    for( uint32_t i { 0 }; i < measurementCount; ++i )
    {
        util::BufferWrapper pointsDevice { regularizer.uploadPoints( points ) };                                    // Upload points
        measurements[i] = regularizer.regularize( pointsDevice.buffer(), pointCount, kernelRadius, iterations );    // Regularize
        pointsDevice.destroy();                                                                                     // Destroy buffer
    }

    // NOTE: To retrieve the regularized points, use Regularizer::downloadPoints( ... )
    // const auto regularized { regularizer.downloadPoints( pointsDevice, pointCount ) };

    // --- Compute median measurement --- //
    std::sort( measurements.begin(), measurements.end() );
    const double milliseconds { measurements[measurementCount / 2] / 1'000'000 };
    std::cout << "Finished regularization of " << pointCount << " points with kernelRadius = " << kernelRadius << " and iterations = " << iterations << " in " << milliseconds << " ms!" << std::endl;
}