// array_lambda.cu

#include <cuda_runtime.h>

#include <cstddef>
#include <format>
#include <iomanip>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>


//---------------------------------------------------------------------------
// CUDA error checking using C++20 std::source_location
//---------------------------------------------------------------------------

inline void cuda_check(
    cudaError_t error,
    std::string_view operation,
    std::source_location location = std::source_location::current())
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::format(
                "{}:{}: {} failed: {}",
                location.file_name(),
                location.line(),
                operation,
                cudaGetErrorString(error)));
    }
}


//---------------------------------------------------------------------------
// Lightweight, non-owning 2D array view.
//
// This object contains only:
//
//   * a pointer to the data
//   * the array dimensions
//
// It can therefore be copied inexpensively into a CUDA kernel or captured
// by value in a device lambda.
//---------------------------------------------------------------------------

template <typename T>
class Array2DView
{
public:
    __host__ __device__
    Array2DView(T* data, int nx, int ny)
        : data_(data),
          nx_(nx),
          ny_(ny)
    {
    }

    __host__ __device__
    T& operator()(int i, int j) const
    {
        return data_[j * nx_ + i];
    }

    __host__ __device__
    int nx() const
    {
        return nx_;
    }

    __host__ __device__
    int ny() const
    {
        return ny_;
    }

private:
    T* data_{nullptr};

    int nx_{0};
    int ny_{0};
};


//---------------------------------------------------------------------------
// RAII owner of a contiguous GPU allocation.
//
// The object owns the memory returned by cudaMalloc(). Its destructor
// releases that memory automatically with cudaFree().
//
// Copying is disabled because there must be exactly one owner. Moving is
// allowed and transfers ownership.
//---------------------------------------------------------------------------

template <typename T>
class DeviceArray2D
{
public:
    DeviceArray2D() = default;

    DeviceArray2D(int nx, int ny)
        : nx_(nx),
          ny_(ny)
    {
        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&data_),
                size() * sizeof(T)),
            "cudaMalloc");
    }

    ~DeviceArray2D()
    {
        // Destructors should not throw. Therefore, do not call cuda_check()
        // here. For a small example, silently ignoring a cudaFree() error is
        // preferable to throwing during stack unwinding.
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceArray2D(DeviceArray2D const&) = delete;

    DeviceArray2D& operator=(DeviceArray2D const&) = delete;

    DeviceArray2D(DeviceArray2D&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          nx_(std::exchange(other.nx_, 0)),
          ny_(std::exchange(other.ny_, 0))
    {
    }

    DeviceArray2D& operator=(DeviceArray2D&& other) noexcept
    {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }

            data_ = std::exchange(other.data_, nullptr);
            nx_ = std::exchange(other.nx_, 0);
            ny_ = std::exchange(other.ny_, 0);
        }

        return *this;
    }

    [[nodiscard]]
    Array2DView<T> view() const
    {
        return Array2DView<T>{data_, nx_, ny_};
    }

    [[nodiscard]]
    T* data() const
    {
        return data_;
    }

    [[nodiscard]]
    int nx() const
    {
        return nx_;
    }

    [[nodiscard]]
    int ny() const
    {
        return ny_;
    }

    [[nodiscard]]
    std::size_t size() const
    {
        return static_cast<std::size_t>(nx_) *
               static_cast<std::size_t>(ny_);
    }

private:
    T* data_{nullptr};

    int nx_{0};
    int ny_{0};
};


//---------------------------------------------------------------------------
// Generic two-dimensional CUDA kernel.
//
// Each CUDA thread determines one pair of array indices (i, j) and invokes
// the supplied function for that array zone.
//---------------------------------------------------------------------------

template <typename F>
__global__
void parallel_for_kernel(int nx, int ny, F function)
{
    int const i =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    int const j =
        static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (i < nx && j < ny) {
        function(i, j);
    }
}


//---------------------------------------------------------------------------
// AMReX-like host-side launch function
//---------------------------------------------------------------------------

template <typename F>
void parallel_for(int nx, int ny, F function)
{
    constexpr unsigned int threads_x = 16;
    constexpr unsigned int threads_y = 16;

    dim3 const threads_per_block{
        threads_x,
        threads_y
    };

    dim3 const number_of_blocks{
        (static_cast<unsigned int>(nx) + threads_x - 1) / threads_x,
        (static_cast<unsigned int>(ny) + threads_y - 1) / threads_y
    };

    parallel_for_kernel<<<number_of_blocks, threads_per_block>>>(
        nx, ny, function);

    // This checks whether the kernel was launched successfully. It does not
    // necessarily detect an error that occurs later while the kernel runs.
    cuda_check(cudaGetLastError(), "parallel_for_kernel launch");
}


//---------------------------------------------------------------------------
// Main program
//---------------------------------------------------------------------------

int main()
{
    try {
        constexpr int nx = 8;
        constexpr int ny = 6;

        // This object owns the GPU allocation. The allocation is released
        // automatically when array goes out of scope.
        DeviceArray2D<double> array(nx, ny);

        // The view is a small, non-owning object containing a device pointer
        // and the array dimensions.
        auto const array_view = array.view();

        // Capture the view by value. This copies the pointer and dimensions
        // into the device lambda, but does not copy the array elements.
        parallel_for(
            nx, ny,
            [=] __device__ (int i, int j)
            {
                array_view(i, j) = 100.0 * j + i;
            });

        // Wait for the kernel to finish. Runtime errors encountered during
        // execution are reported here.
        cuda_check(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize");

        // Copy the contiguous GPU data back to the host for printing.
        std::vector<double> host_data(array.size());

        cuda_check(
            cudaMemcpy(
                host_data.data(),
                array.data(),
                array.size() * sizeof(double),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host");

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                std::cout
                    << std::setw(6)
                    << host_data[static_cast<std::size_t>(j) * nx + i];
            }

            std::cout << '\n';
        }

        // No cudaFree() call is needed. The DeviceArray2D destructor handles
        // it automatically.
    }
    catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
