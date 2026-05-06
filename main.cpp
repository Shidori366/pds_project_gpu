#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/device_library_decls.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>

typedef struct {
    int8_t* arr;
    int64_t size;
} Array;

typedef struct {
    Array template_c;
    size_t segmentCount;
} Information;

__global__ void processSegment(int64_t segmentStart, Array template_t,
                               Array output) {
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (global_idx > template_t.size) {
        return;
    }

    if (template_t.arr[global_idx] == 0) {
        return;
    }

    int64_t multiplier = (segmentStart + global_idx - 1) / global_idx;

    while (global_idx * multiplier - segmentStart < output.size) {
        output.arr[global_idx * multiplier - segmentStart] = 0;
        ++multiplier;
    }
}

Information getInformation(int64_t n) {
    int64_t m = std::sqrt(n) + 1;
    Array template_c = Array{new int8_t[m], m};

    int64_t p = 2;

    for (int64_t i = 0; i < m; ++i) {
        if (i < 2) {
            template_c.arr[i] = 0;
            continue;
        }

        template_c.arr[i] = 1;
    }

    for (int64_t i = 2; i < m; ++i) {
        int64_t multiplier = 2;
        if (template_c.arr[i] == 1) {
            p = i;
            while (p * multiplier < m) {
                template_c.arr[multiplier * p] = 0;
                ++multiplier;
            }
        }
    }

    size_t segmentCount = n / m;

    if (n % m != 0) {
        segmentCount = n / m + 1;
    }

    return Information{template_c, segmentCount - 1};
}

void process(int64_t n) {
    using namespace std::chrono;

    auto begin = steady_clock::now();

    Information info = getInformation(n);
    int64_t length = info.template_c.size;
    int64_t segmentStart = length;

    int threads = 256;
    int blocks = (info.template_c.size + threads - 1) / threads;

    // template on gpu
    Array template_g;
    template_g.size = info.template_c.size;
    hipError_t error =
        hipMalloc(&template_g.arr, info.template_c.size * sizeof(int8_t));
    error = hipMemcpy(template_g.arr, info.template_c.arr,
                      template_g.size * sizeof(int8_t), hipMemcpyHostToDevice);

    // output on gpu
    Array output_g;
    output_g.size = length;
    error = hipMalloc(&output_g.arr, length * sizeof(int8_t));

    for (int64_t i = 0; i < info.segmentCount; ++i) {
        segmentStart = length * (i + 1);
        if (i + 1 == info.segmentCount && n - segmentStart < length) {
            length = n - segmentStart + 1;
        }

        output_g.size = length;
        error = hipMemset(output_g.arr, 1, length * sizeof(int8_t));

        processSegment<<<blocks, threads>>>(segmentStart, template_g, output_g);
    }
    error = hipDeviceSynchronize();
    auto end = steady_clock::now();
    std::cout << "Time difference = " << duration_cast<milliseconds>(end - begin).count() << " ms" << std::endl;
    error = hipFree(output_g.arr);
    error = hipFree(template_g.arr);
    delete[] info.template_c.arr;
}

int main() {
    hipDeviceProp_t props;

    hipGetDeviceProperties(&props, 0);

    std::cout << "Device: " << props.name << std::endl;
    int64_t n = 0;
    std::cin >> n;

    process(n);
}
