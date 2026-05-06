#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/device_library_decls.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ostream>

#define DEFAULT_SEGMENT_SIZE (1LL * 1024 * 1024 * 1024)

typedef struct {
    int8_t* arr;
    int64_t size;
} Array;

typedef struct {
    int64_t* arr;
    int64_t size;
} TemplateArray;

typedef struct {
    TemplateArray template_c;
    size_t segment_count;
    int64_t segment_size;
    int64_t segment_start;
} Information;

__global__ void process_segment(int64_t segment_start, TemplateArray template_t,
                                Array output) {
    for (int64_t i = 0; i < template_t.size; ++i) {
        int64_t p = template_t.arr[i];
        int64_t first_multiplier = (segment_start + p - 1) / p;

        for (int64_t j =
                 first_multiplier + blockIdx.x * blockDim.x + threadIdx.x;
             (j * p - segment_start) < output.size;
             j += blockDim.x * gridDim.x) {
            output.arr[j * p - segment_start] = 0;
        }
    }
}

Information get_information(int64_t n) {
    int64_t m = std::sqrt(n) + 1;
    Array template_c = Array{new int8_t[m], m};
    int64_t segment_size = DEFAULT_SEGMENT_SIZE;

    if (segment_size > n - m + 1) {
        segment_size = n - m + 1;
    }

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

    TemplateArray template_optimalized = TemplateArray{new int64_t[m], m};
    int64_t j = 0;

    for (int64_t i = 0; i < template_c.size; ++i) {
        if (template_c.arr[i] == 1) {
            template_optimalized.arr[j++] = i;
        }
    }
    template_optimalized.size = j;

    int64_t nwt = n - template_c.size;
    size_t segment_count = (nwt + segment_size - 1) / segment_size;

    delete[] template_c.arr;
    return Information{template_optimalized, segment_count, segment_size, m};
}

void process(int64_t n) {
    using namespace std::chrono;


    Information info = get_information(n);
    int64_t segment_size = info.segment_size;
    int64_t segment_start = info.segment_start;

    std::cout << "template_size: " << info.template_c.size << std::endl;
    std::cout << "segment_size: " << info.segment_size << std::endl;
    std::cout << "segment_count: " << info.segment_count << std::endl;

    // template on gpu
    TemplateArray template_g;
    template_g.size = info.template_c.size;
    hipError_t error =
        hipMalloc(&template_g.arr, info.template_c.size * sizeof(int64_t));
    error = hipMemcpy(template_g.arr, info.template_c.arr,
                      template_g.size * sizeof(int64_t), hipMemcpyHostToDevice);

    // output on gpu
    Array output_g;
    output_g.size = segment_size;
    error = hipMalloc(&output_g.arr, segment_size * sizeof(int8_t));

    auto begin = steady_clock::now();
    for (int64_t i = 0; i < info.segment_count; ++i) {
        if (i + 1 == info.segment_count && n - segment_start < segment_size) {
            segment_size = n - segment_start + 1;
        }

        output_g.size = segment_size;
        error = hipMemset(output_g.arr, 1, segment_size * sizeof(int8_t));
        int threads = 256;
        int blocks = 216;

        process_segment<<<blocks, threads>>>(segment_start, template_g,
                                             output_g);

        error = hipDeviceSynchronize();
        segment_start += segment_size;
    }
    auto end = steady_clock::now();
    std::cout << "Time = " << duration_cast<milliseconds>(end - begin).count()
              << " ms" << std::endl;

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
