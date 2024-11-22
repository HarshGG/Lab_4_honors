#include <iostream>
#include <chrono>
#include <stdio.h>
#include <vector>

#include <adiak.hpp>
#include <caliper/cali-manager.h>
#include <caliper/cali.h>

/**
 * @brief Implemenation using shared memory for image and constant memory for
 * filter
 * @param a input image
 * @param b output image
 * @param nx image width
 * @param nx image length
 */
 __constant__ float fc[9];
__global__ void filter_shared(unsigned char *a, unsigned char *b, int nx,
                              int ny) {
    __shared__ uchar4 shared_arr[16 + 2][64 + 2];

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int x = blockIdx.x * blockDim.x + thread_x;
    int y = blockIdx.y * blockDim.y + thread_y;

    if (x >= nx || y >= ny) {
      return;
    } 

    shared_arr[thread_y + 1][thread_x + 1] = make_uchar4(a[y * nx + x], 0, 0, 0);

    if (thread_x == 0) {
      shared_arr[thread_y + 1][0] = make_uchar4(a[y * nx + max(x - 1, 0)], 0, 0, 0);
    }
    if (thread_x == blockDim.x - 1) {
      shared_arr[thread_y + 1][blockDim.x + 1] = make_uchar4(a[y * nx + min(x + 1, nx - 1)], 0, 0, 0);
    }
    if (thread_y == 0) {
      shared_arr[0][thread_x + 1] = make_uchar4(a[max(y - 1, 0) * nx + x], 0, 0, 0);
    }
    if (thread_y == blockDim.y - 1) {
      shared_arr[blockDim.y + 1][thread_x + 1] = make_uchar4(a[min(y + 1, ny - 1) * nx + x], 0, 0, 0);
    }


    if (thread_x == 0 && thread_y == 0) {
      shared_arr[0][0] = make_uchar4(a[max(y - 1, 0) * nx + max(x - 1, 0)], 0, 0, 0);
    }
    if (thread_x == blockDim.x - 1 && thread_y == 0) {
      shared_arr[0][blockDim.x + 1] = make_uchar4(a[max(y - 1, 0) * nx + min(x + 1, nx - 1)], 0, 0, 0);
    }
    if (thread_x == 0 && thread_y == blockDim.y - 1) {
      shared_arr[blockDim.y + 1][0] = make_uchar4(a[min(y + 1, ny - 1) * nx + max(x - 1, 0)], 0, 0, 0);
    }
    if (thread_x == blockDim.x - 1 && thread_y == blockDim.y - 1) {
      shared_arr[blockDim.y + 1][blockDim.x + 1] = make_uchar4(a[min(y + 1, ny - 1) * nx + min(x + 1, nx - 1)], 0, 0, 0);
    }

    __syncthreads();

    float sum = 0.0f;

    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        sum += f_c[(i + 1) * 3 + (j + 1)] * shared_arr[thread_y + 1 + i][thread_x + 1 + j].x;
      }
    }

    b[y * nx + x] = (unsigned char)min
    (255, max(0, (int)(sum + 0.5f)));

  }

/**
 * @brief Implemenation using global memory for image and constant memory for
 * filter
 * @param a input image
 * @param b output image
 * @param nx image width
 * @param nx image length
 */
__global__ void filter_constant(unsigned char *a, unsigned char *b, int nx,
                                int ny) {
    auto idx = [&nx](int y,int x){ return y*nx+x; };

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    if(x<0 || y <0 || x >= nx || y >= ny)return;

    int xl = max(0,x-1); int yl = max(0,y-1);
    int xh = min(nx-1,x+1); int yh = min(ny-1,y+1);

    float v = fc[0]*a[idx(yl,xl)] + fc[1]*a[idx(yl,x)] + fc[2]*a[idx(yl,xh)] +
    fc[3]*a[idx(y,xl)] + fc[4]*a[idx(y,x)] + fc[5]*a[idx(y,xh)] +
    fc[6]*a[idx(yh,xl)] + fc[7]*a[idx(yh,x)] + fc[8]*a[idx(yh,xh)];

    uint f = (uint)(v+0.5f);

    b[idx(y,x)] = (unsigned char)min(255,max(0,f));
  }

/**
 * @brief Implemenation using global memory fir filter and image
 * @param a input image
 * @param b output image
 * @param c filter
 * @param nx image width
 * @param nx image length
 */

__global__ void filter_global(unsigned char *a, unsigned char *b, int nx,
                              int ny, float *c) {
  auto idx = [&nx](int y,int x){ return y*nx+x; };

  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;

  if(x<0 || y <0 || x >= nx || y >= ny)return;

  int xl = max(0,x-1); int yl = max(0,y-1);
  int xh = min(nx-1,x+1); int yh = min(ny-1,y+1);

  float v = c[0]*a[idx(yl,xl)] + c[1]*a[idx(yl,x)] + c[2]*a[idx(yl,xh)] +
  c[3]*a[idx(y,xl)] + c[4]*a[idx(y,x)] + c[5]*a[idx(y,xh)] +
  c[6]*a[idx(yh,xl)] + c[7]*a[idx(yh,x)] + c[8]*a[idx(yh,xh)];

  uint f = (uint)(v+0.5f);

  b[idx(y,x)] = (unsigned char)min(255,max(0,f));
}

/**
 * @brief CPU implementation for the filter
 * @param a input image
 * @param b output image
 * @param c filter
 * @param nx image width
 * @param ny image length
 */
void filter_CPU(const std::vector<unsigned char> &a,
                std::vector<unsigned char> &b, int nx, int ny,
                const std::vector<float> &c) {
  auto idx = [&nx](int y, int x) { return y * nx + x; };

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      int xl = std::max(0, x - 1);
      int yl = std::max(0, y - 1);
      int xh = std::min(nx - 1, x + 1);
      int yh = std::min(ny - 1, y + 1);

      float v =
          c[0] * a[idx(yl, xl)] + c[1] * a[idx(yl, x)] + c[2] * a[idx(yl, xh)] +
          c[3] * a[idx(y, xl)] + c[4] * a[idx(y, x)] + c[5] * a[idx(y, xh)] +
          c[6] * a[idx(yh, xl)] + c[7] * a[idx(yh, x)] + c[8] * a[idx(yh, xh)];

      uint f = (uint)(v + 0.5f);
      b[idx(y, x)] =
          (unsigned char)std::min(255, std::max(0, static_cast<int>(f)));
    }
  }
}

int main(int argc, char *argv[]) {
  CALI_CXX_MARK_FUNCTION;

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  // Image size
  int nx = atoi(argv[1]);
  int ny = nx;
  int size = nx * ny;

  float kernel_time_global = 1.0f;
  float kernel_time_constant = 1.0f;
  float kernel_time_shared = 1.0f;

  std::vector<unsigned char> h_a(size, 0); // Input image
  std::vector<unsigned char> h_b_cpu(size, 0); // Output image (CPU)
  std::vector<unsigned char> h_b_gpu(size, 0); // Output image (GPU)
  std::vector<float> h_c = {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f}; // Filter coefficients

  // Initialize input image with random values
  for (int i = 0; i < size; ++i) {
      h_a[i] = rand() % 256;
  }

  // CPU timing
  auto cpu_start = std::chrono::high_resolution_clock::now();
  filter_CPU(h_a, h_b_cpu, nx, ny, h_c);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;
  std::cout << "CPU time: " << cpu_elapsed.count() << "s\n";

  // Allocate device memory
  unsigned char *d_a = nullptr, *d_b = nullptr;
  cudaMalloc(&d_a, size * sizeof(unsigned char));
  cudaMalloc(&d_b, size * sizeof(unsigned char));
  float* d_c = nullptr;
  cudaMalloc(&d_c, 9 * sizeof(float));
  cudaMemcpy(d_c, h_c.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);

  // Copy data to device
  cudaMemcpyToSymbol(fc, h_c.data(), 9 * sizeof(float));
  cudaMemcpy(d_a, h_a.data(), size * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Define block and grid sizes
  dim3 block(16, 16);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  // dim3 grid((nx + 64 - 1) / 64, (ny + 16 - 1) / 16);

  // global timing
  CALI_MARK_BEGIN("kernel_global");
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // TODO: Launch filter kernel
  filter_global<<<grid, block>>>(d_a, d_b, nx, ny, d_c);

  // global timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CALI_MARK_END("kernel_global");
  cudaEventElapsedTime(&kernel_time_global, start, stop);
  // TODO: Copy result back to host
  cudaMemcpy(h_b_gpu.data(), d_b, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  std::cout << "Global time: " << kernel_time_global / 1000.0f << "s\n";
  // TODO: Check result
  bool flag = true;
  for (int i = 0; i < size; ++i) {
    if (h_b_cpu[i] != h_b_gpu[i]) {
      std::cout << "Verification failed at index " << i << "CPU: " << (int)h_b_cpu[i] << ", GPU: " << (int)h_b_gpu[i] << std::endl;
      flag = false;
      break;
    }
  }

  if (flag) {
    std::cout << "Verification passed" << std::endl;
  }

  // constant timing
  CALI_MARK_BEGIN("kernel_constant");
  cudaEventRecord(start, 0);

  // TODO: Launch filter kernel
  filter_constant<<<grid, block>>>(d_a, d_b, nx, ny);

  // constant timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CALI_MARK_END("kernel_constant");
  cudaEventElapsedTime(&kernel_time_constant, start, stop);
  std::cout << "Constant time: " << kernel_time_constant / 1000.0f << "s\n";

  // TODO: Copy result back to host
  cudaMemcpy(h_b_gpu.data(), d_b, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);

  // TODO: Check result
  flag = true;
  for (int i = 0; i < size; ++i) {
    if (h_b_cpu[i] != h_b_gpu[i]) {
      std::cout << "Verification failed at index " << i << "CPU: " << (int)h_b_cpu[i] << ", Constant: " << (int)h_b_gpu[i] << std::endl;
      flag = false;
      break;
    }
  }

  if (flag) {
    std::cout << "Verification passed" << std::endl;
  }

  // shared timing
  CALI_MARK_BEGIN("kernel_constant");
  cudaEventRecord(start, 0);

  // TODO: Launch filter kernel
  filter_shared<<<grid, block>>>(d_a, d_b, nx, ny);

  // constant timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CALI_MARK_END("kernel_shared");
  cudaEventElapsedTime(&kernel_time_shared, start, stop);
  std::cout << "Shared time: " << kernel_time_shared / 1000.0f << "s\n";

  // TODO: Copy result back to host
  cudaMemcpy(h_b_gpu.data(), d_b, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);

  // TODO: Check result
  flag = true;
  for (int i = 0; i < size; ++i) {
    if (h_b_cpu[i] != h_b_gpu[i]) {
      std::cout << "Verification failed at index " << i << "CPU: " << (int)h_b_cpu[i] << ", Shared: " << (int)h_b_gpu[i] << std::endl;
      flag = false;
      break;
    }
  }

  if (flag) {
    std::cout << "Verification passed" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float numerator = 9 * sizeof(unsigned char) + 1 * sizeof(unsigned char) + 9 * sizeof(float);
  float effective_bandwidth_global = numerator / (kernel_time_global * 1e6);

  numerator = 9 * sizeof(unsigned char) + 1 * sizeof(unsigned char);
  float effective_bandwidth_constant = numerator / (kernel_time_constant * 1e6);
  numerator = 8 * sizeof(unsigned char);
  float effective_bandwidth_shared = numerator / (kernel_time_shared * 1e6);

  adiak::init(NULL);
  adiak::value("image_size", nx);
  adiak::value("effective_bandwidth_global", effective_bandwidth_global);
  adiak::value("effective_bandwidth_constant", effective_bandwidth_constant);
  adiak::value("effective_bandwidth_shared", effective_bandwidth_shared);
  adiak::value("kernel_time_global", kernel_time_global);
  adiak::value("kernel_time_constant", kernel_time_constant);
  adiak::value("kernel_time_shared", kernel_time_shared);
  

  // Flush Caliper output
  mgr.stop();
  mgr.flush();

  std::cout << "End" << "\n";
  return 0;
}
