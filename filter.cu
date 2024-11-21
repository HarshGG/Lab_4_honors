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
__global__ void filter_shared(unsigned char *a, unsigned char *b, int nx,
                              int ny) {}

/**
 * @brief Implemenation using global memory for image and constant memory for
 * filter
 * @param a input image
 * @param b output image
 * @param nx image width
 * @param nx image length
 */
__global__ void filter_constant(unsigned char *a, unsigned char *b, int nx,
                                int ny) {}

/**
 * @brief Implemenation using global memory fir filter and image
 * @param a input image
 * @param b output image
 * @param c filter
 * @param nx image width
 * @param nx image length
 */
__constant__ float fc[9];
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

int main() {
  CALI_CXX_MARK_FUNCTION;

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  // Image size
  int nx = 1024;
  int ny = 1024;
  int size = nx * ny;

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

  // GPU timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // TODO: Launch filter kernel
  filter_global<<<grid, block>>>(d_a, d_b, nx, ny, d_c);

  // GPU timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float gpu_elapsed = 0;
  cudaEventElapsedTime(&gpu_elapsed, start, stop);
  std::cout << "GPU time: " << gpu_elapsed / 1000.0f << "s\n";

  // TODO: Copy result back to host
  cudaMemcpy(h_b_gpu.data(), d_b, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);

  // TODO: Check result
  bool flag = false;
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

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Flush Caliper output
  mgr.stop();
  mgr.flush();

  std::cout << "End" << "\n";
  return 0;
}
