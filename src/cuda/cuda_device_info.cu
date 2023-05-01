#include "cuda/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>

void DumpDeviceInfo(int device_id) {
  cudaDeviceProp prop;
  CUDART_CALL(cudaGetDeviceProperties(&prop, device_id));
  std::cout << "===\nDevice " << device_id << ": " << std::endl;
  std::cout << "Device Name: " << prop.name << std::endl;
  std::cout << "Arch: sm" << prop.major << prop.minor << std::endl;
  std::cout << "Global Memory: " << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GB" << std::endl;
  std::cout << "Clock Rate: " << prop.clockRate / 1000 << " Mhz" << std::endl;
  std::cout << "Num of SM: " << prop.multiProcessorCount << std::endl;
  std::cout << "Cuda Core Per SM: " << _ConvertSMVer2Cores(prop.major, prop.minor) << std::endl;
  std::cout << "Max Blocks Per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
  std::cout << "Max Threads Per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
}

int main() {
  int device_count = 0;
  CUDART_CALL(cudaGetDeviceCount(&device_count));
  std::cout << "Found " << device_count << " devices on this machine" << std::endl;

  for (auto i = 0; i < device_count; ++i) {
    DumpDeviceInfo(i);
  }

  return 0;
}
