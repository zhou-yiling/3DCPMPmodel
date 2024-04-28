
#include <cuda_runtime.h>  // CUDA runtime API
#include <iostream>        // for std::cout and std::endl
#include <iomanip>         // for std::put_time
#include <chrono>          // for std::chrono::system_clock
#include "utility.h"

using namespace std;

void printcudaMemoryInfo() 
{
  size_t free_byte;
  size_t total_byte;
  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
  if (cudaSuccess != cuda_status) {
    std::cout << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
    exit(1);
  }
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  std::cout << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0 << "MB, free = " << free_db / 1024.0 / 1024.0 << "MB, total = " << total_db / 1024.0 / 1024.0 << "MB" << std::endl;
}


void printTimeStamp()
{
	auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
	cout << "Time: " << put_time(localtime(&now), "%Y-%m-%d %X") << endl;
}



