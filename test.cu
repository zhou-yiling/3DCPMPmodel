#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cstring>

using namespace std;

#define DX 20
#define DY 20

__global__  void test(int *a)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    a[I] += 1;
}

dim3 block(20, 1, 1);
dim3 grid(20, 1, 1);

int main()
{
	cudaSetDevice(0);
    int * a;
    cudaMallocManaged(&a, sizeof(int) * DX * DY);

    cudaGetLastError();
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
    }

    test<<< grid, block >>>(a);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    for(int i = 0; i < DX * DY; i++)
    {
        cout << a[i] << " ";
    }

    // int b[DX * DY];
    // cudaMemcpy(b, a, sizeof(int) * DX * DY, cudaMemcpyDeviceToHost);
    // cout << endl;

    int b[DX * DY];
    for(int i = 0; i < DX * DY; i++)
    {
        b[i] = i;
    }
    FILE *File = fopen("test.txt", "a+");
    fwrite(b, sizeof(int), DX * DY, File);

    rewind(File);

    cout << " fread " << endl;
    fread(a, sizeof(int), DX * DY, File);
    cudaDeviceSynchronize();
    test<<< grid, block >>>(a);
    cudaDeviceSynchronize();
    for(int i = 0; i < DX * DY; i++)
    {
        cout << a[i] << " ";
    }
}