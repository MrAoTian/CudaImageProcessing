#pragma once
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(err)		__check(err, __FILE__, __LINE__)
#define CheckMsg(msg)	__checkMsg(msg, __FILE__, __LINE__)
#define CUDA_SAFE_FREE(a) if (a != nullptr) CHECK(cudaFree(a))


/* Check cuda runtime api, and print error. */
inline void __check(cudaError err, const char* file, const int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CHECK() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
}


/* Check cuda runtime api, and print error with Message. */
inline void __checkMsg(const char* msg, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CheckMsg() CUDA error: %s in file <%s>, line %i : %s.\n", msg, file, line, cudaGetErrorString(err));
		exit(-1);
	}
}


/* Initialize device properties */
inline bool initDevice(int dev)
{
	int device_count = 0;
	CHECK(cudaGetDeviceCount(&device_count));
	if (device_count == 0)
	{
		fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
		return false;
	}
	dev = std::max<int>(0, std::min<int>(dev, device_count - 1));
	cudaDeviceProp device_prop;
	CHECK(cudaGetDeviceProperties(&device_prop, dev));
	if (device_prop.major < 1)
	{
		fprintf(stderr, "error: device does not support CUDA.\n");
		return false;
	}
	CHECK(cudaSetDevice(dev));

	int driver_version = 0;
	int runtime_version = 0;
	CHECK(cudaDriverGetVersion(&driver_version));
	CHECK(cudaRuntimeGetVersion(&runtime_version));
	fprintf(stderr, "Using Device %d: %s, CUDA Driver Version: %d.%d, Runtime Version: %d.%d\n", dev, device_prop.name,
		driver_version / 1000, driver_version % 1000, runtime_version / 1000, runtime_version % 1000);
	return true;
}


/* Get CPU timer(microsecond level). */
inline long long cpuTimer()
{
	std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds>(
		std::chrono::system_clock::now().time_since_epoch()
		);
	return ms.count();
}


/* GPU timer(microsecond level). */
class GpuTimer 
{
public:
	GpuTimer(cudaStream_t stream_ = 0) : stream(stream_)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	float read() 
	{
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}
private:
	cudaEvent_t start, stop;
	cudaStream_t stream;
};


/* Shuffle down, copy data from threadIdx to threadIdx - delta */
template <class T> 
__device__ __inline__ T shiftDown(T var, unsigned int delta, int width = 32) 
{
#if (CUDART_VERSION >= 9000)
	return __shfl_down_sync(0xffffffff, var, delta, width);		// Not defined but can be compiled
#else
	return __shfl_down(var, delta, width);
#endif
}


/* Shuffle up, copy data from threadIdx to threadIdx + delta */
template <class T>
__device__ __inline__ T shiftUp(T var, unsigned int delta, int width = 32) {
#if (CUDART_VERSION >= 9000)
	return __shfl_up_sync(0xffffffff, var, delta, width);
#else
	return __shfl_up(var, delta, width);
#endif
}


/* Shuffle or broadcast, set all data of current thread warp to data at thread NO.lane */
template <class T>
__device__ __inline__ T shuffle(T var, unsigned int lane, int width = 32) {
#if (CUDART_VERSION >= 9000)
	return __shfl_sync(0xffffffff, var, lane, width);
#else
	return __shfl(var, lane, width);
#endif
}


/* Align up */
inline int iAlignUp(const int a, const int b)
{
	return (a % b != 0) ? (a - a % b + b) : a;
}


/* Divide up */
inline int iDivUp(const int a, const int b)
{
    return (a + b - 1) / b;
}


