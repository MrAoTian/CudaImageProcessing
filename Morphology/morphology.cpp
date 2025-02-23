#include "morphology.h"
#include "image_process.h"
#include <opencv2/highgui.hpp>



CudaMorphology::CudaMorphology(/* args */)
{
}


CudaMorphology::~CudaMorphology()
{
}


void CudaMorphology::init(int width_, int height_)
{
    width = width_;
    height = height_;
    this->allocMemory();
}


void CudaMorphology::run(unsigned char* src, unsigned char* dst, int ksz, int mode)
{
    const int radius = ksz >> 1;
    if (radius <= 20)
    {
        hMorphology(src, dst, hset1, radius, mode, width, height, wstride);
    }
    else
    {
        // Compute min/max set
        hCalcMset(src, hset1, hset2, ksz, mode, width, height, wstride);
        if (false)
        {
            cv::Mat result(height, width, CV_8UC1);
            CHECK(cudaMemcpy2D(result.data, width * sizeof(unsigned char), hset1, wstride * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost));
            cv::imwrite("../data/hset1.png", result);
            CHECK(cudaMemcpy2D(result.data, width * sizeof(unsigned char), hset2, wstride * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost));
            cv::imwrite("../data/hset2.png", result);
        }

        // Compute row min/max
        hCalcRowM(hset1, hset2, hmop, ksz, mode, width, height, wstride);

        // Transpose
        hTranspose(hmop, vmop, width, height, wstride, hstride);

        // Compute min/max set
        hCalcMset(vmop, vset1, vset2, ksz, mode, height, width, hstride);

        // Compute row min/max
        hCalcRowM(vset1, vset2, vmop, ksz, mode, height, width, hstride);

        // Transpose
        hTranspose(vmop, dst, height, width, hstride, wstride);
    }
}


void CudaMorphology::allocMemory()
{
    this->freeMemory();

    size_t wspitch = width * sizeof(unsigned char);
    size_t wdpitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&hset1), &wdpitch, wspitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&hset2), &wdpitch, wspitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&hmop), &wdpitch, wspitch, height));
    wstride = wdpitch / sizeof(unsigned char);

    size_t hspitch = height * sizeof(unsigned char);
    size_t hdpitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&vset1), &hdpitch, hspitch, width));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&vset2), &hdpitch, hspitch, width));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&vmop), &hdpitch, hspitch, width));
    hstride = hdpitch / sizeof(unsigned char);
}


void CudaMorphology::freeMemory()
{
    CUDA_SAFE_FREE(hset1);
    CUDA_SAFE_FREE(hset2);
    CUDA_SAFE_FREE(hmop);
    CUDA_SAFE_FREE(vset1);
    CUDA_SAFE_FREE(vset2);
    CUDA_SAFE_FREE(vmop);
}


