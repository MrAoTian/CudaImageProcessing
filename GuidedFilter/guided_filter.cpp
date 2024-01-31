#include "guided_filter.h"
#include "guided_filter_d.h"
#include <opencv2/highgui.hpp>



GuidedFilter::GuidedFilter()
{
}


GuidedFilter::~GuidedFilter()
{
    this->freeMemory();
}


void GuidedFilter::init(const int _width, const int _height, const int _guided_channel, const int _src_channel)
{
    swhcs.w = _width; swhcs.x = _height; swhcs.y = _src_channel;
    gwhcs.w = _width; gwhcs.x = _height; gwhcs.y = _guided_channel;
    iwhcs.w = _width + 1; iwhcs.x = _height + 1; iwhcs.y = _src_channel;

    this->allocMemory();
}


void GuidedFilter::run(float* guidiance, float* src, float* dst, const int r, const float eps)
{
    // Box filter
    hBoxFilter(src, pm, buffer, swhcs, iwhcs, r);
    if (false)
    {
        cv::Mat blur_src(swhcs.x, swhcs.w, CV_32FC(swhcs.y));
        const size_t dpitch = swhcs.w * swhcs.y * sizeof(float);
        const size_t spitch = swhcs.z * sizeof(float);
        CHECK(cudaMemcpy2D(blur_src.data, dpitch, pm, spitch, dpitch, swhcs.x, cudaMemcpyDeviceToHost));
        CHECK(cudaDeviceSynchronize());
        blur_src.convertTo(blur_src, CV_8U, 255.0);
        bool ret = cv::imwrite("../data/blur_src.png", blur_src);        
        std::cout << (ret ? "Success" : "Failed") << " to apply box filter on src" << std::endl;
    }   

    hBoxFilter(guidiance, im, buffer, gwhcs, iwhcs, r);
    if (false)
    {
        cv::Mat blur_gui(gwhcs.x, gwhcs.w, CV_32FC(gwhcs.y));
        const size_t dpitch = gwhcs.w * gwhcs.y * sizeof(float);
        const size_t spitch = gwhcs.z * sizeof(float);
        CHECK(cudaMemcpy2D(blur_gui.data, dpitch, im, spitch, dpitch, gwhcs.x, cudaMemcpyDeviceToHost));
        CHECK(cudaDeviceSynchronize());
        blur_gui.convertTo(blur_gui, CV_8U, 255.0);
        bool ret = cv::imwrite("../data/blur_gui.png", blur_gui);
        std::cout << (ret ? "Success" : "Failed") << " to apply box filter on guidiance" << std::endl;
    }    

    hMultiply(src, guidiance, ipm, swhcs, gwhcs);
    hMultiply(guidiance, guidiance, iim, gwhcs, gwhcs);
    hBoxFilter(ipm, ipm, buffer, swhcs, iwhcs, r);
    hBoxFilter(iim, iim, buffer, gwhcs, iwhcs, r);
    hCalcA(a, pm, im, ipm, iim, swhcs, gwhcs, eps);
    hCalcB(b, a, pm, im, swhcs, gwhcs);
    hBoxFilter(a, am, buffer, swhcs, iwhcs, r);
    hBoxFilter(b, bm, buffer, swhcs, iwhcs, r);
    hLinearTransform(guidiance, dst, am, bm, gwhcs, swhcs);
}



void GuidedFilter::allocMemory()
{
    this->freeMemory();

    size_t spitch_real = swhcs.w * swhcs.y * sizeof(float);
    size_t gpitch_real = gwhcs.w * gwhcs.y * sizeof(float);
    size_t spitch = 0;
    size_t gpitch = 0;
    size_t ipitch = 0;

    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&pm), &spitch, spitch_real, swhcs.x));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&im), &gpitch, gpitch_real, gwhcs.x));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&ipm), &spitch, spitch_real, swhcs.x));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&iim), &gpitch, gpitch_real, gwhcs.x));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&a), &spitch, spitch_real, swhcs.x));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&b), &spitch, spitch_real, swhcs.x));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&am), &spitch, spitch_real, swhcs.x));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&bm), &spitch, spitch_real, swhcs.x));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&buffer), &ipitch, iwhcs.w * iwhcs.y * sizeof(float), iwhcs.x));

    swhcs.z = spitch / sizeof(float);
    gwhcs.z = gpitch / sizeof(float);
    iwhcs.z = ipitch / sizeof(float);
}


void GuidedFilter::freeMemory()
{
    CUDA_SAFE_FREE(pm);
    CUDA_SAFE_FREE(im);
    CUDA_SAFE_FREE(ipm);
    CUDA_SAFE_FREE(iim);
    CUDA_SAFE_FREE(buffer);
    CUDA_SAFE_FREE(a);
    CUDA_SAFE_FREE(b);
    CUDA_SAFE_FREE(am);
    CUDA_SAFE_FREE(bm);
}

