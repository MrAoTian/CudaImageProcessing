#pragma once
#include "cuda_utils.h"


class GuidedFilter
{
public:
    GuidedFilter();
    ~GuidedFilter();

    /*  
    Initialization - Allocate memory for temp variables.
    @param:
        _width: Width of images.
        _height: Height of images.
        _guided_channel: Channel of guidiance image. 1 or 3. 
        _src_channel: Channel of source and destination image. 1 or 3.
    */
    void init(const int _width, const int _height, const int _guided_channel = 3, const int _src_channel = 3);

    /*
    Run filter.
    @param:
        guidiance: Guidiance image
        src: Source image
        dst: Destination image
        r: Size of mean-filter
        eps: Regularization parameter
    */
    void run(float* guidiance, float* src, float* dst, const int r, const float eps);

private:
    // Size
    int4 swhcs; // Size of source image (width, height, channel, stride)
    int4 gwhcs; // Size of guided image (width, height, channel, stride)
    int4 iwhcs; // Size of integral image (width, height, channel, stride)

    // Buffer
    float* pm = nullptr;
    float* im = nullptr;
    float* ipm = nullptr;
    float* iim = nullptr;
    float* buffer = nullptr;
    float* a = nullptr;
    float* b = nullptr;
    float* am = nullptr;
    float* bm = nullptr;

    /* Allocate memory */
    void allocMemory();

    /* Free memory */
    void freeMemory();

};


