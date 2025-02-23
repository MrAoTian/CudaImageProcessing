#pragma once
#include "cuda_utils.h"


/* Integral image with extra buffer */
void hIntegral(unsigned char* src, int* integral, int* buff, int width, int height, int sstride, int dstride);

/* Integral image without extra buffer, but the size of result must be aligned by 4 */
void hAligned4Integral(unsigned char* src, int* integral, int swidth, int sheight, int sstride, int dwidth, int dheight);

/* Initialize random seed */
void hInitRand(curandState* rand_state, int seed, int num);

/* Fill data with random data */
void hRandFill(unsigned char* data, curandState* rand_state, int width, int height, int stride);

/* Compute the max absolute difference of two image */
void hCmpMaxAbsDiff(int* nppi_res, int* cuda_res, int width, int height);
