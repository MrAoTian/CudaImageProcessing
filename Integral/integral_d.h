#pragma once
#include "cuda_utils.h"

void hIntegral(unsigned char* src, int* integral, int* buff, int width, int height, int sstride, int dstride);
void hInitRand(curandState* rand_state, int seed, int num);
void hRandFill(unsigned char* data, curandState* rand_state, int width, int height, int stride);
void hCmpMaxAbsDiff(int* nppi_res, int* cuda_res, int width, int height);
