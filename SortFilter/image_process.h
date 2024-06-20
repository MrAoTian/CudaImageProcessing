#pragma once
#include "cuda_utils.h"


/* Compute min/max set */
void hCalcMset(unsigned char* src, unsigned char* mset1, unsigned char* mset2, int ksz, int mode, int width, int height, int stride);

/* Compute row min/max */
void hCalcRowM(unsigned char* mset1, unsigned char* mset2, unsigned char* dst, int ksz, int mode, int width, int height, int stride);

/* Transpose */
void hTranspose(unsigned char* src, unsigned char* dst, int width, int height, int sstride, int dstride);


