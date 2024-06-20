#include "image_process.h"

#define NX 32
#define NY 16
#define X1 256


__global__ void gCalcMinSet(unsigned char* src, unsigned char* mset1, unsigned char* mset2, int ksz, int width, int height, int stride)
{
    int ix = blockIdx.x * X1 + threadIdx.x;
    if (ix >= width)
    {
        return;
    }

    int ystart = blockIdx.y * ksz;
    int iy = ystart;
    
    // 1
    int yend = min(ystart + ksz, height);
    int idx0 = ystart * stride + ix;
    int curr_idx = idx0;
    int next_idx = 0;
    mset1[curr_idx] = src[curr_idx];
    for (iy = ystart + 1; iy < yend; iy++)
    {
        next_idx = curr_idx + stride;
        mset1[next_idx] = min(src[next_idx], mset1[curr_idx]);
        curr_idx = next_idx;
    }

    // 2
    mset2[idx0] = mset1[curr_idx];
    mset2[curr_idx] = src[curr_idx];
    for (iy = yend - 2; iy > ystart; iy--)
    {
        next_idx = curr_idx - stride;
        mset2[next_idx] = min(src[next_idx], mset2[curr_idx]);
        curr_idx = next_idx;
    }
}


__global__ void gCalcMaxSet(unsigned char* src, unsigned char* mset1, unsigned char* mset2, int ksz, int width, int height, int stride)
{
    int ix = blockIdx.x * X1 + threadIdx.x;
    if (ix >= width)
    {
        return;
    }

    int ystart = blockIdx.y * ksz;
    int iy = ystart;
    
    // 1
    int yend = min(ystart + ksz, height);
    int idx0 = ystart * stride + ix;
    int curr_idx = idx0;
    int next_idx = 0;
    mset1[curr_idx] = src[curr_idx];
    for (iy = ystart + 1; iy < yend; iy++)
    {
        next_idx = curr_idx + stride;
        mset1[next_idx] = max(src[next_idx], mset1[curr_idx]);
        curr_idx = next_idx;
    }

    // 2
    mset2[idx0] = mset1[curr_idx];
    mset2[curr_idx] = src[curr_idx];
    for (iy = yend - 2; iy > ystart; iy--)
    {
        next_idx = curr_idx - stride;
        mset2[next_idx] = max(src[next_idx], mset2[curr_idx]);
        curr_idx = next_idx;
    }
}


__global__ void gCalRowMin(unsigned char* mset1, unsigned char* mset2, unsigned char* dst, int radius, int width, int height, int stride)
{
    int ix = blockIdx.x * NX + threadIdx.x;
    int iy = blockIdx.y * NY + threadIdx.y;
    if (ix >= width || iy >= height)
    {
        return;
    }

    int ytop = max(iy - radius, 0);
    int ybot = min(iy + radius, height - 1);
    dst[iy * stride + ix] = min(mset1[ybot * stride + ix], mset2[ytop * stride + ix]);
}


__global__ void gCalRowMax(unsigned char* mset1, unsigned char* mset2, unsigned char* dst, int radius, int width, int height, int stride)
{
    int ix = blockIdx.x * NX + threadIdx.x;
    int iy = blockIdx.y * NY + threadIdx.y;
    if (ix >= width || iy >= height)
    {
        return;
    }
    int ytop = max(iy - radius, 0);
    int ybot = min(iy + radius, height - 1);
    dst[iy * stride + ix] = max(mset1[ybot * stride + ix], mset2[ytop * stride + ix]);
}


__global__ void gTransposeUnroll4Col(unsigned char* src, unsigned char* dst, int width, int height, int sstride, int dstride)
{
    unsigned int ix_ = blockIdx.x * NX * 4 + threadIdx.x;
    unsigned int iy = blockIdx.y * NY + threadIdx.y;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        unsigned int ix = ix_ + i * NX;
        if (iy < height && ix < width)
        {
            dst[ix * dstride + iy] = src[iy * sstride + ix];
        }        
    }
}





void hCalcMset(unsigned char* src, unsigned char* mset1, unsigned char* mset2, int ksz, int mode, int width, int height, int stride)
{
    dim3 block(X1, 1);
    dim3 grid(iDivUp(width, X1), iDivUp(height, ksz));
    if (mode == 0)
        gCalcMinSet<<<grid, block>>>(src, mset1, mset2, ksz, width, height, stride);
    else
        gCalcMaxSet<<<grid, block>>>(src, mset1, mset2, ksz, width, height, stride);
}


void hCalcRowM(unsigned char* mset1, unsigned char* mset2, unsigned char* dst, int ksz, int mode, int width, int height, int stride)
{
    int radius = ksz / 2;
    dim3 block(NX, NY);
    dim3 grid(iDivUp(width, NX), iDivUp(height, NY));
    if (mode == 0)
        gCalRowMin<<<grid, block>>>(mset1, mset2, dst, radius, width, height, stride);
    else
        gCalRowMax<<<grid, block>>>(mset1, mset2, dst, radius, width, height, stride);
}


void hTranspose(unsigned char* src, unsigned char* dst, int width, int height, int sstride, int dstride)
{
    dim3 block(NX, NY);
    dim3 grid(iDivUp(width, NX), iDivUp(height, NY));
    gTransposeUnroll4Col<<<grid, block>>>(src, dst, width, height, sstride, dstride);
}

