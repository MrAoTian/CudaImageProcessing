#include "image_process.h"

#define NX 32
#define NY 16
#define X1 256





__device__ unsigned char minUchar(unsigned char a, unsigned char b)
{
    return min(a, b);
}


__device__ unsigned char maxUchar(unsigned char a, unsigned char b)
{
    return max(a, b);
}


typedef unsigned char(*MorphFunction)(unsigned char, unsigned char);


__device__ MorphFunction MorphFuncArray[] = { minUchar, maxUchar };




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


__global__ void gCalRowMin(unsigned char* mset1, unsigned char* mset2, unsigned char* dst, int radius, int ksz, int last_kernel, int width, int height, int stride)
{
    int ix = blockIdx.x * NX + threadIdx.x;
    int iy = blockIdx.y * NY + threadIdx.y;
    if (ix >= width || iy >= height)
    {
        return;
    }

    
    int ytop = iy - radius;
    int ybot = iy + radius;
    if (ytop < 0)
    {
        dst[iy * stride + ix] = mset1[ybot * stride + ix];    
    }
    else if (iy / ksz == last_kernel && iy % ksz >= radius)
    {       
        dst[iy * stride + ix] = mset2[ytop * stride + ix];
    }
    else
    {
        dst[iy * stride + ix] = min(mset1[min(height - 1, ybot) * stride + ix], mset2[ytop * stride + ix]);
    }
}


__global__ void gCalRowMax(unsigned char* mset1, unsigned char* mset2, unsigned char* dst, int radius, int ksz, int last_kernel, int width, int height, int stride)
{
    int ix = blockIdx.x * NX + threadIdx.x;
    int iy = blockIdx.y * NY + threadIdx.y;
    if (ix >= width || iy >= height)
    {
        return;
    }
    
    int ytop = iy - radius;
    int ybot = iy + radius;
    if (ytop < 0)
    {
        dst[iy * stride + ix] = mset1[ybot * stride + ix];    
    }
    else if (iy / ksz == last_kernel && iy % ksz >= radius)
    {       
        dst[iy * stride + ix] = mset2[ytop * stride + ix];
    }
    else
    {
        dst[iy * stride + ix] = max(mset1[min(height - 1, ybot) * stride + ix], mset2[ytop * stride + ix]);
    }
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


template <int RADIUS, int KX>
__global__ void gMorphSplit(unsigned char* src, unsigned char* dst, int mode, int width, int height, int stride)
{
    __shared__ unsigned char pmem[RADIUS * 2][KX + RADIUS * 2];
    __shared__ unsigned char smem[RADIUS * 4][KX];

    const int R2 = RADIUS << 1;
    const int R4 = RADIUS << 2;
    const int R8 = RADIUS << 3;
    const int tix = threadIdx.x;
    const int tiy = threadIdx.y;
    const int ix = blockIdx.x * KX + tix;
    const int iy = blockIdx.y * R8 + tiy;
    int xs[R2 + 1];
    xs[RADIUS] = min(ix, width - 1);
    for (int i = 1; i <= RADIUS; ++i)
    {
        xs[RADIUS - i] = max(0, min(ix - i, width - 1));
        xs[RADIUS + i] = min(ix + i, width - 1);
    }
    const int six = tix + RADIUS;
    const int pix = tix + R2;
    const bool left_cond = tix < RADIUS;
    const bool righ_cond = six >= KX || ix + RADIUS >= width;
    MorphFunction kernelFunc = MorphFuncArray[mode];

    // First radius range - Load data to shared memory
    int offset = max(0, min(iy - RADIUS, height - 1)) * stride;
    pmem[tiy][six] = src[offset + xs[RADIUS]];
    if (left_cond) pmem[tiy][tix] = src[offset + xs[0]];
    if (righ_cond) pmem[tiy][pix] = src[offset + xs[R2]];
    __syncthreads();

    // Second ~ 4th radius range
#pragma unroll
    for (int range = RADIUS; range < R4; range += RADIUS)
    {
        // Load data to shared memory
        int piy = (tiy + range) % R2;
        offset = min(iy + range - RADIUS, height - 1) * stride;
        pmem[piy][six] = src[offset + xs[RADIUS]];
        if (left_cond) pmem[piy][tix] = src[offset + xs[0]];
        if (righ_cond) pmem[piy][pix] = src[offset + xs[R2]];

        // Reduce row for last-radius 
        int siy = tiy + range - RADIUS;
        piy = siy % R2;
        unsigned char res = pmem[piy][six];
        for (int i = 1; i <= RADIUS; ++i)
        {
            res = kernelFunc(res, kernelFunc(pmem[piy][six - i], pmem[piy][six + i]));
        }
        smem[siy][tix] = res;
        __syncthreads();
    }

    // 4th -> radius range 
#pragma unroll
    for (int range = RADIUS; range < R8 - RADIUS; range += RADIUS)
    {
        // Load data to shared memory
        int siy = tiy + range;
        int piy = (siy + RADIUS) % R2;  // (siy + R3) % R2;
        offset = min(iy + range + R2, height - 1) * stride;
        pmem[piy][six] = src[offset + xs[RADIUS]];
        if (left_cond) pmem[piy][tix] = src[offset + xs[0]];
        if (righ_cond) pmem[piy][pix] = src[offset + xs[R2]];

        // Reduce col and save result to dst
        int ny = iy + range - RADIUS;
        if (ix < width && ny < height)
        {
            unsigned char res = smem[siy % R4][tix];
            for (int i = 1; i <= RADIUS; ++i)
            {
                res = kernelFunc(res, kernelFunc(smem[(siy - i) % R4][tix], smem[(siy + i) % R4][tix]));
            }
            dst[ny * stride + ix] = res;
        }    

        // Reduce row for last-radius
        siy += R2;
        piy = siy % R2;
        unsigned char res = pmem[piy][six];
        for (int i = 1; i <= RADIUS; ++i)
        {
            res = kernelFunc(res, kernelFunc(pmem[piy][six - i], pmem[piy][six + i]));
        }
        smem[siy % R4][tix] = res;
    
        __syncthreads();
    }

    // Last 2 range
    if (ix < width)
    {
        // Reduce row for last-radius
        int siy = tiy + R8 + RADIUS;
        int piy = siy % R2;
        unsigned char res = pmem[piy][six];
        for (int i = 1; i <= RADIUS; ++i)
        {
            res = kernelFunc(res, kernelFunc(pmem[piy][six - i], pmem[piy][six + i]));
        }
        smem[siy % R4][tix] = res;

        int ny = iy + R8 - R2;
        if (ny >= height) return;        
        siy = tiy + R8 - RADIUS;
        res = smem[siy % R4][tix];
        for (int i = 1; i <= RADIUS; ++i)
        {
            res = kernelFunc(res, kernelFunc(smem[(siy - i) % R4][tix], smem[(siy + i) % R4][tix]));
        }
        dst[ny * stride + ix] = res;

        ny += RADIUS;
        if (ny >= height) return;        
        siy += RADIUS;
        res = smem[siy % R4][tix];
        for (int i = 1; i <= RADIUS; ++i)
        {
            res = kernelFunc(res, kernelFunc(smem[(siy - i) % R4][tix], smem[(siy + i) % R4][tix]));
        }
        dst[ny * stride + ix] = res;
    }
}


__global__ void gMorphRow(unsigned char* src, unsigned char* dst, int radius, int ksz, int mode, int width, int height, int stride)
{
    extern __shared__ unsigned char sdata[];

    int slen = blockDim.x * ksz;
    unsigned char* oppo = sdata;
    unsigned char* vivo = sdata + slen + radius;
    MorphFunction kernelFunc = MorphFuncArray[mode];

    int s1 = threadIdx.x * ksz;
    int s2 = s1 + radius;
    int ix = blockIdx.x * slen + s1;
    int ystart = blockIdx.y * stride;
    unsigned char v1, v2;
    v1 = src[ystart + min(width - 1, ix)];
    vivo[s1] = v1;
    for (int i = 1; i < ksz; i++)
    {
        v2 = src[ystart + min(width - 1, ix + i)];
        oppo[s2 + i] = v2;
        v1 = kernelFunc(v1, v2);
        vivo[s1 + i] = v1;
    }
    oppo[s2] = v1;
    for (int i = ksz - 2; i > 0; i--)
    {
        v2 = kernelFunc(oppo[s2 + i], v2);
        oppo[s2 + i] = v2;
    }
    if (threadIdx.x == 0)
    {
        v1 = src[ystart + max(0, ix - 1)];
        oppo[radius - 1] = v1;
        for (int i = 2; i <= radius; i++)
        {
            v1 = kernelFunc(v1, src[ystart + max(0, ix - i)]);
            oppo[radius - i] = v1;
        }
    }
    if (threadIdx.x == blockDim.x - 1)
    {
        v1 = src[ystart + min(width - 1, ix + ksz)];
        vivo[slen] = v1;
        for (int i = 1; i < radius; i++)
        {
            v1 = kernelFunc(v1, src[ystart + min(width - 1, ix + ksz + i)]);
            vivo[slen + i] = v1;
        }
    }
    __syncthreads();
    
    for (int i = 0; i < ksz; i++)
    {
        if (ix + i >= width)
        {
            return;
        }
        dst[ystart + ix + i] = kernelFunc(oppo[s1 + i], vivo[s2 + i]);
    }
}


__global__ void gMorphCol(unsigned char* src, unsigned char* dst, int radius, int ksz, int mode, int width, int height, int stride)
{
    extern __shared__ unsigned char sdata[];

    int slen = blockDim.y * ksz;
    unsigned char* oppo = sdata;
    unsigned char* vivo = sdata + slen + radius;
    MorphFunction kernelFunc = MorphFuncArray[mode];

    int s1 = threadIdx.y * ksz;
    int s2 = s1 + radius;
    int iy = blockIdx.y * slen + s1;
    int ix = blockIdx.x;
    unsigned char v1, v2;
    v1 = src[min(height - 1, iy) * stride + ix];
    vivo[s1] = v1;
    for (int i = 1; i < ksz; i++)
    {
        v2 = src[min(height - 1, iy + i) * stride + ix];
        oppo[s2 + i] = v2;
        v1 = kernelFunc(v1, v2);
        vivo[s1 + i] = v1;
    }
    oppo[s2] = v1;
    for (int i = ksz - 2; i > 0; i--)
    {
        v2 = kernelFunc(oppo[s2 + i], v2);
        oppo[s2 + i] = v2;
    }
    if (threadIdx.y == 0)
    {
        v1 = src[max(0, iy - 1) * stride + ix];
        oppo[radius - 1] = v1;
        for (int i = 2; i <= radius; i++)
        {
            v1 = kernelFunc(v1, src[max(0, iy - i) * stride + ix]);
            oppo[radius - i] = v1;
        }
    }
    if (threadIdx.y == blockDim.y - 1)
    {
        v1 = src[min(height - 1, iy + ksz) * stride + ix];
        vivo[slen] = v1;
        for (int i = 1; i < radius; i++)
        {
            v1 = kernelFunc(v1, src[min(height - 1, iy + ksz + i) * stride + ix]);
            vivo[slen + i] = v1;
        }
    }
    __syncthreads();
    
    for (int i = 0; i < ksz; i++)
    {
        if (iy + i >= height)
        {
            return;
        }
        dst[(iy + i) * stride + ix] = kernelFunc(oppo[s1 + i], vivo[s2 + i]);
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
    CHECK(cudaDeviceSynchronize());
}


void hCalcRowM(unsigned char* mset1, unsigned char* mset2, unsigned char* dst, int ksz, int mode, int width, int height, int stride)
{
    int radius = ksz / 2;
    int last_kernel = height / ksz;
    if (height % ksz == 0)
        last_kernel--;
    dim3 block(NX, NY);
    dim3 grid(iDivUp(width, NX), iDivUp(height, NY));
    if (mode == 0)
        gCalRowMin<<<grid, block>>>(mset1, mset2, dst, radius, ksz, last_kernel, width, height, stride);
    else
        gCalRowMax<<<grid, block>>>(mset1, mset2, dst, radius, ksz, last_kernel, width, height, stride);
    CHECK(cudaDeviceSynchronize());
}


void hTranspose(unsigned char* src, unsigned char* dst, int width, int height, int sstride, int dstride)
{
    dim3 block(NX, NY);
    dim3 grid(iDivUp(width, NX), iDivUp(height, NY));
    gTransposeUnroll4Col<<<grid, block>>>(src, dst, width, height, sstride, dstride);
    CHECK(cudaDeviceSynchronize());
}


void hMorphology(unsigned char* src, unsigned char* dst, unsigned char* buffer, int radius, int mode, int width, int height, int stride)
{
    const int ksize = 2 * radius + 1;
    dim3 block(1, radius);
    dim3 grid(1, iDivUp(height, 8 * radius));
    switch (radius)
    {
    case 1:
        block.x = 128; grid.x = iDivUp(width, block.x);
        gMorphSplit<1, 128><<<grid, block>>>(src, dst, mode, width, height, stride);
        break;
    case 2:
        block.x = 128; grid.x = iDivUp(width, block.x);
        gMorphSplit<2, 128><<<grid, block>>>(src, dst, mode, width, height, stride);
        break;
    case 3:
        block.x = 128; grid.x = iDivUp(width, block.x);
        gMorphSplit<3, 128><<<grid, block>>>(src, dst, mode, width, height, stride);
        break;
    case 4:
        block.x = 128; grid.x = iDivUp(width, block.x);
        gMorphSplit<4, 128><<<grid, block>>>(src, dst, mode, width, height, stride);
        break;
    case 5:
        block.x = 64; grid.x = iDivUp(width, block.x);
        gMorphSplit<5, 64><<<grid, block>>>(src, dst, mode, width, height, stride);
        break;
    case 6:
        block.x = 64; grid.x = iDivUp(width, block.x);
        gMorphSplit<6, 64><<<grid, block>>>(src, dst, mode, width, height, stride);
        break;
    // case 7:
    //     block.x = 64; grid.x = iDivUp(width, block.x);
    //     gMorphSplit<7, 64><<<grid, block>>>(src, dst, mode, width, height, stride);
    //     break;
    default:
    {
        dim3 row_block(256, 1);
        dim3 row_grid(iDivUp(width, row_block.x * ksize), height);
        dim3 col_block(1, 256);
        dim3 col_grid(width, iDivUp(height, col_block.y * ksize));
        size_t row_bytes = (256 * ksize + radius) * 2 * sizeof(unsigned char);
        size_t col_bytes = (256 * ksize + radius) * 2 * sizeof(unsigned char);
        gMorphRow<<<row_grid, row_block, row_bytes>>>(src, buffer, radius, ksize, mode, width, height, stride);
        CHECK(cudaDeviceSynchronize());
        gMorphCol<<<col_grid, col_block, col_bytes>>>(buffer, dst, radius, ksize, mode, width, height, stride);
        CHECK(cudaDeviceSynchronize());
        break;
    }
    }
    CHECK(cudaDeviceSynchronize());
}



