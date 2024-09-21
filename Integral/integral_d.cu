#include "integral_d.h"


#define NTX 2   // Number of tile on X direction per block
#define NTY 2   // Number of tile on X direction per block
#define NTT 16   // Number of threads of a tile: 4, 8, 16
#define MTT 15   // Max thread index in tile
#define NTT2 32   // Number of threads of a tile
#define NTT3 48   // Number of threads of a tile
#define NTT4 64   // Number of threads of a tile


__global__ void gIntegralInTile(unsigned char* src, int* dst, int width, int height, int sstride, int dstride)
{
    __shared__ int sdata[NTY * NTT][NTX * NTT];
    unsigned int tidx_block = threadIdx.x;                          // The thread index in block
    unsigned int idx_tile = tidx_block / NTT;                       // The index of tile in block
    unsigned int iy0_block = idx_tile / NTX * NTT;                  // The start x coordinate of current tile opposite to block
    unsigned int ix0_block = (idx_tile & (NTX - 1)) * NTT;                  // The start y coordinate of current tile opposite to block
    unsigned int iy0_image = blockIdx.y * NTY * NTT + iy0_block;    // The start x coordinate of current tile opposite to original point
    unsigned int ix0_image = blockIdx.x * NTX * NTT + ix0_block;    // The start y coordinate of current tile opposite to original point
    unsigned int tidx_tile = tidx_block & MTT;                      // The thread index in tile

    unsigned int iy = iy0_image + tidx_tile;
    unsigned int id_block = iy0_block + tidx_tile;
    unsigned int ix = ix0_image;
    unsigned int irow = iy * sstride;

    // Cumulative by row
    if (iy < height && ix < width)
    {
        sdata[id_block][ix0_block] = src[irow + ix];
        ix++;
#pragma unroll
        for (int i = 1; i < NTT && ix < width; i++, ix++)
        {
            sdata[id_block][ix0_block + i] = sdata[id_block][ix0_block + i - 1] + src[irow + ix];             
        }
    }
    __syncthreads();

    // Cumulative by col
    ix = ix0_image + tidx_tile;
    iy = iy0_image;
    id_block = ix0_block + tidx_tile;    
    if (iy < height && ix < width)
    {
        unsigned int idx = iy * dstride + ix;
        dst[idx] = sdata[iy0_block][id_block];
        iy++;
#pragma unroll
        for (int i = 1; i < NTT && iy < height; i++, iy++)
        {
            sdata[iy0_block + i][id_block] += sdata[iy0_block + i - 1][id_block];
            idx += width;
            dst[idx] = sdata[iy0_block + i][id_block];
        }        
    }
}


__global__ void gIntegralTileLastCol(int* data, int* trs, int width, int height, int stride, int xtiles)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int iy = blockIdx.x;
    unsigned int ystart = iy * stride;

    unsigned int stid = tid << 1;
	unsigned int stidp = stid + 1;
	unsigned int stidq = stid + 2;
    unsigned int snum = blockDim.x << 1;

    unsigned int pack_ix = stid * NTT + MTT;
    unsigned int pack_ixp = pack_ix + NTT;

    sdata[stid] = pack_ix < width ? data[ystart + pack_ix] : 0;
    sdata[stidp] = pack_ixp < width ? data[ystart + pack_ixp] : 0;
    int vlast = sdata[stidp];
        
    int shift = 0;    
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            sdata[(stidq << shift) - 1] += sdata[(stidp << shift) - 1];
        }
        shift++;
    }

    if (tid == 0)
    {
        sdata[snum - 1] = 0;
    }
    
    for (unsigned int d = 1; d < snum; d <<= 1)
    {
        shift--;
        __syncthreads();
        if (tid < d)
        {
            int ai = (stidp << shift) - 1;
            int bi = (stidq << shift) - 1;
            int t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    if (pack_ix >= width) return;
    int didx = ((iy / NTT) * xtiles + stid) * NTT + (iy & MTT);
    trs[didx] = sdata[stidp];
    if (pack_ixp >= width) return;
    trs[didx + NTT] = stidq < snum ? sdata[stidq] : (vlast + sdata[stidp]);
}


__global__ void gIntegralTileLastCol2(int* data, int* trs, int width, int height, int stride, int xtiles)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int iy = blockIdx.x;

    // Pack 4
    unsigned int pack_ix = (tid << 2) * NTT + NTT - 1;
    unsigned int pack_idx = iy * stride + pack_ix;

    unsigned int stid = tid << 1;
	unsigned int stidp = stid + 1;
	unsigned int stidq = stid + 2;
    unsigned int snum = blockDim.x << 1;

    // Copy data to shared memory and apply binary tree parallel cumulation
    int a0 = pack_ix < width ? data[pack_idx] : 0;
    sdata[stid] = (pack_ix + NTT) < width ? a0 + data[pack_idx + NTT] : 0;
    int b0 = (pack_ix + NTT2) < width ? data[pack_idx + NTT2] : 0;
    int b1 = (pack_ix + NTT3) < width ? b0 + data[pack_idx + NTT3] : 0;
    sdata[stidp] = b1;

    unsigned int shift = 0;    
    for (unsigned int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            sdata[(stidq << shift) - 1] += sdata[(stidp << shift) - 1];
        }
        shift++;
    }

    if (tid == 0)
    {
        sdata[snum - 1] = 0;
    }
    
    for (unsigned int d = 1; d < snum; d <<= 1)
    {
        shift--;
        __syncthreads();
        if (tid < d)
        {
            unsigned int ai = (stidp << shift) - 1;
            unsigned int bi = (stidq << shift) - 1;
            int t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    // Unpack results
    if (pack_ix >= width) return;
    int didx = ((iy / NTT) * xtiles + (tid << 2)) * NTT + (iy & MTT);
    trs[didx] = a0 + sdata[stid];
    if (pack_ix + NTT >= width) return;
    trs[didx + NTT] = sdata[stidp];
    if (pack_ix + NTT2 >= width) return;
    trs[didx + NTT2] = b0 + sdata[stidp];
    if (pack_ix + NTT3 >= width) return;
    trs[didx + NTT3] = stidq < snum ? sdata[stidq] : (b1 + sdata[stidp]);
}


__global__ void gIntegralTileLastRow(int* data, int* trs, int width, int height, int stride, int xtiles)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int ix = blockIdx.x;

    int stid = tid << 1;
	int stidp = stid + 1;
	int stidq = stid + 2;
    int snum = blockDim.x << 1;

    // Pack 4
    int OFFSET = NTT * stride;
    int pack_iy = stid * NTT + MTT;
    int pack_iyp = pack_iy + NTT;
    int pack_yoffset = pack_iy * stride;
    int pack_idx = pack_yoffset + ix;

    // Copy data to shared memory and apply binary tree parallel cumulation
    int didx = (stid * xtiles + (ix / NTT - 1)) * NTT + MTT;
    sdata[stid] = pack_iy < height ? (data[pack_idx] + (ix >= NTT ? trs[didx] : 0)) : 0;
    int b = pack_iyp < height ? (data[pack_idx + OFFSET] + (ix >= NTT ? trs[didx + NTT * xtiles] : 0)) : 0;
    sdata[stidp] = b;

    int shift = 0;    
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            sdata[(stidq << shift) - 1] += sdata[(stidp << shift) - 1];
        }
        shift++;
    }

    if (tid == 0)
    {
        sdata[snum - 1] = 0;
    }
    
    for (int d = 1; d < snum; d <<= 1)
    {
        shift--;
        __syncthreads();
        if (tid < d)
        {
            int ai = (stidp << shift) - 1;
            int bi = (stidq << shift) - 1;
            int t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    // Unpack results
    if (pack_iy >= height) return;
    data[pack_idx] = sdata[stidp];
    if (pack_iyp >= height) return;
    data[pack_idx + OFFSET] = stidq < snum ? sdata[stidq] : (b + sdata[stidp]);
}


__global__ void gIntegralTileLastRow2(int* data, int* trs, int width, int height, int stride, int xtiles)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int ix = blockIdx.x;

    // Pack 4
    int OFFSET = NTT * stride;
    int OFFSET2 = NTT2 * stride;
    int OFFSET3 = NTT3 * stride;
    int pack_iy = (tid << 2) * NTT + NTT - 1;
    int pack_yoffset = pack_iy * stride;
    int pack_idx = pack_yoffset + ix;

    int stid = tid << 1;
	int stidp = stid + 1;
	int stidq = stid + 2;
    int snum = blockDim.x << 1;

    int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
    if (ix >= NTT)
    {
        int didx = ((tid << 2) * xtiles + (ix / NTT - 1)) * NTT + MTT;
        if (pack_iy < height)
            a0 = data[pack_idx] + trs[didx];
        if (pack_iy + NTT < height)
            a1 = a0 + data[pack_idx + OFFSET] + trs[didx + NTT * xtiles];
        if (pack_iy + NTT2 < height)
            b0 = data[pack_idx + OFFSET2] + trs[didx + NTT2 * xtiles];
        if (pack_iy + NTT3 < height)
            b1 = b0 + data[pack_idx + OFFSET3] + trs[didx + NTT3 * xtiles];
    }
    else
    {
        if (pack_iy < height)
            a0 = data[pack_idx];
        if (pack_iy + NTT < height)
            a1 = a0 + data[pack_idx + OFFSET];
        if (pack_iy + NTT2 < height)
            b0 = data[pack_idx + OFFSET2];
        if (pack_iy + NTT3 < height)
            b1 = b0 + data[pack_idx + OFFSET3];
    }

    // Copy data to shared memory and apply binary tree parallel cumulation
    sdata[stid] = a1;    
    sdata[stidp] = b1;

    int shift = 0;    
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            sdata[(stidq << shift) - 1] += sdata[(stidp << shift) - 1];
        }
        shift++;
    }

    if (tid == 0)
    {
        sdata[snum - 1] = 0;
    }
    
    for (int d = 1; d < snum; d <<= 1)
    {
        shift--;
        __syncthreads();
        if (tid < d)
        {
            int ai = (stidp << shift) - 1;
            int bi = (stidq << shift) - 1;
            int t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    // Unpack results
    if (pack_iy >= height) return;
    data[pack_idx] = a0 + sdata[stid];
    if (pack_iy + NTT >= height) return;
    data[pack_idx + OFFSET] = sdata[stidp];
    if (pack_iy + NTT2 >= height) return;
    data[pack_idx + OFFSET2] = b0 + sdata[stidp];
    if (pack_iy + NTT3 >= height) return;
    data[pack_idx + OFFSET3] = stidq < snum ? sdata[stidq] : (b1 + sdata[stidp]);
}


__global__ void gIntegralInGlobal(int* data, int* trs, int width, int height, int stride, int xtiles)
{
    __shared__ int vleft[NTY * NTX * MTT];

    unsigned int tidx_block = threadIdx.x;                          // The thread index in block
    unsigned int tidx_tile = tidx_block & MTT;                      // The thread index in tile
    unsigned int idx_tile = tidx_block / NTT;                       // The index of tile in block
    unsigned int tiy_tile = idx_tile / NTX;                         // The y index of tile
    unsigned int tix_tile = idx_tile & (NTX - 1);                   // The x index of tile
    unsigned int tix = blockIdx.x * NTX + tix_tile;
    unsigned int tiy = blockIdx.y * NTY + tiy_tile;
    unsigned int tix0_image = tix * NTT;                            // The x0 of tile opposite to global
    unsigned int tiy0_image = tiy * NTT;                            // The y0 of tile opposite to global
    unsigned int tidx0_image = tiy0_image * stride + tix0_image;
    unsigned int sidx0 = (tiy_tile * NTX + tix_tile) * MTT;

    if (tidx_tile < MTT)
        vleft[sidx0 + tidx_tile] = tix0_image > 0 ? trs[(tiy * xtiles + tix - 1) * NTT + tidx_tile] : 0;
    __syncthreads();

    if (tix0_image + tidx_tile >= width)
        return;
    
    int vtop = tiy0_image > 0 ? data[tidx0_image + tidx_tile - stride] : 0;
#pragma unroll
    for (int i = 0; i < MTT; i++)
    {
        if (tiy0_image >= height)
            return;
        data[tidx0_image + tidx_tile] += vtop + vleft[sidx0 + i];
        tidx0_image += stride;
        tiy0_image++;
    }
}


__global__ void gInitRand(curandState* state, int seed, int num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num)
    {
        curand_init(seed, tid, 0, &state[tid]);
    }    
}


__global__ void gRandFill(unsigned char* data, curandState* rand_state, int width, int height, int stride)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height)
    {
        return;
    }

    int idx = iy * stride + ix;
    data[idx] = curand(&rand_state[idx]) % 256;    
}


__global__ void gCmpMaxAbsDiff(int* nppi_res, int* cuda_res, int width, int height)
{
    __shared__ int sdata[1024];
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int tid = iy * blockDim.x + ix;
    int ystart = 0;
    int pystart = 0;
    sdata[tid] = 0;
    while (iy < height)
    {
        ystart = iy * width;
        pystart = (iy + 1) * (width + 1);
        while (ix < width)
        {
            sdata[tid] = max(sdata[tid], abs(nppi_res[pystart + ix + 1] - cuda_res[ystart + ix]));            
            ix += blockDim.x;
        }        
        iy += blockDim.y;
    }
    __syncthreads();

    if (tid < 512)
        sdata[tid] = max(sdata[tid], sdata[tid + 512]);
    if (tid < 256)
        sdata[tid] = max(sdata[tid], sdata[tid + 256]);
    if (tid < 128)
        sdata[tid] = max(sdata[tid], sdata[tid + 128]);
    if (tid < 64)
        sdata[tid] = max(sdata[tid], sdata[tid + 64]);
    if (tid < 32)
    {
        int v = max(sdata[tid], sdata[tid + 32]);
        v = max(v, __shfl_down_sync(0xffffffff, v, 16));
        v = max(v, __shfl_down_sync(0xffffffff, v, 8));
        v = max(v, __shfl_down_sync(0xffffffff, v, 4));
        v = max(v, __shfl_down_sync(0xffffffff, v, 2));
        v = max(v, __shfl_down_sync(0xffffffff, v, 1));
        if (tid == 0)
            cuda_res[0] = v;
    }    
}










void hIntegral(unsigned char* src, int* integral, int* buff, int width, int height, int sstride, int dstride)
{
    // First step: Integral in each tile
    dim3 block1(NTY * NTX * NTT);
    dim3 grid1(iDivUp(width, NTX * NTT), iDivUp(height, NTY * NTT));
    gIntegralInTile<<<grid1, block1>>>(src, integral, width, height, sstride, dstride);
    // CHECK(cudaDeviceSynchronize());

    // Second step: Integral the last column of each tile
    const int xtiles = iDivUp(width, NTT);          // Number of tiles on x direction
    const int xtile_stride = iDivUp(dstride, NTT);
    const int xpsz = iExp2Up(xtiles) / 2;           // Padding size on x direction
    dim3 block2(xpsz / 2);                          // Avoid too much thread and shared memory by /4, /2 for binary-tree parallel
    dim3 grid2(height);
    gIntegralTileLastCol2<<<grid2, block2, xpsz * sizeof(int)>>>(integral, buff, width, height, dstride, xtile_stride);
    // CHECK(cudaDeviceSynchronize());

    // Third step: Integral the last row of each tile
    const int ytiles = iDivUp(height, NTT);
    const int ypsz = iExp2Up(ytiles) / 2;
    dim3 block3(ypsz / 2);                          // Avoid too much thread and shared memory by /4, /2 for binary-tree parallel
    dim3 grid3(width);
    gIntegralTileLastRow2<<<grid3, block3, ypsz * sizeof(int)>>>(integral, buff, width, height, dstride, xtile_stride);
    // CHECK(cudaDeviceSynchronize());

    // We add a assist column to avoid assign extra buffer.
    dim3 block4(NTY * NTX * NTT);
    dim3 grid4(iDivUp(width, NTX * NTT), iDivUp(height, NTY * NTT));
    gIntegralInGlobal<<<grid4, block4>>>(integral, buff, width, height, dstride, xtile_stride);
    // CHECK(cudaDeviceSynchronize());
}


void hInitRand(curandState* rand_state, int seed, int num)
{
    gInitRand<<<iDivUp(num, 1024), 1024>>>(rand_state, seed, num);
}


void hRandFill(unsigned char* data, curandState* rand_state, int width, int height, int stride)
{
    dim3 block(32, 8);
    dim3 grid(iDivUp(width, block.x), iDivUp(height, block.y));
    gRandFill<<<grid, block>>>(data, rand_state, width, height, stride);
}


void hCmpMaxAbsDiff(int* nppi_res, int* cuda_res, int width, int height)
{
    dim3 block(32, 32);
    gCmpMaxAbsDiff<<<1, block>>>(nppi_res, cuda_res, width, height);
}


