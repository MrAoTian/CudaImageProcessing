#include "image_process.h"


////////////////// Histogram equalization //////////////////


#define X2 32
#define Y2 32


__global__ void gCalcHistUnroll8(unsigned char* src, int* hist, int width, int height, int stride)
{
    __shared__ int shist[256];
    int _ix = blockIdx.x * X2 * 8 + threadIdx.x;
    int iy = blockIdx.y * Y2 + threadIdx.y;
    if (_ix >= width || iy >= height)
    {
        return;
    }

    // Initialization
    int tid = threadIdx.y * X2 + threadIdx.x;
    if (tid < 256)
    {
        shist[tid] = 0;
    }
    __syncthreads();

    // Statistical
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int ix = _ix + i * X2;
        if (ix >= width)
        {
            return;
        }
        atomicAdd(shist + src[iy * stride + ix], 1);
    }
    __syncthreads();

    // Add to global memory
    if (tid < 256)
    {
        atomicAdd(hist + tid, shist[tid]);
    }
}


__global__ void gCalcHeTable(int* hist, unsigned char* table, float factor/* = 256.f / size*/)
{
    __shared__ int cumu_hist[256];

    int tdx = threadIdx.x;
    int offset = 1;
    int tdx2 = tdx + tdx;
    int tdx2p = tdx2 + 1;
    int tdx2pp = tdx2 + 2;
    cumu_hist[tdx2] = hist[tdx2];
    cumu_hist[tdx2p] = hist[tdx2p];
    for (int d = 128; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tdx < d)
        {
            int ai = offset * tdx2p - 1;
            int bi = offset * tdx2pp - 1;
            cumu_hist[bi] += cumu_hist[ai];
        }
        offset <<= 1;
    }

    if (tdx == 0)
    {
        cumu_hist[255] = 0;
    }
    
    for (int d = 1; d < 256; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (tdx < d)
        {
            int ai = offset * tdx2p - 1;
            int bi = offset * tdx2pp - 1;
            int t = cumu_hist[ai];
            cumu_hist[ai] = cumu_hist[bi];
            cumu_hist[bi] += t;
        }
    }
    __syncthreads();

    table[tdx2] = __float2int_rn(fminf(255.f, cumu_hist[tdx2p] * factor));
    if (tdx2p == 255) 
    {
        table[tdx2p] = __float2int_rn(fminf(255.f, (cumu_hist[255] + hist[255]) * factor));
    }
    else
    {
        table[tdx2p] = __float2int_rn(fminf(255.f, cumu_hist[tdx2pp] * factor));
    }
}


__global__ void gMapping(unsigned char* src, unsigned char* dst, unsigned char* table, int width, int height, int stride)
{
    int ix = blockIdx.x * X2 + threadIdx.x;
    int iy = blockIdx.y * Y2 + threadIdx.y;
    if (ix < width && iy < height)
    {
        int idx = iy * stride + ix;
        dst[idx] = table[src[idx]];
    }
}



void hCalcHist(unsigned char* src, int* hist, int width, int height, int stride)
{
    dim3 block(X2, Y2);
    dim3 grid(iDivUp(width, X2 * 8), iDivUp(height, Y2));
    gCalcHistUnroll8<<<grid, block>>>(src, hist, width, height, stride);
}


void hCalcHeTable(int* hist, unsigned char* table, float fatcor)
{
    gCalcHeTable<<<1, 128>>>(hist, table, fatcor);
}


void hMapping(unsigned char* src, unsigned char* dst, unsigned char* table, int width, int height, int stride)
{
    dim3 block(X2, Y2);
    dim3 grid(iDivUp(width, X2), iDivUp(height, Y2));
    gMapping<<<grid, block>>>(src, dst, table, width, height, stride);
}





////////////////// CLAHE //////////////////

#define NX 16
#define NY 16


__inline__ __device__ int dLimitSize(int x, int sz)
{
    if (x < 0) return -x;
    else if (x >= sz) return sz + sz - 2 - x;
    else return x;
}


__global__ void gCalcTileHists(unsigned char* src, int* hists, int xtiles, int ytiles, int tile_width, int tile_height, int pad_left, int pad_top, int width, int height, int stride)
{
    __shared__ int shist[256];
    int ix = blockIdx.x * NX + threadIdx.x;
    int iy = blockIdx.y * NY + threadIdx.y;
    int tid = threadIdx.y * NX + threadIdx.x;
    
    for (int i = 0; i < ytiles; i++)
    {
        int systart = dLimitSize(iy + i * tile_height - pad_top, height) * stride;
        for (int j = 0; j < xtiles; j++)
        {
            int sidx = systart + dLimitSize(ix + j * tile_width - pad_left, width);

            shist[tid] = 0;
            __syncthreads();
            
            if (ix < tile_width && iy < tile_height)
            {
                atomicAdd(shist + src[sidx], 1);
            }
            __syncthreads();

            int* curr_hist = hists + ((i * xtiles + j) << 8);
            atomicAdd(curr_hist + tid, shist[tid]);
            __syncthreads();
        }        
    }
    
    
}


__global__ void gClipLimit(int* hists, int limit)
{
    __shared__ int steal;
    int tid = threadIdx.x;
    int* curr_hist = hists + (blockIdx.x << 8);

    if (tid == 0)
    {
        steal = 0;
    }
    __syncthreads();

    if (curr_hist[tid] > limit)
    {
        atomicAdd(&steal, curr_hist[tid] - limit);
        curr_hist[tid] = limit;
    }
    __syncthreads();

    int bonus = steal >> 8;
    int residual = steal - (bonus << 8);
    atomicAdd(curr_hist + tid, bonus);
    if (tid < residual)
    {
        atomicAdd(curr_hist + (tid << 8) / residual, 1);
    }
}


__global__ void gCreateTable(int* hists, float* table, float fr)
{
    __shared__ int cumu_hist[256];

    int bid = blockIdx.x;
    int tdx = threadIdx.x;
    int offset = 1;
    int tdx2 = tdx + tdx;
    int tdx2p = tdx2 + 1;
    int tdx2pp = tdx2 + 2;

    int* curr_hist = hists + (bid << 8);
    cumu_hist[tdx2] = curr_hist[tdx2];
    cumu_hist[tdx2p] = curr_hist[tdx2p];
    for (int d = 128; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tdx < d)
        {
            int ai = offset * tdx2p - 1;
            int bi = offset * tdx2pp - 1;
            cumu_hist[bi] += cumu_hist[ai];
        }
        offset <<= 1;
    }

    if (tdx == 0)
    {
        cumu_hist[255] = 0;
    }

    for (int d = 1; d < 256; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (tdx < d)
        {
            int ai = offset * tdx2p - 1;
            int bi = offset * tdx2pp - 1;
            int t = cumu_hist[ai];
            cumu_hist[ai] = cumu_hist[bi];
            cumu_hist[bi] += t;
        }
    }
    __syncthreads();

    float* curr_table = table + (bid << 8);
    curr_table[tdx2] = __fmul_rn(cumu_hist[tdx2p], fr);
    if (tdx < 127)
    {
        curr_table[tdx2p] = __fmul_rn(cumu_hist[tdx2pp], fr);    
    }
    else
    {
        curr_table[tdx2p] = __fmul_rn(cumu_hist[tdx2p] + curr_hist[tdx2p], fr);
    }    
}


__global__ void gInterpolateMapping(unsigned char* src, unsigned char* dst, float* tables, int xtiles, int ytiles, int tile_width, int tile_height, int pad_left, int pad_top, int width, int height, int stride)
{
    int ix = blockIdx.x * NX + threadIdx.x;
    int iy = blockIdx.y * NY + threadIdx.y;
    if (ix >= width || iy >= height)
    {
        return;
    }

    int idx = iy * stride + ix;
    int pix = ix + pad_left;
    int piy = iy + pad_top;
    int htw = tile_width >> 1;
    int hth = tile_height >> 1;
    int ymode = piy < hth ? 0 : (piy >= ytiles * tile_height - hth ? 2 : 1);
    int xmode = pix < htw ? 0 : (pix >= xtiles * tile_width - htw ? 2 : 1);
    int mode = ymode * 3 + xmode;
    int tidx = src[idx];

    switch (mode)
    {
    case 0: // Top-Left
    {
        float* curr_table = tables + 0;
        dst[idx] = __float2int_rn(curr_table[tidx]);
        break;
    }
    case 1: // Top-Mid
    {
        int wbi = (pix - htw) / tile_width;
        float* table0 = tables + (wbi << 8);
        float* table1 = table0 + 256;
        float p = __fdiv_rn(pix - (wbi * tile_width + htw), tile_width);
        dst[idx] = __float2int_rn(__fmaf_rn(1 - p, table0[tidx], __fmul_rn(p, table1[tidx])));
        break;
    }
    case 2: // Top-Right
    {
        float* curr_table = tables + ((xtiles - 1) << 8);
        dst[idx] = __float2int_rn(curr_table[tidx]);
        break;
    }
    case 3: // Mid-Left
    {
        int hbi = (piy - hth) / tile_height;
        float* table0 = tables + ((hbi * xtiles) << 8);
        float* table1 = table0 + (xtiles << 8);
        float p = __fdiv_rn(piy - (hbi * tile_height + hth), tile_height);
        dst[idx] = __float2int_rn(__fmaf_rn(1 - p, table0[tidx], __fmul_rn(p, table1[tidx])));
        break;
    }
    case 4: // Mid-Mid
    {
        int hbi = (piy - hth) / tile_height;
        int wbi = (pix - htw) / tile_width;
        float* table0 = tables + ((hbi * xtiles + wbi) << 8);
        float* table1 = table0 + 256;
        float* table2 = table0 + (xtiles << 8);
        float* table3 = table2 + 256;
        float p = __fdiv_rn(piy - (hbi * tile_height + hth), tile_height);
        float q = __fdiv_rn(pix - (wbi * tile_width + htw), tile_width);
        dst[idx] = __float2int_rn((1 - p) * ((1 - q) * table0[tidx] + q * table1[tidx]) + p * ((1 - q) * table2[tidx] + q * table3[tidx]));
        break;
    }
    case 5: // Mid-Right
    {
        int hbi = (piy - hth) / tile_height;
        float* table0 = tables + ((hbi * xtiles + xtiles - 1) << 8);
        float* table1 = table0 + (xtiles << 8);
        float p = __fdiv_rn(piy - (hbi * tile_height + hth), tile_height);
        dst[idx] = __float2int_rn(__fmaf_rn(1 - p, table0[tidx], __fmul_rn(p, table1[tidx])));
        break;
    }
    case 6: // Bot-Left
    {
        float* curr_table =  tables + ((ytiles * xtiles - xtiles) << 8);
        dst[idx] = __float2int_rn(curr_table[tidx]);
        break;
    }
    case 7: // Bot-Mid
    {
        int wbi = (pix - htw) / tile_width;
        float* table0 = tables + ((ytiles * xtiles - xtiles + wbi) << 8);
        float* table1 = table0 + 256;
        float p = __fdiv_rn(pix - (wbi * tile_width + htw), tile_width);
        dst[idx] = __float2int_rn(__fmaf_rn(1 - p, table0[tidx], __fmul_rn(p, table1[tidx])));
        break;
    }
    case 8: // Bot-Right
    {
        float* curr_table =  tables + ((ytiles * xtiles - 1) << 8);
        dst[idx] = __float2int_rn(curr_table[tidx]);
        break;
    }
    }    
}



void hCalcTileHists(unsigned char* src, int* hists, int xtiles, int ytiles, int tile_width, int tile_height, int pad_left, int pad_top, int width, int height, int stride)
{
    dim3 block(NX, NY);
    dim3 grid(iDivUp(tile_width, NX), iDivUp(tile_height, NY));
    gCalcTileHists<<<grid, block>>>(src, hists, xtiles, ytiles, tile_width, tile_height, pad_left, pad_top, width, height, stride);
}


void hClipLimit(int* hists, int limit, int ntiles)
{
    gClipLimit<<<ntiles, 256>>>(hists, limit);
}


void hCreateTable(int* hists, float* tables, int tile_pixels, int ntiles)
{
    float fr = 255.f / tile_pixels;
    gCreateTable<<<ntiles, 128>>>(hists, tables, fr);
}


void hInterpolateMapping(unsigned char* src, unsigned char* dst, float* tables, int xtiles, int ytiles, int tile_width, int tile_height, int pad_left, int pad_top, int width, int height, int stride)
{
    dim3 block(NX, NY);
    dim3 grid(iDivUp(width, NX), iDivUp(height, NY));
    gInterpolateMapping<<<grid, block>>>(src, dst, tables, xtiles, ytiles, tile_width, tile_height, pad_left, pad_top, width, height, stride);
}
