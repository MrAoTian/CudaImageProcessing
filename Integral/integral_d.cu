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


__global__ void gAligned4IntegralInTile(unsigned char* src, int4* pack_integral, int swidth, int sheight, int sstride, int dpack_width, int dheight)
{
    int pack_ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = (blockIdx.y * blockDim.y + threadIdx.y) << 2;
    if (pack_ix < dpack_width && iy < dheight)
    {
        // Load data
        int six = pack_ix << 2;
        int systart = 0; 
        unsigned char tile[16] = { 0 };
        for (int i = 0; i < min(4, sheight - iy); i++)
        {
            systart = (iy + i) * sstride;
            for (int j = 0; j < min(4, swidth - six); j++)
            {
                tile[i * 4 + j] = src[systart + six + j];
            }
        }

        // Integral by row
        int4 irow1, irow2, irow3, irow4;
        irow1.x = tile[0]; 
        irow2.x = tile[4]; 
        irow3.x = tile[8]; 
        irow4.x = tile[12]; 
        irow1.y = tile[0] + tile[1]; 
        irow2.y = tile[4] + tile[5]; 
        irow3.y = tile[8] + tile[9]; 
        irow4.y = tile[12] + tile[13]; 
        irow1.z = irow1.y + tile[2]; 
        irow2.z = irow2.y + tile[6]; 
        irow3.z = irow3.y + tile[10]; 
        irow4.z = irow4.y + tile[14]; 
        irow1.w = irow1.z + tile[3]; 
        irow2.w = irow2.z + tile[7]; 
        irow3.w = irow3.z + tile[11]; 
        irow4.w = irow4.z + tile[15]; 

        // Integral by col
        irow2.x += irow1.x; irow2.y += irow1.y; irow2.z += irow1.z; irow2.w += irow1.w;
        irow3.x += irow2.x; irow3.y += irow2.y; irow3.z += irow2.z; irow3.w += irow2.w;
        irow4.x += irow3.x; irow4.y += irow3.y; irow4.z += irow3.z; irow4.w += irow3.w;

        // Save result
        int idx = iy * dpack_width + pack_ix;
        pack_integral[idx] = irow1;
        pack_integral[idx + dpack_width] = irow2;
        pack_integral[idx + dpack_width * 2] = irow3;
        pack_integral[idx + dpack_width * 3] = irow4;
    }
}


__global__ void gAligned4IntegralInTile2(unsigned char* src, int4* pack_integral, int swidth, int sheight, int sstride, int dheight, int xtiles, int ntiles)
{
    int itile = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_iy = 0, tile_ix = 0, ix = 0, iy = 0, idx = 0;
    int tile[16] = { 0 };
    int4* irows = reinterpret_cast<int4*>(tile);
    for (int tile_idx = itile; tile_idx < ntiles; tile_idx += gridDim.x * blockDim.x)
    {
        tile_iy = tile_idx / xtiles;
        tile_ix = tile_idx % xtiles;
        ix = tile_ix << 2;
        iy = tile_iy << 2;
        idx = iy * sstride + ix;
        irows[0] = make_int4(0, 0, 0, 0);
        irows[1] = make_int4(0, 0, 0, 0);
        irows[2] = make_int4(0, 0, 0, 0);
        irows[3] = make_int4(0, 0, 0, 0);
        for (int i = 0; i < min(4, sheight - iy); i++)
        {
            for (int j = 0; j < min(4, swidth - ix); j++)
            {
                tile[i * 4 + j] = src[idx + j];
            }
            idx += sstride;
        }

        // Integral by row
        tile[1] += tile[0]; tile[5] += tile[4]; tile[9] += tile[8]; tile[13] += tile[12]; 
        tile[2] += tile[1]; tile[6] += tile[5]; tile[10] += tile[9]; tile[14] += tile[13]; 
        tile[3] += tile[2]; tile[7] += tile[6]; tile[11] += tile[10]; tile[15] += tile[14]; 

        // Integral by col
        tile[4] += tile[0]; tile[5] += tile[1]; tile[6] += tile[2]; tile[7] += tile[3];
        tile[8] += tile[4]; tile[9] += tile[5]; tile[10] += tile[6]; tile[11] += tile[7];
        tile[12] += tile[8]; tile[13] += tile[9]; tile[14] += tile[10]; tile[15] += tile[11];

        // Save result
        idx = iy * xtiles + tile_ix;
        pack_integral[idx] = irows[0];
        pack_integral[idx + xtiles] = irows[1];
        pack_integral[idx + xtiles * 2] = irows[2];
        pack_integral[idx + xtiles * 3] = irows[3];
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


__inline__ __device__ void dAddInt4Inplace(int4& a, const int4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}


__inline__ __device__ int4 dAddInt4(const int4& a, const int4& b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}


__global__ void gIntegralAtTileLastColUnroll2(int* integral, int width, int height)
{
    extern __shared__ int4 pack_sdata[];

    const int tid = threadIdx.x;
    const int piy = blockIdx.x << 2;
    const int pix = (tid << 4) + 3;
    const int pidx1 = piy * width + pix;
    const int pidx2 = pidx1 + width;
    const int pidx3 = pidx2 + width;
    const int pidx4 = pidx3 + width;
    const int stid = tid << 1;
    const int stidp = stid + 1;
    const int stidq = stid + 2;
    const int snum = blockDim.x << 1;

    // Copy data to shared memory and integral the pack values in register
    int4 a0 = make_int4(0, 0, 0, 0);
    int4 a1 = make_int4(0, 0, 0, 0);
    int4 a2 = make_int4(0, 0, 0, 0);
    pack_sdata[stid] = make_int4(0, 0, 0, 0);
    if (pix < width)
    {
        a0 = make_int4(integral[pidx1], integral[pidx2], integral[pidx3], integral[pidx4]);
    }
    if (pix + 4 < width)
    {
        pack_sdata[stid] = make_int4(
            a0.x + integral[pidx1 + 4], a0.y + integral[pidx2 + 4],
            a0.z + integral[pidx3 + 4], a0.w + integral[pidx4 + 4]);
    }
    if (pix + 8 < width)
    {
        a1 = make_int4(integral[pidx1 + 8], integral[pidx2 + 8], 
            integral[pidx3 + 8], integral[pidx4 + 8]);
    }
    if (pix + 12 < width)
    {
        a2 = make_int4(a1.x + integral[pidx1 + 12], a1.y + integral[pidx2 + 12],
            a1.z + integral[pidx3 + 12], a1.w + integral[pidx4 + 12]);
    }
    pack_sdata[stidp] = a2;

    // Cumulative binary tree node
    int shift = 0, si = 0, sj = 0;
    int4 temp;
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            dAddInt4Inplace(pack_sdata[(stidq << shift) - 1], pack_sdata[(stidp << shift) - 1]);
        }
        shift++;
    }

    if (tid == 0)
    {
        pack_sdata[snum - 1] = make_int4(0, 0, 0, 0);
    }
    
    for (int d = 1; d < snum; d <<= 1)
    {
        shift--;
        __syncthreads();
        if (tid < d)
        {
            si = (stidp << shift) - 1;
            sj = (stidq << shift) - 1;
            temp = pack_sdata[si];
            pack_sdata[si] = pack_sdata[sj];
            dAddInt4Inplace(pack_sdata[sj], temp);
        }
    }
    __syncthreads();

    // Save results to global memory
    if (pix >= width) return;
    temp = pack_sdata[stid];
    integral[pidx1] = a0.x + temp.x;
    integral[pidx2] = a0.y + temp.y;
    integral[pidx3] = a0.z + temp.z;
    a0.w += temp.w;
    integral[pidx4] = a0.w;
    
    if (pix + 4 >= width) return;
    integral[pidx4 + 1] += a0.w;
    integral[pidx4 + 2] += a0.w;
    integral[pidx4 + 3] += a0.w;

    a0 = pack_sdata[stidp];
    integral[pidx1 + 4] = a0.x;
    integral[pidx2 + 4] = a0.y;
    integral[pidx3 + 4] = a0.z;
    integral[pidx4 + 4] = a0.w;

    if (pix + 8 >= width) return;
    integral[pidx4 + 5] += a0.w;
    integral[pidx4 + 6] += a0.w;
    integral[pidx4 + 7] += a0.w;

    integral[pidx1 + 8] = a1.x + a0.x;
    integral[pidx2 + 8] = a1.y + a0.y;
    integral[pidx3 + 8] = a1.z + a0.z;
    a1.w += a0.w;
    integral[pidx4 + 8] = a1.w;

    if (pix + 12 >= width) return;
    integral[pidx4 +  9] += a1.w;
    integral[pidx4 + 10] += a1.w;
    integral[pidx4 + 11] += a1.w;

    if (stidq < snum)
    {
        a2 = pack_sdata[stidq];
        integral[pidx1 + 12] = a2.x;
        integral[pidx2 + 12] = a2.y;
        integral[pidx3 + 12] = a2.z;
        integral[pidx4 + 12] = a2.w;
    }
    else
    {
        integral[pidx1 + 12] = a2.x + a0.x;
        integral[pidx2 + 12] = a2.y + a0.y;
        integral[pidx3 + 12] = a2.z + a0.z;
        a2.w += a0.w;
        integral[pidx4 + 12] = a2.w;
    }

    if (pix + 16 < width)
    {
        integral[pidx4 + 13] += a2.w;
        integral[pidx4 + 14] += a2.w;
        integral[pidx4 + 15] += a2.w;
    }
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


__global__ void gIntegralAtTileLastRowUnroll2(int* integral, int4* pack_integral, int pack_width, int width, int height)
{
    extern __shared__ int4 pack_sdata[];

    int tid = threadIdx.x;
    const int pix = blockIdx.x;
    const int piy = (tid << 4) + 3;
    const int pidx = piy * pack_width + pix;
    const int OFFSET_4 = pack_width * 4;
    const int OFFSET_8 = pack_width * 8;
    const int OFFSET_12 = pack_width * 12;
    const int stid = tid << 1;
    const int stidp = stid + 1;
    const int stidq = stid + 2;
    const int snum = blockDim.x << 1;

    // Copy data to shared memory and integral the pack values in register
    int4 a0, a1, a2, temp;
    a0 = make_int4(0, 0, 0, 0);
    if (piy < height)
    {
        a0 = pack_integral[pidx];
    }

    pack_sdata[stid] = make_int4(0, 0, 0, 0);
    if (piy + 4 < height)
    {
        a1 = pack_integral[pidx + OFFSET_4];
        pack_sdata[stid] = dAddInt4(a0, a1);
    }

    a1 = make_int4(0, 0, 0, 0);
    if (piy + 8 < height)
    {
        a1 = pack_integral[pidx + OFFSET_8];
    }
    
    a2 = make_int4(0, 0, 0, 0);
    if (piy + 12 < height)
    {
        a2 = pack_integral[pidx + OFFSET_12];
        dAddInt4Inplace(a2, a1);
    }
    pack_sdata[stidp] = a2;

    // Cumulative binary tree node
    int shift = 0, si = 0, sj = 0;
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            si = (stidq << shift) - 1;
            sj = (stidp << shift) - 1;
            dAddInt4Inplace(pack_sdata[si], pack_sdata[sj]);
        }
        shift++;
    }

    if (tid == 0)
    {
        pack_sdata[snum - 1] = make_int4(0, 0, 0, 0);
    }
    
    for (unsigned int d = 1; d < snum; d <<= 1)
    {
        shift--;
        __syncthreads();
        if (tid < d)
        {
            si = (stidp << shift) - 1;
            sj = (stidq << shift) - 1;
            temp = pack_sdata[si];
            pack_sdata[si] = pack_sdata[sj];
            dAddInt4Inplace(pack_sdata[sj], temp);
        }
    }
    __syncthreads();

    int& idx = tid;
    idx = (pidx << 2) + 3;
        
    // Save results to global memory
    if (piy >= height) return;
    dAddInt4Inplace(a0, pack_sdata[stid]);
    pack_integral[pidx] = a0;
    
    if (piy + 4 >= height) return;
    integral[idx + width] += a0.w;
    integral[idx + width * 2] += a0.w;
    integral[idx + width * 3] += a0.w;

    a0 = pack_sdata[stidp];
    pack_integral[pidx + OFFSET_4] = a0;

    if (piy + 8 >= height) return;
    integral[idx + width * 5] += a0.w;
    integral[idx + width * 6] += a0.w;
    integral[idx + width * 7] += a0.w;

    dAddInt4Inplace(a1, a0);
    pack_integral[pidx + OFFSET_8] = a1;

    if (piy + 12 >= height) return;
    integral[idx + width *  9] += a1.w;
    integral[idx + width * 10] += a1.w;
    integral[idx + width * 11] += a1.w;

    a2 = stidq < snum ? pack_sdata[stidq] : dAddInt4(a2, a0);
    pack_integral[pidx + OFFSET_12] = a2;

    if (piy + 16 < height)
    {
        integral[idx + width * 13] += a2.w;
        integral[idx + width * 14] += a2.w;
        integral[idx + width * 15] += a2.w;
    }
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


__global__ void gIntegralGlobalPack(int* integral, int4* pack_integral, int pack_width, int width, int height)
{
    const int pack_ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = (blockIdx.y * blockDim.y + threadIdx.y) << 2;
    if (pack_ix < pack_width && iy < height)
    {
        int pack_idx = iy * pack_width + pack_ix;
        const int idx = pack_idx << 2;
        int4 vtop = iy > 0 ? pack_integral[pack_idx - pack_width] : make_int4(0, 0, 0, 0);
        int3 vleft = pack_ix > 0 ? make_int3(integral[idx - 1], integral[idx - 1 + width], integral[idx - 1 + (width << 1)]) : make_int3(0, 0, 0);
        int2 vtopleft = (iy > 0 && pack_ix > 0) ? make_int2(integral[idx - 1 - width], integral[idx - 1 - width]) : make_int2(0, 0);
        
        int4 val = pack_integral[pack_idx];
        val.x += vtop.x + vleft.x - vtopleft.x; 
        val.y += vtop.y + vleft.x - vtopleft.x; 
        val.z += vtop.z + vleft.x - vtopleft.x;
        pack_integral[pack_idx] = val;

        pack_idx += pack_width;
        val = pack_integral[pack_idx];
        val.x += vtop.x + vleft.y - vtopleft.x; 
        val.y += vtop.y + vleft.y - vtopleft.x; 
        val.z += vtop.z + vleft.y - vtopleft.x;
        pack_integral[pack_idx] = val;

        pack_idx += pack_width;
        val = pack_integral[pack_idx];
        val.x += vtop.x + vleft.z - vtopleft.x; 
        val.y += vtop.y + vleft.z - vtopleft.x; 
        val.z += vtop.z + vleft.z - vtopleft.x;
        pack_integral[pack_idx] = val;
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


void hAligned4Integral(unsigned char* src, int* integral, int swidth, int sheight, int sstride, int dwidth, int dheight)
{
    const int xtiles = dwidth / 4;  // Count of tiles on direction X
    const int ytiles = dheight / 4; // Count of tiles on direction Y
    const int ntiles = xtiles * ytiles;
    int4* pack_integral = reinterpret_cast<int4*>(integral);

    // 1. Compute integral in 4x4 tile
    dim3 block1(32, 8);
    dim3 grid1(iDivUp(xtiles, block1.x), iDivUp(ytiles, block1.y));
    gAligned4IntegralInTile<<<grid1, block1>>>(src, pack_integral, swidth, sheight, sstride, xtiles, dheight);

    // dim3 block1_2(256);
    // dim3 grid1_2(ntiles / (block1_2.x * 4));
    // gAligned4IntegralInTile2<<<grid1_2, block1_2>>>(src, pack_integral, swidth, sheight, sstride, dheight, xtiles, ntiles);

    // 2. Compute integral of last tile col
    const int xnodes = iExp2Up(xtiles) / 2;             // Node count of binary tree, /2 to avoid too much shared memory
    const size_t xbytes = xnodes * sizeof(int4);        // Size of shared memory
    dim3 block2(xnodes / 2);                            // The thread count / shared memory = 1 / 2
    dim3 grid2(ytiles);                                 // Pack tile column to avoid another buffer
    gIntegralAtTileLastColUnroll2<<<grid2, block2, xbytes>>>(integral, dwidth, dheight);

    // 3. Compute integral of last tile row
    const int ynodes = iExp2Up(ytiles) / 2;
    const size_t ybytes = ynodes * sizeof(int4);
    dim3 block3(ynodes / 2);
    dim3 grid3(xtiles);
    gIntegralAtTileLastRowUnroll2<<<grid3, block3, ybytes>>>(integral, pack_integral, xtiles, dwidth, dheight);

    // 4. Broadcast integral result from tile to global
    dim3 block4(32, 8);
    dim3 grid4(iDivUp(xtiles, block4.x), iDivUp(ytiles, block4.y));
    gIntegralGlobalPack<<<grid4, block4>>>(integral, pack_integral, xtiles, dwidth, dheight);
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


