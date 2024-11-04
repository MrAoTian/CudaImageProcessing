#include "guided_filter_d.h"
#include <opencv2/highgui.hpp>


#define ILEN 1024
#define X2   32


template <int CHANNEL>
__global__ void gScanLongRow(float* src, float* dst, int width, int height, int src_stride, int dst_stride, int steps)
{
	__shared__ float smem[ILEN * CHANNEL];
	__shared__ float cumu_sums[CHANNEL];

	int tid = threadIdx.x;
	int tid2 = tid << 1;
	int tid2p = tid2 + 1;
	int tid2pp = tid2 + 2;
	int ts2 = tid2 * CHANNEL;
	int ts2p = ts2 + CHANNEL;

	int iy = blockIdx.x;
	int systart = iy * src_stride;
	int dystart = (iy + 1) * dst_stride;
	const int wc = width * CHANNEL;

#pragma unroll
	for (int c = 0; c < CHANNEL; c++)
	{
		cumu_sums[c] = 0;
	}

	for (int i = 0; i < steps; i++)
	{
		int ix2 = i * ILEN + tid2;
		int ix2p = ix2 + 1;
		int is2 = ix2 * CHANNEL;
		int is2p = is2 + CHANNEL;
		
		// load source to shared memory
#pragma unroll
		for (int c = 0; c < CHANNEL; c++)
		{
			smem[ts2 + c] = ix2 < width ? src[systart + is2 + c] : 0.f;
			smem[ts2p + c] = ix2p < width ? src[systart + is2p + c] : 0.f;
		}		

		// Accumulate
		int offset = 0;
		for (int d = ILEN >> 1; d > 1; d >>= 1)
		{
			__syncthreads();
			if (tid < d)
			{
				int ai = ((tid2p << offset) - 1) * CHANNEL;
				int bi = ((tid2pp << offset) - 1) * CHANNEL;
#pragma unroll
				for (int c = 0; c < CHANNEL; c++)
				{
					smem[bi + c] += smem[ai + c];	
				}
			}
			offset++;
		}

		if (tid == 0)
		{
			int last_idx = (ILEN - 1) * CHANNEL;
#pragma unroll
			for (int c = 0; c < CHANNEL; c++)
			{
				smem[last_idx + c] = 0.f;
			}
		}

		for (int d = 1; d < ILEN; d <<= 1)
		{
			__syncthreads();
			if (tid < d)
			{
				int ai = ((tid2p << offset) - 1) * CHANNEL;
				int bi = ((tid2pp << offset) - 1) * CHANNEL;
#pragma unroll
				for (int c = 0; c < CHANNEL; c++)
				{
					float t = smem[ai + c];
					smem[ai + c] = smem[bi + c];
					smem[bi + c] += t;
				}
			}
			offset--;
		}
		__syncthreads();

		// Add cumulative sum
#pragma unroll
		for (int c = 0; c < CHANNEL; c++)
		{
			smem[ts2 + c] += cumu_sums[c];
			smem[ts2p + c] += cumu_sums[c];
		}
		__syncthreads();

		// Store results
		if (ix2 < width)
		{
#pragma unroll
			for (int c = 0; c < CHANNEL; c++)
			{
				dst[dystart + is2 + c] = smem[ts2 + c];
			}
			if (ix2 == width - 1)
			{				
#pragma unroll
				for (int c = 0; c < CHANNEL; c++)
				{
					dst[dystart + wc + c] = smem[ts2 + c] + src[systart + wc - CHANNEL + c];
				}
			}
		}
		if (ix2p < width)
		{
#pragma unroll
			for (int c = 0; c < CHANNEL; c++)
			{
				dst[dystart + is2p + c] = smem[ts2p + c];
			}
			if (ix2p == width - 1)
			{
#pragma unroll
				for (int c = 0; c < CHANNEL; c++)
				{
					dst[dystart + wc + c] = smem[ts2p + c] + src[systart + wc - CHANNEL + c];
				}
			}
		}

		// Cumulative to next step
		if ((tid == blockDim.x - 1) && (ix2p < width))	// Cumulative sum
		{
#pragma unroll
			for (int c = 0; c < CHANNEL; c++)
			{
				cumu_sums[c] = smem[ts2p + c] + src[systart + is2p + c];
			}		
		}
		__syncthreads();
	}
}


__global__ void gScanLongCol(float* data, int width, int height, int channel, int stride, int steps)
{
	__shared__ float smem[ILEN];
	__shared__ float cumu_sum;

	int tid = threadIdx.x;
	int tid2 = tid << 1;
	int tid2p = tid2 + 1;
	int tid2pp = tid2 + 2;
	int ix = blockIdx.x + channel;
	
	cumu_sum = 0;
	for (int i = 0; i < steps; i++)
	{
		int iy2 = i * ILEN + tid2 + 1;	// 1
		int iy2p = iy2 + 1;				// 2
		int idx2 = iy2 * stride + ix;
		int idx2p = idx2 + stride;

		// load source to shared memory
		smem[tid2] = iy2 < height ? data[idx2] : 0.f;
		smem[tid2p] = iy2p < height ? data[idx2p] : 0.f;

		// Accumulate
		int offset = 0;
		for (int d = ILEN >> 1; d > 1; d >>= 1)
		{
			__syncthreads();
			if (tid < d)
			{
				int ai = (tid2p << offset) - 1;
				int bi = (tid2pp << offset) - 1;
				smem[bi] += smem[ai];
			}
			offset++;
		}

		if (tid == 0)
		{
			smem[ILEN - 1] = 0.f;
		}

		for (int d = 1; d < ILEN; d <<= 1)
		{
			__syncthreads();
			if (tid < d)
			{
				int ai = (tid2p << offset) - 1;
				int bi = (tid2pp << offset) - 1;
				float t = smem[ai];
				smem[ai] = smem[bi];
				smem[bi] += t;
			}
			offset--;
		}
		__syncthreads();

		smem[tid2] += cumu_sum;
		smem[tid2p] += cumu_sum;
		__syncthreads();

		if (iy2 < height)
		{
			data[idx2 - stride] = smem[tid2];
			if (iy2 == height - 1)
			{
				data[idx2] += smem[tid2];
				// data[idx2p] = smem[tid2] + data[idx2];
			}
		}
		if (iy2p < height)
		{
			data[idx2] = smem[tid2p];
			if (iy2p == height - 1)
			{
				data[idx2p] += smem[tid2p];
				//data[idx2p + p] = smem[tid2p] + data[idx2p];
			}
		}

		if (tid == blockDim.x - 1 && iy2p < height)	// Cumulative sum
		{
			cumu_sum = smem[tid2p] + data[idx2p];
		}
		__syncthreads();
	}
}


template <int CHANNEL>
__global__ void gIntegralToMean(float* p_mean, float* p_intergral, int width, int height, int stride, int istride, int radius)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix >= width || iy >= height)
	{
		return;
	}

	int lef = max(0, ix - radius);
	int top = max(0, iy - radius);
	int rig = min(width, ix + 1 + radius);
	int bot = min(height, iy + 1 + radius);
	int iystart0 = top * istride;
	int iystart1 = bot * istride;
	int sidx0 = iystart0 + lef * CHANNEL;
	int sidx1 = iystart0 + rig * CHANNEL;
	int sidx2 = iystart1 + lef * CHANNEL;
	int sidx3 = iystart1 + rig * CHANNEL;
	int midx = iy * stride + ix * CHANNEL;
	float inv_area = __fdiv_rn(1.f, __int2float_rn((bot - top) * (rig - lef)));
	float* dst = p_mean + midx;

#pragma unroll 
	for (int c = 0; c < CHANNEL; c++)
	{
		dst[c] = __fmul_rn(p_intergral[sidx0 + c] + p_intergral[sidx3 + c] - p_intergral[sidx1 + c] - p_intergral[sidx2 + c], inv_area);
	}
}


__global__ void gMultiply(float* a, float* b, float* c, int width, int height, int channel, int stride)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix < width && iy < height)
	{
		int idx = iy * stride + ix * channel;
		for (int i = 0; i < channel; i++)
		{
			c[idx] = __fmul_rn(a[idx], b[idx]);
			idx++;
		}
	}
}


__global__ void gMultiplyCN1(float* a, float* b, float* c, int width, int height, int channel, int stride1, int stride2)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix < width && iy < height)
	{
		int idx = iy * stride1 + ix * channel;
		float vb = b[iy * stride2 + ix];
		for (int i = 0; i < channel; i++)
		{
			c[idx] = __fmul_rn(a[idx], vb);
			idx++;
		}
	}
}


__global__ void gCalcA(float* a, float* pm, float* im, float* ipm, float* iim, float eps, int width, int height, int channel, int stride)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix < width && iy < height)
	{
		int idx = iy * stride + ix * channel;
		float vim = 0.f, numerator = 0.f, denominator = 0.f;
		for (int i = 0; i < channel; i++)
		{
			vim = im[idx];
			numerator = __fmaf_rn(pm[idx], -vim, ipm[idx]);
			denominator = __fmaf_rn(-vim, vim, __fadd_rn(iim[idx], eps));
			a[idx] = __fdiv_rn(numerator, denominator);
			idx++;
		}
	}
}


__global__ void gCalcACN1(float* a, float* pm, float* im, float* ipm, float* iim, float eps, int width, int height, int channel, int stride1, int stride2)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix < width && iy < height)
	{
		int xs = ix * channel;
		int sidx = iy * stride1 + xs;
		int gidx = iy * stride2 + ix;
		float vim = im[gidx];
		float viim = __fadd_rn(iim[gidx], eps);
		float numerator = 0.f, denominator = 0.f;
		for (int i = 0; i < channel; i++)
		{
			numerator = __fmaf_rn(pm[sidx], -vim, ipm[sidx]);
			denominator = __fmaf_rn(-vim, vim, viim);
			a[sidx] = __fdiv_rn(numerator, denominator);
			sidx++;
		}
	}
}


__global__ void gCalcB(float* b, float* a, float* pm, float* im, int width, int height, int channel, int stride)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix < width && iy < height)
	{
		int idx = iy * stride + ix * channel;		
		for (int i = 0; i < channel; i++)
		{
			b[idx] = __fmaf_rn(a[idx], -im[idx], pm[idx]);
			idx++;
		}
	}
}


__global__ void gCalcBCN1(float* b, float* a, float* pm, float* im, int width, int height, int channel, int stride1, int stride2)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix < width && iy < height)
	{
		int idx = iy * stride1 + ix * channel;
		int vim = -im[idx];
		for (int i = 0; i < channel; i++)
		{
			b[idx] = __fmaf_rn(a[idx], vim, pm[idx]);
			idx++;
		}
	}
}


__global__ void gLinearTransform(float* src, float* dst, float* a, float* b, int width, int height, int channel, int stride)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix < width && iy < height)
	{
		int idx = iy * stride + ix * channel;		
		for (int i = 0; i < channel; i++)
		{
			dst[idx] = __fmaf_rn(src[idx], a[idx], b[idx]);
			idx++;
		}
	}
}


__global__ void gLinearTransformCN1(float* src, float* dst, float* a, float* b, int width, int height, int channel, int stride1, int stride2)
{
	int ix = blockIdx.x * X2 + threadIdx.x;
	int iy = blockIdx.y * X2 + threadIdx.y;
	if (ix < width && iy < height)
	{
		int idx = iy * stride1 + ix * channel;
		float vi = src[iy * stride2 + ix];
		for (int i = 0; i < channel; i++)
		{
			dst[idx] = __fmaf_rn(vi, a[idx], b[idx]);
			idx++;
		}
	}
}


__device__ __inline__ int reflectBorder(int x, int sz)
{
    return x < 0 ? -x : (x >= sz ? sz + sz - 2 - x : x);
}


template <const int RADIUS, const int KX>
__global__ void gCalcAB(const float* __restrict__ guidiance, const float* __restrict__ src, float* __restrict__ A, float* __restrict__ B, float eps, float coef, int width, int height, int stride)
{
    __shared__ float P[RADIUS * 2][KX + RADIUS * 2];
    __shared__ float I[RADIUS * 2][KX + RADIUS * 2];
    __shared__ float Pm[RADIUS * 4][KX];
    __shared__ float Im[RADIUS * 4][KX];
    __shared__ float IPm[RADIUS * 4][KX];
    __shared__ float IIm[RADIUS * 4][KX];

    const int R2 = RADIUS << 1;
    const int R4 = RADIUS << 2;
    const int R8 = RADIUS << 3;
    const int tix = threadIdx.x;
    const int tiy = threadIdx.y;
    const int ix = blockIdx.x * KX + tix;
    const int iy = blockIdx.y * R8 + tiy;
    int xs[R2 + 1];
    xs[RADIUS] = reflectBorder(ix, width);
    for (int i = 1; i <= RADIUS; ++i)
    {
        xs[RADIUS - i] = reflectBorder(ix - i, width);
        xs[RADIUS + i] = reflectBorder(ix + i, width);
    }
    const int six = tix + RADIUS;
    const int pix = tix + R2;
    const bool left_cond = tix < RADIUS;
    const bool righ_cond = six >= KX || ix + RADIUS >= width;
    int piy, siy, niy, offset, ny;
    float p1, p2, i1, i2, pmu, imu, ipmu, iimu;

    // First radius range - Load data to shared memory
    offset = reflectBorder(iy - RADIUS, height) * stride;
    P[tiy][six] = src[offset + xs[RADIUS]];
    I[tiy][six] = guidiance[offset + xs[RADIUS]];
    if (left_cond)
    {
        P[tiy][tix] = src[offset + xs[0]];
        I[tiy][tix] = guidiance[offset + xs[0]];
    }
    if (righ_cond)
    {
        P[tiy][pix] = src[offset + xs[R2]];
        I[tiy][pix] = guidiance[offset + xs[R2]];
    }
    __syncthreads();

    // Second ~ 4th radius range
#pragma unroll
    for (int range = RADIUS; range < R4; range += RADIUS)
    {
        // Load data to shared memory
        piy = (tiy + range) % R2;
        offset = reflectBorder(iy + range - RADIUS, height) * stride;
        P[piy][six] = src[offset + xs[RADIUS]];
        I[piy][six] = guidiance[offset + xs[RADIUS]];
        if (left_cond)
        {
            P[piy][tix] = src[offset + xs[0]];
            I[piy][tix] = guidiance[offset + xs[0]];
        }
        if (righ_cond)
        {
            P[piy][pix] = src[offset + xs[R2]];
            I[piy][pix] = guidiance[offset + xs[R2]];
        }

        // Reduce row for last-radius 
        siy = tiy + range - RADIUS;
        piy = siy % R2;
        p1 = P[piy][six];
        i1 = I[piy][six];
        pmu = p1;
        imu = i1;
        ipmu = p1 * i1;
        iimu = i1 * i1;
        for (int i = 1; i <= RADIUS; ++i)
        {
            p1 = P[piy][six - i]; p2 = P[piy][six + i];
            i1 = I[piy][six - i]; i2 = I[piy][six + i];
            pmu += p1 + p2;
            imu += i1 + i2;
            ipmu += p1 * i1 + p2 * i2;
            iimu += i1 * i1 + i2 * i2;
        }
        Pm[siy][tix] = pmu;
        Im[siy][tix] = imu;
        IPm[siy][tix] = ipmu;
        IIm[siy][tix] = iimu;
        __syncthreads();
    }

    // 4th -> radius range 
#pragma unroll
    for (int range = RADIUS; range < R8 - RADIUS; range += RADIUS)
    {
        // Load data to shared memory
        siy = tiy + range;
        piy = (siy + RADIUS) % R2;  // (siy + R3) % R2;
        offset = reflectBorder(iy + range + R2, height) * stride;
        P[piy][six] = src[offset + xs[RADIUS]];
        I[piy][six] = guidiance[offset + xs[RADIUS]];
        if (left_cond)
        {
            P[piy][tix] = src[offset + xs[0]];
            I[piy][tix] = guidiance[offset + xs[0]];
        }
        if (righ_cond)
        {
            P[piy][pix] = src[offset + xs[R2]];
            I[piy][pix] = guidiance[offset + xs[R2]];
        }

        // Reduce col and save result to dst
        ny = iy + range - RADIUS;
        if (ix < width && ny < height)
        {
            piy = siy % R4;
            pmu = Pm[piy][tix];
            imu = Im[piy][tix];
            ipmu = IPm[piy][tix];
            iimu = IIm[piy][tix];
            for (int i = 1; i <= RADIUS; ++i)
            {
                piy = (siy - i) % R4;
                niy = (siy + i) % R4;
                pmu += Pm[piy][tix] + Pm[niy][tix];
                imu += Im[piy][tix] + Im[niy][tix];
                ipmu += IPm[piy][tix] + IPm[niy][tix];
                iimu += IIm[piy][tix] + IIm[niy][tix];
            }
            pmu *= coef;
            imu *= coef;
            ipmu *= coef;
            iimu *= coef;
            ipmu = (ipmu - pmu * imu) / (iimu - imu * imu + eps);   // ipmu -> a
            niy = ny * stride + ix; // niy -> index
            A[niy] = ipmu;
            B[niy] = pmu - ipmu * imu;
        }    

        // Reduce row for last-radius
        siy += R2;
        piy = siy % R2;
        siy = siy % R4;
        p1 = P[piy][six];
        i1 = I[piy][six];
        pmu = p1;
        imu = i1;
        ipmu = p1 * i1;
        iimu = i1 * i1;
        for (int i = 1; i <= RADIUS; ++i)
        {
            p1 = P[piy][six - i]; p2 = P[piy][six + i];
            i1 = I[piy][six - i]; i2 = I[piy][six + i];
            pmu += p1 + p2;
            imu += i1 + i2;
            ipmu += p1 * i1 + p2 * i2;
            iimu += i1 * i1 + i2 * i2;
        }
        Pm[siy][tix] = pmu;
        Im[siy][tix] = imu;
        IPm[siy][tix] = ipmu;
        IIm[siy][tix] = iimu;
        __syncthreads();
    }

    // Last 2 range
    if (ix < width)
    {
        // Reduce row for last-radius
        siy = tiy + R8 + RADIUS;
        piy = siy % R2;
        siy = siy % R4;
        p1 = P[piy][six];
        i1 = I[piy][six];
        pmu = p1;
        imu = i1;
        ipmu = p1 * i1;
        iimu = i1 * i1;
        for (int i = 1; i <= RADIUS; ++i)
        {
            p1 = P[piy][six - i]; p2 = P[piy][six + i];
            i1 = I[piy][six - i]; i2 = I[piy][six + i];
            pmu += p1 + p2;
            imu += i1 + i2;
            ipmu += p1 * i1 + p2 * i2;
            iimu += i1 * i1 + i2 * i2;
        }
        Pm[siy][tix] = pmu;
        Im[siy][tix] = imu;
        IPm[siy][tix] = ipmu;
        IIm[siy][tix] = iimu;

        // Reduce col and save result to dst. -2
        ny = iy + R8 - R2;
        if (ny >= height) return;        
        siy = tiy + R8 - RADIUS;
        piy = siy % R4;
        pmu = Pm[piy][tix];
        imu = Im[piy][tix];
        ipmu = IPm[piy][tix];
        iimu = IIm[piy][tix];
        for (int i = 1; i <= RADIUS; ++i)
        {
            piy = (siy - i) % R4;
            niy = (siy + i) % R4;
            pmu += Pm[piy][tix] + Pm[niy][tix];
            imu += Im[piy][tix] + Im[niy][tix];
            ipmu += IPm[piy][tix] + IPm[niy][tix];
            iimu += IIm[piy][tix] + IIm[niy][tix];
        }
        pmu *= coef;
        imu *= coef;
        ipmu *= coef;
        iimu *= coef;
        ipmu = (ipmu - pmu * imu) / (iimu - imu * imu + eps);   // ipmu -> a
        niy = ny * stride + ix; // niy -> index
        A[niy] = ipmu;
        B[niy] = pmu - ipmu * imu;

        // Reduce col and save result to dst. -1
        ny += RADIUS;
        if (ny >= height) return;        
        siy += RADIUS;
        piy = siy % R4;
        pmu = Pm[piy][tix];
        imu = Im[piy][tix];
        ipmu = IPm[piy][tix];
        iimu = IIm[piy][tix];
        for (int i = 1; i <= RADIUS; ++i)
        {
            piy = (siy - i) % R4;
            niy = (siy + i) % R4;
            pmu += Pm[piy][tix] + Pm[niy][tix];
            imu += Im[piy][tix] + Im[niy][tix];
            ipmu += IPm[piy][tix] + IPm[niy][tix];
            iimu += IIm[piy][tix] + IIm[niy][tix];
        }
        pmu *= coef;
        imu *= coef;
        ipmu *= coef;
        iimu *= coef;
        ipmu = (ipmu - pmu * imu) / (iimu - imu * imu + eps);   // ipmu -> a
        niy = ny * stride + ix; // niy -> index
        A[niy] = ipmu;
        B[niy] = pmu - ipmu * imu;
    }
}


template <const int RADIUS, const int KX>
__global__ void gWeightByABm(const float* __restrict__ guidiance, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ dst, float coef, int width, int height, int stride)
{
    __shared__ float As[RADIUS * 2][KX + RADIUS * 2];
    __shared__ float Bs[RADIUS * 2][KX + RADIUS * 2];
    __shared__ float Am[RADIUS * 4][KX];
    __shared__ float Bm[RADIUS * 4][KX];

    const int R2 = RADIUS << 1;
    const int R4 = RADIUS << 2;
    const int R8 = RADIUS << 3;
    const int tix = threadIdx.x;
    const int tiy = threadIdx.y;
    const int ix = blockIdx.x * KX + tix;
    const int iy = blockIdx.y * R8 + tiy;
    int xs[R2 + 1];
    xs[RADIUS] = reflectBorder(ix, width);
    for (int i = 1; i <= RADIUS; ++i)
    {
        xs[RADIUS - i] = reflectBorder(ix - i, width);
        xs[RADIUS + i] = reflectBorder(ix + i, width);
    }
    const int six = tix + RADIUS;
    const int pix = tix + R2;
    const bool left_cond = tix < RADIUS;
    const bool righ_cond = six >= KX || ix + RADIUS >= width;
    int piy, siy, niy, offset, ny;
    float amu, bmu;

    // First radius range - Load data to shared memory
    offset = reflectBorder(iy - RADIUS, height) * stride;
    As[tiy][six] = A[offset + xs[RADIUS]];
    Bs[tiy][six] = B[offset + xs[RADIUS]];
    if (left_cond)
    {
        As[tiy][tix] = A[offset + xs[0]];
        Bs[tiy][tix] = B[offset + xs[0]];
    }
    if (righ_cond)
    {
        As[tiy][pix] = A[offset + xs[R2]];
        Bs[tiy][pix] = B[offset + xs[R2]];
    }
    __syncthreads();

    // Second ~ 4th radius range
#pragma unroll
    for (int range = RADIUS; range < R4; range += RADIUS)
    {
        // Load data to shared memory
        piy = (tiy + range) % R2;
        offset = reflectBorder(iy + range - RADIUS, height) * stride;
        As[piy][six] = A[offset + xs[RADIUS]];
        Bs[piy][six] = B[offset + xs[RADIUS]];
        if (left_cond)
        {
            As[piy][tix] = A[offset + xs[0]];
            Bs[piy][tix] = B[offset + xs[0]];
        }
        if (righ_cond)
        {
            As[piy][pix] = A[offset + xs[R2]];
            Bs[piy][pix] = B[offset + xs[R2]];
        }

        // Reduce row for last-radius 
        siy = tiy + range - RADIUS;
        piy = siy % R2;
        amu = As[piy][six];
        bmu = Bs[piy][six];
        for (int i = 1; i <= RADIUS; ++i)
        {
            amu += As[piy][six - i] + As[piy][six + i];
            bmu += Bs[piy][six - i] + Bs[piy][six + i];
        }
        Am[siy][tix] = amu;
        Bm[siy][tix] = bmu;
        __syncthreads();
    }

    // 4th -> radius range 
#pragma unroll
    for (int range = RADIUS; range < R8 - RADIUS; range += RADIUS)
    {
        // Load data to shared memory
        siy = tiy + range;
        piy = (siy + RADIUS) % R2;  // (siy + R3) % R2;
        offset = reflectBorder(iy + range + R2, height) * stride;
        As[piy][six] = A[offset + xs[RADIUS]];
        Bs[piy][six] = B[offset + xs[RADIUS]];
        if (left_cond)
        {
            As[piy][tix] = A[offset + xs[0]];
            Bs[piy][tix] = B[offset + xs[0]];
        }
        if (righ_cond)
        {
            As[piy][pix] = A[offset + xs[R2]];
            Bs[piy][pix] = B[offset + xs[R2]];
        }

        // Reduce col and save result to dst
        ny = iy + range - RADIUS;
        if (ix < width && ny < height)
        {
            piy = siy % R4;
            amu = Am[piy][tix];
            bmu = Bm[piy][tix];
            for (int i = 1; i <= RADIUS; ++i)
            {
                piy = (siy - i) % R4;
                niy = (siy + i) % R4;
                amu += Am[piy][tix] + Am[niy][tix];
                bmu += Bm[piy][tix] + Bm[niy][tix];
            }
            niy = ny * stride + ix; // niy -> index
            dst[niy] = (amu * guidiance[niy] + bmu) * coef;
        }    

        // Reduce row for last-radius
        siy += R2;
        piy = siy % R2;
        siy = siy % R4;
        amu = As[piy][six];
        bmu = Bs[piy][six];
        for (int i = 1; i <= RADIUS; ++i)
        {
            amu += As[piy][six - i] + As[piy][six + i];
            bmu += Bs[piy][six - i] + Bs[piy][six + i];
        }
        Am[siy][tix] = amu;
        Bm[siy][tix] = bmu;
        __syncthreads();
    }

    // Last 2 range
    if (ix < width)
    {
        // Reduce row for last-radius
        siy = tiy + R8 + RADIUS;
        piy = siy % R2;
        siy = siy % R4;
        amu = As[piy][six];
        bmu = Bs[piy][six];
        for (int i = 1; i <= RADIUS; ++i)
        {
            amu += As[piy][six - i] + As[piy][six + i];
            bmu += Bs[piy][six - i] + Bs[piy][six + i];
        }
        Am[siy][tix] = amu;
        Bm[siy][tix] = bmu;

        // Reduce col and save result to dst. -2
        ny = iy + R8 - R2;
        if (ny >= height) return;        
        siy = tiy + R8 - RADIUS;
        piy = siy % R4;
        amu = Am[piy][tix];
        bmu = Bm[piy][tix];
        for (int i = 1; i <= RADIUS; ++i)
        {
            piy = (siy - i) % R4;
            niy = (siy + i) % R4;
            amu += Am[piy][tix] + Am[niy][tix];
            bmu += Bm[piy][tix] + Bm[niy][tix];
        }
        niy = ny * stride + ix; // niy -> index
        dst[niy] = (amu * guidiance[niy] + bmu) * coef;

        // Reduce col and save result to dst. -1
        ny += RADIUS;
        if (ny >= height) return;        
        siy += RADIUS;
        piy = siy % R4;
        amu = Am[piy][tix];
        bmu = Bm[piy][tix];
        for (int i = 1; i <= RADIUS; ++i)
        {
            piy = (siy - i) % R4;
            niy = (siy + i) % R4;
            amu += Am[piy][tix] + Am[niy][tix];
            bmu += Bm[piy][tix] + Bm[niy][tix];
        }
        niy = ny * stride + ix; // niy -> index
        dst[niy] = (amu * guidiance[niy] + bmu) * coef;
    }
}









void hBoxFilter(float* src, float* dst, float* integral, const int4& swhcs, const int4& iwhcs, const int r)
{
	const int& width = swhcs.w;
	const int& height = swhcs.x;
	const int& channel = swhcs.y;
	const int& stride = swhcs.z;
	
	const int& iwidth = iwhcs.w;
	const int& iheight = iwhcs.x;
	const int& istride = iwhcs.z;

	// Scan row
	dim3 block1(ILEN / 2);
	dim3 grid1(height);
	const int steps1 = (width + ILEN - 1) / ILEN;
	if (channel == 1)
	{
		gScanLongRow<1> << <grid1, block1 >> > (src, integral, width, height, stride, istride, steps1);
	}
	else if (channel == 3)
	{
		gScanLongRow<3> << <grid1, block1 >> > (src, integral, width, height, stride, istride, steps1);
	}
	else
	{
		printf("gScanLongRow Do not support channel: %d\n", channel);
		return;
	}
	CHECK(cudaDeviceSynchronize());

	// Scan col
	dim3 block2(ILEN / 2);
	dim3 grid2(width * channel);
	const int steps2 = (height + ILEN - 1) / ILEN;
	gScanLongCol << <grid2, block2 >> > (integral, iwidth, iheight, channel, istride, steps2);
	CHECK(cudaDeviceSynchronize());

	// Integral to mean
	dim3 block3(X2, X2);
	dim3 grid3((width + X2 - 1) / X2, (height + X2 - 1) / X2);
	if (channel == 1)
	{
		gIntegralToMean<1> << <grid3, block3 >> > (dst, integral, width, height, stride, istride, r);
	}
	else if (channel == 3)
	{
		gIntegralToMean<3> << <grid3, block3 >> > (dst, integral, width, height, stride, istride, r);
	}
	else
	{
		printf("gIntegralToMean Do not support channel: %d\n", channel);
		return;
	}
	
	CHECK(cudaDeviceSynchronize());
	CheckMsg("hBoxFilter() execution failed\n");
}


void hMultiply(float* a, float* b, float* c, const int4& awhcs, const int4& bwhcs)
{
	const int& width = awhcs.w;
	const int& height = awhcs.x;
	const int& channel1 = awhcs.y;
	const int& stride1 = awhcs.z;
	const int& channel2 = bwhcs.y;
	const int& stride2 = bwhcs.z;

	dim3 block(X2, X2);
	dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
	if (channel1 == channel2)
	{
		gMultiply<<<grid, block>>>(a, b, c, width, height, channel1, stride1);
	}
	else if (channel2 == 1)
	{
		gMultiplyCN1<<<grid, block>>>(a, b, c, width, height, channel1, stride1, stride2);
	}
	else
	{
		printf("gMultiply Do not support channel: %d, %d\n", channel1, channel2);
		return;
	}

	CHECK(cudaDeviceSynchronize());
	CheckMsg("hMultiply() execution failed\n");
}


void hCalcA(float* a, float* pm, float* im, float* ipm, float* iim, const int4& swhcs, const int4& gwhcs, const float eps)
{
	const int& width = swhcs.w;
	const int& height = swhcs.x;
	const int& channel1 = swhcs.y;
	const int& stride1 = swhcs.z;
	const int& channel2 = gwhcs.y;
	const int& stride2 = gwhcs.z;

	dim3 block(X2, X2);
	dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
	if (channel1 == channel2)
	{
		gCalcA<<<grid, block>>>(a, pm, im, ipm, iim, eps, width, height, channel1, stride1);
	}
	else if (channel2 == 1)
	{
		gCalcACN1<<<grid, block>>>(a, pm, im, ipm, iim, eps, width, height, channel1, stride1, stride2);
	}
	else
	{
		printf("gCalcA Do not support channel: %d, %d\n", channel1, channel2);
		return;
	}

	CHECK(cudaDeviceSynchronize());
	CheckMsg("hCalcA() execution failed\n");
}


void hCalcB(float* b, float* a, float* pm, float* im, const int4& swhcs, const int4& gwhcs)
{
	const int& width = swhcs.w;
	const int& height = swhcs.x;
	const int& channel1 = swhcs.y;
	const int& stride1 = swhcs.z;
	const int& channel2 = gwhcs.y;
	const int& stride2 = gwhcs.z;

	dim3 block(X2, X2);
	dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
	if (channel1 == channel2)
	{
		gCalcB<<<grid, block>>>(b, a, pm, im, width, height, channel1, stride1);
	}
	else if (channel2 == 1)
	{
		gCalcBCN1<<<grid, block>>>(b, a, pm, im, width, height, channel1, stride1, stride2);
	}
	else
	{
		printf("gCalcB Do not support channel: %d, %d\n", channel1, channel2);
		return;
	}

	CHECK(cudaDeviceSynchronize());
	CheckMsg("hCalcB() execution failed\n");
}


void hLinearTransform(float* src, float* dst, float* a, float* b, const int4& swhcs, const int4& dwhcs)
{
	const int& width = dwhcs.w;
	const int& height = dwhcs.x;
	const int& channel1 = dwhcs.y;
	const int& stride1 = dwhcs.z;
	const int& channel2 = swhcs.y;
	const int& stride2 = swhcs.z;

	dim3 block(X2, X2);
	dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
	if (channel1 == channel2)
	{
		gLinearTransform<<<grid, block>>>(src, dst, a, b, width, height, channel1, stride1);
	}
	else if (channel2 == 1)
	{
		gLinearTransformCN1<<<grid, block>>>(src, dst, a, b, width, height, channel1, stride1, stride2);
	}
	else
	{
		printf("gLinearTransform Do not support channel: %d, %d\n", channel1, channel2);
		return;
	}

	CHECK(cudaDeviceSynchronize());
	CheckMsg("hLinearTransform() execution failed\n");
}


void hGuidedFilter(float* d_guided, float* d_src, float* d_dst, float* d_A, float* d_B, float eps, int radius, int width, int height, int stride)
{	
    dim3 block(1, radius);
    dim3 grid(1, iDivUp(height, 8 * radius));
	const int ksz = 2 * radius + 1;
    const float coef = 1.f / static_cast<float>(ksz * ksz);
	switch (radius)
	{
	case 1:
		block.x = 128; grid.x = iDivUp(width, block.x);
		gCalcAB<1, 128><<<grid, block>>>(d_guided, d_src, d_A, d_B, eps, coef, width, height, stride);
		gWeightByABm<1, 128><<<grid, block>>>(d_guided, d_A, d_B, d_dst, coef, width, height, stride);
		break;
	case 2:
		block.x = 64; grid.x = iDivUp(width, block.x);
		gCalcAB<2, 64><<<grid, block>>>(d_guided, d_src, d_A, d_B, eps, coef, width, height, stride);
		gWeightByABm<2, 64><<<grid, block>>>(d_guided, d_A, d_B, d_dst, coef, width, height, stride);
		break;
	case 3:
		block.x = 64; grid.x = iDivUp(width, block.x);
		gCalcAB<3, 64><<<grid, block>>>(d_guided, d_src, d_A, d_B, eps, coef, width, height, stride);
		gWeightByABm<3, 64><<<grid, block>>>(d_guided, d_A, d_B, d_dst, coef, width, height, stride);
		break;
	case 4:
		block.x = 64; grid.x = iDivUp(width, block.x);
		gCalcAB<4, 64><<<grid, block>>>(d_guided, d_src, d_A, d_B, eps, coef, width, height, stride);
		gWeightByABm<4, 64><<<grid, block>>>(d_guided, d_A, d_B, d_dst, coef, width, height, stride);
		break;
	case 5:
		block.x = 32; grid.x = iDivUp(width, block.x);
		gCalcAB<5, 32><<<grid, block>>>(d_guided, d_src, d_A, d_B, eps, coef, width, height, stride);
		gWeightByABm<5, 32><<<grid, block>>>(d_guided, d_A, d_B, d_dst, coef, width, height, stride);
		break;
	case 6:
		block.x = 32; grid.x = iDivUp(width, block.x);
		gCalcAB<6, 32><<<grid, block>>>(d_guided, d_src, d_A, d_B, eps, coef, width, height, stride);
		gWeightByABm<6, 32><<<grid, block>>>(d_guided, d_A, d_B, d_dst, coef, width, height, stride);
		break;
	case 7:
		block.x = 32; grid.x = iDivUp(width, block.x);
		gCalcAB<7, 32><<<grid, block>>>(d_guided, d_src, d_A, d_B, eps, coef, width, height, stride);
		gWeightByABm<7, 32><<<grid, block>>>(d_guided, d_A, d_B, d_dst, coef, width, height, stride);
		break;
	default:
		break;
	}
}
