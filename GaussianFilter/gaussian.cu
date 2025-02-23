#include "gaussian.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafilters.hpp>
#include <nppi.h>


__constant__ float d_const_kernel[256];



__global__ void gWarmUp(float* src, float* dst, int width, int height, int stride)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < width && iy < height)
	{
		unsigned int idx = iy * stride + ix;
		float v = src[idx];
		dst[idx] = (v < 128.f) ? (v * 10.f) : (v * 0.1f);
	}
}


__global__ void gGaussNaive(float* src, float* dst, float* kernel, int radius, int width, int height, int stride)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height)
    {
        float wsum = 0.f;
        int kidx = 0;
        for (int yoffset = -radius; yoffset <= radius; yoffset++)
        {
            const int ny = reflectBorder(iy + yoffset, height);
            const float* psrc = src + ny * stride;
            for (int xoffset = -radius; xoffset <= radius; xoffset++)
            {
                const int nx = reflectBorder(ix + xoffset, width);
                wsum += psrc[nx] * kernel[kidx];
                kidx++;
            }
        }
        dst[iy * stride + ix] = wsum;
    }
}


__global__ void gGaussConst(const float* __restrict__ src, float* __restrict__ dst, int radius, int width, int height, int stride)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < width && iy < height)
    {
        float wsum = 0.f;
        int kidx = 0;
        for (int yoffset = -radius; yoffset <= radius; yoffset++)
        {
            const int ny = reflectBorder(iy + yoffset, height);
            const float* psrc = src + ny * stride;
            for (int xoffset = -radius; xoffset <= radius; xoffset++)
            {
                const int nx = reflectBorder(ix + xoffset, width);
                wsum += psrc[nx] * d_const_kernel[kidx];
                kidx++;
            }
        }
        dst[iy * stride + ix] = wsum;
    }
}


__global__ void gGaussShare(const float* __restrict__ src, float* __restrict__ dst, int radius, int width, int height, int stride)
{
    extern __shared__ float smem[];
    const int tix = threadIdx.x;
    const int tiy = threadIdx.y;
    const int ix = blockIdx.x * blockDim.x + tix;
    const int iy = blockIdx.y * blockDim.y + tiy;
    const int ix0 = reflectBorder(ix - radius, width);
    const int ix1 = reflectBorder(ix, width);
    const int ix2 = reflectBorder(ix + radius, width);
    const int iy0 = reflectBorder(iy - radius, height);
    const int iy1 = reflectBorder(iy, height);
    const int iy2 = reflectBorder(iy + radius, height);
    const int r2 = 2 * radius;
    const int six = tix + radius;
    const int pix = tix + r2;
    const int siy = tiy + radius;
    const int piy = tiy + r2;
    const int scol = blockDim.x + r2;
    // const int srow = blockDim.y + r2;
    const bool lef_cond = tix < radius;
    const bool rig_cond = six >= blockDim.x || ix + radius >= width;
    const bool top_cond = tiy < radius;
    const bool bot_cond = siy >= blockDim.y || iy + radius >= height;
    
    // Copy data to shared memory
    smem[siy * scol + six] = src[iy1 * stride + ix1];
    if (lef_cond) smem[siy * scol + tix] = src[iy1 * stride + ix0];
    if (rig_cond) smem[siy * scol + pix] = src[iy1 * stride + ix2];
    if (top_cond) smem[tiy * scol + six] = src[iy0 * stride + ix1];
    if (bot_cond) smem[piy * scol + six] = src[iy2 * stride + ix1];
    if (lef_cond && top_cond) smem[tiy * scol + tix] = src[iy0 * stride + ix0];
    if (lef_cond && bot_cond) smem[piy * scol + tix] = src[iy2 * stride + ix0];
    if (rig_cond && top_cond) smem[tiy * scol + pix] = src[iy0 * stride + ix2];
    if (rig_cond && bot_cond) smem[piy * scol + pix] = src[iy2 * stride + ix2];
    __syncthreads();

    // Computation
    if (ix < width && iy < height)
    {
        float wsum = 0.f;
        int kidx = 0;
        for (int yoffset = -radius; yoffset <= radius; yoffset++)
        {
            const float* sdata = smem + (siy + yoffset) * scol;
            for (int xoffset = -radius; xoffset <= radius; xoffset++)
            {
                wsum += sdata[six + xoffset] * d_const_kernel[kidx];
                kidx++;
            }
        }
        dst[iy * stride + ix] = wsum;
    }
}


__global__ void gGaussSplit(const float* __restrict__ src, float* __restrict__ dst, int radius, int width, int height, int stride)
{
    extern __shared__ float smem[];
    const int tix = threadIdx.x;
    const int tiy = threadIdx.y;
    const int ix = blockIdx.x * blockDim.x + tix;
    const int iy = blockIdx.y * blockDim.y + tiy;

    // Weighted by row
    const int siy = tiy + radius;    
    const int sidx = siy * blockDim.x + tix;
    const float* psrc = src + reflectBorder(iy, height) * stride;
    smem[sidx] = psrc[reflectBorder(ix, width)] * d_const_kernel[0];
    for (int i = 1; i <= radius; i++)
    {
        smem[sidx] += (psrc[reflectBorder(ix - i, width)] + psrc[reflectBorder(ix + i, width)]) * d_const_kernel[i];
    }
    if (tiy < radius)   // Top
    {
        const int tidx = tiy * blockDim.x + tix;
        const float* tsrc = src + reflectBorder(iy - radius, height) * stride;
        smem[tidx] = tsrc[reflectBorder(ix, width)] * d_const_kernel[0];
        for (int i = 1; i <= radius; i++)
        {
            smem[tidx] += (tsrc[reflectBorder(ix - i, width)] + tsrc[reflectBorder(ix + i, width)]) * d_const_kernel[i];
        }
    }
    if (siy >= blockDim.y || iy + radius >= height) // Bottom
    {
        const int bidx = (siy + radius) * blockDim.x + tix;
        const float* bsrc = src + reflectBorder(iy + radius, height) * stride;
        smem[bidx] = bsrc[reflectBorder(ix, width)] * d_const_kernel[0];
        for (int i = 1; i <= radius; i++)
        {
            smem[bidx] += (bsrc[reflectBorder(ix - i, width)] + bsrc[reflectBorder(ix + i, width)]) * d_const_kernel[i];
        }
    }
    __syncthreads();

    // Weight by column
    if (ix < width && iy < height)
    {
        float wsum = smem[sidx] * d_const_kernel[0];
        for (int i = 1; i <= radius; i++)
        {
            wsum += (smem[sidx - i * blockDim.x] + smem[sidx + i * blockDim.x]) * d_const_kernel[i];
        }        
        dst[iy * stride + ix] = wsum;
    }
}


template <int RADIUS, int KX>
__global__ void gGaussOptim(const float* __restrict__ src, float* __restrict__ dst, int width, int height, int stride)
{
    __shared__ float pmem[RADIUS * 2][KX + RADIUS * 2];
    __shared__ float smem[RADIUS * 4][KX];

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

    // First radius range - Load data to shared memory
    int offset = reflectBorder(iy - RADIUS, height) * stride;
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
        offset = reflectBorder(iy + range - RADIUS, height) * stride;
        pmem[piy][six] = src[offset + xs[RADIUS]];
        if (left_cond) pmem[piy][tix] = src[offset + xs[0]];
        if (righ_cond) pmem[piy][pix] = src[offset + xs[R2]];

        // Reduce row for last-radius 
        int siy = tiy + range - RADIUS;
        piy = siy % R2;
        float res = __fmul_rn(d_const_kernel[0], pmem[piy][six]);
        for (int i = 1; i <= RADIUS; ++i)
        {
            res += __fmul_rn(d_const_kernel[i], pmem[piy][six - i] + pmem[piy][six + i]);
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
        offset = reflectBorder(iy + range + R2, height) * stride;
        pmem[piy][six] = src[offset + xs[RADIUS]];
        if (left_cond) pmem[piy][tix] = src[offset + xs[0]];
        if (righ_cond) pmem[piy][pix] = src[offset + xs[R2]];

        // Reduce col and save result to dst
        int ny = iy + range - RADIUS;
        if (ix < width && ny < height)
        {
            float res = __fmul_rn(d_const_kernel[0], smem[siy % R4][tix]);
            for (int i = 1; i <= RADIUS; ++i)
            {
                res += __fmul_rn(d_const_kernel[i], smem[(siy - i) % R4][tix] + smem[(siy + i) % R4][tix]);
            }
            dst[ny * stride + ix] = res;
        }    

        // Reduce row for last-radius
        siy += R2;
        piy = siy % R2;
        float res = __fmul_rn(d_const_kernel[0], pmem[piy][six]);
        for (int i = 1; i <= RADIUS; ++i)
        {
            res += __fmul_rn(d_const_kernel[i], pmem[piy][six - i] + pmem[piy][six + i]);
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
        float res = __fmul_rn(d_const_kernel[0], pmem[piy][six]);
        for (int i = 1; i <= RADIUS; ++i)
        {
            res += __fmul_rn(d_const_kernel[i], pmem[piy][six - i] + pmem[piy][six + i]);
        }
        smem[siy % R4][tix] = res;

        int ny = iy + R8 - R2;
        if (ny >= height) return;        
        siy = tiy + R8 - RADIUS;
        res = __fmul_rn(d_const_kernel[0], smem[siy % R4][tix]);
        for (int i = 1; i <= RADIUS; ++i)
        {
            res += __fmul_rn(d_const_kernel[i], smem[(siy - i) % R4][tix] + smem[(siy + i) % R4][tix]);
        }
        dst[ny * stride + ix] = res;

        ny += RADIUS;
        if (ny >= height) return;        
        siy += RADIUS;
        res = __fmul_rn(d_const_kernel[0], smem[siy % R4][tix]);
        for (int i = 1; i <= RADIUS; ++i)
        {
            res += __fmul_rn(d_const_kernel[i], smem[(siy - i) % R4][tix] + smem[(siy + i) % R4][tix]);
        }
        dst[ny * stride + ix] = res;
    }
}


void calcMaxOccupancyGridBlock(dim3& block, dim3& grid, const int width, const int height)
{
    // 每个SM的最大线程数: 1536; 设置总线程数小于等于512可以被整除
    float max_occupancy = 0.f;
    float min_whr = FLT_MAX;
    const float area = static_cast<float>(width * height);
    for (int total_threads = 512; total_threads >= 128; total_threads >>= 1)
    {
        for (int bx = total_threads; bx >= 1; bx >>= 1)
        {
            const int by = total_threads / bx;
            const int gx = iDivUp(width, bx);
            const int gy = iDivUp(height, by);
            const float whr = static_cast<float>(max(bx, by)) / static_cast<float>(min(bx, by));
            const float active_occupancy = area / static_cast<float>(gx * bx * gy * by);
            if (active_occupancy > max_occupancy)
            {
                min_whr = whr;
                max_occupancy = active_occupancy;
                block.x = bx; block.y = by;
                grid.x = gx; grid.y = gy;
            }
            else if (abs(active_occupancy - max_occupancy) < 1e-6f && 
                bx * by == block.x * block.y && 
                whr < min_whr)
            {
                min_whr = whr;
                max_occupancy = active_occupancy;
                block.x = bx; block.y = by;
                grid.x = gx; grid.y = gy;
            }
        }
    }
}


void calcMaxOccupancyGridBlockWithShare(dim3& block, dim3& grid, const int width, const int height, const int radius, int (*func)(int, int, int))
{
    // 每个SM的最大线程数: 1536; 设置总线程数小于等于512可以被整除
    float max_occupancy = 0.f;
    float min_whr = FLT_MAX;
    const float area = static_cast<float>(width * height);
    for (int total_threads = 512; total_threads >= 128; total_threads >>= 1)
    {
        const int blocks_per_sm = 1536 / total_threads;
        for (int bx = total_threads; bx >= 32; bx >>= 1)
        {
            const int by = total_threads / bx;
            const int gx = iDivUp(width, bx);
            const int gy = iDivUp(height, by);
            const int snum = func(bx, by, radius);
            const int sbytes = static_cast<int>(snum * sizeof(float));
            const int blocks_can_launched_per_sm = 100 * 1024 / sbytes;
            const float sm_occupancy = fminf(1.f, static_cast<float>(blocks_can_launched_per_sm) / static_cast<float>(blocks_per_sm));
            const float share_occupancy = static_cast<float>(bx * by) / static_cast<float>(snum);
            const float active_occupancy = fminf(sm_occupancy, fminf(share_occupancy, area / static_cast<float>(gx * bx * gy * by)));
            const float whr = static_cast<float>(max(bx, by)) / static_cast<float>(min(bx, by));
            if (active_occupancy > max_occupancy)
            {
                min_whr = whr;
                max_occupancy = active_occupancy;
                block.x = bx; block.y = by;
                grid.x = gx; grid.y = gy;
            }
            else if (abs(active_occupancy - max_occupancy) < 1e-6f && 
                bx * by == block.x * block.y && 
                whr < min_whr)
            {
                min_whr = whr;
                max_occupancy = active_occupancy;
                block.x = bx; block.y = by;
                grid.x = gx; grid.y = gy;
            }
        }
    }
}


double calcMaxAbsDiff(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat diff;
    double max_diff = DBL_MAX;
    cv::absdiff(a, b, diff);
    cv::minMaxLoc(diff, NULL, &max_diff);
    return max_diff;
}


inline int pad2Area(int x, int y, int r)
{
    return (x + r + r) * (y + r + r);
}


inline int pad1Area(int x, int y, int r)
{
    return x * (y + r + r);
}


void gaussianComparasion(int argc, char** argv)
{
    // Parse config
    int width = 3840;
    int height = 2160;
    int radius = 1;
    float sigma = 0.5f;
    int nrepeats = 1;
    std::string src_path = "";  // ../data/sample.png
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) radius = atoi(argv[3]);
    if (argc > 4) sigma = atof(argv[4]);
    if (argc > 5) nrepeats = atoi(argv[5]);
    if (argc > 6) src_path = argv[6];

    // Random generate on host
    cv::Mat image, h_src;
    if (src_path.empty())
    {
        cv::RNG rng;
        h_src = cv::Mat(height, width, CV_32FC1);
        rng.fill(h_src, rng.UNIFORM, 0, 1, true);
    }
    else
    {
        image = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
        image.convertTo(h_src, CV_32F, 1.0 / 255.0);
        // cv::resize(h_src, h_src, cv::Size(width, height));
        height = h_src.rows;
        width = h_src.cols;
    }

    // Generate gauss kernel
    const int ksz = 2 * radius + 1;
    const cv::Size kernel_size(ksz, ksz);
    const cv::Mat kernel_1d = cv::getGaussianKernel(ksz, sigma, CV_32FC1);
    const cv::Mat kernel_2d = kernel_1d * kernel_1d.t();

    // Gaussian blurring on host
    cv::Mat h_dst;
    cv::GaussianBlur(h_src, h_dst, kernel_size, sigma, sigma);

    // Allocate data on device
    std::unique_ptr<float, DeviceBufferDeleter> d_pk1d(nullptr);
    std::unique_ptr<float, DeviceBufferDeleter> d_pk2d(nullptr);
    std::unique_ptr<float, DeviceBufferDeleter> d_src(nullptr);
    std::unique_ptr<float, DeviceBufferDeleter> d_dst_naive(nullptr);
    std::unique_ptr<float, DeviceBufferDeleter> d_dst_const(nullptr);
    std::unique_ptr<float, DeviceBufferDeleter> d_dst_share(nullptr);
    std::unique_ptr<float, DeviceBufferDeleter> d_dst_split(nullptr);
    std::unique_ptr<float, DeviceBufferDeleter> d_dst_optim(nullptr);
    std::unique_ptr<float, DeviceBufferDeleter> d_dst_nppi(nullptr);
    cv::cuda::GpuMat d_src_mat(height, width, CV_32FC1);
    cv::cuda::GpuMat d_dst_mat(height, width, CV_32FC1);
    size_t spitch = width * sizeof(float);
    size_t dpitch = 0;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pk1d), ksz * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pk2d), ksz * ksz * sizeof(float)));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst_naive), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst_const), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst_share), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst_split), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst_optim), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst_nppi), &dpitch, spitch, height));
    d_src_mat.upload(h_src);
    const int stride = static_cast<int>(dpitch / sizeof(float));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_pk1d.get(), kernel_1d.data, ksz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pk2d.get(), kernel_2d.data, ksz * ksz * sizeof(float), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_src.get(), h_src.data, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy2D(d_src.get(), dpitch, h_src.data, spitch, spitch, height, cudaMemcpyHostToDevice));

	// Warm up
	dim3 block_warm(32, 32);
	dim3 grid_warm(iDivUp(width, block_warm.x), iDivUp(height, block_warm.y));
	for (int i = 0; i < 10; i++)
	{
		gWarmUp<<<grid_warm, block_warm>>>(d_src.get(), d_dst_naive.get(), width, height, stride);
	}
	CHECK(cudaDeviceSynchronize());
	
    // Naive gauss
    dim3 block_naive, grid_naive;
    calcMaxOccupancyGridBlock(block_naive, grid_naive, width, height);
    printf("Launch config of naive gauss: block(%d, %d), grid(%d, %d)\n", block_naive.x, block_naive.y, grid_naive.x, grid_naive.y);
    GpuTimer timer(0);
    for (int i = 0; i < nrepeats; i++)
    {
        gGaussNaive<<<grid_naive, block_naive>>>(d_src.get(), d_dst_naive.get(), d_pk2d.get(), radius, width, height, stride);
    }
    CHECK(cudaDeviceSynchronize());
    float t_naive_elapsed = timer.read();

    // Gauss using constant memory
    dim3 block_const = block_naive;
    dim3 grid_const = grid_naive;
    CHECK(cudaMemcpyToSymbol(d_const_kernel, d_pk2d.get(), ksz * ksz * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    printf("Launch config of const gauss: block(%d, %d), grid(%d, %d)\n", block_const.x, block_const.y, grid_const.x, grid_const.y);
    float t_const_start = timer.read();
    for (int i = 0; i < nrepeats; i++)
    {
        gGaussConst<<<grid_const, block_const>>>(d_src.get(), d_dst_const.get(), radius, width, height, stride);
    }
    CHECK(cudaDeviceSynchronize());
    float t_const_elapsed = timer.read() - t_const_start;
    
    // Gauss using shared memory
    dim3 block_share, grid_share;
    calcMaxOccupancyGridBlockWithShare(block_share, grid_share, width, height, radius, pad2Area);
    printf("Launch config of share gauss: block(%d, %d), grid(%d, %d)\n", block_share.x, block_share.y, grid_share.x, grid_share.y);
    size_t sbytes = pad2Area(block_share.x, block_share.y, radius) * sizeof(float);
    float t_share_start = timer.read();
    for (int i = 0; i < nrepeats; i++)
    {
        gGaussShare<<<grid_share, block_share, sbytes>>>(d_src.get(), d_dst_share.get(), radius, width, height, stride);
    }
    CHECK(cudaDeviceSynchronize());
    float t_share_elapsed = timer.read() - t_share_start;

    // Gauss using split kernel
    dim3 block_split, grid_split;
    CHECK(cudaMemcpyToSymbol(d_const_kernel, d_pk1d.get() + radius, (radius + 1) * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    calcMaxOccupancyGridBlockWithShare(block_split, grid_split, width, height, radius, pad1Area);
    printf("Launch config of split gauss: block(%d, %d), grid(%d, %d)\n", block_split.x, block_split.y, grid_split.x, grid_split.y);
    sbytes = pad1Area(block_split.x, block_split.y, radius) * sizeof(float);
    float t_split_start = timer.read();
    for (int i = 0; i < nrepeats; i++)
    {
        gGaussSplit<<<grid_split, block_split, sbytes>>>(d_src.get(), d_dst_split.get(), radius, width, height, stride);
    }
    CHECK(cudaDeviceSynchronize()); 
    float t_split_elapsed = timer.read() - t_split_start;

    // Gauss using reusable shared memory
    dim3 block_optim(1, radius), grid_optim(1, iDivUp(height, 8 * radius));
    float t_optim_start = timer.read();
    for (int i = 0; i < nrepeats; i++)
    {
        switch (radius)
        {
        case 1:
            block_optim.x = 128; grid_optim.x = iDivUp(width, block_optim.x);
            gGaussOptim<1, 128><<<grid_optim, block_optim>>>(d_src.get(), d_dst_optim.get(), width, height, stride);
            break;
        case 2:
            block_optim.x = 128; grid_optim.x = iDivUp(width, block_optim.x);
            gGaussOptim<2, 128><<<grid_optim, block_optim>>>(d_src.get(), d_dst_optim.get(), width, height, stride);
            break;
        case 3:
            block_optim.x = 128; grid_optim.x = iDivUp(width, block_optim.x);
            gGaussOptim<3, 128><<<grid_optim, block_optim>>>(d_src.get(), d_dst_optim.get(), width, height, stride);
            break;
        case 4:
            block_optim.x = 128; grid_optim.x = iDivUp(width, block_optim.x);
            gGaussOptim<4, 128><<<grid_optim, block_optim>>>(d_src.get(), d_dst_optim.get(), width, height, stride);
            break;
        case 5:
            block_optim.x = 64; grid_optim.x = iDivUp(width, block_optim.x);
            gGaussOptim<5, 64><<<grid_optim, block_optim>>>(d_src.get(), d_dst_optim.get(), width, height, stride);
            break;
        case 6:
            block_optim.x = 64; grid_optim.x = iDivUp(width, block_optim.x);
            gGaussOptim<6, 64><<<grid_optim, block_optim>>>(d_src.get(), d_dst_optim.get(), width, height, stride);
            break;
        case 7:
            block_optim.x = 64; grid_optim.x = iDivUp(width, block_optim.x);
            gGaussOptim<7, 64><<<grid_optim, block_optim>>>(d_src.get(), d_dst_optim.get(), width, height, stride);
            break;
        default:
            break;
        }
    }
    printf("Launch config of optim gauss: block(%d, %d), grid(%d, %d)\n", block_optim.x, block_optim.y, grid_optim.x, grid_optim.y);
    CHECK(cudaDeviceSynchronize());   
    float t_optim_elapsed = timer.read() - t_optim_start;

    // Gauss by OpenCV-CUDA
    cv::Ptr<cv::cuda::Filter> cvcu_gauss_filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, kernel_size, sigma, sigma);
    float t_cvcu_start = timer.read();
    for (int i = 0; i < nrepeats; i++)
    {
        cvcu_gauss_filter->apply(d_src_mat, d_dst_mat);
    }
    CHECK(cudaDeviceSynchronize());
    float t_cvcu_elapsed = timer.read() - t_cvcu_start;

    // Gauss by NPPI
    NppiSize src_roi{width, height};
    NppiPoint src_offset{0, 0};
    float t_nppi_start = timer.read();
    for (int i = 0; i < nrepeats; i++)
    {
        nppiFilterGaussAdvancedBorder_32f_C1R(d_src.get(), dpitch, src_roi, src_offset, d_dst_nppi.get(), dpitch, src_roi, ksz, d_pk1d.get(), NppiBorderType::NPP_BORDER_MIRROR);
    }
    CHECK(cudaDeviceSynchronize());
    float t_nppi_elapsed = timer.read() - t_nppi_start;
    
    // Copy data from device to host
    cv::Mat h_dst_naive(height, width, CV_32FC1);
    cv::Mat h_dst_const(height, width, CV_32FC1);
    cv::Mat h_dst_share(height, width, CV_32FC1);
    cv::Mat h_dst_split(height, width, CV_32FC1);
    cv::Mat h_dst_optim(height, width, CV_32FC1);
    cv::Mat h_dst_nppi(height, width, CV_32FC1);
    cv::Mat h_dst_cvcu;
    CHECK(cudaMemcpy2D(h_dst_naive.data, spitch, d_dst_naive.get(), dpitch, spitch, height, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2D(h_dst_const.data, spitch, d_dst_const.get(), dpitch, spitch, height, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2D(h_dst_share.data, spitch, d_dst_share.get(), dpitch, spitch, height, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2D(h_dst_split.data, spitch, d_dst_split.get(), dpitch, spitch, height, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2D(h_dst_optim.data, spitch, d_dst_optim.get(), dpitch, spitch, height, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2D(h_dst_nppi.data, spitch, d_dst_nppi.get(), dpitch, spitch, height, cudaMemcpyDeviceToHost));
    d_dst_mat.download(h_dst_cvcu);

    // Verify output
    double maxdiff_naive = calcMaxAbsDiff(h_dst, h_dst_naive);
    double maxdiff_const = calcMaxAbsDiff(h_dst, h_dst_const);
    double maxdiff_share = calcMaxAbsDiff(h_dst, h_dst_share);
    double maxdiff_split = calcMaxAbsDiff(h_dst, h_dst_split);
    double maxdiff_optim = calcMaxAbsDiff(h_dst, h_dst_optim);
    double maxdiff_nppi = calcMaxAbsDiff(h_dst, h_dst_nppi);
    double maxdiff_cvcu = calcMaxAbsDiff(h_dst, h_dst_cvcu);
    printf("max difference of naive gauss: %lf, time cost: %fms\n", maxdiff_naive, t_naive_elapsed / nrepeats);
    printf("max difference of const gauss: %lf, time cost: %fms\n", maxdiff_const, t_const_elapsed / nrepeats);
    printf("max difference of share gauss: %lf, time cost: %fms\n", maxdiff_share, t_share_elapsed / nrepeats);
    printf("max difference of split gauss: %lf, time cost: %fms\n", maxdiff_split, t_split_elapsed / nrepeats);
    printf("max difference of optim gauss: %lf, time cost: %fms\n", maxdiff_optim, t_optim_elapsed / nrepeats);
    printf("max difference of NPPI gauss: %lf, time cost: %fms\n", maxdiff_nppi, t_nppi_elapsed / nrepeats);
    printf("max difference of OpenCV-CUDA: %lf, time cost: %fms\n", maxdiff_cvcu, t_cvcu_elapsed / nrepeats);

    if (!src_path.empty())
    {
        h_src.convertTo(h_src, CV_8U, 255.0);
        h_dst.convertTo(h_dst, CV_8U, 255.0);
        h_dst_naive.convertTo(h_dst_naive, CV_8U, 255.0);
        h_dst_const.convertTo(h_dst_const, CV_8U, 255.0);
        h_dst_share.convertTo(h_dst_share, CV_8U, 255.0);
        h_dst_split.convertTo(h_dst_split, CV_8U, 255.0);
        h_dst_optim.convertTo(h_dst_optim, CV_8U, 255.0);
        h_dst_nppi.convertTo(h_dst_nppi, CV_8U, 255.0);
        h_dst_cvcu.convertTo(h_dst_cvcu, CV_8U, 255.0);
        std::string base_path = src_path.substr(0, src_path.rfind("."));
        cv::imwrite(base_path + "_gray.png", h_src);
        cv::imwrite(base_path + "_cv.png", h_dst);
        cv::imwrite(base_path + "_naive.png", h_dst_naive);
        cv::imwrite(base_path + "_const.png", h_dst_const);
        cv::imwrite(base_path + "_share.png", h_dst_share);
        cv::imwrite(base_path + "_split.png", h_dst_split);
        cv::imwrite(base_path + "_optim.png", h_dst_optim);
        cv::imwrite(base_path + "_nppi.png", h_dst_nppi);
        cv::imwrite(base_path + "_cvcu.png", h_dst_cvcu);
    }
}


int main(int argc, char** argv)
{
	gaussianComparasion(argc, argv);

	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}


