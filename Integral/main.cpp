#include "integral_d.h"
#include <nppi.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>


/* Use this function to compare speed with OpenCV and NPPI */
void integralDemo(int argc, char** argv);
/* Use this function to verify the result */
void autoTestDemo(int argc, char** argv);


int main(int argc, char** argv)
{
    integralDemo(argc, argv);
    // autoTestDemo(argc, argv);
    
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}


void integralDemo(int argc, char** argv)
{
    int width = 3840;
    int height = 2160;
    int nrepeats = 1;
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) nrepeats = atoi(argv[3]);

    // Initialize image
    cv::Mat image = cv::Mat::ones(height, width, CV_8UC1);
    cv::RNG rng;
    rng.fill(image, rng.UNIFORM, 0, 256, true);

    // Integral on host
    cv::Mat h_integral;
    cv::integral(image, h_integral);

    // Allocate memory
    unsigned char* d_src = nullptr;
    int* d_nppi_res = nullptr;
    int* d_cuda_res = nullptr;
    int* d_cuda_res2 = nullptr;
    int* d_temp_buf = nullptr;
    const int align_height = iDivUp(height, 4) * 4;
    const int align_width = iDivUp(width, 4) * 4;
    size_t sbytes = height * width * sizeof(unsigned char);
    size_t dbytes = (height + 1) * (width + 1) * sizeof(int);
    size_t dbytes2 = height * width * sizeof(int);
    size_t dbytes3 = align_height * align_width * sizeof(int);
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_src), sbytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nppi_res), dbytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cuda_res), dbytes2));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cuda_res2), dbytes3));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buf), dbytes2));
    CHECK(cudaMemcpy(d_src, image.data, sbytes, cudaMemcpyHostToDevice));

    cv::cuda::GpuMat d_image(height, width, CV_8UC1, d_src);
    cv::cuda::GpuMat d_integral(height + 1, width + 1, CV_32SC1);

    // NPPI
    NppiSize osz;
    osz.width = width;
    osz.height = height;
    int sstep = width * sizeof(unsigned char);
    int dstep = (width + 1) * sizeof(int);
    GpuTimer timer(0);
    for (int i = 0; i < nrepeats; i++)
    {
        nppiIntegral_8u32s_C1R(d_src, sstep, d_nppi_res, dstep, osz, 0);
    }
    CHECK(cudaDeviceSynchronize());
    float t0 = timer.read();

    // OpenCV
    for (int i = 0; i < nrepeats; i++)
    {
        cv::cuda::integral(d_image, d_integral);
    }
    CHECK(cudaDeviceSynchronize());
    float t1 = timer.read();

    // CUDA
    for (int i = 0; i < nrepeats; i++)
    {
        hIntegral(d_src, d_cuda_res, d_temp_buf, width, height, width, width);
    }
    CHECK(cudaDeviceSynchronize());
    float t2 = timer.read();

    // CUDA 2
    for (int i = 0; i < nrepeats; i++)
    {
        hAligned4Integral(d_src, d_cuda_res2, width, height, width, align_width, align_height);
    }
    CHECK(cudaDeviceSynchronize());
    float t3 = timer.read();

    // Copy to host
    cv::Mat h_nppi_res(height + 1, width + 1, CV_32SC1);
    cv::Mat h_cv_res(height + 1, width + 1, CV_32SC1);
    cv::Mat h_cuda_res(height, width, CV_32SC1);
    cv::Mat h_cuda_res2(align_height, align_width, CV_32SC1);
    CHECK(cudaMemcpy(h_nppi_res.data, d_nppi_res, dbytes, cudaMemcpyDeviceToHost));
    d_integral.download(h_cv_res);
    CHECK(cudaMemcpy(h_cuda_res.data, d_cuda_res, height * width * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_cuda_res2.data, d_cuda_res2, align_height * align_width * sizeof(int), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 16; i++)
    // {
    //     for (int j = 0; j < 16; j++)
    //     {
    //         printf("%5d ", h_cuda_res2.at<int>(i, j));
    //     }
    //     printf("\n");
    // }
    
    // Find max absdiff
    cv::Mat nppi_diff, cv_diff, cuda_diff, cuda_diff2;
    cv::absdiff(h_integral, h_nppi_res, nppi_diff);
    cv::absdiff(h_integral, h_cv_res, cv_diff);
    cv::absdiff(h_integral.rowRange(1, height + 1).colRange(1, width + 1), h_cuda_res, cuda_diff);
    cv::absdiff(h_integral.rowRange(1, height + 1).colRange(1, width + 1), h_cuda_res2.rowRange(0, height).colRange(0, width), cuda_diff2);

    double nppi_maxdiff = 1000000;
    double cv_maxdiff = 1000000;
    double cuda_maxdiff = 1000000;
    double cuda_maxdiff2 = 1000000;
    cv::minMaxLoc(nppi_diff, NULL, &nppi_maxdiff);
    cv::minMaxLoc(cv_diff, NULL, &cv_maxdiff);
    cv::minMaxLoc(cuda_diff, NULL, &cuda_maxdiff);
    cv::minMaxLoc(cuda_diff2, NULL, &cuda_maxdiff2);

    printf("Image Size: (%d, %d)\n", width, height);
    printf("Time of NPPI: %fms\n", t0 / nrepeats);
    printf("Time of OpenCV: %fms\n", (t1 - t0) / nrepeats);
    printf("Time of CUDA: %fms\n", (t2 - t1) / nrepeats);
    printf("Time of aligned CUDA: %fms\n", (t3 - t2) / nrepeats);
    printf("Max difference of NPPI: %lf\n", nppi_maxdiff);
    printf("Max difference of OpenCV: %lf\n", cv_maxdiff);
    printf("Max difference of CUDA: %lf\n", cuda_maxdiff);
    printf("Max difference of aligned CUDA: %lf\n", cuda_maxdiff2);

    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_nppi_res);
    CUDA_SAFE_FREE(d_cuda_res);
    CUDA_SAFE_FREE(d_cuda_res2);
    CUDA_SAFE_FREE(d_temp_buf);
}


void autoTestDemo(int argc, char** argv)
{
    int min_size = 64;
    int max_size = 6000;
    int nrepeats = 10;
    int mrepeats = 5;
    if (argc > 1) min_size = atoi(argv[1]);
    if (argc > 2) max_size = atoi(argv[2]);
    if (argc > 3) nrepeats = atoi(argv[3]);
    if (argc > 4) mrepeats = atoi(argv[4]);

    unsigned char* d_src = nullptr;
    int* d_nppi_res = nullptr;
    int* d_cuda_res = nullptr;
    int* d_temp_buf = nullptr;
    size_t sbytes = max_size * max_size * sizeof(unsigned char);
    size_t dbytes = (max_size + 1) * (max_size + 1) * sizeof(int);
    size_t dbytes2 = max_size * max_size * sizeof(int);
    size_t dbytes3 = max_size * iDivUp(max_size, 8) * sizeof(int);
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_src), sbytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nppi_res), dbytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cuda_res), dbytes2));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buf), dbytes3));

    curandState* rand_state = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&rand_state), max_size * max_size * sizeof(curandState)));
    srand(time(0));

    // Initialize rand state
    hInitRand(rand_state, rand(), max_size * max_size);

    const char* path = "../res.log";
    FILE *fp = fopen(path, "a");
    if (fp == nullptr)
        return;

    cv::RNG rng;
    for (int i = 0; i < nrepeats; i++)
    {
        const int width = rng.uniform(min_size, max_size);
        const int height = rng.uniform(min_size, max_size);
        NppiSize osz { width, height };
        int sstep = width * sizeof(unsigned char);
        int dstep = (width + 1) * sizeof(int);

        // Initialization
        hRandFill(d_src, rand_state, width, height, width);

        for (int i = 0; i < mrepeats; i++)
        {
            // NPPI
            nppiIntegral_8u32s_C1R(d_src, sstep, d_nppi_res, dstep, osz, 0);

            // CUDA
            hIntegral(d_src, d_cuda_res, d_temp_buf, width, height, width, width);

            // max-abs-diff
            hCmpMaxAbsDiff(d_nppi_res, d_cuda_res, width, height);
            CHECK(cudaDeviceSynchronize());

            int max_diff = 4564545;
            CHECK(cudaMemcpy(&max_diff, d_cuda_res, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK(cudaDeviceSynchronize());

            fprintf(fp, "Size: (%d, %d), Max difference of NPPI and CUDA: %d\n", width, height, max_diff);
        }

        if (i % 10 == 0)
        {
            fclose(fp);
            fp = fopen(path, "a");
            if (fp == nullptr)
                return;
        }
    }

    fclose(fp); 
    
    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_nppi_res);
    CUDA_SAFE_FREE(d_cuda_res);
    CUDA_SAFE_FREE(d_temp_buf);
    CUDA_SAFE_FREE(rand_state);
}


