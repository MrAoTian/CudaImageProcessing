#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "guided_filter.h"
#include "cuda_utils.h"


void cvGuidedFilterDemo(int argc, char** argv);
void cudaGuidedFilterDemo(int argc, char** argv);


int main(int argc, char** argv)
{
    initDevice(0);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    cvGuidedFilterDemo(argc, argv);
    cudaGuidedFilterDemo(argc, argv);

    CHECK(cudaDeviceReset());
    printf("Procedure finished\n");
    return EXIT_SUCCESS;
}



void guidedFilter(cv::Mat& I, cv::Mat& p, cv::Mat& dst, const int r, float eps)
{
    // Guided Filtering 
    const cv::Size wsz(r, r);
    cv::Mat pm, Im, Ipm, IIm;
    cv::blur(p, pm, wsz);
    cv::blur(I, Im, wsz);
    cv::blur(p.mul(I), Ipm, wsz);
    cv::blur(I.mul(I), IIm, wsz);

    cv::Mat a, b, am, bm;
    a = (Ipm- pm.mul(Im)) / (IIm - Im.mul(Im) + eps);
    b = pm - a.mul(Im);
    cv::blur(a, am, wsz);
    cv::blur(b, bm, wsz);

    dst = am.mul(I) + bm;    
}


void cvGuidedFilterDemo(int argc, char** argv)
{
    std::string guidian_path = "../data/adobe_gt_4.jpg";
    std::string src_path = "../data/adobe_image_4.jpg";
    std::string dst_path = "../data/adobe_result_4.jpg";
    if (argc > 1)
        guidian_path = argv[1];
    if (argc > 2)
        src_path = argv[2];
    if (argc > 3)
        dst_path = argv[3];

    cv::Mat guidiance = cv::imread(guidian_path);
    cv::Mat src = cv::imread(src_path);
    cv::Mat dst;

    cv::resize(src, src, cv::Size(1920, 1080));
    cv::resize(guidiance, guidiance, cv::Size(1920, 1080));
    // Convert to float
    cv::Mat p, I;
    src.convertTo(p, CV_32F, 1.0 / 255.0);
    guidiance.convertTo(I, CV_32F, 1.0 / 255.0);

    auto t_start = cpuTimer();
    for (int i = 0; i < 10; i++)
    {
        guidedFilter(I, p, dst, 15, 0.3f);
    }   
    auto t_elapsed = cpuTimer() - t_start;

    printf("Time of cpu guided filter: %fms\n", t_elapsed * 0.001f / 10);

    dst.convertTo(dst, CV_8U, 255.0);
    cv::imwrite(dst_path, dst);
}


void cudaGuidedFilterDemo(int argc, char** argv)
{
    // Get image path
    std::string guidian_path = "../data/adobe_gt_4.jpg";
    std::string src_path = "../data/adobe_image_4.jpg";
    std::string dst_path = "../data/adobe_cuda_result_4.jpg";
    if (argc > 1)
        guidian_path = argv[1];
    if (argc > 2)
        src_path = argv[2];
    if (argc > 3)
        dst_path = argv[3];

    // Read image
    cv::Mat guidiance = cv::imread(guidian_path, cv::IMREAD_GRAYSCALE);
    cv::Mat src = cv::imread(src_path);
    if (src.rows != guidiance.rows || src.cols != guidiance.cols)
    {
        std::cout << "Size of source and guidiance image was not equivalence" << std::endl;
        return;
    }

    // Size
    cv::resize(src, src, cv::Size(1920, 1080));
    cv::resize(guidiance, guidiance, cv::Size(1920, 1080));
    const int height = src.rows;
    const int width = src.cols;
    const int schannels = src.channels();
    const int gchannels = guidiance.channels();
    int sstride = 0;
    int gstride = 0;

    // To float
    cv::Mat fsrc, fguidiance;
    src.convertTo(fsrc, CV_32F, 1.0 / 255.0);
    guidiance.convertTo(fguidiance, CV_32F, 1.0 / 255.0);

    // Allocate memory on device
    float* d_src = nullptr;
    float* d_guidiance = nullptr;
    float* d_dst = nullptr;
    const size_t spitch1 = width * schannels * sizeof(float);
    const size_t gpitch1 = width * gchannels * sizeof(float);
    size_t spitch2 = 0;
    size_t gpitch2 = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src), &spitch2, spitch1, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_guidiance), &gpitch2, gpitch1, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst), &spitch2, spitch1, height));
    CHECK(cudaMemcpy2D(d_src, spitch2, fsrc.data, spitch1, spitch1, height, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy2D(d_guidiance, gpitch2, fguidiance.data, gpitch1, gpitch1, height, cudaMemcpyHostToDevice));
    sstride = spitch2 / sizeof(float);
    gstride = gpitch2 / sizeof(float);

    // Initialize filter
    std::shared_ptr<GuidedFilter> gfilter = std::make_shared<GuidedFilter>();
    gfilter->init(width, height, gchannels, schannels);

    // Forward
    gfilter->run(d_guidiance, d_src, d_dst, 3, 0.3f);

    GpuTimer timer(0);
    for (int i = 0; i < 100; i++)
    {
        gfilter->run(d_guidiance, d_src, d_dst, 7, 0.3f);
    }    
    float t_elapsed = timer.read();
    printf("Time of cuda guided filter: %fms\n", t_elapsed / 100);

    // Copy data from device to host
    cv::Mat hres(height, width, CV_32FC(schannels));
    CHECK(cudaMemcpy2D(hres.data, spitch1, d_dst, spitch2, spitch1, height, cudaMemcpyDeviceToHost));
    hres.convertTo(hres, CV_8U, 255.0);
    cv::imwrite(dst_path, hres);

    // Free memory
    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_guidiance);
    CUDA_SAFE_FREE(d_dst);
}



