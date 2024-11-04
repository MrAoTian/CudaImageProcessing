#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include "guided_filter.h"
#include "guided_filter_d.h"


void cvGuidedFilterDemo(int argc, char** argv);
void cudaGuidedFilterDemo(int argc, char** argv);
void cudaSmallGuidedDemo(int argc, char** argv);


int main(int argc, char** argv)
{
    initDevice(0);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    // cvGuidedFilterDemo(argc, argv);
    // cudaGuidedFilterDemo(argc, argv);
    cudaSmallGuidedDemo(argc, argv);

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


double calcMaxAbsDiff(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat c;
    cv::absdiff(a, b, c);
    double maxdiff = DBL_MAX;
    cv::minMaxLoc(c, NULL, &maxdiff);
    return maxdiff;
}


void cudaSmallGuidedDemo(int argc, char** argv)
{
    // Parse config
    int radius = 1;
    float eps = 0.3f;
    int nrepeats = 1;
    std::string src_path = "../data/adobe_image_4.jpg";
    std::string guided_path = "../data/adobe_gt_4.jpg";
    if (argc > 1) radius      = std::atoi(argv[1]);
    if (argc > 2) eps         = std::atof(argv[2]);
    if (argc > 3) nrepeats    = std::atoi(argv[3]);
    if (argc > 4) src_path    = argv[4];
    if (argc > 5) guided_path = argv[5];

    // Prepare data on host
    cv::Mat h_src = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    if (h_src.empty())
    {
        printf("Can not read source image from: %s\n", src_path.c_str());
        return;
    }
    cv::Mat h_guided = cv::imread(guided_path, cv::IMREAD_GRAYSCALE);
    if (h_guided.empty())
    {
        printf("Guided image is missing. We use median-filtered image as guided-image\n");
        cv::medianBlur(h_src, h_guided, 3);
    }
    h_src.convertTo(h_src, CV_32FC1, 1.0 / 255.0);
    h_guided.convertTo(h_guided, CV_32FC1, 1.0 / 255.0);
    const int height = 2160;
    const int width = 3840;
    const int ksz = 2 * radius + 1;
    cv::resize(h_src, h_src, cv::Size(width, height));
    cv::resize(h_guided, h_guided, cv::Size(width, height));    

    // Allocate data on device
    float* d_src = nullptr;
    float* d_guided = nullptr;
    float* d_dst_cuda = nullptr;
    float* d_A = nullptr;
    float* d_B = nullptr;
    size_t spitch = width * sizeof(float);
    size_t dpitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_guided), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst_cuda), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_A), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_B), &dpitch, spitch, height));
    const int stride = static_cast<int>(dpitch / sizeof(float));

    // Copy data from host to device
    CHECK(cudaMemcpy2D(d_src, dpitch, h_src.data, spitch, spitch, height, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy2D(d_guided, dpitch, h_guided.data, spitch, spitch, height, cudaMemcpyHostToDevice));

    // Guided filtering on host
    cv::Mat h_dst;
    cv::ximgproc::guidedFilter(h_guided, h_src, h_dst, radius, eps);

    // Guided filtering by myself
    cv::Mat h_dst_mycv, h_A_mycv, h_B_mycv, h_Pm_mycv, h_Im_mycv, h_IPm_mycv, h_IIm_mycv;
    {
        const cv::Size wsz(ksz, ksz);
        
        cv::blur(h_src, h_Pm_mycv, wsz);
        cv::blur(h_guided, h_Im_mycv, wsz);
        cv::blur(h_src.mul(h_guided), h_IPm_mycv, wsz);
        cv::blur(h_guided.mul(h_guided), h_IIm_mycv, wsz);

        cv::Mat am, bm;
        h_A_mycv = (h_IPm_mycv- h_Pm_mycv.mul(h_Im_mycv)) / (h_IIm_mycv - h_Im_mycv.mul(h_Im_mycv) + eps);
        h_B_mycv = h_Pm_mycv- h_A_mycv.mul(h_Im_mycv);
        cv::blur(h_A_mycv, am, wsz);
        cv::blur(h_B_mycv, bm, wsz);
        h_dst_mycv = am.mul(h_guided) + bm;
    }
    
    // Warm up
    for (int i = 0; i < 100; i++)
    {
        hGuidedFilter(d_guided, d_src, d_dst_cuda, d_A, d_B, eps, radius, width, height, stride);
    }

    // Guided filter by CUDA
    GpuTimer timer(0);
    for (int i = 0; i < nrepeats; i++)
    {
        hGuidedFilter(d_guided, d_src, d_dst_cuda, d_A, d_B, eps, radius, width, height, stride);
    }
    CHECK(cudaDeviceSynchronize());
    float t_elapsed = timer.read();

    // Copy data from device to host
    cv::Mat h_dst_cuda(height, width, CV_32FC1);
    cv::Mat h_A_cuda(height, width, CV_32FC1);
    cv::Mat h_B_cuda(height, width, CV_32FC1);
    cv::Mat h_Pm_cuda(height, width, CV_32FC1);
    cv::Mat h_Im_cuda(height, width, CV_32FC1);
    cv::Mat h_IPm_cuda(height, width, CV_32FC1);
    cv::Mat h_IIm_cuda(height, width, CV_32FC1);
    CHECK(cudaMemcpy2D(h_dst_cuda.data, spitch, d_dst_cuda, dpitch, spitch, height, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2D(h_A_cuda.data, spitch, d_A, dpitch, spitch, height, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2D(h_B_cuda.data, spitch, d_B, dpitch, spitch, height, cudaMemcpyDeviceToHost));

    // Verify
    double maxdiff_cuda = calcMaxAbsDiff(h_dst, h_dst_cuda);
    double maxdiff_mycv = calcMaxAbsDiff(h_dst, h_dst_mycv);
    double maxdiff_mycu = calcMaxAbsDiff(h_dst_cuda, h_dst_mycv);
    double maxdiff_A = calcMaxAbsDiff(h_A_cuda, h_A_mycv);
    double maxdiff_B = calcMaxAbsDiff(h_B_cuda, h_B_mycv);
    printf("max difference between host and device: %lf\n", maxdiff_cuda);
    printf("max difference between host and myself: %lf\n", maxdiff_mycv);
    printf("max difference between cuda and myself: %lf\n", maxdiff_mycu);
    printf("max difference between host and device for A: %lf\n", maxdiff_A);
    printf("max difference between host and device for B: %lf\n", maxdiff_B);
    printf("Time cost of CUDA guided filter: %fms\n", t_elapsed / nrepeats);

    // Save result
    h_dst.convertTo(h_dst, CV_8U, 255.0);
    h_dst_cuda.convertTo(h_dst_cuda, CV_8U, 255.0);
    h_dst_mycv.convertTo(h_dst_mycv, CV_8U, 255.0);
    std::string base_path = src_path.substr(0, src_path.rfind("."));
    std::string cvres_path = base_path + "_cvres.png";
    std::string cures_path = base_path + "_cures.png";
    std::string myres_path = base_path + "_myres.png";
    cv::imwrite(cvres_path, h_dst);
    cv::imwrite(cures_path, h_dst_cuda);
    cv::imwrite(myres_path, h_dst_mycv);

    // Free memory
    CUDA_SAFE_FREE(d_guided);
    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_dst_cuda);
    CUDA_SAFE_FREE(d_A);
    CUDA_SAFE_FREE(d_B);
}



