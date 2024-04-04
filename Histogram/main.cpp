#include "hist_equalization.h"
#include "clahe.h"
#include "cuda_utils.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


void heDemo(int argc, char** argv);
void claheDemo(int argc, char** argv);


int main(int argc, char** argv)
{
    // heDemo(argc, argv);
    claheDemo(argc, argv);
    
    return EXIT_SUCCESS;
}


void heDemo(int argc, char** argv)
{
    // Load configure
    std::string src_path = "../data/night.png";
    if (argc > 1) src_path = argv[1];

    // Read image
    cv::Mat image = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        printf("Failed to read image from: %s\n", src_path.c_str());
        return;
    }
    const int width = image.cols;
    const int height = image.rows;
    
    // Allocate memory on device 
    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    size_t spitch = width * sizeof(unsigned char);
    size_t dpitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst), &dpitch, spitch, height));
    CHECK(cudaMemcpy2D(d_src, dpitch, image.data, spitch, spitch, height, cudaMemcpyHostToDevice));    
    const int stride = dpitch / sizeof(unsigned char);
    
    // Initialize histogram equalization
    std::shared_ptr<HistEqualizer> hist_equalizer = std::make_shared<HistEqualizer>();
    hist_equalizer->init();

    // Forward
    hist_equalizer->run(d_src, d_dst, width, height, stride);

    // Copy result to host
    cv::Mat hres(height, width, CV_8UC1);
    cv::Mat dres(height, width, CV_8UC1);
    cv::equalizeHist(image, hres);
    CHECK(cudaMemcpy2D(dres.data, spitch, d_dst, dpitch, spitch, height, cudaMemcpyDeviceToHost));

    std::string dst_path1 = "../data/night_gray.png";
    std::string dst_path2 = "../data/night_cvhe.png";
    std::string dst_path3 = "../data/night_cudahe.png";
    cv::imwrite(dst_path1, image);
    cv::imwrite(dst_path2, hres);
    cv::imwrite(dst_path3, dres);

    // Free
    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_dst);
}


void claheDemo(int argc, char** argv)
{
    // Load configure
    float clip_limit = 1.0;
    int xtiles = 8;
    int ytiles = 8;
    std::string src_path = "../data/night.png";
    if (argc > 1) src_path = argv[1];
    if (argc > 2) clip_limit = atof(argv[2]);
    if (argc > 3) xtiles = atoi(argv[3]);
    if (argc > 4) ytiles = atoi(argv[4]);

    // Read image
    cv::Mat image = cv::imread(src_path);
    if (image.empty())
    {
        printf("Failed to read image from: %s\n", src_path.c_str());
        return;
    }
    const int width = image.cols;
    const int height = image.rows;

    cv::Mat lab;
    std::vector<cv::Mat> lab_channels;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    cv::split(lab, lab_channels);
    
    // Allocate memory on device 
    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    size_t spitch = width * sizeof(unsigned char);
    size_t dpitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst), &dpitch, spitch, height));
    CHECK(cudaMemcpy2D(d_src, dpitch, lab_channels[0].data, spitch, spitch, height, cudaMemcpyHostToDevice));    
    const int stride = dpitch / sizeof(unsigned char);
    
    // Initialize histogram equalization
    std::shared_ptr<Claher> cuda_claher = std::make_shared<Claher>();
    cuda_claher->init(clip_limit, xtiles, ytiles);

    // Forward
    cuda_claher->run(d_src, d_dst, width, height, stride);

    // Copy result to host
    cv::Mat hres(height, width, CV_8UC1);
    cv::Mat dres(height, width, CV_8UC1);
    cv::Ptr<cv::CLAHE> cv_claher = cv::createCLAHE(clip_limit, cv::Size(xtiles, ytiles));
    cv_claher->apply(lab_channels[0], hres);
    CHECK(cudaMemcpy2D(dres.data, spitch, d_dst, dpitch, spitch, height, cudaMemcpyDeviceToHost));

    cv::Mat hlab, dlab;
    cv::merge(std::vector<cv::Mat>{hres, lab_channels[1], lab_channels[2]}, hlab);
    cv::merge(std::vector<cv::Mat>{dres, lab_channels[1], lab_channels[2]}, dlab);

    cv::Mat hbgr, dbgr;
    cv::cvtColor(hlab, hbgr, cv::COLOR_Lab2BGR);
    cv::cvtColor(dlab, dbgr, cv::COLOR_Lab2BGR);

    std::string dst_path1 = "../data/night_L.png";
    std::string dst_path2 = "../data/night_cv_clahe.png";
    std::string dst_path3 = "../data/night_cuda_clahe.png";
    std::string dst_path4 = "../data/night_bgr_cv_clahe.png";
    std::string dst_path5 = "../data/night_bgr_cuda_clahe.png";
    cv::imwrite(dst_path1, lab_channels[0]);
    cv::imwrite(dst_path2, hres);
    cv::imwrite(dst_path3, dres);
    cv::imwrite(dst_path4, hbgr);
    cv::imwrite(dst_path5, dbgr);

    // Free
    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_dst);
}





