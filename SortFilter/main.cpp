#include "cuda_utils.h"
#include "min_max_filter.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


void minMaxFilterDemo(int argc, char** argv);
void minMaxFilterRGBDemo(int argc, char** argv);
void minMaxFilterLABDemo(int argc, char** argv);


int main(int argc, char** argv)
{
    minMaxFilterDemo(argc, argv);
    minMaxFilterRGBDemo(argc, argv);
    minMaxFilterLABDemo(argc, argv);
    
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}



void minMaxFilterDemo(int argc, char** argv)
{
    int radius = 23;
    int mode = 1;
    if (argc > 1) radius = atoi(argv[1]);
    if (argc > 2) mode = atoi(argv[2]);

    std::string src_path = "../data/sea.png";
    cv::Mat image = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        printf("Can not read image from: %s\n", src_path.c_str());
        return;
    }

    const int width = image.cols;
    const int height = image.rows;

    // Initialize min-max-filter
    std::shared_ptr<MinMaxFilter> p_filter = std::make_shared<MinMaxFilter>();
    p_filter->init(width, height);

    // Allocate memory on device 
    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    size_t spitch = width * sizeof(unsigned char);
    size_t dpitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst), &dpitch, spitch, height));
    CHECK(cudaMemcpy2D(d_src, dpitch, image.data, spitch, spitch, height, cudaMemcpyHostToDevice));
    const int stride = dpitch / sizeof(unsigned char);

    // Run
    const int ksz = 2 * radius + 1;
    p_filter->run(d_src, d_dst, ksz, mode);

    // Copy result back to host
    cv::Mat dres(height, width, CV_8UC1);
    CHECK(cudaMemcpy2D(dres.data, spitch, d_dst, dpitch, spitch, height, cudaMemcpyDeviceToHost));

    // Result by OpenCV
    cv::Mat hres;
    if (mode == 0)
    {
        cv::erode(image, hres, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksz, ksz)));
    }
    else
    {
        cv::dilate(image, hres, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksz, ksz)));
    }
    
    std::string hres_path = "../data/cv_gray_" + std::to_string(mode) + ".png";
    std::string dres_path = "../data/cuda_gray_" + std::to_string(mode) + ".png";
    cv::imwrite(dres_path, dres);
    cv::imwrite(hres_path, hres);

    cv::Mat diff;
    cv::absdiff(hres, dres, diff);

    double maxv;
    cv::Point maxp;
    cv::minMaxLoc(diff, NULL, &maxv, NULL, &maxp);
    printf("Max difference: %lf, at (%d, %d)\n", maxv, maxp.x, maxp.y);

    // Free
    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_dst);
}


void minMaxFilterRGBDemo(int argc, char** argv)
{
    int radius = 23;
    int mode = 0;
    if (argc > 1) radius = atoi(argv[1]);
    if (argc > 2) mode = atoi(argv[2]);

    std::string src_path = "../data/sea.png";
    cv::Mat image = cv::imread(src_path, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        printf("Can not read image from: %s\n", src_path.c_str());
        return;
    }

    const int width = image.cols;
    const int height = image.rows;

    // Initialize min-max-filter
    std::shared_ptr<MinMaxFilter> p_filter = std::make_shared<MinMaxFilter>();
    p_filter->init(width, height);

    // Allocate memory on device 
    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    size_t spitch = width * sizeof(unsigned char);
    size_t dpitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst), &dpitch, spitch, height));
    const int stride = dpitch / sizeof(unsigned char);

    // Run
    const int ksz = 2 * radius + 1;
    std::vector<cv::Mat> split_channels;
    cv::split(image, split_channels);
    for (int i = 0; i < split_channels.size(); i++)
    {
        CHECK(cudaMemcpy2D(d_src, dpitch, split_channels[i].data, spitch, spitch, height, cudaMemcpyHostToDevice));
        p_filter->run(d_src, d_dst, ksz, mode);
        CHECK(cudaMemcpy2D(split_channels[i].data, spitch, d_dst, dpitch, spitch, height, cudaMemcpyDeviceToHost));
    }   
    cv::Mat dres;
    cv::merge(split_channels, dres);


    // Result by OpenCV
    cv::Mat hres;
    if (mode == 0)
    {
        cv::erode(image, hres, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksz, ksz)));
    }
    else
    {
        cv::dilate(image, hres, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksz, ksz)));
    }
    
    std::string hres_path = "../data/cv_bgr_" + std::to_string(mode) + ".png";
    std::string dres_path = "../data/cuda_bgr_" + std::to_string(mode) + ".png";
    cv::imwrite(dres_path, dres);
    cv::imwrite(hres_path, hres);

    // Free
    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_dst);
}


void minMaxFilterLABDemo(int argc, char** argv)
{
    int radius = 23;
    int mode = 0;
    if (argc > 1) radius = atoi(argv[1]);
    if (argc > 2) mode = atoi(argv[2]);

    std::string src_path = "../data/sea.png";
    cv::Mat image = cv::imread(src_path, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        printf("Can not read image from: %s\n", src_path.c_str());
        return;
    }

    const int width = image.cols;
    const int height = image.rows;
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

    // Initialize min-max-filter
    std::shared_ptr<MinMaxFilter> p_filter = std::make_shared<MinMaxFilter>();
    p_filter->init(width, height);

    // Allocate memory on device 
    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    size_t spitch = width * sizeof(unsigned char);
    size_t dpitch = 0;
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src), &dpitch, spitch, height));
    CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst), &dpitch, spitch, height));
    const int stride = dpitch / sizeof(unsigned char);

    // Run
    const int ksz = 2 * radius + 1;
    std::vector<cv::Mat> split_channels;
    cv::split(image, split_channels);
    CHECK(cudaMemcpy2D(d_src, dpitch, split_channels[0].data, spitch, spitch, height, cudaMemcpyHostToDevice));
    p_filter->run(d_src, d_dst, ksz, mode);
    CHECK(cudaMemcpy2D(split_channels[0].data, spitch, d_dst, dpitch, spitch, height, cudaMemcpyDeviceToHost));  
    cv::Mat dres;
    cv::merge(split_channels, dres);

    // Result by OpenCV
    cv::Mat hres;
    if (mode == 0)
    {
        cv::erode(image, hres, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksz, ksz)));
    }
    else
    {
        cv::dilate(image, hres, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksz, ksz)));
    }
    
    std::string hres_path = "../data/cv_lab_" + std::to_string(mode) + ".png";
    std::string dres_path = "../data/cuda_lab_" + std::to_string(mode) + ".png";
    cv::imwrite(dres_path, dres);
    cv::imwrite(hres_path, hres);

    // Free
    CUDA_SAFE_FREE(d_src);
    CUDA_SAFE_FREE(d_dst);
}


