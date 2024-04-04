#include "hist_equalization.h"
#include "image_process.h"


HistEqualizer::HistEqualizer()
{
}


HistEqualizer::~HistEqualizer()
{
    this->freeMemory();
}


void HistEqualizer::freeMemory()
{
    CUDA_SAFE_FREE(hist);
    CUDA_SAFE_FREE(table);
}


void HistEqualizer::allocMemory()
{
    this->freeMemory();
    CHECK(cudaMalloc(reinterpret_cast<void**>(&hist), 256 * sizeof(int)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&table), 256 * sizeof(unsigned char)));
}


void HistEqualizer::init()
{
    this->allocMemory();
}


void HistEqualizer::run(unsigned char* src, unsigned char* dst, int width, int height, int stride)
{
    // Statistical histogram
    CHECK(cudaMemset(hist, 0, 256 * sizeof(int)));
    hCalcHist(src, hist, width, height, stride);

    if (false)
    {
        int h_hist[256] = {0};
        CHECK(cudaMemcpy(h_hist, hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        printf("Histogram: ");
        for (int i = 0; i < 256; i++)
        {
            if (i % 16 == 0)
                printf("\n");
            printf("%5d ", h_hist[i]);    
        }
        printf("\n");
    }

    // Compute look up table
    const float factor = 256.f / static_cast<float>(width * height);
    hCalcHeTable(hist, table, factor);

    if (false)
    {
        unsigned char h_table[256] = {0};
        CHECK(cudaMemcpy(h_table, table, 256 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        printf("Table: ");
        for (int i = 0; i < 256; i++)
        {
            if (i % 16 == 0)
                printf("\n");
            printf("%5d ", (int)h_table[i]);    
        }
        printf("\n");
    }

    // Mapping
    hMapping(src, dst, table, width, height, stride);
}

