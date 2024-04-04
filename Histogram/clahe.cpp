#include "clahe.h"
#include "image_process.h"


Claher::Claher()
{
}


Claher::~Claher()
{
    this->freeMemory();
}


void Claher::init(float clip_limit_, int xtiles_, int ytiles_)
{
    clip_limit = clip_limit_;
    xtiles = xtiles_;
    ytiles = ytiles_;

    this->allocMemory();
}


void Claher::run(unsigned char* src, unsigned char* dst, int width, int height, int stride)
{
    const int tile_width = iDivUp(width, xtiles);
    const int tile_height = iDivUp(height, ytiles);
    const int pad_width = tile_width * xtiles;
    const int pad_height = tile_height * ytiles;
    const int residual_width = pad_width - width;
    const int residual_height = pad_height - height;
    int4 pad_border;
    pad_border.x = residual_width >> 1;
    pad_border.y = residual_height >> 1;
    pad_border.z = residual_width - pad_border.x;
    pad_border.w = residual_height - pad_border.y;

    // Statistic histogram
    hCalcTileHists(src, hist, xtiles, ytiles, tile_width, tile_height, pad_border.x, pad_border.y, width, height, stride);

    if (false)
    {
        int h_hist[256] = {0};
        for (int i = 0; i < ytiles * xtiles; i++)
        {
            CHECK(cudaMemcpy(h_hist, hist + i * 256, 256 * sizeof(int), cudaMemcpyDeviceToHost));
            int hsum = 0;
            printf("Histogram%d: ", i);
            for (int j = 0; j < 256; j++)
            {
                // printf("%d ", h_hist[j]);
                hsum += h_hist[j];
            }
            printf("%d\n", hsum);
        }
    }

    // Clip limit
    const int limit = static_cast<int>(tile_width * tile_height * clip_limit / 256 + 0.5f);
    hClipLimit(hist, limit, ytiles * xtiles);

    if (false)
    {
        int h_hist[256] = {0};
        for (int i = 0; i < ytiles * xtiles; i++)
        {
            CHECK(cudaMemcpy(h_hist, hist + i * 256, 256 * sizeof(int), cudaMemcpyDeviceToHost));
            int hsum = 0;
            printf("Histogram%d: ", i);
            for (int j = 0; j < 256; j++)
            {
                printf("%d ", h_hist[j]);
                hsum += h_hist[j];
            }
            printf("%d\n", hsum);
        }
    }

    // Create table
    hCreateTable(hist, table, tile_width * tile_height, ytiles * xtiles);

    if (false)
    {
        float h_table[256] = {0};
        for (int i = 0; i < ytiles * xtiles; i++)
        {
            CHECK(cudaMemcpy(h_table, table + i * 256, 256 * sizeof(float), cudaMemcpyDeviceToHost));
            printf("Table%d: ", i);
            for (int j = 0; j < 256; j++)
            {
                printf("%.2f ", h_table[j]);
            }
            printf("\n");
        }
    }

    // Look up and interpolation
    hInterpolateMapping(src, dst, table, xtiles, ytiles, tile_width, tile_height, pad_border.x, pad_border.y, width, height, stride);
}


void Claher::freeMemory()
{
    CUDA_SAFE_FREE(hist);
    CUDA_SAFE_FREE(table);
}


void Claher::allocMemory()
{
    this->freeMemory();

    const int ntiles = xtiles * ytiles;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&hist), ntiles * 256 * sizeof(int)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&table), ntiles * 256 * sizeof(float)));
}


