#pragma once


class Claher
{   
public:
    Claher();
    ~Claher();

    /* Initialization */
    void init(float clip_limit_, int xtiles_, int ytiles_);

    /* Run */
    void run(unsigned char* src, unsigned char* dst, int width, int height, int stride);

private:
    // Contrast limit
    float clip_limit = 1.f;
    // Number of tiles in x axis
    int xtiles;
    // Number of tiles in y axis
    int ytiles;
    // Histograms
    int* hist = nullptr;
    // Look up table
    float* table = nullptr;
    
    /* Free memory */
    void freeMemory();

    /* Allocate Memory */
    void allocMemory();

};

