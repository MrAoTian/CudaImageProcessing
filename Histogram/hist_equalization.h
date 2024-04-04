#pragma once



class HistEqualizer
{
private:
    int width = 0;
    int height = 0;
    int stride = 0;
    int *hist = nullptr;
    unsigned char *table = nullptr;

    /* Free memory */
    void freeMemory();

    /* Allocate memory */
    void allocMemory();

public:
    HistEqualizer();
    ~HistEqualizer();

    /* Initialization */
    void init();

    /* Run */
    void run(unsigned char* src, unsigned char* dst, int width, int height, int stride);

};




