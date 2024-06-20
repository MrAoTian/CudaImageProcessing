#pragma once



class MinMaxFilter
{
public:
    MinMaxFilter();
    ~MinMaxFilter();

    /* Initialization */
    void init(int width_, int height_);

    /* Apply filter */
    void run(unsigned char* src, unsigned char* dst, int ksz, int mode);

private:
    int width;
    int height;
    int wstride;
    int hstride;
    unsigned char* hset1 = nullptr;
    unsigned char* hset2 = nullptr;
    unsigned char* hmop = nullptr;
    unsigned char* vset1 = nullptr;
    unsigned char* vset2 = nullptr;
    unsigned char* vmop = nullptr;

    /* Allocate memory */
    void allocMemory();

    /* Free memory */
    void freeMemory();

};



