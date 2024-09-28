#pragma once
#include "cuda_utils.h"


////////////////// Warm up //////////////////
void hWarmUp(unsigned char* src, unsigned char* dst, int width, int height, int stride);


////////////////// Histogram equalization //////////////////

// Compute histogram
void hCalcHist(unsigned char* src, int* hist, int width, int height, int stride);

// Compute histogram equalization table
void hCalcHeTable(int* hist, unsigned char* table, float fatcor);

// Mapping table
void hMapping(unsigned char* src, unsigned char* dst, unsigned char* table, int width, int height, int stride);



////////////////// CLAHE //////////////////

// Compute tile histogram
void hCalcTileHists(unsigned char* src, int* hists, int xtiles, int ytiles, int tile_width, int tile_height, int pad_left, int pad_top, int width, int height, int stride);

// Limit Clip
void hClipLimit(int* hists, int limit, int ntiles);

// Create LUT
void hCreateTable(int* hists, float* tables, int tile_pixels, int ntiles);

// Interpolation mapping
void hInterpolateMapping(unsigned char* src, unsigned char* dst, float* tables, int xtiles, int ytiles, int tile_width, int tile_height, int pad_left, int pad_top, int width, int height, int stride);
