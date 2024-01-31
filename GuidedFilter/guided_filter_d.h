#pragma once
#include "cuda_utils.h"


/* Box filter */
void hBoxFilter(float* src, float* dst, float* integral, const int4& swhcs, const int4& iwhcs, const int r);

/* Multiplication */
void hMultiply(float* a, float* b, float* c, const int4& awhcs, const int4& bwhcs);

/* Compute A */
void hCalcA(float* a, float* pm, float* im, float* ipm, float* iim, const int4& swhcs, const int4& gwhcs, const float eps);

/* Compute B */
void hCalcB(float* b, float* a, float* pm, float* im, const int4& swhcs, const int4& gwhcs);

/* Linear transform */
void hLinearTransform(float* src, float* dst, float* a, float* b, const int4& swhcs, const int4& dwhcs);

