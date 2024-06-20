# CUDA implementation of histogram equalization and CLAHE.
## Build & Run
```
mkdir build
cd build
cmake ..
make
./CudaHistogram
```
## Histogram Equalization Example
- Original image
![Original](data/night_gray.png)
- Histogram equalized by OpenCv
![OpenCv-HE](data/night_cvhe.png)
- Histogram equalized by CUDA
![CUDA-HE](data/night_cudahe.png)
## CLAHE Example
- Original image
![Original](data/night.png)
- Histogram equalized by OpenCv
![OpenCv-CLAHE](data/night_bgr_cv_clahe.png)
- Histogram equalized by CUDA
![CUDA-CLAHE](data/night_bgr_cuda_clahe.png)
