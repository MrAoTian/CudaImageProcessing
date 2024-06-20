# CUDA implementation of guided filter.
## Build & Run
```
mkdir build
cd build
cmake..
make
./cuda_guided_filter
```
## Example
- Original image
![Original](data/adobe_image_4.jpg)
- Guided image
![Guidiance](data/adobe_gt_4.jpg)
- Guided filter by C++
![C++](data/adobe_result_4.jpg)
- Guided filter by CUDA
![CUDA](data/adobe_cuda_result_4.jpg)
