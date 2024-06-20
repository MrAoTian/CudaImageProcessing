# CUDA implementation of min, max filter
## Build & Run
``` shell
mkdir build
cd build
cmake ..
make
./cuda_sort_filter radius mode
```
## Example
- Original image
![Original](data/sea.png)
- A min filter example
![min](data/cuda_bgr_0.png)
- A max filter example
![max](data/cuda_bgr_1.png)

