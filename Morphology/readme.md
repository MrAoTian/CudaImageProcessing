# CUDA implementation of erode and dilate
## Build & Run
``` shell
mkdir build
cd build
cmake ..
make
./cuda_morphology radius mode nrepeats
```
## Example
- Original image
![Original](data/sea.png)
- A min filter example
![min](data/cuda_bgr_0.png)
- A max filter example
![max](data/cuda_bgr_1.png)

