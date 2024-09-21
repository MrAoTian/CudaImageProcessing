# CUDA implementation of integral

## 编译运行指令
``` shell
mkdir build
cd build
cmake ..
make
./cuda_integral 'options'
```

## 速度实验
激活函数 **integralDemo**, 重复1000次实验，每次实验对同一张4K图像使用OpenCV、NPPI和本项目代码计算积分图:
``` shell
# Command
$ cd path-to-build
$ ./cuda_integral 3840 2160 1000

# Output
Image Size: (3840, 2160)
Time of NPPI: 1.929810ms
Time of OpenCV: 2.685752ms
Time of CUDA: 0.596972ms
Max difference of NPPI: 0.000000
Max difference of OpenCV: 0.000000
Max difference of CUDA: 0.000000
```
本项目实现的积分图计算速度远快于NPPI和OpenCV。

## 正确性实验
激活函数**autoTestDemo**，重复若干次实验，每次实验随机生成不同尺寸的图像，对比NPPI的结果均一致，详情见[res.log](res.log)

