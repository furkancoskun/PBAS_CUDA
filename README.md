# PBAS_CUDA
This is CUDA C++ implementation of PBAS(Pixel Based Adaptive Segmenter) [1] algorithm. This repository includes codes for:
* Single and Multithreaded CPU version of PBAS [2]
* CUDA C++ implementation of PBAS

PBAS_CUDA does not need any OpenCV or OpenCV-CUDA function, all pre-processing steps are, also, written in seperate CUDA kernels in PBAS_CUDA.cu file. You can use PBAS_CUDA in your project by just copying PBAS_CUDA.cu PBAS_CUDA.h and PBAS_Params.h files.  

You can see the video that includes code overview and discussion on results from the link below:  

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/4vAxb0s96Vg/0.jpg)](https://www.youtube.com/watch?v=4vAxb0s96Vg)  

https://www.youtube.com/watch?v=4vAxb0s96Vg

## Requirements
* nvcc, CudaRT and CuRAND (Installing Cuda Toolkit satisfies this requirement)  
https://developer.nvidia.com/cuda-toolkit
* OpenCV > 3.0  
Although original PBAS CPU implementation requires OpenCV for matrix operations, PBAS_CUDA does not need any OpenCV function natively. In this repository, OpenCV is required for IO operations like video reading and writing etc. and for running CPU version for benchmarking.



## Building the PBAS_CUDA
!! You should set width and height of your test video to PbasParams.h file before compilation !!  

### For Linux
* make -j$(nproc)  

### For Windows
Not tested yet! It should be compiled with appropriate include of Cuda and OpenCV dependencies.

## Running the PBAS_CUDA
for video input:
* ./pbas -v $(video-name)  

for sequence input:
* ./pbas -s $(sequence-folder)

## Performance
* RTX 2080 Super & i7 CPU @ 2.30GHz 
  
  |   | 1920 x 1080 | 1280 x 720 |
  | ------------- | ------------- |  ------------- |
  | PBAS-CPU Single Thread  | 300 ms  |  170 ms  |
  | PBAS-CPU Multi Thread   | 120 ms  |  65 ms  |
  | PBAS-CUDA   | 9 ms  |  4.5 ms  |
  
## References
[1] M. Hofmann, P.Tiefenbacher, G. Rigoll "Background Segmentation with Feedback: The Pixel-Based Adaptive Segmenter", in proc of IEEE Workshop on Change Detection, 2012  

[2] https://sites.google.com/site/pbassegmenter/home

## Contact
Furkan Coskun - furkan.coskun@metu.edu.tr