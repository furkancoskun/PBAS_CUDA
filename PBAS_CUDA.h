#ifndef PBAS_CUDA_H
#define PBAS_CUDA_H

#include <iostream>
#include <string>  
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PBAS_Params.h"
#include <curand.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

class PBAS_CUDA
{
public:
	PBAS_CUDA();
	~PBAS_CUDA();

    void Process(cv::Mat &inputFrame, cv::Mat &segmentationResult);
    void Reset();

private:
    void CreateRandomNumberArrays();
    void SetupCurandStates();
    void GetGradientMagnitudeFeatures();
    void ApplyGaussianBlur();
    void CalculateMeanGradientMagnitude();

    int* counter_;
    unsigned char* inputImage_;
    unsigned char* inputImageBlurred_;
    unsigned char* segmentationResultImage_;
    unsigned char* pixelValueBackModel_;
    float* inputImageGradientMagnitude_;
    float* gradientMagnitudeBackModel_;
    float* R_Array_;
    float* T_Array_;
    float* d_min_Array_;
    float* d_min_History_Array_;
    float* maxNormArray_;
    float* maxNormSumReductionArray_;
    int* maxNormCounterArray_;
    int* maxNormCounterSumReductionArray_;
    float meanGradMag_;
    cv::Mat resultImage_;
    curandState* curandStates_;

    // Random Number Arrays
    int* Random_N_Array_1_; // Random array for updating pixel's own model and d_min_history array
    int* Random_N_Array_2_; // Random array for updating pixel's neighbour model
    int* Random_X_Array_; // Random array for X coordinate for updating neighbour model
    int* Random_Y_Array_; // Random array for X coordinate for updating neighbour model

    // Initialization Variables
    float* h_R_Array_Initialization;
    float* h_T_Array_Initialization;
    float* h_d_min_Array_Initialization;
    
    // Cuda kernel call variables
    int pbasKernelGridSize_;
    dim3 GradMagKernelGridSize;
    dim3 GradMagKernelBlockSize;
    dim3 GaussianBlurKernelGridSize;
    dim3 GaussianBlurKernelBlockSize;    
    cudaStream_t cudaStream_;
    int iterationCounter_;
};
 
#endif //PBAS_CUDA_H