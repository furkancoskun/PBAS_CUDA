#ifndef PBAS_PARAMS_H
#define PBAS_PARAMS_H

#include "cuda_runtime.h"

// PBAS Params
const int width = 1280;
const int height = 720;
const int N_historyQueueSize = 20;
const int BackgroundDecisionCount = 2; // (#min in paper)
const int T_Update_Rate = 18;
const int T_Update_Rate_Lower_Boundary = 2;
const int T_Update_Rate_Upper_Boundary = 200;
const float h_R_Threshold = 18.0f;
__device__ const float R_Threshold = h_R_Threshold;
__device__ const float R_Threshold_Scale_Constant = 5.0f;
__device__ const float R_Threshold_Inc_Dec_Rate = 5.0f;
__device__ const float T_Update_Rate_Inc_Rate = 0.05f;
__device__ const float T_Update_Rate_Dec_Rate = 1.0f;
__device__ const float alpha = 10.0f;
__device__ const float beta = 1.0f;

// Number of threads in a grid
const int pbasKernelBlockSize = 16; // PBAS(main) kernel block size i.e. number of threads
const int gradMagKernelBlockSize = 16; // Gradient Magnitude Calculation kernel block size i.e. number of threads
const int randomNumberArrayCreateKernelBlockSize = 16; // Create Random Int Number Arrays kernel block size i.e. number of threads
const int gaussianBlurKernelBlockSize = 16; // Gradient Blur kernel block size i.e. number of threads
const int meanGradMagCalculationKernelBlockSize = 512; // Calculate Mean Gradient kernel(sum reduction) block size i.e. number of threads
const int randomNumberArraysSize = 100000;

#endif //PBAS_PARAMS_H