#include "PBAS_CUDA.h"

#define CUDA_CALL(cmd) do {                       \
  cudaError_t e = cmd;                            \
  if( e != cudaSuccess ) {                        \
    printf("Failed: Cuda Error %s:%d '%s'\n",     \
        __FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);                           \
  }                                               \
} while(0)

#define CURAND_CALL(cmd) do {                       \
  curandStatus_t e = cmd;                           \
  if( e != CURAND_STATUS_SUCCESS ) {                \
    printf("Failed: CuRand error %s:%d '%s'\n",     \
        __FILE__,__LINE__,curandGetErrorString(e)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0) 

// Parser for cuRAND API errors
static const char *curandGetErrorString(curandStatus_t error) {
    switch (error) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }

  return "<unknown>";
}

__global__ void gaussianBlur(unsigned char* image, unsigned char* gaussianBlurImage, int width, int height) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        float gaussian;
        if( x > 0 && y > 0 && x < width-1 && y < height-1) 
        {
            gaussian = ((1.0f/16.0f) * float(image[(y-1)*width+(x-1)])) + ((1.0f/8.0f) * float(image[y*width+(x-1)])) + ((1.0f/16.0f) * float(image[(y+1)*width+(x-1)])) +
                       ( (1.0f/8.0f) * float(image[(y-1)*width+(x  )])) + ((1.0f/4.0f) * float(image[y*width+(x  )])) + ( (1.0f/8.0f) * float(image[(y+1)*width+(x  )])) +
                       ((1.0f/16.0f) * float(image[(y-1)*width+(x+1)])) + ((1.0f/8.0f) * float(image[y*width+(x+1)])) + ((1.0f/16.0f) * float(image[(y+1)*width+(x+1)])) ;


            gaussianBlurImage[y*width + x] = (unsigned char)(gaussian);
        }
        else
        {
            gaussianBlurImage[y*width + x] = image[y*width + x];
        }
    }
}

__global__ void gradientMagnitude(unsigned char* image, float* gradMag, int width, int height) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        float dx, dy;
        if( x > 0 && y > 0 && x < width-1 && y < height-1) 
        {
            dx = (-1* image[(y-1)*width + (x-1)]) + (-2*image[y*width+(x-1)]) + (-1*image[(y+1)*width+(x-1)]) +
                 (    image[(y-1)*width + (x+1)]) + ( 2*image[y*width+(x+1)]) + (   image[(y+1)*width+(x+1)]);

            dy = (    image[(y-1)*width + (x-1)]) + ( 2*image[(y-1)*width+x]) + (   image[(y-1)*width+(x+1)]) +
                 (-1* image[(y+1)*width + (x-1)]) + (-2*image[(y+1)*width+x]) + (-1*image[(y+1)*width+(x+1)]);

            gradMag[y*width + x] = sqrt((dx*dx) + (dy*dy));
        }
        else
        {
            gradMag[y*width + x] = 0;
        }
    }
}

__device__ void warpReduceFloat(volatile float* shmem_ptr, int tid) {
	shmem_ptr[tid] += shmem_ptr[tid + 32];
	shmem_ptr[tid] += shmem_ptr[tid + 16];
	shmem_ptr[tid] += shmem_ptr[tid + 8];
	shmem_ptr[tid] += shmem_ptr[tid + 4];
	shmem_ptr[tid] += shmem_ptr[tid + 2];
	shmem_ptr[tid] += shmem_ptr[tid + 1];
}

__global__ void sumReductionFloat(float* reductionArray, float* sumValue, int totalSize) 
{
    __shared__ float sdata[meanGradMagCalculationKernelBlockSize];

    int tid = threadIdx.x;
    int i = (blockIdx.x*blockDim.x*2) + threadIdx.x;

    if (i >= totalSize)
    {
        sdata[tid] = 0.0f;
    }
    else if ((i+blockDim.x) >= totalSize)
    {
        sdata[tid] = reductionArray[i];
    }
    else
    {
        sdata[tid] = reductionArray[i] + reductionArray[i+blockDim.x];
    }
    __syncthreads();

    //// Loop Unrolling Creates Problems with Optimization Flags. It does not give correct results!
    //// TO-DO : Investigate Loop Unrolling with O3 flag!
    ////
    // for (int s=blockDim.x/2; s>32; s>>=1)
    // {
    //     if (tid < s)
    //     {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }
    // if (tid < 32)
    // {
    //     warpReduceFloat(sdata, tid);
    // }

	for (int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
		if (tid < s) 
        {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

    if (tid == 0) 
    {
        sumValue[blockIdx.x] = sdata[0];
    }
}

__device__ void warpReduceInt(volatile int* shmem_ptr, int tid) {
	shmem_ptr[tid] += shmem_ptr[tid + 32];
	shmem_ptr[tid] += shmem_ptr[tid + 16];
	shmem_ptr[tid] += shmem_ptr[tid + 8];
	shmem_ptr[tid] += shmem_ptr[tid + 4];
	shmem_ptr[tid] += shmem_ptr[tid + 2];
	shmem_ptr[tid] += shmem_ptr[tid + 1];
}

__global__ void sumReductionInt(int* reductionArray, int* sumValue, int totalSize) 
{
    __shared__ int sdata[meanGradMagCalculationKernelBlockSize];

    int tid = threadIdx.x;
    int i = (blockIdx.x*blockDim.x*2) + threadIdx.x;

    if (i >= totalSize)
    {
        sdata[tid] = 0;
    }
    else if ((i+blockDim.x) >= totalSize)
    {
        sdata[tid] = reductionArray[i];
    }
    else
    {
        sdata[tid] = reductionArray[i] + reductionArray[i+blockDim.x];
    }
    __syncthreads();

    //// Loop Unrolling Creates Problems with Optimization Flags. It does not give correct results!
    //// TO-DO : Investigate Loop Unrolling with O3 flag!
    ////
    // for (int s=blockDim.x/2; s>32; s>>=1)
    // {
    //     if (tid < s)
    //     {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }
    // if (tid < 32)
    // {
    //     warpReduceInt(sdata, tid);
    // }

	for (int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
		if (tid < s) 
        {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

    if (tid == 0) 
    {
        sumValue[blockIdx.x] = sdata[0];
    }
}

__global__ void generateRandomIntArrayInRangeAtoB(float* auxiliaryArray, int* randomIntArray, int A, int B, int totalSize)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check limits
    if (i >= totalSize)
    {
        return;
    }

    float randomValue = auxiliaryArray[i];
    randomValue *= (B - A + 0.999999);
    randomValue += A;
    randomIntArray[i] = (int)truncf(randomValue);
}

__global__ void setupCurandStates(curandState* curandStates)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundaries
    if (i >= (width * height))
    {
        return;
    }

    curand_init(1234+i, 0, 0, &curandStates[i]);
}

__device__ void getRandomEntryNumber(curandState& curand_state, int& randomEntryNumber)
{ 
    float randomNumber = curand_uniform(&curand_state);
    randomNumber *= float(randomNumberArraysSize - 10);
    randomNumber += 5.0f;
    randomEntryNumber = (int)truncf(randomNumber);
}

__device__ void getRandomUpdateCoefficient(curandState& curand_state, int& randomUpdateCoefficient, float& T_Coefficient)
{
    float randomNumber = curand_uniform(&curand_state);
    randomNumber *= T_Coefficient;
    randomUpdateCoefficient = (int)truncf(randomNumber); 
}

__device__ void checkValid(int& idx, int& X_difference, int& Y_difference)
{
    int row = idx / width;
    int col = idx - (row * width);
    row += X_difference;
    col += Y_difference;
    
    if (row < 0 || row >= height)
    {
        X_difference = 0;
    }

    if (col < 0 || col >= width)
    {
        Y_difference = 0;
    }
}

__global__ void processOneChannelInitializationStage(unsigned char* inputImage, float* inputGradMagImage, unsigned char* pixelValueBackModel,
                                                     float* gradMagBackModel, unsigned char* segResultImage, float* R_Array, float* T_Array, 
                                                     float* d_min_Array, float* d_min_History_Array, float meanGradMag, float* maxNormArray, int* maxNormCounterArray,
                                                     int* random_N_Array_1, int* random_N_Array_2, int* random_X_Array, int* random_Y_Array, int* counter)
{
    __shared__ unsigned char inputImagePixelValueCache[pbasKernelBlockSize];
    __shared__ unsigned char pixelValueBackModelCache[pbasKernelBlockSize * N_historyQueueSize];
    __shared__ float inputImageGradMagCache[pbasKernelBlockSize];
    __shared__ float gradMagModelCache[pbasKernelBlockSize * N_historyQueueSize];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundaries
    if (i >= (width * height))
    {
        return;
    }

    // Store detection data to cache for fast comparison 
    inputImagePixelValueCache[tid] = inputImage[i];
    inputImageGradMagCache[tid] = inputGradMagImage[i];

    // Update Pixel and GradientMagnitude Background Models
    pixelValueBackModel[(i * N_historyQueueSize) + *counter] = inputImagePixelValueCache[tid];  // = inputImage[i];
    gradMagBackModel[(i * N_historyQueueSize) + *counter] = inputImageGradMagCache[tid]; // = inputGradMagImage[i];

    for (int N_id = 0; N_id < ((*counter)+1); N_id++)
    {
        pixelValueBackModelCache[(tid * N_historyQueueSize) + N_id] = pixelValueBackModel[(i * N_historyQueueSize) + N_id];
        gradMagModelCache[(tid * N_historyQueueSize) + N_id] = gradMagBackModel[(i * N_historyQueueSize) + N_id];
    }

    // Wait for finishing copy operations
    __syncthreads();

    //  --Detection Part--
    int count = 0;
    float abs_diff_pixel_value, abs_diff_mag_grad;
    float distance = 0.0f;
    float maxNorm = 0.0f;
    float R_ = R_Array[i];
    float T_ = T_Array[i];
    float minDist = 1000.0f;
    int maxNormCounter = 0;

    for (int N_id = 0; N_id < ((*counter)+1); N_id++)
    {
        abs_diff_pixel_value = abs(int(pixelValueBackModelCache[(tid * N_historyQueueSize) + N_id]) - int(inputImagePixelValueCache[tid]));
        abs_diff_mag_grad = abs(gradMagModelCache[(tid * N_historyQueueSize) + N_id] - inputImageGradMagCache[tid]);

        distance = ((alpha * abs_diff_mag_grad / meanGradMag) + (beta * abs_diff_pixel_value));

        if (distance < R_)
        {   
            if (distance < minDist)
            {
                minDist = distance;
            }
            count++;
        }
        else
        {
            maxNorm += abs_diff_mag_grad;
            maxNormCounter++;
        }

        if (count==BackgroundDecisionCount)
        {
            break;
        }
    }
    
    // Update average minimum distance that will be used while updating R and T
    float average_d_min = d_min_Array[i];
    d_min_History_Array[(i * N_historyQueueSize) + (*counter)] = minDist;
    average_d_min = (average_d_min * (*counter) + minDist) / ((*counter)+1);
    d_min_Array[i] = average_d_min;

    // If The  pixel is Background
    if (count >= BackgroundDecisionCount)
    {
        segResultImage[i] = (unsigned char)(0);

        // Update R
        float tempR = R_;
        if (R_ < (average_d_min * R_Threshold_Scale_Constant))
        {
            tempR += R_ * R_Threshold_Inc_Dec_Rate;

        }
        else
        {
            tempR -= R_ * R_Threshold_Inc_Dec_Rate;
        }

        if (tempR >= R_Threshold)
        {
            R_Array[i] = tempR;
        }
        else
        {
            R_Array[i] = R_Threshold;
        }

        //Update T
        float tempT = T_;
        tempT -= T_Update_Rate_Inc_Rate / (average_d_min + 1);
        if (tempT > T_Update_Rate_Lower_Boundary && tempT < T_Update_Rate_Upper_Boundary)
        {
            T_Array[i] = tempT;
        }
    }

    // Else -> The  pixel is Foreground
    else
    {
        segResultImage[i] = (unsigned char)(255);

        // Update R
        float tempR = R_;
        if (R_ < (average_d_min * R_Threshold_Scale_Constant))
        {
            tempR += R_ * R_Threshold_Inc_Dec_Rate / 10.0;
        }
        {
            tempR -= R_ * R_Threshold_Inc_Dec_Rate / 10.0;
        }

        if (tempR >= R_Threshold)
        {
            R_Array[i] = tempR;
        }
        else
        {
            R_Array[i] = R_Threshold;
        }

        //Update T
        float tempT = T_;
        tempT += T_Update_Rate_Dec_Rate / (average_d_min + 1);
        if (tempT > T_Update_Rate_Lower_Boundary && tempT < T_Update_Rate_Upper_Boundary)
        {
            T_Array[i] = tempT;
        }
    }

    maxNormArray[i] = maxNorm;
    maxNormCounterArray[i] = maxNormCounter;    
}



__global__ void processOneChannel(unsigned char* inputImage, float* inputGradMagImage, unsigned char* pixelValueBackModel, float* gradMagBackModel, 
                                  unsigned char* segResultImage, float* R_Array, float* T_Array, float* d_min_Array, float* d_min_History_Array, 
                                  float meanGradMag, float* maxNormArray, int* maxNormCounterArray,
                                  int* random_N_Array_1, int* random_N_Array_2, int* random_X_Array, int* random_Y_Array, curandState* curandStates)
{
    __shared__ unsigned char inputImagePixelValueCache[pbasKernelBlockSize];
    __shared__ unsigned char pixelValueBackModelCache[pbasKernelBlockSize * N_historyQueueSize];
    __shared__ float inputImageGradMagCache[pbasKernelBlockSize];
    __shared__ float gradMagModelCache[pbasKernelBlockSize * N_historyQueueSize];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundaries
    if (i >= (width * height))
    {
        return;
    }

    // Store detection data to cache for fast comparison 
    inputImagePixelValueCache[tid] = inputImage[i];
    inputImageGradMagCache[tid] = inputGradMagImage[i];
    for (int N_id = 0; N_id < N_historyQueueSize; N_id++)
    {
        pixelValueBackModelCache[(tid * N_historyQueueSize) + N_id] = pixelValueBackModel[(i * N_historyQueueSize) + N_id];
        gradMagModelCache[(tid * N_historyQueueSize) + N_id] = gradMagBackModel[(i * N_historyQueueSize) + N_id];
    }

    // Wait for finishing copy operations
    __syncthreads();

    //  --Detection Part--
    int count = 0;
    float abs_diff_pixel_value, abs_diff_mag_grad;
    float distance = 0.0f;
    float maxNorm = 0.0f;
    float R_ = R_Array[i];
    float T_ = T_Array[i];
    float minDist = 1000.0f;
    int maxNormCounter = 0;


    for (int N_id = 0; N_id < N_historyQueueSize; N_id++)
    {
        abs_diff_pixel_value = abs(int(pixelValueBackModelCache[(tid * N_historyQueueSize) + N_id]) - int(inputImagePixelValueCache[tid]));
        abs_diff_mag_grad = abs(gradMagModelCache[(tid * N_historyQueueSize) + N_id] - inputImageGradMagCache[tid]);

        distance = ((alpha * abs_diff_mag_grad / meanGradMag) + (beta * abs_diff_pixel_value));

        if (distance < R_)
        {   
            if (distance < minDist)
            {
                minDist = distance;
            }
            count++;
        }
        else
        {
            maxNorm += abs_diff_mag_grad;
            maxNormCounter++;
        }

        if (count==BackgroundDecisionCount)
        {
            break;
        }
    }

    float average_d_min = d_min_Array[i];
    int randomEntryNumber;
    int randomUpdateCoeff;
    int randomEntry;

    // If The  pixel is Background
    if (count >= BackgroundDecisionCount)
    {
        segResultImage[i] = (unsigned char)(0);

        getRandomEntryNumber(curandStates[i], randomEntryNumber);

        // Update Pixel's Own Background Model and Minimum Distance Array
        getRandomUpdateCoefficient(curandStates[i], randomUpdateCoeff, T_);
        if (randomUpdateCoeff < 1)
        {
            randomEntry = random_N_Array_1[randomEntryNumber];

            // Update Pixel and GradientMagnitude Background Models
            pixelValueBackModel[(i * N_historyQueueSize) + randomEntry] = inputImagePixelValueCache[tid]; // = inputImage[i];
            gradMagBackModel[(i * N_historyQueueSize) + randomEntry] = inputImageGradMagCache[tid]; // = inputGradMagImage[i];

            // Update average minimum distance that will be used while updating R and T
            float oldDistance = d_min_History_Array[(i * N_historyQueueSize) + randomEntry];
            d_min_History_Array[(i * N_historyQueueSize) + randomEntry] = minDist;
            average_d_min = ((average_d_min * (N_historyQueueSize - 1)) - oldDistance + minDist) / N_historyQueueSize;
            d_min_Array[i] = average_d_min;
        }

        // Update Randomly Selected Neighbour's Pixel Background Model
        getRandomUpdateCoefficient(curandStates[i], randomUpdateCoeff, T_);
        if (randomUpdateCoeff < 1)
        {
            randomEntry = random_N_Array_2[randomEntryNumber];
            int randomX_Difference = random_X_Array[randomEntryNumber];
            int randomY_Difference = random_Y_Array[randomEntryNumber];

            checkValid(i, randomX_Difference, randomY_Difference);

            int neighborIndex = i + randomX_Difference + (randomY_Difference * width);
            pixelValueBackModel[(neighborIndex * N_historyQueueSize) + randomEntry]  = inputImage[neighborIndex];
            gradMagBackModel[(neighborIndex * N_historyQueueSize) + randomEntry]  = inputGradMagImage[neighborIndex];
        }

        // Update R
        float tempR = R_;
        if (R_ < (average_d_min* R_Threshold_Scale_Constant))
        {
            tempR += R_ * R_Threshold_Inc_Dec_Rate;

        }
        else
        {
            tempR -= R_ * R_Threshold_Inc_Dec_Rate;
        }

        if (tempR >= R_Threshold)
        {
            R_Array[i] = tempR;
        }
        else
        {
            R_Array[i] = R_Threshold;
        }

        //Update T
        float tempT = T_;
        tempT -= T_Update_Rate_Inc_Rate / (average_d_min + 1);
        if (tempT > T_Update_Rate_Lower_Boundary && tempT < T_Update_Rate_Upper_Boundary)
        {
            T_Array[i] = tempT;
        }        
    }

    // Else -> The  pixel is Foreground
    else
    { 
        segResultImage[i] = (unsigned char)(255);

        // Update R
        float tempR = R_;
        if (R_ < (average_d_min * R_Threshold_Scale_Constant))
        {
            tempR += R_ * R_Threshold_Inc_Dec_Rate / 10.0;
        }
        {
            tempR -= R_ * R_Threshold_Inc_Dec_Rate / 10.0;
        }

        if (tempR >= R_Threshold)
        {
            R_Array[i] = tempR;
        }
        else
        {
            R_Array[i] = R_Threshold;
        }

        //Update T
        float tempT = T_;
        tempT += T_Update_Rate_Dec_Rate / (average_d_min + 1);
        if (tempT > T_Update_Rate_Lower_Boundary && tempT < T_Update_Rate_Upper_Boundary)
        {
            T_Array[i] = tempT;
        }
    }

    maxNormArray[i] = maxNorm;
    maxNormCounterArray[i] = maxNormCounter;
}


PBAS_CUDA::PBAS_CUDA()
{
    CUDA_CALL(cudaStreamCreate(&cudaStream_));

    pbasKernelGridSize_ = width * height / pbasKernelBlockSize;
    if ((width * height) % pbasKernelBlockSize != 0)
    {
        pbasKernelGridSize_++;
    }

    GradMagKernelBlockSize = dim3(gradMagKernelBlockSize, gradMagKernelBlockSize, 1);
    GradMagKernelGridSize = dim3(ceil(width/gradMagKernelBlockSize), ceil(height/gradMagKernelBlockSize), 1);

    GaussianBlurKernelBlockSize = dim3(gaussianBlurKernelBlockSize, gaussianBlurKernelBlockSize, 1);
    GaussianBlurKernelGridSize = dim3(ceil(width/gaussianBlurKernelBlockSize), ceil(height/gaussianBlurKernelBlockSize), 1);

    CUDA_CALL(cudaMalloc((void**)&counter_, size_t(sizeof(int))));
    CUDA_CALL(cudaMalloc((void**)&inputImage_, size_t(width * height * sizeof(unsigned char))));
    CUDA_CALL(cudaMalloc((void**)&inputImageBlurred_, size_t(width * height * sizeof(unsigned char))));
    CUDA_CALL(cudaMalloc((void**)&inputImageGradientMagnitude_, size_t(width * height * sizeof(float))));
    CUDA_CALL(cudaMalloc((void**)&segmentationResultImage_, size_t(width * height * sizeof(unsigned char))));
    CUDA_CALL(cudaMalloc((void**)&pixelValueBackModel_, size_t(N_historyQueueSize * width * height * sizeof(unsigned char))));
    CUDA_CALL(cudaMalloc((void**)&gradientMagnitudeBackModel_, N_historyQueueSize * width * height * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&R_Array_, size_t(width * height * sizeof(float))));
    CUDA_CALL(cudaMalloc((void**)&T_Array_, size_t(width * height * sizeof(float))));
    CUDA_CALL(cudaMalloc((void**)&d_min_Array_, size_t(width * height * sizeof(float))));
    CUDA_CALL(cudaMalloc((void**)&d_min_History_Array_, size_t(N_historyQueueSize * width * height * sizeof(float))));
    CUDA_CALL(cudaMalloc((void**)&maxNormArray_, size_t(width * height * sizeof(float))));
    CUDA_CALL(cudaMalloc((void**)&maxNormSumReductionArray_, size_t(width * height * sizeof(float))));
    CUDA_CALL(cudaMalloc((void**)&maxNormCounterArray_, size_t(width * height * sizeof(int))));
    CUDA_CALL(cudaMalloc((void**)&maxNormCounterSumReductionArray_, size_t(width * height * sizeof(int))));
    CUDA_CALL(cudaMalloc((void**)&curandStates_, size_t(width * height * sizeof(curandState))));
    CUDA_CALL(cudaMalloc((void**)&Random_N_Array_1_, size_t(randomNumberArraysSize * sizeof(int))));
    CUDA_CALL(cudaMalloc((void**)&Random_N_Array_2_, size_t(randomNumberArraysSize * sizeof(int))));
    CUDA_CALL(cudaMalloc((void**)&Random_X_Array_, size_t(randomNumberArraysSize * sizeof(int))));
    CUDA_CALL(cudaMalloc((void**)&Random_Y_Array_, size_t(randomNumberArraysSize * sizeof(int))));

    h_R_Array_Initialization = (float*)malloc(size_t(width * height * sizeof(float)));
    h_T_Array_Initialization = (float*)malloc(size_t(width * height * sizeof(float)));
    h_d_min_Array_Initialization = (float*)malloc(size_t(width * height * sizeof(float)));
    for (int i = 0; i < (width * height); i++)
    {
        h_R_Array_Initialization[i] = 18.0;
        h_T_Array_Initialization[i] = 18.0;
        h_d_min_Array_Initialization[i] = 0.0;
    }

    resultImage_ = cv::Mat::zeros(height, width, CV_8UC1);

    SetupCurandStates();
    
    CreateRandomNumberArrays();
    
    Reset();
}


void PBAS_CUDA::Reset()
{
    iterationCounter_ = 0;
    meanGradMag_ = 1.0f;
    CUDA_CALL(cudaMemcpy(R_Array_, h_R_Array_Initialization, size_t(width * height * sizeof(float)), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(T_Array_, h_T_Array_Initialization, size_t(width * height * sizeof(float)), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_min_Array_, h_d_min_Array_Initialization, size_t(width * height * sizeof(float)), cudaMemcpyHostToDevice));
}


PBAS_CUDA::~PBAS_CUDA()
{
    CUDA_CALL(cudaFree(counter_));
    CUDA_CALL(cudaFree(inputImage_));
    CUDA_CALL(cudaFree(inputImageBlurred_));
    CUDA_CALL(cudaFree(inputImageGradientMagnitude_));
    CUDA_CALL(cudaFree(segmentationResultImage_));
    CUDA_CALL(cudaFree(pixelValueBackModel_));
    CUDA_CALL(cudaFree(gradientMagnitudeBackModel_));
    CUDA_CALL(cudaFree(R_Array_));
    CUDA_CALL(cudaFree(T_Array_));
    CUDA_CALL(cudaFree(d_min_Array_));
    CUDA_CALL(cudaFree(d_min_History_Array_));
    CUDA_CALL(cudaFree(maxNormArray_));
    CUDA_CALL(cudaFree(maxNormSumReductionArray_));
    CUDA_CALL(cudaFree(maxNormCounterArray_));
    CUDA_CALL(cudaFree(maxNormCounterSumReductionArray_));
    CUDA_CALL(cudaFree(Random_N_Array_1_));
    CUDA_CALL(cudaFree(Random_N_Array_2_));
    CUDA_CALL(cudaFree(Random_X_Array_));
    CUDA_CALL(cudaFree(Random_Y_Array_));
}


void PBAS_CUDA::Process(cv::Mat& inputFrame, cv::Mat &segmentationResult)
{
    CUDA_CALL(cudaMemcpy(inputImage_, inputFrame.data, size_t(width * height * sizeof(unsigned char)), cudaMemcpyHostToDevice));

    ApplyGaussianBlur();

    GetGradientMagnitudeFeatures();

    CUDA_CALL(cudaMemcpy(counter_, &iterationCounter_, size_t(sizeof(int)), cudaMemcpyHostToDevice));

    if (iterationCounter_ < N_historyQueueSize)
    {
        processOneChannelInitializationStage << <pbasKernelGridSize_, pbasKernelBlockSize, 0, cudaStream_ >> > (inputImageBlurred_, inputImageGradientMagnitude_, pixelValueBackModel_,
            gradientMagnitudeBackModel_, segmentationResultImage_, R_Array_, T_Array_, d_min_Array_, d_min_History_Array_, meanGradMag_, maxNormArray_, maxNormCounterArray_,
            Random_N_Array_1_, Random_N_Array_2_, Random_X_Array_, Random_Y_Array_, counter_);
    }
    else
    {
        processOneChannel << <pbasKernelGridSize_, pbasKernelBlockSize, 0, cudaStream_ >> > (inputImageBlurred_, inputImageGradientMagnitude_, pixelValueBackModel_,
                gradientMagnitudeBackModel_, segmentationResultImage_, R_Array_, T_Array_, d_min_Array_, d_min_History_Array_, meanGradMag_, maxNormArray_, maxNormCounterArray_,
                Random_N_Array_1_, Random_N_Array_2_, Random_X_Array_, Random_Y_Array_, curandStates_);
    }

    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    CUDA_CALL(cudaMemcpy(resultImage_.data, segmentationResultImage_, size_t(width * height * sizeof(unsigned char)), cudaMemcpyDeviceToHost));
    segmentationResult = resultImage_;

    meanGradMag_ = 75.0f;
    //CalculateMeanGradientMagnitude();

    iterationCounter_++;
}


void PBAS_CUDA::CreateRandomNumberArrays()
{
    int randomNumberArrayCreateKernelGridSize = randomNumberArraysSize / randomNumberArrayCreateKernelBlockSize;
    if ((randomNumberArraysSize) % randomNumberArrayCreateKernelGridSize != 0)
    {
        randomNumberArrayCreateKernelGridSize++;
    }
    
    float* auxiliaryArray;
    CUDA_CALL(cudaMalloc((void**)&auxiliaryArray, size_t(randomNumberArraysSize * sizeof(float))));

    curandGenerator_t curandRandNumGenerator;
    CURAND_CALL(curandCreateGenerator(&curandRandNumGenerator, CURAND_RNG_PSEUDO_MTGP32));

    // Fill Random_N_Array_1_
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(curandRandNumGenerator, (unsigned long long)clock()));
    CURAND_CALL(curandGenerateUniform(curandRandNumGenerator, auxiliaryArray, size_t(randomNumberArraysSize)));
    generateRandomIntArrayInRangeAtoB<< <randomNumberArrayCreateKernelGridSize, randomNumberArrayCreateKernelBlockSize, 0, cudaStream_ >> >
                                     (auxiliaryArray, Random_N_Array_1_, 0, N_historyQueueSize, randomNumberArraysSize);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    // Fill Random_N_Array_2_
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(curandRandNumGenerator, (unsigned long long)clock()));
    CURAND_CALL(curandGenerateUniform(curandRandNumGenerator, auxiliaryArray, size_t(randomNumberArraysSize)));
    generateRandomIntArrayInRangeAtoB<< <randomNumberArrayCreateKernelGridSize, randomNumberArrayCreateKernelBlockSize, 0, cudaStream_ >> >
                                     (auxiliaryArray, Random_N_Array_2_, 0, N_historyQueueSize, randomNumberArraysSize);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    // Fill Random_X_Array_
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(curandRandNumGenerator, (unsigned long long)clock()));
    CURAND_CALL(curandGenerateUniform(curandRandNumGenerator, auxiliaryArray, size_t(randomNumberArraysSize)));
    generateRandomIntArrayInRangeAtoB<< <randomNumberArrayCreateKernelGridSize, randomNumberArrayCreateKernelBlockSize, 0, cudaStream_ >> >
                                     (auxiliaryArray, Random_X_Array_, 0, -1, 1);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    // Fill Random_Y_Array_
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(curandRandNumGenerator, (unsigned long long)clock()));
    CURAND_CALL(curandGenerateUniform(curandRandNumGenerator, auxiliaryArray, size_t(randomNumberArraysSize)));
    generateRandomIntArrayInRangeAtoB<< <randomNumberArrayCreateKernelGridSize, randomNumberArrayCreateKernelBlockSize, 0, cudaStream_ >> >
                                     (auxiliaryArray, Random_Y_Array_, 0, -1, 1);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    CUDA_CALL(cudaFree(auxiliaryArray));
    CURAND_CALL(curandDestroyGenerator(curandRandNumGenerator));
}


void PBAS_CUDA::GetGradientMagnitudeFeatures()
{
    gradientMagnitude<< <GradMagKernelGridSize, GradMagKernelBlockSize, 0, cudaStream_ >> >(inputImageBlurred_, inputImageGradientMagnitude_, width, height);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));
}


void PBAS_CUDA::SetupCurandStates()
{
    setupCurandStates << <pbasKernelGridSize_, pbasKernelBlockSize, 0, cudaStream_ >> >(curandStates_);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));
}


void PBAS_CUDA::ApplyGaussianBlur()
{
    gaussianBlur<< <GaussianBlurKernelGridSize, GaussianBlurKernelBlockSize, 0, cudaStream_ >> >(inputImage_, inputImageBlurred_, width, height);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));
}


void PBAS_CUDA::CalculateMeanGradientMagnitude()
{
    // cv::Mat dummy = cv::Mat(720,1280,CV_32SC1);
    // CUDA_CALL(cudaMemcpy(dummy.data, maxNormCounterArray_, size_t(width*height*sizeof(int)), cudaMemcpyDeviceToHost));
    // double s = cv::sum(dummy)[0];
    // std::cout << "Opencv SUM: " << s << std::endl;

    // cv::Mat dummy2 = cv::Mat(720,1280,CV_32FC1);
    // CUDA_CALL(cudaMemcpy(dummy2.data, maxNormArray_, size_t(width*height*sizeof(float)), cudaMemcpyDeviceToHost));
    // double s2 = cv::sum(dummy2)[0];
    // std::cout << "Opencv SUM FLOAT: " << s2 << std::endl;
    // std::cout << "Opencv NORM: " << float(s2)/float(s) << std::endl;

    float* floatArray, *floatArrayHost;
    int* intArray, *intArrayHost;

    floatArrayHost = (float*)malloc(size_t(width * height * sizeof(float)));
    intArrayHost = (int*)malloc(size_t(width * height * sizeof(int)));

    for (int i=0; i<(width*height); i++)
    {
        floatArrayHost[i] = 1.0f;
        intArrayHost[i] = 1;
    }

    CUDA_CALL(cudaMalloc((void**)&floatArray, size_t(width * height * sizeof(float))));
    CUDA_CALL(cudaMalloc((void**)&intArray, size_t(width * height * sizeof(int))));

    cudaMemcpy(intArray, intArrayHost, size_t(width * height * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(floatArray, floatArrayHost, size_t(width * height * sizeof(float)), cudaMemcpyHostToDevice);


    int totalSize = width*height;
    int gridSize = ceil(totalSize / 2 / meanGradMagCalculationKernelBlockSize);

    //sumReductionFloat<< <gridSize, meanGradMagCalculationKernelBlockSize, 0, cudaStream_ >> >(maxNormArray_, maxNormSumReductionArray_, totalSize);
    sumReductionFloat<< <gridSize, meanGradMagCalculationKernelBlockSize, 0, cudaStream_ >> >(floatArray, maxNormSumReductionArray_, totalSize);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    //sumReductionInt<< <gridSize, meanGradMagCalculationKernelBlockSize, 0, cudaStream_ >> >(maxNormCounterArray_, maxNormCounterSumReductionArray_, totalSize);
    sumReductionInt<< <gridSize, meanGradMagCalculationKernelBlockSize, 0, cudaStream_ >> >(intArray, maxNormCounterSumReductionArray_, totalSize);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    totalSize = gridSize;
    gridSize = ceil(float(totalSize) / 2 / meanGradMagCalculationKernelBlockSize);

    while (gridSize > 1)
    {
        sumReductionFloat<< <gridSize, meanGradMagCalculationKernelBlockSize, 0, cudaStream_ >> >(maxNormSumReductionArray_, maxNormSumReductionArray_, totalSize);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaStreamSynchronize(cudaStream_));  

        sumReductionInt<< <gridSize, meanGradMagCalculationKernelBlockSize, 0, cudaStream_ >> >(maxNormCounterSumReductionArray_, maxNormCounterSumReductionArray_, totalSize);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaStreamSynchronize(cudaStream_));  

        totalSize = gridSize;
        gridSize = ceil(float(totalSize) / 2 / meanGradMagCalculationKernelBlockSize);
    }    

    sumReductionFloat<< <1, meanGradMagCalculationKernelBlockSize, 0, cudaStream_ >> >(maxNormSumReductionArray_, maxNormSumReductionArray_, totalSize);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    sumReductionInt<< <1, meanGradMagCalculationKernelBlockSize, 0, cudaStream_ >> >(maxNormCounterSumReductionArray_, maxNormCounterSumReductionArray_, totalSize);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaStreamSynchronize(cudaStream_));

    float totalNorm;
    int totalCounter;
    CUDA_CALL(cudaMemcpy(&totalNorm, maxNormSumReductionArray_, size_t(sizeof(float)), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&totalCounter, maxNormCounterSumReductionArray_, size_t(sizeof(int)), cudaMemcpyDeviceToHost));

    assert(totalCounter == 921600);

    //assert(totalNorm == 921600);

    std::cout << "GPU - totalNorm: " << totalNorm << std::endl;
    std::cout << "GPU - totalCounter: " << totalCounter << std::endl;
    std::cout << "GPU - NORM: " << totalNorm/(totalCounter+1) << std::endl;
    
}