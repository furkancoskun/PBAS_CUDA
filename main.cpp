#include <iostream>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

#include "PBAS.h"
#include "PBAS_CUDA.h"

static void show_usage()
{
    std::cerr << "Usage: " << std::endl
        << "\t-h,--help\tShow this help message" << std::endl
        << "\t-s,--sequence \tPath to Sequence\tSpecify the sequence path" << std::endl
        << "\t-v,--video \tPath to Video \t\tSpecify the video path" << std::endl;
}


void ProcessCPU_Multithreaded(PBAS& pbasChannel1, PBAS& pbasChannel2, PBAS& pbasChannel3, cv::Mat& inputImage, cv::Mat& segResult)
{
    // Adapted from original PBAS Code

    // PRE-PROCESSING
    cv::Mat blurImage;
    cv::GaussianBlur(inputImage, blurImage, cv::Size(3, 3), 3); // use gaussian blur
    //cv::bilateralFilter(inputImage, blurImage, 5, 15, 15); // or use bilateral filter
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(blurImage, rgbChannels);

    // Run algorithm for each channel in seperate threads
    cv::Mat pbasResult1, pbasResult2, pbasResult3;
    std::future<bool> pbasThread1 = std::async(std::launch::async, &PBAS::process, &pbasChannel1, &rgbChannels.at(0), &pbasResult1);
    std::future<bool> pbasThread2 = std::async(std::launch::async, &PBAS::process, &pbasChannel2, &rgbChannels.at(1), &pbasResult2);
    std::future<bool> pbasThread3 = std::async(std::launch::async, &PBAS::process, &pbasChannel3, &rgbChannels.at(2), &pbasResult3);
    pbasThread1.wait();
    pbasThread2.wait();
    pbasThread3.wait();

    // or all foreground results of each rgb channel
    cv::bitwise_or(pbasResult1, pbasResult3, pbasResult1);
    cv::bitwise_or(pbasResult1, pbasResult2, pbasResult1);

    // copy or result of 3 channel into output variable
    pbasResult1.copyTo(segResult);
}


void ProcessCPU(PBAS& pbasChannel1, PBAS& pbasChannel2, PBAS& pbasChannel3, cv::Mat& inputImage, cv::Mat& segResult)
{
    // Adapted from original PBAS Code

    // PRE-PROCESSING
    cv::Mat blurImage;
    cv::GaussianBlur(inputImage, blurImage, cv::Size(3, 3), 3); // use gaussian blur
    //cv::bilateralFilter(inputImage, blurImage, 5, 15, 15); // or use bilateral filter
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(blurImage, rgbChannels);

    // Run algorithm for each channel in seperate threads
    cv::Mat pbasResult1, pbasResult2, pbasResult3;
    std::future<bool> pbasThread1 = std::async(std::launch::async, &PBAS::process, &pbasChannel1, &rgbChannels.at(0), &pbasResult1);
    pbasThread1.wait();
    std::future<bool> pbasThread2 = std::async(std::launch::async, &PBAS::process, &pbasChannel2, &rgbChannels.at(1), &pbasResult2);
    pbasThread2.wait();
    std::future<bool> pbasThread3 = std::async(std::launch::async, &PBAS::process, &pbasChannel3, &rgbChannels.at(2), &pbasResult3);
    pbasThread3.wait();

    // or all foreground results of each rgb channel
    cv::bitwise_or(pbasResult1, pbasResult3, pbasResult1);
    cv::bitwise_or(pbasResult1, pbasResult2, pbasResult1);

    // copy or result of 3 channel into output variable
    pbasResult1.copyTo(segResult);
}


void ProcessCUDA(PBAS_CUDA& PBAS_CUDA_Channel1, PBAS_CUDA& PBAS_CUDA_Channel2, PBAS_CUDA& PBAS_CUDA_Channel3, cv::Mat& inputImage, cv::Mat& segResult)
{
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(inputImage, rgbChannels);

    // Run algorithm for each channel in seperate threads
    cv::Mat pbasCudaResult1, pbasCudaResult2, pbasCudaResult3;
    std::future<void> pbasCudaThread1 = std::async(std::launch::async, &PBAS_CUDA::Process, &PBAS_CUDA_Channel1, std::ref(rgbChannels.at(0)), std::ref(pbasCudaResult1));
    std::future<void> pbasCudaThread2 = std::async(std::launch::async, &PBAS_CUDA::Process, &PBAS_CUDA_Channel2, std::ref(rgbChannels.at(1)), std::ref(pbasCudaResult2));
    std::future<void> pbasCudaThread3 = std::async(std::launch::async, &PBAS_CUDA::Process, &PBAS_CUDA_Channel3, std::ref(rgbChannels.at(2)), std::ref(pbasCudaResult3));
    pbasCudaThread1.wait();
    pbasCudaThread2.wait();
    pbasCudaThread3.wait();
 
    // or all foreground results of each rgb channel
    cv::bitwise_or(pbasCudaResult1, pbasCudaResult3, pbasCudaResult1);
    cv::bitwise_or(pbasCudaResult1, pbasCudaResult2, pbasCudaResult1);

    // copy or result of 3 channel into output variable
    pbasCudaResult1.copyTo(segResult);
}


int main(int argc, char* argv[])
{
    if (argc != 3 || std::string(argv[1]) == "-h") {
        show_usage();
        return 0;
    }

    cv::Mat inputFrame;
    cv::Mat outputFrameSingleCpuImp;
    cv::Mat outputFrameMultiCpuImp;
    cv::Mat outputFrameCudaImp;

    //PBAS Algorithm Parameters
    int N_historyQueueSize = 20;
    int BackgroundDecisionCount = 2; // (#min in paper)
    double R_Threshold = 18.0;
    double R_Threshold_Scale_Constant = 5.0;
    double R_Threshold_Inc_Dec_Rate = 5.0;
    int T_Update_Rate = 18;
    int T_Update_Rate_Lower_Boundary = 2;
    int T_Update_Rate_Upper_Boundary = 200;
    double T_Update_Rate_Inc_Rate = 0.05;
    double T_Update_Rate_Dec_Rate = 1.0;
    double alpha = 10.0;
    double beta = 1.0;
  
    // Create and construct PBAS(cpu) objects for each color channel.
    PBAS pbas_R, pbas_G, pbas_B;
    pbas_R.initialization(N_historyQueueSize, R_Threshold, BackgroundDecisionCount, T_Update_Rate, alpha, beta, R_Threshold_Scale_Constant, 
                          R_Threshold_Inc_Dec_Rate, T_Update_Rate_Inc_Rate, T_Update_Rate_Dec_Rate, T_Update_Rate_Lower_Boundary, T_Update_Rate_Upper_Boundary);
    pbas_G.initialization(N_historyQueueSize, R_Threshold, BackgroundDecisionCount, T_Update_Rate, alpha, beta, R_Threshold_Scale_Constant,
                          R_Threshold_Inc_Dec_Rate, T_Update_Rate_Inc_Rate, T_Update_Rate_Dec_Rate, T_Update_Rate_Lower_Boundary, T_Update_Rate_Upper_Boundary);
    pbas_B.initialization(N_historyQueueSize, R_Threshold, BackgroundDecisionCount, T_Update_Rate, alpha, beta, R_Threshold_Scale_Constant,
                          R_Threshold_Inc_Dec_Rate, T_Update_Rate_Inc_Rate, T_Update_Rate_Dec_Rate, T_Update_Rate_Lower_Boundary, T_Update_Rate_Upper_Boundary);

    //Create Objects for Multithreaded Version
    PBAS pbas_R_Multithreaded, pbas_G_Multithreaded, pbas_B_Multithreaded;
    pbas_R_Multithreaded.initialization(N_historyQueueSize, R_Threshold, BackgroundDecisionCount, T_Update_Rate, alpha, beta, R_Threshold_Scale_Constant, 
                          R_Threshold_Inc_Dec_Rate, T_Update_Rate_Inc_Rate, T_Update_Rate_Dec_Rate, T_Update_Rate_Lower_Boundary, T_Update_Rate_Upper_Boundary);
    pbas_G_Multithreaded.initialization(N_historyQueueSize, R_Threshold, BackgroundDecisionCount, T_Update_Rate, alpha, beta, R_Threshold_Scale_Constant,
                          R_Threshold_Inc_Dec_Rate, T_Update_Rate_Inc_Rate, T_Update_Rate_Dec_Rate, T_Update_Rate_Lower_Boundary, T_Update_Rate_Upper_Boundary);
    pbas_B_Multithreaded.initialization(N_historyQueueSize, R_Threshold, BackgroundDecisionCount, T_Update_Rate, alpha, beta, R_Threshold_Scale_Constant,
                          R_Threshold_Inc_Dec_Rate, T_Update_Rate_Inc_Rate, T_Update_Rate_Dec_Rate, T_Update_Rate_Lower_Boundary, T_Update_Rate_Upper_Boundary);

    // Create and construct PBAS_CUDA(gpu) objects for each color channel.
    PBAS_CUDA pbasCuda_R;
    PBAS_CUDA pbasCuda_G;
    PBAS_CUDA pbasCuda_B;

    // Sequence Input
    // TO-DO : Sequence input will be implemented!
    if (std::string(argv[1]) == "-s" || std::string(argv[1]) == "--sequence")
    {
        std::string seqPath = std::string(argv[2]);
        std::cout << "Sequence Input Path: " << seqPath << std::endl;
        std::cout << "Sequence input feature is not implemented yet! Please use video input feature!" << std::endl;
        return 0;
    }

    // Video Input
    if (std::string(argv[1]) == "-v" || std::string(argv[1]) == "--video")
    {
        std::string videoPath = std::string(argv[2]);
        std::cout << "Video Input Path: " << videoPath << std::endl;
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened())
        {
            std::cout << "!!! Failed to open video: " << videoPath << std::endl;
            return 0;
        }

        double totalTimeCpu = 0.0;
        double totalTimeCpuMultithreaded = 0.0;
        double totalTimeGpu = 0.0;

#ifdef TIME_RECORDER        
        std::ofstream singleCPUTime ("singleCPUTime.txt");
        std::ofstream multiCPUTime ("multiCPUTime.txt");
        std::ofstream GPUTime ("GPUTime.txt");
#endif

#ifdef OUTPUT_VIDEO_WRITER
        cv::VideoWriter inputVideo("inputVideo.avi",CV_FOURCC('M','J','P','G'),20, cv::Size(width,height));
        cv::VideoWriter gpuOutputVideo("gpuOut.avi",CV_FOURCC('M','J','P','G'),20, cv::Size(width,height),0);
        cv::VideoWriter singleCpuOutputVideo("cpuSingleOut.avi",CV_FOURCC('M','J','P','G'),20, cv::Size(width,height),0);
        cv::VideoWriter multiCpuOutputVideo("cpuMultiOut.avi",CV_FOURCC('M','J','P','G'),20, cv::Size(width,height),0);
#endif

        int frameCounter = 0;
        while(true)
        {
            cap.read(inputFrame);
            if (inputFrame.empty()) 
            {
                break;
            }

            auto processCPU_Start_time = std::chrono::high_resolution_clock::now();
            ProcessCPU(pbas_R, pbas_G, pbas_B, inputFrame, outputFrameSingleCpuImp);
            auto processCPU_Elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::high_resolution_clock::now() - processCPU_Start_time).count();
            totalTimeCpu += processCPU_Elapsed_time;


            auto processCPU_Multithreaded_Start_time = std::chrono::high_resolution_clock::now();
            ProcessCPU_Multithreaded(pbas_R_Multithreaded, pbas_G_Multithreaded, pbas_B_Multithreaded, inputFrame, outputFrameMultiCpuImp);
            auto processCPU_Multithreaded_Elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::high_resolution_clock::now() - processCPU_Multithreaded_Start_time).count();
            totalTimeCpuMultithreaded += processCPU_Multithreaded_Elapsed_time;  


            auto processGPU_Start_time = std::chrono::high_resolution_clock::now();
            ProcessCUDA(pbasCuda_R, pbasCuda_G, pbasCuda_B, inputFrame, outputFrameCudaImp);
            auto processGPU_Elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::high_resolution_clock::now() - processGPU_Start_time).count();
            totalTimeGpu += processGPU_Elapsed_time;

#ifdef SHOW_OUTPUT_VIDEO
            cv::imshow("Input",inputFrame);
            cv::imshow("SingleCPU-Result",outputFrameSingleCpuImp);
            cv::imshow("MultiCPU-Result",outputFrameMultiCpuImp);
            cv::imshow("GPU-Result",outputFrameCudaImp);
            cv::waitKey(1);
#endif

#ifdef OUTPUT_VIDEO_WRITER
            inputVideo << inputFrame;
            singleCpuOutputVideo << outputFrameSingleCpuImp;
            multiCpuOutputVideo << outputFrameMultiCpuImp;
            gpuOutputVideo << outputFrameCudaImp;
#endif

#ifdef OUTPUT_FRAME_WRITER
            cv::imwrite("SingleCpuResult_" + std::to_string(frameCounter) + ".png", outputFrameSingleCpuImp);
            cv::imwrite("MultiCpuResult_" + std::to_string(frameCounter) + ".png", outputFrameMultiCpuImp);
            cv::imwrite("CudaResult_" + std::to_string(frameCounter) + ".png", outputFrameCudaImp);
#endif

#ifdef TIME_RECORDER  
            singleCPUTime << double(processCPU_Elapsed_time)/1000.0 << std::endl;
            multiCPUTime << double(processCPU_Multithreaded_Elapsed_time)/1000.0 << std::endl;
            GPUTime << double(processGPU_Elapsed_time)/1000.0 << std::endl;
#endif

            std::cout << "-------------------------------------------------------" << std::endl;
            std::cout << "Processed Frame Id: " << frameCounter << std::endl;
            std::cout << "Process Time Single-Cpu: " << processCPU_Elapsed_time << std::endl;
            std::cout << "Process Time Multithreaded-Cpu: " << processCPU_Multithreaded_Elapsed_time << std::endl;
            std::cout << "Process Time Gpu: " << processGPU_Elapsed_time << std::endl << std::endl;

            frameCounter++;  
        }

        std::cout << "-------------------------------------------------------" << std::endl;   
        std::cout << "#######################################################" << std::endl;       
        std::cout << "totalTimeCpu: " << totalTimeCpu << std::endl;
        std::cout << "totalTimeCpu-Multithreaded: " << totalTimeCpuMultithreaded << std::endl;
        std::cout << "totalTimeGpu: " << totalTimeGpu << std::endl;
        std::cout << "Average Elapsed Time for Single-Cpu: " << totalTimeCpu/frameCounter << std::endl;
        std::cout << "Average Elapsed Time for  Multithreaded-Cpu: " << totalTimeCpuMultithreaded/frameCounter << std::endl;
        std::cout << "Average Elapsed Time for Gpu: " << totalTimeGpu/frameCounter << std::endl;
        std::cout << "#######################################################" << std::endl;      
        std::cout << "-------------------------------------------------------" << std::endl;   

        return 0;
    }

    show_usage();
    return 0;
}