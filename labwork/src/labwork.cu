#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <time.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            printf("labwork 1 OpenMP ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            //labwork.labwork5_CPU();
            //labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
        printf("GPU #%d\n", i);
        printf("GPU name: %s\n", prop.name);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Number of cores: %d\n", getSPcores(prop));
        printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Memory Clock Rate: %d\n", prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\nDevices", prop.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

}

// Write a grayscale kernel
__global__ void grayscale(uchar3 *input, uchar3 *output) {
    // this will execute in a device core
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y +input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
 
    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount *sizeof(uchar3));

    // Copy InputImage from CPU (host) to GPU (device)
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing : launch the kernel
    int blockSize = 1024;
    int numBlock = pixelCount / blockSize;  
    grayscale<<<numBlock, blockSize>>>(devInput, devOutput);

    // Copy CUDA Memory from GPU to CPU
    // allocate memory for the output on the host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));  
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);   

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

// Write a grayscale kernel
__global__ void grayscaleVer2D(uchar3 *input, uchar3 *output, int width, int height) {
    // this will execute in a device core
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy >= height) return;

    int tid = tidx + tidy * width;
    output[tid].x = (input[tid].x + input[tid].y +input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
 
    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount *sizeof(uchar3));

    // Copy InputImage from CPU (host) to GPU (device)
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing : launch the kernel
    // int blockSize = 1024;
    // int numBlock = pixelCount / blockSize;  
    // grayscale<<<numBlock, blockSize>>>(devInput, devOutput);
    dim3 blockSize = dim3(32, 32);
    //dim3 gridSize = dim3(8, 8);
    dim3 gridSize = dim3((inputImage->width + blockSize.x -1) / blockSize.x, (inputImage->height + blockSize.y -1) / blockSize.y);
    grayscaleVer2D<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    // allocate memory for the output on the host
    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));  
    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);   

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}

void Labwork::labwork5_GPU() {
}

void Labwork::labwork6_GPU() {
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























