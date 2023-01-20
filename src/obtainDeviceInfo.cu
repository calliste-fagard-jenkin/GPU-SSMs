#include "../include/obtainDeviceInfo.cuh"
#include <iostream>

void getInfo(){
    /*
    purpose : This function prints out information about the available CUDA devices on 
              the system
    */
    int count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&count);
    std::cout << "This program has identified " << count << " Nvidia GPUs" << std::endl;
    
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "The first device is called " << prop.name << std::endl;
}

void obtainDeviceInfo(){
    /*
    purpose : Acts as a wrapper for getInfo so that this function can be called
              from a regular C++ file
    */
    getInfo();
}
