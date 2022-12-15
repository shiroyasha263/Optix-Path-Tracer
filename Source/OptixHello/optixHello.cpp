#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>
//#include <sutil/CUDAOutputBuffer.h>
//#include <sampleConfig.h>
//
//#include "optixHello.h"
//
//#include <iomanip>
//#include <iostream>
//#include <string>

//Initialises optix and checks for errors
void initOptix() {
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if(numDevices == 0)
		throw std::runtime_error("No CUDA Capable Device");
	std::cout << "Found " << numDevices << "Cuda Devices\n";

	OPTIX_CHECK( optixInit() );
}

int main(int ac, char **av) {
	try {
		std::cout << "Initializing optix.. \n";

		optixInit();

		std::cout << "Succesfully Initialized \n";

		std::cout << "Done. Clean Exit \n";
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << "\n";
		exit(1);
	}
	return 0;
}