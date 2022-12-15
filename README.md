# Optix-Ray-Tracer

## The first code sample

This is the starting point of the optix api, all we do in the first code is initialise optix and check if it has been initialised as planned

I am still kind of new to this, so I have copied most of the repositories from the OPTIX SDK, and modified the CMakeLists a bit to add one subdirectory and create a cmake folder for that specifying how to make the files present inside it

Now we make type out our first line of code, we start by adding the include libraries - 

```#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>
```

Then we start with creating a function to initialize cuda, check the number of devices and create a optixInit

```void initOptix() {
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if(numDevices == 0)
		throw std::runtime_error("No CUDA Capable Device");
	std::cout << "Found " << numDevices << "Cuda Devices\n";

	OPTIX_CHECK( optixInit() );
}
```

Now finally we make our main function where will try to check if optix is initialised without any issues or not

```int main(int ac, char **av) {
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
```