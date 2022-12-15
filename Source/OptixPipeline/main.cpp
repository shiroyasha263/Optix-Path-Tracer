#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "Params.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sutil/CUDAOutputBuffer.h>
#include <sampleConfig.h>

#include "RenderState.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iomanip>
#include <iostream>
#include <string>

int main(int ac, char** av)
{
    try {
        const unsigned int width = 1200;
        const unsigned int height = 1024;
        RenderState sample(width, height);
        sample.resize(width, height);
        sample.launchSubFrame();

        std::cout << "Done with launchSubFrame";
        std::vector<uint32_t> pixels(width * height);
        sample.downloadPixels(pixels.data());

        const std::string fileName = "osc_example2.png";
        stbi_write_png(fileName.c_str(), width, height, 4,
            pixels.data(), width * sizeof(uint32_t));
        std::cout << std::endl
            << "Image rendered, and saved to " << fileName << " ... done." << std::endl
            << std::endl;
    }
    catch (std::runtime_error& e) {
        std::cout << "FATAL ERROR: " << e.what()
            << std::endl;
        exit(1);
    }
    return 0;
}