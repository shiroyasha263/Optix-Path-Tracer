#pragma once

#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
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

class RenderState
{
public:
    RenderState(unsigned int width, unsigned int height);
    void initLaunchParams();
    void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer);
    void createContext();
    void buildMeshAccel();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();
    void cleanUp();

public:

    OptixDeviceContext context = 0;

    OptixTraversableHandle         gas_handle = 0;  // Traversable handle for triangle AS
    CUdeviceptr                    d_gas_output_buffer = 0;  // Triangle AS memory
    CUdeviceptr                    d_vertices = 0;

    OptixModule                    ptx_module = 0;
    OptixModule                    sphere_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              raygen_prog_group = 0;
    OptixProgramGroup              radiance_miss_group = 0;
    OptixProgramGroup              radiance_hit_group = 0;

    CUstream                       stream = 0;
    Params                         params;
    Params* d_params;

    OptixShaderBindingTable        sbt = {};

    CUdeviceptr                    d_vertex_buffer = 0;
    CUdeviceptr                    d_radius_buffer = 0;
};