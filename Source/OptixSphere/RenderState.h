#pragma once

#include <optix.h>
#include <sutil/sutil.h>
#include <sutil/CUDAOutputBuffer.h>
#include "Params.h"
#include <sutil/vec_math.h>
#include <optix_stubs.h>
#include <iomanip>
#include <optix_stack_size.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>

class RenderState {
public:
    RenderState();
	RenderState(unsigned int width, unsigned int height);
	void launchSubFrame(sutil::CUDAOutputBuffer<uchar4>& output_buffer);
	void resize(unsigned int width, unsigned int height);
public:
    void initLaunchParams(unsigned int width, unsigned int height);

    void createContext();

    void createModule();

    void createRaygenPrograms();

    void createMissPrograms();

    void createHitgroupPrograms();

    void createPipeline();

    void createSBT();

    void buildAccel();
public:
    CUstream           stream;
   
    OptixDeviceContext optixContext;

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    OptixModule                 module;
    OptixModule                 sphere_module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    std::vector<OptixProgramGroup> raygenPGs;
    CUdeviceptr raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUdeviceptr missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUdeviceptr hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    Params params;
    Params*   d_params;
    /*! @} */

    CUdeviceptr colorBuffer;

    /*! the camera we are to render with. */
    sutil::Camera camera;

    /*! the model we are going to trace rays against */
    CUdeviceptr vertexBuffer;
    CUdeviceptr indexBuffer;
    CUdeviceptr asBuffer;
};