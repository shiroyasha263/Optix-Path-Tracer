#pragma once

#include <optix.h>
#include <sutil/sutil.h>
#include <sutil/CUDAOutputBuffer.h>
#include "Params.h"


class RenderState {
public:
    RenderState();

    RenderState(unsigned int width, unsigned int height);

    void initLaunchParams(unsigned int width, unsigned int height);

    void launchSubFrame();

    //void displaySubFrame();

    void createContext();

    void createModule();

    void createProgramGroup();

    void createPipeline();

    void createSBT();

    void resize(const unsigned int width, const unsigned int height);

    void downloadPixels(uint32_t pixels[]);

    ~RenderState();

private:
    OptixDeviceContext             optixContext = 0;

    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              raygen_prog = 0;
    OptixProgramGroup              miss_prog = 0;
    OptixProgramGroup              hitgroup_prog = 0;

    CUstream                       stream = 0;
    Params                         params;
    Params*                        d_params;

    OptixShaderBindingTable        sbt = {};
    CUdeviceptr                    color_buffer;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    int objectID;
};

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};