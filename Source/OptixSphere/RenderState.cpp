#include "RenderState.h"

# define PRINT(var) std::cout << #var << "=" << var << std::endl;

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
};

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

RenderState::RenderState() {
    initLaunchParams(768, 768);

    std::cout << "creating optix context ..." << std::endl;
    createContext();

    std::cout << "creating optix AS ..." << std::endl;
    buildAccel();

    std::cout << "setting up module ..." << std::endl;
    createModule();

    std::cout << "creating program groups ..." << std::endl;
    createRaygenPrograms();
    createMissPrograms();
    createHitgroupPrograms();

    std::cout << "setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "building SBT ..." << std::endl;
    createSBT();
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
}

RenderState::RenderState(unsigned int width, unsigned int height) {
    initLaunchParams(width, height);

    std::cout << "creating optix context ..." << std::endl;
    createContext();

    std::cout << "creating optix AS ..." << std::endl;
    buildAccel();
    
    std::cout << "setting up module ..." << std::endl;
    createModule();

    std::cout << "creating program groups ..." << std::endl;
    createRaygenPrograms();
    createMissPrograms();
    createHitgroupPrograms();

    std::cout << "setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "building SBT ..." << std::endl;
    createSBT();
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
}

void RenderState::initLaunchParams(unsigned int width, unsigned int height) {
    params.img_width = width;
    params.img_height = height;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.img_width * params.img_height * sizeof(float4)
    ));
    params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    params.samples_per_launch = 10;
    params.subframe_index = 0u;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
}

void RenderState::createContext() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &optixContext));
}

void RenderState::buildAccel() {

    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // sphere build input

    float3 sphereVertex = make_float3(0.f, 0.f, 0.f);
    float  sphereRadius = 1.5f;

    CUdeviceptr d_vertex_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex_buffer), sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertex_buffer), &sphereVertex,
        sizeof(float3), cudaMemcpyHostToDevice));

    CUdeviceptr d_radius_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius_buffer), sizeof(float)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius_buffer), &sphereRadius, sizeof(float),
        cudaMemcpyHostToDevice));

    OptixBuildInput sphere_input = {};

    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.vertexBuffers = &d_vertex_buffer;
    sphere_input.sphereArray.numVertices = 1;
    sphere_input.sphereArray.radiusBuffers = &d_radius_buffer;

    uint32_t sphere_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    sphere_input.sphereArray.flags = sphere_input_flags;
    sphere_input.sphereArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accel_options, &sphere_input, 1, &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(optixContext,
        0,  // CUDA stream
        &accel_options, &sphere_input,
        1,  // num build inputs
        d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes, &gas_handle,
        &emitProperty,  // emitted property list
        1               // num emitted properties
    ));

    d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
    CUDA_CHECK(cudaFree((void*)d_vertex_buffer));
    CUDA_CHECK(cudaFree((void*)d_radius_buffer));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(optixContext, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }

    params.handle = gas_handle;
}

void RenderState::createModule() {
    moduleCompileOptions.maxRegisterCount           =   50;
    moduleCompileOptions.optLevel                   =   OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel                 =   OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipelineCompileOptions                          =   {};
    pipelineCompileOptions.traversableGraphFlags    =   OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur           =   false;
    pipelineCompileOptions.numPayloadValues         =   2;
    pipelineCompileOptions.numAttributeValues       =   2;
    pipelineCompileOptions.exceptionFlags           =   OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.usesPrimitiveTypeFlags   =   OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    pipelineLinkOptions.maxTraceDepth               =   2;

    size_t      input_size  = 0;
    const char* input       = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "deviceProgram.cu", input_size);
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
        optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        input,
        input_size,
        log, &sizeof_log,
        &module));


    OptixBuiltinISOptions builtin_is_options = {};

    builtin_is_options.usesMotionBlur = false;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(optixContext, &moduleCompileOptions, &pipelineCompileOptions,
        &builtin_is_options, &sphere_module));
}

void RenderState::createRaygenPrograms() {
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &raygenPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void RenderState::createMissPrograms()
{
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void RenderState::createHitgroupPrograms() {
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    pgDesc.hitgroup.moduleIS = sphere_module;
    pgDesc.hitgroup.entryFunctionNameIS = nullptr;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void RenderState::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log, &sizeof_log,
        &pipeline
    ));
    if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
    (/* [in] The pipeline to configure the stack size for */
        pipeline,
        /* [in] The direct stack size requirement for direct
           callables invoked from IS or AH. */
        2 * 1024,
        /* [in] The direct stack size requirement for direct
           callables invoked from RG, MS, or CH.  */
        2 * 1024,
        /* [in] The continuation stack requirement. */
        2 * 1024,
        /* [in] The maximum depth of a traversable graph
           passed to trace. */
        1));
    if (sizeof_log > 1) PRINT(log);
}

void RenderState::createSBT() {
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygenRecordsBuffer), raygenRecords.size() * sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygenRecordsBuffer), &raygenRecords, raygenRecords.size() * sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    sbt.raygenRecord = raygenRecordsBuffer;

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&missRecordsBuffer), missRecords.size() * sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(missRecordsBuffer), &missRecords, missRecords.size() * sizeof(MissRecord), cudaMemcpyHostToDevice));
    sbt.missRecordBase = missRecordsBuffer;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    int numObjects = 1;
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i = 0; i < numObjects; i++) {
        int objectType = 0;
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
        rec.objectID = i;
        hitgroupRecords.push_back(rec);
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroupRecordsBuffer), hitgroupRecords.size() * sizeof(HitgroupRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroupRecordsBuffer), &hitgroupRecords, hitgroupRecords.size() * sizeof(HitgroupRecord), cudaMemcpyHostToDevice));
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void RenderState::launchSubFrame(sutil::CUDAOutputBuffer<uchar4>& output_buffer) {

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(d_params),
        &params, sizeof(Params),
        cudaMemcpyHostToDevice, stream
    ));

    OPTIX_CHECK(optixLaunch(
        pipeline,
        stream,
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(Params),
        &sbt,
        params.img_width,   // launch width
        params.img_height,  // launch height
        1                     // launch depth
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void RenderState::resize(unsigned int width, unsigned int height) {
    if (width == 0 | height == 0) return;

    // resize our cuda frame buffer
    if (colorBuffer) cudaFree(reinterpret_cast<void*>(colorBuffer));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&colorBuffer), width * height * sizeof(uchar4)));

    // update the launch parameters that we'll pass to the optix
    // launch:
    params.img_width = width;
    params.img_height = height;
    params.frame_buffer = reinterpret_cast<uchar4*>(colorBuffer);
}