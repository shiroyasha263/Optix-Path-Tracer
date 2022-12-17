#include "RenderState.h"
#include <optix_stubs.h>
#include <iomanip>
#include <optix_stack_size.h>

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
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

void RenderState::createContext() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    optixContext = context;
}

void RenderState::createModule() {
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "deviceProgram.cu", inputSize);

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        optixContext,
        &module_compile_options,
        &pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &ptx_module
    ));
}

void RenderState::createProgramGroup() {
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof(log);

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__renderFrame";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            optixContext, &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog
        ));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            optixContext, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log, &sizeof_log,
            &miss_prog
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            optixContext,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &hitgroup_prog
        ));
    }
}

void RenderState::createPipeline() {
    OptixProgramGroup program_groups[] =
    {
        raygen_prog,
        miss_prog,
        hitgroup_prog
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        optixContext,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &pipeline
    ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(raygen_prog, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(miss_prog, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(hitgroup_prog, &stack_sizes));

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}

void RenderState::createSBT() {
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof(RaygenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    RaygenRecord rg_sbt;
    rg_sbt.data = nullptr;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));


    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

    MissRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog, &ms_sbt));
    ms_sbt.data = nullptr;

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_records),
        &ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));

    const int MAT_COUNT = 1;
    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT
    ));

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * MAT_COUNT];
    for (int i = 0; i < MAT_COUNT; ++i) {

        int objectType = 0;  // SBT for radiance ray-type for ith material

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog, &hitgroup_records[objectType]));
        hitgroup_records[objectType].objectID = 1;
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records,
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT,
        cudaMemcpyHostToDevice
    ));

    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_records;
    sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    sbt.missRecordCount = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase = d_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    sbt.hitgroupRecordCount = RAY_TYPE_COUNT * MAT_COUNT;
}

void RenderState::resize(const unsigned int width, const unsigned int height) {
    if (width == 0 || height == 0) return;

    if (color_buffer) cudaFree(reinterpret_cast<void*>(color_buffer));

    //Error might be here somewhere
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&color_buffer), width * height * sizeof(uchar4)));
    params.img_width = width;
    params.img_height = height;
    params.frame_buffer = reinterpret_cast<uchar4*>(color_buffer);
}

void RenderState::downloadPixels(uint32_t pixels[]) {
    CUDA_CHECK(cudaMemcpy(pixels, reinterpret_cast<void*>(color_buffer), params.img_width * params.img_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

RenderState::RenderState() {

    initLaunchParams(768, 768);

    std::cout << "creating optix context ..." << std::endl;
    createContext();

    std::cout << "setting up module ..." << std::endl;
    createModule();

    std::cout << "creating program groups ..." << std::endl;
    createProgramGroup();

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

    std::cout << "setting up module ..." << std::endl;
    createModule();

    std::cout << "creating program groups ..." << std::endl;
    createProgramGroup();

    std::cout << "setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "building SBT ..." << std::endl;
    createSBT();
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
}

RenderState::~RenderState() {
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog));
    OPTIX_CHECK(optixModuleDestroy(ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(optixContext));


    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}