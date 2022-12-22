#include "RenderState.h"
#include <cuda/sphere.h>

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


struct Vertex
{
    float x, y, z, pad;
};


struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};


struct Instance
{
    float transform[12];
};

const int32_t TRIANGLE_COUNT = 32;
const int32_t MAT_COUNT = 1;

const static std::array<Vertex, TRIANGLE_COUNT * 3> g_vertices =
{ {
        // Floor  -- white lambert
        {    0.0f,    0.0f,    0.0f, 0.0f },
        {    0.0f,    0.0f,  559.2f, 0.0f },
        {  556.0f,    0.0f,  559.2f, 0.0f },
        {    0.0f,    0.0f,    0.0f, 0.0f },
        {  556.0f,    0.0f,  559.2f, 0.0f },
        {  556.0f,    0.0f,    0.0f, 0.0f },

        // Ceiling -- white lambert
        {    0.0f,  548.8f,    0.0f, 0.0f },
        {  556.0f,  548.8f,    0.0f, 0.0f },
        {  556.0f,  548.8f,  559.2f, 0.0f },

        {    0.0f,  548.8f,    0.0f, 0.0f },
        {  556.0f,  548.8f,  559.2f, 0.0f },
        {    0.0f,  548.8f,  559.2f, 0.0f },

        // Back wall -- white lambert
        {    0.0f,    0.0f,  559.2f, 0.0f },
        {    0.0f,  548.8f,  559.2f, 0.0f },
        {  556.0f,  548.8f,  559.2f, 0.0f },

        {    0.0f,    0.0f,  559.2f, 0.0f },
        {  556.0f,  548.8f,  559.2f, 0.0f },
        {  556.0f,    0.0f,  559.2f, 0.0f },

        // Right wall -- green lambert
        {    0.0f,    0.0f,    0.0f, 0.0f },
        {    0.0f,  548.8f,    0.0f, 0.0f },
        {    0.0f,  548.8f,  559.2f, 0.0f },

        {    0.0f,    0.0f,    0.0f, 0.0f },
        {    0.0f,  548.8f,  559.2f, 0.0f },
        {    0.0f,    0.0f,  559.2f, 0.0f },

        // Left wall -- red lambert
        {  556.0f,    0.0f,    0.0f, 0.0f },
        {  556.0f,    0.0f,  559.2f, 0.0f },
        {  556.0f,  548.8f,  559.2f, 0.0f },

        {  556.0f,    0.0f,    0.0f, 0.0f },
        {  556.0f,  548.8f,  559.2f, 0.0f },
        {  556.0f,  548.8f,    0.0f, 0.0f },

        // Short block -- white lambert
        {  130.0f,  165.0f,   65.0f, 0.0f },
        {   82.0f,  165.0f,  225.0f, 0.0f },
        {  242.0f,  165.0f,  274.0f, 0.0f },

        {  130.0f,  165.0f,   65.0f, 0.0f },
        {  242.0f,  165.0f,  274.0f, 0.0f },
        {  290.0f,  165.0f,  114.0f, 0.0f },

        {  290.0f,    0.0f,  114.0f, 0.0f },
        {  290.0f,  165.0f,  114.0f, 0.0f },
        {  240.0f,  165.0f,  272.0f, 0.0f },

        {  290.0f,    0.0f,  114.0f, 0.0f },
        {  240.0f,  165.0f,  272.0f, 0.0f },
        {  240.0f,    0.0f,  272.0f, 0.0f },

        {  130.0f,    0.0f,   65.0f, 0.0f },
        {  130.0f,  165.0f,   65.0f, 0.0f },
        {  290.0f,  165.0f,  114.0f, 0.0f },

        {  130.0f,    0.0f,   65.0f, 0.0f },
        {  290.0f,  165.0f,  114.0f, 0.0f },
        {  290.0f,    0.0f,  114.0f, 0.0f },

        {   82.0f,    0.0f,  225.0f, 0.0f },
        {   82.0f,  165.0f,  225.0f, 0.0f },
        {  130.0f,  165.0f,   65.0f, 0.0f },

        {   82.0f,    0.0f,  225.0f, 0.0f },
        {  130.0f,  165.0f,   65.0f, 0.0f },
        {  130.0f,    0.0f,   65.0f, 0.0f },

        {  240.0f,    0.0f,  272.0f, 0.0f },
        {  240.0f,  165.0f,  272.0f, 0.0f },
        {   82.0f,  165.0f,  225.0f, 0.0f },

        {  240.0f,    0.0f,  272.0f, 0.0f },
        {   82.0f,  165.0f,  225.0f, 0.0f },
        {   82.0f,    0.0f,  225.0f, 0.0f },

        // Tall block -- white lambert
        {  423.0f,  330.0f,  247.0f, 0.0f },
        {  265.0f,  330.0f,  296.0f, 0.0f },
        {  314.0f,  330.0f,  455.0f, 0.0f },

        {  423.0f,  330.0f,  247.0f, 0.0f },
        {  314.0f,  330.0f,  455.0f, 0.0f },
        {  472.0f,  330.0f,  406.0f, 0.0f },

        {  423.0f,    0.0f,  247.0f, 0.0f },
        {  423.0f,  330.0f,  247.0f, 0.0f },
        {  472.0f,  330.0f,  406.0f, 0.0f },

        {  423.0f,    0.0f,  247.0f, 0.0f },
        {  472.0f,  330.0f,  406.0f, 0.0f },
        {  472.0f,    0.0f,  406.0f, 0.0f },

        {  472.0f,    0.0f,  406.0f, 0.0f },
        {  472.0f,  330.0f,  406.0f, 0.0f },
        {  314.0f,  330.0f,  456.0f, 0.0f },

        {  472.0f,    0.0f,  406.0f, 0.0f },
        {  314.0f,  330.0f,  456.0f, 0.0f },
        {  314.0f,    0.0f,  456.0f, 0.0f },

        {  314.0f,    0.0f,  456.0f, 0.0f },
        {  314.0f,  330.0f,  456.0f, 0.0f },
        {  265.0f,  330.0f,  296.0f, 0.0f },

        {  314.0f,    0.0f,  456.0f, 0.0f },
        {  265.0f,  330.0f,  296.0f, 0.0f },
        {  265.0f,    0.0f,  296.0f, 0.0f },

        {  265.0f,    0.0f,  296.0f, 0.0f },
        {  265.0f,  330.0f,  296.0f, 0.0f },
        {  423.0f,  330.0f,  247.0f, 0.0f },

        {  265.0f,    0.0f,  296.0f, 0.0f },
        {  423.0f,  330.0f,  247.0f, 0.0f },
        {  423.0f,    0.0f,  247.0f, 0.0f },

        // Ceiling light -- emmissive
        {  343.0f,  548.6f,  227.0f, 0.0f },
        {  213.0f,  548.6f,  227.0f, 0.0f },
        {  213.0f,  548.6f,  332.0f, 0.0f },

        {  343.0f,  548.6f,  227.0f, 0.0f },
        {  213.0f,  548.6f,  332.0f, 0.0f },
        {  343.0f,  548.6f,  332.0f, 0.0f }
    } };

static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices = { {
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- white lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
    3, 3                           // Ceiling light -- emmissive
} };



const std::array<float3, MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f,  0.0f },

} };


const std::array<float3, MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
} };

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void RenderState::initLaunchParams()
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)
    ));
    params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    //Make this a variable instead
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    params.samples_per_launch = 1;
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    params.subframe_index = 0u;

    params.light.emission = make_float3(15.0f, 15.0f, 5.0f);
    params.light.corner = make_float3(343.0f, 548.5f, 227.0f);
    params.light.v1 = make_float3(0.0f, 0.0f, 105.0f);
    params.light.v2 = make_float3(-130.0f, 0.0f, 0.0f);
    params.light.normal = normalize(cross(params.light.v1, params.light.v2));
    params.handle = gas_handle;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));

}

void RenderState::launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
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
        params.width,   // launch width
        params.height,  // launch height
        1                     // launch depth
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void RenderState::createContext()
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
}

void RenderState::buildMeshAccel()
{
    //
    // copy mesh data to device
    //
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // sphere build input

    std::vector<float3> sphereVertex;
    std::vector<float> sphereRadius;

    sphereVertex.push_back(make_float3(0.0f));
    sphereVertex.push_back(make_float3(0.0f, -100.5f, 0.0f));

    sphereRadius.push_back(0.5f);
    sphereRadius.push_back(100.0f);

    d_vertex_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex_buffer), sphereVertex.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertex_buffer), sphereVertex.data(),
        sphereVertex.size() * sizeof(float3), cudaMemcpyHostToDevice));

    d_radius_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius_buffer), sphereRadius.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius_buffer), sphereRadius.data(), sphereRadius.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    OptixBuildInput sphere_input = {};

    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.vertexBuffers = &d_vertex_buffer;
    sphere_input.sphereArray.numVertices = (int)sphereVertex.size();
    sphere_input.sphereArray.radiusBuffers = &d_radius_buffer;

    uint32_t sphere_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    sphere_input.sphereArray.flags = sphere_input_flags;

    // in this example we have one SBT entry, and no per-primitive
    // materials:
    sphere_input.sphereArray.numSbtRecords = 1;
    sphere_input.sphereArray.sbtIndexOffsetBuffer = 0; // THE HELL IS THIS?
    sphere_input.sphereArray.sbtIndexOffsetSizeInBytes = 0;
    sphere_input.sphereArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &sphere_input, 1, &gas_buffer_sizes));
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

    OPTIX_CHECK(optixAccelBuild(context,
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
        OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
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
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

    size_t      inputSize = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "deviceProgram.cu", inputSize);

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &ptx_module
    ));

    OptixBuiltinISOptions builtin_is_options = {};

    builtin_is_options.usesMotionBlur = false;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        &builtin_is_options,
        &sphere_module));
}

void RenderState::createProgramGroups() {
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof(log);

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context, &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group
        ));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log, &sizeof_log,
            &radiance_miss_group
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        hit_prog_group_desc.hitgroup.moduleAH = nullptr;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        hit_prog_group_desc.hitgroup.moduleIS = sphere_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &radiance_hit_group
        ));
    }
}

void RenderState::createPipeline() {
    OptixProgramGroup program_groups[] =
    {
        raygen_prog_group,
        radiance_miss_group,
        radiance_hit_group,
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        context,
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
    OPTIX_CHECK(optixUtilAccumulateStackSizes(raygen_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(radiance_miss_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(radiance_hit_group, &stack_sizes));

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

    // For a single raygen program, we create a record storage in the table
    // where we assign this raygen record data to it
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    // For a single miss program, we create a record storage in the table
    // where we assign this miss record data to it
    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

    MissRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(radiance_miss_group, &ms_sbt));
    ms_sbt.data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_records),
        &ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));

    //Things change here as compared to before, here we do the same thing but for
    //every different type of rays and object
    //In this example we are only using one type of object so not an issue
    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT
    ));

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * MAT_COUNT];
    for (int i = 0; i < MAT_COUNT; ++i)
    {
        //Since we only have one type of object
        int objectType = 0;
        HitGroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(radiance_hit_group, &rec));
        rec.data.diffuse_color = make_float3(0.5f);
        rec.data.vertex = reinterpret_cast<float3*>(d_vertex_buffer);
        rec.data.radius = reinterpret_cast<float*>(d_radius_buffer);
        hitgroup_records[i] = rec;
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

void RenderState::cleanUp() {
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(radiance_hit_group));
    OPTIX_CHECK(optixModuleDestroy(ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));


    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}

RenderState::RenderState(unsigned int width, unsigned int height) {
    params.width = width;
    params.height = height;

    createContext();
    buildMeshAccel();
    createModule();
    createProgramGroups();
    createPipeline();
    createSBT();
    initLaunchParams();
}