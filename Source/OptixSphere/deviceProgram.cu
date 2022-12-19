#include <optix.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include "Params.h"

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ Params params;

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

static __forceinline__ __device__ void* unpackPointer(unsigned int u0, unsigned int u1) {
    const unsigned long long uptr = static_cast<unsigned long long>(u0) << 32 | u1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void* packPointer(void* ptr, unsigned int& u0, unsigned int& u1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    u0 = uptr >> 32;
    u1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return (reinterpret_cast<T*>(unpackPointer(u0, u1)));
}

extern "C" __global__ void __closesthit__radiance() {
    float3& result = *(float3*)getPRD<float3>();
    result = make_float3(0.0f);
}


//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    float3& result = *(float3*)getPRD<float3>();
    result = make_float3(1.0f);
}



//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{

    const unsigned int w = params.img_width;
    const unsigned int h = params.img_height;
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int subframe_index = params.subframe_index;

    float3 result = make_float3(0.0f);
    int i = params.samples_per_launch;
    
    const float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(w),
        static_cast<float>(idx.y) / static_cast<float>(h)) - 1.0f;

    float3 rayDir = normalize(d.x * U + d.y * V + W);
    float3 origin = eye;

    unsigned int u0, u1;
    packPointer(&result, u0, u1);

    optixTrace(
        params.handle,
        origin,
        rayDir,
        0.0f,
        1e20f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,
        1,
        0,
        u0, u1
    );

    if (params.subframe_index == 0 &&
        optixGetLaunchIndex().x == 0 &&
        optixGetLaunchIndex().y == 0) {
        // we could of course also have used optixGetLaunchDims to query
        // the launch size, but accessing the optixLaunchParams here
        // makes sure they're not getting optimized away (because
        // otherwise they'd not get used)
        printf("############################################\n");
        printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
            params.img_width,
            params.img_height);
        printf("############################################\n");
    }

    // ------------------------------------------------------------------
    // for this example, produce a simple test pattern:
    // ------------------------------------------------------------------

    // compute a test pattern based on pixel ID

    // and write to frame buffer ...
    const uint32_t fbIndex = idx.x + idx.y * params.img_width;
    params.frame_buffer[fbIndex] = make_color(result);
}