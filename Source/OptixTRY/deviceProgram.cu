#include <optix.h>

#include "Params.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

extern "C" {
    __constant__ Params params;
}



//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    float3       color;
    unsigned int seed;
    int          countEmitted;
    int          done;
    int          pad;
};


struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct HitData {
    float3 center;
    float radius;
    float3 normal;
    float3 N;
    float3 P;
    float t_hit;
};

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}


static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>(unpackPointer(u0, u1));
}


static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
{
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}


static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    RadiancePRD* prd
)
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,        // SBT offset
        RAY_TYPE_COUNT,           // SBT stride
        RAY_TYPE_RADIANCE,        // missSBTIndex
        u0, u1);
}


static __forceinline__ __device__ void diffuse_scatter(const Ray& ray_in, const HitData& hitData, RadiancePRD* prd,
    float3& attenuation) {
    unsigned int seed = prd->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);

    float3 w_in;
    cosine_sample_hemisphere(z1, z2, w_in);
    Onb onb(hitData.N);
    onb.inverse_transform(w_in);
    prd->direction = w_in;
    prd->origin = hitData.P;
}

static __forceinline__ __device__ void specular_scatter(const Ray& ray_in, const HitData& hitData, RadiancePRD* prd, 
    float3& attenuation, float f) {
    
    float fuzz = f < 1 ? f : 1;

    unsigned int seed = prd->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);

    float3 unit_dir = normalize(ray_in.direction);
    float3 reflected = reflect(unit_dir, hitData.N);
    
    float3 w_in;
    cosine_sample_hemisphere(z1, z2, w_in);
    Onb onb(reflected);
    onb.inverse_transform(w_in);

    if (dot(reflected + fuzz * w_in, hitData.N) < 0) {
        prd->done = true;
        attenuation = make_float3(0.0f);
    }
    prd->origin = hitData.P;
    prd->direction = reflected + fuzz * w_in;
}

static __forceinline__ __device__ void dielectric_scatter(const Ray& ray_in, const HitData& hitData, RadiancePRD* prd,
    float3& attenuation, float index_of_refraction) {
    
    attenuation = make_float3(1.0f);
    float refraction_index = dot(hitData.N, hitData.normal) > 0 ? (1.0 / index_of_refraction) : index_of_refraction;

    float3 unit_dir = normalize(ray_in.direction);

    float cos_theta = fmin(dot(-unit_dir, hitData.normal), 1.0f);
    float3 r_out_prep = refraction_index * (unit_dir + cos_theta * hitData.normal);
    float3 r_out_parallel = -sqrt(fabs(1.0f - length(r_out_prep) * length(r_out_prep))) * hitData.N;
    float3 refracted = r_out_prep + r_out_parallel;

    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_index * sin_theta > 1.0;
    float3 direction;

    float r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    float reflectance = r0 + (1 - r0) * pow((1 - cos_theta), 5);

    unsigned int seed = prd->seed;
    const float rand_float = rnd(seed);

    if (cannot_refract || reflectance > rand_float)
        direction = reflect(unit_dir, hitData.N);
    else
        direction = refracted;

    prd->origin = hitData.P;
    prd->direction = refracted;
}

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w = params.width;
    const int    h = params.height;
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);

    float3 result = make_float3(0.0f);
    int i = params.samples_per_launch;

    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)
        ) - 1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        RadiancePRD prd;
        prd.color = make_float3(0.f);
        prd.radiance = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;

        int depth = 0;
        for (;;) {
            traceRadiance(
                params.handle,
                ray_origin,
                ray_direction,
                0.001f,  // tmin       // TODO: smarter offset
                1e16f,  // tmax
                &prd);

            if (prd.done || depth >= 50) // TODO RR, variable for depth
                break;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }

        result += prd.attenuation;

    } while (--i);

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;
    float3         accum_color = result / static_cast<float>(params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float                 a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    params.frame_buffer[image_index] = make_color(accum_color);
}

extern "C" __global__ void __miss__radiance()
{
    RadiancePRD* prd = getPRD();
    prd->attenuation *= make_float3(0.5, 0.7, 1.0);
    prd->done = true;
}

extern "C" __global__ void __closesthit__radiance()
{
    const HitGroupData& sbtData
        = *(const HitGroupData*)optixGetSbtDataPointer();

    const int primID = optixGetPrimitiveIndex();

    const float3 center = sbtData.vertex[primID];
    const float radius = sbtData.radius[primID];

    float  t_hit = optixGetRayTmax();

    // Backface hit not used.
    //float  t_hit2 = __uint_as_float( optixGetAttribute_0() ); 

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = normalize(optixGetWorldRayDirection());

    float3 P = ray_orig + ray_dir * t_hit;
    float3 normal = normalize(P - center) * (radius / fabs(radius));

    const float3 N = faceforward(normal, -ray_dir, normal);

    Ray ray_in;
    ray_in.origin = ray_orig;
    ray_in.direction = ray_dir;

    HitData hitData;
    hitData.center = center;
    hitData.radius = radius;
    hitData.N = N;
    hitData.P = P;
    hitData.normal = normal;

    RadiancePRD* prd = getPRD();

    if (prd->countEmitted)
        prd->emitted = make_float3(0.0f);
    else
        prd->emitted = make_float3(0.0f);
    {
        float3 attenuation = sbtData.diffuse_color;
        if (sbtData.material == SPECULAR) {
            bool test = false;
            specular_scatter(ray_in, hitData, prd, attenuation, 0.5f);
        }
        else if(sbtData.material == DIFFUSE) {
            diffuse_scatter(ray_in, hitData, prd, attenuation);
        }
        else if (sbtData.material == DIELECTRIC) {
            dielectric_scatter(ray_in, hitData, prd, attenuation, 1.5f);
        }
        prd->attenuation *= attenuation;
    }
}