#pragma once
#include <vector>

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};

enum MeshType {
    SPHERICAL,
    TRIANGULAR
};

enum MaterialType {
    DIFFUSE,
    SPECULAR,
    DIELECTRIC
};

enum PrimitiveTyoe {
    TRIANGLE,
    SPHERE
};

struct Material {
    float3 emission;
    float3 diffuse_color;
    float fuzz;
    float eta;
};

struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};


struct Params
{
    unsigned int subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    ParallelogramLight     light; // TODO: make light list
    OptixTraversableHandle handle;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    float3  emission_color;
    float3  diffuse_color;
    float3 vertex;
    float radius;
    float3* vertices;
    int3* indices;
    MaterialType materialType;
    Material material;
    MeshType meshType;
};

struct SphereicalMesh {
    float3 center;
    float radius;
    float3 diffuse_color;
    MaterialType materialType;
    Material material;
};

struct TriangularMesh {
    std::vector<float3> vertices;
    std::vector<int3>   indices;
    MaterialType materialType;
    Material material;
};