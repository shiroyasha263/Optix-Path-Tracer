#pragma once
#include <sutil/vec_math.h>

struct Params {

public:

	unsigned int subframe_index;

	float4* accum_buffer;
	uchar4* frame_buffer;

	float3 eye = make_float3(0.f, 0.f, 3.0f);
	float3 U, V, W;

	unsigned int img_width;
	unsigned int img_height;

	unsigned int samples_per_launch;
	OptixTraversableHandle handle = 0;
};

enum RayType
{
	RAY_TYPE_RADIANCE = 0,
	RAY_TYPE_COUNT
};