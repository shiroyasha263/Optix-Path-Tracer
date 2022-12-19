#pragma once

struct Params {
	unsigned int subframe_index;

	float4* accum_buffer;
	uchar4* frame_buffer;

	float3 eye;
	float3 U, V, W;

	unsigned int img_width;
	unsigned int img_height;

	unsigned int samples_per_launch;
};