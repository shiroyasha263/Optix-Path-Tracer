#pragma once

struct Params {
	unsigned int subframe_index;
	
	float4* accum_buffer;
	uint32_t* frame_buffer;

	unsigned int img_width;
	unsigned int img_height;

	unsigned int samples_per_launch;
};