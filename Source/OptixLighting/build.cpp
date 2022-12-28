//#include "build.h"
//
//void sphereBuild::build(OptixBuildInput& build_input, uint32_t& build_input_flags) const {
//	build_input = {};
//	build_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
//
//	build_input.sphereArray.vertexBuffers = &d_vertex;
//	build_input.sphereArray.numVertices = 1;
//	build_input.sphereArray.radiusBuffers = &d_radius;
//
//	build_input_flags = 0;
//
//	build_input.sphereArray.flags = &build_input_flags;
//	build_input.sphereArray.numSbtRecords = 1;
//	build_input.sphereArray.sbtIndexOffsetBuffer = 0;
//	build_input.sphereArray.sbtIndexOffsetSizeInBytes = 0;
//	build_input.sphereArray.sbtIndexOffsetStrideInBytes = 0;
//}