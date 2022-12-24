// Creating this To tackle the problem that is creating buildInputs, 
// with this I will be able to handle multiple build inputs with a class

#pragma once
#include <memory.h>
#include "Params.h"

using std::shared_ptr;
using std::make_shared;

class buildInput {
public:
	virtual void build(OptixBuildInput& build_input, uint32_t& build_input_flags) const = 0;
	virtual SphereicalMesh get_sphere() const{
		return spherical_mesh;
	}
public:
	SphereicalMesh spherical_mesh;
};

class sphereBuild : public buildInput {
public:
	sphereBuild(float3 center, float radius, MaterialType material, float3 diffuse_color) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex), sizeof(float3)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertex), &center,
			sizeof(float3), cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius), sizeof(float)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius), &radius,
			sizeof(float), cudaMemcpyHostToDevice));

		spherical_mesh.center = center;
		spherical_mesh.radius = radius;
		spherical_mesh.diffuse_color = diffuse_color;
		spherical_mesh.material = material;
	};

	virtual void build(OptixBuildInput& build_input, uint32_t& build_input_flags) const override;

public:
	CUdeviceptr d_vertex;
	CUdeviceptr d_radius;
};

class triangleBuild : public buildInput {

public:
	CUdeviceptr d_vertices;
	CUdeviceptr d_indices;
};

class buildInputList {
public:
	buildInputList() {}
	buildInputList(shared_ptr<buildInput> object) { add(object); }

	void clear() { objects.clear(); }
	void add(shared_ptr<buildInput> object) { objects.push_back(object); }

public:
	std::vector<shared_ptr<buildInput>> objects;
};