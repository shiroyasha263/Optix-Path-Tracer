OPTIX_add_sample_executable( optixHello target_name
  ${SAMPLES_CUDA_DIR}/helpers.h
  draw_solid_color.cu
  optixHello.cpp
  optixHello.h

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)