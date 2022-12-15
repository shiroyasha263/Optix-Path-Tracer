OPTIX_add_sample_executable( optixHello target_name
  optixHello.cpp

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)