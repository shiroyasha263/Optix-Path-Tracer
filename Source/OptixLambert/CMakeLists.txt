OPTIX_add_sample_executable( OptixTRY target_name
  main.cpp
  params.h
  RenderState.h
  RenderState.cpp
  deviceProgram.cu

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
${optix_LIBRARY}
${CUDA_LIBRARIES}
${CUDA_CUDA_LIBRARY}
)