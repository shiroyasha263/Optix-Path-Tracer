
project(OptiX-Samples)

cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

cmake_policy(VERSION 2.8.12)

if( POLICY CMP0072 )
  # FindOpenGL prefers GLVND by default when available
  cmake_policy(SET CMP0072 NEW)
endif()

if( POLICY CMP0074 )
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/../Source/CMake"
  ${CMAKE_MODULE_PATH}
  )

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

# CUDA 8 is broken for generating dependencies during configure
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)

# CUDA 11.7 adds support for OptiX targetted input files.
if( CUDA_VERSION VERSION_GREATER_EQUAL 11.7 )
  option(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON)
else()
  # You can try to force this on
  option(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF)
endif()

# This code looks funny, but CMake doesn't have an equality operator for boolean types
# (only integer and string). By doing NOT NOT VAL, you can force the original value into 0
# or 1 and allow the EQUAL operator to function correctly.
if( NOT ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT ) ) EQUAL ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ) ) )
  message( STATUS "Unsetting values associated with OptiX code generation" )
  # This allows us to reset dependent options if you change it.
  unset( SAMPLES_INPUT_GENERATE_OPTIXIR CACHE )
  unset( SAMPLES_INPUT_GENERATE_PTX CACHE )
  unset( GENERATE_DEBUG_DEVICE_CODE CACHE )
endif()
set(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ${SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT} CACHE INTERNAL "Previous configured value" FORCE)

if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF )
else()
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON )
endif()

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

function(optix_add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config} ${CUDA_NVCC_FLAGS${config}} CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endfunction()

function(optix_add_cuda_flag flag)
  optix_add_cuda_flag_config( "" ${flag} )
endfunction()

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if( NOT PASSED_FIRST_CONFIGURE )
  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  optix_add_cuda_flag("--use_fast_math")
  optix_add_cuda_flag("-lineinfo")
  foreach( config ${CMAKE_CONFIGURATION_TYPES} )
    if( ${config} STREQUAL "Debug" )
      optix_add_cuda_flag_config( _${config} "-G" )
      optix_add_cuda_flag_config( _${config} "-O0" )
    endif()
  endforeach()


  if( CUDA_VERSION VERSION_LESS "3.0" )
    optix_add_cuda_flag("--keep")
  endif()

  # Some CUDA 11.x toolkits erroneously complain about sm_50 being deprecated
  if(CUDA_VERSION VERSION_GREATER "11.0")
    optix_add_cuda_flag("-Wno-deprecated-gpu-targets")
  endif()

  if(CMAKE_CXX_STANDARD EQUAL 11)
    set(SAMPLES_NVRTC_CXX "-std=c++11")
  else()
    set(SAMPLES_NVRTC_CXX "")
  endif()
  set(CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -use_fast_math -lineinfo -default-device -rdc true -D__x86_64 CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
endif()

mark_as_advanced(CUDA_NVRTC_FLAGS)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../" CACHE PATH "Path to OptiX installed location.")

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
  "${OptiX_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  )

# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")
if( CUDA_NVRTC_ENABLED AND SAMPLES_INPUT_GENERATE_OPTIXIR )
  if( NOT SAMPLES_INPUT_GENERATE_PTX )
    message( SEND_ERROR "CUDA_NVRTC_ENABLED is only compatible with SAMPLES_INPUT_GENERATE_PTX. Please disable CUDA_NVRTC_ENABLED unless enabling SAMPLES_INPUT_GENERATE_PTX." )
  endif()
endif()


##################################################################
# SUtil compilation

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

if( WIN32 )
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
else( WIN32 )
  if( USING_GNU_C AND NOT APPLE )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
  endif()
endif( WIN32 )

set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")

set(SAMPLES_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Source/support")

# NVRTC include paths relative to the sample path
set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \"lib/DemandLoading/include\", \\
  \"lib/ImageSource/include\", \\
  \".\", ")

# NVRTC absolute include paths to the headers used to build the samples
set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

# Build a null-terminated option list for NVRTC
set(CUDA_NVRTC_OPTIONS)
foreach(flag ${CUDA_NVRTC_FLAGS})
  set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS} \\\n  \"${flag}\",")
endforeach()
set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS}")

configure_file(sampleConfig.h.in sampleConfig.h @ONLY)

# Path to sutil.h that all the samples need
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     "${CMAKE_BINARY_DIR}/include"
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS}
                     )

set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

# Helper macro to generate PTX from the CUDA files in sutil.
macro(OPTIX_sutil_compile_to_optix_input generated_files)
  if( SAMPLES_INPUT_GENERATE_OPTIXIR )
    message("sutil OPTIXIR")
    CUDA_WRAP_SRCS( sutil OPTIXIR generated_files2 ${ARGN} )
    list(APPEND ${generated_files} ${generated_files2})
  endif()
  if( SAMPLES_INPUT_GENERATE_PTX AND NOT CUDA_NVRTC_ENABLED)
    message("sutil PTX")
    CUDA_WRAP_SRCS( sutil PTX generated_files3 ${ARGN} )
    list(APPEND ${generated_files} ${generated_files3})
  endif()
  message("${generated_files} = ${${generated_files}}")
endmacro()

# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
macro(OPTIX_add_source_groups)
  if( SAMPLES_INPUT_GENERATE_PTX )
    if( NOT CUDA_NVRTC_ENABLED )
      source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
    endif()
  endif()
  if( SAMPLES_INPUT_GENERATE_OPTIXIR )
    source_group("OptixIR Files"  REGULAR_EXPRESSION ".+\\.optixir$")
  endif()
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
endmacro()

#########################################################
# OPTIX_add_sample_executable
#
# Convience function for adding samples to the code.  You can copy the contents of this
# funtion into your individual project if you wish to customize the behavior.  Note that
# in CMake, functions have their own scope, whereas macros use the scope of the caller.
function(OPTIX_add_sample_executable target_name_base target_name_var)

  set( target_name ${target_name_base} )
  set( ${target_name_var} ${target_name} PARENT_SCOPE )

  OPTIX_add_source_groups()

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
  set(cu_obj_source_files)
  set(cu_optix_source_files)
  foreach(file ${source_files})
    get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
    if(${_cuda_source_format} MATCHES "OBJ")
      list(APPEND cu_obj_source_files ${file})
    else()
      list(APPEND cu_optix_source_files ${file})
    endif()
  endforeach()

  # Create the rules to build the OBJ from the CUDA files.
  CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )

  # Create the rules to build the PTX and/or OPTIX files.
  if( SAMPLES_INPUT_GENERATE_OPTIXIR )
    CUDA_WRAP_SRCS( ${target_name} OPTIXIR generated_files2 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
    list(APPEND generated_files ${generated_files2})
  endif()
  if( SAMPLES_INPUT_GENERATE_PTX AND NOT CUDA_NVRTC_ENABLED)
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files3 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
    list(APPEND generated_files ${generated_files3})
  endif()

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  add_executable(${target_name}
    ${source_files}
    ${generated_files}
    ${cmake_options}
    )

  # Most of the samples link against the sutil library and the optix library.  Here is the
  # rule that specifies this linkage.
  target_link_libraries( ${target_name}
    ${GLFW_LIB_NAME}
    imgui
    sutil_7_sdk
    )

  set_target_properties( ${target_name} PROPERTIES
    COMPILE_DEFINITIONS
    "OPTIX_SAMPLE_NAME_DEFINE=${target_name};OPTIX_SAMPLE_DIR_DEFINE=${target_name}" )

  if( UNIX AND NOT APPLE )
    # Force using RPATH instead of RUNPATH on Debian
    target_link_libraries( ${target_name} "-Wl,--disable-new-dtags" )
  endif()

  if(USING_GNU_CXX)
    target_link_libraries( ${target_name} m ) # Explicitly link against math library (C samples don't do that by default)
  endif()
endfunction()

#########################################################
#  List of samples found in subdirectories.
#
# If you wish to start your own sample, you can copy one of the sample's directories.
# Just make sure you rename all the occurances of the sample's name in the C code as well
# and the CMakeLists.txt file.

add_subdirectory(OptixHello)
add_subdirectory(OptixPipeline)
add_subdirectory(OptixWindow)
add_subdirectory(OptixSphere)
add_subdirectory(OptixTRY)

#add_subdirectory( optixBoundValues       )
#add_subdirectory( optixCallablePrograms  )
#add_subdirectory( optixCompileWithTasks  )
#add_subdirectory( optixCurves            )
#add_subdirectory( optixCustomPrimitive   )
#add_subdirectory( optixCutouts           )
#add_subdirectory( optixDemandLoadSimple  )
#add_subdirectory( optixDemandTexture     )
#add_subdirectory( optixDenoiser          )
#add_subdirectory( optixDynamicGeometry   )
#add_subdirectory( optixDynamicMaterials  )
#add_subdirectory( optixHair              )
#add_subdirectory( optixHello             )
#add_subdirectory( optixMeshViewer        )
#add_subdirectory( optixModuleCreateAbort )
#add_subdirectory( optixMotionGeometry    )
#add_subdirectory( optixMultiGPU          )
#add_subdirectory( optixNVLink            )
#add_subdirectory( optixOpticalFlow       )
#add_subdirectory( optixPathTracer        )
#add_subdirectory( optixRaycasting        )
#add_subdirectory( optixSimpleMotionBlur  )
#add_subdirectory( optixSphere            )
#add_subdirectory( optixTriangle          )
#add_subdirectory( optixVolumeViewer      )
#add_subdirectory( optixWhitted           )

# Our sutil library.  The rules to build it are found in the subdirectory.
add_subdirectory(sutil)
# Our other libraries.
add_subdirectory( lib/DemandLoading )
add_subdirectory( lib/ImageSource )
# Third-party support libraries.
add_subdirectory(support)

#################################################################

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")
