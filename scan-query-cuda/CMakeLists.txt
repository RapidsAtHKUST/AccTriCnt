cmake_minimum_required(VERSION 3.6)
project(scan-query)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH}) # To allow CMake to locate our Find*.cmake files
add_definitions(-DUSE_LOG)

include(SetSIMDFlages)
include(SetLibraries)
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
add_definitions("-DGIT_SHA1=\"${GIT_SHA1}\"")

include_directories(../dependencies/sparsepp)

if (USER_CC_CAP)
    message("User CC (CUDA): ${USER_CC_CAP}")
    set(COMPUTE_CAP ${USER_CC_CAP})
    set(EXTRA_COMPATIBILITY_DEFINES "-DNO_CO_PROCESS")
else()
    set(COMPUTE_CAP "61")
endif ()

#### CUDA Related Files
find_package("CUDA")
option(CUDA_ALL "Build all CUDA exec." OFF)
option(LEGACY "Build CUDA LEGACY exec." OFF)
#set(CUDA_SEPARABLE_COMPILATION ON)
if (CUDA_FOUND)
    message("--- CUDA Found, Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}/")
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include) # locate libraries installed with toolkit, e.g, "thrust"

    # modify options globally
    macro(cuda_add_compile_options options)
        list(APPEND CUDA_NVCC_FLAGS "${options}")
    endmacro()
    cuda_add_compile_options("-gencode arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}")
    cuda_add_compile_options("-std=c++11 -O3 -g -lineinfo")
    cuda_add_compile_options("-DUNIFIED_MEM -DUSE_LOG -DCUDA -DBIT_VEC")
    cuda_add_compile_options("-Xcompiler ${OpenMP_CXX_FLAGS} -Xcompiler -march=native")

    # Attention: do not know why not able to pass defintions to Xcompiler
    function(build_our_cuda_exec suffix extra_options)
        ## 1) Default Merge 4-8
        if (CUDA_MERGE_RELEASE)
            cuda_add_executable(scan-xp-cuda${suffix} ${CUDA_SOURCE_FILES})
            message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
        endif ()
        ## Merge Impl (different configurations)
        macro(build_scanxp_shared_differ_cmp tile_src tile_dst)
            cuda_add_executable(scan-xp-cuda-${tile_src}-${tile_dst}${suffix} ${CUDA_SOURCE_FILES} OPTIONS "-DTILE_SRC=${tile_src} -DTILE_DST=${tile_dst} ${extra_options}")
            message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
        endmacro()
        if (CUDA_ALL)
            build_scanxp_shared_differ_cmp(4 8)
            build_scanxp_shared_differ_cmp(2 16)
            build_scanxp_shared_differ_cmp(1 32)
        endif ()

        ## 2) Bitmap Impl
        cuda_add_executable(scan-xp-cuda-bitmap-warp-per-vertex${suffix} ${CUDA_SOURCE_FILES} OPTIONS "-DUSE_BITMAP_KERNEL  ${extra_options}")
        cuda_add_executable(scan-xp-cuda-bitmap-1D${suffix} ${CUDA_SOURCE_FILES} OPTIONS "-DUSE_BITMAP_KERNEL -DBITMAP_1D ${extra_options}")
        message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

        ## 3) BaseLines: Edge per Thread (Simpliest)
        if (CUDA_ALL)
            cuda_add_executable(scan-xp-cuda-baseline-hybrid${suffix} ${CUDA_SOURCE_FILES} OPTIONS "-DBASELINE_HYBRID ${extra_options}")
            message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
        endif ()

        ## 4) Hybrid Galloping (Thread-Level-Baseline & SIMD-Warp-Level Merge)
        if (suffix MATCHES ".*-multi-gpu-multi-pass")
            message("to compile scan-xp-cuda-hybrid-kernels${suffix}")
            cuda_add_executable(scan-xp-cuda-hybrid-kernels${suffix} ${CUDA_SOURCE_FILES} OPTIONS "-DUSE_HYBRID_KERNELS ${extra_options}")
        endif ()

    endfunction()

    set(UTIL_FILES util/graph.cu set-inter/csr_set_intersection_serial.cpp util/log.cpp util/stat.cpp)

    if (LEGACY)
        # Deprecated: since not general engough to support super large datasets
        # 1st: single-gpu
        #   cuda_add_compile_options("-DVERIFY_WITH_CPU")
        set(CUDA_SOURCE_FILES experimental/scan_xp.cu main_scanxp.cpp ${UTIL_FILES})
        build_our_cuda_exec("" "-DVERIFY_WITH_CPU")

        # 2nd: multi-gpu
        set(CUDA_SOURCE_FILES experimental/scan_xp_multigpu.cu main_scanxp.cpp ${UTIL_FILES})
        build_our_cuda_exec("-multi-gpu" "-DMULTI_GPU -DVERIFY_WITH_CPU")

        # 4th: single-gpu-multi-pass
        # Deprecated: single-gpu-multi-pass is a special case of multi-GPU without task-assignment
        set(CUDA_SOURCE_FILES experimental/scan_xp_multipass.cu main_scanxp.cpp ${UTIL_FILES})
        build_our_cuda_exec("-single-gpu-multi-pass" "-DVERIFY_WITH_CPU")
    endif ()

    # 3rd: multi-gpu-multi-pass
    if (LEGACY)
        set(CUDA_SOURCE_FILES experimental/scan_xp_multigpu_multipass.cu main_scanxp.cpp ${UTIL_FILES})
        #    build_our_cuda_exec("-multi-gpu-multi-pass" "-DMULTI_GPU -DVERIFY_WITH_CPU")
        build_our_cuda_exec("-multi-gpu-multi-pass" "-DMULTI_GPU -DVERIFY_WITH_CPU")

        # 5th: dynamic load balance
        set(CUDA_SOURCE_FILES experimental/scan_xp_multigpu_multipass_dynamic.cu main_scanxp.cpp ${UTIL_FILES})
        #    build_our_cuda_exec("-multi-gpu-multi-pass-dynamic-lb" "-DMULTI_GPU -DVERIFY_WITH_CPU")
        build_our_cuda_exec("-multi-gpu-multi-pass-dynamic-lb" "-DMULTI_GPU")
    endif ()

    # experimental: varying num-of-pass for snap_friendster
    set(CUDA_SOURCE_FILES experimental_scan_xp_multigpu_multipass.cu experimental/main_scanxp.cpp ${UTIL_FILES})
    cuda_add_executable(scan-xp-cuda-varying-parameters-bitmap ${CUDA_SOURCE_FILES} OPTIONS "-DUSE_BITMAP_KERNEL -DMULTI_GPU -DVARY_BLOCK_SIZE ${EXTRA_COMPATIBILITY_DEFINES}")
    cuda_add_executable(scan-xp-cuda-varying-parameters-bitmap1D ${CUDA_SOURCE_FILES} OPTIONS "-DUSE_BITMAP_KERNEL -DMULTI_GPU -DBITMAP_1D -DVARY_BLOCK_SIZE ${EXTRA_COMPATIBILITY_DEFINES}")
    cuda_add_executable(scan-xp-cuda-varying-parameters-hybrid-kernels ${CUDA_SOURCE_FILES} OPTIONS "-DUSE_HYBRID_KERNELS -DMULTI_GPU -DVARY_BLOCK_SIZE ${EXTRA_COMPATIBILITY_DEFINES}")
endif ()