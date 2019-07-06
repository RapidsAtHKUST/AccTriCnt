cmake_minimum_required(VERSION 3.6)
project(scan-query)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH}) # To allow CMake to locate our Find*.cmake files
add_definitions(-DUSE_LOG)

option(BUILD_TRI_CNT "build-tri-cnt" ON)
option(ENABLE_HBW "Build With HBW for KNL." ON)
option(UTILS "enable utils: converter " ON)
option(BUILD_HASH "build-hash" OFF)
option(BUILD_LEGACY "build-legacy" OFF)
option(KNL "enable knl compilation" OFF)

include(SetSIMDFlages)
include(SetLibraries)
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
add_definitions("-DGIT_SHA1=\"${GIT_SHA1}\"")

# compilation-time info
if (SHOW_OPT_INFO)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
        message(icpc)
        add_compile_options("-qopt-report=5")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options("-fopt-info-vec-all")
        message(g++)
    endif ()
endif ()

set(UTIL_FILES util/graph.cpp
        set-inter/csr_set_intersection_serial.cpp
        set-inter/csr_set_intersection_simd.cpp
        set-inter/csr_set_intersection_hybrid.cpp
        util/log.cpp util/stat.cpp)
set(SOURCE_FILES scan_xp.cpp ${UTIL_FILES} main_scanxp.cpp tools/pkc.cpp)
set(BASELINE_SOURCE_FILES scan_xp_baseline.cpp ${UTIL_FILES} main_scanxp.cpp)

include_directories(../dependencies/sparsepp)
include_directories(../dependencies/CRoaring)
include_directories(../dependencies/libpopcnt)
include_directories(set-inter/emptyheaded/include)

function(build_our_exec suffix)
    message("${suffix}: ${SOURCE_FILES}")

    if (KNL)
        # 1st: merge vs pivot
        add_executable(scan-xp-avx512-merge${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-avx512-merge${suffix} PRIVATE -march=native -O3 -g)
        target_compile_definitions(scan-xp-avx512-merge${suffix} PRIVATE AVX512=1 HAS_AVX512=1)

        if (BUILD_LEGACY)
            add_executable(scan-xp-avx512-merge-popcnt${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx512-merge-popcnt${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx512-merge-popcnt${suffix} PRIVATE AVX512_POPCNT=1 HAS_AVX512=1)

            add_executable(scan-xp-avx512-pivot${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx512-pivot${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx512-pivot${suffix} PRIVATE PIVOT_KNL=1 HAS_AVX512=1)

            # 2nd: single-dir galloping vs double-dir
            add_executable(scan-xp-avx512-galloping-single${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx512-galloping-single${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx512-galloping-single${suffix} PRIVATE GALLOPING_AVX512=1 HAS_AVX512=1)

            add_executable(scan-xp-avx512-galloping-double${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx512-galloping-double${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx512-galloping-double${suffix} PRIVATE GALLOPING_DOUBLE_AVX512=1 HAS_AVX512=1)

            # experimental no-blk-merge no-galloping
            add_executable(scan-xp-avx512-hybrid-no-blk-merge${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx512-hybrid-no-blk-merge${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx512-hybrid-no-blk-merge${suffix} PRIVATE HYBRID_NO_MERGE=1 HAS_AVX512=1)

            add_executable(scan-xp-avx512-hybrid-no-galloping${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx512-hybrid-no-galloping${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx512-hybrid-no-galloping${suffix} PRIVATE HYBRID_NO_GALLOPING=1 HAS_AVX512=1)

            # 4th: HUGE_PAGE
            add_executable(scan-xp-avx512-hybrid-hugepage${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx512-hybrid-hugepage${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx512-hybrid-hugepage${suffix} PRIVATE ADVANCED=1 HAS_AVX512=1 -DHUGE_PAGE=1)
        endif ()
        # 3rd: hybrid galloping-single-dir and merge
        add_executable(scan-xp-avx512-hybrid${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-avx512-hybrid${suffix} PRIVATE -march=native -O3 -g)
        target_compile_definitions(scan-xp-avx512-hybrid${suffix} PRIVATE ADVANCED=1 HAS_AVX512=1)
    endif ()

    if (KNL OR AVX2)
        # 1st: merge vs pivot
        add_executable(scan-xp-avx2-merge${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-avx2-merge${suffix} PRIVATE -march=native -O3 -g)
        target_compile_definitions(scan-xp-avx2-merge${suffix} PRIVATE AVX2=1 HAS_AVX2=1)

        if (BUILD_LEGACY)
            add_executable(scan-xp-avx2-merge-popcnt${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx2-merge-popcnt${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx2-merge-popcnt${suffix} PRIVATE AVX2_POPCNT=1 HAS_AVX2=1)

            add_executable(scan-xp-avx2-pivot${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx2-pivot${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx2-pivot${suffix} PRIVATE PIVOT_AVX2=1 HAS_AVX2=1)
            # 2nd: single-dir galloping vs double-dir
            add_executable(scan-xp-avx2-galloping-single${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx2-galloping-single${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx2-galloping-single${suffix} PRIVATE AVX2_GALLOPING_SINGLE=1 HAS_AVX2=1)

            add_executable(scan-xp-avx2-galloping-double${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-avx2-galloping-double${suffix} PRIVATE -march=native -O3 -g)
            target_compile_definitions(scan-xp-avx2-galloping-double${suffix} PRIVATE AVX2_GALLOPING_DOUBLE=1 HAS_AVX2=1)
        endif ()

        # 3rd: hybrid galloping-single-dir and merge
        add_executable(scan-xp-avx2-hybrid${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-avx2-hybrid${suffix} PRIVATE -march=native -O3 -g)
        target_compile_definitions(scan-xp-avx2-hybrid${suffix} PRIVATE CPU_ADVANCED=1 HAS_AVX2=1)
    endif ()

    #### TetZank: AVX/AVX2/ASM
    function(build_tetzank name macro)
        add_executable(scan-xp-tetzank-${name}${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-tetzank-${name}${suffix} PRIVATE -O3 -g -march=native)
        target_compile_definitions(scan-xp-tetzank-${name}${suffix} PRIVATE ${macro} TETZANK=1)

    endfunction()
    if (AVX OR AVX2 OR KNL)
        message(AVX...Build TETZANK)
        build_tetzank(avx TETZANK_AVX=1)
    endif ()
    if (AVX2 OR KNL)
        message(AVX2...Build TETZANK)
        build_tetzank(avx2 TETZANK_AVX2=1)
        build_tetzank(avx2-asm TETZANK_AVX2_ASM=1)
    endif ()

    #### Han: BSR...BMiss/Qfilter/Hier
    function(build_compact_forward name macro)
        add_executable(scan-xp-compact-forward-${name} ${SOURCE_FILES} set-inter/lemire/intersection.cpp
                set-inter/lemire/benchintersection.cpp set-inter/han/intersection_algos.cpp
                set-inter/han/utils/util.cpp)
        target_compile_options(scan-xp-compact-forward-${name} PRIVATE -O3 -g -march=native)
        target_compile_definitions(scan-xp-compact-forward-${name} PRIVATE COMPACT_FORWARD=1 ${macro})
    endfunction()
    if (AVX2 OR KNL)
        build_compact_forward("avx2-lemire" "LEMIRE_AVX2=1")
        build_compact_forward("avx2-asm-tetzank" "TETZANK_AVX2_ASM=1;TETZANK=1")
        build_compact_forward("avx2-tetzank" "TETZANK_AVX2=1;TETZANK=1")
    endif ()
    if (SSE4)
        add_executable(scan-xp-cf-hybrid ${SOURCE_FILES})
        target_compile_options(scan-xp-cf-hybrid PRIVATE -O3 -g -march=native)
        target_compile_definitions(scan-xp-cf-hybrid PRIVATE HYBRID_CF_NORMAL=1)

        build_compact_forward("deg" DEG_DESCENDING_REORDERING=1)
        build_compact_forward("kcore" K_CORE_REORDERING=1)
        build_compact_forward("reorder" "")
        build_compact_forward(qfilter HAN_QFILTER=1)
        build_compact_forward(bmiss HAN_BMISS=1)
        build_compact_forward(bmiss-sttni HAN_BMISS_STTNI=1)
        build_compact_forward(bmp BITMAP=1)
        build_compact_forward(bmp-naive-pack "BITMAP=1;PACK=1")

        function(build_han name macro)
            add_executable(scan-xp-sse-han-${name}${suffix} ${SOURCE_FILES}
                    set-inter/han/intersection_algos.cpp
                    set-inter/han/utils/util.cpp)
            target_compile_options(scan-xp-sse-han-${name}${suffix} PRIVATE -O3 -g -march=native)
            target_compile_definitions(scan-xp-sse-han-${name}${suffix} PRIVATE ${macro})
        endfunction()

        build_han(qfilter HAN_QFILTER=1)
        build_han(bmiss HAN_BMISS=1)
        build_han(bmiss-sttni HAN_BMISS_STTNI=1)
        build_han(bsr-qfilter BSR=1)

        build_han(bsr-scalar-merge "BSR=1;SCALAR_MERGE_BSR=1")
        build_han(bsr-scalar-galloping "BSR=1;SCALAR_GALLOPING_BSR=1")
        build_han(bsr-sse-galloping "BSR=1;SIMD_GALLOPING_BSR=1")
        build_han(bsr-sse-shuffle "BSR=1;SHUFFLE_BSR=1")

        function(build_emptyheaded name macro other)
            add_executable(scan-xp-sse-emptyheaded-${name}${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-sse-emptyheaded-${name}${suffix} PRIVATE -O3 -g -march=native)
            target_compile_definitions(scan-xp-sse-emptyheaded-${name}${suffix} PRIVATE ${macro} ${other} NUM_THREADS=1)
        endfunction()
        build_emptyheaded(layout-hybrid EMPTY_HEADED=1 DUMMY=1)
        build_emptyheaded(layout-uint EMPTY_HEADED=1 EH_LAYOUT_UINT=1)
    endif ()

    #### Lemire: SSE4/Roaring
    if (SSE4)
        add_executable(scan-xp-sse-lemire${suffix} ${SOURCE_FILES} set-inter/lemire/intersection.cpp
                set-inter/lemire/benchintersection.cpp)
        target_compile_options(scan-xp-sse-lemire${suffix} PRIVATE -O3 -g -march=native)
        target_compile_definitions(scan-xp-sse-lemire${suffix} PRIVATE LEMIRE_SSE=1)

        add_executable(scan-xp-sse-roaring${suffix} ${SOURCE_FILES} ../dependencies/CRoaring/roaring.c)
        target_compile_options(scan-xp-sse-roaring${suffix} PRIVATE -O3 -g -march=native)
        target_compile_definitions(scan-xp-sse-roaring${suffix} PRIVATE ROARING=1 USESSE4=1)

        add_executable(scan-xp-sse-bm-lemire${suffix} ${SOURCE_FILES} set-inter/lemire/intersection.cpp
                set-inter/lemire/benchintersection.cpp)
        target_compile_options(scan-xp-sse-bm-lemire${suffix} PRIVATE -O3 -g -march=native)
        target_compile_definitions(scan-xp-sse-bm-lemire${suffix} PRIVATE LEMIRE_HIGHLY_SCALABLE=1)
    endif ()

    if (AVX2 OR KNL)
        # lemire
        add_executable(scan-xp-avx2-lemire${suffix} ${SOURCE_FILES} set-inter/lemire/intersection.cpp)
        target_compile_options(scan-xp-avx2-lemire${suffix} PRIVATE -O3 -g -march=native)
        target_compile_definitions(scan-xp-avx2-lemire${suffix} PRIVATE LEMIRE_AVX2=1)

        add_executable(scan-xp-avx2-roaring${suffix} ${SOURCE_FILES} ../dependencies/CRoaring/roaring.c)
        target_compile_options(scan-xp-avx2-roaring${suffix} PRIVATE -O3 -g -march=native)
        target_compile_definitions(scan-xp-avx2-roaring${suffix} PRIVATE ROARING=1 USEAVX=1)
    endif ()

    if (SSE4)
        # 1st: merge vs pivot
        add_executable(scan-xp-sse-merge${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-sse-merge${suffix} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-sse-merge${suffix} PRIVATE SSE_MERGE=1)

        if (BUILD_LEGACY)
            add_executable(scan-xp-sse-pivot${suffix} ${SOURCE_FILES})
            target_compile_options(scan-xp-sse-pivot${suffix} PRIVATE -O3 -g)
            target_compile_definitions(scan-xp-sse-pivot${suffix} PRIVATE SSE_PIVOT=1)
        endif ()

        # 2nd: hybrid hybrid galloping-single-dir and merge
        add_executable(scan-xp-sse-hybrid${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-sse-hybrid${suffix} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-sse-hybrid${suffix} PRIVATE SSE_HYBRID=1)

        add_executable(scan-xp-sse-bins${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-sse-bins${suffix} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-sse-bins${suffix} PRIVATE SSE_BIN=1)
    endif ()

    ####### without simd instructions
    # 1st: merge vs pivot
    if (BUILD_LEGACY)
        add_executable(scan-xp-naive-pivot${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-naive-pivot${suffix} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-naive-pivot${suffix} PRIVATE NAIVE=1)

        # 2nd: single-dir galloping vs
        add_executable(scan-xp-naive-galloping-single${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-naive-galloping-single${suffix} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-naive-galloping-single${suffix} PRIVATE GALLOPING_SINGLE=1)

        add_executable(scan-xp-naive-galloping-double${suffix} ${SOURCE_FILES})
        target_compile_options(scan-xp-naive-galloping-double${suffix} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-naive-galloping-double${suffix} PRIVATE GALLOPING_DOUBLE=1)
    endif ()

    add_executable(scan-xp-naive-merge${suffix} ${SOURCE_FILES})
    target_compile_options(scan-xp-naive-merge${suffix} PRIVATE -O3 -g)
    target_compile_definitions(scan-xp-naive-merge${suffix} PRIVATE NAIVE_STD_MERGE=1)

    # 3rd: hybrid galloping-single-dir and merge
    add_executable(scan-xp-naive-hybrid${suffix} ${SOURCE_FILES})
    target_compile_options(scan-xp-naive-hybrid${suffix} PRIVATE -O3 -g)
    target_compile_definitions(scan-xp-naive-hybrid${suffix} PRIVATE NAIVE_HYBRID=1)

    # hash related:
    function(build_hash_bitmap type macro)
        if (BUILD_HASH)
            ## unordered_map
            add_executable(scan-xp-naive-hash${suffix}${type} ${SOURCE_FILES})
            target_compile_options(scan-xp-naive-hash${suffix}${type} PRIVATE -O3 -g)
            target_compile_definitions(scan-xp-naive-hash${suffix}${type} PRIVATE HASH=1 ${macro})

            ## spp
            add_executable(scan-xp-naive-hash-spp${suffix}${type} ${SOURCE_FILES})
            target_compile_options(scan-xp-naive-hash-spp${suffix}${type} PRIVATE -O3 -g)
            target_compile_definitions(scan-xp-naive-hash-spp${suffix}${type} PRIVATE HASH_SPP=1 ${macro})
        endif ()

        ## bitmap, cost: num of threads * (number of vertices / 8) bytes
        add_executable(scan-xp-naive-bitvec${suffix}${type} ${SOURCE_FILES})
        target_compile_options(scan-xp-naive-bitvec${suffix}${type} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-naive-bitvec${suffix}${type} PRIVATE BIT_VEC=1 ${macro})

        add_executable(scan-xp-naive-bitvec-adv${suffix}${type} ${SOURCE_FILES})
        target_compile_options(scan-xp-naive-bitvec-adv${suffix}${type} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-naive-bitvec-adv${suffix}${type} PRIVATE BIT_ADVANCED=1 ${macro})

        add_executable(scan-xp-naive-bitvec-rfadv${suffix}${type} ${SOURCE_FILES})
        target_compile_options(scan-xp-naive-bitvec-rfadv${suffix}${type} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-naive-bitvec-rfadv${suffix}${type} PRIVATE BIT_ADVANCED=1 RANGE_FILTERING=1 ${macro})

        add_executable(scan-xp-naive-bitvec-op${suffix}${type} ${SOURCE_FILES})
        target_compile_options(scan-xp-naive-bitvec-op${suffix}${type} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-naive-bitvec-op${suffix}${type} PRIVATE BIT_ONLINE_PACK=1 ${macro})

        add_executable(scan-xp-naive-bitvec-2d${suffix}${type} ${SOURCE_FILES})
        target_compile_options(scan-xp-naive-bitvec-2d${suffix}${type} PRIVATE -O3 -g)
        target_compile_definitions(scan-xp-naive-bitvec-2d${suffix}${type} PRIVATE BIT_VEC=1 BIT_VEC_INDEX=1 ${macro})

        # try the high-bandwidth-memory
        if (KNL)
            if (ENABLE_HBW)
                add_executable(scan-xp-naive-bitvec-hbw${suffix}${type} ${SOURCE_FILES})
                target_compile_options(scan-xp-naive-bitvec-hbw${suffix}${type} PRIVATE -O3 -g)
                target_compile_definitions(scan-xp-naive-bitvec-hbw${suffix}${type} PRIVATE BIT_VEC=1 HBW=1 ${macro})

                add_executable(scan-xp-naive-bitvec-hbw-2d${suffix}${type} ${SOURCE_FILES})
                target_compile_options(scan-xp-naive-bitvec-hbw-2d${suffix}${type} PRIVATE -O3 -g)
                target_compile_definitions(scan-xp-naive-bitvec-hbw-2d${suffix}${type} PRIVATE BIT_VEC=1 HBW=1 BIT_VEC_INDEX=1 ${macro})
            endif ()
        endif ()
    endfunction()

    build_hash_bitmap("" "")
endfunction()

build_our_exec("")

function(build_our_exec_tri_cnt)
    function(build_tri_cnt name macro extra_files)
        set(TRI_CNT_SOURCE_FILES
                scan_xp_triangle_cnt.cpp scan_xp.cpp ${UTIL_FILES} main_triangle_count.cpp tools/pkc.cpp)
        add_executable(tri-cnt-${name} ${TRI_CNT_SOURCE_FILES} set-inter/lemire/intersection.cpp ${extra_files}
                set-inter/lemire/benchintersection.cpp)
        target_compile_options(tri-cnt-${name} PRIVATE -O3 -g -march=native)
        target_compile_definitions(tri-cnt-${name} PRIVATE COMPACT_FORWARD=1 ${macro})
    endfunction()

    if (SSE4)
        build_tri_cnt(sse-bmerge "DEG_DESCENDING_REORDERING=1;SSE_MERGE=1" "")
        build_tri_cnt(sse-hybrid "DEG_DESCENDING_REORDERING=1;SSE_HYBRID=1" "")
        build_tri_cnt(sse-pivot "DEG_DESCENDING_REORDERING=1;SSE_PIVOT=1" "")

        build_tri_cnt(sse-lemire "DEG_DESCENDING_REORDERING=1;LEMIRE_SSE=1" "")
        build_tri_cnt(sse-lemire-high "DEG_DESCENDING_REORDERING=1;LEMIRE_HIGHLY_SCALABLE=1" "")

        build_tri_cnt(sse-roaring "DEG_DESCENDING_REORDERING=1;ROARING=1;USESSE4=1" "../dependencies/CRoaring/roaring.c")

        build_tri_cnt(sse-eh-hybrid "DEG_DESCENDING_REORDERING=1;EMPTY_HEADED=1;NUM_THREADS=1" "")
        build_tri_cnt(sse-eh-uint "DEG_DESCENDING_REORDERING=1;EMPTY_HEADED=1;NUM_THREADS=1;EH_LAYOUT_UINT=1" "")

        build_tri_cnt(sse-bsr-galloping "DEG_DESCENDING_REORDERING=1;SIMD_GALLOPING_BSR=1;BSR=1"
                "set-inter/han/intersection_algos.cpp;set-inter/han/utils/util.cpp")
        build_tri_cnt(sse-bsr-shuffle "DEG_DESCENDING_REORDERING=1;SHUFFLE_BSR=1;BSR=1"
                "set-inter/han/intersection_algos.cpp;set-inter/han/utils/util.cpp")
        build_tri_cnt(sse-bsr-qfilter "DEG_DESCENDING_REORDERING=1;QFILTER_BSR=1;BSR=1"
                "set-inter/han/intersection_algos.cpp;set-inter/han/utils/util.cpp")

        build_tri_cnt(sse-han-qfilter "DEG_DESCENDING_REORDERING=1;HAN_QFILTER=1;"
                "set-inter/han/intersection_algos.cpp;set-inter/han/utils/util.cpp")
        build_tri_cnt(sse-han-bmiss "DEG_DESCENDING_REORDERING=1;HAN_BMISS=1;"
                "set-inter/han/intersection_algos.cpp;set-inter/han/utils/util.cpp")
        build_tri_cnt(sse-han-bmiss-sttni "DEG_DESCENDING_REORDERING=1;HAN_BMISS_STTNI=1;"
                "set-inter/han/intersection_algos.cpp;set-inter/han/utils/util.cpp")
    endif ()

    build_tri_cnt(naive-bsr-scalar-merge "DEG_DESCENDING_REORDERING=1;SCALAR_MERGE_BSR=1;BSR=1"
            "set-inter/han/intersection_algos.cpp;set-inter/han/utils/util.cpp")
    build_tri_cnt(naive-bsr-galloping "DEG_DESCENDING_REORDERING=1;SCALAR_GALLOPING_BSR=1;BSR=1"
            "set-inter/han/intersection_algos.cpp;set-inter/han/utils/util.cpp")

    if (BUILD_LEGACY)
        build_tri_cnt(naive-pivot "DEG_DESCENDING_REORDERING=1;NAIVE=1" "")
        build_tri_cnt(naive-galloping "DEG_DESCENDING_REORDERING=1;NAIVE_GALLOPING=1" "")
    endif ()
    build_tri_cnt(naive-merge "DEG_DESCENDING_REORDERING=1;NAIVE_MERGE=1" "")
    build_tri_cnt(naive-hybrid "DEG_DESCENDING_REORDERING=1;NAIVE_HYBRID=1" "")

    if (BUILD_HASH)
        build_tri_cnt(naive-hash "DEG_DESCENDING_REORDERING=1;HASH=1" "")
        build_tri_cnt(naive-hash-spp "DEG_DESCENDING_REORDERING=1;HASH=1;HASH_SPP=1" "")
    endif ()
    build_tri_cnt(naive-bitvec "DEG_DESCENDING_REORDERING=1;BIT_VEC=1" "")
    build_tri_cnt(naive-bitvec-2d "DEG_DESCENDING_REORDERING=1;BIT_VEC=1;BIT_VEC_INDEX=1" "")
    build_tri_cnt(naive-bitvec-adv "DEG_DESCENDING_REORDERING=1;BIT_VEC_ADV=1" "")
    build_tri_cnt(naive-bitvec-op "DEG_DESCENDING_REORDERING=1;BIT_VEC_OP=1" "")

    if (AVX)
        build_tri_cnt(avx-tetzank "DEG_DESCENDING_REORDERING=1;TETZANK_AVX=1;TETZANK=1" "")
    endif ()

    if (AVX2)
        build_tri_cnt(avx2-bmerge "DEG_DESCENDING_REORDERING=1;AVX2=1;HAS_AVX2=1" "")
        build_tri_cnt(avx2-pivot "DEG_DESCENDING_REORDERING=1;AVX2_PIVOT=1;HAS_AVX2=1" "")
        build_tri_cnt(avx2-galloping "DEG_DESCENDING_REORDERING=1;AVX2_GALLOPING_SINGLE=1;HAS_AVX2=1" "")
        build_tri_cnt(avx2-hybrid "DEG_DESCENDING_REORDERING=1;AVX2_HYBRID=1;HAS_AVX2=1" "")

        build_tri_cnt(avx2-tetzank "DEG_DESCENDING_REORDERING=1;TETZANK_AVX2=1;TETZANK=1" "")
        build_tri_cnt(avx2-asm-tetzank "DEG_DESCENDING_REORDERING=1;TETZANK_AVX2_ASM=1;TETZANK=1" "")

        build_tri_cnt(avx2-lemire "DEG_DESCENDING_REORDERING=1;LEMIRE_AVX2=1" "")
        build_tri_cnt(avx2-roaring "DEG_DESCENDING_REORDERING=1;ROARING=1;USEAVX=1" "../dependencies/CRoaring/roaring.c")
    endif ()
endfunction()

if (BUILD_TRI_CNT)
    build_our_exec_tri_cnt()
endif ()

if (UTILS)
    add_subdirectory(converter)
endif ()
