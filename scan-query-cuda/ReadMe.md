## Compile

* `EXTRA_INCLUDE_DIR` is for `tcmalloc`

```zsh
mkdir build && cd build
cmake .. -DEXTRA_INCLUDE_DIR=/homes/ywangby/workspace/yche/local/include
make -j
make -j
```

* extra options

```
option(CUDA_ALL "Build all CUDA exec." OFF)
option(LEGACY "Build CUDA LEGACY exec." OFF)
```

## Profiling

```zsh
CUDA_VISIBLE_DEVICES=6 nvprof ./scan-xp-cuda-varying-parameters-bitmap /export/data/set-inter-datasets/data/dataset/snap_livejournal 0.8 5 64 3 32 tmp.txt
```

### Parameters

```
auto EPSILON = strtod(argv[2], nullptr);
auto MY_U = atoi(argv[3]);
auto NUMT = atoi(argv[4]);
auto num_pass = atoi(argv[5]);
num_pass_global = num_pass;
num_of_warps_global = atoi(argv[6]);
```

## Code Structures

* Experimental codes

File | Comment
--- | ---
[util](util) | grpah, log, stat, union-find, util, serialization
[set-inter](set-inter) | set intersections on CPUs
[cuda_utils](cuda_utils) | device util and set intersection functions
[experimental](experimental) | some legacy codes
[scan_xp.h](scan_xp.h), [scan_xp_common.cu](scan_xp_common.cu) | SCAN codes
[scan_xp_multi_gpu_multi_pass_common.cu](scan_xp_multi_gpu_multi_pass_common.cu) | counting kernels
[experimental_scan_xp_multigpu_multipass.cu](experimental_scan_xp_multigpu_multipass.cu) | kernel invocation wrappers

* Legacy codes: see [experimental](experimental).

## Others

## UM Techniques

* https://devblogs.nvidia.com/unified-memory-cuda-beginners/
* https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
* https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/

### Co-Processing

* Require CC 6.0+, support co-processing


