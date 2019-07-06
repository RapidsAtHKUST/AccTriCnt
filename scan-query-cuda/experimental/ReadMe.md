## Multi-GPU Dynamic Scheduling Pitfalls

* occupancy: ensure make the minimum number of blocks to satisfy the utilization requirement `2048/128 = 16` at leat, but still tail-effects...
* unified-memory-prefetch: ensure much more regular page access pattern, do not split into too much tasks (maybe just `8`) 

## File Organization

file | description
--- | ---
[scan_xp_common.cu](scan_xp_common.cu) | scan-xp common algorithmic functions, directly copy and paste into separate files
[set_inter_device_functions.cuh](set_inter_device_functions.cuh) | set-intersection scalar device functions
[cuda_util.cuh](cuda_util.cuh) | cuda utilities
[scan_xp_multigpu_multipass.cu](scan_xp_multigpu_multipass.cu) | multi-gpu multiple-pass implementation for `snap-friendster` and larger datasets

file | description
--- | ---
[scan_xp.cu](scan_xp.cu) | `warning: depracated`: single-gpu implementation
[scan_xp_multipass.cu](scan_xp_multipass.cu) | single-gpu multiple-pass for `snap-friendster` and larger datasets
[scan_xp_multigpu.cu](scan_xp_multigpu.cu) | `warning: depracated`: a special case of [scan_xp_multigpu_multipass.cu](scan_xp_multigpu_multipass.cu)

## Compile (CUDA-ALL, with MergeBased and Baseline)

see [CMakeLists.txt](CMakeLists.txt)

attention: first time `make -j` may report error, then `make -j` again to build all cuda executables

```zsh
cmake .. -DEXTRA_INCLUDE_DIR=/homes/ywangby/workspace/yche/local/include
make -j
make -j
```

## Run 

```zsh
CUDA_VISIBLE_DEVICES=1 ./scan-xp-cuda-bitmap-warp-per-vertex-multi-gpu-multi-pass /export/data/set-inter-datasets/data/dataset/webgraph_twitter/rev_deg 0.2 5 56
```