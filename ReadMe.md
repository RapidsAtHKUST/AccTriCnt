## Title 

Accelerating All-Edge Common Neighbor Counting on Three Processors  

## Abstract

We propose to accelerate an important but time-consuming operation 
in online graph analytics, which is the counting of common
neighbors for each pair of adjacent vertices (u,v), or edge (u,v),
on three modern processors of different architectures. We study
two representative algorithms for this problem: (1) a merge-based
pivot-skip algorithm (MPS) that intersects the two sets of neighbor
vertices of each edge (u,v) to obtain the count; and (2) a bitmap-based 
algorithm (BMP), which dynamically constructs a bitmap
index on the neighbor set of each vertex u, and for each neighbor
v of u, looks up v’s neighbors in u’s bitmap. We parallelize and
optimize both algorithms on a multicore CPU, an Intel Xeon Phi
Knights Landing processor (KNL), and an NVIDIA GPU. Our experi-
ments show that (1) Both the CPU and the GPU favor BMP whereas
MPS wins on the KNL; (2) Across all datasets, the best performer is
either MPS on the KNL or BMP on the GPU; and (3) Our optimized
algorithms can complete the operation within tens of seconds on
billion-edge Twitter graphs, enabling online analytics.

## Papers To Cite

If you use the codes in your research, please kindly cite the following papers.

* Yulin Che, Zhuohang Lai, Shixuan Sun, Qiong Luo, and Yue Wang. 2019.
Accelerating All-Edge Common Neighbor Counting on Three Processors.
In 48th International Conference on Parallel Processing (ICPP 2019), August
5–8, 2019, Kyoto, Japan. ACM, New York, NY, USA, 10 pages. 
* Yulin Che, Shixuan Sun, Qiong Luo. 2018. Parallelizing Pruning-based Graph Structural Clustering. 
In ICPP 2018: 47th International Conference on Parallel Processing, August 13–16, 2018, 
Eugene, OR, USA. ACM, New York, NY, USA, 10 pages.

## Folder Organization

Folder | Comment
--- | ---
[graph-reordering](graph-reordering) | graph reordering utilities
[scan-query](scan-query) | SCAN (part of which is all-edge TC) / TC on CPUs and KNL
[scan-query-cuda](scan-query-cuda) | SCAN (part of which is all-edge TC) on GPUs (single-GPU/multi-GPU)

## Git Modules

* First time to init this repo: build from scratch, see the following 

File | Comment
--- | ---
[load_gitmodules.sh](load_gitmodules.sh) | load from scratch [.gitmodules](.gitmodules) without git indexing

* Otherwise, do as follows

```
git submodule init
git submodule update
```

* For lemire's `CRoaring` run `amalgamation.sh` (to have a single source file)