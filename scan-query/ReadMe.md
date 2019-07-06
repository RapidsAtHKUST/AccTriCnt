## Build (With Compiler Supporting C++11)

```zsh
mkdir -p build && cd build
cmake ..
```

* to turn off or on options, just add e.g. `-DBUILD_TRI_CNT=OFF` after `cmake ..`

```
option(BUILD_TRI_CNT "build-tri-cnt" ON)
option(ENABLE_HBW "Build With HBW for KNL." ON)
option(UTILS "enable utils: converter " ON)
option(BUILD_HASH "build-hash" OFF)
option(BUILD_LEGACY "build-legacy" OFF)
option(KNL "enable knl compilation" OFF)
```

* for knl building, add `-DKNL=ON`, and use latest g++ supporting `AVX-512`, via `scl enable devtoolset-7 zsh`

## Usage (TC/SCAN)

```zsh
./tri-cnt-naive-bitvec /mnt/storage1/yche/datasets/snap_livejournal 40 deg
./scan-xp-naive-bitvec /mnt/storage1/yche/datasets/snap_livejournal/ 0.2 5 40 deg
```

## Reordering Methods

Conditions | Methods
--- | ---
without dictionary | deg, kcore, random
with dictionary |  gro, cache, bfsr, dfs, hybrid, rcm-cache, slashburn
else | orginal order

## Code Structure

Folder | Comment
--- | ---
[converter](converter) | converting edge list into our graph format
[set-inter](set-inter) | emptyheaded/han/lemire/tetzank/our set intersection methods
[tools](tools) | k-core decomposition utility
[util](util) | lemire's bitmap, search, parallel sort (twice memory), graph, log, stat, timer, union-find, serailization
[scan_xp.cpp](scan_xp.cpp) | SCAN: structral graph clustering on networks
[scan_xp_triangle_cnt.cpp](scan_xp_triangle_cnt.cpp) | triangle counting
[main_scanxp.cpp](main_scanxp.cpp), [main_triangle_count.cpp](main_triangle_count.cpp) | SCAN/TC main entries

