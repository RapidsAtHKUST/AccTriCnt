//
// Created by yche on 10/31/17.
//

#ifndef SCAN_QUERY__SET_INTERSECTION_H
#define SCAN_QUERY__SET_INTERSECTION_H

#include <unordered_set>
#include <xmmintrin.h>

#ifdef HASH_SPP

#include <sparsepp/spp.h>

#endif

#include "../util/graph.h"

#include "../util/fake_header.h"


template<typename T>
uint32_t BinarySearchForGallopingSearch(const T *array, uint32_t offset_beg, uint32_t offset_end, int val) {
    while (offset_end - offset_beg >= 32) {
        auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_beg) + offset_end) / 2);
        _mm_prefetch((char *) &array[(static_cast<unsigned long>(mid + 1) + offset_end) / 2], _MM_HINT_T0);
        _mm_prefetch((char *) &array[(static_cast<unsigned long>(offset_beg) + mid) / 2], _MM_HINT_T0);
        if (array[mid] == val) {
            return mid;
        } else if (array[mid] < val) {
            offset_beg = mid + 1;
        } else {
            offset_end = mid;
        }
    }

    // linear search fallback
    for (auto offset = offset_beg; offset < offset_end; offset++) {
        if (array[offset] >= val) {
            return offset;
        }
    }
    return offset_end;
}

template<typename T>
uint32_t GallopingSearch(T *array, uint32_t offset_beg, uint32_t offset_end, int val) {
    if (array[offset_end - 1] < val) {
        return offset_end;
    }
    // galloping
    if (array[offset_beg] >= val) {
        return offset_beg;
    }
    if (array[offset_beg + 1] >= val) {
        return offset_beg + 1;
    }
    if (array[offset_beg + 2] >= val) {
        return offset_beg + 2;
    }

    auto jump_idx = 4u;
    while (true) {
        auto peek_idx = offset_beg + jump_idx;
        if (peek_idx >= offset_end) {
            return BinarySearchForGallopingSearch(array, (jump_idx >> 1) + offset_beg + 1, offset_end, val);
        }
        if (array[peek_idx] < val) {
            jump_idx <<= 1;
        } else {
            return array[peek_idx] == val ? peek_idx :
                   BinarySearchForGallopingSearch(array, (jump_idx >> 1) + offset_beg + 1, peek_idx + 1, val);
        }
    }
}

int ComputeCNNaive(Graph *g, int u, int v);

int ComputeCNNaiveStdMerge(Graph *g, int u, int v);

int ComputeCNGallopingSingleDir(Graph *g, int u, int v);

int ComputeCNGallopingDoubleDir(Graph *g, int u, int v);

int ComputeCNHash(Graph *g, int u, int v, unordered_set<int> &neighbor_sets);

#ifdef HASH_SPP

int ComputeCNHashSPP(Graph *g, int u, int v, spp::sparse_hash_set<int> &neighbor_sets);

#endif

template <typename T>
int ComputeCNHashBitVec(Graph *g, int u, int v, T &neighbor_bits) {
    auto cn_count = 0;
    for (auto offset = g->node_off[v]; offset < g->node_off[v + 1]; offset++) {
        if (neighbor_bits[g->edge_dst[offset]]) {
            cn_count++;
        }
    }
    return cn_count;
}

#endif
