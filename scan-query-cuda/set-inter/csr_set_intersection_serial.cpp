//
// Created by yche on 10/31/17.
//


#include "csr_set_intersection_serial.h"

#include <immintrin.h>  //AVX

int ComputeCNHash(Graph *g, int u, int v, unordered_set<int> &neighbor_sets) {
    auto cn_count = 0;
    for (auto offset = g->node_off[v]; offset < g->node_off[v + 1]; offset++) {
        if (neighbor_sets.find(g->edge_dst[offset]) != neighbor_sets.end()) {
            cn_count++;
        }
    }
    return cn_count;
}

#ifdef HASH_SPP

int ComputeCNHashSPP(Graph *g, int u, int v, spp::sparse_hash_set<int> &neighbor_sets) {
    auto cn_count = 0;
    for (auto offset = g->node_off[v]; offset < g->node_off[v + 1]; offset++) {
        if (neighbor_sets.find(g->edge_dst[offset]) != neighbor_sets.end()) {
            cn_count++;
        }
    }
    return cn_count;
}
#endif

int ComputeCNNaive(Graph *g, int u, int v) {

    auto cn_count = 0;
    auto offset_nei_u = g->node_off[u], offset_nei_v = g->node_off[v];
    auto off_u_end = g->node_off[u + 1], off_v_end = g->node_off[v + 1];
    while (true) {
        while (g->edge_dst[offset_nei_u] < g->edge_dst[offset_nei_v]) {
            ++offset_nei_u;
            if (offset_nei_u >= off_u_end) {
                return cn_count;
            }
        }

        while (g->edge_dst[offset_nei_u] > g->edge_dst[offset_nei_v]) {
            ++offset_nei_v;
            if (offset_nei_v >= off_v_end) {
                return cn_count;
            }
        }

        if (g->edge_dst[offset_nei_u] == g->edge_dst[offset_nei_v]) {
            cn_count++;
            ++offset_nei_u;
            ++offset_nei_v;
            if (offset_nei_u >= off_u_end || offset_nei_v >= off_v_end) {
                return cn_count;
            }
        }
    }
}

int ComputeCNNaiveStdMerge(Graph *g, int u, int v) {

    auto cn_count = 0;
    auto offset_nei_u = g->node_off[u], offset_nei_v = g->node_off[v];
    auto off_u_end = g->node_off[u + 1], off_v_end = g->node_off[v + 1];
    while (true) {
        if (g->edge_dst[offset_nei_u] < g->edge_dst[offset_nei_v]) {
            ++offset_nei_u;
            if (offset_nei_u >= off_u_end) {
                return cn_count;
            }
        } else if (g->edge_dst[offset_nei_u] > g->edge_dst[offset_nei_v]) {
            ++offset_nei_v;
            if (offset_nei_v >= off_v_end) {
                return cn_count;
            }
        } else {
            cn_count++;
            ++offset_nei_u;
            ++offset_nei_v;
            if (offset_nei_u >= off_u_end || offset_nei_v >= off_v_end) {
                return cn_count;
            }
        }
    }
}

int ComputeCNGallopingDoubleDir(Graph *g, int u, int v) {

    auto cn_count = 0;
    auto offset_nei_u = g->node_off[u], offset_nei_v = g->node_off[v];
    auto off_u_end = g->node_off[u + 1], off_v_end = g->node_off[v + 1];
    while (true) {
        offset_nei_u = GallopingSearch(g->edge_dst, offset_nei_u, off_u_end, g->edge_dst[offset_nei_v]);
        if (offset_nei_u >= off_u_end) {
            return cn_count;
        }

        offset_nei_v = GallopingSearch(g->edge_dst, offset_nei_v, off_v_end, g->edge_dst[offset_nei_u]);
        if (offset_nei_v >= off_v_end) {
            return cn_count;
        }

        if (g->edge_dst[offset_nei_u] == g->edge_dst[offset_nei_v]) {
            cn_count++;
            ++offset_nei_u;
            ++offset_nei_v;
            if (offset_nei_u >= off_u_end || offset_nei_v >= off_v_end) {
                return cn_count;
            }
        }
    }
}

int ComputeCNGallopingSingleDir(Graph *g, int u, int v) {

    auto cn_count = 0;
    if (g->degree[u] > g->degree[v]) {
        auto tmp = u;
        u = v;
        v = tmp;
    }
    auto offset_nei_u = g->node_off[u], offset_nei_v = g->node_off[v];
    auto off_u_end = g->node_off[u + 1], off_v_end = g->node_off[v + 1];

    while (true) {
        while (g->edge_dst[offset_nei_u] < g->edge_dst[offset_nei_v]) {
            ++offset_nei_u;
            if (offset_nei_u >= off_u_end) {
                return cn_count;
            }
        }

        offset_nei_v = GallopingSearch(g->edge_dst, offset_nei_v, off_v_end, g->edge_dst[offset_nei_u]);
        if (offset_nei_v >= off_v_end) {
            return cn_count;
        }

        if (g->edge_dst[offset_nei_u] == g->edge_dst[offset_nei_v]) {
            cn_count++;
            ++offset_nei_u;
            ++offset_nei_v;
            if (offset_nei_u >= off_u_end || offset_nei_v >= off_v_end) {
                return cn_count;
            }
        }
    }
}
