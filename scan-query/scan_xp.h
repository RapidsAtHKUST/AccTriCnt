#ifndef SCAN_QUERY_SCANXP
#define SCAN_QUERY_SCANXP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <omp.h> //OpenMP

#include <iostream>
#include <vector>
#include <fstream>
#include <set>
#include <atomic> //CAS

#include "set-inter/csr_set_intersection_serial.h"
#include "set-inter/csr_set_intersection_simd.h"
#include "set-inter/csr_set_intersection_hybrid.h"

#include "util/fake_header.h"
#include "util/pretty_print.h"

#ifdef EMPTY_HEADED

#include "emptyheaded.hpp"

#endif

#include "util/util.h"
#include "util/lemire/EWAHBoolArray/headers/boolarray.h"
#include "util/timer.h"
//#if defined(HASH) || defined(HASH_SPP) || defined(BIT_VEC)
//#define DEG_DESCENDING_REORDERING
//#endif

using namespace std;

//#define STAT

#include "util/stat.h"
#include "util/log.h"

inline void atomic_add(int *ptr, int add_val) {
    int64_t old_val;
    int64_t new_val;
    do {
        old_val = *ptr;
        new_val = old_val + add_val;
    } while (!__sync_bool_compare_and_swap(ptr, old_val, new_val));
}

struct BSRSet {
    int *base_;
    int *states_;
    int size_;

    BSRSet() = default;
};

template<typename T>
vector<int32_t> core_val_histogram(int n, T &core, bool is_print = false) {
    Timer histogram_timer;
    // core-value histogram
    int max_core_val = 0;
    vector<int32_t> histogram;
#pragma omp parallel
    {
#pragma omp for reduction(max:max_core_val)
        for (auto u = 0; u < n; u++) {
            max_core_val = max(max_core_val, core[u]);
        }
#pragma omp single
        {
            log_info("max value: %d", max_core_val);
            histogram = vector<int32_t>(max_core_val + 1, 0);
        }
        vector<int32_t> local_histogram(histogram.size());

#pragma omp for
        for (auto u = 0; u < n; u++) {
            auto core_val = core[u];
            local_histogram[core_val]++;
        }

        // local_histogram[i] is immutable here.
        for (auto i = 0; i < local_histogram.size(); i++) {
#pragma omp atomic
            histogram[i] += local_histogram[i];
        }
    }
    if (is_print) {
        if (histogram.size() < 400) {
            stringstream ss;
            ss << pretty_print_array(&histogram.front(), histogram.size());
            log_info("values histogram: %s", ss.str().c_str());
        } else {
            {
                stringstream ss;
                ss << pretty_print_array(&histogram.front(), 100);
                log_info("first100 values histogram: %s", ss.str().c_str());
            }
            {

                stringstream ss;
                ss << pretty_print_array(&histogram.front() + histogram.size() - 100, 100);
                log_info("last100 values histogram: %s", ss.str().c_str());
            }
        }
    }
    log_info("Histogram Time: %.9lf s", histogram_timer.elapsed());

    auto &bins = histogram;
    auto bin_cnt = 0;
    int64_t acc = 0;
    auto thresh = n / 10;
    auto last = 0;

    for (auto i = 0; i < histogram.size(); i++) {
        if (bins[i] > 0) {
            bin_cnt++;
            acc += bins[i];
            if (acc > thresh || i == histogram.size() - 1) {
                log_info("bin[%d - %d]: %s", last, i, FormatWithCommas(acc).c_str());
                last = i + 1;
                acc = 0;
            }
        }
    }
    log_info("Reversed Bins...");
    last = histogram.size() - 1;
    acc = 0;
    for (int32_t i = histogram.size() - 1; i > -1; i--) {
        if (bins[i] > 0) {
            bin_cnt++;
            acc += bins[i];
            if (acc > thresh || i == 0) {
                log_info("bin[%d - %d]: %s", i, last, FormatWithCommas(acc).c_str());
                last = i + 1;
                acc = 0;
            }
        }
    }
    log_info("total bin counts: %d", bin_cnt);
    return histogram;
}

#ifdef EMPTY_HEADED

template<typename T>
const Set<T> *NewSet(Graph *g, int u) {
//    static thread_local Set<T> *buffer = (Set<T> *) malloc(2048 * 1024 * 4);
    static thread_local Set<T> *buffer = (Set<T> *) malloc(1024 * 1024 * 32);
    T tmp;

    auto du = g->node_off[u + 1] - g->node_off[u];

    Set<T> *set_u;

    buffer->from_array((uint8_t *) buffer + sizeof(Set<T>),
                       reinterpret_cast<uint32_t *>(g->edge_dst + g->node_off[u]), du);
    if (buffer->number_of_bytes + sizeof(Set<T>) >= 2048 * 1024 * 4) {
        log_fatal("exceed size (>8MB): %s", FormatWithCommas(buffer->number_of_bytes + sizeof(Set<T>)).c_str());
    }
    assert(buffer->number_of_bytes + sizeof(Set<T>) < 1024 * 1024 * 32);
    set_u = (Set<T> *) malloc(buffer->number_of_bytes + sizeof(Set<T>));
    mempcpy(set_u, buffer, buffer->number_of_bytes + sizeof(Set<T>));
    return set_u;
}

template<typename T>
const Set<T> *NewSet(Graph *g, uint32_t offset_beg, uint32_t size) {
    static thread_local Set<T> *buffer = (Set<T> *) malloc(2048 * 1024 * 4);
    T tmp;

    Set<T> *set_u;

    buffer->from_array((uint8_t *) buffer + sizeof(Set<T>),
                       reinterpret_cast<uint32_t *>(g->edge_dst + offset_beg), size);
    assert(buffer->number_of_bytes + sizeof(Set<T>) < 2048 * 1024 * 4);
    set_u = (Set<T> *) malloc(buffer->number_of_bytes + sizeof(Set<T>));
    mempcpy(set_u, buffer, buffer->number_of_bytes + sizeof(Set<T>));
    return set_u;
}

#endif

class SCAN_XP {
private:
    int thread_num_;
    int min_u_;
    double epsilon_;
#ifdef STAT
    vector<InterSectStat> global_bins;
    vector<LowerBoundStat> global_lb_bins;
#endif

public:
    Graph g;
    UnionFind *uf_ptr;

    vector<int> old_vid_dict;
    vector<int> new_vid_dict;

#define ENABLE_REORDERING
#if defined(DEG_DESCENDING_REORDERING) || defined(K_CORE_REORDERING) || defined(ENABLE_REORDERING)

    void ReorderKCoreDegeneracy(Graph &yche_graph);

    void ReorderDegDescending(Graph &g);

    void ReorderRandom(Graph &g);

    void Reorder(Graph &g);

#define REORDERING
#endif

private:
    bool CheckHub(Graph *g, UnionFind *uf, int a);

    uint32_t BinarySearch(int *array, uint32_t offset_beg, uint32_t offset_end, int val);

    int FindSrc(Graph *g, int u, uint32_t edge_idx);

private:
    void KCliqueSortedArray(int k);

public:
    void KClique(int k);

private:
    void TriCntSortedArray();

    void TriCntHash();

    void TriCntBitmap();

    void TriCntBitmapAdv();

    void TriCntBitmapOp();

    void TriCntEmptyHeaded();

    void TriCntRoaring();

    void TriCntBSR();

public:
    void TriCnt();

private:
    using word_type = uint64_t;

    static constexpr uint32_t word_in_bits = sizeof(word_type) * 8;

    void PackVertex(Graph *g, vector<vector<int>> &partition_id_lst,
                    vector<vector<word_type>> &bitmap_in_partition_lst, int u, int &packed_num);

private:
    void CheckCoreEmptyHeaded(Graph *g);

    void CheckCoreBSR(Graph *g);

    void CheckCoreRoaring(Graph *g);

    void CheckCoreHash(Graph *g);

    void CheckCoreBitmap(Graph *g);

    void CheckCoreBitmapAdvanced(Graph *g);

    void CheckCoreBitmapOnlinePack(Graph *g);

    void CheckCoreCompactForward(Graph *g);

    void CheckCoreHybridCFNormal(Graph *g);

    void CheckCoreSortedArray(Graph *g);

private:
    void CheckCore(Graph *g);

    void CheckCoreCompSimCore(Graph *g);

    void ClusterCore();

    void LabelNonCore();

    void PostProcess();

    void MarkClusterMinEleAsId(UnionFind *union_find_ptr);

    void PrepareResultOutput();

public:
    SCAN_XP(int thread_num, int min_u, double epsilon, char *dir);

    SCAN_XP(int thread_num, char *dir);

    ~SCAN_XP();

    void Execute();
};

#endif

