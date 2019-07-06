#include "scan_xp.h"

#include <chrono>
#include <unordered_set>
#include <random>

#include <sparsepp/spp.h>

#include "libpopcnt.h"
#include "set-inter/lemire/intersection.h"
#include "set-inter/han/intersection_algos.hpp"

#ifdef HBW

#include <memkind.h>
#include <hbw_allocator.h>

#endif

#ifdef TBB

#include <tbb/parallel_sort.h>
//#include <tbb/parallel_scan.h>

#endif

#ifdef ROARING

#include "roaring.hh"

#endif

#ifdef TETZANK

#include "set-inter/tetzank/intersection/avx.hpp"

#endif

#if defined(TETZANK) && defined(__AVX2__)

#include "set-inter/tetzank/intersection/avx2.hpp"

#endif

#include "util/stat.h"
#include "util/util.h"
#include "util/yche_serialization.h"
#include "util/fake_header.h"
#include "util/pretty_print.h"
#include "util/sort/parasort_cmp.h"
#include "util/search/search.h"

#define NUM_OF_BINS (10)
#define MAX_DEGREE (2048 * 1024)

using namespace std::chrono;

SCAN_XP::SCAN_XP(int thread_num, int min_u, double epsilon, char *dir) :
        thread_num_(thread_num), min_u_(min_u), epsilon_(epsilon),
#ifdef STAT
global_bins(NUM_OF_BINS),
global_lb_bins(NUM_OF_BINS),
#endif
        g(Graph(dir)), uf_ptr(nullptr) {

    log_info("Cur #of threads: %d, max-cap (env): %d", omp_get_num_threads(), omp_get_max_threads());
    log_info("thread num: %d", thread_num_);
    log_info("graph, n: %s, m: %s", FormatWithCommas(g.nodemax).c_str(), FormatWithCommas(g.edgemax).c_str());

    uf_ptr = new UnionFind(g.nodemax);

    // pre-processing
//#pragma omp parallel for num_threads(thread_num_) schedule(dynamic, 100000)
#pragma omp parallel for schedule(dynamic, 100000)
    for (auto i = 0u; i < g.nodemax; i++) {
        g.core_count[i] = 0;
        g.label[i] = UNCLASSIFIED;
        for (auto n = g.node_off[i]; n < g.node_off[i + 1]; n++) {
            g.common_node_num[n] = 2;
        }
    }
}

SCAN_XP::~SCAN_XP() {
    delete uf_ptr;
}

uint32_t SCAN_XP::BinarySearch(int *array, uint32_t offset_beg, uint32_t offset_end, int val) {
    auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_beg) + offset_end) / 2);
    if (array[mid] == val) { return mid; }
    return val < array[mid] ? BinarySearch(array, offset_beg, mid, val) : BinarySearch(array, mid + 1, offset_end, val);
}

int SCAN_XP::FindSrc(Graph *g, int u, uint32_t edge_idx) {
    if (edge_idx >= g->node_off[u + 1]) {
        // update last_u, preferring galloping instead of binary search because not large range here
        u = GallopingSearch(g->node_off, static_cast<uint32_t>(u) + 1, g->nodemax + 1, edge_idx);
        // 1) first > , 2) has neighbor
        if (g->node_off[u] > edge_idx) {
            while (g->degree[u - 1] == 1) { u--; }
            u--;
        } else {
            // g->node_off[u] == i
            while (g->degree[u] == 1) {
                u++;
            }
        }
    }
    return u;
}

void SCAN_XP::CheckCoreCompSimCore(Graph *g) {
    auto start = high_resolution_clock::now();
//#pragma omp parallel for num_threads(thread_num_)
#pragma omp parallel for
    for (auto i = 0u; i < g->edgemax; i++) {
        static thread_local auto u = 0;
        u = FindSrc(g, u, i);

        long double du = g->node_off[u + 1] - g->node_off[u] + 1;
        long double dv = g->node_off[g->edge_dst[i] + 1] - g->node_off[g->edge_dst[i]] + 1;
        auto sim_value = static_cast<double>((long double) g->common_node_num[i] / sqrt(du * dv));
        if (sim_value >= epsilon_) {
            g->core_count[u]++;
            g->similarity[i] = true;
        } else {
            g->similarity[i] = false;
        }
    }

//#pragma omp parallel for num_threads(thread_num_)
#pragma omp parallel for
    for (auto i = 0u; i < g->nodemax; i++) {
        if (g->core_count[i] >= min_u_) {
            g->label[i] = CORE;
        };
    }
    auto end = high_resolution_clock::now();
    log_info("core-checking sim-core-comp cost : %.3lf s, Mem Usage: %s KB",
             duration_cast<milliseconds>(end - start).count() / 1000.0, FormatWithCommas(getValue()).c_str());
}

void SCAN_XP::CheckCoreHash(Graph *g) {
#pragma omp parallel num_threads(thread_num_)
    {
#ifdef STAT
        vector<InterSectStat> bins(NUM_OF_BINS);
        vector<LowerBoundStat> lb_bins(NUM_OF_BINS);
#endif

#pragma omp for schedule(dynamic, 6000)
        for (auto i = 0u; i < g->edgemax; i++) {
            // FindSrc.
            // Remove edge_src optimization, assuming task scheduling in FIFO-queue-mode
            static thread_local auto u = 0;
            u = FindSrc(g, u, i);

            static thread_local auto last_u = -1;
#if defined(HASH)
            static thread_local unordered_set<int> hash_table;
#elif defined(HASH_SPP)
            static thread_local spp::sparse_hash_set<int> hash_table;
#endif
#if defined(HASH) || defined(HASH_SPP)
            if (last_u != u) {
                last_u = u;
                // Clear & Construct hash table.
                hash_table.clear();
                for (auto offset = g->node_off[u]; offset < g->node_off[u + 1]; offset++) {
                    hash_table.emplace(g->edge_dst[offset]);
                }
            }
#endif

            auto v = g->edge_dst[i];
#if defined(DEG_DESCENDING_REORDERING)
            if (u < v) {
#else
            if (g->degree[u] > g->degree[v] || ((g->degree[u] == g->degree[v]) && (u < v))) {
#endif
#ifdef STAT
                auto clk_beg = high_resolution_clock::now();
#endif
                // Compute.
#if defined(HASH)
                g->common_node_num[i] += ComputeCNHash(g, u, v, hash_table);
#elif defined(HASH_SPP)
                g->common_node_num[i] += ComputeCNHashSPP(g, u, v, hash_table);
#endif

#ifdef STAT
                auto clk_end = high_resolution_clock::now();
                long time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);

                clk_beg = high_resolution_clock::now();
#endif
                // Symmetrically Assign.
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                u)] = g->common_node_num[i];
#ifdef STAT
                clk_end = high_resolution_clock::now();
                time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                lb_bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);
#endif
            }
        }
#ifdef STAT
#pragma omp critical
        {
            for (int i = 0; i < 10; i++) {
                global_bins[i].MergeStat(bins[i]);
                global_lb_bins[i].MergeStat(lb_bins[i]);
            }
        }
#endif
    }
}

void SCAN_XP::CheckCoreBitmap(Graph *g) {
    auto clk_beg = high_resolution_clock::now();
#pragma omp parallel num_threads(thread_num_)
    {

#ifdef STAT
        vector<InterSectStat> bins(NUM_OF_BINS);
        vector<LowerBoundStat> lb_bins(NUM_OF_BINS);
#endif
#pragma omp for schedule(dynamic, 6000)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local auto u = 0;
            u = FindSrc(g, u, i);

            static thread_local auto last_u = -1;

#if defined(HBW)
            static thread_local auto bits_vec = vector<bool, hbw::allocator<bool>>(g->nodemax, false);
#else
            static thread_local auto bits_vec = vector<bool>(g->nodemax, false);
#endif

#if defined(BIT_VEC_INDEX) && defined(HBW)
            static thread_local auto index_bits_vec = vector<bool, hbw::allocator<bool>>(
                    (g->nodemax + INDEX_RANGE - 1) / INDEX_RANGE, false);
#elif  defined(BIT_VEC_INDEX)
            static thread_local auto index_bits_vec = vector<bool>(
                    (g->nodemax + INDEX_RANGE - 1) / INDEX_RANGE, false);
#endif

#if defined(BIT_VEC_INDEX)
            if (last_u != u) {
                // clear previous
                if (last_u != -1) {
                    for (auto offset = g->node_off[last_u]; offset < g->node_off[last_u + 1]; offset++) {
                        bits_vec[g->edge_dst[offset]] = false;
                    }
                }
                index_bits_vec.assign(index_bits_vec.size(), false);
                for (auto offset = g->node_off[u]; offset < g->node_off[u + 1]; offset++) {
                    bits_vec[g->edge_dst[offset]] = true;
                    index_bits_vec[g->edge_dst[offset] >> INDEX_BIT_SCALE_LOG] = true;
                }
                last_u = u;
            }
#else
            if (last_u != u) {
                // clear previous
                if (last_u != -1) {
                    for (auto offset = g->node_off[last_u]; offset < g->node_off[last_u + 1]; offset++) {
                        bits_vec[g->edge_dst[offset]] = false;
                    }
                }
            for (auto offset = g->node_off[u]; offset < g->node_off[u + 1]; offset++) {
                bits_vec[g->edge_dst[offset]] = true;
            }
            last_u = u;
        }
#endif
            auto v = g->edge_dst[i];
#if defined(DEG_DESCENDING_REORDERING)
            if (u < v) {
#else
            if (g->degree[u] > g->degree[v] || ((g->degree[u] == g->degree[v]) && (u < v))) {
#endif
#ifdef STAT
                auto clk_beg = high_resolution_clock::now();
#endif
#if defined(BIT_VEC_INDEX)
//                uint32_t rev_edge_idx;
//                g->common_node_num[i] += ComputeCNHashBitVec2DMemIdx(g, u, v, bits_vec, index_bits_vec, rev_edge_idx);
                g->common_node_num[i] += ComputeCNHashBitVec2D(g, u, v, bits_vec, index_bits_vec);
#else
                g->common_node_num[i] += ComputeCNHashBitVec(g, u, v, bits_vec);
#endif
#ifdef STAT
                auto clk_end = high_resolution_clock::now();
                long time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);

                clk_beg = high_resolution_clock::now();
#endif
                // Symmetrically Assign.
#if defined(BIT_VEC_INDEX)
//                g->common_node_num[rev_edge_idx] = g->common_node_num[i];
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                u)] = g->common_node_num[i];
#else
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                u)] = g->common_node_num[i];
#endif
#ifdef STAT
                clk_end = high_resolution_clock::now();
                time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                lb_bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);
#endif
            }
        }
#ifdef STAT
#pragma omp critical
        {
            for (int i = 0; i < 10; i++) {
                global_bins[i].MergeStat(bins[i]);
                global_lb_bins[i].MergeStat(lb_bins[i]);
            }
        }
#endif

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("All-Edge Computation Cost (Bitmap): %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            log_info("Finish Bitmap Checking.");
        }
    }
}

void SCAN_XP::PackVertex(Graph *g, vector<vector<int>> &partition_id_lst,
                         vector<vector<SCAN_XP::word_type>> &bitmap_in_partition_lst, int u, int &packed_num) {
    auto prev_blk_id = -1;
    auto num_blks = 0;
    auto pack_num_u = 0;
    for (auto off = g->node_off[u]; off < g->node_off[u + 1]; off++) {
        auto v = g->edge_dst[off];
        auto cur_blk_id = v / word_in_bits;
        if (cur_blk_id == prev_blk_id) {
            pack_num_u++;
        } else {
            prev_blk_id = cur_blk_id;
            num_blks++;
        }
    }

    prev_blk_id = -1;
    if ((g->node_off[u + 1] - g->node_off[u]) >= 16 && (g->node_off[u + 1] - g->node_off[u]) / num_blks > 2) {
        packed_num++;
        for (auto off = g->node_off[u]; off < g->node_off[u + 1]; off++) {
            auto v = g->edge_dst[off];
            auto cur_blk_id = v / word_in_bits;
            if (cur_blk_id == prev_blk_id) {
                pack_num_u++;
            } else {
                prev_blk_id = cur_blk_id;
                num_blks++;
                partition_id_lst[u].emplace_back(cur_blk_id);
                bitmap_in_partition_lst[u].emplace_back(0);
            }
            bitmap_in_partition_lst[u].back() |= static_cast<word_type>(1u) << (v % word_in_bits);
        }
    }
}

void SCAN_XP::CheckCoreBitmapOnlinePack(Graph *g) {
    int packed_num = 0;

    auto clk_beg = high_resolution_clock::now();

    uint64_t empty_intersection_num = 0;
    uint64_t empty_index_word_num = 0;

    uint64_t check_word_num = 0;
    uint64_t tc_num = 0;

    using word_type = uint64_t;
    vector<vector<int>> partition_id_lst(g->nodemax);
    vector<vector<word_type>> bitmap_in_partition_lst(g->nodemax);

#pragma omp parallel num_threads(thread_num_)
    {

        auto bool_arr = BoolArray<word_type>(g->nodemax);

        // Pre-Process: Indexing Words.
#pragma omp for schedule(dynamic, 100) reduction(+:packed_num)
        for (auto u = 0; u < g->nodemax; u++) {
            PackVertex(g, partition_id_lst, bitmap_in_partition_lst, u, packed_num);
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("Packed#: %s", FormatWithCommas(packed_num).c_str());
            log_info("PreProcess-BitmapOP Cost: %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            clk_beg = high_resolution_clock::now();
        }

#pragma omp for schedule(dynamic, 6000) reduction(+:check_word_num) reduction(+:tc_num) reduction(+:empty_intersection_num) \
reduction(+:empty_index_word_num)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local auto u = 0;
            u = FindSrc(g, u, i);

            static thread_local auto last_u = -1;
            if (last_u != u) {
                // clear previous
                if (last_u != -1) {
                    for (auto offset = g->node_off[last_u]; offset < g->node_off[last_u + 1]; offset++) {
                        auto v = g->edge_dst[offset];
                        bool_arr.setWord(v / word_in_bits, 0);
                    }
                }
                for (auto offset = g->node_off[u]; offset < g->node_off[u + 1]; offset++) {
                    auto v = g->edge_dst[offset];
                    bool_arr.set(v);
                }
                last_u = u;
            }

            auto v = g->edge_dst[i];

            if (g->degree[u] > g->degree[v] || ((g->degree[u] == g->degree[v]) && (u < v))) {
                auto local_cnt = 0;

                word_type bitmap_pack = 0;
                if (!partition_id_lst[v].empty()) {
                    for (auto wi = 0; wi < partition_id_lst[v].size(); wi++) {
                        auto index_word = bool_arr.getWord(partition_id_lst[v][wi]);
                        auto res = index_word & bitmap_in_partition_lst[v][wi];
                        if (index_word == 0) {
                            empty_index_word_num++;
                        }
                        if (res == 0) {
                            empty_intersection_num++;
                        }
                        local_cnt += popcnt(&res, sizeof(word_type));
                    }
                    check_word_num += partition_id_lst[v].size();

                } else if (g->node_off[v + 1] - g->node_off[v] > 0) {
                    auto prev_blk_id = g->edge_dst[g->node_off[v]] / word_in_bits;

                    for (auto off = g->node_off[v]; off < g->node_off[v + 1]; off++) {
                        auto w = g->edge_dst[off];
                        // Online pack.
                        auto cur_blk_id = w / word_in_bits;
                        if (cur_blk_id != prev_blk_id) {
                            // Execute prev.
                            auto index_word = bool_arr.getWord(prev_blk_id);
                            auto res = index_word & bitmap_pack;
                            if (index_word == 0) {
                                empty_index_word_num++;
                            }
                            if (res == 0) {
                                empty_intersection_num++;
                            }
                            local_cnt += popcnt(&res, sizeof(word_type));

                            check_word_num++;
                            bitmap_pack = 0;
                            prev_blk_id = cur_blk_id;
                        }
                        bitmap_pack |= static_cast<word_type >(1u) << (w % word_in_bits);
                    }
//                    // Execute last.
                    auto index_word = bool_arr.getWord(prev_blk_id);
                    auto res = index_word & bitmap_pack;
                    if (index_word == 0) {
                        empty_index_word_num++;
                    }
                    if (res == 0) {
                        empty_intersection_num++;
                    }
                    check_word_num++;
                    local_cnt += popcnt(&res, sizeof(word_type));
                }

                tc_num += local_cnt;
                g->common_node_num[i] += local_cnt;
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                u)] = g->common_node_num[i];
            }
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("All-Edge Computation Cost (Bitmap-OP): %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            log_debug("Check Word Num: %s (Bits: %s), TC: %s/%s, Valid/Check: %.6lf",
                      FormatWithCommas(check_word_num).c_str(),
                      FormatWithCommas(check_word_num * word_in_bits).c_str(),
                      FormatWithCommas(tc_num).c_str(),
                      FormatWithCommas(tc_num / 3).c_str(),
                      static_cast<double>(tc_num) / (check_word_num * word_in_bits));
            log_debug("Empty Num: %s (Empty Idx: %s) (EmptyIdx/All, Empty/All: %.6lf, %.6lf)",
                      FormatWithCommas(empty_intersection_num).c_str(),
                      FormatWithCommas(empty_index_word_num).c_str(),
                      static_cast<double>(empty_index_word_num) / check_word_num,
                      static_cast<double>(empty_intersection_num) / check_word_num);
            log_debug("Valid/Check-RF: %.6lf (Hits: %.6lf)",
                      static_cast<double>(tc_num) / ((check_word_num - empty_index_word_num) * word_in_bits),
                      static_cast<double>(tc_num) / ((check_word_num - empty_index_word_num)));
        }
    }
}

void SCAN_XP::CheckCoreBitmapAdvanced(Graph *g) {
    auto clk_beg = high_resolution_clock::now();

    int packed_num = 0;

    uint64_t check_word_num = 0;
    uint64_t dense_tc_num = 0;
    uint64_t sparse_tc_num = 0;
    uint64_t max_num = 0;
    uint64_t filter_num = 0;
    uint64_t other_num = 0;

    vector<vector<int>> partition_id_lst(g->nodemax);
    vector<vector<word_type>> bitmap_in_partition_lst(g->nodemax);
    {
        auto clk_end = high_resolution_clock::now();
        auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
        log_info("MemAlloc Cost: %.9lf s, %s KB", time / pow(10, 9), FormatWithCommas(getValue()).c_str());
    }
#pragma omp parallel num_threads(thread_num_)
    {
        auto bool_arr = BoolArray<word_type>(g->nodemax);

#ifdef RANGE_FILTERING
        constexpr uint32_t max_range_bits = 1024 * 32 * 8;
        auto range_idx_arr = BoolArray<word_type>(max_range_bits);
#endif

        // Pre-Process: Indexing Words.
#pragma omp for schedule(dynamic, 100) reduction(+:packed_num)
        for (auto u = 0; u < g->nodemax; u++) {
            PackVertex(g, partition_id_lst, bitmap_in_partition_lst, u, packed_num);
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("Packed#: %s", FormatWithCommas(packed_num).c_str());
            log_info("PreProcess-BitmapAdvanced Cost: %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            clk_beg = high_resolution_clock::now();
        }

#pragma omp for schedule(dynamic, 6000) reduction(+:dense_tc_num) reduction(+:sparse_tc_num) \
reduction(+:filter_num) reduction(+:other_num) reduction(+:check_word_num) reduction(+:max_num)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local auto u = 0;
            u = FindSrc(g, u, i);

            int min_ele, max_ele, range_gap;
            static thread_local auto last_u = -1;
            if (last_u != u) {
                if (last_u != -1) {
                    for (auto offset = g->node_off[last_u]; offset < g->node_off[last_u + 1]; offset++) {
                        auto v = g->edge_dst[offset];
                        bool_arr.setWord(v / word_in_bits, 0);
                    }
#ifdef RANGE_FILTERING
                    range_idx_arr.reset();
#endif
                }
#ifdef RANGE_FILTERING
                min_ele = g->edge_dst[g->node_off[u]];
                max_ele = g->edge_dst[g->node_off[u + 1] - 1] + 1;
                range_gap = (max_ele - min_ele) / max_range_bits + 1;
#endif
                for (auto offset = g->node_off[u]; offset < g->node_off[u + 1]; offset++) {
                    auto v = g->edge_dst[offset];
#ifdef RANGE_FILTERING
                    range_idx_arr.set((v - min_ele) / range_gap);
#endif
                    bool_arr.set(v);
                }
                last_u = u;
            }

            auto v = g->edge_dst[i];
            if (g->degree[u] > g->degree[v] || ((g->degree[u] == g->degree[v]) && (u < v))) {
                auto local_cnt = 0;
                if (!partition_id_lst[v].empty()) {
                    for (auto wi = 0; wi < partition_id_lst[v].size(); wi++) {
                        auto res = bool_arr.getWord(partition_id_lst[v][wi]) & bitmap_in_partition_lst[v][wi];
                        local_cnt += popcnt(&res, sizeof(word_type));
                    }
                    dense_tc_num += local_cnt;
                    check_word_num += partition_id_lst[v].size();
                } else {

#ifdef RANGE_FILTERING
                    max_num += g->node_off[v + 1] - g->node_off[v];
                    auto off_beg = LinearSearch(g->edge_dst, g->node_off[v], g->node_off[v + 1], min_ele);
                    for (auto off = off_beg; off < g->node_off[v + 1]; off++) {
                        auto w = g->edge_dst[off];
                        if (w >= max_ele) {
                            break;
                        }
                        if (range_idx_arr.get((w - min_ele) / range_gap)) {
                            other_num++;
                            if (bool_arr.get(w))
                                local_cnt++;
                        } else {
                            filter_num++;
                        }
                    }
#else
                    for (auto off = g->node_off[v]; off < g->node_off[v + 1]; off++) {
                        auto w = g->edge_dst[off];
                        if (bool_arr.get(w))
                            local_cnt++;
                    }
#endif
                    sparse_tc_num += local_cnt;
                }

                // Symmetrically Assign.
                if (local_cnt > 0) {
                    g->common_node_num[i] += local_cnt;
                    g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                    u)] = g->common_node_num[i];
                }
            }
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("All-Edge Computation Cost (Bitmap-ADV): %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            log_debug("Dense Tc: %s, Sparse Tc: %s, Total: %s, Tri: %s ",
                      FormatWithCommas(dense_tc_num).c_str(),
                      FormatWithCommas(sparse_tc_num).c_str(),
                      FormatWithCommas(dense_tc_num + sparse_tc_num).c_str(),
                      FormatWithCommas((dense_tc_num + sparse_tc_num) / 3).c_str());
            log_debug("Check Word Num: %s (Bits: %s), Valid/Check: %.6lf", FormatWithCommas(check_word_num).c_str(),
                      FormatWithCommas(check_word_num * word_in_bits).c_str(),
                      static_cast<double>(dense_tc_num) / (check_word_num * word_in_bits));
#ifdef RANGE_FILTERING
            log_debug(
                    "All Num: %s (At Most: %s, Ratio: %.6lf), Filter Num: %s, Other Num: %s, Left/All: %.6lf, Valid/Left: %.6lf",
                    FormatWithCommas(filter_num + other_num).c_str(),
                    FormatWithCommas(max_num).c_str(),
                    static_cast<double>(filter_num + other_num) / max_num,
                    FormatWithCommas(filter_num).c_str(),
                    FormatWithCommas(other_num).c_str(),
                    static_cast<double>(other_num ) / (filter_num + other_num),
                    static_cast<double>(sparse_tc_num) / other_num);
#endif
        }
    }
}


void SCAN_XP::CheckCoreSortedArray(Graph *g) {
    auto clk_beg = high_resolution_clock::now();

#pragma omp parallel num_threads(thread_num_)
    {
#ifdef STAT
        vector<InterSectStat> bins(NUM_OF_BINS);
        vector<LowerBoundStat> lb_bins(NUM_OF_BINS);
#endif
#pragma omp for schedule(dynamic, 6000)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local auto u = 0;
            static thread_local uint32_t *my_ptr = new uint32_t[1024 * 1024 * 2];
            static thread_local int *my_ptr_han = new int[1024 * 1024 * 2];
            u = FindSrc(g, u, i);

            auto v = g->edge_dst[i];
            if (u < v) {
#ifdef STAT
                auto clk_beg = high_resolution_clock::now();
#endif

#if defined(NAIVE)
                g->common_node_num[i] += ComputeCNNaive(g, u, v);
#elif defined(TETZANK_AVX)
                //                g->common_node_num[i] += intersect_vector_avx(
                                g->common_node_num[i] += intersect_vector_avx_count(
                                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                                        g->node_off[u + 1] - g->node_off[u],
                                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                                        g->node_off[v + 1] - g->node_off[v]
                                        );
                //                        , my_ptr);
#elif defined(TETZANK_AVX2)
                //                g->common_node_num[i] += intersect_vector_avx2(
                                g->common_node_num[i] += intersect_vector_avx2_count(
                                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                                        g->node_off[u + 1] - g->node_off[u],
                                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                                        g->node_off[v + 1] - g->node_off[v]
                                        );
                //                        , my_ptr);
#elif defined(TETZANK_AVX2_ASM)
                //                g->common_node_num[i] += intersect_vector_avx2_asm(
                                g->common_node_num[i] += intersect_vector_avx2_asm_count(
                                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                                        g->node_off[u + 1] - g->node_off[u],
                                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                                        g->node_off[v + 1] - g->node_off[v]
                                        );
                //                        , my_ptr);
#elif defined(LEMIRE_SSE)
                g->common_node_num[i] += SIMDCompressionLib::SIMDintersection(
                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                        g->node_off[u + 1] - g->node_off[u],
                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                        g->node_off[v + 1] - g->node_off[v],
                        my_ptr);
#elif defined(LEMIRE_AVX2)
                g->common_node_num[i] += SIMDCompressionLib::SIMDintersection_avx2(
                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                        g->node_off[u + 1] - g->node_off[u],
                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                        g->node_off[v + 1] - g->node_off[v],
                        my_ptr);
#elif defined(LEMIRE_HIGHLY_SCALABLE)
                g->common_node_num[i] += SIMDCompressionLib::lemire_highlyscalable_intersect_SIMD(
                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                        g->node_off[u + 1] - g->node_off[u],
                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                        g->node_off[v + 1] - g->node_off[v],
                        my_ptr);
#elif defined(HAN_QFILTER)
                g->common_node_num[i] += intersect_qfilter_uint_b4_v2(
                        g->node_off[u] + g->edge_dst,
                        g->node_off[u + 1] - g->node_off[u],
                        g->node_off[v] + g->edge_dst,
                        g->node_off[v + 1] - g->node_off[v],
                        my_ptr_han);
#elif defined(HAN_BMISS)
                g->common_node_num[i] += intersect_bmiss_uint_b4(
                        g->node_off[u] + g->edge_dst,
                        g->node_off[u + 1] - g->node_off[u],
                        g->node_off[v] + g->edge_dst,
                        g->node_off[v + 1] - g->node_off[v],
                        my_ptr_han);
#elif defined(HAN_BMISS_STTNI)
                g->common_node_num[i] += intersect_bmiss_uint_sttni_b8(
                        g->node_off[u] + g->edge_dst,
                        g->node_off[u + 1] - g->node_off[u],
                        g->node_off[v] + g->edge_dst,
                        g->node_off[v + 1] - g->node_off[v],
                        my_ptr_han);
#elif defined(HAN_HIER)
                g->common_node_num[i] += intersect_hierainter_uint_sttni(
                        g->node_off[u] + g->edge_dst,
                        g->node_off[u + 1] - g->node_off[u],
                        g->node_off[v] + g->edge_dst,
                        g->node_off[v + 1] - g->node_off[v],
                        my_ptr_han);
#elif defined(GALLOPING_SINGLE)
                g->common_node_num[i] += ComputeCNGallopingSingleDir(g, u, v);
#elif defined(GALLOPING_DOUBLE)
                g->common_node_num[i] += ComputeCNGallopingDoubleDir(g, u, v);
#elif defined(NAIVE_HYBRID)
                g->common_node_num[i] += ComputeCNHybrid(g, u, v);
#elif defined(SSE_MERGE)
                g->common_node_num[i] += ComputeCNSSE4(g, u, v);
#elif defined(SSE_HYBRID)
                g->common_node_num[i] += ComputeCNSSEHybrid(g, u, v);
#elif defined(SSE_PIVOT)
                g->common_node_num[i] += ComputeCNPivotSSE4(g, u, v);
#elif defined(SSE_BIN)
                g->common_node_num[i] += ComputeCNLowerBoundSSE(g, u, v);
#elif defined(AVX2)
                g->common_node_num[i] += ComputeCNAVX2(g, u, v);
#elif defined(AVX2_POPCNT)
                g->common_node_num[i] += ComputeCNAVX2PopCnt(g, u, v);
#elif defined(PIVOT_AVX2)
                g->common_node_num[i] += ComputeCNPivotAVX2(g, u, v);
#elif defined(AVX2_GALLOPING_SINGLE)
                g->common_node_num[i] += ComputeCNGallopingSingleDirAVX2(g, u, v);
#elif defined(AVX2_GALLOPING_DOUBLE)
                g->common_node_num[i] += ComputeCNGallopingDoubleDirAVX2(g, u, v);
#elif defined(CPU_ADVANCED)
                g->common_node_num[i] += ComputeCNHybridAVX2(g, u, v);
#elif defined(GALLOPING_AVX512)
                g->common_node_num[i] += ComputeCNGallopingSingleDirAVX512(g, u, v);
#elif defined(GALLOPING_DOUBLE_AVX512)
                g->common_node_num[i] += ComputeCNGallopingDoubleDirAVX512(g, u, v);
#elif defined(PIVOT_KNL)
                g->common_node_num[i] += ComputeCNPivotAVX512(g, u, v);
#elif defined(AVX512_POPCNT)
                g->common_node_num[i] += ComputeCNAVX512PopCnt(g, u, v);
#elif defined(AVX512)
                g->common_node_num[i] += ComputeCNAVX512(g, u, v);
#elif defined(ADVANCED)
                g->common_node_num[i] += ComputeCNHybridAVX512(g, u, v);
#elif defined(HYBRID_NO_MERGE)
                g->common_node_num[i] += ComputeCNHybridAVX512NoBlkMerge(g, u, v);
#elif defined(HYBRID_NO_GALLOPING)
                g->common_node_num[i] += ComputeCNHybridAVX512NoGalloping(g, u, v);
#else
                g->common_node_num[i] += ComputeCNNaiveStdMerge(g, u, v);
#endif
#ifdef STAT
                auto clk_end = high_resolution_clock::now();
                long time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);

                clk_beg = high_resolution_clock::now();
#endif
                // Symmetrically Assign.
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                u)] = g->common_node_num[i];
#ifdef STAT
                clk_end = high_resolution_clock::now();
                time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                lb_bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);
#endif
            }
        }

#ifdef STAT
#pragma  omp critical
        {
            for (int i = 0; i < 10; i++) {
                global_bins[i].MergeStat(bins[i]);
                global_lb_bins[i].MergeStat(lb_bins[i]);
            }
        }
#endif

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("All-Edge Computation Cost (Sorted Array): %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            log_info("Finish Sorted Array Computation.");
        }
    }
}

void SCAN_XP::CheckCoreEmptyHeaded(Graph *g) {
#ifdef EMPTY_HEADED
    log_info("Before EmptyHeaded... %s KB", FormatWithCommas(getValue()).c_str());
#ifdef EH_LAYOUT_UINT
    using Layout= uinteger;
#else
    using Layout= hybrid;
#endif
    vector<const Set<Layout> *> eh_sets(g->nodemax);

    auto clk_beg = high_resolution_clock::now();
    size_t max_size = 0;
    size_t type0 = 0;
    size_t type1 = 0;
    size_t max_mem_consumption = 0;
#pragma omp parallel for schedule(dynamic, 100), reduction(max: max_size), reduction(+:type0), reduction(+:type1)
    for (int i = 0; i < g->nodemax; i++) {
        eh_sets[i] = NewSet<Layout>(g, i);
        max_size = max(max_size, eh_sets[i]->number_of_bytes + sizeof(Set<Layout>));
        if (eh_sets[i]->type == 0) {
            type0++;
        } else {
            type1++;
        }
        max_mem_consumption = max(max_mem_consumption, eh_sets[i]->number_of_bytes + sizeof(Set<Layout>));
    }
    log_info("type0 (RANGE_BITSET): %s, type1 (UINTEGER): %s, Max Mem: %s KB", FormatWithCommas(type0).c_str(),
             FormatWithCommas(type1).c_str(), FormatWithCommas(max_mem_consumption).c_str());
    auto clk_end = high_resolution_clock::now();
    auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
    log_info("PreProcess-EmptyHeaded Cost: %.9lf s, %s KB", time / pow(10, 9),
             FormatWithCommas(getValue()).c_str());

    clk_beg = high_resolution_clock::now();
#pragma omp parallel num_threads(thread_num_)
    {
#ifdef STAT
        vector<InterSectStat> bins(NUM_OF_BINS);
        vector<LowerBoundStat> lb_bins(NUM_OF_BINS);
#endif
        Set<Layout> *res = (Set<Layout> *) malloc(max_size);
#pragma omp for schedule(dynamic, 6000)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local auto u = 0;
            u = FindSrc(g, u, i);
            auto v = g->edge_dst[i];
            if (u < v) {
#ifdef STAT
                auto clk_beg = high_resolution_clock::now();
#endif
                Set<Layout> *ptr = ops::set_intersect(res, eh_sets[u], eh_sets[v]);
                g->common_node_num[i] += ptr->cardinality;

#ifdef STAT
                auto clk_end = high_resolution_clock::now();
                long time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);

                clk_beg = high_resolution_clock::now();
#endif
                // Symmetrically Assign.
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                u)] = g->common_node_num[i];
#ifdef STAT
                clk_end = high_resolution_clock::now();
                time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                lb_bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);
#endif
            }
        }

#ifdef STAT
#pragma  omp critical
        {
            for (int i = 0; i < 10; i++) {
                global_bins[i].MergeStat(bins[i]);
                global_lb_bins[i].MergeStat(lb_bins[i]);
            }
        }
#endif
#pragma omp single
        {
            clk_end = high_resolution_clock::now();
            time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("All-Edge Computation Cost (EmptyHeaded): %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            log_info("Finish EmptyHead Computation.");
        }
    }

#endif
}

void SCAN_XP::CheckCoreBSR(Graph *g) {
#ifdef BSR
    log_info("Before BSRSet... %s KB", FormatWithCommas(getValue()).c_str());
    vector<BSRSet> bsrs(g->nodemax);

    auto clk_beg = high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < g->nodemax; i++) {
        static thread_local int *tmp_base = new int[MAX_DEGREE];
        static thread_local int *tmp_state = new int[MAX_DEGREE];

        auto degree = g->node_off[i + 1] - g->node_off[i];
        auto tmp_size = offline_uint_trans_bsr(g->edge_dst + g->node_off[i], degree, tmp_base, tmp_state);

        bsrs[i].base_ = new int[degree];
        bsrs[i].states_ = new int[degree];
        bsrs[i].size_ = tmp_size;
        memcpy(bsrs[i].base_, tmp_base, static_cast<size_t>(tmp_size) * sizeof(int));
        memcpy(bsrs[i].states_, tmp_state, static_cast<size_t>(tmp_size) * sizeof(int));
    }
    auto clk_end = high_resolution_clock::now();
    auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
    log_info("PreProcess-BSRSet Cost: %.9lf s, %s KB", time / pow(10, 9), FormatWithCommas(getValue()).c_str());

    clk_beg = high_resolution_clock::now();
#pragma omp parallel num_threads(thread_num_)
    {
#ifdef STAT
        vector<InterSectStat> bins(NUM_OF_BINS);
        vector<LowerBoundStat> lb_bins(NUM_OF_BINS);
#endif
#pragma omp for schedule(dynamic, 6000)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local auto u = 0;
            u = FindSrc(g, u, i);
            static thread_local int *tmp_base = new int[MAX_DEGREE];
            static thread_local int *tmp_state = new int[MAX_DEGREE];
            auto v = g->edge_dst[i];
            if (u < v) {
#ifdef STAT
                auto clk_beg = high_resolution_clock::now();
#endif

#if defined(SCALAR_MERGE_BSR)
                auto size_c = intersect_scalarmerge_bsr(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                                                        bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,
                                                        tmp_base, tmp_state);
#elif defined(SCALAR_GALLOPING_BSR)
                auto size_c = intersect_scalargalloping_bsr(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                        bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,tmp_base, tmp_state);
#elif defined(SIMD_GALLOPING_BSR)
                auto size_c = intersect_simdgalloping_bsr(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                                                          bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,
                                                          tmp_base, tmp_state);
#elif defined(SHUFFLE_BSR)
                auto size_c = intersect_shuffle_bsr_b4(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                                                       bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,
                                                       tmp_base, tmp_state);
#else
                auto size_c = intersect_qfilter_bsr_b4_v2(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                                                          bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,
                                                          tmp_base, tmp_state);
#endif

                // popcnt here
                auto cnt = popcnt(tmp_state, size_c * sizeof(int));
                g->common_node_num[i] += cnt;
#ifdef STAT
                auto clk_end = high_resolution_clock::now();
                long time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);

                clk_beg = high_resolution_clock::now();
#endif
                // Symmetrically Assign.
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                u)] = g->common_node_num[i];
#ifdef STAT
                clk_end = high_resolution_clock::now();
                time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                lb_bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);
#endif
            }
        }

#ifdef STAT
#pragma  omp critical
        {
            for (int i = 0; i < 10; i++) {
                global_bins[i].MergeStat(bins[i]);
                global_lb_bins[i].MergeStat(lb_bins[i]);
            }
        }
#endif

#pragma omp single
        {
            clk_end = high_resolution_clock::now();
            time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("All-Edge Computation Cost (BSR): %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            log_info("Finish BSR Computation.");
        }
    }
#endif
}


void SCAN_XP::CheckCoreRoaring(Graph *g) {
#ifdef ROARING

    log_info("Before Roaring... %s KB", FormatWithCommas(getValue()).c_str());

    vector<Roaring> roarings(g->nodemax);

    long cnt = 0;
    auto clk_beg = high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100), reduction(+:cnt)
    for (int i = 0; i < g->nodemax; i++) {
        roarings[i] = Roaring(g->node_off[i + 1] - g->node_off[i],
                              reinterpret_cast<const uint32_t *>(g->node_off[i] + g->edge_dst));
        cnt += roarings[i].runOptimize() ? 1 : 0;
        roarings[i].shrinkToFit();
    }
    auto clk_end = high_resolution_clock::now();
    auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
    log_info("PreProcess-Roaring Cost: %.9lf s, %s KB", time / pow(10, 9),
             FormatWithCommas(getValue()).c_str());

    log_info("Run: %s", FormatWithCommas(cnt).c_str());

    clk_beg = high_resolution_clock::now();
#pragma omp parallel num_threads(thread_num_)
    {
#ifdef STAT
        vector<InterSectStat> bins(NUM_OF_BINS);
        vector<LowerBoundStat> lb_bins(NUM_OF_BINS);
#endif
#pragma omp for schedule(dynamic, 6000)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local auto u = 0;
            u = FindSrc(g, u, i);

            auto v = g->edge_dst[i];
            if (u < v) {
#ifdef STAT
                auto clk_beg = high_resolution_clock::now();
#endif
                g->common_node_num[i] += roaring_bitmap_and_cardinality(&roarings[u].roaring,
                                                                        &roarings[v].roaring);

#ifdef STAT
                auto clk_end = high_resolution_clock::now();
                long time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);

                clk_beg = high_resolution_clock::now();
#endif
                // Symmetrically Assign.
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                u)] = g->common_node_num[i];
#ifdef STAT
                clk_end = high_resolution_clock::now();
                time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
                lb_bins[min<int>(floor(log10(time)), 10 - 1)].AddStat(time, g, i, u, v);
#endif
            }
        }

#ifdef STAT
#pragma  omp critical
        {
            for (int i = 0; i < 10; i++) {
                global_bins[i].MergeStat(bins[i]);
                global_lb_bins[i].MergeStat(lb_bins[i]);
            }
        }
#endif
#pragma omp single
        {
            clk_end = high_resolution_clock::now();
            time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("All-Edge Computation Cost (Roar): %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            log_info("Finish Roaring Computation.");
        }
    }
#endif

}

#if defined(REORDERING)

void SCAN_XP::Reorder(Graph &g) {
    auto start = high_resolution_clock::now();

    new_vid_dict = vector<int>(g.degree.size());
    for (auto i = 0; i < g.degree.size(); i++) {
        new_vid_dict[old_vid_dict[i]] = i;
    }
    // new-deg
    vector<int> new_deg(g.nodemax);
    for (auto new_id = 0; new_id < g.nodemax; new_id++) {
        new_deg[new_id] = g.degree[old_vid_dict[new_id]];
        assert(new_deg[new_id] >= 1);
    }

    // verify permutation
    for (auto i = 0; i < std::min<int32_t>(5, static_cast<int32_t>(new_vid_dict.size())); i++) {
        log_info("old->new %d -> %d", i, new_vid_dict[i]);
    }
    vector<int> verify_map(new_vid_dict.size(), 0);
    int cnt = 0;
//#pragma omp parallel num_threads(thread_num_)
#pragma omp parallel
    {
#pragma omp for reduction(+:cnt)
        for (auto i = 0; i < new_vid_dict.size(); i++) {
            if (verify_map[new_vid_dict[i]] == 0) {
                cnt++;
                verify_map[new_vid_dict[i]] = 1;
            } else {
                assert(false);
            }
        }
#pragma omp single
        log_info("%d, %d", cnt, new_vid_dict.size());
        assert(cnt == new_vid_dict.size());
    }
    // 1st CSR: new_off, new_neighbors
    vector<uint32_t> new_off(g.nodemax + 1);
    new_off[0] = 0;
    assert(new_off.size() == g.nodemax + 1);
    for (auto i = 0u; i < g.nodemax; i++) { new_off[i + 1] = new_off[i] + new_deg[i] - 1; }
    log_info("%zu", new_off[g.nodemax]);
    assert(new_off[g.nodemax] == g.edgemax);

    vector<int> new_neighbors(g.edgemax);
    auto end = high_resolution_clock::now();
    log_info("init ordering structures time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

    // 2nd Parallel Transform
//#pragma omp parallel num_threads(thread_num_)
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 100)
        for (auto i = 0; i < g.nodemax; i++) {
            auto origin_i = old_vid_dict[i];
            // transform
            auto cur_idx = new_off[i];
            for (auto my_old_off = g.node_off[origin_i]; my_old_off < g.node_off[origin_i + 1]; my_old_off++) {
                if (cur_idx > g.edgemax) {
                    log_info("%d, i: %d", cur_idx, i);
                }
                assert(cur_idx <= g.edgemax);
                assert(my_old_off <= g.edgemax);
                assert(g.edge_dst[my_old_off] < g.nodemax);
                new_neighbors[cur_idx] = new_vid_dict[g.edge_dst[my_old_off]];
                cur_idx++;
            }
            // sort the local ranges
            sort(begin(new_neighbors) + new_off[i], begin(new_neighbors) + new_off[i + 1]);
        }
    }
    auto end2 = high_resolution_clock::now();
    log_info("parallel transform and sort: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);

    memcpy(g.edge_dst, &new_neighbors.front(), g.edgemax * sizeof(int32_t));
    memcpy(g.node_off, &new_off.front(), (g.nodemax + 1) * sizeof(uint32_t));
    g.degree = new_deg;
}

// extern functions.
void PKC(graph_t *g, int *deg, int num_of_threads);

void SCAN_XP::ReorderKCoreDegeneracy(Graph &yche_graph) {
    graph_t g;
    g.adj = yche_graph.edge_dst;
    g.num_edges = yche_graph.node_off;
    g.n = yche_graph.nodemax;
    g.m = yche_graph.edgemax;

    auto clk_beg = high_resolution_clock::now();

    vector<int> core_val(g.n, 0);
    PKC(&g, &core_val.front(), thread_num_);
    core_val_histogram(g.n, core_val);
    old_vid_dict = vector<int>(g.n);
    for (auto i = 0; i < old_vid_dict.size(); i++) { old_vid_dict[i] = i; }
#ifdef DUAL_DEG_CORE
    sort(begin(old_vid_dict), end(old_vid_dict),
         [&core_val, &yche_graph](int l, int r) -> bool {
             return static_cast<int64_t>(yche_graph.degree[l]) * core_val[l] >
                    static_cast<int64_t>(yche_graph.degree[r]) * core_val[r];
         });
#else
    sort(begin(old_vid_dict), end(old_vid_dict),
         [&core_val](int l, int r) -> bool {
             return core_val[l] > core_val[r];
         });
#endif
    auto clk_end = high_resolution_clock::now();
    auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
    log_info("PKC (k-core) time:  %.9lf s", time / pow(10, 9));

    Reorder(yche_graph);
}

void SCAN_XP::ReorderRandom(Graph &g) {
    auto clk_beg = high_resolution_clock::now();

    old_vid_dict = vector<int>(g.degree.size());
    for (auto i = 0; i < old_vid_dict.size(); i++) { old_vid_dict[i] = i; }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::shuffle(begin(old_vid_dict), end(old_vid_dict), gen);

    auto clk_end = high_resolution_clock::now();
    auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
    log_info("Random shuffling time:  %.9lf s", time / pow(10, 9));

    Reorder(g);
}


void SCAN_XP::ReorderDegDescending(Graph &g) {
    auto clk_beg = high_resolution_clock::now();

    old_vid_dict = vector<int>(g.degree.size());
    for (auto i = 0; i < old_vid_dict.size(); i++) { old_vid_dict[i] = i; }

#ifdef TBB
    log_info("Use TBB parallel sort");
    tbb::parallel_sort(begin(old_vid_dict), end(old_vid_dict),
                       [&g](int l, int r) -> bool { return g.degree[l] > g.degree[r]; });
#else
    log_info("Use parallel sort (parasort)");
    parasort(old_vid_dict.size(), &old_vid_dict.front(),
                 [&g](int l, int r) -> bool { return g.degree[l] > g.degree[r]; }, thread_num_);
//    sort(begin(old_vid_dict), end(old_vid_dict),
//         [&g](int l, int r) -> bool { return g.degree[l] > g.degree[r]; });
#endif
    auto clk_end = high_resolution_clock::now();
    auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
    log_info("Deg-descending time:  %.9lf s", time / pow(10, 9));

    Reorder(g);
}

#endif

void SCAN_XP::CheckCoreCompactForward(Graph *g) {
#ifdef COMPACT_FORWARD
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g->nodemax + 1));

    auto max_d = 0;
    uint64_t triangle_cnt = 0;
    int64_t max_support = 0;
    uint64_t half_dup_tri_cnt = 0;
    uint64_t emtpy_res_cnt = 0;
    uint64_t mismatch = 0;
    uint64_t reduced_bin_search_cnt = 0;

#if defined(BITMAP) && defined(PACK)
    int packed_num = 0;
    vector<vector<int>> partition_id_lst(g->nodemax);
    vector<vector<word_type>> bitmap_in_partition_lst(g->nodemax);
#endif
    auto clk_beg = high_resolution_clock::now();

#pragma omp parallel num_threads(thread_num_)
    {
        auto *my_ptr = new uint32_t[1024 * 1024 * 2];
        auto *my_ptr_han = new int[1024 * 1024 * 2];
#ifdef BITMAP
        auto bool_arr = BoolArray<word_type>(g->nodemax);
#endif

        auto start = high_resolution_clock::now();

#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g->nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g->edge_dst + g->node_off[u], g->edge_dst + g->node_off[u + 1], u) - g->edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g->node_off[u]);
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_debug("Finish node_off_end (DODG), max d: %d, time: %.6lf s", max_d, time / pow(10, 9));
            clk_beg = high_resolution_clock::now();
        }

#if defined(BITMAP) && defined(PACK)
        // Pre-Process: Indexing Words.
#pragma omp for schedule(dynamic, 100) reduction(+:packed_num)
        for (auto u = 0; u < g->nodemax; u++) {
            PackVertex(g, partition_id_lst, bitmap_in_partition_lst, u, packed_num);
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("Packed#: %s", FormatWithCommas(packed_num).c_str());
            log_info("PreProcess-BitmapPack Cost: %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            clk_beg = high_resolution_clock::now();
        }
#endif

        // Listing on the Degree-Ordered-Directed-Graph (View)
#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt) reduction(+:emtpy_res_cnt) reduction(+:mismatch)
        for (auto u = 0u; u < g->nodemax; u++) {
#ifdef BITMAP
            static thread_local auto last_u = -1;
            if (last_u != u) {
                if (last_u != -1) {
                    for (auto offset = g->node_off[last_u]; offset < node_off_end[last_u + 1]; offset++) {
                        auto v = g->edge_dst[offset];
                        bool_arr.setWord(v / word_in_bits, 0);
                    }
                }
                for (auto offset = g->node_off[u]; offset < node_off_end[u + 1]; offset++) {
                    auto v = g->edge_dst[offset];

                    bool_arr.set(v);
                }
                last_u = u;
            }
#endif
            for (auto edge_idx = g->node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g->edge_dst[edge_idx];
#ifdef EXPREIMENTAL
                g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1], u)] = edge_idx;
#endif

#ifdef LEMIRE_AVX2
                auto size = SIMDCompressionLib::SIMDintersection_avx2(
                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                        node_off_end[u + 1] - g->node_off[u],
                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                        node_off_end[v + 1] - g->node_off[v],
                        my_ptr);
#elif defined(TETZANK_AVX2_ASM)
                auto size =  intersect_vector_avx2_asm(
                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                        node_off_end[u + 1] - g->node_off[u],
                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                        node_off_end[v + 1] - g->node_off[v]
                        , my_ptr);
#elif defined(TETZANK_AVX2)
                auto size =  intersect_vector_avx2(
                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                        node_off_end[u + 1] - g->node_off[u],
                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                        node_off_end[v + 1] - g->node_off[v]
                        , my_ptr);
#elif defined(HAN_QFILTER)
                auto size = intersect_qfilter_uint_b4_v2(
                        g->node_off[u] + g->edge_dst,
                        node_off_end[u + 1] - g->node_off[u],
                        g->node_off[v] + g->edge_dst,
                        node_off_end[v + 1] - g->node_off[v],
                        my_ptr_han);
#elif defined(HAN_BMISS)
                auto size =  intersect_bmiss_uint_b4(
                        g->node_off[u] + g->edge_dst,
                        node_off_end[u + 1] - g->node_off[u],
                        g->node_off[v] + g->edge_dst,
                        node_off_end[v + 1] - g->node_off[v],
                        my_ptr_han);
#elif defined(HAN_BMISS_STTNI)
                auto size =  intersect_bmiss_uint_sttni_b8(
                        g->node_off[u] + g->edge_dst,
                        node_off_end[u + 1] - g->node_off[u],
                        g->node_off[v] + g->edge_dst,
                        node_off_end[v + 1] - g->node_off[v],
                        my_ptr_han);
#elif defined(BITMAP)
                auto size = 0;
#if defined(PACK)
                if (!partition_id_lst[v].empty()) {
                    for (auto wi = 0; wi < partition_id_lst[v].size(); wi++) {
                        auto res = bool_arr.getWord(partition_id_lst[v][wi]) & bitmap_in_partition_lst[v][wi];
                        if (res != 0) {
                            uint32_t begin_pos = partition_id_lst[v][wi] * word_in_bits;
                            for (uint32_t i = 0; i < word_in_bits; i++) {
                                if (res & (static_cast<word_type>(1) << i)) {
                                    my_ptr[size++] = begin_pos + i;
                                }
                            }
                        }
                    }
                } else {
                    for (auto off = g->node_off[v]; off < node_off_end[v + 1]; off++) {
                        auto w = g->edge_dst[off];
                        if (bool_arr.get(w))
                            my_ptr[size++] = w;
                    }
                }
#else
                for (auto off = g->node_off[v]; off < node_off_end[v + 1]; off++) {
                    auto w = g->edge_dst[off];
                    if (bool_arr.get(w))
                        my_ptr[size++] = w;
                    else
                        mismatch++;
                }
#endif
#else
                auto size = SIMDCompressionLib::SIMDintersection(
                        reinterpret_cast<const uint32_t *>(g->node_off[u] + g->edge_dst),
                        node_off_end[u + 1] - g->node_off[u],
                        reinterpret_cast<const uint32_t *>(g->node_off[v] + g->edge_dst),
                        node_off_end[v + 1] - g->node_off[v], my_ptr);
#endif
                if (size != 0) {
                    triangle_cnt += size;
                    // atomic updates:
                    // 1st: p->q
                    atomic_add(&g->common_node_num[edge_idx], static_cast<int>(size));

                    // 2nd: iterate p->r
                    auto prev_it_u = g->node_off[u];
                    for (auto iter_res = 0; iter_res < size; iter_res++) {

#if defined(HAN_QFILTER) || defined(HAN_BMISS) || defined(HAN_BMISS_STTNI)
                        prev_it_u = LinearSearch(g->edge_dst, prev_it_u, node_off_end[u + 1],
                                                 my_ptr_han[iter_res]);
#else
                        prev_it_u = LinearSearch(g->edge_dst, prev_it_u, node_off_end[u + 1],
                                                 my_ptr[iter_res]);
#endif
                        atomic_add(g->common_node_num + prev_it_u, 1);
                    }

                    // 3rd: iterate q->r
                    auto prev_it_v = g->node_off[v];
                    for (auto iter_res = 0; iter_res < size; iter_res++) {
#if defined(HAN_QFILTER) || defined(HAN_BMISS) || defined(HAN_BMISS_STTNI)
                        prev_it_v = LinearSearch(g->edge_dst, prev_it_v, node_off_end[v + 1],
                                                 my_ptr_han[iter_res]);
#else
                        prev_it_v = LinearSearch(g->edge_dst, prev_it_v, node_off_end[v + 1],
                                                 my_ptr[iter_res]);
#endif
                        atomic_add(g->common_node_num + prev_it_v, 1);
                    }
                } else {
                    emtpy_res_cnt++;
                }
            }
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            auto total_workload = triangle_cnt * 3 + mismatch;
            log_debug("MisMatch: %s, Total: %s, Ratio: %.6lf",
                      FormatWithCommas(mismatch).c_str(), FormatWithCommas(total_workload).c_str(),
                      static_cast<double>(mismatch) / total_workload);
            log_debug("Forward cost: %.3lf s, Mem Usage: %s KB",
                      duration_cast<milliseconds>(end - start).count() / 1000.0,
                      FormatWithCommas(getValue()).c_str());
            log_debug("Triangle Cnt: %s, EmptyCnt/All %s / %s, Ratio: %.6lf", FormatWithCommas(triangle_cnt).c_str(),
                      FormatWithCommas(emtpy_res_cnt).c_str(),
                      FormatWithCommas(g->edgemax / 2).c_str(),
                      static_cast<double>(emtpy_res_cnt) / (g->edgemax / 2));
        }
#pragma omp single
        log_info("finish init listing");
        // Binary Search Or LowerBound
#pragma omp for schedule(dynamic, 1000) reduction(max:max_support) reduction(+:half_dup_tri_cnt) reduction(+:reduced_bin_search_cnt)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local int u = 0;
            u = FindSrc(g, u, i);
            auto v = g->edge_dst[i];
#ifdef EXPREIMENTAL
            if (u < v) {
                auto rev_i = (uint32_t) g->common_node_num[i];
                g->common_node_num[i] = g->common_node_num[rev_i];

                auto cnt = (g->common_node_num[i] - 2);
                max_support = max<int64_t>(max_support, cnt);
                half_dup_tri_cnt += cnt;
            }
#else
            if (u > v) {
                if (g->common_node_num[i] > 2) {
#ifdef PREFETCH_OPT
                    auto my_off =
                            branchfree_search(g->edge_dst + g->node_off[v], g->node_off[v + 1] - g->node_off[v], u) +
                            g->node_off[v];
                    g->common_node_num[my_off] = g->common_node_num[i];
#else
                    g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                    u)] = g->common_node_num[i];
#endif
                } else {
                    reduced_bin_search_cnt++;
                }
                auto cnt = (g->common_node_num[i] - 2);
                max_support = max<int64_t>(max_support, cnt);
                half_dup_tri_cnt += cnt;
            }
#endif
        }
        delete[]my_ptr;

        auto end2 = high_resolution_clock::now();
#pragma omp single
        {
            log_debug("LowerBound cost: %.3lf s, Mem Usage: %s KB",
                      duration_cast<milliseconds>(end2 - end).count() / 1000.0,
                      FormatWithCommas(getValue()).c_str());
            log_debug("Reduced BinS Cnt： %s / %s, Ratio: %.6lf", FormatWithCommas(reduced_bin_search_cnt).c_str(),
                      FormatWithCommas(g->edgemax / 2).c_str(),
                      static_cast<double>(reduced_bin_search_cnt) / (g->edgemax / 2));
            log_info("Max support: %s; total: %s", FormatWithCommas(max_support).c_str(),
                     FormatWithCommas(half_dup_tri_cnt).c_str());
            assert(half_dup_tri_cnt == 3 * triangle_cnt);

            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_debug("All-Edge Computation Cost (CompactForward): %.9lf s, %s KB", time / pow(10, 9),
                      FormatWithCommas(getValue()).c_str());
            log_info("Finish CompactForward Computation.");
        }
    };

    free(node_off_end);
#endif
}

void SCAN_XP::CheckCoreHybridCFNormal(Graph *g) {
    auto clk_beg = high_resolution_clock::now();

    int packed_num = 0;
    auto max_d = 0;

    uint64_t check_word_num = 0;
    uint64_t dense_tc_num = 0;
    uint64_t sparse_tc_num = 0;

    uint64_t triangle_cnt = 0;
    int64_t max_support = 0;
    uint64_t half_dup_tri_cnt = 0;
    uint64_t emtpy_res_cnt = 0;

    vector<vector<int>> partition_id_lst(g->nodemax);
    vector<vector<word_type>> bitmap_in_partition_lst(g->nodemax);
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g->nodemax + 1));
    {
        auto clk_end = high_resolution_clock::now();
        auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
        log_info("MemAlloc Cost: %.9lf s, %s KB", time / pow(10, 9),
                 FormatWithCommas(getValue()).c_str());
    }

#pragma omp parallel num_threads(thread_num_)
    {
        auto bool_arr = BoolArray<word_type>(g->nodemax);
        auto *my_ptr = new uint32_t[1024 * 1024 * 2];

        // Pre-Process: Indexing Words.
#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g->nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g->edge_dst + g->node_off[u], g->edge_dst + g->node_off[u + 1], u) - g->edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g->node_off[u]);
        }
#pragma omp for schedule(dynamic, 100) reduction(+:packed_num)
        for (auto u = 0; u < g->nodemax; u++) {
            PackVertex(g, partition_id_lst, bitmap_in_partition_lst, u, packed_num);
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("Packed#: %s, MaxDeg: %s", FormatWithCommas(packed_num).c_str(), FormatWithCommas(max_d).c_str());
            log_info("PreProcess-BitmapAdvanced Cost: %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            clk_beg = high_resolution_clock::now();
        }

        // CF: V2-V2
#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt) reduction(+:emtpy_res_cnt)
        for (auto u = 0u; u < g->nodemax; u++) {
            for (auto offset = g->node_off[u]; offset < node_off_end[u + 1]; offset++) {
                auto v = g->edge_dst[offset];
                bool_arr.set(v);
            }
            for (auto edge_idx = g->node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g->edge_dst[edge_idx];
                if (!partition_id_lst[u].empty() && !partition_id_lst[v].empty()) {
                    continue;
                }
                auto size = 0;

                for (auto off = g->node_off[v]; off < node_off_end[v + 1]; off++) {
                    auto w = g->edge_dst[off];
                    if (bool_arr.get(w))
                        my_ptr[size++] = w;
                }

                if (size != 0) {
                    triangle_cnt += size;
                    // atomic updates:
                    // 1st: p->q
                    if (partition_id_lst[u].empty() && partition_id_lst[v].empty())
                        atomic_add(&g->common_node_num[edge_idx], size);

                    // 2nd: iterate p->r
                    auto prev_it_u = g->node_off[u];
                    for (auto iter_res = 0; iter_res < size; iter_res++) {
                        prev_it_u = LinearSearch(g->edge_dst, prev_it_u, node_off_end[u + 1],
                                                 my_ptr[iter_res]);
                        if (partition_id_lst[u].empty() && partition_id_lst[g->edge_dst[prev_it_u]].empty())
                            atomic_add(g->common_node_num + prev_it_u, 1);
                    }

                    // 3rd: iterate q->r
                    auto prev_it_v = g->node_off[v];
                    for (auto iter_res = 0; iter_res < size; iter_res++) {
                        prev_it_v = LinearSearch(g->edge_dst, prev_it_v, node_off_end[v + 1],
                                                 my_ptr[iter_res]);
                        if (partition_id_lst[v].empty() && partition_id_lst[g->edge_dst[prev_it_v]].empty())
                            atomic_add(g->common_node_num + prev_it_v, 1);
                    }
                } else {
                    emtpy_res_cnt++;
                }
            }
            for (auto offset = g->node_off[u]; offset < node_off_end[u + 1]; offset++) {
                auto v = g->edge_dst[offset];
                bool_arr.setWord(v / word_in_bits, 0);
            }
        }

#pragma omp single
        {
            log_debug("TC (CF): %s", FormatWithCommas(triangle_cnt).c_str());
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_debug("Finish CF: V2-V2: %.9lf s, %s KB", time / pow(10, 9),
                      FormatWithCommas(getValue()).c_str());
        }

#pragma omp for schedule(dynamic, 1000) reduction(max:max_support) reduction(+:half_dup_tri_cnt)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local int u = 0;
            u = FindSrc(g, u, i);
            auto v = g->edge_dst[i];
            if (u > v && partition_id_lst[u].empty() && partition_id_lst[v].empty()) {
                if (g->common_node_num[i] > 2) {
                    g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                    u)] = g->common_node_num[i];
                }
                auto cnt = (g->common_node_num[i] - 2);
                max_support = max<int64_t>(max_support, cnt);
                half_dup_tri_cnt += cnt;
            }
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_debug("Finish LB for V2-V2: %.9lf s, %s KB", time / pow(10, 9),
                      FormatWithCommas(getValue()).c_str());
        }

        // BMP: V1-V1, V1-V2
#pragma omp for schedule(dynamic, 6000) reduction(+:dense_tc_num) reduction(+:sparse_tc_num) \
reduction(+:check_word_num)
        for (auto i = 0u; i < g->edgemax; i++) {
            static thread_local auto u = 0;
            u = FindSrc(g, u, i);
            static thread_local auto last_u = -1;
            if (last_u != u) {
                if (last_u != -1) {
                    for (auto offset = g->node_off[last_u]; offset < g->node_off[last_u + 1]; offset++) {
                        auto v = g->edge_dst[offset];
                        bool_arr.setWord(v / word_in_bits, 0);
                    }
                }
                for (auto offset = g->node_off[u]; offset < g->node_off[u + 1]; offset++) {
                    auto v = g->edge_dst[offset];
                    bool_arr.set(v);
                }
                last_u = u;
            }

            auto v = g->edge_dst[i];
            if (g->degree[u] > g->degree[v] || ((g->degree[u] == g->degree[v]) && (u < v))) {
                if (partition_id_lst[u].empty() && partition_id_lst[v].empty()) {
                    continue;
                }
                auto local_cnt = 0;
                if (!partition_id_lst[v].empty()) {
                    for (auto wi = 0; wi < partition_id_lst[v].size(); wi++) {
                        auto res = bool_arr.getWord(partition_id_lst[v][wi]) & bitmap_in_partition_lst[v][wi];
                        local_cnt += popcnt(&res, sizeof(word_type));
                    }
                    dense_tc_num += local_cnt;
                    check_word_num += partition_id_lst[v].size();
                } else {
                    for (auto off = g->node_off[v]; off < g->node_off[v + 1]; off++) {
                        auto w = g->edge_dst[off];
                        if (bool_arr.get(w))
                            local_cnt++;
                    }
                    sparse_tc_num += local_cnt;
                }

                // Symmetrically Assign.
                if (local_cnt > 0) {
                    g->common_node_num[i] += local_cnt;
                    g->common_node_num[BinarySearch(g->edge_dst, g->node_off[v], g->node_off[v + 1],
                                                    u)] = g->common_node_num[i];
                }
            }
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("TC (OverEstimation): %s",
                     FormatWithCommas(triangle_cnt + (dense_tc_num + sparse_tc_num) / 3).c_str());
            log_debug("Dense Tc: %s, Sparse Tc: %s, Total: %s, Tri: %s ",
                      FormatWithCommas(dense_tc_num).c_str(),
                      FormatWithCommas(sparse_tc_num).c_str(),
                      FormatWithCommas(dense_tc_num + sparse_tc_num).c_str(),
                      FormatWithCommas((dense_tc_num + sparse_tc_num) / 3).c_str());
            log_debug("Check Word Num: %s (Bits: %s), Valid/Check: %.6lf", FormatWithCommas(check_word_num).c_str(),
                      FormatWithCommas(check_word_num * word_in_bits).c_str(),
                      static_cast<double>(dense_tc_num) / (check_word_num * word_in_bits));
            log_debug("All-Edge Computation Cost (CF-Normal): %.9lf s, %s KB", time / pow(10, 9),
                      FormatWithCommas(getValue()).c_str());
        }
    }
}

extern string reorder_method;
extern string dir;

void SCAN_XP::CheckCore(Graph *g) {
    auto clk_beg = high_resolution_clock::now();

#ifdef DEG_DESCENDING_REORDERING
    reorder_method = "deg";
#endif
    if (reorder_method == "deg") {
        // Counting on the Degree-Ordered-Directed-Graph (View).
        log_info("Re-ordering... Method: DODG");
        ReorderDegDescending(*g);
    } else if (reorder_method == "kcore") {
        log_info("Re-ordering... Method: Kcore");
        ReorderKCoreDegeneracy(*g);
    } else if (reorder_method == "random") {
        log_info("Re-ordering... Method: Random");
        ReorderRandom(*g);
    } else if (reorder_method == "gro" || reorder_method == "cache" ||
               reorder_method == "bfsr" || reorder_method == "dfs" ||
               reorder_method == "hybrid" || reorder_method == "rcm-cache"
               || reorder_method == "slashburn") {
        log_info("Re-ordering... Method: %s", reorder_method.c_str());
        string reorder_file_path = dir + "/" + reorder_method + ".dict";
        FILE *pFile = fopen(reorder_file_path.c_str(), "r");
        YcheSerializer serializer;
        serializer.read_array(pFile, new_vid_dict);
        fclose(pFile);
        log_info("Finish loading %s", reorder_file_path.c_str());

        old_vid_dict = vector<int>(g->degree.size());
        assert(new_vid_dict.size() == old_vid_dict.size());
#pragma omp parallel for
        for (auto v = 0; v < g->nodemax; v++) {
            old_vid_dict[new_vid_dict[v]] = v;
        }
        Reorder(*g);
    } else {
        old_vid_dict = vector<int>(g->nodemax);
        for (auto i = 0; i < g->nodemax; i++) {
            old_vid_dict[i] = i;
        }
        log_info("Preserving the original order...");
    }
    auto clk_end = high_resolution_clock::now();
    auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
    log_info("PreProcess Reordering Cost: %.9lf s, %s KB", time / pow(10, 9), FormatWithCommas(getValue()).c_str());

    auto start = high_resolution_clock::now();

#if defined(COMPACT_FORWARD)
    CheckCoreCompactForward(g);
#elif defined(HYBRID_CF_NORMAL)
    CheckCoreHybridCFNormal(g);
#elif defined(EMPTY_HEADED)
    CheckCoreEmptyHeaded(g);
#elif defined(BSR)
    CheckCoreBSR(g);
#elif defined(ROARING)
    CheckCoreRoaring(g);
#elif defined(HASH) || defined(HASH_SPP)
    CheckCoreHash(g);
#elif defined(BIT_VEC)
    CheckCoreBitmap(g);
#elif defined(BIT_ADVANCED)
    CheckCoreBitmapAdvanced(g);
#elif defined(BIT_ONLINE_PACK)
    CheckCoreBitmapOnlinePack(g);
#else
    CheckCoreSortedArray(g);
#endif

    auto end = high_resolution_clock::now();
    log_debug("All-Edge CN-Cnt Cost: %.3lf s, Mem Usage: %s KB",
              duration_cast<milliseconds>(end - start).count() / 1000.0,
              FormatWithCommas(getValue()).c_str());
#ifdef STAT
    for (int i = 0; i < 10; i++) {
        if (global_bins[i].cnt_ != 0) {
            log_info("bins[%d]: %s, avg: %.3lf us, total: %.9lf s; "
                     "min: %.3lf us, max: %.3lf us, "
                     "avg-deg: (%d, %d), "
                     "avg-sel-ratio: %.6lf, avg-sel#: %d, min-max-sel#: (%d, %d), "
                     "avg-skew#: %d, min-max-skew#: (%d, %d)", i,
                     FormatWithCommas(global_bins[i].cnt_).c_str(),
                     static_cast<double>(global_bins[i].acc_time_) / global_bins[i].cnt_ / pow(10, 3),
                     global_bins[i].acc_time_ / pow(10, 9),
                     static_cast<double>(global_bins[i].min_time_) / pow(10, 3),
                     static_cast<double>(global_bins[i].max_time_) / pow(10, 3),
                     global_bins[i].acc_min_degree_ / global_bins[i].cnt_,
                     global_bins[i].acc_max_degree_ / global_bins[i].cnt_,
                     global_bins[i].acc_select_ratio_ / global_bins[i].cnt_,
                     global_bins[i].acc_select_ / global_bins[i].cnt_,
                     global_bins[i].min_select_,
                     global_bins[i].max_select_,
                     global_bins[i].acc_skew_ / global_bins[i].cnt_,
                     global_bins[i].min_skew_,
                     global_bins[i].max_skew_);
        }
    }
    for (int i = 0; i < 10; i++) {
        if (global_lb_bins[i].cnt_ != 0) {
            log_info("LB bins[%d]: %s, avg: %.3lf us, min: %.3lf us, total: %.9lf s; ", i,
                     FormatWithCommas(global_lb_bins[i].cnt_).c_str(),
                     static_cast<double>(global_lb_bins[i].acc_time_) / global_lb_bins[i].cnt_ / pow(10, 3),
                     static_cast<double>(global_lb_bins[i].min_time_) / pow(10, 3),
                     global_lb_bins[i].acc_time_ / pow(10, 9));
        }
    }
#endif
}

void SCAN_XP::ClusterCore() {
//#pragma omp parallel for num_threads(thread_num_) schedule(dynamic, 2000)
#pragma omp parallel for schedule(dynamic, 2000)
    for (auto i = 0u; i < g.nodemax; i++) {
        if (g.label[i] == CORE) {
            for (auto edge_idx = g.node_off[i]; edge_idx < g.node_off[i + 1]; edge_idx++) {
                // yche: fix bug, only union when g.edge_dst[edge_idx] is a core vertex
                if (g.label[g.edge_dst[edge_idx]] == CORE && g.similarity[edge_idx]) {
                    uf_ptr->UnionThreadSafe(i, g.edge_dst[edge_idx]);
                }
            }
        }
    }
}

bool SCAN_XP::CheckHub(Graph *g, UnionFind *uf, int a) {
    set<int> c;

    for (auto i = g->node_off[a]; i < g->node_off[a + 1]; i++) {
        if (g->label[g->edge_dst[i]] != CORE)continue;
        c.insert((*uf).FindRoot(g->edge_dst[i]));
    }
    return c.size() >= 2;
}

void SCAN_XP::LabelNonCore() {
    int core_num = 0u;
//#pragma omp parallel for num_threads(thread_num_) schedule(dynamic, 1000), reduction(+:core_num)
#pragma omp parallel for schedule(dynamic, 1000), reduction(+:core_num)
    for (auto i = 0u; i < g.nodemax; i++) {
        if (g.label[i] == CORE) {
            core_num++;
            continue;
        }

        if (CheckHub(&g, uf_ptr, i)) {
            g.label[i] = HUB;
        }
    }
    log_info("Core: %s", FormatWithCommas(core_num).c_str());
}

void SCAN_XP::PostProcess() {
    int cluster_num;
    int hub_num = 0u;
    int out_num = 0u;

    set<int> c;
    for (auto i = 0u; i < g.nodemax; i++) {
        if (g.label[i] == CORE) {
            c.emplace(uf_ptr->FindRoot(i));
        }
    }
    cluster_num = static_cast<int>(c.size());

//#pragma omp parallel for num_threads(thread_num_) reduction(+:hub_num, out_num)
#pragma omp parallel for reduction(+:hub_num, out_num)
    for (auto i = 0u; i < g.nodemax; i++) {
        if (g.label[i] == HUB) {
            hub_num++;
        } else if (g.label[i] == UNCLASSIFIED) {
            out_num++;
        }
    }
    log_info("Cluster: %s, Hub: %s, Outlier: %s", FormatWithCommas(cluster_num).c_str(),
             FormatWithCommas(hub_num).c_str(), FormatWithCommas(out_num).c_str());
}

void SCAN_XP::MarkClusterMinEleAsId(UnionFind *union_find_ptr) {
    auto start = high_resolution_clock::now();
    g.cluster_dict = vector<int>(g.nodemax);
    std::fill(g.cluster_dict.begin(), g.cluster_dict.end(), g.nodemax);

//#pragma omp parallel for num_threads(thread_num_)
#pragma omp parallel for
    for (auto i = 0u; i < g.nodemax; i++) {
#ifndef REORDERING
        if (g.label[i] == CORE) {
            int x = union_find_ptr->FindRoot(i);
            int cluster_min_ele;
            do {
                cluster_min_ele = g.cluster_dict[x];
                if (i >= g.cluster_dict[x]) {
                    break;
                }
            } while (!__sync_bool_compare_and_swap(&(g.cluster_dict[x]), cluster_min_ele, i));
        }
#else
        if (g.label[i] == CORE) {
            int x = union_find_ptr->FindRoot(i);
            int cluster_min_ele;
            do {
                cluster_min_ele = g.cluster_dict[x];
                if (old_vid_dict[i] >= g.cluster_dict[x]) {
                    break;
                }
            } while (!__sync_bool_compare_and_swap(&(g.cluster_dict[x]), cluster_min_ele, old_vid_dict[i]));
        }
#endif

    }
    auto end = high_resolution_clock::now();
    log_info("Step4 - cluster id initialization cost: %.3lf s, Mem Usage: %s KB",
             duration_cast<milliseconds>(end - start).count() / 1000.0,
             FormatWithCommas(getValue()).c_str());
}

void SCAN_XP::PrepareResultOutput() {
    // prepare output
    MarkClusterMinEleAsId(uf_ptr);

    auto start = high_resolution_clock::now();
//#pragma omp parallel num_threads(thread_num_)
#pragma omp parallel
    {
        vector<pair<int, int>> local_non_core_cluster;
#pragma omp for nowait
        for (auto i = 0u; i < g.nodemax; i++) {
            if (g.label[i] == CORE) {
                for (auto j = g.node_off[i]; j < g.node_off[i + 1]; j++) {
                    auto v = g.edge_dst[j];
                    if (g.label[v] != CORE && g.similarity[j]) {
#ifndef REORDERING
                        local_non_core_cluster.emplace_back(g.cluster_dict[uf_ptr->FindRoot(i)], v);
#else
                        local_non_core_cluster.emplace_back(g.cluster_dict[uf_ptr->FindRoot(i)],
                                                            old_vid_dict[v]);
#endif
                    }
                }
            }
        }
#pragma omp critical
        {
            for (auto ele: local_non_core_cluster) {
                g.noncore_cluster.emplace_back(ele);
            }
        };

    };
    auto end = high_resolution_clock::now();
    log_info("Step4 - prepare results: %.3lf s, Mem Usage: %s KB",
             duration_cast<milliseconds>(end - start).count() / 1000.0,
             FormatWithCommas(getValue()).c_str());

    auto epsilon_str = to_string(epsilon_);
    epsilon_str.erase(epsilon_str.find_last_not_of("0u") + 1);

    start = high_resolution_clock::now();

#ifndef REORDERING
    g.Output(epsilon_str.c_str(), to_string(min_u_).c_str(), uf_ptr);
#else
    g.Output(epsilon_str.c_str(), to_string(min_u_).c_str(), uf_ptr, old_vid_dict);
#endif
    end = high_resolution_clock::now();
    log_info("Step4 - output to the disk cost: %.3lf s, Mem Usage: %s KB",
             duration_cast<milliseconds>(end - start).count() / 1000.0,
             FormatWithCommas(getValue()).c_str());
}

void SCAN_XP::Execute() {
#ifdef CNT_STAT
    int64_t galloping_cnt = 0;
    int64_t blk_merge_cnt = 0;

#pragma omp parallel for num_threads(thread_num_) schedule(dynamic, 6000) reduction(+:galloping_cnt) reduction(+:blk_merge_cnt)
    for (auto i = 0u; i < g.edgemax; i++) {
        // remove edge_src optimization, assuming task scheduling in FIFO-queue-mode
        static thread_local auto u = 0;
        u = FindSrc(&g, u, i);

        auto v = g.edge_dst[i];
        if (u < v) {
            if (g.degree[u] / 50 > g.degree[v] || g.degree[v] / 50 > g.degree[u]) {
               galloping_cnt++;
            } else {
                blk_merge_cnt++;
            }
        }
    }
    log_info("skew: %s, not-skew: %s, total: %s", FormatWithCommas(galloping_cnt).c_str(), FormatWithCommas(blk_merge_cnt).c_str(),
            FormatWithCommas(galloping_cnt + blk_merge_cnt).c_str());
#else
    //step1 CheckCore
    double s1 = omp_get_wtime();
    CheckCore(&g);
    CheckCoreCompSimCore(&g);
    double e1 = omp_get_wtime();
    log_info("Step1 - CheckCore: %.3lf s, Mem Usage: %s KB", e1 - s1, FormatWithCommas(getValue()).c_str());
#endif

    //step2 ClusterCore
    double s2 = omp_get_wtime();
    ClusterCore();
    double e2 = omp_get_wtime();
    log_info("Step2 - ClusterCore: %.3lf s, Mem Usage: %s KB", e2 - s2,
             FormatWithCommas(getValue()).c_str());

    //step3 LabelNonCore
    double s3 = omp_get_wtime();
    LabelNonCore();
    double e3 = omp_get_wtime();
    log_info("Step3 - LabelNonCore: %.3lf s, Mem Usage: %s KB", e3 - s3,
             FormatWithCommas(getValue()).c_str());

    // post-processing, prepare result and output
    PostProcess();
    PrepareResultOutput();
}